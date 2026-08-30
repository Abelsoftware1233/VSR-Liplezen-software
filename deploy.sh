#!/usr/bin/env bash
#
# deploy.sh — Visual Speech Recognition (VSR) Multi-Client Stack
# ==================================================================
# Zet de Python API en de JS web-UI op als systemd-services:
#
#   - API (FastAPI/uvicorn):   0.0.0.0:6040
#   - Web-UI (statisch):       0.0.0.0:6041
#
# LET OP — AFWIJKING VAN DE ORIGINELE REPO-OPZET:
#   De repo (README.md / MULTI_CLIENT_README.md / api/server.py) is
#   ontworpen om UITSLUITEND op 127.0.0.1 (localhost) te draaien: "no data
#   leaves your machine". Dit deploy-script bindt bewust op 0.0.0.0, dus
#   de API en web-UI worden bereikbaar voor ELK apparaat dat de host op
#   het netwerk kan bereiken — inclusief video-uploads die op de server
#   worden verwerkt. Dat is een expliciete keuze; als dat niet de bedoeling
#   is, verander ELF_LISTEN_HOST hieronder naar 127.0.0.1 en draai dan
#   zonder --uninstall opnieuw.
#
# Wat dit script verder doet (idempotent — veilig om opnieuw te draaien):
#   1. Root-check (nodig voor systemd + gebruikersbeheer).
#   2. Controleert ffmpeg (systeemvereiste, niet via pip installeerbaar
#      op een manier die overal werkt — het script waarschuwt en stopt
#      niet hard, zodat je zelf kunt kiezen hoe je dat oplost).
#   3. Maakt een dedicated systeemgebruiker aan.
#   4. Kopieert het project naar een installatie-directory.
#   5. Zet een Python-venv op en installeert requirements.txt (dit bevat
#      torch — dit KAN LANG DUREN en veel schijfruimte kosten,
#      zie README.md: ~6-8 GB).
#   6. Downloadt de pretrained modelgewichten (download_models.py),
#      éénmalig — wordt overgeslagen als model_weights/ al gevuld is.
#   7. Genereert TWEE systemd-services (API + web-UI) en start ze.
#   8. Health-check op de API.
#
# Gebruik:
#   sudo ./deploy.sh                    # installeren/updaten + starten
#   sudo ./deploy.sh --skip-models      # sla de modellen-download over
#                                       #   (handig bij herhaald testen)
#   sudo ./deploy.sh --uninstall        # beide services stoppen/verwijderen
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuratie
# ---------------------------------------------------------------------------
API_SERVICE_NAME="vsr-api"
WEB_SERVICE_NAME="vsr-web"
API_PORT="6040"
WEB_PORT="6041"
LISTEN_HOST="0.0.0.0"     # bewust netwerk-breed — zie waarschuwing hierboven
APP_USER="vsr"
APP_GROUP="vsr"
INSTALL_DIR="/opt/vsr-app"
VENV_DIR="${INSTALL_DIR}/venv"
API_SERVICE_FILE="/etc/systemd/system/${API_SERVICE_NAME}.service"
WEB_SERVICE_FILE="/etc/systemd/system/${WEB_SERVICE_NAME}.service"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SKIP_MODELS=0
for arg in "$@"; do
    if [[ "$arg" == "--skip-models" ]]; then
        SKIP_MODELS=1
    fi
done

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
    C_GREEN='\033[0;32m'; C_YELLOW='\033[0;33m'; C_RED='\033[0;31m'; C_BLUE='\033[0;34m'; C_RESET='\033[0m'
else
    C_GREEN=''; C_YELLOW=''; C_RED=''; C_BLUE=''; C_RESET=''
fi
log()  { echo -e "${C_BLUE}[deploy]${C_RESET} $1"; }
ok()   { echo -e "${C_GREEN}[ok]${C_RESET} $1"; }
warn() { echo -e "${C_YELLOW}[let op]${C_RESET} $1"; }
err()  { echo -e "${C_RED}[fout]${C_RESET} $1" >&2; }

# Bij elke fout: toon exact welke regel/commando faalde, zodat je niet meer
# hoeft te gokken (of een screenshot hoeft te sturen) om te weten waar het
# misging. Nu err() bekend is, kan de trap 'm gebruiken.
on_error() {
    err "Script gestopt op regel ${1} — commando: ${2}"
    err "Dit is de ECHTE oorzaak. Los dit op en draai daarna gewoon opnieuw: sudo ./deploy.sh"
    err "(het script is idempotent — al voltooide stappen worden overgeslagen)"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

# ---------------------------------------------------------------------------
# --uninstall pad
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--uninstall" ]]; then
    log "Services ${API_SERVICE_NAME} en ${WEB_SERVICE_NAME} verwijderen…"
    systemctl stop "${API_SERVICE_NAME}.service" 2>/dev/null || true
    systemctl stop "${WEB_SERVICE_NAME}.service" 2>/dev/null || true
    systemctl disable "${API_SERVICE_NAME}.service" 2>/dev/null || true
    systemctl disable "${WEB_SERVICE_NAME}.service" 2>/dev/null || true
    rm -f "${API_SERVICE_FILE}" "${WEB_SERVICE_FILE}"
    systemctl daemon-reload
    warn "Services verwijderd. Installatiemap (${INSTALL_DIR}, inclusief gedownloade"
    warn "modelgewichten van meerdere GB's) en gebruiker (${APP_USER}) zijn NIET verwijderd."
    warn "Handmatig opruimen indien gewenst:"
    echo "    sudo rm -rf ${INSTALL_DIR}"
    echo "    sudo userdel ${APP_USER}"
    exit 0
fi

# ---------------------------------------------------------------------------
# 1. Root-check
# ---------------------------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
    err "Dit script heeft root nodig (systemd-unit + poortbinding + gebruikersbeheer)."
    err "Start opnieuw met: sudo ./deploy.sh"
    exit 1
fi

echo ""
warn "Dit zet de API (poort ${API_PORT}) en web-UI (poort ${WEB_PORT}) neer op"
warn "${LISTEN_HOST} — dus bereikbaar vanaf het netwerk, NIET localhost-only"
warn "zoals de originele repo beschrijft. Video's die mensen uploaden worden"
warn "op deze server verwerkt. Ctrl+C nu als dat niet de bedoeling is."
echo ""
sleep 3

# ---------------------------------------------------------------------------
# 2. Vereisten controleren
# ---------------------------------------------------------------------------
log "Vereisten controleren…"

if ! command -v python3 >/dev/null 2>&1; then
    err "python3 niet gevonden. Installeer eerst Python 3.10 of 3.12 (zie README.md)."
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
log "Python versie gevonden: ${PY_VERSION}"
if [[ "${PY_VERSION}" != "3.10" && "${PY_VERSION}" != "3.12" ]]; then
    warn "README.md vraagt om Python 3.10 of 3.12 — gevonden: ${PY_VERSION}."
    warn "torch kan hierop falen. Ga door op eigen risico."
fi

if ! python3 -m venv --help >/dev/null 2>&1; then
    err "python3-venv ontbreekt. Installeer met bv.: apt install python3-venv"
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    warn "ffmpeg niet gevonden op PATH — vereist voor video/frame-decoding (zie README.md)."
    warn "Installeer bv. met: apt install ffmpeg   (of het equivalent voor jouw distro)"
    warn "De installatie gaat door, maar transcriptie zal falen zonder ffmpeg."
else
    ok "ffmpeg gevonden."
fi

if ! command -v git >/dev/null 2>&1; then
    err "git niet gevonden — nodig voor download_models.py en eventuele git-based dependencies."
    exit 1
fi

if command -v systemctl >/dev/null 2>&1; then
    HAS_SYSTEMD=1
else
    err "systemctl niet gevonden. Dit script vereist systemd voor de gevraagde service-opzet."
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Systeemgebruiker aanmaken (idempotent)
# ---------------------------------------------------------------------------
if id "${APP_USER}" >/dev/null 2>&1; then
    log "Gebruiker '${APP_USER}' bestaat al, hergebruiken."
else
    log "Systeemgebruiker '${APP_USER}' aanmaken (geen login, geen eigen home)…"
    useradd --system --no-create-home --shell /usr/sbin/nologin "${APP_USER}"
    ok "Gebruiker '${APP_USER}' aangemaakt."
fi

# ---------------------------------------------------------------------------
# 4. Bestanden naar installatiemap kopiëren
# ---------------------------------------------------------------------------
log "Bestanden kopiëren naar ${INSTALL_DIR}…"
mkdir -p "${INSTALL_DIR}"

if command -v rsync >/dev/null 2>&1; then
    rsync -a \
        --exclude 'venv' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.git' \
        --exclude 'model_weights' \
        --exclude 'third_party' \
        "${SCRIPT_DIR}/" "${INSTALL_DIR}/"
else
    cp -r "${SCRIPT_DIR}/." "${INSTALL_DIR}/"
    rm -rf "${INSTALL_DIR}/venv" "${INSTALL_DIR}/__pycache__"
fi
ok "Bestanden gekopieerd (model_weights/ blijft ongemoeid als het al bestaat, zodat"
ok "eerder gedownloade gewichten van meerdere GB's niet worden overschreven)."

# ---------------------------------------------------------------------------
# 5. Python-venv opzetten + dependencies installeren
# ---------------------------------------------------------------------------
if [[ -d "${VENV_DIR}" ]]; then
    log "Virtuele omgeving bestaat al, dependencies bijwerken…"
else
    log "Virtuele omgeving aanmaken in ${VENV_DIR}…"
    python3 -m venv "${VENV_DIR}"
fi

log "Dependencies installeren — dit bevat torch en kan lang duren"
log "(README.md noemt ~6-8 GB schijfruimte voor dependencies + gewichten)…"

# ---------------------------------------------------------------------------
# fairseq/AV-HuBERT (secundair ensemble-model) is UITGESCHAKELD.
# ---------------------------------------------------------------------------
# fairseq (2022, niet meer bijgewerkt) eist omegaconf<2.1 en
# antlr4-python3-runtime==4.8 — onverenigbaar met de rest van deze
# requirements.txt (pytorch-lightning/hydra-core trekken nieuwere versies
# van diezelfde pakketten binnen). Eerdere pogingen om fairseq er via een
# aparte pip-stap toch bij te installeren, installeerden wel zonder
# expliciete fout, maar braken de omgeving alsnog STIL: pip's resolver
# waarschuwt dan alleen achteraf ("does not currently take into account
# all the packages that are installed") en fairseq crasht pas bij het
# eerste daadwerkelijke gebruik. Vandaar: fairseq staat nu uitgecommentarieerd
# in requirements.txt, en dit script installeert het NIET meer. De code in
# api/server.py importeert AVHubertTranscriber defensief (try/except), dus
# de API start gewoon door zonder het secundaire model.
#
# Wil je dit ooit alsnog gebruiken: installeer fairseq in een VOLLEDIG
# apart venv (nooit in ${VENV_DIR}) en roep het aan via een subprocess.
"${VENV_DIR}/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/bin/pip" install -r "${INSTALL_DIR}/requirements.txt"
ok "Dependencies geïnstalleerd."

# ---------------------------------------------------------------------------
# Validatie: bevestig dat er geen resterende dependency-conflicten zijn.
# `pip check` vangt precies het soort stille conflict dat eerder de
# API-service liet crashen zonder duidelijke installatiefout.
# ---------------------------------------------------------------------------
log "Dependency-conflicten controleren (pip check)…"
if "${VENV_DIR}/bin/pip" check; then
    ok "Geen dependency-conflicten gevonden."
else
    err "pip check vond conflicten (zie hierboven). De installatie gaat door,"
    err "maar de API kan hierdoor alsnog falen bij opstarten. Los dit op voor"
    err "je verdergaat — dit is exact het soort probleem dat eerder de"
    err "'vsr-api'-service liet crashen zonder duidelijke oorzaak."
fi

# ---------------------------------------------------------------------------
# 6. Modelgewichten downloaden (eenmalig, overslaan met --skip-models)
# ---------------------------------------------------------------------------
if [[ $SKIP_MODELS -eq 1 ]]; then
    warn "--skip-models opgegeven: modellen-download overgeslagen."
elif [[ -d "${INSTALL_DIR}/model_weights" ]] && [[ -n "$(ls -A "${INSTALL_DIR}/model_weights" 2>/dev/null)" ]]; then
    log "model_weights/ bevat al bestanden, download overgeslagen (idempotent)."
    log "Forceer opnieuw downloaden met: sudo rm -rf ${INSTALL_DIR}/model_weights && sudo ./deploy.sh"
else
    log "Pretrained modelgewichten downloaden (eenmalig, ~2-3 GB)…"
    (cd "${INSTALL_DIR}" && "${VENV_DIR}/bin/python" download_models.py) \
        || { err "Downloaden van modelgewichten is mislukt. Los dit handmatig op en draai het script opnieuw."; exit 1; }
    ok "Modelgewichten gedownload."
fi

# ---------------------------------------------------------------------------
# 7. Eigendom/rechten zetten
# ---------------------------------------------------------------------------
log "Eigendom instellen op ${APP_USER}:${APP_GROUP}…"
chown -R "${APP_USER}:${APP_GROUP}" "${INSTALL_DIR}"
ok "Rechten gezet."

# ---------------------------------------------------------------------------
# 8. systemd unit-files genereren
# ---------------------------------------------------------------------------
log "systemd-service voor de API schrijven naar ${API_SERVICE_FILE}…"
cat > "${API_SERVICE_FILE}" <<EOF
[Unit]
Description=VSR Local API (FastAPI) — lip-reading transcriptie-backend
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${APP_USER}
Group=${APP_GROUP}
WorkingDirectory=${INSTALL_DIR}
ExecStart=${VENV_DIR}/bin/uvicorn api.server:app --host ${LISTEN_HOST} --port ${API_PORT}
Restart=on-failure
RestartSec=5
# Modelinferentie kan geheugen-/tijdsintensief zijn; ruimere timeout dan default.
TimeoutStartSec=120

# --- Hardening ---
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=${INSTALL_DIR}
# Poort ${API_PORT} is unprivileged (>1024) — geen extra capabilities nodig.

[Install]
WantedBy=multi-user.target
EOF
ok "API-service geschreven."

log "systemd-service voor de web-UI schrijven naar ${WEB_SERVICE_FILE}…"
cat > "${WEB_SERVICE_FILE}" <<EOF
[Unit]
Description=VSR Web UI (statische bestanden, python http.server)
After=network-online.target ${API_SERVICE_NAME}.service
Wants=network-online.target
Requires=${API_SERVICE_NAME}.service

[Service]
Type=simple
User=${APP_USER}
Group=${APP_GROUP}
WorkingDirectory=${INSTALL_DIR}/web
ExecStart=${VENV_DIR}/bin/python3 -m http.server ${WEB_PORT} --bind ${LISTEN_HOST}
Restart=on-failure
RestartSec=3

# --- Hardening ---
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=${INSTALL_DIR}

[Install]
WantedBy=multi-user.target
EOF
ok "Web-UI-service geschreven."

log "systemd herladen…"
systemctl daemon-reload

log "Services enablen (autostart bij boot)…"
systemctl enable "${API_SERVICE_NAME}.service" --quiet
systemctl enable "${WEB_SERVICE_NAME}.service" --quiet

log "API-service (her)starten — model laadt bij opstarten in het geheugen, dit kan even duren…"
systemctl restart "${API_SERVICE_NAME}.service"

# Geef de API tijd om het model te laden voor we de web-service (die er van
# afhangt) starten en voor de health-check.
log "Wachten tot de API klaar is (max 60s)…"
API_READY=0
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${API_PORT}/health" >/dev/null 2>&1; then
        API_READY=1
        break
    fi
    sleep 2
done

if [[ $API_READY -eq 1 ]]; then
    ok "API reageert op /health."
else
    warn "API reageerde niet binnen 60s. Mogelijk laadt het model nog (grote checkpoints"
    warn "kunnen langer duren op CPU). Check de status handmatig:"
    echo "    sudo journalctl -u ${API_SERVICE_NAME} -f"
fi

log "Web-UI-service (her)starten…"
systemctl restart "${WEB_SERVICE_NAME}.service"
sleep 1

if systemctl is-active --quiet "${API_SERVICE_NAME}.service"; then
    ok "Service '${API_SERVICE_NAME}' draait."
else
    err "Service '${API_SERVICE_NAME}' is niet actief. Logs bekijken met:"
    echo "    sudo journalctl -u ${API_SERVICE_NAME} -n 50 --no-pager"
fi

if systemctl is-active --quiet "${WEB_SERVICE_NAME}.service"; then
    ok "Service '${WEB_SERVICE_NAME}' draait."
else
    err "Service '${WEB_SERVICE_NAME}' is niet actief. Logs bekijken met:"
    echo "    sudo journalctl -u ${WEB_SERVICE_NAME} -n 50 --no-pager"
fi

echo ""
ok "Klaar."
echo ""
echo "  Web-UI:  http://<server-ip>:${WEB_PORT}/"
echo "  API:     http://<server-ip>:${API_PORT}/health"
echo ""
warn "Nogmaals: dit draait op ${LISTEN_HOST}, dus bereikbaar vanaf het netwerk."
warn "Zorg voor passende firewall-regels als dat niet voor iedereen open moet staan,"
warn "bv.: sudo ufw allow from <toegestaan-subnet> to any port ${API_PORT},${WEB_PORT} proto tcp"
echo ""
echo "Nuttige commando's:"
echo "  sudo systemctl status ${API_SERVICE_NAME} ${WEB_SERVICE_NAME}"
echo "  sudo systemctl restart ${API_SERVICE_NAME} ${WEB_SERVICE_NAME}"
echo "  sudo journalctl -u ${API_SERVICE_NAME} -f     # live logs API (incl. model-load)"
echo "  sudo journalctl -u ${WEB_SERVICE_NAME} -f     # live logs web-UI"
echo "  sudo ./deploy.sh --uninstall                  # beide services verwijderen"
echo ""
echo "De Java desktop-app (desktop-java/) blijft een LOKALE client — die praat"
echo "vanaf het apparaat waarop hij draait met de API. Als dat apparaat niet"
echo "deze server is, moet je in VSRApiClient.java 127.0.0.1 vervangen door"
echo "het IP-adres van deze server."
echo ""

# ---------------------------------------------------------------------------
# 9. Optioneel: nginx reverse proxy, zodat het domein zonder poortnummer werkt
# ---------------------------------------------------------------------------
if command -v nginx >/dev/null 2>&1; then
    log "nginx gevonden — reverse-proxyconfig aanmaken zodat het domein zonder"
    log "poortnummer werkt (bv. http://vsr.abelsoftware123.com i.p.v. :${WEB_PORT})…"

    NGINX_SITE_FILE="/etc/nginx/sites-available/vsr-app"
    cat > "${NGINX_SITE_FILE}" <<EOF
# VSR-app reverse proxy — gegenereerd door deploy.sh
# Web-UI op poort ${WEB_PORT}, API op poort ${API_PORT}, beide via /api/ proxied
# zodat de browser alles op poort 80 binnenkrijgt (geen CORS-gedoe, geen
# zichtbare poortnummers voor bezoekers).
server {
    listen 80;
    listen [::]:80;
    server_name _;   # vervang door je domeinnaam, bv. vsr.abelsoftware123.com

    client_max_body_size 500M;   # video-uploads kunnen groot zijn

    location /api/ {
        proxy_pass http://127.0.0.1:${API_PORT}/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300s;   # modelinferentie kan lang duren
    }

    location / {
        proxy_pass http://127.0.0.1:${WEB_PORT}/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
}
EOF
    ln -sf "${NGINX_SITE_FILE}" /etc/nginx/sites-enabled/vsr-app
    if nginx -t 2>&1 | grep -q "successful"; then
        systemctl reload nginx
        ok "nginx-config geactiveerd. Site draait nu ook op poort 80."
        warn "web/app.js gebruikt nog steeds poort ${API_PORT} rechtstreeks voor API-calls,"
        warn "niet de /api/ proxy-route hierboven. Voor volledige poort-loze werking via"
        warn "nginx moet API_BASE in web/app.js naar '/api' wijzen i.p.v. ':${API_PORT}'."
        echo "  Site (via nginx):  http://<server-ip-of-domeinnaam>/"
    else
        err "nginx-configtest faalde — config NIET geactiveerd. Controleer handmatig met: nginx -t"
    fi
else
    log "nginx niet gevonden — reverse-proxystap overgeslagen (optioneel)."
    log "Zie het einde van dit script voor handmatige nginx-installatie-instructies."
fi
