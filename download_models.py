"""
One-time setup script: downloads existing pretrained VSR model weights and
clones the upstream inference code they depend on. This script does NOT
train anything — it only fetches checkpoints that were already trained by
their original authors.

Usage:
    python download_models.py                  # Auto-AVSR only (recommended default)
    python download_models.py --with-av-hubert  # also fetch AV-HuBERT for ensembling
    python download_models.py --with-lm         # also fetch the subword n-gram language model

Sources:
    - Auto-AVSR weights & code: https://github.com/mpc001/auto_avsr (Apache 2.0 code license;
      weights carry LRS3 dataset terms — non-commercial, see docs/LICENSES.md)
    - AV-HuBERT weights & code: https://github.com/facebookresearch/av_hubert
    - Language model: bundled with the Auto-AVSR / Chaplin release on Hugging Face Hub
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
THIRD_PARTY = ROOT / "third_party"
WEIGHTS = ROOT / "model_weights"

AUTO_AVSR_REPO = "https://github.com/mpc001/auto_avsr.git"
AV_HUBERT_REPO = "https://github.com/facebookresearch/av_hubert.git"

# Hugging Face Hub repo met de officiële, door de Auto-AVSR-onderzoeksgroep
# zelf gepubliceerde checkpoints. Voorheen wees dit naar een derde-partij
# repo ("amanvirparhar/chaplin-models") die als makkelijke re-host diende,
# maar die geeft nu een 401/RepositoryNotFoundError (verwijderd, hernoemd,
# of op privé gezet — niet iets wat wij kunnen beïnvloeden). Dit is de
# officiële bron: een HF SPACE (geen gewoon model-repo!) van mpc001, de
# oorspronkelijke Auto-AVSR-auteurs, met de checkpoints onder
# benchmarks/LRS3/models/LRS3_V_WER19.1/.
HF_AUTO_AVSR_REPO_ID = "mpc001/auto_avsr"
HF_AUTO_AVSR_REPO_TYPE = "space"
HF_AUTO_AVSR_SUBDIR = "benchmarks/LRS3/models/LRS3_V_WER19.1"


def run(cmd: list[str], cwd: Path | None = None):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def clone_if_missing(repo_url: str, target_dir: Path):
    if target_dir.exists():
        print(f"[skip] {target_dir} already exists")
        return
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", repo_url, str(target_dir)])


def download_auto_avsr():
    print("\n=== Auto-AVSR (primary model) ===")
    clone_if_missing(AUTO_AVSR_REPO, THIRD_PARTY / "auto_avsr")

    dest = WEIGHTS / "auto_avsr"
    dest.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    print(f"Downloading pretrained checkpoint from Hugging Face Hub "
          f"({HF_AUTO_AVSR_REPO_ID}, subdir: {HF_AUTO_AVSR_SUBDIR})...")
    # snapshot_download i.p.v. losse hf_hub_download-aanroepen per bestand:
    # dit is een HF SPACE (geen model-repo), en we weten de exacte
    # bestandsnamen binnen de checkpoint-map niet met zekerheid (kan per
    # Auto-AVSR-versie verschillen). allow_patterns haalt alleen de relevante
    # submap op, niet de volledige ~3.7 GB Space.
    local_dir = Path(snapshot_download(
        repo_id=HF_AUTO_AVSR_REPO_ID,
        repo_type=HF_AUTO_AVSR_REPO_TYPE,
        allow_patterns=[f"{HF_AUTO_AVSR_SUBDIR}/*", "configs/LRS3_V_WER19.1.ini"],
        local_dir=str(dest / "_hf_snapshot"),
    ))

    # configs/default.yaml verwacht de checkpoint op een vast, plat pad
    # (model_weights/auto_avsr/lrs3_v_wer19.1.pth) — onafhankelijk van hoe
    # de bron-repo intern is gestructureerd. Zoek het .pth-bestand in de
    # gedownloade submap en symlink het naar dat vaste pad, in plaats van
    # de exacte upstream bestandsnaam te hardcoden (die kan wijzigen).
    checkpoint_dir = local_dir / HF_AUTO_AVSR_SUBDIR
    pth_candidates = list(checkpoint_dir.glob("*.pth"))
    if not pth_candidates:
        raise FileNotFoundError(
            f"Geen .pth-checkpoint gevonden in {checkpoint_dir} na download. "
            f"Bekijk de mapinhoud handmatig: ls {checkpoint_dir}"
        )
    checkpoint_target = dest / "lrs3_v_wer19.1.pth"
    if not checkpoint_target.exists():
        checkpoint_target.symlink_to(pth_candidates[0].resolve())
    print(f"  -> checkpoint: {checkpoint_target} -> {pth_candidates[0]}")

    config_source = local_dir / "configs" / "LRS3_V_WER19.1.ini"
    config_target = dest / "lrs3_v_wer19.1.ini"
    if config_source.exists() and not config_target.exists():
        config_target.symlink_to(config_source.resolve())
        print(f"  -> config: {config_target} -> {config_source}")
    elif not config_source.exists():
        print(f"  [waarschuwing] configbestand niet gevonden op {config_source}")

    print("Auto-AVSR ready.")


def download_av_hubert():
    print("\n=== AV-HuBERT (optional secondary model) ===")
    clone_if_missing(AV_HUBERT_REPO, THIRD_PARTY / "av_hubert")

    fairseq_dir = THIRD_PARTY / "av_hubert" / "fairseq"
    if not fairseq_dir.exists():
        run(["git", "clone", "--depth", "1",
             "https://github.com/pytorch/fairseq.git", str(fairseq_dir)])
        run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=fairseq_dir)

    dest = WEIGHTS / "av_hubert"
    dest.mkdir(parents=True, exist_ok=True)
    print(
        "NOTE: AV-HuBERT's official LRS3-finetuned checkpoint requires "
        "accepting Meta's model license via their form:\n"
        "  https://github.com/facebookresearch/av_hubert#model-checkpoints\n"
        "This script cannot auto-accept that license for you. Download "
        "'large_lrs3.pt' manually from the link above and place it at:\n"
        f"  {dest / 'large_lrs3.pt'}"
    )


def download_language_model():
    print("\n=== Subword n-gram language model (optioneel) ===")
    print(
        "SKIPPED: dit bestand (lm_en_subword.arpa) kwam oorspronkelijk uit "
        "dezelfde derde-partij HuggingFace-repo als het primaire model "
        "(amanvirparhar/chaplin-models), die niet meer bereikbaar is (401). "
        "De officiële mpc001/auto_avsr-repo bevat dit specifieke bestand "
        "niet — het lijkt Chaplin-eigen te zijn, niet standaard Auto-AVSR.\n"
        "Taalmodel-rescoring is een optionele verbetering bovenop de kern-"
        "transcriptie (zie configs/default.yaml -> language_model.enabled), "
        "geen vereiste. Zet 'enabled: false' in die config, of vraag om dit "
        "verder uit te zoeken als je 'm alsnog nodig hebt."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-av-hubert", action="store_true",
                         help="Also download AV-HuBERT for ensemble transcription")
    parser.add_argument("--with-lm", action="store_true", default=True,
                         help="Also download the n-gram language model (default: on)")
    parser.add_argument("--skip-lm", action="store_true",
                         help="Skip language model download")
    args = parser.parse_args()

    WEIGHTS.mkdir(exist_ok=True)
    THIRD_PARTY.mkdir(exist_ok=True)

    download_auto_avsr()

    if args.with_av_hubert:
        download_av_hubert()

    if args.with_lm and not args.skip_lm:
        download_language_model()

    print("\nAll done. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
