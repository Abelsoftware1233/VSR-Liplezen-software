# Multi-Client Setup: Python API + JavaScript Web UI + Java Desktop App

This extends the original Streamlit app with a local REST API and two
separate clients — a JavaScript web UI and a Java (JavaFX) desktop app —
that both talk to the same model.

**Everything still runs 100% locally.** The "API" is a Python process on
your own machine listening on `127.0.0.1:6040` — it never leaves your
computer and is not reachable from the internet.

```
┌─────────────────────┐     ┌─────────────────────┐
│  JS Web UI (browser)│     │ Java Desktop (JavaFX)│
│  web/index.html      │     │ desktop-java/         │
└──────────┬───────────┘     └──────────┬────────────┘
           │  HTTP (localhost only)      │  HTTP (localhost only)
           └──────────────┬───────────────┘
                           ▼
              ┌─────────────────────────┐
              │  Python API (FastAPI)    │
              │  127.0.0.1:6040           │
              │  api/server.py            │
              │                            │
              │  Loads Auto-AVSR + │
              │  AV-HuBERT + LM once,     │
              │  keeps them warm in       │
              │  memory                   │
              └─────────────────────────┘
```

## Why this structure

- The model is only loaded **once**, in the Python process, instead of
  being duplicated per client.
- Both the web UI and the desktop app become thin clients — they just
  upload a video and display JSON back.
- You still don't need a hosted backend or cloud infrastructure — this is
  three local processes on one machine talking over `localhost`.

## 1. Start the Python API

```bash
# from the repo root, with your existing venv activated
pip install fastapi "uvicorn[standard]" python-multipart

uvicorn api.server:app --host 127.0.0.1 --port 6040
```

You should see:
```
VSR API ready on http://127.0.0.1:6040
```

Leave this running in its own terminal. Both clients below depend on it.

Quick check it's alive:
```bash
curl http://127.0.0.1:6040/health
```

## 2. JavaScript Web UI

No build step, no npm install needed — it's a static page that calls the
local API directly via `fetch()`.

```bash
cd web
python3 -m http.server 8080
# or: npx serve .
```

Open `http://localhost:8080` in your browser. Upload a video, click
Transcribe.

> Note: opening `web/index.html` directly as a `file://` URL will hit CORS
> issues in some browsers — serving it over `http://localhost:8080` (as
> above) avoids that.

## 3. Java Desktop App

Requires Java 17+ and Maven.

```bash
cd desktop-java
mvn clean javafx:run
```

A native window opens. It pings `127.0.0.1:6040/health` on startup to
confirm the API is running, then lets you pick a video file and transcribe
it the same way the web UI does.

To build a standalone jar instead of running via Maven each time:
```bash
mvn clean package
java -jar target/vsr-desktop-1.0.0.jar
```
(You'll need the JavaFX runtime on your module path for a plain `java -jar`
run — `mvn javafx:run` handles that for you automatically, which is the
simpler option for local use.)

## Port 6040 — why it's configurable in one place

The port is set in exactly three places, all currently pointed at **6040**:

| File | Line |
|---|---|
| `api/server.py` | `PORT = 6040` |
| `web/app.js` | `const API_BASE = "http://127.0.0.1:6040";` |
| `desktop-java/.../VSRApiClient.java` | `public static final String API_BASE = "http://127.0.0.1:6040";` |

If you ever need to change the port, update it in all three.

## Everything else

The actual VSR pipeline (Auto-AVSR, AV-HuBERT, MediaPipe face tracking, LM
rescoring, accuracy limitations) is unchanged from the original Streamlit
version — see the main `README.md` and `docs/ACCURACY_NOTES.md`. This
document only covers the new client/server split.
