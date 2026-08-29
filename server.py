"""
Local REST API for the Visual Speech Recognition pipeline.

Runs entirely on your own machine (localhost) — this is NOT a hosted
backend. It exists purely so the JavaScript web UI and the Java desktop
app can both talk to a single, already-loaded model instead of each
re-implementing/re-loading the pipeline separately.

Run with:
    uvicorn api.server:app --host 127.0.0.1 --port 6040

Then:
    - the JS web UI (web/) points at http://127.0.0.1:6040
    - the Java desktop app (desktop-java/) points at http://127.0.0.1:6040

No data leaves your machine — 127.0.0.1 never touches the network.
"""

from __future__ import annotations
import tempfile
import shutil
from pathlib import Path

import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from utils.video_io import VideoReader, probe_video
from utils.face_tracking import LipTracker
from utils.roi_extraction import build_roi_sequence
from models.auto_avsr_wrapper import AutoAVSRTranscriber
from models.av_hubert_wrapper import AVHubertTranscriber
from models.lm_rescore import LMRescorer

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"
PORT = 6040

app = FastAPI(title="VSR Local API", version="1.0")

# CORS: allows the JS web UI (served from a browser, possibly a different
# local port like 5173 or 8080) to call this API. Restricted to localhost
# origins only — this API is not meant to be reachable from outside your
# machine.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:6040",
        "http://127.0.0.1:6040",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Models are loaded once at startup and kept warm in memory ----
_state: dict = {}


@app.on_event("startup")
def load_everything():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    _state["cfg"] = cfg

    _state["primary_model"] = AutoAVSRTranscriber(
        checkpoint_path=cfg["models"]["primary"]["checkpoint"],
        config_path=cfg["models"]["primary"]["config"],
        device=cfg["device"],
    )

    _state["secondary_model"] = None
    if cfg["models"]["secondary_ensemble"]["enabled"]:
        _state["secondary_model"] = AVHubertTranscriber(
            checkpoint_path=cfg["models"]["secondary_ensemble"]["checkpoint"],
            device=cfg["device"],
        )

    _state["lm"] = None
    if cfg["language_model"]["enabled"]:
        lm_cfg = cfg["language_model"]
        _state["lm"] = LMRescorer(
            backend=lm_cfg["backend"],
            lm_path=lm_cfg["lm_path"],
            llm_rescoring=lm_cfg.get("llm_rescoring"),
        )

    print(f"VSR API ready on http://127.0.0.1:{PORT}")


@app.get("/health")
def health():
    """Simple check the JS/Java clients can ping before offering the upload UI."""
    return {
        "status": "ok",
        "device": _state["cfg"]["device"],
        "secondary_model_enabled": _state["secondary_model"] is not None,
        "language_model_enabled": _state["lm"] is not None,
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), use_secondary: bool = False):
    """
    Accepts a video file upload, runs the full VSR pipeline, returns JSON:
        {
          "primary_transcript": "...",
          "secondary_transcript": "..." | null,
          "final_transcript": "...",
          "frames_total": int,
          "frames_with_face_detected": int
        }
    """
    cfg = _state["cfg"]
    suffix = Path(file.filename).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        meta = probe_video(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read video: {e}")

    try:
        with VideoReader(
            tmp_path,
            target_fps=cfg["video"]["target_fps"],
            max_duration_seconds=cfg["video"]["max_duration_seconds"],
        ) as reader:
            frames = list(reader.frames())

        if not frames:
            raise HTTPException(status_code=422, detail="No frames could be read from this video.")

        with LipTracker(
            min_detection_confidence=cfg["face_tracking"]["min_detection_confidence"],
            min_tracking_confidence=cfg["face_tracking"]["min_tracking_confidence"],
        ) as tracker:
            frames_with_boxes = [(f, tracker.detect_mouth(f)) for f in frames]

        detected_count = sum(1 for _, box in frames_with_boxes if box is not None)
        if detected_count == 0:
            raise HTTPException(
                status_code=422,
                detail="No face was detected in this video. Make sure the speaker's "
                       "face and mouth are clearly visible and reasonably front-facing.",
            )

        roi_sequence = build_roi_sequence(
            frames_with_boxes,
            crop_size=tuple(cfg["roi"]["crop_size"]),
            grayscale=cfg["roi"]["grayscale"],
            normalize=cfg["roi"]["normalize"],
        )

        primary_transcript = _state["primary_model"].transcribe(roi_sequence)
        final_text = primary_transcript

        secondary_transcript = None
        if use_secondary and _state["secondary_model"] is not None:
            secondary_transcript = _state["secondary_model"].transcribe(roi_sequence)
            if len(secondary_transcript) > len(final_text):
                final_text = secondary_transcript

        if _state["lm"] is not None:
            final_text = _state["lm"].rescore(final_text)

        return JSONResponse({
            "primary_transcript": primary_transcript,
            "secondary_transcript": secondary_transcript,
            "final_transcript": final_text,
            "frames_total": len(frames_with_boxes),
            "frames_with_face_detected": detected_count,
            "video_meta": meta,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="127.0.0.1", port=PORT, reload=False)
