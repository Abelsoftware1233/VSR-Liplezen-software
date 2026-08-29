"""
Streamlit app: upload a silent video, get a text transcription generated
purely from lip movement (no audio is read at any point in this pipeline).

Run with:
    streamlit run app.py

Everything below runs locally in this one process — there is no backend
server and no data leaves your machine (aside from the one-time model
download in download_models.py, and the optional Ollama LM rescoring step
which also runs locally if you use it).
"""

from __future__ import annotations
import tempfile
from pathlib import Path

import streamlit as st
import yaml

from utils.video_io import VideoReader, probe_video
from utils.face_tracking import LipTracker
from utils.roi_extraction import build_roi_sequence
from models.auto_avsr_wrapper import AutoAVSRTranscriber
from models.av_hubert_wrapper import AVHubertTranscriber
from models.lm_rescore import LMRescorer

CONFIG_PATH = Path(__file__).parent / "configs" / "default.yaml"


@st.cache_resource
def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_primary_model(_cfg):
    return AutoAVSRTranscriber(
        checkpoint_path=_cfg["models"]["primary"]["checkpoint"],
        config_path=_cfg["models"]["primary"]["config"],
        device=_cfg["device"],
    )


@st.cache_resource
def load_secondary_model(_cfg):
    if not _cfg["models"]["secondary_ensemble"]["enabled"]:
        return None
    return AVHubertTranscriber(
        checkpoint_path=_cfg["models"]["secondary_ensemble"]["checkpoint"],
        device=_cfg["device"],
    )


@st.cache_resource
def load_lm_rescorer(_cfg):
    lm_cfg = _cfg["language_model"]
    if not lm_cfg["enabled"]:
        return None
    return LMRescorer(
        backend=lm_cfg["backend"],
        lm_path=lm_cfg["lm_path"],
        llm_rescoring=lm_cfg.get("llm_rescoring"),
    )


def run_pipeline(video_path: str, cfg: dict, use_secondary: bool) -> dict:
    """Runs the full face-tracking -> ROI -> transcription -> LM rescoring pipeline."""
    results = {"primary_transcript": None, "secondary_transcript": None, "final_transcript": None}

    with VideoReader(
        video_path,
        target_fps=cfg["video"]["target_fps"],
        max_duration_seconds=cfg["video"]["max_duration_seconds"],
    ) as reader:
        frames = list(reader.frames())

    if not frames:
        raise ValueError("No frames could be read from this video.")

    with LipTracker(
        min_detection_confidence=cfg["face_tracking"]["min_detection_confidence"],
        min_tracking_confidence=cfg["face_tracking"]["min_tracking_confidence"],
    ) as tracker:
        frames_with_boxes = [(f, tracker.detect_mouth(f)) for f in frames]

    detected_count = sum(1 for _, box in frames_with_boxes if box is not None)
    if detected_count == 0:
        raise ValueError(
            "No face was detected in this video. Make sure the speaker's "
            "face and mouth are clearly visible and reasonably front-facing."
        )
    if detected_count < len(frames_with_boxes) * 0.5:
        st.warning(
            f"Face was only detected in {detected_count}/{len(frames_with_boxes)} frames. "
            f"Accuracy will likely be lower than usual — try a more front-facing angle "
            f"or better lighting."
        )

    roi_sequence = build_roi_sequence(
        frames_with_boxes,
        crop_size=tuple(cfg["roi"]["crop_size"]),
        grayscale=cfg["roi"]["grayscale"],
        normalize=cfg["roi"]["normalize"],
    )

    primary_model = load_primary_model(cfg)
    results["primary_transcript"] = primary_model.transcribe(roi_sequence)
    final_text = results["primary_transcript"]

    if use_secondary:
        secondary_model = load_secondary_model(cfg)
        if secondary_model is not None:
            results["secondary_transcript"] = secondary_model.transcribe(roi_sequence)
            # Simple ensembling: prefer the longer, more confident-looking
            # output; a proper ROVER/confusion-network merge would need
            # word-level confidence scores from both decoders, which is a
            # further improvement left for docs/ACCURACY_NOTES.md.
            if len(results["secondary_transcript"]) > len(final_text):
                final_text = results["secondary_transcript"]

    lm = load_lm_rescorer(cfg)
    if lm is not None:
        final_text = lm.rescore(final_text)

    results["final_transcript"] = final_text
    return results


def main():
    st.set_page_config(page_title="Visual Speech Recognition", page_icon="👄", layout="centered")
    st.title("👄 Visual Speech Recognition")
    st.caption("Upload a silent video. The mouth movement is transcribed to text — no audio is read.")

    cfg = load_config()

    with st.expander("⚠️ Accuracy expectations — please read", expanded=False):
        st.markdown(
            "This app combines open-source pretrained models "
            "([Auto-AVSR](https://github.com/mpc001/auto_avsr), optionally "
            "[AV-HuBERT](https://github.com/facebookresearch/av_hubert)) — "
            "the strongest realistic setup available today. Even so, the best "
            "published word error rate on clean lab video is **~20%**, meaning "
            "roughly 1 in 5 words can be wrong. Poor lighting, side angles, "
            "facial hair, or fast speech will make this worse. **English only.** "
            "See `docs/ACCURACY_NOTES.md` for details."
        )

    use_secondary = st.checkbox(
        "Also run AV-HuBERT and combine results (slower on CPU, may slightly improve accuracy)",
        value=cfg["models"]["secondary_ensemble"]["enabled"],
    )

    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.video(tmp_path)

        try:
            meta = probe_video(tmp_path)
            st.caption(
                f"{meta['width']}x{meta['height']} · {meta['duration_seconds']:.1f}s · "
                f"{meta['fps']:.0f} fps"
            )
        except Exception as e:
            st.error(f"Could not read this video file: {e}")
            return

        if st.button("Transcribe", type="primary"):
            with st.spinner("Running face tracking, ROI extraction, and transcription... "
                             "this can take a while on CPU."):
                try:
                    results = run_pipeline(tmp_path, cfg, use_secondary)
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    return

            st.subheader("Transcript")
            st.success(results["final_transcript"] or "(empty output)")

            with st.expander("Raw model output (before language model correction)"):
                st.text(f"Auto-AVSR: {results['primary_transcript']}")
                if results["secondary_transcript"]:
                    st.text(f"AV-HuBERT: {results['secondary_transcript']}")

            st.download_button(
                "Download transcript (.txt)",
                data=results["final_transcript"] or "",
                file_name=f"{Path(uploaded_file.name).stem}_transcript.txt",
            )


if __name__ == "__main__":
    main()
