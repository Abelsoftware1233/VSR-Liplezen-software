# Visual Speech Recognition (Lip Reading) — Local Streamlit App

Upload a silent (or muted) video and get a text transcription generated purely
from lip movement, using existing open-source pretrained models — **no
training required**.

This app runs **100% locally**. No backend server, no cloud API calls for
inference. Everything happens on your own machine inside one Streamlit
process.

---

## ⚠️ Read this before you expect "perfect" translation

Lip reading from video alone is a fundamentally ambiguous problem — many
words look identical on the lips (e.g. "pack", "back", "mac"). This is true
for humans and for AI. There is currently **no model, open-source or
commercial, that transcribes lip movement perfectly.**

Realistic numbers from the best published open-source systems, on **clean,
front-facing, well-lit English video**:

| Model | Word Error Rate (WER) on LRS3 benchmark |
|---|---|
| Auto-AVSR (visual only) | ~19.5–20.3% |
| AV-HuBERT Large (visual only) | ~28.6% |
| SLAM-LLM (LLM-decoder VSR) | ~28.3% |

A WER of ~20% means roughly **1 in 5 words will be wrong** even under ideal
conditions. Poor lighting, side angles, fast speech, facial hair, or
non-English speech will make it worse. This app stacks several techniques
(described below) to push accuracy as high as realistically possible, but it
will never be perfect, and you should not rely on it for anything
safety-critical, legal, or medical.

**Language:** all bundled pretrained models are trained on English (LRS3 /
LRS2 datasets, BBC/TED footage). Non-English video will not transcribe
correctly.

---

## How it works (pipeline)

```
Video upload
    │
    ▼
1. Face & lip detection ──────────── MediaPipe Face Mesh
    │
    ▼
2. Mouth ROI extraction ──────────── OpenCV (crop, grayscale, normalize)
    │
    ▼
3. Visual feature encoding ────────── Auto-AVSR ResNet-18 + Conformer front-end
    │
    ▼
4. Sequence-to-text decoding ──────── Auto-AVSR Conformer encoder/decoder (CTC + attention)
    │
    ▼
5. Language model rescoring ───────── optional n-gram / transformer LM rescoring
    │
    ▼
Final transcript (displayed + downloadable)
```

## Libraries and pretrained models bundled

This project deliberately wires together **every actively-maintained
open-source VSR component** currently available, rather than picking just
one:

- **[Auto-AVSR](https://github.com/mpc001/auto_avsr)** (Ma et al., 2023) —
  primary transcription model. Best published open-source WER (~20% on
  LRS3). Provides pretrained checkpoints, no training needed.
- **[AV-HuBERT](https://github.com/facebookresearch/av_hubert)** (Meta AI,
  Shi et al., 2022) — used as a secondary/ensemble model. Self-supervised,
  pretrained on VoxCeleb2 + LRS3.
- **[Chaplin](https://github.com/amanvirparhar/chaplin)** — reference
  implementation showing how Auto-AVSR weights integrate with MediaPipe for
  real-time lip cropping; this repo adapts the same weight format for
  file-based (uploaded video) inference instead of live webcam.
- **[MediaPipe Face Mesh](https://github.com/google/mediapipe)** (Google) —
  face and lip landmark detection / ROI tracking.
- **OpenCV** — video I/O, frame extraction, ROI cropping and normalization.
- **PyTorch / torchaudio / torchvision** — model runtime.
- **espnet-style Conformer + CTC/attention decoder** (bundled inside
  Auto-AVSR) — sequence modeling of lip motion over time.
- **Language model rescoring** — an optional pass using a subword n-gram LM
  (trained on LRS3 transcripts) or an external LLM (e.g. a local Ollama
  model) to smooth ungrammatical or ambiguous output, similar to what
  Chaplin does with `qwen3` locally.

You do **not** need to train anything. Setup only downloads existing
pretrained checkpoints from Hugging Face Hub / the official model releases.

---

## Requirements

- Python 3.10 or 3.12
- ~6–8 GB free disk space (pretrained weights + dependencies)
- CPU is enough to run this (no GPU required). A GPU (NVIDIA, with CUDA)
  will make it noticeably faster, but everything here defaults to CPU mode.
- ffmpeg installed and on your PATH (for video/frame decoding)

## Setup

```bash
# 1. Clone this repo
git clone <your-repo-url>
cd vsr-repo

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download pretrained model weights (one-time, ~2-3 GB)
python download_models.py

# 5. Launch the app
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`. Upload a
video file (mp4, mov, avi) with a clearly visible, front-facing mouth, and
click **Transcribe**.

## Project structure

```
vsr-repo/
├── app.py                  # Streamlit UI — upload, run pipeline, show result
├── download_models.py      # One-time script to fetch pretrained checkpoints
├── requirements.txt
├── models/
│   ├── auto_avsr_wrapper.py   # Loads Auto-AVSR checkpoint, runs inference
│   ├── av_hubert_wrapper.py   # Optional secondary model for ensembling
│   └── lm_rescore.py          # Language-model correction pass
├── utils/
│   ├── face_tracking.py    # MediaPipe-based face/lip detection
│   ├── roi_extraction.py   # Crop, grayscale, normalize mouth ROI per frame
│   └── video_io.py         # Frame extraction / video handling (OpenCV)
├── configs/
│   └── default.yaml        # Model paths, thresholds, frame rate settings
├── model_weights/           # Downloaded checkpoints land here (gitignored)
└── docs/
    └── ACCURACY_NOTES.md    # Honest breakdown of when/why this will fail
```

## Known limitations (please read)

- English only.
- Struggles with: side profile angles, poor lighting, facial hair covering
  the mouth, fast/mumbled speech, multiple speakers in frame, low
  resolution video.
- Sentence-level context helps but does not eliminate ambiguity — visually
  identical word groups ("viseme clashes") remain a hard limit of the
  underlying signal, not a bug in this code.
- This is a research-grade pipeline assembled from academic
  models, not a polished commercial product. Expect to tune thresholds in
  `configs/default.yaml` for your specific camera setup.

## License note

Auto-AVSR and AV-HuBERT pretrained weights carry their own licenses derived
from the LRS3 dataset (TED terms of use, CC BY-NC-ND 4.0) — **non-commercial
use only** unless you retrain on your own licensed data. Check
`docs/LICENSES.md` before any commercial use.
