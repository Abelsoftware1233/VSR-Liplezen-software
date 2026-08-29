"""
Wrapper around the pretrained Auto-AVSR model (mpc001/auto_avsr).

This is the primary transcription model in this pipeline. It is NOT trained
here — it loads an existing checkpoint downloaded by download_models.py and
runs inference only.

Reference: Ma, P., Haliassos, A., Fernandez-Lopez, A., Chen, H., Petridis,
S., & Pantic, M. (2023). Auto-AVSR: Audio-Visual Speech Recognition with
Automatic Labels. ICASSP 2023.

Best published visual-only WER: 19.5-20.3% on LRS3.

NOTE: This wrapper expects the actual auto_avsr package (cloned from
https://github.com/mpc001/auto_avsr) to be importable — see
download_models.py, which clones it into third_party/auto_avsr and adds it
to sys.path. This file defines the interface this project uses to call
into that code; it deliberately does not vendor/copy Auto-AVSR's model code
itself, since that lives in their repository and should be pulled from the
source of truth (and kept up to date with their license terms).
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np


class AutoAVSRTranscriber:
    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cpu"):
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.device = device
        self._model = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Auto-AVSR checkpoint not found at {self.checkpoint_path}.\n"
                f"Run `python download_models.py` first to fetch pretrained weights."
            )

        third_party_path = Path(__file__).parent.parent / "third_party" / "auto_avsr"
        if not third_party_path.exists():
            raise FileNotFoundError(
                f"Auto-AVSR source not found at {third_party_path}.\n"
                f"Run `python download_models.py` first — it clones "
                f"https://github.com/mpc001/auto_avsr into third_party/."
            )
        sys.path.insert(0, str(third_party_path))

        # Import happens here (not at module top-level) because the auto_avsr
        # package only exists on disk after download_models.py has run.
        from lightning import ModelModule  # provided by mpc001/auto_avsr
        import torch

        self._model = ModelModule.load_from_checkpoint(
            str(self.checkpoint_path),
            map_location=torch.device(self.device),
        )
        self._model.eval()
        self._model.to(self.device)
        self._loaded = True

    def transcribe(self, roi_sequence: np.ndarray) -> str:
        """
        roi_sequence: (T, H, W) float32 array of preprocessed mouth crops,
        as produced by utils.roi_extraction.build_roi_sequence.

        Returns the raw decoded transcript string (before any LM rescoring).
        """
        self._ensure_loaded()
        import torch

        with torch.no_grad():
            video_tensor = torch.from_numpy(roi_sequence).unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)
            video_tensor = video_tensor.to(self.device)
            output = self._model(video_tensor)
            transcript = self._model.decode(output)

        return transcript.strip()
