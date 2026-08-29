"""
Wrapper around Meta AI's pretrained AV-HuBERT model (facebookresearch/av_hubert).

Used as an optional secondary/ensemble model — enable via
configs/default.yaml -> models.secondary_ensemble.enabled: true.

Running two independently-trained models and combining their outputs (e.g.
via ROVER-style voting or simple confidence comparison) tends to reduce
error rate slightly versus either model alone, at the cost of roughly 2x
inference time. On CPU this is meaningfully slower, so it's off by default.

Reference: Shi, B., Hsu, W.-N., Lakhotia, K., & Mohamed, A. (2022). Learning
Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction.
Visual-only WER on LRS3: ~28.6% (large model, fine-tuned on 433h LRS3).

Like auto_avsr_wrapper.py, this loads an existing pretrained checkpoint —
no training happens here.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np


class AVHubertTranscriber:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self._model = None
        self._task = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"AV-HuBERT checkpoint not found at {self.checkpoint_path}.\n"
                f"Run `python download_models.py --with-av-hubert` first."
            )

        third_party_path = Path(__file__).parent.parent / "third_party" / "av_hubert"
        if not third_party_path.exists():
            raise FileNotFoundError(
                f"AV-HuBERT source not found at {third_party_path}.\n"
                f"Run `python download_models.py --with-av-hubert` — it clones "
                f"https://github.com/facebookresearch/av_hubert into third_party/."
            )
        sys.path.insert(0, str(third_party_path))
        sys.path.insert(0, str(third_party_path / "fairseq"))

        import fairseq
        from fairseq import checkpoint_utils

        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [str(self.checkpoint_path)]
        )
        self._model = models[0]
        self._model.eval()
        self._task = task
        self._loaded = True

    def transcribe(self, roi_sequence: np.ndarray) -> str:
        """
        roi_sequence: (T, H, W) float32 array of preprocessed mouth crops.
        Returns the raw decoded transcript string, visual-modality only
        (no audio stream is used anywhere in this project).
        """
        self._ensure_loaded()
        import torch

        with torch.no_grad():
            video_tensor = torch.from_numpy(roi_sequence).unsqueeze(0).unsqueeze(0)
            video_tensor = video_tensor.to(self.device)

            sample = {
                "net_input": {
                    "source": {"video": video_tensor, "audio": None},
                    "padding_mask": None,
                }
            }
            generator = self._task.build_generator([self._model], cfg={"beam": 5})
            hypos = generator.generate([self._model], sample)
            transcript = self._task.target_dictionary.string(hypos[0][0]["tokens"])

        return transcript.strip()
