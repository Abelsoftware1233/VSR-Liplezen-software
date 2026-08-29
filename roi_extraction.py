"""
Region of Interest (ROI) extraction.

Step 2 of the pipeline: given a mouth bounding box per frame, crop it,
resize to a fixed size, convert to grayscale, and normalize — matching the
preprocessing Auto-AVSR and AV-HuBERT were trained on.
"""

from __future__ import annotations
import cv2
import numpy as np

from utils.face_tracking import MouthBox


def extract_roi(
    frame_bgr: np.ndarray,
    box: MouthBox,
    crop_size: tuple[int, int] = (96, 96),
    grayscale: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Crop the mouth region from a frame and preprocess it for the model."""
    h, w = frame_bgr.shape[:2]
    x_min = max(0, box.x_min)
    y_min = max(0, box.y_min)
    x_max = min(w, box.x_max)
    y_max = min(h, box.y_max)

    if x_max <= x_min or y_max <= y_min:
        # Degenerate box (tracking glitch) — return a blank frame rather than crash.
        crop = np.zeros((crop_size[1], crop_size[0], 3), dtype=np.uint8)
    else:
        crop = frame_bgr[y_min:y_max, x_min:x_max]
        crop = cv2.resize(crop, crop_size, interpolation=cv2.INTER_AREA)

    if grayscale:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    crop = crop.astype(np.float32)

    if normalize:
        # Per-frame zero-mean, unit-variance normalization (standard for Auto-AVSR input)
        mean = crop.mean()
        std = crop.std()
        if std > 1e-6:
            crop = (crop - mean) / std
        else:
            crop = crop - mean

    return crop


def build_roi_sequence(
    frames_with_boxes: list[tuple[np.ndarray, MouthBox | None]],
    crop_size: tuple[int, int] = (96, 96),
    grayscale: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Build a (T, H, W) or (T, H, W, C) numpy array of mouth ROIs across a video.

    Frames where no face/mouth was detected reuse the previous valid box so
    the sequence doesn't have gaps — a brief tracking dropout (blink,
    momentary head turn) shouldn't break the whole clip.
    """
    rois = []
    last_box: MouthBox | None = None

    for frame, box in frames_with_boxes:
        active_box = box if box is not None else last_box
        if active_box is None:
            # No face ever found yet — skip until we get a first detection.
            continue
        last_box = active_box
        roi = extract_roi(frame, active_box, crop_size, grayscale, normalize)
        rois.append(roi)

    if not rois:
        raise ValueError(
            "No face/mouth was detected in any frame of this video. "
            "Make sure the speaker's face is visible and reasonably front-facing."
        )

    return np.stack(rois, axis=0)
