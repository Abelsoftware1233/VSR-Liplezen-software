"""
Face & lip tracking using Google's MediaPipe Face Mesh.

This is the "Step 1" component from the pipeline description: find the face
in each frame and locate the mouth region precisely enough to crop it.

MediaPipe is used instead of dlib because it needs no separate landmark
model download, runs faster on CPU, and is what the reference Chaplin
implementation (github.com/amanvirparhar/chaplin) uses for the same task.
A dlib fallback path is stubbed below in case MediaPipe fails to install on
a given machine.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False


# MediaPipe Face Mesh landmark indices that outline the outer + inner lips.
# These indices are fixed by the MediaPipe face mesh topology (468 points).
_MOUTH_LANDMARK_IDS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
    95, 185, 40, 39, 37, 0, 267, 269, 270, 409,
]


@dataclass
class MouthBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def expanded(self, margin_ratio: float = 0.35) -> "MouthBox":
        """Pad the tight landmark box so the crop includes some context around the lips."""
        w = self.x_max - self.x_min
        h = self.y_max - self.y_min
        mx = int(w * margin_ratio)
        my = int(h * margin_ratio)
        return MouthBox(
            x_min=max(0, self.x_min - mx),
            y_min=max(0, self.y_min - my),
            x_max=self.x_max + mx,
            y_max=self.y_max + my,
        )


class LipTracker:
    """Wraps MediaPipe Face Mesh to return a mouth bounding box per frame."""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        if not _MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "mediapipe is not installed. Run `pip install -r requirements.txt` first."
            )
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect_mouth(self, frame_bgr: np.ndarray) -> MouthBox | None:
        """Return a MouthBox for the given BGR frame, or None if no face was found."""
        import cv2

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        xs = [landmarks[i].x * w for i in _MOUTH_LANDMARK_IDS]
        ys = [landmarks[i].y * h for i in _MOUTH_LANDMARK_IDS]

        box = MouthBox(
            x_min=int(min(xs)), y_min=int(min(ys)),
            x_max=int(max(xs)), y_max=int(max(ys)),
        )
        return box.expanded()

    def close(self):
        self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
