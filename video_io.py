"""
Video I/O utilities.

Handles reading an uploaded video file, resampling to the target frame rate
the pretrained models expect (25 fps), and yielding frames as numpy arrays.
Uses OpenCV — no backend/server involved, this all runs in-process.
"""

from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Iterator


class VideoReader:
    """Reads a video file and yields frames resampled to a target FPS."""

    def __init__(self, video_path: str | Path, target_fps: int = 25, max_duration_seconds: int = 60):
        self.video_path = str(video_path)
        self.target_fps = target_fps
        self.max_duration_seconds = max_duration_seconds

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(
                f"Could not open video file: {self.video_path}. "
                f"Check that ffmpeg is installed and the file is a valid video."
            )

        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_seconds = self.frame_count / self.source_fps if self.source_fps else 0

        if self.duration_seconds > self.max_duration_seconds:
            raise ValueError(
                f"Video is {self.duration_seconds:.1f}s long, which exceeds the "
                f"configured max of {self.max_duration_seconds}s. Trim the video "
                f"or raise `max_duration_seconds` in configs/default.yaml — note "
                f"longer videos take proportionally longer on CPU."
            )

    def frames(self) -> Iterator[np.ndarray]:
        """Yield BGR frames resampled to target_fps via simple frame-index sampling."""
        if self.source_fps <= 0:
            step = 1.0
        else:
            step = self.source_fps / self.target_fps

        next_frame_pos = 0.0
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_idx >= next_frame_pos:
                yield frame
                next_frame_pos += step
            frame_idx += 1

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def probe_video(video_path: str | Path) -> dict:
    """Quick metadata probe, used by the Streamlit UI before running the full pipeline."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps else 0
    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration,
    }
