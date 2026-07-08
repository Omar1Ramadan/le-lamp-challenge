from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from time import monotonic_ns
from typing import Any

import numpy as np
from numpy.typing import NDArray

from social_lamp.domain.contracts import ComponentHealth


@dataclass(frozen=True)
class CapturedFrame:
    image: NDArray[np.uint8]
    mono_ns: int


class LatestFrameBuffer:
    def __init__(self, *, capacity: int = 3, max_age_ns: int = 300_000_000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._frames: deque[CapturedFrame] = deque(maxlen=capacity)
        self._lock = Lock()
        self.dropped = 0
        self.max_age_ns = max_age_ns

    def put(self, image: NDArray[np.uint8], mono_ns: int) -> None:
        with self._lock:
            if len(self._frames) == self._frames.maxlen:
                self.dropped += 1
            self._frames.append(CapturedFrame(image.copy(), mono_ns))

    def latest(self, *, now_mono_ns: int | None = None) -> CapturedFrame | None:
        with self._lock:
            if not self._frames:
                return None
            frame = self._frames[-1]
            if now_mono_ns is not None and now_mono_ns - frame.mono_ns > self.max_age_ns:
                return None
            return frame

    def health(self, *, now_mono_ns: int) -> ComponentHealth:
        with self._lock:
            if not self._frames:
                return ComponentHealth(component="camera", status="degraded", detail="no frames")
            if now_mono_ns - self._frames[-1].mono_ns > self.max_age_ns:
                return ComponentHealth(component="camera", status="degraded", detail="stale frame")
            return ComponentHealth(component="camera", status="ok")


def _probe_camera() -> int:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - environment-specific
        print(f"camera_unavailable import_error={exc.__class__.__name__}")
        return 0

    capture = cv2.VideoCapture(0)
    try:
        if not capture.isOpened():
            print("camera_unavailable")
            return 0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        print(f"camera_available width={width} height={height} fps={fps:.2f}")
        return 0
    finally:
        capture.release()


if __name__ == "__main__":
    raise SystemExit(_probe_camera())


class OpenCVCameraSource:
    def __init__(self, *, camera_index: int = 0) -> None:
        self._camera_index = camera_index
        self._capture: Any | None = None
        self.health_detail = "camera not started"

    def open(self) -> bool:
        try:
            import cv2
        except Exception as exc:
            self.health_detail = f"camera unavailable: {exc.__class__.__name__}"
            return False

        capture = cv2.VideoCapture(self._camera_index)
        if not capture.isOpened():
            capture.release()
            self.health_detail = "camera unavailable"
            return False

        self._capture = capture
        self.health_detail = "camera available"
        return True

    def read(self) -> CapturedFrame | None:
        capture = self._capture
        if capture is None:
            self.health_detail = "camera unavailable"
            return None
        ok, frame = capture.read()
        if not ok or frame is None:
            self.health_detail = "camera frame unavailable"
            return None
        self.health_detail = "camera available"
        return CapturedFrame(np.asarray(frame, dtype=np.uint8), mono_ns=monotonic_ns())

    def close(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()
        self.health_detail = "camera stopped"
