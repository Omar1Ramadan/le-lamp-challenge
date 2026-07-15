from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from social_lamp.capture.frames import CapturedFrame
from social_lamp.domain.contracts import ComponentHealth
from social_lamp.perception.faces import FaceResult
from social_lamp.perception.location import BBox
from social_lamp.perception.objects import Detection


def make_face(
    *,
    confidence: float = 0.92,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    gaze_score: float = 0.8,
    gaze_quality: float = 0.9,
    face_area_ratio: float = 0.12,
    pose_source: str = "mediapipe_matrix",
    pose_quality: float = 0.95,
) -> FaceResult:
    return FaceResult(
        face_confidence=confidence,
        yaw_degrees=yaw,
        pitch_degrees=pitch,
        gaze_score=gaze_score,
        gaze_quality=gaze_quality,
        face_area_ratio=face_area_ratio,
        pose_source=pose_source,
        roll_degrees=roll,
        pose_quality=pose_quality,
    )


def make_detection(
    *,
    label: str = "object",
    confidence: float = 0.9,
    bbox: BBox | None = None,
    mono_ns: int = 0,
) -> Detection:
    return Detection(
        label=label,
        confidence=confidence,
        bbox=bbox or (0.35, 0.35, 0.55, 0.65),
        mono_ns=mono_ns,
    )


def make_frame(*, mono_ns: int = 0, shape: tuple[int, int, int] = (4, 4, 3)) -> CapturedFrame:
    return CapturedFrame(np.zeros(shape, dtype=np.uint8), mono_ns=mono_ns)


class SequenceFaceProcessor:
    def __init__(self, sequence: Sequence[tuple[FaceResult, ...]]) -> None:
        self._sequence = list(sequence)
        self._index = 0
        self.call_count = 0

    def process(self, frame: CapturedFrame, *, now_mono_ns: int) -> tuple[FaceResult, ...]:
        self.call_count += 1
        del frame, now_mono_ns
        if self._index < len(self._sequence):
            result = self._sequence[self._index]
            self._index += 1
            return result
        return self._sequence[-1] if self._sequence else ()


class FailingFaceProcessor:
    def __init__(self, fail_on_call: int = 1) -> None:
        self._fail_on_call = fail_on_call
        self.call_count = 0

    def process(self, frame: CapturedFrame, *, now_mono_ns: int) -> tuple[FaceResult, ...]:
        self.call_count += 1
        del frame, now_mono_ns
        if self.call_count == self._fail_on_call:
            raise RuntimeError("face model crashed")
        return (make_face(),)


class SequenceObjectDetector:
    def __init__(self, sequence: Sequence[tuple[Detection, ...]]) -> None:
        self._sequence = list(sequence)
        self._index = 0
        self.call_count = 0

    def detect(self, image: NDArray[np.uint8]) -> tuple[Detection, ...]:
        self.call_count += 1
        del image
        if self._index < len(self._sequence):
            result = self._sequence[self._index]
            self._index += 1
            return result
        return self._sequence[-1] if self._sequence else ()

    def health(self) -> ComponentHealth:
        return ComponentHealth(component="object_detector", status="active")


class TimedObjectDetector:
    def __init__(self, detections: tuple[Detection, ...]) -> None:
        self.detections = detections
        self.call_count = 0

    def detect(self, image: NDArray[np.uint8]) -> tuple[Detection, ...]:
        self.call_count += 1
        del image
        return self.detections

    def health(self) -> ComponentHealth:
        return ComponentHealth(component="object_detector", status="active")
