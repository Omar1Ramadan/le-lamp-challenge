from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from social_lamp.capture.frames import CapturedFrame
from social_lamp.perception.engagement import EngagementSignals

MAX_FRAME_AGE_NS = 300_000_000


@dataclass(frozen=True)
class FaceResult:
    face_confidence: float
    yaw_degrees: float
    pitch_degrees: float
    gaze_score: float
    gaze_quality: float
    face_area_ratio: float


class FaceLandmarker(Protocol):
    def detect(self, image: NDArray[np.uint8]) -> list[FaceResult]: ...


def _clamp(value: float, *, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def face_result_to_signals(
    *,
    face_confidence: float,
    yaw_degrees: float,
    pitch_degrees: float,
    gaze_score: float,
    gaze_quality: float,
    face_area_ratio: float,
) -> EngagementSignals:
    head = _clamp(1.0 - abs(yaw_degrees) / 45.0 - abs(pitch_degrees) / 40.0)
    proximity = _clamp(face_area_ratio / 0.15)
    gaze = _clamp(gaze_score) if gaze_quality >= 0.45 else None
    return EngagementSignals(
        face_presence=_clamp(face_confidence),
        head_toward=head,
        gaze_toward=gaze,
        proximity=proximity,
        directed_speech=0.0,
        confidence=min(_clamp(face_confidence), max(_clamp(gaze_quality), 0.45)),
    )


class MediaPipeFaceAdapter:
    def __init__(self, *, landmarker: FaceLandmarker) -> None:
        self._landmarker = landmarker

    def process(
        self, frame: CapturedFrame, *, now_mono_ns: int
    ) -> tuple[FaceResult, ...]:
        if now_mono_ns - frame.mono_ns > MAX_FRAME_AGE_NS:
            return ()
        return tuple(self._landmarker.detect(frame.image))
