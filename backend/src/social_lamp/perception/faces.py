from dataclasses import dataclass
from pathlib import Path
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


def _opencv_attention_proxy(
    *, horizontal_offset: float, vertical_offset: float, eye_count: int
) -> float:
    centeredness = _clamp(1.0 - horizontal_offset / 0.35 - vertical_offset / 0.45)
    eye_bonus = 0.2 if eye_count >= 2 else 0.0
    return _clamp(0.15 + 0.75 * centeredness + eye_bonus)


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

    def process(self, frame: CapturedFrame, *, now_mono_ns: int) -> tuple[FaceResult, ...]:
        if now_mono_ns - frame.mono_ns > MAX_FRAME_AGE_NS:
            return ()
        return tuple(self._landmarker.detect(frame.image))


class OpenCvFaceProcessor:
    def __init__(self) -> None:
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError(f"face model unavailable: {exc.__class__.__name__}") from exc

        cascade_path = Path(str(cv2.data.haarcascades)) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            raise RuntimeError(f"face model unavailable: missing cascade {cascade_path}")
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            raise RuntimeError("face model unavailable: cascade load failed")
        eye_path = Path(str(cv2.data.haarcascades)) / "haarcascade_eye_tree_eyeglasses.xml"
        eye_cascade = cv2.CascadeClassifier(str(eye_path)) if eye_path.exists() else None
        self._cv2 = cv2
        self._cascade = cascade
        self._eye_cascade = (
            eye_cascade if eye_cascade is not None and not eye_cascade.empty() else None
        )
        self.last_debug: dict[str, object] | None = None

    def process(self, frame: CapturedFrame, *, now_mono_ns: int) -> tuple[FaceResult, ...]:
        self.last_debug = None
        if now_mono_ns - frame.mono_ns > MAX_FRAME_AGE_NS:
            return ()

        grayscale = self._cv2.cvtColor(frame.image, self._cv2.COLOR_BGR2GRAY)
        detections = self._cascade.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=5)
        height, width = grayscale.shape
        frame_area = max(width * height, 1)
        sorted_detections = sorted(
            detections, key=lambda item: int(item[2]) * int(item[3]), reverse=True
        )

        results: list[FaceResult] = []
        for index, (x, y, w, h) in enumerate(sorted_detections):
            center_x = (float(x) + float(w) / 2.0) / float(width)
            center_y = (float(y) + float(h) / 2.0) / float(height)
            horizontal_offset = abs(center_x - 0.5)
            vertical_offset = abs(center_y - 0.45)
            yaw_degrees = min(45.0, horizontal_offset * 120.0)
            pitch_degrees = min(40.0, vertical_offset * 100.0)
            eyes = self._detect_eyes(grayscale, int(x), int(y), int(w), int(h))
            attention_proxy = _opencv_attention_proxy(
                horizontal_offset=horizontal_offset,
                vertical_offset=vertical_offset,
                eye_count=len(eyes),
            )
            gaze_quality = 0.75 if len(eyes) >= 2 else 0.45
            face_area_ratio = (w * h) / frame_area
            if index == 0:
                self.last_debug = {
                    "attention_proxy": round(attention_proxy, 3),
                    "box": {
                        "x": float(x) / float(width),
                        "y": float(y) / float(height),
                        "width": float(w) / float(width),
                        "height": float(h) / float(height),
                    },
                    "eye_count": len(eyes),
                    "target": {"x": center_x, "y": center_y},
                    "yaw_degrees": round(yaw_degrees, 1),
                    "pitch_degrees": round(pitch_degrees, 1),
                }
            results.append(
                FaceResult(
                    face_confidence=0.75,
                    yaw_degrees=yaw_degrees,
                    pitch_degrees=pitch_degrees,
                    gaze_score=attention_proxy,
                    gaze_quality=gaze_quality,
                    face_area_ratio=face_area_ratio,
                )
            )
        return tuple(results)

    def _detect_eyes(
        self, grayscale: NDArray[np.uint8], x: int, y: int, w: int, h: int
    ) -> tuple[tuple[int, int, int, int], ...]:
        if self._eye_cascade is None:
            return ()
        upper_face = grayscale[y : y + max(h // 2, 1), x : x + w]
        if upper_face.size == 0:
            return ()
        detections = self._eye_cascade.detectMultiScale(
            upper_face, scaleFactor=1.1, minNeighbors=4
        )
        return tuple((int(ex), int(ey), int(ew), int(eh)) for ex, ey, ew, eh in detections)


class HeuristicFaceProcessor:
    detail = "face model unavailable; using browser-frame heuristic"

    def __init__(self) -> None:
        self.last_debug: dict[str, object] | None = None

    def process(self, frame: CapturedFrame, *, now_mono_ns: int) -> tuple[FaceResult, ...]:
        self.last_debug = None
        if now_mono_ns - frame.mono_ns > MAX_FRAME_AGE_NS:
            return ()

        image = frame.image
        if image.size == 0:
            return ()

        height, width = image.shape[:2]
        if height < 24 or width < 24:
            return ()

        center_crop = image[height // 5 : height * 4 // 5, width // 5 : width * 4 // 5]
        if (
            center_crop.size == 0
            or float(np.std(center_crop)) < 8.0
            or float(np.mean(center_crop)) < 20.0
        ):
            return ()

        b = image[:, :, 0].astype(np.int16)
        g = image[:, :, 1].astype(np.int16)
        r = image[:, :, 2].astype(np.int16)
        skin = (r > 75) & (g > 45) & (b > 25) & (r > g) & (r > b) & ((r - b) > 20)
        y_indices, x_indices = np.nonzero(skin)
        if y_indices.size == 0 or x_indices.size == 0:
            return ()

        x_min = int(x_indices.min())
        x_max = int(x_indices.max())
        y_min = int(y_indices.min())
        y_max = int(y_indices.max())
        face_width = max(x_max - x_min + 1, 1)
        face_height = max(y_max - y_min + 1, 1)
        face_area_ratio = (face_width * face_height) / float(width * height)
        skin_coverage = float(y_indices.size) / float(skin.size)
        if skin_coverage < 0.04 or face_area_ratio < 0.03:
            return ()

        center_x = (x_min + x_max) / 2.0 / float(width)
        center_y = (y_min + y_max) / 2.0 / float(height)
        horizontal_offset = abs(center_x - 0.5)
        vertical_offset = abs(center_y - 0.45)
        aspect_ratio = face_width / float(face_height)
        centeredness = max(0.0, 1.0 - horizontal_offset / 0.35 - vertical_offset / 0.45)
        frontal_shape = max(0.0, 1.0 - abs(aspect_ratio - 0.78) / 0.55)
        attention_proxy = min(centeredness, frontal_shape)
        yaw_degrees = min(45.0, horizontal_offset * 120.0)
        pitch_degrees = min(40.0, vertical_offset * 100.0)
        self.last_debug = {
            "attention_proxy": round(attention_proxy, 3),
            "box": {
                "x": x_min / float(width),
                "y": y_min / float(height),
                "width": face_width / float(width),
                "height": face_height / float(height),
            },
            "target": {"x": center_x, "y": center_y},
            "yaw_degrees": round(yaw_degrees, 1),
            "pitch_degrees": round(pitch_degrees, 1),
        }

        return (
            FaceResult(
                face_confidence=0.55,
                yaw_degrees=yaw_degrees,
                pitch_degrees=pitch_degrees,
                gaze_score=attention_proxy,
                gaze_quality=0.45,
                face_area_ratio=min(face_area_ratio, 0.15),
            ),
        )
