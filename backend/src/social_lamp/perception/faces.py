from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from social_lamp.capture.frames import CapturedFrame
from social_lamp.perception.engagement import EngagementSignals

MAX_FRAME_AGE_NS = 300_000_000
DEFAULT_FACE_LANDMARKER_MODEL = (
    Path(__file__).resolve().parents[3] / "assets" / "mediapipe" / "face_landmarker.task"
)


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
    eye_bonus = 0.2 if eye_count >= 2 else -0.25
    return _clamp(0.15 + 0.75 * centeredness + eye_bonus)


def _opencv_eye_attention(
    eyes: tuple[tuple[int, int, int, int], ...], *, face_width: int, face_height: int
) -> tuple[float, tuple[tuple[float, float, float, float], ...]]:
    if not eyes or face_width <= 0 or face_height <= 0:
        return 0.2, ()

    normalized = tuple(
        (
            float(ex) / float(face_width),
            float(ey) / float(face_height),
            float(ew) / float(face_width),
            float(eh) / float(face_height),
        )
        for ex, ey, ew, eh in eyes
    )
    eye_centers = sorted(
        (
            (x + width / 2.0, y + height / 2.0)
            for x, y, width, height in normalized
            if 0.12 <= y + height / 2.0 <= 0.58
        ),
        key=lambda center: center[0],
    )
    if len(eye_centers) < 2:
        return 0.42, normalized

    left, right = eye_centers[0], eye_centers[-1]
    separation = right[0] - left[0]
    average_y = (left[1] + right[1]) / 2.0
    vertical_delta = abs(left[1] - right[1])
    separation_score = _clamp(1.0 - abs(separation - 0.36) / 0.28)
    height_score = _clamp(1.0 - abs(average_y - 0.32) / 0.22)
    level_score = _clamp(1.0 - vertical_delta / 0.16)
    attention = _clamp(
        0.15 + 0.35 * separation_score + 0.3 * height_score + 0.2 * level_score
    )
    return attention, normalized


def _blendshape_gaze(blendshapes: object) -> tuple[float, float, dict[str, object]]:
    scores: dict[str, float] = {}
    for category in blendshapes or ():
        name = getattr(category, "category_name", "")
        score = getattr(category, "score", 0.0)
        if isinstance(name, str) and isinstance(score, float | int):
            scores[name] = float(score)

    look_names = (
        "eyeLookDownLeft",
        "eyeLookDownRight",
        "eyeLookInLeft",
        "eyeLookInRight",
        "eyeLookOutLeft",
        "eyeLookOutRight",
        "eyeLookUpLeft",
        "eyeLookUpRight",
    )
    blink_names = ("eyeBlinkLeft", "eyeBlinkRight")
    look_away = max((scores.get(name, 0.0) for name in look_names), default=0.0)
    blink = max((scores.get(name, 0.0) for name in blink_names), default=0.0)
    if not scores:
        return 0.5, 0.55, {"eye_look_away": None, "eye_blink": None}
    gaze_score = _clamp(1.0 - 1.45 * look_away - 0.75 * blink)
    return (
        gaze_score,
        0.9,
        {
            "eye_blink": round(blink, 3),
            "eye_look_away": round(look_away, 3),
        },
    )


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


class MediaPipeFaceLandmarkerProcessor:
    def __init__(self, *, model_path: Path = DEFAULT_FACE_LANDMARKER_MODEL) -> None:
        if not model_path.exists():
            raise RuntimeError(f"face landmarker unavailable: missing model {model_path}")
        try:
            import mediapipe as mp
            from mediapipe.tasks.python.core import base_options
            from mediapipe.tasks.python.vision import face_landmarker
        except Exception as exc:
            raise RuntimeError(f"face landmarker unavailable: {exc.__class__.__name__}") from exc

        options = face_landmarker.FaceLandmarkerOptions(
            base_options=base_options.BaseOptions(
                model_asset_path=str(model_path),
                delegate=base_options.BaseOptions.Delegate.CPU,
            ),
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._mp = mp
        self._landmarker = face_landmarker.FaceLandmarker.create_from_options(options)
        self.last_debug: dict[str, object] | None = None

    def process(self, frame: CapturedFrame, *, now_mono_ns: int) -> tuple[FaceResult, ...]:
        self.last_debug = None
        if now_mono_ns - frame.mono_ns > MAX_FRAME_AGE_NS:
            return ()

        rgb = frame.image[:, :, ::-1].copy()
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return ()

        landmarks = result.face_landmarks[0]
        xs = [float(landmark.x) for landmark in landmarks]
        ys = [float(landmark.y) for landmark in landmarks]
        x_min = _clamp(min(xs))
        x_max = _clamp(max(xs))
        y_min = _clamp(min(ys))
        y_max = _clamp(max(ys))
        width = max(x_max - x_min, 0.001)
        height = max(y_max - y_min, 0.001)
        center_x = x_min + width / 2.0
        center_y = y_min + height / 2.0
        horizontal_offset = abs(center_x - 0.5)
        vertical_offset = abs(center_y - 0.45)
        yaw_degrees = min(45.0, horizontal_offset * 120.0)
        pitch_degrees = min(40.0, vertical_offset * 100.0)

        blendshapes = result.face_blendshapes[0] if result.face_blendshapes else ()
        gaze_score, gaze_quality, gaze_detail = _blendshape_gaze(blendshapes)
        if gaze_quality < 0.85:
            gaze_score = min(
                gaze_score,
                _opencv_attention_proxy(
                    horizontal_offset=horizontal_offset,
                    vertical_offset=vertical_offset,
                    eye_count=2,
                ),
            )

        self.last_debug = {
            "attention_proxy": round(gaze_score, 3),
            "box": {"x": x_min, "y": y_min, "width": width, "height": height},
            "gaze_source": "mediapipe_blendshapes",
            "landmark_count": len(landmarks),
            "target": {"x": center_x, "y": center_y},
            "yaw_degrees": round(yaw_degrees, 1),
            "pitch_degrees": round(pitch_degrees, 1),
            **gaze_detail,
        }
        return (
            FaceResult(
                face_confidence=0.92,
                yaw_degrees=yaw_degrees,
                pitch_degrees=pitch_degrees,
                gaze_score=gaze_score,
                gaze_quality=gaze_quality,
                face_area_ratio=min(width * height, 0.15),
            ),
        )


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
            head_attention = _opencv_attention_proxy(
                horizontal_offset=horizontal_offset,
                vertical_offset=vertical_offset,
                eye_count=len(eyes),
            )
            eye_attention, normalized_eyes = _opencv_eye_attention(
                eyes, face_width=int(w), face_height=int(h)
            )
            attention_proxy = _clamp(0.5 * head_attention + 0.5 * eye_attention)
            gaze_quality = 0.82 if len(normalized_eyes) >= 2 else 0.42
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
                    "eyes": [
                        {"x": ex, "y": ey, "width": ew, "height": eh}
                        for ex, ey, ew, eh in normalized_eyes
                    ],
                    "eye_count": len(eyes),
                    "gaze_source": "opencv_eye_geometry",
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
