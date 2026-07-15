from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from social_lamp.capture.frames import CapturedFrame
from social_lamp.config import Settings
from social_lamp.perception.engagement import EngagementSignals


@dataclass
class FaceProcessorMetadata:
    name: str
    status: str  # "active" | "degraded" | "disabled"
    detail: str | None = None
    model_path: str | None = None


class FaceDetectorMode(StrEnum):
    AUTO = "auto"
    MEDIAPIPE = "mediapipe"
    OPENCV = "opencv"
    HEURISTIC = "heuristic"
    DISABLED = "disabled"


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
    pose_source: str = "unavailable"
    roll_degrees: float = 0.0
    pose_quality: float = 0.0


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
    attention = _clamp(0.15 + 0.35 * separation_score + 0.3 * height_score + 0.2 * level_score)
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


def _euler_from_mediapipe_matrix(
    matrix: NDArray[np.float32],
) -> tuple[float, float, float]:
    r = np.array(matrix, dtype=np.float64).reshape(4, 4)
    R = r[:3, :3]

    pitch = float(np.arcsin(np.clip(-R[1, 2], -1.0, 1.0)))
    yaw = float(np.arctan2(R[0, 2], R[2, 2]))
    roll = float(np.arctan2(R[1, 0], R[1, 1]))

    yaw_deg = float(np.degrees(yaw))
    pitch_deg = float(np.degrees(pitch))
    roll_deg = float(np.degrees(roll))

    return yaw_deg, pitch_deg, roll_deg


def _estimate_pose_from_landmarks(
    landmarks: object,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float] | None:
    try:
        import cv2
    except Exception:
        return None

    indices_2d = (1, 168, 33, 362, 61, 291, 152)
    model_3d = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -35.0, 12.0),
            (-33.0, -30.0, -8.0),
            (33.0, -30.0, -8.0),
            (-28.0, 40.0, -5.0),
            (28.0, 40.0, -5.0),
            (0.0, 65.0, 5.0),
        ],
        dtype=np.float64,
    )

    points_2d: list[tuple[float, float]] = []
    for idx in indices_2d:
        lm = landmarks[idx]
        points_2d.append((float(lm.x) * image_width, float(lm.y) * image_height))

    if not points_2d:
        return None

    image_points = np.array(points_2d, dtype=np.float64)
    focal_length = max(image_width, image_height)
    camera_matrix = np.array(
        [
            [focal_length, 0, image_width / 2.0],
            [0, focal_length, image_height / 2.0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, _ = cv2.solvePnP(
        model_3d, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    pitch = float(np.arcsin(np.clip(-rmat[1, 2], -1.0, 1.0)))
    yaw = float(np.arctan2(rmat[0, 2], rmat[2, 2]))
    roll = float(np.arctan2(rmat[1, 0], rmat[1, 1]))

    return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))


def _box_proxy_pose(
    center_x: float, center_y: float
) -> tuple[float, float]:
    horizontal_offset = center_x - 0.5
    vertical_offset = center_y - 0.45
    yaw = horizontal_offset * 120.0
    pitch = vertical_offset * 100.0
    return yaw, pitch


def face_result_to_signals(
    *,
    face_confidence: float,
    yaw_degrees: float,
    pitch_degrees: float,
    gaze_score: float,
    gaze_quality: float,
    face_area_ratio: float,
    pose_source: str = "unavailable",
    pose_quality: float = 0.0,
) -> EngagementSignals:
    head: float | None
    if pose_quality < 0.5:
        head = None
    else:
        head = _clamp(1.0 - abs(yaw_degrees) / 45.0 - abs(pitch_degrees) / 40.0)

    proximity = _clamp(face_area_ratio / 0.15)
    gaze = _clamp(gaze_score) if gaze_quality >= 0.45 else None

    base_conf = _clamp(face_confidence)
    if pose_quality < 0.5:
        confidence = min(base_conf, 0.35)
    else:
        confidence = min(base_conf, max(_clamp(gaze_quality), 0.45))

    return EngagementSignals(
        face_presence=_clamp(face_confidence),
        head_toward=head,
        gaze_toward=gaze,
        proximity=proximity,
        directed_speech=0.0,
        confidence=confidence,
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
        self.metadata = FaceProcessorMetadata(
            name="mediapipe_face_landmarker", status="active", model_path=str(model_path)
        )
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

        yaw_degrees: float
        pitch_degrees: float
        roll_degrees: float = 0.0
        pose_source: str
        pose_quality: float

        h, w = frame.image.shape[:2]

        if result.facial_transformation_matrixes:
            matrix = np.array(result.facial_transformation_matrixes[0], dtype=np.float32)
            yaw_degrees, pitch_degrees, roll_degrees = _euler_from_mediapipe_matrix(matrix)
            pose_source = "mediapipe_matrix"
            pose_quality = 0.95
        else:
            pnp_result = _estimate_pose_from_landmarks(landmarks, w, h)
            if pnp_result is not None:
                yaw_degrees, pitch_degrees, roll_degrees = pnp_result
                pose_source = "landmark_pnp"
                pose_quality = 0.85
            else:
                yaw_degrees, pitch_degrees = _box_proxy_pose(center_x, center_y)
                pose_source = "box_proxy"
                pose_quality = 0.35

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
            "pose_source": pose_source,
            "pose_quality": round(pose_quality, 2),
            "yaw_degrees": round(yaw_degrees, 1),
            "pitch_degrees": round(pitch_degrees, 1),
            "roll_degrees": round(roll_degrees, 1),
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
                pose_source=pose_source,
                roll_degrees=roll_degrees,
                pose_quality=pose_quality,
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
        self.metadata = FaceProcessorMetadata(name="opencv_haar", status="active")
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
            yaw_degrees, pitch_degrees = _box_proxy_pose(center_x, center_y)
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
                    "pose_source": "box_proxy",
                    "pose_quality": 0.4,
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
                    pose_source="box_proxy",
                    pose_quality=0.4,
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
        detections = self._eye_cascade.detectMultiScale(upper_face, scaleFactor=1.1, minNeighbors=4)
        return tuple((int(ex), int(ey), int(ew), int(eh)) for ex, ey, ew, eh in detections)


class HeuristicFaceProcessor:
    detail = "face model unavailable; using low-reliability heuristic fallback"

    def __init__(self) -> None:
        self.metadata = FaceProcessorMetadata(
            name="heuristic_skin_region", status="degraded", detail=self.detail
        )
        self.last_debug: dict[str, object] | None = None
        self._cv2: object | None = None

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _ensure_cv2(self) -> object | None:
        if self._cv2 is None:
            try:
                import cv2

                self._cv2 = cv2
            except Exception:
                self._cv2 = False
        return self._cv2 if self._cv2 is not False else None

    def _skin_mask(self, image: NDArray[np.uint8]) -> NDArray[np.bool_]:
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)

        bgr_skin = (r > 70) & (g > 40) & (b > 20) & (r > g) & (r > b) & ((r - b) > 15)

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cr = r - g
        cb = b - g
        ycrcb_skin = (
            (y > 40)
            & (cr > -10) & (cr < 45)
            & (cb > -30) & (cb < 15)
            & ((cr - cb) > 3)
        )

        return bgr_skin | ycrcb_skin

    def _morphological_open(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        cv2_mod = self._ensure_cv2()
        if cv2_mod is not None:
            kernel = cv2_mod.getStructuringElement(cv2_mod.MORPH_ELLIPSE, (5, 5))
            opened = cv2_mod.morphologyEx(
                mask.astype(np.uint8), cv2_mod.MORPH_OPEN, kernel
            )
            return opened.astype(np.bool_)

        m = mask.copy()
        for _ in range(2):
            e = m.copy()
            e[1:, :] &= m[:-1, :]
            e[:-1, :] &= m[1:, :]
            e[:, 1:] &= m[:, :-1]
            e[:, :-1] &= m[:, 1:]
            m = e
        for _ in range(2):
            d = m.copy()
            d[1:, :] |= m[:-1, :]
            d[:-1, :] |= m[1:, :]
            d[:, 1:] |= m[:, :-1]
            d[:, :-1] |= m[:, 1:]
            m = d
        return m

    def _find_largest_bbox(
        self, mask: NDArray[np.bool_]
    ) -> tuple[int, int, int, int] | None:
        cv2_mod = self._ensure_cv2()
        if cv2_mod is not None:
            num_labels, labels, stats, centroids = cv2_mod.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8
            )
            if num_labels < 2:
                return None
            areas = stats[1:, cv2_mod.CC_STAT_AREA]
            if len(areas) == 0:
                return None
            largest_idx = int(np.argmax(areas)) + 1
            x = int(stats[largest_idx, cv2_mod.CC_STAT_LEFT])
            y = int(stats[largest_idx, cv2_mod.CC_STAT_TOP])
            w = int(stats[largest_idx, cv2_mod.CC_STAT_WIDTH])
            h = int(stats[largest_idx, cv2_mod.CC_STAT_HEIGHT])
            return (x, y, x + w - 1, y + h - 1)

        y_indices, x_indices = np.nonzero(mask)
        if len(y_indices) < 30:
            return None
        return (
            int(x_indices.min()),
            int(y_indices.min()),
            int(x_indices.max()),
            int(y_indices.max()),
        )

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

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

        mask = self._skin_mask(image)
        mask = self._morphological_open(mask)

        total_pixels = float(mask.size)
        skin_count = float(np.sum(mask))
        skin_coverage = skin_count / total_pixels
        heuristic_reasons: list[str] = []

        bbox = self._find_largest_bbox(mask)
        if bbox is None:
            heuristic_reasons.append("no_skin_region")
            self.last_debug = {
                "skin_coverage": round(skin_coverage, 4),
                "heuristic_reason": heuristic_reasons,
            }
            return ()

        x_min, y_min, x_max, y_max = bbox
        face_width = max(x_max - x_min + 1, 1)
        face_height = max(y_max - y_min + 1, 1)
        face_area = face_width * face_height
        frame_area = float(width * height)
        face_area_ratio = face_area / frame_area
        aspect_ratio = face_width / max(face_height, 1)

        if face_area_ratio > 0.30:
            heuristic_reasons.append("too_large")
        if face_area_ratio < 0.015:
            heuristic_reasons.append("too_small")
        if aspect_ratio < 0.35 or aspect_ratio > 1.6:
            heuristic_reasons.append("bad_aspect")

        center_x = (x_min + x_max) / 2.0 / float(width)
        center_y = (y_min + y_max) / 2.0 / float(height)
        if center_y > 0.75 or center_y < 0.1:
            heuristic_reasons.append("bad_position")

        box_slice = mask[y_min : y_max + 1, x_min : x_max + 1]
        box_skin_count = float(np.sum(box_slice))
        box_density = box_skin_count / max(face_area, 1)
        if box_density < 0.2:
            heuristic_reasons.append("low_density")

        if heuristic_reasons:
            self.last_debug = {
                "skin_coverage": round(skin_coverage, 4),
                "face_area_ratio": round(face_area_ratio, 4),
                "aspect_ratio": round(aspect_ratio, 3),
                "heuristic_reason": heuristic_reasons,
            }
            return ()

        horizontal_offset = abs(center_x - 0.5)
        vertical_offset = abs(center_y - 0.45)
        centeredness = max(0.0, 1.0 - horizontal_offset / 0.35 - vertical_offset / 0.45)
        frontal_shape = max(0.0, 1.0 - abs(aspect_ratio - 0.78) / 0.55)
        attention_proxy = min(centeredness, frontal_shape)
        yaw_degrees, pitch_degrees = _box_proxy_pose(center_x, center_y)

        self.last_debug = {
            "attention_proxy": round(attention_proxy, 3),
            "box": {
                "x": x_min / float(width),
                "y": y_min / float(height),
                "width": face_width / float(width),
                "height": face_height / float(height),
            },
            "target": {"x": center_x, "y": center_y},
            "gaze_source": "heuristic_skin_region",
            "pose_source": "box_proxy",
            "pose_quality": 0.4,
            "yaw_degrees": round(yaw_degrees, 1),
            "pitch_degrees": round(pitch_degrees, 1),
            "skin_coverage": round(skin_coverage, 4),
            "face_area_ratio": round(face_area_ratio, 4),
            "aspect_ratio": round(aspect_ratio, 3),
            "box_density": round(box_density, 3),
            "heuristic_reason": heuristic_reasons or None,
        }

        return (
            FaceResult(
                face_confidence=0.45,
                yaw_degrees=yaw_degrees,
                pitch_degrees=pitch_degrees,
                gaze_score=round(attention_proxy * 0.5, 3),
                gaze_quality=0.35,
                face_area_ratio=min(face_area_ratio, 0.15),
                pose_source="box_proxy",
                pose_quality=0.4,
            ),
        )


def build_face_detector(settings: Settings) -> tuple[object | None, FaceProcessorMetadata]:
    mode = FaceDetectorMode(settings.face_detector_mode)

    if mode == FaceDetectorMode.DISABLED:
        return (
            None,
            FaceProcessorMetadata(
                name="none", status="disabled", detail="face detection disabled by config"
            ),
        )

    if mode == FaceDetectorMode.HEURISTIC:
        processor = HeuristicFaceProcessor()
        return (processor, processor.metadata)

    if mode == FaceDetectorMode.OPENCV:
        try:
            processor = OpenCvFaceProcessor()
            return (processor, processor.metadata)
        except RuntimeError:
            fallback = HeuristicFaceProcessor()
            meta = FaceProcessorMetadata(
                name=fallback.metadata.name,
                status="degraded",
                detail=fallback.metadata.detail,
            )
            return (fallback, meta)

    if mode == FaceDetectorMode.MEDIAPIPE:
        try:
            processor = MediaPipeFaceLandmarkerProcessor()
            return (processor, processor.metadata)
        except RuntimeError:
            fallback = HeuristicFaceProcessor()
            meta = FaceProcessorMetadata(
                name=fallback.metadata.name,
                status="degraded",
                detail=fallback.metadata.detail,
            )
            return (fallback, meta)

    # auto: try mediapipe -> opencv -> heuristic
    try:
        processor = MediaPipeFaceLandmarkerProcessor()
        return (processor, processor.metadata)
    except RuntimeError:
        pass
    try:
        processor = OpenCvFaceProcessor()
        return (processor, processor.metadata)
    except RuntimeError:
        pass
    fallback = HeuristicFaceProcessor()
    meta = FaceProcessorMetadata(
        name=fallback.metadata.name,
        status="degraded",
        detail=fallback.metadata.detail,
    )
    return (fallback, meta)
