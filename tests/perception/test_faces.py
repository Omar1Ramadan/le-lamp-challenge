import numpy as np
from social_lamp.capture.frames import CapturedFrame
from social_lamp.config import Settings
from social_lamp.perception.faces import (
    FaceResult,
    HeuristicFaceProcessor,
    MediaPipeFaceAdapter,
    OpenCvFaceProcessor,
    build_face_detector,
    face_result_to_signals,
    _box_proxy_pose,
    _euler_from_mediapipe_matrix,
)


# ---------------------------------------------------------------------------
# Existing tests updated for new FaceResult / face_result_to_signals API
# ---------------------------------------------------------------------------


def test_low_quality_eyes_disable_gaze_signal() -> None:
    signals = face_result_to_signals(
        face_confidence=0.92,
        yaw_degrees=4.0,
        pitch_degrees=-3.0,
        gaze_score=0.8,
        gaze_quality=0.2,
        face_area_ratio=0.12,
        pose_quality=0.95,
    )
    assert signals.gaze_toward is None
    assert signals.head_toward > 0.8


def test_face_mapping_clamps_values() -> None:
    signals = face_result_to_signals(
        face_confidence=1.4,
        yaw_degrees=80.0,
        pitch_degrees=80.0,
        gaze_score=3.0,
        gaze_quality=0.9,
        face_area_ratio=0.5,
        pose_quality=0.95,
    )
    assert signals.face_presence == 1.0
    assert signals.head_toward == 0.0
    assert signals.gaze_toward == 1.0
    assert signals.proximity == 1.0


def test_mediapipe_adapter_skips_stale_frames_without_calling_landmarker() -> None:
    class Landmarker:
        called = False

        def detect(self, image: np.ndarray) -> list[object]:
            self.called = True
            return []

    landmarker = Landmarker()
    adapter = MediaPipeFaceAdapter(landmarker=landmarker)
    frame = CapturedFrame(np.zeros((2, 2, 3), dtype=np.uint8), mono_ns=1)
    assert adapter.process(frame, now_mono_ns=400_000_001) == ()
    assert not landmarker.called


def test_heuristic_face_processor_requires_skin_tone_center() -> None:
    processor = HeuristicFaceProcessor()
    empty_room = np.full((96, 96, 3), (40, 80, 40), dtype=np.uint8)

    assert processor.process(CapturedFrame(empty_room, mono_ns=1), now_mono_ns=1) == ()


def test_heuristic_face_processor_detects_person_like_center() -> None:
    processor = HeuristicFaceProcessor()
    image = np.full((96, 96, 3), 45, dtype=np.uint8)
    image[18:78, 26:72] = (95, 120, 145)

    faces = processor.process(CapturedFrame(image, mono_ns=1), now_mono_ns=1)

    assert len(faces) == 1
    assert faces[0].gaze_quality < 0.45  # degraded, below usable threshold
    assert faces[0].gaze_score < 0.55  # conservative, not high
    assert faces[0].face_confidence < 0.5  # modest confidence


def test_build_face_detector_disabled_returns_none() -> None:
    settings = Settings(_env_file=None, face_detector_mode="disabled")
    processor, metadata = build_face_detector(settings)
    assert processor is None
    assert metadata.status == "disabled"


def test_build_face_detector_heuristic_returns_heuristic() -> None:
    settings = Settings(_env_file=None, face_detector_mode="heuristic")
    processor, metadata = build_face_detector(settings)
    assert isinstance(processor, HeuristicFaceProcessor)
    assert metadata.name == "heuristic_skin_region"
    assert metadata.status == "degraded"


def test_build_face_detector_opencv_returns_opencv() -> None:
    settings = Settings(_env_file=None, face_detector_mode="opencv")
    processor, metadata = build_face_detector(settings)
    assert isinstance(processor, OpenCvFaceProcessor)
    assert metadata.name == "opencv_haar"
    assert metadata.status == "active"


def test_build_face_detector_metadata_matches_processor() -> None:
    settings = Settings(_env_file=None, face_detector_mode="heuristic")
    _, metadata = build_face_detector(settings)
    assert metadata.name == "heuristic_skin_region"
    assert metadata.status == "degraded"
    assert metadata.detail is not None


def test_heuristic_face_processor_lowers_attention_when_off_center() -> None:
    processor = HeuristicFaceProcessor()
    image = np.full((96, 96, 3), 45, dtype=np.uint8)
    image[18:78, 6:52] = (95, 120, 145)

    faces = processor.process(CapturedFrame(image, mono_ns=1), now_mono_ns=1)

    assert len(faces) == 1
    assert faces[0].gaze_score < 0.45
    assert abs(faces[0].yaw_degrees) > 20.0


# ---------------------------------------------------------------------------
# New tests for Phase 1: FaceResult defaults
# ---------------------------------------------------------------------------


def test_face_result_has_pose_fields() -> None:
    result = FaceResult(
        face_confidence=0.9,
        yaw_degrees=10.0,
        pitch_degrees=5.0,
        gaze_score=0.8,
        gaze_quality=0.7,
        face_area_ratio=0.1,
    )
    assert result.pose_source == "unavailable"
    assert result.roll_degrees == 0.0
    assert result.pose_quality == 0.0


# ---------------------------------------------------------------------------
# New tests for Phase 2: _euler_from_mediapipe_matrix
# ---------------------------------------------------------------------------


def test_matrix_identity_gives_near_zero_angles() -> None:
    identity = np.eye(4, dtype=np.float32)
    yaw, pitch, roll = _euler_from_mediapipe_matrix(identity)
    assert abs(yaw) < 1e-6
    assert abs(pitch) < 1e-6
    assert abs(roll) < 1e-6


def test_matrix_positive_yaw() -> None:
    theta = np.radians(30.0)
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.array(
        [
            [cos, 0.0, sin, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin, 0.0, cos, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    yaw, pitch, roll = _euler_from_mediapipe_matrix(R)
    assert abs(yaw - 30.0) < 1.0
    assert abs(pitch) < 1.0
    assert abs(roll) < 1.0


def test_matrix_negative_yaw() -> None:
    theta = np.radians(-20.0)
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.array(
        [
            [cos, 0.0, sin, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin, 0.0, cos, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    yaw, pitch, roll = _euler_from_mediapipe_matrix(R)
    assert abs(yaw - (-20.0)) < 1.0
    assert abs(pitch) < 1.0
    assert abs(roll) < 1.0


def test_matrix_positive_pitch() -> None:
    theta = np.radians(15.0)
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos, -sin, 0.0],
            [0.0, sin, cos, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    yaw, pitch, roll = _euler_from_mediapipe_matrix(R)
    assert abs(yaw) < 1.0
    assert abs(pitch - 15.0) < 1.0
    assert abs(roll) < 1.0


def test_matrix_positive_roll() -> None:
    theta = np.radians(10.0)
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.array(
        [
            [cos, -sin, 0.0, 0.0],
            [sin, cos, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    yaw, pitch, roll = _euler_from_mediapipe_matrix(R)
    assert abs(yaw) < 1.0
    assert abs(pitch) < 1.0
    assert abs(roll - 10.0) < 1.0


def test_matrix_4x4_input() -> None:
    matrix_4x4 = np.eye(4, dtype=np.float32)
    yaw, pitch, roll = _euler_from_mediapipe_matrix(matrix_4x4)
    assert abs(yaw) < 1e-6
    assert abs(pitch) < 1e-6
    assert abs(roll) < 1e-6


# ---------------------------------------------------------------------------
# New tests for _box_proxy_pose
# ---------------------------------------------------------------------------


def test_box_proxy_centered() -> None:
    yaw, pitch = _box_proxy_pose(0.5, 0.45)
    assert abs(yaw) < 0.1
    assert abs(pitch) < 0.1


def test_box_proxy_left_offset() -> None:
    yaw, pitch = _box_proxy_pose(0.3, 0.45)
    assert yaw < 0.0
    assert abs(pitch) < 0.1


def test_box_proxy_right_offset() -> None:
    yaw, pitch = _box_proxy_pose(0.7, 0.45)
    assert yaw > 0.0
    assert abs(pitch) < 0.1


def test_box_proxy_down_offset() -> None:
    yaw, pitch = _box_proxy_pose(0.5, 0.65)
    assert abs(yaw) < 0.1
    assert pitch > 0.0


# ---------------------------------------------------------------------------
# New tests for Phase 5: face_result_to_signals with pose quality
# ---------------------------------------------------------------------------


def test_low_pose_quality_removes_head_toward() -> None:
    signals = face_result_to_signals(
        face_confidence=0.92,
        yaw_degrees=4.0,
        pitch_degrees=-3.0,
        gaze_score=0.8,
        gaze_quality=0.9,
        face_area_ratio=0.12,
        pose_quality=0.3,
    )
    assert signals.head_toward is None
    assert signals.gaze_toward is not None
    assert signals.face_presence is not None
    assert signals.proximity is not None


def test_low_pose_quality_lowers_confidence() -> None:
    signals = face_result_to_signals(
        face_confidence=0.92,
        yaw_degrees=0.0,
        pitch_degrees=0.0,
        gaze_score=0.9,
        gaze_quality=0.9,
        face_area_ratio=0.12,
        pose_quality=0.3,
    )
    assert signals.confidence <= 0.35


def test_high_pose_quality_preserves_head_toward() -> None:
    signals = face_result_to_signals(
        face_confidence=0.92,
        yaw_degrees=5.0,
        pitch_degrees=3.0,
        gaze_score=0.8,
        gaze_quality=0.9,
        face_area_ratio=0.12,
        pose_quality=0.95,
    )
    assert signals.head_toward is not None
    assert signals.head_toward > 0.7


def test_high_yaw_lowers_head_toward() -> None:
    signals = face_result_to_signals(
        face_confidence=0.92,
        yaw_degrees=40.0,
        pitch_degrees=0.0,
        gaze_score=0.8,
        gaze_quality=0.9,
        face_area_ratio=0.12,
        pose_quality=0.95,
    )
    assert signals.head_toward is not None
    assert signals.head_toward < 0.2


def test_mid_pose_quality_boundary() -> None:
    signals_above = face_result_to_signals(
        face_confidence=0.9,
        yaw_degrees=0.0,
        pitch_degrees=0.0,
        gaze_score=0.8,
        gaze_quality=0.8,
        face_area_ratio=0.1,
        pose_quality=0.5,
    )
    assert signals_above.head_toward is not None

    signals_below = face_result_to_signals(
        face_confidence=0.9,
        yaw_degrees=0.0,
        pitch_degrees=0.0,
        gaze_score=0.8,
        gaze_quality=0.8,
        face_area_ratio=0.1,
        pose_quality=0.49,
    )
    assert signals_below.head_toward is None


def test_off_center_face_with_matrix_pose_not_treated_as_turned() -> None:
    signals = face_result_to_signals(
        face_confidence=0.92,
        yaw_degrees=2.0,
        pitch_degrees=1.0,
        gaze_score=0.8,
        gaze_quality=0.9,
        face_area_ratio=0.12,
        pose_source="mediapipe_matrix",
        pose_quality=0.95,
    )
    assert signals.head_toward is not None
    assert signals.head_toward > 0.85


# ---------------------------------------------------------------------------
# New tests for pose_source propagation
# ---------------------------------------------------------------------------


def test_default_pose_source_unavailable() -> None:
    signals = face_result_to_signals(
        face_confidence=0.9,
        yaw_degrees=0.0,
        pitch_degrees=0.0,
        gaze_score=0.5,
        gaze_quality=0.5,
        face_area_ratio=0.1,
    )
    assert signals.head_toward is None


def test_pose_quality_threshold_edge() -> None:
    signals = face_result_to_signals(
        face_confidence=0.9,
        yaw_degrees=0.0,
        pitch_degrees=0.0,
        gaze_score=0.5,
        gaze_quality=0.5,
        face_area_ratio=0.1,
        pose_quality=0.5,
    )
    assert signals.head_toward is not None
    assert signals.confidence > 0.35


# ---------------------------------------------------------------------------
# New tests for processor debug output
# ---------------------------------------------------------------------------


def test_heuristic_debug_contains_debug_fields() -> None:
    processor = HeuristicFaceProcessor()
    image = np.full((96, 96, 3), 45, dtype=np.uint8)
    image[18:78, 26:72] = (95, 120, 145)
    processor.process(CapturedFrame(image, mono_ns=1), now_mono_ns=1)
    assert processor.last_debug is not None
    assert processor.last_debug["gaze_source"] == "heuristic_skin_region"
    assert processor.last_debug["pose_source"] == "box_proxy"
    assert processor.last_debug["pose_quality"] == 0.4
    assert "skin_coverage" in processor.last_debug
    assert "face_area_ratio" in processor.last_debug
    assert "aspect_ratio" in processor.last_debug
    assert "heuristic_reason" in processor.last_debug


def test_face_result_fields_default() -> None:
    result = FaceResult(
        face_confidence=0.9,
        yaw_degrees=10.0,
        pitch_degrees=5.0,
        gaze_score=0.8,
        gaze_quality=0.7,
        face_area_ratio=0.1,
    )
    assert result.pose_source == "unavailable"
    assert result.roll_degrees == 0.0
    assert result.pose_quality == 0.0


# ---------------------------------------------------------------------------
# Phase 7: Heuristic false-positive rejection
# ---------------------------------------------------------------------------


def test_heuristic_rejects_uniform_warm_background() -> None:
    processor = HeuristicFaceProcessor()
    uniform_warm = np.full((96, 96, 3), (100, 120, 140), dtype=np.uint8)
    faces = processor.process(CapturedFrame(uniform_warm, mono_ns=1), now_mono_ns=1)
    assert faces == ()


def test_heuristic_rejects_large_skin_block() -> None:
    processor = HeuristicFaceProcessor()
    image = np.full((96, 96, 3), 45, dtype=np.uint8)
    image[5:90, 5:90] = (95, 120, 145)
    faces = processor.process(CapturedFrame(image, mono_ns=1), now_mono_ns=1)
    assert faces == ()


def test_heuristic_rejects_tiny_skin_patch() -> None:
    processor = HeuristicFaceProcessor()
    image = np.full((96, 96, 3), 45, dtype=np.uint8)
    image[46:51, 46:51] = (95, 120, 145)
    faces = processor.process(CapturedFrame(image, mono_ns=1), now_mono_ns=1)
    assert faces == ()


def test_heuristic_rejects_off_center_blob() -> None:
    processor = HeuristicFaceProcessor()
    image = np.full((96, 96, 3), 45, dtype=np.uint8)
    image[18:78, 6:52] = (95, 120, 145)
    faces = processor.process(CapturedFrame(image, mono_ns=1), now_mono_ns=1)
    assert len(faces) == 1
    assert faces[0].gaze_score < 0.35


# ---------------------------------------------------------------------------
# New tests for getattr-based landmark access safety
# ---------------------------------------------------------------------------


def test_euler_identity_matrix_various_shapes() -> None:
    flat_16 = np.arange(16, dtype=np.float32).reshape(4, 4)
    flat_16[3, :3] = 0.0
    flat_16[:3, 3] = 0.0
    flat_16[3, 3] = 1.0
    identity = np.eye(4, dtype=np.float32)
    yaw, pitch, roll = _euler_from_mediapipe_matrix(identity)
    assert abs(yaw) < 1e-6
    assert abs(pitch) < 1e-6
    assert abs(roll) < 1e-6
