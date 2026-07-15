import numpy as np
from social_lamp.capture.frames import CapturedFrame
from social_lamp.config import Settings
from social_lamp.perception.faces import (
    FaceDetectorMode,
    FaceProcessorMetadata,
    HeuristicFaceProcessor,
    MediaPipeFaceAdapter,
    OpenCvFaceProcessor,
    build_face_detector,
    face_result_to_signals,
)


def test_low_quality_eyes_disable_gaze_signal() -> None:
    signals = face_result_to_signals(
        face_confidence=0.92,
        yaw_degrees=4.0,
        pitch_degrees=-3.0,
        gaze_score=0.8,
        gaze_quality=0.2,
        face_area_ratio=0.12,
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
    assert faces[0].gaze_score > 0.75


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
    assert metadata.status == "active"


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
    assert metadata.status == "active"
    assert metadata.detail is not None


def test_heuristic_face_processor_lowers_attention_when_off_center() -> None:
    processor = HeuristicFaceProcessor()
    image = np.full((96, 96, 3), 45, dtype=np.uint8)
    image[18:78, 6:52] = (95, 120, 145)

    faces = processor.process(CapturedFrame(image, mono_ns=1), now_mono_ns=1)

    assert len(faces) == 1
    assert faces[0].gaze_score < 0.45
    assert faces[0].yaw_degrees > 20.0
