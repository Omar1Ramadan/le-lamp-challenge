import numpy as np
from social_lamp.capture.frames import CapturedFrame
from social_lamp.perception.faces import MediaPipeFaceAdapter, face_result_to_signals


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
