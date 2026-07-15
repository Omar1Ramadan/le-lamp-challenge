from pathlib import Path

import numpy as np
from social_lamp.config import Settings
from social_lamp.perception.objects import (
    Detection,
    FastObjectDetector,
    NullObjectDetector,
    ObjectTrack,
    YoloObjectDetector,
)
from social_lamp.perception.location import BBox


class _FakeModel:
    def predict(self, image: np.ndarray) -> list:
        return []


def test_null_detector_reports_disabled_health() -> None:
    detector = NullObjectDetector()
    health = detector.health()
    assert health.component == "object_detector"
    assert health.status == "disabled"
    assert "not configured" in (health.detail or "").lower()


def test_active_detector_reports_ok_health() -> None:
    model = _FakeModel()
    detector = FastObjectDetector(model=model)
    health = detector.health()
    assert health.component == "object_detector"
    assert health.status == "active"


def test_active_detector_returns_empty_on_no_objects() -> None:
    model = _FakeModel()
    detector = FastObjectDetector(model=model)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detector.detect(image)
    assert detections == ()


def _fake_model_path() -> Path:
    return Path("/tmp/__not_a_real_model.pt")


def test_yolo_detector_reports_degraded_when_model_missing() -> None:
    detector = YoloObjectDetector(model_path=_fake_model_path(), confidence=0.45)
    health = detector.health()
    assert health.status == "degraded"


def test_yolo_detector_does_not_crash_on_empty_image() -> None:
    detector = YoloObjectDetector(model_path=_fake_model_path(), confidence=0.45)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detector.detect(image)
    assert isinstance(detections, tuple)


# ---------------------------------------------------------------------------
# ObjectTrack stability tests
# ---------------------------------------------------------------------------

BBOX: BBox = (0.35, 0.35, 0.55, 0.65)


def _detections(label: str, n: int, *, start_ns: int = 0, confidence: float = 0.9) -> list[Detection]:
    return [
        Detection(label=label, confidence=confidence, bbox=BBOX, mono_ns=start_ns + i)
        for i in range(n)
    ]


def test_track_not_stable_with_fewer_than_5_detections() -> None:
    track = ObjectTrack(track_id="track-test")
    for d in _detections("keys", 4):
        track.add(d)
    assert track.is_stable is False


def test_track_stable_with_5_consistent_detections_within_1s() -> None:
    track = ObjectTrack(track_id="track-test")
    for d in _detections("keys", 5, start_ns=0):
        track.add(d)
    assert track.is_stable is True


def test_track_not_stable_when_time_span_exceeds_1s() -> None:
    track = ObjectTrack(track_id="track-test")
    dets = _detections("keys", 4, start_ns=0) + [
        Detection(label="keys", confidence=0.9, bbox=BBOX, mono_ns=2_000_000_000),
    ]
    for d in dets:
        track.add(d)
    assert track.is_stable is False


def test_track_not_stable_with_low_confidence() -> None:
    track = ObjectTrack(track_id="track-test")
    for d in _detections("keys", 5, start_ns=0, confidence=0.3):
        track.add(d)
    assert track.is_stable is False


def test_track_not_stable_with_flapping_labels() -> None:
    track = ObjectTrack(track_id="track-test")
    labels = ["keys", "keys", "bottle", "keys", "bottle"]
    for i, label in enumerate(labels):
        track.add(Detection(label=label, confidence=0.9, bbox=BBOX, mono_ns=i))
    assert track.is_stable is False


def test_track_label_majority_vote() -> None:
    track = ObjectTrack(track_id="track-test")
    labels = ["keys", "keys", "bottle", "keys", "keys"]
    for i, label in enumerate(labels):
        track.add(Detection(label=label, confidence=0.9, bbox=BBOX, mono_ns=i))
    assert track.label == "keys"
    assert track.is_stable is True


def test_track_stable_boundary_labels_75_percent() -> None:
    track = ObjectTrack(track_id="track-test")
    labels = ["keys", "keys", "keys", "keys", "bottle"]
    for i, label in enumerate(labels):
        track.add(Detection(label=label, confidence=0.9, bbox=BBOX, mono_ns=i))
    assert track.is_stable is True


def test_track_not_stable_boundary_labels_below_75_percent() -> None:
    track = ObjectTrack(track_id="track-test")
    labels = ["keys", "keys", "keys", "bottle", "bottle"]
    for i, label in enumerate(labels):
        track.add(Detection(label=label, confidence=0.9, bbox=BBOX, mono_ns=i))
    assert track.is_stable is False


def test_track_deque_evicts_oldest() -> None:
    track = ObjectTrack(track_id="track-test")
    for d in _detections("keys", 25):
        track.add(d)
    assert len(track.detections) == 20


def test_track_stability_uses_last_5_detections() -> None:
    track = ObjectTrack(track_id="track-test")
    for d in _detections("keys", 16, start_ns=0):
        track.add(d)
    dets = _detections("bottle", 5, start_ns=1_000)
    for d in dets:
        track.add(d)
    assert track.is_stable is True
    assert track.label == "keys"


def test_object_detection_settings_default_to_disabled() -> None:
    settings = Settings()
    assert settings.enable_object_detection is False
    assert settings.object_detector_model == "yolov8n.pt"
    assert settings.object_detection_confidence == 0.45
    assert settings.object_detection_max_fps == 8
    assert settings.object_detection_classes is None
