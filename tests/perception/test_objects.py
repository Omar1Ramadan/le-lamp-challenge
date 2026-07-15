from pathlib import Path

import numpy as np
from social_lamp.config import Settings
from social_lamp.domain.contracts import ComponentHealth
from social_lamp.perception.objects import FastObjectDetector, NullObjectDetector, YoloObjectDetector


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


def test_object_detection_settings_default_to_disabled() -> None:
    settings = Settings()
    assert settings.enable_object_detection is False
    assert settings.object_detector_model == "yolov8n.pt"
    assert settings.object_detection_confidence == 0.45
    assert settings.object_detection_max_fps == 8
    assert settings.object_detection_classes is None
