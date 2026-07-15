import numpy as np
from social_lamp.domain.contracts import ComponentHealth
from social_lamp.perception.objects import FastObjectDetector, NullObjectDetector


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
