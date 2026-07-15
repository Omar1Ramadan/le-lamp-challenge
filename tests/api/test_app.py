import base64

import numpy as np
from fastapi.testclient import TestClient
from social_lamp.api.app import create_app
from social_lamp.capture.frames import CapturedFrame
from social_lamp.domain.contracts import ComponentHealth
from social_lamp.perception.faces import FaceResult


class FakeFaceProcessor:
    def process(self, frame: CapturedFrame, *, now_mono_ns: int) -> tuple[FaceResult, ...]:
        del frame, now_mono_ns
        return (
            FaceResult(
                face_confidence=0.9,
                yaw_degrees=0.0,
                pitch_degrees=0.0,
                gaze_score=0.8,
                gaze_quality=0.9,
                face_area_ratio=0.12,
                pose_source="mediapipe_matrix",
                pose_quality=0.95,
            ),
        )


class FakeObjectDetector:
    def __init__(self, health_status: str = "active") -> None:
        self._health_status = health_status

    def detect(self, image: np.ndarray) -> tuple[()]:
        del image
        return ()

    def health(self) -> ComponentHealth:
        return ComponentHealth(component="object_detector", status=self._health_status)


def test_health_and_initial_snapshot_are_available() -> None:
    with TestClient(create_app()) as client:
        assert client.get("/api/health").json() == {"status": "healthy"}
        response = client.get("/api/world")
        assert response.status_code == 200
        assert response.json()["social_state"] == "idle"


def test_websocket_receives_initial_snapshot() -> None:
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws") as socket:
            message = socket.receive_json()
            assert message["type"] == "world_snapshot"
            assert message["body"]["revision"] >= 0


def test_replays_endpoint_lists_local_fixtures() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/replays")
        assert response.status_code == 200
        replay_ids = {item["id"] for item in response.json()["replays"]}
        assert "core-journey" in replay_ids


def test_browser_vision_frame_updates_people() -> None:
    with TestClient(create_app()) as client:
        client.app.state.browser_face_processor = FakeFaceProcessor()
        client.app.state.browser_object_detector = FakeObjectDetector()

        response = client.post(
            "/api/vision/frame",
            json={"image_base64": _encoded_test_jpeg()},
        )

        assert response.status_code == 200
        assert len(response.json()["world_snapshot"]["people"]) == 1
        world = client.get("/api/world").json()
        assert world["people"] == [
            {
                "person_id": "person-1",
                "engagement_score": 0.76,
                "engagement_confidence": 0.9,
                "is_active_speaker": False,
            }
        ]


def test_browser_vision_frame_rejects_invalid_image() -> None:
    with TestClient(create_app()) as client:
        response = client.post("/api/vision/frame", json={"image_base64": "not-base64"})

        assert response.status_code == 400


def test_browser_vision_frame_with_disabled_detector_reports_disabled_health() -> None:
    with TestClient(create_app()) as client:
        client.app.state.browser_face_processor = FakeFaceProcessor()
        client.app.state.browser_object_detector = FakeObjectDetector(health_status="disabled")

        response = client.post(
            "/api/vision/frame",
            json={"image_base64": _encoded_test_jpeg()},
        )
        assert response.status_code == 200
        health = response.json()["world_snapshot"]["health"]
        assert any(
            h["component"] == "object_detector" and h["status"] == "disabled" for h in health
        )


def test_browser_vision_frame_falls_back_when_face_model_missing(monkeypatch) -> None:
    monkeypatch.setenv("FACE_DETECTOR_MODE", "disabled")

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/vision/frame",
            json={"image_base64": _encoded_person_like_jpeg()},
        )

        assert response.status_code == 200
        # With FACE_DETECTOR_MODE=disabled, no people should be tracked
        assert len(response.json()["world_snapshot"]["people"]) == 0


def test_browser_vision_frame_includes_vision_status() -> None:
    with TestClient(create_app()) as client:
        client.app.state.browser_face_processor = FakeFaceProcessor()
        client.app.state.browser_object_detector = FakeObjectDetector()
        from social_lamp.perception.faces import FaceProcessorMetadata

        client.app.state.browser_face_processor_metadata = FaceProcessorMetadata(
            name="test_detector", status="active"
        )

        response = client.post(
            "/api/vision/frame",
            json={"image_base64": _encoded_test_jpeg()},
        )
        assert response.status_code == 200
        data = response.json()
        assert "vision_status" in data
        vs = data["vision_status"]
        assert vs["face_detector"]["name"] == "test_detector"
        assert vs["face_detector"]["status"] == "active"
        assert vs["object_detector"]["name"] == "object_detector"
        assert vs["object_detector"]["status"] == "active"


def _encoded_test_jpeg() -> str:
    import cv2

    ok, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _encoded_person_like_jpeg() -> str:
    import cv2

    image = np.full((96, 96, 3), 45, dtype=np.uint8)
    image[18:78, 26:72] = (95, 120, 145)
    image[35:42, 34:42] = (40, 40, 40)
    image[35:42, 54:62] = (40, 40, 40)
    image[58:62, 40:56] = (55, 55, 55)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")
