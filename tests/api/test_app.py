from fastapi.testclient import TestClient
from social_lamp.api.app import create_app


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
            assert message["body"]["revision"] == 0
