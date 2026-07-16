from pathlib import Path

from fastapi.testclient import TestClient
from social_lamp.api.app import create_app
from social_lamp.domain.contracts import ComponentHealth


def test_websocket_receives_sequenced_replay_updates(tmp_path: Path) -> None:
    with TestClient(create_app(database_path=tmp_path / "memory.db")) as client:
        with client.websocket_connect("/ws") as socket:
            initial = socket.receive_json()
            assert initial["seq"] == 1
            assert initial["type"] == "world_snapshot"

            response = client.post(
                "/api/replay", json={"directory": "evaluation/fixtures/core-engagement"}
            )
            assert response.status_code == 200

            timeline = socket.receive_json()
            snapshot = socket.receive_json()
            metric = socket.receive_json()

            assert [timeline["seq"], snapshot["seq"], metric["seq"]] == [2, 3, 4]
            assert timeline["type"] == "behavior_timeline"
            assert snapshot["type"] == "world_snapshot"
            assert snapshot["body"]["social_state"] == "engaged"
            assert metric == {
                "seq": 4,
                "type": "metric",
                "body": {"name": "social_transition", "labels": {"state": "engaged"}},
            }


def test_simulator_acknowledgements_are_recorded(tmp_path: Path) -> None:
    with TestClient(create_app(database_path=tmp_path / "memory.db")) as client:
        with client.websocket_connect("/ws") as socket:
            socket.receive_json()
            client.post("/api/replay", json={"directory": "evaluation/fixtures/core-engagement"})
            timeline = socket.receive_json()
            timeline_id = timeline["body"]["timeline_id"]

            for ack_type in ("timeline_received", "first_visible_frame", "timeline_complete"):
                socket.send_json(
                    {
                        "type": "simulator_ack",
                        "body": {"timeline_id": timeline_id, "ack_type": ack_type},
                    }
                )

            expected_acks = ["timeline_received", "first_visible_frame", "timeline_complete"]
            assert client.get(f"/api/simulator/timelines/{timeline_id}").json() == {
                "timeline_id": timeline_id,
                "acknowledgements": expected_acks,
            }


def test_missing_browser_degrades_adapter_health(tmp_path: Path) -> None:
    with TestClient(create_app(database_path=tmp_path / "memory.db")) as client:
        response = client.post(
            "/api/replay", json={"directory": "evaluation/fixtures/core-engagement"}
        )
        assert response.status_code == 200
        world = client.get("/api/world").json()
        assert (
            ComponentHealth(
                component="simulator", status="degraded", detail="no browser client connected"
            ).model_dump(mode="json")
            in world["health"]
        )
