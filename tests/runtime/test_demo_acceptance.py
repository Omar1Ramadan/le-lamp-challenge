from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from social_lamp.api.app import create_app
from social_lamp.capture.frames import CapturedFrame
from social_lamp.memory.repository import MemoryRepository
from social_lamp.perception.objects import Detection
from social_lamp.runtime.live import build_live_runtime


class FakeFaceProcessor:
    def process(self, frame: CapturedFrame, *, now_mono_ns: int) -> tuple[()]:
        del frame, now_mono_ns
        return ()


class FakeObjectDetector:
    def __init__(self, detections: tuple[Detection, ...]) -> None:
        self._detections = detections

    def detect(self, image: np.ndarray) -> tuple[Detection, ...]:
        del image
        return self._detections


def test_core_replay_streams_backend_evidence_and_recall(tmp_path: Path) -> None:
    with TestClient(create_app(database_path=tmp_path / "memory.db")) as client:
        with client.websocket_connect("/ws") as socket:
            assert socket.receive_json()["type"] == "world_snapshot"

            response = client.post(
                "/api/replay", json={"directory": "evaluation/fixtures/core-journey"}
            )
            assert response.status_code == 200

            messages = [socket.receive_json() for _ in range(13)]
            message_types = [message["type"] for message in messages]
            assert "behavior_timeline" in message_types
            assert "memory_result" in message_types
            assert any(
                message["type"] == "world_snapshot" and message["body"]["social_state"] == "engaged"
                for message in messages
            )
            assert any(
                message["type"] == "metric"
                and message["body"]["name"] == "attention_level"
                and message["body"].get("value") == 1
                for message in messages
            )

            recall = client.post("/api/text", json={"text": "Where are my keys?"}).json()
            assert recall["response"]["status"] == "found"
            assert "right side of the desk" in recall["response"]["text"]
            assert recall["response"]["evidence_ids"] == ["observation-core-keys-2"]


@pytest.mark.asyncio
async def test_live_runtime_acceptance_uses_fakes_without_physical_hardware(tmp_path: Path) -> None:
    coordinator = await build_live_runtime(database_path=tmp_path / "memory.db")
    try:
        detection = Detection(
            label="keys", confidence=0.91, bbox=(0.72, 0.60, 0.92, 0.90), mono_ns=10
        )
        for step in range(5):
            await coordinator.process_vision_frame(
                CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10 + step),
                face_processor=FakeFaceProcessor(),
                object_detector=FakeObjectDetector((detection,)),
                anchors={"desk": (0.0, 0.55, 1.0, 1.0)},
            )
        answer = await coordinator.submit_text("Where are my keys?")
        assert answer.status == "found"
        assert "right side of the desk" in answer.text
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_replay_memory_persists_as_normal_runtime_evidence(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    with TestClient(create_app(database_path=database)) as client:
        client.post("/api/replay", json={"directory": "evaluation/fixtures/core-journey"})

    repository = await MemoryRepository.open(database)
    try:
        result = await repository.find_last_seen("keys")
        assert result.status == "found"
        assert result.evidence_ids == ("observation-core-keys-2",)
    finally:
        await repository.close()
