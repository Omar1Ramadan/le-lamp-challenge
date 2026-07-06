from pathlib import Path

import anyio
import numpy as np
import pytest
from fastapi.testclient import TestClient
from social_lamp.api.app import create_app
from social_lamp.capture.frames import CapturedFrame
from social_lamp.memory.repository import MemoryRepository
from social_lamp.perception.objects import Detection
from social_lamp.replay.trace import TraceReader
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


@pytest.mark.asyncio
async def test_live_runtime_persists_stable_object_memory(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    coordinator = await build_live_runtime(database_path=database)
    try:
        detection = Detection(
            label="keys", confidence=0.91, bbox=(0.35, 0.35, 0.55, 0.65), mono_ns=10
        )
        for step in range(5):
            await coordinator.process_vision_frame(
                CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10 + step),
                face_processor=FakeFaceProcessor(),
                object_detector=FakeObjectDetector((detection,)),
                anchors={"desk": (0.3, 0.3, 0.7, 0.8)},
            )
    finally:
        await coordinator.stop()

    repository = await MemoryRepository.open(database)
    try:
        result = await repository.find_last_seen("keys")
        assert result.status == "found"
        assert result.anchor_name == "desk"
    finally:
        await repository.close()


def test_app_clear_memory_and_text_recall_use_active_repository(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    from social_lamp.memory.repository import ObservationWrite

    async def seed_memory() -> None:
        repository = await MemoryRepository.open(database)
        try:
            await repository.record(ObservationWrite.example("keys", "right"))
        finally:
            await repository.close()

    anyio.run(seed_memory)

    with TestClient(create_app(database_path=database)) as client:
        response = client.post("/api/text", json={"text": "Where are my keys?"}).json()
        assert response["response"]["status"] == "found"
        assert "right" in response["response"]["text"]

        assert client.post("/api/memory/clear").json() == {"ok": True}
        response = client.post("/api/text", json={"text": "Where are my keys?"}).json()
        assert response["response"]["status"] == "not_found"


@pytest.mark.asyncio
async def test_export_trace_uses_active_runtime_session(tmp_path: Path) -> None:
    coordinator = await build_live_runtime(database_path=tmp_path / "memory.db")
    try:
        detection = Detection(label="mug", confidence=0.91, bbox=(0.1, 0.1, 0.4, 0.4), mono_ns=10)
        for step in range(5):
            await coordinator.process_vision_frame(
                CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=20 + step),
                face_processor=FakeFaceProcessor(),
                object_detector=FakeObjectDetector((detection,)),
                anchors={},
            )
        export = coordinator.export_trace(tmp_path / "trace-export")
    finally:
        await coordinator.stop()

    reader = TraceReader(tmp_path / "trace-export")
    assert export["checksum_valid"] is True
    assert reader.manifest().session_id == str(coordinator.world.snapshot.session_id)
    records = list(reader.records())
    assert records[-1].record_type == "snapshot"
    assert records[-1].body["revision"] == coordinator.world.snapshot.revision
