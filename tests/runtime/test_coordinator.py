from pathlib import Path
from time import monotonic_ns

import pytest
from social_lamp.audio.stream import MicrophoneChunk
from social_lamp.domain.contracts import (
    AudioMode,
    ComponentHealth,
    PersonState,
    SocialState,
    WorldSnapshot,
)
from social_lamp.runtime.coordinator import RuntimeCoordinator
from social_lamp.runtime.testing import FakeSimulator, FakeMetrics, MutableWorldModel, TestMemory
from social_lamp.world.commands import Source


@pytest.mark.asyncio
async def test_engagement_replay_reaches_simulator_adapter(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    await coordinator.start()
    try:
        await coordinator.replay(Path("evaluation/fixtures/core-engagement"))
        assert coordinator.world.snapshot.social_state.value == "engaged"
        assert coordinator.simulator.executed[-1].duration_ms == 700
        assert coordinator.metrics.counter("social_transition", state="engaged") == 1
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_stop_neutralizes_adapter_and_closes_memory(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    await coordinator.start()
    await coordinator.stop()
    assert coordinator.simulator.neutralized
    assert coordinator.memory.closed


@pytest.mark.asyncio
async def test_text_submission_acknowledges_command(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    await coordinator.start()
    try:
        response = await coordinator.submit_text("Where are my keys?")
        assert response.status in {"not_found", "unsupported"}
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_set_health_duplicate_does_not_publish_repeatedly(tmp_path: Path) -> None:
    publish_count = 0
    published: list[dict] = []

    async def publisher(body: dict) -> None:
        nonlocal publish_count
        publish_count += 1
        published.append(body)

    from social_lamp.domain.clock import FakeClock
    from uuid6 import uuid7

    clock = FakeClock(100, "2026-07-04T12:00:00Z")
    world = MutableWorldModel(session_id=uuid7(), clock=clock)
    coordinator = RuntimeCoordinator(
        world=world,
        simulator=FakeSimulator(),
        metrics=FakeMetrics(),
        memory=TestMemory(tmp_path / "memory.db"),
        snapshot_publisher=publisher,
    )

    await coordinator._set_health("camera", "ok", None)
    assert publish_count == 1

    await coordinator._set_health("camera", "ok", None)
    assert publish_count == 1

    await coordinator._set_health("camera", "degraded", "no signal")
    assert publish_count == 2

    await coordinator.stop()


@pytest.mark.asyncio
async def test_vision_exception_marks_health_degraded(tmp_path: Path) -> None:
    from social_lamp.capture.frames import CapturedFrame
    import numpy as np

    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")

    class BrokenProcessor:
        def process(self, frame, *, now_mono_ns):
            raise RuntimeError("model failure")

    frame = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=100)
    result = await coordinator.process_vision_frame(
        frame,
        face_processor=BrokenProcessor(),
        object_detector=object(),
        anchors={},
    )
    assert result is None
    assert ComponentHealth(component="vision", status="degraded", detail="model failure") in coordinator.world.snapshot.health


@pytest.mark.asyncio
async def test_audio_updates_preserve_vision_people_when_no_speaker(tmp_path: Path) -> None:
    from social_lamp.capture.frames import CapturedFrame
    from social_lamp.perception.faces import FaceResult
    import numpy as np

    class FakeFaceProcessor:
        def process(self, frame, *, now_mono_ns):
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
        def detect(self, image):
            return ()
        def health(self):
            return ComponentHealth(component="object_detector", status="disabled")

    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")

    frame = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=100)
    await coordinator.process_vision_frame(
        frame,
        face_processor=FakeFaceProcessor(),
        object_detector=FakeObjectDetector(),
        anchors={},
    )
    assert len(coordinator.world.snapshot.people) == 1
    vision_person = coordinator.world.snapshot.people[0]

    class SilentClassifier:
        def classify(self, pcm, sample_rate):
            from social_lamp.audio.analysis import AudioClass, VoiceFrame
            return VoiceFrame(False, AudioClass.NOISE, 0.1)

    await coordinator.process_audio_chunk(
        MicrophoneChunk(pcm=b"", mono_ns=200), classifier=SilentClassifier()
    )

    snapshot = coordinator.world.snapshot
    assert len(snapshot.people) == 1
    assert snapshot.people[0].person_id == vision_person.person_id


@pytest.mark.asyncio
async def test_vision_frame_only_publishes_meaningful_change(tmp_path: Path) -> None:
    publish_count = 0

    async def publisher(body: dict) -> None:
        nonlocal publish_count
        publish_count += 1

    from social_lamp.capture.frames import CapturedFrame
    from social_lamp.domain.clock import FakeClock
    from social_lamp.perception.faces import FaceResult
    from uuid6 import uuid7
    import numpy as np

    clock = FakeClock(100, "2026-07-04T12:00:00Z")
    world = MutableWorldModel(session_id=uuid7(), clock=clock)
    coordinator = RuntimeCoordinator(
        world=world,
        simulator=FakeSimulator(),
        metrics=FakeMetrics(),
        memory=TestMemory(tmp_path / "memory.db"),
        snapshot_publisher=publisher,
    )

    face = FaceResult(
        face_confidence=0.9,
        yaw_degrees=0.0,
        pitch_degrees=0.0,
        gaze_score=0.8,
        gaze_quality=0.9,
        face_area_ratio=0.12,
        pose_source="mediapipe_matrix",
        pose_quality=0.95,
    )
    frame1 = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=100)
    await coordinator.process_vision_frame(
        frame1,
        face_processor=_FakeFaceProcessor((face,)),
        object_detector=_FakeObjectDetector(()),
        anchors={},
    )
    assert publish_count == 1

    frame2 = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=200)
    await coordinator.process_vision_frame(
        frame2,
        face_processor=_FakeFaceProcessor((face,)),
        object_detector=_FakeObjectDetector(()),
        anchors={},
    )
    assert publish_count == 1

    await coordinator.stop()


class _FakeFaceProcessor:
    def __init__(self, faces):
        self.faces = faces

    def process(self, frame, *, now_mono_ns):
        return self.faces


class _FakeObjectDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, image):
        return self.detections
