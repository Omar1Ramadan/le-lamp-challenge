from pathlib import Path

import numpy as np
import pytest
from social_lamp.capture.frames import CapturedFrame, LatestFrameBuffer
from social_lamp.domain.contracts import ComponentHealth, ObjectState, PersonState
from social_lamp.memory.repository import MemoryRepository
from social_lamp.perception.faces import FaceResult
from social_lamp.perception.location import BBox
from social_lamp.perception.objects import Detection
from social_lamp.runtime.coordinator import RuntimeCoordinator
from social_lamp.runtime.testing import build_test_runtime


class FakeFaceProcessor:
    def __init__(self, faces: tuple[FaceResult, ...] = ()) -> None:
        self.faces = faces

    def process(self, frame: CapturedFrame, *, now_mono_ns: int) -> tuple[FaceResult, ...]:
        del frame, now_mono_ns
        return self.faces


class FakeObjectDetector:
    def __init__(self, detections: tuple[Detection, ...] = ()) -> None:
        self.detections = detections

    def detect(self, image: np.ndarray) -> tuple[Detection, ...]:
        del image
        return self.detections


@pytest.mark.asyncio
async def test_live_frame_updates_world_with_face_and_stable_object(tmp_path: Path) -> None:
    memory = await MemoryRepository.open(tmp_path / "memory.db")
    coordinator = build_test_runtime(tmp_path / "unused.db")
    coordinator.memory = memory
    try:
        face = FaceResult(
            face_confidence=0.9,
            yaw_degrees=0.0,
            pitch_degrees=0.0,
            gaze_score=0.8,
            gaze_quality=0.9,
            face_area_ratio=0.12,
        )
        bbox: BBox = (0.35, 0.35, 0.55, 0.65)
        detection = Detection(label="keys", confidence=0.92, bbox=bbox, mono_ns=10)

        for step in range(5):
            frame = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10 + step)
            await coordinator.process_vision_frame(
                frame,
                face_processor=FakeFaceProcessor((face,)),
                object_detector=FakeObjectDetector((detection,)),
                anchors={"desk": (0.3, 0.3, 0.7, 0.8)},
            )

        snapshot = coordinator.world.snapshot
        assert snapshot.revision > 0
        assert snapshot.people == (
            PersonState(person_id="person-1", engagement_score=0.76, engagement_confidence=0.9),
        )
        assert snapshot.objects == (
            ObjectState(
                track_id="track-keys",
                label="keys",
                confidence=0.92,
                horizontal_region="center",
                depth_band="midground",
                anchor_name="desk",
            ),
        )
        assert await memory.count_observations() == 1
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_live_frame_degrades_health_when_model_fails(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")

    class BrokenDetector:
        def detect(self, image: np.ndarray) -> tuple[Detection, ...]:
            del image
            raise RuntimeError("model unavailable")

    frame = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10)
    await coordinator.process_vision_frame(
        frame,
        face_processor=FakeFaceProcessor(),
        object_detector=BrokenDetector(),
        anchors={},
    )

    assert (
        ComponentHealth(component="vision", status="degraded", detail="model unavailable")
        in coordinator.world.snapshot.health
    )


@pytest.mark.asyncio
async def test_live_frame_preserves_person_for_brief_missed_detection(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    try:
        face = FaceResult(
            face_confidence=0.9,
            yaw_degrees=0.0,
            pitch_degrees=0.0,
            gaze_score=0.8,
            gaze_quality=0.9,
            face_area_ratio=0.12,
        )
        frame1 = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10_000_000)
        await coordinator.process_vision_frame(
            frame1,
            face_processor=FakeFaceProcessor((face,)),
            object_detector=FakeObjectDetector(),
            anchors={},
        )
        assert len(coordinator.world.snapshot.people) == 1

        frame2 = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10_100_000)
        await coordinator.process_vision_frame(
            frame2,
            face_processor=FakeFaceProcessor(),
            object_detector=FakeObjectDetector(),
            anchors={},
        )

        people = coordinator.world.snapshot.people
        assert len(people) == 1
        assert people[0].person_id == "person-1"
        assert people[0].engagement_confidence == round(0.9 * 0.5, 2)
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_live_frame_clears_person_after_missing_expiry(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    try:
        face = FaceResult(
            face_confidence=0.9,
            yaw_degrees=0.0,
            pitch_degrees=0.0,
            gaze_score=0.8,
            gaze_quality=0.9,
            face_area_ratio=0.12,
        )
        frame1 = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=0)
        await coordinator.process_vision_frame(
            frame1,
            face_processor=FakeFaceProcessor((face,)),
            object_detector=FakeObjectDetector(),
            anchors={},
        )
        assert len(coordinator.world.snapshot.people) == 1

        frame2 = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=2_000_000_000)
        await coordinator.process_vision_frame(
            frame2,
            face_processor=FakeFaceProcessor(),
            object_detector=FakeObjectDetector(),
            anchors={},
        )
        assert coordinator.world.snapshot.people == ()
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_missing_face_does_not_create_new_engagement_transition(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    try:
        face = FaceResult(
            face_confidence=0.9,
            yaw_degrees=0.0,
            pitch_degrees=0.0,
            gaze_score=0.8,
            gaze_quality=0.9,
            face_area_ratio=0.12,
        )
        frame1 = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10_000_000)
        await coordinator.process_vision_frame(
            frame1,
            face_processor=FakeFaceProcessor((face,)),
            object_detector=FakeObjectDetector(),
            anchors={},
        )
        first_state = coordinator.world.snapshot.social_state

        frame2 = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10_100_000)
        await coordinator.process_vision_frame(
            frame2,
            face_processor=FakeFaceProcessor(),
            object_detector=FakeObjectDetector(),
            anchors={},
        )
        assert coordinator.world.snapshot.social_state == first_state
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_disabled_detector_reports_disabled_health(tmp_path: Path) -> None:
    coordinator = build_test_runtime(tmp_path / "unused.db")
    try:
        frame = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10)
        await coordinator.process_vision_frame(
            frame,
            face_processor=FakeFaceProcessor(),
            object_detector=FakeObjectDetector(),
            anchors={},
        )
        assert any(
            h.component == "object_detector" and h.status == "disabled"
            for h in coordinator.world.snapshot.health
        )
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_degraded_detector_sets_degraded_health(tmp_path: Path) -> None:
    coordinator = build_test_runtime(tmp_path / "unused.db")

    class DegradedDetector:
        def detect(self, image: np.ndarray) -> tuple[Detection, ...]:
            del image
            return ()

        def health(self) -> ComponentHealth:
            return ComponentHealth(
                component="object_detector",
                status="degraded",
                detail="object model crashed",
            )

    frame = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10)
    await coordinator.process_vision_frame(
        frame,
        face_processor=FakeFaceProcessor(),
        object_detector=DegradedDetector(),
        anchors={},
    )
    assert any(
        h.component == "object_detector" and h.status == "degraded"
        for h in coordinator.world.snapshot.health
    )


@pytest.mark.asyncio
async def test_active_detector_with_objects_updates_world(tmp_path: Path) -> None:
    coordinator = build_test_runtime(tmp_path / "unused.db")

    class ActiveDetector:
        def __init__(self) -> None:
            self._call_count = 0

        def detect(self, image: np.ndarray) -> tuple[Detection, ...]:
            self._call_count += 1
            bbox: BBox = (0.1, 0.1, 0.5, 0.5)
            return (
                Detection(label="bottle", confidence=0.9, bbox=bbox, mono_ns=10 + self._call_count),
            )  # noqa: E501

        def health(self) -> ComponentHealth:
            return ComponentHealth(component="object_detector", status="active")

    try:
        for step in range(5):
            frame = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10 + step)
            await coordinator.process_vision_frame(
                frame,
                face_processor=FakeFaceProcessor(),
                object_detector=ActiveDetector(),
                anchors={"desk": (0.0, 0.0, 1.0, 1.0)},
            )
        snapshot = coordinator.world.snapshot
        assert any(obj.label == "bottle" for obj in snapshot.objects)
        assert any(
            h.component == "object_detector" and h.status == "active" for h in snapshot.health
        )
    finally:
        await coordinator.stop()


def test_latest_frame_buffer_reports_stale_frames() -> None:
    buffer = LatestFrameBuffer(capacity=1, max_age_ns=5)
    buffer.put(np.zeros((2, 2, 3), dtype=np.uint8), mono_ns=10)

    assert buffer.latest(now_mono_ns=20) is None
    assert buffer.health(now_mono_ns=20) == ComponentHealth(
        component="camera", status="degraded", detail="stale frame"
    )
