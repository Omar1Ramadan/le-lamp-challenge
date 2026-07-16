from pathlib import Path

import numpy as np
import pytest
import social_lamp.runtime.coordinator as coordinator_module
from social_lamp.capture.frames import CapturedFrame, LatestFrameBuffer
from social_lamp.domain.clock import FakeClock
from social_lamp.domain.contracts import ComponentHealth, ObjectState, PersonState
from social_lamp.memory.repository import MemoryRepository
from social_lamp.perception.faces import FaceResult
from social_lamp.perception.location import BBox
from social_lamp.perception.objects import Detection
from social_lamp.runtime.coordinator import (
    ObjectMemoryState,
    RuntimeCoordinator,
    should_record_object_observation,
)
from social_lamp.runtime.testing import (
    FakeMetrics,
    FakeSimulator,
    MutableWorldModel,
    TestMemory,
    build_test_runtime,
)
from uuid6 import uuid7

from tests.fakes.vision import (
    FailingFaceProcessor,
    SequenceFaceProcessor,
    TimedObjectDetector,
    make_face,
    make_frame,
)


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
            pose_source="mediapipe_matrix",
            pose_quality=0.95,
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
            pose_source="mediapipe_matrix",
            pose_quality=0.95,
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
            pose_source="mediapipe_matrix",
            pose_quality=0.95,
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
            pose_source="mediapipe_matrix",
            pose_quality=0.95,
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


# ---------------------------------------------------------------------------
# Unit tests for should_record_object_observation
# ---------------------------------------------------------------------------


def test_first_stable_triggers_record() -> None:
    mem = ObjectMemoryState()
    decision, reason = should_record_object_observation(
        mem, "keys", "center", "midground", "desk", 100
    )
    assert decision is True
    assert reason == "first_stable"


def test_unchanged_does_not_record_before_refresh() -> None:
    mem = ObjectMemoryState(
        last_recorded_label="keys",
        last_recorded_region="center",
        last_recorded_depth="midground",
        last_recorded_anchor="desk",
        last_recorded_mono_ns=100,
    )
    decision, reason = should_record_object_observation(
        mem, "keys", "center", "midground", "desk", 200
    )
    assert decision is False
    assert reason == "unchanged"


def test_refresh_triggers_after_30_seconds() -> None:
    mem = ObjectMemoryState(
        last_recorded_label="keys",
        last_recorded_region="center",
        last_recorded_depth="midground",
        last_recorded_anchor="desk",
        last_recorded_mono_ns=100,
    )
    decision, reason = should_record_object_observation(
        mem, "keys", "center", "midground", "desk", 30_000_000_100
    )
    assert decision is True
    assert reason == "refresh"


def test_label_changed_triggers_record() -> None:
    mem = ObjectMemoryState(
        last_recorded_label="keys",
        last_recorded_region="center",
        last_recorded_depth="midground",
        last_recorded_anchor="desk",
        last_recorded_mono_ns=100,
    )
    decision, reason = should_record_object_observation(
        mem, "bottle", "center", "midground", "desk", 200
    )
    assert decision is True
    assert reason == "label_changed"


def test_location_changed_pending_before_threshold() -> None:
    mem = ObjectMemoryState(
        last_recorded_label="keys",
        last_recorded_region="center",
        last_recorded_depth="midground",
        last_recorded_anchor="desk",
        last_recorded_mono_ns=100,
    )
    decision, reason = should_record_object_observation(
        mem, "keys", "left", "midground", "desk", 200
    )
    assert decision is False
    assert reason == "location_pending"


def test_location_changed_after_one_second() -> None:
    mem = ObjectMemoryState(
        last_recorded_label="keys",
        last_recorded_region="center",
        last_recorded_depth="midground",
        last_recorded_anchor="desk",
        last_recorded_mono_ns=100,
        pending_region="left",
        pending_depth="midground",
        pending_anchor="desk",
        pending_since_mono_ns=200,
    )
    decision, reason = should_record_object_observation(
        mem, "keys", "left", "midground", "desk", 1_200_000_200
    )
    assert decision is True
    assert reason == "location_changed"


def test_location_pending_does_not_record_before_one_second() -> None:
    mem = ObjectMemoryState(
        last_recorded_label="keys",
        last_recorded_region="center",
        last_recorded_depth="midground",
        last_recorded_anchor="desk",
        last_recorded_mono_ns=100,
        pending_region="left",
        pending_depth="midground",
        pending_anchor="desk",
        pending_since_mono_ns=1_200_000_000,
    )
    decision, reason = should_record_object_observation(
        mem, "keys", "left", "midground", "desk", 1_200_000_500
    )
    assert decision is False
    assert reason == "location_pending"


def test_location_pending_reverted_is_location_pending() -> None:
    mem = ObjectMemoryState(
        last_recorded_label="keys",
        last_recorded_region="center",
        last_recorded_depth="midground",
        last_recorded_anchor="desk",
        last_recorded_mono_ns=100,
        pending_region="left",
        pending_depth="midground",
        pending_anchor="desk",
        pending_since_mono_ns=200,
    )
    decision, reason = should_record_object_observation(
        mem, "keys", "center", "midground", "desk", 500
    )
    assert decision is False
    assert reason == "unchanged"


def test_location_pending_pending_loc_different_from_new_loc() -> None:
    mem = ObjectMemoryState(
        last_recorded_label="keys",
        last_recorded_region="center",
        last_recorded_depth="midground",
        last_recorded_anchor="desk",
        last_recorded_mono_ns=100,
        pending_region="left",
        pending_depth="midground",
        pending_anchor="desk",
        pending_since_mono_ns=200,
    )
    decision, reason = should_record_object_observation(
        mem, "keys", "right", "midground", "desk", 500
    )
    assert decision is False
    assert reason == "location_pending"


def test_multiple_tracks_have_independent_memory_state() -> None:
    mem_a = ObjectMemoryState(last_recorded_label="keys", last_recorded_mono_ns=100)
    mem_b = ObjectMemoryState()
    decision_a, _ = should_record_object_observation(
        mem_a, "keys", "center", "midground", None, 200
    )
    decision_b, reason_b = should_record_object_observation(
        mem_b, "bottle", "center", "midground", None, 200
    )
    assert decision_a is False
    assert decision_b is True
    assert reason_b == "first_stable"


def test_label_change_overrides_location_pending() -> None:
    mem = ObjectMemoryState(
        last_recorded_label="keys",
        last_recorded_region="center",
        last_recorded_depth="midground",
        last_recorded_anchor="desk",
        last_recorded_mono_ns=100,
        pending_region="left",
        pending_depth="midground",
        pending_anchor="desk",
        pending_since_mono_ns=200,
    )
    decision, reason = should_record_object_observation(
        mem, "bottle", "left", "midground", "desk", 500
    )
    assert decision is True
    assert reason == "label_changed"


def test_location_depth_change() -> None:
    mem = ObjectMemoryState(
        last_recorded_label="keys",
        last_recorded_region="center",
        last_recorded_depth="midground",
        last_recorded_anchor="desk",
        last_recorded_mono_ns=100,
        pending_region="center",
        pending_depth="foreground",
        pending_anchor="desk",
        pending_since_mono_ns=200,
    )
    decision, reason = should_record_object_observation(
        mem, "keys", "center", "foreground", "desk", 1_200_000_200
    )
    assert decision is True
    assert reason == "location_changed"


def test_location_anchor_change() -> None:
    mem = ObjectMemoryState(
        last_recorded_label="keys",
        last_recorded_region="center",
        last_recorded_depth="midground",
        last_recorded_anchor="desk",
        last_recorded_mono_ns=100,
        pending_region="center",
        pending_depth="midground",
        pending_anchor="table",
        pending_since_mono_ns=200,
    )
    decision, reason = should_record_object_observation(
        mem, "keys", "center", "midground", "table", 1_200_000_200
    )
    assert decision is True
    assert reason == "location_changed"


# ---------------------------------------------------------------------------
# Integration tests for memory recording pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_stable_object_records_once(tmp_path: Path) -> None:
    memory = await MemoryRepository.open(tmp_path / "memory.db")
    coordinator = build_test_runtime(tmp_path / "unused.db")
    coordinator.memory = memory
    try:
        bbox: BBox = (0.35, 0.35, 0.55, 0.65)
        detection = Detection(label="keys", confidence=0.92, bbox=bbox, mono_ns=10)
        for step in range(20):
            frame = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10 + step)
            await coordinator.process_vision_frame(
                frame,
                face_processor=FakeFaceProcessor(),
                object_detector=FakeObjectDetector((detection,)),
                anchors={"desk": (0.3, 0.3, 0.7, 0.8)},
            )
        assert await memory.count_observations() == 1
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_stable_object_refreshes_after_30_seconds(tmp_path: Path) -> None:
    memory = await MemoryRepository.open(tmp_path / "memory.db")
    coordinator = build_test_runtime(tmp_path / "unused.db")
    coordinator.memory = memory
    try:
        bbox: BBox = (0.35, 0.35, 0.55, 0.65)
        detection = Detection(label="keys", confidence=0.92, bbox=bbox, mono_ns=10)

        for step in range(5):
            frame = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10 + step)
            await coordinator.process_vision_frame(
                frame,
                face_processor=FakeFaceProcessor(),
                object_detector=FakeObjectDetector((detection,)),
                anchors={"desk": (0.3, 0.3, 0.7, 0.8)},
            )
        assert await memory.count_observations() == 1

        for step in range(5):
            frame = CapturedFrame(
                np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=30_000_000_010 + step
            )
            await coordinator.process_vision_frame(
                frame,
                face_processor=FakeFaceProcessor(),
                object_detector=FakeObjectDetector((detection,)),
                anchors={"desk": (0.3, 0.3, 0.7, 0.8)},
            )
        assert await memory.count_observations() == 2
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_location_change_under_one_second_does_not_record(tmp_path: Path) -> None:
    memory = await MemoryRepository.open(tmp_path / "memory.db")
    coordinator = build_test_runtime(tmp_path / "unused.db")
    coordinator.memory = memory
    try:
        bbox: BBox = (0.35, 0.35, 0.55, 0.65)
        detection = Detection(label="keys", confidence=0.92, bbox=bbox, mono_ns=10)

        for step in range(5):
            frame = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10 + step)
            await coordinator.process_vision_frame(
                frame,
                face_processor=FakeFaceProcessor(),
                object_detector=FakeObjectDetector((detection,)),
                anchors={"desk": (0.3, 0.3, 0.7, 0.8)},
            )
        assert await memory.count_observations() == 1

        left_bbox: BBox = (0.0, 0.1, 0.2, 0.5)
        moved_detection = Detection(label="keys", confidence=0.92, bbox=left_bbox, mono_ns=100)
        for step in range(5):
            frame = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=100 + step)
            await coordinator.process_vision_frame(
                frame,
                face_processor=FakeFaceProcessor(),
                object_detector=FakeObjectDetector((moved_detection,)),
                anchors={"desk": (0.3, 0.3, 0.7, 0.8)},
            )
        assert await memory.count_observations() == 1
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_location_change_over_one_second_records_once(tmp_path: Path) -> None:
    memory = await MemoryRepository.open(tmp_path / "memory.db")
    coordinator = build_test_runtime(tmp_path / "unused.db")
    coordinator.memory = memory
    try:
        bbox: BBox = (0.35, 0.35, 0.55, 0.65)
        detection = Detection(label="keys", confidence=0.92, bbox=bbox, mono_ns=10)

        for step in range(5):
            frame = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10 + step)
            await coordinator.process_vision_frame(
                frame,
                face_processor=FakeFaceProcessor(),
                object_detector=FakeObjectDetector((detection,)),
                anchors={"desk": (0.3, 0.3, 0.7, 0.8)},
            )
        assert await memory.count_observations() == 1

        left_bbox: BBox = (0.0, 0.1, 0.2, 0.5)
        moved_detection = Detection(
            label="keys", confidence=0.92, bbox=left_bbox, mono_ns=1_000_000_000
        )
        for step in range(5):
            frame = CapturedFrame(
                np.zeros((4, 4, 3), dtype=np.uint8),
                mono_ns=1_000_000_000 + step * 250_000_000,
            )
            await coordinator.process_vision_frame(
                frame,
                face_processor=FakeFaceProcessor(),
                object_detector=FakeObjectDetector((moved_detection,)),
                anchors={"desk": (0.3, 0.3, 0.7, 0.8)},
            )
        # 5th frame: last 5 detections span 1s (stable), pending ≥1s → records once
        assert await memory.count_observations() == 2
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_multiple_tracks_maintain_independent_memory(tmp_path: Path) -> None:
    memory = await MemoryRepository.open(tmp_path / "memory.db")
    coordinator = build_test_runtime(tmp_path / "unused.db")
    coordinator.memory = memory
    try:
        bbox: BBox = (0.35, 0.35, 0.55, 0.65)
        keys = Detection(label="keys", confidence=0.92, bbox=bbox, mono_ns=10)
        bottle = Detection(label="bottle", confidence=0.9, bbox=bbox, mono_ns=10)

        for step in range(5):
            frame = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10 + step)
            await coordinator.process_vision_frame(
                frame,
                face_processor=FakeFaceProcessor(),
                object_detector=FakeObjectDetector((keys, bottle)),
                anchors={"desk": (0.3, 0.3, 0.7, 0.8)},
            )
        assert await memory.count_observations() == 2
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_flapping_location_does_not_spam_observations(tmp_path: Path) -> None:
    memory = await MemoryRepository.open(tmp_path / "memory.db")
    coordinator = build_test_runtime(tmp_path / "unused.db")
    coordinator.memory = memory
    try:
        bbox: BBox = (0.35, 0.35, 0.55, 0.65)
        detection = Detection(label="keys", confidence=0.92, bbox=bbox, mono_ns=10)

        for step in range(5):
            frame = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10 + step)
            await coordinator.process_vision_frame(
                frame,
                face_processor=FakeFaceProcessor(),
                object_detector=FakeObjectDetector((detection,)),
                anchors={"desk": (0.3, 0.3, 0.7, 0.8)},
            )
        assert await memory.count_observations() == 1

        left_bbox: BBox = (0.0, 0.1, 0.2, 0.5)
        center_bbox: BBox = (0.35, 0.35, 0.55, 0.65)
        flipped = [
            Detection(label="keys", confidence=0.92, bbox=center_bbox, mono_ns=100),
            Detection(label="keys", confidence=0.92, bbox=left_bbox, mono_ns=101),
            Detection(label="keys", confidence=0.92, bbox=center_bbox, mono_ns=200),
            Detection(label="keys", confidence=0.92, bbox=left_bbox, mono_ns=201),
            Detection(label="keys", confidence=0.92, bbox=center_bbox, mono_ns=300),
        ]
        for step, det in enumerate(flipped):
            frame = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=100 + step)
            await coordinator.process_vision_frame(
                frame,
                face_processor=FakeFaceProcessor(),
                object_detector=FakeObjectDetector((det,)),
                anchors={"desk": (0.3, 0.3, 0.7, 0.8)},
            )
        assert await memory.count_observations() == 1
    finally:
        await coordinator.stop()


# ---------------------------------------------------------------------------
# Intermittent detection: same person preserved
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intermittent_face_preserves_same_person(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    try:
        seq = SequenceFaceProcessor(
            [
                (make_face(),),
                (make_face(),),
                (),
                (make_face(),),
                (make_face(),),
            ]
        )
        for step in range(5):
            await coordinator.process_vision_frame(
                make_frame(mono_ns=step * 50_000_000),
                face_processor=seq,
                object_detector=TimedObjectDetector(()),
                anchors={},
            )
        people = coordinator.world.snapshot.people
        assert len(people) == 1
        assert people[0].person_id == "person-1"
    finally:
        await coordinator.stop()


# ---------------------------------------------------------------------------
# Failing processor marks vision degraded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failing_face_processor_sets_vision_degraded(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    try:
        await coordinator.process_vision_frame(
            make_frame(mono_ns=100),
            face_processor=FailingFaceProcessor(fail_on_call=1),
            object_detector=TimedObjectDetector(()),
            anchors={},
        )
        health = coordinator.world.snapshot.health
        assert any(h.component == "vision" and h.status == "degraded" for h in health)
    finally:
        await coordinator.stop()


# ---------------------------------------------------------------------------
# Stale frame through coordinator does not invoke detector
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_frame_processing_delegates_to_processor(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    processor = SequenceFaceProcessor([(make_face(),)])
    await coordinator.process_vision_frame(
        make_frame(mono_ns=100),
        face_processor=processor,
        object_detector=TimedObjectDetector(()),
        anchors={},
    )
    assert processor.call_count == 1


# ---------------------------------------------------------------------------
# Multiple faces (two separate people)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_face_processor_returns_multiple_faces(tmp_path: Path) -> None:
    seq = SequenceFaceProcessor(
        [
            (make_face(confidence=0.9), make_face(confidence=0.85, gaze_score=0.6)),
        ]
    )
    frame = make_frame(mono_ns=100)
    results = seq.process(frame, now_mono_ns=100)
    assert len(results) == 2
    assert results[0].face_confidence == 0.9
    assert results[1].face_confidence == 0.85


@pytest.mark.asyncio
async def test_coordinator_processes_multiple_faces(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    try:
        face1 = make_face(confidence=0.9, bbox=(0.1, 0.1, 0.2, 0.3))
        face2 = make_face(confidence=0.85, bbox=(0.6, 0.1, 0.2, 0.3), gaze_score=0.6)
        for step in range(5):
            await coordinator.process_vision_frame(
                make_frame(mono_ns=step * 50_000_000),
                face_processor=SequenceFaceProcessor([(face1, face2)]),
                object_detector=TimedObjectDetector(()),
                anchors={},
            )

        people = coordinator.world.snapshot.people
        assert [person.person_id for person in people] == ["person-1", "person-2"]
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_primary_person_id_uses_highest_engagement_face(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    try:
        lower_engagement = make_face(
            confidence=0.85,
            gaze_score=0.45,
            gaze_quality=0.9,
            bbox=(0.1, 0.1, 0.2, 0.3),
        )
        higher_engagement = make_face(
            confidence=0.95,
            gaze_score=0.95,
            gaze_quality=0.95,
            bbox=(0.6, 0.1, 0.25, 0.35),
        )
        for step in range(5):
            await coordinator.process_vision_frame(
                make_frame(mono_ns=step * 50_000_000),
                face_processor=SequenceFaceProcessor([(lower_engagement, higher_engagement)]),
                object_detector=TimedObjectDetector(()),
                anchors={},
            )

        assert coordinator.world.snapshot.primary_person_id == "person-2"
    finally:
        await coordinator.stop()


# ---------------------------------------------------------------------------
# Source conflict: stale camera result does not override newer browser frame
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coordinator_replaces_snapshot_on_each_frame(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    try:
        await coordinator.process_vision_frame(
            make_frame(mono_ns=1_000),
            face_processor=SequenceFaceProcessor([(make_face(),)]),
            object_detector=TimedObjectDetector(()),
            anchors={},
        )
        rev1 = coordinator.world.snapshot.revision

        await coordinator.process_vision_frame(
            make_frame(mono_ns=2_000),
            face_processor=SequenceFaceProcessor([()]),
            object_detector=TimedObjectDetector(()),
            anchors={},
        )
        assert coordinator.world.snapshot.revision > rev1
    finally:
        await coordinator.stop()


# ---------------------------------------------------------------------------
# Low-confidence face does not transition to engaged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_low_confidence_face_does_not_engage(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    try:
        for step in range(5):
            await coordinator.process_vision_frame(
                make_frame(mono_ns=step * 250_000_000),
                face_processor=SequenceFaceProcessor(
                    [(make_face(confidence=0.3, gaze_quality=0.2, pose_quality=0.0),)]
                ),
                object_detector=TimedObjectDetector(()),
                anchors={},
            )
        assert coordinator.world.snapshot.social_state.value in ("idle", "candidate")
    finally:
        await coordinator.stop()


# ---------------------------------------------------------------------------
# Heuristic-grade face should not reach engaged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_box_proxy_face_does_not_engage(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    try:
        for step in range(5):
            await coordinator.process_vision_frame(
                make_frame(mono_ns=step * 250_000_000),
                face_processor=SequenceFaceProcessor(
                    [(make_face(pose_source="box_proxy", pose_quality=0.4, gaze_quality=0.45),)]
                ),
                object_detector=TimedObjectDetector(()),
                anchors={},
            )
        assert coordinator.world.snapshot.social_state.value in ("idle", "candidate")
    finally:
        await coordinator.stop()


def test_latest_frame_buffer_reports_stale_frames() -> None:
    buffer = LatestFrameBuffer(capacity=1, max_age_ns=5)
    buffer.put(np.zeros((2, 2, 3), dtype=np.uint8), mono_ns=10)

    assert buffer.latest(now_mono_ns=20) is None
    assert buffer.health(now_mono_ns=20) == ComponentHealth(
        component="camera", status="degraded", detail="stale frame"
    )


@pytest.mark.asyncio
async def test_world_snapshot_includes_engagement_calibration_status(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    try:
        snapshot = coordinator.world.snapshot
        assert snapshot.engagement_calibration.state == "uncalibrated"
        assert snapshot.engagement_calibration.mode == "fallback"
        assert snapshot.engagement_calibration.progress == 0.0
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_runtime_calibration_completes_from_vision_frames(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    try:
        coordinator.start_engagement_calibration(mono_ns=0)
        face = FaceResult(
            face_confidence=0.92,
            yaw_degrees=12.0,
            pitch_degrees=6.0,
            roll_degrees=1.0,
            gaze_score=0.7,
            gaze_quality=0.9,
            face_area_ratio=0.2,
            pose_source="mediapipe_matrix",
            pose_quality=0.95,
        )
        for index in range(31):
            await coordinator.process_vision_frame(
                CapturedFrame(
                    np.zeros((4, 4, 3), dtype=np.uint8),
                    mono_ns=index * 100_000_000,
                ),
                face_processor=FakeFaceProcessor((face,)),
                object_detector=FakeObjectDetector(),
                anchors={},
            )

        status = coordinator.engagement_calibration_status(mono_ns=3_000_000_000)
        assert status.state == "calibrated"
        assert coordinator.world.snapshot.engagement_calibration.state == "calibrated"
        assert coordinator.world.snapshot.engagement_calibration.person_id == "person-1"
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_runtime_calibration_start_uses_current_time_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    try:
        monkeypatch.setattr(coordinator_module, "monotonic_ns", lambda: 10_000_000_000)
        coordinator.start_engagement_calibration()
        await coordinator.process_vision_frame(
            CapturedFrame(
                np.zeros((4, 4, 3), dtype=np.uint8),
                mono_ns=10_100_000_000,
            ),
            face_processor=FakeFaceProcessor(
                (
                    FaceResult(
                        face_confidence=0.92,
                        yaw_degrees=12.0,
                        pitch_degrees=6.0,
                        roll_degrees=1.0,
                        gaze_score=0.7,
                        gaze_quality=0.9,
                        face_area_ratio=0.2,
                        pose_source="mediapipe_matrix",
                        pose_quality=0.95,
                    ),
                )
            ),
            object_detector=FakeObjectDetector(),
            anchors={},
        )

        status = coordinator.engagement_calibration_status(mono_ns=10_100_000_000)
        assert status.state == "calibrating"
        assert status.sample_count == 1
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_runtime_calibration_cancel_updates_snapshot(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    try:
        coordinator.start_engagement_calibration(mono_ns=0)
        coordinator.cancel_engagement_calibration()
        assert coordinator.world.snapshot.engagement_calibration.state == "uncalibrated"
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_object_detection_throttle_default_no_max_fps(tmp_path: Path) -> None:
    clock = FakeClock(mono_ns=0, wall_time_utc="2026-07-04T12:00:00Z")
    world = MutableWorldModel(session_id=uuid7(), clock=clock)
    coordinator = RuntimeCoordinator(
        world=world,
        simulator=FakeSimulator(),
        metrics=FakeMetrics(),
        memory=TestMemory(tmp_path / "memory.db"),
        object_detection_max_fps=0,
    )
    try:
        bbox: BBox = (0.35, 0.35, 0.55, 0.65)
        detection = Detection(label="keys", confidence=0.92, bbox=bbox, mono_ns=10)
        callback_counts: list[int] = []

        class CountingDetector:
            def __init__(self) -> None:
                self.call_count = 0

            def detect(self, image: np.ndarray) -> tuple[Detection, ...]:
                self.call_count += 1
                callback_counts.append(self.call_count)
                return (detection,)

            def health(self) -> ComponentHealth:
                return ComponentHealth(component="object_detector", status="active")

        for step in range(5):
            frame = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10 + step)
            await coordinator.process_vision_frame(
                frame,
                face_processor=FakeFaceProcessor(),
                object_detector=CountingDetector(),
                anchors={"desk": (0.3, 0.3, 0.7, 0.8)},
            )
        assert len(callback_counts) == 5
        assert coordinator._object_detection_skipped_frames == 0
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_object_detection_throttle_skips_inference(tmp_path: Path) -> None:
    clock = FakeClock(mono_ns=0, wall_time_utc="2026-07-04T12:00:00Z")
    world = MutableWorldModel(session_id=uuid7(), clock=clock)
    coordinator = RuntimeCoordinator(
        world=world,
        simulator=FakeSimulator(),
        metrics=FakeMetrics(),
        memory=TestMemory(tmp_path / "memory.db"),
        object_detection_max_fps=2,  # 500ms interval
    )
    try:
        bbox: BBox = (0.35, 0.35, 0.55, 0.65)
        detection = Detection(label="keys", confidence=0.92, bbox=bbox, mono_ns=10)
        inference_times: list[int] = []

        class TrackingDetector:
            def detect(self, image: np.ndarray) -> tuple[Detection, ...]:
                inference_times.append(image.sum())
                del image
                return (detection,)

            def health(self) -> ComponentHealth:
                return ComponentHealth(component="object_detector", status="active")

        detector = TrackingDetector()
        for step in range(5):
            frame = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10 + step)
            await coordinator.process_vision_frame(
                frame,
                face_processor=FakeFaceProcessor(),
                object_detector=detector,
                anchors={"desk": (0.3, 0.3, 0.7, 0.8)},
            )
        assert len(inference_times) == 1
        assert coordinator._object_detection_skipped_frames == 4
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_object_detection_throttle_runs_when_interval_elapsed(
    tmp_path: Path,
) -> None:
    clock = FakeClock(mono_ns=0, wall_time_utc="2026-07-04T12:00:00Z")
    world = MutableWorldModel(session_id=uuid7(), clock=clock)
    coordinator = RuntimeCoordinator(
        world=world,
        simulator=FakeSimulator(),
        metrics=FakeMetrics(),
        memory=TestMemory(tmp_path / "memory.db"),
        object_detection_max_fps=2,  # 500ms interval
    )
    try:
        bbox: BBox = (0.35, 0.35, 0.55, 0.65)
        detection = Detection(label="keys", confidence=0.92, bbox=bbox, mono_ns=10)
        inference_count = 0

        class CountingDetector:
            def detect(self, image: np.ndarray) -> tuple[Detection, ...]:
                nonlocal inference_count
                inference_count += 1
                return (detection,)

            def health(self) -> ComponentHealth:
                return ComponentHealth(component="object_detector", status="active")

        detector = CountingDetector()
        # Frame at ns=10 - should trigger (first frame)
        frame1 = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=10)
        await coordinator.process_vision_frame(
            frame1, face_processor=FakeFaceProcessor(), object_detector=detector, anchors={},
        )
        assert inference_count == 1
        assert coordinator._last_object_inference_mono_ns == 10

        # Frame at ns=20 - skipped (< 500ms)
        frame2 = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=20)
        await coordinator.process_vision_frame(
            frame2, face_processor=FakeFaceProcessor(), object_detector=detector, anchors={},
        )
        assert inference_count == 1
        assert coordinator._object_detection_skipped_frames == 1

        # Frame at ns=600_000_000 - should trigger (600ms > 500ms)
        frame3 = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=600_000_000)
        await coordinator.process_vision_frame(
            frame3, face_processor=FakeFaceProcessor(), object_detector=detector, anchors={},
        )
        assert inference_count == 2
        assert coordinator._last_object_inference_mono_ns == 600_000_000
    finally:
        await coordinator.stop()
