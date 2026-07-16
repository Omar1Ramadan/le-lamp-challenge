from pathlib import Path

import numpy as np
import pytest

from social_lamp.capture.frames import CapturedFrame
from social_lamp.domain.contracts import ComponentHealth, SocialState
from social_lamp.perception.faces import FaceResult
from social_lamp.runtime.coordinator import RuntimeCoordinator
from social_lamp.runtime.testing import (
    FakeEvidencePublisher,
    FakeMetrics,
    FakeSimulator,
    MutableWorldModel,
    TestMemory,
)


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
    def health(self):
        return ComponentHealth(component="object_detector", status="disabled")


def _make_test_coordinator(tmp_path: Path, captured: FakeEvidencePublisher) -> RuntimeCoordinator:
    from social_lamp.domain.clock import FakeClock
    from uuid6 import uuid7

    clock = FakeClock(200_000_000, "2026-07-04T12:00:00Z")
    world = MutableWorldModel(session_id=uuid7(), clock=clock)
    return RuntimeCoordinator(
        world=world,
        simulator=FakeSimulator(),
        metrics=FakeMetrics(),
        memory=TestMemory(tmp_path / "memory.db"),
        evidence_publisher=captured,
    )


@pytest.mark.asyncio
async def test_engagement_transition_emits_event(tmp_path: Path) -> None:
    captured = FakeEvidencePublisher()
    coordinator = _make_test_coordinator(tmp_path, captured)
    coordinator._previous_social_state_for_event = SocialState.IDLE

    face = FaceResult(
        face_confidence=0.9, yaw_degrees=0.0, pitch_degrees=0.0,
        gaze_score=0.8, gaze_quality=0.9, face_area_ratio=0.12,
        pose_source="mediapipe_matrix", pose_quality=0.95,
    )
    frame = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=200_000_000)
    await coordinator.process_vision_frame(
        frame,
        face_processor=_FakeFaceProcessor((face,)),
        object_detector=_FakeObjectDetector(()),
        anchors={},
    )

    engagement_events = [e for e in captured.events if e["event_type"] == "engagement_transition"]
    assert len(engagement_events) >= 1
    assert "metadata" in engagement_events[0]


@pytest.mark.asyncio
async def test_behavior_events_during_vision_frame(tmp_path: Path) -> None:
    captured = FakeEvidencePublisher()
    coordinator = _make_test_coordinator(tmp_path, captured)
    coordinator._previous_social_state_for_event = SocialState.IDLE

    face = FaceResult(
        face_confidence=0.9, yaw_degrees=0.0, pitch_degrees=0.0,
        gaze_score=0.8, gaze_quality=0.9, face_area_ratio=0.12,
        pose_source="mediapipe_matrix", pose_quality=0.95,
    )
    frame = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=200_000_000)
    await coordinator.process_vision_frame(
        frame,
        face_processor=_FakeFaceProcessor((face,)),
        object_detector=_FakeObjectDetector(()),
        anchors={},
    )

    types = {e["event_type"] for e in captured.events}
    assert "behavior_selected" in types or "behavior_suppressed" in types
    assert "engagement_transition" in types


@pytest.mark.asyncio
async def test_behavior_suppressed_event_includes_reason(tmp_path: Path) -> None:
    captured = FakeEvidencePublisher()
    coordinator = _make_test_coordinator(tmp_path, captured)
    coordinator._previous_social_state_for_event = SocialState.ENGAGED

    face = FaceResult(
        face_confidence=0.9, yaw_degrees=0.0, pitch_degrees=0.0,
        gaze_score=0.8, gaze_quality=0.9, face_area_ratio=0.12,
        pose_source="mediapipe_matrix", pose_quality=0.95,
    )
    frame = CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=200_000_000)
    await coordinator.process_vision_frame(
        frame,
        face_processor=_FakeFaceProcessor((face,)),
        object_detector=_FakeObjectDetector(()),
        anchors={},
    )

    suppressed = [e for e in captured.events if e["event_type"] == "behavior_suppressed"]
    if suppressed:
        meta = suppressed[0].get("metadata", {})
        assert "reason" in meta
