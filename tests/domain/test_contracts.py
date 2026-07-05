from uuid import UUID

import pytest
from pydantic import ValidationError
from social_lamp.domain.clock import FakeClock
from social_lamp.domain.contracts import (
    ObservationEvent,
    ObservationSource,
    SocialState,
    WorldSnapshot,
)


def test_observation_rejects_invalid_confidence() -> None:
    clock = FakeClock(mono_ns=10, wall_time_utc="2026-07-04T12:00:00Z")
    with pytest.raises(ValidationError):
        ObservationEvent.create(
            clock=clock,
            session_id=UUID("018f0000-0000-7000-8000-000000000001"),
            correlation_id=UUID("018f0000-0000-7000-8000-000000000002"),
            source=ObservationSource.FACE,
            kind="face_presence",
            confidence=1.1,
            payload={"person_id": "person-a"},
        )


def test_world_snapshot_is_immutable() -> None:
    snapshot = WorldSnapshot.empty(
        session_id=UUID("018f0000-0000-7000-8000-000000000001"), mono_ns=10
    )
    assert snapshot.social_state is SocialState.IDLE
    with pytest.raises(ValidationError):
        snapshot.social_state = SocialState.ENGAGED
