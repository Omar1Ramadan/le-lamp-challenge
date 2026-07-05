from uuid import UUID

from social_lamp.domain.clock import FakeClock
from social_lamp.domain.contracts import ObservationEvent, ObservationSource, SocialState
from social_lamp.world.model import WorldModel


def test_world_model_advances_revision_only_on_stable_change() -> None:
    clock = FakeClock(0, "2026-07-04T12:00:00Z")
    session_id = UUID("018f0000-0000-7000-8000-000000000001")
    correlation_id = UUID("018f0000-0000-7000-8000-000000000002")
    model = WorldModel(session_id=session_id, clock=clock)
    event = ObservationEvent.create(
        clock=clock,
        session_id=session_id,
        correlation_id=correlation_id,
        source=ObservationSource.SYSTEM,
        kind="social_state_changed",
        confidence=0.9,
        payload={"state": "engaged", "primary_person_id": "person-a"},
    )
    changed = model.apply(event)
    unchanged = model.apply(event)
    assert changed.social_state is SocialState.ENGAGED
    assert changed.revision == 1
    assert unchanged.revision == 1
