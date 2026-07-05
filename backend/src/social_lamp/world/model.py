from uuid import UUID

from uuid6 import uuid7

from social_lamp.domain.clock import Clock
from social_lamp.domain.contracts import ObservationEvent, SocialState, WorldSnapshot


class WorldModel:
    def __init__(self, *, session_id: UUID, clock: Clock) -> None:
        self._clock = clock
        self._snapshot = WorldSnapshot.empty(session_id=session_id, mono_ns=clock.mono_ns)

    @property
    def snapshot(self) -> WorldSnapshot:
        return self._snapshot

    def replace(self, snapshot: WorldSnapshot) -> None:
        self._snapshot = snapshot

    def apply(self, event: ObservationEvent) -> WorldSnapshot:
        if event.kind != "social_state_changed":
            return self._snapshot
        state = SocialState(event.payload["state"])
        primary = event.payload.get("primary_person_id")
        if state is self._snapshot.social_state and primary == self._snapshot.primary_person_id:
            return self._snapshot
        self._snapshot = self._snapshot.model_copy(
            update={
                "snapshot_id": uuid7(),
                "revision": self._snapshot.revision + 1,
                "as_of_mono_ns": self._clock.mono_ns,
                "social_state": state,
                "primary_person_id": primary,
            }
        )
        return self._snapshot
