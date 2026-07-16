from uuid import UUID

from uuid6 import uuid7

from social_lamp.domain.clock import Clock
from social_lamp.domain.contracts import (
    ComponentHealth,
    ObservationEvent,
    PersonState,
    SocialState,
    WorldSnapshot,
)
from social_lamp.world.commands import AudioUpdate, HealthUpdate, VisionUpdate


def _replace_health(
    current: tuple[ComponentHealth, ...], item: ComponentHealth
) -> tuple[ComponentHealth, ...]:
    return tuple(existing for existing in current if existing.component != item.component) + (item,)


class WorldModel:
    def __init__(self, *, session_id: UUID, clock: Clock) -> None:
        self._clock = clock
        self._snapshot = WorldSnapshot.empty(session_id=session_id, mono_ns=clock.mono_ns)
        self._last_vision_ns: int = 0
        self._last_audio_ns: int = 0
        self._last_health_ns: dict[str, int] = {}
        self._last_positive_vision_ns: int = 0

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

    def apply_health(self, update: HealthUpdate) -> WorldSnapshot | None:
        last_ns = self._last_health_ns.get(update.component, 0)
        if update.as_of_mono_ns < last_ns:
            return None

        for existing in self._snapshot.health:
            if (
                existing.component == update.component
                and existing.status == update.status
                and existing.detail == update.detail
            ):
                return None

        new_health = _replace_health(
            self._snapshot.health,
            ComponentHealth(component=update.component, status=update.status, detail=update.detail),
        )
        self._snapshot = self._snapshot.model_copy(
            update={
                "snapshot_id": uuid7(),
                "revision": self._snapshot.revision + 1,
                "as_of_mono_ns": update.as_of_mono_ns,
                "health": new_health,
            }
        )
        self._last_health_ns[update.component] = update.as_of_mono_ns
        return self._snapshot

    def apply_vision(self, update: VisionUpdate) -> WorldSnapshot | None:
        if update.as_of_mono_ns < self._last_vision_ns:
            return None

        is_positive = len(update.people) > 0
        if not is_positive and self._last_positive_vision_ns > update.as_of_mono_ns:
            return None

        new_health = self._snapshot.health
        for hu in update.health_updates:
            new_health = _replace_health(new_health, hu)

        objects = update.objects if update.objects is not None else self._snapshot.objects

        candidate = self._snapshot.model_copy(
            update={
                "snapshot_id": uuid7(),
                "revision": self._snapshot.revision + 1,
                "as_of_mono_ns": update.as_of_mono_ns,
                "social_state": update.social_state,
                "primary_person_id": update.primary_person_id,
                "people": update.people,
                "objects": objects,
                "health": new_health,
            }
        )

        if (
            candidate.people == self._snapshot.people
            and candidate.social_state == self._snapshot.social_state
            and candidate.primary_person_id == self._snapshot.primary_person_id
            and candidate.objects == self._snapshot.objects
            and candidate.health == self._snapshot.health
        ):
            return None

        self._last_vision_ns = update.as_of_mono_ns
        if is_positive:
            self._last_positive_vision_ns = update.as_of_mono_ns
        self._snapshot = candidate
        return candidate

    def apply_audio(self, update: AudioUpdate) -> WorldSnapshot | None:
        if update.as_of_mono_ns < self._last_audio_ns:
            return None

        new_health = self._snapshot.health
        if update.health_update is not None:
            new_health = _replace_health(new_health, update.health_update)

        if update.active_speaker_id is not None and self._snapshot.people:
            new_people = tuple(
                person.model_copy(
                    update={"is_active_speaker": person.person_id == update.active_speaker_id}
                )
                for person in self._snapshot.people
            )
        elif update.active_speaker_id is not None:
            new_people = (
                PersonState(
                    person_id=update.active_speaker_id,
                    engagement_score=0.0,
                    engagement_confidence=0.0,
                    is_active_speaker=True,
                ),
            )
        else:
            new_people = tuple(
                person.model_copy(update={"is_active_speaker": False})
                for person in self._snapshot.people
            )

        candidate = self._snapshot.model_copy(
            update={
                "snapshot_id": uuid7(),
                "revision": self._snapshot.revision + 1,
                "as_of_mono_ns": update.as_of_mono_ns,
                "audio_mode": update.audio_mode,
                "people": new_people,
                "health": new_health,
            }
        )

        if (
            candidate.audio_mode == self._snapshot.audio_mode
            and candidate.people == self._snapshot.people
            and candidate.health == self._snapshot.health
        ):
            return None

        self._last_audio_ns = update.as_of_mono_ns
        self._snapshot = candidate
        return candidate

    def replace_from_replay(self, snapshot: WorldSnapshot) -> WorldSnapshot:
        self._snapshot = snapshot
        return snapshot
