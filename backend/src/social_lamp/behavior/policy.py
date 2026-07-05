from dataclasses import dataclass

from uuid6 import uuid7

from social_lamp.domain.contracts import BehaviorIntent, SocialState, WorldSnapshot


@dataclass(frozen=True)
class AttentionIntent:
    parameters: dict[str, int]


class AttentionSchedule:
    OFFSETS_NS = (5_000_000_000, 15_000_000_000, 35_000_000_000)
    RESTART_COOLDOWN_NS = 60_000_000_000

    def __init__(self, *, disengaged_at_ns: int) -> None:
        self._start = disengaged_at_ns
        self._emitted = 0
        self.exhausted = False
        self.suppression_reason: str | None = None
        self._exhausted_at_ns: int | None = None

    def suppress(self, reason: str, *, mono_ns: int | None = None) -> None:
        self.suppression_reason = reason
        self.exhausted = True
        self._exhausted_at_ns = self._start if mono_ns is None else mono_ns

    def intent_at(self, mono_ns: int) -> AttentionIntent | None:
        if self.exhausted or self._emitted >= len(self.OFFSETS_NS):
            return None
        due = self._start + self.OFFSETS_NS[self._emitted]
        if mono_ns < due:
            return None
        self._emitted += 1
        level = self._emitted
        if level == len(self.OFFSETS_NS):
            self.exhausted = True
            self._exhausted_at_ns = mono_ns
        return AttentionIntent(parameters={"level": level})

    def can_restart_at(self, mono_ns: int) -> bool:
        if not self.exhausted:
            return False
        exhausted_at = self._exhausted_at_ns if self._exhausted_at_ns is not None else self._start
        return mono_ns >= exhausted_at + self.RESTART_COOLDOWN_NS


class BehaviorPolicy:
    def on_transition(
        self, previous: WorldSnapshot, current: WorldSnapshot
    ) -> BehaviorIntent | None:
        if previous.social_state is current.social_state:
            return None
        mapping = {
            SocialState.ENGAGED: ("acknowledge", 60),
            SocialState.DISENGAGED: ("disengage", 60),
            SocialState.SEEKING_ATTENTION: ("seek_attention", 40),
            SocialState.IDLE: ("return_neutral", 20),
        }
        selected = mapping.get(current.social_state)
        if selected is None:
            return None
        kind, urgency = selected
        return BehaviorIntent(
            intent_id=uuid7(),
            correlation_id=uuid7(),
            session_id=current.session_id,
            kind=kind,
            urgency=urgency,
            created_at_mono_ns=current.as_of_mono_ns,
            expires_at_mono_ns=current.as_of_mono_ns + 2_000_000_000,
            target_person_id=current.primary_person_id,
        )
