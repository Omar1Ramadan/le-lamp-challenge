from uuid6 import uuid7

from social_lamp.domain.contracts import BehaviorIntent, SocialState, WorldSnapshot


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
