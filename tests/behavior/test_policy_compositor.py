from uuid import UUID

from social_lamp.behavior.compositor import BehaviorCompositor
from social_lamp.behavior.policy import BehaviorPolicy
from social_lamp.domain.contracts import SocialState, WorldSnapshot


def test_engagement_produces_acknowledge_timeline() -> None:
    session_id = UUID("018f0000-0000-7000-8000-000000000001")
    previous = WorldSnapshot.empty(session_id=session_id, mono_ns=0)
    current = previous.model_copy(
        update={"revision": 1, "social_state": SocialState.ENGAGED, "as_of_mono_ns": 1}
    )
    intent = BehaviorPolicy().on_transition(previous, current)
    assert intent is not None
    assert intent.kind == "acknowledge"
    timeline = BehaviorCompositor().compose(intent, current_pose={})
    assert timeline.intent_id == intent.intent_id
    assert {track.channel for track in timeline.motion_tracks} == {
        "head_yaw",
        "head_pitch",
    }
    assert timeline.duration_ms == 700
