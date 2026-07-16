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
    decision = BehaviorPolicy().evaluate(current, previous.social_state)
    assert decision.intent is not None
    assert not decision.suppressed
    assert decision.intent.kind == "acknowledge_engagement"
    timeline = BehaviorCompositor().compose(decision.intent, current_pose={})
    assert timeline.intent_id == decision.intent.intent_id
    assert {track.channel for track in timeline.motion_tracks} == {
        "base_yaw",
        "head_yaw",
        "head_pitch",
    }
    assert timeline.duration_ms == 900


def test_candidate_attention_produces_orient_timeline() -> None:
    session_id = UUID("018f0000-0000-7000-8000-000000000001")
    previous = WorldSnapshot.empty(session_id=session_id, mono_ns=0)
    current = previous.model_copy(
        update={"revision": 1, "social_state": SocialState.CANDIDATE, "as_of_mono_ns": 1}
    )
    decision = BehaviorPolicy().evaluate(current, previous.social_state)
    assert decision.intent is not None
    assert decision.intent.kind == "orient"
    timeline = BehaviorCompositor().compose(decision.intent, current_pose={})
    assert timeline.duration_ms == 800
    assert {track.channel for track in timeline.motion_tracks} == {
        "base_yaw",
        "head_yaw",
        "head_pitch",
    }


def test_idle_transition_from_engaged_uses_idle_settle() -> None:
    session_id = UUID("018f0000-0000-7000-8000-000000000001")
    previous = WorldSnapshot.empty(session_id=session_id, mono_ns=0)
    current = previous.model_copy(
        update={"revision": 1, "social_state": SocialState.IDLE, "as_of_mono_ns": 1}
    )
    decision = BehaviorPolicy().evaluate(
        current, previous_social_state=SocialState.ENGAGED
    )
    assert decision.intent is not None
    assert decision.intent.kind == "idle_settle"


def test_no_transition_returns_no_intent() -> None:
    session_id = UUID("018f0000-0000-7000-8000-000000000001")
    current = WorldSnapshot.empty(session_id=session_id, mono_ns=0)
    decision = BehaviorPolicy().evaluate(current, previous_social_state=None)
    assert decision.suppressed
    assert decision.suppression_reason == "no_candidate_intent"


def test_attention_suppressed_when_no_visible_person() -> None:
    session_id = UUID("018f0000-0000-7000-8000-000000000001")
    current = WorldSnapshot.empty(session_id=session_id, mono_ns=0)
    policy = BehaviorPolicy()
    policy.reset_idle_timer(0)
    decision = policy.evaluate(
        current,
        previous_social_state=None,
        mono_ns=31_000_000_000,
        has_visible_person=False,
    )
    assert decision.suppressed
    assert decision.suppression_reason is not None


def test_attention_seek_after_idle_threshold() -> None:
    session_id = UUID("018f0000-0000-7000-8000-000000000001")
    current = WorldSnapshot.empty(session_id=session_id, mono_ns=0)
    policy = BehaviorPolicy()
    policy.reset_idle_timer(0)
    decision = policy.evaluate(
        current,
        previous_social_state=None,
        mono_ns=31_000_000_000,
        has_visible_person=True,
    )
    assert decision.intent is not None
    assert decision.intent.kind == "attention_seek"


def test_higher_priority_replaces_lower() -> None:
    session_id = UUID("018f0000-0000-7000-8000-000000000001")
    current = WorldSnapshot.empty(session_id=session_id, mono_ns=0)
    policy = BehaviorPolicy()
    decision = policy.evaluate(
        current,
        previous_social_state=None,
        mono_ns=0,
        active_timeline_id="timeline-1",
        active_timeline_kind="idle_settle",
        active_timeline_priority=10,
        active_timeline_cancellable=True,
    )
    assert decision.suppressed
    assert decision.suppression_reason == "no_candidate_intent"


def test_suppression_reason_is_emitted() -> None:
    session_id = UUID("018f0000-0000-7000-8000-000000000001")
    previous = WorldSnapshot.empty(session_id=session_id, mono_ns=0)
    current = previous.model_copy(
        update={"revision": 1, "social_state": SocialState.ENGAGED, "as_of_mono_ns": 1}
    )
    policy = BehaviorPolicy()
    _ = policy.evaluate(
        current,
        previous.social_state,
        audio_suppressed=True,
    )
    assert len(policy.decisions) >= 1
    last = policy.decisions[-1]
    assert last.suppressed
    assert last.suppression_reason is not None
