from hypothesis import given
from hypothesis import strategies as st
from social_lamp.domain.contracts import SocialState
from social_lamp.perception.engagement import EngagementEstimator, EngagementSignals


def test_missing_gaze_renormalizes_available_signals() -> None:
    estimator = EngagementEstimator()
    sample = estimator.sample(
        EngagementSignals(
            face_presence=1.0,
            head_toward=1.0,
            gaze_toward=None,
            proximity=1.0,
            directed_speech=0.0,
            confidence=0.9,
        ),
        mono_ns=0,
    )
    assert 0.79 < sample.raw_score < 0.81


def test_engagement_requires_entry_dwell_and_exit_hysteresis() -> None:
    estimator = EngagementEstimator(smoothing_ms=0)
    attentive = EngagementSignals(1.0, 1.0, 1.0, 1.0, 0.0, 0.9)
    away = EngagementSignals(1.0, 0.0, 0.0, 1.0, 0.0, 0.9)
    assert estimator.sample(attentive, 0).state is SocialState.CANDIDATE
    assert estimator.sample(attentive, 699_000_000).state is SocialState.CANDIDATE
    assert estimator.sample(attentive, 700_000_000).state is SocialState.ENGAGED
    assert estimator.sample(away, 1_899_000_000).state is SocialState.ENGAGED
    assert estimator.sample(away, 1_900_000_000).state is SocialState.DISENGAGED


maybe_signal = st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0))


def test_heuristic_confidence_does_not_transition_to_engaged() -> None:
    estimator = EngagementEstimator(smoothing_ms=0)
    heuristic = EngagementSignals(
        face_presence=0.45,
        head_toward=None,
        gaze_toward=None,
        proximity=0.4,
        directed_speech=0.0,
        confidence=0.35,
    )
    for mono_ns in range(0, 2_000_000_000, 100_000_000):
        sample = estimator.sample(heuristic, mono_ns)
        assert sample.state is not SocialState.ENGAGED


@given(
    face_presence=maybe_signal,
    head_toward=maybe_signal,
    gaze_toward=maybe_signal,
    proximity=maybe_signal,
    directed_speech=maybe_signal,
    confidence=st.floats(min_value=0.0, max_value=1.0),
)
def test_scores_stay_in_unit_range(
    face_presence: float | None,
    head_toward: float | None,
    gaze_toward: float | None,
    proximity: float | None,
    directed_speech: float | None,
    confidence: float,
) -> None:
    estimator = EngagementEstimator(smoothing_ms=0)
    sample = estimator.sample(
        EngagementSignals(
            face_presence,
            head_toward,
            gaze_toward,
            proximity,
            directed_speech,
            confidence,
        ),
        mono_ns=0,
    )
    assert 0.0 <= sample.raw_score <= 1.0
    assert 0.0 <= sample.smoothed_score <= 1.0
