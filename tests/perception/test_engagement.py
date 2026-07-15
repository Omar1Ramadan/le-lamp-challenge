from hypothesis import given
from hypothesis import strategies as st
from social_lamp.domain.contracts import SocialState
from social_lamp.perception.engagement import EngagementEstimator, EngagementSignals
from social_lamp.perception.faces import FaceResult, face_result_to_signals


def _calibration_face(
    *,
    confidence: float = 0.92,
    yaw: float = 2.0,
    pitch: float = 4.0,
    roll: float = 1.0,
    gaze: float = 0.75,
    gaze_quality: float = 0.9,
    area: float = 0.18,
) -> FaceResult:
    return FaceResult(
        face_confidence=confidence,
        yaw_degrees=yaw,
        pitch_degrees=pitch,
        roll_degrees=roll,
        gaze_score=gaze,
        gaze_quality=gaze_quality,
        face_area_ratio=area,
        pose_source="mediapipe_matrix",
        pose_quality=0.95,
    )


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


def test_calibration_starts_and_cancels() -> None:
    estimator = EngagementEstimator()
    status = estimator.start_calibration("person-1", mono_ns=100)
    assert status.state == "calibrating"
    assert status.person_id == "person-1"
    assert status.sample_count == 0
    assert status.progress == 0.0

    cancelled = estimator.cancel_calibration()
    assert cancelled.state == "uncalibrated"
    assert cancelled.person_id is None
    assert cancelled.mode == "fallback"


def test_calibration_fails_without_enough_samples_after_window() -> None:
    estimator = EngagementEstimator()
    estimator.start_calibration("person-1", mono_ns=0)
    estimator.observe_calibration(_calibration_face(), mono_ns=0)

    status = estimator.calibration_status(mono_ns=3_000_000_000)
    assert status.state == "failed"
    assert status.sample_count == 1
    assert status.quality == "failed"
    assert status.failure_reason == "not enough valid samples"
    assert status.mode == "fallback"


def test_calibration_completes_with_median_baselines() -> None:
    estimator = EngagementEstimator()
    estimator.start_calibration("person-1", mono_ns=0)
    for index in range(20):
        estimator.observe_calibration(
            _calibration_face(
                yaw=10.0 if index == 19 else 2.0,
                pitch=5.0,
                roll=1.0,
                area=0.18,
                gaze=0.7,
            ),
            mono_ns=index * 100_000_000,
        )

    status = estimator.calibration_status(mono_ns=3_000_000_000)
    assert status.state == "calibrated"
    assert status.sample_count == 20
    assert status.quality == "good"
    assert status.mode == "calibrated"
    assert estimator._calibration.neutral_yaw == 2.0
    assert estimator._calibration.neutral_pitch == 5.0
    assert estimator._calibration.neutral_face_scale == 0.18


def test_calibration_ignores_low_confidence_samples() -> None:
    estimator = EngagementEstimator()
    estimator.start_calibration("person-1", mono_ns=0)
    for index in range(25):
        estimator.observe_calibration(
            _calibration_face(confidence=0.2),
            mono_ns=index * 100_000_000,
        )

    status = estimator.calibration_status(mono_ns=3_000_000_000)
    assert status.state == "failed"
    assert status.sample_count == 0
    assert status.failure_reason == "not enough valid samples"


def test_calibration_high_pose_variance_fails() -> None:
    estimator = EngagementEstimator()
    estimator.start_calibration("person-1", mono_ns=0)
    for index in range(20):
        yaw = -35.0 if index % 2 == 0 else 35.0
        estimator.observe_calibration(
            _calibration_face(yaw=yaw),
            mono_ns=index * 100_000_000,
        )

    status = estimator.calibration_status(mono_ns=3_000_000_000)
    assert status.state == "failed"
    assert status.quality == "failed"
    assert status.failure_reason == "pose moved too much"


def _completed_calibrator() -> EngagementEstimator:
    estimator = EngagementEstimator()
    estimator.start_calibration("person-1", mono_ns=0)
    for index in range(20):
        estimator.observe_calibration(_calibration_face(yaw=20.0, pitch=8.0, area=0.2), index)
    estimator.calibration_status(mono_ns=3_000_000_000)
    return estimator


def test_calibrated_head_pose_uses_neutral_offsets() -> None:
    estimator = _completed_calibrator()
    face = _calibration_face(yaw=20.0, pitch=8.0, area=0.2)
    calibrated = face_result_to_signals(face, calibration=estimator._calibration)
    fallback = face_result_to_signals(face)

    assert calibrated.head_toward == 1.0
    assert fallback.head_toward is not None
    assert fallback.head_toward < 1.0
    assert calibrated.confidence == 0.9900000000000001


def test_calibrated_proximity_uses_neutral_face_scale() -> None:
    estimator = _completed_calibrator()
    face = _calibration_face(yaw=20.0, pitch=8.0, area=0.1)
    signals = face_result_to_signals(face, calibration=estimator._calibration)

    assert signals.proximity == 0.5


def test_calibration_without_gaze_reports_partial_mode() -> None:
    estimator = EngagementEstimator()
    estimator.start_calibration("person-1", mono_ns=0)
    for index in range(20):
        estimator.observe_calibration(_calibration_face(gaze_quality=0.1), index)
    status = estimator.calibration_status(mono_ns=3_000_000_000)

    assert status.state == "calibrated"
    assert status.mode == "partial"
