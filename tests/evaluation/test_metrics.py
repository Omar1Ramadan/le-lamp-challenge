from social_lamp.evaluation.metrics import (
    ClassificationCounts,
    evaluate_gates,
    percentile,
)


def test_classification_metrics_exclude_ambiguous_labels() -> None:
    counts = ClassificationCounts.from_pairs(
        [
            ("engaged", "engaged"),
            ("engaged", "not_engaged"),
            ("not_engaged", "not_engaged"),
            ("ambiguous", "engaged"),
        ]
    )
    assert counts.true_positive == 1
    assert counts.false_negative == 1
    assert counts.true_negative == 1
    assert counts.f1 == 2 / 3


def test_latency_uses_nearest_rank_p95_and_gates_fail_closed() -> None:
    assert percentile(list(range(1, 101)), 0.95) == 95
    result = evaluate_gates(
        engagement_f1=0.84,
        false_transitions_per_two_minutes=0.0,
        frame_to_observation_p95_ms=100,
        state_to_visible_p95_ms=100,
        memory_accuracy=0.95,
        grounding_rate=1.0,
        max_normal_frame_age_ms=200,
    )
    assert not result.passed
    assert result.failures == ("engagement_f1",)


def test_gate_passes_when_all_thresholds_are_met() -> None:
    result = evaluate_gates(
        engagement_f1=0.85,
        false_transitions_per_two_minutes=1.0,
        frame_to_observation_p95_ms=200,
        state_to_visible_p95_ms=150,
        memory_accuracy=0.90,
        grounding_rate=1.0,
        max_normal_frame_age_ms=299,
    )
    assert result.passed
    assert result.failures == ()
