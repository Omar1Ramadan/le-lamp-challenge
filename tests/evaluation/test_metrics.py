from social_lamp.evaluation.metrics import (
    ClassificationCounts,
    ConfusionMatrix,
    evaluate_engagement,
    evaluate_gates,
    evaluate_grounding,
    evaluate_latency,
    evaluate_memory,
    evaluate_transitions,
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
        false_transitions_per_minute=0.0,
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
        false_transitions_per_minute=0.5,
        frame_to_observation_p95_ms=200,
        state_to_visible_p95_ms=150,
        memory_accuracy=0.90,
        grounding_rate=1.0,
        max_normal_frame_age_ms=299,
    )
    assert result.passed
    assert result.failures == ()


def test_engagement_metrics_perfect_match() -> None:
    observed = [(0, "idle"), (100, "engaged"), (500, "engaged"), (1000, "disengaged")]
    segments = [(100, 900, "engaged")]
    result = evaluate_engagement(observed, segments)
    assert result["f1"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_engagement_metrics_no_match() -> None:
    observed = [(0, "disengaged"), (100, "disengaged")]
    segments = [(50, 200, "engaged")]
    result = evaluate_engagement(observed, segments)
    assert result["f1"] == 0.0


def test_engagement_metrics_partial_match() -> None:
    observed = [(0, "engaged")]
    segments = [(0, 1000, "engaged")]
    result = evaluate_engagement(observed, segments)
    assert result["f1"] == 1.0


def test_engagement_metrics_empty_segments() -> None:
    observed = [(0, "idle"), (100, "engaged")]
    result = evaluate_engagement(observed, [])
    assert result["f1"] == 0.0
    assert result["support"] == 0


def test_transition_metrics_perfect_match() -> None:
    observed = [(100, "idle", "candidate"), (500, "candidate", "engaged")]
    expected = [(100, "idle", "candidate", 200), (500, "candidate", "engaged", 200)]
    result = evaluate_transitions(observed, expected, duration_ms=2000)
    assert result["true_positives"] == 2
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0


def test_transition_metrics_false_positive() -> None:
    observed = [
        (100, "idle", "candidate"), (300, "candidate", "engaged"),
        (600, "engaged", "disengaged"),
    ]
    expected = [(100, "idle", "candidate", 200), (300, "candidate", "engaged", 200)]
    result = evaluate_transitions(observed, expected, duration_ms=2000)
    assert result["true_positives"] == 2
    assert result["false_positives"] == 1  # extra transition to disengaged
    assert result["false_negatives"] == 0


def test_transition_metrics_false_negative() -> None:
    observed = [(100, "idle", "candidate")]
    expected = [(100, "idle", "candidate", 200), (500, "candidate", "engaged", 200)]
    result = evaluate_transitions(observed, expected, duration_ms=2000)
    assert result["true_positives"] == 1
    assert result["false_negatives"] == 1  # missed engaged transition


def test_transition_metrics_timing_error() -> None:
    observed = [(300, "idle", "candidate")]
    expected = [(100, "idle", "candidate", 100)]
    result = evaluate_transitions(observed, expected, duration_ms=2000)
    # 300-100=200 > 100 tolerance so this is a false positive
    assert result["true_positives"] == 0
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1


def test_transition_metrics_false_positive_rate() -> None:
    observed = [
        (100, "idle", "candidate"), (300, "candidate", "engaged"),
        (500, "engaged", "disengaged"),
    ]
    expected = [(100, "idle", "candidate", 200), (300, "candidate", "engaged", 200)]
    result = evaluate_transitions(observed, expected, duration_ms=60_000)
    # 1 FP over 60s = 1.0 per minute
    assert result["false_transitions_per_minute"] == 1.0


def test_latency_metrics() -> None:
    input_events = [(0, "observation"), (100, "snapshot")]
    output_events = [(10, "observation"), (120, "snapshot")]
    result = evaluate_latency(input_events, output_events)
    assert result["p50_ms"] == 10.0
    assert result["missing_output_count"] == 0


def test_latency_metrics_missing_output() -> None:
    result = evaluate_latency([(0, "observation")], [])
    assert result["missing_output_count"] == 1
    assert result["max_ms"] == 0.0


def test_memory_metrics_perfect() -> None:
    observed = [
        {"type": "memory_result", "label": "keys", "location": "desk"},
        {"type": "object_seen", "label": "keys", "location": "desk"},
    ]
    expected = [
        ("memory_result", "keys", [0, 10000], "desk"),
        ("object_seen", "keys", [0, 10000], "desk"),
    ]
    result = evaluate_memory(observed, expected)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["extra_memories"] == 0


def test_memory_metrics_extra() -> None:
    observed = [
        {"type": "memory_result", "label": "keys", "location": "desk"},
        {"type": "object_seen", "label": "cup", "location": "table"},
    ]
    expected = [
        ("memory_result", "keys", [0, 10000], "desk"),
    ]
    result = evaluate_memory(observed, expected)
    assert result["precision"] == 0.5  # one correct out of two observed
    assert result["recall"] == 1.0
    assert result["f1"] == 2 / 3
    assert result["extra_memories"] == 1


def test_memory_metrics_missing() -> None:
    observed: list[dict] = []
    expected = [("memory_result", "keys", [0, 10000], "desk")]
    result = evaluate_memory(observed, expected)
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    assert result["false_negatives"] == 1


def test_grounding_metrics_evidenced() -> None:
    observed = [
        {
            "query": "Where is my cup?",
            "text": "Your cup is on the table.",
            "evidence_types": ["object_seen"],
            "memory_labels": ["cup"],
        }
    ]
    expected = [("Where is my cup?", ["table"], ["object_seen"], ["cup"])]
    result = evaluate_grounding(observed, expected)
    assert result["grounded_rate"] == 1.0
    assert result["grounded_count"] == 1
    assert result["unsupported_count"] == 0


def test_grounding_metrics_unsupported() -> None:
    observed = [
        {
            "query": "Where is my cup?",
            "text": "I don't know.",
            "evidence_types": [],
            "memory_labels": [],
        }
    ]
    expected = [("Where is my cup?", ["table"], ["object_seen"], ["cup"])]
    result = evaluate_grounding(observed, expected)
    assert result["grounded_rate"] == 0.0
    assert result["grounded_count"] == 0
    assert result["unsupported_count"] == 1


def test_grounding_metrics_no_observed_answer() -> None:
    result = evaluate_grounding([], [("Where is my cup?", ["table"], ["object_seen"], ["cup"])])
    assert result["grounded_rate"] == 0.0
    assert result["unsupported_count"] == 1


def test_confusion_matrix_accumulates() -> None:
    cm = ConfusionMatrix()
    cm.engaged_engaged = 10
    cm.engaged_disengaged = 2
    cm.disengaged_disengaged = 5
    assert cm.accuracy == 15 / 17
