from __future__ import annotations

import statistics
from dataclasses import dataclass
from math import ceil
from typing import Any


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, ceil(quantile * len(ordered)) - 1)
    return ordered[index]


@dataclass(frozen=True)
class ClassificationCounts:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @classmethod
    def from_pairs(cls, pairs: list[tuple[str, str]]) -> ClassificationCounts:
        filtered = [(truth, prediction) for truth, prediction in pairs if truth != "ambiguous"]
        return cls(
            sum(t == "engaged" and p == "engaged" for t, p in filtered),
            sum(t == "not_engaged" and p == "engaged" for t, p in filtered),
            sum(t == "not_engaged" and p == "not_engaged" for t, p in filtered),
            sum(t == "engaged" and p == "not_engaged" for t, p in filtered),
        )

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0


@dataclass
class ConfusionMatrix:
    engaged_engaged: int = 0
    engaged_disengaged: int = 0
    engaged_candidate: int = 0
    engaged_idle: int = 0
    disengaged_engaged: int = 0
    disengaged_disengaged: int = 0
    disengaged_candidate: int = 0
    disengaged_idle: int = 0
    candidate_engaged: int = 0
    candidate_disengaged: int = 0
    candidate_candidate: int = 0
    candidate_idle: int = 0
    idle_engaged: int = 0
    idle_disengaged: int = 0
    idle_candidate: int = 0
    idle_idle: int = 0

    def by_class(self, truth: str, prediction: str) -> int:
        return getattr(self, f"{truth}_{prediction}", 0)

    @property
    def accuracy(self) -> float:
        correct = (
            self.engaged_engaged
            + self.disengaged_disengaged
            + self.candidate_candidate
            + self.idle_idle
        )
        total = sum(self.__dict__.values())
        return correct / total if total else 0.0


def _binary_engaged(state: str) -> str:
    return "engaged" if state == "engaged" else "not_engaged"


def evaluate_engagement(
    observed_states: list[tuple[int, str]],
    segments: list[tuple[int, int, str]],
    interval_ms: int = 250,
) -> dict[str, Any]:
    max_time = max(
        max((s[1] for s in segments), default=0),
        max((t for t, _ in observed_states), default=0),
    )
    if max_time == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "support": 0,
            "confusion_matrix": {},
        }

    # Evaluate at a common grid of timestamps
    grid = list(range(0, max_time + interval_ms, interval_ms))

    # Label at each grid point
    def _label_at(t: int) -> str | None:
        for start_ms, end_ms, state in segments:
            if start_ms <= t < end_ms:
                return state
        return None

    # Observed state at each grid point (hold-last interpolation)
    def _observed_at(t: int) -> str | None:
        last: str | None = None
        for obs_t, obs_s in observed_states:
            if obs_t > t:
                break
            last = obs_s
        return last

    pairs: list[tuple[str, str]] = []
    for t in grid:
        truth = _label_at(t)
        pred = _observed_at(t)
        if truth is not None and pred is not None:
            pairs.append((_binary_engaged(truth), _binary_engaged(pred)))

    counts = ClassificationCounts.from_pairs(pairs)

    return {
        "precision": counts.precision,
        "recall": counts.recall,
        "f1": counts.f1,
        "support": counts.true_positive + counts.false_negative,
        "confusion_matrix": {
            "engaged_engaged": counts.true_positive,
            "engaged_not_engaged": counts.false_positive,
            "not_engaged_engaged": counts.false_negative,
            "not_engaged_not_engaged": counts.true_negative,
        },
    }


def evaluate_transitions(
    observed_transitions: list[tuple[int, str, str]],
    expected: list[tuple[int, str, str, int]],
    duration_ms: int,
) -> dict[str, Any]:
    fp = 0
    fn = 0
    timing_errors: list[float] = []
    tp = 0

    used_expected = [False] * len(expected)

    for obs_t, obs_from, obs_to in observed_transitions:
        matched = False
        for i, (exp_t, exp_from, exp_to, tolerance) in enumerate(expected):
            if used_expected[i]:
                continue
            if exp_from == obs_from and exp_to == obs_to and abs(obs_t - exp_t) <= tolerance:
                tp += 1
                timing_errors.append(float(abs(obs_t - exp_t)))
                used_expected[i] = True
                matched = True
                break
        if not matched:
            fp += 1

    fn = sum(1 for used in used_expected if not used)

    duration_minutes = duration_ms / 60_000.0 if duration_ms > 0 else 1.0
    fp_per_minute = fp / duration_minutes if duration_minutes > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "false_transitions_per_minute": fp_per_minute,
        "median_timing_error_ms": statistics.median(timing_errors) if timing_errors else 0.0,
        "max_timing_error_ms": max(timing_errors) if timing_errors else 0.0,
        "missed_transitions": fn,
    }


def evaluate_latency(
    input_events: list[tuple[int, str]],
    output_events: list[tuple[int, str]],
) -> dict[str, Any]:
    latencies: list[float] = []
    missing: int = 0

    for inp_t, inp_type in input_events:
        matched = False
        for out_t, out_type in output_events:
            if out_type == inp_type and out_t >= inp_t:
                latencies.append(float(out_t - inp_t))
                matched = True
                break
        if not matched:
            missing += 1

    if not latencies:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "missing_output_count": missing}

    return {
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "max_ms": max(latencies),
        "missing_output_count": missing,
    }


def evaluate_memory(
    observed: list[dict[str, Any]],
    expected: list[tuple[str, str, list[int], str | None]],
) -> dict[str, Any]:
    tp = 0
    fn = 0
    extra = 0

    used_expected = [False] * len(expected)
    for obs in observed:
        matched_expected = False
        for i, (exp_type, exp_label, _within_ms, location) in enumerate(expected):
            if used_expected[i]:
                continue
            obs_type = obs.get("type", "")
            obs_label = obs.get("label", "")
            obs_location = obs.get("location")
            if obs_type != exp_type or obs_label != exp_label:
                continue
            if location and obs_location != location:
                continue
            tp += 1
            used_expected[i] = True
            matched_expected = True
            break
        if not matched_expected:
            extra += 1

    fn = sum(1 for used in used_expected if not used)

    total_expected = tp + fn
    precision = tp / (tp + extra) if (tp + extra) > 0 else 0.0
    recall = tp / total_expected if total_expected > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_negatives": fn,
        "extra_memories": extra,
    }


def evaluate_grounding(
    observed_answers: list[dict[str, Any]],
    expected: list[tuple[str, list[str], list[str], list[str]]],
) -> dict[str, Any]:
    grounded = 0
    unsupported = 0
    unknown_correct = 0
    total = len(expected)

    for query, answer_contains, evidence_types, memory_labels in expected:
        obs = _find_answer(observed_answers, query)
        if obs is None:
            if not answer_contains and not evidence_types:
                unknown_correct += 1
                continue
            unsupported += 1
            continue

        answer_text = obs.get("text", "")
        obs_evidence = obs.get("evidence_types", [])
        obs_memory = obs.get("memory_labels", [])

        has_text = (
            any(phrase in answer_text for phrase in answer_contains) if answer_contains else True
        )
        has_evidence = all(et in obs_evidence for et in evidence_types) if evidence_types else True
        has_memory = all(ml in obs_memory for ml in memory_labels) if memory_labels else True

        if has_text and has_evidence and has_memory:
            grounded += 1
        else:
            unsupported += 1

    grounded_rate = grounded / total if total > 0 else 0.0

    return {
        "grounded_rate": grounded_rate,
        "grounded_count": grounded,
        "unsupported_count": unsupported,
        "unknown_correct_count": unknown_correct,
        "total_expected": total,
    }


def _find_answer(
    answers: list[dict[str, Any]], query: str
) -> dict[str, Any] | None:
    for a in answers:
        if a.get("query", "").strip().lower() == query.strip().lower():
            return a
    return None


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: tuple[str, ...]


def evaluate_gates(**metrics: float) -> GateResult:
    checks = {
        "engagement_f1": metrics["engagement_f1"] >= 0.85,
        "false_transitions": metrics["false_transitions_per_minute"] <= 0.5,
        "frame_latency": metrics["frame_to_observation_p95_ms"] <= 200,
        "reaction_latency": metrics["state_to_visible_p95_ms"] <= 150,
        "memory_accuracy": metrics["memory_accuracy"] >= 0.90,
        "grounding_rate": metrics["grounding_rate"] >= 1.0,
        "frame_freshness": metrics["max_normal_frame_age_ms"] < 300,
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    return GateResult(not failures, failures)
