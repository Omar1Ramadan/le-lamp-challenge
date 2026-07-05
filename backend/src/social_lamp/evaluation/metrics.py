from dataclasses import dataclass
from math import ceil


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
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
    def from_pairs(cls, pairs: list[tuple[str, str]]) -> "ClassificationCounts":
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


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: tuple[str, ...]


def evaluate_gates(**metrics: float) -> GateResult:
    checks = {
        "engagement_f1": metrics["engagement_f1"] >= 0.85,
        "false_transitions": metrics["false_transitions_per_two_minutes"] <= 1.0,
        "frame_latency": metrics["frame_to_observation_p95_ms"] <= 200,
        "reaction_latency": metrics["state_to_visible_p95_ms"] <= 150,
        "memory_accuracy": metrics["memory_accuracy"] >= 0.90,
        "grounding_rate": metrics["grounding_rate"] == 1.0,
        "frame_freshness": metrics["max_normal_frame_age_ms"] < 300,
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    return GateResult(not failures, failures)
