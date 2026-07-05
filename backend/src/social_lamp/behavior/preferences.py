from dataclasses import dataclass


@dataclass(frozen=True)
class PreferenceAudit:
    context: str
    behavior: str
    outcome: str
    previous_score: float
    new_score: float
    correlation_id: str | None = None


class PreferenceModel:
    DELTAS = {
        "reengaged": 0.10,
        "positive": 0.20,
        "rejected": -0.25,
        "muted": -0.25,
        "no_response": -0.05,
    }

    def __init__(self, *, exploration_enabled: bool = True) -> None:
        self._scores: dict[tuple[str, str], float] = {}
        self.audit: list[PreferenceAudit] = []
        self.exploration_enabled = exploration_enabled

    def score(self, context: str, behavior: str) -> float:
        return self._scores.get((context, behavior), 1.0)

    def record(
        self,
        context: str,
        behavior: str,
        outcome: str,
        *,
        correlation_id: str | None = None,
    ) -> None:
        if not self.exploration_enabled:
            return
        previous = self.score(context, behavior)
        updated = min(1.5, max(0.5, previous + self.DELTAS[outcome]))
        self._scores[(context, behavior)] = updated
        self.audit.append(
            PreferenceAudit(context, behavior, outcome, previous, updated, correlation_id)
        )

    def start_session(self) -> None:
        for key, score in tuple(self._scores.items()):
            self._scores[key] = 1.0 + (score - 1.0) * 0.95

    def disable_exploration(self) -> None:
        self.exploration_enabled = False
