from dataclasses import dataclass

from social_lamp.domain.contracts import SocialState


@dataclass(frozen=True)
class EngagementSignals:
    face_presence: float | None
    head_toward: float | None
    gaze_toward: float | None
    proximity: float | None
    directed_speech: float | None
    confidence: float


@dataclass(frozen=True)
class EngagementSample:
    raw_score: float
    smoothed_score: float
    confidence: float
    state: SocialState


class EngagementEstimator:
    WEIGHTS = (0.20, 0.30, 0.25, 0.10, 0.15)

    def __init__(self, *, smoothing_ms: int = 250) -> None:
        self._smoothing_ms = smoothing_ms
        self._smoothed: float | None = None
        self._last_ns: int | None = None
        self._state = SocialState.IDLE
        self._candidate_since_ns: int | None = None
        self._away_since_ns: int | None = None
        self._engaged_since_ns: int | None = None

    def _fuse(self, signals: EngagementSignals) -> float:
        values = (
            signals.face_presence,
            signals.head_toward,
            signals.gaze_toward,
            signals.proximity,
            signals.directed_speech,
        )
        available = [
            (value, weight)
            for value, weight in zip(values, self.WEIGHTS, strict=True)
            if value is not None
        ]
        weight_sum = sum(weight for _, weight in available)
        if not weight_sum:
            return 0.0
        return sum(value * weight for value, weight in available) / weight_sum

    def sample(self, signals: EngagementSignals, mono_ns: int) -> EngagementSample:
        raw = self._fuse(signals)
        if self._smoothed is None or self._smoothing_ms == 0 or self._last_ns is None:
            self._smoothed = raw
        else:
            elapsed_ms = max(0.0, (mono_ns - self._last_ns) / 1_000_000)
            alpha = min(1.0, elapsed_ms / self._smoothing_ms)
            self._smoothed += alpha * (raw - self._smoothed)
        self._last_ns = mono_ns
        score = self._smoothed
        if signals.confidence < 0.45:
            return EngagementSample(raw, score, signals.confidence, self._state)
        if self._state in {SocialState.IDLE, SocialState.CANDIDATE}:
            if score >= 0.68:
                if self._candidate_since_ns is None:
                    self._candidate_since_ns = mono_ns
                self._state = SocialState.CANDIDATE
                if mono_ns - self._candidate_since_ns >= 700_000_000:
                    self._state = SocialState.ENGAGED
                    self._engaged_since_ns = mono_ns
                    self._candidate_since_ns = None
                    self._away_since_ns = None
            elif score >= 0.45:
                self._state = SocialState.CANDIDATE
                self._candidate_since_ns = self._candidate_since_ns or mono_ns
            elif score < 0.35:
                self._state = SocialState.IDLE
                self._candidate_since_ns = None
        elif self._state is SocialState.ENGAGED:
            if score < 0.38:
                if self._away_since_ns is None:
                    self._away_since_ns = mono_ns
                exit_since_ns = self._engaged_since_ns or self._away_since_ns
                if mono_ns - exit_since_ns >= 1_200_000_000:
                    self._state = SocialState.DISENGAGED
                    self._engaged_since_ns = None
            else:
                self._away_since_ns = None
        elif self._state is SocialState.DISENGAGED and score >= 0.62:
            if self._candidate_since_ns is None:
                self._candidate_since_ns = mono_ns
            if mono_ns - self._candidate_since_ns >= 500_000_000:
                self._state = SocialState.ENGAGED
                self._engaged_since_ns = mono_ns
        return EngagementSample(raw, score, signals.confidence, self._state)
