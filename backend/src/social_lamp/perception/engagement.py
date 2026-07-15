from dataclasses import dataclass
from statistics import median, pstdev
from typing import Literal

from social_lamp.domain.contracts import SocialState

CalibrationState = Literal["uncalibrated", "calibrating", "calibrated", "failed", "expired"]
CalibrationMode = Literal["fallback", "calibrated", "partial"]


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


@dataclass(frozen=True)
class EngagementCalibrationStatus:
    state: CalibrationState
    person_id: str | None
    sample_count: int
    quality: str
    failure_reason: str | None
    mode: CalibrationMode
    progress: float


@dataclass(frozen=True)
class EngagementCalibrationSample:
    yaw: float
    pitch: float
    roll: float
    face_scale: float
    gaze: float | None


@dataclass
class EngagementCalibration:
    state: CalibrationState = "uncalibrated"
    person_id: str | None = None
    started_mono_ns: int | None = None
    completed_mono_ns: int | None = None
    sample_count: int = 0
    neutral_yaw: float | None = None
    neutral_pitch: float | None = None
    neutral_roll: float | None = None
    neutral_face_scale: float | None = None
    gaze_baseline: float | None = None
    quality: str = "unavailable"
    failure_reason: str | None = None


class EngagementEstimator:
    WEIGHTS = (0.20, 0.30, 0.25, 0.10, 0.15)
    CALIBRATION_DURATION_NS = 3_000_000_000
    CALIBRATION_MIN_SAMPLES = 20
    CALIBRATION_MIN_FACE_CONFIDENCE = 0.45
    CALIBRATION_LIMITED_STD_DEGREES = 15.0
    CALIBRATION_FAIL_STD_DEGREES = 25.0

    def __init__(self, *, smoothing_ms: int = 250) -> None:
        self._smoothing_ms = smoothing_ms
        self._smoothed: float | None = None
        self._last_ns: int | None = None
        self._state = SocialState.IDLE
        self._candidate_since_ns: int | None = None
        self._away_since_ns: int | None = None
        self._engaged_since_ns: int | None = None
        self._calibration = EngagementCalibration()
        self._calibration_samples: list[EngagementCalibrationSample] = []

    def start_calibration(
        self,
        person_id: str | None,
        mono_ns: int,
    ) -> EngagementCalibrationStatus:
        self._calibration = EngagementCalibration(
            state="calibrating",
            person_id=person_id,
            started_mono_ns=mono_ns,
            quality="unavailable",
        )
        self._calibration_samples = []
        return self.calibration_status(mono_ns)

    def cancel_calibration(self) -> EngagementCalibrationStatus:
        self._calibration = EngagementCalibration()
        self._calibration_samples = []
        return self.calibration_status()

    def observe_calibration(self, face: object, mono_ns: int) -> None:
        if self._calibration.state != "calibrating":
            return
        if self._calibration.started_mono_ns is None:
            return
        if mono_ns - self._calibration.started_mono_ns >= self.CALIBRATION_DURATION_NS:
            self._complete_calibration(mono_ns)
            return
        face_confidence = float(getattr(face, "face_confidence", 0.0))
        if face_confidence < self.CALIBRATION_MIN_FACE_CONFIDENCE:
            return
        gaze_quality = float(getattr(face, "gaze_quality", 0.0))
        self._calibration_samples.append(
            EngagementCalibrationSample(
                yaw=float(getattr(face, "yaw_degrees", 0.0)),
                pitch=float(getattr(face, "pitch_degrees", 0.0)),
                roll=float(getattr(face, "roll_degrees", 0.0)),
                face_scale=float(getattr(face, "face_area_ratio", 0.0)),
                gaze=float(getattr(face, "gaze_score", 0.0)) if gaze_quality >= 0.45 else None,
            )
        )
        self._calibration.sample_count = len(self._calibration_samples)

    def calibration_status(self, mono_ns: int | None = None) -> EngagementCalibrationStatus:
        if mono_ns is not None and self._calibration.state == "calibrating":
            if self._calibration.started_mono_ns is not None:
                elapsed = mono_ns - self._calibration.started_mono_ns
                if elapsed >= self.CALIBRATION_DURATION_NS:
                    self._complete_calibration(mono_ns)
        return EngagementCalibrationStatus(
            state=self._calibration.state,
            person_id=self._calibration.person_id,
            sample_count=self._calibration.sample_count,
            quality=self._calibration.quality,
            failure_reason=self._calibration.failure_reason,
            mode=self._calibration_mode(),
            progress=self._calibration_progress(mono_ns),
        )

    def _calibration_mode(self) -> CalibrationMode:
        if self._calibration.state != "calibrated":
            return "fallback"
        if self._calibration.gaze_baseline is None:
            return "partial"
        return "calibrated"

    def _calibration_progress(self, mono_ns: int | None) -> float:
        if self._calibration.state == "calibrated":
            return 1.0
        if self._calibration.state != "calibrating" or self._calibration.started_mono_ns is None:
            return 0.0
        if mono_ns is None:
            return 0.0
        elapsed = max(0, mono_ns - self._calibration.started_mono_ns)
        return round(min(1.0, elapsed / self.CALIBRATION_DURATION_NS), 2)

    def _complete_calibration(self, mono_ns: int) -> None:
        samples = self._calibration_samples
        if len(samples) < self.CALIBRATION_MIN_SAMPLES:
            self._calibration.state = "failed"
            self._calibration.completed_mono_ns = mono_ns
            self._calibration.sample_count = len(samples)
            self._calibration.quality = "failed"
            self._calibration.failure_reason = "not enough valid samples"
            return

        yaws = [sample.yaw for sample in samples]
        pitches = [sample.pitch for sample in samples]
        yaw_std = pstdev(yaws) if len(yaws) > 1 else 0.0
        pitch_std = pstdev(pitches) if len(pitches) > 1 else 0.0
        if (
            yaw_std > self.CALIBRATION_FAIL_STD_DEGREES
            or pitch_std > self.CALIBRATION_FAIL_STD_DEGREES
        ):
            self._calibration.state = "failed"
            self._calibration.completed_mono_ns = mono_ns
            self._calibration.sample_count = len(samples)
            self._calibration.quality = "failed"
            self._calibration.failure_reason = "pose moved too much"
            return

        gaze_values = [sample.gaze for sample in samples if sample.gaze is not None]
        self._calibration.state = "calibrated"
        self._calibration.completed_mono_ns = mono_ns
        self._calibration.sample_count = len(samples)
        self._calibration.neutral_yaw = median(yaws)
        self._calibration.neutral_pitch = median(pitches)
        self._calibration.neutral_roll = median(sample.roll for sample in samples)
        self._calibration.neutral_face_scale = median(sample.face_scale for sample in samples)
        self._calibration.gaze_baseline = median(gaze_values) if gaze_values else None
        self._calibration.quality = (
            "limited"
            if yaw_std > self.CALIBRATION_LIMITED_STD_DEGREES
            or pitch_std > self.CALIBRATION_LIMITED_STD_DEGREES
            else "good"
        )
        self._calibration.failure_reason = None

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
