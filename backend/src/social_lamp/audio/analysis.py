from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class AudioClass(StrEnum):
    DIRECT_SPEECH = "direct_speech"
    CONVERSATION_BACKGROUND = "conversation_background"
    TELEVISION_MEDIA = "television_media"
    MUSIC = "music"
    OTHER = "other"


@dataclass(frozen=True)
class VoiceFrame:
    voiced: bool
    audio_class: AudioClass
    confidence: float
    speaker_id: str | None = None
    valence_tendency: float = 0.0
    arousal: float = 0.0


@dataclass(frozen=True)
class AudioState:
    speech_active: bool
    suppress_unsolicited_sound: bool
    speaker_id: str | None
    listen_priority: int | None = None


@dataclass(frozen=True)
class VocalAffectObservation:
    valence_tendency: float
    arousal: float
    confidence: float
    window_ms: int
    speaker_id: str | None


@dataclass(frozen=True)
class MicrophoneHealth:
    status: str
    detail: str

    @classmethod
    def from_device_available(cls, available: bool) -> MicrophoneHealth:
        if available:
            return cls(status="healthy", detail="microphone available")
        return cls(status="missing", detail="microphone device is not available")


class SimulatorAudioInterruption(Protocol):
    def cancel(self, reason: str) -> None: ...


@dataclass
class SimulatorSpeechInterruption:
    cancelled: bool = False
    reason: str | None = None

    def cancel(self, reason: str) -> None:
        self.cancelled = True
        self.reason = reason


class AudioAnalyzer:
    def __init__(
        self,
        *,
        frame_ms: int = 20,
        interruption: SimulatorAudioInterruption | None = None,
    ) -> None:
        self._frame_ms = frame_ms
        self._voiced_ms = 0
        self._silence_ms = 0
        self._active = False
        self._simulator_speaking = False
        self._interruption = interruption

    def set_simulator_speaking(self, speaking: bool) -> None:
        self._simulator_speaking = speaking

    def push(self, frame: VoiceFrame) -> AudioState:
        if frame.voiced:
            self._voiced_ms += self._frame_ms
            self._silence_ms = 0
            self._active = self._active or self._voiced_ms >= 120
        else:
            self._silence_ms += self._frame_ms
            if self._silence_ms >= 500:
                self._active = False
                self._voiced_ms = 0
        media = frame.audio_class in {AudioClass.TELEVISION_MEDIA, AudioClass.MUSIC}
        speaker = frame.speaker_id if frame.audio_class is AudioClass.DIRECT_SPEECH else None
        listen_priority: int | None = None
        interrupted_simulator = (
            self._simulator_speaking
            and frame.voiced
            and frame.audio_class is AudioClass.DIRECT_SPEECH
        )
        if interrupted_simulator:
            listen_priority = 90
            if self._interruption is not None:
                self._interruption.cancel("human speech interrupted simulator audio")
            self._simulator_speaking = False
        suppress_unsolicited_sound = media and frame.confidence >= 0.65
        return AudioState(self._active, suppress_unsolicited_sound, speaker, listen_priority)


class ActiveSpeakerScorer:
    def __init__(self, *, threshold: float = 0.65) -> None:
        self._threshold = threshold

    def associate(self, candidates: dict[str, dict[str, float]]) -> str | None:
        best_id: str | None = None
        best_score = 0.0
        for person_id, features in candidates.items():
            score = (
                0.5 * features.get("mouth_correlation", 0.0)
                + 0.3 * features.get("visual_plausibility", 0.0)
                + 0.2 * features.get("continuity", 0.0)
            )
            if score > best_score:
                best_score = score
                best_id = person_id
        if best_score < self._threshold:
            return None
        return best_id


class VocalAffectWindow:
    def __init__(self, *, frame_ms: int = 20) -> None:
        self._frame_ms = frame_ms
        self._frames: list[VoiceFrame] = []

    def push(self, frame: VoiceFrame) -> None:
        if not frame.voiced:
            return
        self._frames.append(frame)
        max_frames = 8_000 // self._frame_ms
        if len(self._frames) > max_frames:
            self._frames = self._frames[-max_frames:]

    def observation(self) -> VocalAffectObservation | None:
        window_ms = len(self._frames) * self._frame_ms
        if window_ms < 3_000 or window_ms > 8_000 or not self._frames:
            return None
        confidence = sum(frame.confidence for frame in self._frames) / len(self._frames)
        if confidence < 0.60:
            return None
        valence = _bounded_average(frame.valence_tendency for frame in self._frames)
        arousal = _bounded_average(frame.arousal for frame in self._frames)
        speaker_id = _single_speaker(frame.speaker_id for frame in self._frames)
        return VocalAffectObservation(valence, arousal, confidence, window_ms, speaker_id)

    def clear(self) -> None:
        self._frames.clear()


class PcmChunkSource(Protocol):
    def chunks(self) -> Iterator[bytes]: ...


class VadClassifier(Protocol):
    def classify(self, pcm: bytes, sample_rate: int) -> VoiceFrame: ...


@dataclass(frozen=True)
class MonoPcmDeviceAdapter:
    source: PcmChunkSource
    classifier: VadClassifier
    sample_rate: int = 16_000

    def frames(self) -> Iterator[VoiceFrame]:
        for chunk in self.source.chunks():
            yield self.classifier.classify(chunk, self.sample_rate)


def _bounded_average(values: Iterable[float]) -> float:
    values_tuple = tuple(values)
    if not values_tuple:
        return 0.0
    average = sum(values_tuple) / len(values_tuple)
    return max(-1.0, min(1.0, average))


def _single_speaker(speakers: Iterable[str | None]) -> str | None:
    unique = {speaker for speaker in speakers if speaker is not None}
    if len(unique) == 1:
        return next(iter(unique))
    return None
