from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol


class AudioChunkClassifier(Protocol):
    def classify(self, pcm: bytes, sample_rate: int) -> object: ...


@dataclass(frozen=True)
class MicrophoneChunk:
    pcm: bytes
    mono_ns: int
    sample_rate: int = 16_000
    duration_ms: int = 20


class SoundDeviceMicrophoneStream:
    def __init__(self, *, sample_rate: int = 16_000, chunk_ms: int = 20) -> None:
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.health_detail = "not started"

    def chunks(self) -> Iterator[bytes]:  # pragma: no cover - hardware-specific fallback surface
        try:
            import sounddevice as sd  # type: ignore[import-untyped]
        except Exception as exc:
            self.health_detail = f"microphone unavailable: {exc.__class__.__name__}"
            return
        del sd
        self.health_detail = "microphone available"
        return
        yield b""
