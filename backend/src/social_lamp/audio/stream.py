from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Any, Protocol, cast

import numpy as np

from social_lamp.audio.analysis import AudioClass, VoiceFrame


class AudioChunkClassifier(Protocol):
    def classify(self, pcm: bytes, sample_rate: int) -> VoiceFrame: ...


@dataclass(frozen=True)
class MicrophoneChunk:
    pcm: bytes
    mono_ns: int
    sample_rate: int = 16_000
    duration_ms: int = 20


class SoundDeviceMicrophoneStream:
    def __init__(
        self, *, sample_rate: int = 16_000, chunk_ms: int = 20, max_chunks: int = 32
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self._frames_per_chunk = int(sample_rate * chunk_ms / 1000)
        self._queue: Queue[bytes] = Queue(maxsize=max_chunks)
        self._stream: object | None = None
        self.health_detail = "not started"

    def start(self) -> bool:
        try:
            import sounddevice as sd  # type: ignore[import-untyped]
        except Exception as exc:
            self.health_detail = f"microphone unavailable: {exc.__class__.__name__}"
            return False

        def callback(indata: object, frames: int, time: object, status: object) -> None:
            del frames, time
            if status:
                self.health_detail = f"microphone warning: {status}"
            payload = bytes(cast(Any, indata))
            try:
                self._queue.put_nowait(payload)
            except Full:
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
                self._queue.put_nowait(payload)

        stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self._frames_per_chunk,
            channels=1,
            dtype="int16",
            callback=callback,
        )
        try:
            stream.start()
        except Exception as exc:
            self.health_detail = f"microphone unavailable: {exc}"
            stream.close()
            return False

        self._stream = stream
        self.health_detail = "microphone available"
        return True

    def read_chunk(self, *, timeout_s: float = 0.5) -> bytes | None:
        try:
            return self._queue.get(timeout=timeout_s)
        except Empty:
            return None

    def close(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            self.health_detail = "microphone stopped"
            return
        stop = getattr(stream, "stop", None)
        if callable(stop):
            stop()
        close = getattr(stream, "close", None)
        if callable(close):
            close()
        self.health_detail = "microphone stopped"

    def chunks(self) -> Iterator[bytes]:  # pragma: no cover - hardware-specific fallback surface
        if not self.start():
            return
        try:
            while True:
                chunk = self.read_chunk(timeout_s=1.0)
                if chunk is None:
                    continue
                yield chunk
        finally:
            self.close()


class SimpleVadClassifier:
    def __init__(self, *, aggressiveness: int = 2) -> None:
        try:
            import webrtcvad  # type: ignore[import-untyped]
        except Exception:
            self._vad = None
        else:
            self._vad = webrtcvad.Vad(aggressiveness)

    def classify(self, pcm: bytes, sample_rate: int) -> VoiceFrame:
        amplitude = _normalized_amplitude(pcm)
        if self._vad is not None:
            try:
                voiced = bool(self._vad.is_speech(pcm, sample_rate))
            except Exception:
                voiced = amplitude >= 0.015
        else:
            voiced = amplitude >= 0.015
        confidence = min(0.98, 0.55 + amplitude * 4.0) if voiced else max(0.2, 0.5 - amplitude)
        return VoiceFrame(
            voiced=voiced,
            audio_class=AudioClass.DIRECT_SPEECH if voiced else AudioClass.OTHER,
            confidence=round(confidence, 2),
        )


def _normalized_amplitude(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    return float(np.abs(samples).mean() / 32768.0)
