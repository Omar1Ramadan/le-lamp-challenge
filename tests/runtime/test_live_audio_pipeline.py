from pathlib import Path

import pytest
from social_lamp.audio.analysis import AudioClass, SimulatorSpeechInterruption, VoiceFrame
from social_lamp.audio.stream import MicrophoneChunk
from social_lamp.domain.contracts import AudioMode, ComponentHealth, PersonState
from social_lamp.runtime.coordinator import RuntimeCoordinator


class FakeClassifier:
    def __init__(self, frames: list[VoiceFrame]) -> None:
        self.frames = frames

    def classify(self, pcm: bytes, sample_rate: int) -> VoiceFrame:
        del pcm, sample_rate
        return self.frames.pop(0)


@pytest.mark.asyncio
async def test_microphone_chunks_drive_runtime_audio_state(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    classifier = FakeClassifier(
        [VoiceFrame(True, AudioClass.DIRECT_SPEECH, 0.9, speaker_id="person-1") for _ in range(6)]
    )

    for step in range(6):
        await coordinator.process_audio_chunk(
            MicrophoneChunk(pcm=b"voice", mono_ns=step * 20_000_000), classifier=classifier
        )

    snapshot = coordinator.world.snapshot
    assert snapshot.audio_mode is AudioMode.LISTENING
    assert snapshot.people == (
        PersonState(
            person_id="person-1",
            engagement_score=0.0,
            engagement_confidence=0.0,
            is_active_speaker=True,
        ),
    )
    assert ComponentHealth(component="microphone", status="ok") in snapshot.health


@pytest.mark.asyncio
async def test_simulator_speech_interruption_cancels_and_listens(
    tmp_path: Path,
) -> None:
    interruption = SimulatorSpeechInterruption()
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    coordinator.configure_audio(interruption=interruption)
    coordinator.set_simulator_speaking(True)

    await coordinator.process_audio_chunk(
        MicrophoneChunk(pcm=b"voice", mono_ns=0),
        classifier=FakeClassifier([VoiceFrame(True, AudioClass.DIRECT_SPEECH, 0.95)]),
    )

    assert interruption.cancelled
    assert coordinator.audio_state.listen_priority == 90


@pytest.mark.asyncio
async def test_microphone_failure_degrades_health_without_crashing(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")

    class BrokenClassifier:
        def classify(self, pcm: bytes, sample_rate: int) -> VoiceFrame:
            del pcm, sample_rate
            raise RuntimeError("microphone unavailable")

    await coordinator.process_audio_chunk(
        MicrophoneChunk(pcm=b"", mono_ns=0), classifier=BrokenClassifier()
    )

    assert (
        ComponentHealth(component="microphone", status="degraded", detail="microphone unavailable")
        in coordinator.world.snapshot.health
    )
