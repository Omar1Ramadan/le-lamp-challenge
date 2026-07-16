from pathlib import Path

import numpy as np
import pytest
from social_lamp.audio.analysis import AudioClass, SimulatorSpeechInterruption, VoiceFrame
from social_lamp.audio.stream import MicrophoneChunk
from social_lamp.capture.frames import CapturedFrame
from social_lamp.domain.contracts import AudioMode, ComponentHealth, PersonState
from social_lamp.perception.faces import FaceResult
from social_lamp.runtime.coordinator import RuntimeCoordinator


class FakeClassifier:
    def __init__(self, frames: list[VoiceFrame]) -> None:
        self.frames = frames

    def classify(self, pcm: bytes, sample_rate: int) -> VoiceFrame:
        del pcm, sample_rate
        return self.frames.pop(0)


class FakeFaceProcessor:
    def __init__(self, faces: tuple[FaceResult, ...]) -> None:
        self.faces = faces

    def process(self, frame: CapturedFrame, *, now_mono_ns: int) -> tuple[FaceResult, ...]:
        del frame, now_mono_ns
        return self.faces


class FakeObjectDetector:
    def detect(self, image: np.ndarray) -> tuple[()]:
        del image
        return ()


def _face(*, yaw: float = 0.0, confidence: float = 0.9) -> FaceResult:
    return FaceResult(
        face_confidence=confidence,
        yaw_degrees=yaw,
        pitch_degrees=0.0,
        gaze_score=0.8,
        gaze_quality=0.9,
        face_area_ratio=0.12,
        pose_source="mediapipe_matrix",
        pose_quality=0.95,
    )


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

    classifier = FakeClassifier(
        [VoiceFrame(True, AudioClass.DIRECT_SPEECH, 0.95) for _ in range(13)]
    )
    for step in range(13):
        await coordinator.process_audio_chunk(
            MicrophoneChunk(pcm=b"voice", mono_ns=step * 20_000_000),
            classifier=classifier,
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


@pytest.mark.asyncio
async def test_unknown_speech_with_one_visible_person_marks_active_speaker(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    await coordinator.process_vision_frame(
        CapturedFrame(np.zeros((4, 4, 3), dtype=np.uint8), mono_ns=0),
        face_processor=FakeFaceProcessor((_face(),)),
        object_detector=FakeObjectDetector(),
        anchors={},
    )

    classifier = FakeClassifier([VoiceFrame(True, AudioClass.DIRECT_SPEECH, 0.9) for _ in range(6)])
    for step in range(6):
        await coordinator.process_audio_chunk(
            MicrophoneChunk(pcm=b"voice", mono_ns=100_000_000 + step * 20_000_000),
            classifier=classifier,
        )

    assert coordinator.world.snapshot.audio_mode is AudioMode.LISTENING
    assert coordinator.world.snapshot.people[0].person_id == "person-1"
    assert coordinator.world.snapshot.people[0].is_active_speaker is True


@pytest.mark.asyncio
async def test_background_media_is_suppressed_without_listening(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    classifier = FakeClassifier(
        [VoiceFrame(True, AudioClass.TELEVISION_MEDIA, 0.9) for _ in range(8)]
    )

    for step in range(8):
        await coordinator.process_audio_chunk(
            MicrophoneChunk(pcm=b"media", mono_ns=step * 20_000_000),
            classifier=classifier,
        )

    snapshot = coordinator.world.snapshot
    assert snapshot.audio_mode is AudioMode.SILENT
    assert any(
        h.component == "vad" and h.status == "background_media" for h in snapshot.health
    )
    assert coordinator.replay_messages[-1][0] == "audio_event"
    assert coordinator.replay_messages[-1][1]["kind"] == "background_media_suppressed"
