from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

from uuid6 import uuid7

from social_lamp.audio.analysis import AudioAnalyzer, AudioState, SimulatorAudioInterruption
from social_lamp.audio.stream import MicrophoneChunk
from social_lamp.behavior.compositor import BehaviorCompositor
from social_lamp.behavior.policy import BehaviorPolicy
from social_lamp.capture.frames import CapturedFrame
from social_lamp.conversation.base import ConversationProvider, ConversationResponse
from social_lamp.conversation.template import TemplateConversationProvider
from social_lamp.domain.contracts import (
    AudioMode,
    BehaviorTimeline,
    ComponentHealth,
    MemoryQuery,
    MemoryResult,
    ObjectState,
    PersonState,
    SocialState,
    WorldSnapshot,
)
from social_lamp.memory.repository import ObservationWrite
from social_lamp.perception.engagement import EngagementEstimator
from social_lamp.perception.faces import FaceResult, face_result_to_signals
from social_lamp.perception.location import BBox, locate_box
from social_lamp.perception.objects import Detection, ObjectTrack
from social_lamp.replay.trace import TraceReader


class SimulatorPort(Protocol):
    @property
    def pose(self) -> dict[str, float]: ...

    async def execute(self, timeline: BehaviorTimeline) -> object: ...

    async def neutralize(self) -> None: ...


class MetricsPort(Protocol):
    def increment(self, name: str, **labels: str) -> None: ...


class WorldPort(Protocol):
    @property
    def snapshot(self) -> WorldSnapshot: ...

    def replace(self, snapshot: WorldSnapshot) -> None: ...


class MemoryPort(Protocol):
    async def close(self) -> None: ...

    async def find_last_seen(
        self,
        object_label: str,
        *,
        session_scope: str | None = None,
        before_utc: str | None = None,
    ) -> MemoryResult: ...

    async def clear(self) -> None: ...


class RuntimeCoordinator:
    def __init__(
        self,
        *,
        world: WorldPort,
        simulator: SimulatorPort,
        metrics: MetricsPort,
        memory: MemoryPort,
        conversation: ConversationProvider | None = None,
        policy: BehaviorPolicy | None = None,
        compositor: BehaviorCompositor | None = None,
    ) -> None:
        self.world = world
        self.simulator = simulator
        self.metrics = metrics
        self.memory = memory
        self.conversation = conversation or TemplateConversationProvider(self._query_memory)
        self._policy = policy or BehaviorPolicy()
        self._compositor = compositor or BehaviorCompositor()
        self._running = False
        self._tasks: set[asyncio.Task[None]] = set()
        self._resources_closed = False
        self._object_tracks: dict[str, ObjectTrack] = {}
        self._recorded_stable_tracks: set[str] = set()
        self._audio_analyzer = AudioAnalyzer()
        self.audio_state = AudioState(False, False, None)
        self.bonuses_enabled = False

    @classmethod
    def for_test(cls, *, database: Path) -> RuntimeCoordinator:
        from social_lamp.runtime.testing import build_test_runtime

        return build_test_runtime(database)

    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        if self._resources_closed:
            raise RuntimeError("runtime resources are already closed")
        self._running = True

    async def stop(self) -> None:
        if self._resources_closed:
            self._running = False
            return
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
        await self.simulator.neutralize()
        await self.conversation.close("runtime stopping")
        await self.memory.close()
        self._resources_closed = True

    async def replay(self, directory: Path) -> None:
        previous = self.world.snapshot
        for record in TraceReader(directory).records():
            if record.record_type == "snapshot":
                state_value = record.body.get("social_state")
                if not isinstance(state_value, str):
                    continue
                revision_value = record.body.get("revision", previous.revision + 1)
                if isinstance(revision_value, int):
                    revision = revision_value
                else:
                    revision = previous.revision + 1
                current = previous.model_copy(
                    update={
                        "snapshot_id": uuid7(),
                        "revision": revision,
                        "as_of_mono_ns": record.recorded_at_mono_ns,
                        "social_state": SocialState(state_value),
                    }
                )
                self.world.replace(current)
            elif record.record_type == "observation":
                current = previous
            else:
                continue

            if current.revision == previous.revision:
                previous = current
                continue
            intent = self._policy.on_transition(previous, current)
            if intent is not None:
                timeline = self._compositor.compose(intent, self.simulator.pose)
                await self.simulator.execute(timeline)
            self.metrics.increment("social_transition", state=current.social_state.value)
            previous = current

    async def submit_text(self, text: str) -> ConversationResponse:
        return await self.conversation.handle_text(str(uuid7()), text)

    async def neutralize(self) -> None:
        await self.simulator.neutralize()

    async def clear_memory(self) -> None:
        await self.memory.clear()

    def configure_audio(self, *, interruption: SimulatorAudioInterruption | None = None) -> None:
        self._audio_analyzer = AudioAnalyzer(interruption=interruption)

    def set_simulator_speaking(self, speaking: bool) -> None:
        self._audio_analyzer.set_simulator_speaking(speaking)

    async def process_audio_chunk(self, chunk: MicrophoneChunk, *, classifier: object) -> None:
        snapshot = self.world.snapshot
        try:
            frame = classifier.classify(chunk.pcm, chunk.sample_rate)  # type: ignore[attr-defined]
        except Exception as exc:
            self.world.replace(
                snapshot.model_copy(
                    update={
                        "revision": snapshot.revision + 1,
                        "as_of_mono_ns": chunk.mono_ns,
                        "health": _replace_health(
                            snapshot.health,
                            ComponentHealth(
                                component="microphone", status="degraded", detail=str(exc)
                            ),
                        ),
                    }
                )
            )
            return
        self.audio_state = self._audio_analyzer.push(frame)
        people = snapshot.people
        if self.audio_state.speaker_id is not None:
            people = (
                PersonState(
                    person_id=self.audio_state.speaker_id,
                    engagement_score=0.0,
                    engagement_confidence=0.0,
                    is_active_speaker=True,
                ),
            )
        self.world.replace(
            snapshot.model_copy(
                update={
                    "snapshot_id": uuid7(),
                    "revision": snapshot.revision + 1,
                    "as_of_mono_ns": chunk.mono_ns,
                    "audio_mode": AudioMode.LISTENING
                    if self.audio_state.speech_active
                    else AudioMode.SILENT,
                    "people": people,
                    "health": _replace_health(
                        snapshot.health, ComponentHealth(component="microphone", status="ok")
                    ),
                }
            )
        )

    async def process_vision_frame(
        self,
        frame: CapturedFrame,
        *,
        face_processor: object,
        object_detector: object,
        anchors: dict[str, BBox],
    ) -> None:
        snapshot = self.world.snapshot
        try:
            faces = tuple(face_processor.process(frame, now_mono_ns=frame.mono_ns))  # type: ignore[attr-defined]
            detections = tuple(object_detector.detect(frame.image))  # type: ignore[attr-defined]
        except Exception as exc:
            self.world.replace(
                snapshot.model_copy(
                    update={
                        "revision": snapshot.revision + 1,
                        "as_of_mono_ns": frame.mono_ns,
                        "health": _replace_health(
                            snapshot.health,
                            ComponentHealth(component="vision", status="degraded", detail=str(exc)),
                        ),
                    }
                )
            )
            return

        people = (_person_from_face(faces[0]),) if faces else snapshot.people
        objects: list[ObjectState] = []
        for detection in detections:
            track_id = f"track-{detection.label.strip().lower().replace(' ', '-')}"
            track = self._object_tracks.setdefault(track_id, ObjectTrack(track_id=track_id))
            track.add(detection)
            if not track.is_stable:
                continue
            location = locate_box(detection.bbox, anchors=anchors)
            state = ObjectState(
                track_id=track.track_id,
                label=track.label,
                confidence=detection.confidence,
                horizontal_region=location.horizontal_region,
                depth_band=location.depth_band,
                anchor_name=location.anchor_name,
            )
            objects.append(state)
            should_record = track.track_id not in self._recorded_stable_tracks and hasattr(
                self.memory, "record"
            )
            if should_record:
                await self.memory.record(  # type: ignore[attr-defined]
                    _observation_write(
                        state=state,
                        detection=detection,
                        snapshot=snapshot,
                        captured_at_mono_ns=frame.mono_ns,
                    )
                )
                self._recorded_stable_tracks.add(track.track_id)

        self.world.replace(
            snapshot.model_copy(
                update={
                    "snapshot_id": uuid7(),
                    "revision": snapshot.revision + 1,
                    "as_of_mono_ns": frame.mono_ns,
                    "people": people,
                    "objects": tuple(objects) if objects else snapshot.objects,
                    "health": _replace_health(
                        snapshot.health, ComponentHealth(component="vision", status="ok")
                    ),
                }
            )
        )

    def set_bonuses(self, enabled: bool) -> bool:
        self.bonuses_enabled = enabled
        return self.bonuses_enabled

    def export_trace(self, directory: Path) -> dict[str, object]:
        reader = TraceReader(directory)
        return {
            "manifest": reader.manifest().__dict__,
            "records": [record.__dict__ for record in reader.records()],
            "checksum_valid": reader.verify_checksum(),
        }

    async def _query_memory(self, query: MemoryQuery) -> MemoryResult:
        if query.kind != "last_seen":
            return MemoryResult.not_found()
        scope = str(query.session_scope) if query.session_scope is not None else None
        return await self.memory.find_last_seen(
            query.object_label,
            session_scope=scope,
            before_utc=query.before_utc,
        )


def _person_from_face(face: FaceResult) -> PersonState:
    signals = face_result_to_signals(
        face_confidence=face.face_confidence,
        yaw_degrees=face.yaw_degrees,
        pitch_degrees=face.pitch_degrees,
        gaze_score=face.gaze_score,
        gaze_quality=face.gaze_quality,
        face_area_ratio=face.face_area_ratio,
    )
    sample = EngagementEstimator(smoothing_ms=0).sample(signals, 0)
    return PersonState(
        person_id="person-1",
        engagement_score=round(sample.raw_score, 2),
        engagement_confidence=signals.confidence,
    )


def _replace_health(
    current: tuple[ComponentHealth, ...], item: ComponentHealth
) -> tuple[ComponentHealth, ...]:
    return tuple(existing for existing in current if existing.component != item.component) + (item,)


def _observation_write(
    *, state: ObjectState, detection: Detection, snapshot: WorldSnapshot, captured_at_mono_ns: int
) -> ObservationWrite:
    return ObservationWrite(
        observation_id=f"vision-{state.track_id}-{captured_at_mono_ns}",
        track_id=state.track_id,
        session_id=str(snapshot.session_id),
        observed_at_utc="2026-07-04T12:00:00Z",
        observed_at_mono_ns=captured_at_mono_ns,
        label=state.label,
        label_source="vision",
        detection_confidence=detection.confidence,
        bbox=detection.bbox,
        horizontal_region=state.horizontal_region,
        depth_band=state.depth_band,
        anchor_name=state.anchor_name,
        location_confidence=0.8,
        frame_ref=None,
        snapshot_path=None,
        correlation_id=str(uuid7()),
    )
