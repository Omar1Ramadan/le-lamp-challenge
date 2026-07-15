from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from time import monotonic_ns
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
from social_lamp.replay.trace import TraceManifest, TraceReader, TraceRecord, write_trace


class SimulatorPort(Protocol):
    @property
    def pose(self) -> dict[str, float]: ...

    async def execute(self, timeline: BehaviorTimeline) -> object: ...

    async def neutralize(self) -> None: ...


class MetricsPort(Protocol):
    def increment(self, name: str, **labels: str) -> None: ...


class CameraSource(Protocol):
    health_detail: str

    def open(self) -> bool: ...

    def read(self) -> CapturedFrame | None: ...

    def close(self) -> None: ...


class AudioSource(Protocol):
    health_detail: str

    def start(self) -> bool: ...

    def read_chunk(self, *, timeout_s: float = 0.5) -> bytes | None: ...

    def close(self) -> None: ...


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

    async def record(self, observation: ObservationWrite) -> str: ...


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
        camera_source: CameraSource | None = None,
        face_processor: object | None = None,
        object_detector: object | None = None,
        anchors: dict[str, BBox] | None = None,
        audio_source: AudioSource | None = None,
        audio_classifier: object | None = None,
        snapshot_publisher: Callable[[dict[str, object]], Awaitable[None]] | None = None,
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
        self._engagement_estimator = EngagementEstimator()
        self.audio_state = AudioState(False, False, None)
        self.bonuses_enabled = False
        self.replay_messages: list[tuple[str, dict[str, object]]] = []
        self._camera_source = camera_source
        self._face_processor = face_processor
        self._object_detector = object_detector
        self._anchors = anchors or {}
        self._audio_source = audio_source
        self._audio_classifier = audio_classifier
        self._snapshot_publisher = snapshot_publisher

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
        if self._running:
            return
        self._running = True
        if (
            self._camera_source is not None
            and self._face_processor is not None
            and self._object_detector is not None
        ):
            self._spawn(self._run_camera_loop())
        if self._audio_source is not None and self._audio_classifier is not None:
            self._spawn(self._run_audio_loop())

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
        if self._camera_source is not None:
            await asyncio.to_thread(self._camera_source.close)
        if self._audio_source is not None:
            await asyncio.to_thread(self._audio_source.close)
        await self.simulator.neutralize()
        await self.conversation.close("runtime stopping")
        await self.memory.close()
        self._resources_closed = True

    def _spawn(self, coroutine: Coroutine[object, object, None]) -> None:
        task: asyncio.Task[None] = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_camera_loop(self) -> None:
        source = self._camera_source
        face_processor = self._face_processor
        object_detector = self._object_detector
        if source is None or face_processor is None or object_detector is None:
            return
        opened = await asyncio.to_thread(source.open)
        if not opened:
            await self._set_health("camera", "degraded", source.health_detail)
            return
        while self._running:
            frame = await asyncio.to_thread(source.read)
            if frame is None:
                await self._set_health("camera", "degraded", source.health_detail)
                await asyncio.sleep(0.1)
                continue
            await self._set_health("camera", "ok", None)
            await self.process_vision_frame(
                frame,
                face_processor=face_processor,
                object_detector=object_detector,
                anchors=self._anchors,
            )
            await asyncio.sleep(0.05)

    async def _run_audio_loop(self) -> None:
        source = self._audio_source
        classifier = self._audio_classifier
        if source is None or classifier is None:
            return
        started = await asyncio.to_thread(source.start)
        if not started:
            await self._set_health("microphone", "degraded", source.health_detail)
            return
        while self._running:
            chunk_bytes = await asyncio.to_thread(source.read_chunk, timeout_s=0.5)
            if chunk_bytes is None:
                continue
            await self.process_audio_chunk(
                MicrophoneChunk(pcm=chunk_bytes, mono_ns=monotonic_ns()),
                classifier=classifier,
            )

    async def _set_health(self, component: str, status: str, detail: str | None) -> None:
        snapshot = self.world.snapshot
        updated = snapshot.model_copy(
            update={
                "snapshot_id": uuid7(),
                "revision": snapshot.revision + 1,
                "as_of_mono_ns": monotonic_ns(),
                "health": _replace_health(
                    snapshot.health,
                    ComponentHealth(component=component, status=status, detail=detail),
                ),
            }
        )
        self.world.replace(updated)
        await self._publish_snapshot(updated)

    async def _publish_snapshot(self, snapshot: WorldSnapshot) -> None:
        if self._snapshot_publisher is None:
            return
        await self._snapshot_publisher(snapshot.model_dump(mode="json"))

    async def replay(self, directory: Path) -> None:
        self.replay_messages.clear()
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
                self.replay_messages.append(("world_snapshot", current.model_dump(mode="json")))
            elif record.record_type == "observation":
                current = previous
            elif record.record_type == "memory":
                await self._record_replay_memory(record)
                current = previous
            elif record.record_type == "memory_result":
                evidence_ids = record.body.get("evidence_ids", ())
                if not isinstance(evidence_ids, list):
                    evidence_ids = []
                result = MemoryResult(
                    status=str(record.body.get("status", "not_found")),
                    canonical_label="keys" if record.body.get("status") == "found" else None,
                    horizontal_region="right" if record.body.get("status") == "found" else None,
                    depth_band="foreground" if record.body.get("status") == "found" else None,
                    anchor_name="desk" if record.body.get("status") == "found" else None,
                    observed_at_utc="2026-07-04T12:00:00Z"
                    if record.body.get("status") == "found"
                    else None,
                    evidence_ids=tuple(str(item) for item in evidence_ids),
                )
                self.replay_messages.append(("memory_result", result.model_dump(mode="json")))
                current = previous
            elif record.record_type == "intent":
                if record.body.get("kind") == "seek_attention":
                    level = record.body.get("level", 1)
                    level_value = level if isinstance(level, int) else 1
                    self.replay_messages.append(
                        (
                            "metric",
                            {
                                "name": "attention_level",
                                "value": level_value,
                            },
                        )
                    )
                current = previous
            elif record.record_type == "timeline":
                current = previous
            elif record.record_type == "bonus":
                self.replay_messages.append(
                    ("metric", {"name": str(record.body.get("name", "bonus_event")), "value": 1})
                )
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
                self.replay_messages.append(("behavior_timeline", timeline.model_dump(mode="json")))
                health = getattr(self.simulator, "health", None)
                if isinstance(health, ComponentHealth):
                    current = current.model_copy(
                        update={"health": _replace_health(current.health, health)}
                    )
                    self.world.replace(current)
            self.metrics.increment("social_transition", state=current.social_state.value)
            self.replay_messages.append(
                (
                    "metric",
                    {"name": "social_transition", "labels": {"state": current.social_state.value}},
                )
            )
            if current.social_state is SocialState.ENGAGED:
                self.replay_messages.append(("metric", {"name": "engagement_seen", "value": 1}))
            previous = current

    async def _record_replay_memory(self, record: TraceRecord) -> None:
        observation_id = str(record.body.get("observation_id", "observation-replay"))
        label = str(record.body.get("label", "object"))
        horizontal_region = record.body.get("horizontal_region")
        anchor_name = record.body.get("anchor_name")
        write = ObservationWrite(
            observation_id=observation_id,
            track_id=f"track-{label}",
            session_id=str(self.world.snapshot.session_id),
            observed_at_utc="2026-07-04T12:00:00Z",
            observed_at_mono_ns=record.recorded_at_mono_ns,
            label=label,
            label_source="replay",
            detection_confidence=0.95,
            bbox=(0.72, 0.60, 0.92, 0.90),
            horizontal_region=str(horizontal_region) if horizontal_region is not None else None,
            depth_band="foreground",
            anchor_name=str(anchor_name) if anchor_name is not None else None,
            location_confidence=0.95,
            frame_ref=None,
            snapshot_path=None,
            correlation_id=str(uuid7()),
        )
        try:
            await self.memory.record(write)
        except sqlite3.IntegrityError:
            pass
        result = MemoryResult(
            status="found",
            canonical_label=label,
            horizontal_region=write.horizontal_region,
            depth_band=write.depth_band,
            anchor_name=write.anchor_name,
            observed_at_utc=write.observed_at_utc,
            evidence_ids=(observation_id,),
        )
        self.replay_messages.append(("memory_result", result.model_dump(mode="json")))

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
            await self._publish_snapshot(self.world.snapshot)
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
        await self._publish_snapshot(self.world.snapshot)

    async def process_vision_frame(
        self,
        frame: CapturedFrame,
        *,
        face_processor: object,
        object_detector: object,
        anchors: dict[str, BBox],
    ) -> BehaviorTimeline | None:
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
            await self._publish_snapshot(self.world.snapshot)
            return None

        if faces:
            person, social_state = _person_from_face(
                faces[0], self._engagement_estimator, frame.mono_ns
            )
            people = (person,)
            primary_person_id = "person-1"
        else:
            people = ()
            social_state = SocialState.DISENGAGED if snapshot.people else SocialState.IDLE
            primary_person_id = None

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
            should_record = track.track_id not in self._recorded_stable_tracks
            if should_record:
                await self.memory.record(
                    _observation_write(
                        state=state,
                        detection=detection,
                        snapshot=snapshot,
                        captured_at_mono_ns=frame.mono_ns,
                    )
                )
                self._recorded_stable_tracks.add(track.track_id)

        detector_health = (
            object_detector.health()
            if hasattr(object_detector, "health")
            else ComponentHealth(component="object_detector", status="disabled")
        )
        current = snapshot.model_copy(
            update={
                "snapshot_id": uuid7(),
                "revision": snapshot.revision + 1,
                "as_of_mono_ns": frame.mono_ns,
                "social_state": social_state,
                "primary_person_id": primary_person_id,
                "people": people,
                "objects": tuple(objects) if objects else snapshot.objects,
                "health": _replace_health(
                    _replace_health(
                        snapshot.health,
                        ComponentHealth(component="vision", status="ok"),
                    ),
                    detector_health,
                ),
            }
        )
        self.world.replace(current)
        await self._publish_snapshot(current)

        intent = self._policy.on_transition(snapshot, current)
        if intent is None:
            return None
        timeline = self._compositor.compose(intent, self.simulator.pose)
        await self.simulator.execute(timeline)
        return timeline

    def set_bonuses(self, enabled: bool) -> bool:
        self.bonuses_enabled = enabled
        return self.bonuses_enabled

    def export_trace(self, directory: Path) -> dict[str, object]:
        snapshot = self.world.snapshot
        write_trace(
            directory,
            manifest=TraceManifest(
                schema_version="1.0",
                application_version="runtime",
                session_id=str(snapshot.session_id),
                configuration_hash="local",
            ),
            records=(
                TraceRecord(
                    sequence=1,
                    record_type="snapshot",
                    recorded_at_mono_ns=snapshot.as_of_mono_ns,
                    body=snapshot.model_dump(mode="json"),
                ),
            ),
        )
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


def _person_from_face(
    face: FaceResult, estimator: EngagementEstimator, mono_ns: int
) -> tuple[PersonState, SocialState]:
    signals = face_result_to_signals(
        face_confidence=face.face_confidence,
        yaw_degrees=face.yaw_degrees,
        pitch_degrees=face.pitch_degrees,
        gaze_score=face.gaze_score,
        gaze_quality=face.gaze_quality,
        face_area_ratio=face.face_area_ratio,
    )
    sample = estimator.sample(signals, mono_ns)
    return (
        PersonState(
            person_id="person-1",
            engagement_score=round(sample.smoothed_score, 2),
            engagement_confidence=signals.confidence,
        ),
        sample.state,
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
