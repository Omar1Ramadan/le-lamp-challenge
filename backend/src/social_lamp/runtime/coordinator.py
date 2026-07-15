from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
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
    EngagementCalibrationSnapshot,
    MemoryQuery,
    MemoryResult,
    ObjectState,
    PersonState,
    SocialState,
    WorldSnapshot,
)
from social_lamp.memory.repository import ObservationWrite
from social_lamp.perception.engagement import EngagementCalibrationStatus, EngagementEstimator
from social_lamp.perception.faces import face_result_to_signals
from social_lamp.perception.location import BBox, locate_box
from social_lamp.perception.objects import Detection, ObjectTrack
from social_lamp.perception.tracker import PersonTrack, PersonTracker
from social_lamp.replay.trace import TraceManifest, TraceReader, TraceRecord, write_trace
from social_lamp.world.commands import AudioUpdate, HealthUpdate, Source, VisionUpdate

FACE_MISSING_GRACE_MS = 600
FACE_TRACK_EXPIRE_MS = 1500
LOCATION_STABLE_NS = 1_000_000_000
OBJECT_REFRESH_NS = 30_000_000_000


@dataclass
class ObjectMemoryState:
    last_recorded_label: str | None = None
    last_recorded_region: str | None = None
    last_recorded_depth: str | None = None
    last_recorded_anchor: str | None = None
    last_recorded_mono_ns: int | None = None
    pending_region: str | None = None
    pending_depth: str | None = None
    pending_anchor: str | None = None
    pending_since_mono_ns: int | None = None


class SimulatorPort(Protocol):
    @property
    def pose(self) -> dict[str, float]: ...

    async def execute(self, timeline: BehaviorTimeline) -> object: ...

    async def neutralize(self) -> None: ...

    def handle_ack(self, body: dict[str, object]) -> None: ...


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

    def apply_health(self, update: HealthUpdate) -> WorldSnapshot | None: ...

    def apply_vision(self, update: VisionUpdate) -> WorldSnapshot | None: ...

    def apply_audio(self, update: AudioUpdate) -> WorldSnapshot | None: ...

    def replace_from_replay(self, snapshot: WorldSnapshot) -> WorldSnapshot: ...


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
        object_detection_max_fps: int = 0,
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
        self._object_memory_state: dict[str, ObjectMemoryState] = {}
        self._audio_analyzer = AudioAnalyzer()
        self._engagement_estimators: dict[str, EngagementEstimator] = {}
        self._engagement_calibration = EngagementEstimator()
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
        self._last_social_state = SocialState.IDLE
        self._person_tracker = PersonTracker(
            track_expire_ns=FACE_TRACK_EXPIRE_MS * 1_000_000
        )
        self._previous_primary_person_id: str | None = None
        self._primary_candidate_person_id: str | None = None
        self._primary_candidate_frames = 0
        self._object_detection_max_fps = object_detection_max_fps
        self._last_object_inference_mono_ns: int = 0
        self._object_detection_skipped_frames: int = 0

    @classmethod
    def for_test(cls, *, database: Path) -> RuntimeCoordinator:
        from social_lamp.runtime.testing import build_test_runtime

        return build_test_runtime(database)

    @property
    def running(self) -> bool:
        return self._running

    def start_engagement_calibration(
        self,
        *,
        mono_ns: int | None = None,
    ) -> EngagementCalibrationSnapshot:
        snapshot = self.world.snapshot
        status = self._engagement_calibration.start_calibration(
            snapshot.primary_person_id,
            mono_ns if mono_ns is not None else snapshot.as_of_mono_ns,
        )
        return self._set_engagement_calibration_status(status)

    def cancel_engagement_calibration(self) -> EngagementCalibrationSnapshot:
        status = self._engagement_calibration.cancel_calibration()
        return self._set_engagement_calibration_status(status)

    def engagement_calibration_status(
        self,
        *,
        mono_ns: int | None = None,
    ) -> EngagementCalibrationSnapshot:
        status = self._engagement_calibration.calibration_status(mono_ns)
        return self._set_engagement_calibration_status(status)

    def _set_engagement_calibration_status(
        self,
        status: EngagementCalibrationStatus,
    ) -> EngagementCalibrationSnapshot:
        calibration = _calibration_snapshot(status)
        self.world.replace(
            self.world.snapshot.model_copy(
                update={
                    "revision": self.world.snapshot.revision + 1,
                    "engagement_calibration": calibration,
                }
            )
        )
        return calibration

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

    async def _set_health(
        self, component: str, status: str, detail: str | None, *, mono_ns: int | None = None
    ) -> None:
        update = HealthUpdate(
            component=component,
            status=status,
            detail=detail,
            as_of_mono_ns=mono_ns or monotonic_ns(),
        )
        current = self.world.apply_health(update)
        if current is not None:
            await self._publish_snapshot(current)

    def _should_run_object_detection(self, mono_ns: int) -> bool:
        if self._object_detection_max_fps <= 0:
            return True
        if self._last_object_inference_mono_ns == 0:
            return True
        min_interval_ns = 1_000_000_000 // self._object_detection_max_fps
        return (mono_ns - self._last_object_inference_mono_ns) >= min_interval_ns

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
                self.world.replace_from_replay(current)
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
                    self.world.replace_from_replay(current)
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
        try:
            frame = classifier.classify(chunk.pcm, chunk.sample_rate)  # type: ignore[attr-defined]
        except Exception as exc:
            await self._set_health("microphone", "degraded", str(exc), mono_ns=chunk.mono_ns)
            return
        self.audio_state = self._audio_analyzer.push(frame)
        update = AudioUpdate(
            audio_mode=AudioMode.LISTENING if self.audio_state.speech_active else AudioMode.SILENT,
            as_of_mono_ns=chunk.mono_ns,
            active_speaker_id=self.audio_state.speaker_id,
            health_update=ComponentHealth(component="microphone", status="ok"),
        )
        current = self.world.apply_audio(update)
        if current is not None:
            await self._publish_snapshot(current)

    async def process_vision_frame(
        self,
        frame: CapturedFrame,
        *,
        face_processor: object,
        object_detector: object,
        anchors: dict[str, BBox],
    ) -> BehaviorTimeline | None:
        previous = self.world.snapshot
        try:
            faces = tuple(face_processor.process(frame, now_mono_ns=frame.mono_ns))  # type: ignore[attr-defined]
        except Exception as exc:
            await self._set_health("vision", "degraded", str(exc), mono_ns=frame.mono_ns)
            return None

        detections: tuple[Detection, ...] = ()
        if self._should_run_object_detection(frame.mono_ns):
            try:
                detections = tuple(object_detector.detect(frame.image))  # type: ignore[attr-defined]
                self._last_object_inference_mono_ns = frame.mono_ns
            except Exception as exc:
                await self._set_health("vision", "degraded", str(exc), mono_ns=frame.mono_ns)
                return None
        else:
            self._object_detection_skipped_frames += 1

        tracks = self._person_tracker.update(list(faces), frame.mono_ns)
        visible_tracks = [track for track in tracks if track.visible]
        engagement_scores: dict[str, float] = {}
        track_states: dict[str, SocialState] = {}
        people_list: list[PersonState] = []

        for face in faces:
            self._engagement_calibration.observe_calibration(face, frame.mono_ns)

        for track in visible_tracks:
            if track.last_face is None:
                continue
            estimator = self._engagement_estimators.setdefault(
                track.person_id, EngagementEstimator()
            )
            person, social_state = _person_from_track(
                track,
                estimator,
                frame.mono_ns,
                calibration=self._engagement_calibration._calibration,
            )
            people_list.append(person)
            engagement_scores[track.person_id] = person.engagement_score
            track_states[track.person_id] = social_state

        for track in tracks:
            if track.visible:
                continue
            previous_person = next(
                (person for person in previous.people if person.person_id == track.person_id),
                None,
            )
            if previous_person is None:
                continue
            elapsed_ms = (frame.mono_ns - track.last_seen_mono_ns) / 1_000_000
            decay_factor = 0.5 if elapsed_ms < FACE_MISSING_GRACE_MS else 0.2
            people_list.append(
                previous_person.model_copy(
                    update={
                        "engagement_confidence": round(
                            previous_person.engagement_confidence * decay_factor, 2
                        ),
                    }
                )
            )

        active_track_ids = {track.person_id for track in tracks}
        for person_id in list(self._engagement_estimators):
            if person_id not in active_track_ids:
                del self._engagement_estimators[person_id]

        primary_person_id = self._select_primary_person_id(visible_tracks, engagement_scores)
        if (
            primary_person_id is not None
            and self._engagement_calibration._calibration.state == "calibrating"
            and self._engagement_calibration._calibration.person_id is None
        ):
            self._engagement_calibration._calibration.person_id = primary_person_id
        if primary_person_id is not None:
            social_state = track_states.get(primary_person_id, self._last_social_state)
            self._last_social_state = social_state
        elif people_list:
            social_state = self._last_social_state
        else:
            social_state = SocialState.DISENGAGED if previous.people else SocialState.IDLE
            self._last_social_state = social_state

        people = tuple(people_list)

        objects: list[ObjectState] = []
        for detection in detections:
            track_id = f"track-{detection.label.strip().lower().replace(' ', '-')}"
            object_track = self._object_tracks.setdefault(
                track_id, ObjectTrack(track_id=track_id)
            )
            object_track.add(detection)
            if not object_track.is_stable:
                continue
            location = locate_box(detection.bbox, anchors=anchors)
            state = ObjectState(
                track_id=object_track.track_id,
                label=object_track.label,
                confidence=detection.confidence,
                horizontal_region=location.horizontal_region,
                depth_band=location.depth_band,
                anchor_name=location.anchor_name,
            )
            objects.append(state)

            mem = self._object_memory_state.setdefault(track_id, ObjectMemoryState())
            decision, reason = should_record_object_observation(
                mem,
                state.label,
                state.horizontal_region,
                state.depth_band,
                state.anchor_name,
                frame.mono_ns,
            )
            if decision:
                await self.memory.record(
                    _observation_write(
                        state=state,
                        detection=detection,
                        snapshot=previous,
                        captured_at_mono_ns=frame.mono_ns,
                    )
                )
                mem.last_recorded_label = state.label
                mem.last_recorded_region = state.horizontal_region
                mem.last_recorded_depth = state.depth_band
                mem.last_recorded_anchor = state.anchor_name
                mem.last_recorded_mono_ns = frame.mono_ns
                mem.pending_region = None
                mem.pending_depth = None
                mem.pending_anchor = None
                mem.pending_since_mono_ns = None
            else:
                loc = (state.horizontal_region, state.depth_band, state.anchor_name)
                last_loc = (
                    mem.last_recorded_region,
                    mem.last_recorded_depth,
                    mem.last_recorded_anchor,
                )
                if loc != last_loc:
                    pending_loc = (
                        mem.pending_region,
                        mem.pending_depth,
                        mem.pending_anchor,
                    )
                    if pending_loc != loc:
                        mem.pending_region = state.horizontal_region
                        mem.pending_depth = state.depth_band
                        mem.pending_anchor = state.anchor_name
                        mem.pending_since_mono_ns = frame.mono_ns
                else:
                    mem.pending_region = None
                    mem.pending_depth = None
                    mem.pending_anchor = None
                    mem.pending_since_mono_ns = None

        detector_health = (
            object_detector.health()
            if hasattr(object_detector, "health")
            else ComponentHealth(component="object_detector", status="disabled")
        )
        metadata = getattr(face_processor, "metadata", None)
        if metadata is not None:
            face_health = ComponentHealth(
                component="face_detector",
                status=metadata.status,
                detail=f"{metadata.name}: {metadata.detail}" if metadata.detail else metadata.name,
            )
        else:
            face_health = ComponentHealth(
                component="face_detector", status="unknown", detail="no metadata available"
            )
        update = VisionUpdate(
            people=people,
            social_state=social_state,
            primary_person_id=primary_person_id,
            as_of_mono_ns=frame.mono_ns,
            source=Source.CAMERA,
            objects=tuple(objects) if objects else None,
            health_updates=(
                ComponentHealth(component="vision", status="ok"),
                face_health,
                detector_health,
            ),
        )
        current = self.world.apply_vision(update)
        if current is None:
            return None
        current = current.model_copy(
            update={
                "engagement_calibration": _calibration_snapshot(
                    self._engagement_calibration.calibration_status(frame.mono_ns)
                )
            }
        )
        self.world.replace(current)
        await self._publish_snapshot(current)

        intent = self._policy.on_transition(previous, current)
        if intent is None:
            return None
        timeline = self._compositor.compose(intent, self.simulator.pose)
        await self.simulator.execute(timeline)
        sim_health = getattr(self.simulator, "health", None)
        if isinstance(sim_health, ComponentHealth):
            current = current.model_copy(
                update={"health": _replace_health(current.health, sim_health)}
            )
            self.world.replace(current)
            await self._publish_snapshot(current)
        return timeline

    def _select_primary_person_id(
        self, visible_tracks: list[PersonTrack], engagement_scores: dict[str, float]
    ) -> str | None:
        if not visible_tracks:
            self._previous_primary_person_id = None
            self._primary_candidate_person_id = None
            self._primary_candidate_frames = 0
            return None

        candidate = max(
            visible_tracks,
            key=lambda track: (
                engagement_scores.get(track.person_id, 0.0),
                track.bbox[2] * track.bbox[3],
            ),
        ).person_id
        visible_ids = {track.person_id for track in visible_tracks}
        previous = self._previous_primary_person_id
        if previous is None or previous not in visible_ids:
            self._previous_primary_person_id = candidate
            self._primary_candidate_person_id = None
            self._primary_candidate_frames = 0
            return candidate
        if candidate == previous:
            self._primary_candidate_person_id = None
            self._primary_candidate_frames = 0
            return previous
        if self._primary_candidate_person_id == candidate:
            self._primary_candidate_frames += 1
        else:
            self._primary_candidate_person_id = candidate
            self._primary_candidate_frames = 1
        if self._primary_candidate_frames >= 3:
            self._previous_primary_person_id = candidate
            self._primary_candidate_person_id = None
            self._primary_candidate_frames = 0
            return candidate
        return previous

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


def _person_from_track(
    track: PersonTrack,
    estimator: EngagementEstimator,
    mono_ns: int,
    *,
    calibration: object | None = None,
) -> tuple[PersonState, SocialState]:
    face = track.last_face
    if face is None:
        return (
            PersonState(
                person_id=track.person_id,
                engagement_score=0.0,
                engagement_confidence=0.0,
            ),
            SocialState.IDLE,
        )
    signals = face_result_to_signals(
        face_confidence=face.face_confidence,
        yaw_degrees=face.yaw_degrees,
        pitch_degrees=face.pitch_degrees,
        gaze_score=face.gaze_score,
        gaze_quality=face.gaze_quality,
        face_area_ratio=face.face_area_ratio,
        pose_source=face.pose_source,
        pose_quality=face.pose_quality,
        calibration=calibration,
    )
    sample = estimator.sample(signals, mono_ns)
    return (
        PersonState(
            person_id=track.person_id,
            engagement_score=round(sample.smoothed_score, 2),
            engagement_confidence=signals.confidence,
        ),
        sample.state,
    )


def _calibration_snapshot(status: EngagementCalibrationStatus) -> EngagementCalibrationSnapshot:
    return EngagementCalibrationSnapshot(
        state=status.state,
        person_id=status.person_id,
        sample_count=status.sample_count,
        quality=status.quality,
        failure_reason=status.failure_reason,
        mode=status.mode,
        progress=status.progress,
    )


def _replace_health(
    current: tuple[ComponentHealth, ...], item: ComponentHealth
) -> tuple[ComponentHealth, ...]:
    return tuple(existing for existing in current if existing.component != item.component) + (item,)


def should_record_object_observation(
    mem: ObjectMemoryState,
    current_label: str,
    current_region: str | None,
    current_depth: str | None,
    current_anchor: str | None,
    mono_ns: int,
) -> tuple[bool, str]:
    if mem.last_recorded_mono_ns is None:
        return True, "first_stable"
    if current_label != mem.last_recorded_label:
        return True, "label_changed"

    loc = (current_region, current_depth, current_anchor)
    last_loc = (mem.last_recorded_region, mem.last_recorded_depth, mem.last_recorded_anchor)
    if loc != last_loc:
        pending_loc = (mem.pending_region, mem.pending_depth, mem.pending_anchor)
        if (
            pending_loc == loc
            and mem.pending_since_mono_ns is not None
            and mono_ns - mem.pending_since_mono_ns >= LOCATION_STABLE_NS
        ):
            return True, "location_changed"
        return False, "location_pending"

    if mono_ns - mem.last_recorded_mono_ns >= OBJECT_REFRESH_NS:
        return True, "refresh"

    return False, "unchanged"


def _observation_write(
    *, state: ObjectState, detection: Detection, snapshot: WorldSnapshot, captured_at_mono_ns: int
) -> ObservationWrite:
    return ObservationWrite(
        observation_id=f"vision-{state.track_id}-{captured_at_mono_ns}",
        track_id=state.track_id,
        session_id=str(snapshot.session_id),
        observed_at_utc=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
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
