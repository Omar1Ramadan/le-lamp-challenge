from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from uuid6 import uuid7

from social_lamp.adapters.simulator import SimulatorAdapter
from social_lamp.api.hub import ConnectionHub
from social_lamp.audio.stream import SimpleVadClassifier, SoundDeviceMicrophoneStream
from social_lamp.capture.frames import OpenCVCameraSource
from social_lamp.config import Settings
from social_lamp.conversation.base import ConversationProvider
from social_lamp.domain.clock import SystemClock
from social_lamp.domain.contracts import ComponentHealth
from social_lamp.memory.repository import MemoryRepository
from social_lamp.perception.faces import MediaPipeFaceLandmarkerProcessor, OpenCvFaceProcessor
from social_lamp.perception.location import BBox
from social_lamp.perception.objects import NullObjectDetector, YoloObjectDetector
from social_lamp.runtime.coordinator import RuntimeCoordinator
from social_lamp.runtime.providers import build_conversation_provider
from social_lamp.world.model import WorldModel


class RuntimeMetrics:
    def __init__(self) -> None:
        self._counters: defaultdict[tuple[str, tuple[tuple[str, str], ...]], int] = defaultdict(int)

    def increment(self, name: str, **labels: str) -> None:
        self._counters[(name, tuple(sorted(labels.items())))] += 1

    def counter(self, name: str, **labels: str) -> int:
        return self._counters[(name, tuple(sorted(labels.items())))]


def _build_object_detector(settings: Settings) -> NullObjectDetector | YoloObjectDetector:
    if not settings.enable_object_detection:
        return NullObjectDetector()
    try:
        class_ids = None
        if settings.object_detection_classes is not None:
            class_ids = [int(c.strip()) for c in settings.object_detection_classes.split(",")]
        return YoloObjectDetector(
            model_path=settings.object_detector_model,
            confidence=settings.object_detection_confidence,
            classes=class_ids,
        )
    except Exception:
        return NullObjectDetector()


async def build_live_runtime(
    *,
    settings: Settings | None = None,
    database_path: Path | None = None,
    hub: ConnectionHub | None = None,
    conversation: ConversationProvider | None = None,
) -> RuntimeCoordinator:
    resolved_settings = settings or Settings()
    if database_path is not None:
        resolved_settings = resolved_settings.model_copy(update={"database_path": database_path})

    clock = SystemClock()
    world = WorldModel(session_id=uuid7(), clock=clock)
    memory = await MemoryRepository.open(resolved_settings.database_path)
    resolved_hub = hub or ConnectionHub()
    simulator = SimulatorAdapter(resolved_hub)
    metrics = RuntimeMetrics()
    camera_source = None
    face_processor = None
    object_detector = None
    audio_source = None
    audio_classifier = None
    anchors: dict[str, BBox] = {"desk": (0.25, 0.45, 0.75, 0.95)}

    if resolved_settings.enable_live_capture:
        camera_source = OpenCVCameraSource(camera_index=resolved_settings.camera_index)
        object_detector = _build_object_detector(resolved_settings)
        world.replace(
            world.snapshot.model_copy(
                update={
                    "revision": world.snapshot.revision + 1,
                    "health": world.snapshot.health + (object_detector.health(),),
                }
            )
        )
        audio_source = SoundDeviceMicrophoneStream()
        audio_classifier = SimpleVadClassifier()
        landmark_exc: RuntimeError | None = None
        if resolved_settings.enable_mediapipe_face_landmarker:
            try:
                face_processor = MediaPipeFaceLandmarkerProcessor()
            except RuntimeError as exc:
                landmark_exc = exc
        if face_processor is None:
            try:
                face_processor = OpenCvFaceProcessor()
            except RuntimeError as fallback_exc:
                detail = str(fallback_exc)
                if landmark_exc is not None:
                    detail = f"{landmark_exc}; {fallback_exc}"
                world.replace(
                    world.snapshot.model_copy(
                        update={
                            "revision": world.snapshot.revision + 1,
                            "health": (
                                ComponentHealth(
                                    component="vision",
                                    status="degraded",
                                    detail=detail,
                                ),
                            ),
                        }
                    )
                )
    else:
        world.replace(
            world.snapshot.model_copy(
                update={
                    "revision": world.snapshot.revision + 1,
                    "health": (
                        ComponentHealth(
                            component="camera",
                            status="disabled",
                            detail="set ENABLE_LIVE_CAPTURE=true to use local devices",
                        ),
                        ComponentHealth(
                            component="microphone",
                            status="disabled",
                            detail="set ENABLE_LIVE_CAPTURE=true to use local devices",
                        ),
                    ),
                }
            )
        )

    coordinator = RuntimeCoordinator(
        world=world,
        simulator=simulator,
        metrics=metrics,
        memory=memory,
        conversation=conversation,
        camera_source=camera_source,
        face_processor=face_processor,
        object_detector=object_detector,
        anchors=anchors,
        audio_source=audio_source,
        audio_classifier=audio_classifier,
        snapshot_publisher=lambda body: resolved_hub.broadcast(
            {"type": "world_snapshot", "body": body}
        ),
    )
    if conversation is None:
        coordinator.conversation = build_conversation_provider(
            resolved_settings,
            query=coordinator._query_memory,
        )
    return coordinator
