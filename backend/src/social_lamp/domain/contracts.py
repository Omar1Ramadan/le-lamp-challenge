from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from uuid6 import uuid7

from social_lamp.domain.clock import Clock


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ObservationSource(StrEnum):
    CAPTURE = "capture"
    FACE = "face"
    GAZE = "gaze"
    AUDIO = "audio"
    OBJECT = "object"
    ENRICHMENT = "enrichment"
    OPERATOR = "operator"
    SYSTEM = "system"


class SocialState(StrEnum):
    IDLE = "idle"
    CANDIDATE = "candidate"
    ENGAGED = "engaged"
    DISENGAGED = "disengaged"
    SEEKING_ATTENTION = "seeking_attention"


class AudioMode(StrEnum):
    SILENT = "silent"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class ObservationEvent(FrozenModel):
    schema_version: str = "1.0"
    event_id: UUID
    correlation_id: UUID
    session_id: UUID
    source: ObservationSource
    kind: str = Field(min_length=1, max_length=80)
    captured_at_mono_ns: int = Field(ge=0)
    emitted_at_mono_ns: int = Field(ge=0)
    wall_time_utc: str
    confidence: float = Field(ge=0.0, le=1.0)
    frame_ref: str | None = None
    payload: dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        clock: Clock,
        session_id: UUID,
        correlation_id: UUID,
        source: ObservationSource,
        kind: str,
        confidence: float,
        payload: dict[str, Any],
        captured_at_mono_ns: int | None = None,
        frame_ref: str | None = None,
    ) -> "ObservationEvent":
        now = clock.mono_ns
        captured = now if captured_at_mono_ns is None else captured_at_mono_ns
        return cls(
            event_id=uuid7(),
            correlation_id=correlation_id,
            session_id=session_id,
            source=source,
            kind=kind,
            captured_at_mono_ns=captured,
            emitted_at_mono_ns=now,
            wall_time_utc=clock.wall_time_utc,
            confidence=confidence,
            frame_ref=frame_ref,
            payload=payload,
        )


class PersonState(FrozenModel):
    person_id: str
    engagement_score: float = Field(ge=0.0, le=1.0)
    engagement_confidence: float = Field(ge=0.0, le=1.0)
    is_active_speaker: bool = False


class EngagementCalibrationSnapshot(FrozenModel):
    state: str = "uncalibrated"
    person_id: str | None = None
    sample_count: int = Field(default=0, ge=0)
    quality: str = "unavailable"
    failure_reason: str | None = None
    mode: str = "fallback"
    progress: float = Field(default=0.0, ge=0.0, le=1.0)


class ObjectState(FrozenModel):
    track_id: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    horizontal_region: str | None = None
    depth_band: str | None = None
    anchor_name: str | None = None


class ComponentHealth(FrozenModel):
    component: str
    status: str
    detail: str | None = None


class WorldSnapshot(FrozenModel):
    schema_version: str = "1.0"
    snapshot_id: UUID
    revision: int = Field(ge=0)
    session_id: UUID
    as_of_mono_ns: int = Field(ge=0)
    social_state: SocialState
    audio_mode: AudioMode
    primary_person_id: str | None
    people: tuple[PersonState, ...]
    engagement_calibration: EngagementCalibrationSnapshot = Field(
        default_factory=EngagementCalibrationSnapshot
    )
    objects: tuple[ObjectState, ...]
    health: tuple[ComponentHealth, ...]

    @classmethod
    def empty(cls, *, session_id: UUID, mono_ns: int) -> "WorldSnapshot":
        return cls(
            snapshot_id=uuid7(),
            revision=0,
            session_id=session_id,
            as_of_mono_ns=mono_ns,
            social_state=SocialState.IDLE,
            audio_mode=AudioMode.SILENT,
            primary_person_id=None,
            people=(),
            objects=(),
            health=(),
        )


class BehaviorIntent(FrozenModel):
    intent_id: UUID
    correlation_id: UUID
    session_id: UUID
    kind: str
    priority: int = Field(ge=0, le=100)
    created_at_mono_ns: int = Field(ge=0)
    expires_at_mono_ns: int = Field(ge=0)
    target_person_id: str | None = None
    cancellable: bool = True
    replace_policy: str = "replace_lower_priority"
    suppression_reason: str | None = None
    source_event_ids: tuple[str, ...] = ()
    parameters: dict[str, Any] = Field(default_factory=dict)


class BehaviorDecision(FrozenModel):
    intent: BehaviorIntent | None = None
    suppressed: bool = False
    suppression_reason: str | None = None
    active_timeline_id: str | None = None
    replacement: str | None = None


class MotionKeyframe(FrozenModel):
    offset_ms: int = Field(ge=0)
    value: float = Field(ge=-1.0, le=1.0)
    easing: str = "ease_in_out"


class MotionTrack(FrozenModel):
    channel: str
    keyframes: tuple[MotionKeyframe, ...]


class LightKeyframe(FrozenModel):
    offset_ms: int = Field(ge=0)
    rgb: tuple[float, float, float]
    brightness: float = Field(ge=0.0, le=1.0)


class BehaviorTimeline(FrozenModel):
    timeline_id: UUID
    intent_id: UUID
    correlation_id: UUID
    priority: int = Field(ge=0, le=100)
    duration_ms: int = Field(gt=0)
    cancellable: bool
    motion_tracks: tuple[MotionTrack, ...]
    light_track: tuple[LightKeyframe, ...] = ()
    audio_resource_id: str | None = None


class MemoryQuery(FrozenModel):
    kind: str
    object_label: str = Field(min_length=1, max_length=80)
    session_scope: UUID | str | None = None
    before_utc: str | None = None
    limit: int = Field(default=1, ge=1, le=20)


class MemoryResult(FrozenModel):
    status: str
    canonical_label: str | None = None
    horizontal_region: str | None = None
    depth_band: str | None = None
    anchor_name: str | None = None
    observed_at_utc: str | None = None
    evidence_ids: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()

    @classmethod
    def not_found(cls) -> "MemoryResult":
        return cls(status="not_found")


class ObservationSummary(FrozenModel):
    id: str
    type: str = "object_seen"
    summary: str
    observed_at_utc: str
    evidence_ids: tuple[str, ...] = ()


class ToolCallRecord(FrozenModel):
    name: str
    status: str = "success"
    evidence_ids: tuple[str, ...] = ()
    detail: str | None = None


class GroundingValidation(FrozenModel):
    valid: bool
    evidence_ids: tuple[str, ...] = ()
    reason: str | None = None


class EvidenceEvent(FrozenModel):
    event_id: str
    event_type: str
    correlation_id: str | None = None
    occurred_at_mono_ns: int = Field(ge=0)
    source: str = "runtime"
    summary: str
    severity: str = "info"
    entity_refs: tuple[dict[str, Any], ...] = ()
    evidence_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
