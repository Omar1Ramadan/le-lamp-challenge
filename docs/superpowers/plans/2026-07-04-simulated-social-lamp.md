# Simulated Social Lamp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portfolio-quality simulated social lamp that detects engagement, expresses behavior through a 3D six-channel lamp, remembers objects, recalls evidence conversationally, reports evaluation metrics, and supports all four optional bonuses without requiring physical hardware.

**Architecture:** A Python 3.12 FastAPI modular monolith owns capture, perception, the single-writer world model, SQLite memory, behavior policy, conversation ports, replay, and telemetry. A React/TypeScript/React Three Fiber client renders typed snapshots and timelines received over WebSocket; it never owns domain decisions. Work proceeds as vertical TDD slices, beginning with deterministic replay and a simulator adapter before adding live CV, audio, cloud, and bonus modules.

**Tech Stack:** Python 3.12, uv, FastAPI, Pydantic 2, asyncio, aiosqlite, OpenCV, MediaPipe, Ultralytics, NumPy, OpenAI Python SDK, sounddevice, WebRTC VAD; React, TypeScript, Vite, React Three Fiber, Drei, Vitest, Testing Library, Playwright; pytest, pytest-asyncio, Hypothesis, Ruff, mypy.

---

## Implementation Rules

- Read the corresponding file in `docs/design/` before beginning each task.
- Keep raw pixels/audio in bounded buffers; publish references and typed observations only.
- Use monotonic nanoseconds for ordering and latency. Use UTC only for persistence and display.
- Keep core behavior operational with cloud, live camera, live microphone, or WebGL unavailable.
- Run the exact focused test after each implementation step, then the full backend/frontend suite before each commit.
- Never place model inference, database I/O, or network calls on the world-model task.
- Do not weaken a design threshold to make a test pass; record measured exceptions in an approved design amendment.

## Target File Map

```text
backend/src/social_lamp/
  api/app.py                    FastAPI lifecycle, HTTP, and WebSocket endpoints
  domain/contracts.py           Versioned cross-module Pydantic contracts
  domain/clock.py               Clock port and deterministic fake clock
  events/bus.py                 Bounded typed fan-out bus
  world/model.py                Single-writer stable state reducer
  perception/engagement.py      Fusion, smoothing, dwell, and hysteresis
  perception/faces.py           MediaPipe face/head/gaze adapter
  perception/objects.py         Fast detector, tracking, and enrichment queue
  perception/location.py        Scene-relative region/anchor calculations
  memory/repository.py          SQLite migrations, writes, and deterministic queries
  behavior/policy.py            Intent selection, escalation, and suppression
  behavior/preferences.py       Bounded adaptive preference updates
  behavior/compositor.py        Authored intent-to-timeline conversion
  adapters/base.py              LampAdapter protocol
  adapters/simulator.py         WebSocket-backed adapter
  conversation/base.py          ConversationProvider protocol and tools
  conversation/template.py      Offline deterministic provider
  conversation/grounding.py     Evidence validator
  conversation/openai_realtime.py Cloud provider adapter
  audio/analysis.py             VAD, background media, speaker, and affect evidence
  replay/trace.py               JSONL manifests, recording, and replay
  evaluation/metrics.py         Accuracy, transition, memory, and latency metrics
  main.py                       Application entry point

frontend/src/
  contracts/generated.ts        Generated backend contract types
  state/store.ts                Snapshot/event reducer and reconnect state
  scene/LampScene.tsx           Six-channel articulated lamp
  components/PerceptionPanel.tsx
  components/EvidenceTimeline.tsx
  components/Inspector.tsx
  components/DemoRail.tsx
  App.tsx

tests/                           Backend unit, contract, replay, and API tests
frontend/src/**/*.test.tsx       Frontend component and rig tests
frontend/e2e/                    Playwright full-scenario tests
evaluation/fixtures/             Public-safe deterministic replay fixtures
```

## Phase 1: Deterministic Vertical Skeleton

### Task 1: Scaffold reproducible backend and frontend workspaces

**Files:**
- Create: `.python-version`, `.gitignore`, `pyproject.toml`, `pnpm-workspace.yaml`, `package.json`
- Create: `backend/src/social_lamp/__init__.py`, `backend/src/social_lamp/main.py`
- Create: `tests/test_smoke.py`
- Create: `frontend/` with Vite React TypeScript scaffold

- [ ] **Step 1: Write the failing backend smoke test**

```python
# tests/test_smoke.py
from social_lamp.main import create_app


def test_application_has_expected_title() -> None:
    app = create_app()
    assert app.title == "Simulated Social Lamp"
```

- [ ] **Step 2: Run the test to verify the package is absent**

Run: `uv run pytest tests/test_smoke.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'social_lamp'`.

- [ ] **Step 3: Initialize the workspaces and create the minimal application**

Run:

```powershell
uv init --bare --python 3.12
uv add fastapi pydantic uvicorn uuid6 aiosqlite numpy
uv add --dev pytest pytest-asyncio hypothesis httpx mypy ruff
pnpm create vite frontend --template react-ts
pnpm --dir frontend add three @react-three/fiber @react-three/drei
pnpm --dir frontend add -D vitest jsdom @testing-library/react @testing-library/jest-dom @playwright/test
```

Add this package configuration to `pyproject.toml`:

```toml
[project]
name = "simulated-social-lamp"
version = "0.1.0"
requires-python = ">=3.12,<3.13"

[tool.uv]
package = true

[tool.pytest.ini_options]
pythonpath = ["backend/src"]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "ASYNC"]

[tool.mypy]
python_version = "3.12"
strict = true
packages = ["social_lamp"]
mypy_path = "backend/src"
```

```python
# backend/src/social_lamp/main.py
from fastapi import FastAPI


def create_app() -> FastAPI:
    return FastAPI(title="Simulated Social Lamp", version="0.1.0")


app = create_app()
```

Root scripts:

```json
{
  "name": "simulated-social-lamp",
  "private": true,
  "scripts": {
    "dev:web": "pnpm --dir frontend dev",
    "test:web": "pnpm --dir frontend test -- --run",
    "typecheck:web": "pnpm --dir frontend exec tsc --noEmit",
    "e2e": "pnpm --dir frontend exec playwright test"
  }
}
```

```yaml
# pnpm-workspace.yaml
packages:
  - frontend
```

```text
# .python-version
3.12
```

```gitignore
.venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
frontend/node_modules/
frontend/dist/
```

- [ ] **Step 4: Verify the initial toolchain**

Run:

```powershell
uv run pytest tests/test_smoke.py -v
uv run ruff check backend tests
uv run mypy
pnpm --dir frontend exec tsc --noEmit
pnpm --dir frontend build
```

Expected: one backend test passes, lint/type checks pass, and Vite produces `frontend/dist`.

- [ ] **Step 5: Commit the scaffold**

```powershell
git add .python-version .gitignore pyproject.toml uv.lock package.json pnpm-workspace.yaml frontend backend tests
git commit -m "build: scaffold social lamp workspaces"
```

### Task 2: Define and validate canonical domain contracts

**Files:**
- Create: `backend/src/social_lamp/domain/contracts.py`
- Create: `backend/src/social_lamp/domain/clock.py`
- Create: `tests/domain/test_contracts.py`

- [ ] **Step 1: Write failing contract tests**

```python
# tests/domain/test_contracts.py
from uuid import UUID

import pytest
from pydantic import ValidationError

from social_lamp.domain.clock import FakeClock
from social_lamp.domain.contracts import (
    ObservationEvent,
    ObservationSource,
    SocialState,
    WorldSnapshot,
)


def test_observation_rejects_invalid_confidence() -> None:
    clock = FakeClock(mono_ns=10, wall_time_utc="2026-07-04T12:00:00Z")
    with pytest.raises(ValidationError):
        ObservationEvent.create(
            clock=clock,
            session_id=UUID("018f0000-0000-7000-8000-000000000001"),
            correlation_id=UUID("018f0000-0000-7000-8000-000000000002"),
            source=ObservationSource.FACE,
            kind="face_presence",
            confidence=1.1,
            payload={"person_id": "person-a"},
        )


def test_world_snapshot_is_immutable() -> None:
    snapshot = WorldSnapshot.empty(
        session_id=UUID("018f0000-0000-7000-8000-000000000001"), mono_ns=10
    )
    assert snapshot.social_state is SocialState.IDLE
    with pytest.raises(ValidationError):
        snapshot.social_state = SocialState.ENGAGED
```

- [ ] **Step 2: Run the tests to verify missing contracts**

Run: `uv run pytest tests/domain/test_contracts.py -v`

Expected: FAIL with import errors for `social_lamp.domain`.

- [ ] **Step 3: Implement the clock and base contracts**

```python
# backend/src/social_lamp/domain/clock.py
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    @property
    def mono_ns(self) -> int: ...

    @property
    def wall_time_utc(self) -> str: ...


class SystemClock:
    @property
    def mono_ns(self) -> int:
        import time

        return time.monotonic_ns()

    @property
    def wall_time_utc(self) -> str:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass
class FakeClock:
    mono_ns: int
    wall_time_utc: str

    def advance_ms(self, milliseconds: int) -> None:
        self.mono_ns += milliseconds * 1_000_000
```

Implement `contracts.py` with frozen Pydantic models and these exact public names:

```python
# backend/src/social_lamp/domain/contracts.py
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
```

- [ ] **Step 4: Run contract and static checks**

Run:

```powershell
uv run pytest tests/domain/test_contracts.py -v
uv run ruff check backend tests
uv run mypy
```

Expected: two tests pass and static checks report no errors.

- [ ] **Step 5: Commit the contracts**

```powershell
git add backend/src/social_lamp/domain tests/domain
git commit -m "feat: add versioned domain contracts"
```

### Task 3: Add bounded event fan-out and the single-writer world model

**Files:**
- Create: `backend/src/social_lamp/events/bus.py`
- Create: `backend/src/social_lamp/world/model.py`
- Create: `tests/events/test_bus.py`
- Create: `tests/world/test_model.py`

- [ ] **Step 1: Write failing backpressure and reducer tests**

```python
# tests/events/test_bus.py
import pytest

from social_lamp.events.bus import EventBus


@pytest.mark.asyncio
async def test_bus_drops_oldest_noncritical_event() -> None:
    bus: EventBus[int] = EventBus(capacity=2)
    subscription = bus.subscribe("world")
    await bus.publish(1)
    await bus.publish(2)
    await bus.publish(3)
    assert await subscription.get() == 2
    assert await subscription.get() == 3
    assert bus.dropped("world") == 1


@pytest.mark.asyncio
async def test_bus_never_silently_drops_critical_event() -> None:
    bus: EventBus[int] = EventBus(capacity=1)
    bus.subscribe("world")
    await bus.publish(1)
    with pytest.raises(RuntimeError, match="critical queue overflow"):
        await bus.publish(2, critical=True)
```

```python
# tests/world/test_model.py
from uuid import UUID

from social_lamp.domain.contracts import ObservationEvent, ObservationSource, SocialState
from social_lamp.domain.clock import FakeClock
from social_lamp.world.model import WorldModel


def test_world_model_advances_revision_only_on_stable_change() -> None:
    clock = FakeClock(0, "2026-07-04T12:00:00Z")
    session_id = UUID("018f0000-0000-7000-8000-000000000001")
    correlation_id = UUID("018f0000-0000-7000-8000-000000000002")
    model = WorldModel(session_id=session_id, clock=clock)
    event = ObservationEvent.create(
        clock=clock,
        session_id=session_id,
        correlation_id=correlation_id,
        source=ObservationSource.SYSTEM,
        kind="social_state_changed",
        confidence=0.9,
        payload={"state": "engaged", "primary_person_id": "person-a"},
    )
    changed = model.apply(event)
    unchanged = model.apply(event)
    assert changed.social_state is SocialState.ENGAGED
    assert changed.revision == 1
    assert unchanged.revision == 1
```

- [ ] **Step 2: Run tests to confirm missing implementations**

Run: `uv run pytest tests/events/test_bus.py tests/world/test_model.py -v`

Expected: FAIL with import errors for `events.bus` and `world.model`.

- [ ] **Step 3: Implement bounded fan-out and immutable snapshot reduction**

```python
# backend/src/social_lamp/events/bus.py
import asyncio
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class Subscription(Generic[T]):
    queue: asyncio.Queue[T]

    async def get(self) -> T:
        return await self.queue.get()


class EventBus(Generic[T]):
    def __init__(self, capacity: int = 32) -> None:
        self._capacity = capacity
        self._queues: dict[str, asyncio.Queue[T]] = {}
        self._dropped: dict[str, int] = {}

    def subscribe(self, name: str) -> Subscription[T]:
        queue: asyncio.Queue[T] = asyncio.Queue(maxsize=self._capacity)
        self._queues[name] = queue
        self._dropped[name] = 0
        return Subscription(queue)

    async def publish(self, event: T, *, critical: bool = False) -> None:
        for name, queue in self._queues.items():
            if queue.full():
                if critical:
                    raise RuntimeError(f"critical queue overflow: {name}")
                queue.get_nowait()
                self._dropped[name] += 1
            queue.put_nowait(event)

    def dropped(self, name: str) -> int:
        return self._dropped[name]
```

```python
# backend/src/social_lamp/world/model.py
from uuid import UUID

from uuid6 import uuid7

from social_lamp.domain.clock import Clock
from social_lamp.domain.contracts import ObservationEvent, SocialState, WorldSnapshot


class WorldModel:
    def __init__(self, *, session_id: UUID, clock: Clock) -> None:
        self._clock = clock
        self._snapshot = WorldSnapshot.empty(session_id=session_id, mono_ns=clock.mono_ns)

    @property
    def snapshot(self) -> WorldSnapshot:
        return self._snapshot

    def apply(self, event: ObservationEvent) -> WorldSnapshot:
        if event.kind != "social_state_changed":
            return self._snapshot
        state = SocialState(event.payload["state"])
        primary = event.payload.get("primary_person_id")
        if state is self._snapshot.social_state and primary == self._snapshot.primary_person_id:
            return self._snapshot
        self._snapshot = self._snapshot.model_copy(
            update={
                "snapshot_id": uuid7(),
                "revision": self._snapshot.revision + 1,
                "as_of_mono_ns": self._clock.mono_ns,
                "social_state": state,
                "primary_person_id": primary,
            }
        )
        return self._snapshot
```

- [ ] **Step 4: Verify focused and full backend tests**

Run:

```powershell
uv run pytest tests/events/test_bus.py tests/world/test_model.py -v
uv run pytest -q
uv run ruff check backend tests
uv run mypy
```

Expected: all tests and static checks pass.

- [ ] **Step 5: Commit the event/world slice**

```powershell
git add backend/src/social_lamp/events backend/src/social_lamp/world tests/events tests/world
git commit -m "feat: add bounded events and world model"
```

### Task 4: Add behavior intent, compositor, and simulator adapter contracts

**Files:**
- Modify: `backend/src/social_lamp/domain/contracts.py`
- Create: `backend/src/social_lamp/behavior/policy.py`
- Create: `backend/src/social_lamp/behavior/compositor.py`
- Create: `backend/src/social_lamp/adapters/base.py`
- Create: `tests/behavior/test_policy_compositor.py`

- [ ] **Step 1: Write the failing vertical behavior test**

```python
# tests/behavior/test_policy_compositor.py
from uuid import UUID

from social_lamp.behavior.compositor import BehaviorCompositor
from social_lamp.behavior.policy import BehaviorPolicy
from social_lamp.domain.contracts import SocialState, WorldSnapshot


def test_engagement_produces_acknowledge_timeline() -> None:
    session_id = UUID("018f0000-0000-7000-8000-000000000001")
    previous = WorldSnapshot.empty(session_id=session_id, mono_ns=0)
    current = previous.model_copy(
        update={"revision": 1, "social_state": SocialState.ENGAGED, "as_of_mono_ns": 1}
    )
    intent = BehaviorPolicy().on_transition(previous, current)
    assert intent is not None
    assert intent.kind == "acknowledge"
    timeline = BehaviorCompositor().compose(intent, current_pose={})
    assert timeline.intent_id == intent.intent_id
    assert {track.channel for track in timeline.motion_tracks} == {
        "head_yaw",
        "head_pitch",
    }
    assert timeline.duration_ms == 700
```

- [ ] **Step 2: Run the test to verify behavior modules are absent**

Run: `uv run pytest tests/behavior/test_policy_compositor.py -v`

Expected: FAIL during import of `social_lamp.behavior`.

- [ ] **Step 3: Add behavior contracts and minimal authored behavior**

Append these exact models to `domain/contracts.py`:

```python
class BehaviorIntent(FrozenModel):
    intent_id: UUID
    correlation_id: UUID
    session_id: UUID
    kind: str
    urgency: int = Field(ge=0, le=100)
    created_at_mono_ns: int = Field(ge=0)
    expires_at_mono_ns: int = Field(ge=0)
    target_person_id: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


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
```

```python
# backend/src/social_lamp/behavior/policy.py
from uuid6 import uuid7

from social_lamp.domain.contracts import BehaviorIntent, SocialState, WorldSnapshot


class BehaviorPolicy:
    def on_transition(
        self, previous: WorldSnapshot, current: WorldSnapshot
    ) -> BehaviorIntent | None:
        if previous.social_state is current.social_state:
            return None
        mapping = {
            SocialState.ENGAGED: ("acknowledge", 60),
            SocialState.DISENGAGED: ("disengage", 60),
            SocialState.SEEKING_ATTENTION: ("seek_attention", 40),
            SocialState.IDLE: ("return_neutral", 20),
        }
        selected = mapping.get(current.social_state)
        if selected is None:
            return None
        kind, urgency = selected
        return BehaviorIntent(
            intent_id=uuid7(),
            correlation_id=uuid7(),
            session_id=current.session_id,
            kind=kind,
            urgency=urgency,
            created_at_mono_ns=current.as_of_mono_ns,
            expires_at_mono_ns=current.as_of_mono_ns + 2_000_000_000,
            target_person_id=current.primary_person_id,
        )
```

```python
# backend/src/social_lamp/behavior/compositor.py
from uuid6 import uuid7

from social_lamp.domain.contracts import (
    BehaviorIntent,
    BehaviorTimeline,
    LightKeyframe,
    MotionKeyframe,
    MotionTrack,
)


class BehaviorCompositor:
    def compose(
        self, intent: BehaviorIntent, current_pose: dict[str, float]
    ) -> BehaviorTimeline:
        if intent.kind != "acknowledge":
            return BehaviorTimeline(
                timeline_id=uuid7(),
                intent_id=intent.intent_id,
                correlation_id=intent.correlation_id,
                priority=intent.urgency,
                duration_ms=500,
                cancellable=True,
                motion_tracks=(),
            )
        tracks = tuple(
            MotionTrack(
                channel=channel,
                keyframes=(
                    MotionKeyframe(offset_ms=0, value=current_pose.get(channel, 0.0)),
                    MotionKeyframe(offset_ms=350, value=target),
                    MotionKeyframe(offset_ms=700, value=0.0),
                ),
            )
            for channel, target in (("head_yaw", 0.12), ("head_pitch", 0.25))
        )
        return BehaviorTimeline(
            timeline_id=uuid7(),
            intent_id=intent.intent_id,
            correlation_id=intent.correlation_id,
            priority=intent.urgency,
            duration_ms=700,
            cancellable=True,
            motion_tracks=tracks,
            light_track=(
                LightKeyframe(offset_ms=0, rgb=(1.0, 0.55, 0.2), brightness=0.2),
                LightKeyframe(offset_ms=350, rgb=(1.0, 0.55, 0.2), brightness=0.8),
                LightKeyframe(offset_ms=700, rgb=(1.0, 0.55, 0.2), brightness=0.3),
            ),
        )
```

```python
# backend/src/social_lamp/adapters/base.py
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from social_lamp.domain.contracts import BehaviorTimeline


class AdapterCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)
    motion_channels: tuple[str, ...]
    supports_light: bool
    supports_audio: bool


class LampAdapter(Protocol):
    async def capabilities(self) -> AdapterCapabilities: ...
    async def execute(self, timeline: BehaviorTimeline) -> UUID: ...
    async def cancel(self, execution_id: UUID, reason: str) -> None: ...
    async def neutralize(self, reason: str) -> UUID: ...
```

- [ ] **Step 4: Verify the behavior slice and contract validation**

Run:

```powershell
uv run pytest tests/behavior/test_policy_compositor.py tests/domain/test_contracts.py -v
uv run pytest -q
uv run ruff check backend tests
uv run mypy
```

Expected: focused and full suites pass.

- [ ] **Step 5: Commit behavior foundations**

```powershell
git add backend/src/social_lamp/domain backend/src/social_lamp/behavior backend/src/social_lamp/adapters tests/behavior
git commit -m "feat: add behavior policy and compositor"
```

### Task 5: Expose snapshots and timelines through FastAPI and WebSocket

**Files:**
- Create: `backend/src/social_lamp/api/app.py`
- Create: `backend/src/social_lamp/api/hub.py`
- Create: `backend/src/social_lamp/adapters/simulator.py`
- Modify: `backend/src/social_lamp/main.py`
- Create: `tests/api/test_app.py`

- [ ] **Step 1: Write failing API and simulator-adapter tests**

```python
# tests/api/test_app.py
from fastapi.testclient import TestClient

from social_lamp.api.app import create_app


def test_health_and_initial_snapshot_are_available() -> None:
    with TestClient(create_app()) as client:
        assert client.get("/api/health").json() == {"status": "healthy"}
        response = client.get("/api/world")
        assert response.status_code == 200
        assert response.json()["social_state"] == "idle"


def test_websocket_receives_initial_snapshot() -> None:
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws") as socket:
            message = socket.receive_json()
            assert message["type"] == "world_snapshot"
            assert message["body"]["revision"] == 0
```

- [ ] **Step 2: Run tests to confirm the API package is missing**

Run: `uv run pytest tests/api/test_app.py -v`

Expected: FAIL importing `social_lamp.api.app`.

- [ ] **Step 3: Implement the connection hub and application lifecycle**

```python
# backend/src/social_lamp/api/hub.py
from fastapi import WebSocket


class ConnectionHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    async def broadcast(self, message: dict[str, object]) -> None:
        failed: list[WebSocket] = []
        for client in self._clients:
            try:
                await client.send_json(message)
            except RuntimeError:
                failed.append(client)
        for client in failed:
            self.disconnect(client)
```

```python
# backend/src/social_lamp/api/app.py
from contextlib import asynccontextmanager
from uuid6 import uuid7

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from social_lamp.api.hub import ConnectionHub
from social_lamp.domain.clock import SystemClock
from social_lamp.world.model import WorldModel


def create_app() -> FastAPI:
    clock = SystemClock()
    world = WorldModel(session_id=uuid7(), clock=clock)
    hub = ConnectionHub()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield

    app = FastAPI(title="Simulated Social Lamp", version="0.1.0", lifespan=lifespan)
    app.state.world = world
    app.state.hub = hub

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/api/world")
    async def current_world() -> dict[str, object]:
        return world.snapshot.model_dump(mode="json")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await hub.connect(websocket)
        await websocket.send_json(
            {"type": "world_snapshot", "body": world.snapshot.model_dump(mode="json")}
        )
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            hub.disconnect(websocket)

    return app
```

Update `main.py` to import `create_app` from `social_lamp.api.app`. Implement `SimulatorAdapter` as an adapter that broadcasts `{"type": "behavior_timeline", "body": timeline.model_dump(mode="json")}` and returns the timeline ID as the execution ID.

- [ ] **Step 4: Verify HTTP and WebSocket behavior**

Run:

```powershell
uv run pytest tests/api/test_app.py -v
uv run pytest -q
uv run ruff check backend tests
uv run mypy
```

Expected: API tests pass; the full backend suite remains green.

- [ ] **Step 5: Commit the runnable backend slice**

```powershell
git add backend/src/social_lamp/api backend/src/social_lamp/adapters/simulator.py backend/src/social_lamp/main.py tests/api
git commit -m "feat: stream world state and simulator timelines"
```

### Task 6: Render the six-channel lamp from server timelines

**Files:**
- Create: `frontend/src/contracts/domain.ts`
- Create: `frontend/src/state/store.ts`
- Create: `frontend/src/scene/pose.ts`
- Create: `frontend/src/scene/LampScene.tsx`
- Modify: `frontend/src/App.tsx`
- Create: `frontend/src/scene/pose.test.ts`

- [ ] **Step 1: Write the failing pose-mapping test**

```typescript
// frontend/src/scene/pose.test.ts
import { describe, expect, it } from "vitest";
import { poseToRotations } from "./pose";

describe("poseToRotations", () => {
  it("maps all six normalized channels to bounded radians", () => {
    const rotations = poseToRotations({
      base_yaw: 1,
      shoulder_pitch: -1,
      elbow_pitch: 0.5,
      wrist_pitch: 0.25,
      head_yaw: -0.5,
      head_pitch: 0,
    });
    expect(Object.keys(rotations)).toHaveLength(6);
    expect(rotations.base_yaw).toBeCloseTo(Math.PI / 2);
    expect(rotations.shoulder_pitch).toBeCloseTo(-Math.PI / 3);
  });
});
```

- [ ] **Step 2: Run Vitest to verify the pose module is absent**

Run: `pnpm --dir frontend test -- --run src/scene/pose.test.ts`

Expected: FAIL resolving `./pose`.

- [ ] **Step 3: Implement typed pose mapping and the articulated scene**

```typescript
// frontend/src/contracts/domain.ts
export type MotionChannel =
  | "base_yaw"
  | "shoulder_pitch"
  | "elbow_pitch"
  | "wrist_pitch"
  | "head_yaw"
  | "head_pitch";

export type LampPose = Record<MotionChannel, number>;

export interface MotionKeyframe {
  offset_ms: number;
  value: number;
  easing: string;
}

export interface MotionTrack {
  channel: MotionChannel;
  keyframes: MotionKeyframe[];
}

export interface BehaviorTimeline {
  timeline_id: string;
  intent_id: string;
  duration_ms: number;
  motion_tracks: MotionTrack[];
}
```

```typescript
// frontend/src/scene/pose.ts
import type { LampPose } from "../contracts/domain";

const LIMITS: LampPose = {
  base_yaw: Math.PI / 2,
  shoulder_pitch: Math.PI / 3,
  elbow_pitch: Math.PI / 2,
  wrist_pitch: Math.PI / 2,
  head_yaw: Math.PI / 2,
  head_pitch: Math.PI / 3,
};

export function neutralPose(): LampPose {
  return {
    base_yaw: 0,
    shoulder_pitch: 0,
    elbow_pitch: 0,
    wrist_pitch: 0,
    head_yaw: 0,
    head_pitch: 0,
  };
}

export function poseToRotations(pose: LampPose): LampPose {
  return Object.fromEntries(
    Object.entries(pose).map(([channel, value]) => [
      channel,
      Math.max(-1, Math.min(1, value)) * LIMITS[channel as keyof LampPose],
    ]),
  ) as LampPose;
}
```

Implement `LampScene.tsx` as nested groups in this exact order: base yaw, shoulder pitch, elbow pitch, wrist pitch, head yaw, head pitch. Use `useFrame` to interpolate the current pose and render primitive cylinder/box/sphere meshes. `App.tsx` must place the scene in `<Canvas>` and show a connection banner outside the canvas.

- [ ] **Step 4: Verify pose logic, type checking, and production build**

Run:

```powershell
pnpm --dir frontend test -- --run src/scene/pose.test.ts
pnpm --dir frontend exec tsc --noEmit
pnpm --dir frontend build
```

Expected: pose test, type check, and build pass.

- [ ] **Step 5: Commit the first visible simulator**

```powershell
git add frontend/src
git commit -m "feat: render six-channel simulated lamp"
```

## Phase 2: Engagement, Objects, and Evidence Memory

### Task 7: Implement engagement fusion and hysteresis with a fake clock

**Files:**
- Create: `backend/src/social_lamp/perception/engagement.py`
- Create: `tests/perception/test_engagement.py`

- [ ] **Step 1: Write failing score and dwell tests**

```python
# tests/perception/test_engagement.py
from social_lamp.domain.contracts import SocialState
from social_lamp.perception.engagement import EngagementEstimator, EngagementSignals


def test_missing_gaze_renormalizes_available_signals() -> None:
    estimator = EngagementEstimator()
    sample = estimator.sample(
        EngagementSignals(
            face_presence=1.0,
            head_toward=1.0,
            gaze_toward=None,
            proximity=1.0,
            directed_speech=0.0,
            confidence=0.9,
        ),
        mono_ns=0,
    )
    assert 0.79 < sample.raw_score < 0.81


def test_engagement_requires_entry_dwell_and_exit_hysteresis() -> None:
    estimator = EngagementEstimator(smoothing_ms=0)
    attentive = EngagementSignals(1.0, 1.0, 1.0, 1.0, 0.0, 0.9)
    away = EngagementSignals(1.0, 0.0, 0.0, 1.0, 0.0, 0.9)
    assert estimator.sample(attentive, 0).state is SocialState.CANDIDATE
    assert estimator.sample(attentive, 699_000_000).state is SocialState.CANDIDATE
    assert estimator.sample(attentive, 700_000_000).state is SocialState.ENGAGED
    assert estimator.sample(away, 1_899_000_000).state is SocialState.ENGAGED
    assert estimator.sample(away, 1_900_000_000).state is SocialState.DISENGAGED
```

- [ ] **Step 2: Run tests to verify the estimator is absent**

Run: `uv run pytest tests/perception/test_engagement.py -v`

Expected: FAIL importing `perception.engagement`.

- [ ] **Step 3: Implement normalized fusion, EMA, and duration-based transitions**

```python
# backend/src/social_lamp/perception/engagement.py
from dataclasses import dataclass

from social_lamp.domain.contracts import SocialState


@dataclass(frozen=True)
class EngagementSignals:
    face_presence: float | None
    head_toward: float | None
    gaze_toward: float | None
    proximity: float | None
    directed_speech: float | None
    confidence: float


@dataclass(frozen=True)
class EngagementSample:
    raw_score: float
    smoothed_score: float
    confidence: float
    state: SocialState


class EngagementEstimator:
    WEIGHTS = (0.20, 0.30, 0.25, 0.10, 0.15)

    def __init__(self, *, smoothing_ms: int = 250) -> None:
        self._smoothing_ms = smoothing_ms
        self._smoothed: float | None = None
        self._last_ns: int | None = None
        self._state = SocialState.IDLE
        self._candidate_since_ns: int | None = None
        self._away_since_ns: int | None = None

    def _fuse(self, signals: EngagementSignals) -> float:
        values = (
            signals.face_presence,
            signals.head_toward,
            signals.gaze_toward,
            signals.proximity,
            signals.directed_speech,
        )
        available = [(value, weight) for value, weight in zip(values, self.WEIGHTS) if value is not None]
        weight_sum = sum(weight for _, weight in available)
        return sum(value * weight for value, weight in available) / weight_sum if weight_sum else 0.0

    def sample(self, signals: EngagementSignals, mono_ns: int) -> EngagementSample:
        raw = self._fuse(signals)
        if self._smoothed is None or self._smoothing_ms == 0 or self._last_ns is None:
            self._smoothed = raw
        else:
            elapsed_ms = max(0.0, (mono_ns - self._last_ns) / 1_000_000)
            alpha = min(1.0, elapsed_ms / self._smoothing_ms)
            self._smoothed += alpha * (raw - self._smoothed)
        self._last_ns = mono_ns
        score = self._smoothed
        if signals.confidence < 0.45:
            return EngagementSample(raw, score, signals.confidence, self._state)
        if self._state in {SocialState.IDLE, SocialState.CANDIDATE}:
            if score >= 0.68:
                if self._candidate_since_ns is None:
                    self._candidate_since_ns = mono_ns
                self._state = SocialState.CANDIDATE
                if mono_ns - self._candidate_since_ns >= 700_000_000:
                    self._state = SocialState.ENGAGED
                    self._candidate_since_ns = None
                    self._away_since_ns = None
            elif score >= 0.45:
                self._state = SocialState.CANDIDATE
                self._candidate_since_ns = self._candidate_since_ns or mono_ns
            elif score < 0.35:
                self._state = SocialState.IDLE
                self._candidate_since_ns = None
        elif self._state is SocialState.ENGAGED:
            if score < 0.38:
                if self._away_since_ns is None:
                    self._away_since_ns = mono_ns
                if mono_ns - self._away_since_ns >= 1_200_000_000:
                    self._state = SocialState.DISENGAGED
            else:
                self._away_since_ns = None
        elif self._state is SocialState.DISENGAGED and score >= 0.62:
            if self._candidate_since_ns is None:
                self._candidate_since_ns = mono_ns
            if mono_ns - self._candidate_since_ns >= 500_000_000:
                self._state = SocialState.ENGAGED
        return EngagementSample(raw, score, signals.confidence, self._state)
```

- [ ] **Step 4: Verify boundary cases and property ranges**

Add Hypothesis coverage asserting every combination of available `[0,1]` signals yields scores in `[0,1]`, then run:

```powershell
uv run pytest tests/perception/test_engagement.py -v
uv run pytest -q
uv run ruff check backend tests
uv run mypy
```

Expected: all engagement, full-suite, and static checks pass.

- [ ] **Step 5: Commit the engagement state machine**

```powershell
git add backend/src/social_lamp/perception/engagement.py tests/perception/test_engagement.py
git commit -m "feat: add temporal engagement estimator"
```

### Task 8: Integrate bounded camera capture and MediaPipe face evidence

**Files:**
- Create: `backend/src/social_lamp/capture/frames.py`
- Create: `backend/src/social_lamp/perception/faces.py`
- Create: `tests/capture/test_frames.py`
- Create: `tests/perception/test_faces.py`
- Modify: `pyproject.toml`, `uv.lock`

- [ ] **Step 1: Write failing latest-frame and face-mapping tests**

```python
# tests/capture/test_frames.py
import numpy as np

from social_lamp.capture.frames import LatestFrameBuffer


def test_latest_frame_replaces_oldest_without_queueing() -> None:
    buffer = LatestFrameBuffer(capacity=3)
    for index in range(5):
        buffer.put(np.full((2, 2, 3), index, dtype=np.uint8), mono_ns=index)
    latest = buffer.latest()
    assert latest is not None
    assert latest.mono_ns == 4
    assert int(latest.image[0, 0, 0]) == 4
    assert buffer.dropped == 2
```

```python
# tests/perception/test_faces.py
from social_lamp.perception.faces import face_result_to_signals


def test_low_quality_eyes_disable_gaze_signal() -> None:
    signals = face_result_to_signals(
        face_confidence=0.92,
        yaw_degrees=4.0,
        pitch_degrees=-3.0,
        gaze_score=0.8,
        gaze_quality=0.2,
        face_area_ratio=0.12,
    )
    assert signals.gaze_toward is None
    assert signals.head_toward > 0.8
```

- [ ] **Step 2: Run tests before installing CV dependencies**

Run: `uv run pytest tests/capture/test_frames.py tests/perception/test_faces.py -v`

Expected: FAIL importing capture and face modules.

- [ ] **Step 3: Add CV dependencies and implement testable adapters**

Run: `uv add opencv-python mediapipe`

```python
# backend/src/social_lamp/capture/frames.py
from collections import deque
from dataclasses import dataclass
from threading import Lock

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CapturedFrame:
    image: NDArray[np.uint8]
    mono_ns: int


class LatestFrameBuffer:
    def __init__(self, *, capacity: int = 3) -> None:
        self._frames: deque[CapturedFrame] = deque(maxlen=capacity)
        self._lock = Lock()
        self.dropped = 0

    def put(self, image: NDArray[np.uint8], mono_ns: int) -> None:
        with self._lock:
            if len(self._frames) == self._frames.maxlen:
                self.dropped += 1
            self._frames.append(CapturedFrame(image.copy(), mono_ns))

    def latest(self) -> CapturedFrame | None:
        with self._lock:
            return self._frames[-1] if self._frames else None
```

```python
# backend/src/social_lamp/perception/faces.py
from social_lamp.perception.engagement import EngagementSignals


def face_result_to_signals(
    *,
    face_confidence: float,
    yaw_degrees: float,
    pitch_degrees: float,
    gaze_score: float,
    gaze_quality: float,
    face_area_ratio: float,
) -> EngagementSignals:
    head = max(0.0, min(1.0, 1.0 - abs(yaw_degrees) / 35.0 - abs(pitch_degrees) / 30.0))
    proximity = max(0.0, min(1.0, face_area_ratio / 0.15))
    gaze = max(0.0, min(1.0, gaze_score)) if gaze_quality >= 0.45 else None
    return EngagementSignals(
        face_presence=max(0.0, min(1.0, face_confidence)),
        head_toward=head,
        gaze_toward=gaze,
        proximity=proximity,
        directed_speech=0.0,
        confidence=min(face_confidence, max(gaze_quality, 0.45)),
    )
```

Implement `MediaPipeFaceAdapter.process(frame)` behind a protocol-injected landmarker. It must return pure face result records, skip frames older than 300 ms, and never import or initialize a model in unit tests. Add a command-line camera probe that prints resolution/FPS and exits without writing files.

- [ ] **Step 4: Verify adapters with fakes and a manual probe**

Run:

```powershell
uv run pytest tests/capture/test_frames.py tests/perception/test_faces.py -v
uv run pytest -q
uv run ruff check backend tests
uv run mypy
uv run python -m social_lamp.capture.frames --probe
```

Expected: automated tests pass. The probe reports a camera or exits with a clear `camera_unavailable` status without crashing.

- [ ] **Step 5: Commit live face-perception infrastructure**

```powershell
git add pyproject.toml uv.lock backend/src/social_lamp/capture backend/src/social_lamp/perception/faces.py tests/capture tests/perception/test_faces.py
git commit -m "feat: add bounded capture and face evidence"
```

### Task 9: Add object tracking and scene-relative localization

**Files:**
- Create: `backend/src/social_lamp/perception/location.py`
- Create: `backend/src/social_lamp/perception/objects.py`
- Create: `tests/perception/test_objects.py`
- Modify: `pyproject.toml`, `uv.lock`

- [ ] **Step 1: Write failing stability and location tests**

```python
# tests/perception/test_objects.py
from social_lamp.perception.location import locate_box
from social_lamp.perception.objects import Detection, ObjectTrack


def test_scene_relative_location_uses_normalized_regions() -> None:
    location = locate_box((0.72, 0.60, 0.92, 0.90), anchors={"desk": (0.0, 0.55, 1.0, 1.0)})
    assert location.horizontal_region == "right"
    assert location.anchor_name == "desk"


def test_track_becomes_stable_after_five_consistent_detections() -> None:
    track = ObjectTrack(track_id="object-1")
    for index in range(5):
        track.add(
            Detection("mug", 0.8, (0.1, 0.1, 0.3, 0.5), index * 200_000_000)
        )
    assert track.is_stable
    assert track.label == "mug"
```

- [ ] **Step 2: Run tests to verify object modules are absent**

Run: `uv run pytest tests/perception/test_objects.py -v`

Expected: FAIL importing location and object modules.

- [ ] **Step 3: Implement pure localization and bounded tracking**

Run: `uv add ultralytics`

```python
# backend/src/social_lamp/perception/location.py
from dataclasses import dataclass

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class SceneLocation:
    horizontal_region: str
    depth_band: str
    anchor_name: str | None


def _intersection_over_box(box: BBox, anchor: BBox) -> float:
    x1, y1, x2, y2 = box
    ax1, ay1, ax2, ay2 = anchor
    width = max(0.0, min(x2, ax2) - max(x1, ax1))
    height = max(0.0, min(y2, ay2) - max(y1, ay1))
    area = max(0.000001, (x2 - x1) * (y2 - y1))
    return width * height / area


def locate_box(box: BBox, *, anchors: dict[str, BBox]) -> SceneLocation:
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    horizontal = "left" if center_x < 1 / 3 else "right" if center_x >= 2 / 3 else "center"
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    depth = "foreground" if area >= 0.20 else "midground" if area >= 0.06 else "background"
    candidates = [name for name, anchor in anchors.items() if _intersection_over_box(box, anchor) >= 0.50]
    return SceneLocation(horizontal, depth, sorted(candidates)[0] if candidates else None)
```

```python
# backend/src/social_lamp/perception/objects.py
from collections import Counter, deque
from dataclasses import dataclass, field

from social_lamp.perception.location import BBox


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: BBox
    mono_ns: int


@dataclass
class ObjectTrack:
    track_id: str
    detections: deque[Detection] = field(default_factory=lambda: deque(maxlen=20))

    def add(self, detection: Detection) -> None:
        self.detections.append(detection)

    @property
    def label(self) -> str:
        return Counter(item.label for item in self.detections).most_common(1)[0][0]

    @property
    def is_stable(self) -> bool:
        if len(self.detections) < 5:
            return False
        recent = list(self.detections)[-5:]
        if recent[-1].mono_ns - recent[0].mono_ns > 1_000_000_000:
            return False
        if sum(item.confidence for item in recent) / len(recent) < 0.55:
            return False
        count = Counter(item.label for item in recent).most_common(1)[0][1]
        return count / len(recent) >= 0.75
```

Implement `FastObjectDetector` behind an injected model protocol and a capacity-two `EnrichmentQueue` that coalesces requests by track ID. Tests use fakes; model downloads occur only in an explicit setup command.

- [ ] **Step 4: Verify objects and overload behavior**

Add tests for exact horizontal/depth boundaries, conflicting labels, old detections, and enrichment coalescing, then run:

```powershell
uv run pytest tests/perception/test_objects.py -v
uv run pytest -q
uv run ruff check backend tests
uv run mypy
```

Expected: all tests and static checks pass without downloading a model.

- [ ] **Step 5: Commit object perception**

```powershell
git add pyproject.toml uv.lock backend/src/social_lamp/perception/location.py backend/src/social_lamp/perception/objects.py tests/perception/test_objects.py
git commit -m "feat: add stable object tracking and localization"
```

### Task 10: Persist evidence and answer deterministic memory queries

**Files:**
- Modify: `backend/src/social_lamp/domain/contracts.py`
- Create: `backend/src/social_lamp/memory/repository.py`
- Create: `backend/src/social_lamp/memory/migrations/001_initial.sql`
- Create: `tests/memory/test_repository.py`

- [ ] **Step 1: Write failing transactional memory tests**

```python
# tests/memory/test_repository.py
from pathlib import Path

import pytest

from social_lamp.memory.repository import MemoryRepository, ObservationWrite


@pytest.mark.asyncio
async def test_last_seen_returns_newest_grounded_observation(tmp_path: Path) -> None:
    repository = await MemoryRepository.open(tmp_path / "memory.db")
    await repository.record(
        ObservationWrite.example("keys", "left", observed_at_mono_ns=10)
    )
    newest = await repository.record(
        ObservationWrite.example("keys", "right", observed_at_mono_ns=20)
    )
    result = await repository.find_last_seen("keys")
    assert result.status == "found"
    assert result.horizontal_region == "right"
    assert result.evidence_ids == (newest,)
    await repository.close()


@pytest.mark.asyncio
async def test_unknown_object_returns_not_found(tmp_path: Path) -> None:
    repository = await MemoryRepository.open(tmp_path / "memory.db")
    result = await repository.find_last_seen("wallet")
    assert result.status == "not_found"
    assert result.evidence_ids == ()
    await repository.close()
```

- [ ] **Step 2: Run tests to verify memory infrastructure is absent**

Run: `uv run pytest tests/memory/test_repository.py -v`

Expected: FAIL importing `social_lamp.memory.repository`.

- [ ] **Step 3: Create the schema and repository**

```sql
-- backend/src/social_lamp/memory/migrations/001_initial.sql
PRAGMA foreign_keys = ON;
CREATE TABLE schema_meta(version INTEGER PRIMARY KEY, migrated_at_utc TEXT NOT NULL);
CREATE TABLE sessions(
  session_id TEXT PRIMARY KEY,
  started_at_utc TEXT NOT NULL,
  ended_at_utc TEXT,
  source_mode TEXT NOT NULL,
  config_hash TEXT NOT NULL
);
CREATE TABLE object_tracks(
  track_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(session_id),
  first_seen_utc TEXT NOT NULL,
  last_seen_utc TEXT NOT NULL,
  current_label TEXT NOT NULL,
  current_label_confidence REAL NOT NULL,
  active INTEGER NOT NULL
);
CREATE TABLE observations(
  observation_id TEXT PRIMARY KEY,
  track_id TEXT NOT NULL REFERENCES object_tracks(track_id),
  session_id TEXT NOT NULL REFERENCES sessions(session_id),
  observed_at_utc TEXT NOT NULL,
  observed_at_mono_ns INTEGER NOT NULL,
  label TEXT NOT NULL,
  label_source TEXT NOT NULL,
  detection_confidence REAL NOT NULL,
  bbox_json TEXT NOT NULL,
  horizontal_region TEXT,
  depth_band TEXT,
  anchor_name TEXT,
  location_confidence REAL NOT NULL,
  frame_ref TEXT,
  snapshot_path TEXT,
  correlation_id TEXT NOT NULL
);
CREATE TABLE last_known_objects(
  canonical_label TEXT PRIMARY KEY,
  observation_id TEXT NOT NULL REFERENCES observations(observation_id),
  updated_at_utc TEXT NOT NULL
);
CREATE TABLE behavior_preferences(
  context_key TEXT NOT NULL,
  behavior_key TEXT NOT NULL,
  score REAL NOT NULL,
  evidence_count INTEGER NOT NULL,
  updated_at_utc TEXT NOT NULL,
  PRIMARY KEY(context_key, behavior_key)
);
CREATE INDEX observations_label_time ON observations(label, observed_at_mono_ns DESC);
CREATE INDEX observations_session_time ON observations(session_id, observed_at_mono_ns DESC);
CREATE INDEX observations_track_time ON observations(track_id, observed_at_mono_ns DESC);
```

Enable WAL mode on open. `record` must insert the append-only observation and upsert `last_known_objects` in one transaction. On any exception, roll back both writes.

Add these immutable models to `domain/contracts.py`:

```python
class MemoryQuery(FrozenModel):
    kind: str
    object_label: str = Field(min_length=1, max_length=80)
    session_scope: UUID | None = None
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
```

Implement exact-alias lookup before normalized canonical-label lookup. `find_last_seen` returns only fields from the selected row and includes its observation ID in `evidence_ids`.

`ObservationWrite.example` is a test factory with deterministic UUIDv7-compatible constants; production code constructs records explicitly. Parameter-bind all SQL values.

- [ ] **Step 4: Prove rollback, ambiguity, alias, and ordering behavior**

Add tests that inject a failure between observation insert and last-known update, verify the transaction leaves neither change, then cover aliases, ambiguous labels, session scope, and `before_utc`. Run:

```powershell
uv run pytest tests/memory/test_repository.py -v
uv run pytest -q
uv run ruff check backend tests
uv run mypy
```

Expected: all repository and full-suite tests pass.

- [ ] **Step 5: Commit evidence memory**

```powershell
git add backend/src/social_lamp/domain/contracts.py backend/src/social_lamp/memory tests/memory
git commit -m "feat: add transactional evidence memory"
```

## Phase 3: Grounded Recall, Replay, and Explainable Dashboard

### Task 11: Add offline grounded conversation and read-only memory tools

**Files:**
- Create: `backend/src/social_lamp/conversation/base.py`
- Create: `backend/src/social_lamp/conversation/grounding.py`
- Create: `backend/src/social_lamp/conversation/template.py`
- Create: `tests/conversation/test_template.py`

- [ ] **Step 1: Write failing grounded-response tests**

```python
# tests/conversation/test_template.py
import pytest

from social_lamp.conversation.grounding import validate_grounding
from social_lamp.conversation.template import TemplateConversationProvider
from social_lamp.domain.contracts import MemoryResult


@pytest.mark.asyncio
async def test_template_provider_answers_from_evidence() -> None:
    evidence = MemoryResult(
        status="found",
        canonical_label="keys",
        horizontal_region="right",
        depth_band="foreground",
        anchor_name="desk",
        observed_at_utc="2026-07-04T12:00:00Z",
        evidence_ids=("observation-1",),
        alternatives=(),
    )
    provider = TemplateConversationProvider(lambda _: evidence)
    response = await provider.handle_text("turn-1", "Where are my keys?")
    assert response.text == "I last saw the keys on the right side of the desk."
    assert response.evidence_ids == ("observation-1",)


def test_grounding_rejects_location_not_present_in_evidence() -> None:
    evidence = MemoryResult.not_found()
    assert not validate_grounding("Your wallet is on the shelf.", evidence)
```

- [ ] **Step 2: Run tests to confirm conversation modules are absent**

Run: `uv run pytest tests/conversation/test_template.py -v`

Expected: FAIL importing `social_lamp.conversation`.

- [ ] **Step 3: Implement the provider port, parser, templates, and validator**

```python
# backend/src/social_lamp/conversation/base.py
from dataclasses import dataclass
from typing import AsyncIterator, Protocol


@dataclass(frozen=True)
class ConversationResponse:
    text: str
    evidence_ids: tuple[str, ...]
    status: str


class ConversationProvider(Protocol):
    async def handle_text(self, turn_id: str, text: str) -> ConversationResponse: ...
    async def handle_audio(self, turn_id: str, chunks: AsyncIterator[bytes]) -> ConversationResponse: ...
    async def interrupt(self, turn_id: str, reason: str) -> None: ...
    async def close(self, reason: str) -> None: ...
```

```python
# backend/src/social_lamp/conversation/template.py
import re
from collections.abc import Awaitable, Callable

from social_lamp.conversation.base import ConversationResponse
from social_lamp.domain.contracts import MemoryQuery, MemoryResult


class TemplateConversationProvider:
    def __init__(
        self, query: Callable[[MemoryQuery], Awaitable[MemoryResult] | MemoryResult]
    ) -> None:
        self._query = query

    async def handle_text(self, turn_id: str, text: str) -> ConversationResponse:
        match = re.search(r"(?:where|when).*?(?:my|the|a)\s+([a-zA-Z][a-zA-Z -]{0,40})[?.!]*$", text.strip(), re.I)
        if match is None:
            return ConversationResponse(
                "I can answer where I last saw an object.", (), "unsupported"
            )
        label = match.group(1).strip().lower()
        query = MemoryQuery(kind="last_seen", object_label=label, limit=1)
        result_or_awaitable = self._query(query)
        if isinstance(result_or_awaitable, MemoryResult):
            result = result_or_awaitable
        else:
            result = await result_or_awaitable
        if result.status == "not_found":
            return ConversationResponse(
                f"I do not have reliable evidence for the {label}.", (), "not_found"
            )
        if result.status == "ambiguous":
            choices = ", ".join(result.alternatives)
            return ConversationResponse(
                f"I found more than one possible match: {choices}.", (), "ambiguous"
            )
        location = " ".join(
            part
            for part in (
                f"on the {result.horizontal_region} side" if result.horizontal_region else "",
                f"of the {result.anchor_name}" if result.anchor_name else "",
            )
            if part
        )
        return ConversationResponse(
            f"I last saw the {result.canonical_label} {location}.",
            result.evidence_ids,
            "found",
        )

    async def interrupt(self, turn_id: str, reason: str) -> None:
        return None

    async def close(self, reason: str) -> None:
        return None
```

```python
# backend/src/social_lamp/conversation/grounding.py
import re

from social_lamp.domain.contracts import MemoryResult

LOCATION_TOKENS = {"left", "center", "right", "foreground", "midground", "background"}


def validate_grounding(text: str, evidence: MemoryResult) -> bool:
    normalized = set(re.findall(r"[a-z]+", text.lower()))
    if evidence.status == "not_found":
        phrases = ("do not", "don't", "no evidence", "not know")
        return any(phrase in text.lower() for phrase in phrases) and not (normalized & LOCATION_TOKENS)
    allowed = {
        value
        for value in (
            evidence.horizontal_region,
            evidence.depth_band,
            evidence.anchor_name,
        )
        if value is not None
    }
    mentioned_locations = normalized & LOCATION_TOKENS
    if not mentioned_locations.issubset(allowed):
        return False
    if evidence.anchor_name is not None:
        return evidence.anchor_name.lower() in text.lower() or not mentioned_locations
    return bool(evidence.evidence_ids)
```

- [ ] **Step 4: Verify found, ambiguous, unsupported, and not-found turns**

Run:

```powershell
uv run pytest tests/conversation/test_template.py -v
uv run pytest -q
uv run ruff check backend tests
uv run mypy
```

Expected: conversation and full backend tests pass.

- [ ] **Step 5: Commit deterministic grounded recall**

```powershell
git add backend/src/social_lamp/conversation tests/conversation
git commit -m "feat: add offline grounded memory recall"
```

### Task 12: Record and deterministically replay correlated traces

**Files:**
- Create: `backend/src/social_lamp/replay/trace.py`
- Create: `tests/replay/test_trace.py`
- Create: `evaluation/fixtures/core-engagement/manifest.json`
- Create: `evaluation/fixtures/core-engagement/events.jsonl`

- [ ] **Step 1: Write failing trace round-trip and checksum tests**

```python
# tests/replay/test_trace.py
from pathlib import Path

import pytest

from social_lamp.replay.trace import TraceManifest, TraceReader, TraceWriter


@pytest.mark.asyncio
async def test_trace_round_trip_preserves_order_and_checksum(tmp_path: Path) -> None:
    writer = TraceWriter(tmp_path)
    await writer.open(TraceManifest.example())
    await writer.append("observation", 20, {"kind": "face_presence"})
    await writer.append("snapshot", 30, {"revision": 1})
    await writer.close()
    records = list(TraceReader(tmp_path).records())
    assert [record.sequence for record in records] == [1, 2]
    assert [record.record_type for record in records] == ["observation", "snapshot"]
    assert TraceReader(tmp_path).verify_checksum()
```

- [ ] **Step 2: Run tests to verify replay infrastructure is absent**

Run: `uv run pytest tests/replay/test_trace.py -v`

Expected: FAIL importing `social_lamp.replay.trace`.

- [ ] **Step 3: Implement JSONL recording and schema-gated reading**

```python
# backend/src/social_lamp/replay/trace.py
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class TraceManifest:
    schema_version: str
    application_version: str
    session_id: str
    configuration_hash: str

    @classmethod
    def example(cls) -> "TraceManifest":
        return cls("1.0", "test", "session-test", "config-test")


@dataclass(frozen=True)
class TraceRecord:
    sequence: int
    record_type: str
    recorded_at_mono_ns: int
    body: dict[str, object]


@dataclass(frozen=True)
class ReplayTrace:
    manifest: TraceManifest
    records: tuple[TraceRecord, ...]


class TraceWriter:
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._records: list[str] = []

    async def open(self, manifest: TraceManifest) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        (self._directory / "manifest.json").write_text(
            json.dumps(asdict(manifest), sort_keys=True), encoding="utf-8"
        )

    async def append(self, record_type: str, mono_ns: int, body: dict[str, object]) -> None:
        record = TraceRecord(len(self._records) + 1, record_type, mono_ns, body)
        self._records.append(json.dumps(asdict(record), sort_keys=True))

    async def close(self) -> None:
        content = "\n".join(self._records) + "\n"
        (self._directory / "events.jsonl").write_text(content, encoding="utf-8")
        digest = hashlib.sha256(content.encode()).hexdigest()
        (self._directory / "events.sha256").write_text(digest, encoding="ascii")


class TraceReader:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def records(self) -> Iterator[TraceRecord]:
        for line in (self._directory / "events.jsonl").read_text(encoding="utf-8").splitlines():
            yield TraceRecord(**json.loads(line))

    def verify_checksum(self) -> bool:
        content = (self._directory / "events.jsonl").read_bytes()
        expected = (self._directory / "events.sha256").read_text(encoding="ascii")
        return hashlib.sha256(content).hexdigest() == expected
```

Reject manifests whose major schema version is not `1`. Add the committed core fixture with a face observation, candidate snapshot, engaged snapshot, acknowledge intent, and simulator timeline using fixed IDs and monotonic times.

- [ ] **Step 4: Verify round-trip, incompatible schema, and fixture replay**

Run:

```powershell
uv run pytest tests/replay/test_trace.py -v
uv run pytest -q
uv run ruff check backend tests
uv run mypy
```

Expected: trace tests and full backend suite pass.

- [ ] **Step 5: Commit replay support and the first fixture**

```powershell
git add backend/src/social_lamp/replay tests/replay evaluation/fixtures/core-engagement
git commit -m "feat: add deterministic event replay"
```

### Task 13: Generate frontend contracts and build the explainable dashboard

**Files:**
- Create: `tools/export_contract_schemas.py`
- Create: `frontend/src/contracts/generated.ts`
- Create: `frontend/src/state/store.ts`
- Create: `frontend/src/components/PerceptionPanel.tsx`
- Create: `frontend/src/components/EvidenceTimeline.tsx`
- Create: `frontend/src/components/Inspector.tsx`
- Create: `frontend/src/components/DemoRail.tsx`
- Modify: `frontend/src/App.tsx`
- Create: `frontend/src/state/store.test.ts`
- Create: `frontend/src/components/Dashboard.test.tsx`

- [ ] **Step 1: Write failing reducer and evidence-display tests**

```typescript
// frontend/src/state/store.test.ts
import { describe, expect, it } from "vitest";
import { initialState, reduceServerMessage } from "./store";

describe("reduceServerMessage", () => {
  it("replaces state from a full snapshot and detects sequence gaps", () => {
    const first = reduceServerMessage(initialState, {
      sequence: 1,
      type: "world_snapshot",
      body: { revision: 4, social_state: "engaged", people: [], objects: [], health: [] },
    });
    expect(first.world?.revision).toBe(4);
    const gap = reduceServerMessage(first, {
      sequence: 3,
      type: "metric",
      body: { name: "frame_age_ms", value: 20 },
    });
    expect(gap.needsResync).toBe(true);
  });
});
```

```typescript
// frontend/src/components/Dashboard.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Inspector } from "./Inspector";

describe("Inspector", () => {
  it("shows evidence and degraded health without color-only meaning", () => {
    render(
      <Inspector
        state="engaged"
        evidence={[{ id: "observation-1", label: "keys", location: "right side of desk" }]}
        health={[{ component: "cloud", status: "degraded", detail: "offline fallback" }]}
      />,
    );
    expect(screen.getByText("keys")).toBeVisible();
    expect(screen.getByText(/degraded/i)).toBeVisible();
    expect(screen.getByText(/offline fallback/i)).toBeVisible();
  });
});
```

- [ ] **Step 2: Run frontend tests to confirm dashboard modules are absent**

Run: `pnpm --dir frontend test -- --run src/state/store.test.ts src/components/Dashboard.test.tsx`

Expected: FAIL resolving store and dashboard components.

- [ ] **Step 3: Export contracts and implement dashboard state/components**

Add `json-schema-to-typescript` as a frontend dev dependency. `tools/export_contract_schemas.py` must call `model_json_schema()` for `WorldSnapshot`, `BehaviorTimeline`, `MemoryResult`, and `ObservationEvent`, write one combined schema to `frontend/src/contracts/domain.schema.json`, then run `json2ts` to produce `generated.ts`.

Implement the reducer with this base shape and extend its message union with the generated contract bodies:

```typescript
// frontend/src/state/store.ts
import type {
  BehaviorTimeline,
  MemoryResult,
  ObservationEvent,
  WorldSnapshot,
} from "../contracts/generated";

interface MetricBody { name: string; value: number }
interface FaultBody { component: string; detail: string }

export interface DashboardState {
  world: WorldSnapshot | null;
  timeline: BehaviorTimeline | null;
  evidence: MemoryResult[];
  metrics: Record<string, number>;
  faults: FaultBody[];
  lastSequence: number;
  needsResync: boolean;
}

export type ServerMessage =
  | { sequence: number; type: "world_snapshot"; body: WorldSnapshot }
  | { sequence: number; type: "behavior_timeline"; body: BehaviorTimeline }
  | { sequence: number; type: "observation"; body: ObservationEvent }
  | { sequence: number; type: "memory_result"; body: MemoryResult }
  | { sequence: number; type: "metric"; body: MetricBody }
  | { sequence: number; type: "fault"; body: FaultBody };

export const initialState: DashboardState = {
  world: null,
  timeline: null,
  evidence: [],
  metrics: {},
  faults: [],
  lastSequence: 0,
  needsResync: false,
};

export function reduceServerMessage(state: DashboardState, message: ServerMessage): DashboardState {
  const gap = state.lastSequence !== 0 && message.sequence !== state.lastSequence + 1;
  const next = { ...state, lastSequence: message.sequence, needsResync: state.needsResync || gap };
  switch (message.type) {
    case "world_snapshot":
      return { ...next, world: message.body };
    case "behavior_timeline":
      return { ...next, timeline: message.body };
    case "memory_result":
      return { ...next, evidence: [...state.evidence, message.body] };
    case "metric":
      return { ...next, metrics: { ...state.metrics, [message.body.name]: message.body.value } };
    case "fault":
      return { ...next, faults: [...state.faults, message.body] };
    case "observation":
      return next;
  }
}
```

Build the four dashboard regions from the design and keep operator controls acknowledgement-driven.

- [ ] **Step 4: Verify generated types, components, accessibility, and build**

Run:

```powershell
uv run python tools/export_contract_schemas.py
pnpm --dir frontend exec json2ts -i src/contracts/domain.schema.json -o src/contracts/generated.ts
pnpm --dir frontend test -- --run
pnpm --dir frontend exec tsc --noEmit
pnpm --dir frontend build
```

Expected: schema generation produces no diff on a second run; all frontend tests and build pass.

- [ ] **Step 5: Commit the explainable dashboard**

```powershell
git add tools frontend/src frontend/package.json pnpm-lock.yaml
git commit -m "feat: add typed explainable dashboard"
```

## Phase 4: Live Orchestration, Audio, Cloud, and Bonuses

### Task 14: Orchestrate the complete replay-to-simulator pipeline

**Files:**
- Create: `backend/src/social_lamp/runtime/coordinator.py`
- Create: `backend/src/social_lamp/runtime/testing.py`
- Modify: `backend/src/social_lamp/api/app.py`
- Create: `tests/runtime/test_coordinator.py`

- [ ] **Step 1: Write a failing end-to-end coordinator test**

```python
# tests/runtime/test_coordinator.py
from pathlib import Path

import pytest

from social_lamp.runtime.coordinator import RuntimeCoordinator


@pytest.mark.asyncio
async def test_engagement_replay_reaches_simulator_adapter(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    await coordinator.start()
    await coordinator.replay(
        Path("evaluation/fixtures/core-engagement")
    )
    assert coordinator.world.snapshot.social_state.value == "engaged"
    assert coordinator.simulator.executed[-1].duration_ms == 700
    assert coordinator.metrics.counter("social_transition", state="engaged") == 1
    await coordinator.stop()
```

- [ ] **Step 2: Run the test to confirm runtime orchestration is absent**

Run: `uv run pytest tests/runtime/test_coordinator.py -v`

Expected: FAIL importing `social_lamp.runtime.coordinator`.

- [ ] **Step 3: Implement owned tasks and dependency injection**

```python
# backend/src/social_lamp/runtime/coordinator.py
from pathlib import Path

from social_lamp.behavior.compositor import BehaviorCompositor
from social_lamp.behavior.policy import BehaviorPolicy
from social_lamp.replay.trace import TraceReader


class RuntimeCoordinator:
    def __init__(self, *, world, simulator, metrics, memory) -> None:
        self.world = world
        self.simulator = simulator
        self.metrics = metrics
        self.memory = memory
        self._policy = BehaviorPolicy()
        self._compositor = BehaviorCompositor()
        self._running = False

    @classmethod
    def for_test(cls, *, database: Path) -> "RuntimeCoordinator":
        from social_lamp.runtime.testing import build_test_runtime

        return build_test_runtime(database)

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False
        await self.memory.close()

    async def replay(self, directory: Path) -> None:
        previous = self.world.snapshot
        for record in TraceReader(directory).records():
            if record.record_type != "observation":
                continue
            from social_lamp.domain.contracts import ObservationEvent

            current = self.world.apply(ObservationEvent.model_validate(record.body))
            if current.revision == previous.revision:
                continue
            intent = self._policy.on_transition(previous, current)
            if intent is not None:
                timeline = self._compositor.compose(intent, self.simulator.pose)
                await self.simulator.execute(timeline)
            self.metrics.increment("social_transition", state=current.social_state.value)
            previous = current
```

Create `runtime/testing.py` with fake simulator/metrics and an in-memory-compatible repository. Production `RuntimeCoordinator` receives capture, perception, world, memory, policy, compositor, conversation, trace, metrics, and adapter dependencies explicitly. It owns their asyncio tasks through one `TaskGroup`; stop cancels producers, drains critical memory writes, neutralizes the adapter, then closes resources.

Wire one coordinator into FastAPI lifespan. Add typed endpoints for starting/stopping sessions, selecting replay, submitting text, neutralizing, clearing memory, toggling bonuses, and exporting traces. Restrict the first release to loopback clients.

- [ ] **Step 4: Verify replay, lifecycle shutdown, and API command acknowledgement**

Run:

```powershell
uv run pytest tests/runtime/test_coordinator.py tests/api/test_app.py -v
uv run pytest -q
uv run ruff check backend tests
uv run mypy
```

Expected: replay reaches the simulator exactly once, shutdown leaves no pending tasks, and API tests pass.

- [ ] **Step 5: Commit the integrated deterministic runtime**

```powershell
git add backend/src/social_lamp/runtime backend/src/social_lamp/api tests/runtime tests/api
git commit -m "feat: orchestrate replay-to-simulator pipeline"
```

### Task 15: Add voice activity, interruption, speaker association, and affect evidence

**Files:**
- Create: `backend/src/social_lamp/audio/analysis.py`
- Create: `tests/audio/test_analysis.py`
- Modify: `pyproject.toml`, `uv.lock`

- [ ] **Step 1: Write failing audio-window and interruption tests**

```python
# tests/audio/test_analysis.py
from social_lamp.audio.analysis import AudioAnalyzer, AudioClass, VoiceFrame


def test_speech_starts_after_120_ms_and_ends_after_500_ms_silence() -> None:
    analyzer = AudioAnalyzer(frame_ms=20)
    states = [analyzer.push(VoiceFrame(True, AudioClass.DIRECT_SPEECH, 0.9)) for _ in range(6)]
    assert states[-1].speech_active
    for _ in range(24):
        state = analyzer.push(VoiceFrame(False, AudioClass.OTHER, 0.9))
    assert state.speech_active
    state = analyzer.push(VoiceFrame(False, AudioClass.OTHER, 0.9))
    assert not state.speech_active


def test_television_suppresses_unsolicited_sound() -> None:
    analyzer = AudioAnalyzer(frame_ms=20)
    state = analyzer.push(VoiceFrame(True, AudioClass.TELEVISION_MEDIA, 0.85))
    assert state.suppress_unsolicited_sound
    assert state.speaker_id is None
```

- [ ] **Step 2: Run tests to confirm audio analysis is absent**

Run: `uv run pytest tests/audio/test_analysis.py -v`

Expected: FAIL importing `social_lamp.audio.analysis`.

- [ ] **Step 3: Implement deterministic audio state before device adapters**

```python
# backend/src/social_lamp/audio/analysis.py
from dataclasses import dataclass
from enum import StrEnum


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


@dataclass(frozen=True)
class AudioState:
    speech_active: bool
    suppress_unsolicited_sound: bool
    speaker_id: str | None


@dataclass(frozen=True)
class VocalAffectObservation:
    valence_tendency: float
    arousal: float
    confidence: float
    window_ms: int
    speaker_id: str | None


class AudioAnalyzer:
    def __init__(self, *, frame_ms: int = 20) -> None:
        self._frame_ms = frame_ms
        self._voiced_ms = 0
        self._silence_ms = 0
        self._active = False

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
        return AudioState(self._active, media and frame.confidence >= 0.65, speaker)
```

Run `uv add sounddevice webrtcvad-wheels`. Add device adapters that produce 20 ms mono PCM chunks and inject VAD/classifier protocols. Implement active-speaker association as a pure scorer combining mouth correlation, visual plausibility, and primary continuity; require `0.65`. Implement affect output only after 3-8 voiced seconds with bounded valence/arousal and confidence; discard it at session end.

- [ ] **Step 4: Verify thresholds, cancellation event, session cleanup, and device absence**

Add tests for speech during simulator audio producing an immediate priority-90 listen intent and cancellation call, anonymous speaker uncertainty, affect confidence below `0.60`, and missing microphone health. Run:

```powershell
uv run pytest tests/audio/test_analysis.py tests/behavior -v
uv run pytest -q
uv run ruff check backend tests
uv run mypy
```

Expected: audio and behavior tests pass without requiring a microphone.

- [ ] **Step 5: Commit local audio intelligence**

```powershell
git add pyproject.toml uv.lock backend/src/social_lamp/audio tests/audio backend/src/social_lamp/behavior
git commit -m "feat: add interruption-aware audio analysis"
```

### Task 16: Add cloud conversation behind the provider port

**Files:**
- Create: `backend/src/social_lamp/conversation/openai_realtime.py`
- Create: `backend/src/social_lamp/config.py`
- Create: `backend/src/social_lamp/runtime/providers.py`
- Create: `tests/conversation/test_openai_adapter.py`
- Modify: `pyproject.toml`, `uv.lock`

- [ ] **Step 1: Write failing provider-selection and grounding-fallback tests**

```python
# tests/conversation/test_openai_adapter.py
import pytest

from social_lamp.config import Settings
from social_lamp.conversation.openai_realtime import OpenAIRealtimeProvider
from social_lamp.conversation.template import TemplateConversationProvider
from social_lamp.runtime.providers import build_conversation_provider


def test_missing_api_key_selects_template_provider() -> None:
    provider = build_conversation_provider(Settings(openai_api_key=None))
    assert isinstance(provider, TemplateConversationProvider)


@pytest.mark.asyncio
async def test_ungrounded_cloud_answer_is_replaced_by_template(fake_openai_client) -> None:
    fake_openai_client.answer = "The keys are on the shelf."
    provider = OpenAIRealtimeProvider(client=fake_openai_client, model="test-model")
    response = await provider.handle_grounded_turn(
        turn_id="turn-1",
        text="Where are my keys?",
        evidence_result=fake_openai_client.not_found_result,
    )
    assert response.status == "not_found"
    assert "do not have reliable evidence" in response.text
```

- [ ] **Step 2: Run tests before installing the SDK**

Run: `uv run pytest tests/conversation/test_openai_adapter.py -v`

Expected: FAIL importing cloud adapter/configuration.

- [ ] **Step 3: Implement explicit settings and an injected-client adapter**

Run: `uv add openai pydantic-settings`

`Settings` reads `OPENAI_API_KEY`, `OPENAI_REALTIME_MODEL`, database/snapshot paths, camera index, retention days, and feature flags. It must never print secret values. `build_conversation_provider` chooses the template provider unless both provider name and key are present.

`OpenAIRealtimeProvider` exposes only the three read-only memory tools, caps tool results at ten observations, validates every final response with `validate_grounding`, replaces invalid output with the template result, limits one tool retry, and implements interruption by cancelling provider and adapter audio. Tests inject a fake client; no automated test contacts the network.

- [ ] **Step 4: Verify provider selection, timeout, retry, interruption, and grounding**

Run:

```powershell
uv run pytest tests/conversation/test_openai_adapter.py tests/conversation/test_template.py -v
uv run pytest -q
uv run ruff check backend tests
uv run mypy
```

Expected: all cloud tests pass using fakes; core tests pass with no API key.

- [ ] **Step 5: Commit the optional cloud provider**

```powershell
git add pyproject.toml uv.lock backend/src/social_lamp/config.py backend/src/social_lamp/conversation backend/src/social_lamp/runtime/providers.py tests/conversation
git commit -m "feat: add grounded cloud conversation adapter"
```

### Task 17: Complete attention escalation and adaptive preferences

**Files:**
- Create: `backend/src/social_lamp/behavior/preferences.py`
- Modify: `backend/src/social_lamp/behavior/policy.py`
- Modify: `backend/src/social_lamp/memory/repository.py`
- Create: `tests/behavior/test_attention_preferences.py`

- [ ] **Step 1: Write failing escalation and learning tests**

```python
# tests/behavior/test_attention_preferences.py
from social_lamp.behavior.preferences import PreferenceModel
from social_lamp.behavior.policy import AttentionSchedule


def test_attention_schedule_escalates_then_exhausts() -> None:
    schedule = AttentionSchedule(disengaged_at_ns=0)
    assert schedule.intent_at(4_999_000_000) is None
    assert schedule.intent_at(5_000_000_000).parameters["level"] == 1
    assert schedule.intent_at(15_000_000_000).parameters["level"] == 2
    assert schedule.intent_at(35_000_000_000).parameters["level"] == 3
    assert schedule.intent_at(36_000_000_000) is None
    assert schedule.exhausted


def test_preferences_are_bounded_auditable_and_decay() -> None:
    model = PreferenceModel()
    for _ in range(10):
        model.record("seek_attention:quiet", "light-pulse", "reengaged")
    assert model.score("seek_attention:quiet", "light-pulse") == 1.5
    audit = model.audit[-1]
    assert audit.previous_score <= audit.new_score
    model.start_session()
    assert model.score("seek_attention:quiet", "light-pulse") == 1.475
```

- [ ] **Step 2: Run tests to verify preference module and schedule are absent**

Run: `uv run pytest tests/behavior/test_attention_preferences.py -v`

Expected: FAIL importing `PreferenceModel` or `AttentionSchedule`.

- [ ] **Step 3: Implement exact escalation, suppression, and bounded updates**

```python
# additions to backend/src/social_lamp/behavior/policy.py
from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionIntent:
    parameters: dict[str, int]


class AttentionSchedule:
    OFFSETS_NS = (5_000_000_000, 15_000_000_000, 35_000_000_000)

    def __init__(self, *, disengaged_at_ns: int) -> None:
        self._start = disengaged_at_ns
        self._emitted = 0
        self.exhausted = False
        self.suppression_reason: str | None = None

    def suppress(self, reason: str) -> None:
        self.suppression_reason = reason
        self.exhausted = True

    def intent_at(self, mono_ns: int) -> AttentionIntent | None:
        if self.exhausted or self._emitted >= len(self.OFFSETS_NS):
            return None
        due = self._start + self.OFFSETS_NS[self._emitted]
        if mono_ns < due:
            return None
        self._emitted += 1
        level = self._emitted
        if level == len(self.OFFSETS_NS):
            self.exhausted = True
        return AttentionIntent(parameters={"level": level})
```

Directed speech, renewed engagement, speaking mode, operator suppression, and television/media call `suppress` immediately with the corresponding reason. A new schedule cannot be created until 60 seconds after exhaustion.

```python
# backend/src/social_lamp/behavior/preferences.py
from dataclasses import dataclass


@dataclass(frozen=True)
class PreferenceAudit:
    context: str
    behavior: str
    outcome: str
    previous_score: float
    new_score: float


class PreferenceModel:
    DELTAS = {
        "reengaged": 0.10,
        "positive": 0.20,
        "rejected": -0.25,
        "muted": -0.25,
        "no_response": -0.05,
    }

    def __init__(self) -> None:
        self._scores: dict[tuple[str, str], float] = {}
        self.audit: list[PreferenceAudit] = []

    def score(self, context: str, behavior: str) -> float:
        return self._scores.get((context, behavior), 1.0)

    def record(self, context: str, behavior: str, outcome: str) -> None:
        previous = self.score(context, behavior)
        updated = min(1.5, max(0.5, previous + self.DELTAS[outcome]))
        self._scores[(context, behavior)] = updated
        self.audit.append(PreferenceAudit(context, behavior, outcome, previous, updated))

    def start_session(self) -> None:
        for key, score in tuple(self._scores.items()):
            self._scores[key] = 1.0 + (score - 1.0) * 0.95
```

Persist every audit update transactionally with context, behavior, outcome, old/new scores, and correlation ID. Disable preference exploration during replay and evaluation.

- [ ] **Step 4: Verify every suppression rule and persistence restart**

Run:

```powershell
uv run pytest tests/behavior/test_attention_preferences.py tests/behavior/test_policy_compositor.py -v
uv run pytest -q
uv run ruff check backend tests
uv run mypy
```

Expected: all policy, persistence, and full-suite tests pass.

- [ ] **Step 5: Commit bonus behavior adaptation**

```powershell
git add backend/src/social_lamp/behavior backend/src/social_lamp/memory/repository.py tests/behavior
git commit -m "feat: add adaptive attention-seeking behavior"
```

## Phase 5: Evaluation, End-to-End Proof, and Delivery

### Task 18: Build the metric engine and threshold-gated evaluation CLI

**Files:**
- Create: `backend/src/social_lamp/evaluation/metrics.py`
- Create: `backend/src/social_lamp/evaluation/cli.py`
- Create: `tests/evaluation/test_metrics.py`
- Create: `evaluation/manifest.json`

- [ ] **Step 1: Write failing metric and threshold tests**

```python
# tests/evaluation/test_metrics.py
from social_lamp.evaluation.metrics import (
    ClassificationCounts,
    evaluate_gates,
    percentile,
)


def test_classification_metrics_exclude_ambiguous_labels() -> None:
    counts = ClassificationCounts.from_pairs(
        [
            ("engaged", "engaged"),
            ("engaged", "not_engaged"),
            ("not_engaged", "not_engaged"),
            ("ambiguous", "engaged"),
        ]
    )
    assert counts.true_positive == 1
    assert counts.false_negative == 1
    assert counts.true_negative == 1
    assert counts.f1 == 2 / 3


def test_latency_uses_nearest_rank_p95_and_gates_fail_closed() -> None:
    assert percentile(list(range(1, 101)), 0.95) == 95
    result = evaluate_gates(
        engagement_f1=0.84,
        false_transitions_per_two_minutes=0.0,
        frame_to_observation_p95_ms=100,
        state_to_visible_p95_ms=100,
        memory_accuracy=0.95,
        grounding_rate=1.0,
        max_normal_frame_age_ms=200,
    )
    assert not result.passed
    assert result.failures == ("engagement_f1",)
```

- [ ] **Step 2: Run tests to verify evaluation modules are absent**

Run: `uv run pytest tests/evaluation/test_metrics.py -v`

Expected: FAIL importing `social_lamp.evaluation.metrics`.

- [ ] **Step 3: Implement pure metrics and a machine-readable report**

```python
# backend/src/social_lamp/evaluation/metrics.py
from dataclasses import dataclass
from math import ceil


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = max(0, ceil(quantile * len(ordered)) - 1)
    return ordered[index]


@dataclass(frozen=True)
class ClassificationCounts:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @classmethod
    def from_pairs(cls, pairs: list[tuple[str, str]]) -> "ClassificationCounts":
        filtered = [(truth, prediction) for truth, prediction in pairs if truth != "ambiguous"]
        return cls(
            sum(t == "engaged" and p == "engaged" for t, p in filtered),
            sum(t == "not_engaged" and p == "engaged" for t, p in filtered),
            sum(t == "not_engaged" and p == "not_engaged" for t, p in filtered),
            sum(t == "engaged" and p == "not_engaged" for t, p in filtered),
        )

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: tuple[str, ...]


def evaluate_gates(**metrics: float) -> GateResult:
    checks = {
        "engagement_f1": metrics["engagement_f1"] >= 0.85,
        "false_transitions": metrics["false_transitions_per_two_minutes"] <= 1.0,
        "frame_latency": metrics["frame_to_observation_p95_ms"] <= 200,
        "reaction_latency": metrics["state_to_visible_p95_ms"] <= 150,
        "memory_accuracy": metrics["memory_accuracy"] >= 0.90,
        "grounding_rate": metrics["grounding_rate"] == 1.0,
        "frame_freshness": metrics["max_normal_frame_age_ms"] < 300,
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    return GateResult(not failures, failures)
```

Implement `evaluation.cli` to load dataset manifest/checksums, execute sensor or event replay, calculate stratified engagement, transition, association, memory, grounding, and latency metrics, write `output/evaluation/report.json` plus Markdown, and exit nonzero when core gates fail. Include application commit, configuration hash, model IDs, dataset version, and hardware probe output.

- [ ] **Step 4: Verify metrics, reports, and fail/pass exit codes**

Run:

```powershell
uv run pytest tests/evaluation/test_metrics.py -v
uv run python -m social_lamp.evaluation.cli --fixture evaluation/fixtures/core-engagement --output output/evaluation
uv run pytest -q
uv run ruff check backend tests
uv run mypy
```

Expected: metric tests pass, the deterministic fixture writes both reports, and full checks pass. The fixture report may be marked `sample_only`; release gates require the completed labeled suite.

- [ ] **Step 5: Commit measurable evaluation**

```powershell
git add backend/src/social_lamp/evaluation tests/evaluation evaluation/manifest.json
git commit -m "feat: add threshold-gated evaluation reports"
```

### Task 19: Prove the four-step journey and bonuses in Playwright

**Files:**
- Create: `frontend/playwright.config.ts`
- Create: `frontend/e2e/core-journey.spec.ts`
- Create: `frontend/e2e/bonuses.spec.ts`
- Create: `evaluation/fixtures/core-journey/`
- Modify: `package.json`

- [ ] **Step 1: Write the failing core-journey browser test**

```typescript
// frontend/e2e/core-journey.spec.ts
import { expect, test } from "@playwright/test";

test("replay demonstrates engagement, attention seeking, memory, and recall", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Load core journey replay" }).click();
  await expect(page.getByTestId("demo-step-engagement")).toHaveAttribute("data-complete", "true");
  await expect(page.getByText("Seeking attention: level 1")).toBeVisible();
  await expect(page.getByRole("article", { name: /memory: keys/i })).toContainText("right side");
  await page.getByLabel("Ask the lamp").fill("Where are my keys?");
  await page.getByRole("button", { name: "Ask" }).click();
  await expect(page.getByTestId("lamp-answer")).toContainText("right side of the desk");
  await page.getByRole("button", { name: "Show evidence" }).click();
  await expect(page.getByText("observation-core-keys-2")).toBeVisible();
});
```

- [ ] **Step 2: Run Playwright to verify missing fixture controls**

Run: `pnpm e2e -- core-journey.spec.ts`

Expected: FAIL because the replay control or core fixture is absent.

- [ ] **Step 3: Add deterministic fixtures and demo controls**

Create a core-journey trace containing idle, candidate, engaged, disengaged, seeking level 1, stable keys observation on the right side of the desk, a moved/removed object state, a memory query, grounded result, and recall timeline. Add dashboard controls that call the replay endpoint and mark demo steps complete only from correlated backend evidence.

Create bonus fixtures and tests for Person A/B active-speaker association, affect confidence gating, preference score change/reset, speech interruption cancellation latency, and television suppression. Configure Playwright `webServer` entries to start `uv run uvicorn social_lamp.main:app --port 8000` and `pnpm dev --host 127.0.0.1 --port 5173`; base URL is `http://127.0.0.1:5173`.

- [ ] **Step 4: Run all browser, frontend, and backend suites**

Run:

```powershell
uv run pytest -q
pnpm --dir frontend test -- --run
pnpm --dir frontend exec tsc --noEmit
pnpm --dir frontend build
pnpm e2e
```

Expected: backend, frontend, build, four-step journey, and bonus journeys all pass.

- [ ] **Step 5: Commit end-to-end proof**

```powershell
git add frontend/playwright.config.ts frontend/e2e evaluation/fixtures/core-journey frontend/src package.json
git commit -m "test: prove core and bonus demo journeys"
```

### Task 20: Add CI, clean-machine setup, privacy controls, and delivery documentation

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.env.example`
- Create: `README.md`
- Create: `docs/PRIVACY.md`
- Create: `docs/DEMO.md`
- Create: `docs/LIMITATIONS.md`
- Modify: `.gitignore`

- [ ] **Step 1: Write the failing documentation/configuration verification test**

```python
# tests/test_delivery_files.py
from pathlib import Path


def test_delivery_files_cover_required_operations() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    privacy = Path("docs/PRIVACY.md").read_text(encoding="utf-8")
    demo = Path("docs/DEMO.md").read_text(encoding="utf-8")
    limitations = Path("docs/LIMITATIONS.md").read_text(encoding="utf-8")
    assert "uv sync" in readme and "pnpm install" in readme
    assert "offline" in readme.lower() and "replay" in readme.lower()
    assert "seven days" in privacy.lower() and "clear all memory" in privacy.lower()
    assert all(step in demo for step in ("Engagement", "Attention seeking", "Memory formation", "Memory recall"))
    assert "monocular" in limitations.lower() and "session-only" in limitations.lower()
```

- [ ] **Step 2: Run the test to verify delivery files are missing**

Run: `uv run pytest tests/test_delivery_files.py -v`

Expected: FAIL with `FileNotFoundError` for `README.md`.

- [ ] **Step 3: Add exact setup, operation, privacy, and CI gates**

Document Windows prerequisites, `uv sync`, `pnpm install`, optional model download, live run commands, replay/offline commands, cloud environment variables, tests, evaluation, troubleshooting, architecture links, privacy retention/deletion, demo choreography, and explicit limitations.

Use this CI shape:

```yaml
name: CI
on:
  pull_request:
  push:
    branches: [master]
jobs:
  backend:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: "3.12"
      - run: uv sync --locked
      - run: uv run ruff check backend tests
      - run: uv run mypy
      - run: uv run pytest -q
  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "22"
          cache: pnpm
      - run: pnpm install --frozen-lockfile
      - run: pnpm --dir frontend test -- --run
      - run: pnpm --dir frontend exec tsc --noEmit
      - run: pnpm --dir frontend build
```

`.env.example` contains non-secret names and safe defaults. `.gitignore` excludes `.env`, model weights, local databases, snapshots, raw private media, output reports, and build caches while retaining public-safe replay fixtures.

- [ ] **Step 4: Run the complete release verification**

Run:

```powershell
uv sync --locked
pnpm install --frozen-lockfile
uv run ruff check backend tests
uv run ruff format --check backend tests
uv run mypy
uv run pytest -q
pnpm --dir frontend test -- --run
pnpm --dir frontend exec tsc --noEmit
pnpm --dir frontend build
pnpm e2e
uv run python -m social_lamp.evaluation.cli --fixture evaluation/fixtures/core-journey --output output/evaluation
git diff --check
```

Expected: every command exits zero; evaluation reports identify the fixture as deterministic evidence; `git status --short` lists only intended delivery files before commit.

- [ ] **Step 5: Commit the release-ready project**

```powershell
git add .github .env.example .gitignore README.md docs tests/test_delivery_files.py
git commit -m "docs: add reproducible delivery and demo guidance"
```

## Final Acceptance Checklist

- [ ] Run the full release verification from Task 20 on a clean checkout.
- [ ] Run one live webcam/microphone session and export its trace and hardware profile.
- [ ] Run with network disabled and prove replay, simulator, memory, text recall, and reports still work.
- [ ] Confirm every factual recall response has evidence IDs or an explicit uncertainty status.
- [ ] Confirm the dashboard visibly reports camera, microphone, cloud, adapter, and trace health.
- [ ] Confirm raw media, secrets, databases, snapshots, and generated private reports are not tracked.
- [ ] Record the final demo from the exact commit/configuration/dataset versions named in its report.
- [ ] Compare measured gates against `docs/design/09-evaluation-delivery.md`; document any approved exception before release.
