from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from uuid6 import uuid7

from social_lamp.domain.clock import FakeClock
from social_lamp.domain.contracts import (
    BehaviorTimeline,
    MemoryResult,
    ObservationSummary,
    WorldSnapshot,
)
from social_lamp.memory.repository import ObservationWrite
from social_lamp.runtime.coordinator import RuntimeCoordinator
from social_lamp.world.model import WorldModel


class MutableWorldModel(WorldModel):
    def replace(self, snapshot: WorldSnapshot) -> None:
        self._snapshot = snapshot


@dataclass
class FakeSimulator:
    pose: dict[str, float] = field(default_factory=dict)
    executed: list[BehaviorTimeline] = field(default_factory=list)
    neutralized: bool = False

    async def execute(self, timeline: BehaviorTimeline) -> UUID:
        self.executed.append(timeline)
        return timeline.timeline_id

    def handle_ack(self, body: dict[str, object]) -> None:
        pass

    async def neutralize(self) -> None:
        self.neutralized = True


@dataclass
class FakeMetrics:
    _counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = field(default_factory=dict)

    def increment(self, name: str, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        self._counters[key] = self._counters.get(key, 0) + 1

    def counter(self, name: str, **labels: str) -> int:
        return self._counters.get((name, tuple(sorted(labels.items()))), 0)


class TestMemory:
    def __init__(self, database: Path) -> None:
        self.database = database
        self.closed = False
        self.cleared = False

    async def close(self) -> None:
        self.closed = True

    async def clear(self) -> None:
        self.cleared = True

    async def record(self, observation: ObservationWrite) -> str:
        return observation.observation_id

    async def find_last_seen(
        self,
        object_label: str,
        *,
        session_scope: str | None = None,
        before_utc: str | None = None,
    ) -> MemoryResult:
        del object_label, session_scope, before_utc
        return MemoryResult.not_found()

    async def find_location(
        self,
        entity_label: str,
        *,
        session_scope: str | None = None,
    ) -> MemoryResult:
        del entity_label, session_scope
        return MemoryResult.not_found()

    async def list_recent_observations(
        self,
        *,
        limit: int = 10,
        before_utc: str | None = None,
    ) -> tuple[ObservationSummary, ...]:
        del limit, before_utc
        return ()


def build_test_runtime(database: Path) -> RuntimeCoordinator:
    clock = FakeClock(mono_ns=0, wall_time_utc="2026-07-04T12:00:00Z")
    world = MutableWorldModel(session_id=uuid7(), clock=clock)
    return RuntimeCoordinator(
        world=world,
        simulator=FakeSimulator(),
        metrics=FakeMetrics(),
        memory=TestMemory(database),
    )
