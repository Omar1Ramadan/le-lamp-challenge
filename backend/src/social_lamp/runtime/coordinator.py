from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

from uuid6 import uuid7

from social_lamp.behavior.compositor import BehaviorCompositor
from social_lamp.behavior.policy import BehaviorPolicy
from social_lamp.conversation.base import ConversationProvider, ConversationResponse
from social_lamp.conversation.template import TemplateConversationProvider
from social_lamp.domain.contracts import (
    BehaviorTimeline,
    MemoryQuery,
    MemoryResult,
    SocialState,
    WorldSnapshot,
)
from social_lamp.replay.trace import TraceReader


class SimulatorPort(Protocol):
    pose: dict[str, float]

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
        self.bonuses_enabled = False

    @classmethod
    def for_test(cls, *, database: Path) -> RuntimeCoordinator:
        from social_lamp.runtime.testing import build_test_runtime

        return build_test_runtime(database)

    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
        await self.simulator.neutralize()
        await self.conversation.close("runtime stopping")
        await self.memory.close()

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
