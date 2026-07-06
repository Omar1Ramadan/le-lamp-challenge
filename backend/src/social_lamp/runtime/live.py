from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from uuid6 import uuid7

from social_lamp.adapters.simulator import SimulatorAdapter
from social_lamp.api.hub import ConnectionHub
from social_lamp.config import Settings
from social_lamp.conversation.base import ConversationProvider
from social_lamp.domain.clock import SystemClock
from social_lamp.memory.repository import MemoryRepository
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
    simulator = SimulatorAdapter(hub or ConnectionHub())
    metrics = RuntimeMetrics()

    coordinator = RuntimeCoordinator(
        world=world,
        simulator=simulator,
        metrics=metrics,
        memory=memory,
        conversation=conversation,
    )
    if conversation is None:
        coordinator.conversation = build_conversation_provider(
            resolved_settings,
            query=coordinator._query_memory,
        )
    return coordinator
