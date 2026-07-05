from pathlib import Path

import pytest
from social_lamp.runtime.coordinator import RuntimeCoordinator


@pytest.mark.asyncio
async def test_engagement_replay_reaches_simulator_adapter(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    await coordinator.start()
    try:
        await coordinator.replay(Path("evaluation/fixtures/core-engagement"))
        assert coordinator.world.snapshot.social_state.value == "engaged"
        assert coordinator.simulator.executed[-1].duration_ms == 700
        assert coordinator.metrics.counter("social_transition", state="engaged") == 1
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_stop_neutralizes_adapter_and_closes_memory(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    await coordinator.start()
    await coordinator.stop()
    assert coordinator.simulator.neutralized
    assert coordinator.memory.closed


@pytest.mark.asyncio
async def test_text_submission_acknowledges_command(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
    await coordinator.start()
    try:
        response = await coordinator.submit_text("Where are my keys?")
        assert response.status in {"not_found", "unsupported"}
    finally:
        await coordinator.stop()
