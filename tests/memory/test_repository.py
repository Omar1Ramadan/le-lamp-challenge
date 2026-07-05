from pathlib import Path

import pytest
from social_lamp.memory.repository import MemoryRepository, ObservationWrite


@pytest.mark.asyncio
async def test_last_seen_returns_newest_grounded_observation(tmp_path: Path) -> None:
    repository = await MemoryRepository.open(tmp_path / "memory.db")
    await repository.record(ObservationWrite.example("keys", "left", observed_at_mono_ns=10))
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


@pytest.mark.asyncio
async def test_alias_lookup_precedes_canonical_lookup(tmp_path: Path) -> None:
    repository = await MemoryRepository.open(tmp_path / "memory.db")
    obs_id = await repository.record(
        ObservationWrite.example("remote control", "center", aliases=("clicker",))
    )
    result = await repository.find_last_seen("clicker")
    assert result.status == "found"
    assert result.canonical_label == "remote control"
    assert result.evidence_ids == (obs_id,)
    await repository.close()


@pytest.mark.asyncio
async def test_session_scope_and_before_utc_filter_results(tmp_path: Path) -> None:
    repository = await MemoryRepository.open(tmp_path / "memory.db")
    await repository.record(
        ObservationWrite.example("keys", "left", session_id="session-a", observed_at_mono_ns=10)
    )
    scoped = await repository.record(
        ObservationWrite.example(
            "keys",
            "right",
            session_id="session-b",
            observed_at_mono_ns=20,
            observed_at_utc="2026-07-04T12:00:20Z",
        )
    )
    result = await repository.find_last_seen("keys", session_scope="session-b")
    assert result.evidence_ids == (scoped,)
    before = await repository.find_last_seen("keys", before_utc="2026-07-04T12:00:15Z")
    assert before.horizontal_region == "left"
    await repository.close()


@pytest.mark.asyncio
async def test_record_rolls_back_when_last_known_update_fails(tmp_path: Path) -> None:
    repository = await MemoryRepository.open(tmp_path / "memory.db")
    repository.fail_after_observation_insert = True
    with pytest.raises(RuntimeError, match="injected failure"):
        await repository.record(ObservationWrite.example("keys", "left"))
    assert await repository.count_observations() == 0
    assert (await repository.find_last_seen("keys")).status == "not_found"
    await repository.close()
