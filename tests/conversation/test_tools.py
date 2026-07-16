import pytest
from social_lamp.conversation.tools import (
    execute_find_last_seen,
    execute_find_location,
    execute_list_recent_observations,
    get_tool_definitions,
)
from social_lamp.domain.contracts import MemoryResult, ObservationSummary


class _FakeMemory:
    def __init__(self) -> None:
        self._records: dict[str, MemoryResult] = {}
        self._observations: tuple[ObservationSummary, ...] = ()

    def with_result(self, label: str, result: MemoryResult) -> "_FakeMemory":
        self._records[label.lower()] = result
        return self

    def with_observations(self, observations: tuple[ObservationSummary, ...]) -> "_FakeMemory":
        self._observations = observations
        return self

    async def find_last_seen(
        self,
        object_label: str,
        *,
        session_scope: str | None = None,
        before_utc: str | None = None,
    ) -> MemoryResult:
        del session_scope, before_utc
        return self._records.get(object_label.lower(), MemoryResult.not_found())

    async def find_location(
        self,
        entity_label: str,
        *,
        session_scope: str | None = None,
    ) -> MemoryResult:
        del session_scope
        return self._records.get(entity_label.lower(), MemoryResult.not_found())

    async def list_recent_observations(
        self,
        *,
        limit: int = 10,
        before_utc: str | None = None,
    ) -> tuple[ObservationSummary, ...]:
        del before_utc
        return self._observations[:limit]


@pytest.mark.asyncio
async def test_find_last_seen_returns_newest_matching() -> None:
    memory = _FakeMemory().with_result(
        "cup",
        MemoryResult(
            status="found",
            canonical_label="cup",
            horizontal_region="right",
            anchor_name="desk",
            observed_at_utc="2026-07-04T12:00:00Z",
            evidence_ids=("mem-1",),
        ),
    )
    result, memory_result = await execute_find_last_seen(memory, object_label="cup")
    assert result.status == "success"
    assert result.output["found"] is True
    assert result.output["object_label"] == "cup"
    assert result.evidence_ids == ("mem-1",)
    assert memory_result.status == "found"


@pytest.mark.asyncio
async def test_find_last_seen_returns_not_found() -> None:
    memory = _FakeMemory()
    result, _ = await execute_find_last_seen(memory, object_label="wallet")
    assert result.status == "not_found"
    assert result.output["found"] is False


@pytest.mark.asyncio
async def test_find_last_seen_returns_ambiguous() -> None:
    memory = _FakeMemory().with_result(
        "key",
        MemoryResult(status="ambiguous", alternatives=("keys", "keyboard")),
    )
    result, _ = await execute_find_last_seen(memory, object_label="key")
    assert result.status == "ambiguous"
    assert result.output["alternatives"] == ["keys", "keyboard"]


@pytest.mark.asyncio
async def test_find_location_returns_location_with_evidence() -> None:
    memory = _FakeMemory().with_result(
        "cup",
        MemoryResult(
            status="found",
            canonical_label="cup",
            horizontal_region="center",
            anchor_name="desk",
            evidence_ids=("mem-1",),
        ),
    )
    result, _ = await execute_find_location(memory, entity_label="cup")
    assert result.status == "success"
    assert result.output["found"] is True
    assert result.output["horizontal_region"] == "center"
    assert result.output["anchor_name"] == "desk"
    assert result.evidence_ids == ("mem-1",)


@pytest.mark.asyncio
async def test_find_location_not_found() -> None:
    memory = _FakeMemory()
    result, _ = await execute_find_location(memory, entity_label="wallet")
    assert result.status == "not_found"
    assert result.output["found"] is False


@pytest.mark.asyncio
async def test_list_recent_observations_respects_limit() -> None:
    observations = tuple(
        ObservationSummary(
            id=f"obs-{i}",
            type="object_seen",
            summary=f"object {i}",
            observed_at_utc=f"2026-07-04T12:0{i}:00Z",
            evidence_ids=(f"obs-{i}",),
        )
        for i in range(5)
    )
    memory = _FakeMemory().with_observations(observations)
    result, actual = await execute_list_recent_observations(memory, limit=3)
    assert result.status == "success"
    assert len(actual) == 3


@pytest.mark.asyncio
async def test_list_recent_observations_empty() -> None:
    memory = _FakeMemory()
    result, actual = await execute_list_recent_observations(memory, limit=10)
    assert result.status == "success"
    assert len(actual) == 0
    assert result.output["total"] == 0


def test_tool_definitions_are_returned() -> None:
    defs = get_tool_definitions()
    assert len(defs) == 3
    names = {d["name"] for d in defs}
    assert names == {"find_last_seen", "find_location", "list_recent_observations"}
    for d in defs:
        assert "description" in d
        assert "input_schema" in d
