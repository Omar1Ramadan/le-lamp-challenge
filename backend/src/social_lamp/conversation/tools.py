from __future__ import annotations

from typing import Any, Protocol

from social_lamp.domain.contracts import MemoryQuery, MemoryResult, ObservationSummary


class MemoryQueryPort(Protocol):
    async def find_last_seen(
        self,
        object_label: str,
        *,
        session_scope: str | None = None,
        before_utc: str | None = None,
    ) -> MemoryResult: ...

    async def find_location(
        self,
        entity_label: str,
        *,
        session_scope: str | None = None,
    ) -> MemoryResult: ...

    async def list_recent_observations(
        self,
        *,
        limit: int = 10,
        before_utc: str | None = None,
    ) -> tuple[ObservationSummary, ...]: ...


_TOOL_DEFINITIONS: tuple[dict[str, object], ...] = (
    {
        "name": "find_last_seen",
        "description": "Look up the last reliable observation for an object label.",
        "input_schema": {
            "type": "object",
            "properties": {
                "object_label": {
                    "type": "string",
                    "description": "The object label to look up (e.g. 'cup', 'keys').",
                }
            },
            "required": ["object_label"],
        },
    },
    {
        "name": "find_location",
        "description": "Look up the current or most recent known location for an entity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_label": {
                    "type": "string",
                    "description": "The entity label to find the location of.",
                }
            },
            "required": ["entity_label"],
        },
    },
    {
        "name": "list_recent_observations",
        "description": "Retrieve recent observation summaries for grounding context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum observations to return (default 10).",
                }
            },
        },
    },
)


def get_tool_definitions() -> tuple[dict[str, object], ...]:
    return _TOOL_DEFINITIONS


class ToolCallResult:
    def __init__(
        self,
        name: str,
        status: str = "success",
        evidence_ids: tuple[str, ...] = (),
        detail: str | None = None,
        output: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.status = status
        self.evidence_ids = evidence_ids
        self.detail = detail
        self.output = output if output is not None else {}


async def execute_find_last_seen(
    memory: MemoryQueryPort,
    *,
    object_label: str,
    session_scope: str | None = None,
    before_utc: str | None = None,
) -> tuple[ToolCallResult, MemoryResult]:
    result = await memory.find_last_seen(
        object_label,
        session_scope=session_scope,
        before_utc=before_utc,
    )
    if result.status == "not_found":
        return ToolCallResult(
            name="find_last_seen",
            status="not_found",
            output={"found": False, "object_label": object_label},
        ), result
    if result.status == "ambiguous":
        return ToolCallResult(
            name="find_last_seen",
            status="ambiguous",
            evidence_ids=result.evidence_ids,
            output={
                "found": False,
                "object_label": object_label,
                "alternatives": list(result.alternatives),
            },
        ), result
    return ToolCallResult(
        name="find_last_seen",
        status="success",
        evidence_ids=result.evidence_ids,
        output={
            "found": True,
            "object_label": result.canonical_label,
            "horizontal_region": result.horizontal_region,
            "depth_band": result.depth_band,
            "anchor_name": result.anchor_name,
            "observed_at_utc": result.observed_at_utc,
            "evidence_ids": list(result.evidence_ids),
        },
    ), result


async def execute_find_location(
    memory: MemoryQueryPort,
    *,
    entity_label: str,
    session_scope: str | None = None,
) -> tuple[ToolCallResult, MemoryResult]:
    result = await memory.find_location(
        entity_label,
        session_scope=session_scope,
    )
    if result.status == "not_found":
        return ToolCallResult(
            name="find_location",
            status="not_found",
            output={"found": False, "entity_label": entity_label},
        ), result
    return ToolCallResult(
        name="find_location",
        status="success",
        evidence_ids=result.evidence_ids,
        output={
            "found": True,
            "entity_label": result.canonical_label,
            "horizontal_region": result.horizontal_region,
            "depth_band": result.depth_band,
            "anchor_name": result.anchor_name,
            "confidence": _location_confidence(result),
            "evidence_ids": list(result.evidence_ids),
        },
    ), result


async def execute_list_recent_observations(
    memory: MemoryQueryPort,
    *,
    limit: int = 10,
    before_utc: str | None = None,
) -> tuple[ToolCallResult, tuple[ObservationSummary, ...]]:
    observations = await memory.list_recent_observations(
        limit=limit,
        before_utc=before_utc,
    )
    all_evidence: list[str] = []
    obs_output: list[dict[str, object]] = []
    for obs in observations:
        all_evidence.extend(obs.evidence_ids)
        obs_output.append(
            {
                "id": obs.id,
                "type": obs.type,
                "summary": obs.summary,
                "observed_at_utc": obs.observed_at_utc,
                "evidence_ids": list(obs.evidence_ids),
            }
        )
    return ToolCallResult(
        name="list_recent_observations",
        status="success",
        evidence_ids=tuple(all_evidence),
        output={"observations": obs_output, "total": len(observations)},
    ), observations


def _location_confidence(result: MemoryResult) -> float:
    if result.horizontal_region or result.depth_band or result.anchor_name:
        return 0.8
    return 0.0


def query_to_tool_results(
    memory: MemoryQueryPort,
    query: MemoryQuery,
) -> tuple[MemoryResult, ToolCallResult]:
    raise NotImplementedError("use execute_ functions for async tool dispatch")
