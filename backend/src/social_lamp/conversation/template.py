from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from social_lamp.conversation.base import ConversationResponse
from social_lamp.conversation.tools import (
    MemoryQueryPort,
    execute_find_last_seen,
    execute_list_recent_observations,
)
from social_lamp.domain.contracts import ObservationSummary, ToolCallRecord

EventEmitter = Callable[[str, dict[str, Any]], Awaitable[None]]


class TemplateConversationProvider:
    def __init__(
        self,
        memory: MemoryQueryPort,
        event_emitter: EventEmitter | None = None,
    ) -> None:
        self._memory = memory
        self._event_emitter = event_emitter

    async def handle_text(self, turn_id: str, text: str) -> ConversationResponse:
        await self._emit("think", {"turn_id": turn_id, "query": text.strip()})

        stripped = text.strip().lower()

        if re.search(r"(?:what|anything).*(?:seen|observed|notice)", stripped, re.I):
            response = await self._handle_recent_observations(turn_id)
        else:
            response = await self._handle_last_seen(turn_id, stripped)

        if response.grounded:
            await self._emit(
                "recall_success",
                {
                    "turn_id": turn_id,
                    "evidence_ids": list(response.evidence_ids),
                    "source": response.source,
                    "tool_names": [t.name for t in response.tool_calls],
                },
            )
        elif response.status in ("not_found", "unsupported", "ambiguous"):
            await self._emit(
                "recall_unknown",
                {
                    "turn_id": turn_id,
                    "status": response.status,
                    "source": response.source,
                    "tool_names": [t.name for t in response.tool_calls],
                },
            )

        return response

    async def _handle_last_seen(
        self, turn_id: str, text: str
    ) -> ConversationResponse:
        match = re.search(
            r"(?:where|when).*?(?:is|are|my|the|a)\s+([a-zA-Z][a-zA-Z -]{0,40})[?.!]*$",
            text,
            re.I,
        )
        if match is None:
            return ConversationResponse(
                text="I can answer where I last saw an object.",
                evidence_ids=(),
                status="unsupported",
                grounded=False,
                source="template",
                tool_calls=(ToolCallRecord(name="parse", status="unsupported"),),
            )

        label = match.group(1).strip().lower()
        tr, memory = await execute_find_last_seen(self._memory, object_label=label)

        if tr.status == "not_found":
            text = f"I do not have reliable evidence for the {label}."
            return ConversationResponse(
                text=text,
                evidence_ids=(),
                status="not_found",
                grounded=False,
                source="template",
                tool_calls=(ToolCallRecord(name=tr.name, status="not_found"),),
            )

        if tr.status == "ambiguous":
            choices = ", ".join(str(a) for a in tr.output.get("alternatives", []))
            text = f"I found more than one possible match: {choices}."
            return ConversationResponse(
                text=text,
                evidence_ids=tr.evidence_ids,
                status="ambiguous",
                grounded=False,
                source="template",
                tool_calls=(
                    ToolCallRecord(name=tr.name, status="ambiguous", evidence_ids=tr.evidence_ids),
                ),
            )

        parts = []
        output = tr.output
        if output.get("horizontal_region"):
            parts.append(f"on the {output['horizontal_region']} side")
        if output.get("anchor_name"):
            parts.append(f"of the {output['anchor_name']}")
        location = " ".join(parts)
        obj_label = output.get("object_label", "object")
        if location:
            text = f"I last saw the {obj_label} {location}."
        else:
            text = f"I last saw the {obj_label}."

        return ConversationResponse(
            text=text,
            evidence_ids=tr.evidence_ids,
            status="found",
            grounded=True,
            source="deterministic_template",
            tool_calls=(
                ToolCallRecord(
                    name=tr.name,
                    status="success",
                    evidence_ids=tr.evidence_ids,
                ),
            ),
        )

    async def _handle_recent_observations(
        self, turn_id: str
    ) -> ConversationResponse:
        tr, observations = await execute_list_recent_observations(
            self._memory, limit=10
        )

        if not observations:
            return ConversationResponse(
                text="I have not observed anything recently.",
                evidence_ids=(),
                status="not_found",
                grounded=False,
                source="template",
                tool_calls=(
                    ToolCallRecord(name=tr.name, status="success", evidence_ids=tr.evidence_ids),
                ),
            )

        labels = _summarize_observations(observations)
        evidence_ids = tr.evidence_ids
        text = f"Recently, I saw {labels}."

        return ConversationResponse(
            text=text,
            evidence_ids=evidence_ids,
            status="found",
            grounded=True,
            source="deterministic_template",
            tool_calls=(
                ToolCallRecord(
                    name=tr.name,
                    status="success",
                    evidence_ids=evidence_ids,
                ),
            ),
        )

    async def handle_audio(self, turn_id: str, chunks: object) -> ConversationResponse:
        del turn_id, chunks
        return ConversationResponse(
            text="I can answer text questions for now.",
            evidence_ids=(),
            status="unsupported",
        )

    async def interrupt(self, turn_id: str, reason: str) -> None:
        del turn_id, reason

    async def close(self, reason: str) -> None:
        del reason

    async def _emit(self, event: str, data: dict[str, Any]) -> None:
        if self._event_emitter is not None:
            await self._event_emitter(event, data)


def _summarize_observations(observations: tuple[ObservationSummary, ...]) -> str:
    labels: list[str] = []
    for obs in observations:
        label = obs.summary.split()[0] if obs.summary.split() else "something"
        loc = " ".join(obs.summary.split()[1:]) if len(obs.summary.split()) > 1 else ""
        labels.append(f"a {label} {loc}".strip() if loc else f"a {label}")
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"
