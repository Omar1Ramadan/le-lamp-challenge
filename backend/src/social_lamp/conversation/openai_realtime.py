from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from social_lamp.conversation.base import ConversationResponse
from social_lamp.conversation.grounding import validate_grounding_detailed
from social_lamp.conversation.template import TemplateConversationProvider
from social_lamp.conversation.tools import (
    MemoryQueryPort,
    execute_list_recent_observations,
    get_tool_definitions,
)
from social_lamp.domain.contracts import (
    MemoryResult,
    ObservationSummary,
    ToolCallRecord,
)


class OpenAIRealtimeClient(Protocol):
    async def answer_text(
        self,
        *,
        model: str,
        turn_id: str,
        text: str,
        evidence_result: MemoryResult,
        tools: tuple[dict[str, object], ...],
    ) -> str: ...

    async def cancel(self, turn_id: str, reason: str) -> None: ...

    async def close(self, reason: str) -> None: ...


class _EvidenceMemory:
    def __init__(
        self, evidence: MemoryResult, observations: tuple[ObservationSummary, ...]
    ) -> None:
        self._evidence = evidence
        self._observations = observations

    async def find_last_seen(
        self,
        object_label: str,
        *,
        session_scope: str | None = None,
        before_utc: str | None = None,
    ) -> MemoryResult:
        if (
            self._evidence.status == "found"
            and self._evidence.canonical_label
            and object_label.lower() in self._evidence.canonical_label.lower()
        ):
            return self._evidence
        return MemoryResult.not_found()

    async def find_location(
        self,
        entity_label: str,
        *,
        session_scope: str | None = None,
    ) -> MemoryResult:
        return await self.find_last_seen(entity_label, session_scope=session_scope)

    async def list_recent_observations(
        self,
        *,
        limit: int = 10,
        before_utc: str | None = None,
    ) -> tuple[ObservationSummary, ...]:
        return self._observations[:limit]


class OpenAIRealtimeProvider:
    def __init__(
        self,
        *,
        client: OpenAIRealtimeClient,
        model: str,
        memory: MemoryQueryPort | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._memory = memory

    async def handle_text(self, turn_id: str, text: str) -> ConversationResponse:
        return await self.handle_grounded_turn(
            turn_id=turn_id,
            text=text,
            evidence_result=MemoryResult.not_found(),
        )

    async def handle_audio(
        self, turn_id: str, chunks: AsyncIterator[bytes]
    ) -> ConversationResponse:
        async for _ in chunks:
            break
        return await self.handle_text(turn_id, "")

    async def handle_grounded_turn(
        self,
        *,
        turn_id: str,
        text: str,
        evidence_result: MemoryResult,
    ) -> ConversationResponse:
        capped_evidence = _cap_evidence(evidence_result, limit=10)
        observations: tuple[ObservationSummary, ...] = ()
        if self._memory is not None:
            _, observations = await execute_list_recent_observations(
                self._memory, limit=5
            )

        evidence_memory = _EvidenceMemory(capped_evidence, observations)
        answer = ""
        for _ in range(2):
            answer = await self._client.answer_text(
                model=self._model,
                turn_id=turn_id,
                text=text,
                evidence_result=capped_evidence,
                tools=memory_tools(),
            )
            validation = validate_grounding_detailed(
                answer, capped_evidence, observations
            )
            if validation.valid:
                all_ids = _dedup_ids(capped_evidence.evidence_ids, validation.evidence_ids)
                return ConversationResponse(
                    answer,
                    all_ids,
                    capped_evidence.status,
                    grounded=True,
                    source="openai",
                    tool_calls=(
                        ToolCallRecord(
                            name="answer_text",
                            status="success",
                            evidence_ids=all_ids,
                        ),
                    ),
                )
        del answer
        return await _template_fallback(text, capped_evidence, observations, evidence_memory)

    async def interrupt(self, turn_id: str, reason: str) -> None:
        await self._client.cancel(turn_id, reason)

    async def close(self, reason: str) -> None:
        await self._client.close(reason)


def memory_tools() -> tuple[dict[str, object], ...]:
    return get_tool_definitions()


async def _template_fallback(
    text: str,
    evidence: MemoryResult,
    observations: tuple[ObservationSummary, ...],
    evidence_memory: _EvidenceMemory,
) -> ConversationResponse:
    provider = TemplateConversationProvider(evidence_memory)
    return await provider.handle_text("fallback", text)


def _cap_evidence(evidence: MemoryResult, *, limit: int) -> MemoryResult:
    if len(evidence.evidence_ids) <= limit:
        return evidence
    return evidence.model_copy(update={"evidence_ids": evidence.evidence_ids[:limit]})


def _dedup_ids(*id_tuples: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for ids in id_tuples:
        for eid in ids:
            if eid not in seen:
                seen.add(eid)
                result.append(eid)
    return tuple(result)
