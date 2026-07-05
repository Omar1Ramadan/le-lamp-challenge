from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from social_lamp.conversation.base import ConversationResponse
from social_lamp.conversation.grounding import validate_grounding
from social_lamp.conversation.template import TemplateConversationProvider
from social_lamp.domain.contracts import MemoryQuery, MemoryResult


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


class OpenAIRealtimeProvider:
    def __init__(self, *, client: OpenAIRealtimeClient, model: str) -> None:
        self._client = client
        self._model = model
        self._template = TemplateConversationProvider(lambda _: MemoryResult.not_found())

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
        answer = ""
        for _ in range(2):
            answer = await self._client.answer_text(
                model=self._model,
                turn_id=turn_id,
                text=text,
                evidence_result=capped_evidence,
                tools=memory_tools(),
            )
            if validate_grounding(answer, capped_evidence):
                return ConversationResponse(
                    answer,
                    capped_evidence.evidence_ids,
                    capped_evidence.status,
                )
        del answer
        return await _template_fallback(text, capped_evidence)

    async def interrupt(self, turn_id: str, reason: str) -> None:
        await self._client.cancel(turn_id, reason)

    async def close(self, reason: str) -> None:
        await self._client.close(reason)


def memory_tools() -> tuple[dict[str, object], ...]:
    return (
        {
            "name": "find_last_seen",
            "description": "Read-only lookup for the last reliable object observation.",
        },
        {
            "name": "find_aliases",
            "description": "Read-only lookup of known aliases for an object label.",
        },
        {
            "name": "list_recent_observations",
            "description": "Read-only listing of recent grounded observations.",
        },
    )


async def _template_fallback(text: str, evidence: MemoryResult) -> ConversationResponse:
    async def query(_: MemoryQuery) -> MemoryResult:
        return evidence

    return await TemplateConversationProvider(query).handle_text("fallback", text)


def _cap_evidence(evidence: MemoryResult, *, limit: int) -> MemoryResult:
    if len(evidence.evidence_ids) <= limit:
        return evidence
    return evidence.model_copy(update={"evidence_ids": evidence.evidence_ids[:limit]})
