from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from openai import AsyncOpenAI

from social_lamp.config import Settings
from social_lamp.conversation.base import ConversationProvider
from social_lamp.conversation.openai_realtime import OpenAIRealtimeClient, OpenAIRealtimeProvider
from social_lamp.conversation.template import TemplateConversationProvider
from social_lamp.conversation.tools import MemoryQueryPort
from social_lamp.domain.contracts import MemoryQuery, MemoryResult

QueryFunction = Callable[[MemoryQuery], Awaitable[MemoryResult] | MemoryResult]
EventEmitter = Callable[[str, dict[str, Any]], Awaitable[None]]


def build_conversation_provider(
    settings: Settings,
    *,
    openai_client: OpenAIRealtimeClient | None = None,
    query: QueryFunction | None = None,
    memory: MemoryQueryPort | None = None,
    event_emitter: EventEmitter | None = None,
) -> ConversationProvider:
    memory_query = query or _not_found
    provider_name = settings.conversation_provider.strip().lower()
    if provider_name != "openai" or settings.openai_api_key is None:
        return TemplateConversationProvider(
            memory or _QueryFunctionAdapter(memory_query),
            event_emitter=event_emitter,
        )

    client = openai_client or _OpenAIClient(settings.openai_api_key.get_secret_value())
    return OpenAIRealtimeProvider(client=client, model=settings.openai_realtime_model)


def _not_found(_: MemoryQuery) -> MemoryResult:
    return MemoryResult.not_found()


class _QueryFunctionAdapter:
    def __init__(self, query: QueryFunction) -> None:
        self._query = query

    async def find_last_seen(
        self,
        object_label: str,
        *,
        session_scope: str | None = None,
        before_utc: str | None = None,
    ) -> MemoryResult:
        return await self._query(
            MemoryQuery(kind="last_seen", object_label=object_label, limit=1)
        )

    async def find_location(
        self,
        entity_label: str,
        *,
        session_scope: str | None = None,
    ) -> MemoryResult:
        return await self._query(
            MemoryQuery(kind="last_seen", object_label=entity_label, limit=1)
        )

    async def list_recent_observations(
        self,
        *,
        limit: int = 10,
        before_utc: str | None = None,
    ) -> Any:
        del limit, before_utc
        return ()


class _OpenAIClient:
    def __init__(self, api_key: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)

    async def answer_text(
        self,
        *,
        model: str,
        turn_id: str,
        text: str,
        evidence_result: MemoryResult,
        tools: tuple[dict[str, object], ...],
    ) -> str:
        del turn_id, evidence_result, tools
        response = await self._client.responses.create(
            model=model,
            input=text,
            timeout=10.0,
        )
        return response.output_text

    async def cancel(self, turn_id: str, reason: str) -> None:
        del turn_id, reason

    async def close(self, reason: str) -> None:
        del reason
        await self._client.close()
