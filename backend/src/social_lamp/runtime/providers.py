from __future__ import annotations

from collections.abc import Awaitable, Callable

from openai import AsyncOpenAI

from social_lamp.config import Settings
from social_lamp.conversation.base import ConversationProvider
from social_lamp.conversation.openai_realtime import OpenAIRealtimeClient, OpenAIRealtimeProvider
from social_lamp.conversation.template import TemplateConversationProvider
from social_lamp.domain.contracts import MemoryQuery, MemoryResult

QueryFunction = Callable[[MemoryQuery], Awaitable[MemoryResult] | MemoryResult]


def build_conversation_provider(
    settings: Settings,
    *,
    openai_client: OpenAIRealtimeClient | None = None,
    query: QueryFunction | None = None,
) -> ConversationProvider:
    memory_query = query or _not_found
    provider_name = settings.conversation_provider.strip().lower()
    if provider_name != "openai" or settings.openai_api_key is None:
        return TemplateConversationProvider(memory_query)

    client = openai_client or _OpenAIClient(settings.openai_api_key.get_secret_value())
    return OpenAIRealtimeProvider(client=client, model=settings.openai_realtime_model)


def _not_found(_: MemoryQuery) -> MemoryResult:
    return MemoryResult.not_found()


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
