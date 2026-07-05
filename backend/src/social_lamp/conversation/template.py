import inspect
import re
from collections.abc import Awaitable, Callable

from social_lamp.conversation.base import ConversationResponse
from social_lamp.domain.contracts import MemoryQuery, MemoryResult

QueryFunction = Callable[[MemoryQuery], Awaitable[MemoryResult] | MemoryResult]


class TemplateConversationProvider:
    def __init__(self, query: QueryFunction) -> None:
        self._query = query

    async def handle_text(self, turn_id: str, text: str) -> ConversationResponse:
        del turn_id
        match = re.search(
            r"(?:where|when).*?(?:my|the|a)\s+([a-zA-Z][a-zA-Z -]{0,40})[?.!]*$",
            text.strip(),
            re.I,
        )
        if match is None:
            return ConversationResponse(
                "I can answer where I last saw an object.", (), "unsupported"
            )

        label = match.group(1).strip().lower()
        query = MemoryQuery(kind="last_seen", object_label=label, limit=1)
        result_or_awaitable = self._query(query)
        if inspect.isawaitable(result_or_awaitable):
            result = await result_or_awaitable
        else:
            result = result_or_awaitable

        if result.status == "not_found":
            return ConversationResponse(
                f"I do not have reliable evidence for the {label}.", (), "not_found"
            )
        if result.status == "ambiguous":
            choices = ", ".join(result.alternatives)
            return ConversationResponse(
                f"I found more than one possible match: {choices}.", (), "ambiguous"
            )

        location = " ".join(
            part
            for part in (
                f"on the {result.horizontal_region} side"
                if result.horizontal_region
                else "",
                f"of the {result.anchor_name}" if result.anchor_name else "",
            )
            if part
        )
        return ConversationResponse(
            f"I last saw the {result.canonical_label} {location}.",
            result.evidence_ids,
            "found",
        )

    async def handle_audio(self, turn_id: str, chunks: object) -> ConversationResponse:
        del turn_id, chunks
        return ConversationResponse("I can answer text questions for now.", (), "unsupported")

    async def interrupt(self, turn_id: str, reason: str) -> None:
        del turn_id, reason

    async def close(self, reason: str) -> None:
        del reason
