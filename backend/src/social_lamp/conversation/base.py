from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

from social_lamp.domain.contracts import ToolCallRecord


@dataclass(frozen=True)
class ConversationResponse:
    text: str
    evidence_ids: tuple[str, ...]
    status: str
    grounded: bool = False
    source: str = "template"
    tool_calls: tuple[ToolCallRecord, ...] = field(default_factory=tuple)


class ConversationProvider(Protocol):
    async def handle_text(self, turn_id: str, text: str) -> ConversationResponse: ...

    async def handle_audio(
        self, turn_id: str, chunks: AsyncIterator[bytes]
    ) -> ConversationResponse: ...

    async def interrupt(self, turn_id: str, reason: str) -> None: ...

    async def close(self, reason: str) -> None: ...
