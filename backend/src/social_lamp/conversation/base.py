from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ConversationResponse:
    text: str
    evidence_ids: tuple[str, ...]
    status: str


class ConversationProvider(Protocol):
    async def handle_text(self, turn_id: str, text: str) -> ConversationResponse: ...

    async def handle_audio(
        self, turn_id: str, chunks: AsyncIterator[bytes]
    ) -> ConversationResponse: ...

    async def interrupt(self, turn_id: str, reason: str) -> None: ...

    async def close(self, reason: str) -> None: ...
