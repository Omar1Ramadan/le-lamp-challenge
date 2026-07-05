from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from social_lamp.domain.contracts import BehaviorTimeline


class AdapterCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)
    motion_channels: tuple[str, ...]
    supports_light: bool
    supports_audio: bool


class LampAdapter(Protocol):
    async def capabilities(self) -> AdapterCapabilities: ...
    async def execute(self, timeline: BehaviorTimeline) -> UUID: ...
    async def cancel(self, execution_id: UUID, reason: str) -> None: ...
    async def neutralize(self, reason: str) -> UUID: ...
