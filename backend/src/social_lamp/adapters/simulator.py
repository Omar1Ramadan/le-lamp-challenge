from uuid import UUID

from social_lamp.api.hub import ConnectionHub
from social_lamp.domain.contracts import BehaviorTimeline


class SimulatorAdapter:
    def __init__(self, hub: ConnectionHub) -> None:
        self._hub = hub

    async def execute(self, timeline: BehaviorTimeline) -> UUID:
        await self._hub.broadcast(
            {"type": "behavior_timeline", "body": timeline.model_dump(mode="json")}
        )
        return timeline.timeline_id

    @property
    def pose(self) -> dict[str, float]:
        return {}

    async def neutralize(self) -> None:
        await self._hub.broadcast({"type": "neutralize", "body": {}})
