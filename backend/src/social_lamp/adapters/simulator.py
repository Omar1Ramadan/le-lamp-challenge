from uuid import UUID

from social_lamp.api.hub import ConnectionHub
from social_lamp.domain.contracts import BehaviorTimeline, ComponentHealth


class SimulatorAdapter:
    def __init__(self, hub: ConnectionHub) -> None:
        self._hub = hub
        self.health = ComponentHealth(
            component="simulator", status="degraded", detail="no browser client connected"
        )

    async def execute(self, timeline: BehaviorTimeline) -> UUID:
        if self._hub.client_count == 0:
            self.health = ComponentHealth(
                component="simulator", status="degraded", detail="no browser client connected"
            )
            return timeline.timeline_id
        self.health = ComponentHealth(component="simulator", status="ok")
        await self._hub.broadcast(
            {"type": "behavior_timeline", "body": timeline.model_dump(mode="json")}
        )
        return timeline.timeline_id

    @property
    def pose(self) -> dict[str, float]:
        return {}

    async def neutralize(self) -> None:
        await self._hub.broadcast({"type": "neutralize", "body": {}})
