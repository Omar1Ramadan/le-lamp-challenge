from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from uuid6 import uuid7

from social_lamp.api.hub import ConnectionHub
from social_lamp.domain.clock import SystemClock
from social_lamp.world.model import WorldModel


def create_app() -> FastAPI:
    clock = SystemClock()
    world = WorldModel(session_id=uuid7(), clock=clock)
    hub = ConnectionHub()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    app = FastAPI(title="Simulated Social Lamp", version="0.1.0", lifespan=lifespan)
    app.state.world = world
    app.state.hub = hub

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/api/world")
    async def current_world() -> dict[str, object]:
        return world.snapshot.model_dump(mode="json")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await hub.connect(websocket)
        await websocket.send_json(
            {"type": "world_snapshot", "body": world.snapshot.model_dump(mode="json")}
        )
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            hub.disconnect(websocket)

    return app
