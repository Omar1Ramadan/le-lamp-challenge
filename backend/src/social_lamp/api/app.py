from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from social_lamp.api.hub import ConnectionHub
from social_lamp.runtime.coordinator import RuntimeCoordinator


class ReplayRequest(BaseModel):
    directory: str


class TextRequest(BaseModel):
    text: str


class ToggleRequest(BaseModel):
    enabled: bool


class TraceExportRequest(BaseModel):
    directory: str


LOOPBACK_HOSTS = {"127.0.0.1", "::1", "testclient", "localhost"}


def create_app() -> FastAPI:
    coordinator = RuntimeCoordinator.for_test(database=Path(".runtime/memory.db"))
    hub = ConnectionHub()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await coordinator.start()
        try:
            yield
        finally:
            await coordinator.stop()

    app = FastAPI(title="Simulated Social Lamp", version="0.1.0", lifespan=lifespan)
    app.state.world = coordinator.world
    app.state.hub = hub
    app.state.coordinator = coordinator

    @app.middleware("http")
    async def loopback_only(
        request: Request, call_next: Callable[[Request], Awaitable[Any]]
    ) -> Any:
        client = request.client.host if request.client is not None else "testclient"
        if client not in LOOPBACK_HOSTS:
            raise HTTPException(status_code=403, detail="loopback clients only")
        return await call_next(request)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/api/world")
    async def current_world() -> dict[str, object]:
        return cast(dict[str, object], coordinator.world.snapshot.model_dump(mode="json"))

    @app.post("/api/session/start")
    async def start_session() -> dict[str, object]:
        await coordinator.start()
        return {"ok": True, "running": coordinator.running}

    @app.post("/api/session/stop")
    async def stop_session() -> dict[str, object]:
        await coordinator.stop()
        return {"ok": True, "running": coordinator.running}

    @app.post("/api/replay")
    async def replay(request: ReplayRequest) -> dict[str, object]:
        await coordinator.replay(Path(request.directory))
        return {"ok": True, "revision": coordinator.world.snapshot.revision}

    @app.post("/api/text")
    async def submit_text(request: TextRequest) -> dict[str, object]:
        response = await coordinator.submit_text(request.text)
        return {"ok": True, "response": response.__dict__}

    @app.post("/api/neutralize")
    async def neutralize() -> dict[str, object]:
        await coordinator.neutralize()
        return {"ok": True}

    @app.post("/api/memory/clear")
    async def clear_memory() -> dict[str, object]:
        await coordinator.clear_memory()
        return {"ok": True}

    @app.post("/api/bonuses")
    async def toggle_bonuses(request: ToggleRequest) -> dict[str, object]:
        return {"ok": True, "enabled": coordinator.set_bonuses(request.enabled)}

    @app.post("/api/traces/export")
    async def export_trace(request: TraceExportRequest) -> dict[str, object]:
        return {"ok": True, "trace": coordinator.export_trace(Path(request.directory))}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await hub.connect(websocket)
        await websocket.send_json(
            {"type": "world_snapshot", "body": coordinator.world.snapshot.model_dump(mode="json")}
        )
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            hub.disconnect(websocket)

    return app
