from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from social_lamp.api.hub import ConnectionHub
from social_lamp.runtime.coordinator import RuntimeCoordinator
from social_lamp.runtime.live import build_live_runtime


class ReplayRequest(BaseModel):
    directory: str


class TextRequest(BaseModel):
    text: str


class ToggleRequest(BaseModel):
    enabled: bool


class TraceExportRequest(BaseModel):
    directory: str


LOOPBACK_HOSTS = {"127.0.0.1", "::1", "testclient", "localhost"}


def create_app(*, database_path: Path | None = None) -> FastAPI:
    hub = ConnectionHub()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        coordinator = await build_live_runtime(database_path=database_path, hub=hub)
        app.state.world = coordinator.world
        app.state.coordinator = coordinator
        await coordinator.start()
        try:
            yield
        finally:
            await coordinator.stop()

    app = FastAPI(title="Simulated Social Lamp", version="0.1.0", lifespan=lifespan)
    app.state.hub = hub

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
    async def current_world(request: Request) -> dict[str, object]:
        coordinator = _coordinator(request.app)
        return cast(dict[str, object], coordinator.world.snapshot.model_dump(mode="json"))

    @app.post("/api/session/start")
    async def start_session(request: Request) -> dict[str, object]:
        coordinator = _coordinator(request.app)
        await coordinator.start()
        return {"ok": True, "running": coordinator.running}

    @app.post("/api/session/stop")
    async def stop_session(request: Request) -> dict[str, object]:
        coordinator = _coordinator(request.app)
        await coordinator.stop()
        return {"ok": True, "running": coordinator.running}

    @app.post("/api/replay")
    async def replay(request: Request, body: ReplayRequest) -> dict[str, object]:
        coordinator = _coordinator(request.app)
        await coordinator.replay(Path(body.directory))
        await hub.broadcast(
            {"type": "world_snapshot", "body": coordinator.world.snapshot.model_dump(mode="json")}
        )
        await hub.broadcast(
            {
                "type": "metric",
                "body": {"name": "social_transition", "labels": {"state": "engaged"}},
            }
        )
        return {"ok": True, "revision": coordinator.world.snapshot.revision}

    @app.get("/api/simulator/timelines/{timeline_id}")
    async def simulator_timeline(timeline_id: str) -> dict[str, object]:
        timeline = timeline_id
        return {
            "timeline_id": timeline,
            "acknowledgements": hub.timeline_acknowledgements(timeline),
        }

    @app.post("/api/text")
    async def submit_text(request: Request, body: TextRequest) -> dict[str, object]:
        coordinator = _coordinator(request.app)
        response = await coordinator.submit_text(body.text)
        return {"ok": True, "response": response.__dict__}

    @app.post("/api/neutralize")
    async def neutralize(request: Request) -> dict[str, object]:
        coordinator = _coordinator(request.app)
        await coordinator.neutralize()
        return {"ok": True}

    @app.post("/api/memory/clear")
    async def clear_memory(request: Request) -> dict[str, object]:
        coordinator = _coordinator(request.app)
        await coordinator.clear_memory()
        return {"ok": True}

    @app.post("/api/bonuses")
    async def toggle_bonuses(request: Request, body: ToggleRequest) -> dict[str, object]:
        coordinator = _coordinator(request.app)
        return {"ok": True, "enabled": coordinator.set_bonuses(body.enabled)}

    @app.post("/api/traces/export")
    async def export_trace(request: Request, body: TraceExportRequest) -> dict[str, object]:
        coordinator = _coordinator(request.app)
        return {"ok": True, "trace": coordinator.export_trace(Path(body.directory))}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        coordinator = _coordinator(websocket.app)
        await hub.connect(websocket)
        await hub.send(
            websocket, "world_snapshot", coordinator.world.snapshot.model_dump(mode="json")
        )
        try:
            while True:
                message = await websocket.receive_json()
                if message.get("type") == "simulator_ack":
                    body = message.get("body", {})
                    if isinstance(body, dict):
                        hub.record_ack(str(body.get("timeline_id")), str(body.get("stage")))
        except WebSocketDisconnect:
            hub.disconnect(websocket)

    return app


def _coordinator(app: FastAPI) -> RuntimeCoordinator:
    return cast(RuntimeCoordinator, app.state.coordinator)
