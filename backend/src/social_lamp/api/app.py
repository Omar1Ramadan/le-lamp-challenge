from __future__ import annotations

import base64
import binascii
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from time import monotonic_ns
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from social_lamp.api.hub import ConnectionHub
from social_lamp.capture.frames import CapturedFrame
from social_lamp.config import Settings
from social_lamp.perception.faces import (
    HeuristicFaceProcessor,
    MediaPipeFaceLandmarkerProcessor,
    OpenCvFaceProcessor,
)
from social_lamp.runtime.coordinator import RuntimeCoordinator
from social_lamp.perception.objects import NullObjectDetector
from social_lamp.runtime.live import build_live_runtime


class ReplayRequest(BaseModel):
    directory: str


class TextRequest(BaseModel):
    text: str


class ToggleRequest(BaseModel):
    enabled: bool


class TraceExportRequest(BaseModel):
    directory: str


class VisionFrameRequest(BaseModel):
    image_base64: str


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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    @app.get("/api/replays")
    def replays() -> dict[str, object]:
        fixture_root = Path("evaluation/fixtures")
        items = []
        if fixture_root.exists():
            for directory in sorted(path for path in fixture_root.iterdir() if path.is_dir()):
                items.append(
                    {
                        "id": directory.name,
                        "label": directory.name.replace("-", " ").title(),
                        "directory": str(directory),
                    }
                )
        return {"replays": items}

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
        replay_messages = getattr(coordinator, "replay_messages", [])
        backend_proof_messages = [
            item
            for item in replay_messages
            if item[0] in {"memory_result", "fault"}
            or (
                item[0] == "metric"
                and item[1].get("name") not in {"social_transition", "engagement_seen"}
            )
        ]
        if backend_proof_messages:
            for message_type, message_body in replay_messages:
                if message_type == "behavior_timeline":
                    continue
                await hub.broadcast({"type": message_type, "body": message_body})
            response_messages = [
                {"seq": 10_000 + index, "type": message_type, "body": message_body}
                for index, (message_type, message_body) in enumerate(replay_messages, start=1)
                if message_type != "behavior_timeline"
            ]
        else:
            await hub.broadcast(
                {
                    "type": "world_snapshot",
                    "body": coordinator.world.snapshot.model_dump(mode="json"),
                }
            )
            await hub.broadcast(
                {
                    "type": "metric",
                    "body": {"name": "social_transition", "labels": {"state": "engaged"}},
                }
            )
            response_messages = [
                {
                    "seq": 10_001,
                    "type": "world_snapshot",
                    "body": coordinator.world.snapshot.model_dump(mode="json"),
                }
            ]
        return {
            "ok": True,
            "revision": coordinator.world.snapshot.revision,
            "messages": response_messages,
        }

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
        await hub.broadcast(
            {
                "type": "metric",
                "body": {"name": f"recall_{response.status}", "value": 1},
            }
        )
        return {"ok": True, "response": response.__dict__}

    @app.post("/api/vision/frame")
    async def vision_frame(request: Request, body: VisionFrameRequest) -> dict[str, object]:
        coordinator = _coordinator(request.app)
        frame = _decode_browser_frame(body.image_base64)
        timeline = await coordinator.process_vision_frame(
            frame,
            face_processor=_browser_face_processor(request.app),
            object_detector=_browser_object_detector(request.app),
            anchors={"desk": (0.25, 0.45, 0.75, 0.95)},
        )
        degraded_detail = getattr(request.app.state, "browser_face_processor_degraded", None)
        if isinstance(degraded_detail, str):
            await coordinator._set_health("vision_model", "degraded", degraded_detail)
        return {
            "ok": True,
            "revision": coordinator.world.snapshot.revision,
            "world_snapshot": coordinator.world.snapshot.model_dump(mode="json"),
            "behavior_timeline": timeline.model_dump(mode="json") if timeline is not None else None,
            "vision_debug": getattr(_browser_face_processor(request.app), "last_debug", None),
        }

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


def _decode_browser_frame(image_base64: str) -> CapturedFrame:
    try:
        encoded = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid image_base64") from exc

    try:
        import cv2
        import numpy as np
    except Exception as exc:
        raise HTTPException(status_code=503, detail="vision decoder unavailable") from exc

    data = np.frombuffer(encoded, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="invalid image data")
    return CapturedFrame(image=image, mono_ns=monotonic_ns())


def _browser_face_processor(
    app: FastAPI,
) -> MediaPipeFaceLandmarkerProcessor | OpenCvFaceProcessor | HeuristicFaceProcessor:
    processor = getattr(app.state, "browser_face_processor", None)
    if processor is None:
        landmark_exc: RuntimeError | None = None
        if Settings().enable_mediapipe_face_landmarker:
            try:
                processor = MediaPipeFaceLandmarkerProcessor()
                app.state.browser_face_processor_degraded = None
            except RuntimeError as exc:
                landmark_exc = exc
        if processor is None:
            try:
                processor = OpenCvFaceProcessor()
                app.state.browser_face_processor_degraded = (
                    str(landmark_exc) if landmark_exc is not None else None
                )
            except RuntimeError as opencv_exc:
                processor = HeuristicFaceProcessor()
                details = [str(exc) for exc in (landmark_exc, opencv_exc) if exc is not None]
                app.state.browser_face_processor_degraded = "; ".join(details)
        app.state.browser_face_processor = processor
    return cast(
        MediaPipeFaceLandmarkerProcessor | OpenCvFaceProcessor | HeuristicFaceProcessor,
        processor,
    )


def _browser_object_detector(app: FastAPI) -> NullObjectDetector:
    detector = getattr(app.state, "browser_object_detector", None)
    if detector is None:
        detector = NullObjectDetector()
        app.state.browser_object_detector = detector
    return cast(NullObjectDetector, detector)
