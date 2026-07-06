from collections import defaultdict
from typing import Any

from fastapi import WebSocket


class ConnectionHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._sequence = 0
        self._acks: dict[str, list[str]] = defaultdict(list)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def envelope(self, message_type: str, body: dict[str, Any]) -> dict[str, object]:
        self._sequence += 1
        return {"seq": self._sequence, "type": message_type, "body": body}

    async def send(self, websocket: WebSocket, message_type: str, body: dict[str, Any]) -> None:
        await websocket.send_json(self.envelope(message_type, body))

    async def broadcast(self, message: dict[str, object]) -> None:
        body = message.get("body", {})
        if not isinstance(body, dict):
            body = {"value": body}
        envelope = self.envelope(str(message["type"]), body)
        failed: list[WebSocket] = []
        for client in self._clients:
            try:
                await client.send_json(envelope)
            except RuntimeError:
                failed.append(client)
        for client in failed:
            self.disconnect(client)

    def record_ack(self, timeline_id: str, stage: str) -> None:
        stages = self._acks[timeline_id]
        if stage not in stages:
            stages.append(stage)

    def timeline_acknowledgements(self, timeline_id: str) -> list[str]:
        return list(self._acks.get(timeline_id, []))
