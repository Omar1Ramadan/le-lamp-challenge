from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic_ns
from typing import Any
from uuid import UUID

from social_lamp.api.hub import ConnectionHub
from social_lamp.domain.contracts import BehaviorTimeline, ComponentHealth


TIMELINE_RECEIVED_TIMEOUT_MS = 500
FIRST_VISIBLE_TIMEOUT_MS = 1500
COMPLETION_TIMEOUT_MS = 5000


@dataclass
class SimulatorTimelineStatus:
    timeline_id: str
    correlation_id: str
    issued_mono_ns: int
    duration_ms: int
    behavior_id: str | None = None
    received_ack_mono_ns: int | None = None
    first_visible_mono_ns: int | None = None
    completed_mono_ns: int | None = None
    cancelled_mono_ns: int | None = None
    cancellation_reason: str | None = None

    @property
    def received_latency_ms(self) -> float | None:
        if self.received_ack_mono_ns is not None:
            return (self.received_ack_mono_ns - self.issued_mono_ns) / 1_000_000
        return None

    @property
    def first_visible_latency_ms(self) -> float | None:
        if self.first_visible_mono_ns is not None:
            return (self.first_visible_mono_ns - self.issued_mono_ns) / 1_000_000
        return None

    @property
    def is_complete(self) -> bool:
        return self.completed_mono_ns is not None

    @property
    def is_cancelled(self) -> bool:
        return self.cancelled_mono_ns is not None

    def now_ns(self) -> int:
        return monotonic_ns()


class SimulatorAdapter:
    def __init__(self, hub: ConnectionHub) -> None:
        self._hub = hub
        self._timelines: dict[str, SimulatorTimelineStatus] = {}
        self.health = ComponentHealth(
            component="simulator", status="degraded", detail="no browser client connected"
        )

    async def execute(self, timeline: BehaviorTimeline) -> UUID:
        now = monotonic_ns()
        tid = str(timeline.timeline_id)
        self._timelines[tid] = SimulatorTimelineStatus(
            timeline_id=tid,
            correlation_id=str(timeline.correlation_id),
            issued_mono_ns=now,
            duration_ms=timeline.duration_ms,
        )
        if self._hub.client_count == 0:
            self.health = ComponentHealth(
                component="simulator", status="degraded", detail="no browser client connected"
            )
            return timeline.timeline_id
        await self._hub.broadcast(
            {"type": "behavior_timeline", "body": timeline.model_dump(mode="json")}
        )
        self._update_health()
        return timeline.timeline_id

    def handle_ack(self, ack_body: dict[str, Any]) -> None:
        timeline_id = str(ack_body.get("timeline_id", ""))
        ack_type = str(ack_body.get("ack_type", ""))
        now = monotonic_ns()

        status = self._timelines.get(timeline_id)
        if status is None:
            return

        if ack_type == "timeline_received":
            if status.received_ack_mono_ns is None:
                status.received_ack_mono_ns = now
        elif ack_type == "first_visible_frame":
            if status.first_visible_mono_ns is None:
                status.first_visible_mono_ns = now
        elif ack_type == "timeline_complete":
            if status.completed_mono_ns is None:
                status.completed_mono_ns = now
        elif ack_type == "timeline_cancelled":
            if status.cancelled_mono_ns is None:
                status.cancelled_mono_ns = now
                status.cancellation_reason = str(ack_body.get("reason", "unknown"))

        self._update_health()

    def _update_health(self) -> None:
        if self._hub.client_count == 0:
            self.health = ComponentHealth(
                component="simulator", status="degraded", detail="no browser client connected"
            )
            return

        now = monotonic_ns()
        degraded_details: list[str] = []

        for status in self._timelines.values():
            if status.is_complete or status.is_cancelled:
                continue
            elapsed = (now - status.issued_mono_ns) / 1_000_000

            if status.received_ack_mono_ns is None and elapsed > TIMELINE_RECEIVED_TIMEOUT_MS:
                degraded_details.append(
                    f"missing timeline_received for {status.timeline_id}"
                )
            elif status.first_visible_mono_ns is None and elapsed > FIRST_VISIBLE_TIMEOUT_MS:
                degraded_details.append(
                    f"missing first_visible_frame for {status.timeline_id}"
                )

        if degraded_details:
            self.health = ComponentHealth(
                component="simulator",
                status="degraded",
                detail="; ".join(degraded_details[:3]),
            )
        else:
            self.health = ComponentHealth(component="simulator", status="ok")

    def timeline_status(self, timeline_id: str) -> SimulatorTimelineStatus | None:
        return self._timelines.get(timeline_id)

    def ack_latencies(self) -> dict[str, dict[str, float | None]]:
        result: dict[str, dict[str, float | None]] = {}
        for tid, status in self._timelines.items():
            result[tid] = {
                "received_latency_ms": status.received_latency_ms,
                "first_visible_latency_ms": status.first_visible_latency_ms,
            }
        return result

    @property
    def pose(self) -> dict[str, float]:
        return {}

    async def neutralize(self) -> None:
        for status in self._timelines.values():
            if not status.is_complete and not status.is_cancelled:
                status.cancelled_mono_ns = monotonic_ns()
                status.cancellation_reason = "neutralize"
        await self._hub.broadcast({"type": "neutralize", "body": {}})
