from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    @property
    def mono_ns(self) -> int: ...

    @property
    def wall_time_utc(self) -> str: ...


class SystemClock:
    @property
    def mono_ns(self) -> int:
        import time

        return time.monotonic_ns()

    @property
    def wall_time_utc(self) -> str:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass
class FakeClock:
    mono_ns: int
    wall_time_utc: str

    def advance_ms(self, milliseconds: int) -> None:
        self.mono_ns += milliseconds * 1_000_000
