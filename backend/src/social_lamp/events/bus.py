import asyncio
from dataclasses import dataclass


@dataclass
class Subscription[T]:
    queue: asyncio.Queue[T]

    async def get(self) -> T:
        return await self.queue.get()


class EventBus[T]:
    def __init__(self, capacity: int = 32) -> None:
        self._capacity = capacity
        self._queues: dict[str, asyncio.Queue[T]] = {}
        self._dropped: dict[str, int] = {}

    def subscribe(self, name: str) -> Subscription[T]:
        queue: asyncio.Queue[T] = asyncio.Queue(maxsize=self._capacity)
        self._queues[name] = queue
        self._dropped[name] = 0
        return Subscription(queue)

    async def publish(self, event: T, *, critical: bool = False) -> None:
        for name, queue in self._queues.items():
            if queue.full():
                if critical:
                    raise RuntimeError(f"critical queue overflow: {name}")
                queue.get_nowait()
                self._dropped[name] += 1
            queue.put_nowait(event)

    def dropped(self, name: str) -> int:
        return self._dropped[name]
