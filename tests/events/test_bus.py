import pytest
from social_lamp.events.bus import EventBus


@pytest.mark.asyncio
async def test_bus_drops_oldest_noncritical_event() -> None:
    bus: EventBus[int] = EventBus(capacity=2)
    subscription = bus.subscribe("world")
    await bus.publish(1)
    await bus.publish(2)
    await bus.publish(3)
    assert await subscription.get() == 2
    assert await subscription.get() == 3
    assert bus.dropped("world") == 1


@pytest.mark.asyncio
async def test_bus_never_silently_drops_critical_event() -> None:
    bus: EventBus[int] = EventBus(capacity=1)
    bus.subscribe("world")
    await bus.publish(1)
    with pytest.raises(RuntimeError, match="critical queue overflow"):
        await bus.publish(2, critical=True)
