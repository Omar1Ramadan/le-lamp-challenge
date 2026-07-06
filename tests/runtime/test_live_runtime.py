from pathlib import Path

from fastapi.testclient import TestClient
from social_lamp.adapters.simulator import SimulatorAdapter
from social_lamp.api.app import create_app
from social_lamp.memory.repository import MemoryRepository
from social_lamp.runtime import testing as testing_runtime
from social_lamp.runtime.coordinator import RuntimeCoordinator
from social_lamp.runtime.live import RuntimeMetrics, build_live_runtime


def test_create_app_uses_live_runtime_dependencies(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "memory.db"))

    with TestClient(create_app()) as client:
        coordinator = client.app.state.coordinator

        assert isinstance(coordinator, RuntimeCoordinator)
        assert isinstance(coordinator.memory, MemoryRepository)
        assert isinstance(coordinator.simulator, SimulatorAdapter)
        assert isinstance(coordinator.metrics, RuntimeMetrics)
        assert not isinstance(coordinator.memory, testing_runtime.TestMemory)
        assert not isinstance(coordinator.simulator, testing_runtime.FakeSimulator)
        assert client.get("/api/health").json() == {"status": "healthy"}


def test_live_runtime_builder_uses_explicit_settings_and_loopback_adapter(tmp_path: Path) -> None:
    database = tmp_path / "configured" / "memory.db"

    with TestClient(create_app(database_path=database)) as client:
        coordinator = client.app.state.coordinator

        assert isinstance(coordinator.memory, MemoryRepository)
        assert isinstance(coordinator.simulator, SimulatorAdapter)
        assert database.exists()
        assert client.get("/api/world").json()["social_state"] == "idle"


def test_live_runtime_start_stop_are_idempotent(tmp_path: Path) -> None:
    with TestClient(create_app(database_path=tmp_path / "memory.db")) as client:
        assert client.post("/api/session/start").json() == {"ok": True, "running": True}
        assert client.post("/api/session/start").json() == {"ok": True, "running": True}
        assert client.post("/api/session/stop").json() == {"ok": True, "running": False}
        assert client.post("/api/session/stop").json() == {"ok": True, "running": False}


async def test_build_live_runtime_directly_constructs_real_ports(tmp_path: Path) -> None:
    coordinator = await build_live_runtime(database_path=tmp_path / "memory.db")
    try:
        assert isinstance(coordinator.memory, MemoryRepository)
        assert isinstance(coordinator.simulator, SimulatorAdapter)
        assert isinstance(coordinator.metrics, RuntimeMetrics)
    finally:
        await coordinator.stop()
