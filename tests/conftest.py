import pytest


@pytest.fixture(autouse=True)
def disable_live_capture_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_LIVE_CAPTURE", "false")
