# Agent Task 09 — Stable object tracking and localization

Child roles spawned via Hermes delegation:
- Task 9 spec checker

## RED

Command:

```bash
PYTHONPATH= uv run pytest tests/perception/test_objects.py -v
```

Expected failure observed:

```text
ModuleNotFoundError: No module named 'social_lamp.perception.location'
```

## GREEN

Commands:

```bash
PYTHONPATH= uv run pytest tests/perception/test_objects.py -v
PYTHONPATH= uv run pytest -q
PYTHONPATH= uv run ruff check backend tests
PYTHONPATH= uv run mypy
```

Results:

- focused pytest: 5 passed
- full pytest: 22 passed, 1 third-party deprecation warning
- ruff: All checks passed
- mypy: Success, no issues in 25 source files
