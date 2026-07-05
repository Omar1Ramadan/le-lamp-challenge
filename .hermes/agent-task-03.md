# Agent Task 03 — Bounded events and world model

Child model: `ollama/qwen3-coder:30b` launched as patch-proposal process `proc_18a547d408fe`.

## RED

Command:

```bash
PYTHONPATH= uv run pytest tests/events/test_bus.py tests/world/test_model.py -v
```

Expected failures observed:

```text
ModuleNotFoundError: No module named 'social_lamp.events'
ModuleNotFoundError: No module named 'social_lamp.world'
```

## GREEN

Commands:

```bash
PYTHONPATH= uv run pytest tests/events/test_bus.py tests/world/test_model.py -v
PYTHONPATH= uv run pytest -q
PYTHONPATH= uv run ruff check backend tests
PYTHONPATH= uv run mypy
```

Results:

- focused pytest: 3 passed
- full pytest: 6 passed
- ruff: All checks passed
- mypy: Success, no issues in 9 source files

## Parent corrections

- Used Python 3.12 generic class syntax for `EventBus[T]` and `Subscription[T]` because the configured ruff `UP` rules require it.
