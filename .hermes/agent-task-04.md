# Agent Task 04 — Behavior policy and compositor

Child model: `ollama/qwen3-coder:30b` launched as patch-proposal process `proc_0ea9573aa761`.

## RED

Command:

```bash
PYTHONPATH= uv run pytest tests/behavior/test_policy_compositor.py -v
```

Expected failure observed:

```text
ModuleNotFoundError: No module named 'social_lamp.behavior'
```

## GREEN

Commands:

```bash
PYTHONPATH= uv run pytest tests/behavior/test_policy_compositor.py tests/domain/test_contracts.py -v
PYTHONPATH= uv run pytest -q
PYTHONPATH= uv run ruff check backend tests
PYTHONPATH= uv run mypy
```

Results:

- focused pytest: 3 passed
- full pytest: 7 passed
- ruff: All checks passed after line wrapping
- mypy: Success, no issues in 14 source files
