# Agent Task 02 — Domain contracts

Child model: `ollama/qwen3-coder:30b` launched as patch-proposal process `proc_8cd892074a31`.

## RED

Command:

```bash
PYTHONPATH= uv run pytest tests/domain/test_contracts.py -v
```

Expected failure observed:

```text
ModuleNotFoundError: No module named 'social_lamp.domain'
```

## GREEN

Commands:

```bash
PYTHONPATH= uv run pytest tests/domain/test_contracts.py -v
PYTHONPATH= uv run ruff check backend tests
PYTHONPATH= uv run mypy
```

Results:

- `pytest`: 2 passed
- `ruff`: All checks passed after import sort fix
- `mypy`: Success, no issues in 5 source files
