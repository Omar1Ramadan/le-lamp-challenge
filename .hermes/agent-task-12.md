# Agent Task 12 — Deterministic event replay

Child role spawned via Hermes delegation:
- no-tool Task 12 spec reviewer

## RED

Command:

```bash
PYTHONPATH= uv run pytest tests/replay/test_trace.py -v
```

Expected failure observed:

```text
ModuleNotFoundError: No module named 'social_lamp.replay'
```

## GREEN

Commands:

```bash
PYTHONPATH= uv run pytest tests/replay/test_trace.py -v
PYTHONPATH= uv run pytest -q
PYTHONPATH= uv run ruff check backend tests
PYTHONPATH= uv run mypy
```

Results:

- focused pytest: 3 passed
- full pytest: 33 passed, 1 third-party deprecation warning
- ruff: All checks passed
- mypy: Success, no issues in 34 source files

## Parent corrections

- Hash events from written bytes so checksum is stable on Windows newline handling.
- Validate manifest schema before dataclass construction so incomplete incompatible manifests raise ValueError.
