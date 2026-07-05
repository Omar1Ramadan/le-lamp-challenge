# Agent Task 10 — Transactional evidence memory

Child roles spawned via Hermes delegation:
- Task 10 memory repository spec checker

## RED

Command:

```bash
PYTHONPATH= uv run pytest tests/memory/test_repository.py -v
```

Expected failure observed:

```text
ModuleNotFoundError: No module named 'social_lamp.memory'
```

## GREEN

Commands:

```bash
PYTHONPATH= uv run pytest tests/memory/test_repository.py -v
PYTHONPATH= uv run pytest -q
PYTHONPATH= uv run ruff check backend tests
PYTHONPATH= uv run mypy
pnpm --dir frontend test -- --run src/scene/pose.test.ts
pnpm --dir frontend exec tsc --noEmit
pnpm --dir frontend build
```

Results:

- memory pytest: 5 passed
- full backend pytest: 27 passed, 1 third-party deprecation warning
- ruff: All checks passed
- mypy: Success, no issues in 28 source files
- frontend pose test: 1 passed
- frontend typecheck/build: passed with existing Three/R3F large chunk warning

## Coverage

- newest grounded observation ordering
- not found
- exact alias lookup
- session scope
- before_utc filtering
- injected rollback between observation insert and last-known update
