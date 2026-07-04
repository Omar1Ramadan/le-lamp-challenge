# Agent Task 01 — Scaffold reproducible workspaces

Child model used: `ollama/qwen3-coder:30b` via `ollama run` as a patch-proposal child.
Child output: `.hermes/qwen-task-01-output.txt` in the task worktree during orchestration; parent applied corrected implementation.

## RED

Command:

```bash
uv run pytest tests/test_smoke.py -v
```

Expected failure observed before backend package existed:

```text
ModuleNotFoundError: No module named 'social_lamp'
```

## GREEN / verification

Hermes injects `PYTHONPATH` pointing at its own app venv, which contaminates project subprocess imports on this machine. Project verification was run with `PYTHONPATH=` cleared so uv uses the task worktree `.venv`.

Commands:

```bash
PYTHONPATH= uv run pytest tests/test_smoke.py -v
PYTHONPATH= uv run ruff check backend tests
PYTHONPATH= uv run mypy
pnpm --dir frontend exec tsc --noEmit
pnpm --dir frontend build
```

Results:

- `pytest`: 1 passed
- `ruff`: All checks passed
- `mypy`: Success, no issues in 2 source files
- `tsc --noEmit`: passed
- `vite build`: passed

## Parent corrections to Qwen proposal

- Kept `.python-version` as exact plan value `3.12`, not Qwen's proposed `3.12.4`.
- Used exact plan root package scripts.
- Used exact smoke test importing `create_app`, not app directly.
- Added explicit Hatchling build backend because `[tool.uv] package = true` otherwise used an incompatible implicit setuptools backend under Python 3.12.
- Added root `node_modules/` ignore because pnpm workspace install creates root `node_modules`.
