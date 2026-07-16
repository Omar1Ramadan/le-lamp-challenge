# Verification

## Quick start

```bash
pnpm verify
```

Runs all release gates locally. Exits non-zero if any gate fails.

## What's checked

| # | Gate | Command |
|---|------|---------|
| 1 | Python lint | `uv run ruff check backend tests` |
| 2 | Python format | `uv run ruff format --check backend tests` |
| 3 | Python types | `uv run mypy` |
| 4 | Python tests | `uv run pytest -q` |
| 5 | Frontend lint | `pnpm --dir frontend lint` (oxlint) |
| 6 | Frontend types | `pnpm --dir frontend exec tsc -b` |
| 7 | Frontend tests | `pnpm --dir frontend test -- --run` |
| 8 | Frontend build | `pnpm --dir frontend build` |
| 9 | Contract freshness | `bash scripts/check-contracts` |
| 10 | Git diff check | `git diff --check` |

## Contract generation

When Pydantic model schemas change (`backend/src/social_lamp/domain/contracts.py`),
regenerate the frontend contract types:

```bash
pnpm contracts:generate
```

This runs two steps:
1. `tools/export_contract_schemas.py` — exports JSON Schema from Pydantic models to `domain.schema.json`
2. `json-schema-to-typescript` — converts `domain.schema.json` to `frontend/src/contracts/generated.ts`

To verify contracts are fresh without regenerating:

```bash
pnpm verify:contracts
```

## CI pipeline

Pushes and PRs to `main` trigger `.github/workflows/verify.yml` on GitHub Actions.
It runs the same gates as `pnpm verify` on `ubuntu-latest` with Python 3.12 and Node 22.

## Known pre-existing failures

These issues exist in the codebase and cause non-zero exit codes in `pnpm verify`.
They are tracked as tech debt and will be resolved separately:

- `uv run mypy` fails on `backend/src/social_lamp/behavior/policy.py` and
  `backend/src/social_lamp/perception/faces.py`
- `uv run ruff check .` flags line-length and import-ordering violations in
  unrelated files
- "Full release verification" in the README includes additional acceptance-level
  tests not covered by `pnpm verify` (e2e, evaluation CLI, etc.)

## What to do when a gate fails

1. **Lint / format**: run `uv run ruff check --fix backend tests` or
   `uv run ruff format backend tests`
2. **Types**: fix type annotations, add missing stubs
3. **Tests**: `uv run pytest tests/path/to/test.py -v` to debug
4. **Contracts stale**: `pnpm contracts:generate` then commit the updated
   `generated.ts`
5. **Frontend**: follow the specific tool's output (oxlint, tsc, vitest, vite)
6. **Git dirty**: `git diff --check` shows whitespace errors; fix and re-stage
