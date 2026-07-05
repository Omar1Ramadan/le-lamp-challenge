# Agent Task 13 — Typed explainable dashboard

Child role spawned via Hermes delegation:
- no-tool Task 13 spec reviewer

## RED

Command:

```bash
pnpm --dir frontend test -- --run src/state/store.test.ts src/components/Dashboard.test.tsx
```

Expected failures observed:

```text
reduceServerMessage is not a function
Cannot find module './Inspector'
```

## GREEN

Commands:

```bash
PYTHONPATH= uv run python tools/export_contract_schemas.py
pnpm --dir frontend exec json2ts -i src/contracts/domain.schema.json -o src/contracts/generated.ts
pnpm --dir frontend test -- --run
pnpm --dir frontend exec tsc --noEmit
pnpm --dir frontend build
PYTHONPATH= uv run pytest -q
PYTHONPATH= uv run ruff check backend tests tools
PYTHONPATH= uv run mypy
```

Results:

- frontend tests: 3 passed
- frontend typecheck/build: passed with existing large Three/R3F bundle warning
- backend pytest: 33 passed, 1 third-party deprecation warning
- ruff: All checks passed
- mypy: Success, no issues in 34 source files
- schema generation: second run produced no diff

## Parent corrections

- Flattened nested Pydantic `$defs` for json2ts.
- Added jsdom and jest-dom setup for component tests.
- Made replay checksum verification tolerate CRLF checkout normalization.
