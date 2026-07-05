# Simulated Social Lamp

A portfolio-quality simulated social lamp that detects engagement, expresses state through a six-channel 3D lamp, stores grounded object evidence, answers deterministic recall questions, and reports evaluation metrics. The project is designed to run without physical lamp hardware: replay, simulator, memory, text recall, and reports all work offline.

## What is included

- FastAPI backend with typed Pydantic contracts, bounded event flow, single-writer world model, SQLite evidence memory, deterministic replay, evaluation reports, and optional cloud conversation behind a provider port.
- React/TypeScript/Vite frontend with React Three Fiber lamp rendering, dashboard panels, replay controls, and Playwright demo journeys.
- Public-safe deterministic replay fixtures under `evaluation/fixtures/`.
- Privacy, demo, and limitations documentation in `docs/`.

## Prerequisites

Tested on Windows with Git Bash/MSYS, Python 3.12, `uv`, Node 22, and `pnpm`.

Install tools if needed:

```bash
# Python package/runtime manager
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Node package manager
corepack enable
corepack prepare pnpm@latest --activate
```

Optional live features require a webcam, microphone, and local permission grants. The simulator and replay path do not require them.

## Clean-machine setup

From a fresh clone:

```bash
git clone <repo-url> simulated-social-lamp
cd simulated-social-lamp
uv sync --locked
pnpm install --frozen-lockfile
```

If you are developing and need to refresh generated contract types:

```bash
uv run python tools/export_contract_schemas.py
pnpm --dir frontend exec json2ts -i src/contracts/domain.schema.json -o src/contracts/generated.ts
```

## Environment configuration

Copy the example file and only fill in secrets locally:

```bash
cp .env.example .env
```

Important variables:

- `CONVERSATION_PROVIDER=template` keeps recall deterministic and offline.
- `ENABLE_CLOUD_CONVERSATION=true` plus `OPENAI_API_KEY` enables the optional cloud provider.
- `DATABASE_PATH` points at the local SQLite evidence memory.
- `SNAPSHOT_PATH` points at private local snapshots if enabled.
- `RETENTION_DAYS` controls cleanup policy for private runtime data.

Do not commit `.env`, local databases, raw media, snapshots, model weights, or generated private reports.

## Run the app

Start the backend and frontend in separate terminals:

```bash
uv run uvicorn social_lamp.main:app --port 8000
pnpm --dir frontend dev --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173`. The API is loopback-only for this release.

## Offline replay workflow

The deterministic replay path is the primary delivery proof and works with network disabled:

```bash
uv run uvicorn social_lamp.main:app --port 8000
pnpm --dir frontend dev --host 127.0.0.1 --port 5173
```

Then use the dashboard button **Load core journey replay**, or call the backend directly:

```bash
curl -X POST http://127.0.0.1:8000/api/replay \
  -H 'content-type: application/json' \
  -d '{"directory":"evaluation/fixtures/core-journey"}'
```

The replay demonstrates engagement, attention seeking, memory formation, and memory recall without camera, microphone, cloud, or physical hardware.

## Tests and validation

Focused backend check for delivery docs:

```bash
uv run pytest tests/test_delivery_files.py -v
```

Full release verification:

```bash
uv sync --locked
pnpm install --frozen-lockfile
uv run ruff check backend tests
uv run ruff format --check backend tests
uv run mypy
uv run pytest -q
pnpm --dir frontend test -- --run
pnpm --dir frontend exec tsc --noEmit
pnpm --dir frontend build
pnpm e2e
uv run python -m social_lamp.evaluation.cli --fixture evaluation/fixtures/core-journey --output output/evaluation
git diff --check
```

The evaluation command writes:

- `output/evaluation/report.json`
- `output/evaluation/report.md`

These reports are intentionally ignored because local runs may include private hardware and configuration details.

## Optional live and model setup

The public demo does not require model downloads. For live perception work, install the locked dependencies with `uv sync --locked`, then run the explicit camera probe:

```bash
uv run python -m social_lamp.capture.frames --probe
```

Object model downloads should be performed only by explicit setup commands for live experiments, not during automated tests.

## Troubleshooting

- **`uv sync --locked` fails:** confirm Python 3.12 is installed and on PATH.
- **`pnpm install --frozen-lockfile` fails:** run `corepack enable`, then retry with Node 22.
- **Playwright cannot start servers:** ensure ports `8000` and `5173` are free.
- **Camera or microphone unavailable:** use replay/offline mode; the core simulator remains operational.
- **Cloud responses unavailable:** keep `CONVERSATION_PROVIDER=template` for deterministic grounded recall.
- **WebGL unavailable:** backend replay, memory, text recall, and reports still work; use dashboard health indicators to show the degraded frontend state.

## Architecture and design links

- Overall system: `docs/design/01-overall-system-design.md`
- Events/world model: `docs/design/02-events-world-model-observability.md`
- Engagement perception: `docs/design/03-engagement-perception.md`
- Object perception and memory: `docs/design/04-object-perception-memory.md`
- Behavior and preferences: `docs/design/05-behavior-adaptive-preferences.md`
- Output compositor/adapters: `docs/design/06-output-compositor-adapters.md`
- Conversation/audio: `docs/design/07-conversation-audio-intelligence.md`
- Simulator/dashboard: `docs/design/08-simulator-dashboard.md`
- Evaluation/delivery: `docs/design/09-evaluation-delivery.md`

## Privacy summary

The app stores typed observations and evidence references, not raw camera/audio buffers, for normal operation. Local memory can be cleared with the dashboard control or `POST /api/memory/clear`. See `docs/PRIVACY.md` for retention, deletion, and offline behavior.
