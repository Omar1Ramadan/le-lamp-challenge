# Demo Guide

This demo can run entirely from deterministic replay fixtures. Use live camera and microphone only after the replay path has passed.

## Setup

```bash
uv sync --locked
pnpm install --frozen-lockfile
uv run uvicorn social_lamp.main:app --port 8000
pnpm --dir frontend dev --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173` and confirm the dashboard shows backend health and a connected WebSocket state. Replay proof controls are populated from `GET /api/replays`; proof checkmarks must come from backend replay messages, persisted memory, and socket/API evidence, not local click handlers.

## Four-step core journey

1. **Engagement**
   - Click **Load core journey replay**.
   - Confirm the demo rail marks engagement complete from backend replay evidence (`engagement_seen`) rather than the button click.
   - Watch the lamp move from idle/candidate to engaged and acknowledge the person.

2. **Attention seeking**
   - Continue the replay into disengagement.
   - Confirm the dashboard displays `Seeking attention: level 1` from the backend replay metric.
   - Point out that speech, renewed engagement, television/media, or operator suppression can prevent escalation.

3. **Memory formation**
   - Confirm the replay creates a memory article for keys.
   - Verify the evidence states the keys were on the right side of the desk and remains queryable from the SQLite runtime memory.
   - Use **Show evidence** to display the supporting observation ID.

4. **Memory recall**
   - Type `Where are my keys?` into **Ask the lamp**.
   - Click **Ask**.
   - Confirm the answer says the keys were on the right side of the desk and includes the evidence ID.

## Live local-first acceptance

For the functional local-first demo proof, run one hardware-backed session when devices are available and one hardware-free fake/live-runtime CI path:

- Live path: grant webcam/microphone permission, start the app, verify camera/microphone/cloud/simulator health, form a stable object memory, ask a recall question, and export a trace.
- CI/fake path: `uv run pytest tests/runtime/test_demo_acceptance.py -v` injects fake vision frames into the live runtime, persists evidence, recalls it through the normal conversation path, and proves replay/live coexist without physical hardware.
- Offline path: disable network after dependencies are installed, load the core replay, confirm visible simulator output and WebSocket updates, ask the recall question, and export the evaluation report.

## Bonus journey

Run the Playwright bonus fixture or use the dashboard bonus controls to show:

- Person A/B active-speaker association.
- Coarse affect evidence only when confidence is high enough.
- Preference score updates and reset behavior.
- User interruption cancelling lamp speech and switching to listening.
- Television/background media suppressing unsolicited sound while typed recall still works.

## Evaluation artifact

After the demo, produce a deterministic report:

```bash
uv run python -m social_lamp.evaluation.cli --fixture evaluation/fixtures/core-journey --output output/evaluation
```

Record the commit hash, configuration hash, fixture path, dataset version, and whether the report is marked sample-only. Keep private reports out of git.

## Offline proof

To prove the offline path, disable network access after dependencies are installed, keep `CONVERSATION_PROVIDER=template`, restart the backend/frontend, load the core journey replay, ask the recall question, and regenerate the evaluation report.
