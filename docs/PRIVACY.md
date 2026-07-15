# Privacy Controls

## Data minimization

The simulated social lamp is designed to keep raw pixels and audio out of durable storage during normal operation. Capture and audio modules use bounded in-memory buffers; domain modules publish typed observations, timestamps, health states, and evidence identifiers. Public replay fixtures contain synthetic or public-safe events only.

## What can be stored locally

Local runtime data may include:

- SQLite evidence memory at `DATABASE_PATH`.
- Object labels, scene-relative locations, monotonic timestamps, UTC display timestamps, and evidence IDs.
- Anonymous session-local person IDs such as `person-1` and `person-2` while the runtime is active.
- Optional private snapshots if `SNAPSHOT_PATH` is enabled for a local live demo.
- Evaluation reports under `output/evaluation/` that may include hardware profile and configuration hash.

Local runtime data must not include committed secrets, raw private media, model weights, or private snapshots.

## Face tracking and identity

Multi-person face tracking uses only short-lived spatial/temporal association between face bounding boxes in nearby frames. It does not perform face recognition, create biometric embeddings, or persist identity across sessions. Runtime labels such as `person-1` and `person-2` are anonymous and reset with the session.

## Anonymous person tracking

Live face tracking uses anonymous session-local labels such as `person-1` and `person-2`. These IDs are assigned from face bounding-box movement within the current runtime session only. They are not biometric identities, are not face-recognition results, and are not designed to persist across restarts or separate sessions.

The tracker does not generate or store face embeddings. It matches detections spatially over time using bounding-box overlap and confidence signals so the runtime can keep engagement state stable while multiple people are visible.

## Engagement calibration

Engagement calibration is session-local. During a three-second calibration window, the runtime may compute neutral head pose, face scale, and gaze baseline for the current anonymous person track. These calibration baselines are not persisted, are not exposed in normal UI/API responses, and are not used for identity recognition. If calibration is cancelled, fails, or is unavailable, the system uses fallback engagement thresholds.

## Retention

The sample `.env.example` sets `RETENTION_DAYS=7`. Treat this as a seven days default for private demo data unless a shorter event-specific policy is chosen. Retention cleanup should remove local databases, private snapshots, raw media captures, and generated private reports that are older than the configured period.

## Clear all memory

Use the dashboard **Clear all memory** control or call:

```bash
curl -X POST http://127.0.0.1:8000/api/memory/clear
```

This clears the local evidence memory used for recall. For a complete privacy reset, also delete `.runtime/`, private snapshots, raw media captures, and `output/evaluation/`.

## Offline mode

Offline replay uses deterministic fixtures and the template conversation provider. It does not require webcam, microphone, cloud APIs, or network access after dependencies are installed. In offline mode the lamp can still replay behavior, form sample memory evidence, answer text recall from evidence, and produce reports.

## Cloud provider

Cloud conversation is optional. Leave `CONVERSATION_PROVIDER=template` and `ENABLE_CLOUD_CONVERSATION=false` for private or offline demos. If cloud conversation is enabled, keep API keys only in `.env` and validate every factual response against memory evidence before displaying it.

## Consent guidance

Before a live webcam/microphone session, tell participants what is being observed, whether snapshots are enabled, how long private data is retained, and how to request deletion. Avoid recording bystanders, screens with secrets, or private rooms unless everyone involved has consented.
