# Simulated Social Lamp Design Set

This directory is the authoritative design source for the simulated social lamp. The documents define a local-first perception-to-action system that can drive a browser simulator now and physical hardware through an adapter later.

## Status

All documents were reviewed and approved on 2026-07-04. The design set is frozen as the baseline for TDD implementation planning; later design changes require a documented amendment.

| Order | Document | Depends on | Status |
| --- | --- | --- | --- |
| 1 | [Overall system design](01-overall-system-design.md) | Challenge brief | Approved |
| 2 | [Events, world model, and observability](02-events-world-model-observability.md) | 1 | Approved |
| 3 | [Engagement perception](03-engagement-perception.md) | 1-2 | Approved |
| 4 | [Object perception and memory](04-object-perception-memory.md) | 1-2 | Approved |
| 5 | [Behavior and adaptive preferences](05-behavior-adaptive-preferences.md) | 1-4 | Approved |
| 6 | [Output compositor and adapters](06-output-compositor-adapters.md) | 1-2, 5 | Approved |
| 7 | [Conversation and audio intelligence](07-conversation-audio-intelligence.md) | 1-6 | Approved |
| 8 | [Simulator and dashboard](08-simulator-dashboard.md) | 1-7 | Approved |
| 9 | [Evaluation and delivery](09-evaluation-delivery.md) | 1-8 | Approved |

## Dependency Rules

- Perception modules publish observations; they never command outputs or generate conversational claims.
- The world model is the only writer of stable runtime state.
- Behavior policy consumes stable state and emits intent; it does not address a simulator or motor directly.
- The compositor owns output arbitration and emits timelines through `LampAdapter`.
- Memory stores evidence. Conversation can query evidence but cannot create observations.
- The web client renders server state and sends explicit operator commands; it is not a second decision engine.
- Bonus capabilities are feature-gated and cannot change core behavior when disabled.

## Shared Glossary

- **Observation:** Timestamped, confidence-bearing evidence produced by a sensor or perception module.
- **World model:** The current stable interpretation of people, objects, engagement, audio activity, and system health.
- **Intent:** A semantic behavior request such as acknowledge, listen, or seek attention.
- **Timeline:** Synchronized motion, light, and sound keyframes produced from an intent.
- **Track:** A session-scoped anonymous identity assigned to a visible person or object.
- **Memory:** Append-only observations plus derived last-known state used for deterministic recall.
- **Scene-relative location:** Human-readable horizontal region, apparent depth band, and optional named anchor such as desk or shelf.
- **Replay:** Deterministic reprocessing of recorded inputs or events without requiring live sensors or cloud access.
- **Core scenario:** Engagement, attention seeking, memory formation, and grounded recall.
- **Bonus capability:** Anonymous multi-user interaction, coarse vocal affect, adaptive behavior preferences, or interruption awareness.

## Decision Record

The approved defaults are a Python modular monolith, a React/Three.js dashboard, local authoritative state and memory, a cloud conversation provider behind a port, anonymous session identities, scene-relative localization, and an abstract six-channel motion contract. Physical construction, metric 3D localization, persistent biometrics, and online model retraining are excluded.
