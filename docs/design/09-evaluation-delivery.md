# Evaluation and Delivery

## Responsibility

Define reproducible datasets, labels, metrics, acceptance gates, automated tests, demonstration evidence, and the final technical writeup. A claim is included in the portfolio only when its trace and metric inputs can be reproduced.

## Evaluation Modes

- **Unit fixtures:** synthetic scores, events, memory rows, intents, and adapter responses.
- **Event replay:** recorded observations exercise world state, policy, memory, conversation, compositor, and dashboard deterministically.
- **Sensor replay:** labeled video/audio exercises actual perception models.
- **Live acceptance:** webcam/microphone sessions validate permissions, timing, cloud degradation, and human-visible behavior.

Cloud responses are replaced by recorded or deterministic fakes for automated gates. A separate live-cloud smoke test is informational because external latency is not controlled.

## Dataset Structure

```text
evaluation/
  manifest.json
  scenarios/<scenario-id>/
    media.mp4
    audio.wav                 # only when explicitly recorded
    labels.jsonl
    expected-memories.json
    expected-queries.json
    metadata.json
```

The manifest records dataset version and checksums. Scenario metadata includes lighting, distance, glasses, occlusion, number of people, background audio, object set, and consent/recording provenance. Private evaluation media is excluded from public source control unless intentionally released.

## Engagement Labels and Metrics

Labels are sampled at 100 ms intervals as `engaged`, `not_engaged`, or `ambiguous`, with primary person and active speaker when known. Ambiguous intervals are excluded from headline classification metrics and reported separately.

Required metrics:

- Precision, recall, and F1 for engaged intervals.
- False engagement transitions per minute.
- Missed disengagement transitions.
- Median and p95 entry/exit delay against labeled transitions.
- Primary-person and active-speaker association accuracy.
- Results stratified by lighting, distance, glasses, profile, and multi-user condition.

Release gates are engagement F1 at least `0.85` and no more than one false engagement transition per two minutes across the core suite.

## Object and Memory Metrics

- Stable-track precision and recall on the curated object set.
- Canonical label accuracy for fast and enriched labels separately.
- Horizontal-region, depth-band, and named-anchor accuracy.
- Last-seen query accuracy after objects remain, move, disappear, and reappear.
- Ambiguity accuracy for visually similar or aliased objects.
- Grounding rate: proportion of factual answers whose fields exactly match selected evidence.

The core gate is at least `0.90` correct last-seen object plus defensible scene-relative location on scripted queries. Grounding must be `1.00`; a correct explicit uncertainty response passes when evidence is absent.

## Latency Metrics

All latency uses monotonic timestamps and correlated events:

- Frame capture to perception observation: p50, p95, maximum.
- Observation to stable state transition, excluding configured dwell and including it separately.
- Stable state to intent.
- Intent to adapter acknowledgement.
- Stable state to first visible simulator frame.
- Speech end to transcript finalization, evidence result, first response text, and first audible output.
- Memory query to evidence result.

Core gates:

- p95 frame-to-observation at most `200 ms` on the documented development machine.
- p95 stable-state-to-first-visible-response at most `150 ms`.
- Normally processed frame age below `300 ms`.

The report names CPU, GPU if available, memory, camera resolution/frame rate, model versions, and whether acceleration was active.

## Reliability and Degradation Matrix

Automated or manual scenarios cover:

- Camera absent, permission denied, disconnect, low light, and stale frames.
- Microphone absent, background conversation, television, music, and user interruption.
- Detector/enrichment failure and overload.
- Database lock, snapshot failure, and retention cleanup.
- Cloud unconfigured, timeout, disconnect, invalid output, and grounding rejection.
- Browser disconnect, reconnect, missing WebGL, and audio autoplay restriction.
- Adapter partial capabilities, acknowledgement timeout, cancellation, and neutralization failure.

Core simulation must remain operable with network access disabled. Each scenario has an expected health state, visible fault, fallback, and trace event.

## Automated Test Gates

- Backend: pytest unit, contract, property, database, replay, and API tests.
- Frontend: Vitest reducer/component/rig tests and accessibility checks.
- End to end: Playwright against event-replay fixtures.
- Schema: backend JSON Schema generation compared with generated TypeScript types.
- Quality: Ruff formatting/linting, mypy strict checks for domain contracts, ESLint, and TypeScript no-emit checks.
- Evaluation: one command produces machine-readable metrics and fails on core threshold regression.

Model-heavy sensor replay may run in a scheduled or explicitly enabled job; deterministic event replay remains part of every normal test run.

## Four-Step Demo Script

1. **Engagement:** begin idle, look toward the camera, show score/dwell transition, and acknowledge animation. Look away and show stable disengagement.
2. **Attention seeking:** remain disengaged, demonstrate subtle motion followed by light while dashboard explains cooldown and suppression state.
3. **Memory formation:** show a curated object, wait for a stable track, display the stored memory and evidence snapshot, then move or remove it.
4. **Memory recall:** ask where the object was seen, display the query evidence, and hear an answer matching the stored location.

The demo can switch to a deterministic replay at any point without restarting the application.

## Bonus Demo Script

- Two anonymous people appear; the dashboard tracks Person A/B and associates a directed speaker.
- A controlled voice clip produces coarse affect tendencies and visibly bounded behavior modulation.
- Repeated attention-seeking outcomes update and expose a preference score, followed by a reset demonstration.
- The user interrupts lamp speech; audio stops, listening begins, and the trace shows cancellation latency.
- Television/background audio suppresses unsolicited sound without disabling typed recall.

Bonus results are reported separately and cannot compensate for a failed core gate.

## Technical Writeup

The final writeup includes:

- Problem definition and explicit interpretation of engagement, location, and simulated 6-DOF output.
- High- and low-level architecture/data-flow diagrams.
- Module responsibilities and typed contracts.
- Alternatives considered and rationale for the modular monolith, local memory, deterministic policy, cascaded detection, and adapter boundary.
- Evaluation protocol, hardware/software environment, results, limitations, and failure analysis.
- Privacy model and future physical-adapter path.

Diagrams and claims link to the corresponding design section and evaluation artifact. Limitations explicitly include monocular location heuristics, session-only identity, affect uncertainty, curated object coverage, and cloud dependency for natural voice.

## Delivery Artifacts

- Source repository and reproducible setup instructions.
- Approved design set and implementation plan.
- Architecture diagrams and generated contract reference.
- Versioned replay fixtures or public-safe samples.
- Machine-readable and human-readable evaluation reports.
- Demo video showing the required four steps and separate bonus evidence.
- Known-limitations and privacy documentation.

## Acceptance Review

Before recording the final video, run all automated gates, a clean-machine setup test, offline fallback, a full deterministic demo replay, and one live session. The reported commit hash, configuration hash, model identifiers, and evaluation dataset version must match the recorded evidence.
