# Events, World Model, and Observability

## Responsibility

This subsystem defines the contracts connecting all modules, owns stable runtime state through a single writer, prevents stale-work accumulation, and produces replayable evidence for debugging and evaluation.

## Time and Identity

- UUIDv7 identifiers are used for events, correlations, sessions, tracks, intents, timelines, and queries.
- `captured_at_mono_ns` and `emitted_at_mono_ns` use one process monotonic clock for ordering and latency.
- `wall_time_utc` is an RFC 3339 UTC string used for display and persisted history, never latency arithmetic.
- Every event includes `schema_version`, initially `1.0`.
- A correlation begins at a sensor frame, user command, or conversation turn and survives through resulting state, intent, timeline, adapter, and metric events.

## ObservationEvent

```text
ObservationEvent
  schema_version: "1.0"
  event_id: UUIDv7
  correlation_id: UUIDv7
  session_id: UUIDv7
  source: capture | face | gaze | audio | object | enrichment | operator | system
  kind: discriminated payload name
  captured_at_mono_ns: integer
  emitted_at_mono_ns: integer
  wall_time_utc: timestamp
  confidence: float [0, 1]
  frame_ref: optional opaque frame identifier
  payload: kind-specific typed object
```

Pixels and audio buffers do not travel through the event bus. `frame_ref` addresses an immutable item in a bounded ring buffer. Consumers must tolerate a reference expiring and emit a metric rather than block capture.

## WorldSnapshot

```text
WorldSnapshot
  schema_version: "1.0"
  snapshot_id: UUIDv7
  revision: monotonically increasing integer
  session_id: UUIDv7
  as_of_mono_ns: integer
  social_state: idle | candidate | engaged | disengaged | seeking_attention
  audio_mode: silent | listening | thinking | speaking
  primary_person_id: optional session track ID
  people: list[PersonState]
  objects: list[ObjectState]
  audio: AudioState
  health: list[ComponentHealth]
```

The world-model task is the only writer. Other modules consume immutable snapshots. It keeps short observation windows, applies freshness limits, and increments `revision` only when stable state changes.

## Entity Lifetimes

- Person tracks are anonymous and expire after 10 seconds without supporting evidence.
- Object tracks become inactive after 5 seconds without evidence but remain queryable through memory.
- Audio evidence expires after 500 ms unless a longer utterance state is maintained.
- Component health expires into `unknown` after three expected heartbeat intervals.
- A session begins when live or replay capture starts and ends explicitly or after 15 minutes without sensor or operator activity.

## Behavior Contracts

```text
BehaviorIntent
  intent_id, correlation_id, session_id
  kind: acknowledge | listen | disengage | seek_attention | think | recall_success |
        recall_unknown | return_neutral | express_affect | fault
  target_person_id: optional
  urgency: integer [0, 100]
  created_at_mono_ns, expires_at_mono_ns
  parameters: typed per-kind values

BehaviorTimeline
  timeline_id, intent_id, correlation_id
  priority: integer [0, 100]
  duration_ms: positive integer
  cancellable: boolean
  motion_tracks: list[MotionTrack]
  light_track: optional LightTrack
  audio_track: optional AudioTrack
```

## Queue and Backpressure Policy

| Path | Capacity | Overflow behavior |
| --- | ---: | --- |
| Camera frame ring | 3 | Replace oldest; inference reads newest |
| Raw audio chunks | 50 x 20 ms | Drop oldest and emit overrun metric |
| Observation bus per consumer | 32 | Drop oldest non-fault event |
| Memory write queue | 256 | Reject new noncritical refresh; never drop first sighting or movement change silently |
| Slow enrichment queue | 2 | Coalesce requests by object track |
| Conversation turn queue | 8 | Reject with busy response after capacity |
| Adapter command queue | 1 active + 1 pending | Priority replacement or explicit rejection |

Fault, state-transition, cancellation, and persistence-failure events are never intentionally discarded. If their queue cannot accept data, the affected component enters degraded health and the session trace records a synthetic overflow fault.

## ReplayTrace

A trace is UTF-8 JSON Lines with a manifest. The manifest records schema version, application version, configuration hash, model identifiers, source media checksums, session ID, and monotonic origin. Each line is one typed event or expected output in monotonic order.

```text
ReplayTraceManifest
  schema_version, application_version, session_id
  configuration_hash, model_identifiers
  source_media_checksums, monotonic_origin_ns
  created_at_utc

ReplayTraceRecord
  sequence: monotonically increasing integer
  record_type: observation | snapshot | intent | timeline | adapter | query | metric | fault
  recorded_at_mono_ns
  body: corresponding versioned contract
```

Two replay modes are required:

- **Sensor replay:** feed recorded media and labels through perception for accuracy and performance measurement.
- **Event replay:** bypass models and feed observations into world state, behavior, memory, and UI for deterministic integration tests.

Cloud responses and adapter acknowledgements are recorded as fixtures in deterministic replay. Live secrets and raw authentication data are excluded.

## Metrics

- Counters: frames captured/dropped, observations dropped, transitions, false transitions, memories written, query outcomes, faults, interruptions.
- Gauges: queue depth, frame age, active tracks, engagement score, component health.
- Histograms: capture-to-observation, observation-to-state, state-to-intent, intent-to-adapter, query-to-evidence, speech-end-to-first-output.
- Structured logs include event ID, correlation ID, session ID, component, severity, and safe details.

Metrics are available in process for the dashboard and exportable as JSON. Raw image data, transcripts, and secrets are not metric labels.

## Operator Commands

The API accepts typed commands to start/end a session, select live or replay input, submit text, enable a bonus flag, clear memory, and request neutralization. Commands are authenticated only by local-origin restriction in the first release. Non-local binding requires an explicit configuration change and token authentication.

## Failure Behavior

- Missing or expired frame references skip that inference and increment a counter.
- An invalid event is rejected at the producing boundary and emits a schema fault.
- A world-model exception preserves the last valid snapshot, marks health degraded, and restarts the task with a bounded backoff.
- Trace write failure disables recording but not control, surfaces a persistent dashboard fault, and never pretends evaluation evidence exists.
- Replay rejects incompatible major schema versions and reports the exact supported range.

## Tests

- Round-trip serialization for every event payload.
- Rejection of invalid confidence, timestamps, enums, and incompatible versions.
- Deterministic world snapshots from identical event traces.
- Queue overflow tests proving newest-frame and never-silently-drop-fault policies.
- Correlation continuity from observation through adapter acknowledgement.
- Replay checksum and ordering validation.
