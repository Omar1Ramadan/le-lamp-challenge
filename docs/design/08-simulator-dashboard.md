# Simulator and Dashboard

## Responsibility

Provide the portfolio-facing 3D lamp, explain the system's real-time decisions, expose memory and evaluation evidence, support live and replay operation, and surface degraded components without becoming a second source of domain truth.

## Technology and Runtime

- React with TypeScript and Vite.
- React Three Fiber and Drei for the articulated scene.
- A generated TypeScript contract package derived from backend JSON Schema.
- One WebSocket for snapshots, timelines, metrics, traces, and faults.
- HTTP endpoints for session commands, replay selection, text turns, configuration, memory deletion, and exports.

The client reconnects with exponential backoff and requests the latest full snapshot after reconnect. Event sequence gaps trigger resynchronization rather than speculative state.

## Desktop Layout

The default demo view has four regions:

1. **Scene:** large 3D lamp with visible light, animated six-channel rig, current social/audio state, and adapter health.
2. **Perception:** live camera with face, gaze/head direction, person labels, object tracks, regions, confidence, frame age, and privacy indicator.
3. **Evidence timeline:** engagement transitions, behaviors, interruptions, object memories, queries, and faults correlated by color and ID.
4. **Inspector:** tabs for world state, memory cards, live metrics, component health, adaptive preferences, and replay controls.

At narrower widths, scene and perception become stacked while the inspector moves below. The demo target is desktop; mobile is readable but not a primary interaction surface.

## 3D Lamp

The lamp is an original articulated model built from simple geometry, avoiding dependency on the physical LeLamp asset. Its hierarchy maps exactly to `base_yaw`, `shoulder_pitch`, `elbow_pitch`, `wrist_pitch`, `head_yaw`, and `head_pitch`.

Timelines from the server drive interpolation. The client reports timeline receipt, first rendered changed frame, completion, and cancellation. Camera controls are limited during guided demo mode so motion remains legible.

Light color and intensity follow `LightTrack`. Audio plays only through resources approved by the server and respects cancellation. Reduced-motion mode replaces large movement with light cues and minimal pose changes.

## Explainability

- Engagement shows the fused score, confidence, threshold bands, dwell progress, and available signal contributions.
- Every behavior card shows its triggering transition, chosen variant, priority, and any preference influence.
- Every memory card shows label, timestamp, scene-relative location, confidence, and snapshot evidence.
- Every conversational answer has a “show evidence” action.
- Faults identify the component, effect, fallback, and correlation ID.

Diagnostic detail can be hidden for a clean demo view without stopping collection.

## Operator Controls

- Start/end session.
- Choose live camera, sensor replay, or event replay.
- Calibrate engagement.
- Submit a text question and enable/disable microphone input.
- Toggle each bonus capability.
- Request neutral pose or mute.
- Inspect/reset adaptive preferences.
- Clear current-session memory or all memory with confirmation.
- Export trace, metrics, and evaluation summary.

Controls send typed commands and wait for server acknowledgement. The UI never changes authoritative state optimistically for safety-sensitive actions.

## Demo Mode

Guided demo mode presents a four-step progress rail: engagement, attention seeking, memory formation, and recall. Each step completes only from correlated backend evidence. Bonus badges activate when their scenarios are demonstrated. Replay can reset to a known seed so a failed live sensor does not end the presentation.

## Accessibility and Privacy

- All status colors have text/icon equivalents.
- Keyboard operation covers controls and evidence navigation.
- Captions display lamp and user speech.
- Reduced motion and mute are first-class settings.
- Camera/microphone/cloud indicators remain visible whenever active.
- Snapshot deletion and retention status are visible from memory cards.

## Failure and Degradation

- WebSocket disconnect freezes the last view with a prominent disconnected state; it does not animate new behavior locally.
- Camera permission or backend capture failure offers replay selection.
- Missing 3D support falls back to a 2D articulated SVG driven by the same timelines.
- Audio autoplay restriction shows an explicit enable-audio control.
- Missing snapshot renders structured evidence with an unavailable marker.
- Export failure preserves the current session and shows an actionable fault.

## Tests

- Generated contract compatibility against backend schemas.
- Reducer tests for full snapshot, ordered event, sequence gap, and reconnect behavior.
- Rig tests proving each logical channel moves only its expected joint hierarchy.
- Timeline acknowledgement and first-visible-frame reporting.
- UI tests for all world states, memory statuses, faults, and provider fallbacks.
- Accessibility tests for keyboard navigation, labels, contrast, captions, and reduced motion.
- Playwright end-to-end tests for the four-step demo, replay fallback, bonus toggles, memory clearing, and exports.
