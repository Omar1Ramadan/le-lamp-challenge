# Output Compositor and Adapters

## Responsibility

Translate semantic intents into synchronized, safe motion/light/audio timelines; arbitrate active output; and execute through a capability-aware adapter that can represent the browser simulator now and physical hardware later.

## Abstract Six-Channel Motion Model

The logical channels are normalized to `[-1, 1]`:

1. `base_yaw`
2. `shoulder_pitch`
3. `elbow_pitch`
4. `wrist_pitch`
5. `head_yaw`
6. `head_pitch`

Zero is the neutral authored pose, not necessarily a physical servo midpoint. Each channel has configured position, velocity, and acceleration limits. The simulator implements all six. A hardware adapter maps supported logical channels to calibrated joints and reports omissions or coupled mappings through capabilities.

## Keyframe Tracks

```text
MotionKeyframe(offset_ms, value, easing)
MotionTrack(channel, keyframes)
LightKeyframe(offset_ms, rgb, brightness, easing)
LightTrack(keyframes, optional pattern)
AudioTrack(kind: clip | speech, resource_id, start_offset_ms, gain)
```

Offsets begin at zero, are strictly increasing per track, and cannot exceed timeline duration. Supported easing is `linear`, `ease_in`, `ease_out`, `ease_in_out`, or `spring_soft`; physical adapters may approximate unsupported easing while reporting the mapping.

## Compositor

The compositor owns one active timeline and at most one pending timeline. It:

- Resolves an intent to a versioned authored variant.
- Applies bounded affect and preference parameters.
- Validates positions, velocities, duration, and resources.
- Blends from the adapter's latest reported pose to avoid discontinuities.
- Cancels or queues according to intent priority.
- Emits command, acknowledgement, first-visible-output, completion, cancellation, and fault events.

Audio interruption is immediate. Motion cancellation uses a 150 ms controlled blend to the new target unless the adapter reports an emergency stop.

## LampAdapter Port

```text
LampCapabilities
  adapter_id, adapter_version
  motion_channels: map[channel, limits]
  supports_light, supports_audio, supports_pose_feedback
  supported_easing, minimum_command_interval_ms

LampHealth
  status: healthy | degraded | unavailable
  last_ack_mono_ns
  pose: optional map[channel, value]
  faults: list[typed fault]

LampAdapter
  async capabilities() -> LampCapabilities
  async execute(timeline: BehaviorTimeline) -> ExecutionHandle
  async cancel(execution_id, reason) -> CancellationResult
  async neutralize(reason) -> ExecutionHandle
  async health() -> LampHealth
```

An adapter must be idempotent for repeated `execute` calls with the same timeline ID and may execute only one timeline at a time.

## Simulator Adapter

The server-side simulator adapter validates timelines and publishes them to the WebSocket gateway. The client acknowledges receipt and reports the first rendered animation frame, completion, and cancellation. If no client is connected, the adapter remains degraded but records intended timelines for replay.

The browser uses a fixed articulated rig with the same logical channel names. Rendering interpolation follows server keyframes; the browser does not reinterpret behavior intent.

## Future Hardware Adapter

Hardware startup loads calibration, queries capabilities, verifies neutral pose, and refuses motion when required calibration is absent. Logical channels may map one-to-one, couple into multiple joints, or be unsupported. Unsupported optional channels are omitted with a structured degradation event. A behavior requiring a missing essential capability is rejected before movement.

The hardware adapter must enforce its own hard joint, velocity, temperature, and communication limits in addition to compositor validation. Connection loss triggers controlled neutralization when possible and an unavailable health state.

## Light and Audio

Light uses linear RGB values plus brightness; adapters perform device-specific color correction. Patterns are expanded into keyframes by the compositor for deterministic replay.

Audio resources are either versioned local clips or speech handles produced by conversation. The adapter never receives arbitrary filesystem paths or unbounded audio. Speech cancellation must stop buffered playback and emit the last played timestamp.

## Failure and Degradation

- Invalid timeline: reject before adapter execution and emit the validation reason.
- Missing optional channel: omit and record mapping degradation.
- Missing required channel/resource: reject the behavior and allow policy fallback.
- Adapter acknowledgement timeout: mark degraded, cancel, and neutralize if available.
- Browser disconnect: stop claiming visible output latency; retain command trace.
- Physical communication loss: stop new commands and surface unavailable state.
- Neutralization failure: persistent highest-severity fault requiring operator acknowledgement.

## Tests

- Timeline schema, ordering, range, duration, and resource validation.
- Priority replacement and controlled cancellation timing.
- Deterministic intent-to-timeline snapshots for every authored variant.
- Capability mapping for full, partial, coupled, and incompatible adapters.
- Adapter idempotency and acknowledgement timeout behavior.
- Simulator first-frame correlation used for reaction latency.
- Fake hardware tests proving limits cannot be exceeded by authored or adjusted behavior.
