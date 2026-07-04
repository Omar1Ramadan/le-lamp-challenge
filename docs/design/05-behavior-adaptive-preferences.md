# Behavior and Adaptive Preferences

## Responsibility

Convert stable world state into semantic, explainable behavior intents; coordinate attention-seeking escalation; and adapt selection among safe authored variants without learning arbitrary motion or modifying perception.

## Authored Behavior Library

| Intent | Default expression | Purpose |
| --- | --- | --- |
| `acknowledge` | Small orienting tilt and warm light pulse | Confirm engagement |
| `listen` | Face primary person, steady low light | Signal attention without interrupting |
| `disengage` | Slow release toward neutral | Avoid an abrupt social cutoff |
| `seek_attention` | Subtle motion, then light, then optional sound | Attempt re-engagement |
| `think` | Low-amplitude cyclic head motion | Cover grounded query latency |
| `recall_success` | Small nod and brief brightening | Confirm evidence found |
| `recall_unknown` | Gentle asymmetric tilt and dim pulse | Communicate uncertainty |
| `express_affect` | Bounded tempo/amplitude adjustment | Reflect coarse vocal affect |
| `return_neutral` | Smooth neutral pose and idle light | Restore baseline |
| `fault` | Safe neutralization and visible status light | Signal degraded output |

Each intent has named variants with fixed safe ranges. The policy chooses a variant; the compositor owns exact keyframes.

## Policy Inputs

- Current and previous `WorldSnapshot`.
- Stable state-transition events.
- Audio mode, interruption suppression, and coarse affect hints.
- Recent behavior outcomes and cooldown ledger.
- Feature flags and operator overrides.
- Session-scoped preference scores.

The policy is deterministic in replay mode. Equal candidates use a fixed lexical tie-break.

## Priority and Arbitration

| Priority | Intent class |
| ---: | --- |
| 100 | Emergency neutralization or adapter fault |
| 90 | User interruption and listening |
| 80 | Direct response acknowledgement |
| 60 | Engagement/disengagement transition |
| 40 | Attention seeking |
| 20 | Ambient affect or idle expression |

Higher-priority intents may cancel a lower-priority cancellable timeline. An equal-priority intent replaces only when newer and materially different. Expired intents are discarded before composition.

## Attention-Seeking Schedule

Seeking begins after five seconds of stable disengagement. It has at most three attempts:

1. Small motion only.
2. Motion plus light after a ten-second cooldown.
3. Motion, light, and a short nonverbal sound after a twenty-second cooldown.

Any directed speech, confident background-media condition, active lamp speech, operator suppression, or renewed engagement cancels the schedule. After exhaustion, the policy returns to idle for at least 60 seconds before a new schedule may begin.

## Interruption Awareness

When the user starts speaking, the policy immediately emits `listen` at priority 90 and requests cancellation of lamp audio. When speech is likely television or unrelated background media, unsolicited sound is prohibited and attention-seeking motion is limited to the first subtle level. The policy records the reason for every suppression decision.

## Coarse Affect Use

Accepted affect hints may adjust only tempo, light warmth, and motion amplitude within authored limits:

- High arousal reduces attention-seeking intensity and increases listening stillness.
- Low arousal permits slower, softer motion.
- Negative-valence tendency avoids playful or celebratory variants.
- Low-confidence affect has no effect.

The dashboard describes these as signal tendencies, not detected emotions.

## Adaptive Preferences Bonus

Preferences rank safe variants per context key such as `seek_attention:quiet` or `acknowledge:engaged`. Each variant starts at score `1.0`, is clamped to `[0.5, 1.5]`, and stores an evidence count.

Outcomes update the chosen variant:

- Re-engagement within three seconds: `+0.10`.
- Explicit positive feedback: `+0.20`.
- Explicit rejection or immediate mute: `-0.25`.
- No response through the outcome window: `-0.05`.

Other variants are unchanged. Scores decay 5% toward `1.0` at each new session so early noise cannot dominate permanently. Live mode selects the highest eligible score; every fifth eligible selection may choose the next-best underrepresented variant when exploration is enabled. Replay and evaluation disable exploration.

The dashboard provides reset and inspection controls. Preference changes are persisted with triggering outcome, previous score, new score, and correlation ID. The learner cannot create behaviors, exceed safety bounds, change escalation count, or modify interruption rules.

## Failure and Degradation

- Missing world updates: emit no new social intent and allow current timeline to complete or expire.
- Invalid preference data: ignore the row, use score `1.0`, and emit a fault.
- Compositor unavailable: retain intent metrics but do not retry stale expressive actions after recovery.
- Conflicting signals: interruption and safety priorities win; uncertainty results in less movement and no unsolicited sound.
- Bonus failure: disable only that bonus and continue the core deterministic policy.

## Tests

- Every stable social transition maps to the expected intent.
- Priority, replacement, expiry, cooldown, and cancellation boundaries.
- Complete three-step attention schedule and every suppression condition.
- Affect confidence and range constraints.
- Preference update arithmetic, clamping, decay, exploration gating, reset, and audit history.
- Replay determinism with identical snapshots and configuration.
- Property test proving no policy output can reference an unknown authored behavior.

## Reference

- [ELEGNT: Expressive and Functional Movement Design for Non-Anthropomorphic Robot](https://machinelearning.apple.com/research/elegnt-expressive-functional-movement)
