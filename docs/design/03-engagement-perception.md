# Engagement Perception

## Responsibility

Estimate whether one or more visible people are attending to the lamp, stabilize noisy evidence over time, select an anonymous primary person, and emit evidence suitable for reliability evaluation. This subsystem does not choose lamp behavior.

## Inputs and Outputs

Inputs are current camera frame references, audio activity observations, calibration settings, and active session tracks. Outputs are face detections, head pose, gaze tendency, apparent proximity, directed-speech evidence, anonymous person tracks, fused engagement samples, and stable transition observations.

## Visual Pipeline

1. MediaPipe Face Landmarker detects faces and landmarks in live-stream mode.
2. Landmark geometry estimates head yaw, pitch, and roll. OpenCV PnP is the fallback or validation implementation.
3. Eye blendshapes and iris/eye landmarks estimate a coarse toward/away gaze tendency.
4. Face area and vertical position provide an apparent-proximity signal, not metric distance.
5. A lightweight tracker associates detections across frames using position, scale, and landmark similarity. It creates anonymous `person-<session suffix>-<counter>` IDs.

No face embeddings are persisted. Tracks reset at session end.

## Per-Session Calibration

Calibration is optional but recommended. During a three-second prompt, the user looks toward the camera from their intended position. The system records neutral head pose, typical face scale, and usable eye-signal variance. If calibration is skipped, conservative defaults apply. A calibration is valid only for the current session and camera configuration.

The runtime exposes calibration as `uncalibrated`, `calibrating`, `calibrated`, `failed`, or `expired`. Calibration applies only to the current anonymous `primary_person_id`; if the primary person changes, scoring must not claim identity transfer. Raw neutral baselines stay backend-internal and are not persisted by default.

When calibrated, head-pose scoring uses yaw/pitch offsets from the neutral baseline, proximity uses the calibrated face-scale ratio, and gaze uses the calibrated baseline when available. If gaze calibration is unavailable, pose and scale calibration can still operate in partial mode. Uncalibrated, failed, or cancelled calibration uses the existing fallback thresholds.

The UI shows whether head, gaze, proximity, and audio signals are available. Unavailable signals are removed and remaining weights are normalized rather than treated as negative evidence.

## Fused Score

Each person receives a score in `[0, 1]`:

```text
0.20 * face_presence
+ 0.30 * head_toward_lamp
+ 0.25 * gaze_toward_lamp
+ 0.10 * proximity_quality
+ 0.15 * directed_speech
```

Defaults are configurable but versioned in replay manifests. Visual samples are smoothed with an exponential moving average using a 250 ms time constant. Directed speech boosts only the visually plausible speaker or, when association is unavailable, the current primary person at reduced weight.

## State Thresholds

- Enter `candidate` when the primary score is at least `0.45` for 200 ms.
- Enter `engaged` when it is at least `0.68` for 700 ms.
- Return `candidate -> idle` below `0.35` for 500 ms.
- Enter `disengaged` below `0.38` for 1,200 ms after engagement.
- Return `disengaged -> engaged` at or above `0.62` for 500 ms.
- Enter `seeking_attention` after five seconds of continuous disengagement when interruption suppression is not active.

Threshold comparisons use monotonic duration, not frame count. Brief blinks, glances, and dropped frames therefore do not cause transitions.

## Primary Person and Multi-User Bonus

The primary person is the eligible track with the highest smoothed engagement score plus a small continuity bonus for the existing primary. A new track must exceed the current primary by `0.15` for one second before replacing it. The active speaker overrides this only while directed speech is confidently associated.

The dashboard labels people as Person A, Person B, and so on. These labels are session-local and are not claimed as real identity.

## Directed Speech Evidence

Voice activity detection finds speech intervals. Active-speaker association combines mouth-motion correlation, visible-person plausibility, and recent primary-person continuity. With one visible person, speech may be associated when visual plausibility is adequate. With multiple people and insufficient evidence, the speaker remains unknown rather than being guessed.

## Confidence and Uncertainty

The fused confidence reflects input availability, landmark confidence, calibration quality, and cross-signal agreement. A high score with confidence below `0.45` cannot create a stable engagement transition. The world snapshot exposes both score and confidence so the dashboard can explain decisions.

## Failure and Degradation

- No camera: engagement is unknown; replay and explicit text controls remain available.
- Multiple overlapping faces: retain existing tracks when possible and lower association confidence.
- Eyes obscured or glasses glare: remove gaze weight and rely on head pose, face, proximity, and speech.
- Low light or motion blur: lower visual confidence; do not infer disengagement immediately.
- Stale frames older than 300 ms: skip inference and emit freshness metrics.
- Model exception: mark the component degraded, preserve the last stable state until evidence expires, and retry with bounded backoff.

## Evaluation Labels

Replay clips have 100 ms interval labels: `engaged`, `not_engaged`, or `ambiguous`, plus person track, directed speaker, lighting, glasses, distance, head-pose condition, and occlusion tags. Ambiguous intervals are excluded from headline F1 but reported separately.

Transition metrics include precision, recall, F1, false engagements per minute, missed disengagements, entry delay, exit delay, and primary-person switch accuracy. The release gate is F1 at least 0.85 and at most one false engagement transition per two minutes across the core replay suite.

## Tests

- Score normalization when signals are unavailable.
- Exact dwell and hysteresis boundary behavior using a fake monotonic clock.
- Blink, brief glance, dropped-frame, and low-confidence sequences.
- Primary-person continuity and controlled switching.
- Session reset proving anonymous identity is not persisted.
- Recorded replay tests across lighting, glasses, distance, profile, multi-user, and background-speech conditions.

## References

- [MediaPipe Face Landmarker for Python](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)
- [OpenCV PnP pose computation](https://docs.opencv.org/master/d5/d1f/calib3d_solvePnP.html)
