# Limitations

## Perception

- Location is scene-relative and monocular. Depth bands are heuristics from bounding-box size, not metric 3D distance.
- Object memory is strongest for curated demo objects and labels. Similar objects can produce ambiguity and should be reported as uncertainty.
- Live engagement quality depends on lighting, camera placement, occlusion, glasses, and model availability.
- Face detection falls back through MediaPipe → OpenCV Haar cascade → heuristic skin-region detection. Each fallback reduces accuracy.
  - MediaPipe (face_landmarker.task model) gives the best gaze and head-pose estimates.
  - OpenCV Haar cascade is a fast CPU detector but lacks blendshape gaze tracking.
  - **Heuristic skin-region fallback is a last-resort safety net, not a reliable detector.** It uses combined BGR + approximate YCrCb skin-color thresholding and morphological cleanup, but has no real face geometry, no landmark tracking, and no gaze estimation.
  - Heuristic fallback always reports `status=degraded`. It cannot trigger high-confidence engagement transitions on its own.
  - Heuristic gaze quality is set below the usable threshold, so gaze signals are always `None` when using the fallback.
  - Heuristic confidence is capped at 0.45 and engagement confidence at 0.35, preventing accidental ENGAGED state transitions.
  - False-positive rejection includes: aspect-ratio validation (0.35–1.6), area bounds (1.5%–30% of frame), position checks (upper/middle frame), box-density threshold (>20%), and largest-connected-component isolation.
  - Use `FACE_DETECTOR_MODE=mediapipe` to force MediaPipe and get a clear degradation report if it fails.
  - Use `FACE_DETECTOR_MODE=opencv` or `heuristic` to test lower-quality paths.
  - Common MediaPipe failures: missing `face_landmarker.task` model file, unsupported Python/platform, or import errors.
- The active detector is reported in world health (`face_detector` component) and in `/api/vision/frame` response (`vision_status.face_detector`).
- The `/api/vision/frame` response also includes `vision_status.object_detector` with the object detector's health.
- The dashboard Perception panel shows status badges ("Active", "Degraded", "Disabled") for both face and object detectors. The Device panel overlay shows the active gaze source, pose source, and yaw/pitch angles when available.
- `FACE_DETECTION_MAX_FACES` controls the MediaPipe maximum face count. The default is 4 and valid values are 1 through 8.
- Multiple visible faces are tracked as anonymous session-local person IDs (`person-1`, `person-2`, etc.). Tracking is spatial/temporal only, based on face bounding-box overlap between frames.
- `primary_person_id` selects one active person for behaviors that still need a single target. Selection favors stronger engagement, then larger face area, and avoids rapid switching.
- Engagement calibration improves robustness for the current camera/person setup but is still sensitive to large posture changes, lighting changes, occlusion, and primary-person switches. It is session-local and falls back to default thresholds when unavailable, failed, cancelled, or not applicable.
- The heuristic fallback is not suitable for evaluation claims and exists only to prevent the demo from failing completely when proper models are unavailable.

## Identity and speakers

- Person identity is session-only. The system should not claim durable identity across restarts or separate sessions.
- Person IDs are anonymous runtime labels only. They are not face recognition IDs and must not be interpreted as real identity.
- Active-speaker association is probabilistic and may be anonymous when visual/audio evidence is insufficient.
- Affect evidence is coarse, bounded, and confidence-gated; it is not an emotion detector.

## Conversation

- The deterministic template provider answers a small set of memory-recall questions.
- Natural voice/cloud conversation is optional and depends on external service availability, latency, and configuration.
- Every factual recall should either cite evidence IDs or state explicit uncertainty.

## Simulator and hardware

- The 3D lamp is a simulator. It proves behavior timelines and adapter boundaries, not physical motor torque, calibration, or safety.
- WebGL or browser audio restrictions can degrade the visible demo, but backend replay, memory, text recall, and reports remain available.

## Evaluation

- Public fixtures are deterministic evidence for the software journey. They do not replace a larger labeled sensor dataset.
- Sample-only reports cannot prove the final release gates by themselves.
- Live results vary by hardware, camera, microphone, OS permissions, and room conditions.

## Privacy

- Local runtime data can contain sensitive context even when raw media is not retained.
- Private snapshots, local databases, raw media, `.env`, model weights, and generated private reports must remain untracked.
