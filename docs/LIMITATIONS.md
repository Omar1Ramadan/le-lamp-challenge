# Limitations

## Perception

- Location is scene-relative and monocular. Depth bands are heuristics from bounding-box size, not metric 3D distance.
- Object memory is strongest for curated demo objects and labels. Similar objects can produce ambiguity and should be reported as uncertainty.
- Live engagement quality depends on lighting, camera placement, occlusion, glasses, and model availability.
- Face detection falls back through MediaPipe → OpenCV Haar cascade → heuristic skin-color region detection. Each fallback reduces accuracy.
  - MediaPipe (face_landmarker.task model) gives the best gaze and head-pose estimates.
  - OpenCV Haar cascade is a fast CPU detector but lacks blendshape gaze tracking.
  - Heuristic skin-region fallback is a coarse pixel-based detector with no real face geometry.
  - Use `FACE_DETECTOR_MODE=mediapipe` to force MediaPipe and get a clear degradation report if it fails.
  - Use `FACE_DETECTOR_MODE=opencv` or `heuristic` to test lower-quality paths.
  - Common MediaPipe failures: missing `face_landmarker.task` model file, unsupported Python/platform, or import errors.
- The active detector is reported in world health (`face_detector` component) and in `/api/vision/frame` response (`vision_status.face_detector`).

## Identity and speakers

- Person identity is session-only. The system should not claim durable identity across restarts or separate sessions.
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
