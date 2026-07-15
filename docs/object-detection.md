# Object Detection

## Default State

Object detection is **disabled by default**. The UI will show "Object detection disabled" and no objects will be tracked.

## Enabling Object Detection

Set the following environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_OBJECT_DETECTION` | `false` | Set to `true` to enable |
| `OBJECT_DETECTOR_MODEL` | `yolov8n.pt` | Path or model name (auto-downloaded by ultralytics) |
| `OBJECT_DETECTION_CONFIDENCE` | `0.45` | Detection confidence threshold |
| `OBJECT_DETECTION_MAX_FPS` | `8` | Inference throttle |
| `OBJECT_DETECTION_CLASSES` | (none) | Optional comma-separated COCO class IDs to allowlist |

## Behavior

- If the model loads successfully, the UI shows "N objects tracked" when objects are detected, and "0 objects detected" when active but nothing found.
- If the model fails to load, the UI shows "Object detector degraded" with the error detail. The app continues running.
- Object tracks must be stable (5 detections within 1 second, confidence >= 0.55, label agreement >= 75%) before appearing in world state.
- Memory records are written once per stable track (first stabilization only).

## Privacy

Object labels and bounding boxes are stored in memory. Raw frames are not retained after inference.
