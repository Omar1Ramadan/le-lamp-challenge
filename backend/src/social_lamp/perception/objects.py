from collections import Counter, OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic_ns
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from social_lamp.domain.contracts import ComponentHealth
from social_lamp.perception.location import BBox


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: BBox
    mono_ns: int


@dataclass
class ObjectTrack:
    track_id: str
    detections: deque[Detection] = field(default_factory=lambda: deque(maxlen=20))

    def add(self, detection: Detection) -> None:
        self.detections.append(detection)

    @property
    def label(self) -> str:
        return Counter(item.label for item in self.detections).most_common(1)[0][0]

    @property
    def is_stable(self) -> bool:
        if len(self.detections) < 5:
            return False
        recent = list(self.detections)[-5:]
        if recent[-1].mono_ns - recent[0].mono_ns > 1_000_000_000:
            return False
        if sum(item.confidence for item in recent) / len(recent) < 0.55:
            return False
        count = Counter(item.label for item in recent).most_common(1)[0][1]
        return count / len(recent) >= 0.75


class NullObjectDetector:
    def detect(self, image: object) -> tuple[Detection, ...]:
        del image
        return ()

    def health(self) -> ComponentHealth:
        return ComponentHealth(
            component="object_detector",
            status="disabled",
            detail="Object detection model is not configured",
        )


class ObjectDetectorModel(Protocol):
    def predict(self, image: NDArray[np.uint8]) -> list[Detection]: ...


class FastObjectDetector:
    def __init__(self, *, model: ObjectDetectorModel) -> None:
        self._model = model

    def detect(self, image: NDArray[np.uint8]) -> tuple[Detection, ...]:
        return tuple(self._model.predict(image))

    def health(self) -> ComponentHealth:
        return ComponentHealth(
            component="object_detector",
            status="active",
        )


class YoloObjectDetector:
    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        confidence: float = 0.45,
        classes: list[int] | None = None,
    ) -> None:
        self._model_path = str(model_path)
        self._confidence = confidence
        self._classes = classes
        self._model: Any = None
        self._load_error: str | None = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO

            self._model = YOLO(self._model_path)
        except Exception as exc:
            self._load_error = f"Object detection model load failed: {exc}"
            self._model = None

    def health(self) -> ComponentHealth:
        if self._load_error is not None:
            return ComponentHealth(
                component="object_detector",
                status="degraded",
                detail=self._load_error,
            )
        return ComponentHealth(
            component="object_detector",
            status="active",
        )

    def detect(self, image: NDArray[np.uint8]) -> tuple[Detection, ...]:
        if self._model is None:
            return ()
        try:
            results = self._model(
                image,
                conf=self._confidence,
                classes=self._classes,
                verbose=False,
            )
            now = monotonic_ns()
            detections: list[Detection] = []
            for result in results:
                if result.boxes is None:
                    continue
                for box, conf, cls_id in zip(
                    result.boxes.xyxyn,
                    result.boxes.conf,
                    result.boxes.cls,
                ):
                    x1, y1, x2, y2 = box.tolist()
                    label = result.names[int(cls_id)]
                    detections.append(
                        Detection(
                            label=label,
                            confidence=float(conf),
                            bbox=(x1, y1, x2, y2),
                            mono_ns=now,
                        )
                    )
            return tuple(detections)
        except Exception:
            return ()


class EnrichmentQueue:
    def __init__(self, *, capacity: int = 2) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        self._items: OrderedDict[str, Mapping[str, object]] = OrderedDict()

    def put(self, track_id: str, request: Mapping[str, object]) -> None:
        if track_id in self._items:
            del self._items[track_id]
        elif len(self._items) >= self._capacity:
            self._items.popitem(last=False)
        self._items[track_id] = dict(request)

    def pop(self) -> tuple[str, Mapping[str, object]] | None:
        if not self._items:
            return None
        return self._items.popitem(last=False)
