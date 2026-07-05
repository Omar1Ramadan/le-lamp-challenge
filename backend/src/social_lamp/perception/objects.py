from collections import Counter, OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

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


class ObjectDetectorModel(Protocol):
    def predict(self, image: NDArray[np.uint8]) -> list[Detection]: ...


class FastObjectDetector:
    def __init__(self, *, model: ObjectDetectorModel) -> None:
        self._model = model

    def detect(self, image: NDArray[np.uint8]) -> tuple[Detection, ...]:
        return tuple(self._model.predict(image))


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
