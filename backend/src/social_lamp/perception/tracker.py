from __future__ import annotations

from dataclasses import dataclass

from social_lamp.perception.faces import FaceResult

BoundingBox = tuple[float, float, float, float]


@dataclass
class PersonTrack:
    person_id: str
    last_seen_mono_ns: int
    bbox: BoundingBox
    face_confidence: float
    visible: bool = True
    missed_frames: int = 0
    last_face: FaceResult | None = None


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return intersection / union


class PersonTracker:
    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        track_expire_ns: int = 1_500_000_000,
        max_tracks: int = 8,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._track_expire_ns = track_expire_ns
        self._max_tracks = max_tracks
        self._tracks: dict[str, PersonTrack] = {}
        self._next_id = 1

    @property
    def active_tracks(self) -> list[PersonTrack]:
        return list(self._tracks.values())

    def update(self, detections: list[FaceResult], mono_ns: int) -> list[PersonTrack]:
        matched_track_ids: set[str] = set()
        visible_order: list[PersonTrack] = []

        for detection in detections:
            match = self._best_match(detection, matched_track_ids)
            if match is None:
                track = self._create_track(detection, mono_ns)
            else:
                track = self._tracks[match]
                track.last_seen_mono_ns = mono_ns
                track.bbox = detection.bbox
                track.face_confidence = detection.face_confidence
                track.visible = True
                track.missed_frames = 0
                track.last_face = detection
            matched_track_ids.add(track.person_id)
            visible_order.append(track)

        lost_tracks: list[PersonTrack] = []
        for person_id in list(self._tracks):
            if person_id in matched_track_ids:
                continue
            track = self._tracks[person_id]
            if mono_ns - track.last_seen_mono_ns >= self._track_expire_ns:
                del self._tracks[person_id]
                continue
            track.visible = False
            track.missed_frames += 1
            lost_tracks.append(track)

        return visible_order + lost_tracks

    def _best_match(self, detection: FaceResult, matched_track_ids: set[str]) -> str | None:
        best_id: str | None = None
        best_iou = self._iou_threshold
        for person_id, track in self._tracks.items():
            if person_id in matched_track_ids:
                continue
            score = _iou(detection.bbox, track.bbox)
            if score > best_iou:
                best_id = person_id
                best_iou = score
        return best_id

    def _create_track(self, detection: FaceResult, mono_ns: int) -> PersonTrack:
        if len(self._tracks) >= self._max_tracks:
            oldest_id = min(
                self._tracks,
                key=lambda person_id: self._tracks[person_id].last_seen_mono_ns,
            )
            del self._tracks[oldest_id]

        person_id = f"person-{self._next_id}"
        self._next_id += 1
        if self._next_id > self._max_tracks:
            self._next_id = 1

        track = PersonTrack(
            person_id=person_id,
            last_seen_mono_ns=mono_ns,
            bbox=detection.bbox,
            face_confidence=detection.face_confidence,
            visible=True,
            missed_frames=0,
            last_face=detection,
        )
        self._tracks[person_id] = track
        return track
