from social_lamp.perception.faces import FaceResult
from social_lamp.perception.tracker import PersonTracker


def _face(bbox: tuple[float, float, float, float], confidence: float = 0.9) -> FaceResult:
    return FaceResult(
        face_confidence=confidence,
        yaw_degrees=0.0,
        pitch_degrees=0.0,
        gaze_score=0.8,
        gaze_quality=0.9,
        face_area_ratio=bbox[2] * bbox[3],
        bbox=bbox,
        pose_source="mediapipe_matrix",
        pose_quality=0.95,
    )


def test_two_faces_create_two_tracks() -> None:
    tracker = PersonTracker()

    tracks = tracker.update(
        [
            _face((0.1, 0.1, 0.2, 0.3)),
            _face((0.6, 0.1, 0.2, 0.3)),
        ],
        mono_ns=1_000_000,
    )

    assert [track.person_id for track in tracks] == ["person-1", "person-2"]
    assert all(track.visible for track in tracks)


def test_nearby_movement_keeps_track_id() -> None:
    tracker = PersonTracker(iou_threshold=0.3)
    first = tracker.update([_face((0.1, 0.1, 0.2, 0.3))], mono_ns=1_000_000)

    moved = tracker.update([_face((0.12, 0.11, 0.2, 0.3))], mono_ns=2_000_000)

    assert moved[0].person_id == first[0].person_id
    assert moved[0].visible is True
    assert moved[0].missed_frames == 0


def test_lost_track_stays_invisible_until_expiry() -> None:
    tracker = PersonTracker(track_expire_ns=10_000_000)
    tracker.update([_face((0.1, 0.1, 0.2, 0.3))], mono_ns=1_000_000)

    tracks = tracker.update([], mono_ns=2_000_000)

    assert len(tracks) == 1
    assert tracks[0].person_id == "person-1"
    assert tracks[0].visible is False
    assert tracks[0].missed_frames == 1


def test_lost_track_expires_after_timeout() -> None:
    tracker = PersonTracker(track_expire_ns=5_000_000)
    tracker.update([_face((0.1, 0.1, 0.2, 0.3))], mono_ns=1_000_000)

    tracks = tracker.update([], mono_ns=7_000_000)

    assert tracks == []


def test_track_reappears_within_timeout_keeps_id() -> None:
    tracker = PersonTracker(track_expire_ns=10_000_000)
    face = _face((0.1, 0.1, 0.2, 0.3))
    tracker.update([face], mono_ns=1_000_000)
    tracker.update([], mono_ns=2_000_000)

    tracks = tracker.update([face], mono_ns=3_000_000)

    assert len(tracks) == 1
    assert tracks[0].person_id == "person-1"
    assert tracks[0].visible is True
    assert tracks[0].missed_frames == 0


def test_detection_order_controls_visible_track_order() -> None:
    tracker = PersonTracker(iou_threshold=0.3)
    left = _face((0.1, 0.1, 0.2, 0.3))
    right = _face((0.6, 0.1, 0.2, 0.3))
    tracker.update([left, right], mono_ns=1_000_000)

    tracks = tracker.update([right, left], mono_ns=2_000_000)

    assert [track.person_id for track in tracks] == ["person-2", "person-1"]
