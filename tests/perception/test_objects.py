from social_lamp.perception.location import locate_box
from social_lamp.perception.objects import Detection, EnrichmentQueue, ObjectTrack


def test_scene_relative_location_uses_normalized_regions() -> None:
    location = locate_box((0.72, 0.60, 0.92, 0.90), anchors={"desk": (0.0, 0.55, 1.0, 1.0)})
    assert location.horizontal_region == "right"
    assert location.anchor_name == "desk"


def test_location_boundaries_and_depth_bands() -> None:
    left = locate_box((0.0, 0.0, 0.30, 0.10), anchors={})
    center = locate_box((0.34, 0.0, 0.66, 0.20), anchors={})
    right = locate_box((0.67, 0.0, 0.80, 0.10), anchors={})
    foreground = locate_box((0.0, 0.0, 0.50, 0.50), anchors={})
    assert left.horizontal_region == "left"
    assert center.horizontal_region == "center"
    assert right.horizontal_region == "right"
    assert right.depth_band == "background"
    assert center.depth_band == "midground"
    assert foreground.depth_band == "foreground"


def test_track_becomes_stable_after_five_consistent_detections() -> None:
    track = ObjectTrack(track_id="object-1")
    for index in range(5):
        track.add(Detection("mug", 0.8, (0.1, 0.1, 0.3, 0.5), index * 200_000_000))
    assert track.is_stable
    assert track.label == "mug"


def test_track_rejects_old_or_conflicting_detections() -> None:
    old_track = ObjectTrack(track_id="old")
    for index in range(5):
        old_track.add(Detection("mug", 0.8, (0.1, 0.1, 0.3, 0.5), index * 300_000_000))
    assert not old_track.is_stable

    conflict = ObjectTrack(track_id="conflict")
    for item in (
        Detection("mug", 0.8, (0.1, 0.1, 0.3, 0.5), 0),
        Detection("phone", 0.8, (0.1, 0.1, 0.3, 0.5), 1),
        Detection("mug", 0.8, (0.1, 0.1, 0.3, 0.5), 2),
        Detection("phone", 0.8, (0.1, 0.1, 0.3, 0.5), 3),
        Detection("book", 0.8, (0.1, 0.1, 0.3, 0.5), 4),
    ):
        conflict.add(item)
    assert not conflict.is_stable


def test_enrichment_queue_coalesces_by_track_id_and_bounds_capacity() -> None:
    queue = EnrichmentQueue(capacity=2)
    queue.put("track-1", {"label": "mug"})
    queue.put("track-2", {"label": "keys"})
    queue.put("track-1", {"label": "cup"})
    queue.put("track-3", {"label": "wallet"})
    assert queue.pop() == ("track-1", {"label": "cup"})
    assert queue.pop() == ("track-3", {"label": "wallet"})
    assert queue.pop() is None
