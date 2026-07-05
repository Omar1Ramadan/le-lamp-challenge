from dataclasses import dataclass

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class SceneLocation:
    horizontal_region: str
    depth_band: str
    anchor_name: str | None


def _intersection_over_box(box: BBox, anchor: BBox) -> float:
    x1, y1, x2, y2 = box
    ax1, ay1, ax2, ay2 = anchor
    width = max(0.0, min(x2, ax2) - max(x1, ax1))
    height = max(0.0, min(y2, ay2) - max(y1, ay1))
    area = max(0.000001, (x2 - x1) * (y2 - y1))
    return width * height / area


def locate_box(box: BBox, *, anchors: dict[str, BBox]) -> SceneLocation:
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    horizontal = "left" if center_x < 1 / 3 else "right" if center_x >= 2 / 3 else "center"
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    depth = "foreground" if area >= 0.20 else "midground" if area >= 0.06 else "background"
    candidates = [
        name for name, anchor in anchors.items() if _intersection_over_box(box, anchor) >= 0.50
    ]
    return SceneLocation(horizontal, depth, sorted(candidates)[0] if candidates else None)
