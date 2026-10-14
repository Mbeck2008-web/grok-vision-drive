"""Road-center geometry for the sim ground-truth log.

Pure functions. BeamNG is not imported here. A road is a centerline of
``middle`` points plus a width. The ego lane is the lane of that road whose
center is closest to the car.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

LANE_W_M = 3.5
SHOULDER_M = 0.6
# Farther than this from every centerline, the car is not on the mapped road.
OFF_NETWORK_M = 12.0


@dataclass
class RoadPoly:
    road_id: str
    xy: list[tuple[float, float]]
    width_m: float


@dataclass
class RoadHit:
    road_id: str
    offset_m: float
    heading_err_rad: float | None
    lane_id: int
    lane_offset_m: float
    lane_width_m: float
    off_road: bool
    dist_m: float


def heading_from_dir(dx: float, dy: float) -> float:
    """World heading, radians, atan2(y, x)."""
    return math.atan2(float(dy), float(dx))


def wrap_rad(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _xy3(raw: object) -> tuple[float, float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        return float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None


def parse_road_edges(road_id: str, edges: object) -> RoadPoly | None:
    """Centerline from BeamNG ``get_road_edges`` records.

    Each record may carry ``middle`` / ``left`` / ``right`` as xyz. A missing
    middle is the mean of the two edges. Width is the left-right distance,
    or the record's ``width`` when the edges are absent.
    """
    if not isinstance(edges, list):
        return None
    xy: list[tuple[float, float]] = []
    widths: list[float] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        left = _xy3(edge.get("left"))
        right = _xy3(edge.get("right"))
        mid = _xy3(edge.get("middle") or edge.get("mid") or edge.get("center"))
        if mid is None and left is not None and right is not None:
            mid = ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)
        if mid is None:
            continue
        xy.append(mid)
        if left is not None and right is not None:
            widths.append(math.hypot(right[0] - left[0], right[1] - left[1]))
        elif edge.get("width") is not None:
            try:
                widths.append(abs(float(edge["width"])))
            except (TypeError, ValueError):
                pass
    if len(xy) < 2:
        return None
    width = sum(widths) / len(widths) if widths else 2.0 * LANE_W_M
    if width < 2.0:
        width = LANE_W_M
    return RoadPoly(road_id=str(road_id), xy=xy, width_m=float(width))


def lane_count(width_m: float) -> int:
    return max(1, int(round(float(width_m) / LANE_W_M)))


def lane_center_offsets(width_m: float) -> list[float]:
    """Signed offsets of each lane center from the road middle.

    Index 0 is the leftmost lane. Positive is to the right of the road
    direction. A 7 m two-lane road is at about −1.75 m and +1.75 m.
    """
    n = lane_count(width_m)
    lane_w = float(width_m) / n
    return [-float(width_m) / 2.0 + (i + 0.5) * lane_w for i in range(n)]


def project_road(
    roads: list[RoadPoly],
    x: float,
    y: float,
    heading_rad: float | None,
) -> RoadHit | None:
    """Nearest centerline sample for a vehicle at ``(x, y)``.

    The segment is flipped so it points with the vehicle when a heading is
    known. ``offset_m`` is positive to the right of that direction.
    ``heading_err_rad`` is positive when the vehicle points left of the road
    (counter-clockwise). ``off_road`` is outside the road plus a shoulder,
    or farther than ``OFF_NETWORK_M`` from every centerline.
    """
    best: tuple[float, RoadPoly, int, float, float, float] | None = None
    px, py = float(x), float(y)
    for road in roads:
        pts = road.xy
        for i in range(len(pts) - 1):
            ax, ay = pts[i]
            bx, by = pts[i + 1]
            dx, dy = bx - ax, by - ay
            length2 = dx * dx + dy * dy
            if length2 < 1e-6:
                continue
            t = ((px - ax) * dx + (py - ay) * dy) / length2
            t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
            cx = ax + t * dx
            cy = ay + t * dy
            dist = math.hypot(px - cx, py - cy)
            if best is None or dist < best[0]:
                best = (dist, road, i, cx, cy, t)
    if best is None:
        return None
    dist, road, i, cx, cy, _t = best
    ax, ay = road.xy[i]
    bx, by = road.xy[i + 1]
    fx, fy = bx - ax, by - ay
    flen = math.hypot(fx, fy) or 1.0
    fx, fy = fx / flen, fy / flen
    if heading_rad is not None:
        vx, vy = math.cos(float(heading_rad)), math.sin(float(heading_rad))
        if fx * vx + fy * vy < 0.0:
            fx, fy = -fx, -fy
    # Right of the road direction: forward × up.
    rx, ry = fy, -fx
    offset = (px - cx) * rx + (py - cy) * ry
    heading_err = None
    if heading_rad is not None:
        vx, vy = math.cos(float(heading_rad)), math.sin(float(heading_rad))
        heading_err = math.atan2(fx * vy - fy * vx, fx * vx + fy * vy)
    centers = lane_center_offsets(road.width_m)
    lane_id = min(range(len(centers)), key=lambda k: abs(offset - centers[k]))
    lane_w = road.width_m / len(centers)
    lane_off = offset - centers[lane_id]
    off = dist > OFF_NETWORK_M or abs(offset) > road.width_m * 0.5 + SHOULDER_M
    return RoadHit(
        road_id=road.road_id,
        offset_m=float(offset),
        heading_err_rad=None if heading_err is None else float(heading_err),
        lane_id=int(lane_id),
        lane_offset_m=float(lane_off),
        lane_width_m=float(lane_w),
        off_road=bool(off),
        dist_m=float(dist),
    )
