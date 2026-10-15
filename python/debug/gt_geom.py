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


# A heading jump bigger than this in one tick is a different road, not a yaw.
# gt_ctrl1 and gt_ctrl2 spiked by ~1.5 rad on a crossing polyline. A bend at
# a join is smaller than that. 1.0 rad still rejects a cross street (~1.57).
PREVIEW_JUMP_RAD = 1.0
# Hold the last good curve this long, then slow down without centering the wheel.
# A real run-off (past the tip, no continuation) does not wait this out.
PREVIEW_LOST_S = 1.5
# A road whose endpoint sits this close to ours, and whose heading is within
# this angle, is the continuation. A cross street at the same point is not.
# 8 m / 0.5 rad missed the join on gt_ctrl3 and the car parked in the lane.
CONNECT_M = 18.0
CONNECT_HEADING = 1.0
AHEAD_MIN_M = 4.0
# Metres past the polyline tip before a missing continuation is a dead end.
# gt_ctrl3 set the brake while still on the lane, longitudinal miss ~0.
PAST_TIP_M = 3.0
# Walk this far along the centerline for curvature. A kink tighter than 8 m
# is clamped; that is already near full lock.
CURVATURE_AHEAD_M = 12.0
CURVATURE_CAP = 0.125


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
    curvature: float = 0.0
    ahead_m: float = 0.0
    # The mesh ended and nothing ahead continues this heading. gt_ctrl2
    # drove 13 m past that point at 12 m/s before the hold aged out.
    discontinuous: bool = False


@dataclass
class RoadFollow:
    """Which polyline the wheel is currently on, and which way it runs."""

    road_id: str = ""
    forward_heading: float | None = None


@dataclass
class PreviewMemory:
    """Last curve sample the wheel is allowed to follow."""

    offset_m: float | None = None
    heading_err_rad: float = 0.0
    curvature: float = 0.0
    road_id: str = ""
    age_s: float = 0.0


@dataclass
class CurvePreview:
    offset_m: float
    heading_err_rad: float
    curvature: float
    lane_x: float
    held: bool
    lost: bool
    road_id: str


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


def ego_center_x(lane_offset_m: float, heading_err_rad: float | None, y_m: float = 8.0) -> float:
    """Where the lane center sits in the ego frame, x right, y forward.

    ``lane_offset_m`` is positive when the car is right of the center, so the
    center is to the left (negative x) at the bumper. ``heading_err_rad`` is
    positive when the car points left of the road, which swings the center
    ahead toward the right. This is the straight-road lookahead the GT log
    compares with ``lane_x``.
    """
    h = 0.0 if heading_err_rad is None else float(heading_err_rad)
    return -float(lane_offset_m) + float(y_m) * math.sin(h)


def lane_center_offsets(width_m: float) -> list[float]:
    """Signed offsets of each lane center from the road middle.

    Index 0 is the leftmost lane. Positive is to the right of the road
    direction. A 7 m two-lane road is at about −1.75 m and +1.75 m.
    """
    n = lane_count(width_m)
    lane_w = float(width_m) / n
    return [-float(width_m) / 2.0 + (i + 0.5) * lane_w for i in range(n)]


def _project_onto(
    road: RoadPoly,
    x: float,
    y: float,
    heading_rad: float | None,
) -> RoadHit | None:
    """Closest sample on one polyline. None when the road has no segment."""
    best: tuple[float, int, float, float] | None = None
    px, py = float(x), float(y)
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
            best = (dist, i, cx, cy)
    if best is None:
        return None
    dist, i, cx, cy = best
    ax, ay = pts[i]
    bx, by = pts[i + 1]
    fx, fy = bx - ax, by - ay
    flen = math.hypot(fx, fy) or 1.0
    fx, fy = fx / flen, fy / flen
    if heading_rad is not None:
        vx, vy = math.cos(float(heading_rad)), math.sin(float(heading_rad))
        if fx * vx + fy * vy < 0.0:
            fx, fy = -fx, -fy
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
    kappa, ahead = curvature_ahead(road, i, cx, cy, fx, fy, CURVATURE_AHEAD_M)
    return RoadHit(
        road_id=road.road_id,
        offset_m=float(offset),
        heading_err_rad=None if heading_err is None else float(heading_err),
        lane_id=int(lane_id),
        lane_offset_m=float(lane_off),
        lane_width_m=float(lane_w),
        off_road=bool(off),
        dist_m=float(dist),
        curvature=float(kappa),
        ahead_m=float(ahead),
    )


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

    This is the raw nearest polyline. Through a junction it will pick a
    crossing road. ``project_follow`` stays on the one the car is already on.
    """
    best: RoadHit | None = None
    for road in roads:
        hit = _project_onto(road, x, y, heading_rad)
        if hit is None:
            continue
        if best is None or hit.dist_m < best.dist_m:
            best = hit
    return best


def _road_heading(vehicle_heading: float | None, heading_err: float | None) -> float | None:
    if vehicle_heading is None or heading_err is None:
        return None
    return wrap_rad(float(vehicle_heading) - float(heading_err))


def _remember(follow: RoadFollow, hit: RoadHit, vehicle_heading: float | None) -> None:
    follow.road_id = str(hit.road_id)
    road_h = _road_heading(vehicle_heading, hit.heading_err_rad)
    if road_h is not None:
        follow.forward_heading = road_h


def _forward_tip(
    road: RoadPoly,
    x: float,
    y: float,
    heading_rad: float | None,
) -> tuple[float, float, float] | None:
    """Point and heading at the end of ``road`` in the direction of travel."""
    px, py = float(x), float(y)
    pts = road.xy
    best: tuple[float, int, float, float, float, float] | None = None
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
        flen = math.hypot(dx, dy) or 1.0
        fx, fy = dx / flen, dy / flen
        if best is None or dist < best[0]:
            best = (dist, i, cx, cy, fx, fy)
    if best is None:
        return None
    _dist, i, cx, cy, fx, fy = best
    if heading_rad is not None:
        vx, vy = math.cos(float(heading_rad)), math.sin(float(heading_rad))
        if fx * vx + fy * vy < 0.0:
            fx, fy = -fx, -fy
    forward = (pts[i + 1][0] - pts[i][0]) * fx + (pts[i + 1][1] - pts[i][1]) * fy >= 0.0
    idx = i + 1 if forward else i
    step = 1 if forward else -1
    last_hx, last_hy = fx, fy
    tip_x, tip_y = cx, cy
    guard = 0
    while 0 <= idx < len(pts) and guard < 400:
        guard += 1
        nx, ny = pts[idx]
        dx, dy = nx - tip_x, ny - tip_y
        seg = math.hypot(dx, dy)
        if seg >= 1e-6:
            last_hx, last_hy = dx / seg, dy / seg
            tip_x, tip_y = nx, ny
        idx += step
    return tip_x, tip_y, math.atan2(last_hy, last_hx)


def _successor(
    roads: list[RoadPoly],
    current: RoadPoly,
    x: float,
    y: float,
    heading_rad: float | None,
) -> RoadPoly | None:
    """The polyline that continues past the end of ``current``.

    Shares a neighbourhood with the forward tip and leaves along nearly
    the same heading. A cross street at the junction fails the heading test.
    """
    tip = _forward_tip(current, x, y, heading_rad)
    if tip is None:
        return None
    tip_x, tip_y, tip_h = tip
    best: tuple[float, RoadPoly] | None = None
    for road in roads:
        if road.road_id == current.road_id or len(road.xy) < 2:
            continue
        # Either end can be the junction. The neighbour point is the
        # direction we would drive after joining there.
        ends = (
            (road.xy[0], road.xy[1]),
            (road.xy[-1], road.xy[-2]),
        )
        for end, nxt in ends:
            if math.hypot(end[0] - tip_x, end[1] - tip_y) > CONNECT_M:
                continue
            leave = math.atan2(nxt[1] - end[1], nxt[0] - end[0])
            delta = abs(wrap_rad(leave - tip_h))
            if delta > CONNECT_HEADING:
                continue
            if best is None or delta < best[0]:
                best = (delta, road)
    return None if best is None else best[1]


# Parking brake on gt_ctrl1 shed about 4.5 m/s^2. The mesh has to be at
# least a stop shorter than that or the brake starts too late.
PARK_DECEL_MPS2 = 4.5


def _stop_ahead_m(speed_mps: float) -> float:
    try:
        speed = float(speed_mps)
    except (TypeError, ValueError):
        speed = 0.0
    if speed != speed or speed < 0.0:
        speed = 0.0
    return max(AHEAD_MIN_M, speed * speed / (2.0 * PARK_DECEL_MPS2))


def project_follow(
    roads: list[RoadPoly],
    x: float,
    y: float,
    heading_rad: float | None,
    follow: RoadFollow,
    speed_mps: float = 0.0,
) -> RoadHit | None:
    """Stay on the polyline we are already following, then its continuation.

    ``project_road`` picks the globally nearest centerline. gt_ctrl2 did
    that at three junctions (heading error jumped by ~1.5 rad for one
    tick) and again at the end of road 119441, where the next nearest
    polyline was 19 m away. This keeps the current road while the car is
    still beside it. When that mesh runs out it takes a road that leaves
    the tip along the same heading, and otherwise marks the hit
    discontinuous so the caller brakes without adopting the cross street.
    """
    current = None
    if follow.road_id:
        for road in roads:
            if road.road_id == follow.road_id:
                current = road
                break
    if current is not None:
        hit = _project_onto(current, x, y, heading_rad)
        if hit is not None and lane_still_beside(hit):
            # ahead_m stops growing at the curvature lookahead, so a full
            # walk means the end is not in sight. A short walk is the real
            # distance left. Being near that end is not a dead end: gt_ctrl3
            # was centered in the lane with the tip still ahead, and the
            # brake held it there for the rest of the minute.
            sees_end = hit.ahead_m < CURVATURE_AHEAD_M - 0.05
            if not sees_end or hit.ahead_m >= _stop_ahead_m(speed_mps):
                _remember(follow, hit, heading_rad)
                return hit
            nxt = _successor(roads, current, x, y, heading_rad)
            if nxt is not None:
                chained = _project_onto(nxt, x, y, heading_rad)
                if (
                    chained is not None
                    and chained.heading_err_rad is not None
                    and abs(chained.heading_err_rad) <= CONNECT_HEADING
                ):
                    _remember(follow, chained, heading_rad)
                    return chained
            # Still on the polyline. Only a car that has run past the tip
            # with nothing continuous ahead is the gt_ctrl2 departure.
            along = math.sqrt(max(0.0, hit.dist_m * hit.dist_m - hit.offset_m * hit.offset_m))
            if along < PAST_TIP_M and hit.dist_m <= OFF_NETWORK_M:
                _remember(follow, hit, heading_rad)
                return hit
            hit.discontinuous = True
            return hit
    nearest = project_road(roads, x, y, heading_rad)
    if nearest is None:
        return None
    road_h = _road_heading(heading_rad, nearest.heading_err_rad)
    if (
        follow.forward_heading is not None
        and road_h is not None
        and abs(wrap_rad(road_h - follow.forward_heading)) > PREVIEW_JUMP_RAD
    ):
        nearest.discontinuous = True
        # Do not adopt the crossing road. The caller holds the last curve.
        nearest.road_id = follow.road_id or nearest.road_id
        return nearest
    _remember(follow, nearest, heading_rad)
    return nearest


def curvature_ahead(
    road: RoadPoly,
    seg_index: int,
    cx: float,
    cy: float,
    fx: float,
    fy: float,
    distance_m: float,
) -> tuple[float, float]:
    """Signed curvature along the centerline, and how far the walk got.

    Positive is a left bend (road heading increases counter-clockwise).
    ``fx, fy`` is the forward unit vector already aligned with the vehicle.
    A walk that dies at the end of the polyline returns 0 so the caller
    can keep the previous curvature instead of commanding straight.
    """
    pts = road.xy
    n = len(pts)
    distance = float(distance_m)
    if n < 2 or distance <= 0.5 or seg_index < 0 or seg_index >= n - 1:
        return 0.0, 0.0
    ax, ay = pts[seg_index]
    bx, by = pts[seg_index + 1]
    forward = (bx - ax) * fx + (by - ay) * fy >= 0.0
    h0 = math.atan2(fy, fx)
    travelled = 0.0
    px, py = float(cx), float(cy)
    idx = seg_index + 1 if forward else seg_index
    step = 1 if forward else -1
    last_hx, last_hy = fx, fy
    guard = 0
    while travelled < distance - 1e-6 and guard < 400:
        guard += 1
        if idx < 0 or idx >= n:
            break
        nx, ny = pts[idx]
        dx, dy = nx - px, ny - py
        seg = math.hypot(dx, dy)
        if seg < 1e-6:
            idx += step
            continue
        last_hx, last_hy = dx / seg, dy / seg
        if travelled + seg >= distance:
            travelled = distance
            break
        travelled += seg
        px, py = nx, ny
        idx += step
    if travelled < 2.0:
        return 0.0, travelled
    kappa = wrap_rad(math.atan2(last_hy, last_hx) - h0) / travelled
    return max(-CURVATURE_CAP, min(CURVATURE_CAP, kappa)), travelled


def lane_still_beside(hit: RoadHit) -> bool:
    """True when the car is still next to its lane, including past a polyline end.

    ``off_road`` is also true when the nearest centerline sample is an
    endpoint more than ``OFF_NETWORK_M`` away. gt_ctrl1 did that at lane
    offset 0 and the controller zeroed the wheel at 14 m/s.
    """
    if hit.lane_offset_m is None:
        return False
    width = hit.lane_width_m if hit.lane_width_m and hit.lane_width_m > 0.0 else LANE_W_M
    return abs(float(hit.lane_offset_m)) <= float(width) * 0.5 + SHOULDER_M


def update_curve_preview(
    memory: PreviewMemory,
    *,
    offset_m: float | None,
    heading_err_rad: float | None,
    curvature: float,
    road_id: str,
    beside: bool,
    dt_s: float,
    discontinuous: bool = False,
) -> CurvePreview | None:
    """Accept a sample, or keep the last one through a jump or a short gap.

    A heading change past ``PREVIEW_JUMP_RAD`` is rejected even when the
    new road's lane offset is small. gt_ctrl1 had that at t=46 s: offset
    −0.8 m, heading error −1.56 rad. A discontinuity (the mesh ended, or
    that jump) is lost immediately so the caller brakes. gt_ctrl2 waited
    out ``PREVIEW_LOST_S`` at 12 m/s and was 19 m onto the next polyline
    before the brake. The steer command stays the held curve either way.
    """
    try:
        dt = float(dt_s)
    except (TypeError, ValueError):
        dt = 0.1
    if dt != dt or dt < 0.0:
        dt = 0.1
    fresh = bool(beside) and offset_m is not None and heading_err_rad is not None
    heading = 0.0
    offset = 0.0
    if fresh:
        try:
            offset = float(offset_m)
            heading = float(heading_err_rad)
        except (TypeError, ValueError):
            fresh = False
        else:
            if offset != offset or heading != heading:
                fresh = False
    jumped = False
    if fresh and memory.offset_m is not None:
        if abs(wrap_rad(heading - memory.heading_err_rad)) > PREVIEW_JUMP_RAD:
            fresh = False
            jumped = True
    if discontinuous:
        fresh = False
    if fresh:
        try:
            kappa = float(curvature)
        except (TypeError, ValueError):
            kappa = 0.0
        if kappa != kappa:
            kappa = 0.0
        memory.offset_m = offset
        memory.heading_err_rad = heading
        memory.curvature = max(-CURVATURE_CAP, min(CURVATURE_CAP, kappa))
        memory.road_id = str(road_id)
        memory.age_s = 0.0
        held = False
    elif memory.offset_m is not None:
        memory.age_s += dt
        held = True
    else:
        return None
    return CurvePreview(
        offset_m=float(memory.offset_m),
        heading_err_rad=float(memory.heading_err_rad),
        curvature=float(memory.curvature),
        lane_x=ego_center_x(float(memory.offset_m), float(memory.heading_err_rad), 8.0),
        held=held,
        lost=bool(discontinuous or jumped or memory.age_s > PREVIEW_LOST_S),
        road_id=memory.road_id,
    )
