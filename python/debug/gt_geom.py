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
    # lanesLeft + lanesRight when the dump has them. 0 derives a count from width.
    lanes: int = 0
    # Height of each ``xy`` point, per point: an entry is None where that edge
    # record had no height. None when no record had one. Only the join filter
    # reads it (stacked garage floors share x, y); a segment with a missing
    # end height has no dz, so that one join is judged in plan only.
    z: list[float | None] | None = None


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
# Gap-first join filter for the excursion walk (``road_ahead_split``). A
# candidate whose end lies inside CONNECT_M / CONNECT_HEADING is dropped,
# before the heading pick, when any of these fails:
# * JOIN_GAP_M: distance from the tip to its projection on the candidate
#   road, which is the jump the walk makes onto it. Picking by heading alone
#   took 892630 (9.97 m away, 3.49 deg) over 892725 (0.18 m, 3.59 deg) at the
#   end of west_coast_usa road 894247, and the sent weave jumped 5-10 m
#   sideways (rec5b lines 3281 and 27739).
# * JOIN_HEADING_RAD: leave heading against the tip heading. This also cuts
#   some real junctions sharper than 30 deg. That is deliberate: a cut walk
#   only shortens the line (refused short / no_path); a wrong join sends one.
# * JOIN_DZ_M: height of that projected point against the tip, when both
#   roads carry z. Parking levels of the same garage sit ~4 m apart.
# Thresholds from replaying every west_coast_usa tip against a
# re-implementation of map.lua's link rules (see the PR that added them).
JOIN_GAP_M = 2.0
JOIN_HEADING_RAD = math.radians(30.0)
JOIN_DZ_M = 1.5
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
    """Which polyline the wheel is currently on, and which way it runs.

    ``lane_id`` stays put while the car is still inside that lane, so a
    sample on the painted line does not swap the label to the next lane.
    """

    road_id: str = ""
    forward_heading: float | None = None
    lane_id: int | None = None
    lane_center_m: float | None = None
    # Heading of the edge the label is locked to. A kink in one segment does
    # not replace this while the car still points along it.
    tangent_heading: float | None = None
    last_lane_off: float | None = None
    last_gt: float | None = None
    last_xy: tuple[float, float] | None = None
    label_fresh: bool = False


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


def _z3(raw: object) -> float | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    try:
        value = float(raw[2])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _xy3(raw: object) -> tuple[float, float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        return float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None


def _positive(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out <= 0.0 or out > 80.0:
        return None
    return out


def _edge_width(edge: dict, left: tuple[float, float] | None, right: tuple[float, float] | None) -> float | None:
    """Full road width. A node ``radius`` is half of that.

    Left/right span is the painted width. Some Tech dumps store the same
    distance again as ``width``, and some store the radius under ``width``.
    A value much smaller than the span is treated as a radius. The largest
    plausible reading wins so a narrow edge pair does not shrink the road.
    """
    span = None
    if left is not None and right is not None:
        span = math.hypot(right[0] - left[0], right[1] - left[1])
        if span < 0.5:
            span = None
    cands: list[float] = []
    if span is not None:
        cands.append(span)
    radius = _positive(edge.get("radius"))
    if radius is not None:
        cands.append(radius * 2.0)
    width = _positive(edge.get("width"))
    if width is not None:
        cands.append(width)
        if span is not None and width < span * 0.75:
            cands.append(width * 2.0)
    if not cands:
        return None
    return max(cands)


def road_width_floor(poly: RoadPoly, road: object) -> RoadPoly:
    """Raise ``poly.width_m`` from a road-level radius, width, or lane count."""
    if not isinstance(road, dict):
        return poly
    extra: list[float] = []
    radius = _positive(road.get("radius"))
    if radius is not None:
        extra.append(radius * 2.0)
    width = _positive(road.get("width"))
    if width is not None:
        extra.append(width)
    lanes = road.get("lanes")
    try:
        count = int(lanes)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        count = 0
    if 1 <= count <= 6:
        extra.append(count * LANE_W_M)
    if extra:
        poly.width_m = float(min(40.0, max([poly.width_m, *extra])))
    return poly


def parse_road_edges(road_id: str, edges: object) -> RoadPoly | None:
    """Centerline from BeamNG ``get_road_edges`` records.

    Each record may carry ``middle`` / ``left`` / ``right`` as xyz. A missing
    middle is the mean of the two edges. Width is the left-right distance,
    or the record's ``width`` when the edges are absent.
    """
    if not isinstance(edges, list):
        return None
    xy: list[tuple[float, float]] = []
    zs: list[float | None] = []
    widths: list[float] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        left = _xy3(edge.get("left"))
        right = _xy3(edge.get("right"))
        mid_raw = edge.get("middle") or edge.get("mid") or edge.get("center")
        mid = _xy3(mid_raw)
        mid_z = _z3(mid_raw)
        if mid is None and left is not None and right is not None:
            mid = ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)
            mid_z = None
        if mid_z is None:
            # A middle without a height still gets the edges' mean height.
            lz, rz = _z3(edge.get("left")), _z3(edge.get("right"))
            mid_z = None if lz is None or rz is None else 0.5 * (lz + rz)
        if mid is None:
            continue
        xy.append(mid)
        zs.append(mid_z)
        got = _edge_width(edge, left, right)
        if got is not None:
            widths.append(got)
    if len(xy) < 2:
        return None
    # The widest node is a junction flare. The median span is the road the
    # car is on; using the flare made a 6 m two-lane road look 10 m wide and
    # put the lane boundary under the car.
    if widths:
        ordered = sorted(widths)
        width = ordered[len(ordered) // 2]
    else:
        width = 2.0 * LANE_W_M
    if width < 2.0:
        width = LANE_W_M
    # Per point: one record without a height costs that point, not the road.
    z = None if all(value is None for value in zs) else [None if v is None else float(v) for v in zs]
    return RoadPoly(road_id=str(road_id), xy=xy, width_m=float(width), z=z)


@dataclass(frozen=True)
class HeightCoverage:
    """How much of a parsed road network carries heights (``RoadPoly.z``)."""

    roads: int
    roads_without: int  # no height at any point: no dz for any join on them
    roads_partial: int  # some points without a height
    points: int
    points_without: int

    @property
    def roads_without_share(self) -> float:
        return self.roads_without / self.roads if self.roads else 0.0

    @property
    def roads_partial_share(self) -> float:
        return self.roads_partial / self.roads if self.roads else 0.0

    @property
    def points_without_share(self) -> float:
        return self.points_without / self.points if self.points else 0.0


def height_coverage(roads: list[RoadPoly]) -> HeightCoverage:
    n = without = partial = points = points_without = 0
    for road in roads:
        n += 1
        count = len(road.xy)
        points += count
        z = road.z
        if z is None or len(z) != count:
            without += 1
            points_without += count
            continue
        missing = sum(1 for value in z if value is None)
        points_without += missing
        if missing == count:
            without += 1
        elif missing:
            partial += 1
    return HeightCoverage(n, without, partial, points, points_without)


def road_heights_line(level: str, roads: list[RoadPoly], logged: set[str]) -> str | None:
    """The once-per-map height log line, or None when ``level`` was logged already.

    The join filter's dz check (JOIN_DZ_M) is skipped wherever a height is
    missing, silently. This line says how often that is on this map.
    """
    key = str(level or "current")
    if key in logged:
        return None
    logged.add(key)
    cov = height_coverage(roads)
    return (
        f"collect road heights {key} roads={cov.roads} "
        f"without={cov.roads_without} ({100.0 * cov.roads_without_share:.1f}%) "
        f"partial={cov.roads_partial} ({100.0 * cov.roads_partial_share:.1f}%) "
        f"points_without={cov.points_without}/{cov.points} ({100.0 * cov.points_without_share:.1f}%)"
    )


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


def lane_center_offsets(width_m: float, lanes: int | None = None) -> list[float]:
    """Signed offsets of each lane center from the road middle.

    Index 0 is the leftmost lane. Positive is to the right of the road
    direction. A 7 m two-lane road is at about −1.75 m and +1.75 m.
    ``lanes`` is the dump's lane count when it is known.
    """
    n = int(lanes) if lanes is not None and 1 <= int(lanes) <= 8 else lane_count(width_m)
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
    centers = lane_center_offsets(road.width_m, road.lanes)
    lane_id = min(range(len(centers)), key=lambda k: abs(offset - centers[k]))
    lane_w = road.width_m / len(centers)
    lane_off = offset - centers[lane_id]
    # Lateral only. A car still inside the lane but past a short edge is on
    # the next piece, not off the road. Distance-to-endpoint was marking
    # about 9% of the west-coast cruise off-road.
    off = abs(offset) > road.width_m * 0.5 + SHOULDER_M
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


def _acquire(
    roads: list[RoadPoly],
    x: float,
    y: float,
    heading_rad: float | None,
) -> RoadHit | None:
    """The road the car is in, not merely the nearest centerline.

    A narrow spline beside the carriageway can be closer than the road
    center. Prefer a centerline the car actually sits inside, aligned with
    its heading, and closest to a lane center.
    """
    nearest: RoadHit | None = None
    inside: RoadHit | None = None
    inside_key: tuple[float, float, float] | None = None
    for road in roads:
        hit = _project_onto(road, x, y, heading_rad)
        if hit is None:
            continue
        if nearest is None or hit.dist_m < nearest.dist_m:
            nearest = hit
        contained = abs(hit.offset_m) <= road.width_m * 0.5 + SHOULDER_M and hit.dist_m <= OFF_NETWORK_M
        heading_pen = 0.0 if hit.heading_err_rad is None else abs(hit.heading_err_rad)
        aligned = heading_rad is None or heading_pen <= CONNECT_HEADING
        if not contained or not aligned:
            continue
        key = (heading_pen, abs(hit.lane_offset_m), hit.dist_m)
        if inside_key is None or key < inside_key:
            inside_key = key
            inside = hit
    return inside if inside is not None else nearest


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


def _nearest_on(road: RoadPoly, x: float, y: float) -> tuple[float, int, float, float, float] | None:
    """``(dist, segment, t, px, py)``: the nearest point of ``road`` to (x, y).

    Same pick as ``excursion._locate`` (first segment wins a tie, segments
    under 1 mm skipped), so ``dist`` is the jump the walk makes onto it.
    """
    pts = road.xy
    best: tuple[float, int, float, float, float] | None = None
    px, py = float(x), float(y)
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
            best = (dist, i, t, cx, cy)
    return best


def _z_on(road: RoadPoly, segment: int, t: float) -> float | None:
    z = road.z
    if z is None or len(z) != len(road.xy) or not 0 <= segment < len(z) - 1:
        return None
    za, zb = z[segment], z[segment + 1]
    if za is None or zb is None:
        return None
    return float(za) + float(t) * (float(zb) - float(za))


@dataclass
class JoinCandidate:
    """One end of another road near this tip, and why the walk would not take it."""

    road: RoadPoly
    end: int  # 0: first point, -1: last point
    end_m: float  # tip to that end point
    gap_m: float  # tip to its projection on the road: the jump the walk makes
    heading_rad: float  # |leave heading - tip heading|
    dz_m: float | None  # projected point height - tip height; None where either lacks z
    # None (a continuation), "join_gap", "join_heading", "join_dz", or
    # "not_ahead" (outside CONNECT_HEADING: a side road, never a continuation).
    refused: str | None


def join_candidates(
    roads: list[RoadPoly],
    road: RoadPoly,
    x: float,
    y: float,
    heading_rad: float | None,
) -> tuple[tuple[float, float, float] | None, list[JoinCandidate]]:
    """The forward tip of ``road`` and every road end within CONNECT_M of it."""
    tip = _forward_tip(road, x, y, heading_rad)
    if tip is None:
        return None, []
    tip_x, tip_y, tip_h = tip
    tip_near = _nearest_on(road, tip_x, tip_y)
    tip_z = None if tip_near is None else _z_on(road, tip_near[1], tip_near[2])
    out: list[JoinCandidate] = []
    for other in roads:
        if other.road_id == road.road_id or len(other.xy) < 2:
            continue
        ends = (
            (0, other.xy[0], other.xy[1]),
            (-1, other.xy[-1], other.xy[-2]),
        )
        near: tuple[float, int, float, float, float] | None = None
        for which, end, nxt in ends:
            end_m = math.hypot(end[0] - tip_x, end[1] - tip_y)
            if end_m > CONNECT_M:
                continue
            leave = math.atan2(nxt[1] - end[1], nxt[0] - end[0])
            delta = abs(wrap_rad(leave - tip_h))
            if near is None:
                near = _nearest_on(other, tip_x, tip_y)
            gap = end_m if near is None else near[0]
            other_z = None if near is None else _z_on(other, near[1], near[2])
            dz = None if tip_z is None or other_z is None else other_z - tip_z
            if delta > CONNECT_HEADING:
                refused: str | None = "not_ahead"
            elif gap > JOIN_GAP_M:
                refused = "join_gap"
            elif delta > JOIN_HEADING_RAD:
                refused = "join_heading"
            elif dz is not None and abs(dz) > JOIN_DZ_M:
                refused = "join_dz"
            else:
                refused = None
            out.append(JoinCandidate(other, which, end_m, gap, delta, dz, refused))
    return tip, out


def road_ahead_split(
    roads: list[RoadPoly],
    road: RoadPoly,
    x: float,
    y: float,
    heading_rad: float | None,
) -> tuple[RoadPoly | None, bool]:
    """Colinear continuation, and whether another road leaves this tip.

    A piece boundary with one continuation is the same road. A second
    leave-heading is a junction. The continuation itself is not counted.

    Gap first: a candidate is dropped when the walk would jump more than
    JOIN_GAP_M onto it, turn more than JOIN_HEADING_RAD, or change height by
    more than JOIN_DZ_M. Only the survivors are picked by heading. The fork
    count is unchanged: a road whose end meets this tip (4 m) leaving
    between CONNECT_HEADING and 2.6 rad.
    """
    tip, cands = join_candidates(roads, road, x, y, heading_rad)
    if tip is None:
        return None, False
    cont: RoadPoly | None = None
    cont_delta: float | None = None
    others = 0
    for cand in cands:
        if cand.refused is None:
            if cont_delta is None or cand.heading_rad < cont_delta:
                cont_delta = cand.heading_rad
                cont = cand.road
        elif cand.refused == "not_ahead" and cand.heading_rad < 2.6 and cand.end_m <= 4.0:
            # A road that merely passes nearby is not a fork. The end has
            # to meet this tip.
            others += 1
    return cont, others > 0


LABEL_JUMP_M = 2.0


def label_center_x(hit: RoadHit | None, y_m: float = 5.0) -> float | None:
    """Ego-frame lane center at ``y_m``. This is ``lane_gt[0]`` at 5 m."""
    if hit is None or hit.off_road or hit.lane_offset_m is None:
        return None
    return ego_center_x(float(hit.lane_offset_m), hit.heading_err_rad, y_m)


def hold_lane_label(previous: RoadHit | None, proposed: RoadHit | None) -> RoadHit | None:
    """Drop a one-tick lane swap.

    The 6-minute run's cruise jumps were a lane-index flip of about 3 m, or
    a polyline heading flip, while the car moved less than a metre. A real
    lane change shifts this center by a fraction of a metre per tick.
    """
    if proposed is None:
        return previous
    if previous is None or proposed.off_road:
        return proposed
    old = label_center_x(previous)
    new = label_center_x(proposed)
    if old is None or new is None:
        return proposed
    if abs(new - old) > LABEL_JUMP_M:
        return previous
    return proposed


def stabilize_lane(hit: RoadHit, follow: RoadFollow) -> RoadHit:
    """Keep the lane id until the car has clearly left that lane."""
    if follow.road_id != hit.road_id or follow.lane_center_m is None or follow.lane_id is None:
        follow.lane_id = int(hit.lane_id)
        follow.lane_center_m = float(hit.offset_m - hit.lane_offset_m)
        return hit
    new_center = float(hit.offset_m - hit.lane_offset_m)
    # Stay with the held lane unless the other center is clearly closer.
    # The midpoint between two lanes is where a nearest-lane pick flips.
    # Half a lane is not enough: the car sits on the boundary and a 4 cm
    # step flips the label by a full lane. Require the other center to be
    # closer by two metres, which is past the painted line.
    if abs(hit.offset_m - new_center) + 1.5 >= abs(float(hit.offset_m) - float(follow.lane_center_m)):
        lane_off = float(hit.offset_m) - float(follow.lane_center_m)
        return RoadHit(
            road_id=hit.road_id,
            offset_m=hit.offset_m,
            heading_err_rad=hit.heading_err_rad,
            lane_id=int(follow.lane_id),
            lane_offset_m=lane_off,
            lane_width_m=hit.lane_width_m,
            off_road=hit.off_road,
            dist_m=hit.dist_m,
            curvature=hit.curvature,
            ahead_m=hit.ahead_m,
            discontinuous=hit.discontinuous,
        )
    follow.lane_id = int(hit.lane_id)
    follow.lane_center_m = new_center
    return hit


# A label may follow a successor only when that edge points within about 30
# degrees of the car. A cross street is ~90 degrees and is not a candidate.
LABEL_ALIGN_RAD = math.radians(30.0)
# Published heading error stays inside this. Above it the label is invalid
# rather than a snapped edge. 0.30 rad is the cruise bar from the live runs.
LABEL_HEADING_RAD = 0.30
LABEL_BESIDE_M = 18.0
LABEL_JOIN_M = 8.0


def _segment_views(
    road: RoadPoly,
    x: float,
    y: float,
    heading_rad: float,
) -> list[tuple[float, float, float, float]]:
    """``(dist, heading_err, offset, road_heading)`` for each segment."""
    vx, vy = math.cos(heading_rad), math.sin(heading_rad)
    out: list[tuple[float, float, float, float]] = []
    pts = road.xy
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        if length2 < 1e-6:
            continue
        t = ((x - ax) * dx + (y - ay) * dy) / length2
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        cx = ax + t * dx
        cy = ay + t * dy
        dist = math.hypot(x - cx, y - cy)
        flen = math.sqrt(length2)
        fx, fy = dx / flen, dy / flen
        if fx * vx + fy * vy < 0.0:
            fx, fy = -fx, -fy
        herr = math.atan2(fx * vy - fy * vx, fx * vx + fy * vy)
        rx, ry = fy, -fx
        offset = (x - cx) * rx + (y - cy) * ry
        out.append((dist, herr, offset, math.atan2(fy, fx)))
    return out


def _beside_view(
    road: RoadPoly,
    x: float,
    y: float,
    heading_rad: float | None,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]] | None:
    """Closest foot on ``road``, and the nearby segment that best matches heading.

    None when the car is not laterally on this road. The heading match may
    still be large; the caller decides whether that is a label.
    """
    if heading_rad is None or not math.isfinite(float(heading_rad)):
        return None
    views = _segment_views(road, x, y, float(heading_rad))
    if not views:
        return None
    foot = min(views, key=lambda item: item[0])
    if foot[0] > LABEL_BESIDE_M:
        return None
    if abs(foot[2]) > road.width_m * 0.5 + SHOULDER_M:
        return None
    near = [item for item in views if item[0] <= foot[0] + 8.0]
    aligned = min(near, key=lambda item: abs(item[1]))
    return foot, aligned


def _chord_heading(
    road: RoadPoly,
    x: float,
    y: float,
    heading_rad: float,
    half_m: float = 8.0,
) -> float | None:
    """Heading of the centerline across ``half_m`` behind and ahead of the car.

    One kinked segment swings the nearest tangent. The chord across the
    neighbouring nodes does not.
    """
    pts = road.xy
    best: tuple[float, int, float] | None = None
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        if length2 < 1e-6:
            continue
        t = ((x - ax) * dx + (y - ay) * dy) / length2
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        dist = math.hypot(x - (ax + t * dx), y - (ay + t * dy))
        if best is None or dist < best[0]:
            best = (dist, i, t)
    if best is None:
        return None
    _dist, i, t = best

    def walk(sign: float) -> tuple[float, float]:
        rem = float(half_m)
        if sign > 0.0:
            cx = pts[i][0] + t * (pts[i + 1][0] - pts[i][0])
            cy = pts[i][1] + t * (pts[i + 1][1] - pts[i][1])
            left = math.hypot(pts[i + 1][0] - cx, pts[i + 1][1] - cy)
            if left >= rem:
                flen = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) or 1.0
                return cx + (pts[i + 1][0] - pts[i][0]) / flen * rem, cy + (pts[i + 1][1] - pts[i][1]) / flen * rem
            rem -= left
            j = i + 1
            while j < len(pts) - 1 and rem > 1e-6:
                dx = pts[j + 1][0] - pts[j][0]
                dy = pts[j + 1][1] - pts[j][1]
                seg = math.hypot(dx, dy)
                if seg < 1e-6:
                    j += 1
                    continue
                if seg >= rem:
                    return pts[j][0] + dx / seg * rem, pts[j][1] + dy / seg * rem
                rem -= seg
                j += 1
            return pts[-1]
        cx = pts[i][0] + t * (pts[i + 1][0] - pts[i][0])
        cy = pts[i][1] + t * (pts[i + 1][1] - pts[i][1])
        left = math.hypot(cx - pts[i][0], cy - pts[i][1])
        if left >= rem:
            flen = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) or 1.0
            return cx - (pts[i + 1][0] - pts[i][0]) / flen * rem, cy - (pts[i + 1][1] - pts[i][1]) / flen * rem
        rem -= left
        j = i
        while j > 0 and rem > 1e-6:
            dx = pts[j][0] - pts[j - 1][0]
            dy = pts[j][1] - pts[j - 1][1]
            seg = math.hypot(dx, dy)
            if seg < 1e-6:
                j -= 1
                continue
            if seg >= rem:
                return pts[j][0] - dx / seg * rem, pts[j][1] - dy / seg * rem
            rem -= seg
            j -= 1
        return pts[0]

    back = walk(-1.0)
    fore = walk(1.0)
    road_h = math.atan2(fore[1] - back[1], fore[0] - back[0])
    vx, vy = math.cos(heading_rad), math.sin(heading_rad)
    if math.cos(road_h) * vx + math.sin(road_h) * vy < 0.0:
        road_h = wrap_rad(road_h + math.pi)
    return road_h


def _roads_meet(left: RoadPoly, right: RoadPoly, limit_m: float = LABEL_JOIN_M) -> bool:
    """True when an endpoint of each road sits on the same junction."""
    for ax, ay in (left.xy[0], left.xy[-1]):
        for bx, by in (right.xy[0], right.xy[-1]):
            if math.hypot(ax - bx, ay - by) <= limit_m:
                return True
    return False


def _road_grid(roads: list[RoadPoly]) -> dict[tuple[int, int], list[RoadPoly]]:
    """Coarse cells so a tick does not walk every edge on the map.

    Cached on the list identity. A road is stored in the cells its vertices
    and long segments pass through, plus the neighbouring cell.
    """
    key = id(roads)
    cached = _GRID_CACHE.get(key)
    if cached is not None and cached[0] == len(roads):
        return cached[1]
    cell = 40.0
    grid: dict[tuple[int, int], list[RoadPoly]] = {}
    for road in roads:
        pts = road.xy
        placed: set[tuple[int, int]] = set()

        def add(px: float, py: float) -> None:
            ix = int(math.floor(px / cell))
            iy = int(math.floor(py / cell))
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    slot = (ix + ox, iy + oy)
                    if slot in placed:
                        continue
                    placed.add(slot)
                    grid.setdefault(slot, []).append(road)

        for px, py in pts:
            add(px, py)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            span = math.hypot(x1 - x0, y1 - y0)
            if span <= cell:
                continue
            steps = int(span // cell)
            for step in range(1, steps + 1):
                t = step / (steps + 1)
                add(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
    _GRID_CACHE.clear()
    _GRID_CACHE[key] = (len(roads), grid)
    return grid


_GRID_CACHE: dict[int, tuple[int, dict[tuple[int, int], list[RoadPoly]]]] = {}


def _cell_roads(
    grid: dict[tuple[int, int], list[RoadPoly]],
    x: float,
    y: float,
) -> list[RoadPoly]:
    return grid.get((int(float(x) // 40.0), int(float(y) // 40.0)), [])


def _heading_for_label(
    aligned_err: float,
    road_heading: float,
    vehicle_heading: float,
    follow: RoadFollow,
    same_road: bool,
) -> float | None:
    """Heading error to publish, or None when this tick has no honest label.

    A segment kink can flip the tangent by several tenths of a radian while
    the car has not yawed. Keep the tangent the car was already following.
    """
    if same_road and follow.tangent_heading is not None:
        held = wrap_rad(vehicle_heading - follow.tangent_heading)
        swung = abs(wrap_rad(road_heading - follow.tangent_heading))
        if swung > 0.15 and abs(held) <= LABEL_HEADING_RAD:
            return held
    if abs(aligned_err) <= LABEL_HEADING_RAD:
        follow.tangent_heading = road_heading
        return aligned_err
    if (
        same_road
        and follow.tangent_heading is not None
        and abs(wrap_rad(vehicle_heading - follow.tangent_heading)) <= LABEL_HEADING_RAD
    ):
        return wrap_rad(vehicle_heading - follow.tangent_heading)
    return None


def _hit_from_view(
    road: RoadPoly,
    x: float,
    y: float,
    heading_rad: float,
    herr: float,
) -> RoadHit | None:
    """Lane numbers from the foot of the perpendicular, heading from ``herr``."""
    hit = _project_onto(road, x, y, heading_rad)
    if hit is None:
        return None
    hit.heading_err_rad = float(herr)
    hit.off_road = abs(hit.offset_m) > road.width_m * 0.5 + SHOULDER_M
    return hit


def label_edge(
    roads: list[RoadPoly],
    x: float,
    y: float,
    heading_rad: float | None,
    follow: RoadFollow,
) -> RoadHit | None:
    """The edge the car is on, or None when that edge is ambiguous.

    The lock stays on the current road while the car is laterally inside it.
    A successor is taken only when its heading is within about 30 degrees of
    travel and it is the only such road. A crossing edge is never selected.
    """
    if heading_rad is None or not roads:
        return None
    heading = float(heading_rad)
    grid = _road_grid(roads)
    bag = _cell_roads(grid, x, y)
    current = next((road for road in roads if road.road_id == follow.road_id), None) if follow.road_id else None
    chosen: RoadPoly | None = None
    view = None
    if current is not None:
        view = _beside_view(current, x, y, heading)
        if view is not None:
            chosen = current
        else:
            successors = []
            for road in bag:
                if road.road_id == current.road_id or not _roads_meet(current, road):
                    continue
                got = _beside_view(road, x, y, heading)
                if got is None or abs(got[1][1]) > LABEL_ALIGN_RAD:
                    continue
                successors.append((abs(got[1][1]), got[0][0], road, got))
            successors.sort()
            if len(successors) == 1 or (
                len(successors) > 1 and successors[1][0] > successors[0][0] + 0.12
            ):
                chosen = successors[0][2]
                view = successors[0][3]
                if chosen.road_id != follow.road_id:
                    follow.lane_id = None
                    follow.lane_center_m = None
            elif len(successors) > 1:
                # Two continuations within 30 degrees. Keep the lock and publish nothing.
                return None
            else:
                # The locked edge ended and nothing connected continues. Look again.
                follow.road_id = ""
                follow.tangent_heading = None
                follow.lane_id = None
                follow.lane_center_m = None
    if chosen is None or view is None:
        ranked = []
        for road in bag:
            got = _beside_view(road, x, y, heading)
            if got is None or abs(got[1][1]) > LABEL_ALIGN_RAD:
                continue
            centers = lane_center_offsets(road.width_m, road.lanes)
            lane_fit = min(abs(got[0][2] - center) for center in centers)
            ranked.append((lane_fit, got[0][0], abs(got[1][1]), road, got))
        ranked.sort()
        if not ranked:
            follow.road_id = ""
            follow.tangent_heading = None
            return None
        chosen = ranked[0][3]
        view = ranked[0][4]
        if chosen.road_id != follow.road_id:
            follow.lane_id = None
            follow.lane_center_m = None
    _foot, aligned = view
    chord = _chord_heading(chosen, x, y, heading, 8.0)
    if chord is None:
        aligned_err, road_h = aligned[1], aligned[3]
    else:
        road_h = chord
        aligned_err = wrap_rad(heading - chord)
    herr = _heading_for_label(
        aligned_err,
        road_h,
        heading,
        follow,
        follow.tangent_heading is not None,
    )
    follow.road_id = chosen.road_id
    if herr is None:
        return None
    return _hit_from_view(chosen, x, y, heading, herr)


def follow_label(
    roads: list[RoadPoly],
    x: float,
    y: float,
    heading_rad: float | None,
    follow: RoadFollow,
    speed_mps: float = 0.0,
    last: RoadHit | None = None,
) -> tuple[RoadHit | None, RoadHit | None, bool]:
    """Project the car onto the locked edge this tick.

    Returns ``(label, memory, refused)``. The lane id sticks across the
    painted line. A tick whose edge is ambiguous or points the wrong way
    returns no label, and does not adopt that edge. ``speed_mps`` and
    ``last`` stay in the signature so older callers keep working.
    """
    del speed_mps, last
    prev_road = follow.road_id
    prev_off = follow.last_lane_off
    hit = label_edge(roads, x, y, heading_rad, follow)
    if hit is None or hit.off_road or hit.lane_offset_m is None:
        follow.label_fresh = False
        return hit, None, hit is None
    if hit.road_id != prev_road and prev_off is not None and follow.lane_center_m is None:
        # The new edge has no held lane yet. Continue the previous lane offset
        # so a piece boundary does not jump the label by a lane width.
        road = next((item for item in roads if item.road_id == hit.road_id), None)
        if road is not None:
            centers = lane_center_offsets(road.width_m, road.lanes)
            center = min(centers, key=lambda item: abs((hit.offset_m - item) - prev_off))
            follow.lane_center_m = float(center)
            follow.lane_id = centers.index(center)
    stab = stabilize_lane(hit, follow)
    gt = label_center_x(stab)
    moved = None
    if follow.last_xy is not None:
        moved = math.hypot(float(x) - follow.last_xy[0], float(y) - follow.last_xy[1])
    if (
        follow.label_fresh
        and follow.last_gt is not None
        and gt is not None
        and abs(gt - follow.last_gt) > LABEL_JUMP_M
        and moved is not None
        and moved < 2.0
    ):
        # The car barely moved and the 5 m lane center stepped by more than 2 m.
        # That is a bad edge, not a lane change. Skip this tick; the lock stays.
        follow.label_fresh = False
        return None, None, True
    if stab.lane_offset_m is not None:
        follow.last_lane_off = float(stab.lane_offset_m)
    follow.last_gt = gt
    follow.last_xy = (float(x), float(y))
    follow.label_fresh = True
    return stab, hit, False


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
    nearest = _acquire(roads, x, y, heading_rad)
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
