"""Hardcoded route from what the car can see, then a path a controller can follow.

Geometry and rules only. No network, no checkpoint, no wheel angle.

What is in frame is a sample, not the whole road:

* A lane line that stops early, or only one side of it, is continued in the
  direction it was already going. A missing side is the last measured width,
  or a default width, off the side we do have. A frame with no lines can
  still continue a prior lane the caller still remembers.
* A car that is only partly in frame is given a full footprint. A car that
  just dropped out is coasted with its last yaw and speed, and the path
  stops short of that predicted footprint.
* Lane points that lie on a circular arc are a roundabout we have not seen
  all of. The path leaves the visible arc and keeps going around that circle.
* A stop sign, a yield, or a red light on the route ends the path there.
  An unknown lamp is not treated as red. A sign that is only partly seen,
  or briefly missed, still counts: signs do not move.

The output is a polyline in the ego frame (+x right, +y forward) plus the
speed flags ``plan_command`` already turns into steer and throttle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from python.planning.speed import plan_speed

LANE_W_DEFAULT = 3.5
LANE_W_MIN = 2.4
LANE_W_MAX = 5.2
# A pair outside this band is two different roads, not the ego lane.
# The efc25e3 run held det (−4.11, +6.64), a 10.7 m gap, and called it a pair.
LANE_PAIR_W_MIN = 2.7
LANE_PAIR_W_MAX = 4.5
# A paint point farther than this is not this road's lane. Two widths at the
# bumper, opening slightly with distance. -16 m at y=8 is past it. A lane
# sitting one width beside the car is not.
LANE_OUTLIER_WIDTHS = 2.0
LANE_OUTLIER_PER_M = 0.35
# Ego-lane center memory. The PR 82 run's lookahead flipped between about
# -1.1 m and +1.5 m in one tick (one boundary, side chosen by the sign of x,
# center jumping by a full lane). A step inside the gate is the same lane.
# A larger step has to repeat, then creeps across. The creep is under 2 m,
# which is the jump the replay refuses.
CENTER_GATE_M = 1.25
CENTER_HYST_TICKS = 3
CENTER_SLEW_M = 0.35
# A single frame may not move the center by more than this. The gt_run5
# lane_x series jumped about ±1 m inside the old 1.25 m instant-take gate,
# and the wheel followed the noise.
CENTER_INNOV_M = 0.40
CENTER_BLEND = 0.5
# One visible line within this band of the last side decision keeps that
# side. Crossing it is the other boundary.
ONE_SIDE_DEADBAND_M = 0.75
LENGTH_M = 36.0
STEP_M = 1.0
EGO_WIDTH = 1.9
VEHICLE_LENGTH = 4.5
VEHICLE_WIDTH = 1.8
STANDOFF_M = 4.0
SIGN_STANDOFF_M = 3.0
SHOULDER_M = 4.5
LIGHT_GATE_M = 6.5
V_CAP = 18.0
ROUNDABOUT_R_MIN = 7.0
ROUNDABOUT_R_MAX = 35.0
ROUNDABOUT_EXTRA_RAD = 1.35
ROUNDABOUT_RMS_MAX = 0.45
ROUNDABOUT_SPAN_MIN = 0.35

_OCCUPANTS = frozenset({"vehicle", "car", "truck", "bus", "pedestrian", "bike"})
# Full brake only when closing speed is real. A short gap alone does not slam it.
_FULL_BRAKE_TTC = 1.2
_COMFORT_DECEL = 2.5
_SIGN_MISS_MAX = 8
_PARTIAL_GATE_M = 1.6
_TURN_RAD = 0.35
_LANE_CHANGE_M = 2.2


@dataclass
class PathPlan:
    path_ego: list[dict[str, float]]
    path_width: float = LANE_W_DEFAULT
    path_conf: float = 0.0
    curvature: float = 0.0
    drivable: bool = False
    target_v: float = 0.0
    ttc_lead: float | None = None
    aeb: str = "off"
    cipv_id: int | None = None
    stop_reason: str = "none"
    prediction: str = "none"
    occupied: list[dict[str, Any]] = field(default_factory=list)
    # Polyline length in meters. The supervisor graph plots this as predicted throttle.
    path_length_m: float = 0.0
    # Brake the path follower would command, in [0, 1]. Not a wheel reading.
    pred_brake: float = 0.0
    # Background turn signal from the route: off | left | right. Not a wheel reading.
    blinker: str = "off"
    # pair | one_left | one_right, plus /hold or /slew when the center memory
    # did not take the raw midpoint. none when there is no route.
    lane_source: str = "none"

    def planner_dict(self) -> dict[str, Any]:
        return {
            "corridor_width": self.path_width,
            "curvature": self.curvature,
            "target_v": self.target_v,
            "ttc_lead": self.ttc_lead,
            "aeb": self.aeb,
            "cipv_id": self.cipv_id,
            "stop_reason": self.stop_reason,
            "prediction": self.prediction,
            "path_length_m": self.path_length_m,
            "pred_brake": self.pred_brake,
            "blinker": self.blinker,
            "lane_source": self.lane_source,
        }


def reckon_ego_x(x_m: float, y_m: float, *, yaw_rate: float, dt: float) -> float:
    """Move a point fixed in the world into the ego frame after ``dt``.

    ``yaw_rate`` is counterclockwise, the same sign as the ground-truth
    heading. A right turn is negative, so a point ahead of the car moves
    toward −x. Speed does not change x while the lane is parallel to the car.
    """
    theta = float(yaw_rate) * float(dt)
    return float(x_m) * math.cos(theta) + float(y_m) * math.sin(theta)


def reckon_path(
    path: list[dict[str, float]] | None,
    *,
    speed_mps: float,
    yaw_rate: float,
    dt: float,
) -> list[dict[str, float]]:
    """Rigid ego update of a remembered route. Points that fall behind drop out."""
    theta = float(yaw_rate) * float(dt)
    c = math.cos(theta)
    s = math.sin(theta)
    vdt = max(0.0, float(speed_mps)) * float(dt)
    out: list[dict[str, float]] = []
    for pt in path or []:
        x = float(pt["x"])
        y = float(pt["y"])
        x2 = x * c + y * s
        y2 = -x * s + y * c - vdt
        if y2 > 0.4:
            out.append({"x": float(x2), "y": float(y2), "z": 0.0})
    return out


def lane_x_limit(y_m: float) -> float:
    """Largest |x| a lane point at this forward distance may have."""
    try:
        y = float(y_m)
    except (TypeError, ValueError):
        y = 0.0
    if y != y:
        y = 0.0
    return LANE_OUTLIER_WIDTHS * LANE_W_DEFAULT + LANE_OUTLIER_PER_M * max(0.0, y)


def lane_point_plausible(x_m: float, y_m: float, *, widths: float | None = None) -> bool:
    """False when this point is too far sideways to be a lane boundary."""
    try:
        x = float(x_m)
        y = float(y_m)
    except (TypeError, ValueError):
        return False
    if x != x or x in (float("inf"), float("-inf")):
        return False
    if widths is None:
        limit = lane_x_limit(y)
    else:
        # Flat. The per-metre term would keep a +8 m miss at the 8 m row
        # and still drop the beside line at x=-6.4 m that the ground test paints.
        limit = float(widths) * LANE_W_DEFAULT
    return abs(x) <= limit


@dataclass
class LaneTrack:
    """Ego-lane center remembered across ticks. Owned by the perception loop."""

    center_x: float | None = None
    # One-sided: "left" means the visible line is the left boundary.
    side: str | None = None
    pending_x: float | None = None
    pending_n: int = 0
    source: str = "none"

    def reset(self) -> None:
        self.center_x = None
        self.side = None
        self.pending_x = None
        self.pending_n = 0
        self.source = "none"


def follow_lane_center(
    track: LaneTrack,
    measured_x: float,
    *,
    count: bool = True,
) -> tuple[float, str]:
    """Center x to drive, and take | hold | slew.

    ``measured_x`` is the midpoint the lines imply this tick. The first
    sample is stored. A later sample inside ``CENTER_INNOV_M`` is blended
    in. A larger step does not land until it has repeated
    ``CENTER_HYST_TICKS`` times, and then moves at most ``CENTER_SLEW_M``
    per call. ``count`` false holds the memory still (a prior-lane fallback
    must not vote).
    """
    measured = float(measured_x)
    if track.center_x is None:
        track.center_x = measured
        track.pending_x = None
        track.pending_n = 0
        return measured, "take"
    # A prior-lane sample is already in a stale ego frame. Do not absorb it,
    # even when it still sits inside the gate.
    if not count:
        return float(track.center_x), "hold"
    innov = measured - float(track.center_x)
    if abs(innov) <= CENTER_INNOV_M:
        track.center_x = float(track.center_x) + CENTER_BLEND * innov
        track.pending_x = None
        track.pending_n = 0
        return float(track.center_x), "take"
    if track.pending_x is None or abs(measured - track.pending_x) > CENTER_GATE_M:
        track.pending_x = measured
        track.pending_n = 1
    else:
        track.pending_n += 1
    if track.pending_n < CENTER_HYST_TICKS:
        return float(track.center_x), "hold"
    step = max(-CENTER_SLEW_M, min(CENTER_SLEW_M, track.pending_x - track.center_x))
    track.center_x = float(track.center_x) + step
    if abs(track.pending_x - track.center_x) <= CENTER_GATE_M:
        track.center_x = float(track.pending_x)
        track.pending_x = None
        track.pending_n = 0
    return float(track.center_x), "slew"


def follow_path(plan: PathPlan, *, ego_speed_mps: float, seq: int = 0) -> Any:
    """Steer and pedals from the predicted route. The wheel is not an input."""
    from python.control.actuate import plan_command

    return plan_command(
        path_ego=plan.path_ego,
        planner=plan.planner_dict(),
        ego_speed_mps=float(ego_speed_mps),
        seq=int(seq),
    )


def predict_path(
    *,
    lanes_bev: list | None = None,
    lane_conf: float = 0.0,
    curvature: float = 0.0,
    tracks: list | None = None,
    signs: list | None = None,
    ego_speed_mps: float = 0.0,
    prior_lanes: list | None = None,
    length_m: float = LENGTH_M,
    step_m: float = STEP_M,
    lane_track: LaneTrack | None = None,
    remember: bool = True,
) -> PathPlan:
    """Predict a drivable route from the scene. Wheel angle is not an argument.

    ``lane_track`` keeps the ego-lane center across calls. ``remember``
    false still shifts the route onto that center, but the sample does not
    count toward a lane switch.
    """
    del curvature  # image-slope hint is not the route; the polyline is
    step = max(0.4, float(step_m))
    horizon = max(8.0, float(length_m))
    live = _polylines(lanes_bev)
    used_prior = False
    polys = live
    if not polys:
        polys = _polylines(prior_lanes)
        used_prior = bool(polys)

    # Rear and repeater lines sit behind the bumper. A building edge beside
    # the car is not a smooth forward lane. Either one, mixed in, makes the
    # whole set look like a roundabout (src=roundabout at t=8.9 s).
    # Drop far points first so one bad vertex does not throw away the line.
    polys = _drop_lane_outliers(polys)
    ahead = [p for p in polys if max(pt[1] for pt in p) > 1.0] or list(polys)
    smooth = [p for p in ahead if _smooth_forward(p)]
    route_polys = smooth if smooth and _is_forward_lane(smooth) else ahead
    circle = None if _is_forward_lane(route_polys) else (_fit_roundabout(route_polys) if route_polys else None)
    ring_curv = 0.0
    lane_source = "none"
    if circle is not None:
        raw, width, ring_curv = _roundabout_route(circle, polys, step)
        prediction = "roundabout"
        lane_source = "roundabout"
        extended = True
    else:
        built = (
            _lane_route(route_polys, horizon, step, lane_track=lane_track, remember=remember)
            if polys
            else None
        )
        if built is None:
            return _expose_pedals(_empty(), ego_speed_mps)
        raw, width, extended, lane_source = built
        if used_prior:
            prediction = "context"
            if lane_source != "none":
                lane_source = f"prior/{lane_source}"
        elif extended:
            prediction = "extended"
        else:
            prediction = "lanes"

    path = [_pt(x, y) for x, y in raw]
    if len(path) < 2:
        return _expose_pedals(_empty(), ego_speed_mps)

    curv = ring_curv if prediction == "roundabout" and ring_curv > 1e-4 else _path_curvature(path)
    occupants = _predict_occupants(tracks or [])
    path, stop_reason, cipv_id, ttc, occupied, stop_dist, closing = _apply_scene(
        path,
        occupants,
        signs or [],
        width,
        ego_speed_mps=float(ego_speed_mps),
    )
    aeb, target, ttc = _speed_flags(
        curv=curv,
        stop_reason=stop_reason,
        ttc=ttc,
        stop_dist=stop_dist,
        closing_mps=closing,
        ego_speed_mps=float(ego_speed_mps),
        signs=signs or [],
        path=path,
    )
    conf = _confidence(prediction, float(lane_conf), extended=extended or used_prior)
    plan = PathPlan(
        path_ego=path,
        path_width=float(width),
        path_conf=conf,
        curvature=curv,
        drivable=True,
        target_v=target,
        ttc_lead=ttc,
        aeb=aeb,
        cipv_id=cipv_id,
        stop_reason=stop_reason,
        prediction=prediction,
        occupied=occupied,
        lane_source=lane_source,
    )
    if lane_track is not None:
        lane_track.source = lane_source
    return _expose_pedals(plan, ego_speed_mps)


def path_length_m(path: list | None) -> float:
    """Arc length of a route polyline, in meters.

    Ego frame: +x right, +y forward, +z up. Each step is the straight
    distance between consecutive points, including z when a point has it.
    """
    if not path:
        return 0.0
    total = 0.0
    prev: tuple[float, float, float] | None = None
    for raw in path:
        if isinstance(raw, dict):
            x, y, z = raw.get("x"), raw.get("y"), raw.get("z", 0.0)
        else:
            try:
                x, y = raw[0], raw[1]
                z = raw[2] if len(raw) > 2 else 0.0
            except (TypeError, IndexError):
                continue
        try:
            point = (float(x), float(y), float(z or 0.0))
        except (TypeError, ValueError):
            continue
        if prev is not None:
            total += math.dist(prev, point)
        prev = point
    return float(total)


def _expose_pedals(plan: PathPlan, ego_speed_mps: float) -> PathPlan:
    """Record path length, blinker, and the brake the follower would send."""
    plan.path_length_m = path_length_m(plan.path_ego)
    plan.blinker = blinker_for_path(plan.path_ego, plan.prediction)
    commanded = follow_path(plan, ego_speed_mps=float(ego_speed_mps))
    plan.pred_brake = float(commanded.brake)
    return plan


def _is_forward_lane(polys: list[list[tuple[float, float]]]) -> bool:
    """True when every visible line is still a forward lane, not a closing circle.

    Order does not matter. A bend stays a smooth graph of y, including one
    sample a few centimetres out of order. A roundabout is a point set that,
    once ordered by y, still jumps sideways because the road has folded.
    """
    if not polys:
        return False
    return all(_smooth_forward(poly) for poly in polys)


def _smooth_forward(poly: list[tuple[float, float]]) -> bool:
    if len(poly) < 2:
        return False
    ordered = sorted(poly, key=lambda p: (p[1], p[0]))
    if ordered[-1][1] - ordered[0][1] < 1.5:
        return False
    for a, b in zip(ordered, ordered[1:]):
        dy = abs(b[1] - a[1])
        dx = abs(b[0] - a[0])
        # A folded circle puts both sides of the ring at nearly the same y.
        if dx > 1.25 and dx > max(dy, 0.05) * 4.0:
            return False
    return True


def blinker_for_path(path: list[dict[str, float]] | None, prediction: str = "") -> str:
    """Turn signal from the route. Straight stays off.

    A roundabout exit signals right (leave the circle). A lane change is a
    lateral shift that straightens up. A turn is a heading change that stays.
    """
    if prediction == "roundabout":
        return "right"
    if not path or len(path) < 4:
        return "off"
    headings: list[float] = []
    for a, b in zip(path, path[1:]):
        dx = float(b["x"]) - float(a["x"])
        dy = float(b["y"]) - float(a["y"])
        if dx * dx + dy * dy < 0.04:
            continue
        headings.append(math.atan2(dx, dy))
    if len(headings) < 2:
        return "off"
    n = max(1, len(headings) // 5)
    h0 = sum(headings[:n]) / n
    h1 = sum(headings[-n:]) / n
    turn = _wrap(h1 - h0)
    shift = float(path[-1]["x"]) - float(path[0]["x"])
    if abs(shift) >= _LANE_CHANGE_M and abs(h1) < 0.22 and abs(turn) < 0.28:
        return "right" if shift > 0 else "left"
    if turn > _TURN_RAD:
        return "right"
    if turn < -_TURN_RAD:
        return "left"
    return "off"


def _empty() -> PathPlan:
    return PathPlan(path_ego=[], path_conf=0.35, prediction="none", drivable=False)


def _pt(x: float, y: float) -> dict[str, float]:
    return {"x": round(float(x), 4), "y": round(float(y), 4), "z": 0.0}


def _cls(obj: dict[str, Any]) -> str:
    return str(obj.get("cls") or obj.get("class") or "").strip().lower()


def _f(obj: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if obj.get(key) is None:
            continue
        try:
            return float(obj[key])
        except (TypeError, ValueError):
            continue
    return default


def _polylines(lanes: list | None) -> list[list[tuple[float, float]]]:
    out: list[list[tuple[float, float]]] = []
    for poly in lanes or []:
        pts: list[tuple[float, float]] = []
        for p in poly or []:
            if isinstance(p, dict):
                x, y = p.get("x"), p.get("y")
            else:
                try:
                    x, y = p[0], p[1]
                except (TypeError, IndexError):
                    continue
            if x is None or y is None:
                continue
            pts.append((float(x), float(y)))
        if len(pts) >= 2:
            out.append(pts)
    return out


def _mean_x(pts: list[tuple[float, float]]) -> float:
    return sum(p[0] for p in pts) / len(pts)


def _near_x(pts: list[tuple[float, float]]) -> float:
    """Lateral position of the line next to the car, not the mean of the whole arc."""
    ordered = _sort_y(pts)
    y0 = ordered[0][1]
    near = [p for p in ordered if p[1] <= y0 + 6.0]
    if not near:
        near = ordered[:1]
    return sum(p[0] for p in near) / len(near)


def _sort_y(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted(pts, key=lambda p: p[1])
    dedup: list[tuple[float, float]] = []
    for x, y in ordered:
        if dedup and abs(y - dedup[-1][1]) < 0.05:
            px, py = dedup[-1]
            dedup[-1] = ((px + x) * 0.5, py)
            continue
        dedup.append((x, y))
    return dedup


def _interp(pts: list[tuple[float, float]], y: float) -> float | None:
    if len(pts) < 2:
        return None
    if y <= pts[0][1]:
        return pts[0][0]
    if y >= pts[-1][1]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if (y0 <= y <= y1) or (y1 <= y <= y0):
            if abs(y1 - y0) < 1e-6:
                return x0
            t = (y - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)
    return None


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _circumcircle(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> tuple[float, float, float] | None:
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-8:
        return None
    s1 = x1 * x1 + y1 * y1
    s2 = x2 * x2 + y2 * y2
    s3 = x3 * x3 + y3 * y3
    ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    radius = math.hypot(x1 - ux, y1 - uy)
    if not math.isfinite(radius):
        return None
    return ux, uy, radius


def _fit_poly(pts: list[tuple[float, float]]) -> tuple[float, float, float, float, float] | None:
    if len(pts) < 5:
        return None
    circ = _circumcircle(pts[0], pts[len(pts) // 2], pts[-1])
    if circ is None:
        return None
    cx, cy, radius = circ
    if not (ROUNDABOUT_R_MIN <= radius <= ROUNDABOUT_R_MAX):
        return None
    err = [abs(math.hypot(x - cx, y - cy) - radius) for x, y in pts]
    rms = math.sqrt(sum(e * e for e in err) / len(err))
    if rms > ROUNDABOUT_RMS_MAX:
        return None
    angs = [math.atan2(y - cy, x - cx) for x, y in pts]
    rel = [_wrap(a - angs[0]) for a in angs]
    span = max(rel) - min(rel)
    if span < ROUNDABOUT_SPAN_MIN:
        return None
    return cx, cy, radius, rms, span


def _fit_roundabout(
    polys: list[list[tuple[float, float]]],
) -> tuple[float, float, float] | None:
    fits = []
    for poly in polys:
        fit = _fit_poly(poly)
        if fit is not None:
            fits.append(fit)
    if not fits:
        return None
    if len(fits) >= 2:
        a, b = fits[0], fits[1]
        if math.hypot(a[0] - b[0], a[1] - b[1]) <= 4.0:
            return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5)
    best = min(fits, key=lambda f: f[3])
    return best[0], best[1], best[2]


def _roundabout_route(
    circle: tuple[float, float, float],
    polys: list[list[tuple[float, float]]],
    step: float,
) -> tuple[list[tuple[float, float]], float, float]:
    """Follow the ring in the direction the visible points already run.

    No radial chord from the bumper to the paint: that cut toward the center.
    Curvature is 1/radius so a 12 m ring is not given a highway speed.
    """
    cx, cy, radius = circle
    cloud = [p for line in polys for p in line]
    angs = [math.atan2(y - cy, x - cx) for x, y in cloud]
    nearest = min(range(len(cloud)), key=lambda i: math.hypot(cloud[i][0], cloud[i][1]))
    a_near = angs[nearest]
    rels = [_wrap(a - a_near) for a in angs]
    mean_rel = sum(rels) / max(1, len(rels))
    # Continue away from the visible mass, into the unseen part of the ring.
    # The point set decides this, so a y-sorted copy of the same arc agrees.
    direction = -1.0 if mean_rel > 1e-3 else 1.0
    ext = max((rel * direction) for rel in rels)
    goal = max(0.0, ext) + ROUNDABOUT_EXTRA_RAD
    path: list[tuple[float, float]] = []
    theta = a_near
    traveled = 0.0
    while traveled <= goal * radius + 1e-6 and len(path) < 96:
        path.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
        theta += direction * step / max(radius, 1e-3)
        traveled += step
    width = LANE_W_DEFAULT
    if len(polys) >= 2:
        radii = []
        for line in polys:
            radii.append(sum(math.hypot(x - cx, y - cy) for x, y in line) / len(line))
        if len(radii) >= 2:
            width = max(LANE_W_MIN, min(LANE_W_MAX, abs(radii[0] - radii[1])))
    return path, width, 1.0 / max(radius, 1e-3)


def _near_separation(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    y_start = max(left[0][1], right[0][1])
    y_end = min(left[-1][1], right[-1][1])
    if y_end - y_start < 0.3:
        return _mean_x(right) - _mean_x(left)
    acc: list[float] = []
    y = y_start
    for _ in range(5):
        if y > y_end + 1e-6:
            break
        acc.append(_boundary_x(right, y) - _boundary_x(left, y))
        y += 1.0
    if not acc:
        return _mean_x(right) - _mean_x(left)
    return sum(acc) / len(acc)


def _one_side(near_x: float, track: LaneTrack | None) -> str:
    """Which boundary a single line is. The last side sticks across x=0.

    A line just left of the car and the same line just right of the car are
    not two lanes. The center is half a width off this line either way, so
    flipping the side jumps the center by a full lane.
    """
    prev = None if track is None else track.side
    if prev == "left" and near_x < ONE_SIDE_DEADBAND_M:
        return "left"
    if prev == "right" and near_x > -ONE_SIDE_DEADBAND_M:
        return "right"
    return "left" if near_x <= 0.0 else "right"


def _one_close_line(
    polys: list[list[tuple[float, float]]],
    track: LaneTrack | None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], float, str] | None:
    """One boundary within 2.5 m, when no pair has a real lane width."""
    nearest: list[tuple[float, float]] | None = None
    nearest_abs: float | None = None
    for poly in polys:
        ordered = _sort_y(poly)
        if len(ordered) < 2:
            continue
        dist = abs(_near_x(ordered))
        if nearest is None or nearest_abs is None or dist < nearest_abs:
            nearest = ordered
            nearest_abs = dist
    # One lane width. The curb-and-wall case sits near 3 m; a 10 m pair's
    # nearer line is past this and is not a boundary.
    if nearest is None or nearest_abs is None or nearest_abs > LANE_W_DEFAULT:
        return None
    side = _one_side(_near_x(nearest), track)
    if side == "left":
        left = nearest
        right = [(x + LANE_W_DEFAULT, y) for x, y in nearest]
    else:
        right = nearest
        left = [(x - LANE_W_DEFAULT, y) for x, y in nearest]
    if track is not None:
        track.side = side
    return left, right, LANE_W_DEFAULT, f"one_{side}"


def _pair(
    polys: list[list[tuple[float, float]]],
    track: LaneTrack | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], float, str] | None:
    if not polys:
        return None
    if len(polys) == 1:
        only = _sort_y(polys[0])
        if len(only) < 2:
            return None
        side = _one_side(_near_x(only), track)
        # Half a lane toward the center. The other boundary is the full
        # width away, so the midpoint is not the paint line itself.
        if side == "left":
            left = only
            right = [(x + LANE_W_DEFAULT, y) for x, y in only]
        else:
            right = only
            left = [(x - LANE_W_DEFAULT, y) for x, y in only]
        if track is not None:
            track.side = side
        return left, right, LANE_W_DEFAULT, f"one_{side}"

    best: tuple[tuple, list, list, float] | None = None
    for i, a in enumerate(polys):
        for b in polys[i + 1 :]:
            sa, sb = _sort_y(a), _sort_y(b)
            if len(sa) < 2 or len(sb) < 2:
                continue
            left, right = (sa, sb) if _near_x(sa) <= _near_x(sb) else (sb, sa)
            # Width at the bumper, not the mean. A camera lane converges at the
            # horizon, so the average gap collapses even when the near field is
            # a full lane.
            width = _near_separation(left, right)
            # 10 m between the two lines is not one lane. Skip it so a closer
            # pair can still win, and so the average of two far lines is not
            # stored as the center.
            if width < LANE_PAIR_W_MIN or width > LANE_PAIR_W_MAX:
                continue
            # A pair that still reaches ahead of the bumper beats a pair that
            # lives entirely behind, even when the rear pair is closer to 3.5 m.
            # A rear-only or beside-only scene still pairs: nothing ahead loses
            # that key, and the width keys below are unchanged.
            ahead = left[-1][1] > 0.0 and right[-1][1] > 0.0
            straddles = _mean_x(left) <= 0.6 and _mean_x(right) >= -0.6
            score = (
                0 if ahead else 1,
                0 if straddles else 1,
                abs(width - LANE_W_DEFAULT),
                abs((_mean_x(left) + _mean_x(right)) * 0.5),
            )
            if best is None or score < best[0]:
                best = (score, left, right, width)
    if best is None:
        return _one_close_line(polys, track)
    left, right, width = best[1], best[2], best[3]
    # x next to the bumper, not the far end of a rear line.
    near_l = min(left, key=lambda p: abs(p[1]))[0]
    near_r = min(right, key=lambda p: abs(p[1]))[0]
    # Two edges on the same side of the car are a curb and a wall, not the
    # ego lane. The 47b430e/ede174a run kept a positive center while the car
    # crossed the road to the right. Use the nearer line, offset by half a
    # lane toward the car, when that line is close enough to be our boundary.
    same_side = near_l > 0.35 or near_r < -0.35
    bad_width = width < LANE_W_MIN or width > LANE_W_MAX
    nearer = left if abs(near_l) <= abs(near_r) else right
    if (same_side or bad_width) and min(abs(near_l), abs(near_r)) < 2.5:
        only = _sort_y(nearer)
        side = _one_side(_near_x(only), track)
        if side == "left":
            left = only
            right = [(x + LANE_W_DEFAULT, y) for x, y in only]
        else:
            right = only
            left = [(x - LANE_W_DEFAULT, y) for x, y in only]
        if track is not None:
            track.side = side
        return left, right, LANE_W_DEFAULT, f"one_{side}"
    if bad_width:
        width = LANE_W_DEFAULT
    return left, right, float(width), "pair"


# Logged lane lines only. A curve at 18 m is wider than this and still a
# lane, so the route keeps the looser per-metre gate. 1.9 widths is 6.65 m:
# the ecf5d35 det_right_x of +8..+16 m is not the ego lane.
LANE_BOUNDARY_WIDTHS = 1.9


def _drop_lane_outliers(
    polys: list[list[tuple[float, float]]],
    *,
    widths: float | None = None,
) -> list[list[tuple[float, float]]]:
    """Drop paint points too far sideways to be this lane, before the fit."""
    kept: list[list[tuple[float, float]]] = []
    for poly in polys:
        pts = [(x, y) for x, y in poly if lane_point_plausible(x, y, widths=widths)]
        if len(pts) >= 2:
            kept.append(pts)
    return kept


def _steer_ref(samples: list[tuple[float, float]]) -> tuple[float, float]:
    """The lookahead the wheel reads: first point at 8 m, else the farthest ahead."""
    ahead = [p for p in samples if p[1] >= 8.0]
    if ahead:
        return ahead[0]
    forward = [p for p in samples if p[1] > 0.0]
    if forward:
        return max(forward, key=lambda p: p[1])
    return samples[0]


def _lane_route(
    polys: list[list[tuple[float, float]]],
    horizon: float,
    step: float,
    lane_track: LaneTrack | None = None,
    remember: bool = True,
) -> tuple[list[tuple[float, float]], float, bool, str] | None:
    polys = _drop_lane_outliers(polys)
    if not polys:
        return None
    prev_side = None if lane_track is None else lane_track.side
    paired = _pair(polys, lane_track)
    if paired is None:
        return None
    left, right, width, base = paired
    # Overlap may start beside or behind the car. Do not clamp it to the bumper.
    y0 = max(left[0][1], right[0][1])
    y1 = min(left[-1][1], right[-1][1])
    if y1 - y0 < 0.5:
        y0 = min(left[0][1], right[0][1])
        y1 = max(left[-1][1], right[-1][1])
    observed: list[tuple[float, float]] = []
    y = y0
    guard = 0
    y_stop = min(y1, horizon)
    while y <= y_stop + 1e-6 and guard < 200:
        xl = _boundary_x(left, y)
        xr = _boundary_x(right, y)
        observed.append(((xl + xr) * 0.5, y))
        y += step
        guard += 1
    if len(observed) < 2:
        return None
    heading = _end_heading(observed)
    curv = _sample_curvature(observed)
    samples = list(observed)
    # Backfill to the bumper so a short visible piece still starts at the car.
    if samples[0][1] > step:
        x0 = samples[0][0]
        prefix: list[tuple[float, float]] = []
        yb = 0.0
        while yb < samples[0][1] - 1e-6:
            prefix.append((x0, yb))
            yb += step
        samples = prefix + samples

    extended = y1 < horizon - step
    x, y, h = observed[-1][0], observed[-1][1], heading
    # Extend along the visible heading while it still points forward. A lane
    # that is still behind the bumper keeps stepping, so the route is not
    # stuck aft of the car. A step that falls behind the bumper without
    # moving forward is the road turning back on itself, and that stops.
    remain = max(0.0, horizon - max(0.0, y))
    guard = 0
    while remain > step * 0.5 and guard < 80:
        h_next = h + curv * step
        if math.cos(h_next) < 0.2:
            break
        h = h_next
        y_prev = y
        x = x + math.sin(h) * step
        y = y + math.cos(h) * step
        if y < -0.5 and y <= y_prev:
            break
        samples.append((x, y))
        extended = True
        remain -= step
        guard += 1
    one_sided = len(polys) < 2
    x_ref, y_ref = _steer_ref(samples)
    source = base
    if not lane_point_plausible(x_ref, y_ref):
        # The fit ran off. Keep the last center if we have one. Do not
        # publish a lookahead tens of metres off the road.
        if lane_track is None or lane_track.center_x is None:
            return None
        chosen = float(lane_track.center_x)
        mode = "hold"
    elif lane_track is None:
        chosen = x_ref
        mode = "take"
    else:
        chosen, mode = follow_lane_center(lane_track, x_ref, count=remember)
    shift = chosen - x_ref
    if abs(shift) > 1e-9:
        samples = [(x + shift, y) for x, y in samples]
    if mode == "hold" and lane_track is not None:
        # A rejected sample must not flip which boundary the next tick trusts.
        lane_track.side = prev_side
    if mode != "take":
        source = f"{base}/{mode}"
    return samples, width, extended or one_sided, source


def _boundary_x(pts: list[tuple[float, float]], y: float) -> float:
    hit = _interp(pts, y)
    if hit is not None:
        return hit
    if y <= pts[0][1]:
        return pts[0][0]
    return pts[-1][0]


def _end_heading(samples: list[tuple[float, float]]) -> float:
    b = samples[-1]
    a = samples[max(0, len(samples) - 4)]
    return math.atan2(b[0] - a[0], b[1] - a[1])


def _sample_curvature(samples: list[tuple[float, float]]) -> float:
    """Curvature from the visible piece. Three points are enough for a tight arc."""
    if len(samples) >= 3:
        circ = _circumcircle(samples[0], samples[len(samples) // 2], samples[-1])
        if circ is not None:
            _cx, _cy, radius = circ
            if 4.0 <= radius <= 80.0:
                ax, ay = samples[0]
                bx, by = samples[len(samples) // 2]
                cx, cy = samples[-1]
                cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                if abs(cross) > 1e-3:
                    # Positive curvature turns right (heading increases).
                    sign = -1.0 if cross > 0.0 else 1.0
                    return max(-0.25, min(0.25, sign / radius))
    if len(samples) < 4:
        return 0.0
    mid = len(samples) // 2
    h0 = math.atan2(samples[mid][0] - samples[0][0], samples[mid][1] - samples[0][1])
    h1 = math.atan2(samples[-1][0] - samples[mid][0], samples[-1][1] - samples[mid][1])
    dist = math.hypot(samples[-1][0] - samples[0][0], samples[-1][1] - samples[0][1])
    if dist < 1.0:
        return 0.0
    return max(-0.25, min(0.25, _wrap(h1 - h0) / dist))


def _path_curvature(path: list[dict[str, float]]) -> float:
    if len(path) < 4:
        return 0.0
    mid = len(path) // 2
    h0 = math.atan2(path[mid]["x"] - path[0]["x"], path[mid]["y"] - path[0]["y"])
    h1 = math.atan2(path[-1]["x"] - path[mid]["x"], path[-1]["y"] - path[mid]["y"])
    dist = 0.0
    for a, b in zip(path, path[1:]):
        dist += math.hypot(b["x"] - a["x"], b["y"] - a["y"])
    if dist < 1.0:
        return 0.0
    return _wrap(h1 - h0) / dist


def _confidence(prediction: str, lane_conf: float, *, extended: bool) -> float:
    if prediction == "roundabout":
        return 0.62
    if prediction == "context" or extended:
        return max(0.5, min(0.8, 0.5 + 0.25 * lane_conf))
    return max(0.5, min(1.0, 0.45 + 0.5 * max(0.0, lane_conf)))


def _unseen_s(track: dict[str, Any]) -> float:
    if track.get("unseen_s") is not None:
        try:
            return max(0.0, float(track["unseen_s"]))
        except (TypeError, ValueError):
            return 0.0
    try:
        misses = float(track.get("misses") or 0.0)
    except (TypeError, ValueError):
        misses = 0.0
    tick = 0.1
    if track.get("tick_s") is not None:
        try:
            tick = max(0.02, float(track["tick_s"]))
        except (TypeError, ValueError):
            tick = 0.1
    return max(0.0, misses) * tick


def _predict_occupants(tracks: list) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in tracks or []:
        if not isinstance(raw, dict):
            continue
        kind = _cls(raw)
        if kind not in _OCCUPANTS:
            continue
        yaw = _f(raw, "yaw", "heading", default=math.pi / 2.0)
        speed = max(0.0, _f(raw, "speed_mps", "speed"))
        unseen = _unseen_s(raw)
        x = _f(raw, "x") + math.cos(yaw) * speed * unseen
        y = _f(raw, "y") + math.sin(yaw) * speed * unseen
        length = _f(raw, "length", "length_m", default=VEHICLE_LENGTH)
        width = _f(raw, "width", "width_m", default=VEHICLE_WIDTH)
        partial = bool(raw.get("partial")) or _f(raw, "seen_fraction", default=1.0) < 0.99
        if partial or length <= 0.0:
            length = max(length, VEHICLE_LENGTH)
        if partial or width <= 0.0:
            width = max(width, VEHICLE_WIDTH)
        if kind == "pedestrian":
            length = max(length, 0.6)
            width = max(width, 0.6)
        out.append(
            {
                "id": raw.get("id"),
                "cls": kind,
                "x": x,
                "y": y,
                "yaw": yaw,
                "length": length,
                "width": width,
                "unseen_s": unseen,
                "partial": partial,
                "speed_mps": speed,
            }
        )
    return out


def _along(path: list[dict[str, float]]) -> list[float]:
    s = [0.0]
    for a, b in zip(path, path[1:]):
        s.append(s[-1] + math.hypot(b["x"] - a["x"], b["y"] - a["y"]))
    return s


def _nearest(path: list[dict[str, float]], x: float, y: float) -> int:
    best_i, best_d = 0, 1e18
    for i, p in enumerate(path):
        d = (p["x"] - x) ** 2 + (p["y"] - y) ** 2
        if d < best_d:
            best_i, best_d = i, d
    return best_i


def _apply_scene(
    path: list[dict[str, float]],
    occupants: list[dict[str, Any]],
    signs: list,
    width: float,
    *,
    ego_speed_mps: float,
) -> tuple[list[dict[str, float]], str, int | None, float | None, list[dict[str, Any]], float | None, float | None]:
    cuts: list[tuple[float, str, int | None, float | None, float]] = []
    occupied: list[dict[str, Any]] = []
    station = _along(path)
    for occ in occupants:
        idx = _nearest(path, occ["x"], occ["y"])
        lateral = math.hypot(path[idx]["x"] - occ["x"], path[idx]["y"] - occ["y"])
        reach = (EGO_WIDTH + float(occ["width"])) * 0.5 + 0.2
        if lateral > reach:
            continue
        # Behind the bumper only. y=0 is in the path; a nearest vertex at the origin
        # must not hide it.
        if float(occ["y"]) < -0.05:
            continue
        along = float(occ.get("speed_mps") or 0.0) * math.sin(float(occ.get("yaw") or (math.pi / 2.0)))
        closing = float(ego_speed_mps) - along
        # Matching speed is not a stop. Anything actually in front of the bumper is.
        if closing <= 0.4 and float(occ["y"]) > 4.0:
            continue
        back = float(occ["length"]) * 0.5 + STANDOFF_M
        cut_s = station[idx] - back
        occupied.append(dict(occ))
        ttc = None if closing <= 0.05 else max(0.0, cut_s) / max(closing, 0.1)
        ident = occ.get("id")
        try:
            ident = None if ident is None else int(ident)
        except (TypeError, ValueError):
            ident = None
        kind = str(occ.get("cls") or "vehicle")
        reason = kind if kind in ("pedestrian", "bike") else "vehicle"
        cuts.append((cut_s, reason, ident, ttc, closing))

    for raw in signs or []:
        if not isinstance(raw, dict):
            continue
        if _sign_misses(raw) > _SIGN_MISS_MAX:
            continue
        kind = _cls(raw)
        x, y = _f(raw, "x"), _f(raw, "y")
        idx = _nearest(path, x, y)
        lateral = math.hypot(path[idx]["x"] - x, path[idx]["y"] - y)
        if station[idx] < 0.3:
            continue
        if kind in ("stop_sign", "stop"):
            if lateral > _sign_gate(SHOULDER_M, raw):
                continue
            cuts.append((
                station[idx] - SIGN_STANDOFF_M, "stop_sign", None,
                _sign_ttc(station[idx], ego_speed_mps), float(ego_speed_mps),
            ))
        elif kind in ("yield", "yield_sign"):
            if lateral > _sign_gate(SHOULDER_M, raw):
                continue
            # Same reach as a stop sign on this route. Brake is scaled from the
            # remaining distance, not held at one value.
            cuts.append((station[idx] - SIGN_STANDOFF_M, "yield", None, None, float(ego_speed_mps)))
        elif kind in ("traffic_light", "light"):
            if lateral > _sign_gate(LIGHT_GATE_M, raw):
                continue
            state = str(raw.get("state") or "").strip().lower()
            if state in ("red", "r"):
                cuts.append((
                    station[idx] - SIGN_STANDOFF_M, "red_light", None,
                    _sign_ttc(station[idx], ego_speed_mps), float(ego_speed_mps),
                ))
            elif state in ("yellow", "amber") and station[idx] < 22.0:
                cuts.append((
                    station[idx] - SIGN_STANDOFF_M, "yellow_light", None,
                    _sign_ttc(station[idx], ego_speed_mps), float(ego_speed_mps),
                ))
    if not cuts:
        return path, "none", None, None, occupied, None, None
    cut_s, reason, cipv_id, ttc, closing = min(cuts, key=lambda c: c[0])
    stop_dist = max(0.0, cut_s)
    trimmed = [p for p, s in zip(path, station) if s <= stop_dist + 1e-6]
    if len(trimmed) < 2:
        trimmed = [path[0], path[1 if len(path) > 1 else 0]]
        if trimmed[0] is trimmed[1]:
            trimmed = [path[0], _pt(path[0]["x"], path[0]["y"] + 0.4)]
    return trimmed, reason, cipv_id, ttc, occupied, stop_dist, closing


def _limit_mps(signs: list, path: list[dict[str, float]]) -> float | None:
    if not path:
        return None
    cap = None
    for raw in signs or []:
        if not isinstance(raw, dict):
            continue
        if _cls(raw) not in ("speed_limit", "speed"):
            continue
        limit = raw.get("limit_mps", raw.get("speed_mps", raw.get("limit")))
        if limit is None:
            continue
        try:
            value = float(limit)
        except (TypeError, ValueError):
            continue
        idx = _nearest(path, _f(raw, "x"), _f(raw, "y"))
        lateral = math.hypot(path[idx]["x"] - _f(raw, "x"), path[idx]["y"] - _f(raw, "y"))
        if lateral > SHOULDER_M:
            continue
        cap = value if cap is None else min(cap, value)
    return cap


def _sign_misses(raw: dict[str, Any]) -> int:
    try:
        return max(0, int(raw.get("misses") or 0))
    except (TypeError, ValueError):
        return 0


def _sign_partial(raw: dict[str, Any]) -> bool:
    if bool(raw.get("partial")):
        return True
    try:
        frac = float(raw.get("seen_fraction"))
    except (TypeError, ValueError):
        return False
    return frac < 0.99


def _sign_gate(base: float, raw: dict[str, Any]) -> float:
    """A partly seen sign may sit just outside the measured shoulder."""
    return base + (_PARTIAL_GATE_M if _sign_partial(raw) else 0.0)


def _sign_ttc(station: float, ego_speed_mps: float) -> float | None:
    if ego_speed_mps <= 0.05:
        return None
    return max(0.0, station - SIGN_STANDOFF_M) / ego_speed_mps


def _speed_flags(
    *,
    curv: float,
    stop_reason: str,
    ttc: float | None,
    stop_dist: float | None,
    closing_mps: float | None,
    ego_speed_mps: float,
    signs: list,
    path: list[dict[str, float]],
) -> tuple[str, float, float | None]:
    """Brake grows as the gap shrinks and as closing speed rises. Far stops do not slam to 1."""
    dist = None if stop_dist is None else max(0.0, float(stop_dist))
    closing = None if closing_mps is None else float(closing_mps)
    if closing is None and dist is not None and ttc is not None and ttc > 0.05:
        closing = dist / ttc
    hard = False
    if dist is not None and stop_reason not in ("none", "yield"):
        close = 0.0 if closing is None else closing
        lead = float(ego_speed_mps) - close
        stopped = lead <= 0.4
        # A bumper gap is a full stop only when that lead is actually stopped.
        # A 1 m/s close inside 1.5 m stays a scaled brake.
        if stopped and dist <= 1.5 and float(ego_speed_mps) > 0.5:
            hard = True
        elif ttc is not None and ttc < _FULL_BRAKE_TTC and close >= 3.0:
            hard = True
    # plan_command treats ttc under 1.2 as a full brake. Keep that number only
    # when this stop is actually in that band.
    reported_ttc = ttc if hard else (None if ttc is not None and ttc < _FULL_BRAKE_TTC else ttc)
    aeb = "brake" if hard else "off"
    speed = plan_speed(
        ego_speed_mps=ego_speed_mps,
        curvature=curv,
        ttc_lead=reported_ttc,
        aeb=aeb,
        v_cap=V_CAP,
    )
    target = speed.target_v
    if dist is not None and stop_reason not in ("none",) and not hard:
        if stop_reason == "yield":
            target = min(target, max(1.5, 0.45 * dist))
        else:
            target = min(target, math.sqrt(max(0.0, 2.0 * _COMFORT_DECEL * dist)))
    limit = _limit_mps(signs, path)
    if limit is not None:
        target = min(target, max(0.0, limit))
    if aeb == "brake":
        target = 0.0
    return aeb, float(target), reported_ttc
