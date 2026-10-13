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

from python.perception.track import PRIOR_RATE_SIGMA_MPS, _RATE_VALID_SIGMA
from python.planning.speed import plan_speed

LANE_W_DEFAULT = 3.5
LANE_W_MIN = 2.4
LANE_W_MAX = 5.2
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
_VEHICLES = frozenset({"vehicle", "car", "truck", "bus"})
_VRU = frozenset({"pedestrian", "bike"})
# Time-to-collision full brake. A vehicle rate can arm this. A pedestrian
# or bike in the path arms it from the gap alone.
_FULL_BRAKE_TTC = 1.2
# Brake 1.0 is 8 m/s². The range law is required decel from a conservative
# closing speed, divided by that 8. A small request stays under the comfort
# decel. A large one saturates at 1. There is no distance wall.
_BRAKE_DECEL = 8.0
_RANGE_RESERVE_M = 3.0
_COMFORT_DECEL = 2.5
# Conservative closing is the range-rate mean minus k standard deviations
# (more negative means closing faster). k near 1. 0.85 keeps a ±2 m
# pacing lead at a 1.5 s gap inside a 1 m/s speed change; 1.0 shed just
# over that on the first uncertain frames.
_CLOSING_K = 0.85
# A pedestrian or bike already inside this remaining distance keeps the
# brake until the car is stopped, even when the close has fallen off.
# It is the standoff box in front of the bumper, not a lead-gap gate.
_VRU_HOLD_M = STANDOFF_M + VEHICLE_LENGTH * 0.5
# A slower lead is matched only after the filtered close has held this
# long. Correlated range noise reaches a second of mild closing and lets
# go. A real 2 m/s close does not.
_CLOSE_HOLD_S = 1.5
# Route-gap drop. A 0.40 s curvature under ±1 m is about 31 m/s² of
# noise, so a real 6 m/s² brake never clears it. The arm is the least-
# squares slope over a second, which has to be steeper than a 1.2 m wave
# on a 4 s period (about −1.9 m/s). A half-second slope sees the same
# brake sooner, and inside the last 12 m that slope can be milder because
# the wave lives farther out. Once either slope holds, the pedal stays
# down until the filtered rate has been flat.
_DROP_WIN_S = 1.0
_DROP_ARM_MPS = -2.3
_DROP_FAST_S = 0.50
_DROP_FAST_MPS = -2.8
# One frame at 10 Hz and at 15 Hz, inside the short gap only. Farther
# out, one noisy half-second at ±2 m is not a lead brake, so that arm
# waits. A second frame of the same slope was the difference between
# stopping short of a 12.5 m lead and touching it.
_DROP_ARM_S = 0.06
_DROP_HOLD_S = 0.15
# Inside this station a held 5 m/s² still loses to a lead braking at 6.
# The half-second slope is still catching up, so the short gap arms on a
# milder slope once the filtered rate is already closing. The cap stays
# under the line that promotes aeb. A 27 m wander is outside this station.
_DROP_HARD_M = 12.0
_DROP_HARD_A = 7.6
_DROP_SHORT_MPS = -1.7
# A locked decel inside this station. 18 m was a ±2 m grab on a matched
# 18 m pace, and the 5 m/s² floor then shed more than that pace can spare.
_DROP_LEAD_M = 15.0
# Release only after the filtered rate has been flat or opening this long.
# One shallow slope window is not a release, and neither is a single frame.
# A rate that is clearly opening, or a route slope that has been opening
# for this long, lets go on that frame instead of waiting out a flat spell.
_DROP_RELEASE = -0.25
_DROP_RELEASE_S = 0.50
_DROP_OPEN_MPS = 0.5
# Inside 15 m the arm has to clear the noise, not a fixed 1 m/s. A ±2 m
# grab at 12 m has a sigma near 2, and −1 m/s of mean is still inside it.
_DROP_ARM_PAD = 0.4
# Comfort is a wall. The lower bound on closing (mean minus k sigma,
# written as a positive closing speed) has to clear this, or the drop
# has to be locked. A 4 m same-track jump peaks near 2.1 m/s on a
# matched lead and has to follow that lead instead.
_COMFORT_CLOSE_MPS = 2.5
# An unconfirmed rate must not promote aeb. Inside this gap the bumper
# is close enough that the full brake still stands.
_AEB_CONFIRM_M = 5.0
# Slow-side closing (mean plus k sigma, as a positive close) that is
# still a real stop while sigma itself is not valid yet. Below this, a
# short gap contributes no pedal: ±2 m at 12 m looks like several m/s
# of close for most of a second. A longer gap keeps a light pedal so a
# stopped lead is already slowing when the filter settles.
_UNCERTAIN_CLOSE_MPS = 3.0
_UNCERTAIN_LIGHT_M = 18.0
# Under the 0.99 line that promotes aeb. The drop alone must not disengage.
_DROP_CAP = 0.98
_SWAP_JUMP_M = 2.2
# Comfort stop once the cut is inside this remaining distance. Farther out,
# matching a lead that still has road speed keeps a wander from being a wall.
# 16 m of cut is a lead whose bumper is still around 22 m: the 16 m and 20 m
# brakes live here, and a 27 m wander does not.
_COMFORT_CUT_M = 16.0
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
    # Extra brake in [0, 1] from the range/range-rate law. 0 leaves the
    # speed-error brake alone. It reaches 1 when the gap demands a full stop.
    range_brake: float = 0.0
    # Full polyline before a lead trims it. Lateral tests use this, because
    # the trimmed end sits short of the bumper and misses the car.
    route_ego: list[dict[str, float]] = field(default_factory=list)

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
            "range_brake": self.range_brake,
        }


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
) -> PathPlan:
    """Predict a drivable route from the scene. Wheel angle is not an argument."""
    del curvature  # image-slope hint is not the route; the polyline is
    step = max(0.4, float(step_m))
    horizon = max(8.0, float(length_m))
    live = _polylines(lanes_bev)
    used_prior = False
    polys = live
    if not polys:
        polys = _polylines(prior_lanes)
        used_prior = bool(polys)

    circle = None if _is_forward_lane(polys) else (_fit_roundabout(polys) if polys else None)
    ring_curv = 0.0
    if circle is not None:
        raw, width, ring_curv = _roundabout_route(circle, polys, step)
        prediction = "roundabout"
        extended = True
    else:
        built = _lane_route(polys, horizon, step) if polys else None
        if built is None:
            return _expose_pedals(_empty(), ego_speed_mps)
        raw, width, extended = built
        if used_prior:
            prediction = "context"
        elif extended:
            prediction = "extended"
        else:
            prediction = "lanes"

    path = [_pt(x, y) for x, y in raw]
    if len(path) < 2:
        return _expose_pedals(_empty(), ego_speed_mps)
    route = list(path)

    # On the full route, before a lead trims it. The trimmed end sits short
    # of the bumper, and a lateral test against that end would miss the car.
    kalman_brake, drop_brake = _lead_range_brake(tracks or [], route, float(ego_speed_mps))
    curv = ring_curv if prediction == "roundabout" and ring_curv > 1e-4 else _path_curvature(path)
    occupants = _predict_occupants(tracks or [])
    path, stop_reason, cipv_id, ttc, occupied, stop_dist, closing, comfort_stop = _apply_scene(
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
        comfort_stop=comfort_stop,
    )
    # Kalman required-decel can still be a full stop once the rate sigma is
    # inside the valid band, or the gap is already inside 5 m. Farther out,
    # an unconfirmed rate is capped at 0.98 in _required_brake, so this 0.99
    # line cannot promote aeb from noise. A route-gap drop is graded and
    # capped under that, so it cannot promote aeb on its own.
    range_brake = 0.0
    if aeb != "brake":
        if kalman_brake >= 0.99:
            aeb = "brake"
            target = 0.0
        else:
            range_brake = max(kalman_brake, min(_DROP_CAP, drop_brake))
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
        range_brake=range_brake,
        route_ego=route,
    )
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


def _pair(
    polys: list[list[tuple[float, float]]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], float] | None:
    if not polys:
        return None
    if len(polys) == 1:
        only = _sort_y(polys[0])
        if len(only) < 2:
            return None
        if _near_x(only) <= 0.0:
            left, right = only, [(x + LANE_W_DEFAULT, y) for x, y in only]
        else:
            right, left = only, [(x - LANE_W_DEFAULT, y) for x, y in only]
        return left, right, LANE_W_DEFAULT

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
            if width < 1.2:
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
        return None
    width = best[3]
    if width < LANE_W_MIN or width > LANE_W_MAX:
        width = LANE_W_DEFAULT
    return best[1], best[2], float(width)


def _lane_route(
    polys: list[list[tuple[float, float]]],
    horizon: float,
    step: float,
) -> tuple[list[tuple[float, float]], float, bool] | None:
    paired = _pair(polys)
    if paired is None:
        return None
    left, right, width = paired
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
    return samples, width, extended or one_sided


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


def _rate_and_sigma(raw: dict[str, Any], ego_speed_mps: float) -> tuple[float, float]:
    """Signed range-rate and its standard deviation.

    A missing ``range_rate_valid`` is an older caller that already put a
    road speed on the track: sigma is 0 and the rate is speed minus ego.
    ``False`` with no sigma is the first-sample prior, rate 0, not a stop.
    """
    valid = raw.get("range_rate_valid")
    sigma_raw = raw.get("range_rate_sigma")
    rate_raw = raw.get("range_rate_mps")
    has_sigma = False
    sigma = 0.0
    if sigma_raw is not None:
        try:
            sigma = max(0.0, float(sigma_raw))
            has_sigma = True
        except (TypeError, ValueError):
            has_sigma = False
    has_rate = False
    rate = 0.0
    if rate_raw is not None:
        try:
            rate = float(rate_raw)
            has_rate = True
        except (TypeError, ValueError):
            has_rate = False
    # A filter publishes both, including while road speed is still withheld.
    # The mean is the closing estimate. Sigma is the uncertainty around it.
    if has_rate or has_sigma:
        if not has_sigma:
            sigma = PRIOR_RATE_SIGMA_MPS if valid is False else 0.0
        return rate, sigma
    if valid is False:
        return 0.0, PRIOR_RATE_SIGMA_MPS
    return _f(raw, "speed_mps", "speed") - float(ego_speed_mps), 0.0


def _num(value: Any, default: float | None = None) -> float | None:
    """Finite float, or None when the value is missing or not a number."""
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def vehicle_on_route(
    x: float,
    y: float,
    path: list[dict[str, float]],
    width_m: float | None = None,
) -> bool:
    """Same lateral test the range brake uses. Off the polyline is not a lead.

    A straight ``|x| < 1.4`` tube still contains a car in the next lane once
    an 80 m curve has bent the route out from under it.
    """
    if not path:
        return False
    if not math.isfinite(x) or not math.isfinite(y) or y < -0.05:
        return False
    idx = _nearest(path, x, y)
    px = _num(path[idx].get("x"))
    py = _num(path[idx].get("y"))
    if px is None or py is None:
        return False
    lateral = math.hypot(px - x, py - y)
    if not math.isfinite(lateral):
        return False
    half = VEHICLE_WIDTH if width_m is None else width_m
    if not math.isfinite(half) or half <= 0.0:
        half = VEHICLE_WIDTH
    reach = max(
        (EGO_WIDTH + VEHICLE_WIDTH) * 0.5 + 0.2,
        (EGO_WIDTH + half) * 0.5 + 0.2,
    )
    return lateral <= reach


def _required_brake(gap_m: float, rate_mps: float, sigma_mps: float) -> float:
    """Brake in [0, 1] from required decel.

    ``closing`` is the range-rate mean pulled toward a faster close by
    ``k`` standard deviations. ``a_req = closing² / (2 · (gap − reserve))``.
    Brake is ``a_req / 8``: under the comfort decel while the request is
    small, and 1 when the request reaches the brake limit. The reserve is
    the gap the formula keeps, not an 8 m wall. A gap inside the reserve
    is already at the bumper. A non-finite gap is not a bumper.
    """
    gap = _num(gap_m)
    rate = _num(rate_mps)
    sigma = _num(sigma_mps)
    if gap is None or rate is None or sigma is None:
        return 0.0
    if gap <= _RANGE_RESERVE_M:
        return 1.0
    usable = max(0.25, gap - _RANGE_RESERVE_M)
    # A confirmed rate brakes the fast side of the noise. An unconfirmed
    # rate uses that same side, but the pedal stays at or under 0.98 so
    # it cannot promote aeb. The slow side has to show a real close
    # before that pedal is allowed. A short gap whose slow side is still
    # inside the noise is not a stop. A longer gap keeps a light pedal
    # so a stopped lead is already slowing while sigma settles.
    uncertain = sigma > _RATE_VALID_SIGMA and gap > _AEB_CONFIRM_M
    signed = max(0.0, sigma)
    closing = max(0.0, -(rate - _CLOSING_K * signed))
    confirmed_close = max(0.0, -(rate + _CLOSING_K * signed))
    if not math.isfinite(closing) or not math.isfinite(usable) or usable <= 0.0:
        return 0.0
    a_req = (closing * closing) / (2.0 * usable)
    if not math.isfinite(a_req):
        return 0.0
    brake = max(0.0, min(1.0, a_req / _BRAKE_DECEL))
    if uncertain:
        if confirmed_close >= _UNCERTAIN_CLOSE_MPS:
            cap = _DROP_CAP
        elif gap >= _UNCERTAIN_LIGHT_M:
            cap = 0.20
        else:
            cap = 0.0
        brake = min(brake, cap)
    return brake


def _route_station(path: list[dict[str, float]], x: float, y: float) -> float | None:
    """Arc length to the closest point on the polyline, not the nearest vertex."""
    if len(path) < 2:
        return None
    best_d = 1e18
    best_s = None
    travelled = 0.0
    for a, b in zip(path, path[1:]):
        ax = _num(a.get("x"))
        ay = _num(a.get("y"))
        bx = _num(b.get("x"))
        by = _num(b.get("y"))
        if ax is None or ay is None or bx is None or by is None:
            continue
        dx = bx - ax
        dy = by - ay
        seg2 = dx * dx + dy * dy
        seg = math.sqrt(seg2) if seg2 > 0.0 else 0.0
        if seg2 < 1e-12:
            t = 0.0
        else:
            t = ((x - ax) * dx + (y - ay) * dy) / seg2
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
        px = ax + t * dx
        py = ay + t * dy
        dist = (px - x) ** 2 + (py - y) ** 2
        if dist < best_d:
            best_d = dist
            best_s = travelled + seg * t
        travelled += seg
    return best_s


def _window_points(
    series: list, now: float, window: float,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for item in series:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        t = _num(item[0])
        s = _num(item[1])
        if t is None or s is None:
            continue
        if now - window - 1e-9 <= t <= now + 1e-6:
            out.append((t, s))
    return out


def _ls_slope(pts: list[tuple[float, float]]) -> float | None:
    if len(pts) < 3:
        return None
    span = pts[-1][0] - pts[0][0]
    if span < 0.12:
        return None
    n = float(len(pts))
    mt = sum(t for t, _ in pts) / n
    ms = sum(s for _, s in pts) / n
    var_t = sum((t - mt) ** 2 for t, _ in pts)
    if var_t < 1e-8:
        return None
    cov = sum((t - mt) * (s - ms) for t, s in pts)
    return cov / var_t


def _smooth_gap(series: list, k: int = 3) -> list[tuple[float, float]]:
    """Causal moving average so one noisy grab does not flip the curvature."""
    raw: list[tuple[float, float]] = []
    for item in series:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        t = _num(item[0])
        s = _num(item[1])
        if t is None or s is None:
            continue
        raw.append((t, s))
    out: list[tuple[float, float]] = []
    for i, (t, _s) in enumerate(raw):
        chunk = raw[max(0, i - k + 1): i + 1]
        out.append((t, sum(v for _, v in chunk) / len(chunk)))
    return out


def _record_route_gap(raw: dict[str, Any], station: float, now: float) -> list | None:
    """Append this frame's arc station. A swap clears the list first."""
    series = raw.get("route_gaps")
    state = raw.get("drop_state")
    if not isinstance(series, list) or not isinstance(state, list) or len(state) < 2:
        return None
    if series:
        prev_t = _num(series[-1][0]) if isinstance(series[-1], (tuple, list)) and series[-1] else None
        prev_s = _num(series[-1][1]) if isinstance(series[-1], (tuple, list)) and len(series[-1]) > 1 else None
        if prev_t is not None and abs(prev_t - now) < 1e-4:
            return series
        rate_now = _num(raw.get("range_rate_mps"))
        already_closing = raw.get("range_rate_valid") is True and (
            rate_now is not None and rate_now <= -0.4
        )
        if prev_s is not None and abs(station - prev_s) >= _SWAP_JUMP_M and not already_closing:
            series.clear()
            for i in range(len(state)):
                state[i] = 0.0
    series.append((float(now), float(station)))
    if len(series) > 48:
        del series[:-48]
    return series


def _drop_slope(series: list, now: float, window: float) -> float | None:
    """Closing rate of the route gap over ``window`` seconds.

    A window that has not covered most of that span is not a slope yet.
    One grab, including a 1.6 m step, does not fill a half-second slope
    by itself once the span test is the window.
    """
    pts = _window_points(_smooth_gap(series, 3), now, window)
    if len(pts) < 3 or pts[-1][0] - pts[0][0] < 0.65 * window:
        return None
    slope = _ls_slope(pts)
    if slope is None or not math.isfinite(slope):
        return None
    return slope


def _drop_brake(raw: dict[str, Any], station: float, now: float) -> float:
    """Graded brake from a one-second route-gap slope, then held.

    Arms when that slope is steeper than a 1.2 m / 4 s wave and stays
    there. A swap clears the series before this runs. The pedal stays
    down across a shallow window. It comes off only after the filtered
    range-rate has been flat or opening for half a second. Nothing here
    is stored as a full stop, and nothing here sets ``aeb``.
    """
    series = _record_route_gap(raw, station, now)
    state = raw.get("drop_state")
    if series is None or not isinstance(state, list) or len(state) < 2:
        return 0.0
    while len(state) < 3:
        state.append(0.0)
    slow = _drop_slope(series, now, _DROP_WIN_S)
    fast = _drop_slope(series, now, _DROP_FAST_S)
    rate = _num(raw.get("range_rate_mps"))
    valid = raw.get("range_rate_valid") is True
    dt = 0.1
    if len(series) >= 2:
        t1 = _num(series[-1][0]) if isinstance(series[-1], (tuple, list)) else None
        t0 = _num(series[-2][0]) if isinstance(series[-2], (tuple, list)) else None
        if t1 is not None and t0 is not None:
            dt = max(1e-3, min(0.25, t1 - t0))
    locked = _num(state[0]) or 0.0
    hot = _num(state[1]) or 0.0
    flat_s = _num(state[2]) or 0.0
    sigma = _num(raw.get("range_rate_sigma"))
    if sigma is None or not math.isfinite(sigma):
        sigma = 0.0
    sigma = max(0.0, sigma)
    # Clearly opening: the pedal comes off on this frame. A positive route
    # slope has to hold, the same way a flat rate does, before it releases.
    slope_open = (slow is not None and slow > 0.0) or (
        slow is None and fast is not None and fast > 0.0
    )
    if locked >= 1.0:
        if valid and rate is not None and rate > _DROP_OPEN_MPS:
            state[0] = 0.0
            state[1] = 0.0
            state[2] = 0.0
            return 0.0
        rate_flat = valid and rate is not None and rate >= _DROP_RELEASE
        if rate_flat or slope_open:
            flat_s += dt
        else:
            flat_s = 0.0
        if flat_s >= _DROP_RELEASE_S:
            state[0] = 0.0
            state[1] = 0.0
            state[2] = 0.0
            return 0.0
    else:
        flat_s = 0.0
    long_steep = (fast is not None and fast <= _DROP_FAST_MPS) or (
        slow is not None and slow <= _DROP_ARM_MPS
    )
    # A pace whose rate is still flat can throw a steep half-second.
    # That is not a lead brake. Once the filtered rate is closing, it is.
    rate_closing = valid and rate is not None and rate <= -0.8
    # Inside 15 m the mean has to clear k sigma plus a small pad before
    # a new lock. A fixed −1 m/s line sits inside a ±2 m grab. The short
    # stop still arms on −1 m/s once the slope is already steep: its
    # rate is only about −1.2 on the frame the gap would otherwise be
    # given up, and the floor below does not stay unless that close
    # keeps deepening.
    beyond = -(_DROP_ARM_PAD + _CLOSING_K * sigma)
    rate_beyond = valid and rate is not None and rate <= beyond
    short_zone = station <= _DROP_LEAD_M
    rate_early = valid and rate is not None and rate <= -1.0
    rate_hard = rate_early if station <= _DROP_HARD_M else (
        rate_beyond if short_zone else rate_closing
    )
    # The 12.5 m stop is already gone by the time the half-second slope
    # clears −2.8. Inside that station the milder slope is enough, and
    # only after the filtered rate itself is closing past the noise.
    short_steep = (
        station <= _DROP_HARD_M
        and rate_hard
        and fast is not None
        and fast <= _DROP_SHORT_MPS
    )
    steep = long_steep or short_steep
    if steep and (locked >= 1.0 or rate_hard):
        hot += dt
    elif locked < 1.0:
        hot = max(0.0, hot - dt)
    was_locked = locked >= 1.0
    if hot >= (_DROP_ARM_S if station <= _DROP_HARD_M else _DROP_HOLD_S) and steep and (
        locked >= 1.0 or rate_hard
    ):
        # Farther out, the first lock is 5 m/s². Inside the last 12 m the
        # same slope is a lead that is already braking harder than that,
        # so the hold goes to the cap. Either way the pedal stays at 0.98.
        floor = _DROP_HARD_A if station <= _DROP_HARD_M else 5.0
        locked = min(_DROP_HARD_A, max(locked, floor))
    state[0] = float(locked)
    state[1] = float(hot)
    state[2] = float(flat_s)
    if locked < 1.0:
        return 0.0
    if station <= _RANGE_RESERVE_M:
        return _DROP_CAP
    usable = max(0.25, station - _RANGE_RESERVE_M)
    slope = slow if fast is None else fast if slow is None else min(slow, fast)
    closing = 0.0 if slope is None else max(0.0, -slope)
    # A noise dip can match a lead brake for one frame and then let go.
    # The floor stays for that opening frame. It stays after that when
    # the one-second slope is already a close and the filtered rate is
    # still past 1.5 m/s, or when the rate itself has become a hard
    # close. A ±2 m grab does not keep both. An opening rate drops the
    # floor on this frame.
    deep_close = valid and rate is not None and rate <= -2.5
    # hot can sit at one frame after the slope lets go. That is not a
    # new arm. Only the frame that raises the lock counts.
    just_armed = locked >= 1.0 and not was_locked
    # Inside the short gap the one-second slope only has to be a real
    # close. Farther out, a wander sits on about −2 m/s and must not
    # keep the floor; a lead brake is steeper than that.
    if station <= _DROP_HARD_M:
        # Both windows. A dip whose half-second slope has already
        # turned is not still a lead brake, even if the one-second
        # slope is still negative.
        slow_closing = (
            slow is not None
            and slow <= -1.2
            and fast is not None
            and fast <= _DROP_SHORT_MPS
        )
    else:
        # A 14 m wander reached about −3.5 m/s on the one-second slope.
        # A lead braking at 6 m/s² is steeper than −4 once that window fills.
        slow_closing = slow is not None and slow <= -4.0
    rate_held = valid and rate is not None and rate <= -1.5
    still_closing = deep_close or (
        just_armed and valid and rate is not None and rate < 0.0
    ) or (slow_closing and rate_held)
    lead_a = (
        min(locked, _DROP_HARD_A)
        if station <= _DROP_LEAD_M and still_closing
        else 0.0
    )
    # The opening frame of a noise dip is not yet the 7.6 m/s² hold.
    # A real brake is already on the one-second slope, or it deepens
    # on the next frames and the hold catches up.
    if just_armed and not deep_close and not (slow_closing and rate_held):
        lead_a = min(lead_a, 5.0)
    a_req = lead_a + (closing * closing) / (2.0 * usable)
    if not math.isfinite(a_req):
        return 0.0
    return max(0.0, min(_DROP_CAP, a_req / _BRAKE_DECEL))


def _lead_range_brake(
    tracks: list, path: list[dict[str, float]], ego_speed_mps: float,
) -> tuple[float, float]:
    """Kalman brake, and a graded route-gap brake that cannot arm AEB alone.

    Anything still in front of the bumper counts, including a lead under
    4 m. Confirmed and unconfirmed leads both use the same Kalman law.
    The second value uses arc distance along ``path``. Ego-frame y shrinks
    in a curve while that arc length does not, so a bend is not a drop.
    """
    ego = _num(ego_speed_mps)
    if not path or ego is None or ego < 0.05:
        return 0.0, 0.0
    kalman = 0.0
    drop = 0.0
    for raw in tracks or []:
        if not isinstance(raw, dict):
            continue
        if _cls(raw) not in _VEHICLES:
            continue
        y = _num(raw.get("y"))
        x = _num(raw.get("x"))
        if y is None or x is None or y < -0.05:
            continue
        half = _num(raw.get("width"), None)
        if half is None:
            half = _num(raw.get("width_m"), VEHICLE_WIDTH)
        if not vehicle_on_route(x, y, path, half):
            continue
        rate, sigma = _rate_and_sigma(raw, ego_speed_mps)
        station = _route_station(path, x, y)
        # The filter is on ego-frame range. Its brake stays on that range.
        # The drop below is the arc length, which does not shrink in a bend.
        kalman = max(kalman, _required_brake(y, rate, sigma))
        samples = raw.get("samples")
        now = None
        if isinstance(samples, list) and samples:
            last = samples[-1]
            if isinstance(last, (tuple, list)) and last:
                now = _num(last[0])
        if station is not None and now is not None:
            drop = max(drop, _drop_brake(raw, station, now))
    return kalman, drop


def _predict_occupants(tracks: list) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in tracks or []:
        if not isinstance(raw, dict):
            continue
        kind = _cls(raw)
        if kind not in _OCCUPANTS:
            continue
        # A vehicle at speed 0 before a range-rate exists is not a stopped
        # lead. Pedestrians and bikes in the path still are: the gap alone
        # is the brake, on the first sample.
        unconfirmed = raw.get("range_rate_valid") is False
        if unconfirmed and kind not in _VRU:
            continue
        yaw = _f(raw, "yaw", "heading", default=math.pi / 2.0)
        speed = 0.0 if unconfirmed else max(0.0, _f(raw, "speed_mps", "speed"))
        sigma = 0.0
        if raw.get("range_rate_sigma") is not None:
            try:
                sigma = max(0.0, float(raw.get("range_rate_sigma")))
            except (TypeError, ValueError):
                sigma = 0.0
            if not math.isfinite(sigma):
                sigma = 0.0
        unseen = _unseen_s(raw)
        x0 = _num(raw.get("x"))
        y0 = _num(raw.get("y"))
        if x0 is None or y0 is None:
            continue
        x = x0 + math.cos(yaw) * speed * unseen
        y = y0 + math.sin(yaw) * speed * unseen
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        length = _f(raw, "length", "length_m", default=VEHICLE_LENGTH)
        width = _f(raw, "width", "width_m", default=VEHICLE_WIDTH)
        if not math.isfinite(length):
            length = VEHICLE_LENGTH
        if not math.isfinite(width):
            width = VEHICLE_WIDTH
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
                "range_rate_mps": raw.get("range_rate_mps"),
                "range_rate_sigma": sigma,
                "close_hold_s": raw.get("close_hold_s") or 0.0,
                "drop_state": raw.get("drop_state"),
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
) -> tuple[
    list[dict[str, float]], str, int | None, float | None,
    list[dict[str, Any]], float | None, float | None, bool,
]:
    cuts: list[tuple[float, str, int | None, float | None, float, bool]] = []
    occupied: list[dict[str, Any]] = []
    station = _along(path)
    for occ in occupants:
        if not math.isfinite(float(occ["x"])) or not math.isfinite(float(occ["y"])):
            continue
        idx = _nearest(path, occ["x"], occ["y"])
        lateral = math.hypot(path[idx]["x"] - occ["x"], path[idx]["y"] - occ["y"])
        if not math.isfinite(lateral):
            continue
        reach = (EGO_WIDTH + float(occ["width"])) * 0.5 + 0.2
        if lateral > reach:
            continue
        # Behind the bumper only. y=0 is in the path; a nearest vertex at the origin
        # must not hide it.
        if float(occ["y"]) < -0.05:
            continue
        # speed_mps is the lead's road speed. Closing is ego minus that,
        # and time-to-collision is the gap divided by closing.
        along = float(occ.get("speed_mps") or 0.0) * math.sin(float(occ.get("yaw") or (math.pi / 2.0)))
        closing = float(ego_speed_mps) - along
        kind = str(occ.get("cls") or "vehicle")
        sigma = float(occ.get("range_rate_sigma") or 0.0)
        # Matching speed is not a stop. A rate that is still inside its own
        # noise is not a close either: the mean can read 1 m/s of closing on
        # a lead that is keeping pace, and the comfort stop would then shed
        # speed the range law never asked for. The brake law uses the other
        # side of the same sigma. A pedestrian or bike already inside the
        # stop box still counts.
        optimistic = closing - _CLOSING_K * sigma
        # Under about 1 m/s of closing past the noise, the mean is a wander,
        # not a car we have to stop for. The range law still brakes a real
        # close. Cutting the path here used to turn that wander into a wall.
        if optimistic <= 0.4 and float(occ["y"]) > 4.0:
            if kind not in _VRU:
                continue
            rough = float(occ["y"]) - float(occ["length"]) * 0.5 - STANDOFF_M
            if rough > _VRU_HOLD_M:
                continue
        back = float(occ["length"]) * 0.5 + STANDOFF_M
        cut_s = station[idx] - back
        occupied.append(dict(occ))
        ttc = None if closing <= 0.05 else max(0.0, cut_s) / max(closing, 0.1)
        # A closing rate has to hold before we match it. A correlated
        # wander crosses 1 m/s and lets go. A real slower lead does not.
        try:
            hold_s = float(occ.get("close_hold_s") or 0.0)
        except (TypeError, ValueError):
            hold_s = 0.0
        if not math.isfinite(hold_s):
            hold_s = 0.0
        reported = closing
        if kind not in _VRU and hold_s < _CLOSE_HOLD_S and closing < 3.5:
            reported = min(closing, 0.9)
        ident = occ.get("id")
        try:
            ident = None if ident is None else int(ident)
        except (TypeError, ValueError):
            ident = None
        reason = kind if kind in ("pedestrian", "bike") else "vehicle"
        # Lower bound on closing: the mean minus k sigma. A jump can cut
        # the path while that bound is only a metre and a half. Comfort is
        # a wall, so it waits for a real close or for the drop to lock.
        drop = occ.get("drop_state")
        locked = (
            isinstance(drop, list)
            and bool(drop)
            and (_num(drop[0]) or 0.0) >= 1.0
        )
        # A stored lock is not a wall once the filtered rate has let up.
        # The floor itself already dropped. Comfort waits for a hard close.
        rate_now = _num(occ.get("range_rate_mps"))
        lock_hard = locked and rate_now is not None and rate_now <= -2.5
        bound = closing - _CLOSING_K * max(0.0, sigma)
        comfort = bool(lock_hard or bound >= _COMFORT_CLOSE_MPS)
        cuts.append((cut_s, reason, ident, ttc, reported, comfort))

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
                _sign_ttc(station[idx], ego_speed_mps), float(ego_speed_mps), True,
            ))
        elif kind in ("yield", "yield_sign"):
            if lateral > _sign_gate(SHOULDER_M, raw):
                continue
            # Same reach as a stop sign on this route. Brake is scaled from the
            # remaining distance, not held at one value.
            cuts.append((
                station[idx] - SIGN_STANDOFF_M, "yield", None, None,
                float(ego_speed_mps), True,
            ))
        elif kind in ("traffic_light", "light"):
            if lateral > _sign_gate(LIGHT_GATE_M, raw):
                continue
            state = str(raw.get("state") or "").strip().lower()
            if state in ("red", "r"):
                cuts.append((
                    station[idx] - SIGN_STANDOFF_M, "red_light", None,
                    _sign_ttc(station[idx], ego_speed_mps), float(ego_speed_mps), True,
                ))
            elif state in ("yellow", "amber") and station[idx] < 22.0:
                cuts.append((
                    station[idx] - SIGN_STANDOFF_M, "yellow_light", None,
                    _sign_ttc(station[idx], ego_speed_mps), float(ego_speed_mps), True,
                ))
    if not cuts:
        return path, "none", None, None, occupied, None, None, False
    cut_s, reason, cipv_id, ttc, closing, comfort = min(cuts, key=lambda c: c[0])
    stop_dist = max(0.0, cut_s)
    trimmed = [p for p, s in zip(path, station) if s <= stop_dist + 1e-6]
    if len(trimmed) < 2:
        trimmed = [path[0], path[1 if len(path) > 1 else 0]]
        if trimmed[0] is trimmed[1]:
            trimmed = [path[0], _pt(path[0]["x"], path[0]["y"] + 0.4)]
    return trimmed, reason, cipv_id, ttc, occupied, stop_dist, closing, bool(comfort)


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
    comfort_stop: bool = False,
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
        # Below 0.5 m/s, and whenever closing is under 3 m/s, the two laws
        # above both let go. A pedestrian or bike still inside the stop
        # box holds the brake to a standstill even when the close is gone.
        # A vehicle that is only a little slower stays on the scaled brake.
        if (
            float(ego_speed_mps) > 0.05
            and stop_reason in _VRU
            and dist <= _VRU_HOLD_M
        ):
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
            close_now = 0.0 if closing is None else closing
            lead_v = float(ego_speed_mps) - close_now
            comfort = math.sqrt(max(0.0, 2.0 * _COMFORT_DECEL * dist))
            # The comfort target is a wall. It applies when the lower
            # bound on closing is a real close, or the drop is locked and
            # the filtered rate is still a hard close. A stored lock whose
            # rate has already let up is not a wall. A same-track jump can
            # still cut the path while that bound is only a couple of m/s;
            # matching the lead's speed there does not shed the way the
            # wall does. A stop or a light sets the flag itself, and a
            # lead that is already near a standstill still uses the wall.
            if comfort_stop and (dist <= _COMFORT_CUT_M or lead_v <= 1.0):
                target = min(target, comfort)
            else:
                target = min(target, lead_v)
    limit = _limit_mps(signs, path)
    if limit is not None:
        target = min(target, max(0.0, limit))
    if aeb == "brake":
        target = 0.0
    return aeb, float(target), reported_ttc
