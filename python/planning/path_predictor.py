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
_HARD_STOPS = frozenset({"vehicle", "stop_sign", "red_light"})


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

    circle = _fit_roundabout(polys) if polys else None
    if circle is not None:
        raw, width = _roundabout_route(circle, polys, step)
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

    curv = _path_curvature(path)
    occupants = _predict_occupants(tracks or [])
    path, stop_reason, cipv_id, ttc, occupied = _apply_scene(
        path,
        occupants,
        signs or [],
        width,
        ego_speed_mps=float(ego_speed_mps),
    )
    aeb, target = _speed_flags(
        curv=curv,
        stop_reason=stop_reason,
        ttc=ttc,
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
    """Record path length and the brake the follower would send. The route stays as built."""
    plan.path_length_m = path_length_m(plan.path_ego)
    commanded = follow_path(plan, ego_speed_mps=float(ego_speed_mps))
    plan.pred_brake = float(commanded.brake)
    return plan


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
) -> tuple[list[tuple[float, float]], float]:
    cx, cy, radius = circle
    cloud = [p for poly in polys for p in poly]
    angs = [math.atan2(y - cy, x - cx) for x, y in cloud]
    nearest = min(range(len(cloud)), key=lambda i: math.hypot(cloud[i][0], cloud[i][1]))
    a_near = angs[nearest]
    rels = [_wrap(a - a_near) for a in angs]
    mean_rel = sum(rels) / len(rels)
    direction = -1.0 if mean_rel > 1e-3 else 1.0
    ext = max(rel * direction for rel in rels)
    goal = ext + ROUNDABOUT_EXTRA_RAD
    entry = (cx + radius * math.cos(a_near), cy + radius * math.sin(a_near))
    path: list[tuple[float, float]] = []
    gap = math.hypot(entry[0], entry[1])
    if gap > step:
        n = max(1, int(gap / step))
        for i in range(n):
            t = i / n
            path.append((entry[0] * t, entry[1] * t))
    theta = a_near
    traveled = 0.0
    arc_len = goal * radius
    while traveled <= arc_len + 1e-6 and len(path) < 96:
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        if not path or math.hypot(x - path[-1][0], y - path[-1][1]) >= step * 0.4:
            path.append((x, y))
        theta += direction * step / radius
        traveled += step
    width = LANE_W_DEFAULT
    if len(polys) >= 2:
        radii = []
        for poly in polys:
            radii.append(sum(math.hypot(x - cx, y - cy) for x, y in poly) / len(poly))
        if len(radii) >= 2:
            width = max(LANE_W_MIN, min(LANE_W_MAX, abs(radii[0] - radii[1])))
    return path, width


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
        if _mean_x(only) <= 0.0:
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
            left, right = (sa, sb) if _mean_x(sa) <= _mean_x(sb) else (sb, sa)
            # Width at the bumper, not the mean. A camera lane converges at the
            # horizon, so the average gap collapses even when the near field is
            # a full lane.
            width = _near_separation(left, right)
            if width < 1.2:
                continue
            straddles = _mean_x(left) <= 0.6 and _mean_x(right) >= -0.6
            score = (
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
    y0 = max(left[0][1], right[0][1], 0.0)
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
    guard = 0
    while y < horizon - 1e-6 and guard < 200:
        h = h + curv * step
        x = x + math.sin(h) * step
        y = y + math.cos(h) * step
        samples.append((x, y))
        extended = True
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
    if len(samples) < 4:
        return 0.0
    mid = len(samples) // 2
    h0 = math.atan2(samples[mid][0] - samples[0][0], samples[mid][1] - samples[0][1])
    h1 = math.atan2(samples[-1][0] - samples[mid][0], samples[-1][1] - samples[mid][1])
    dist = math.hypot(samples[-1][0] - samples[0][0], samples[-1][1] - samples[0][1])
    if dist < 1.0:
        return 0.0
    return max(-0.08, min(0.08, _wrap(h1 - h0) / dist))


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
) -> tuple[list[dict[str, float]], str, int | None, float | None, list[dict[str, Any]]]:
    cuts: list[tuple[float, str, int | None, float | None, dict[str, Any] | None]] = []
    occupied: list[dict[str, Any]] = []
    station = _along(path)
    for occ in occupants:
        idx = _nearest(path, occ["x"], occ["y"])
        lateral = math.hypot(path[idx]["x"] - occ["x"], path[idx]["y"] - occ["y"])
        reach = (EGO_WIDTH + float(occ["width"])) * 0.5 + 0.2
        if lateral > reach:
            continue
        if station[idx] < 1.5:
            continue
        back = float(occ["length"]) * 0.5 + STANDOFF_M
        cut_s = station[idx] - back
        occupied.append(dict(occ))
        closing = max(ego_speed_mps, 0.0)
        ttc = None if closing <= 0.05 else max(0.0, cut_s) / closing
        ident = occ.get("id")
        try:
            ident = None if ident is None else int(ident)
        except (TypeError, ValueError):
            ident = None
        cuts.append((cut_s, "vehicle", ident, ttc, occ))

    for raw in signs or []:
        if not isinstance(raw, dict):
            continue
        kind = _cls(raw)
        x, y = _f(raw, "x"), _f(raw, "y")
        idx = _nearest(path, x, y)
        lateral = math.hypot(path[idx]["x"] - x, path[idx]["y"] - y)
        if station[idx] < 1.5:
            continue
        if kind in ("stop_sign", "stop"):
            if lateral > SHOULDER_M:
                continue
            cuts.append((station[idx] - SIGN_STANDOFF_M, "stop_sign", None, None, None))
        elif kind in ("yield", "yield_sign"):
            if lateral > SHOULDER_M:
                continue
            cuts.append((station[idx] - SIGN_STANDOFF_M, "yield", None, None, None))
        elif kind in ("traffic_light", "light"):
            if lateral > LIGHT_GATE_M:
                continue
            state = str(raw.get("state") or "").strip().lower()
            if state in ("red", "r"):
                cuts.append((station[idx] - SIGN_STANDOFF_M, "red_light", None, None, None))
            elif state in ("yellow", "amber"):
                if station[idx] < 22.0:
                    cuts.append((station[idx] - SIGN_STANDOFF_M, "yellow_light", None, None, None))
    if not cuts:
        return path, "none", None, None, occupied
    cut_s, reason, cipv_id, ttc, _occ = min(cuts, key=lambda c: c[0])
    trimmed = [p for p, s in zip(path, station) if s <= max(0.0, cut_s) + 1e-6]
    if len(trimmed) < 2:
        trimmed = [path[0], path[1 if len(path) > 1 else 0]]
        if trimmed[0] is trimmed[1]:
            trimmed = [path[0], _pt(path[0]["x"], path[0]["y"] + 0.4)]
    return trimmed, reason, cipv_id, ttc, occupied


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


def _speed_flags(
    *,
    curv: float,
    stop_reason: str,
    ttc: float | None,
    ego_speed_mps: float,
    signs: list,
    path: list[dict[str, float]],
) -> tuple[str, float]:
    aeb = "off"
    if stop_reason in _HARD_STOPS or stop_reason == "yellow_light":
        aeb = "brake"
    elif stop_reason == "yield":
        aeb = "warn"
    speed = plan_speed(
        ego_speed_mps=ego_speed_mps,
        curvature=curv,
        ttc_lead=ttc,
        aeb=aeb,
        v_cap=V_CAP,
    )
    target = speed.target_v
    if stop_reason == "yield":
        target = min(target, 3.0)
    limit = _limit_mps(signs, path)
    if limit is not None:
        target = min(target, max(0.0, limit))
    if aeb == "brake":
        target = 0.0
    return aeb, float(target)
