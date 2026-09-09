"""Lane graph + road edges for the viz layer (ego frame: x right, y forward).

Honesty: `estimate_lanes` only recovers the ego lane's left/right boundaries from Hough
segments. Everything wider than that is a *lateral offset* of a boundary we actually saw,
tagged `kind="predicted"`, and is only produced when a detected boundary exists to anchor
it. No detected lane → no predicted lane, no road edge. Offsets are flat-road and assume
the neighbouring lanes run parallel to ours; that holds in the near field and is why the
app draws predicted geometry dimmer and dashed.

Nothing here feeds the planner. Corridor, CIPV and AEB keep using `lanes_bev` as before.
"""

from __future__ import annotations

from typing import Any

LANE_W_MIN = 2.6
LANE_W_MAX = 4.6
LANE_W_DEFAULT = 3.5
LANE_CONF_MIN = 0.25       # below this the Hough fit is too weak to hang predictions on
EDGE_SHOULDER_M = 0.4      # curb sits just outside the outermost predicted boundary
MAX_POLYS = 8


def _points(poly: Any) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
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
        out.append({"x": round(float(x), 2), "y": round(float(y), 2)})
    out.sort(key=lambda q: q["y"])
    return out


def _mean_x(points: list[dict[str, float]]) -> float:
    return sum(p["x"] for p in points) / max(1, len(points))


def _shift(points: list[dict[str, float]], dx: float) -> list[dict[str, float]]:
    return [{"x": round(p["x"] + dx, 2), "y": p["y"]} for p in points]


def measure_lane_width(left: list[dict[str, float]] | None,
                       right: list[dict[str, float]] | None) -> tuple[float, bool]:
    """(width_m, measured). Falls back to a default when the pair is missing or implausible."""
    if not left or not right:
        return LANE_W_DEFAULT, False
    w = _mean_x(right) - _mean_x(left)
    if LANE_W_MIN <= w <= LANE_W_MAX:
        return round(w, 2), True
    return LANE_W_DEFAULT, False


def lanes_ext(lanes_bev: Any, lane_conf: float, *, neighbours: int = 1) -> list[dict[str, Any]]:
    """Ego-lane boundaries as detected, plus `neighbours` predicted boundaries per side.

    `index` counts boundaries out from the ego lane: -1 / +1 are its own edges, -2 / +2 the
    far side of the neighbouring lane, and so on. `style` stays "unknown" — the Hough fit
    says nothing about solid vs dashed paint.
    """
    detected: list[dict[str, Any]] = []
    for poly in (lanes_bev or [])[:MAX_POLYS]:
        pts = _points(poly)
        if len(pts) > 1:
            detected.append({"points": pts, "mean_x": _mean_x(pts)})
    if not detected:
        return []

    left = min((d for d in detected if d["mean_x"] < 0), key=lambda d: abs(d["mean_x"]), default=None)
    right = min((d for d in detected if d["mean_x"] >= 0), key=lambda d: abs(d["mean_x"]), default=None)
    out: list[dict[str, Any]] = []

    def add(points, kind, side, index):
        out.append({
            "points": points,
            "kind": kind,
            "side": side,
            "style": "unknown",
            "index": index,
        })

    if left:
        add(left["points"], "detected", "left", -1)
    if right:
        add(right["points"], "detected", "right", 1)
    for d in detected:
        if d is not left and d is not right:
            side = "left" if d["mean_x"] < 0 else "right"
            add(d["points"], "detected", side, -2 if side == "left" else 2)

    if float(lane_conf or 0.0) < LANE_CONF_MIN:
        return out

    width, _measured = measure_lane_width(left["points"] if left else None,
                                          right["points"] if right else None)
    # Fill in the boundary we did not see, then step outwards one lane at a time.
    if left and not right:
        add(_shift(left["points"], width), "predicted", "right", 1)
        right = {"points": _shift(left["points"], width)}
    elif right and not left:
        add(_shift(right["points"], -width), "predicted", "left", -1)
        left = {"points": _shift(right["points"], -width)}
    for i in range(1, max(0, neighbours) + 1):
        if left:
            add(_shift(left["points"], -width * i), "predicted", "left", -(1 + i))
        if right:
            add(_shift(right["points"], width * i), "predicted", "right", 1 + i)
    return out


def road_edges(lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Curb line just outside the outermost boundary on each side.

    Always `kind="predicted"`: nothing in the stack detects a kerb, so this is the edge of
    the road we *assume* given the lanes we can see.
    """
    edges: list[dict[str, Any]] = []
    for side, sign in (("left", -1.0), ("right", 1.0)):
        cands = [l for l in lanes if l.get("side") == side and l.get("points")]
        if not cands:
            continue
        outer = max(cands, key=lambda l: abs(int(l.get("index", 0))))
        edges.append({
            "points": _shift(outer["points"], sign * EDGE_SHOULDER_M),
            "kind": "predicted",
            "side": side,
        })
    return edges
