"""Hindsight turn commands from a recorded trajectory (training labels only).

The command on a frame approaching a junction is the turn the car actually
took afterwards. Runtime never imports this module; live commands come from
GPS ``route_intent``. See docs/SPEC_route_conditioning.md.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from python.sensors.route_intent import (
    AMBIGUOUS_HI,
    AMBIGUOUS_LO,
    APPROACH_M,
    Graph,
    LOOKAHEAD_M,
    SNAP_M,
    classify_turn,
)

EXIT_M = 15.0
NEAR_M = 25.0
UTURN_DEG = 135.0
ROUNDABOUT_DEG = 270.0
CMDS = ("follow", "left", "right", "straight", "mask")


def _wrap(rad: float) -> float:
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def _xy(row: Mapping[str, Any]) -> tuple[float, float] | None:
    pos = row.get("pos")
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return None
    try:
        x, y = float(pos[0]), float(pos[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _heading_of(row: Mapping[str, Any], pts: list[tuple[float, float]], index: int) -> float:
    direc = row.get("dir")
    if isinstance(direc, (list, tuple)) and len(direc) >= 2:
        try:
            dx, dy = float(direc[0]), float(direc[1])
            if math.hypot(dx, dy) > 1e-6:
                return math.atan2(dy, dx)
        except (TypeError, ValueError):
            pass
    if index + 1 < len(pts):
        dx = pts[index + 1][0] - pts[index][0]
        dy = pts[index + 1][1] - pts[index][1]
        if math.hypot(dx, dy) > 1e-6:
            return math.atan2(dy, dx)
    if index > 0:
        dx = pts[index][0] - pts[index - 1][0]
        dy = pts[index][1] - pts[index - 1][1]
        if math.hypot(dx, dy) > 1e-6:
            return math.atan2(dy, dx)
    return 0.0


def _sidecar(t: Any, cmd: str, next_jct_m: float | None, reason: str | None = None) -> dict[str, Any]:
    label = cmd if cmd in CMDS else "mask"
    dist: float | None
    if next_jct_m is None:
        dist = None
    else:
        try:
            dist = float(next_jct_m)
            if not math.isfinite(dist):
                dist = None
        except (TypeError, ValueError):
            dist = None
    row: dict[str, Any] = {"t": t, "cmd": label, "next_jct_m": dist, "src": "hindsight"}
    if label == "mask" and reason:
        row["reason"] = reason
    return row



def _neighbors(graph: Graph, node: str) -> list[str]:
    names: list[str] = []
    for a, b, oneway in graph.edges:
        if a == node:
            names.append(b)
        elif b == node and not oneway:
            names.append(a)
    return names


def _dist_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    if denom < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / denom))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def _closest_neighbor(graph: Graph, node: str, x: float, y: float) -> str | None:
    jx, jy = graph.nodes[node]
    best: str | None = None
    best_d: float | None = None
    for name in _neighbors(graph, node):
        nx, ny = graph.nodes[name]
        dist = _dist_point_to_segment(x, y, jx, jy, nx, ny)
        if best_d is None or dist < best_d:
            best, best_d = name, dist
    return best


def _along_to_node(
    xs: list[float], ys: list[float], start: int, jx: float, jy: float
) -> tuple[float, int]:
    """Along-track metres from ``start`` to the closest approach to the node."""
    best_i = start
    best_d = math.hypot(xs[start] - jx, ys[start] - jy)
    acc = 0.0
    at_best = 0.0
    for i in range(start + 1, len(xs)):
        acc += math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
        if acc > LOOKAHEAD_M + SNAP_M:
            break
        d = math.hypot(xs[i] - jx, ys[i] - jy)
        if d < best_d:
            best_d = d
            best_i = i
            at_best = acc
    return at_best, best_i


def label_sequence(graph: Graph, rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One sidecar object per input row."""
    material = list(rows)
    pts: list[tuple[float, float]] = []
    keep: list[Mapping[str, Any] | None] = []
    for row in material:
        xy = _xy(row)
        pts.append(xy if xy is not None else (float("nan"), float("nan")))
        keep.append(row if xy is not None else None)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    headings = [
        0.0 if keep[i] is None else _heading_of(keep[i], pts, i)  # type: ignore[arg-type]
        for i in range(len(material))
    ]
    jcts = [(name, graph.nodes[name]) for name in graph.junctions()]
    out: list[dict[str, Any]] = []
    for i, row in enumerate(material):
        t = row.get("t")
        if keep[i] is None:
            out.append(_sidecar(t, "mask", None, "no_future"))
            continue
        next_name = None
        next_dist = None
        hit_i = None
        acc = 0.0
        for j in range(i, len(material)):
            if j > i:
                if not (math.isfinite(xs[j]) and math.isfinite(xs[j - 1])):
                    break
                acc += math.hypot(xs[j] - xs[j - 1], ys[j] - ys[j - 1])
            if acc > LOOKAHEAD_M:
                break
            if not math.isfinite(xs[j]):
                continue
            for name, (jx, jy) in jcts:
                if math.hypot(xs[j] - jx, ys[j] - jy) <= SNAP_M:
                    next_name, next_dist, hit_i = name, acc, j
                    break
            if next_name is not None:
                break
        if next_name is None:
            out.append(_sidecar(t, "follow", None))
            continue
        jx, jy = graph.nodes[next_name]
        next_dist, hit_i = _along_to_node(xs, ys, i, jx, jy)
        if next_dist > APPROACH_M:
            out.append(_sidecar(t, "follow", next_dist))
            continue
        # Heading on the approach ring vs the first sample EXIT_M past the node.
        before_i = i
        for k in range(hit_i, i - 1, -1):
            if math.hypot(xs[k] - jx, ys[k] - jy) >= EXIT_M:
                before_i = k
                break
        after_i = None
        passed = False
        closest = math.hypot(xs[hit_i] - jx, ys[hit_i] - jy)
        for k in range(hit_i, len(material)):
            if not math.isfinite(xs[k]):
                break
            d = math.hypot(xs[k] - jx, ys[k] - jy)
            if d <= closest + 1.0:
                closest = min(closest, d)
                passed = True
                continue
            if passed and d >= EXIT_M:
                after_i = k
                break
        if after_i is None:
            out.append(_sidecar(t, "mask", next_dist, "no_future"))
            continue
        cum = 0.0
        last = after_i
        for k in range(i, last):
            if math.hypot(xs[k] - jx, ys[k] - jy) <= NEAR_M:
                cum += abs(math.degrees(_wrap(headings[k + 1] - headings[k])))
        if cum >= ROUNDABOUT_DEG:
            out.append(_sidecar(t, "mask", next_dist, "roundabout"))
            continue
        entry = _closest_neighbor(graph, next_name, xs[before_i], ys[before_i])
        exit_n = _closest_neighbor(graph, next_name, xs[after_i], ys[after_i])
        if entry is None or exit_n is None:
            out.append(_sidecar(t, "mask", next_dist, "no_future"))
            continue
        if entry == exit_n:
            out.append(_sidecar(t, "mask", next_dist, "uturn"))
            continue
        ex, ey = graph.nodes[entry]
        ox, oy = graph.nodes[exit_n]
        in_h = math.atan2(jy - ey, jx - ex)
        out_h = math.atan2(oy - jy, ox - jx)
        delta = _wrap(out_h - in_h)
        deg = abs(math.degrees(delta))
        if deg >= UTURN_DEG:
            out.append(_sidecar(t, "mask", next_dist, "uturn"))
            continue
        if AMBIGUOUS_LO <= deg < AMBIGUOUS_HI:
            out.append(_sidecar(t, "mask", next_dist, "ambiguous"))
            continue
        out.append(_sidecar(t, classify_turn(delta), next_dist))
    return out
