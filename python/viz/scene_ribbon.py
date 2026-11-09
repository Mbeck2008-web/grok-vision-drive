"""Perceived lane-center ribbon from a scene-ckpt 14-vector.

The polyline only goes through finite centers at 5, 10, 20 and 30 m.
A missing (NaN) lookahead is a break: the stroke does not jump the gap.
Nothing is added past the last finite lookahead or past 30 m. Confidence
under the live floor draws no ribbon. This is not a planner path and must
not be written to path_ego.
"""
from __future__ import annotations

import math
from typing import Any

from python.debug.lane_targets import LOOKAHEADS_M

# Same floor as python.train.scene_lane_runtime.SCENE_LANE_MIN_CONF (0.40).
# Kept local so importing this module does not load the ONNX runtime.
SCENE_RIBBON_MIN_CONF = 0.40
SCENE_RIBBON_MAX_Y_M = 30.0


def _finite(v: Any) -> bool:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x)


def scene_ckpt_centers(lane_gt: Any) -> list[dict[str, float]]:
    """Finite center samples as ``{x, y}``, in lookahead order. Gaps are omitted."""
    if lane_gt is None:
        return []
    try:
        vec = list(lane_gt)
    except TypeError:
        return []
    out: list[dict[str, float]] = []
    for i, y_m in enumerate(LOOKAHEADS_M):
        idx = i * 3
        if idx >= len(vec):
            break
        if not _finite(vec[idx]):
            continue
        y = float(y_m)
        if y > SCENE_RIBBON_MAX_Y_M + 1e-6:
            continue
        out.append({"x": float(vec[idx]), "y": y})
    return out


def center_runs(centers: list[dict[str, float]]) -> list[list[dict[str, float]]]:
    """Split published centers wherever a lookahead in LOOKAHEADS_M is missing."""
    by_y = {round(float(c["y"]), 3): c for c in centers if _finite(c.get("x")) and _finite(c.get("y"))}
    runs: list[list[dict[str, float]]] = []
    cur: list[dict[str, float]] = []
    for y_m in LOOKAHEADS_M:
        key = round(float(y_m), 3)
        hit = by_y.get(key)
        if hit is None:
            if cur:
                runs.append(cur)
                cur = []
            continue
        cur.append({"x": float(hit["x"]), "y": float(hit["y"])})
    if cur:
        runs.append(cur)
    return runs


def _catmull(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _spline_run(run: list[dict[str, float]], samples_per_seg: int = 8) -> list[dict[str, float]]:
    if len(run) < 2:
        return []
    y_lo = min(p["y"] for p in run)
    y_hi = min(SCENE_RIBBON_MAX_Y_M, max(p["y"] for p in run))
    x_lo = min(p["x"] for p in run)
    x_hi = max(p["x"] for p in run)
    out: list[dict[str, float]] = [{"x": float(run[0]["x"]), "y": float(run[0]["y"])}]
    n = len(run)
    for i in range(n - 1):
        p0 = run[i - 1] if i > 0 else run[i]
        p1 = run[i]
        p2 = run[i + 1]
        p3 = run[i + 2] if i + 2 < n else run[i + 1]
        y_a, y_b = (p1["y"], p2["y"]) if p1["y"] <= p2["y"] else (p2["y"], p1["y"])
        for s in range(1, samples_per_seg + 1):
            t = s / float(samples_per_seg)
            x = _catmull(p0["x"], p1["x"], p2["x"], p3["x"], t)
            y = _catmull(p0["y"], p1["y"], p2["y"], p3["y"], t)
            y = min(max(y, y_a), y_b)
            y = min(max(y, y_lo), y_hi)
            # Convex span of the run: do not invent a lateral offset past the samples.
            x = min(max(x, x_lo), x_hi)
            pt = {"x": float(x), "y": float(y)}
            prev = out[-1]
            if abs(prev["x"] - pt["x"]) < 1e-6 and abs(prev["y"] - pt["y"]) < 1e-6:
                continue
            out.append(pt)
    return [p for p in out if p["y"] <= SCENE_RIBBON_MAX_Y_M + 1e-6]


def scene_ckpt_ribbon_polyline(
    centers: list[dict[str, float]] | None,
    confidence: float,
    *,
    min_conf: float = SCENE_RIBBON_MIN_CONF,
) -> list[dict[str, float]]:
    """Smooth polyline through contiguous finite centers, or empty."""
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return []
    if not math.isfinite(conf) or conf < float(min_conf):
        return []
    poly: list[dict[str, float]] = []
    for run in center_runs(list(centers or [])):
        poly.extend(_spline_run(run))
    return poly


def scene_ckpt_caption(confidence: float) -> str:
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return "LOW"
    if math.isfinite(conf) and conf >= SCENE_RIBBON_MIN_CONF:
        return "SHADOW"
    return "LOW"
