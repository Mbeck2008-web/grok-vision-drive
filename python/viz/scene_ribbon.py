"""Perceived lane-center ribbon from a scene-ckpt 14-vector.

Finite centers at 5, 10, 20 and 30 m are the only knots. A monotone cubic
in ground coordinates is sampled about every 0.5 m, then projected. A missing
lookahead is a break: the curve does not cross the gap. Nothing is added
past the last finite lookahead or past 30 m. Confidence under the live floor
draws no ribbon. This is not a planner path and must not be written to path_ego.
"""
from __future__ import annotations

import math
from typing import Any, Callable

from python.debug.lane_targets import LOOKAHEADS_M

# Same floor as python.train.scene_lane_runtime.SCENE_LANE_MIN_CONF (0.40).
# Kept local so importing this module does not load the ONNX runtime.
SCENE_RIBBON_MIN_CONF = 0.40
SCENE_RIBBON_MAX_Y_M = 30.0
SCENE_RIBBON_STEP_M = 0.5
# Perceived ribbon on the chase view and the windshield. BGR #3d7eea.
RIBBON_BGR = (234, 126, 61)


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


def _pchip_slopes(y: list[float], x: list[float]) -> list[float]:
    """Fritsch-Carlson slopes. The cubic on each span stays between the knots."""
    n = len(y)
    if n < 2:
        return []
    h = [y[i + 1] - y[i] for i in range(n - 1)]
    delta = [(x[i + 1] - x[i]) / h[i] for i in range(n - 1)]
    if n == 2:
        return [delta[0], delta[0]]
    m = [0.0] * n
    m[0] = delta[0]
    m[-1] = delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] == 0.0 or delta[i] == 0.0 or delta[i - 1] * delta[i] < 0.0:
            m[i] = 0.0
        else:
            m[i] = (delta[i - 1] + delta[i]) * 0.5
    for i in range(n - 1):
        if delta[i] == 0.0:
            m[i] = 0.0
            m[i + 1] = 0.0
            continue
        a = m[i] / delta[i]
        b = m[i + 1] / delta[i]
        hyp = a * a + b * b
        if hyp > 9.0:
            tau = 3.0 / math.sqrt(hyp)
            m[i] = tau * a * delta[i]
            m[i + 1] = tau * b * delta[i]
    return m


def _hermite(y0: float, y1: float, x0: float, x1: float, m0: float, m1: float, y: float) -> float:
    h = y1 - y0
    t = 0.0 if h <= 1e-12 else (y - y0) / h
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * x0 + h10 * h * m0 + h01 * x1 + h11 * h * m1


def _sample_ys(y0: float, y1: float, knots: list[float], step: float) -> list[float]:
    ys = [y0, y1]
    y = y0
    while y + step < y1 - 1e-9:
        y += step
        ys.append(y)
    for knot in knots:
        if y0 + 1e-9 < knot < y1 - 1e-9:
            ys.append(knot)
    ys.sort()
    out: list[float] = []
    for value in ys:
        if out and value - out[-1] <= 1e-6:
            continue
        out.append(value)
    return out


def _spline_run(run: list[dict[str, float]]) -> list[dict[str, float]]:
    """Monotone cubic through one contiguous run. Empty when the run cannot stroke."""
    if len(run) < 2:
        return []
    pts = [p for p in run if p["y"] <= SCENE_RIBBON_MAX_Y_M + 1e-6]
    if len(pts) < 2:
        return []
    ys = [float(p["y"]) for p in pts]
    xs = [float(p["x"]) for p in pts]
    y_end = min(SCENE_RIBBON_MAX_Y_M, ys[-1])
    if y_end <= ys[0] + 1e-9:
        return []
    slopes = _pchip_slopes(ys, xs)
    samples = _sample_ys(ys[0], y_end, ys, SCENE_RIBBON_STEP_M)
    out: list[dict[str, float]] = []
    span = 0
    for y in samples:
        if y > y_end + 1e-9:
            break
        while span + 1 < len(ys) and y > ys[span + 1] + 1e-9:
            span += 1
        hit = None
        for i, knot in enumerate(ys):
            if abs(y - knot) <= 1e-6:
                hit = xs[i]
                break
        if hit is None:
            span = min(span, len(ys) - 2)
            hit = _hermite(ys[span], ys[span + 1], xs[span], xs[span + 1], slopes[span], slopes[span + 1], y)
        out.append({"x": float(hit), "y": float(y)})
    if out:
        out[0] = {"x": xs[0], "y": ys[0]}
        if abs(out[-1]["y"] - y_end) <= 1e-6:
            # Last published knot at or before 30 m, not a point past it.
            end_x = xs[-1] if abs(ys[-1] - y_end) <= 1e-6 else out[-1]["x"]
            out[-1] = {"x": float(end_x), "y": float(y_end)}
    return out


def paint_tapered_ribbon(
    img: Any,
    points: list[dict[str, float]],
    project: Callable[[float, float], tuple[int, int] | None],
    color: tuple[int, int, int] = RIBBON_BGR,
) -> None:
    """Anti-aliased centerline. Width tapers with forward distance. One glow blend.

    ``project`` is the existing camera map. This function does not move the math.
    A gap in ``points`` is already a break in y; a rejected projection breaks the stroke.
    """
    if img is None or len(points) < 2:
        return
    import cv2
    import numpy as np

    runs: list[list[tuple[int, int, float]]] = []
    cur: list[tuple[int, int, float]] = []
    for p in points:
        uv = project(float(p["x"]), float(p["y"]))
        if uv is None:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
            continue
        cur.append((int(uv[0]), int(uv[1]), float(p["y"])))
    if len(cur) >= 2:
        runs.append(cur)
    if not runs:
        return
    glow = np.zeros_like(img)

    def _width(y_m: float, core: bool) -> int:
        t = min(1.0, max(0.0, (float(y_m) - 5.0) / 25.0))
        px = 7.0 - 5.0 * t
        if not core:
            px += 6.0
        return max(1, int(round(px)))

    for run in runs:
        for a, b in zip(run, run[1:]):
            y_m = 0.5 * (a[2] + b[2])
            cv2.line(glow, (a[0], a[1]), (b[0], b[1]), color, _width(y_m, False), cv2.LINE_AA)
    blended = img.copy()
    cv2.addWeighted(glow, 0.35, img, 0.65, 0, blended)
    painted = np.any(glow != 0, axis=2)
    img[painted] = blended[painted]
    for run in runs:
        for a, b in zip(run, run[1:]):
            y_m = 0.5 * (a[2] + b[2])
            cv2.line(img, (a[0], a[1]), (b[0], b[1]), color, _width(y_m, True), cv2.LINE_AA)


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
