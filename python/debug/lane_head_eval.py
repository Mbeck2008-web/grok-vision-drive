"""Held-out eval: learned lane head vs Hough, on the same frames.

The flag default (``--learned-lanes``) may flip to ON only when this eval,
on real held-out data, says ``learned_beats_hough``. Nothing here is a
result; it is the harness. ``scripts/eval_learned_lanes.py`` is the CLI.

Sample (one ``.npz`` per frame):
  main      uint8 HxWx3 forward image (what Hough reads without a stitch)
  stitch    optional uint8 stitch (``lanes_from_view`` second arg)
  strip     optional uint8 model strip (``compose_strip``) for the learned head
  lane_gt   float[14] in ``python.debug.lane_targets`` layout: per lookahead
            (5, 10, 20, 30 m) center/left/right x, then heading, then conf.
            NaN where a boundary has no painted line.
  has_paint bool: any painted lane line visible (False: a no-paint frame)

Metrics per predictor, over painted boundaries at each lookahead:
  mae_m       mean |x_pred - x_gt| where both exist
  recall      share of GT boundaries with a prediction within ``HIT_M``
  fp_rate     share of no-paint frames where the predictor emitted any lane
  n_*         counts behind each number
"""
from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from python.debug.lane_targets import LANE_GT_DIM, LOOKAHEADS_M, SLOT_LEFT, SLOT_RIGHT, lookahead_slot

HIT_M = 0.5
# Fewer than this many GT boundaries or no-paint frames: no verdict.
MIN_BOUNDARIES = 200
MIN_NO_PAINT = 50


def x_at(poly: Iterable[Mapping[str, float]], y: float) -> float | None:
    """x of one polyline at ego distance ``y`` (linear), None if it does not span y."""
    pts = sorted(((float(p["y"]), float(p["x"])) for p in poly), key=lambda t: t[0])
    for (y0, x0), (y1, x1) in zip(pts, pts[1:]):
        if y0 <= y <= y1 and y1 > y0:
            return x0 + (x1 - x0) * (y - y0) / (y1 - y0)
    return None


def ego_boundaries(lanes_bev: Iterable[Any], y: float) -> tuple[float | None, float | None]:
    """Nearest line left of the car (x < 0) and right of it (x >= 0) at y."""
    left = right = None
    for poly in lanes_bev or []:
        x = x_at(poly, y)
        if x is None or not math.isfinite(x):
            continue
        if x < 0.0 and (left is None or x > left):
            left = x
        if x >= 0.0 and (right is None or x < right):
            right = x
    return left, right


def score(samples: Iterable[Mapping[str, Any]], predict: Callable[[Mapping[str, Any]], Any]) -> dict[str, Any]:
    """``predict(sample)`` returns a LaneResult-like object with ``lanes_bev``."""
    errs: list[float] = []
    n_gt = hits = 0
    n_no_paint = fps = 0
    for sample in samples:
        res = predict(sample)
        lanes = list(getattr(res, "lanes_bev", []) or [])
        if not bool(sample.get("has_paint", True)):
            n_no_paint += 1
            fps += 1 if lanes else 0
            continue
        gt = np.asarray(sample.get("lane_gt"), dtype=np.float64).reshape(-1)
        if gt.size < LANE_GT_DIM:
            continue
        for k, y in enumerate(LOOKAHEADS_M):
            pred_l, pred_r = ego_boundaries(lanes, float(y))
            for slot, pred in ((SLOT_LEFT, pred_l), (SLOT_RIGHT, pred_r)):
                g = float(gt[lookahead_slot(k, slot)])
                if not math.isfinite(g):
                    continue
                n_gt += 1
                if pred is None:
                    continue
                err = abs(pred - g)
                errs.append(err)
                hits += 1 if err <= HIT_M else 0
    return {
        "mae_m": float(np.mean(errs)) if errs else None,
        "recall": hits / n_gt if n_gt else None,
        "fp_rate": fps / n_no_paint if n_no_paint else None,
        "n_boundaries": n_gt,
        "n_matched": len(errs),
        "n_no_paint": n_no_paint,
    }


def verdict(learned: Mapping[str, Any], hough: Mapping[str, Any]) -> dict[str, Any]:
    """``learned_beats_hough`` only with enough data and no metric worse."""
    why: list[str] = []
    if learned["n_boundaries"] < MIN_BOUNDARIES or learned["n_no_paint"] < MIN_NO_PAINT:
        why.append(
            f"insufficient data: {learned['n_boundaries']} boundaries (need {MIN_BOUNDARIES}), "
            f"{learned['n_no_paint']} no-paint frames (need {MIN_NO_PAINT})"
        )
    for key, better in (("recall", "higher"), ("mae_m", "lower"), ("fp_rate", "lower")):
        a, b = learned.get(key), hough.get(key)
        if a is None:
            why.append(f"learned {key} unmeasured")
            continue
        if b is None:
            continue
        worse = a < b if better == "higher" else a > b
        if worse:
            why.append(f"learned {key} {a:.4f} worse than hough {b:.4f}")
    return {"learned_beats_hough": not why, "why_not": why}
