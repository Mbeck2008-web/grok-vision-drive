"""Ground-truth lane targets at several lookaheads.

The 8 m sample in the live log stays. Training also wants the same lane at
5, 10, 20, and 30 m, plus the heading error. The layout is fixed so a
recording, the scene-net head, and the ONNX runtime agree without a clock
rate.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from python.debug.gt_geom import ego_center_x

# Metres ahead of the bumper. Not a frame count and not a fixed Hz.
LOOKAHEADS_M = (5.0, 10.0, 20.0, 30.0)
# Live lane_x is the 8 m sample. It is interpolated from 5 m and 10 m.
LIVE_LANE_Y_M = 8.0
# Per lookahead: center, left, right. Then heading error. Then confidence.
LANE_GT_DIM = len(LOOKAHEADS_M) * 3 + 2
# Offsets inside one lookahead's triple, the same in lane_gt and lane_gt_valid.
SLOT_CENTER = 0
SLOT_LEFT = 1
SLOT_RIGHT = 2


def lookahead_slot(k: int, part: int = SLOT_CENTER) -> int:
    """Index of ``part`` (SLOT_CENTER/LEFT/RIGHT) for lookahead ``k`` in a lane_gt vector."""
    return 3 * int(k) + int(part)


def lookahead_xs(
    lane_offset_m: float,
    heading_err_rad: float | None,
    lane_width_m: float,
    *,
    ys: tuple[float, ...] = LOOKAHEADS_M,
) -> dict[str, list[float]]:
    """Ego-frame x of the lane center and the two boundaries at each y.

    Same geometry as the 8 m ``gt_center_x`` field. Positive x is to the right.
    """
    half = float(lane_width_m) * 0.5
    centers: list[float] = []
    lefts: list[float] = []
    rights: list[float] = []
    for y in ys:
        center = ego_center_x(lane_offset_m, heading_err_rad, float(y))
        centers.append(float(center))
        lefts.append(float(center - half))
        rights.append(float(center + half))
    return {"y_m": [float(y) for y in ys], "center_x": centers, "left_x": lefts, "right_x": rights}


def pack_lane_gt(
    *,
    lane_offset_m: float | None,
    heading_err_rad: float | None,
    lane_width_m: float | None,
    off_road: bool | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(values[14], valid[14])``.

    Values are center/left/right at each lookahead, then heading error, then
    confidence in ``[0, 1]``. Invalid geometry is NaN and its mask is 0, so
    a missing hit is not scored as a centered lane. Confidence is 0.
    """
    values = np.full((LANE_GT_DIM,), np.nan, dtype=np.float32)
    values[-1] = 0.0
    valid = np.zeros((LANE_GT_DIM,), dtype=np.float32)
    usable = (
        lane_offset_m is not None
        and lane_width_m is not None
        and float(lane_width_m) > 0.0
        and off_road is not True
    )
    if not usable:
        return values, valid
    packed = lookahead_xs(float(lane_offset_m), heading_err_rad, float(lane_width_m))
    cursor = 0
    for center, left, right in zip(packed["center_x"], packed["left_x"], packed["right_x"]):
        values[cursor : cursor + 3] = (center, left, right)
        valid[cursor : cursor + 3] = 1.0
        cursor += 3
    if heading_err_rad is not None and math.isfinite(float(heading_err_rad)):
        values[cursor] = float(heading_err_rad)
        valid[cursor] = 1.0
    values[-1] = 1.0
    valid[-1] = 1.0
    return values, valid


def lane_center_is_valid(mask: Any) -> bool:
    """True when the 5 m and 10 m center slots are marked valid.

    The live 8 m sample is interpolated from those two. Invalid geometry is
    NaN, and this mask is what keeps it out of the score.
    """
    arr = np.asarray(mask if mask is not None else [], dtype=np.float64).reshape(-1)
    if arr.size < 4:
        return False
    return float(arr[0]) > 0.5 and float(arr[3]) > 0.5


def lane_x_from_vector(values: Any, y_m: float = LIVE_LANE_Y_M) -> float | None:
    """Center x at ``y_m``, interpolated between the stored lookaheads."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size < LANE_GT_DIM:
        return None
    centers = [float(arr[i * 3]) for i in range(len(LOOKAHEADS_M))]
    ys = list(LOOKAHEADS_M)
    if float(y_m) <= ys[0]:
        return centers[0]
    if float(y_m) >= ys[-1]:
        return centers[-1]
    for i in range(len(ys) - 1):
        if ys[i] <= float(y_m) <= ys[i + 1]:
            span = ys[i + 1] - ys[i]
            mix = (float(y_m) - ys[i]) / span
            return centers[i] + mix * (centers[i + 1] - centers[i])
    return None


def confidence_from_vector(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size < 1 or not math.isfinite(float(arr[-1])):
        return 0.0
    return float(min(1.0, max(0.0, arr[-1])))


def polylines_from_hit(
    lane_offset_m: float,
    heading_err_rad: float | None,
    lane_width_m: float,
) -> list[list[dict[str, float]]]:
    """Left and right boundaries the existing lane head can train on."""
    packed = lookahead_xs(lane_offset_m, heading_err_rad, lane_width_m)
    left = [{"x": x, "y": y, "z": 0.0} for x, y in zip(packed["left_x"], packed["y_m"])]
    right = [{"x": x, "y": y, "z": 0.0} for x, y in zip(packed["right_x"], packed["y_m"])]
    return [left, right]


def tech_from_hit(
    lane_offset_m: float | None,
    heading_err_rad: float | None,
    lane_width_m: float | None,
    off_road: bool | None,
) -> dict[str, Any]:
    """Tech dict stored beside a strip. Missing geometry is an empty target."""
    values, mask = pack_lane_gt(
        lane_offset_m=lane_offset_m,
        heading_err_rad=heading_err_rad,
        lane_width_m=lane_width_m,
        off_road=off_road,
    )
    lanes: list[list[dict[str, float]]] = []
    if lane_offset_m is not None and lane_width_m is not None and off_road is not True:
        lanes = polylines_from_hit(float(lane_offset_m), heading_err_rad, float(lane_width_m))
    return {
        "lanes": lanes,
        "lane_gt": values.tolist(),
        "lane_gt_valid": mask.tolist(),
        "gt_lookahead_m": [float(y) for y in LOOKAHEADS_M],
        "heading_err_rad": None if heading_err_rad is None else float(heading_err_rad),
        "off_road": None if off_road is None else bool(off_road),
    }


def tech_from_painted(lanes: list[tuple[float, float] | None]) -> dict[str, Any]:
    """Tech dict from painted lane edges at each lookahead (vehicle frame).

    Each lookahead carries its own measured center/left/right, so a curve is
    a curve. A lookahead without paint on both sides is NaN with mask 0.
    The 5 m and 10 m slots gate ``lane_center_is_valid`` as before.
    """
    from python.debug.painted_lanes import heading_from_centers

    values = np.full((LANE_GT_DIM,), np.nan, dtype=np.float32)
    valid = np.zeros((LANE_GT_DIM,), dtype=np.float32)
    values[-1] = 0.0
    centers: list[float | None] = []
    polys_l: list[dict[str, float]] = []
    polys_r: list[dict[str, float]] = []
    for k, y in enumerate(LOOKAHEADS_M):
        got = lanes[k] if k < len(lanes) else None
        if got is None:
            centers.append(None)
            continue
        left, right = float(got[0]), float(got[1])
        center = 0.5 * (left + right)
        values[3 * k : 3 * k + 3] = (center, left, right)
        valid[3 * k : 3 * k + 3] = 1.0
        centers.append(center)
        polys_l.append({"x": left, "y": float(y), "z": 0.0})
        polys_r.append({"x": right, "y": float(y), "z": 0.0})
    heading = heading_from_centers(centers[0], centers[1])
    if heading is not None:
        values[-2] = heading
        valid[-2] = 1.0
    ok = valid[0] > 0.5 and valid[3] > 0.5
    values[-1] = 1.0 if ok else 0.0
    valid[-1] = 1.0
    return {
        "lanes": [polys_l, polys_r] if polys_l else [],
        "lane_gt": values.tolist(),
        "lane_gt_valid": valid.tolist(),
        "gt_lookahead_m": [float(y) for y in LOOKAHEADS_M],
        "gt_source": "painted_decals",
        "gt_origin": "state_pos_ref_node",
        "heading_err_rad": heading,
        "off_road": False if ok else None,
    }


def lane_gt_from_record(record: Mapping[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    """Read a stored tech dict, or pack it from the scalar GT fields."""
    record = record or {}
    raw = record.get("lane_gt")
    mask = record.get("lane_gt_valid")
    if raw is not None and mask is not None:
        values = np.zeros((LANE_GT_DIM,), dtype=np.float32)
        valid = np.zeros((LANE_GT_DIM,), dtype=np.float32)
        src = np.asarray(raw, dtype=np.float32).reshape(-1)
        msrc = np.asarray(mask, dtype=np.float32).reshape(-1)
        n = min(LANE_GT_DIM, int(src.size), int(msrc.size))
        values[:n] = src[:n]
        valid[:n] = msrc[:n]
        return values, valid
    return pack_lane_gt(
        lane_offset_m=_num(record.get("lane_offset_m")),
        heading_err_rad=_num(record.get("heading_err_rad", record.get("road_heading_err_rad"))),
        lane_width_m=_num(record.get("lane_width_m")),
        off_road=record.get("off_road") if isinstance(record.get("off_road"), bool) else None,
    )


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def time_holdout(rows: list[Mapping[str, Any]], fraction: float = 0.2) -> tuple[list[Any], list[Any]]:
    """Last ``fraction`` of the time span, not a shuffled suffix.

    Rows should already be in time order. A single timestamp stays in train.
    """
    if not rows:
        return [], []
    times = []
    for row in rows:
        try:
            times.append(float(row.get("t")))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            times.append(None)
    finite = [t for t in times if t is not None]
    if len(finite) < 2:
        return list(rows), []
    start, end = min(finite), max(finite)
    if end <= start:
        return list(rows), []
    cut = start + (1.0 - float(fraction)) * (end - start)
    train = [row for row, t in zip(rows, times) if t is None or t < cut]
    hold = [row for row, t in zip(rows, times) if t is not None and t >= cut]
    return train, hold


def score_centers(pred: Any, truth: Any) -> dict[str, float | int | None]:
    """Pearson r and RMS of a lane center against ground truth. No torch."""
    xs = [float(v) for v in np.asarray(pred, dtype=np.float64).reshape(-1)]
    ys = [float(v) for v in np.asarray(truth, dtype=np.float64).reshape(-1)]
    pairs = [(a, b) for a, b in zip(xs, ys) if math.isfinite(a) and math.isfinite(b)]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "r": None, "rms": None, "rms_mean": None, "bias": None}
    err = [a - b for a, b in pairs]
    rms = math.sqrt(sum(e * e for e in err) / n)
    bias = sum(err) / n
    my = sum(b for _a, b in pairs) / n
    # Predicting the mean of the truth. This is the floor for an uncorrelated guess.
    rms_mean = math.sqrt(sum((b - my) ** 2 for _a, b in pairs) / n)
    base = {"n": n, "r": None, "rms": rms, "rms_mean": rms_mean, "bias": bias}
    if n < 3:
        return base
    mx = sum(a for a, _b in pairs) / n
    vx = sum((a - mx) ** 2 for a, _b in pairs)
    vy = sum((b - my) ** 2 for _a, b in pairs)
    if vx < 1e-12 or vy < 1e-12:
        return base
    cov = sum((a - mx) * (b - my) for a, b in pairs)
    base["r"] = cov / math.sqrt(vx * vy)
    return base
