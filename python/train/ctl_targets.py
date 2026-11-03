"""Control-head targets, masks, baselines and verdicts. Torch-free.

The optional ``ctl`` head (``--heads lane_gt,ctl``, offline only) predicts the
recorded AI controls ``wheel.steer_ai`` (-1..1), ``pedals.throttle_ai`` (0..1)
and ``pedals.brake_ai`` (0..1) now and 1..5 s ahead. Clutch is never a target.

* Future labels are picked by ``ego.sim_t``, never by row index: the nearest
  full-recording row of the same segment within ``PICK_TOL_S`` of
  ``sim_t + h``. h=0 is the row itself.
* A (row, h) pair counts only when the row is valid (finite, in range, not in
  reverse, parking brake off, past warm-up), every full row in (t, t+h] is
  clean (same segment, no settle / stall / recover / impact window, no reverse
  or parking brake, no sim_t step over ``MAX_STEP_S``), the window does not run
  past the segment end, the target row is valid, and the target row stays in
  the row's own train piece or holdout slice. An invalid h never voids a
  shorter one.

Scores (``score_ctl``) are per horizon on the holdout, with the B0-B4
baselines, the city / highway / ramp slices and the PASS / FAIL / UNPROVEN
verdicts; ``val_lines`` prints them.
"""
from __future__ import annotations

import json
import math
from typing import Any, Iterable, Mapping

import numpy as np

CTL_FIELDS = ("steer_ai", "throttle_ai", "brake_ai")
CTL_HORIZONS_S = (0, 1, 2, 3, 4, 5)
CTL_OUTPUT_FIELDS = ("ctl_steer", "ctl_throttle", "ctl_brake")
K = len(CTL_HORIZONS_S)
CTL_RANGES = ((-1.0, 1.0), (0.0, 1.0), (0.0, 1.0))
_PATHS = (("wheel", "steer_ai"), ("pedals", "throttle_ai"), ("pedals", "brake_ai"))

PICK_TOL_S = 0.15
MAX_STEP_S = 1.0
WARMUP_ROWS = 16
WARMUP_GAP_S = 1.0
PBRAKE_ON = 0.5

GRADED = (0, 3, 4, 5)
# Rev 5 (B' series on): h0 is report-only; only h3-h5 are graded.
GRADED_REV5 = (3, 4, 5)
REV4 = "rev4"
REV5 = "rev5"
BARS = (REV4, REV5)
# Rev 5 bars on ref = min(B3, B1 median): (steer, throttle) multipliers per graded horizon.
REV5_BARS = {3: (0.85, 0.85), 4: (0.95, 0.85), 5: (0.95, 0.85)}
B4_GRADED = (3, 4, 5)
MIN_N = 300
MIN_BRAKE_EVENTS = 30
BRAKE_ON = 0.1
LAG_MEDIAN_MS = 50.0
LAG_P99_MS = 100.0
# spawn_type -> val-line key. Bridge is reported in the record, not gated.
GATED_SLICES = (("city", "city"), ("highway", "hwy"), ("ramp", "ramp"))
REPORT_SLICES = ("bridge",)

REASONS = ("null", "range", "reverse", "pbrake")

# B'1 cond input: the controls of the t-dt row (the previous full-recording row), never row t.
COND_FIELDS = ("speed", "steer_ai", "throttle_ai", "brake_ai", "yaw_rate")
COND_DIM = len(COND_FIELDS)
COND_SPEED_NORM_MPS = 30.0
COND_MAX_DT_S = 0.25
COND_FIELDS_TEXT = "speed_mps/30,steer_ai,throttle_ai,brake_ai,yaw_rate"
COND_SOURCE = "t-dt"


class CtlDataError(RuntimeError):
    """The recording cannot give ctl targets (camera lag gate failed)."""


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _get(row: Mapping[str, Any], group: str, key: str) -> Any:
    part = row.get(group)
    return part.get(key) if isinstance(part, Mapping) else None


def row_sim_t(row: Mapping[str, Any]) -> float | None:
    return _num(_get(row, "ego", "sim_t"))


def _reverse(row: Mapping[str, Any]) -> bool:
    value = _num(_get(row, "veh", "reverse"))
    return value is not None and value > 0.0


def _pbrake(row: Mapping[str, Any]) -> bool:
    value = _num(_get(row, "pedals", "parkingbrake_ai"))
    return value is not None and value > PBRAKE_ON


def row_controls(row: Mapping[str, Any]) -> tuple[np.ndarray | None, str | None]:
    """``(steer, throttle, brake)`` and ``None``, or ``None`` and the first reason the row is out."""
    values = [_num(_get(row, group, key)) for group, key in _PATHS]
    if any(value is None for value in values):
        return None, "null"
    if any(not (lo <= value <= hi) for value, (lo, hi) in zip(values, CTL_RANGES)):
        return None, "range"
    if _reverse(row):
        return None, "reverse"
    if _pbrake(row):
        return None, "pbrake"
    return np.asarray(values, dtype=np.float32), None


def cond_from_row(prev: Mapping[str, Any]) -> np.ndarray | None:
    """The cond vector a row gives to the NEXT row: (speed_mps / 30, steer_ai, throttle_ai, brake_ai, yaw_rate).

    ``yaw_rate`` is ``ego.yaw_rate`` as recorded (rad/s, CCW positive); ``imu.yaw_rate`` is never
    read. None when any field is missing, non-finite, or out of range (steer [-1, 1], pedals [0, 1],
    speed >= 0).
    """
    speed = _num(_get(prev, "ego", "speed_mps"))
    yaw = _num(_get(prev, "ego", "yaw_rate"))
    controls = [_num(_get(prev, group, key)) for group, key in _PATHS]
    if speed is None or yaw is None or speed < 0.0 or any(value is None for value in controls):
        return None
    if any(not (lo <= value <= hi) for value, (lo, hi) in zip(controls, CTL_RANGES)):
        return None
    return np.asarray([speed / COND_SPEED_NORM_MPS, *controls, yaw], dtype=np.float32)


def build_cond(
    full_rows: list[Mapping[str, Any]], kept_rows: list[Mapping[str, Any]]
) -> dict[int, tuple[np.ndarray, float]]:
    """Row id -> (cond (5,), ok) for every kept row.

    The source is the previous row of the FULL recording (t-dt), in the same segment, with
    0 < sim_t step <= 0.25 s. Row t's own controls (its h=0 label) are never read. No usable
    t-dt row (segment start or change, a longer step, a bad field): zeros and ok 0.
    Same row copy as ``build_ctl``: non-mapping entries are dropped first, and a repeated
    row id keeps its first copy.
    """
    full = [row for row in full_rows if isinstance(row, Mapping)]
    pos: dict[int, int] = {}
    for k, row in enumerate(full):
        rid = _row_id(row)
        if rid is not None:
            pos.setdefault(rid, k)
    out: dict[int, tuple[np.ndarray, float]] = {}
    for row in kept_rows:
        rid = _row_id(row)
        if rid is None:
            continue
        k = pos.get(rid)
        vec = None
        if k is not None and k > 0:
            cur = full[k]
            prev = full[k - 1]
            st, pst = row_sim_t(cur), row_sim_t(prev)
            if (prev.get("segment") == cur.get("segment") and st is not None and pst is not None
                    and 0.0 < st - pst <= COND_MAX_DT_S + 1e-9):
                vec = cond_from_row(prev)
        if vec is None:
            out[rid] = (np.zeros(COND_DIM, dtype=np.float32), 0.0)
        else:
            out[rid] = (vec, 1.0)
    return out


def cond_line() -> str:
    return f"[GVD] ctl cond: dim={COND_DIM} fields={COND_FIELDS_TEXT} source={COND_SOURCE}"


def lag_stats(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Label-to-frame lag ``t - cams.main.t`` in ms. Shift 0 is fine at median <= 50 and p99 < 100."""
    lags = []
    for row in rows:
        t = _num(row.get("t"))
        cams = row.get("cams")
        main = cams.get("main") if isinstance(cams, Mapping) else None
        cam_t = _num(main.get("t")) if isinstance(main, Mapping) else None
        if t is not None and cam_t is not None:
            lags.append((t - cam_t) * 1000.0)
    if not lags:
        return {"n": 0, "median_ms": None, "p99_ms": None, "ok": False}
    arr = np.asarray(lags, dtype=np.float64)
    median = float(np.median(arr))
    p99 = float(np.percentile(arr, 99))
    return {"n": int(arr.size), "median_ms": median, "p99_ms": p99,
            "ok": bool(median <= LAG_MEDIAN_MS and p99 < LAG_P99_MS)}


def lag_line(stats: Mapping[str, Any]) -> str:
    def fmt(value: Any) -> str:
        return "na" if value is None else f"{float(value):.1f}"

    return (
        f"[GVD] ctl lag: median_ms={fmt(stats.get('median_ms'))} p99_ms={fmt(stats.get('p99_ms'))} "
        f"n={int(stats.get('n') or 0)} shift=0 {'ok' if stats.get('ok') else 'FAIL'}"
    )


def _row_id(row: Mapping[str, Any]) -> int | None:
    try:
        return int(row["i"])
    except (KeyError, TypeError, ValueError):
        return None


def _pieces(kept: list[Mapping[str, Any]], plan: Iterable[Mapping[str, Any]] | None) -> dict[int, tuple[int, float]]:
    """Row id -> (piece id, piece end sim_t). A piece is a run of kept rows with one plan label and segment."""
    labels = None
    if plan is not None:
        from python.data.cam_hold import plan_labels

        labels = plan_labels(plan)
    out: dict[int, tuple[int, float]] = {}
    ends: dict[int, float] = {}
    piece = -1
    prev_key: Any = None
    for row in kept:
        rid = _row_id(row)
        if rid is None:
            prev_key = None
            continue
        label = "all" if labels is None else labels.get(rid)
        if label is None:
            prev_key = None
            continue
        key = (label, row.get("segment"))
        if key != prev_key:
            piece += 1
            prev_key = key
        st = row_sim_t(row)
        if st is not None:
            ends[piece] = max(ends.get(piece, -math.inf), st)
        out[rid] = (piece, math.nan)
    return {rid: (pid, ends.get(pid, -math.inf)) for rid, (pid, _nan) in out.items()}


def _warmup(kept: list[Mapping[str, Any]]) -> set[int]:
    """The kept row after a gap > WARMUP_GAP_S (from ``t``; dt_s may be stale) and the next 15."""
    warm: set[int] = set()
    left = 0
    prev_t: float | None = None
    for row in kept:
        t = _num(row.get("t"))
        if t is not None and prev_t is not None and t - prev_t > WARMUP_GAP_S:
            left = WARMUP_ROWS
        if t is not None:
            prev_t = t
        if left > 0:
            rid = _row_id(row)
            if rid is not None:
                warm.add(rid)
            left -= 1
    return warm


def _empty_counts() -> dict[str, int]:
    out = {"rows": 0, "valid": 0, **{name: 0 for name in REASONS}, "warmup": 0, "events": 0, "unplanned": 0}
    for h in CTL_HORIZONS_S[1:]:
        out[f"fut_h{h}"] = 0
    return out


def build_ctl(
    full_rows: list[Mapping[str, Any]],
    kept_rows: list[Mapping[str, Any]],
    *,
    plan: Iterable[Mapping[str, Any]] | None = None,
    impacts: Iterable[Mapping[str, Any]] = (),
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Row id -> ``(ctl (3, K), ctl_valid (3, K))`` for every kept row, plus the mask counts.

    ``full_rows`` is the whole recording in file order (``load_scene_rows``,
    before any filter); ``kept_rows`` are the rows training sees, in order;
    ``plan`` is the startup pass plan (train pieces and holdout slices).
    """
    from python.train.event_mask import event_mask

    full = [row for row in full_rows if isinstance(row, Mapping)]
    events, _counts = event_mask(full, list(impacts))
    dirty = [bool(ev) or bool(row.get("recover")) or _reverse(row) or _pbrake(row) for ev, row in zip(events, full)]
    event_row = [bool(ev) or bool(row.get("recover")) for ev, row in zip(events, full)]
    controls = [row_controls(row) for row in full]
    sims = [row_sim_t(row) for row in full]
    pos: dict[int, int] = {}
    for k, row in enumerate(full):
        rid = _row_id(row)
        if rid is not None:
            pos.setdefault(rid, k)
    warm = _warmup(kept_rows)
    pieces = _pieces(kept_rows, plan)
    counts: dict[str, Any] = _empty_counts()
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    horizon_max = float(max(CTL_HORIZONS_S))
    for row in kept_rows:
        rid = _row_id(row)
        if rid is None:
            continue
        counts["rows"] += 1
        ctl = np.zeros((3, K), dtype=np.float32)
        valid = np.zeros((3, K), dtype=np.float32)
        out[rid] = (ctl, valid)
        k = pos.get(rid)
        vals, reason = controls[k] if k is not None else row_controls(row)
        if reason is not None:
            counts[reason] += 1
            continue
        if (k is not None and event_row[k]) or bool(row.get("settle")) or bool(row.get("stall")) or bool(row.get("recover")):
            counts["events"] += 1
            continue
        if rid in warm:
            counts["warmup"] += 1
            continue
        if rid not in pieces:
            counts["unplanned"] += 1
            continue
        counts["valid"] += 1
        ctl[:, 0] = vals
        valid[:, 0] = 1.0
        s0 = sims[k] if k is not None else row_sim_t(row)
        if k is None or s0 is None:
            for h in CTL_HORIZONS_S[1:]:
                counts[f"fut_h{h}"] += 1
            continue
        piece_end = pieces[rid][1]
        seg = full[k].get("segment")
        # One walk forward over the full rows; it stops at the first break.
        walked: list[tuple[int, float, bool]] = []
        prev = s0
        m = k + 1
        while m < len(full):
            st = sims[m]
            if full[m].get("segment") != seg or st is None or st < prev or st - prev > MAX_STEP_S:
                break
            walked.append((m, st, dirty[m]))
            prev = st
            if st > s0 + horizon_max + PICK_TOL_S:
                break
            m += 1
        for h in CTL_HORIZONS_S[1:]:
            goal = s0 + float(h)
            best = None
            for idx, (mm, st, _bad) in enumerate(walked):
                gap = abs(st - goal)
                if gap <= PICK_TOL_S and (best is None or gap < best[0]):
                    best = (gap, idx)
            ok = best is not None and any(st >= goal for _mm, st, _bad in walked)
            if ok:
                j_idx = best[1]
                j, st_j, _ = walked[j_idx]
                reach = max(st_j, goal)
                ok = not any(bad for _mm, st, bad in walked if st <= reach)
                j_vals, j_reason = controls[j]
                jid = _row_id(full[j])
                ok = ok and j_reason is None and jid not in warm and st_j <= piece_end
            if ok:
                ctl[:, h] = j_vals
                valid[:, h] = 1.0
            else:
                counts[f"fut_h{h}"] += 1
    counts["lock_deg"] = steer_lock_deg(full)
    return out, counts


def steer_lock_deg(full_rows: Iterable[Mapping[str, Any]]) -> float | None:
    """The last ``veh_meta.steer_lock_deg`` seen (it is only on a spawn's first rows)."""
    lock = None
    for row in full_rows:
        meta = row.get("veh_meta") if isinstance(row, Mapping) else None
        value = _num(meta.get("steer_lock_deg")) if isinstance(meta, Mapping) else None
        if value is not None and value > 0.0:
            lock = value
    return lock


def attach_ctl(
    full_rows: list[Mapping[str, Any]],
    kept_rows: list[dict[str, Any]],
    *,
    plan: Iterable[Mapping[str, Any]] | None = None,
    impacts: Iterable[Mapping[str, Any]] = (),
    cond: bool = False,
) -> dict[str, Any]:
    """Write ``ctl`` and ``ctl_valid`` on every kept row (in place); return the counts.

    ``cond=True`` (B'1) also writes ``ctl_cond`` (5,) and ``ctl_cond_ok`` from :func:`build_cond`.
    A row with no usable t-dt row gets zeros, ``ctl_cond_ok`` 0 and an all-zero ``ctl_valid``
    (no ctl loss or score); it is counted as ``nocond``. Off: nothing changes.
    """
    targets, counts = build_ctl(full_rows, kept_rows, plan=plan, impacts=impacts)
    conds = build_cond(full_rows, kept_rows) if cond else {}
    if cond:
        counts["nocond"] = 0
    for row in kept_rows:
        rid = _row_id(row)
        if rid is not None and rid in targets:
            row["ctl"], row["ctl_valid"] = targets[rid]
        if cond and rid is not None and rid in conds:
            vec, ok = conds[rid]
            row["ctl_cond"], row["ctl_cond_ok"] = vec, ok
            if ok < 0.5:
                counts["nocond"] += 1
                if "ctl_valid" in row:
                    row["ctl_valid"] = np.zeros_like(np.asarray(row["ctl_valid"], dtype=np.float32))
    return counts


def mask_line(counts: Mapping[str, Any]) -> str:
    text = (
        f"[GVD] ctl mask: rows={int(counts['rows'])} valid={int(counts['valid'])} null={int(counts['null'])} "
        f"range={int(counts['range'])} reverse={int(counts['reverse'])} pbrake={int(counts['pbrake'])} "
        f"warmup={int(counts['warmup'])} fut_h3={int(counts['fut_h3'])} fut_h4={int(counts['fut_h4'])} "
        f"fut_h5={int(counts['fut_h5'])}"
    )
    if int(counts.get("events", 0) or 0):
        # Only without --mask-events: event rows are kept but carry no ctl target.
        text += f" events={int(counts['events'])}"
    if "nocond" in counts:
        # Only with --ctl-cond: rows with no t-dt row carry no ctl target.
        text += f" nocond={int(counts['nocond'])}"
    return text


def _unique_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen: set[Any] = set()
    out = []
    for row in rows:
        key = _row_id(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def ctl_scored(row: Mapping[str, Any] | None) -> bool:
    """True when a row counts for ctl scoring and B1: it has targets and passes ``loss_mask``.

    Same gate as training (rows that fail ``scene_net.loss_mask`` get no loss).
    """
    if not row or "ctl" not in row or "ctl_valid" not in row:
        return False
    from python.train.scene_net import loss_mask

    return bool(loss_mask(row))


def b1_constants(train_rows: Iterable[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Per (channel, horizon) median and mean of the valid train-split targets (B1).

    Only rows that pass ``loss_mask`` (:func:`ctl_scored`), as in training.
    """
    median = np.zeros((3, K), dtype=np.float32)
    mean = np.zeros((3, K), dtype=np.float32)
    rows = [row for row in _unique_rows(train_rows) if ctl_scored(row)]
    if not rows:
        return {"median": median, "mean": mean, "n": np.zeros((3, K), dtype=np.int64)}
    ctl = np.stack([np.asarray(row["ctl"], dtype=np.float32) for row in rows])
    valid = np.stack([np.asarray(row["ctl_valid"], dtype=np.float32) for row in rows]) > 0.5
    n = valid.sum(axis=0)
    for c in range(3):
        for h in range(K):
            vals = ctl[valid[:, c, h], c, h]
            if vals.size:
                median[c, h] = float(np.median(vals))
                mean[c, h] = float(vals.mean())
    return {"median": median, "mean": mean, "n": n.astype(np.int64)}


def ctl_bias(mean: np.ndarray) -> np.ndarray:
    """``ctl_head`` bias (18,): atanh of the steer means, logit of the throttle and brake means."""
    mean = np.asarray(mean, dtype=np.float64).reshape(3, K)
    out = np.zeros((3, K), dtype=np.float64)
    out[0] = np.arctanh(np.clip(mean[0], -0.95, 0.95))
    pedal = np.clip(mean[1:], 0.01, 0.99)
    out[1:] = np.log(pedal / (1.0 - pedal))
    return out.reshape(-1).astype(np.float32)


def f1_score(pred: np.ndarray, label: np.ndarray, thr: float = BRAKE_ON) -> float:
    """F1 of ``pred > thr`` against ``label > thr``. 0 when nothing is predicted positive."""
    p = np.asarray(pred, dtype=np.float64) > thr
    y = np.asarray(label, dtype=np.float64) > thr
    tp = float(np.sum(p & y))
    if not p.any() or not y.any() or tp == 0.0:
        return 0.0
    precision = tp / float(p.sum())
    recall = tp / float(y.sum())
    return 2.0 * precision * recall / (precision + recall)


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2:
        return None
    sa, sb = float(a.std()), float(b.std())
    if sa < 1e-12 or sb < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def b2_pairs(rows: list[Mapping[str, Any]], max_dt: float = 0.25) -> list[int | None]:
    """Index of the previous scored row in the same segment within ``max_dt`` s (``t``), else None."""
    out: list[int | None] = []
    for k, row in enumerate(rows):
        if k == 0:
            out.append(None)
            continue
        prev = rows[k - 1]
        t, pt = _num(row.get("t")), _num(prev.get("t"))
        same = prev.get("segment") == row.get("segment")
        out.append(k - 1 if same and t is not None and pt is not None and 0.0 <= t - pt <= max_dt else None)
    return out


def _mae(a: np.ndarray, b: np.ndarray) -> float | None:
    return float(np.mean(np.abs(a - b))) if a.size else None


def _rmse(a: np.ndarray, b: np.ndarray) -> float | None:
    return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else None


def _metrics(y: np.ndarray, p: np.ndarray, b3: np.ndarray, b4: np.ndarray, b2y: np.ndarray, b2p: np.ndarray,
             b1: Mapping[str, np.ndarray], h: int, lock_deg: float | None) -> dict[str, Any]:
    n = int(y.shape[0])
    med = np.asarray(b1["median"], dtype=np.float64)[:, h]
    mean = np.asarray(b1["mean"], dtype=np.float64)[:, h]
    out: dict[str, Any] = {"n": n}
    names = ("steer", "thr", "brk")
    for c, name in enumerate(names):
        out[f"{name}_mae"] = _mae(p[:, c], y[:, c])
        out[f"{name}_rmse"] = _rmse(p[:, c], y[:, c])
        out[f"b1med_{name}"] = _mae(np.full(n, med[c]), y[:, c])
        out[f"b1mean_{name}"] = _rmse(np.full(n, mean[c]), y[:, c])
        out[f"b3_{name}"] = _mae(b3[:, c], y[:, c])
        out[f"b4_{name}"] = _mae(b4[:, c], y[:, c])
        out[f"b2_{name}"] = _mae(b2p[:, c], b2y[:, c]) if b2y.size else None
    out["b2_n"] = int(b2y.shape[0])
    out["steer_r"] = _pearson(p[:, 0], y[:, 0]) if n else None
    out["steer_mae_deg"] = None if out["steer_mae"] is None or lock_deg is None else out["steer_mae"] * float(lock_deg)
    out["brk_events"] = int(np.sum(y[:, 2] > BRAKE_ON)) if n else 0
    out["brk_f1"] = f1_score(p[:, 2], y[:, 2]) if n else None
    out["b0_brk_f1"] = f1_score(np.zeros(n), y[:, 2]) if n else None
    out["b0_brk_mae"] = _mae(np.zeros(n), y[:, 2])
    out["b3_brk_f1"] = f1_score(b3[:, 2], y[:, 2]) if n else None
    out["b4_brk_f1"] = f1_score(b4[:, 2], y[:, 2]) if n else None
    return out


def _ref(m: Mapping[str, Any], name: str) -> float | None:
    vals = [m.get(f"b3_{name}"), m.get(f"b1med_{name}")]
    vals = [float(v) for v in vals if v is not None]
    return min(vals) if vals else None


def _slice_rev5(m: Mapping[str, Any], h: int) -> str:
    """Rev 5: ref = min(B3, B1 med); B4 and h0 report-only; brake graded at >= 30 events.

    Under 30 brake events the slice is UNPROVEN, never PASS (Rev 4's rule kept); a steer or
    throttle miss still FAILs it.
    """
    if h not in REV5_BARS:
        return "REPORT"
    if int(m.get("n") or 0) < MIN_N:
        return "UNPROVEN"
    k_steer, k_thr = REV5_BARS[h]
    fail = m["steer_mae"] > k_steer * _ref(m, "steer") or m["thr_mae"] > k_thr * _ref(m, "thr")
    brake_open = int(m.get("brk_events") or 0) < MIN_BRAKE_EVENTS
    if not brake_open:
        fail = fail or m["brk_f1"] < 0.5 or not (m["brk_mae"] < m["b1med_brk"])
    if fail:
        return "FAIL"
    return "UNPROVEN" if brake_open else "PASS"


def slice_verdict(m: Mapping[str, Any], h: int, *, bar: str = REV4) -> str:
    """PASS / FAIL / UNPROVEN for one slice at one horizon. n < 300 is UNPROVEN, never PASS.

    ``bar`` "rev4" (default; reproduces runs before B') or "rev5" (B' series; "REPORT" off h3-h5).
    """
    if bar == REV5:
        return _slice_rev5(m, h)
    if bar != REV4:
        raise ValueError(f"unknown ctl bar {bar!r}")
    if int(m.get("n") or 0) < MIN_N:
        return "UNPROVEN"
    fail = m["steer_mae"] > 0.85 * m["b1med_steer"] or m["thr_mae"] > 0.85 * m["b1med_thr"]
    if h in B4_GRADED:
        fail = fail or m["steer_mae"] > 0.95 * m["b4_steer"] or m["thr_mae"] > 0.95 * m["b4_thr"]
    brake_open = int(m.get("brk_events") or 0) < MIN_BRAKE_EVENTS
    if not brake_open:
        fail = fail or m["brk_f1"] < 0.5 or not (m["brk_mae"] < m["b1med_brk"])
        if h in B4_GRADED:
            fail = fail or not (m["brk_f1"] > m["b4_brk_f1"])
    if fail:
        return "FAIL"
    return "UNPROVEN" if brake_open else "PASS"


def horizon_verdict(verdicts: Iterable[str]) -> str:
    seen = list(verdicts)
    if "FAIL" in seen:
        return "FAIL"
    return "PASS" if seen and all(v == "PASS" for v in seen) else "UNPROVEN"


def run_verdict(by_h: Mapping[int, str], lane: str, *, bar: str = REV4) -> str:
    """FAIL if any graded horizon or the lane bar fails.

    PASS only if all graded horizons (Rev 4: h0, h3-h5; Rev 5: h3-h5) pass AND the lane bar
    passes; an unproven lane bar (no usable run-A data) caps the run at UNPROVEN.
    """
    graded = [by_h.get(h, "UNPROVEN") for h in (GRADED_REV5 if bar == REV5 else GRADED)]
    if "FAIL" in graded or lane == "FAIL":
        return "FAIL"
    return "PASS" if all(v == "PASS" for v in graded) and lane == "PASS" else "UNPROVEN"


def lane_verdict(rms: float | None, paired_rms: float | None) -> str:
    """Holdout lane rms within 1.05x the paired lane_gt-only run. No paired run: UNPROVEN."""
    if rms is None or paired_rms is None:
        return "UNPROVEN"
    return "PASS" if float(rms) <= 1.05 * float(paired_rms) + 1e-12 else "FAIL"


LANE_BAR = 1.05


def holdout_rows_sha(rows: Any) -> str:
    """sha256 of the ordered holdout row keys ``segment:i``, one per line (the eval JSON ``rows_sha``)."""
    import hashlib

    text = "".join(f"{row.get('segment')}:{row.get('i')}\n" for row in rows)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def lane_bar(result: Mapping[str, Any], ref_path: Any) -> dict[str, Any]:
    """Lane bar from run A's holdout eval JSON (``eval_onnx_holdout --json``, lane_gt-only run).

    PASS when B's holdout lane rms <= 1.05 x A's; FAIL above. UNPROVEN when
    there is no path, the file is missing or unreadable, either rms is not a
    finite number, A has a ctl block (not a lane_gt-only run), A failed its
    ``--expect-epoch`` check, or ``rows_sha`` (the holdout rows) or ``epoch``
    is missing on either side or differs.
    """
    out: dict[str, Any] = {"verdict": "UNPROVEN", "rms": None, "ref_rms": None, "ratio": None, "bar": LANE_BAR,
                           "ref": None if ref_path is None else str(ref_path), "reason": ""}

    def finite(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        return num if math.isfinite(num) else None

    rms = finite(result.get("rms"))
    out["rms"] = rms
    if ref_path is None:
        out["reason"] = "no --lane-ref (run A eval JSON)"
        return out
    try:
        with open(ref_path, encoding="utf-8") as handle:
            ref = json.load(handle)
    except FileNotFoundError:
        out["reason"] = "run A eval JSON missing"
        return out
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        out["reason"] = f"run A eval JSON unreadable ({type(exc).__name__})"
        return out
    if not isinstance(ref, Mapping):
        out["reason"] = "run A eval JSON unreadable (not an object)"
        return out
    ref_rms = finite(ref.get("rms"))
    out["ref_rms"] = ref_rms
    if ref_rms is None or ref_rms <= 0.0 or rms is None:
        out["reason"] = "no rms (run A or this run)"
        return out
    if "ctl" in ref:
        out["reason"] = "run A is not a lane_gt-only run (it has a ctl block)"
        return out
    if ref.get("epoch_ok") is False:
        out["reason"] = "run A failed its --expect-epoch check"
        return out
    a_sha, b_sha = ref.get("rows_sha"), result.get("rows_sha")
    if not a_sha or not b_sha:
        out["reason"] = f"rows_sha missing (A={a_sha} B={b_sha})"
        return out
    if str(a_sha) != str(b_sha):
        out["reason"] = f"holdout differs (rows_sha: A={str(a_sha)[:12]} B={str(b_sha)[:12]})"
        return out
    a_ep, b_ep = ref.get("epoch"), result.get("epoch")
    if a_ep is None or b_ep is None or isinstance(a_ep, bool) or isinstance(b_ep, bool):
        out["reason"] = f"epoch missing (A={a_ep} B={b_ep})"
        return out
    try:
        same_epoch = int(a_ep) == int(b_ep)
    except (TypeError, ValueError):
        same_epoch = False
    if not same_epoch:
        out["reason"] = f"epoch differs (A={a_ep} B={b_ep})"
        return out
    out["ratio"] = rms / ref_rms
    out["verdict"] = "PASS" if rms <= LANE_BAR * ref_rms + 1e-12 else "FAIL"
    out["reason"] = f"rms {rms:.6g} vs {LANE_BAR} x A {ref_rms:.6g}"
    return out


def score_ctl(
    records: list[tuple[Mapping[str, Any], np.ndarray]],
    b1: Mapping[str, np.ndarray],
    *,
    lock_deg: float | None = None,
    lane: str = "UNPROVEN",
    bar: str = REV4,
) -> dict[str, Any]:
    """Per-horizon scores on ``records`` = ``[(row, pred (3, K)), ...]`` in eval order.

    ``bar``: "rev4" (default) or "rev5" (B' series). Rev 5 adds ``ref_steer`` / ``ref_thr`` /
    ``brk_graded`` and marks h0-h2 "REPORT".

    The model and every baseline use the same valid (row, h) set; B2 (report
    only) skips a row without a previous row within 0.25 s.
    """
    # Same gate as the training loss: a row that fails loss_mask is not scored (model or baseline).
    records = [(row, pred) for row, pred in records if ctl_scored(row)]
    rows = [row for row, _pred in records]
    preds = [np.asarray(pred, dtype=np.float64).reshape(3, K) for _row, pred in records]
    prev = b2_pairs(rows)
    if bar not in BARS:
        raise ValueError(f"unknown ctl bar {bar!r}")
    out: dict[str, Any] = {"h": {}, "lane": lane}
    if bar == REV5:
        out["bar"] = REV5

    def extra(m: dict[str, Any]) -> dict[str, Any]:
        if bar == REV5:
            m["ref_steer"], m["ref_thr"] = _ref(m, "steer"), _ref(m, "thr")
            m["brk_graded"] = int(int(m.get("brk_events") or 0) >= MIN_BRAKE_EVENTS)
        return m

    for h in CTL_HORIZONS_S:
        picked = [k for k, row in enumerate(rows)
                  if "ctl_valid" in row and bool(np.all(np.asarray(row["ctl_valid"])[:, h] > 0.5))]

        def arrays(sel: list[int]) -> tuple[np.ndarray, ...]:
            y = np.asarray([np.asarray(rows[k]["ctl"], dtype=np.float64)[:, h] for k in sel]).reshape(-1, 3)
            p = np.asarray([preds[k][:, h] for k in sel]).reshape(-1, 3)
            b3 = np.asarray([np.asarray(rows[k]["ctl"], dtype=np.float64)[:, 0] for k in sel]).reshape(-1, 3)
            b4 = np.asarray([preds[k][:, 0] for k in sel]).reshape(-1, 3)
            b2y, b2p = [], []
            for k in sel:
                j = prev[k]
                if j is None or "ctl_valid" not in rows[j] or not np.all(np.asarray(rows[j]["ctl_valid"])[:, h] > 0.5):
                    continue
                b2y.append(np.asarray(rows[k]["ctl"], dtype=np.float64)[:, h])
                b2p.append(np.asarray(rows[j]["ctl"], dtype=np.float64)[:, h])
            return y, p, b3, b4, np.asarray(b2y).reshape(-1, 3), np.asarray(b2p).reshape(-1, 3)

        overall = extra(_metrics(*arrays(picked), b1, h, lock_deg))
        overall["verdict_all"] = slice_verdict(overall, h, bar=bar)
        slices: dict[str, Any] = {}
        verdicts = [overall["verdict_all"]]
        for spawn, key in GATED_SLICES + tuple((name, name) for name in REPORT_SLICES):
            sel = [k for k in picked if rows[k].get("spawn_type") == spawn]
            m = extra(_metrics(*arrays(sel), b1, h, lock_deg))
            m["verdict"] = slice_verdict(m, h, bar=bar)
            slices[key] = m
            if spawn not in REPORT_SLICES:
                verdicts.append(m["verdict"])
        overall["slices"] = slices
        overall["verdict"] = "REPORT" if bar == REV5 and h not in GRADED_REV5 else horizon_verdict(verdicts)
        out["h"][h] = overall
    out["run"] = run_verdict({h: out["h"][h]["verdict"] for h in CTL_HORIZONS_S}, lane, bar=bar)
    return out


def _fmt(value: Any) -> str:
    if value is None:
        return "na"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    return f"{float(value):.4f}"


def val_lines(result: Mapping[str, Any]) -> list[str]:
    rev5 = result.get("bar") == REV5
    lines = []
    for h in CTL_HORIZONS_S:
        m = result["h"][h] if h in result["h"] else result["h"][str(h)]
        sl = m["slices"]
        ref = (f"ref_steer={_fmt(m.get('ref_steer'))} ref_thr={_fmt(m.get('ref_thr'))} "
               f"brk_events={int(m.get('brk_events') or 0)} brk_graded={int(m.get('brk_graded') or 0)} ") if rev5 else ""
        lines.append(
            f"[GVD] val ctl h={h} n={int(m['n'])} steer_mae={_fmt(m['steer_mae'])} steer_r={_fmt(m['steer_r'])} "
            f"thr_mae={_fmt(m['thr_mae'])} brk_mae={_fmt(m['brk_mae'])} brk_f1={_fmt(m['brk_f1'])} "
            f"b1med_steer={_fmt(m['b1med_steer'])} b1mean_steer={_fmt(m['b1mean_steer'])} "
            f"b0_brk_f1={_fmt(m['b0_brk_f1'])} b2_steer={_fmt(m['b2_steer'])} b3_steer={_fmt(m['b3_steer'])} "
            f"b3_brk_f1={_fmt(m['b3_brk_f1'])} b4_steer={_fmt(m['b4_steer'])} b4_thr={_fmt(m['b4_thr'])} "
            f"b4_brk_f1={_fmt(m['b4_brk_f1'])} city_steer={_fmt(sl['city']['steer_mae'])} "
            f"hwy_steer={_fmt(sl['hwy']['steer_mae'])} ramp_steer={_fmt(sl['ramp']['steer_mae'])} "
            f"{ref}verdict={m['verdict']}"
        )
    hv = {h: (result["h"][h] if h in result["h"] else result["h"][str(h)])["verdict"] for h in GRADED}
    tag = f" bar={REV5}" if rev5 else ""
    lines.append(
        f"[GVD] val ctl run verdict={result['run']}{tag} h0={hv[0]} h3={hv[3]} h4={hv[4]} h5={hv[5]} "
        f"lane={result['lane']}"
    )
    return lines


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def ctl_record(result: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-safe copy (string horizon keys) for metrics.jsonl and the offline eval."""
    return _plain(result)
