"""Summarize a ground-truth JSONL session. No BeamNG."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

IN_LANE_HALF = 0.5
DT_CAP_S = 1.0
STEER_DEAD = 0.02
# A run that stays in the lane because the car is parked is not a pass.
# gt_run9: 120 s engaged, in_lane 100%, driven 36.7 m, mean speed 0.37 m/s.
MIN_DRIVEN_M = 300.0
MIN_MEAN_SPEED_MPS = 4.0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return rows


def _f(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


# LATERAL FRAME: lane_offset_m is the nav-road offset of the state pos (ref node).
def _in_lane(row: dict[str, Any]) -> bool | None:
    """True when the car is inside its lane. None when the offset is missing."""
    if row.get("off_road") is True:
        return False
    off = _f(row, "lane_offset_m")
    width = _f(row, "lane_width_m")
    if off is None or width is None or width <= 0.0:
        return None
    return abs(off) <= width * IN_LANE_HALF


def _times(rows: list[dict[str, Any]]) -> list[float]:
    out: list[float] = []
    for i, row in enumerate(rows):
        t = _f(row, "t")
        out.append(float(i) if t is None else t)
    return out


def _dts(times: list[float]) -> list[float]:
    dts = [0.0]
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if dt < 0.0:
            dt = 0.0
        elif dt > DT_CAP_S:
            dt = DT_CAP_S
        dts.append(dt)
    return dts


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if dx < 1e-12 or dy < 1e-12:
        return None
    return num / (dx * dy)


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
    return ordered[idx]


def steer_sign_changes(steers: list[float]) -> int:
    """Sign changes, ignoring samples inside the deadband."""
    signs: list[int] = []
    for value in steers:
        if abs(value) < STEER_DEAD:
            continue
        signs.append(1 if value > 0.0 else -1)
    return sum(1 for a, b in zip(signs, signs[1:]) if a * b < 0)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Engaged lane-keep, departures, steer chatter, and latency.

    A departure is the rising edge of "engaged and measured and not in the
    lane". An off-road event is the rising edge of ``off_road`` while
    engaged. Time steps longer than a second are capped so a pause does not
    dominate the percentage.
    """
    times = _times(rows)
    dts = _dts(times)
    engaged_s = 0.0
    in_lane_s = 0.0
    known_s = 0.0
    offsets: list[float] = []
    departures: list[dict[str, Any]] = []
    off_events: list[dict[str, Any]] = []
    was_out = False
    was_off = False
    corr_x: list[float] = []
    corr_y: list[float] = []
    steers: list[float] = []
    lats: list[float] = []
    for row, dt, t in zip(rows, dts, times):
        engaged = bool(row.get("engaged"))
        inside = _in_lane(row)
        off_road = bool(row.get("off_road"))
        if engaged:
            engaged_s += dt
            steers.append(_f(row, "steer") or 0.0)
            if inside is not None:
                known_s += dt
                if inside:
                    in_lane_s += dt
            off = _f(row, "lane_offset_m")
            if off is not None:
                offsets.append(abs(off))
            lx = _f(row, "lane_x")
            if off is not None and lx is not None:
                corr_x.append(lx)
                corr_y.append(off)
            ms = _f(row, "frame_cmd_ms")
            if ms is not None and ms >= 0.0:
                lats.append(ms)
            out = inside is False
            if out and not was_out:
                departures.append({
                    "t": t,
                    "lane_offset_m": off,
                    "road_id": row.get("road_id"),
                    "lane_id": row.get("lane_id"),
                })
            was_out = out
            if off_road and not was_off:
                off_events.append({"t": t, "road_id": row.get("road_id"), "dist_m": _f(row, "dist_m")})
            was_off = off_road
        else:
            was_out = False
            was_off = False
    flips = steer_sign_changes(steers)
    pct = None if known_s <= 1e-6 else 100.0 * in_lane_s / known_s
    driven_m, mean_speed = _travel(rows, dts)
    return {
        "ticks": len(rows),
        "engaged_s": engaged_s,
        "in_lane_s": in_lane_s,
        "known_s": known_s,
        "in_lane_pct": pct,
        "mean_abs_offset_m": None if not offsets else sum(offsets) / len(offsets),
        "max_abs_offset_m": None if not offsets else max(offsets),
        "departures": departures,
        "off_road": off_events,
        "steer_sign_changes": flips,
        "steer_sign_per_s": None if engaged_s <= 1e-6 else flips / engaged_s,
        "lane_x_vs_offset_r": pearson(corr_x, corr_y),
        "driven_m": driven_m,
        "mean_speed_mps": mean_speed,
        "frame_cmd_ms_n": len(lats),
        "frame_cmd_ms_mean": None if not lats else sum(lats) / len(lats),
        "frame_cmd_ms_p50": None if not lats else _percentile(lats, 0.50),
        "frame_cmd_ms_p90": None if not lats else _percentile(lats, 0.90),
        **_perception(rows),
    }


def _travel(rows: list[dict[str, Any]], dts: list[float]) -> tuple[float, float | None]:
    """Path length and mean ``speed_mps`` while engaged.

    Distance is the sum of position steps. When a row has no position, that
    step uses ``speed_mps * dt`` instead. The mean is the speed field, which
    is what stays near 0 when the parking brake is holding the car.
    """
    driven = 0.0
    speeds: list[float] = []
    prev: tuple[float, float] | None = None
    for row, dt in zip(rows, dts):
        if not row.get("engaged"):
            prev = None
            continue
        spd = _f(row, "speed_mps")
        if spd is not None:
            speeds.append(spd)
        pos = row.get("pos")
        xy: tuple[float, float] | None = None
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            try:
                xy = (float(pos[0]), float(pos[1]))
            except (TypeError, ValueError):
                xy = None
        if xy is not None and prev is not None:
            driven += math.hypot(xy[0] - prev[0], xy[1] - prev[1])
        elif spd is not None:
            driven += max(0.0, spd) * dt
        prev = xy
    mean = None if not speeds else sum(speeds) / len(speeds)
    return driven, mean


def distance_failure(summary: dict[str, Any]) -> str | None:
    """Why this run did not drive far enough, or None when it did."""
    driven = summary.get("driven_m")
    speed = summary.get("mean_speed_mps")
    driven_v = 0.0 if driven is None else float(driven)
    speed_v = 0.0 if speed is None else float(speed)
    if driven_v >= MIN_DRIVEN_M and speed_v >= MIN_MEAN_SPEED_MPS:
        return None
    return (
        f"driven_m={driven_v:.1f} need>={MIN_DRIVEN_M:.0f} "
        f"mean_speed_mps={speed_v:.2f} need>={MIN_MEAN_SPEED_MPS:.0f}"
    )


def _perception(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """lane_x against the ground-truth center in the ego frame."""
    xs: list[float] = []
    truth: list[float] = []
    for row in rows:
        if not row.get("engaged"):
            continue
        lx = _f(row, "lane_x")
        gt = _f(row, "gt_center_x")
        if lx is None or gt is None:
            continue
        xs.append(lx)
        truth.append(gt)
    if not xs:
        return {
            "perception_n": 0,
            "perception_bias_m": None,
            "perception_rms_m": None,
            "lane_x_vs_gt_center_r": None,
        }
    err = [a - b for a, b in zip(xs, truth)]
    return {
        "perception_n": len(xs),
        "perception_bias_m": sum(err) / len(err),
        "perception_rms_m": math.sqrt(sum(e * e for e in err) / len(err)),
        "lane_x_vs_gt_center_r": pearson(xs, truth),
    }


def format_report(summary: dict[str, Any], *, source: str = "") -> str:
    def num(key: str, digits: int = 2) -> str:
        value = summary.get(key)
        if value is None:
            return "na"
        return f"{float(value):.{digits}f}"

    lines = []
    if source:
        lines.append(f"gt report: {source}")
    pct = summary.get("in_lane_pct")
    pct_s = "na" if pct is None else f"{float(pct):.1f}"
    lines.append(
        f"engaged_s={num('engaged_s')} in_lane_pct={pct_s} "
        f"mean_abs_offset_m={num('mean_abs_offset_m', 3)} "
        f"max_abs_offset_m={num('max_abs_offset_m', 3)} "
        f"driven_m={num('driven_m', 1)} mean_speed_mps={num('mean_speed_mps', 2)}"
    )
    deps = summary.get("departures") or []
    lines.append(f"departures={len(deps)}")
    for event in deps[:8]:
        off = event.get("lane_offset_m")
        off_s = "na" if off is None else f"{float(off):+.2f}"
        lines.append(
            f"  t={float(event['t']):.2f} lane_offset_m={off_s} "
            f"road={event.get('road_id')} lane={event.get('lane_id')}"
        )
    if len(deps) > 8:
        lines.append(f"  ... {len(deps) - 8} more")
    offs = summary.get("off_road") or []
    lines.append(f"off_road={len(offs)}")
    for event in offs[:8]:
        lines.append(f"  t={float(event['t']):.2f} road={event.get('road_id')} dist_m={event.get('dist_m')}")
    lines.append(
        f"steer_sign_changes={summary.get('steer_sign_changes')} "
        f"steer_sign_per_s={num('steer_sign_per_s', 3)} "
        f"lane_x_vs_offset_r={num('lane_x_vs_offset_r', 3)}"
    )
    lines.append(
        f"frame_cmd_ms n={summary.get('frame_cmd_ms_n')} "
        f"mean={num('frame_cmd_ms_mean', 1)} "
        f"p50={num('frame_cmd_ms_p50', 1)} p90={num('frame_cmd_ms_p90', 1)}"
    )
    lines.append(
        f"perception n={summary.get('perception_n')} "
        f"bias_m={num('perception_bias_m', 3)} "
        f"rms_m={num('perception_rms_m', 3)} "
        f"lane_x_vs_gt_center_r={num('lane_x_vs_gt_center_r', 3)}"
    )
    return "\n".join(lines)


def render_plot(rows: list[dict[str, Any]], path: Path) -> Path:
    """Three stacked traces: lane offset, lane_x, steer. Departure ticks are marked."""
    import cv2
    import numpy as np

    times = _times(rows)
    t0 = times[0] if times else 0.0
    xs = [t - t0 for t in times]
    series = [
        ("lane offset m", [_f(r, "lane_offset_m") for r in rows], (40, 90, 200)),
        ("lane_x m", [_f(r, "lane_x") for r in rows], (40, 140, 60)),
        ("steer", [_f(r, "steer") for r in rows], (180, 80, 30)),
    ]
    deps = {round(float(e["t"]) - t0, 3) for e in summarize(rows).get("departures") or []}
    width, panel_h, gap = 960, 180, 28
    height = gap + len(series) * (panel_h + gap)
    img = np.full((height, width, 3), 248, dtype=np.uint8)
    span = max(xs) - min(xs) if len(xs) > 1 else 1.0
    if span <= 0.0:
        span = 1.0

    def xpix(t: float) -> int:
        return 56 + int((t - min(xs)) / span * (width - 80))

    for i, (title, values, color) in enumerate(series):
        top = gap + i * (panel_h + gap)
        cv2.rectangle(img, (48, top), (width - 16, top + panel_h), (230, 230, 230), 1)
        finite = [v for v in values if v is not None]
        lo = min(finite) if finite else -1.0
        hi = max(finite) if finite else 1.0
        if abs(hi - lo) < 1e-6:
            lo -= 1.0
            hi += 1.0
        pad = 0.08 * (hi - lo)
        lo -= pad
        hi += pad

        def ypix(v: float) -> int:
            return top + panel_h - 12 - int((v - lo) / (hi - lo) * (panel_h - 24))

        zero = ypix(0.0) if lo < 0.0 < hi else None
        if zero is not None:
            cv2.line(img, (48, zero), (width - 16, zero), (210, 210, 210), 1)
        pts = []
        for t, v in zip(xs, values):
            if v is None:
                if len(pts) >= 2:
                    cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
                pts = []
                continue
            pts.append((xpix(t), ypix(v)))
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        for mark in deps:
            xx = xpix(mark)
            cv2.line(img, (xx, top), (xx, top + panel_h), (40, 40, 200), 1)
        cv2.putText(
            img, f"{title}  [{lo:+.2f} .. {hi:+.2f}]", (56, top + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA,
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path
