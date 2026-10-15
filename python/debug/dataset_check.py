"""Check a scene-lane recording without opening BeamNG.

Counts, timestamp gaps, NaNs, target ranges, blank camera sectors, and a
few montage PNGs with the ground-truth lane drawn. Pictures go to the
directory the caller passes, which on the driving PC is under
``E:\\GVD_data``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from python.data.strip_writer import CANVAS_H, CANVAS_W, SECTORS, read_state_jsonl
from python.debug.lane_targets import LOOKAHEADS_M, LANE_GT_DIM, lane_gt_from_record

GAP_S = 2.0
CENTER_ABS_MAX_M = 40.0
BLANK_MEAN = 2.0


def sector_is_blank(crop: np.ndarray, *, mean_max: float = BLANK_MEAN) -> bool:
    if crop is None or int(getattr(crop, "size", 0)) == 0:
        return True
    return float(np.asarray(crop).mean()) < float(mean_max)


def lanes_from_vector(values: Any) -> list[list[dict[str, float]]]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    left: list[dict[str, float]] = []
    right: list[dict[str, float]] = []
    for i, y in enumerate(LOOKAHEADS_M):
        base = i * 3
        if base + 2 >= arr.size:
            break
        left.append({"x": float(arr[base + 1]), "y": float(y)})
        right.append({"x": float(arr[base + 2]), "y": float(y)})
    return [left, right]


def _issues_for_row(row: dict[str, Any], prev_t: float | None) -> tuple[list[str], float | None]:
    issues: list[str] = []
    idx = row.get("i")
    try:
        t = float(row.get("t"))
    except (TypeError, ValueError):
        issues.append(f"row {idx}: t is not a number")
        t = None
    if t is not None and not math.isfinite(t):
        issues.append(f"row {idx}: t is not finite")
        t = None
    if t is not None and prev_t is not None:
        gap = t - prev_t
        if gap < 0.0:
            issues.append(f"row {idx}: time went backwards by {gap:.3f}s")
        elif gap > GAP_S:
            issues.append(f"row {idx}: gap {gap:.3f}s")
    raw_dt = row.get("dt_s")
    if raw_dt is not None:
        try:
            dt = float(raw_dt)
        except (TypeError, ValueError):
            issues.append(f"row {idx}: dt_s is not a number")
            dt = None
        if dt is not None and not math.isfinite(dt):
            issues.append(f"row {idx}: dt_s is not finite")
        elif dt is not None and dt < 0.0:
            issues.append(f"row {idx}: dt_s {dt:.3f}s is negative")
    tech = row.get("tech") if isinstance(row.get("tech"), dict) else row
    values, mask = lane_gt_from_record(tech if isinstance(tech, dict) else None)
    if int(values.size) != LANE_GT_DIM:
        issues.append(f"row {idx}: lane_gt length {int(values.size)}")
    for slot, value in enumerate(values.tolist()):
        if not math.isfinite(float(value)):
            issues.append(f"row {idx}: lane_gt[{slot}] is NaN or Inf")
    heading = float(values[12]) if values.size > 12 else 0.0
    if float(mask[12]) > 0.0 and abs(heading) > math.pi + 1e-3:
        issues.append(f"row {idx}: heading {heading:.3f} rad is outside [-pi, pi]")
    conf = float(values[-1]) if values.size else 0.0
    if conf < -1e-3 or conf > 1.0 + 1e-3:
        issues.append(f"row {idx}: confidence {conf:.3f} is outside [0, 1]")
    for i in range(len(LOOKAHEADS_M)):
        base = i * 3
        if base + 2 >= int(mask.size):
            break
        if float(mask[base]) <= 0.0:
            continue
        for slot, name in ((0, "center"), (1, "left"), (2, "right")):
            got = float(values[base + slot])
            if abs(got) > CENTER_ABS_MAX_M:
                issues.append(
                    f"row {idx}: {name} at {LOOKAHEADS_M[i]:.0f}m is {got:.1f} m"
                )
    return issues, t


def _blank_counts(path: Path) -> tuple[dict[str, int] | None, str | None]:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None, f"unreadable {path.name}"
    if int(bgr.shape[0]) != CANVAS_H or int(bgr.shape[1]) != CANVAS_W:
        return None, f"{path.name} is {bgr.shape[1]}x{bgr.shape[0]}, expected {CANVAS_W}x{CANVAS_H}"
    counts = {sec.cam_id: 0 for sec in SECTORS}
    for sec in SECTORS:
        crop = bgr[:, sec.x0 : sec.x1]
        if sector_is_blank(crop):
            counts[sec.cam_id] = 1
    return counts, None


def _draw_montage(path: Path, strip_path: Path, values: np.ndarray, title: str) -> bool:
    import cv2

    from python.debug.gt_logger import save_debug_frame

    bgr = cv2.imread(str(strip_path), cv2.IMREAD_COLOR)
    lanes = lanes_from_vector(values)
    center = None
    if values.size >= 3:
        # 8 m sits between the 5 m and 10 m samples.
        center = float(values[0]) + 0.6 * (float(values[3]) - float(values[0]))
    saved = save_debug_frame(
        path,
        stitch=bgr,
        lanes_bev=lanes,
        hud=[title],
        lane_x=center,
    )
    return saved is not None


def validate_dataset(
    root: Path | str,
    out_dir: Path | str,
    *,
    montage_n: int = 4,
) -> dict[str, Any]:
    """Write ``report.json`` and up to ``montage_n`` PNGs. Returns the report."""
    root = Path(root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_state_jsonl(root / "state.jsonl")
    strips = root / "strips"
    jpegs = sorted(p for p in strips.glob("*.jpg") if p.is_file()) if strips.is_dir() else []
    jpeg_ids = set()
    for path in jpegs:
        head = path.stem
        if head.isdigit():
            jpeg_ids.add(int(head))
    row_ids = []
    for row in rows:
        try:
            row_ids.append(int(row.get("i")))
        except (TypeError, ValueError):
            row_ids.append(None)
    issues: list[str] = []
    missing_jpeg = [i for i in row_ids if i is not None and i not in jpeg_ids]
    extra_jpeg = sorted(jpeg_ids - {i for i in row_ids if i is not None})
    if missing_jpeg:
        issues.append(f"missing jpeg for rows {missing_jpeg[:8]}")
    if extra_jpeg:
        issues.append(f"jpeg with no state row {extra_jpeg[:8]}")
    prev_t: float | None = None
    nan_rows = 0
    for row in rows:
        row_issues, prev_t = _issues_for_row(row, prev_t)
        if any("NaN" in item or "not finite" in item for item in row_issues):
            nan_rows += 1
        issues.extend(row_issues)
    blank = {sec.cam_id: 0 for sec in SECTORS}
    checked = 0
    for path in jpegs:
        counts, problem = _blank_counts(path)
        if problem is not None:
            issues.append(problem)
        if counts is None:
            continue
        checked += 1
        for cid, flag in counts.items():
            blank[cid] = blank.get(cid, 0) + int(flag)
    if checked:
        for cid, n_blank in blank.items():
            if int(n_blank) == checked:
                issues.append(f"{cid} is blank on all {checked} frames")
    montage_rows = []
    for row in rows:
        tech = row.get("tech") if isinstance(row.get("tech"), dict) else None
        values, mask = lane_gt_from_record(tech)
        if float(mask.sum()) <= 0.0 or not bool(np.isfinite(values).all()):
            continue
        try:
            idx = int(row.get("i"))
        except (TypeError, ValueError):
            continue
        montage_rows.append((idx, values))
    if len(montage_rows) > montage_n:
        step = max(1, len(montage_rows) // montage_n)
        montage_rows = montage_rows[::step][:montage_n]
    montages: list[str] = []
    for idx, values in montage_rows:
        dest = out / f"montage_{idx:06d}.png"
        src = strips / f"{idx:06d}.jpg"
        if _draw_montage(dest, src, values, f"i={idx} gt lanes"):
            montages.append(str(dest))
    report = {
        "kind": "validate",
        "root": str(root),
        "rows": len(rows),
        "jpegs": len(jpegs),
        "frames_checked": checked,
        "nan_rows": nan_rows,
        "blank_frames": blank,
        "issues": issues,
        "issue_count": len(issues),
        "montages": montages,
    }
    from python.debug.remote_status import write_json

    write_json(out / "report.json", report)
    return report
