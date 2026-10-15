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

from python.collect.liveness import motion_issue_lines
from python.data.raw_frames import (
    list_raw_images,
    load_scene_rows,
    load_training_bgr,
    raw_count_issues,
    raw_image_counts,
    read_image_with_repair,
)
from python.data.strip_writer import CANVAS_H, CANVAS_W, SECTORS
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


def _xy_of(row: dict[str, Any]) -> tuple[float, float] | None:
    ego = row.get("ego") if isinstance(row.get("ego"), dict) else {}
    pos = ego.get("pos") if isinstance(ego, dict) else None
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        try:
            return float(pos[0]), float(pos[1])
        except (TypeError, ValueError):
            return None
    if isinstance(ego, dict) and ego.get("x") is not None and ego.get("y") is not None:
        try:
            return float(ego["x"]), float(ego["y"])
        except (TypeError, ValueError):
            return None
    return None


def _speed_of(row: dict[str, Any]) -> float | None:
    ego = row.get("ego") if isinstance(row.get("ego"), dict) else {}
    if not isinstance(ego, dict):
        return None
    try:
        speed = float(ego.get("speed_mps"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(speed):
        return None
    return speed


def _recover_break(
    row: dict[str, Any],
    prev_speed: float | None,
    prev_xy: tuple[float, float] | None,
) -> bool:
    """A respawn settle is a segment break, same as a map change.

    The row is marked ``recover`` after a logged respawn. Older logs have
    the settle only as a multi-second gap while the car was already stopped
    and did not drive through the hole.
    """
    if row.get("recover") is True:
        return True
    if str(row.get("recover")).lower() in ("1", "true"):
        return True
    if prev_speed is None or prev_speed >= 1.0 or prev_xy is None:
        return False
    here = _xy_of(row)
    if here is None:
        return False
    moved = math.hypot(here[0] - prev_xy[0], here[1] - prev_xy[1])
    return moved < 25.0


def _issues_for_row(
    row: dict[str, Any],
    prev_t: float | None,
    prev_i: int | None = None,
    prev_segment: str = "",
    prev_speed: float | None = None,
    prev_xy: tuple[float, float] | None = None,
) -> tuple[list[str], float | None, int | None, str, bool]:
    issues: list[str] = []
    idx = row.get("i")
    try:
        t = float(row.get("t"))
    except (TypeError, ValueError):
        issues.append(f"row {idx}: t is not a number")
        t = None
    this_i: int | None
    try:
        this_i = int(idx) if idx is not None else None
    except (TypeError, ValueError):
        this_i = None
    if t is not None and not math.isfinite(t):
        issues.append(f"row {idx}: t is not finite")
        t = None
    segment = str(row.get("segment") or "")
    switched = bool(segment and prev_segment and segment != prev_segment)
    if t is not None and prev_t is not None:
        gap = t - prev_t
        if gap < 0.0:
            issues.append(f"row {idx}: time went backwards by {gap:.3f}s")
        elif gap > GAP_S and not switched and not _recover_break(row, prev_speed, prev_xy):
            step = None if prev_i is None or this_i is None else this_i - prev_i
            if step == 1:
                issues.append(
                    f"row {idx}: gap {gap:.3f}s between recorded samples "
                    f"(index {prev_i}->{this_i}, a time hole, not a missing file number)"
                )
            else:
                issues.append(
                    f"row {idx}: gap {gap:.3f}s (index {prev_i}->{this_i}, step {step})"
                )
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
    masked_row = False
    for slot, value in enumerate(values.tolist()):
        if math.isfinite(float(value)):
            continue
        covered = slot < int(mask.size) and float(mask[slot]) <= 0.5
        if covered:
            masked_row = True
            continue
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
    return issues, t, this_i if t is not None else prev_i, segment or prev_segment, masked_row


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


def _validate_raw(
    root: Path,
    out: Path,
    rows: list[dict[str, Any]],
    raw_images: list[Path],
    *,
    montage_n: int,
) -> dict[str, Any]:
    """Check per-camera files. Montages are written only under ``out``."""
    issues: list[str] = []
    by_cam: dict[str, set[int]] = {}
    for path in raw_images:
        cam_id = path.parent.name
        by_cam.setdefault(cam_id, set()).add(int(path.stem))
    row_ids = []
    for row in rows:
        try:
            row_ids.append(int(row.get("i")))
        except (TypeError, ValueError):
            row_ids.append(None)
    for row, idx in zip(rows, row_ids):
        cams = row.get("cams") if isinstance(row.get("cams"), dict) else {}
        for cam_id, meta in cams.items():
            rel = str((meta or {}).get("file") or "")
            if rel and not (root / rel).is_file():
                issues.append(f"row {idx}: missing {rel}")
            elif cam_id not in by_cam or idx not in by_cam.get(cam_id, set()):
                issues.append(f"row {idx}: missing raw frame for {cam_id}")
    prev_t: float | None = None
    prev_i: int | None = None
    prev_segment = ""
    prev_speed: float | None = None
    prev_xy: tuple[float, float] | None = None
    nan_rows = 0
    masked_rows = 0
    for row in rows:
        row_issues, prev_t, prev_i, prev_segment, masked = _issues_for_row(
            row, prev_t, prev_i, prev_segment, prev_speed, prev_xy
        )
        prev_speed = _speed_of(row)
        prev_xy = _xy_of(row)
        if any("NaN" in item or "not finite" in item for item in row_issues):
            nan_rows += 1
        if masked:
            masked_rows += 1
        issues.extend(row_issues)
    issues.extend(motion_issue_lines(rows))
    counts = raw_image_counts(raw_images)
    issues.extend(raw_count_issues(counts["per_cam"]))
    blank = {cam_id: 0 for cam_id in by_cam}
    checked = 0
    seen_index: set[int] = set()
    for path in raw_images:
        image = _read_raw_image(path)
        if image is None:
            issues.append(f"unreadable {path.name}")
            continue
        seen_index.add(int(path.stem))
        if sector_is_blank(image):
            blank[path.parent.name] = blank.get(path.parent.name, 0) + 1
    checked = len(seen_index)
    if checked:
        for cam_id, n_blank in blank.items():
            n_files = len(by_cam.get(cam_id, ()))
            if n_files and int(n_blank) == n_files:
                issues.append(f"{cam_id} is blank on all {n_files} frames")
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
        try:
            canvas = load_training_bgr(root, idx)
        except (FileNotFoundError, ValueError):
            canvas = None
        if _draw_canvas(dest, canvas, values, f"i={idx} gt lanes"):
            montages.append(str(dest))
    report = {
        "kind": "validate",
        "format": "raw",
        "root": str(root),
        "rows": len(rows),
        "jpegs": 0,
        "raw_frames": counts["raw_frames"],
        "raw_files": counts["raw_files"],
        "per_cam": counts["per_cam"],
        "frames_checked": checked,
        "nan_rows": nan_rows,
        "masked_rows": masked_rows,
        "masked_fraction": 0.0 if not rows else float(masked_rows) / float(len(rows)),
        "blank_frames": blank,
        "issues": issues,
        "issue_count": len(issues),
        "montages": montages,
    }
    from python.debug.remote_status import write_json

    write_json(out / "report.json", report)
    return report


def _read_raw_image(path: Path) -> np.ndarray | None:
    if path.suffix == ".npy":
        try:
            return np.load(path)
        except (OSError, ValueError):
            return None
    return read_image_with_repair(path)


def _draw_canvas(path: Path, canvas: np.ndarray | None, values: np.ndarray, title: str) -> bool:
    from python.debug.gt_logger import save_debug_frame

    lanes = lanes_from_vector(values)
    center = None
    if values.size >= 6 and bool(np.isfinite(values[:6]).all()):
        center = float(values[0]) + 0.6 * (float(values[3]) - float(values[0]))
    saved = save_debug_frame(path, stitch=canvas, lanes_bev=lanes, hud=[title], lane_x=center)
    return saved is not None


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


def _report_without_frames(
    root: Path,
    out: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """State-only folder. Do not invent a ``strips/*.jpg`` path."""
    issues = ["no raw frames and no strips"]
    prev_t: float | None = None
    prev_i: int | None = None
    prev_segment = ""
    prev_speed: float | None = None
    prev_xy: tuple[float, float] | None = None
    nan_rows = 0
    masked_rows = 0
    for row in rows:
        row_issues, prev_t, prev_i, prev_segment, masked = _issues_for_row(
            row, prev_t, prev_i, prev_segment, prev_speed, prev_xy
        )
        prev_speed = _speed_of(row)
        prev_xy = _xy_of(row)
        if any("NaN" in item or "not finite" in item for item in row_issues):
            nan_rows += 1
        if masked:
            masked_rows += 1
        issues.extend(row_issues)
    issues.extend(motion_issue_lines(rows))
    report = {
        "kind": "validate",
        "root": str(root),
        "rows": len(rows),
        "jpegs": 0,
        "raw_frames": 0,
        "raw_files": 0,
        "per_cam": {},
        "frames_checked": 0,
        "nan_rows": nan_rows,
        "masked_rows": masked_rows,
        "masked_fraction": 0.0 if not rows else float(masked_rows) / float(len(rows)),
        "blank_frames": {},
        "issues": issues,
        "issue_count": len(issues),
        "montages": [],
    }
    from python.debug.remote_status import write_json

    write_json(out / "report.json", report)
    return report


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
    rows = load_scene_rows(root)
    raw_images = list_raw_images(root)
    raw_dir = root / "raw"
    if raw_dir.is_dir() or raw_images:
        return _validate_raw(root, out, rows, raw_images, montage_n=montage_n)
    strips = root / "strips"
    jpegs = sorted(p for p in strips.glob("*.jpg") if p.is_file()) if strips.is_dir() else []
    if not jpegs:
        return _report_without_frames(root, out, rows)
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
    prev_i: int | None = None
    prev_segment = ""
    prev_speed: float | None = None
    prev_xy: tuple[float, float] | None = None
    nan_rows = 0
    masked_rows = 0
    for row in rows:
        row_issues, prev_t, prev_i, prev_segment, masked = _issues_for_row(
            row, prev_t, prev_i, prev_segment, prev_speed, prev_xy
        )
        prev_speed = _speed_of(row)
        prev_xy = _xy_of(row)
        if any("NaN" in item or "not finite" in item for item in row_issues):
            nan_rows += 1
        if masked:
            masked_rows += 1
        issues.extend(row_issues)
    issues.extend(motion_issue_lines(rows))
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
        "masked_rows": masked_rows,
        "masked_fraction": 0.0 if not rows else float(masked_rows) / float(len(rows)),
        "blank_frames": blank,
        "issues": issues,
        "issue_count": len(issues),
        "montages": montages,
    }
    from python.debug.remote_status import write_json

    write_json(out / "report.json", report)
    return report
