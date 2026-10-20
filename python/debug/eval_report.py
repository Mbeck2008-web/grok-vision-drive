"""Eval files a shell can open: CSV, worst-frame PNGs, and a plot.

No torch. The eval script fills the rows after the forward pass.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from python.debug.lane_targets import LOOKAHEADS_M, lane_x_from_vector, score_centers
from python.debug.remote_status import json_num, write_json

HEADING_INDEX = 12


def _center_at(values: Any, index: int) -> float | None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    slot = int(index) * 3
    if slot >= arr.size or not math.isfinite(float(arr[slot])):
        return None
    return float(arr[slot])


def _heading(values: Any) -> float | None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= HEADING_INDEX or not math.isfinite(float(arr[HEADING_INDEX])):
        return None
    return float(arr[HEADING_INDEX])


def lookahead_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """r, rms, and bias of the center at each stored lookahead."""
    table: list[dict[str, Any]] = []
    for index, y in enumerate(LOOKAHEADS_M):
        pred = [_center_at(row.get("pred"), index) for row in rows]
        gt = [_center_at(row.get("gt"), index) for row in rows]
        scored = score_centers(
            [float("nan") if v is None else v for v in pred],
            [float("nan") if v is None else v for v in gt],
        )
        table.append(
            {
                "y_m": float(y),
                "n": int(scored["n"] or 0),
                "r": json_num(scored["r"]),
                "rms": json_num(scored["rms"]),
                "bias": json_num(scored["bias"]),
            }
        )
    pred_h = [_heading(row.get("pred")) for row in rows]
    gt_h = [_heading(row.get("gt")) for row in rows]
    scored_h = score_centers(
        [float("nan") if v is None else v for v in pred_h],
        [float("nan") if v is None else v for v in gt_h],
    )
    table.append(
        {
            "y_m": "heading",
            "n": int(scored_h["n"] or 0),
            "r": json_num(scored_h["r"]),
            "rms": json_num(scored_h["rms"]),
            "bias": json_num(scored_h["bias"]),
        }
    )
    return table


def prediction_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        pred = row.get("pred")
        gt = row.get("gt")
        pred_x = lane_x_from_vector(pred) if pred is not None else None
        gt_x = lane_x_from_vector(gt) if gt is not None else None
        err = None
        if pred_x is not None and gt_x is not None:
            err = float(pred_x) - float(gt_x)
        item: dict[str, Any] = {
            "i": row.get("i"),
            "t": json_num(row.get("t")),
            "pred_center_8m": json_num(pred_x),
            "gt_center_8m": json_num(gt_x),
            "err_8m": json_num(err),
            "abs_err_8m": None if err is None else abs(float(err)),
        }
        for index, y in enumerate(LOOKAHEADS_M):
            tag = f"{int(y)}"
            item[f"pred_c{tag}"] = json_num(_center_at(pred, index))
            item[f"gt_c{tag}"] = json_num(_center_at(gt, index))
        item["pred_heading"] = json_num(_heading(pred))
        item["gt_heading"] = json_num(_heading(gt))
        item["pred_conf"] = json_num(row.get("conf"))
        out.append(item)
    return out


def write_predictions_csv(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = prediction_records(rows)
    fields = [
        "i",
        "t",
        "pred_center_8m",
        "gt_center_8m",
        "err_8m",
        "abs_err_8m",
        "pred_c5",
        "gt_c5",
        "pred_c10",
        "gt_c10",
        "pred_c20",
        "gt_c20",
        "pred_c30",
        "gt_c30",
        "pred_heading",
        "gt_heading",
        "pred_conf",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    return path


def worst_indices(rows: Sequence[Mapping[str, Any]], k: int) -> list[int]:
    scored: list[tuple[float, int]] = []
    for index, row in enumerate(prediction_records(rows)):
        err = row.get("abs_err_8m")
        if err is None:
            continue
        scored.append((float(err), index))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [index for _err, index in scored[: max(0, int(k))]]


def _bev(gt_lanes: list, pred_lanes: list, pred_x: float | None, gt_x: float | None) -> Any:
    import cv2

    size = 220
    img = np.full((size, size, 3), 245, dtype=np.uint8)
    cx, cy = size // 2, size - 28
    scale = 6.0

    def pix(x: float, y: float) -> tuple[int, int]:
        return int(cx + x * scale), int(cy - y * scale)

    cv2.line(img, (cx, 8), (cx, size - 8), (220, 220, 220), 1)

    def draw(polys: list, color: tuple[int, int, int]) -> None:
        for poly in polys or []:
            pts = []
            for point in poly or []:
                try:
                    pts.append(pix(float(point["x"]), float(point["y"])))
                except (KeyError, TypeError, ValueError):
                    continue
            if len(pts) >= 2:
                cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)

    draw(gt_lanes, (40, 140, 60))
    draw(pred_lanes, (40, 90, 220))
    if gt_x is not None:
        cv2.circle(img, pix(float(gt_x), 8.0), 4, (40, 140, 60), -1, cv2.LINE_AA)
    if pred_x is not None:
        cv2.circle(img, pix(float(pred_x), 8.0), 4, (40, 90, 220), -1, cv2.LINE_AA)
    return img


def write_worst_pngs(
    out_dir: Path | str,
    rows: Sequence[Mapping[str, Any]],
    *,
    k: int = 8,
) -> list[Path]:
    """Save the worst ``k`` frames. Green is GT, blue is the prediction."""
    import cv2

    from python.debug.dataset_check import lanes_from_vector

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for rank, index in enumerate(worst_indices(rows, k)):
        row = rows[index]
        pred = row.get("pred")
        gt = row.get("gt")
        rec = prediction_records([row])[0]
        bev = _bev(
            lanes_from_vector(gt) if gt is not None else [],
            lanes_from_vector(pred) if pred is not None else [],
            rec.get("pred_center_8m"),
            rec.get("gt_center_8m"),
        )
        image = row.get("image")
        if image is None:
            canvas = bev
        else:
            frame = np.asarray(image)
            canvas = frame.copy() if frame.ndim == 3 else bev
            if canvas.ndim == 3:
                bh, bw = bev.shape[:2]
                h, w = canvas.shape[:2]
                if h >= bh + 8 and w >= bw + 8:
                    canvas[8 : 8 + bh, w - bw - 8 : w - 8] = bev
                else:
                    canvas = bev
        label = (
            f"worst {rank} i={row.get('i')} err={rec.get('err_8m')}"
        )
        cv2.putText(
            canvas, label[:80], (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 3, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, label[:80], (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (20, 20, 20), 1, cv2.LINE_AA,
        )
        dest = out / f"worst_{rank:02d}_i{int(row.get('i') or 0):06d}.png"
        if cv2.imwrite(str(dest), canvas):
            saved.append(dest)
    return saved


def save_score_plot(
    path: Path | str,
    rows: Sequence[Mapping[str, Any]],
    table: Sequence[Mapping[str, Any]],
) -> Path | None:
    import cv2

    records = prediction_records(rows)
    width, height = 640, 360
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(img, "pred (blue) vs GT (green), 8 m", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    xs_pred = [rec.get("pred_center_8m") for rec in records]
    xs_gt = [rec.get("gt_center_8m") for rec in records]
    finite = [v for v in xs_pred + xs_gt if isinstance(v, (int, float))]
    if finite and len(records) >= 2:
        lo = min(finite)
        hi = max(finite)
        span = max(hi - lo, 1e-3)
        top, bot = 40, 200

        def pt(i: int, value: float) -> tuple[int, int]:
            x = 20 + int(i * (width - 40) / max(1, len(records) - 1))
            y = int(bot - (float(value) - lo) / span * (bot - top))
            return x, y

        for series, color in ((xs_gt, (40, 140, 60)), (xs_pred, (200, 80, 40))):
            pts = []
            for i, value in enumerate(series):
                if isinstance(value, (int, float)):
                    pts.append(pt(i, float(value)))
            if len(pts) >= 2:
                cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
    cv2.putText(img, "rms by lookahead", (16, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    bars = [row for row in table if isinstance(row.get("y_m"), float)]
    max_rms = max([float(row["rms"]) for row in bars if row.get("rms") is not None] or [1.0])
    for i, row in enumerate(bars):
        rms = row.get("rms")
        if rms is None:
            continue
        x0 = 20 + i * 80
        h = int(80 * float(rms) / max(max_rms, 1e-6))
        cv2.rectangle(img, (x0, 340 - h), (x0 + 40, 340), (80, 80, 160), -1)
        cv2.putText(
            img, f"{int(row['y_m'])}m", (x0, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1,
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        return None
    return path


def write_eval_bundle(
    out_dir: Path | str,
    rows: Sequence[Mapping[str, Any]],
    *,
    k: int = 8,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = lookahead_table(rows)
    eight = score_centers(
        [rec["pred_center_8m"] if rec["pred_center_8m"] is not None else float("nan") for rec in prediction_records(rows)],
        [rec["gt_center_8m"] if rec["gt_center_8m"] is not None else float("nan") for rec in prediction_records(rows)],
    )
    summary = {
        "kind": "eval",
        "n": int(eight["n"] or 0),
        "r": json_num(eight["r"]),
        "rms": json_num(eight["rms"]),
        "bias": json_num(eight["bias"]),
        "lookahead": table,
    }
    write_json(out / "summary.json", summary)
    write_predictions_csv(out / "predictions.csv", rows)
    write_worst_pngs(out / "worst", rows, k=k)
    save_score_plot(out / "lookahead.png", rows, table)
    return summary
