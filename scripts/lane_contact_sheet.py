#!/usr/bin/env python3
"""Lane overlay contact sheet: K main-camera frames with GT (and predicted) lanes.

Offline, CPU only. Each tile is the recording's main camera image with the
lane_gt center and boundaries at the 4 lookaheads projected onto it through the
rig's ``config/cameras.yaml`` pose (``lanes.pose_pixel_of_ground``). GT is
green and prediction red, the same colours as the ``--val-worst-png``
thumbnails. Each tile is labelled with recording, frame index, loss (from a
val_worst JSON), the 8 m center error when a prediction is shown, and gt_origin.

Frames: ``--pick even`` (default), ``--pick random --seed S``, or
``--val-worst <val_worst...json>`` (its frames, its predictions and losses).
``--onnx`` predicts each frame after a fresh-state warm-up of up to
``--eval-chunk`` rows ending at it (the trainer's 16-frame val windows).

The lane vector is in the label file's frame (tech.gt_origin); the projection
treats it as the camera pose's ego frame, so a ref-node origin can sit a
little off along y.

Output: ``--out sheet.png``; more than ``--per-page`` tiles writes
``sheet_p1.png``, ``sheet_p2.png``, ... K <= 64, tile width <= 480, so a page
is at most 4x4 tiles (about 1.9 x 1.6 kpx).

    py -3 scripts/lane_contact_sheet.py --strips <recording> --k 16 --out sheet.png
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

MAX_K = 64
MAX_PER_PAGE = 16
MAX_TILE_W = 480
LABEL_H = 30


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--strips", required=True, help="Recording directory (state.jsonl, gt.jsonl, raw/main/).")
    p.add_argument("--onnx", default="", help="Scene ONNX: draw its lane_gt prediction in red.")
    p.add_argument("--val-worst", default="", help="A val_worst JSON from train_scene: its frames, losses and predictions.")
    p.add_argument("--pick", choices=("even", "random"), default="even")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=16, help=f"Frames (max {MAX_K}).")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--per-page", type=int, default=MAX_PER_PAGE)
    p.add_argument("--tile-w", type=int, default=320)
    p.add_argument("--cam", default="main")
    p.add_argument("--eval-chunk", type=int, default=16)
    p.add_argument("--out", default="contact_sheet.png")
    return p


def pick_rows(rows: list[dict], k: int, mode: str, seed: int) -> list[int]:
    """Positions in ``rows``: ``k`` evenly spaced (first and last included) or a seeded sample, in order."""
    n = len(rows)
    k = max(0, min(int(k), MAX_K, n))
    if k == 0:
        return []
    if mode == "random":
        return sorted(random.Random(int(seed)).sample(range(n), k))
    if k == 1:
        return [0]
    return sorted({round(j * (n - 1) / (k - 1)) for j in range(k)})


def lane_points(vec: Any, valid: Any = None) -> list[list[tuple[float, float]]]:
    """Ground polylines (x, y) for left, center, right from a lane_gt vector."""
    from python.debug.lane_targets import LOOKAHEADS_M, SLOT_CENTER, SLOT_LEFT, SLOT_RIGHT, lookahead_slot

    lines = []
    for part in (SLOT_LEFT, SLOT_CENTER, SLOT_RIGHT):
        pts = []
        for k, y in enumerate(LOOKAHEADS_M):
            j = lookahead_slot(k, part)
            x = vec[j] if vec is not None and j < len(vec) else None
            ok = valid is None or (j < len(valid) and (valid[j] or 0) > 0.5)
            if x is not None and ok and math.isfinite(float(x)):
                pts.append((float(x), float(y)))
        lines.append(pts)
    return lines


def draw_lanes(img: np.ndarray, pose: Any, vec: Any, color: tuple[int, int, int], valid: Any, src_wh: tuple[int, int]) -> None:
    import cv2

    from python.perception.lanes import pose_pixel_of_ground

    sw, sh = src_wh
    fx = img.shape[1] / float(sw)
    fy = img.shape[0] / float(sh)
    for line in lane_points(vec, valid):
        px = []
        for x, y in line:
            hit = pose_pixel_of_ground(pose, x, y, width=sw, height=sh)
            if hit is not None:
                px.append((int(round(hit[0] * fx)), int(round(hit[1] * fy))))
        if len(px) >= 2:
            cv2.polylines(img, [np.array(px, dtype=np.int32)], False, color, 2, cv2.LINE_AA)


def tile(
    image: np.ndarray | None, pose: Any, tile_w: int, label: str, gt: Any, gt_valid: Any, pred: Any
) -> np.ndarray:
    import cv2

    from python.train.val_worst import GT_BGR, PRED_BGR

    tile_w = max(64, min(int(tile_w), MAX_TILE_W))
    if image is None:
        body = np.full((tile_w * 3 // 4, tile_w, 3), 40, np.uint8)
        cv2.putText(body, "no frame", (8, body.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        h, w = image.shape[:2]
        body = cv2.resize(image, (tile_w, max(1, round(h * tile_w / w))), interpolation=cv2.INTER_AREA)
        draw_lanes(body, pose, gt, GT_BGR, gt_valid, (w, h))
        if pred is not None:
            draw_lanes(body, pose, pred, PRED_BGR, None, (w, h))
    band = np.full((LABEL_H, tile_w, 3), 16, np.uint8)
    for n, text in enumerate(label.split("\n")[:2]):
        cv2.putText(band, text, (4, 12 + 13 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([band, body])


def label_for(recording: str, item: dict) -> str:
    from python.debug.lane_targets import lane_x_from_vector

    parts = [f"{recording} i={item.get('i')}"]
    if item.get("loss") is not None:
        parts.append(f"loss={float(item['loss']):.4f}")
    if item.get("pred") is not None and item.get("gt") is not None:
        a, b = lane_x_from_vector(item["pred"]), lane_x_from_vector(item["gt"])
        if a is not None and b is not None:
            parts.append(f"err8={abs(float(a) - float(b)):.2f}m")
    return " ".join(parts) + f"\ngt_origin={item.get('gt_origin') or 'unknown'}"


def pages(tiles: list[np.ndarray], cols: int, per_page: int) -> list[np.ndarray]:
    per_page = max(1, min(int(per_page), MAX_PER_PAGE))
    cols = max(1, min(int(cols), per_page))
    out = []
    for start in range(0, len(tiles), per_page):
        chunk = tiles[start: start + per_page]
        th = max(t.shape[0] for t in chunk)
        tw = max(t.shape[1] for t in chunk)
        rows_n = math.ceil(len(chunk) / cols)
        sheet = np.zeros((rows_n * th, min(cols, len(chunk)) * tw, 3), np.uint8)
        for j, t in enumerate(chunk):
            r, c = divmod(j, cols)
            sheet[r * th: r * th + t.shape[0], c * tw: c * tw + t.shape[1]] = t
        out.append(sheet)
    return out


def _image(root: Path, row: dict, cam: str) -> np.ndarray | None:
    import cv2

    info = (row.get("cams") or {}).get(cam) or {}
    if not info.get("file"):
        return None
    return cv2.imread(str(root / info["file"]), cv2.IMREAD_COLOR)


def _onnx_preds(root: Path, rows: list[dict], picks: list[int], onnx: str, eval_chunk: int) -> dict[int, Any]:
    from eval_onnx_holdout import OnnxRows

    from python.train.cam_strip import cams_from_metadata
    from python.train.train_scene import _frame_row

    runner = OnnxRows(onnx)
    used = cams_from_metadata(runner.meta) or None
    out: dict[int, Any] = {}
    for pos in picks:
        state = runner.fresh()
        by_name = None
        for row in rows[max(0, pos - max(1, int(eval_chunk)) + 1): pos + 1]:
            loaded = _frame_row(root, row, cams=used, augment=False, scale=runner.scale)
            if loaded is None:
                by_name = None
                continue
            by_name, state = runner.step(loaded[0], loaded[1], state)
        if by_name is not None and "lane_gt" in by_name:
            out[pos] = np.asarray(by_name["lane_gt"], dtype=np.float32).reshape(-1).tolist()
    return out


def render(args: argparse.Namespace) -> list[Path]:
    import cv2

    from python.data.raw_frames import load_scene_rows
    from python.debug.lane_targets import lane_gt_from_record
    from python.perception.lanes import camera_pose

    root = Path(args.strips)
    pose = camera_pose(args.cam)
    if pose is None:
        raise SystemExit(f"[GVD] contact sheet: no pose for camera {args.cam} in config/cameras.yaml")
    rows = [r for r in load_scene_rows(root) if ((r.get("cams") or {}).get(args.cam) or {}).get("file")]
    by_i = {int(r["i"]): n for n, r in enumerate(rows)}
    items: list[dict] = []
    if args.val_worst:
        dump = json.loads(Path(args.val_worst).read_text(encoding="utf-8"))
        for frame in dump.get("frames", [])[:MAX_K]:
            if int(frame.get("i", -1)) in by_i:
                items.append(dict(frame, pos=by_i[int(frame["i"])]))
    else:
        for pos in pick_rows(rows, args.k, args.pick, args.seed):
            tech = rows[pos].get("tech") if isinstance(rows[pos].get("tech"), dict) else None
            gt, valid = lane_gt_from_record(tech)
            items.append({
                "pos": pos, "i": rows[pos].get("i"), "gt": gt.tolist(), "gt_valid": valid.tolist(),
                "gt_origin": (tech or {}).get("gt_origin"),
            })
    if args.onnx:
        preds = _onnx_preds(root, rows, [it["pos"] for it in items], args.onnx, args.eval_chunk)
        for it in items:
            it["pred"] = preds.get(it["pos"])
    recording = root.name
    tiles = [
        tile(_image(root, rows[it["pos"]], args.cam), pose, args.tile_w, label_for(recording, it),
             it.get("gt"), it.get("gt_valid"), it.get("pred"))
        for it in items
    ]
    if not tiles:
        raise SystemExit("[GVD] contact sheet: no frames to draw")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheets = pages(tiles, args.cols, args.per_page)
    paths = [out] if len(sheets) == 1 else [out.with_name(f"{out.stem}_p{n + 1}{out.suffix}") for n in range(len(sheets))]
    for path, sheet in zip(paths, sheets):
        if not cv2.imwrite(str(path), sheet):
            raise SystemExit(f"[GVD] contact sheet: could not write {path}")
    print(f"[GVD] contact sheet: {len(tiles)} frames, {len(paths)} page(s): " + ", ".join(p.name for p in paths), flush=True)
    return paths


def main(argv: list[str] | None = None) -> int:
    render(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
