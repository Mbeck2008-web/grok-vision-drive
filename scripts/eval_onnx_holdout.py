#!/usr/bin/env python3
"""Score an exported scene ONNX on the trainer's own holdout, offline.

Rebuilds the holdout the way ``train_scene`` does (same label file via
``GVD_GT_FILE``, same ``--cams`` / ``--mask-files`` / ``--mask-events`` /
``--holdout-blocks`` / holdout cells file), runs the ONNX row by row with the
GRU cleared every ``--eval-chunk`` rows (the trainer's primary val stream),
and prints one line in the trainer's val-line format:

    [GVD] val (onnx) step=S epoch=N n=... r=... rms=... rms_mean=... bias=... pred_mean=... pred_std=... eval_chunk=16

Step and epoch come from the ONNX metadata written by
``scripts/export_epoch_onnx.py``. Example (run on the dev PC):

    set GVD_GT_FILE=gt_v103.jsonl
    py -3 scripts/eval_onnx_holdout.py --strips <recording> --onnx <file.onnx> --cams main ^
        --holdout-blocks 1 --mask-files mask_v103_hop_delta.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline holdout score of an exported scene ONNX")
    parser.add_argument("--strips", type=Path, required=True, help="Recording folder (state.jsonl, raw/).")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--cams", default="", help="Comma list. Default: the ONNX gvd_cams metadata.")
    parser.add_argument("--mask-files", default="", help="Same as train_scene --mask-files.")
    parser.add_argument("--mask-events", action="store_true", help="Same as train_scene --mask-events.")
    parser.add_argument("--holdout-blocks", type=int, default=None, help="Same as train_scene (default 1 or GVD_HOLDOUT_BLOCKS).")
    parser.add_argument("--holdout-cells-file", type=Path, default=None, help="Load this cell list (never chooses one).")
    parser.add_argument("--eval-chunk", type=int, default=16, help="GRU reset period, as train_scene --eval-chunk (> 0).")
    parser.add_argument("--cache", default="", help="Frame cache folder, as train_scene --cache.")
    parser.add_argument("--json", type=Path, default=None, help="Also write the result as JSON (create-only).")
    return parser


def _meta(session: Any) -> dict[str, str]:
    try:
        return dict(session.get_modelmeta().custom_metadata_map)
    except Exception:
        return {}


def holdout_rows(
    root: Path,
    *,
    cams: tuple[str, ...] | None,
    mask_files: str = "",
    mask_events: bool = False,
    holdout_blocks: int = 1,
    holdout_cells_file: Path | None = None,
) -> tuple[list[list[dict]], str]:
    """The trainer's holdout slices, in order, and its selection line."""
    from python.data.raw_frames import load_scene_rows, refuse_lossy
    from python.train.continual import CHUNK_MAX, pass_plan
    from python.train.event_mask import drop_masked, mask_file_sources, selection_line
    from python.train.scene_loader import select_training_rows

    refuse_lossy(root)
    state_rows = load_scene_rows(root)
    sources = mask_file_sources(root, state_rows, mask_files)
    rows, _missing, _excluded = select_training_rows(root, state_rows, cams=cams, dropped={})
    if mask_events:
        from python.train.event_mask import mask_event_rows

        rows, _counts = mask_event_rows(root, rows)
    before = rows
    if sources:
        rows = drop_masked(rows, set().union(*(source["ids"] for source in sources)))
    text, _meta_sel = selection_line(state_rows, before, rows, sources)
    cells = None
    if holdout_cells_file is not None:
        from python.train.continual import load_holdout_cells

        cells = load_holdout_cells(Path(holdout_cells_file), rows)
    plan = pass_plan(rows, CHUNK_MAX, CHUNK_MAX // 2, blocks=int(holdout_blocks), cells=cells)
    return [list(item["rows"]) for item in plan if item["kind"] == "holdout"], text


def evaluate(
    root: Path | str,
    onnx_path: Path | str,
    *,
    cams: tuple[str, ...] | None = None,
    mask_files: str = "",
    mask_events: bool = False,
    holdout_blocks: int = 1,
    holdout_cells_file: Path | None = None,
    eval_chunk: int = 16,
    cache: Any = None,
) -> dict[str, Any]:
    """Score ``onnx_path`` on the holdout. Returns the val fields plus step/epoch from the metadata."""
    import onnxruntime as ort  # type: ignore

    from python.debug.lane_targets import lane_center_is_valid, lane_gt_from_record, lane_x_from_vector
    from python.train.cam_strip import cams_from_metadata
    from python.train.scene_net import step_dt
    from python.train.train_scene import _frame_row, val_summary

    if int(eval_chunk) <= 0:
        raise ValueError("--eval-chunk must be > 0 (the single-sequence score is not reproduced here)")
    root = Path(root)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    meta = _meta(session)
    used = tuple(cams) if cams else (cams_from_metadata(meta) or None)
    scale = float(meta.get("gvd_scale") or 1.0)
    from python.train.scene_lane_runtime import _base_name
    from python.train.scene_net import RESIDUAL_S

    # The tracer may rename pass-through state inputs (``h.1``); match on the base name.
    inputs = {_base_name(item.name): item for item in session.get_inputs()}
    outputs = [_base_name(item.name) for item in session.get_outputs()]
    state_keys = [key for key in ("h", "tokens", "mask") if key in inputs]

    def fresh() -> dict[str, np.ndarray]:
        # SceneNet.initial_state: zero h and tokens, every residual slot older than RESIDUAL_S.
        state = {}
        for key in state_keys:
            shape = [int(d) for d in inputs[key].shape]
            value = float(RESIDUAL_S) + 1.0 if key == "mask" else 0.0
            state[key] = np.full(shape, value, dtype=np.float32)
        return state

    slices, selection = holdout_rows(
        root,
        cams=used,
        mask_files=mask_files,
        mask_events=mask_events,
        holdout_blocks=holdout_blocks,
        holdout_cells_file=holdout_cells_file,
    )
    pairs: list[tuple[float, float]] = []
    for rows in slices:
        state = fresh()
        for k, row in enumerate(rows):
            if k % int(eval_chunk) == 0:
                state = fresh()
            loaded = _frame_row(root, row, cams=used, cache=cache, augment=False, scale=scale)
            if loaded is None:
                continue
            frame, row = loaded
            arr = np.ascontiguousarray(frame)
            # Same conversion as SceneNet._strip_nchw: uint8 / 255, else /255 only above 1.5.
            image = arr.astype(np.float32)
            if arr.dtype == np.uint8 or float(image.max()) > 1.5:
                image = image / 255.0
            image = np.ascontiguousarray(np.transpose(image, (2, 0, 1))[None, ...])
            dt = np.asarray([step_dt(row.get("dt_s"))], dtype=np.float32)
            feeds = {"strip": image, "dt": dt, **state}
            out = session.run(None, {inputs[key].name: value for key, value in feeds.items() if key in inputs})
            by_name = dict(zip(outputs, out))
            state = {key: np.asarray(by_name[key], dtype=np.float32) for key in state_keys}
            tech = row.get("tech") if isinstance(row.get("tech"), dict) else row
            gt_vec, gt_mask = lane_gt_from_record(tech if isinstance(tech, dict) else None)
            if not lane_center_is_valid(gt_mask):
                continue
            gt_x = lane_x_from_vector(gt_vec)
            pred_x = lane_x_from_vector(np.asarray(by_name["lane_gt"], dtype=np.float32).reshape(-1))
            pairs.append((float("nan") if pred_x is None else float(pred_x), float("nan") if gt_x is None else float(gt_x)))
    scored = val_summary(pairs, None, {"n": 0})
    return {
        "step": int(meta["gvd_ckpt_step"]) if "gvd_ckpt_step" in meta else None,
        "epoch": int(meta["gvd_ckpt_epoch"]) if "gvd_ckpt_epoch" in meta else None,
        "ckpt_sha256": meta.get("gvd_ckpt_sha256"),
        "n": scored.get("n"),
        "r": scored.get("r"),
        "rms": scored.get("rms"),
        "rms_mean": scored.get("rms_mean"),
        "bias": scored.get("bias"),
        "pred_mean": scored.get("pred_mean"),
        "pred_std": scored.get("pred_std"),
        "eval_chunk": int(eval_chunk),
        "slices": len(slices),
        "holdout_rows": sum(len(rows) for rows in slices),
        "selection": selection,
    }


def val_line(result: dict[str, Any]) -> str:
    return (
        f"[GVD] val (onnx) step={result['step']} epoch={result['epoch']} n={result['n']} r={result['r']} "
        f"rms={result['rms']} rms_mean={result['rms_mean']} bias={result['bias']} "
        f"pred_mean={result['pred_mean']} pred_std={result['pred_std']} eval_chunk={result['eval_chunk']}"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from python.train.cam_strip import parse_cams
    from python.train.train_scene import resolve_holdout_blocks

    cache = None
    if args.cache:
        from python.train.frame_cache import FrameCache

        cache = FrameCache(args.cache)
    if args.json is not None and args.json.exists():
        print(f"eval: {args.json} already exists (create-only)", file=sys.stderr)
        return 2
    result = evaluate(
        args.strips,
        args.onnx,
        cams=parse_cams(args.cams) if args.cams else None,
        mask_files=args.mask_files,
        mask_events=bool(args.mask_events),
        holdout_blocks=resolve_holdout_blocks(args.holdout_blocks),
        holdout_cells_file=args.holdout_cells_file,
        eval_chunk=int(args.eval_chunk),
        cache=cache,
    )
    print(result["selection"], flush=True)
    print(f"[GVD] holdout: rows={result['holdout_rows']} slices={result['slices']}", flush=True)
    print(val_line(result), flush=True)
    if args.json is not None:
        with open(args.json, "x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
