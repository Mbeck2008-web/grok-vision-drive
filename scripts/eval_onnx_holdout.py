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
    parser.add_argument(
        "--lane-ref",
        type=Path,
        default=None,
        help="ctl runs: run A's --json from this script (paired lane_gt-only run, same holdout and epoch). "
        "Lane bar: this rms <= 1.05 x A's. Missing or unreadable: lane UNPROVEN, run at best UNPROVEN.",
    )
    parser.add_argument(
        "--expect-epoch",
        type=int,
        default=None,
        help="The ONNX gvd_ckpt_epoch must equal this. Another value or none: ctl run and lane bar UNPROVEN.",
    )
    return parser


def _meta(session: Any) -> dict[str, str]:
    try:
        return dict(session.get_modelmeta().custom_metadata_map)
    except Exception:
        return {}


class OnnxRows:
    """One exported scene ONNX, fed row by row the way the trainer's val stream is."""

    def __init__(self, onnx_path: Path | str) -> None:
        import onnxruntime as ort  # type: ignore

        from python.train.cam_strip import cam_age_flag
        from python.train.scene_lane_runtime import _base_name

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.meta = _meta(self.session)
        self.scale = float(self.meta.get("gvd_scale") or 1.0)
        # The tracer may rename pass-through state inputs (``h.1``); match on the base name.
        self.inputs = {_base_name(item.name): item for item in self.session.get_inputs()}
        self.outputs = [_base_name(item.name) for item in self.session.get_outputs()]
        self.state_keys = [key for key in ("h", "tokens", "mask") if key in self.inputs]
        # gvd_cam_age=1: the graph takes ``ages``; an older export has neither.
        self.cam_age = cam_age_flag(self.meta)
        if self.cam_age != ("ages" in self.inputs):
            raise ValueError(f"{onnx_path}: gvd_cam_age={int(self.cam_age)} disagrees with the graph inputs {sorted(self.inputs)}")
        # gvd_ctl_cond=N (B'1, --ctl-cond): the graph takes ``cond`` (1 x N), the t-dt row's controls.
        self.cond_dim = int(self.meta.get("gvd_ctl_cond") or 0)
        has_cond = "cond" in self.inputs
        if has_cond != bool(self.cond_dim) or (has_cond and [int(d) for d in self.inputs["cond"].shape] != [1, self.cond_dim]):
            raise ValueError(f"{onnx_path}: gvd_ctl_cond={self.cond_dim} disagrees with the graph inputs {sorted(self.inputs)}")

    def fresh(self) -> dict[str, np.ndarray]:
        from python.train.scene_net import RESIDUAL_S

        # SceneNet.initial_state: zero h and tokens, every residual slot older than RESIDUAL_S.
        state = {}
        for key in self.state_keys:
            shape = [int(d) for d in self.inputs[key].shape]
            value = float(RESIDUAL_S) + 1.0 if key == "mask" else 0.0
            state[key] = np.full(shape, value, dtype=np.float32)
        return state

    def step(self, frame: Any, row: dict, state: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """One row through the graph: the outputs by base name and the next state."""
        from python.train.scene_net import step_dt
        from python.train.train_scene import row_ages

        arr = np.ascontiguousarray(frame)
        # Same conversion as SceneNet._strip_nchw: uint8 / 255, else /255 only above 1.5.
        image = arr.astype(np.float32)
        if arr.dtype == np.uint8 or float(image.max()) > 1.5:
            image = image / 255.0
        image = np.ascontiguousarray(np.transpose(image, (2, 0, 1))[None, ...])
        dt = np.asarray([step_dt(row.get("dt_s"))], dtype=np.float32)
        feeds = {"strip": image, "dt": dt, **state}
        if self.cam_age:
            feeds["ages"] = row_ages(row).reshape(1, -1)
        if self.cond_dim:
            # Zeros (and no ctl score: ctl_valid is all zero) when the row has no t-dt row.
            feeds["cond"] = np.asarray(row.get("ctl_cond", np.zeros(self.cond_dim)), dtype=np.float32).reshape(1, self.cond_dim)
        out = self.session.run(None, {self.inputs[key].name: value for key, value in feeds.items() if key in self.inputs})
        by_name = dict(zip(self.outputs, out))
        return by_name, {key: np.asarray(by_name[key], dtype=np.float32) for key in self.state_keys}


def holdout_rows(
    root: Path,
    *,
    cams: tuple[str, ...] | None,
    mask_files: str = "",
    mask_events: bool = False,
    holdout_blocks: int = 1,
    holdout_cells_file: Path | None = None,
    cam_age: bool = False,
    ctl_info: dict | None = None,
) -> tuple[list[list[dict]], str]:
    """The trainer's holdout slices, in order, and its selection line.

    ``cam_age``: each row also gets the trainer's hold-last plan
    (``cam_hold.annotate_hold`` on the same plan), so the ONNX sees the same
    sectors and ages the val row did.
    ``ctl_info`` (a dict): the rows also get the trainer's ctl targets
    (``ctl_targets.attach_ctl`` on the same plan) and the dict gets the
    train-split B1 constants (``b1``) and the steer lock (``lock_deg``).
    """
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
    if cam_age:
        from python.data.cam_hold import annotate_hold

        annotate_hold(state_rows, rows, plan)
    if ctl_info is not None:
        from python.data.cam_hold import plan_labels
        from python.train.ctl_targets import attach_ctl, b1_constants
        from python.train.event_mask import read_impacts

        counts = attach_ctl(state_rows, rows, plan=plan, impacts=read_impacts(root), cond=bool(ctl_info.get("cond")))
        train_ids = {i for i, label in plan_labels(plan).items() if label == "train"}
        ctl_info["b1"] = b1_constants([row for row in rows if row.get("i") in train_ids])
        ctl_info["lock_deg"] = counts.get("lock_deg")
        ctl_info["counts"] = counts
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
    lane_ref: Path | str | None = None,
    expect_epoch: int | None = None,
) -> dict[str, Any]:
    """Score ``onnx_path`` on the holdout. Returns the val fields plus step/epoch from the metadata.

    A ctl graph also gets ``ctl`` and ``lane_bar``: the lane bar compares this
    rms with run A's eval JSON at ``lane_ref`` (:func:`ctl_targets.lane_bar`).
    ``rows_sha`` fingerprints the ordered holdout rows. With ``expect_epoch``,
    ``epoch_ok`` is whether the ONNX ``gvd_ckpt_epoch`` equals it (a missing
    tag is not ok); not ok makes the ctl run and the lane bar UNPROVEN.
    """
    from python.debug.lane_targets import lane_center_is_valid, lane_gt_from_record, lane_x_from_vector
    from python.train.cam_strip import cams_from_metadata
    from python.train.train_scene import _frame_row, val_summary

    if int(eval_chunk) <= 0:
        raise ValueError("--eval-chunk must be > 0 (the single-sequence score is not reproduced here)")
    root = Path(root)
    runner = OnnxRows(onnx_path)
    meta, inputs, outputs = runner.meta, runner.inputs, runner.outputs
    used = tuple(cams) if cams else (cams_from_metadata(meta) or None)
    scale = runner.scale
    cam_age = runner.cam_age
    # gvd_ctl=1: the graph has the three (1, K) control outputs.
    from python.train.ctl_targets import CTL_OUTPUT_FIELDS

    has_ctl = str(meta.get("gvd_ctl", "0")) == "1"
    if has_ctl != all(name in outputs for name in CTL_OUTPUT_FIELDS):
        raise ValueError(f"{onnx_path}: gvd_ctl={meta.get('gvd_ctl', '0')} disagrees with the graph outputs {sorted(outputs)}")
    from python.train.ctl_targets import ctl_scored

    has_cond = runner.cond_dim > 0
    if has_cond and not has_ctl:
        raise ValueError(f"{onnx_path}: a cond input needs the ctl outputs (gvd_ctl=1)")
    ctl_info: dict | None = {"cond": has_cond} if has_ctl else None
    ctl_records: list = []

    fresh = runner.fresh
    slices, selection = holdout_rows(
        root,
        cams=used,
        mask_files=mask_files,
        mask_events=mask_events,
        holdout_blocks=holdout_blocks,
        holdout_cells_file=holdout_cells_file,
        cam_age=cam_age,
        ctl_info=ctl_info,
    )
    if cache is not None and getattr(cache, "stale", False):
        cache.require_fresh()
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
            by_name, state = runner.step(frame, row, state)
            if has_ctl and ctl_scored(row):
                ctl_records.append(
                    (row, np.stack([np.asarray(by_name[name], dtype=np.float32).reshape(-1) for name in CTL_OUTPUT_FIELDS]))
                )
            tech = row.get("tech") if isinstance(row.get("tech"), dict) else row
            gt_vec, gt_mask = lane_gt_from_record(tech if isinstance(tech, dict) else None)
            if not lane_center_is_valid(gt_mask):
                continue
            gt_x = lane_x_from_vector(gt_vec)
            pred_x = lane_x_from_vector(np.asarray(by_name["lane_gt"], dtype=np.float32).reshape(-1))
            pairs.append((float("nan") if pred_x is None else float(pred_x), float("nan") if gt_x is None else float(gt_x)))
    scored = val_summary(pairs, None, {"n": 0})
    from python.train.ctl_targets import holdout_rows_sha

    rows_sha = holdout_rows_sha(row for rows in slices for row in rows)
    epoch = int(meta["gvd_ckpt_epoch"]) if "gvd_ckpt_epoch" in meta else None
    epoch_ok = None if expect_epoch is None else (epoch is not None and epoch == int(expect_epoch))
    ctl = None
    bar = None
    if ctl_info is not None:
        from python.train.ctl_targets import ctl_record, lane_bar, score_ctl

        bar = lane_bar({"rms": scored.get("rms"), "rows_sha": rows_sha, "epoch": epoch}, lane_ref)
        if epoch_ok is False:
            bar = dict(bar, verdict="UNPROVEN", ratio=None,
                       reason=f"--expect-epoch {int(expect_epoch)}: ONNX gvd_ckpt_epoch={epoch}")
        from python.train.ctl_targets import REV4, REV5

        # The B' series (cond on) is scored on the Rev 5 bar; older graphs keep Rev 4.
        ctl = ctl_record(score_ctl(ctl_records, ctl_info["b1"], lock_deg=ctl_info.get("lock_deg"), lane=bar["verdict"],
                                   bar=REV5 if has_cond else REV4))
        if epoch_ok is False:
            ctl["run"] = "UNPROVEN"
    return {
        "step": int(meta["gvd_ckpt_step"]) if "gvd_ckpt_step" in meta else None,
        "epoch": epoch,
        "expect_epoch": None if expect_epoch is None else int(expect_epoch),
        "epoch_ok": epoch_ok,
        "rows_sha": rows_sha,
        "ckpt_sha256": meta.get("gvd_ckpt_sha256"),
        "n": scored.get("n"),
        "r": scored.get("r"),
        "rms": scored.get("rms"),
        "rms_mean": scored.get("rms_mean"),
        "bias": scored.get("bias"),
        "pred_mean": scored.get("pred_mean"),
        "pred_std": scored.get("pred_std"),
        "eval_chunk": int(eval_chunk),
        "cam_age": bool(cam_age),
        "slices": len(slices),
        "holdout_rows": sum(len(rows) for rows in slices),
        "selection": selection,
        **({} if ctl is None else {"ctl": ctl, "lane_bar": bar}),
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
    try:
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
            lane_ref=args.lane_ref,
            expect_epoch=args.expect_epoch,
        )
        print(result["selection"], flush=True)
        print(f"[GVD] holdout: rows={result['holdout_rows']} slices={result['slices']} rows_sha={result['rows_sha']}",
              flush=True)
        if result.get("expect_epoch") is not None:
            print(f"[GVD] val (onnx) epoch check: expect={result['expect_epoch']} onnx={result['epoch']} "
                  f"ok={result['epoch_ok']}", flush=True)
        print(val_line(result), flush=True)
        if "ctl" in result:
            from python.train.ctl_targets import val_lines

            bar = result["lane_bar"]
            print(
                f"[GVD] val (onnx) lane bar verdict={bar['verdict']} rms={bar['rms']} ref_rms={bar['ref_rms']} "
                f"ratio={bar['ratio']} bar={bar['bar']} ({bar['reason']})",
                flush=True,
            )
            for line in val_lines(result["ctl"]):
                print(line.replace("[GVD] val ctl", "[GVD] val (onnx) ctl", 1), flush=True)
        if args.json is not None:
            with open(args.json, "x", encoding="utf-8") as handle:
                json.dump(result, handle, indent=1)
        return 0
    finally:
        if cache is not None:
            cache.close()


if __name__ == "__main__":
    sys.exit(main())
