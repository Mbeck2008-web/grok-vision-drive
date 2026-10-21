#!/usr/bin/env python3
"""Read-only report of the --cam-age hold-last plan on one recording.

For the trainer's own row selection, masks and holdout split (the same
arguments as ``train_scene`` / ``eval_onnx_holdout``), this prints, per side
camera, the share of rows whose sector is still black under the 1.5 s cap
(no earlier frame of that camera in the same run, or older than the cap), for
the train rows and the holdout rows (all, and the lane-center-scored ones),
plus the median and p95 age of the held (non-black) training sectors
(``train_age_s``) and of every training sector with black counted at the cap,
1.533 s (``train_age_capped_s``, what ``live_age_capture.py`` compares).
``black_pct`` covers all 8 cameras (main is never held, so main is black only
on a row without a main frame).

Nothing is written: the recording, its mask files and the cache are only
read. ``--json`` prints the numbers as JSON to stdout instead of the table.
The plan is the one ``cam_hold.annotate_hold`` gives the trainer; a source
frame that later fails to decode would also be black, and is not counted here
(no pixels are read).

    py -3 scripts/cam_hold_report.py --strips <recording> --cams "" ^
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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hold-last (--cam-age) black share and ages, read-only")
    parser.add_argument("--strips", type=Path, required=True, help="Recording folder (state.jsonl, raw/).")
    parser.add_argument("--cams", default="", help="Trained cameras, as train_scene --cams (selection keys on main).")
    parser.add_argument("--mask-files", default="", help="Same as train_scene --mask-files.")
    parser.add_argument("--mask-events", action="store_true", help="Same as train_scene --mask-events.")
    parser.add_argument("--holdout-blocks", type=int, default=None, help="Same as train_scene (default 1 or GVD_HOLDOUT_BLOCKS).")
    parser.add_argument("--holdout-cells-file", type=Path, default=None, help="Load this cell list (never chooses one).")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the table.")
    return parser


def _pct(black: int, total: int) -> float | None:
    return None if total <= 0 else round(100.0 * black / total, 2)


def report(
    root: Path | str,
    *,
    cams: tuple[str, ...] | None = None,
    mask_files: str = "",
    mask_events: bool = False,
    holdout_blocks: int = 1,
    holdout_cells_file: Path | None = None,
) -> dict[str, Any]:
    from eval_onnx_holdout import holdout_rows

    from python.data.cam_hold import BLACK_STAT_AGE_S, CAM_AGE_ORDER, NEVER_HELD, ROW_HOLD_KEY, annotate_hold, stat_age
    from python.data.raw_frames import load_scene_rows, refuse_lossy
    from python.debug.lane_targets import lane_center_is_valid, lane_gt_from_record
    from python.train.continual import CHUNK_MAX, pass_plan
    from python.train.event_mask import drop_masked, mask_file_sources
    from python.train.scene_loader import select_training_rows

    root = Path(root)
    slices, selection = holdout_rows(
        root,
        cams=cams,
        mask_files=mask_files,
        mask_events=mask_events,
        holdout_blocks=holdout_blocks,
        holdout_cells_file=holdout_cells_file,
        cam_age=True,
    )
    held_ids = {int(row["i"]) for rows in slices for row in rows}
    # The train rows: the same selection, every plan row that is not holdout.
    refuse_lossy(root)
    state_rows = load_scene_rows(root)
    sources = mask_file_sources(root, state_rows, mask_files)
    rows, _missing, _excluded = select_training_rows(root, state_rows, cams=cams, dropped={})
    if mask_events:
        from python.train.event_mask import mask_event_rows

        rows, _counts = mask_event_rows(root, rows)
    if sources:
        rows = drop_masked(rows, set().union(*(source["ids"] for source in sources)))
    cells = None
    if holdout_cells_file is not None:
        from python.train.continual import load_holdout_cells

        cells = load_holdout_cells(Path(holdout_cells_file), rows)
    plan = pass_plan(rows, CHUNK_MAX, CHUNK_MAX // 2, blocks=int(holdout_blocks), cells=cells)
    annotate_hold(state_rows, rows, plan)
    train_ids = {int(row["i"]) for item in plan if item.get("kind") != "holdout" for row in item.get("rows") or []}
    train = [row for row in rows if int(row["i"]) in train_ids]
    hold = [row for rows_ in slices for row in rows_]

    def scored(row: dict) -> bool:
        tech = row.get("tech") if isinstance(row.get("tech"), dict) else row
        _vec, mask = lane_gt_from_record(tech if isinstance(tech, dict) else None)
        return bool(lane_center_is_valid(mask))

    groups = {"train": train, "holdout": hold, "holdout_scored": [row for row in hold if scored(row)]}
    side = [cam for cam in CAM_AGE_ORDER if cam not in NEVER_HELD]
    out: dict[str, Any] = {
        "strips": str(root),
        "selection": selection,
        "rows": {name: len(items) for name, items in groups.items()},
        "black_pct": {},
        "black_n": {},
        "train_age_s": {},
        "train_age_capped_s": {},
        "black_stat_age_s": BLACK_STAT_AGE_S,
    }
    for name, items in groups.items():
        out["black_pct"][name] = {}
        out["black_n"][name] = {}
        for cam in CAM_AGE_ORDER:
            black = sum(1 for row in items if (row[ROW_HOLD_KEY][cam][0] is None))
            out["black_n"][name][cam] = black
            out["black_pct"][name][cam] = _pct(black, len(items))
    for cam in side:
        ages = np.asarray(
            [row[ROW_HOLD_KEY][cam][1] for row in train if row[ROW_HOLD_KEY][cam][0] is not None], dtype=np.float64
        )
        out["train_age_s"][cam] = {
            "n": int(ages.size),
            "median": None if ages.size == 0 else round(float(np.median(ages)), 4),
            "p95": None if ages.size == 0 else round(float(np.percentile(ages, 95)), 4),
        }
        capped = np.asarray(
            [stat_age(row[ROW_HOLD_KEY][cam][1] if row[ROW_HOLD_KEY][cam][0] is not None else None) for row in train],
            dtype=np.float64,
        )
        out["train_age_capped_s"][cam] = {
            "n": int(capped.size),
            "median": None if capped.size == 0 else round(float(np.median(capped)), 4),
            "p95": None if capped.size == 0 else round(float(np.percentile(capped, 95)), 4),
        }
    if held_ids & train_ids:
        raise RuntimeError("a holdout row is also a train row")
    return out


def table(result: dict[str, Any]) -> str:
    lines = [result["selection"], f"rows: {json.dumps(result['rows'])}"]
    lines.append(f"{'cam':<9}{'train black %':>15}{'holdout black %':>17}{'scored black %':>16}{'train age median':>18}{'p95':>8}{'capped median':>16}{'p95':>8}")
    for cam in result["black_pct"]["train"]:
        def cell(name: str) -> str:
            value = result["black_pct"][name].get(cam)
            return "n/a" if value is None else f"{value:.2f}"

        age = result["train_age_s"].get(cam) or {}
        capped = result["train_age_capped_s"].get(cam) or {}

        def f(value: Any) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        lines.append(
            f"{cam:<9}{cell('train'):>15}{cell('holdout'):>17}{cell('holdout_scored'):>16}"
            f"{f(age.get('median')):>18}{f(age.get('p95')):>8}{f(capped.get('median')):>16}{f(capped.get('p95')):>8}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from python.train.cam_strip import parse_cams
    from python.train.train_scene import resolve_holdout_blocks

    result = report(
        args.strips,
        # As train_scene: an empty --cams is all eight (main still keys the selection).
        cams=parse_cams(args.cams),
        mask_files=args.mask_files,
        mask_events=bool(args.mask_events),
        holdout_blocks=resolve_holdout_blocks(args.holdout_blocks),
        holdout_cells_file=args.holdout_cells_file,
    )
    print(json.dumps(result, indent=1) if args.json else table(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
