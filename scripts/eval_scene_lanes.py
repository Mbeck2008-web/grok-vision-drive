#!/usr/bin/env python3
"""Score scene-net lane centers on the last 20% of a recording, in time.

The split is the end of the clock, not a shuffled slice. r and rms compare
the predicted 8 m center with the ground-truth center. Without a checkpoint
the script only checks that the recording has lane targets.

    .venv\\Scripts\\python scripts\\eval_scene_lanes.py --strips data\\scene_lanes --check-targets
    .venv\\Scripts\\python scripts\\eval_scene_lanes.py --strips data\\scene_lanes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.data.strip_writer import read_state_jsonl  # noqa: E402
from python.debug.lane_targets import (  # noqa: E402
    lane_gt_from_record,
    lane_x_from_vector,
    score_centers,
    time_holdout,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate lane centers on a time holdout")
    parser.add_argument("--strips", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--check-targets",
        action="store_true",
        help="Check the recording only. Do not load weights.",
    )
    return parser


def _rows(root: Path) -> list[dict]:
    rows = read_state_jsonl(root / "state.jsonl")
    rows.sort(key=lambda row: float(row.get("t") or 0.0))
    return rows


def _centers(rows: list[dict]) -> list[float]:
    out = []
    for row in rows:
        tech = row.get("tech") if isinstance(row.get("tech"), dict) else row
        values, _mask = lane_gt_from_record(tech)
        center = lane_x_from_vector(values)
        out.append(float("nan") if center is None else float(center))
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = _rows(args.strips)
    if len(rows) < 5:
        print(f"eval: {args.strips} has {len(rows)} rows", file=sys.stderr)
        return 2
    train, hold = time_holdout(rows, 0.2)
    gt = _centers(hold)
    finite = sum(1 for value in gt if value == value)
    print(f"eval rows={len(rows)} train={len(train)} holdout={len(hold)} gt_centers={finite}")
    if args.check_targets:
        if finite < 3:
            print("eval: holdout has fewer than 3 ground-truth centers", file=sys.stderr)
            return 2
        print("eval: targets ok")
        return 0
    try:
        import torch
    except ImportError:
        print("eval: PyTorch is not installed. Train on the 1080 Ti with torch 2.14 cu126.", file=sys.stderr)
        return 2
    from python.train.checkpoint import load_checkpoint
    from python.train.scene_net import SceneNet, torch_ready

    if not torch_ready():
        print("eval: PyTorch is not installed.", file=sys.stderr)
        return 2
    payload = load_checkpoint(args.strips)
    weights = None if payload is None else payload.get("weights")
    if args.checkpoint is not None and args.checkpoint.is_file() and str(args.checkpoint).endswith(".pt"):
        blob = torch.load(args.checkpoint, map_location="cpu")
        weights = blob.get("model") if isinstance(blob, dict) and "model" in blob else blob
    if not isinstance(weights, dict) or not weights:
        print(
            f"eval: no weights next to {args.strips}. Train first; the pause file is "
            f"{args.strips.name}.gvd-pause.npz beside that folder.",
            file=sys.stderr,
        )
        return 2
    net = SceneNet()
    state = {
        key: torch.as_tensor(value)
        for key, value in weights.items()
    }
    net.load_state_dict(state)
    net.eval()
    # One sequence. dt is the recorded step, not a fixed period.
    pred: list[float] = []
    hidden = net.initial_state()
    from python.train.train_scene import _load_bgr

    for row in hold:
        image = _load_bgr(args.strips, int(row["i"]))
        dt = row.get("dt_s")
        with torch.no_grad():
            out, hidden = net(image, 0.0 if dt is None else float(dt), hidden)
        center = lane_x_from_vector(out.lane_gt.detach().cpu().numpy())
        pred.append(float("nan") if center is None else float(center))
    scored = score_centers(pred, gt)
    r = scored["r"]
    rms = scored["rms"]
    print(
        f"eval holdout n={scored['n']} r={r if r is None else f'{r:.3f}'} "
        f"rms_m={rms if rms is None else f'{rms:.3f}'} bias_m={scored['bias']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
