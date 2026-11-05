#!/usr/bin/env python3
"""``--val-worst``: the N worst holdout frames of each val, read-only.

* heap: a stream of losses keeps exactly the N largest, ranked, ties keep the
  earlier frame; N is capped.
* run: a tiny CPU run with ``val_worst=4, val_worst_png=True`` writes one
  ``val_worst.run<id>.ep<E>.json`` per val row (named in the row), 4 frames
  each, ranked by loss, every one with recording, frame index, camera,
  gt_origin and a PNG; the worst loss is at least the val mean loss.
* bit-identical: the same run with ``val_worst=0`` writes no dump, and its
  train losses, val rows and checkpoint weights equal the dump-on run's exactly.
* cli: ``--val-worst`` defaults to 16, ``--val-worst-png`` is off.
"""
# gvd-ci: requires torch
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from test_epoch_ckpt import _fixture, _train  # noqa: E402


def check_heap() -> None:
    from python.train.val_worst import MAX_N, WorstFrames

    worst = WorstFrames(3)
    losses = [0.5, 2.0, 0.1, 2.0, 3.0, 1.0, 0.2]
    for i, loss in enumerate(losses):
        worst.note(loss, {"i": i, "tech": {"gt_origin": "body_centreline"}}, None, None, None)
    ranked = worst.ranked()
    assert [(r["i"], r["loss"]) for r in ranked] == [(4, 3.0), (1, 2.0), (3, 2.0)], ranked
    assert ranked[0]["gt_origin"] == "body_centreline"
    assert WorstFrames(10_000).n == MAX_N
    print("heap: keeps the 3 largest of 7, ranked, tie keeps the earlier frame; N capped OK")


def _run(tmp: Path, name: str, n: int) -> dict:
    import torch

    torch.manual_seed(0)
    torch.set_num_threads(1)
    root = tmp / name / "rec"
    _fixture(root)
    status = tmp / name / "_status"
    ckpts = tmp / name / "ckpts"
    with contextlib.redirect_stdout(io.StringIO()):
        _train(root, ckpt_dir=ckpts, status_root=status, metrics_every=1, val_worst=n, val_worst_png=n > 0)
    rows = [json.loads(line) for line in (status / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    weights = []
    for path in sorted(ckpts.glob("*.npz")):
        with np.load(path, allow_pickle=False) as blob:
            weights.append({k: blob[k].tobytes() for k in blob.files if k != "meta"})
    return {"status": status, "rows": rows, "weights": weights}


def _comparable(rows: list[dict]) -> list[dict]:
    drop = {"t", "ckpt", "ckpt_sha256", "val_worst", "elapsed_s", "eta_s", "steps_per_sec", "memory_bytes", "step_s"}  # wall clock
    return [{k: v for k, v in r.items() if k not in drop} for r in rows if r.get("kind") in ("train", "val")]


def check_run_and_identity() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        on = _run(Path(tmp), "on", 4)
        off = _run(Path(tmp), "off", 0)
        _check(on, off)


def _check(on: dict, off: dict) -> None:
    vals = [r for r in on["rows"] if r.get("kind") == "val"]
    assert vals, on["rows"][-3:]
    for rec in vals:
        name = rec.get("val_worst")
        assert name == f"val_worst.run{name.split('.run')[1].split('.ep')[0]}.ep{rec['epoch']}.json", rec
        dump = json.loads((on["status"] / name).read_text(encoding="utf-8"))
        assert dump["epoch"] == rec["epoch"] and dump["step"] == rec["step"] and dump["n"] == 4, dump
        frames = dump["frames"]
        losses = [f["loss"] for f in frames]
        assert losses == sorted(losses, reverse=True), losses
        assert losses[0] >= rec["loss"] - 1e-6, (losses, rec["loss"])
        for f in frames:
            assert f["recording"] == "rec" and f["camera"] == "main" and isinstance(f["i"], int), f
            assert f["gt_origin"] == "unknown" and len(f["pred"]) == len(f["gt"]) == 14, f
            assert (on["status"] / f["png"]).stat().st_size > 0, f
        assert (on["status"] / name).stat().st_size < 64_000
    print(f"run: {len(vals)} val rows, each with a 4-frame ranked dump + PNGs OK")
    assert not list(off["status"].glob("val_worst*")), "val_worst=0 wrote a dump"
    a, b = _comparable(on["rows"]), _comparable(off["rows"])
    assert a == b, "metrics differ with the dump on"
    assert on["weights"] and on["weights"] == off["weights"], "checkpoint weights differ with the dump on"
    n_train = sum(1 for r in a if r["kind"] == "train")
    print(f"bit-identical: {n_train} train rows, {len(vals)} val rows, {len(on['weights'])} checkpoints equal on vs off OK")


def check_cli() -> None:
    from python.train.train_scene import build_parser

    args = build_parser().parse_args([])
    assert args.val_worst == 16 and args.val_worst_png is False, args
    assert build_parser().parse_args(["--val-worst", "0"]).val_worst == 0
    print("cli: --val-worst 16 by default, png off OK")


def main() -> int:
    check_heap()
    check_cli()
    check_run_and_identity()
    print("val worst OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
