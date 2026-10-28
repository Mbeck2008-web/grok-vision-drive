#!/usr/bin/env python3
"""``--seed``: one flag for the training RNGs; the holdout ignores it.

* cli: ``--seed`` defaults to 0; seed 0 maps to the old numpy streams (0, 1).
* reproduce: ``seed=0`` on a tiny CPU fixture (workers=0, augment on) gives
  the first 20 step losses that main (91dc148, fixed seeding) gave on the
  same fixture, and logs ``[GVD] seed=0`` once. The losses were recorded on
  main before the change. The check allows 1e-3 relative for other CPUs; on
  the recording machine they are bit-identical.
* seed changes training: ``seed=67`` gives different losses.
* holdout: ``seed=0`` and ``seed=67`` give identical holdout rows (the plan's
  holdout items) and the same val n. Only ``--holdout-seed`` moves the holdout.

Not bit-reproducible on GPU: cudnn determinism is not switched on.
"""
# gvd-ci: requires torch
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

N_STEPS = 20
# First 20 train-step losses of main 91dc148 on this fixture (workers=0, augment on,
# torch threads 1, CPU, torch 2.14.1), recorded before --seed existed.
MAIN_LOSSES = [
    0.8842411637306213, 0.4393423795700073, 0.23833239078521729, 0.24891415238380432,
    0.4641000032424927, 0.08581289649009705, 0.1262638419866562, 0.33596330881118774,
    0.28853532671928406, 0.12152993679046631, 0.3143858313560486, 0.4122709035873413,
    0.07514192909002304, 0.07887139916419983, 0.3117033541202545, 0.03457614779472351,
    0.04766307771205902, 0.10393509268760681, 0.09006150811910629, 0.10800345987081528,
]


def check_cli() -> None:
    from python.train.train_scene import build_parser, run_rng_seeds

    assert build_parser().parse_args([]).seed == 0
    assert build_parser().parse_args(["--seed", "67"]).seed == 67
    assert run_rng_seeds(0) == (0, 1), run_rng_seeds(0)
    assert len(set(run_rng_seeds(0) + run_rng_seeds(1) + run_rng_seeds(67))) == 6
    print("cli: --seed defaults to 0; seed 0 -> numpy streams (0, 1) OK")


def _run(tmp: Path, seed: int) -> dict:
    import torch

    from test_epoch_ckpt import _fixture
    from python.train import train_scene

    torch.set_num_threads(1)
    root = tmp / f"scene_s{seed}"
    _fixture(root)
    status = tmp / f"_status{seed}"
    holdouts: list = []
    real = train_scene.pass_plan

    def _spy(*args, **kw):
        plan = real(*args, **kw)
        holdouts.append(json.dumps([item for item in plan if item.get("kind") == "holdout"], sort_keys=True, default=str))
        return plan

    out = io.StringIO()
    train_scene.pass_plan = _spy
    try:
        with contextlib.redirect_stdout(out):
            train_scene.train_directory(
                root,
                epochs=1,
                lr=1e-3,
                capability=None,
                export_path=None,
                free_vram_bytes=0,
                free_ram_bytes=8_000_000_000,
                resume=False,
                open_panel=False,
                autostart=True,
                heads=("lane_gt",),
                cams=("main",),
                augment=True,
                workers=0,
                min_valid=0.0,
                holdout_blocks=1,
                metrics_every=1,
                status_root=status,
                seed=seed,
            )
    finally:
        train_scene.pass_plan = real
    rows = [json.loads(line) for line in (status / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "losses": [row["loss"] for row in rows if row.get("kind") == "train"][:N_STEPS],
        "val_n": [row.get("n") for row in rows if row.get("kind") == "val"],
        "holdouts": holdouts,
        "seed_lines": [line for line in out.getvalue().splitlines() if line.startswith("[GVD] seed=")],
    }


def check_runs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        zero = _run(Path(tmp), 0)
        other = _run(Path(tmp), 67)
    assert zero["seed_lines"] == ["[GVD] seed=0"], zero["seed_lines"]
    assert other["seed_lines"] == ["[GVD] seed=67"], other["seed_lines"]
    assert len(zero["losses"]) == N_STEPS, zero["losses"]
    exact = zero["losses"] == MAIN_LOSSES
    worst = max(abs(a - b) / max(abs(b), 1e-12) for a, b in zip(zero["losses"], MAIN_LOSSES))
    assert worst <= 1e-3, f"seed 0 does not reproduce main's first {N_STEPS} losses (worst rel {worst:.3g})"
    print(f"reproduce: seed 0 == main's first {N_STEPS} losses ({'bit-identical' if exact else f'worst rel {worst:.2g}'}) OK")
    moved = max(abs(a - b) / max(abs(b), 1e-12) for a, b in zip(other["losses"], MAIN_LOSSES))
    assert moved > 1e-2, f"seed 67 gave seed 0's losses (worst rel {moved:.3g})"
    print(f"seed 67: losses differ from seed 0 (worst rel {moved:.2g}) OK")
    # The holdout decides when a run stops, so the two seeds may build the plan a
    # different number of times; every plan of both runs must hold out the same rows.
    plans = set(zero["holdouts"]) | set(other["holdouts"])
    assert zero["holdouts"] and other["holdouts"] and len(plans) == 1, f"the holdout moved with --seed ({len(plans)} plans)"
    sizes = set(zero["val_n"]) | set(other["val_n"])
    assert len(sizes) == 1 and all(math.isfinite(float(n)) and n > 0 for n in sizes), (zero["val_n"], other["val_n"])
    print(f"holdout: seed 0 and seed 67 hold out the same rows in all {len(zero['holdouts'])}+{len(other['holdouts'])} plans (val n={sizes.pop()}) OK")


def main() -> None:
    check_cli()
    check_runs()
    print("test_seed: all OK")


if __name__ == "__main__":
    main()
