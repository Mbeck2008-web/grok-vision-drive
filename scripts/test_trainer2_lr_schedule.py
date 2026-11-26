#!/usr/bin/env python3
"""Trainer 2.0 PR 3: warmup + cosine LR schedule and per-group LR (docs/specs/trainer_2.md).

Checks: check_warmup_cosine_shape, check_param_groups_lr_mult,
check_resume_keeps_schedule, check_recipe_off_is_bit_identical (argparse
defaults pinned to main 8ce34d5; this PR adds no flag and no default).
Torch parts skip visibly without torch (CI installs torch CPU).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from gvd_test_env import skip_check  # noqa: E402

DEFAULTS_PIN = ROOT / "tests" / "fixtures" / "trainer2" / "train_scene_defaults_8ce34d5.json"


def _torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        return None
    return torch


def check_warmup_cosine_shape() -> None:
    from python.train.lr_schedule import FLOOR_FRAC, WarmupCosine, stage_peak

    peak = 1e-3
    sch = WarmupCosine(peak, 20_000)
    assert sch.warmup_steps == 1000, sch.warmup_steps  # 5% of 20k > 500
    assert WarmupCosine(peak, 4_000).warmup_steps == 500  # floor of 500 steps
    assert WarmupCosine(peak, 300).warmup_steps == 300  # stage shorter than the warmup
    lrs = [sch.lr_at(s) for s in range(20_000)]
    # Linear warmup: equal increments, reaching the peak on the last warmup step.
    assert math.isclose(lrs[0], peak / 1000.0, rel_tol=1e-12), lrs[0]
    assert math.isclose(lrs[999], peak, rel_tol=1e-12), lrs[999]
    steps = [lrs[i + 1] - lrs[i] for i in range(999)]
    assert max(steps) - min(steps) < 1e-15, (min(steps), max(steps))
    # Cosine: monotone non-increasing after the peak, halfway at the decay midpoint, ending on the floor.
    tail = lrs[1000:]
    assert all(b <= a + 1e-18 for a, b in zip(tail, tail[1:]))
    mid = 1000 + (20_000 - 1000) // 2
    floor = FLOOR_FRAC * peak
    assert math.isclose(lrs[mid], floor + (peak - floor) * 0.5, rel_tol=1e-3), lrs[mid]
    assert math.isclose(sch.lr_at(20_000), floor, rel_tol=1e-12)
    assert math.isclose(sch.lr_at(10**9), floor, rel_tol=1e-12)
    assert min(tail) >= floor - 1e-18  # warmup starts below the floor by design (peak/1000 at step 0)
    assert math.isclose(stage_peak(1e-3, joint=True), 1e-4, rel_tol=1e-12)
    assert stage_peak(1e-3, joint=False) == 1e-3
    for bad in ((0.0, 10), (-1e-3, 10), (float("nan"), 10), (1e-3, 0)):
        try:
            WarmupCosine(*bad)
        except ValueError:
            continue
        raise AssertionError(f"WarmupCosine{bad} not refused")
    print("warmup+cosine: 1000-step linear warmup to 1e-3, cosine to 5e-5, joint peak 1e-4 OK")


def check_param_groups_lr_mult() -> None:
    torch = _torch()
    if torch is None:
        skip_check("check_param_groups_lr_mult", "torch not installed")
        return
    from python.train.lr_schedule import scene_param_groups, stage_lr_mults
    from python.train.scene_net import SceneNet

    net = SceneNet()
    assert not hasattr(net, "param_groups"), "SceneNet.param_groups would change --backbone-lr-mult on scratch nets"
    mults = stage_lr_mults(ssl_done=True, trained_heads=["lane_head", "lane_valid_head"], entering_heads=["sign_head"])
    groups = scene_param_groups(net, 1e-3, mults)
    by_name = {g["name"]: g for g in groups}
    assert by_name["stem"]["lr_mult"] == 0.1 and math.isclose(by_name["stem"]["lr"], 1e-4)
    assert by_name["lane_head"]["lr_mult"] == 0.3 and math.isclose(by_name["lane_head"]["lr"], 3e-4)
    assert by_name["sign_head"]["lr_mult"] == 1.0 and by_name["sign_head"]["lr"] == 1e-3
    assert by_name["trunk"]["lr_mult"] == 1.0
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "a parameter is in two groups"
    assert set(ids) == {id(p) for p in net.parameters() if p.requires_grad}, "a parameter is in no group"
    stem_ids = {id(p) for p in net.stem.parameters()}
    assert {id(p) for p in by_name["stem"]["params"]} == stem_ids
    assert stage_lr_mults(ssl_done=False)["stem"] == 1.0
    try:
        scene_param_groups(net, 1e-3, {"lane_haed": 0.3})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown group name not refused")
    try:
        stage_lr_mults(ssl_done=True, trained_heads=["lane_head"], entering_heads=["lane_head"])
    except ValueError:
        pass
    else:
        raise AssertionError("head both trained and entering not refused")
    opt = torch.optim.AdamW(groups, lr=1e-3)
    assert [g["lr_mult"] for g in opt.param_groups] == [g["lr_mult"] for g in groups]
    print(f"param groups: {len(groups)} groups, every param once, stem x0.1 / trained x0.3 / entering x1.0 OK")


def check_resume_keeps_schedule() -> None:
    torch = _torch()
    if torch is None:
        skip_check("check_resume_keeps_schedule", "torch not installed")
        return
    from python.train.lr_schedule import WarmupCosine, apply_lr, scene_param_groups, stage_lr_mults
    from python.train.scene_net import SceneNet
    from python.train.train_scene import _pin_optimizer_lr

    net = SceneNet()
    mults = stage_lr_mults(ssl_done=True, trained_heads=["lane_head"], entering_heads=["lane_gt_head"])
    sch = WarmupCosine(1e-3, 2000)
    opt = torch.optim.AdamW(scene_param_groups(net, 1e-3, mults), lr=1e-3)
    straight = []
    for step in range(1200):
        apply_lr(opt, sch, step)
        straight.append([g["lr"] for g in opt.param_groups])
    # Resume at step 700: a fresh optimizer from the saved state, the CLI rate re-pinned, then the schedule.
    saved = json.loads(json.dumps(sch.state_dict()))
    opt2 = torch.optim.AdamW(scene_param_groups(net, 1e-3, mults), lr=1e-3)
    opt2.load_state_dict(opt.state_dict())
    _pin_optimizer_lr(opt2, 1e-3)
    pinned = [g["lr"] for g in opt2.param_groups]
    assert pinned == [1e-3 * g["lr_mult"] for g in opt2.param_groups], pinned
    sch2 = WarmupCosine.from_state_dict(saved)
    for step in range(700, 1200):
        apply_lr(opt2, sch2, step)
        assert [g["lr"] for g in opt2.param_groups] == straight[step], step
    print("resume: re-pinned CLI rate x lr_mult, then the restored schedule matches the uninterrupted run (500 steps) OK")


def check_recipe_off_is_bit_identical() -> None:
    torch = _torch()
    if torch is None:
        skip_check("check_recipe_off_is_bit_identical", "torch not installed (train_scene imports torch)")
        return
    from python.train.train_scene import build_parser

    pin = json.loads(DEFAULTS_PIN.read_text(encoding="utf-8"))
    assert pin["base"].startswith("8ce34d5"), pin["base"]
    got = json.loads(json.dumps(vars(build_parser().parse_args([])), sort_keys=True, default=str))
    want = pin["defaults"]
    added = sorted(set(got) - set(want))
    removed = sorted(set(want) - set(got))
    changed = sorted(k for k in set(got) & set(want) if got[k] != want[k])
    assert not (added or removed or changed), f"train_scene defaults drifted: added={added} removed={removed} changed={changed}"
    print(f"recipe off: {len(got)} train_scene argparse defaults == main 8ce34d5 pin OK")


def main() -> int:
    checks = (
        check_warmup_cosine_shape,
        check_param_groups_lr_mult,
        check_resume_keeps_schedule,
        check_recipe_off_is_bit_identical,
    )
    failed = 0
    for check in checks:
        try:
            check()
        except Exception as exc:  # noqa: BLE001 - report every check
            failed += 1
            print(f"FAIL {check.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
