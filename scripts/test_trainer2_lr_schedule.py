#!/usr/bin/env python3
"""Trainer 2.0 PR 3: warmup + cosine LR schedule and per-group LR (docs/specs/trainer_2.md).

Checks: check_warmup_cosine_shape, check_param_groups_lr_mult,
check_apply_lr_scales_each_group, check_state_dict_round_trip,
check_resume_keeps_schedule, check_recipe_off_is_bit_identical (the argparse
defaults that existed at main 8ce34d5 are pinned; this PR changes none of them.
A flag added later does not fail it).
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
    # Cosine, not linear: hand-computed values at the quarter, middle and three-quarter points of the
    # 19000-step decay (steps 5750, 10500, 15250). A linear decay gives 7.6e-4 / 5.25e-4 / 3.4e-4 -- only
    # the middle one agrees, so the quarter points are what tell the shapes apart.
    #   lr = floor + (peak - floor) * (1 + cos(pi * frac)) / 2, peak 1e-3, floor 5e-5
    quarter = 5e-5 + 9.5e-4 * (1.0 + math.sqrt(0.5)) / 2.0  # frac 0.25
    three_q = 5e-5 + 9.5e-4 * (1.0 - math.sqrt(0.5)) / 2.0  # frac 0.75
    assert math.isclose(quarter, 8.608757e-4, rel_tol=1e-6) and math.isclose(three_q, 1.891243e-4, rel_tol=1e-6)
    assert math.isclose(lrs[1000 + 4750], quarter, rel_tol=1e-12), (lrs[1000 + 4750], quarter)
    assert math.isclose(lrs[1000 + 9500], 5.25e-4, rel_tol=1e-12), lrs[1000 + 9500]
    assert math.isclose(lrs[1000 + 14250], three_q, rel_tol=1e-12), (lrs[1000 + 14250], three_q)
    assert not math.isclose(lrs[1000 + 4750], 7.6e-4, rel_tol=1e-2), "decay looks linear"
    # Default warmup rounds 5% up: 20001 steps -> 1000.05 -> 1001.
    assert WarmupCosine(peak, 20_001).warmup_steps == 1001
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
    # Frozen parameters are in no group, and a fully frozen group is dropped.
    frozen_stem = {id(p) for p in net.stem.parameters()}
    for p in net.stem.parameters():
        p.requires_grad_(False)
    lane_params = list(net.lane_head.parameters())
    assert len(lane_params) >= 2
    lane_params[0].requires_grad_(False)
    thawed = scene_param_groups(net, 1e-3, mults)
    thawed_ids = {id(p) for g in thawed for p in g["params"]}
    assert not (thawed_ids & frozen_stem), "frozen stem params landed in a group"
    assert id(lane_params[0]) not in thawed_ids, "a frozen head param landed in a group"
    assert "stem" not in {g["name"] for g in thawed}, "an all-frozen group was kept"
    assert {id(p) for p in lane_params[1:]} <= {id(p) for g in thawed if g["name"] == "lane_head" for p in g["params"]}
    assert thawed_ids == {id(p) for p in net.parameters() if p.requires_grad}
    for p in net.parameters():
        p.requires_grad_(True)
    # A group missing from mults trains at 1.0 (not 0.3, not 0.0): only lane_head is listed here.
    partial = {g["name"]: g for g in scene_param_groups(net, 2e-3, {"lane_head": 0.3})}
    assert partial["lane_head"]["lr_mult"] == 0.3 and math.isclose(partial["lane_head"]["lr"], 6e-4)
    for name, g in partial.items():
        if name != "lane_head":
            assert g["lr_mult"] == 1.0 and g["lr"] == 2e-3, (name, g["lr_mult"], g["lr"])
    assert {"stem", "trunk"} <= set(partial)
    assert scene_param_groups(net, 2e-3, {}) and all(
        g["lr_mult"] == 1.0 and g["lr"] == 2e-3 for g in scene_param_groups(net, 2e-3, {})
    ), "an empty mults mapping must leave every group at 1.0"
    opt = torch.optim.AdamW(groups, lr=1e-3)
    assert [g["lr_mult"] for g in opt.param_groups] == [g["lr_mult"] for g in groups]
    print(f"param groups: {len(groups)} groups, every param once, stem x0.1 / trained x0.3 / entering x1.0 OK")


def check_apply_lr_scales_each_group() -> None:
    torch = _torch()
    if torch is None:
        skip_check("check_apply_lr_scales_each_group", "torch not installed")
        return
    from python.train.lr_schedule import WarmupCosine, apply_lr, scene_param_groups, stage_lr_mults
    from python.train.scene_net import SceneNet

    net = SceneNet()
    mults = stage_lr_mults(ssl_done=True, trained_heads=["lane_head"], entering_heads=["lane_gt_head"])
    opt = torch.optim.AdamW(scene_param_groups(net, 1e-3, mults), lr=1e-3)
    sch = WarmupCosine(1e-3, 2000)
    want = {"stem": 0.1, "trunk": 1.0, "lane_head": 0.3, "lane_gt_head": 1.0}  # literal, not read back from the groups
    for step in (0, 99, 100, 999, 1500, 1999):
        rate = apply_lr(opt, sch, step)
        assert rate == sch.lr_at(step)
        seen = {}
        for g in opt.param_groups:
            if g["name"] in want:
                seen[g["name"]] = g["lr"]
                assert math.isclose(g["lr"], rate * want[g["name"]], rel_tol=1e-12), (step, g["name"], g["lr"], rate)
        assert set(seen) == set(want), seen
    # A group with no lr_mult key counts as 1.0.
    plain = torch.optim.SGD([{"params": [torch.nn.Parameter(torch.zeros(1))]}], lr=5.0)
    apply_lr(plain, sch, 0)
    assert math.isclose(plain.param_groups[0]["lr"], sch.lr_at(0), rel_tol=1e-12)
    print("apply_lr: each group = schedule(step) x literal lr_mult (stem 0.1, lane_head 0.3, entering 1.0) at 6 steps OK")


def check_state_dict_round_trip() -> None:
    from python.train.lr_schedule import WarmupCosine

    for kwargs in ({}, {"warmup_steps": 123}, {"warmup_steps": 0}, {"floor_frac": 0.2, "warmup_steps": 40}):
        sch = WarmupCosine(2e-3, 3000, **kwargs)
        back = WarmupCosine.from_state_dict(json.loads(json.dumps(sch.state_dict())))
        assert back.state_dict() == sch.state_dict(), (kwargs, back.state_dict(), sch.state_dict())
        assert back.warmup_steps == sch.warmup_steps
        assert [back.lr_at(s) for s in range(0, 3100, 7)] == [sch.lr_at(s) for s in range(0, 3100, 7)], kwargs
    assert WarmupCosine(2e-3, 3000, warmup_steps=123).warmup_steps == 123  # custom warmup is not the 5%/500 default
    assert WarmupCosine.from_state_dict(WarmupCosine(2e-3, 3000, warmup_steps=123).state_dict()).warmup_steps == 123
    print("state_dict: custom warmup (123, 0), floor_frac and default all round-trip, lr_at identical OK")


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
    # Pin only the defaults that existed at 8ce34d5. A flag added later (e.g. #263's --no-ssl-pretrain,
    # --ssl-epochs, --ssl-max-rows) is not this PR's to forbid; removing or changing a pinned one is.
    added = sorted(set(got) - set(want))
    removed = sorted(set(want) - set(got))
    changed = sorted(k for k in set(got) & set(want) if got[k] != want[k])
    assert not (removed or changed), f"train_scene pre-existing defaults drifted: removed={removed} changed={changed}"
    print(f"recipe off: {len(want)} pre-existing train_scene argparse defaults == main 8ce34d5 pin OK "
          f"({len(added)} newer flag(s) not pinned)")


def main() -> int:
    checks = (
        check_warmup_cosine_shape,
        check_param_groups_lr_mult,
        check_apply_lr_scales_each_group,
        check_state_dict_round_trip,
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
