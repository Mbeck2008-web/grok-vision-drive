#!/usr/bin/env python3
"""Trainer 2.0 PR 6: ``train_scene --recipe trainer2`` curriculum runner.

docs/specs/trainer_2.md Appendix C PR 6: stages 0-2 / 2b live; 3-6 skip-logged
until Reality View is built and integrated; deploy check; ckpt retention.

Named AC: check_curriculum_order_and_gates, check_student_export_fits_deploy,
check_old_checkpoints_untouched.

This recipe is meant for a live export, so ``--scale < 1`` and ``--augment``
are refused here (deferred from Historian PR 2b; this PR does not invent
``strip_to_nchw`` or touch ``gvd/t2-sceneckpt-8cam``).

The recipe flag itself stays default off (**spec proposal**). Exit / max-epoch
/ non-regression / joint-peak / accum / WEIGHT_LR numbers that the spec marks
proposal are imported as-is and flagged; they are not decided here.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from gvd_test_env import skip_check  # noqa: E402

DEFAULTS_PIN = ROOT / "tests" / "fixtures" / "trainer2" / "train_scene_defaults_8ce34d5.json"


def _torch():
    try:
        import torch
    except ImportError:
        return None
    torch.set_num_threads(2)
    return torch


def _parser():
    from python.train.train_scene import build_parser

    return build_parser()


def _recipe():
    from python.train import trainer2_recipe as rec

    return rec


def check_recipe_default_off() -> None:
    rec = _recipe()
    assert rec.RECIPE_DEFAULT in ("", None, False, "off")
    args = _parser().parse_args([])
    assert getattr(args, "recipe", "") in ("", None, False, "off")
    rec.validate_recipe_args(args)  # off: scale/augment defaults stay legal
    print("recipe default off (spec proposal) OK")


def check_recipe_refuses_scale_lt_1() -> None:
    rec = _recipe()
    # scale == 1 is allowed; anything below is not.
    rec.validate_live_export_recipe(recipe="trainer2", scale=1.0, augment=False)
    rec.validate_live_export_recipe(recipe="trainer2", scale=1.0, augment=False, kd=False)
    for scale in (0.0, 0.5, 0.999, 0.5 * 0.5):
        try:
            rec.validate_live_export_recipe(recipe="trainer2", scale=scale, augment=False)
        except rec.RecipeRefused as exc:
            assert "--scale" in str(exc) and "live export" in str(exc).lower()
        else:
            raise AssertionError(f"scale {scale} must be refused")
    # recipe off still allows --scale < 1 (train-only path).
    rec.validate_live_export_recipe(recipe="", scale=0.5, augment=True)
    rec.validate_recipe_args(SimpleNamespace(recipe="", scale=0.5, augment=True, kd=False))
    args = _parser().parse_args(["--recipe", "trainer2", "--no-augment", "--scale", "0.5"])
    try:
        rec.validate_recipe_args(args)
    except rec.RecipeRefused as exc:
        assert "--scale" in str(exc)
    else:
        raise AssertionError("CLI --scale 0.5 must be refused")
    print("recipe refuses --scale < 1 (and recipe-off still allows it) OK")


def check_recipe_refuses_augment() -> None:
    rec = _recipe()
    try:
        rec.validate_live_export_recipe(recipe="trainer2", scale=1.0, augment=True)
    except rec.RecipeRefused as exc:
        assert "--augment" in str(exc) and "live export" in str(exc).lower()
    else:
        raise AssertionError("augment=True must be refused")
    # Default --augment is True, so bare --recipe trainer2 is refused.
    args = _parser().parse_args(["--recipe", "trainer2"])
    assert args.augment is True
    try:
        rec.validate_recipe_args(args)
    except rec.RecipeRefused as exc:
        assert "--augment" in str(exc)
    else:
        raise AssertionError("default --augment with --recipe trainer2 must be refused")
    rec.validate_recipe_args(_parser().parse_args(["--recipe", "trainer2", "--no-augment"]))
    # recipe off keeps photometric jitter on.
    off = _parser().parse_args([])
    assert off.augment is True
    rec.validate_recipe_args(off)
    print("recipe refuses --augment (default-on is refused; --no-augment required) OK")


def check_kd_refused_in_pr6() -> None:
    rec = _recipe()
    args = _parser().parse_args(["--recipe", "trainer2", "--no-augment", "--kd"])
    try:
        rec.validate_recipe_args(args)
    except rec.RecipeRefused as exc:
        assert "PR 8" in str(exc) or "PR8" in str(exc)
    else:
        raise AssertionError("--kd must be refused in PR 6")
    rec.validate_recipe_args(_parser().parse_args(["--kd"]))  # flag off-path is ignored until PR 8
    print("--kd refused under --recipe trainer2 (PR 8, not this PR) OK")


def check_unknown_recipe_refused() -> None:
    rec = _recipe()
    try:
        rec.validate_live_export_recipe(recipe="magic", scale=1.0, augment=False)
    except rec.RecipeRefused:
        pass
    else:
        raise AssertionError("unknown recipe must be refused")
    print("unknown recipe refused OK")


def check_curriculum_order_and_gates() -> None:
    rec = _recipe()
    assert rec.STAGE_ORDER == ("0", "1", "2", "2b", "3", "4", "5", "6", "7")
    assert rec.REALITY_VIEW_BUILT_AND_INTEGRATED is False
    # Proposal numbers are named and flagged; this PR does not change them.
    assert rec.EXIT_GAIN == 0.01  # spec proposal: no gain > 1% over 3 evals
    assert rec.EXIT_PATIENCE_EVALS == 3
    assert rec.NON_REGRESSION_FRAC == 0.05  # spec proposal
    assert rec.JOINT_HEAD_REGRESSION_FRAC == 0.02  # spec proposal
    assert rec.STAGE_MAX_EPOCHS >= 1  # spec proposal: a max epoch count

    plan = rec.plan_curriculum()
    ids = [s.id for s in plan]
    assert ids == list(rec.STAGE_ORDER), ids
    live = [s for s in plan if s.action == "run"]
    skipped = [s for s in plan if s.action == "skip"]
    assert [s.id for s in live] == ["0", "1", "2", "2b", "7"], [s.id for s in live]
    assert [s.id for s in skipped] == ["3", "4", "5", "6"], [s.id for s in skipped]
    for s in skipped:
        assert "Reality View" in s.reason, s.reason
        assert s.log_skip is True

    # When Reality View is ready but the heads are not on main, still skip (different reason).
    ready = rec.plan_curriculum(reality_view_ready=True, heads_exist=dict.fromkeys(rec.STAGE_ORDER, False))
    for s in ready:
        if s.id in ("3", "4", "5", "6"):
            assert s.action == "skip"
            assert "Reality View" not in s.reason or "not on main" in s.reason or "exist" in s.reason

    # When Reality View is ready and heads+data exist, 3-6 are runnable (entry still needs prev exited).
    all_on = dict.fromkeys(rec.STAGE_ORDER, True)
    go = rec.plan_curriculum(reality_view_ready=True, heads_exist=all_on, data_exist=all_on)
    assert [s.id for s in go if s.action == "run"] == list(rec.STAGE_ORDER)

    # Entry: a later stage is refused until the previous one has exited.
    ok, reason = rec.entry_allowed("2", prev_exited=False, heads_exist=True, data_exist=True)
    assert ok is False and "previous" in reason.lower()
    ok, reason = rec.entry_allowed("2", prev_exited=True, heads_exist=True, data_exist=True)
    assert ok is True, reason
    ok, reason = rec.entry_allowed("0", prev_exited=False, heads_exist=True, data_exist=True)
    assert ok is True, reason  # first stage has no predecessor
    ok, reason = rec.entry_allowed("3", prev_exited=True, heads_exist=True, data_exist=True, reality_view_ready=False)
    assert ok is False and "Reality View" in reason

    # Exit: no gain > 1% over 3 evals (lower-is-better rms). Exactly 1% is not > 1%.
    # history is the eval sequence of the primary metric.
    assert rec.stage_exited("2", [1.00, 0.995, 0.990, 0.985], epochs=4) is True  # 0.5% gains
    assert rec.stage_exited("2", [1.00, 0.995, 0.990], epochs=3) is False  # only 3 evals, not yet
    assert rec.stage_exited("2", [1.00, 0.98, 0.96, 0.94], epochs=4) is False  # 2% gains
    # A late 1.1% gain resets patience.
    assert rec.stage_exited("2", [1.00, 0.995, 0.990, 0.978], epochs=4) is False
    # Max epochs (proposal) exits even if still improving.
    assert rec.stage_exited("2", [1.00, 0.90], epochs=rec.STAGE_MAX_EPOCHS) is True
    assert rec.stage_exited("2", [1.00, 0.90], epochs=rec.STAGE_MAX_EPOCHS - 1) is False
    # Higher-is-better (sem mIoU / lane r): a 2% rise is a gain.
    assert rec.metric_gain(0.50, 0.51, metric="sem_miou") > rec.EXIT_GAIN
    assert rec.metric_gain(0.50, 0.504, metric="sem_miou") <= rec.EXIT_GAIN  # 0.8%

    # Non-regression (proposal): a stage may not make an earlier metric > 5% relative worse.
    # 5% exactly is not >; 5%+epsilon is.
    rec.check_non_regression({"lane_rms": 1.0}, {"lane_rms": 1.05})  # 5.0% worse, allowed
    try:
        rec.check_non_regression({"lane_rms": 1.0}, {"lane_rms": 1.0500001})
    except rec.CheckpointRejected as exc:
        assert "lane_rms" in str(exc)
    else:
        raise AssertionError("just over 5% worse must reject the checkpoint")
    rec.check_non_regression({"lane_r": 0.80}, {"lane_r": 0.76})  # 5.0% drop, allowed
    try:
        rec.check_non_regression({"lane_r": 0.80}, {"lane_r": 0.759})
    except rec.CheckpointRejected:
        pass
    else:
        raise AssertionError("lane_r drop just over 5% must reject")

    # Orchestrator: skip-logs 3-6, rejects a regressing ckpt, does not call train on skipped stages.
    trained: list[str] = []
    logs: list[str] = []

    def train_fn(stage, prev_ckpt):
        trained.append(stage.id)
        # Stage 7 reports a >5% worse earlier-stage rms so the gate fires.
        if stage.id == "7":
            return rec.StageResult(
                ckpt="stage7.ckpt",
                metrics={"lane_rms": 1.2, "lane_r": 0.8, "ssl_pretext": 0.5, "depth_absrel": 0.2, "sem_miou": 0.4},
                evals=[1.2],
                epochs=1,
            )
        metrics = {"lane_rms": 1.0, "lane_r": 0.8, "ssl_pretext": 0.5, "depth_absrel": 0.2, "sem_miou": 0.4}
        return rec.StageResult(ckpt=f"stage{stage.id}.ckpt", metrics=metrics, evals=[1.0, 0.995, 0.99, 0.985], epochs=4)

    try:
        rec.run_curriculum(train_fn, log=logs.append)
    except rec.CheckpointRejected as exc:
        assert "lane_rms" in str(exc), exc
    else:
        raise AssertionError("non-regression must reject the stage-7 checkpoint")
    assert trained == ["0", "1", "2", "2b", "7"], trained
    skip_text = "\n".join(logs)
    for sid in ("3", "4", "5", "6"):
        assert sid in skip_text and "Reality View" in skip_text, skip_text

    print("curriculum order, skip-log of 3-6, entry/exit, non-regression reject OK")


def check_student_export_fits_deploy() -> None:
    rec = _recipe()
    from python.train.model_size import PRESETS

    assert PRESETS["tiny"].est_vram_gb <= 4.0
    assert PRESETS["small"].est_vram_gb <= 4.0
    assert PRESETS["medium"].est_vram_gb > 4.0
    assert rec.DEPLOY_PRESETS == frozenset({"tiny", "small"})
    assert rec.live_vram_fits(4.0) is True
    assert rec.live_vram_fits(4.0001) is False
    assert rec.live_vram_fits(100.0) is False
    rec.student_export_fits_deploy(preset="tiny", student_param_count=100, exported_param_count=100)
    rec.student_export_fits_deploy(preset="small", student_param_count=200, exported_param_count=200)
    for preset in ("medium", "large", "xl"):
        try:
            rec.student_export_fits_deploy(preset=preset, student_param_count=10, exported_param_count=10)
        except rec.RecipeRefused as exc:
            assert preset in str(exc) or "deploy" in str(exc).lower() or "4" in str(exc)
        else:
            raise AssertionError(f"{preset} must not fit the 4 GB live deploy check")
    try:
        rec.student_export_fits_deploy(preset="small", student_param_count=200, exported_param_count=201)
    except rec.RecipeRefused as exc:
        assert "param" in str(exc).lower()
    else:
        raise AssertionError("param-count mismatch must fail the deploy check")
    try:
        rec.student_export_fits_deploy(preset="huge", student_param_count=1, exported_param_count=1)
    except rec.RecipeRefused:
        pass
    else:
        raise AssertionError("unknown preset must fail")
    # Live inference VRAM cap is 4 GB (decided by Michael 2026-11-25).
    assert rec.LIVE_INFER_VRAM_GB == 4.0
    print("student export fits deploy (tiny/small only; param counts must match; VRAM <= 4 GB) OK")


def check_old_checkpoints_untouched() -> None:
    rec = _recipe()
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        old = root / "epoch-000-existing.ckpt"
        payload = b"keep-me-untouched"
        old.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        # Own partial may be unlinked; an existing ckpt may not.
        assert rec.may_unlink_checkpoint(old) is False
        partial = root / "epoch-001.ckpt.partial-1234"
        partial.write_bytes(b"tmp")
        assert rec.may_unlink_checkpoint(partial, owned_partials=(partial,)) is True
        assert rec.may_unlink_checkpoint(partial) is True  # name contains .partial
        try:
            rec.safe_unlink_checkpoint(old)
        except rec.RecipeRefused:
            pass
        else:
            raise AssertionError("safe_unlink must refuse an existing ckpt")
        rec.safe_unlink_checkpoint(partial, owned_partials=(partial,))
        assert not partial.exists()
        assert old.exists() and hashlib.sha256(old.read_bytes()).hexdigest() == digest

        # A recipe write of a new file leaves the existing one byte-identical.
        before = rec.snapshot_checkpoints(root)
        new = root / "epoch-001-new.ckpt"
        new.write_bytes(b"new-ckpt")
        rec.assert_old_checkpoints_untouched(before, rec.snapshot_checkpoints(root))
        # Tampering with the old file is detected.
        old.write_bytes(b"mutated")
        try:
            rec.assert_old_checkpoints_untouched(before, rec.snapshot_checkpoints(root))
        except rec.RecipeRefused:
            pass
        else:
            raise AssertionError("mutated existing ckpt must be detected")
    print("old checkpoints untouched (only owned .partial files may be unlinked) OK")


def check_stage_heads_wiring() -> None:
    rec = _recipe()
    s0 = rec.stage_train_kwargs("0")
    assert s0["kind"] == "ssl"
    assert "depth_hires" not in s0.get("heads", ())

    s1 = rec.stage_train_kwargs("1")
    assert s1["aux_depth"] is True and s1["aux_sem"] is True
    assert s1["heads"] == ()  # aux flags only; default --heads would also train lanes
    assert "depth_hires" not in (s1.get("heads") or ())  # #265 stays opt-in, not a recipe default

    s2 = rec.stage_train_kwargs("2")
    heads2 = set(s2["heads"])
    assert "lanes" in heads2 and "lane_gt" in heads2 and "road_edge" in heads2
    # Explicit --heads curbs is refused on main until a curb_valid labeler exists;
    # this PR does not invent that labeler.
    assert "curbs" not in heads2
    assert s2.get("aux_depth") is False

    s2b = rec.stage_train_kwargs("2b")
    heads2b = set(s2b["heads"])
    assert "signs" in heads2b and "stop_line" in heads2b and "ctl" in heads2b
    assert "lane_gt" in heads2b  # existing ctl refusal: ctl needs lane_gt

    s7 = rec.stage_train_kwargs("7")
    assert s7["joint"] is True
    # Joint peak is 0.1 x (spec proposal, already in lr_schedule.JOINT_PEAK_MULT).
    from python.train.lr_schedule import JOINT_PEAK_MULT, stage_lr_mults, stage_peak

    assert abs(stage_peak(1e-3, joint=True) - 1e-3 * JOINT_PEAK_MULT) < 1e-15
    mults = rec.stage_lr_mults_for("2")
    # After stage 0, stem x0.1; entering lane heads x1.0; already-trained aux x0.3 (proposal).
    assert mults["stem"] == 0.1
    assert any(v == 1.0 for k, v in mults.items() if k not in ("stem", "trunk"))

    # --stage filter keeps order and does not invent stages.
    assert rec.parse_stage_filter("2b") == ("2b",)
    assert rec.parse_stage_filter("0-2") == ("0", "1", "2")
    assert rec.parse_stage_filter("0,2b") == ("0", "2b")
    try:
        rec.parse_stage_filter("9")
    except rec.RecipeRefused:
        pass
    else:
        raise AssertionError("unknown --stage must be refused")
    print("stage heads / flags / --stage filter / no default depth_hires / curbs omitted OK")


def check_accum_only_when_batch_lt_4() -> None:
    rec = _recipe()
    # Spec proposal: used only if the auto-fit batch < 4 windows (floor 2, cap 8).
    assert rec.ACCUM_BATCH_LT == 4 and rec.ACCUM_FLOOR == 2 and rec.ACCUM_CAP == 8
    assert rec.resolve_accum(requested=0, batch=8) == 1
    assert rec.resolve_accum(requested=4, batch=8) == 1  # batch >= 4: not used
    assert rec.resolve_accum(requested=4, batch=2) == 4
    assert rec.resolve_accum(requested=1, batch=2) == rec.ACCUM_FLOOR  # floor 2
    assert rec.resolve_accum(requested=99, batch=2) == rec.ACCUM_CAP
    assert rec.resolve_accum(requested=0, batch=2) == 1  # not requested
    print("accum only when batch < 4 (floor 2, cap 8, proposal) OK")


def check_exit_criteria_pinned_in_manifest() -> None:
    rec = _recipe()
    cfg = rec.recipe_run_config(seed=0, ssl_pretrain=True)
    # Both exit rules are pinned before training (**spec proposal** process).
    assert cfg["exit_gain"] == rec.EXIT_GAIN
    assert cfg["exit_patience_evals"] == rec.EXIT_PATIENCE_EVALS
    assert cfg["stage_max_epochs"] == rec.STAGE_MAX_EPOCHS
    assert cfg["non_regression_frac"] == rec.NON_REGRESSION_FRAC
    assert cfg["recipe"] == "trainer2"
    for key in ("EXIT_GAIN", "EXIT_PATIENCE_EVALS", "STAGE_MAX_EPOCHS", "NON_REGRESSION_FRAC",
                "JOINT_HEAD_REGRESSION_FRAC", "RECIPE_DEFAULT_OFF", "ACCUM"):
        assert key in rec.PROPOSAL_FLAGS
    print("exit / non-regression / max-epochs pinned in run config and flagged PROPOSAL OK")


def check_proposal_defaults_unchanged() -> None:
    """Cross-module PROPOSAL defaults this PR must not decide or edit."""
    from python.train.trainer2_balance import WEIGHT_LR
    from python.train.trainer2_split import BAR_HELDOUT_MAP_FROM_HOLDOUT
    from python.train.stop_events import RED_TO_GREEN_PROCEED_IS_MISS
    from python.train.trainer2_bar import BACKSTOP_ACTIVATIONS_COUNT_AS_INTERVENTIONS, REQUIRE_UNLABELABLE_SLICES_FOR_2_0

    assert WEIGHT_LR == 0.025
    assert BAR_HELDOUT_MAP_FROM_HOLDOUT is True
    assert RED_TO_GREEN_PROCEED_IS_MISS is True
    assert BACKSTOP_ACTIVATIONS_COUNT_AS_INTERVENTIONS is True
    assert REQUIRE_UNLABELABLE_SLICES_FOR_2_0 is True
    print("cross-PR PROPOSAL defaults unchanged OK")


def check_recipe_off_is_bit_identical() -> None:
    torch = _torch()
    if torch is None:
        skip_check("check_recipe_off_is_bit_identical", "torch not installed (train_scene imports torch)")
        return
    pin = json.loads(DEFAULTS_PIN.read_text(encoding="utf-8"))
    assert pin["base"].startswith("8ce34d5"), pin["base"]
    got = json.loads(json.dumps(vars(_parser().parse_args([])), sort_keys=True, default=str))
    want = pin["defaults"]
    added = sorted(set(got) - set(want))
    removed = sorted(set(want) - set(got))
    changed = sorted(k for k in set(got) & set(want) if got[k] != want[k])
    assert not (removed or changed), f"pre-existing train_scene defaults drifted: removed={removed} changed={changed}"
    new = {k: got[k] for k in added}
    want_new = {"no_ssl_pretrain": False, "ssl_epochs": 1, "ssl_max_rows": 4000,
                "recipe": "", "stage": "", "accum": 0, "kd": False}
    assert new == want_new, new
    print(f"recipe off: {len(want)} pre-existing train_scene argparse defaults == main 8ce34d5 pin OK "
          f"({len(added)} newer flag(s) pinned exactly: {added})")


def check_does_not_invent_pr2c_or_touch_pr2b() -> None:
    rec_path = ROOT / "python" / "train" / "trainer2_recipe.py"
    scene = ROOT / "python" / "train" / "train_scene.py"
    text = rec_path.read_text(encoding="utf-8") + scene.read_text(encoding="utf-8")
    assert not (ROOT / "python" / "train" / "strip_tensor.py").exists()
    # Do not invent PR 2c or import Historian PR 2b. Mentions in comments/docstrings are fine.
    for needle in ("from python.train.strip_tensor", "import strip_tensor",
                   "from python.train.scene_ckpt_runtime", "import scene_ckpt_runtime"):
        assert needle not in text, needle
    print("PR 6 does not invent strip_tensor.py or import scene_ckpt_runtime OK")



# ---------------------------------------------------------------- r1 rework (Skeptic 2 on #269)

def check_scale_and_names_edges() -> None:
    rec = _recipe()
    for bad in (float("nan"), float("inf") * -1):
        try:
            rec.validate_live_export_recipe(recipe="trainer2", scale=bad, augment=False)
        except rec.RecipeRefused:
            pass
        else:
            raise AssertionError(f"scale {bad} must be refused")
    try:
        rec.validate_recipe_args(_parser().parse_args(["--recipe", "trainer2", "--no-augment", "--scale", "0"]))
    except rec.RecipeRefused as exc:
        assert "--scale" in str(exc)
    else:
        raise AssertionError("--scale 0 must be refused (not coerced to 1)")
    for off in ("", "off", "false", "0", None):
        rec.validate_live_export_recipe(recipe=off, scale=0.25, augment=True, kd=True)
    try:
        rec.validate_recipe_args(_parser().parse_args(["--recipe", "trainer2", "--no-augment", "--stage", "9"]))
    except rec.RecipeRefused:
        pass
    else:
        raise AssertionError("validate_recipe_args must reject an unknown --stage up front")
    assert rec.parse_stage_filter("2,0,2") == ("0", "2")
    assert rec.parse_stage_filter("") is None
    try:
        rec.parse_stage_filter("2-0")
    except rec.RecipeRefused as exc:
        assert "backwards" in str(exc)
    else:
        raise AssertionError("backwards --stage range must be refused")
    print("scale NaN/-inf/0 refused; off names pass; --stage dedupe/order/backwards OK")


def check_exit_boundaries_and_nonfinite() -> None:
    rec = _recipe()
    assert rec.STAGE_MAX_EPOCHS == 20  # spec proposal value, pinned
    assert rec.stage_exited("2", [100.0, 100.0, 100.0, 99.0], epochs=4) is True  # exactly 1% is not > 1%
    assert rec.stage_exited("2", [100.0, 100.0, 100.0, 98.9], epochs=4) is False  # 1.1%
    assert rec.stage_exited("2", [1.0, 0.9, 0.9, 0.9], epochs=4) is False  # 10% gain inside the 3-step window
    assert rec.stage_exited("2", [1.0, 1.0, float("nan"), 1.0], epochs=4) is False
    assert rec.stage_exited("2", [1.0, 1.0, 1.0, float("inf")], epochs=4) is False
    assert rec.metric_gain(-0.5, -0.4, metric="lane_r") > 0.19  # negative baseline uses |old|
    assert rec.metric_gain(0.0, 0.0, metric="lane_rms") == 0.0
    assert rec.metric_gain(0.0, -0.1, metric="lane_rms") == 1.0
    assert rec.metric_gain(0.0, 0.1, metric="lane_rms") == 0.0
    import math
    assert math.isnan(rec.metric_gain(1.0, float("nan"), metric="lane_rms"))
    for cur in ({"lane_rms": float("nan")}, {"lane_rms": float("inf")}, {}):
        try:
            rec.check_non_regression({"lane_rms": 1.0}, cur)
        except rec.CheckpointRejected:
            pass
        else:
            raise AssertionError(f"{cur} must reject (not measured / non-finite)")
    print("exit: exact 1% boundary, 3-step window, NaN/inf never exit; non-regression rejects NaN and missing metrics OK")


def check_entry_and_plan_head_data() -> None:
    rec = _recipe()
    assert rec.entry_allowed("2", prev_exited=True, heads_exist=False, data_exist=True)[0] is False
    assert rec.entry_allowed("2", prev_exited=True, heads_exist=True, data_exist=False)[0] is False
    on = dict.fromkeys(rec.STAGE_ORDER, True)
    no_head = dict(on, **{"2": False})
    no_data = dict(on, **{"2": False})
    p1 = {s.id: s for s in rec.plan_curriculum(heads_exist=no_head, data_exist=on)}
    assert p1["2"].action == "skip" and "head" in p1["2"].reason, p1["2"]
    p2 = {s.id: s for s in rec.plan_curriculum(heads_exist=on, data_exist=no_data)}
    assert p2["2"].action == "skip" and "data" in p2["2"].reason, p2["2"]
    print("entry and plan refuse a stage whose head / data is not on main OK")


def _res(rec, metrics, *, evals=(1.0, 1.0, 1.0, 1.0), ckpt="c.ckpt", epochs=4):
    return rec.StageResult(ckpt=ckpt, metrics=dict(metrics), evals=list(evals), epochs=epochs)


def check_runner_baseline_chain_joint_and_exit() -> None:
    rec = _recipe()
    on = dict.fromkeys(rec.STAGE_ORDER, True)
    # --stage 2 alone: refused without a baseline / previous checkpoint
    for kw in ({}, {"earlier_metrics": {"lane_rms": 1.0}}, {"start_ckpt": "s1.ckpt"}):
        try:
            rec.run_curriculum(lambda s, p: _res(rec, {"lane_rms": 1.0}), stages=("2",), heads_exist=on, data_exist=on, **kw)
        except rec.RecipeRefused as exc:
            assert "mid-curriculum" in str(exc), exc
        else:
            raise AssertionError(f"--stage 2 with {kw} must be refused")
    # with both: runs, the train_fn sees the start checkpoint, earlier metrics are the baseline
    seen = []
    out = rec.run_curriculum(lambda s, p: (seen.append((s.id, p)), _res(rec, {"lane_rms": 1.04}, ckpt="s2.ckpt"))[1],
                             stages=("2",), heads_exist=on, data_exist=on,
                             earlier_metrics={"lane_rms": 1.0}, start_ckpt="s1.ckpt")
    assert seen == [("2", "s1.ckpt")] and out.last_ckpt == "s2.ckpt" and out.trained == ["2"], (seen, out)
    try:
        rec.run_curriculum(lambda s, p: _res(rec, {"lane_rms": 1.06}), stages=("2",), heads_exist=on, data_exist=on,
                           earlier_metrics={"lane_rms": 1.0}, start_ckpt="s1.ckpt")
    except rec.CheckpointRejected:
        pass
    else:
        raise AssertionError("earlier_metrics must be the non-regression baseline (6% worse rejects)")
    # ckpt chain across stages: each stage gets the previous stage's checkpoint
    chain = []
    metrics = {"lane_rms": 1.0}
    rec.run_curriculum(lambda s, p: (chain.append((s.id, p)), _res(rec, metrics, ckpt=f"s{s.id}.ckpt"))[1],
                       stages=("0", "1", "2"), heads_exist=on, data_exist=on)
    assert chain == [("0", None), ("1", "s0.ckpt"), ("2", "s1.ckpt")], chain
    # joint stage 7: 2% cap (spec proposal), not 5%
    def joint(worse):
        def fn(s, p):
            return _res(rec, {"lane_rms": 1.0 + (worse if s.id == "7" else 0.0)})
        return rec.run_curriculum(fn, stages=("0", "7"), heads_exist=on, data_exist=on)
    joint(0.015)
    try:
        joint(0.025)
    except rec.CheckpointRejected as exc:
        assert "2.00%" in str(exc), exc
    else:
        raise AssertionError("stage 7 2.5% worse must hit the 2% joint cap")
    # a 4% regression is fine at a non-joint stage
    rec.run_curriculum(lambda s, p: _res(rec, {"lane_rms": 1.0 if s.id == "0" else 1.04}), stages=("0", "2"),
                       heads_exist=on, data_exist=on)
    # the last stage must exit too
    try:
        rec.run_curriculum(lambda s, p: _res(rec, {"lane_rms": 1.0}, evals=(1.0, 0.9), epochs=2), stages=("0",),
                           heads_exist=on, data_exist=on)
    except rec.RecipeRefused as exc:
        assert "did not exit" in str(exc)
    else:
        raise AssertionError("a final stage that did not exit must be refused")
    # MetricNotMeasured stops the run after training that stage
    calls = []

    def unmeasured(s, p):
        calls.append(s.id)
        raise rec.MetricNotMeasured("not measured")
    try:
        rec.run_curriculum(unmeasured, stages=("0", "1"), heads_exist=on, data_exist=on)
    except rec.MetricNotMeasured:
        pass
    else:
        raise AssertionError("MetricNotMeasured must stop the run")
    assert calls == ["0"], calls
    print("runner: --stage needs baseline+ckpt, ckpt chain, earlier baseline, joint 2% cap, last stage must exit, unmeasured stops OK")


def check_stage_result_from_stats_is_real() -> None:
    rec = _recipe()
    for sid in ("0", "1", "2b"):
        try:
            rec.stage_result_from_stats(sid, {"holdout_evals": [{"lane_rms": 1.0}] * 4})
        except rec.MetricNotMeasured as exc:
            assert "not measured offline" in str(exc)
        else:
            raise AssertionError(f"stage {sid} has no offline exit metric; must not pass")
    for bad in ({}, {"holdout_evals": []}, {"holdout_evals": [{"lane_rms": None}]},
                {"holdout_evals": [{"lane_rms": 1.0}, {"lane_rms": float("nan")}]}):
        try:
            rec.stage_result_from_stats("2", bad)
        except rec.MetricNotMeasured:
            pass
        else:
            raise AssertionError(f"{bad} must not yield a stage result")
    res = rec.stage_result_from_stats("2", {"holdout_evals": [{"lane_rms": 1.2, "lane_r": 0.5}, {"lane_rms": 1.1, "lane_r": 0.6}],
                                            "epochs_run": 2, "epoch_ckpts": ["a.ckpt", "b.ckpt"]})
    assert res.evals == [1.2, 1.1] and res.metrics == {"lane_rms": 1.1, "lane_r": 0.6} and res.ckpt == "b.ckpt" and res.epochs == 2, res
    print("stage results come from real holdout evals; stages 0/1/2b refuse (not measured offline in PR 6) OK")


def check_deploy_real_export() -> None:
    rec = _recipe()
    try:
        rec.student_export_fits_deploy(preset="small", student_param_count=201, exported_param_count=200)
    except rec.RecipeRefused:
        pass
    else:
        raise AssertionError("student > exported must also fail")
    torch = _torch()
    if torch is None:
        skip_check("check_deploy_real_export", "torch not installed")
        return
    try:
        import onnx  # noqa: F401
    except ImportError:
        skip_check("check_deploy_real_export", "onnx not installed (CI installs it)")
        return
    from python.train.model_size import scene_net_kwargs
    from python.train.scene_net import SceneNet, export_scene_onnx

    with tempfile.TemporaryDirectory() as raw:
        for preset in ("tiny", "small"):
            for perturb in (False, True):
                net = SceneNet(cams=("main",), **scene_net_kwargs(preset))
                if perturb:
                    with torch.no_grad():
                        for prm in net.parameters():
                            prm.add_(torch.randn_like(prm) * 1e-3)
                path = export_scene_onnx(Path(raw) / f"{preset}{int(perturb)}.onnx", net, cams=("main",))
                out = rec.check_export_fits_deploy(net, path, preset=preset)
                assert out["ok"] and out["params"] == rec.onnx_initializer_count(path), out
        # training-only aux heads are not in the export and not in the student count
        aux = SceneNet(cams=("main",), aux_sensors=True, **scene_net_kwargs("small"))
        assert hasattr(aux, "aux_depth_head") and hasattr(aux, "aux_sem_head")
        with torch.no_grad():
            for prm in aux.parameters():
                prm.add_(torch.randn_like(prm) * 1e-3)
        aux_path = export_scene_onnx(Path(raw) / "aux.onnx", aux, cams=("main",))
        out = rec.check_export_fits_deploy(aux, aux_path, preset="small")
        assert out["params"] == rec.onnx_initializer_count(aux_path) < sum(p.numel() for p in aux.parameters()), out
        small = SceneNet(cams=("main",), **scene_net_kwargs("small"))
        tiny_path = Path(raw) / "tiny0.onnx"
        try:
            rec.check_export_fits_deploy(small, tiny_path, preset="small")
        except rec.RecipeRefused as exc:
            assert "param count" in str(exc)
        else:
            raise AssertionError("a small net against a tiny export must fail")
        try:
            rec.check_export_fits_deploy(small, tiny_path, preset="medium")
        except rec.RecipeRefused:
            pass
        else:
            raise AssertionError("medium must fail the deploy check")
        import onnx
        from onnx import TensorProto, helper

        empty = Path(raw) / "empty.onnx"
        g = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
        onnx.save(helper.make_model(g), str(empty))
        try:
            rec.onnx_initializer_count(empty)
        except rec.RecipeRefused as exc:
            assert "no initializers" in str(exc)
        else:
            raise AssertionError("an export with no initializers must refuse")
        bad = Path(raw) / "bad.onnx"
        bad.write_bytes(b"not an onnx file")
        try:
            rec.onnx_initializer_count(bad)
        except rec.RecipeRefused:
            pass
        else:
            raise AssertionError("an unreadable export must refuse, not pass")
        saved = sys.modules.get("onnx")
        sys.modules["onnx"] = None  # import onnx -> ImportError
        try:
            rec.onnx_initializer_count(tiny_path)
        except rec.RecipeRefused as exc:
            assert "onnx package" in str(exc)
        else:
            raise AssertionError("missing onnx must refuse, not pass")
        finally:
            sys.modules["onnx"] = saved
    print("deploy check on real tiny/small exports (fresh, trained-like, with training-only aux heads), wrong net / medium / unreadable / no onnx refused OK")


def check_retention_real_paths() -> None:
    rec = _recipe()
    from python.train.epoch_ckpt import EpochCkptRefused, write_epoch_ckpt

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        old = root / "old.ckpt"
        old.write_bytes(b"keep")
        other_partial = root / "x.ckpt.partial-9"
        other_partial.write_bytes(b"p")
        assert rec.may_unlink_checkpoint(old, owned_partials=(old,)) is False  # owned but not a .partial
        assert rec.may_unlink_checkpoint(other_partial, owned_partials=(root / "y.partial",)) is False
        onnx = root / "m.onnx"
        onnx.write_bytes(b"o")
        (root / "run.ckpt_meta.json").write_text("{}")  # ckpt-named sidecar, other suffix
        (root / "notes.txt").write_text("n")
        snap = rec.snapshot_checkpoints(root)
        assert "m.onnx" in snap and "old.ckpt" in snap and "x.ckpt.partial-9" not in snap, snap
        assert "run.ckpt_meta.json" in snap and "notes.txt" not in snap, snap
        old.unlink()
        try:
            rec.assert_old_checkpoints_untouched(snap, rec.snapshot_checkpoints(root))
        except rec.RecipeRefused as exc:
            assert "removed" in str(exc)
        else:
            raise AssertionError("a deleted checkpoint must be detected")
        old.write_bytes(b"keep")
        # the real create-only writer refuses an existing checkpoint and leaves its bytes alone
        try:
            write_epoch_ckpt(old, {"w": np.zeros(2, np.float32)}, {"epoch": 0, "step": 0})
        except EpochCkptRefused as exc:
            assert "exists" in str(exc)
        else:
            raise AssertionError("write_epoch_ckpt must be create-only")
        assert old.read_bytes() == b"keep"
        from python.train.epoch_ckpt import unlink_own_partial
        try:
            unlink_own_partial(old, owned_partials=(old,))
        except EpochCkptRefused:
            pass
        else:
            raise AssertionError("the real writer's unlink must refuse a non-partial checkpoint")
        assert old.read_bytes() == b"keep"
        # the runner re-checks after each stage: a stage that deletes an old checkpoint stops the run
        on = dict.fromkeys(rec.STAGE_ORDER, True)

        def vandal(stage, prev):
            old.unlink()
            return _res(rec, {"lane_rms": 1.0})
        try:
            rec.run_curriculum(vandal, stages=("0",), heads_exist=on, data_exist=on, ckpt_root=root)
        except rec.RecipeRefused as exc:
            assert "old.ckpt" in str(exc)
        else:
            raise AssertionError("runner must detect a removed checkpoint")
    src = (ROOT / "python" / "train" / "epoch_ckpt.py").read_text(encoding="utf-8")
    assert "unlink_own_partial(partial, owned_partials=(partial,))" in src
    print("retention: owned+partial rule, snapshot onnx/partial, removed detected, real create-only write, runner re-check OK")


def check_more_pins() -> None:
    rec = _recipe()
    assert rec.resolve_accum(requested=4, batch=4) == 1
    assert rec.resolve_accum(requested=4, batch=3) == 4
    assert rec.stage_train_kwargs("7")["heads"] == ("lanes", "objects", "signs", "lane_gt", "road_edge", "stop_line", "ctl")
    cfg = rec.recipe_run_config()
    assert cfg["joint_head_regression_frac"] == 0.02 and cfg["reality_view_built_and_integrated"] is False
    assert rec.stage_lr_mults_for("0")["stem"] == 1.0  # stem trains at x1 in stage 0
    from python.train.scene_net import HEAD_NAMES, head_names
    assert head_names(()) == frozenset() and head_names([]) == frozenset()
    assert head_names(None) == frozenset(HEAD_NAMES)
    # BEV (stages 3/4) is not on this checkout: its head must not be reported as existing
    assert not (ROOT / "python" / "train" / "bev_lss.py").exists()
    assert rec.default_heads_exist()["3"] is False and rec.default_heads_exist()["4"] is False
    # the preset-known and VRAM guards are independent of the deploy-size list (defence in depth)
    from python.train.model_size import PRESETS
    assert float(PRESETS["medium"].est_vram_gb) > rec.LIVE_INFER_VRAM_GB
    saved = rec.DEPLOY_PRESETS
    rec.DEPLOY_PRESETS = frozenset(saved | {"medium", "no_such_preset"})
    try:
        for name, word in (("medium", "GB"), ("no_such_preset", "unknown")):
            try:
                rec.student_export_fits_deploy(preset=name, student_param_count=1, exported_param_count=1)
            except rec.RecipeRefused as exc:
                assert word in str(exc), exc
            else:
                raise AssertionError(f"{name} must be refused even if listed as a deploy size")
    finally:
        rec.DEPLOY_PRESETS = saved
    print("accum batch==4 boundary, stage 7 heads, cfg joint/rv, stage 0 stem lr, head_names(()) == none, "
          "no BEV head, VRAM / unknown-preset guards independent OK")


def check_balanced_loss_keeps_extras() -> None:
    torch = _torch()
    if torch is None:
        skip_check("check_balanced_loss_keeps_extras", "torch not installed")
        return
    from python.train.train_scene import balanced_loss
    from python.train.trainer2_balance import GradNorm, HeadRamp

    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    const = torch.tensor(0.0)  # a disabled head: no graph
    terms = {"lane_gt": a * 1.0, "aux_depth": b * 1.0, "points": const}
    total = terms["lane_gt"] + 0.5 * terms["aux_depth"] + const
    bal = GradNorm(list(terms))
    ramp = HeadRamp({k: -2000 for k in terms})
    loss, names = balanced_loss(total, terms, bal, ramp, step=0, coeffs={"aux_depth": 0.5})
    assert sorted(names) == ["aux_depth", "lane_gt"], names
    assert abs(float(loss) - float(total)) < 1e-6  # all weights 1, ramp 1: same loss
    loss.backward()
    assert float(b.grad) == 0.5 and float(a.grad) == 1.0, (a.grad, b.grad)
    bal.weights["aux_depth"] = 2.0
    a.grad = b.grad = None
    terms = {"lane_gt": a * 1.0, "aux_depth": b * 1.0, "points": const}
    total = terms["lane_gt"] + 0.5 * terms["aux_depth"]
    loss, _ = balanced_loss(total, terms, bal, ramp, step=0, coeffs={"aux_depth": 0.5})
    loss.backward()
    assert abs(float(b.grad) - 1.0) < 1e-6, b.grad  # w=2 x c=0.5
    # a zero-coefficient term is not balanced (it is not in the loss at all)
    terms = {"lane_gt": a * 1.0, "aux_depth": b * 1.0}
    _loss, names = balanced_loss(terms["lane_gt"], terms, bal, ramp, step=0, coeffs={"aux_depth": 0.0})
    assert names == ["lane_gt"], names
    # the ramp applies: a head that enters at step 100 contributes nothing at step 0
    late = HeadRamp({"lane_gt": -2000, "aux_depth": 100})
    a.grad = b.grad = None
    terms = {"lane_gt": a * 1.0, "aux_depth": b * 1.0}
    total = terms["lane_gt"] + 0.5 * terms["aux_depth"]
    loss, _ = balanced_loss(total, terms, bal, late, step=0, coeffs={"aux_depth": 0.5})
    loss.backward()
    assert abs(float(b.grad)) < 1e-9 and float(a.grad) == 1.0, (a.grad, b.grad)
    print("balanced_loss: extras (aux_depth) keep their gradient and coefficient; no-graph / zero-coef terms skipped; ramp applied OK")


def _fixture(base):
    import test_trainer2_depth_hires as T

    return T._hires_recording(base, 70)


def _grad_spy(torch, train_scene, names):
    seen = []
    real = torch.nn.utils.clip_grad_norm_
    holder = {}
    real_bwd = train_scene._backward_and_maybe_step

    def wrap(net, *a, **k):
        holder["net"] = net
        return real_bwd(net, *a, **k)

    def spy(params, *a, **k):
        net = holder.get("net")
        if net is not None:
            seen.append({n: (0.0 if p.grad is None else float(p.grad.abs().sum())) for n, p in net.named_parameters()
                         if any(n.startswith(x) for x in names)})
        return real(params, *a, **k)
    return seen, (real, real_bwd), spy, wrap


def check_recipe_stages_train_on_fixture() -> None:
    """train_directory with the recipe on, every supervised stage, on the repo fixture (CPU). r1 crashed here."""
    torch = _torch()
    if torch is None:
        skip_check("check_recipe_stages_train_on_fixture", "torch not installed")
        return
    try:
        import cv2  # noqa: F401
    except ImportError:
        skip_check("check_recipe_stages_train_on_fixture", "cv2 not installed")
        return
    import contextlib
    import io

    from python.train import train_scene

    rec = _recipe()
    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        strips = _fixture(base)
        for sid in ("1", "2", "2b", "7"):
            kw = rec.stage_train_kwargs(sid)
            seen, real, spy, wrap = _grad_spy(torch, train_scene, ("aux_depth_head", "lane_gt_head"))
            torch.nn.utils.clip_grad_norm_ = spy
            train_scene._backward_and_maybe_step = wrap
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    stats = train_scene.train_directory(
                        strips, epochs=1, lr=1e-3, capability=None, export_path=None, free_vram_bytes=0,
                        free_ram_bytes=8_000_000_000, resume=False, open_panel=False, autostart=True, cams=("main",),
                        heads=kw["heads"], aux_depth=kw["aux_depth"], aux_sem=kw["aux_sem"], aux_depth_weight=1.0,
                        aux_sem_weight=1.0 if kw["aux_sem"] else 0.0, augment=False, workers=0, min_valid=0.0,
                        holdout_blocks=1, scale=1.0, status_root=base / f"st{sid}", ckpt_dir=base / f"ck{sid}",
                        exact_epochs=1, recipe="trainer2", recipe_stage=sid)
            except Exception as exc:
                raise AssertionError(f"stage {sid} crashed: {type(exc).__name__}: {exc}\n{buf.getvalue()[-2000:]}") from exc
            finally:
                torch.nn.utils.clip_grad_norm_, train_scene._backward_and_maybe_step = real
            assert int(stats["windows"]) > 0 and seen, (sid, stats.get("windows"), len(seen))
            if kw["aux_depth"]:
                assert any(g.get("aux_depth_head.weight", 0.0) > 0.0 for g in seen), (sid, seen)
            if "lane_gt" in kw["heads"]:
                assert any(g.get("lane_gt_head.weight", 0.0) > 0.0 for g in seen), (sid, seen)
            if sid == "1":
                assert all(g.get("lane_gt_head.weight", 0.0) == 0.0 for g in seen), "stage 1 heads=() must not train lane_gt"
    print("recipe on: stages 1 / 2 / 2b / 7 train on the fixture; aux_depth and lane_gt heads get gradient; stage 1 trains no lane head OK")


def _prereg(rec, ck, strips, *, seed=0, ssl=False):
    from python.train.trainer2_split import write_split_manifest

    split = rec.build_recipe_split((strips,), strips, holdout_blocks=1)
    sp = write_split_manifest(ck.parent / f"{ck.name}_split.json", split)
    rec.preregister_recipe_run(ck, split_path=sp, git_sha="0" * 40, seed=seed, ssl_pretrain=ssl,
                               registry=ck.parent / f"{ck.name}_reg")
    return split


def check_recipe_manifest_required() -> None:
    torch = _torch()
    if torch is None:
        skip_check("check_recipe_manifest_required", "torch not installed")
        return
    try:
        import cv2  # noqa: F401
    except ImportError:
        skip_check("check_recipe_manifest_required", "cv2 not installed")
        return
    rec = _recipe()
    from python.train.trainer2_split import SplitError

    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        strips = _fixture(base)
        train, hold = rec.legacy_split_rows((strips,), strips, holdout_blocks=1)
        assert train and hold and all(r["source"] for r in train + hold)
        try:
            rec.require_recipe_manifest(None, seed=0, ssl_pretrain=False, train_rows=train, holdout_rows=hold)
        except rec.RecipeRefused as exc:
            assert "--ckpt-dir" in str(exc)
        else:
            raise AssertionError("no --ckpt-dir must refuse")
        ck = base / "ck"
        ck.mkdir()
        try:
            rec.require_recipe_manifest(ck, seed=0, ssl_pretrain=False, train_rows=train, holdout_rows=hold)
        except SplitError:
            pass
        else:
            raise AssertionError("a missing run manifest must refuse (SplitError)")
        _prereg(rec, ck, strips)
        rec.require_recipe_manifest(ck, seed=0, ssl_pretrain=False, train_rows=train, holdout_rows=hold)
        for kw, word in (({"seed": 1}, "seed"), ({"ssl_pretrain": True}, "ssl")):
            args = dict(seed=0, ssl_pretrain=False, train_rows=train, holdout_rows=hold)
            args.update(kw)
            try:
                rec.require_recipe_manifest(ck, **args)
            except rec.RecipeRefused as exc:
                assert word in str(exc).lower(), exc
            else:
                raise AssertionError(f"{kw} must refuse")
        stray = dict(train[0], i=99999)
        try:
            rec.require_recipe_manifest(ck, seed=0, ssl_pretrain=False, train_rows=train + [stray], holdout_rows=hold)
        except SplitError:
            pass
        else:
            raise AssertionError("a train row outside the manifest must refuse")
        try:
            rec.require_recipe_manifest(ck, seed=0, ssl_pretrain=False, train_rows=train, holdout_rows=hold + [stray])
        except rec.RecipeRefused as exc:
            assert "holdout" in str(exc)
        else:
            raise AssertionError("a holdout row outside the manifest must refuse")
        import json as _json
        runm = ck / "run_manifest.json"
        body = _json.loads(runm.read_text())
        body["config"]["exit_gain"] = 0.5
        from python.train.trainer2_split import manifest_sha256
        body["config_sha256"] = "x"
        body["sha256"] = manifest_sha256(body)
        runm.write_text(_json.dumps(body))
        try:
            rec.require_recipe_manifest(ck, seed=0, ssl_pretrain=False, train_rows=train, holdout_rows=hold)
        except rec.RecipeRefused as exc:
            assert "exit_gain" in str(exc)
        else:
            raise AssertionError("a manifest pinning other exit values must refuse")
    print("recipe manifest: needs --ckpt-dir, a pre-registered run, matching pins/seed/SSL, train and holdout rows in the manifest OK")


def check_run_recipe_end_to_end() -> None:
    """Real ``_run_trainer2_recipe`` on the fixture: manifest, stage 1 -> ckpt -> stage 2 init, export, deploy, gates."""
    torch = _torch()
    if torch is None:
        skip_check("check_run_recipe_end_to_end", "torch not installed")
        return
    try:
        import cv2  # noqa: F401
        import onnx  # noqa: F401
    except ImportError:
        skip_check("check_run_recipe_end_to_end", "cv2 / onnx not installed (CI installs both)")
        return
    import contextlib
    import io

    from python.train import train_scene

    rec = _recipe()
    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        strips = _fixture(base)
        ck = base / "ck"
        ck.mkdir()
        keep = ck / "pre-existing.ckpt"
        keep.write_bytes(b"older run")
        _prereg(rec, ck, strips)
        argv = ["--recipe", "trainer2", "--no-augment", "--cams", "main", "--epochs", "1", "--ckpt-dir", str(ck),
                "--min-valid", "0", "--workers", "0", "--no-panel"]

        def run(stage, **kw):
            args = _parser().parse_args(argv + ["--stage", stage])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                try:
                    res = train_scene._run_trainer2_recipe(
                        args, strip_roots=(strips,), holdout_root=strips, cam_names=("main",), net_kw={},
                        preset_name="small", train_precision="auto", plan_precision=None, frame_cache=None,
                        aux_cache=None, pause_dir=None, start_now=True, open_panel=False, **kw)
                    err = None
                except (rec.RecipeRefused, rec.CheckpointRejected) as exc:
                    res, err = None, exc
            return res, err, out.getvalue()

        # stage 1 trains, then refuses to claim an exit it cannot measure
        from python.train.epoch_ckpt import write_epoch_ckpt
        from python.train.model_size import scene_net_kwargs
        from python.train.scene_net import SceneNet

        seed_net = SceneNet(cams=("main",), **scene_net_kwargs("small"))
        seed = base / "stage0_seed.ckpt"  # stands in for a stage-0 checkpoint (outside --ckpt-dir)
        write_epoch_ckpt(seed, {k: v.detach().numpy() for k, v in seed_net.state_dict().items()}, {"epoch": 0, "step": 0},
                         min_free_bytes=0)
        res, err, text = run("1", export_path=None, start_ckpt=str(seed), earlier_metrics={"ssl_pretext": 1.0})
        assert isinstance(err, rec.MetricNotMeasured) and "depth AbsRel" in str(err), (err, text[-1500:])
        s1 = sorted(ck.rglob("*.ckpt_ep*.npz"))
        s1 = [p for p in s1 if p.name != keep.name]
        assert s1, "stage 1 wrote no epoch checkpoint\n" + text[-3000:]
        # stage 2 from stage 1's checkpoint: loads it, exports, deploy-checks the real ONNX, scores real rms
        export = base / "student.onnx"
        res, err, text = run("2", export_path=export, start_ckpt=str(s1[-1]), earlier_metrics={"lane_rms": 1e9})
        assert "init from previous stage" in text, text[-2000:]
        assert "deploy check {'ok': True, 'preset': 'small', 'params': " in text, (err, text[-800:])
        assert export.is_file()
        # the recipe owns the stage exit: <= 1% gain over 3 evals or the 20-epoch cap, scored on real holdout rms
        n_evals = text.count("[GVD] val step=")
        assert 1 <= n_evals <= rec.STAGE_MAX_EPOCHS, n_evals
        assert "holdout stopped" not in text  # the legacy 2% plateau stop is off under the recipe
        assert "recipe stage 2: stage exit" in text, text[-1500:]
        assert err is None, (err, text[-1500:])
        s2 = sorted(ck.rglob("*.ckpt_ep*.npz"), key=lambda q: q.stat().st_mtime)
        assert str(res.last_ckpt) == str(s2[-1]) and res.trained == ["2"], (res, s2[-1])
        # a mid-curriculum start without a baseline is refused
        res, err, text = run("2", export_path=None)
        assert err is not None and "mid-curriculum" in str(err), err
        assert keep.read_bytes() == b"older run"
    print("_run_trainer2_recipe end to end: stage 1 trains then refuses (unmeasured), stage 2 inits from it, recipe-owned exit on real rms, exports, deploy OK, last ckpt chained; old ckpt untouched OK")


def check_main_recipe_without_manifest_exits_2() -> None:
    torch = _torch()
    if torch is None:
        skip_check("check_main_recipe_without_manifest_exits_2", "torch not installed")
        return
    try:
        import cv2  # noqa: F401
    except ImportError:
        skip_check("check_main_recipe_without_manifest_exits_2", "cv2 not installed")
        return
    import contextlib
    import io
    import os

    import python.data.data_root as data_root
    from python.train import train_scene

    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        strips = _fixture(base)
        ck = base / "ck"
        ck.mkdir()
        saved = (data_root.free_bytes, os.environ.get("GVD_DATA_DIR"))
        data_root.free_bytes = lambda _p: 500 * 1024 ** 3
        os.environ["GVD_DATA_DIR"] = str(base)
        out = io.StringIO()
        code = None
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                train_scene.main(["--strips", str(strips), "--recipe", "trainer2", "--no-augment", "--cams", "main",
                                  "--no-panel", "--no-resume", "--ckpt-dir", str(ck)])
        except SystemExit as exc:
            code = exc.code
        finally:
            data_root.free_bytes = saved[0]
            if saved[1] is None:
                os.environ.pop("GVD_DATA_DIR", None)
            else:
                os.environ["GVD_DATA_DIR"] = saved[1]
        text = out.getvalue()
        assert code == 2, (code, text[-1500:])
        assert "Traceback" not in text and "run manifest" in text, text[-1500:]
    print("main --recipe trainer2 without a pre-registered manifest: clean exit 2 OK")

def main() -> int:
    checks = (
        check_recipe_default_off,
        check_recipe_refuses_scale_lt_1,
        check_recipe_refuses_augment,
        check_kd_refused_in_pr6,
        check_unknown_recipe_refused,
        check_curriculum_order_and_gates,
        check_student_export_fits_deploy,
        check_old_checkpoints_untouched,
        check_stage_heads_wiring,
        check_accum_only_when_batch_lt_4,
        check_exit_criteria_pinned_in_manifest,
        check_proposal_defaults_unchanged,
        check_recipe_off_is_bit_identical,
        check_does_not_invent_pr2c_or_touch_pr2b,
        check_scale_and_names_edges,
        check_exit_boundaries_and_nonfinite,
        check_entry_and_plan_head_data,
        check_runner_baseline_chain_joint_and_exit,
        check_stage_result_from_stats_is_real,
        check_deploy_real_export,
        check_retention_real_paths,
        check_more_pins,
        check_balanced_loss_keeps_extras,
        check_recipe_stages_train_on_fixture,
        check_recipe_manifest_required,
        check_run_recipe_end_to_end,
        check_main_recipe_without_manifest_exits_2,
    )
    failed = 0
    for check in checks:
        try:
            check()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {check.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
