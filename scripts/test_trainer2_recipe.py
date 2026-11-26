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

    # A zero baseline (a stage that missed nothing) that gets worse is rejected whatever its size: the 5% rule is
    # relative to |old| and would divide by zero; improving or holding at zero is fine.
    rec.check_non_regression({"missed_stop": 0.0, "false_brake": 0.0}, {"missed_stop": 0.0, "false_brake": 0.0})
    rec.check_non_regression({"missed_stop": 2.0}, {"missed_stop": 1.0})
    rec.check_non_regression({"missed_stop": 1.0}, {"missed_stop": 1.04})  # 4% worse, inside the 5%
    for earlier, now, what in (({"missed_stop": 0.0}, {"missed_stop": 1.0}, "zero baseline"),
                               ({"false_brake": 0.0}, {"false_brake": 0.001}, "zero baseline"),
                               ({"lane_r": 0.0}, {"lane_r": -0.1}, "zero baseline"),  # higher is better: below zero is worse
                               ({"missed_stop": 1.0}, {"missed_stop": 1.06}, "6")):
        try:
            rec.check_non_regression(earlier, now)
        except rec.CheckpointRejected as exc:
            assert what in str(exc), (earlier, now, exc)
        else:
            raise AssertionError(f"{earlier} -> {now} must be rejected")
    rec.check_non_regression({"lane_r": 0.0}, {"lane_r": 0.3})  # higher-is-better metric rising from zero is not worse
    assert rec._worse_frac(0.0, 2.0, metric="missed_stop") == float("inf")
    assert rec._worse_frac(0.0, 0.0, metric="missed_stop") == 0.0
    assert rec._worse_frac(0.0, 2.0, metric="lane_r") < 0.0 and rec._worse_frac(0.0, -2.0, metric="lane_r") == float("inf")
    assert abs(rec._worse_frac(2.0, 3.0, metric="missed_stop") - 0.5) < 1e-12

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


def _stats(evals, ckpts=("a.ckpt", "b.ckpt")):
    return {"holdout_evals": list(evals), "epochs_run": len(evals), "epoch_ckpts": list(ckpts)}


def check_stage_result_from_stats_is_real() -> None:
    rec = _recipe()
    nan = float("nan")
    # stage 0: the SSL run's per-epoch pretext losses, never a placeholder
    losses = [0.9, 0.8, 0.79, 0.79, 0.79]
    res0 = rec.stage_result_from_stats("0", {"ssl": {"on": True, "losses": losses}})
    assert res0.metrics == {"ssl_pretext": 0.79} and res0.evals == losses and res0.epochs == 5 and res0.ckpt is None, res0
    assert res0.evals_by == {"ssl_pretext": losses}
    for bad in ({}, {"ssl": None}, {"ssl": {"on": False, "why": "off"}}, {"ssl": {"losses": []}}, {"ssl": {"losses": None}},
                {"ssl": {"losses": [0.5, nan]}}, {"ssl": {"losses": [0.5, float("inf")]}}):
        try:
            rec.stage_result_from_stats("0", bad)
        except rec.MetricNotMeasured:
            pass
        else:
            raise AssertionError(f"{bad} must not yield a stage-0 result")
    # stage 1: depth AbsRel and sem mIoU on every eval; its (untrained) lane score is not carried
    ev1 = [{"depth_absrel": 0.9, "sem_miou": 0.2, "lane_rms": 3.0, "lane_r": 0.1},
           {"depth_absrel": 0.8, "sem_miou": 0.3, "lane_rms": 3.0, "lane_r": 0.1}]
    res1 = rec.stage_result_from_stats("1", _stats(ev1))
    assert res1.metrics == {"depth_absrel": 0.8, "sem_miou": 0.3}, res1.metrics
    assert res1.evals_by == {"depth_absrel": [0.9, 0.8], "sem_miou": [0.2, 0.3]} and res1.evals == [0.9, 0.8], res1
    assert res1.ckpt == "b.ckpt" and res1.epochs == 2
    for bad in ([], [{"depth_absrel": 0.9, "sem_miou": None}], [{"depth_absrel": nan, "sem_miou": 0.3}],
                [{"depth_absrel": 0.9, "sem_miou": 0.3}, {"depth_absrel": 0.9}], [{"lane_rms": 1.0}] * 4):
        try:
            rec.stage_result_from_stats("1", _stats(bad))
        except rec.MetricNotMeasured as exc:
            assert "not claiming exit" in str(exc), exc
        else:
            raise AssertionError(f"{bad} must not yield a stage-1 result")
    # stage 2b: the three stop metrics from the holdout pass (python/train/stop_pred.py), every one finite on every eval
    ok2b = {"lane_rms": 0.5, "lane_r": 0.9, "depth_absrel": 0.4, "sem_miou": 0.3, "missed_stop": 2.0, "missed_light": 1.0,
            "false_brake": 0.75}
    res = rec.stage_result_from_stats("2b", _stats([dict(ok2b, missed_stop=4.0, missed_light=3.0, false_brake=2.0),
                                                    dict(ok2b, missed_stop=3.0, missed_light=2.0, false_brake=1.0), ok2b]))
    assert res.metrics == ok2b, res.metrics  # the LAST eval, with the lane and aux metrics it keeps measuring
    assert res.evals_by == {"missed_stop": [4.0, 3.0, 2.0], "missed_light": [3.0, 2.0, 1.0], "false_brake": [2.0, 1.0, 0.75]}, res.evals_by
    assert res.evals == [4.0, 3.0, 2.0] and res.epochs == 3
    for key in ("missed_stop", "missed_light", "false_brake"):
        for bad in (None, nan, float("inf")):
            try:
                rec.stage_result_from_stats("2b", _stats([ok2b, dict(ok2b, **{key: bad})]))
            except rec.MetricNotMeasured as exc:
                assert key in str(exc) and "not claiming exit" in str(exc), exc
            else:
                raise AssertionError(f"2b with {key}={bad} must not yield a result")
        gone = dict(ok2b)
        del gone[key]
        try:
            rec.stage_result_from_stats("2b", _stats([ok2b, gone]))
        except rec.MetricNotMeasured:
            pass
        else:
            raise AssertionError(f"2b without {key} must not yield a result")
    # stage 7 re-measures the stop metrics when the pass produced them (a 2b ran before) and does not need them otherwise
    r7 = rec.stage_result_from_stats("7", _stats([ok2b]))
    assert r7.metrics == ok2b, r7.metrics
    r7 = rec.stage_result_from_stats("7", _stats([{k: v for k, v in ok2b.items() if k not in rec.STOP_METRICS}]))
    assert set(r7.metrics) == {"lane_rms", "lane_r", "depth_absrel", "sem_miou"}, r7.metrics
    # the guard for a stage that cannot be measured is still there (nothing is in it today)
    assert rec.UNMEASURABLE_STAGES == {}
    saved = dict(rec.UNMEASURABLE_STAGES)
    rec.UNMEASURABLE_STAGES["2b"] = "because"
    try:
        for fn, args in ((rec.stage_result_from_stats, ("2b", _stats([ok2b]))), (rec.require_measurable_stages, (["0", "2b"],))):
            try:
                fn(*args)
            except rec.MetricNotMeasured as exc:
                assert "because" in str(exc), exc
            else:
                raise AssertionError("an unmeasurable stage must be refused")
        rec.require_measurable_stages(["0", "1", "2"])
    finally:
        rec.UNMEASURABLE_STAGES.clear()
        rec.UNMEASURABLE_STAGES.update(saved)
    # stages 2 and 7: the 8 m lane score, plus the re-measured aux metrics when the net kept its aux heads
    for sid in ("2", "7"):
        for bad in ({}, {"holdout_evals": []}, {"holdout_evals": [{"lane_rms": None}]},
                    {"holdout_evals": [{"lane_rms": 1.0}, {"lane_rms": nan}]}):
            try:
                rec.stage_result_from_stats(sid, bad)
            except rec.MetricNotMeasured:
                pass
            else:
                raise AssertionError(f"stage {sid}: {bad} must not yield a stage result")
        res = rec.stage_result_from_stats(sid, _stats([{"lane_rms": 1.2, "lane_r": 0.5}, {"lane_rms": 1.1, "lane_r": 0.6,
                                                                                          "depth_absrel": 0.7, "sem_miou": 0.4}]))
        assert res.evals == [1.2, 1.1] and res.ckpt == "b.ckpt" and res.epochs == 2, res
        assert res.metrics == {"lane_rms": 1.1, "lane_r": 0.6, "depth_absrel": 0.7, "sem_miou": 0.4}, res.metrics
        only = rec.stage_result_from_stats(sid, _stats([{"lane_rms": 1.1, "lane_r": None}]))
        assert only.metrics == {"lane_rms": 1.1}, only.metrics
    # stage 3-6 have no metric here
    for sid in ("3", "5", "6"):
        try:
            rec.stage_result_from_stats(sid, _stats([{"lane_rms": 1.0}] * 4))
        except rec.MetricNotMeasured:
            pass
        else:
            raise AssertionError(f"stage {sid} must not yield a result")
    # the chain, built only from real stage results: 0 -> 1 -> 2 -> 7 with the 5% / 2% gates
    on = dict.fromkeys(rec.STAGE_ORDER, True)
    flat = lambda **m: [dict(m)] * 4  # noqa: E731
    stats = {"1": flat(depth_absrel=0.80, sem_miou=0.30), "2": flat(lane_rms=1.0, depth_absrel=0.82, sem_miou=0.30),
             "7": flat(lane_rms=1.0, depth_absrel=0.82, sem_miou=0.30)}

    def chain(edit=None, stages=("0", "1", "2", "7")):
        table = {k: [dict(e) for e in v] for k, v in stats.items()}
        if edit:
            edit(table)

        def fn(stage, prev):
            if stage.id == "0":
                return rec.stage_result_from_stats("0", {"ssl": {"losses": [0.5] * 20}})
            return rec.stage_result_from_stats(stage.id, _stats(table[stage.id], ckpts=(f"s{stage.id}.ckpt",)))
        return rec.run_curriculum(fn, stages=stages, heads_exist=on, data_exist=on)

    out = chain(stages=("0", "1", "2", "7"))
    assert out.trained == ["0", "1", "2", "7"] and out.last_ckpt == "s7.ckpt", out
    assert out.metrics == {"depth_absrel": 0.82, "sem_miou": 0.30, "lane_rms": 1.0}, out.metrics  # no ssl_pretext baseline
    # stage 7 goes through stage_result_from_stats and its own 2% cap (stage 2 may be 5% worse, stage 7 may not)
    chain(lambda t: t["2"].__setitem__(slice(None), flat(lane_rms=1.0, depth_absrel=0.83, sem_miou=0.30)))  # +3.75% <= 5%
    for field, bad in (("depth_absrel", 0.8 * 1.06), ("sem_miou", 0.30 * 0.94)):
        try:
            chain(lambda t: t["2"].__setitem__(slice(None), flat(lane_rms=1.0, **{"depth_absrel": 0.80, "sem_miou": 0.30, field: bad})))
        except rec.CheckpointRejected as exc:
            assert field in str(exc), exc
        else:
            raise AssertionError(f"stage 2 {field} 6% worse than stage 1 must be rejected")
    def seven_worse(t):
        t["2"][:] = flat(lane_rms=1.0, depth_absrel=0.80, sem_miou=0.30)
        t["7"][:] = flat(lane_rms=1.03, depth_absrel=0.80, sem_miou=0.30)
    try:
        chain(seven_worse)
    except rec.CheckpointRejected as exc:
        assert "lane_rms" in str(exc) and "2.00%" in str(exc), exc
    else:
        raise AssertionError("stage 7 lane_rms 3% worse must be rejected by the 2% joint cap")
    # a later stage that did not re-measure an earlier metric is rejected, not skipped
    try:
        chain(lambda t: t["2"].__setitem__(slice(None), flat(lane_rms=1.0)))
    except rec.CheckpointRejected as exc:
        assert "depth_absrel" in str(exc) and "not measured" in str(exc), exc
    else:
        raise AssertionError("stage 2 without depth_absrel must be rejected")
    # a caller's baseline may carry ssl_pretext (it cannot be re-measured): ignored, not a rejection
    out2 = rec.run_curriculum(
        lambda s_, p_: rec.stage_result_from_stats("2", _stats(stats["2"], ckpts=("s2.ckpt",))), stages=("2",), heads_exist=on,
        data_exist=on, earlier_metrics={"ssl_pretext": 0.1, "lane_rms": 1.0, "depth_absrel": 0.82, "sem_miou": 0.30},
        start_ckpt="s1.ckpt")
    assert out2.trained == ["2"] and "ssl_pretext" not in out2.metrics, out2
    # stage 1 exits only when EVERY primary metric has plateaued
    t1 = lambda tb: tb["1"].__setitem__(slice(None), [  # noqa: E731
        {"depth_absrel": 0.8, "sem_miou": 0.30}, {"depth_absrel": 0.8, "sem_miou": 0.35},
        {"depth_absrel": 0.8, "sem_miou": 0.40}, {"depth_absrel": 0.8, "sem_miou": 0.45}])
    try:
        chain(t1)
    except rec.RecipeRefused as exc:
        assert "did not exit" in str(exc), exc
    else:
        raise AssertionError("sem mIoU still improving: stage 1 must not exit")
    print("stage results come from real measurements (0 SSL loss, 1 AbsRel/mIoU, 2/7 lane + re-measured aux); 2b result from the three stop metrics; chain gates 5% / 2% / re-measure OK")


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
                ref = SceneNet(cams=("main",), **scene_net_kwargs(preset))
                out = rec.check_export_fits_deploy(net, path, preset=preset, reference=ref)
                assert out["ok"] and out["params"] == rec.onnx_initializer_count(path), out
        # training-only aux heads are not in the export and not in the student count
        aux = SceneNet(cams=("main",), aux_sensors=True, **scene_net_kwargs("small"))
        assert hasattr(aux, "aux_depth_head") and hasattr(aux, "aux_sem_head")
        with torch.no_grad():
            for prm in aux.parameters():
                prm.add_(torch.randn_like(prm) * 1e-3)
        aux_path = export_scene_onnx(Path(raw) / "aux.onnx", aux, cams=("main",))
        aux_ref = SceneNet(cams=("main",), aux_sensors=True, **scene_net_kwargs("small"))
        out = rec.check_export_fits_deploy(aux, aux_path, preset="small", reference=aux_ref)
        assert out["params"] == rec.onnx_initializer_count(aux_path) < sum(p.numel() for p in aux.parameters()), out
        small = SceneNet(cams=("main",), **scene_net_kwargs("small"))
        tiny_path = Path(raw) / "tiny0.onnx"
        try:
            rec.check_export_fits_deploy(small, tiny_path, preset="small", reference=small)
        except rec.RecipeRefused as exc:
            assert "param count" in str(exc)
        else:
            raise AssertionError("a small net against a tiny export must fail")
        try:
            rec.check_export_fits_deploy(small, tiny_path, preset="medium", reference=small)
        except rec.RecipeRefused:
            pass
        else:
            raise AssertionError("medium must fail the deploy check")
        # Bound to the preset: a medium net (its own export, its own count) labelled "small" or "tiny" is refused
        # against a reference built from that preset's kwargs; the same net with its true preset label is a
        # different refusal (not a deploy size), so the label cannot rescue it either way.
        big = SceneNet(cams=("main",), **scene_net_kwargs("medium"))
        big_path = export_scene_onnx(Path(raw) / "medium.onnx", big, cams=("main",))
        for label in ("small", "tiny"):
            try:
                rec.check_export_fits_deploy(big, big_path, preset=label,
                                             reference=SceneNet(cams=("main",), **scene_net_kwargs(label)))
            except rec.RecipeRefused as exc:
                assert "is not a" in str(exc) and label in str(exc), exc
            else:
                raise AssertionError(f"a medium net labelled {label} passed the deploy check")
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
    """48x64 recording, 70 ticks: lane_gt rows, ``main_depth`` (u16 cm) and ``main_sem`` (u8 ids) train-only planes."""
    import cv2
    import test_trainer2_depth_hires as T

    strips = T._hires_recording(base, 70)
    sem_dir = strips / "raw" / "main_sem"
    sem_dir.mkdir(parents=True)
    for index in range(70):
        ids = np.ones((48, 64), np.uint8)
        ids[:, 32:] = 2 + (index % 2)
        ids[:10] = 3
        ids[-3:, :6] = 0  # unknown: masked out of the loss and the mIoU
        ok, buf = cv2.imencode(".png", ids)
        assert ok
        (sem_dir / f"{index:06d}.png").write_bytes(buf.tobytes())
    (strips / "sem_classes.json").write_text(
        json.dumps({"kind": "gvd_sem_classes", "train_only": True, "classes": {"0": "unknown", "1": "road", "2": "car", "3": "sky"}}),
        encoding="utf-8")
    return strips


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
        for sid in ("1", "2", "7"):  # 2b needs sign ids + stop events: check_stage_2b_trains_end_to_end
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
                        exact_epochs=2, recipe="trainer2", recipe_stage=sid, aux_measure=True)
            except Exception as exc:
                raise AssertionError(f"stage {sid} crashed: {type(exc).__name__}: {exc}\n{buf.getvalue()[-2000:]}") from exc
            finally:
                torch.nn.utils.clip_grad_norm_, train_scene._backward_and_maybe_step = real
            assert int(stats["windows"]) > 0 and seen, (sid, stats.get("windows"), len(seen))
            # every holdout pass scores depth AbsRel / sem mIoU (stage 1 trains them; later stages only re-measure)
            evals = stats["holdout_evals"]
            assert len(evals) == 2, (sid, evals)
            for e in evals:
                assert e["depth_absrel"] is not None and np.isfinite(e["depth_absrel"]) and e["depth_absrel"] > 0.0, (sid, e)
                assert e["sem_miou"] is not None and 0.0 <= e["sem_miou"] <= 1.0, (sid, e)
                assert e["depth_cells"] > 0 and e["sem_cells"] > 0, (sid, e)
            # the pooled sums restart each pass: the same holdout rows give the same cell counts (not 2x on the 2nd)
            assert evals[0]["depth_cells"] == evals[1]["depth_cells"] and evals[0]["sem_cells"] == evals[1]["sem_cells"], evals
            if kw["aux_depth"]:
                assert any(g.get("aux_depth_head.weight", 0.0) > 0.0 for g in seen), (sid, seen)
            if "lane_gt" in kw["heads"]:
                assert any(g.get("lane_gt_head.weight", 0.0) > 0.0 for g in seen), (sid, seen)
            if sid == "1":
                assert all(g.get("lane_gt_head.weight", 0.0) == 0.0 for g in seen), "stage 1 heads=() must not train lane_gt"
            if sid == "7":
                # no ego speed / pedals in this recording: the stop metrics are not measured, softly, outside stage 2b
                assert "stop events not measured in stage 7" in buf.getvalue(), buf.getvalue()[-1500:]
                assert all("missed_stop" not in e for e in evals), evals
    print("recipe on: stages 1 / 2 / 7 train on the fixture; aux_depth and lane_gt heads get gradient; stage 1 trains no lane head OK")


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

        from python.train.trainer2_split import manifest_sha256
        # every pinned value is checked: a manifest that pins any other value for any of them is refused
        pinned = ("recipe", "exit_gain", "exit_patience_evals", "stage_max_epochs", "non_regression_frac",
                  "joint_head_regression_frac", "reality_view_built_and_integrated", "stage_order")
        assert tuple(rec.PINNED_CONFIG_KEYS) == pinned, rec.PINNED_CONFIG_KEYS
        for key in pinned:
            ck2 = base / f"ck_{key}"
            ck2.mkdir()
            _prereg(rec, ck2, strips)
            runm = ck2 / "run_manifest.json"
            body = _json.loads(runm.read_text())
            old_value = body["config"][key]
            body["config"][key] = (not old_value) if isinstance(old_value, bool) else (
                list(reversed(old_value)) if isinstance(old_value, list) else (
                    old_value + "x" if isinstance(old_value, str) else old_value + 0.5))
            assert body["config"][key] != old_value
            body["config_sha256"] = "x"
            body["sha256"] = manifest_sha256(body)
            runm.write_text(_json.dumps(body))
            try:
                rec.require_recipe_manifest(ck2, seed=0, ssl_pretrain=False, train_rows=train, holdout_rows=hold)
            except rec.RecipeRefused as exc:
                assert key in str(exc), (key, exc)
            else:
                raise AssertionError(f"a manifest pinning another {key} must refuse")
    print("recipe manifest: needs --ckpt-dir, a pre-registered run, matching pins/seed/SSL, train and holdout rows in the manifest OK")


def _ckpt_of(net, path, **meta):
    from python.train.epoch_ckpt import write_epoch_ckpt

    base = {"epoch": 3, "step": 7, "cams": ["main"], "scale": 1.0, "backbone": "scratch", "cam_age": False,
            "ctl": False, "heads": [], "preset": "small"}
    base.update(meta)
    write_epoch_ckpt(path, {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}, base, min_free_bytes=0)
    return path


def check_camera_only_deploy_export() -> None:
    """The deploy graph of a stage-7-shaped checkpoint (ctl + aux + road_edge + stop_line) is exported ctl-free.

    ``--export`` refuses a ctl net, and stage 7 trains ctl, so the recipe exports the last checkpoint with the ctl
    head dropped and no ctl-cond input (camera-only runtime), create-only, and checks it against the preset.
    """
    torch = _torch()
    if torch is None:
        skip_check("check_camera_only_deploy_export", "torch not installed")
        return
    try:
        import onnx  # noqa: F401
    except ImportError:
        skip_check("check_camera_only_deploy_export", "onnx not installed (CI installs it)")
        return
    from onnx import numpy_helper

    from python.train.epoch_ckpt import load_epoch_ckpt
    from python.train.model_size import scene_net_kwargs
    from python.train.scene_net import SceneNet

    rec = _recipe()
    heads = ["lanes", "objects", "signs", "lane_gt", "road_edge", "stop_line", "ctl"]
    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        net = SceneNet(cams=("main",), ctl=True, aux_sensors=True, road_edge=True, stop_line=True, **scene_net_kwargs("small"))
        with torch.no_grad():
            for prm in net.parameters():
                prm.add_(torch.randn_like(prm) * 1e-2)
        assert hasattr(net, "ctl_head")
        ck = _ckpt_of(net, base / "s7.ckpt_ep3_step7.npz", ctl=True, heads=heads)
        out = base / "deploy.onnx"
        info = rec.export_camera_only_deploy(ck, out, preset="small")
        assert info["ok"] is True and info["onnx"] == str(out) and out.is_file(), info
        model = onnx.load(str(out))
        names = {i.name for i in model.graph.initializer}
        assert not any("ctl_head" in n for n in names), [n for n in names if "ctl" in n]
        assert not any("cond" in i.name.lower() for i in model.graph.input), [i.name for i in model.graph.input]
        assert not any("ctl" in o.name.lower() for o in model.graph.output), [o.name for o in model.graph.output]
        props = {p.key: p.value for p in model.metadata_props}
        assert "ctl" not in props.get("gvd_heads", "ctl-absent").split(","), props
        assert props.get("gvd_ckpt_file") == ck.name and props.get("gvd_ckpt_epoch") == "3", props
        # every other tensor is the checkpoint's own (the last checkpoint, not a re-init)
        w = load_epoch_ckpt(ck)["weights"]
        inits = {i.name.removeprefix("net."): numpy_helper.to_array(i) for i in model.graph.initializer}
        shared = [k for k in w if k in inits and inits[k].shape == np.shape(w[k])]
        assert len(shared) >= 10 and all(np.array_equal(inits[k], np.asarray(w[k])) for k in shared)
        # create-only: an existing file is refused first and is left byte-for-byte alone
        before = out.read_bytes()
        try:
            rec.export_camera_only_deploy(ck, out, preset="small")
        except rec.RecipeRefused as exc:
            assert "create-only" in str(exc), exc
        else:
            raise AssertionError("an existing --export file must be refused, not overwritten")
        assert out.read_bytes() == before
        # a failing deploy check leaves no file and no .partial
        out2 = base / "deploy2.onnx"
        saved = rec.check_export_fits_deploy

        def boom(*a, **k):
            raise rec.RecipeRefused("count differs")

        rec.check_export_fits_deploy = boom
        try:
            rec.export_camera_only_deploy(ck, out2, preset="small")
        except rec.RecipeRefused:
            pass
        else:
            raise AssertionError("a failed deploy check must refuse")
        finally:
            rec.check_export_fits_deploy = saved
        assert not out2.exists() and not list(base.glob("deploy2*")), list(base.iterdir())
        # camera-only: a ctl-cond / route-cond checkpoint is not exported
        cond = SceneNet(cams=("main",), ctl=True, cond_dim=4, **scene_net_kwargs("small"))
        cck = _ckpt_of(cond, base / "cond.ckpt_ep3_step7.npz", ctl=True, ctl_cond_dim=4, heads=heads)
        try:
            rec.export_camera_only_deploy(cck, base / "cond.onnx", preset="small")
        except rec.RecipeRefused as exc:
            assert "camera-only" in str(exc), exc
        else:
            raise AssertionError("--ctl-cond must never reach the deploy export")
        assert not (base / "cond.onnx").exists()
        # route-cond is a ctl-cond style input too
        routed = SceneNet(cams=("main",), ctl=True, route_cond=True, **scene_net_kwargs("small"))
        rck = _ckpt_of(routed, base / "route.ckpt_ep3_step7.npz", ctl=True, route_cond=True, heads=heads)
        try:
            rec.export_camera_only_deploy(rck, base / "route.onnx", preset="small")
        except rec.RecipeRefused as exc:
            assert "camera-only" in str(exc), exc
        else:
            raise AssertionError("a route-cond checkpoint must not be exported")
        # a tensor the camera-only net does not have (and that is not the ctl head) is refused, not dropped
        from python.train.epoch_ckpt import write_epoch_ckpt

        extra_w = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        extra_w["bogus_head.weight"] = np.zeros((2, 2), np.float32)
        write_epoch_ckpt(base / "extra.ckpt_ep3_step7.npz", extra_w,
                         {"epoch": 3, "step": 7, "cams": ["main"], "scale": 1.0, "backbone": "scratch", "cam_age": False,
                          "ctl": True, "heads": heads, "preset": "small"}, min_free_bytes=0)
        try:
            rec.export_camera_only_deploy(base / "extra.ckpt_ep3_step7.npz", base / "extra.onnx", preset="small")
        except rec.RecipeRefused as exc:
            assert "bogus_head" in str(exc), exc
        else:
            raise AssertionError("an unknown tensor must be refused, not silently dropped")
        # bound to the preset: a medium checkpoint labelled small, and a small one asked for as tiny, are refused
        med = SceneNet(cams=("main",), **scene_net_kwargs("medium"))
        for label, preset_asked, case_net in (("small", "small", med), ("medium", "small", med), ("small", "tiny", net)):
            c = _ckpt_of(case_net, base / f"{label}_{preset_asked}.ckpt_ep3_step7.npz", preset=label,
                         ctl=bool(case_net is net), heads=heads if case_net is net else [])
            try:
                rec.export_camera_only_deploy(c, base / f"{label}_{preset_asked}.onnx", preset=preset_asked)
            except rec.RecipeRefused as exc:
                if label == "medium":
                    assert "checkpoint preset is 'medium'" in str(exc), exc  # the label is checked, not only the size
            else:
                raise AssertionError(f"{label}-labelled {('medium' if case_net is med else 'small')} net passed as {preset_asked}")
            assert not (base / f"{label}_{preset_asked}.onnx").exists()
    print("camera-only deploy export: ctl dropped from a stage-7 ckpt, last-ckpt weights, create-only, preset-bound, ctl-cond refused, failed check leaves no file OK")


def check_aux_eval_numbers() -> None:
    """depth AbsRel / sem mIoU sums: hand-computed values, masks, ignored cells (numpy only)."""
    from python.train.aux_eval import AuxEval

    acc = AuxEval()
    assert acc.absrel() is None and acc.miou() is None  # nothing scored is None, never 0.0
    gt = np.array([[2.0, 4.0], [0.0, 10.0]])
    pred = np.array([[3.0, 2.0], [5.0, 10.0]])
    valid = np.array([[1.0, 1.0], [1.0, 0.0]])
    acc.update_depth(pred, gt, valid)  # cells: |3-2|/2 = .5, |2-4|/4 = .5; gt 0 and masked cells are ignored
    assert abs(acc.absrel() - 0.5) < 1e-12 and acc.rel_n == 2
    acc.update_depth(np.array([[1.0, 1.0]]), np.array([[1.0, 2.0]]), np.array([[1.0, 1.0]]))  # + 0 and .5 -> 4 cells, sum 1.5
    assert abs(acc.absrel() - 0.375) < 1e-12 and acc.rel_n == 4
    acc.update_depth(np.array([[9.0]]), np.array([[1.0]]), np.array([[0.0]]))  # masked: unchanged
    assert acc.rel_n == 4
    try:
        acc.update_depth(np.zeros((2, 2)), np.zeros((2, 3)), np.zeros((2, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch must raise, not broadcast")
    try:  # a size-1 pred would broadcast silently against 2 cells: the explicit shape check refuses it
        acc.update_depth(np.array([5.0]), np.array([1.0, 2.0]), np.array([1.0, 1.0]))
    except ValueError as exc:
        assert "shapes differ" in str(exc), exc
    else:
        raise AssertionError("a broadcastable shape mismatch must raise")
    assert acc.rel_n == 4
    sem = AuxEval()
    logits = np.zeros((3, 1, 4))  # (C, H, W): argmax per cell
    for col, cls in enumerate((1, 1, 2, 0)):
        logits[cls, 0, col] = 5.0
    gt_ids = np.array([[1, 2, 2, 1]])
    sem.update_sem(logits, gt_ids, np.array([[1.0, 1.0, 1.0, 0.0]]))  # last cell masked
    # class 1: pred {0,1} gt {0} -> inter 1 union 2; class 2: pred {2} gt {1,2} -> inter 1 union 2
    assert abs(sem.miou() - 0.5) < 1e-12, sem.miou()
    sem.update_sem(logits[None], gt_ids, np.array([[1.0, 1.0, 1.0, 1.0]]))  # NCHW input; last cell class 0 vs gt 1
    assert sem.sem_n == 7 and 0.0 < sem.miou() < 1.0
    # a class only the model predicts counts (IoU 0, union > 0): classes {1, 2, 3} here, 3 is predicted never present
    only_pred = AuxEval()
    lg = np.zeros((4, 1, 2))
    lg[1, 0, 0] = 5.0
    lg[3, 0, 1] = 5.0
    only_pred.update_sem(lg, np.array([[1, 1]]), np.array([[1.0, 1.0]]))
    # class 1: pred {0} gt {0, 1} -> 1/2; class 3: pred {1} gt {} -> 0/1; mean 0.25
    assert abs(only_pred.miou() - 0.25) < 1e-12, only_pred.miou()
    sem.reset()
    assert sem.miou() is None and sem.absrel() is None
    print("aux eval: AbsRel / mIoU hand values, masks, shape mismatch raises, empty is None OK")


def check_eval_series_and_exit_all() -> None:
    rec = _recipe()
    nan = float("nan")
    # a missing value is NaN (never 0.0): an aux-only stage with no lane score never plateaus at 0.0
    assert all(v != v for v in rec.eval_series("2", [{"lane_rms": None}] * 5)["lane_rms"])
    assert all(v != v for v in rec.eval_series("2", [{}] * 5)["lane_rms"])
    assert rec.eval_series("1", [{"depth_absrel": 0.5}])["sem_miou"][0] != 0.0
    series = rec.eval_series("1", [{"depth_absrel": 0.7, "sem_miou": 0.2, "lane_rms": 9.0}])
    assert series == {"depth_absrel": [0.7], "sem_miou": [0.2]}, series
    assert rec.eval_series("2b", [{"lane_rms": 1.0}]) == {k: [nan] for k in rec.STAGE_PRIMARY_METRICS["2b"]} or all(
        v != v for vs in rec.eval_series("2b", [{"lane_rms": 1.0}]).values() for v in vs)
    flat = [0.5, 0.5, 0.5, 0.5]
    assert rec.stage_exited_all("2", {"lane_rms": [nan] * 5}, epochs=5) is False
    assert rec.stage_exited_all("2", {"lane_rms": [0.0] * 5}, epochs=5) is True  # a real flat 0.0 is a plateau; NaN is not
    assert rec.stage_exited_all("2", {"lane_rms": [nan] * 5}, epochs=rec.STAGE_MAX_EPOCHS) is True  # the cap always exits
    assert rec.stage_exited_all("1", {"depth_absrel": flat, "sem_miou": flat}, epochs=4) is True
    assert rec.stage_exited_all("1", {"depth_absrel": flat, "sem_miou": [0.3, 0.35, 0.4, 0.45]}, epochs=4) is False
    assert rec.stage_exited_all("1", {"depth_absrel": flat}, epochs=4) is False  # a metric never evaluated has not plateaued
    assert rec.stage_exited_all("0", {"ssl_pretext": flat}, epochs=4) is True
    assert rec.stage_exited_all("1", {"depth_absrel": flat}, epochs=rec.STAGE_MAX_EPOCHS) is True  # the cap needs no series
    assert rec.stage_exited_all("1", {}, epochs=rec.STAGE_MAX_EPOCHS - 1) is False
    print("eval series: missing is NaN not 0.0; stage exit needs every primary metric flat (or the cap) OK")


def check_stage7_lr_multipliers_pinned() -> None:
    rec = _recipe()
    heads = {"aux_depth_head", "aux_sem_head", "lane_head", "lane_valid_head", "lane_gt_head", "road_edge_head", "sign_head",
             "sign_class_head", "sign_state_head", "ctl_head", "stop_line_head"}
    want = {"stem": 0.1, "trunk": 1.0, **dict.fromkeys(heads, 0.3)}
    assert rec.stage_lr_mults_for("7") == want, rec.stage_lr_mults_for("7")  # joint: every head trains at x0.3, none enters at x1
    assert all(v == 0.3 for k, v in rec.stage_lr_mults_for("2b").items() if k in {"aux_depth_head", "lane_gt_head"})
    assert rec.stage_lr_mults_for("2b")["ctl_head"] == 1.0 and rec.stage_lr_mults_for("1")["aux_depth_head"] == 1.0
    print("stage 7 LR multipliers pinned (stem 0.1, trunk 1.0, every head 0.3); 2b / 1 entering heads at 1.0 OK")


def check_stop_path_keeps_old_checkpoints() -> None:
    """The run that ends on an unmeasurable stage (MetricNotMeasured) still re-checks the old checkpoints."""
    rec = _recipe()
    on = dict.fromkeys(rec.STAGE_ORDER, True)
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        old = root / "run.ckpt_ep1_step1.npz"
        old.write_bytes(b"older run")

        def stop_and_delete(stage, prev):
            if stage.id == "0":
                return _res(rec, {"ssl_pretext": 1.0}, evals=(1.0,) * 4)
            old.unlink()  # a stage that removes an old checkpoint, then stops unmeasured
            raise rec.MetricNotMeasured("not measured")

        try:
            rec.run_curriculum(stop_and_delete, stages=("0", "1"), heads_exist=on, data_exist=on, ckpt_root=root)
        except rec.MetricNotMeasured:
            raise AssertionError("the removed checkpoint must be reported, not hidden behind the stop")
        except rec.RecipeRefused as exc:
            assert "run.ckpt_ep1_step1.npz" in str(exc), exc
        old.write_bytes(b"older run")
        try:
            rec.run_curriculum(lambda s, p: (_ for _ in ()).throw(rec.MetricNotMeasured("x")) if s.id == "1" else
                               _res(rec, {"ssl_pretext": 1.0}, evals=(1.0,) * 4), stages=("0", "1"), heads_exist=on,
                               data_exist=on, ckpt_root=root)
        except rec.MetricNotMeasured:
            pass
        else:
            raise AssertionError("an untouched tree must still propagate the stop")
    print("stop path: an old checkpoint removed by the stopping stage is reported; an untouched tree propagates the stop OK")


def check_ssl_epoch_losses_and_stop() -> None:
    torch = _torch()
    if torch is None:
        skip_check("check_ssl_epoch_losses_and_stop", "torch not installed")
        return
    from python.train import exp_ssl
    from python.train.scene_net import SceneNet

    rows = [{"i": i, "kind": "train", "source": "s"} for i in range(6)]
    rng = np.random.RandomState(0)
    images = {i: (rng.rand(48, 64, 3) * 255).astype(np.uint8) for i in range(6)}
    image_fn = lambda r: images[int(r["i"])]  # noqa: E731
    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        net = SceneNet(cams=("main",), scale=1.0)
        stats = exp_ssl.run_ssl_pretrain(net, rows, image_fn=image_fn, out_path=base / "a.npz", lr=1e-3, seed=0, epochs=3)
        assert len(stats["epoch_losses"]) == 3 and all(np.isfinite(stats["epoch_losses"])), stats["epoch_losses"]
        assert stats["last_loss"] > 0 and stats["n"] == 18.0
        seen = []
        stop = lambda losses: (seen.append(list(losses)), len(losses) >= 2)[1]  # noqa: E731
        stats2 = exp_ssl.run_ssl_pretrain(SceneNet(cams=("main",), scale=1.0), rows, image_fn=image_fn, out_path=base / "b.npz",
                                          lr=1e-3, seed=0, epochs=10, stop_fn=stop)
        assert len(stats2["epoch_losses"]) == 2 and seen == [stats2["epoch_losses"][:1], stats2["epoch_losses"]], (seen, stats2)
        # the epoch loss is the MEAN of its steps (6 rows x 2 epochs, losses 1..12 -> 3.5 and 9.5)
        feed = iter(float(v) for v in range(1, 13))
        real_loss = exp_ssl.masked_token_pretext_loss
        exp_ssl.masked_token_pretext_loss = lambda n, img, rng=None, target=None: sum(
            p.sum() for p in n.stem.parameters()) * 0.0 + next(feed)
        try:
            st = exp_ssl.run_ssl_pretrain(SceneNet(cams=("main",), scale=1.0), rows, image_fn=image_fn, out_path=base / "m.npz",
                                          lr=1e-3, seed=0, epochs=2)
        finally:
            exp_ssl.masked_token_pretext_loss = real_loss
        assert st["epoch_losses"] == [3.5, 9.5] and st["last_loss"] == 12.0, st
        # the loss history is saved create-only beside the stem and a reused stem reports it
        net3 = SceneNet(cams=("main",), scale=1.0)
        path, info = exp_ssl.default_ssl_stem(net3, rows, image_fn=image_fn, out_dir=base, lr=1e-3, seed=0, epochs=3, max_rows=0)
        side = path.with_suffix("").with_name(path.stem + ".losses.json")
        assert not info["reused"] and info["losses"] == info["epoch_losses"] and side.is_file(), (info, list(base.iterdir()))
        before = side.read_text()
        path2, info2 = exp_ssl.default_ssl_stem(SceneNet(cams=("main",), scale=1.0), rows, image_fn=image_fn, out_dir=base, lr=1e-3,
                                                seed=0, epochs=3, max_rows=0)
        assert path2 == path and info2["reused"] and info2["losses"] == info["losses"] and side.read_text() == before
        side.unlink()
        # an existing history is never replaced (the stem write that follows a crash keeps the first file)
        side.write_text(json.dumps({"epoch_losses": [9.0], "key": "old"}))
        (base / path.name).unlink()
        _p0, info0 = exp_ssl.default_ssl_stem(SceneNet(cams=("main",), scale=1.0), rows, image_fn=image_fn, out_dir=base, lr=1e-3,
                                              seed=0, epochs=3, max_rows=0)
        assert not info0["reused"] and json.loads(side.read_text()) == {"epoch_losses": [9.0], "key": "old"}, side.read_text()
        side.unlink()
        _p, info3 = exp_ssl.default_ssl_stem(SceneNet(cams=("main",), scale=1.0), rows, image_fn=image_fn, out_dir=base, lr=1e-3,
                                             seed=0, epochs=3, max_rows=0)
        assert info3["reused"] and info3["losses"] is None  # no history: unknown, never invented
        assert not list(base.glob("*.partial*")), list(base.iterdir())
        # default_ssl_stem hands its stop_fn to the run: a plateau rule ends the stem at 2 of 10 epochs
        sub = base / "stopped"
        sub.mkdir()
        _ps, infos = exp_ssl.default_ssl_stem(SceneNet(cams=("main",), scale=1.0), rows, image_fn=image_fn, out_dir=sub, lr=1e-3,
                                              seed=0, epochs=10, max_rows=0, stop_fn=lambda losses: len(losses) >= 2)
        assert not infos["reused"] and len(infos["losses"]) == 2, infos
    print("SSL: per-epoch pretext losses, stop_fn ends the run, history saved create-only and returned on reuse (None if absent) OK")



def check_measure_aux_unit() -> None:
    """``_measure_aux`` feeds the pooled sums from a prediction and its target (tensors in, numbers out)."""
    torch = _torch()
    if torch is None:
        skip_check("check_measure_aux_unit", "torch not installed")
        return
    from python.train.aux_eval import AuxEval
    from python.train.train_scene import _measure_aux

    depth = torch.tensor([[[2.0, 8.0], [4.0, 4.0]]])  # 1HW like _prediction_at
    logits = torch.zeros(3, 2, 2)
    for k, cls in enumerate((1, 1, 2, 2)):
        logits[cls, k // 2, k % 2] = 4.0
    tgt = {"aux_depth": torch.tensor([[4.0, 4.0], [4.0, 0.0]]), "aux_depth_valid": torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
           "aux_sem": torch.tensor([[1.0, 2.0], [2.0, 2.0]]), "aux_sem_valid": torch.ones(2, 2)}
    acc = AuxEval()
    _measure_aux(acc, SimpleNamespace(aux_depth=depth, aux_depth_far=None, aux_sem_logits=logits), tgt)
    # AbsRel: |2-4|/4 = .5, |8-4|/4 = 1, |4-4|/4 = 0 -> 0.5; mIoU: class 1 pred {0,1} gt {0} -> 1/2, class 2 pred {2,3} gt {1,2,3} -> 2/3
    assert abs(acc.absrel() - 0.5) < 1e-9 and acc.rel_n == 3, (acc.absrel(), acc.rel_n)
    assert abs(acc.miou() - (0.5 + 2 / 3) / 2) < 1e-9, acc.miou()
    # a depth_hires prediction is another grid: it is not scored (and does not crash)
    hi = AuxEval()
    _measure_aux(hi, SimpleNamespace(aux_depth=torch.zeros(1, 4, 4), aux_depth_far=torch.zeros(1, 4, 4), aux_sem_logits=None), tgt)
    assert hi.absrel() is None and hi.rel_n == 0
    # a net without the aux heads, or a target without labels, adds nothing
    none = AuxEval()
    _measure_aux(none, SimpleNamespace(aux_depth=None, aux_depth_far=None, aux_sem_logits=None), tgt)
    _measure_aux(none, SimpleNamespace(aux_depth=depth, aux_depth_far=None, aux_sem_logits=logits), {})
    assert none.absrel() is None and none.miou() is None
    print("_measure_aux: AbsRel / mIoU from tensors, depth_hires and missing heads / labels are not scored OK")


def check_default_ssl_stage_recipe_wiring() -> None:
    """``_default_ssl_stage(recipe_stop=True)``: epoch cap = STAGE_MAX_EPOCHS, plateau stop, own cache key, rows checked first."""
    torch = _torch()
    if torch is None:
        skip_check("check_default_ssl_stage_recipe_wiring", "torch not installed")
        return
    import python.data.data_root as data_root
    from python.train import exp_ssl, train_scene

    rec = _recipe()
    rows = [{"i": i, "kind": "train", "source": "s"} for i in range(3)]
    args = SimpleNamespace(lr=1e-3, seed=0, ssl_epochs=3, ssl_max_rows=0, scale=1.0, holdout_blocks=1)
    order: list = []
    seen: dict = {}

    def fake_stem(net, picked, **kw):
        order.append("stem")
        seen.update(kw)
        return Path("x.npz"), {"key": "k", "rows": 3, "reused": False, "losses": [0.4, 0.3], "last_loss": 0.3, "n": 6.0}

    saved = (train_scene._ssl_train_rows_and_reader, exp_ssl.default_ssl_stem, os.environ.get("GVD_DATA_DIR"))
    with tempfile.TemporaryDirectory() as raw:
        os.environ["GVD_DATA_DIR"] = raw
        train_scene._ssl_train_rows_and_reader = lambda *a, **k: (rows, lambda r: None)
        exp_ssl.default_ssl_stem = fake_stem
        try:
            path, meta = train_scene._default_ssl_stage(args, (Path(raw),), Path(raw), ("main",), {}, recipe_stop=True,
                                                        rows_check=lambda r: order.append(("rows", len(r))))
            assert order == [("rows", 3), "stem"], order  # the rows are checked before the stem trains
            assert seen["epochs"] == rec.STAGE_MAX_EPOCHS and callable(seen["stop_fn"]), seen
            assert seen["key_extra"]["recipe_exit"] == [rec.EXIT_GAIN, rec.EXIT_PATIENCE_EVALS, rec.STAGE_MAX_EPOCHS]
            assert seen["stop_fn"]([0.5] * 3) is False and seen["stop_fn"]([0.5] * 4) is True
            assert seen["stop_fn"]([0.9, 0.8, 0.7, 0.6]) is False  # still improving by > 1% per epoch
            assert meta["losses"] == [0.4, 0.3] and meta["key"] == "k" and meta["rows"] == 3, meta
            # not the recipe: --ssl-epochs applies, no stop rule, no recipe key part, no loss history in the meta
            order.clear(); seen.clear()
            path, meta = train_scene._default_ssl_stage(args, (Path(raw),), Path(raw), ("main",), {})
            assert seen["epochs"] == 3 and seen["stop_fn"] is None and "recipe_exit" not in seen["key_extra"], seen
            assert "losses" not in meta, meta
            # a failing rows check stops before the stem is trained
            order.clear()

            def refuse(rows_):
                raise RuntimeError("rows refused")

            try:
                train_scene._default_ssl_stage(args, (Path(raw),), Path(raw), ("main",), {}, recipe_stop=True, rows_check=refuse)
            except RuntimeError as exc:
                assert "rows refused" in str(exc)
            else:
                raise AssertionError("rows_check must run")
            assert order == [], order
        finally:
            train_scene._ssl_train_rows_and_reader, exp_ssl.default_ssl_stem = saved[0], saved[1]
            if saved[2] is None:
                os.environ.pop("GVD_DATA_DIR", None)
            else:
                os.environ["GVD_DATA_DIR"] = saved[2]
    print("_default_ssl_stage: recipe cap + plateau stop + own cache key + rows checked before the stem; recipe-off unchanged OK")


def check_recipe_stage_exit_is_owned_by_the_recipe() -> None:
    """A real stage-1 train_directory run stops exactly when ``stage_exited_all`` says so, on its two measured series."""
    if not _torch_cv2_onnx("check_recipe_stage_exit_is_owned_by_the_recipe"):
        return
    import contextlib
    import io

    from python.train import train_scene

    rec = _recipe()
    calls: list = []
    real = rec.stage_exited_all

    def spy(stage_id, evals_by, *, epochs):
        out = real(stage_id, evals_by, epochs=epochs)
        calls.append((stage_id, {k: list(v) for k, v in evals_by.items()}, int(epochs), out))
        return out

    rec.stage_exited_all = spy
    buf = io.StringIO()
    try:
        with tempfile.TemporaryDirectory() as raw, contextlib.redirect_stdout(buf):
            base = Path(raw)
            strips = _fixture(base)
            kw = rec.stage_train_kwargs("1")
            stats = train_scene.train_directory(
                strips, epochs=rec.STAGE_MAX_EPOCHS, lr=1e-3, capability=None, export_path=None, free_vram_bytes=0,
                free_ram_bytes=8_000_000_000, resume=False, open_panel=False, autostart=True, cams=("main",),
                heads=kw["heads"], aux_depth=True, aux_sem=True, aux_depth_weight=1.0, aux_sem_weight=1.0, augment=False,
                workers=0, min_valid=0.0, holdout_blocks=1, scale=1.0, status_root=base / "st", ckpt_dir=base / "ck",
                recipe="trainer2", recipe_stage="1", aux_measure=True)
    finally:
        rec.stage_exited_all = real
    evals = stats["holdout_evals"]
    n = len(evals)
    assert 4 <= n <= rec.STAGE_MAX_EPOCHS and stats["epochs_run"] == n and len(calls) == n, (n, stats["epochs_run"], len(calls))
    # one call per eval, on the stage's two metrics as measured (NaN would only stand in for a missing value)
    for k, (sid, series, epochs, out) in enumerate(calls):
        assert sid == "1" and epochs == k + 1, (k, sid, epochs)
        assert set(series) == {"depth_absrel", "sem_miou"}, series
        assert series["depth_absrel"] == [e["depth_absrel"] for e in evals[: k + 1]], (k, series)
        assert series["sem_miou"] == [e["sem_miou"] for e in evals[: k + 1]], (k, series)
        assert out == (k == n - 1), (k, n, out)  # the run stops at the first True, not before and not after
    assert "recipe stage 1: stage exit" in buf.getvalue() or n == rec.STAGE_MAX_EPOCHS
    print(f"recipe-owned exit: a real stage 1 ran {n} epochs and stopped at the first stage_exited_all True on AbsRel + mIoU OK")


def _fake_env(base):
    """Real ``_run_trainer2_recipe`` with the two compute stages replaced by recorders; manifest, curriculum, export and
    deploy check are the real code."""
    from python.train import train_scene
    from python.train.model_size import scene_net_kwargs
    from python.train.scene_net import SceneNet

    rec = _recipe()
    strips = _fixture(base)
    calls: list = []
    env = SimpleNamespace(base=base, strips=strips, calls=calls, rec=rec, ssl_kw=[], on_train=None, raise_in=None, n=0)

    def fake_ssl(args, roots, holdout_root, cams, net_kw, **kw):
        calls.append(("ssl",))
        env.ssl_kw.append(kw)
        return None, {"on": True, "why": "on (default)", "key": "k", "rows": 4, "reused": False, "losses": [0.5] * 20}

    table = {"1": {"depth_absrel": 0.8, "sem_miou": 0.3},
             "2": {"lane_rms": 1.0, "lane_r": 0.9, "depth_absrel": 0.82, "sem_miou": 0.3},
             "2b": {"lane_rms": 1.0, "lane_r": 0.9, "depth_absrel": 0.82, "sem_miou": 0.3, "missed_stop": 1.0,
                    "missed_light": 1.0, "false_brake": 1.0},
             "7": {"lane_rms": 1.0, "lane_r": 0.9, "depth_absrel": 0.82, "sem_miou": 0.3}}

    def fake_train(root, **kw):
        sid = kw["recipe_stage"]
        calls.append(("train", sid, kw["export_path"], kw["aux_measure"], tuple(kw["heads"])))
        if env.on_train is not None:
            env.on_train(sid)
        if env.raise_in == sid:
            raise rec.MetricNotMeasured(f"stage {sid} unmeasured")
        env.n += 1
        net = SceneNet(cams=("main",), ctl="ctl" in kw["heads"], aux_sensors=True, **scene_net_kwargs("small"))
        path = Path(kw["ckpt_dir"]) / f"scene_x.run20260101T00000{env.n}Z.ckpt_ep{env.n}_step{env.n}.npz"
        _ckpt_of(net, path, epoch=env.n, step=env.n, ctl="ctl" in kw["heads"], heads=list(kw["heads"]), preset="small")
        return {"holdout_evals": [dict(table[sid])] * 4, "epoch_ckpts": [str(path)], "epochs_run": 4}

    def run(*, stage="0,1,2,7", export=None, preset="small", extra=(), ck_name="ck", strips_root=None):
        use = strips_root or strips
        ck = base / ck_name
        ck.mkdir(exist_ok=True)
        argv = ["--recipe", "trainer2", "--no-augment", "--cams", "main", "--ckpt-dir", str(ck), "--min-valid", "0",
                "--workers", "0", "--no-panel", "--stage", stage, *extra]
        args = _parser().parse_args(argv)
        saved = (train_scene.train_directory, train_scene._default_ssl_stage, rec._repo_git_sha)
        train_scene.train_directory, train_scene._default_ssl_stage, rec._repo_git_sha = fake_train, fake_ssl, lambda: "0" * 40
        try:
            res = train_scene._run_trainer2_recipe(
                args, strip_roots=(use,), holdout_root=use, cam_names=("main",), net_kw=scene_net_kwargs(preset),
                preset_name=preset, train_precision="auto", plan_precision=None, frame_cache=None, aux_cache=None,
                export_path=export, pause_dir=None, start_now=True, open_panel=False)
            return res, None, ck
        except Exception as exc:  # noqa: BLE001
            return None, exc, ck
        finally:
            train_scene.train_directory, train_scene._default_ssl_stage, rec._repo_git_sha = saved

    env.run = run
    return env


def _torch_cv2_onnx(name):
    if _torch() is None:
        skip_check(name, "torch not installed")
        return False
    try:
        import cv2  # noqa: F401
        import onnx  # noqa: F401
    except ImportError:
        skip_check(name, "cv2 / onnx not installed (CI installs both)")
        return False
    return True


def check_runner_refuses_before_training() -> None:
    """Every refusal the runner owns happens before any stage trains and before any manifest is written."""
    if not _torch_cv2_onnx("check_runner_refuses_before_training"):
        return
    rec = _recipe()
    with tempfile.TemporaryDirectory() as raw:
        env = _fake_env(Path(raw))
        existing = env.base / "existing.onnx"
        existing.write_bytes(b"a model that must survive")
        cases = (
            ("existing export", dict(export=existing), rec.RecipeRefused, "create-only"),
            ("export, SSL-only plan", dict(stage="0", export=env.base / "new.onnx"), rec.RecipeRefused, "supervised stage"),
            ("medium preset", dict(preset="medium"), rec.RecipeRefused, "GB"),
            ("xl preset", dict(preset="xl"), rec.RecipeRefused, "GB"),
            ("--no-ssl-pretrain with stage 0", dict(extra=("--no-ssl-pretrain",)), rec.RecipeRefused, "--no-ssl-pretrain"),
            ("--init-stem with stage 0", dict(extra=("--init-stem", str(env.base / "x.npz"))), rec.RecipeRefused, "--init-stem"),
            ("--holdout-cells with stage 0", dict(extra=("--holdout-cells",)), rec.RecipeRefused, "cell"),
        )
        for k, (label, kw, kind, word) in enumerate(cases):
            res, err, ck = env.run(ck_name=f"ck{k}", **kw)
            assert res is None and isinstance(err, kind) and word in str(err), (label, err)
            assert env.calls == [], (label, env.calls)
            assert not (ck / "run_manifest.json").exists() and not (ck / "split_manifest.json").exists(), label
        assert existing.read_bytes() == b"a model that must survive"
        # the default plan (2b in it) on a recording with no sign ids: refused before SSL / any stage / any manifest
        _r, err, ck2b = env.run(stage="", ck_name="ck_2b")
        assert isinstance(err, rec.RecipeRefused), err
        for word in ("sim_t", "Nothing was trained", "--stage"):  # the plain fixture rows carry no ego speed / time at all
            assert word in str(err), (word, err)
        assert env.calls == [] and not (ck2b / "run_manifest.json").exists(), env.calls
        # dropping 2b from --stage is the supported way to run the rest
        res, err, ck = env.run(stage="0,1,2,7", ck_name="ck_ok")
        assert err is None and res.trained == ["0", "1", "2", "7"], err
        # a recording that has the stop events in its holdout: the default plan runs 2b, in order
        import stage2b_fixture as fx

        stops = fx.build(env.base / "with_stops", fx.N_ROWS, fx.intersections(fx.N_ROWS))
        env.calls.clear()
        res, err, ck = env.run(stage="", ck_name="ck_all", strips_root=stops)
        assert err is None and res.trained == ["0", "1", "2", "2b", "7"], err
        order = [c[1] for c in env.calls if c[0] == "train"]
        assert order == ["1", "2", "2b", "7"], env.calls
        heads_2b = [c[4] for c in env.calls if c[0] == "train" and c[1] == "2b"][0]
        assert "signs" in heads_2b and "stop_line" in heads_2b and "ctl" in heads_2b, heads_2b
        # the same recording with its signs.jsonl removed is refused again (the join has nothing to find)
        bare = fx.build(env.base / "bare", fx.N_ROWS, fx.intersections(fx.N_ROWS), with_signs=False)
        env.calls.clear()
        _r, err, ck_bare = env.run(stage="", ck_name="ck_bare", strips_root=bare)
        assert isinstance(err, rec.RecipeRefused) and env.calls == [], (err, env.calls)
        for word in ("stage 2b", "0 stop-sign/line", "0 red/yellow-light", "vacuous", "Nothing was trained", "--stage"):
            assert word in str(err), (word, err)
        assert not (ck_bare / "run_manifest.json").exists()
    print("runner: existing export / SSL-only export / medium+xl preset / --no-ssl-pretrain / --init-stem / stage 2b without events refuse before anything trains or is written; with events the default plan reaches 2b OK")


def check_runner_exports_once_after_the_last_stage() -> None:
    if not _torch_cv2_onnx("check_runner_exports_once_after_the_last_stage"):
        return
    import onnx

    rec = _recipe()
    with tempfile.TemporaryDirectory() as raw:
        env = _fake_env(Path(raw))
        export = env.base / "student.onnx"
        env.on_train = lambda sid: (_ for _ in ()).throw(AssertionError(f"export exists before stage {sid}")) if export.exists() else None
        res, err, ck = env.run(export=export)
        assert err is None, err
        assert res.trained == ["0", "1", "2", "7"] and export.is_file(), (res.trained, err)
        trains = [c for c in env.calls if c[0] == "train"]
        # no stage exports on its own (train_directory never gets an export path) and every stage keeps the aux heads
        assert [c[1] for c in trains] == ["1", "2", "7"] and all(c[2] is None and c[3] is True for c in trains), trains
        assert "ctl" in trains[-1][4], trains[-1]
        assert env.ssl_kw and env.ssl_kw[0].get("recipe_stop") is True and callable(env.ssl_kw[0].get("rows_check")), env.ssl_kw
        assert sorted(p.name for p in env.base.glob("student*")) == ["student.onnx"], list(env.base.iterdir())
        props = {p.key: p.value for p in onnx.load(str(export)).metadata_props}
        last = sorted(ck.glob("*.ckpt_ep*.npz"))[-1]
        assert props["gvd_ckpt_file"] == last.name == Path(res.last_ckpt).name, (props.get("gvd_ckpt_file"), last.name)
        assert res.deploy and res.deploy["ok"] is True and res.deploy["preset"] == "small", res.deploy
        assert not any("ctl_head" in i.name for i in onnx.load(str(export)).graph.initializer)
        assert set(res.metrics) == {"lane_rms", "lane_r", "depth_absrel", "sem_miou"}, res.metrics  # no ssl_pretext baseline
        # a stop before the last stage exports nothing
        env2 = _fake_env(Path(raw) / "b") if (Path(raw) / "b").mkdir() is None else None
        env2.raise_in = "2"
        e2 = env2.base / "student2.onnx"
        r2, err2, _ = env2.run(export=e2)
        assert isinstance(err2, rec.MetricNotMeasured) and not e2.exists(), (err2, e2.exists())
        # the runner does not take the export's word for it: a result without ok=True is refused
        env3 = _fake_env(Path(raw) / "c") if (Path(raw) / "c").mkdir() is None else None
        saved = rec.export_camera_only_deploy
        rec.export_camera_only_deploy = lambda *a, **k: {"ok": False, "preset": "small"}
        try:
            r3, err3, _ = env3.run(export=env3.base / "student3.onnx")
        finally:
            rec.export_camera_only_deploy = saved
        assert isinstance(err3, rec.RecipeRefused) and "did not pass" in str(err3), err3
        rec.export_camera_only_deploy = lambda *a, **k: {"preset": "small"}  # no "ok" key at all
        env4 = _fake_env(Path(raw) / "d") if (Path(raw) / "d").mkdir() is None else None
        try:
            r4, err4, _ = env4.run(export=env4.base / "student4.onnx")
        finally:
            rec.export_camera_only_deploy = saved
        assert isinstance(err4, rec.RecipeRefused) and "did not pass" in str(err4), err4
    print("runner: one camera-only export after the last stage from the last checkpoint (ctl dropped); a stop exports nothing; ok=True required OK")


def check_runner_ssl_rows_must_be_manifest_train_rows() -> None:
    if not _torch_cv2_onnx("check_runner_ssl_rows_must_be_manifest_train_rows"):
        return
    from python.train.trainer2_split import SplitError

    rec = _recipe()
    with tempfile.TemporaryDirectory() as raw:
        env = _fake_env(Path(raw))
        res, err, ck = env.run(stage="0", ck_name="ck_a")
        assert err is None, err
        check = env.ssl_kw[0]["rows_check"]
        train, hold = rec.legacy_split_rows((env.strips,), env.strips, holdout_blocks=1)
        assert train and hold
        check(train[:5])  # train rows pass
        for bad, label in ((hold[:1], "holdout row"), ([dict(train[0], i=987654)], "row the manifest does not list"),
                           (train[:3] + hold[:1], "train rows with one holdout row")):
            try:
                check(bad)
            except SplitError as exc:
                assert "ssl" in str(exc), exc
            else:
                raise AssertionError(f"SSL {label} must be refused")
        # untagged rows (single-root, no ``source``) are tagged with the recording root like the split's own rows
        run = rec.require_preregistered_for_test(ck) if hasattr(rec, "require_preregistered_for_test") else None
        from python.train.trainer2_split import require_preregistered

        run = require_preregistered(ck)
        bare = [{k: v for k, v in r.items() if k != "source"} for r in train[:3]]
        assert rec.check_ssl_rows_in_manifest(bare, run, holdout_root=env.strips) == 3
        assert rec.check_ssl_rows_in_manifest([], run, holdout_root=env.strips) == 0
        # the real stage-0 path refuses before the stem is trained when the SSL rows include a holdout row
        from python.train import exp_ssl, train_scene

        trained = []
        saved = (train_scene._ssl_train_rows_and_reader, exp_ssl.default_ssl_stem, rec._repo_git_sha)
        train_scene._ssl_train_rows_and_reader = lambda *a, **k: (train[:3] + hold[:1], lambda r: None)
        exp_ssl.default_ssl_stem = lambda *a, **k: trained.append(1)
        rec._repo_git_sha = lambda: "0" * 40
        try:
            from python.train.model_size import scene_net_kwargs

            args = _parser().parse_args(["--recipe", "trainer2", "--no-augment", "--cams", "main", "--ckpt-dir", str(ck),
                                         "--min-valid", "0", "--workers", "0", "--no-panel", "--stage", "0"])
            try:
                train_scene._run_trainer2_recipe(
                    args, strip_roots=(env.strips,), holdout_root=env.strips, cam_names=("main",), net_kw={}, preset_name="small",
                    train_precision="auto", plan_precision=None, frame_cache=None, aux_cache=None, export_path=None,
                    pause_dir=None, start_now=True, open_panel=False)
            except SplitError as exc:
                assert "ssl" in str(exc), exc
            else:
                raise AssertionError("a holdout row among the SSL rows must stop stage 0")
        finally:
            train_scene._ssl_train_rows_and_reader, exp_ssl.default_ssl_stem, rec._repo_git_sha = saved
        assert trained == [], "the stem must not be trained on rows the manifest does not allow"
    print("SSL rows: subset of the manifest's train rows (holdout / unlisted rows refused before the stem trains) OK")


def check_runner_preregisters_the_manifest() -> None:
    """The real path writes the split + run manifests before any stage (create-only), then checks them."""
    if not _torch_cv2_onnx("check_runner_preregisters_the_manifest"):
        return
    import json as _json

    from python.train.trainer2_split import SplitError, require_preregistered

    rec = _recipe()
    with tempfile.TemporaryDirectory() as raw:
        env = _fake_env(Path(raw))
        seen = {}
        env.on_train = lambda sid: seen.setdefault("run_before_first_stage", (env.base / "ck" / "run_manifest.json").is_file())
        res, err, ck = env.run(stage="0,1,2,7")
        assert err is None and seen.get("run_before_first_stage") is True, (err, seen)
        run = require_preregistered(ck)
        assert run["seed"] == 0 and run["ssl_pretrain"] is True and run["git_sha"] == "0" * 40, run
        assert run["config"]["recipe"] == "trainer2" and run["decision_tier"] == "iteration", run
        assert (ck / "split_manifest.json").is_file() and Path(run["split_path"]) == (ck / "split_manifest.json").resolve()
        assert (env.base / "prereg_registry").is_dir() and len(list((env.base / "prereg_registry").glob("*.json"))) == 1
        # a second run in the same folder finds it; the file is not rewritten
        before = (ck / "run_manifest.json").read_bytes()
        res, err, _ = env.run(stage="0,1,2,7")
        assert err is None and (ck / "run_manifest.json").read_bytes() == before
        # another seed in the same folder is a re-roll: refused, nothing rewritten
        res, err, _ = env.run(stage="0,1,2,7", extra=("--seed", "1"))
        assert isinstance(err, rec.RecipeRefused) and "seed" in str(err), err
        assert (ck / "run_manifest.json").read_bytes() == before
        # SSL off in the plan (--stage 1,2,7 is refused mid-curriculum, so use the helper directly) is recorded as such
        ck2 = env.base / "ck2"
        assert rec.ensure_recipe_manifest(ck2, strip_roots=(env.strips,), holdout_root=env.strips, holdout_blocks=1, seed=3,
                                          ssl_pretrain=False, git_sha="1" * 40) is True
        run2 = require_preregistered(ck2)
        assert run2["ssl_pretrain"] is False and run2["seed"] == 3 and run2["git_sha"] == "1" * 40
        assert rec.ensure_recipe_manifest(ck2, strip_roots=(env.strips,), holdout_root=env.strips, holdout_blocks=1, seed=3,
                                          ssl_pretrain=False, git_sha="1" * 40) is False
        # a sibling folder with another seed for the same arm: the shared registry refuses the re-roll
        try:
            rec.ensure_recipe_manifest(env.base / "ck3", strip_roots=(env.strips,), holdout_root=env.strips, holdout_blocks=1,
                                       seed=4, ssl_pretrain=False, git_sha="1" * 40)
        except SplitError as exc:
            assert "re-roll" in str(exc), exc
        else:
            raise AssertionError("a second seed for the same arm must be refused")
        # a different split already in --ckpt-dir is refused, not overwritten
        ck4 = env.base / "ck4"
        ck4.mkdir()
        (ck4 / "split_manifest.json").write_text(_json.dumps({"format": "x"}))
        try:
            rec.ensure_recipe_manifest(ck4, strip_roots=(env.strips,), holdout_root=env.strips, holdout_blocks=1, seed=0,
                                       ssl_pretrain=True, git_sha="1" * 40)
        except Exception as exc:  # noqa: BLE001
            assert isinstance(exc, (SplitError, rec.RecipeRefused, ValueError, KeyError)), exc
        else:
            raise AssertionError("a foreign split_manifest.json must be refused")
        assert _json.loads((ck4 / "split_manifest.json").read_text()) == {"format": "x"}
        try:
            rec.ensure_recipe_manifest(None, strip_roots=(env.strips,), holdout_root=env.strips, holdout_blocks=1, seed=0,
                                       ssl_pretrain=True)
        except rec.RecipeRefused as exc:
            assert "--ckpt-dir" in str(exc)
        else:
            raise AssertionError("no --ckpt-dir must refuse")
        # the split is the trainer's own for the --holdout-blocks given, not a fixed one
        h1 = {(r["source"], int(r["i"])) for r in rec.legacy_split_rows((env.strips,), env.strips, holdout_blocks=1)[1]}
        h2 = {(r["source"], int(r["i"])) for r in rec.legacy_split_rows((env.strips,), env.strips, holdout_blocks=2)[1]}
        assert h1 != h2 and h1 and h2, (len(h1), len(h2))
        assert rec.ensure_recipe_manifest(env.base / "ck5", strip_roots=(env.strips,), holdout_root=env.strips, holdout_blocks=2,
                                          seed=9, ssl_pretrain=True, git_sha="2" * 40) is True
        from python.train.trainer2_split import _load

        got = {(str(k[0]), int(k[1])) for k in _load(env.base / "ck5" / "split_manifest.json", "split manifest")["holdout"]}
        assert got == h2, (len(got), len(h2))
        # the git commit comes from the checkout, hex only; a failed or odd `git` answer refuses
        import subprocess

        real_run = subprocess.run
        for rc, out in ((0, "not-a-sha\n"), (1, "0" * 40 + "\n"), (0, "\n"), (0, "xyz" * 5)):
            subprocess.run = lambda *a, _rc=rc, _out=out, **k: SimpleNamespace(returncode=_rc, stdout=_out, stderr="")
            try:
                rec._repo_git_sha()
            except rec.RecipeRefused:
                pass
            else:
                raise AssertionError(f"git answer ({rc}, {out!r}) must refuse")
            finally:
                subprocess.run = real_run
        sha = rec._repo_git_sha()
        assert len(sha) >= 7 and all(c in "0123456789abcdef" for c in sha), sha
    print("manifest: split + run manifests pre-registered by the real path before stage 1 (create-only, re-roll and foreign split refused) OK")



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
        check_camera_only_deploy_export,
        check_more_pins,
        check_balanced_loss_keeps_extras,
        check_recipe_stages_train_on_fixture,
        check_recipe_manifest_required,
        check_aux_eval_numbers,
        check_measure_aux_unit,
        check_default_ssl_stage_recipe_wiring,
        check_recipe_stage_exit_is_owned_by_the_recipe,
        check_eval_series_and_exit_all,
        check_stage7_lr_multipliers_pinned,
        check_stop_path_keeps_old_checkpoints,
        check_ssl_epoch_losses_and_stop,
        check_runner_refuses_before_training,
        check_runner_exports_once_after_the_last_stage,
        check_runner_ssl_rows_must_be_manifest_train_rows,
        check_runner_preregisters_the_manifest,
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
