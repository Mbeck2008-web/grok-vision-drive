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
                metrics={"lane_rms": 1.2, "lane_r": 0.8},
                evals=[1.2],
                epochs=1,
            )
        metrics = {"lane_rms": 1.0, "lane_r": 0.8, "ssl_pretext": 0.5, "depth_absrel": 0.2, "sem_miou": 0.4}
        return rec.StageResult(ckpt=f"stage{stage.id}.ckpt", metrics=metrics, evals=[1.0, 0.995, 0.99, 0.985], epochs=4)

    try:
        rec.run_curriculum(train_fn, log=logs.append)
    except rec.CheckpointRejected as exc:
        assert "lane_rms" in str(exc) or "rejected" in str(exc).lower()
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
    for flag in ("recipe", "stage", "accum", "kd"):
        assert flag in got, f"missing --{flag}"
    print(f"recipe off: {len(want)} pre-existing train_scene argparse defaults == main 8ce34d5 pin OK "
          f"({len(added)} newer flag(s) not pinned: {added})")


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
