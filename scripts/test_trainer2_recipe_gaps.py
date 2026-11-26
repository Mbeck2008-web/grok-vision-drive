#!/usr/bin/env python3
"""Trainer 2.0 PR 6 (#269) r3 gaps: tests the recipe / CLI suites could not see.

A separate script (not an edit of ``test_trainer2_recipe.py``) so it does not collide with stage 2b's own test edits.
Each check names the mutant it kills; the r3 report is ``/workspace/skeptic2/verdicts/r3_269.md``.

* ``check_aux_eval_gt_only_class``      mIoU keeps a class only the ground truth has (``sem_union_pred``)
* ``check_aux_eval_inf_gt_depth``       +inf GT depth is not scored, AbsRel stays finite (``absrel_finite``)
* ``check_stages_change_weights_at_real_lr`` stages 1 / 2 / 7 move weights at lr 1e-5 (``noop_step``)
* ``check_manifest_records_real_git_sha``    the pre-registered manifest holds ``git rev-parse HEAD`` (``man_sha_const``)
* ``check_export_records_ckpt_sha256``       the exported ONNX carries the checkpoint's real sha256 (``export_meta_sha``)
* ``check_export_race_is_create_only``       a file that appears mid-export is never replaced (``export_replace_overwrites``)
* ``check_aux_eval_only_on_recipe_runs``     non-recipe runs get no AuxEval keys (``aux_ev_recipe_only``)
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from gvd_test_env import skip_check  # noqa: E402

import test_trainer2_recipe as base_tests  # noqa: E402


def check_aux_eval_gt_only_class() -> None:
    """A class that exists only in the ground truth (never predicted) counts as IoU 0 in the mIoU."""
    from python.train.aux_eval import AuxEval

    acc = AuxEval()
    logits = np.zeros((3, 1, 2))
    logits[1, 0, :] = 5.0  # the model says class 1 everywhere
    acc.update_sem(logits, np.array([[1, 2]]), np.array([[1.0, 1.0]]))
    # class 1: pred {0, 1} gt {0} -> 1/2; class 2: pred {} gt {1} -> 0/1 (union > 0, never predicted). mean 0.25, not 0.5
    assert acc.union == {1: 2, 2: 1} and acc.inter == {1: 1, 2: 0}, (acc.union, acc.inter)
    assert abs(acc.miou() - 0.25) < 1e-12, acc.miou()
    # pooled over ticks: the GT-only class keeps its zero inter while its union grows
    acc.update_sem(logits, np.array([[2, 2]]), np.array([[1.0, 1.0]]))
    assert acc.union == {1: 4, 2: 3} and acc.inter == {1: 1, 2: 0}, (acc.union, acc.inter)
    assert abs(acc.miou() - (0.25 + 0.0) / 2) < 1e-12, acc.miou()
    print("AuxEval: a class only the ground truth has counts as IoU 0 in the mIoU OK")


def check_aux_eval_inf_gt_depth() -> None:
    """A +inf GT depth cell is not scored (it would make |p - g| / g NaN and poison the pooled AbsRel)."""
    from python.train.aux_eval import AuxEval

    acc = AuxEval()
    gt = np.array([[2.0, np.inf, np.nan, 4.0]])
    pred = np.array([[3.0, 5.0, 5.0, 2.0]])
    acc.update_depth(pred, gt, np.ones((1, 4)))
    # only the two finite, positive GT cells: |3-2|/2 = .5 and |2-4|/4 = .5
    assert acc.rel_n == 2 and np.isfinite(acc.rel_sum), (acc.rel_n, acc.rel_sum)
    assert abs(acc.absrel() - 0.5) < 1e-12, acc.absrel()
    inf_only = AuxEval()
    inf_only.update_depth(np.array([[1.0]]), np.array([[np.inf]]), np.array([[1.0]]))
    assert inf_only.rel_n == 0 and inf_only.absrel() is None  # nothing scored is None, not NaN
    print("AuxEval: +inf / NaN GT depth is skipped, AbsRel stays finite OK")


def _ready(name: str, *, onnx: bool = False, git: bool = False) -> bool:
    if base_tests._torch() is None:
        skip_check(name, "torch not installed")
        return False
    try:
        import cv2  # noqa: F401
        if onnx:
            import onnx as _onnx  # noqa: F401
    except ImportError:
        skip_check(name, "cv2 / onnx not installed (CI installs both)")
        return False
    if git:
        import shutil

        if shutil.which("git") is None:
            skip_check(name, "git not installed")
            return False
    return True


def _train_stage(torch, train_scene, strips, base, sid, *, lr, recipe="trainer2", epochs=1):
    """One real ``train_directory`` stage on the fixture (CPU). Returns ``(stats, net, weights_before_first_step)``."""
    rec = base_tests._recipe()
    kw = rec.stage_train_kwargs(sid)
    holder: dict = {}
    real = train_scene._backward_and_maybe_step

    def wrap(net, *a, **k):
        if "before" not in holder:  # state at the first optimizer step: the stage's starting point
            holder["net"] = net
            holder["before"] = {n: t.detach().clone() for n, t in net.state_dict().items()}
        return real(net, *a, **k)

    train_scene._backward_and_maybe_step = wrap
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            stats = train_scene.train_directory(
                strips, epochs=epochs, lr=lr, capability=None, export_path=None, free_vram_bytes=0,
                free_ram_bytes=8_000_000_000, resume=False, open_panel=False, autostart=True, cams=("main",),
                heads=kw["heads"], aux_depth=kw["aux_depth"], aux_sem=kw["aux_sem"], augment=False, workers=0,
                min_valid=0.0, holdout_blocks=1, scale=1.0, status_root=base / f"st_{sid}_{recipe or 'off'}",
                ckpt_dir=base / f"ck_{sid}_{recipe or 'off'}", exact_epochs=epochs, recipe=recipe,
                recipe_stage=sid if recipe else None, aux_measure=bool(recipe))
    finally:
        train_scene._backward_and_maybe_step = real
    assert "before" in holder, f"stage {sid} never took an optimizer step"
    return stats, holder["net"], holder["before"]


def _moved(net, before, prefixes) -> float:
    """Largest absolute change of any tensor whose name starts with one of ``prefixes``."""
    after = net.state_dict()
    diffs = [float((after[n].float() - before[n].float()).abs().max()) for n in before if n.startswith(prefixes)]
    assert diffs, f"no tensors named {prefixes}"
    return max(diffs)


def check_stages_change_weights_at_real_lr() -> None:
    """Stages 1 / 2 / 7 really move weights at lr 1e-5 (the full-curriculum CLI test runs at 1e-9 and cannot see a
    no-op optimizer step). Only the heads a stage trains move: stage 2 gives the aux heads no loss, stage 7 gives them 0.1.
    """
    if not _ready("check_stages_change_weights_at_real_lr"):
        return
    torch = base_tests._torch()
    from python.train import train_scene

    lr = 1e-5
    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        strips = base_tests._fixture(base)
        moves = {}
        for sid in ("1", "2", "7"):
            stats, net, before = _train_stage(torch, train_scene, strips, base, sid, lr=lr)
            assert int(stats["windows"]) > 0, (sid, stats.get("windows"))
            moves[sid] = {
                "stem": _moved(net, before, ("stem.", "backbone.", "trunk.")) if any(
                    n.startswith(("stem.", "backbone.", "trunk.")) for n in before) else None,
                "aux": _moved(net, before, ("aux_depth_head.", "aux_sem_head.")),
                "lane": _moved(net, before, ("lane_gt_head.",)) if any(n.startswith("lane_gt_head.") for n in before) else None,
            }
        print("max |dW| at lr 1e-5:", moves)
    # every moved amount is a real, small Adam-sized change (lr 1e-5 over a handful of steps), never zero, never huge
    assert 0.0 < moves["1"]["aux"] < 0.05, moves
    assert moves["2"]["lane"] is not None and 0.0 < moves["2"]["lane"] < 0.05, moves
    assert moves["2"]["aux"] == 0.0, f"stage 2 gives the aux heads no loss, they must not move: {moves}"
    assert moves["7"]["lane"] is not None and 0.0 < moves["7"]["lane"] < 0.05, moves
    for sid in ("1", "2", "7"):  # the shared stem trains in every supervised stage
        assert moves[sid]["stem"] is not None and 0.0 < moves[sid]["stem"] < 0.05, (sid, moves)
    assert moves["1"]["lane"] == 0.0, f"stage 1 trains no lane head: {moves}"
    assert 0.0 < moves["7"]["aux"] < 0.05, f"stage 7 trains the aux heads (weight 0.1): {moves}"
    print("stages 1 / 2 / 7 move exactly the heads they train at lr 1e-5 OK")


def check_aux_eval_only_on_recipe_runs() -> None:
    """AuxEval (depth AbsRel / sem mIoU on every holdout pass) is a recipe-run feature: the same aux-head run without
    ``--recipe trainer2`` records exactly the holdout keys it did on main, no extra keys and no extra CPU work.
    """
    if not _ready("check_aux_eval_only_on_recipe_runs"):
        return
    torch = base_tests._torch()
    from python.train import train_scene

    keys = ("depth_absrel", "sem_miou", "depth_cells", "sem_cells")
    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        strips = base_tests._fixture(base)
        stats_off, _, _ = _train_stage(torch, train_scene, strips, base, "1", lr=1e-5, recipe="")
        stats_on, _, _ = _train_stage(torch, train_scene, strips, base, "1", lr=1e-5, recipe="trainer2")
    off, on = stats_off["holdout_evals"], stats_on["holdout_evals"]
    assert off and on, (off, on)
    for entry in off:
        assert "lane_rms" in entry and "holdout_loss" in entry, entry  # the pass did run and score
        assert not any(k in entry for k in keys), f"a non-recipe run must not record AuxEval keys: {entry}"
    for entry in on:
        assert all(k in entry for k in keys) and entry["depth_cells"] > 0, entry
    assert set(on[0]) - set(off[0]) == set(keys), (sorted(on[0]), sorted(off[0]))
    print("AuxEval keys are recorded on recipe runs only; a non-recipe aux-head run keeps main's holdout record OK")


def _head_sha() -> str:
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().lower()


def check_manifest_records_real_git_sha() -> None:
    """The pre-registered run manifest holds this checkout's real ``git rev-parse HEAD``, not a constant."""
    if not _ready("check_manifest_records_real_git_sha", git=True):
        return
    from python.train import trainer2_recipe as rec
    from python.train.trainer2_split import require_preregistered

    head = _head_sha()
    assert len(head) == 40 and set(head) <= set("0123456789abcdef") and head != "0" * 40, head
    assert rec._repo_git_sha() == head, (rec._repo_git_sha(), head)
    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        strips = base_tests._fixture(base)
        ck = base / "ck"
        # the CLI path: no git_sha given, so ensure_recipe_manifest reads the repo's commit
        assert rec.ensure_recipe_manifest(ck, strip_roots=(strips,), holdout_root=strips, holdout_blocks=1, seed=0,
                                          ssl_pretrain=False) is True
        run = require_preregistered(ck)
        assert run["git_sha"] == head, (run["git_sha"], head)
        # an explicit sha is recorded as given (the override is for tests / replays, never the default)
        ck2 = base / "other_arm" / "ck2"  # its own registry folder: a second seed in `ck`'s folder would be a re-roll
        rec.ensure_recipe_manifest(ck2, strip_roots=(strips,), holdout_root=strips, holdout_blocks=1, seed=1,
                                   ssl_pretrain=False, git_sha="a" * 40)
        assert require_preregistered(ck2)["git_sha"] == "a" * 40
    print("run manifest records the checkout's real git sha (CLI path), and only an explicit override replaces it OK")


def _stage7_ckpt(torch, base: Path):
    """A stage-7-shaped checkpoint (ctl + aux + road_edge + stop_line) with perturbed weights."""
    from python.train.model_size import scene_net_kwargs
    from python.train.scene_net import SceneNet

    heads = ["lanes", "objects", "signs", "lane_gt", "road_edge", "stop_line", "ctl"]
    net = SceneNet(cams=("main",), ctl=True, aux_sensors=True, road_edge=True, stop_line=True, **scene_net_kwargs("small"))
    with torch.no_grad():
        for prm in net.parameters():
            prm.add_(torch.randn_like(prm) * 1e-2)
    return base_tests._ckpt_of(net, base / "s7.ckpt_ep3_step7.npz", ctl=True, heads=heads, run="r3-run", scale=1.0)


def check_export_records_ckpt_sha256() -> None:
    """The exported ONNX names the checkpoint it came from by file, sha256, epoch, step, run and scale."""
    if not _ready("check_export_records_ckpt_sha256", onnx=True):
        return
    import onnx

    torch = base_tests._torch()
    rec = base_tests._recipe()
    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        ck = _stage7_ckpt(torch, base)
        out = base / "deploy.onnx"
        rec.export_camera_only_deploy(ck, out, preset="small")
        props = {p.key: p.value for p in onnx.load(str(out)).metadata_props}
        want = hashlib.sha256(ck.read_bytes()).hexdigest()
        assert props.get("gvd_ckpt_sha256") == want and len(want) == 64, (props.get("gvd_ckpt_sha256"), want)
        assert props.get("gvd_ckpt_file") == ck.name, props
        assert props.get("gvd_ckpt_epoch") == "3" and props.get("gvd_ckpt_step") == "7", props
        assert props.get("gvd_ckpt_run") == "r3-run" and float(props.get("gvd_scale")) == 1.0, props
        # another checkpoint (other random weights) records its own, different hash: the field is the file's, not a constant
        (base / "two").mkdir()
        ck2 = _stage7_ckpt(torch, base / "two")
        out2 = base / "deploy2.onnx"
        rec.export_camera_only_deploy(ck2, out2, preset="small")
        props2 = {p.key: p.value for p in onnx.load(str(out2)).metadata_props}
        want2 = hashlib.sha256(ck2.read_bytes()).hexdigest()
        assert want2 != want and props2["gvd_ckpt_sha256"] == want2, (props2["gvd_ckpt_sha256"], want2, want)
    print("exported ONNX carries the checkpoint's real sha256 / file / epoch / step / run / scale OK")


def check_export_race_is_create_only() -> None:
    """``--export`` is create-only even when the file appears AFTER the up-front exists() check (another process wins
    the race while the graph is being written and checked): refused cleanly, the other file untouched, no partial left.
    """
    if not _ready("check_export_race_is_create_only", onnx=True):
        return
    torch = base_tests._torch()
    rec = base_tests._recipe()
    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        ck = _stage7_ckpt(torch, base)
        out = base / "deploy.onnx"
        theirs = b"another process's model, written during our export"
        real = rec.check_export_fits_deploy

        def racing_check(*a, **k):
            result = real(*a, **k)
            out.write_bytes(theirs)  # lands between the exists() check and the final link
            return result

        rec.check_export_fits_deploy = racing_check
        try:
            rec.export_camera_only_deploy(ck, out, preset="small")
        except rec.RecipeRefused as exc:
            assert "create-only" in str(exc), exc
        else:
            raise AssertionError("a file that appeared mid-export must be refused, not replaced")
        finally:
            rec.check_export_fits_deploy = real
        assert out.read_bytes() == theirs, "the racing writer's file was replaced"
        assert not [p for p in base.iterdir() if ".partial" in p.name], sorted(p.name for p in base.iterdir())
        # without the race the same call exports normally (the refusal above was the race, not a broken export)
        out2 = base / "deploy_ok.onnx"
        assert rec.export_camera_only_deploy(ck, out2, preset="small")["ok"] is True and out2.is_file()
    print("export race: a file that appears mid-export is refused as create-only, never replaced, no .partial left OK")


CHECKS = (
    check_aux_eval_gt_only_class,
    check_aux_eval_inf_gt_depth,
    check_stages_change_weights_at_real_lr,
    check_aux_eval_only_on_recipe_runs,
    check_manifest_records_real_git_sha,
    check_export_records_ckpt_sha256,
    check_export_race_is_create_only,
)


def main() -> int:
    failed = 0
    for check in CHECKS:
        try:
            check()
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            failed += 1
            print(f"FAIL {check.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
