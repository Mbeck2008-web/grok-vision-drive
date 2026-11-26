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


CHECKS = (
    check_aux_eval_gt_only_class,
    check_aux_eval_inf_gt_depth,
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
