#!/usr/bin/env python3
"""The holdout early stop needs at least 3 val points.

``holdout_stopped`` used to fall back to patience 0 below ``window`` points, so
a 2-point history whose latest score was no better than the best stopped the
run. Epoch 0 of a run was a collapsed constant prediction (val 0.7740), epoch 1
scored 0.7921, and the run stopped after epoch 1.

* 1 and 2 points never stop (with the train_scene arguments and the defaults);
* a 3-point plateau inside ``tol`` stops as before;
* 3 points still improving by more than ``tol`` do not stop;
* ``min_points`` is explicit and defaults to 3.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.train.continual import holdout_stopped  # noqa: E402

TRAIN_ARGS = {"tol": 0.02, "window": 3}


def check_short_histories() -> None:
    assert holdout_stopped([], **TRAIN_ARGS) is False
    assert holdout_stopped([0.7740], **TRAIN_ARGS) is False
    assert holdout_stopped([0.7740]) is False
    # The run that stopped after epoch 1: collapsed epoch 0, then a worse epoch 1.
    assert holdout_stopped([0.7740, 0.7921], **TRAIN_ARGS) is False
    assert holdout_stopped([0.7740, 0.7921]) is False
    print("short histories: 0, 1 and 2 points do not stop")


def check_three_points() -> None:
    # Plateau: the last 3 scores inside tol stop, as before.
    assert holdout_stopped([0.7740, 0.7921, 0.7850], **TRAIN_ARGS) is True
    assert holdout_stopped([2.333, 2.327, 2.320], **TRAIN_ARGS) is True
    # Still improving by more than tol over the window: keep going.
    assert holdout_stopped([0.90, 0.80, 0.70], **TRAIN_ARGS) is False
    assert holdout_stopped([0.7740, 0.7921, 0.7000], **TRAIN_ARGS) is False
    # Longer plateau logic is unchanged.
    flat = [2.333, 2.327, 2.290, 2.282, 2.280]
    assert holdout_stopped(flat[:3], **TRAIN_ARGS) is False
    assert holdout_stopped(flat, **TRAIN_ARGS) is True
    # The window=0 fallback (no scored rms) is patience 0 from the 3rd point.
    assert holdout_stopped([0.5, 0.4, 0.45]) is True
    assert holdout_stopped([0.5, 0.4, 0.35]) is False
    print("three points: plateau stops, improvement does not")


def check_min_points_param() -> None:
    sig = inspect.signature(holdout_stopped)
    assert sig.parameters["min_points"].default == 3
    assert holdout_stopped([0.5, 0.6], min_points=2) is True
    assert holdout_stopped([0.5, 0.6, 0.7], min_points=4) is False
    assert holdout_stopped([0.5, 0.6], min_points=0) is True  # never below 2 points
    assert holdout_stopped([0.5], min_points=0) is False
    print("min_points: explicit, default 3, floor 2")


def main() -> None:
    check_short_histories()
    check_three_points()
    check_min_points_param()
    print("holdout min points ok")


if __name__ == "__main__":
    main()
