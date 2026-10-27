#!/usr/bin/env python3
"""A held plan reports its pedals at the current speed.

When the camera bytes repeat, ModularPerception.tick holds the last plan and
dead-reckons its path. The published pred_brake, path_length_m and blinker
must come from that moved path at this tick's ego speed, the same way a fresh
tick computes them. A plan first fitted at 0 m/s must not keep pred_brake 0
once the car is doing 5.5 m/s (check_run_vision_e2e, CI run 40470586816).

blinker_for_path takes no speed. It reads the path's geometry and the
prediction, so it changes on a held tick only through the moved path.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from python.perception.pipeline import ModularPerception  # noqa: E402
from python.planning import path_predictor  # noqa: E402
from python.planning.path_predictor import blinker_for_path, path_length_m  # noqa: E402

ECHO_SPEED = 5.5
# The m6 stop law with target_v 0: brake = min(0.6, 0.05 * speed).
EXPECTED_STOP_BRAKE = min(0.6, 0.05 * ECHO_SPEED)
DT = 0.05


def _black() -> np.ndarray:
    """The stub camera: no lanes, so the plan is an empty-view stop."""
    return np.zeros((96, 160, 3), dtype=np.uint8)


def _road() -> np.ndarray:
    """Two painted lines: a lane pair with a ~36 m path starting at y=0."""
    import cv2

    img = np.full((96, 160, 3), (40, 42, 44), dtype=np.uint8)
    cv2.line(img, (30, 90), (70, 30), (40, 210, 235), 3)
    cv2.line(img, (130, 90), (90, 30), (245, 245, 245), 3)
    return img


def _perc() -> ModularPerception:
    return ModularPerception(allow_synthetic=False, detector_id="empty")


def _tick(perc: ModularPerception, frame: np.ndarray, speed: float, dt: float = DT):
    return perc.tick(frame, stitch_bgr=frame, ego_speed_mps=speed, dt_s=dt)


def _held_after_zero(frame: np.ndarray):
    """Fresh tick at 0 m/s, then the same bytes at ECHO_SPEED: the held tick."""
    perc = _perc()
    first = _tick(perc, frame, 0.0)
    fresh_plan = perc._last_plan
    held = _tick(perc, frame, ECHO_SPEED)
    assert "stale" in str(held.planner["lane_source"]), held.planner
    return first, fresh_plan, held


def check_held_tick_brake_follows_speed() -> None:
    first, _, held = _held_after_zero(_black())
    assert first.planner["pred_brake"] == 0.0, first.planner
    assert held.planner["lane_source"] == "stale", held.planner
    assert float(held.planner["target_v"]) == 0.0, held.planner
    got = float(held.planner["pred_brake"])
    assert abs(got - EXPECTED_STOP_BRAKE) < 1e-9, (got, EXPECTED_STOP_BRAKE, held.planner)
    assert held.planner["path_length_m"] == path_length_m(held.path_ego), held.planner


def check_held_tick_path_length_and_blinker() -> None:
    first, fresh_plan, held = _held_after_zero(_road())
    assert len(first.path_ego) >= 4, first.path_ego
    # Dead reckoning moved the path 5.5 m/s * 0.05 s back and dropped the point at y=0.
    assert len(held.path_ego) < len(first.path_ego), (len(held.path_ego), len(first.path_ego))
    want_len = path_length_m(held.path_ego)
    assert abs(want_len - float(first.planner["path_length_m"])) > 1e-3, (want_len, first.planner)
    assert held.planner["path_length_m"] == want_len, (held.planner["path_length_m"], want_len)
    assert held.planner["blinker"] == blinker_for_path(held.path_ego, fresh_plan.prediction), held.planner


def check_held_matches_fresh_same_plan_and_speed() -> None:
    """Same plan, same speed: the held tick and a fresh tick publish the same pedals."""
    for name, frame in (("black", _black()), ("road", _road())):
        _, fresh_plan, held = _held_after_zero(frame)
        # The fresh-tick path: drive_plan_from_lanes finishes with _expose_pedals.
        same = path_predictor._expose_pedals(replace(fresh_plan, path_ego=list(held.path_ego)), ECHO_SPEED)
        for key in ("pred_brake", "path_length_m", "blinker"):
            assert held.planner[key] == getattr(same, key), (name, key, held.planner[key], getattr(same, key))

    # End to end on the stub view: a held tick at 5.5 m/s against a fresh pipeline's
    # first tick at 5.5 m/s. Both paths are empty, so the plan is the same.
    _, _, held = _held_after_zero(_black())
    fresh = _tick(_perc(), _black(), ECHO_SPEED)
    assert held.path_ego == fresh.path_ego, (held.path_ego, fresh.path_ego)
    for key in ("pred_brake", "path_length_m", "blinker", "target_v"):
        assert held.planner[key] == fresh.planner[key], (key, held.planner[key], fresh.planner[key])


def main() -> None:
    failed: list[str] = []
    for fn in (
        check_held_tick_brake_follows_speed,
        check_held_tick_path_length_and_blinker,
        check_held_matches_fresh_same_plan_and_speed,
    ):
        try:
            fn()
            print(f"{fn.__name__}: ok", flush=True)
        except AssertionError as exc:
            failed.append(fn.__name__)
            print(f"{fn.__name__}: FAIL {exc!r}", flush=True)
    if failed:
        raise SystemExit("test_held_plan_pedals failed: " + ", ".join(failed))
    print("test_held_plan_pedals: OK")


if __name__ == "__main__":
    main()
