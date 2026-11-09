#!/usr/bin/env python3
"""One seen lane line is not a lane, and modular honors the published vetoes.

The live Hough planner used to copy a single line 3.5 m sideways and steer
that center. Paint labels refuse that. path_conf of a drivable plan was also
floored at 0.5, so the 0.15 path veto could never fire, and modular ignored
evaluate_veto entirely.

    py -3 scripts/test_hough_no_invent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.e2e import E2EIntent  # noqa: E402
from python.planning.path_predictor import _confidence, predict_path  # noqa: E402
from python.runtime.shadow import ShadowConfig, shadow_tick  # noqa: E402


class _StubE2E:
    def forward(self, *args, **kwargs) -> E2EIntent:
        del args, kwargs
        return E2EIntent(steer=0.2, throttle=0.3, brake=0.0, accel=0.3, ok=True, backend="stub")


def _left_only():
    return [[{"x": -1.75, "y": float(y), "z": 0.0} for y in range(0, 42, 2)]]


def _path():
    return [{"x": 0.0, "y": float(i)} for i in range(12)]


def check_one_sided_is_not_drivable() -> None:
    plan = predict_path(lanes_bev=_left_only(), lane_conf=0.9, ego_speed_mps=8.0)
    src = str(plan.lane_source or "")
    assert "one_left" not in src and "one_right" not in src, src
    assert plan.drivable is False, (plan.drivable, src, plan.path_ego[:3])
    print("one-sided line is not a steered lane OK")


def check_path_conf_can_fall_below_veto() -> None:
    low = _confidence("lanes", 0.0, extended=False)
    ext = _confidence("lanes", 0.0, extended=True)
    assert low < 0.15, low
    assert ext < 0.15, ext
    assert _confidence("lanes", 0.05, extended=False) < 0.25
    print(f"path_conf at lane_conf=0 is {low:.3f} (below 0.15) OK")


def check_modular_honors_confidence_veto() -> None:
    tick = shadow_tick(
        policy="modular",
        engaged=True,
        heartbeat_ok=True,
        path_debug_preview=False,
        allow_preview_drive=False,
        path_ego=_path(),
        planner={"target_v": 10.0, "aeb": "off"},
        ego_speed_mps=8.0,
        lane_conf=0.05,
        path_conf=0.5,
        seq=3,
        e2e_policy=_StubE2E(),
        cfg=ShadowConfig(),
    )
    assert tick.veto_reason == "low_lane_conf", tick.veto_reason
    assert tick.applied.reason != "ok", tick.applied
    assert tick.applied.throttle == 0.0, tick.applied
    print("modular low_lane_conf vetoes the drive OK")

    tick2 = shadow_tick(
        policy="modular",
        engaged=True,
        heartbeat_ok=True,
        path_debug_preview=False,
        allow_preview_drive=False,
        path_ego=_path(),
        planner={"target_v": 10.0, "aeb": "off"},
        ego_speed_mps=8.0,
        lane_conf=0.9,
        path_conf=0.05,
        seq=4,
        e2e_policy=_StubE2E(),
        cfg=ShadowConfig(),
    )
    assert tick2.veto_reason == "low_path_conf", tick2.veto_reason
    assert tick2.applied.reason != "ok", tick2.applied
    print("modular low_path_conf vetoes the drive OK")


def main() -> None:
    check_one_sided_is_not_drivable()
    check_path_conf_can_fall_below_veto()
    check_modular_honors_confidence_veto()
    print("test_hough_no_invent: OK")


if __name__ == "__main__":
    main()
