#!/usr/bin/env python3
"""CIPV TTC uses signed relative (closing) speed, not abs(lead speed).

The tracker stores speed_mps as along-track velocity relative to ego,
negative when the gap is closing. Taking abs() made a matched-speed lead
(filtered speed ~0) look stopped: TTC = gap / ego_speed, so a car 1.2 s
ahead at the same speed forced aeb=brake. Opening and closing at the same
magnitude were identical.

    py -3 scripts/test_cipv_signed.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.perception.cipv import select_cipv  # noqa: E402
from python.perception.pipeline import fuse_planner_cipv  # noqa: E402
from python.perception.cipv import CipvResult  # noqa: E402


def _lead(y: float, speed: float) -> dict:
    return {
        "id": 1,
        "class": "vehicle",
        "x": 0.0,
        "y": y,
        "yaw": 1.5708,
        "speed_mps": speed,
        "yaw_rate": 0.0,
        "hits": 5,
        "misses": 0,
        "age": 5,
        "conf": 0.9,
    }


def check_matched_speed_is_not_a_stopped_car() -> None:
    # Review probe: y=8 m, ego=8 m/s, relative speed 0 → used to be TTC 1.0, brake.
    res = select_cipv([_lead(8.0, 0.0)], ego_speed_mps=8.0)
    assert res.aeb == "off", res
    assert res.ttc_lead is None, res
    print("matched-speed lead: aeb off OK")


def check_opening_is_not_closing() -> None:
    closing = select_cipv([_lead(20.0, -5.0)], ego_speed_mps=10.0)
    opening = select_cipv([_lead(20.0, 5.0)], ego_speed_mps=10.0)
    assert closing.aeb != opening.aeb or closing.ttc_lead != opening.ttc_lead, (closing, opening)
    assert opening.aeb == "off", opening
    assert opening.ttc_lead is None, opening
    # 20 m / 5 m/s closing = 4.0 s → off (warn band starts at 2.5 s)
    assert closing.ttc_lead is not None and abs(closing.ttc_lead - 4.0) < 1e-6, closing
    assert closing.aeb == "off", closing
    hard = select_cipv([_lead(5.0, -5.0)], ego_speed_mps=10.0)
    assert hard.ttc_lead is not None and abs(hard.ttc_lead - 1.0) < 1e-6, hard
    assert hard.aeb == "brake", hard
    print("signed relative speed: opening != closing OK")


def check_planner_refused_hard_stop_is_not_overridden() -> None:
    # Planner saw a lead, nulled a short TTC, and refused aeb=brake.
    cipv = CipvResult(track=_lead(8.0, -8.0), ttc_lead=1.0, aeb="brake")
    aeb, ttc, target = fuse_planner_cipv(
        plan_aeb="off",
        plan_ttc=None,
        plan_target_v=6.0,
        plan_stop_reason="lead",
        cipv=cipv,
    )
    assert aeb == "off", aeb
    assert ttc is None, ttc
    assert target == 6.0, target
    # No scene stop: CIPV may still fill in a missing lead.
    aeb2, ttc2, target2 = fuse_planner_cipv(
        plan_aeb="off",
        plan_ttc=None,
        plan_target_v=10.0,
        plan_stop_reason="none",
        cipv=cipv,
    )
    assert aeb2 == "brake" and target2 == 0.0, (aeb2, target2)
    assert ttc2 == 1.0, ttc2
    print("planner-refused hard stop is not overridden OK")


def main() -> None:
    check_matched_speed_is_not_a_stopped_car()
    check_opening_is_not_closing()
    check_planner_refused_hard_stop_is_not_overridden()
    print("test_cipv_signed: OK")


if __name__ == "__main__":
    main()
