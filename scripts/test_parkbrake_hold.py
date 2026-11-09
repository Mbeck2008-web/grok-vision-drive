#!/usr/bin/env python3
"""Only a real hold (stop / AEB / parked) maps to the parking brake.

tech_control_kwargs and Lua applyInputs used to turn every brake above 0.01
into parkingbrake=1. A 2 m/s speed error is brake 0.1; the AEB warn band is
0.35. Neither is a hold. Partials must become a proportional SERVICE brake
while moving so the car still decelerates. A service brake with no throttle
at rest is arcade reverse (run 8), so near-zero speed coasts instead of
sending that pedal. brake >= TECH_HOLD_BRAKE stays a parking-brake hold.

    py -3 scripts/test_parkbrake_hold.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import (  # noqa: E402
    SERVICE_BRAKE_MIN_SPEED_MPS,
    TECH_HOLD_BRAKE,
    is_arcade_reverse_hold,
    plan_command,
    stop_command,
    tech_control_kwargs,
)


def _assert_not_reverse(kw: dict) -> None:
    # tech_control_kwargs may emit a service brake while moving; that is
    # forward deceleration. A park hold must release the service pedal.
    assert "gear" not in kw, kw
    if float(kw.get("parkingbrake") or 0.0) >= 0.99:
        assert float(kw.get("brake") or 0.0) == 0.0, kw
        assert float(kw.get("throttle") or 0.0) == 0.0, kw


def check_partial_brake_is_service_while_moving() -> None:
    for brake in (0.1, 0.35, 0.6, 0.8):
        kw = tech_control_kwargs(0.0, 0.0, brake, speed_mps=12.0)
        _assert_not_reverse(kw)
        assert float(kw["parkingbrake"]) == 0.0, (brake, kw)
        assert float(kw["throttle"]) == 0.0, kw
        assert abs(float(kw["brake"]) - float(brake)) < 1e-9, (brake, kw)
        print(f"partial brake {brake} @12 m/s: service brake, no park OK")


def check_partial_brake_at_rest_does_not_reverse() -> None:
    for brake in (0.1, 0.35, 0.6, 0.8):
        kw = tech_control_kwargs(0.0, 0.0, brake, speed_mps=0.0)
        _assert_not_reverse(kw)
        assert float(kw["parkingbrake"]) == 0.0, (brake, kw)
        assert float(kw["throttle"]) == 0.0, kw
        assert float(kw["brake"]) == 0.0, (brake, kw)
        print(f"partial brake {brake} @0 m/s: coast (no reverse pedal) OK")
    assert SERVICE_BRAKE_MIN_SPEED_MPS > 0.0
    # Raw no-throttle service brake is still the arcade reverse shape.
    assert is_arcade_reverse_hold({"throttle": 0.0, "brake": 0.8, "parkingbrake": 0.0})
    edge = tech_control_kwargs(0.0, 0.0, 0.35, speed_mps=SERVICE_BRAKE_MIN_SPEED_MPS)
    assert float(edge["brake"]) == 0.0 and float(edge["parkingbrake"]) == 0.0, edge
    just = tech_control_kwargs(
        0.0, 0.0, 0.35, speed_mps=SERVICE_BRAKE_MIN_SPEED_MPS + 1e-6
    )
    assert abs(float(just["brake"]) - 0.35) < 1e-9, just


def check_full_hold_parks() -> None:
    for brake in (1.0, TECH_HOLD_BRAKE, 0.995):
        kw = tech_control_kwargs(0.2, 0.0, brake)
        _assert_not_reverse(kw)
        assert float(kw["parkingbrake"]) == 1.0, (brake, kw)
        assert float(kw["throttle"]) == 0.0, kw
        assert float(kw["brake"]) == 0.0, kw
        print(f"hold brake {brake}: parking brake OK")


def check_consumers_get_service_brake() -> None:
    path = [{"x": 0.0, "y": float(i), "z": 0.0} for i in range(12)]
    # ego 12 m/s, target 10 m/s → brake 0.1
    cmd = plan_command(
        path_ego=path,
        planner={"target_v": 10.0, "aeb": "off"},
        ego_speed_mps=12.0,
        seq=1,
    )
    assert abs(cmd.brake - 0.1) < 1e-9, cmd
    kw = tech_control_kwargs(cmd.steer, cmd.throttle, cmd.brake, speed_mps=12.0)
    assert float(kw["parkingbrake"]) == 0.0, kw
    assert abs(float(kw["brake"]) - 0.1) < 1e-9, kw
    warn = plan_command(
        path_ego=path,
        planner={"target_v": 10.0, "aeb": "warn"},
        ego_speed_mps=12.0,
        seq=2,
    )
    assert abs(warn.brake - 0.35) < 1e-9, warn
    wkw = tech_control_kwargs(warn.steer, warn.throttle, warn.brake, speed_mps=12.0)
    assert float(wkw["parkingbrake"]) == 0.0, wkw
    assert abs(float(wkw["brake"]) - 0.35) < 1e-9, wkw
    stop_ap = plan_command(
        path_ego=path,
        planner={"target_v": 0.0, "aeb": "off"},
        ego_speed_mps=20.0,
        seq=3,
    )
    assert abs(stop_ap.brake - 0.6) < 1e-9, stop_ap
    skw = tech_control_kwargs(
        stop_ap.steer, stop_ap.throttle, stop_ap.brake, speed_mps=20.0
    )
    assert float(skw["parkingbrake"]) == 0.0, skw
    assert abs(float(skw["brake"]) - 0.6) < 1e-9, skw
    stop = stop_command()
    hkw = tech_control_kwargs(stop.steer, stop.throttle, stop.brake)
    assert float(hkw["parkingbrake"]) == 1.0, hkw
    print("speed-error 0.1 / AEB warn 0.35 / stop approach 0.6 → service; stop parks OK")


def check_lua_threshold_and_service_path() -> None:
    lua = (ROOT / "beamng_mod" / "lua" / "ge" / "extensions" / "gvd" / "main.lua").read_text(
        encoding="utf-8"
    )
    assert "HOLD_BRAKE" in lua, "Lua applyInputs must name the hold threshold"
    assert "0.99" in lua, "Lua hold threshold must match TECH_HOLD_BRAKE"
    assert "SERVICE_MIN_SPEED" in lua or "service" in lua.lower(), (
        "Lua applyInputs must speed-gate the service pedal"
    )
    # Must not zero out every partial into a coast unconditionally.
    assert "VE_APPLY_FMT" in lua
    print("Lua hold threshold + service path present OK")


def main() -> None:
    check_partial_brake_is_service_while_moving()
    check_partial_brake_at_rest_does_not_reverse()
    check_full_hold_parks()
    check_consumers_get_service_brake()
    check_lua_threshold_and_service_path()
    print("test_parkbrake_hold: OK")


if __name__ == "__main__":
    main()
