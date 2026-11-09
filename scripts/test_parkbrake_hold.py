#!/usr/bin/env python3
"""Only a real hold (stop / AEB / parked) maps to the parking brake.

tech_control_kwargs and Lua applyInputs used to turn every brake above 0.01
into parkingbrake=1. A 2 m/s speed error is brake 0.1; the AEB warn band is
0.35. Neither is a hold. A service brake with no throttle is arcade reverse,
so a partial request coasts (throttle 0, service brake 0, parking brake 0).

    py -3 scripts/test_parkbrake_hold.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import (  # noqa: E402
    TECH_HOLD_BRAKE,
    is_arcade_reverse_hold,
    plan_command,
    stop_command,
    tech_control_kwargs,
)


def _assert_not_reverse(kw: dict) -> None:
    assert not is_arcade_reverse_hold(kw), kw
    assert float(kw.get("brake") or 0.0) == 0.0, kw


def check_partial_brake_is_not_park() -> None:
    for brake in (0.1, 0.35, 0.6, 0.8):
        kw = tech_control_kwargs(0.0, 0.0, brake, speed_mps=12.0)
        _assert_not_reverse(kw)
        assert float(kw["parkingbrake"]) == 0.0, (brake, kw)
        assert float(kw["throttle"]) == 0.0, kw
        print(f"partial brake {brake}: coast, no park OK")


def check_full_hold_parks() -> None:
    for brake in (1.0, TECH_HOLD_BRAKE, 0.995):
        kw = tech_control_kwargs(0.2, 0.0, brake)
        _assert_not_reverse(kw)
        assert float(kw["parkingbrake"]) == 1.0, (brake, kw)
        assert float(kw["throttle"]) == 0.0, kw
        print(f"hold brake {brake}: parking brake OK")


def check_speed_error_and_aeb_warn() -> None:
    # ego 12 m/s, target 10 m/s → brake 0.1
    cmd = plan_command(
        path_ego=[{"x": 0.0, "y": float(i), "z": 0.0} for i in range(12)],
        planner={"target_v": 10.0, "aeb": "off"},
        ego_speed_mps=12.0,
        seq=1,
    )
    assert abs(cmd.brake - 0.1) < 1e-9, cmd
    kw = tech_control_kwargs(cmd.steer, cmd.throttle, cmd.brake)
    assert float(kw["parkingbrake"]) == 0.0, kw
    warn = plan_command(
        path_ego=[{"x": 0.0, "y": float(i), "z": 0.0} for i in range(12)],
        planner={"target_v": 10.0, "aeb": "warn"},
        ego_speed_mps=12.0,
        seq=2,
    )
    assert abs(warn.brake - 0.35) < 1e-9, warn
    wkw = tech_control_kwargs(warn.steer, warn.throttle, warn.brake)
    assert float(wkw["parkingbrake"]) == 0.0, wkw
    held = tech_control_kwargs(*((stop_command().steer, stop_command().throttle, stop_command().brake)))
    stop = stop_command()
    hkw = tech_control_kwargs(stop.steer, stop.throttle, stop.brake)
    assert float(hkw["parkingbrake"]) == 1.0, hkw
    print("speed-error 0.1 and AEB warn 0.35 do not park; stop does OK")


def check_lua_threshold_matches() -> None:
    lua = (ROOT / "beamng_mod" / "lua" / "ge" / "extensions" / "gvd" / "main.lua").read_text(encoding="utf-8")
    assert "HOLD_BRAKE" in lua, "Lua applyInputs must name the hold threshold"
    assert "0.99" in lua, "Lua hold threshold must match TECH_HOLD_BRAKE"
    print("Lua hold threshold present OK")


def main() -> None:
    check_partial_brake_is_not_park()
    check_full_hold_parks()
    check_speed_error_and_aeb_warn()
    check_lua_threshold_matches()
    print("test_parkbrake_hold: OK")


if __name__ == "__main__":
    main()
