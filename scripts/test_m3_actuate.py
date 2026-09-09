#!/usr/bin/env python3
"""Offline M3 actuation gates (Spec verify 2–4)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import (
    CmdJsonActuator,
    heartbeat_fresh,
    may_drive,
    read_electrics_speed,
    safe_command,
)


class _Pollable(dict):
    def poll(self):
        return self


def main() -> None:
    # 2) stale heartbeat → zero throttle + brake
    cmd = safe_command(
        engaged=True,
        heartbeat_ok=False,
        path_debug_preview=False,
        allow_preview_drive=False,
        path_ego=[{"x": 0, "y": 5, "z": 0}, {"x": 0.2, "y": 10, "z": 0}, {"x": 0.3, "y": 15, "z": 0}],
        planner={"target_v": 12, "aeb": "off", "ttc_lead": None},
        ego_speed_mps=8.0,
        seq=1,
    )
    assert cmd.throttle == 0.0 and cmd.brake == 1.0 and cmd.reason == "heartbeat_stale", cmd
    assert heartbeat_fresh(__import__("time").time() - 1.0) is False
    assert heartbeat_fresh(__import__("time").time()) is True

    # 3) preview path → no actuate unless flag
    cmd2 = safe_command(
        engaged=True,
        heartbeat_ok=True,
        path_debug_preview=True,
        allow_preview_drive=False,
        path_ego=[{"x": 0, "y": 5, "z": 0}] * 5,
        planner={"target_v": 10, "aeb": "off"},
        ego_speed_mps=5.0,
        seq=2,
    )
    assert cmd2.reason == "preview_blocked" and cmd2.throttle == 0.0 and cmd2.brake == 1.0, cmd2
    ok, _ = may_drive(engaged=True, heartbeat_ok=True, path_debug_preview=True, allow_preview_drive=True)
    assert ok
    cmd3 = safe_command(
        engaged=True,
        heartbeat_ok=True,
        path_debug_preview=True,
        allow_preview_drive=True,
        path_ego=[{"x": 0.5, "y": float(i), "z": 0} for i in range(12)],
        planner={"target_v": 10, "aeb": "off"},
        ego_speed_mps=5.0,
        seq=3,
    )
    assert cmd3.reason == "ok" and cmd3.throttle >= 0.0, cmd3

    # AEB
    cmd_aeb = safe_command(
        engaged=True,
        heartbeat_ok=True,
        path_debug_preview=False,
        allow_preview_drive=False,
        path_ego=[{"x": 0, "y": float(i), "z": 0} for i in range(12)],
        planner={"target_v": 10, "aeb": "brake", "ttc_lead": 0.5},
        ego_speed_mps=12.0,
        seq=4,
    )
    assert cmd_aeb.throttle == 0.0 and cmd_aeb.brake == 1.0, cmd_aeb

    # 4) Electrics mock → speed
    class V:
        def __init__(self):
            self.sensors = _Pollable(
                {"electrics": {"wheelspeed": 7.5, "airspeed": 7.4, "steering_input": 0.1}}
            )

    spd, steeri = read_electrics_speed(V())
    assert spd == 7.5 and steeri == 0.1, (spd, steeri)

    # cmd json actuator writes the file but never claims applied without a Lua ack (M6 bus)
    act = CmdJsonActuator()
    out = act.stop(seq=9, reason="unit_stop")
    assert out.applied is False and out.reason == "cmd_json_idle" and out.brake == 1.0
    from python.control.actuate import cmd_path

    assert cmd_path().is_file()
    import json

    payload = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert payload["engaged"] is False and payload["brake"] == 1.0 and payload["seq"] == 9

    print("test_m3_actuate: OK")


if __name__ == "__main__":
    main()
