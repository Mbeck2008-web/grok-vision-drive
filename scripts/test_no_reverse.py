#!/usr/bin/env python3
"""No engaged command can select arcade reverse, and backward motion parks."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import (  # noqa: E402
    BeamNGPyActuator,
    DriveCommand,
    is_arcade_reverse_hold,
    is_reverse_control,
    plan_command,
    stop_command,
    tech_control_kwargs,
)
from python.control.launch_guard import (  # noqa: E402
    REVERSE_LONG_MPS,
    longitudinal_mps,
    moving_backward,
)


def _assert_forward(kw: dict) -> None:
    assert not is_reverse_control(kw), kw
    assert not is_arcade_reverse_hold(kw), kw
    assert "gear" not in kw
    throttle = float(kw.get("throttle") or 0.0)
    brake = float(kw.get("brake") or 0.0)
    if throttle <= 1e-6:
        assert brake == 0.0, kw


def check_wire_never_sends_the_reverse_pedal() -> None:
    samples = [
        (0.0, 0.0, 0.0),
        (0.2, 0.4, 0.0),
        (0.2, 0.4, 0.3),
        (0.1, 0.0, 0.2),
        (0.1, 0.0, 0.6),
        (0.1, 0.0, 0.8),
        (0.0, 0.0, 1.0),
        (-0.4, 0.0, 0.35),
    ]
    for steer, throttle, brake in samples:
        kw = tech_control_kwargs(steer, throttle, brake, speed_mps=8.0)
        _assert_forward(kw)
        assert float(kw.get("brake") or 0.0) == 0.0, kw
        if throttle <= 1e-6 and brake > 0.01:
            assert kw["parkingbrake"] == 1.0 and kw["brake"] == 0.0, kw
    assert is_arcade_reverse_hold({"throttle": 0.0, "brake": 0.8, "parkingbrake": 0.0})
    assert is_arcade_reverse_hold({"throttle": 0.0, "brake": 0.2})
    assert not is_arcade_reverse_hold({"throttle": 0.0, "brake": 0.0, "parkingbrake": 1.0})
    planned = plan_command(
        path_ego=[{"x": 0.0, "y": float(i), "z": 0.0} for i in range(12)],
        planner={"target_v": 2.0, "aeb": "warn"},
        ego_speed_mps=12.0,
        seq=1,
    )
    _assert_forward(tech_control_kwargs(planned.steer, planned.throttle, planned.brake))
    stopped = stop_command(reason="reverse")
    _assert_forward(tech_control_kwargs(stopped.steer, stopped.throttle, stopped.brake))


def check_reverse_guard_parks() -> None:
    assert REVERSE_LONG_MPS == 0.3
    assert longitudinal_mps((1.0, 0.0), (1.0, 0.0)) == 1.0
    assert longitudinal_mps((-1.0, 0.0), (1.0, 0.0)) == -1.0
    assert abs(longitudinal_mps((0.0, -0.4), (0.0, 1.0)) + 0.4) < 1e-9
    assert moving_backward(-0.31)
    assert not moving_backward(-0.3)
    assert not moving_backward(-0.2)
    assert not moving_backward(1.0)
    assert not moving_backward(None)

    class Veh:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def control(self, **kw) -> None:
            self.calls.append(kw)

        def set_shift_mode(self, mode: str) -> None:
            del mode

    veh = Veh()
    act = BeamNGPyActuator(veh)
    act.note_engaged(True)
    act.apply(DriveCommand(steer=0.1, throttle=0.3, brake=0.0, seq=1, reason="ok"))
    act.note_engaged(False)
    act.note_park_hold()
    held = act.stop(seq=2, reason="reverse")
    assert held.brake == 1.0 and held.throttle == 0.0
    assert veh.calls[-1]["parkingbrake"] == 1.0
    assert veh.calls[-1]["brake"] == 0.0
    assert veh.calls[-1]["throttle"] == 0.0
    _assert_forward(veh.calls[-1])
    again = act.stop(seq=3, reason="not_engaged")
    assert again.brake == 1.0
    assert veh.calls[-1]["parkingbrake"] == 1.0
    act.note_engaged(True)
    assert act._park_hold is False


def _backward_rows(path: Path) -> list[tuple[float, float]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    found = []
    prev = None
    for row in rows:
        if prev is not None and row.get("engaged") and prev.get("pos") and row.get("pos"):
            try:
                dt = float(row["t"]) - float(prev["t"])
                heading = float(row["heading_rad"])
            except (TypeError, ValueError, KeyError):
                prev = row
                continue
            if dt > 1e-3:
                dx = float(row["pos"][0]) - float(prev["pos"][0])
                dy = float(row["pos"][1]) - float(prev["pos"][1])
                along = (dx * math.cos(heading) + dy * math.sin(heading)) / dt
                if along < -REVERSE_LONG_MPS:
                    found.append((float(row["t"]), along))
        prev = row
    return found


def check_logged_runs() -> None:
    run9 = ROOT / "tests" / "fixtures" / "gt_run9" / "gt.jsonl"
    back = _backward_rows(run9)
    assert back == [], back[:5]
    print(f"gt_run9 backward ticks={len(back)}")
    run10 = list((ROOT / "tests" / "fixtures").glob("gt_run10/**/*.jsonl"))
    if not run10:
        print("gt_run10 log is not in this tree")
    else:
        for path in run10:
            found = _backward_rows(path)
            print(f"{path.name} backward ticks={len(found)}")
            assert found == []


def main() -> None:
    check_wire_never_sends_the_reverse_pedal()
    check_reverse_guard_parks()
    check_logged_runs()
    print("no reverse ok")


if __name__ == "__main__":
    main()
