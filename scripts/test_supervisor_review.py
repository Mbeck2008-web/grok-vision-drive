#!/usr/bin/env python3
"""Merge/exit roles, review capture, arcade gear, link stale, camera wedges."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _line(x0: float, x1: float, y0: float = 0.0, y1: float = 40.0, n: int = 9) -> list[dict]:
    return [
        {"x": x0 + (x1 - x0) * i / (n - 1), "y": y0 + (y1 - y0) * i / (n - 1)}
        for i in range(n)
    ]


def _mean_x(pts: list[dict]) -> float:
    return sum(p["x"] for p in pts) / len(pts)


def check_roles() -> None:
    from python.viz.lane_roles import classify_lane_roles
    from python.viz.stage import VizUI, render_stage

    lanes = [
        {"points": _line(-1.75, -1.75), "kind": "detected", "side": "left"},
        {"points": _line(1.75, 1.75), "kind": "detected", "side": "right"},
        {"points": _line(6.2, 2.1), "kind": "detected", "side": "right"},
        {"points": _line(2.1, 7.0), "kind": "detected", "side": "right"},
        {"points": _line(5.25, 5.25), "kind": "detected", "side": "right"},
    ]
    tagged = classify_lane_roles(lanes)
    by_near = {}
    for ln in tagged:
        pts = sorted(ln["points"], key=lambda p: p["y"])
        by_near[round(pts[0]["x"], 2)] = ln["role"]
    assert by_near[-1.75] == "through"
    assert by_near[1.75] == "through"
    assert by_near[6.2] == "merge", by_near
    assert by_near[2.1] == "exit", by_near
    assert by_near[5.25] == "through", by_near

    state = {
        "engaged": False,
        "loop_hz": 12.0,
        "path_ego": [],
        "path_width": 2.0,
        "viz_smoke": False,
        "lanes_ext": lanes,
        "road_edges": [],
        "tracks": [],
        "signs": [],
    }
    ui = VizUI()
    ui.layers = {0}
    ui.show_nerd = False
    ui.debug.viz_forecast = False
    ui.debug.viz_signs = False
    render_stage(state, ui=ui)
    drawn = {round(sorted(ln["points"], key=lambda p: p["y"])[0]["x"], 2): ln for ln in state["viz_drawn_lanes"]}
    merge = drawn[6.2]
    assert merge["role"] == "merge"
    far = max(merge["points"], key=lambda p: p["y"])
    near = min(merge["points"], key=lambda p: p["y"])
    assert abs(far["x"] - 1.75) < abs(near["x"] - 1.75)
    exit_ln = drawn[2.1]
    assert exit_ln["role"] == "exit"
    exit_far = max(exit_ln["points"], key=lambda p: p["y"])
    exit_near = min(exit_ln["points"], key=lambda p: p["y"])
    assert abs(exit_far["x"] - 1.75) > abs(exit_near["x"] - 1.75)
    assert drawn[5.25]["role"] == "through"

    # Most of this line sits inboard of the real boundary, so its mean x is
    # closer to the car. The near end is still outside, and the gap closes.
    inside_pts = [{"x": 5.0, "y": 0.0}] + [
        {"x": 0.3, "y": float(y)} for y in range(2, 41, 2)
    ]
    inside = classify_lane_roles([
        {"points": _line(-1.75, -1.75), "kind": "detected"},
        {"points": _line(1.75, 1.75), "kind": "detected"},
        {"points": inside_pts, "kind": "detected"},
    ])
    assert inside[-1]["role"] == "merge", inside[-1]["role"]
    # Three metres is enough when the gap actually closes.
    short = classify_lane_roles([
        {"points": _line(-1.75, -1.75, y1=3.0), "kind": "detected"},
        {"points": _line(1.75, 1.75, y1=3.0), "kind": "detected"},
        {"points": _line(5.0, 1.9, y1=3.0), "kind": "detected"},
    ])
    assert short[-1]["role"] == "merge", short[-1]["role"]
    # The right line of a pair may leave. It is not forced to stay through.
    leaving = classify_lane_roles([
        {"points": _line(-1.75, -1.75), "kind": "detected"},
        {"points": _line(1.75, 6.0), "kind": "detected"},
    ])
    assert leaving[1]["role"] == "exit", leaving[1]["role"]


def check_review() -> None:
    from python.viz.review_log import ReviewCapture, ReviewLog
    from python.viz.stage import VizUI

    ui = VizUI()
    assert ui.handle_key(ord("R")) is True
    assert ui.debug.review_record is True
    ui.handle_key(ord("r"))
    assert ui.debug.review_record is False
    help_text = (ROOT / "python" / "viz" / "nerd.py").read_text(encoding="utf-8")
    assert "review capture" in help_text
    assert "Documents/GVD/review" in help_text

    folder = Path(tempfile.mkdtemp()) / "session"
    log = ReviewLog(folder)
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    frame[2, 3] = (10, 20, 30)
    dest = log.write_tick(
        lanes=[{"role": "merge", "kind": "detected", "points": _line(6.2, 2.1)}],
        path=[{"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 0.425, "y": 8.0, "z": 0.0}],
        steer=0.17,
        throttle=0.50,
        glance="DRIVE",
        disengage_reason="none",
        frames={"main": frame},
        brake=0.0,
        engaged=True,
        cmd_reason="ok",
    )
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["tick"] == 1
    assert payload["glance"] == "DRIVE"
    assert payload["disengage_reason"] == "none"
    assert payload["steer"] == 0.17
    assert payload["throttle"] == 0.50
    assert payload["lanes"][0]["role"] == "merge"
    assert payload["path"][1]["x"] == 0.425
    assert payload["frames"] == ["tick_000001_main.jpg"]
    jpg = folder / "tick_000001_main.jpg"
    assert jpg.is_file() and jpg.stat().st_size > 0
    assert dest.name == "tick_000001.json"

    cap = ReviewCapture()
    assert cap.write(False, lanes=[], path=[], steer=0.0, throttle=0.0, glance="OFF", disengage_reason="none") is None
    assert cap.log is None


def check_arcade_hold_not_reverse() -> None:
    from python.control.actuate import (
        TECH_SHIFT_MODE,
        BeamNGPyActuator,
        DriveCommand,
        is_arcade_reverse_hold,
        is_reverse_control,
        plan_command,
        tech_control_kwargs,
    )

    assert TECH_SHIFT_MODE == "arcade"

    def _blocked(kw: dict) -> None:
        assert kw.get("gear") != -1
        assert not is_reverse_control(kw)
        assert not is_arcade_reverse_hold(kw), kw
        assert float(kw["throttle"]) == 0.0
        assert float(kw["brake"]) == 0.0
        assert float(kw["parkingbrake"]) == 1.0
        assert int(kw["gear"]) == 0

    _blocked(tech_control_kwargs(0.0, 0.0, 1.0, speed_mps=0.0))
    _blocked(tech_control_kwargs(0.1, 0.0, 1.0, speed_mps=12.0))
    assert is_arcade_reverse_hold({"throttle": 0.0, "brake": 1.0, "gear": 0, "parkingbrake": 0.0})

    class Veh:
        def __init__(self, speed: float) -> None:
            self.calls: list[dict] = []
            self.shifts: list[str] = []
            self.sensors = {"electrics": {"wheelspeed": speed}}

        def set_shift_mode(self, mode: str) -> None:
            self.shifts.append(mode)

        def control(self, **kw) -> None:
            self.calls.append(kw)

    for speed, cmd in (
        (0.0, DriveCommand(steer=0.0, throttle=0.0, brake=1.0, seq=1, reason="ok")),
        (8.0, DriveCommand(steer=0.0, throttle=0.0, brake=1.0, seq=2, reason="stop")),
        (
            12.0,
            plan_command(
                path_ego=[{"x": 0.0, "y": float(i), "z": 0.0} for i in range(12)],
                planner={"target_v": 0.0, "aeb": "brake", "ttc_lead": 0.4},
                ego_speed_mps=12.0,
                seq=3,
            ),
        ),
    ):
        veh = Veh(speed)
        act = BeamNGPyActuator(veh)
        act.note_engaged(True)
        sent = act.apply(cmd)
        assert sent.applied is True
        assert veh.shifts == ["arcade"]
        _blocked(veh.calls[-1])


def check_link_and_gear() -> None:
    from python.control.actuate import HEARTBEAT_STALE_S, TECH_DRIVE_SHIFT_LUA, TECH_SHIFT_MODE

    assert TECH_SHIFT_MODE == "arcade"
    assert "setGearboxMode('arcade')" in TECH_DRIVE_SHIFT_LUA
    assert "realistic" not in TECH_DRIVE_SHIFT_LUA
    assert HEARTBEAT_STALE_S == 1.5
    lua = (ROOT / "beamng_mod" / "lua" / "ge" / "extensions" / "gvd" / "main.lua").read_text(encoding="utf-8")
    assert "local HB_STALE_S = 1.5" in lua
    assert "local CMD_STALE_S = 0.35" in lua


def check_frustums() -> None:
    import python.viz.debug_draw as draw
    from python.viz.debug_draw import frustum_ground_rays, load_frustums

    draw._FRUSTUMS = None
    frs = {fr["id"]: fr for fr in load_frustums()}

    def look(fr: dict) -> tuple[float, float, float, float]:
        (x, y), left, right = frustum_ground_rays(fr)
        dx = 0.5 * ((left[0] - x) + (right[0] - x))
        dy = 0.5 * ((left[1] - y) + (right[1] - y))
        return x, y, dx, dy

    lx, ly, ldx, ldy = look(frs["repeatL"])
    rx, ry, rdx, rdy = look(frs["repeatR"])
    assert lx < -0.4 and rx > 0.4 and lx < rx
    assert 0.2 < ly < 2.2 and 0.2 < ry < 2.2, (ly, ry)
    assert ldx < 0 < rdx, (ldx, rdx)
    assert ldy < 0 and rdy < 0, (ldy, rdy)

    behind = [cid for cid, fr in frs.items() if look(fr)[1] < -2.0]
    assert behind == ["rear"], behind
    assert look(frs["rear"])[3] < 0

    front = [cid for cid, fr in frs.items() if abs(look(fr)[0]) < 0.3 and look(fr)[3] > 0.5]
    assert sorted(front) == ["main", "narrow", "wide"], front
    # Pillars look across the corner. More lateral than forward, still short of a pure side view.
    assert frs["pillarR"]["yaw_deg"] == 68.0
    assert frs["pillarL"]["yaw_deg"] == -68.0
    _, _, prx, pry = look(frs["pillarR"])
    _, _, plx, ply = look(frs["pillarL"])
    assert prx > pry > 0.0, (prx, pry)
    assert plx < 0.0 < ply and abs(plx) > ply, (plx, ply)


def main() -> None:
    check_roles()
    check_review()
    check_arcade_hold_not_reverse()
    check_link_and_gear()
    check_frustums()
    print("test_supervisor_review: OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"test_supervisor_review: FAIL - {exc}")
        sys.exit(1)
