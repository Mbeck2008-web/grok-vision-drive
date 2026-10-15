#!/usr/bin/env python3
"""Steer sign against the 47b430e ground-truth crash window.

Positive applied steer yawed the car clockwise (heading atan2(dir_y, dir_x)
fell) while lane_x stayed positive. That sign chases a point on the right.
It does not reduce heading error versus the heading at engage. The restoring
command does.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import DriveCommand, steer_for_heading_error  # noqa: E402
from python.control.launch_guard import LAUNCH_THROTTLE, LaunchGuard  # noqa: E402
from python.debug.gt_logger import load_road_polys, polys_from_network  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "gt_crash_window.jsonl"


def _rows() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _unwrapped(rows: list[dict]) -> list[float]:
    out = [float(rows[0]["heading_rad"])]
    for row in rows[1:]:
        prev = out[-1]
        cur = float(row["heading_rad"])
        out.append(prev + math.remainder(cur - prev, 2.0 * math.pi))
    return out


def check_logged_steer_increased_heading_error() -> None:
    rows = _rows()
    heading = _unwrapped(rows)
    h0 = heading[0]
    # The window covers the yaw into the building.
    end = heading[-1] - h0
    assert end < -0.8, end
    steered = [r for r in rows if float(r["steer"]) > 0.2]
    assert len(steered) >= 8
    # +steer while the heading error is already clockwise does not reduce it.
    late = [(h - h0, float(r["steer"])) for h, r in zip(heading, rows) if h - h0 < -0.4]
    assert late
    assert all(steer > 0.0 for _err, steer in late)


def check_restoring_steer_reduces_the_error() -> None:
    """Same sign as the error, because +steer yaws clockwise."""
    rows = _rows()
    heading = _unwrapped(rows)
    h0 = heading[0]
    err = heading[-1] - h0
    assert err < 0.0
    restore = steer_for_heading_error(err)
    assert restore < 0.0, restore
    # One second of the restoring command brings the heading back toward 0.
    # The logged positive steer drives it further clockwise. Gain matches the
    # observed ~0.5 rad/s at steer 0.35, so about 1.4 rad/s per unit steer.
    yaw_per_steer = 1.4
    dt = 1.0
    with_log = err - yaw_per_steer * 0.35 * dt
    with_fix = err - yaw_per_steer * restore * dt
    assert abs(with_fix) < abs(err), (with_fix, err)
    assert abs(with_log) > abs(err)


def check_launch_guard_on_this_yaw() -> None:
    rows = _rows()
    heading = _unwrapped(rows)
    guard = LaunchGuard()
    corrected = None
    for prev, row, h in zip(rows, rows[1:], heading[1:]):
        dt = float(row["t"]) - float(prev["t"])
        cmd = DriveCommand(
            steer=float(row["steer"]), throttle=0.55, brake=0.0, seq=1, reason="ok",
        )
        reason = guard.update(
            cmd,
            engaged=True,
            dt_s=dt,
            heading_rad=float(row["heading_rad"]),
            speed_mps=float(row["speed_mps"]),
            damage=float(row["damage"] or 0.0),
            pos_xy=None,
        )
        assert cmd.throttle <= LAUNCH_THROTTLE + 1e-9
        if heading[0] is not None and abs(h - heading[0]) > 0.35 and guard.age_s < 6.0:
            corrected = cmd.steer
            break
    assert corrected is not None and corrected < 0.0, corrected
    # The damage jump at the end of the window disengages.
    guard2 = LaunchGuard()
    cmd = DriveCommand(steer=0.3, throttle=0.2, brake=0.0, seq=1, reason="ok")
    assert guard2.update(
        cmd, engaged=True, dt_s=0.1, heading_rad=-2.3, speed_mps=1.0,
        damage=0.0, pos_xy=(0.0, 0.0),
    ) is None
    reason = guard2.update(
        cmd, engaged=True, dt_s=0.2, heading_rad=-2.3, speed_mps=8.0,
        damage=1200.0, pos_xy=(1.0, 0.0),
    )
    assert reason == "crash", reason


class _Boom(Exception):
    pass


class _Bad:
    def get_roads(self):
        raise _Boom("deprecated GetDecalRoadData")


class _Net:
    def get_road_network(self, include_edges=True, drivable_only=True):
        assert include_edges and drivable_only
        return {
            "highway": {
                "drivability": 1,
                "edges": [
                    {"left": [0, 0, 0], "right": [7, 0, 0], "middle": [3.5, 0, 0]},
                    {"left": [0, 20, 0], "right": [7, 20, 0], "middle": [3.5, 20, 0]},
                ],
            },
            "crack": {"drivability": 0, "edges": [
                {"left": [0, 0, 0], "right": [0.2, 0, 0]},
                {"left": [0, 5, 0], "right": [0.2, 5, 0]},
            ]},
        }


class _Bng:
    def __init__(self):
        self.scenario = _Net()

    def get_roads(self):
        raise _Boom("should not be called once the network works")


def check_road_network_fallback() -> None:
    polys, note = load_road_polys(_Bad())
    assert polys == []
    assert "deprecated GetDecalRoadData" in note, note
    assert "Boom" in note
    polys, note = load_road_polys(_Bng())
    assert note == "", note
    assert len(polys) == 1 and polys[0].road_id == "highway"
    spec = _Net().get_road_network()
    assert len(polys_from_network(spec)) == 1


def main() -> None:
    check_logged_steer_increased_heading_error()
    check_restoring_steer_reduces_the_error()
    check_launch_guard_on_this_yaw()
    check_road_network_fallback()
    print("test_gt_crash_sign: OK")


if __name__ == "__main__":
    main()
