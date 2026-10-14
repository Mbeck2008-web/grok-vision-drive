#!/usr/bin/env python3
"""Replay the ed33ee4 ground-truth run and the exit brake.

Engaged 9.82 s, in lane 83%, mean |offset| 1.37 m, then a crash
(damage 6597). lane_x vs gt_center r=0.651 with bias +3.61 m. Steer
matched pure pursuit of lane_x (about +0.31 while lane_x was +2.27),
so the gain was not the miss. The car yawed right, lane_x then fell to
~0, steer went to 0, and it walked out of the lane at 6 m/s. The launch
cap's 0.2 brake did not hold 4 m/s.

The exit brake has to fire on that lock and on the later walk-out, and
stay off when the lookahead and the yaw are both small.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import (  # noqa: E402
    STEER_LOCK_RAD,
    WHEELBASE_M,
    DriveCommand,
)
from python.control.launch_guard import (  # noqa: E402
    EXIT_BRAKE,
    EXIT_LANE_X_M,
    EXIT_YAW_RAD,
    LAUNCH_S,
    apply_lane_exit_brake,
)

FIXTURE = ROOT / "tests" / "fixtures" / "gt_run7"


def _rows() -> list[dict]:
    path = FIXTURE / "gt.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    assert vx > 1e-9 and vy > 1e-9
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _pursuit(lane_x: float) -> float:
    y = 8.0
    delta = math.atan2(2.0 * WHEELBASE_M * lane_x, lane_x * lane_x + y * y)
    return max(-1.0, min(1.0, delta / STEER_LOCK_RAD))


def _yaw(rows: list[dict]) -> list[float]:
    h0 = float(rows[0]["heading_rad"])
    return [math.remainder(float(r["heading_rad"]) - h0, 2.0 * math.pi) for r in rows]


def check_perception_and_the_walk_out(rows: list[dict]) -> None:
    engaged = [r for r in rows if r.get("engaged")]
    assert len(engaged) >= 40
    span = float(engaged[-1]["t"]) - float(engaged[0]["t"])
    assert 9.0 < span < 11.0, span
    xs = [float(r["lane_x"]) for r in engaged]
    gts = [float(r["gt_center_x"]) for r in engaged]
    offs = [float(r["lane_offset_m"]) for r in engaged]
    corr = _pearson(xs, gts)
    bias = sum(a - b for a, b in zip(xs, gts)) / len(xs)
    rms = math.sqrt(sum((a - b) ** 2 for a, b in zip(xs, gts)) / len(xs))
    assert corr > 0.6, corr
    assert bias > 3.0, bias
    assert rms > 3.5, rms
    assert _pearson(xs, offs) < -0.3

    # Opening lock: lane_x stays at +2.27 while gt_center falls through 0,
    # and the wheel is the pursuit of that positive x.
    lock = [r for r in engaged if abs(float(r["lane_x"]) - 2.27) < 0.02]
    assert len(lock) >= 8, len(lock)
    assert float(lock[0]["gt_center_x"]) > 0.4
    assert float(lock[-1]["gt_center_x"]) < -1.0
    assert all(float(r["steer"]) > 0.2 for r in lock)
    for row in lock:
        assert abs(float(row["steer"]) - _pursuit(float(row["lane_x"]))) < 0.08, row["t"]

    # After the yaw is built, lane_x ~ 0 so pursuit is ~0. The car is still
    # aimed right and the offset grows through the lane edge.
    walk = [
        r for r in engaged
        if float(r["t"]) > float(engaged[0]["t"]) + 4.5
        and abs(float(r["lane_x"])) < 0.1
    ]
    assert len(walk) >= 6, len(walk)
    assert all(abs(float(r["steer"])) < 0.02 for r in walk)
    assert float(walk[-1]["lane_offset_m"]) > float(walk[0]["lane_offset_m"]) + 1.0

    # Launch window is 6 s from engage. Speed still climbs past 4 m/s on
    # the 0.2 brake, and past 6 m/s before the hit.
    t0 = float(engaged[0]["t"])
    during = [r for r in engaged if float(r["t"]) < t0 + LAUNCH_S and float(r["speed_mps"]) > 5.0]
    assert during, "launch cap held 4 m/s"
    departed = [r for r in engaged if r.get("departed")]
    assert len(departed) == 1
    assert float(departed[0]["lane_offset_m"]) > 1.7
    assert abs(float(departed[0]["steer"])) < 0.05
    crashed = [r for r in rows if float(r.get("damage") or 0.0) > 6000.0]
    assert crashed and crashed[0].get("engaged") is False
    assert float(engaged[-1]["lane_offset_m"]) > 5.0
    assert float(engaged[-1]["speed_mps"]) > 6.0


def check_exit_brake_on_this_run(rows: list[dict]) -> None:
    engaged = [r for r in rows if r.get("engaged")]
    yaw = _yaw(engaged)
    t0 = float(engaged[0]["t"])
    locked_brake = 0
    walk_brake = 0
    for row, err in zip(engaged, yaw):
        cmd = DriveCommand(
            steer=float(row["steer"]), throttle=0.4, brake=0.0, seq=1, reason="ok",
        )
        fired = apply_lane_exit_brake(
            cmd, lane_x=float(row["lane_x"]), heading_err_rad=err,
        )
        x = float(row["lane_x"])
        leaving = abs(x) > EXIT_LANE_X_M or abs(err) > EXIT_YAW_RAD
        assert fired is leaving, (row["t"], x, err, fired)
        if abs(x - 2.27) < 0.02:
            assert cmd.throttle == 0.0 and cmd.brake >= EXIT_BRAKE
            locked_brake += 1
        if (
            float(row["t"]) > t0 + 4.5
            and abs(x) < 0.1
            and abs(err) > 0.30
        ):
            assert cmd.throttle == 0.0 and cmd.brake >= EXIT_BRAKE
            walk_brake += 1
    assert locked_brake >= 8, locked_brake
    assert walk_brake >= 6, walk_brake

    quiet = DriveCommand(steer=0.1, throttle=0.4, brake=0.0, seq=1, reason="ok")
    assert apply_lane_exit_brake(quiet, lane_x=0.2, heading_err_rad=0.05) is False
    assert quiet.throttle == 0.4 and quiet.brake == 0.0
    wide = DriveCommand(steer=0.3, throttle=0.4, brake=0.0, seq=1, reason="ok")
    assert apply_lane_exit_brake(wide, lane_x=2.27, heading_err_rad=0.0) is True
    assert wide.brake >= EXIT_BRAKE and wide.throttle == 0.0
    nosed = DriveCommand(steer=0.0, throttle=0.4, brake=0.0, seq=1, reason="ok")
    assert apply_lane_exit_brake(nosed, lane_x=-0.02, heading_err_rad=-0.33) is True
    assert nosed.brake >= EXIT_BRAKE


def main() -> None:
    rows = _rows()
    check_perception_and_the_walk_out(rows)
    check_exit_brake_on_this_run(rows)
    print("test_gt_run7_replay: OK")


if __name__ == "__main__":
    main()
