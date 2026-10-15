#!/usr/bin/env python3
"""Replay the 00457f7 run: arcade reverse into a building.

Spawn (-717.1, 101.0) is in the lane, offset -0.67 m, empty road ahead.
Before engage the fitter stored one_right at +5.41 m. Engage held that
value. |lane_x| was past 1.25 m, so the exit brake set throttle 0 and
brake 0.8. Arcade treats a service brake with no throttle as reverse.
The car backed up, speed reached 11.1 m/s, yawed under full right steer,
and hit a building (damage 4679). Nothing was in the lane. The detector
on that run was empty, so a parked car was not the hit.

The exit brake must be the parking-brake hold, not a service brake.
Until the first fresh pair after engage, a one-sided line must not move
the center off 0.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import (  # noqa: E402
    TECH_HOLD_BRAKE,
    is_arcade_reverse_hold,
    steer_sample_point,
    tech_control_kwargs,
)
from python.control.launch_guard import EXIT_BRAKE  # noqa: E402
from python.planning.path_predictor import LaneTrack, predict_path  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "gt_run8"


def _rows() -> list[dict]:
    return [json.loads(line) for line in (FIXTURE / "gt.jsonl").read_text().splitlines() if line.strip()]


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def check_backed_into_a_building(rows: list[dict]) -> None:
    engaged = [r for r in rows if r.get("engaged")]
    assert len(engaged) >= 20
    span = float(engaged[-1]["t"]) - float(engaged[0]["t"])
    assert 6.0 < span < 8.0, span
    spawn = engaged[0]
    assert abs(float(spawn["lane_offset_m"]) + 0.67) < 0.05
    assert float(spawn["lane_x"]) > 5.0
    assert "hold" in str(spawn["lane_src"])
    assert float(spawn["speed_mps"]) < 0.1
    # Nose points one way. The car travels the other way: reverse.
    backwards = 0
    prev = spawn
    for row in engaged[1:]:
        dx = float(row["pos"][0]) - float(prev["pos"][0])
        dy = float(row["pos"][1]) - float(prev["pos"][1])
        dist = math.hypot(dx, dy)
        if dist > 0.3:
            move = math.atan2(dy, dx)
            gap = abs(math.remainder(move - float(row["heading_rad"]), 2.0 * math.pi))
            if gap > 2.0:
                backwards += 1
        prev = row
    assert backwards >= 15, backwards
    assert max(float(r["speed_mps"]) for r in engaged) > 10.0
    xs = [float(r["lane_x"]) for r in engaged]
    gts = [float(r["gt_center_x"]) for r in engaged]
    bias = sum(a - b for a, b in zip(xs, gts)) / len(xs)
    assert bias > 2.0
    assert _pearson(xs, gts) < 0.5
    departed = [r for r in engaged if r.get("departed")]
    assert len(departed) == 1
    assert float(departed[0]["lane_offset_m"]) < -1.5
    assert departed[0].get("off_road") is False
    crashed = [r for r in rows if float(r.get("damage") or 0.0) > 4000.0]
    assert crashed and crashed[0].get("engaged") is False
    assert float(crashed[0]["damage"]) > 4500.0
    # Still on a road id at the damage sample. The wall is beside the lane,
    # not a vehicle the empty detector would have named.
    assert crashed[0].get("off_road") is False


def check_exit_brake_is_a_parking_hold() -> None:
    assert EXIT_BRAKE >= TECH_HOLD_BRAKE
    held = tech_control_kwargs(0.4, 0.0, EXIT_BRAKE)
    assert held["throttle"] == 0.0
    assert held["brake"] == 0.0
    assert held["parkingbrake"] == 1.0
    assert is_arcade_reverse_hold(held) is False
    # The command this run actually sent. Service brake, throttle 0.
    # Arcade drove that as reverse.
    sent = tech_control_kwargs(0.4, 0.0, 0.8)
    assert sent["brake"] == 0.8 and sent["parkingbrake"] == 0.0
    assert sent["throttle"] == 0.0


def _line(x: float) -> list[dict[str, float]]:
    return [{"x": float(x), "y": float(y)} for y in range(2, 20, 2)]


def check_no_hold_before_the_first_pair() -> None:
    track = LaneTrack()
    track.awaiting_pair = True
    one = predict_path(
        lanes_bev=[_line(5.41)],
        lane_conf=0.8,
        ego_speed_mps=0.0,
        lane_track=track,
    )
    sample = steer_sample_point(one.path_ego)
    assert one.lane_source == "seed", one.lane_source
    assert sample is not None and abs(float(sample["x"])) < 0.05, sample
    assert track.center_x is None
    assert track.awaiting_pair is True
    pair = predict_path(
        lanes_bev=[_line(-1.7), _line(1.8)],
        lane_conf=0.8,
        ego_speed_mps=4.0,
        lane_track=track,
    )
    seeded = steer_sample_point(pair.path_ego)
    assert pair.lane_source == "pair", pair.lane_source
    assert seeded is not None and abs(float(seeded["x"])) < 0.4, seeded
    assert track.awaiting_pair is False
    assert track.center_x is not None


def main() -> None:
    check_backed_into_a_building(_rows())
    check_exit_brake_is_a_parking_hold()
    check_no_hold_before_the_first_pair()
    print("test_gt_run8_replay: OK")


if __name__ == "__main__":
    main()
