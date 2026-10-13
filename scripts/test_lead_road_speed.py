#!/usr/bin/env python3
"""Lead road speed is ego plus the signed range-rate. No BeamNG.

A stopped car approached at 12 m/s used to be stored as speed ~12, ignored
from about 28 m to 4 m with throttle on, and the first brake was 0.6.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.perception.cipv import select_cipv  # noqa: E402
from python.perception.detect import Detection  # noqa: E402
from python.perception.track import IoUTracker, lead_road_speed  # noqa: E402
from python.planning.path_predictor import follow_path, predict_path  # noqa: E402


def _lanes() -> list:
    left = [{"x": -1.75, "y": float(i)} for i in range(0, 80, 2)]
    right = [{"x": 1.75, "y": float(i)} for i in range(0, 80, 2)]
    return [left, right]


def _approach(ego: float, lead_v: float, y0: float = 40.0):
    """Walk a lead at constant road speed. Yield (y, track, plan, command)."""
    tracker = IoUTracker()
    y = y0
    t = 0.0
    dt = 0.1
    rate = lead_v - ego
    rows = []
    for i in range(60):
        det = Detection(
            cls="vehicle", conf=0.9, xyxy=(0, 0, 10, 10), x=0.0, y=y, yaw=math.pi / 2,
        )
        tracker.update([det], t, ego_speed_mps=ego)
        plan = predict_path(
            lanes_bev=_lanes(), lane_conf=0.9, ego_speed_mps=ego, tracks=tracker.as_dicts(),
        )
        cmd = follow_path(plan, ego_speed_mps=ego, seq=i)
        rows.append((y, tracker.tracks[0], plan, cmd))
        y += rate * dt
        t += dt
        if y < 1.0:
            break
    return rows


def check_first_sample_is_not_ego() -> None:
    tracker = IoUTracker()
    det = Detection(
        cls="vehicle", conf=0.9, xyxy=(0, 0, 10, 10), x=0.0, y=28.0, yaw=math.pi / 2,
    )
    tracker.update([det], 0.0, ego_speed_mps=12.0)
    assert tracker.tracks[0].speed_mps == 0.0
    assert tracker.tracks[0].range_rate_mps == 0.0


def check_sign() -> None:
    stopped, rate = lead_road_speed(12.0, -1.2, 0.1)
    assert abs(stopped) < 1e-6, stopped
    assert abs(rate - (-12.0)) < 1e-6, rate
    slower, rate = lead_road_speed(12.0, -0.9, 0.1)
    assert abs(slower - 3.0) < 1e-6, slower
    assert rate < 0
    faster, rate = lead_road_speed(12.0, 0.4, 0.1)
    assert abs(faster - 16.0) < 1e-6, faster
    assert rate > 0
    # The absolute rate of a stopped car is 12. That is not its road speed.
    assert abs(stopped) < 1.0


def check_stopped_lead_aeb() -> None:
    rows = _approach(12.0, 0.0)
    far = [row for row in rows if 4.0 < row[0] <= 28.0]
    assert far, "approach never crossed 28 m"
    for y, tr, plan, cmd in far:
        assert tr.speed_mps < 1.0, (y, tr.speed_mps)
        assert tr.range_rate_mps < -10.0, (y, tr.range_rate_mps)
        assert cmd.throttle == 0.0, (y, cmd.throttle, cmd.brake, plan.aeb)
    aeb = [row for row in far if row[3].brake >= 0.99 and row[2].aeb == "brake"]
    assert aeb, "stopped lead never reached AEB before 4 m"
    y_aeb = max(row[0] for row in aeb)
    assert y_aeb > 8.0, y_aeb
    # The old first brake, at 4 m, was the comfort cap 0.6 with AEB off.
    at_four = min(rows, key=lambda row: abs(row[0] - 4.0))
    assert at_four[2].aeb == "brake" and at_four[3].brake >= 0.99, (
        at_four[0], at_four[2].aeb, at_four[3].brake,
    )
    cipv = select_cipv(
        [{"class": "vehicle", "x": 0.0, "y": 20.0, "speed_mps": 0.0}],
        ego_speed_mps=12.0,
    )
    assert cipv.ttc_lead is not None and abs(cipv.ttc_lead - (20.0 / 12.0)) < 1e-6


def check_slower_lead_aeb() -> None:
    # Ego 12, lead road speed 3. Range-rate is -9. Storing |rate| would
    # report speed 9 and close too late.
    rows = _approach(12.0, 3.0)
    far = [row for row in rows if 4.0 < row[0] <= 28.0]
    assert far
    for y, tr, plan, cmd in far:
        assert abs(tr.speed_mps - 3.0) < 0.5, (y, tr.speed_mps)
        assert abs(tr.speed_mps - 9.0) > 2.0, (y, tr.speed_mps)
        assert tr.range_rate_mps < -8.0, (y, tr.range_rate_mps)
        assert cmd.throttle == 0.0, (y, cmd.throttle, cmd.brake)
    aeb = [row for row in far if row[3].brake >= 0.99 and row[2].aeb == "brake"]
    assert aeb, "slower lead never reached AEB before 4 m"
    assert max(row[0] for row in aeb) > 8.0


def check_faster_lead_is_not_closing() -> None:
    rows = _approach(12.0, 16.0)
    late = [row for row in rows if row[0] > 45.0]
    assert late
    for y, tr, plan, cmd in late:
        assert abs(tr.speed_mps - 16.0) < 0.5, (y, tr.speed_mps)
        assert tr.range_rate_mps > 3.0, (y, tr.range_rate_mps)
        assert plan.aeb == "off" and cmd.brake == 0.0, (y, plan.aeb, cmd.brake)
        assert plan.stop_reason == "none", (y, plan.stop_reason)


def main() -> None:
    check_first_sample_is_not_ego()
    check_sign()
    check_stopped_lead_aeb()
    check_slower_lead_aeb()
    check_faster_lead_is_not_closing()
    print("test_lead_road_speed: OK")


if __name__ == "__main__":
    main()
