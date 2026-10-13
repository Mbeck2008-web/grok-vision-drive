#!/usr/bin/env python3
"""Lead road speed is ego plus the signed range-rate. No BeamNG.

A stopped car approached at 12 m/s used to be stored as speed ~12, ignored
from about 28 m to 4 m with throttle on, and the first brake was 0.6.
"""
from __future__ import annotations

import math
import random
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
        cls="vehicle", conf=0.9, xyxy=(0, 0, 10, 10), x=0.0, y=16.0, yaw=math.pi / 2,
    )
    tracker.update([det], 0.0, ego_speed_mps=12.0)
    tr = tracker.tracks[0]
    assert tr.speed_mps == 0.0
    assert tr.range_rate_mps == 0.0
    assert tr.range_rate_valid is False
    # speed 0 at 16 m used to be a stopped lead and a full brake.
    plan = predict_path(
        lanes_bev=_lanes(), lane_conf=0.9, ego_speed_mps=12.0, tracks=tracker.as_dicts(),
    )
    cmd = follow_path(plan, ego_speed_mps=12.0, seq=0)
    assert plan.aeb != "brake", plan.aeb
    assert cmd.brake < 0.99, cmd.brake
    cipv = select_cipv(tracker.as_dicts(), ego_speed_mps=12.0)
    assert cipv.aeb == "off"


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


def check_noisy_range_rate_does_not_aeb() -> None:
    """±2 m grab noise at 12–20 m must not full-brake. A real stop still does.

    One 4 m jump in 0.1 s is 40 m/s. A 0.25 blend of that is still about
    10 m/s of closing, and at 16 m that is a 1.0 brake. The rate has to
    come from several frames before it can arm AEB.
    """
    ego = 12.0
    dt = 0.1
    tracker = IoUTracker()
    t = 0.0
    for i in range(24):
        jitter = 2.0 if i % 2 == 0 else -2.0
        y = 16.0 + jitter
        assert 12.0 <= y <= 20.0
        det = Detection(
            cls="vehicle", conf=0.9, xyxy=(0, 0, 10, 10), x=0.0, y=y, yaw=math.pi / 2,
        )
        tracker.update([det], t, ego_speed_mps=ego)
        tr = tracker.tracks[0]
        plan = predict_path(
            lanes_bev=_lanes(), lane_conf=0.9, ego_speed_mps=ego, tracks=tracker.as_dicts(),
        )
        cmd = follow_path(plan, ego_speed_mps=ego, seq=i)
        assert plan.aeb != "brake", (i, y, plan.aeb, tr.range_rate_mps, tr.range_rate_valid)
        assert cmd.brake < 0.99, (i, y, cmd.brake, tr.range_rate_mps)
        t += dt
    # A single late jump after a steady 16 m lead is the same rule.
    steady = IoUTracker()
    t = 0.0
    for i in range(10):
        y = 16.0 if i < 9 else 14.0
        det = Detection(
            cls="vehicle", conf=0.9, xyxy=(0, 0, 10, 10), x=0.0, y=y, yaw=math.pi / 2,
        )
        steady.update([det], t, ego_speed_mps=ego)
        plan = predict_path(
            lanes_bev=_lanes(), lane_conf=0.9, ego_speed_mps=ego, tracks=steady.as_dicts(),
        )
        cmd = follow_path(plan, ego_speed_mps=ego, seq=i)
        assert plan.aeb != "brake" and cmd.brake < 0.99, (i, y, plan.aeb, cmd.brake)
        t += dt


def _decel(brake: float) -> float:
    """Tire limit used by the stop-margin sim. Brake 1.0 is 8 m/s²."""
    return 8.0 * max(0.0, min(1.0, float(brake)))


def _drive(tracker: IoUTracker, y: float, t: float, ego: float, seq: int):
    det = Detection(
        cls="vehicle", conf=0.9, xyxy=(0, 0, 10, 10), x=0.0, y=y, yaw=math.pi / 2,
    )
    tracker.update([det], t, ego_speed_mps=ego)
    plan = predict_path(
        lanes_bev=_lanes(), lane_conf=0.9, ego_speed_mps=ego, tracks=tracker.as_dicts(),
    )
    cmd = follow_path(plan, ego_speed_mps=ego, seq=seq)
    return plan, cmd


def check_step_and_random_noise_do_not_full_brake() -> None:
    """A +2/−2 step and random ±2 m must not full-brake a constant lead.

    Six grabs of three high then three low fit about −10 m/s. That used to
    be a 1.0 brake at 12–20 m with ego at 12 m/s. Random ±2 m did it too.
    """
    ego = 12.0
    dt = 0.1
    tracker = IoUTracker()
    t = 0.0
    for i in range(24):
        step = 2.0 if (i // 3) % 2 == 0 else -2.0
        y = 16.0 + step
        assert 12.0 <= y <= 20.0
        plan, cmd = _drive(tracker, y, t, ego, i)
        assert plan.aeb != "brake", (i, y, plan.aeb, tracker.tracks[0].range_rate_mps)
        assert cmd.brake < 0.99, (i, y, cmd.brake)
        t += dt

    rng = random.Random(81)
    full = 0
    for trial in range(400):
        noisy = IoUTracker()
        t = 0.0
        for i in range(24):
            y = 16.0 + rng.uniform(-2.0, 2.0)
            plan, cmd = _drive(noisy, y, t, ego, i)
            if plan.aeb == "brake" or cmd.brake >= 0.99:
                full += 1
                break
            t += dt
    assert full == 0, full


def _full(plan, cmd) -> bool:
    return plan.aeb == "brake" or cmd.brake >= 0.99


def check_ramp_and_reflected_do_not_full_brake() -> None:
    """A 0.5 m ramp and a reflected walk inside 16±2 are not a close.

    Five even steps from 18 m to 14 m fit a slope. That slope used to
    full-brake a lead that never left the noise band. A 14→18 jump used
    to miss the 4 m gate and leave the first track braking.
    """
    ego = 12.0
    ramp = [18.0 - 0.5 * i for i in range(9)]
    assert ramp[0] == 18.0 and ramp[-1] == 14.0
    for hz in (10.0, 15.0):
        dt = 1.0 / hz
        tracker = IoUTracker()
        t = 0.0
        for i, y in enumerate(ramp):
            plan, cmd = _drive(tracker, y, t, ego, i)
            assert not _full(plan, cmd), (hz, i, y, plan.aeb, cmd.brake)
            assert len(tracker.tracks) == 1, (hz, i, len(tracker.tracks))
            t += dt

    rng = random.Random(81)
    full = 0
    for _trial in range(300):
        for hz in (10.0, 15.0):
            dt = 1.0 / hz
            y = 16.0
            sign = -1.0 if rng.random() < 0.5 else 1.0
            noisy = IoUTracker()
            t = 0.0
            hit = False
            for i in range(24):
                y += sign * 0.5
                if y > 18.0:
                    y = 18.0 - (y - 18.0)
                    sign = -1.0
                elif y < 14.0:
                    y = 14.0 + (14.0 - y)
                    sign = 1.0
                y = min(18.0, max(14.0, y))
                plan, cmd = _drive(noisy, y, t, ego, i)
                if _full(plan, cmd) or len(noisy.tracks) != 1:
                    hit = True
                    break
                t += dt
            if hit:
                full += 1
                break
    assert full == 0, full

    jumped = IoUTracker()
    _drive(jumped, 14.0, 0.0, ego, 0)
    plan, cmd = _drive(jumped, 18.0, 0.1, ego, 1)
    assert len(jumped.tracks) == 1, len(jumped.tracks)
    assert jumped.tracks[0].misses == 0
    assert not _full(plan, cmd), (plan.aeb, cmd.brake)


def check_noisy_close_stops_with_margin() -> None:
    """Ego 18 m/s, lead first seen at 30 m, ±0.5 m and ±1 m, must stop.

    The rate may never clear the noise. The brake is the decel the gap
    asks for, so the stop does not wait on a fit.
    """
    def margin(hz: float, noise: float, seed: int) -> float:
        rng = random.Random(seed)
        dt = 1.0 / hz
        ego = 18.0
        s = 0.0
        lead = 30.0
        t = 0.0
        tracker = IoUTracker()
        for i in range(int(10 * hz) + 2):
            true_y = lead - s
            if true_y <= 0.0:
                return true_y
            y = max(0.4, true_y + rng.uniform(-noise, noise))
            _plan, cmd = _drive(tracker, y, t, ego, i)
            a = _decel(cmd.brake)
            if ego <= 0.05 or (a > 1e-6 and a * dt >= ego):
                if a > 1e-6:
                    s += ego * ego / (2.0 * a)
                ego = 0.0
                break
            s += ego * dt - 0.5 * a * dt * dt
            ego = max(0.0, ego - a * dt)
            t += dt
        assert ego <= 0.05, (hz, noise, seed, ego, lead - s)
        return lead - s

    for noise in (0.5, 1.0):
        for hz in (10.0, 15.0):
            left = [margin(hz, noise, 1000 + trial) for trial in range(40)]
            worst = min(left)
            assert worst >= 2.0, (hz, noise, worst, left)


def check_close_vehicle_and_first_pedestrian_brake() -> None:
    """A vehicle at 3.9 m brakes. A first pedestrian at 12 m is AEB."""
    ego = 12.0
    close = predict_path(
        lanes_bev=_lanes(),
        lane_conf=0.9,
        ego_speed_mps=ego,
        tracks=[{
            "class": "vehicle",
            "x": 0.0,
            "y": 3.9,
            "yaw": math.pi / 2,
            "speed_mps": 0.0,
            "range_rate_valid": False,
        }],
    )
    close_cmd = follow_path(close, ego_speed_mps=ego, seq=0)
    assert close_cmd.throttle == 0.0, close_cmd
    assert close_cmd.brake >= 0.99, close_cmd

    person = predict_path(
        lanes_bev=_lanes(),
        lane_conf=0.9,
        ego_speed_mps=ego,
        tracks=[{
            "class": "pedestrian",
            "x": 0.0,
            "y": 12.0,
            "yaw": math.pi / 2,
            "speed_mps": 0.0,
            "range_rate_valid": False,
        }],
    )
    person_cmd = follow_path(person, ego_speed_mps=ego, seq=1)
    assert person.aeb == "brake", person.aeb
    assert person_cmd.brake >= 0.99, person_cmd
    rider = predict_path(
        lanes_bev=_lanes(),
        lane_conf=0.9,
        ego_speed_mps=ego,
        tracks=[{
            "class": "bike",
            "x": 0.0,
            "y": 12.0,
            "yaw": math.pi / 2,
            "speed_mps": 0.0,
            "range_rate_valid": False,
        }],
    )
    rider_cmd = follow_path(rider, ego_speed_mps=ego, seq=2)
    assert rider.aeb == "brake" and rider_cmd.brake >= 0.99, (rider.aeb, rider_cmd.brake)


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
    check_noisy_range_rate_does_not_aeb()
    check_step_and_random_noise_do_not_full_brake()
    check_ramp_and_reflected_do_not_full_brake()
    check_noisy_close_stops_with_margin()
    check_close_vehicle_and_first_pedestrian_brake()
    check_faster_lead_is_not_closing()
    print("test_lead_road_speed: OK")


if __name__ == "__main__":
    main()
