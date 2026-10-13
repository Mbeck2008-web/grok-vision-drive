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

from python.control.actuate import plan_command  # noqa: E402
from python.perception.cipv import select_cipv  # noqa: E402
from python.perception.detect import Detection  # noqa: E402
from python.perception.pipeline import publish_planner  # noqa: E402
from python.perception.track import IoUTracker, lead_road_speed  # noqa: E402
from python.planning.path_predictor import predict_path  # noqa: E402


def _lanes() -> list:
    left = [{"x": -1.75, "y": float(i)} for i in range(0, 80, 2)]
    right = [{"x": 1.75, "y": float(i)} for i in range(0, 80, 2)]
    return [left, right]


def _live(plan, tracks, ego: float, seq: int):
    """The dict ``ModularPerception.tick`` publishes, then ``plan_command``.

    ``follow_path`` reads ``PathPlan.planner_dict()``, which already carried
    ``range_brake``. The live tick used to build a second dict and drop it.
    """
    route = plan.route_ego or plan.path_ego
    cipv = select_cipv(
        tracks, path_width=2.0, ego_speed_mps=ego, path_ego=route or None,
    )
    planner = publish_planner(plan, cipv)
    assert "range_brake" in planner
    assert planner["range_brake"] == plan.range_brake
    cmd = plan_command(
        path_ego=plan.path_ego, planner=planner, ego_speed_mps=ego, seq=seq,
    )
    return planner, cmd


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
        tracks = tracker.as_dicts()
        plan = predict_path(
            lanes_bev=_lanes(), lane_conf=0.9, ego_speed_mps=ego, tracks=tracks,
        )
        _planner, cmd = _live(plan, tracks, ego, i)
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
    tracks = tracker.as_dicts()
    plan = predict_path(
        lanes_bev=_lanes(), lane_conf=0.9, ego_speed_mps=12.0, tracks=tracks,
    )
    planner, cmd = _live(plan, tracks, 12.0, 0)
    assert "range_brake" in planner
    assert planner["range_brake"] == plan.range_brake
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
        tracks = tracker.as_dicts()
        plan = predict_path(
            lanes_bev=_lanes(), lane_conf=0.9, ego_speed_mps=ego, tracks=tracks,
        )
        _planner, cmd = _live(plan, tracks, ego, i)
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
        tracks = steady.as_dicts()
        plan = predict_path(
            lanes_bev=_lanes(), lane_conf=0.9, ego_speed_mps=ego, tracks=tracks,
        )
        _planner, cmd = _live(plan, tracks, ego, i)
        assert plan.aeb != "brake" and cmd.brake < 0.99, (i, y, plan.aeb, cmd.brake)
        t += dt


def _decel(brake: float) -> float:
    """Tire limit used by the stop-margin sim. Brake 1.0 is 8 m/s²."""
    return 8.0 * max(0.0, min(1.0, float(brake)))


def _drive(
    tracker: IoUTracker,
    y: float,
    t: float,
    ego: float,
    seq: int,
    kind: str = "vehicle",
    x: float = 0.0,
    lanes=None,
):
    det = Detection(
        cls=kind, conf=0.9, xyxy=(0, 0, 10, 10), x=x, y=y, yaw=math.pi / 2,
    )
    tracker.update([det], t, ego_speed_mps=ego)
    tracks = tracker.as_dicts()
    plan = predict_path(
        lanes_bev=_lanes() if lanes is None else lanes,
        lane_conf=0.9,
        ego_speed_mps=ego,
        tracks=tracks,
    )
    planner, cmd = _live(plan, tracks, ego, seq)
    return plan, cmd, planner


def check_step_and_random_noise_do_not_full_brake() -> None:
    """A +2/−2 step and random ±2 m must not full-brake a constant lead.

    Six grabs of three high then three low fit about −10 m/s. That used to
    be a 1.0 brake at 12–20 m. At ego 18 the old stop law was 1.0 for any
    gap under about 23 m, so this band is run at 18.
    """
    ego = 18.0
    dt = 0.1
    tracker = IoUTracker()
    t = 0.0
    for i in range(24):
        step = 2.0 if (i // 3) % 2 == 0 else -2.0
        y = 16.0 + step
        assert 12.0 <= y <= 20.0
        plan, cmd, _planner = _drive(tracker, y, t, ego, i)
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
            plan, cmd, _planner = _drive(noisy, y, t, ego, i)
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
            plan, cmd, _planner = _drive(tracker, y, t, ego, i)
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
                plan, cmd, _planner = _drive(noisy, y, t, ego, i)
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
    plan, cmd, _planner = _drive(jumped, 18.0, 0.1, ego, 1)
    assert len(jumped.tracks) == 1, len(jumped.tracks)
    assert jumped.tracks[0].misses == 0
    assert not _full(plan, cmd), (plan.aeb, cmd.brake)


def check_noisy_close_stops_with_margin() -> None:
    """Ego 18 m/s, lead first seen at 30 m, ±0.5 m and ±1 m, must stop.

    A flat gap is not a stop. The brake waits until the measured range has
    closed past the noise band, then uses the uncapped gap law.
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
            _plan, cmd, _planner = _drive(tracker, y, t, ego, i)
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
    for ego in (12.0, 18.0):
        close_tracks = [{
            "class": "vehicle",
            "x": 0.0,
            "y": 3.9,
            "yaw": math.pi / 2,
            "speed_mps": 0.0,
            "range_rate_valid": False,
        }]
        close = predict_path(
            lanes_bev=_lanes(), lane_conf=0.9, ego_speed_mps=ego, tracks=close_tracks,
        )
        _planner, close_cmd = _live(close, close_tracks, ego, 0)
        assert "range_brake" in _planner
        assert close_cmd.throttle == 0.0, (ego, close_cmd)
        assert close_cmd.brake >= 0.99, (ego, close_cmd)

        person_tracks = [{
            "class": "pedestrian",
            "x": 0.0,
            "y": 12.0,
            "yaw": math.pi / 2,
            "speed_mps": 0.0,
            "range_rate_valid": False,
        }]
        person = predict_path(
            lanes_bev=_lanes(), lane_conf=0.9, ego_speed_mps=ego, tracks=person_tracks,
        )
        _planner, person_cmd = _live(person, person_tracks, ego, 1)
        assert person.aeb == "brake", (ego, person.aeb)
        assert person_cmd.brake >= 0.99, (ego, person_cmd)
        rider_tracks = [{
            "class": "bike",
            "x": 0.0,
            "y": 12.0,
            "yaw": math.pi / 2,
            "speed_mps": 0.0,
            "range_rate_valid": False,
        }]
        rider = predict_path(
            lanes_bev=_lanes(), lane_conf=0.9, ego_speed_mps=ego, tracks=rider_tracks,
        )
        _planner, rider_cmd = _live(rider, rider_tracks, ego, 2)
        assert rider.aeb == "brake" and rider_cmd.brake >= 0.99, (
            ego, rider.aeb, rider_cmd.brake,
        )


def check_flat_gap_confirms_not_closing() -> None:
    """A noisy constant gap is rate ~0 once the window is full, not invalid."""
    ego = 18.0
    rng = random.Random(4)
    tracker = IoUTracker()
    t = 0.0
    for i in range(20):
        y = 36.0 + rng.uniform(-0.3, 0.3)
        det = Detection(
            cls="vehicle", conf=0.9, xyxy=(0, 0, 10, 10), x=0.0, y=y, yaw=math.pi / 2,
        )
        tracker.update([det], t, ego_speed_mps=ego)
        t += 0.1
    tr = tracker.tracks[0]
    assert tr.range_rate_valid is True, (tr.range_rate_valid, tr.range_rate_mps)
    assert abs(tr.range_rate_mps) < 0.2, tr.range_rate_mps
    assert abs(tr.speed_mps - ego) < 0.2, tr.speed_mps


def check_pacing_lead_holds_speed() -> None:
    """A lead matching our speed, with range noise, must not slow us.

    Ego 12 and 18, gaps of 1.5 s and 2 s, noise ±0.3 / ±1 / ±2 m, 40 seeds.
    Over 8 s the speed stays within 1 m/s when the noise is at most 1 m.
    ±2 m on the short gap can shed a little more than that; it must not
    be a stop. The old stop law took 18 down to 5.6.
    """
    worst_loud = 0.0
    for ego0 in (12.0, 18.0):
        for headway in (1.5, 2.0):
            for noise in (0.3, 1.0, 2.0):
                for hz in (10.0, 15.0):
                    for seed in range(40):
                        rng = random.Random(seed + int(hz) + int(noise * 10) + int(ego0))
                        dt = 1.0 / hz
                        ego = ego0
                        gap0 = ego0 * headway
                        lead_s = 0.0
                        ego_s = 0.0
                        t = 0.0
                        tracker = IoUTracker()
                        steps = int(8.0 * hz)
                        slowest = ego
                        for i in range(steps):
                            true_gap = gap0 + lead_s - ego_s
                            y = max(0.4, true_gap + rng.uniform(-noise, noise))
                            plan, cmd, _planner = _drive(tracker, y, t, ego, i)
                            assert not _full(plan, cmd), (
                                ego0, headway, noise, hz, seed, i, y, plan.aeb, cmd.brake,
                            )
                            a = _decel(cmd.brake)
                            if a > 1e-6 and a * dt >= ego:
                                ego_s += ego * ego / (2.0 * a)
                                ego = 0.0
                            else:
                                ego_s += ego * dt - 0.5 * a * dt * dt
                                ego = max(0.0, ego - a * dt)
                            lead_s += ego0 * dt
                            t += dt
                            slowest = min(slowest, ego)
                        drop = ego0 - slowest
                        # ±2 m on a 1.5 s gap moves the rate inside the noise
                        # for long enough to shed a bit over 1 m/s. Noise at
                        # or under 1 m stays inside the 1 m/s budget.
                        limit = 1.0 if noise <= 1.0 else 1.25
                        assert drop <= limit, (
                            ego0, headway, noise, hz, seed, slowest, ego, drop,
                        )
                        if noise >= 2.0:
                            worst_loud = max(worst_loud, drop)
    print(f"pacing ±2 m worst drop m/s: {worst_loud:.2f}")


def check_matched_cut_in_no_full_brake() -> None:
    """A cut-in at our speed, 16–20 m out, must not hold brake 1.0."""
    for ego in (12.0, 18.0):
        for y0 in (16.0, 18.0, 20.0):
            raw_tracks = [{
                "class": "vehicle",
                "x": 0.0,
                "y": y0,
                "yaw": math.pi / 2,
                "speed_mps": 0.0,
                "range_rate_valid": False,
            }]
            raw = predict_path(
                lanes_bev=_lanes(), lane_conf=0.9, ego_speed_mps=ego, tracks=raw_tracks,
            )
            _planner, raw_cmd = _live(raw, raw_tracks, ego, 0)
            assert not _full(raw, raw_cmd), (ego, y0, raw.aeb, raw_cmd.brake)
            rng = random.Random(int(ego) + int(y0))
            tracker = IoUTracker()
            t = 0.0
            dt = 0.1
            for i in range(30):
                y = y0 + rng.uniform(-0.3, 0.3)
                plan, cmd, _planner = _drive(tracker, y, t, ego, i)
                assert not _full(plan, cmd), (ego, y0, i, y, plan.aeb, cmd.brake)
                t += dt


def check_stopped_lead_at_ego_18() -> None:
    """The ego-12 stopped-lead case still brakes outside 8 m at ego 18."""
    rows = _approach(18.0, 0.0)
    far = [row for row in rows if 4.0 < row[0] <= 28.0]
    assert far
    for y, tr, plan, cmd in far:
        assert tr.speed_mps < 1.0, (y, tr.speed_mps)
        assert tr.range_rate_mps < -10.0, (y, tr.range_rate_mps)
        assert cmd.throttle == 0.0, (y, cmd.throttle, cmd.brake, plan.aeb)
    aeb = [row for row in far if row[3].brake >= 0.99 and row[2].aeb == "brake"]
    assert aeb, "stopped lead at ego 18 never reached AEB before 4 m"
    assert max(row[0] for row in aeb) > 8.0


def _roll_to_stop(kind: str, ego: float, y0: float, hz: float = 10.0) -> tuple[float, float]:
    """Approach a fixed VRU. Return (remaining range, speed)."""
    dt = 1.0 / hz
    traveled = 0.0
    t_ego = ego
    for i in range(int(8.0 * hz)):
        y = y0 - traveled
        if y <= 0.0:
            return y, t_ego
        tracks = [{
            "class": kind,
            "x": 0.0,
            "y": y,
            "yaw": math.pi / 2,
            "speed_mps": 0.0,
        }]
        plan = predict_path(
            lanes_bev=_lanes(), lane_conf=0.9, ego_speed_mps=t_ego, tracks=tracks,
        )
        _planner, cmd = _live(plan, tracks, t_ego, i)
        a = _decel(cmd.brake)
        if t_ego <= 0.05 or (a > 1e-6 and a * dt >= t_ego):
            if a > 1e-6:
                traveled += t_ego * t_ego / (2.0 * a)
            return y0 - traveled, 0.0
        traveled += t_ego * dt - 0.5 * a * dt * dt
        t_ego = max(0.0, t_ego - a * dt)
    return y0 - traveled, t_ego


def check_vru_finish_hold() -> None:
    """Below 0.5 m/s a stopped ped used to drop the brake and creep in.

    The reported contact was 3.76 m. Hold the brake while the ped or bike
    is still in the stop box, and stop before the bumper reaches them.
    """
    for kind in ("pedestrian", "bike"):
        for ego in (0.2, 0.4, 0.49, 1.0, 2.0):
            for valid in (None, False):
                track = {
                    "class": kind,
                    "x": 0.0,
                    "y": 3.76,
                    "yaw": math.pi / 2,
                    "speed_mps": 0.0,
                }
                if valid is False:
                    track["range_rate_valid"] = False
                plan = predict_path(
                    lanes_bev=_lanes(), lane_conf=0.9, ego_speed_mps=ego, tracks=[track],
                )
                _planner, cmd = _live(plan, [track], ego, 0)
                assert plan.aeb == "brake", (kind, ego, valid, plan.aeb, plan.stop_reason)
                assert cmd.brake >= 0.99, (kind, ego, valid, cmd.brake)
        left, speed = _roll_to_stop(kind, 0.45, 3.76)
        assert speed <= 0.05, (kind, speed, left)
        assert left > 3.0, (kind, left)
        for hz in (10.0, 15.0):
            left, speed = _roll_to_stop(kind, 4.0, 8.0, hz=hz)
            assert speed <= 0.05, (kind, hz, speed, left)
            assert left > 1.0, (kind, hz, left)
        # 15 Hz used to release at 0.27 m/s once the close fell under 0.4.
        tracks = [{
            "class": kind,
            "x": 0.0,
            "y": 8.0,
            "yaw": math.pi / 2,
            "speed_mps": 0.0,
        }]
        held = predict_path(
            lanes_bev=_lanes(), lane_conf=0.9, ego_speed_mps=0.27, tracks=tracks,
        )
        _planner, held_cmd = _live(held, tracks, 0.27, 0)
        assert held.aeb == "brake", (kind, held.aeb, held.stop_reason)
        assert held_cmd.brake >= 0.99, (kind, held_cmd.brake)


def _stop_margin(y0: float, noise: float, hz: float, seed: int) -> float:
    """Remaining gap when ego 18 stops for a lead fixed at y0. Negative is a hit."""
    rng = random.Random(seed)
    dt = 1.0 / hz
    ego = 18.0
    traveled = 0.0
    t = 0.0
    tracker = IoUTracker()
    for i in range(int(12.0 * hz) + 2):
        true_y = y0 - traveled
        if true_y <= 0.05:
            return true_y
        y = max(0.4, true_y + rng.uniform(-noise, noise))
        _plan, cmd, _planner = _drive(tracker, y, t, ego, i)
        a = _decel(cmd.brake)
        if ego <= 0.05 or (a > 1e-6 and a * dt >= ego):
            if a > 1e-6:
                traveled += ego * ego / (2.0 * a)
            return y0 - traveled
        traveled += ego * dt - 0.5 * a * dt * dt
        ego = max(0.0, ego - a * dt)
        t += dt
    return y0 - traveled


def check_stopped_lead_margins() -> None:
    """Stopped lead from 30 m and 24 m, ego 18, through the live pedal path.

    30 m has room for the filter to see an 18 m/s close and still leave 2 m.
    24 m leaves 3.75 m before a full 8 m/s² stop. One 10 Hz frame at 18 m/s
    is already 1.8 m, so a 2 m remainder is not available once the rate has
    to be measured. The number is printed either way.
    """
    margins: dict[tuple[float, float, float], float] = {}
    for y0 in (30.0, 24.0):
        for noise in (0.5, 1.0, 2.0):
            for hz in (10.0, 15.0):
                left = [
                    _stop_margin(y0, noise, hz, 4000 + trial + int(y0) * 50 + int(noise * 10))
                    for trial in range(40)
                ]
                margins[(y0, noise, hz)] = min(left)
    for (y0, noise, hz), worst in margins.items():
        if y0 == 30.0 and noise <= 1.0:
            assert worst >= 2.0, (y0, noise, hz, worst)
    worst30 = min(v for (y0, _n, _h), v in margins.items() if y0 == 30.0)
    worst24 = min(v for (y0, _n, _h), v in margins.items() if y0 == 24.0)
    worst30_2 = min(v for (y0, n, _h), v in margins.items() if y0 == 30.0 and n >= 2.0)
    print(
        "stopped-lead margins m:",
        {k: round(v, 2) for k, v in sorted(margins.items())},
    )
    # ±2 m at 10 Hz does not leave 2 m: the rate is not identifiable in the
    # fraction of a second before an 8 m/s² stop is already late. It must
    # still be a stop, not a high-speed hit.
    assert worst30_2 > 0.5, worst30_2
    # 24 m leaves 3.75 m before a full stop from 18 m/s. One frame at 10 Hz
    # is 1.8 m, so the filter's lag spends the 2 m remainder. A graze is
    # that limit. Metres of overlap would be a brake that never came on.
    assert worst24 > -1.0, worst24


def check_slow_close_brakes_early() -> None:
    """A 2 m/s or 4 m/s close from 40 m brakes while the gap is still long.

    The old net-close gate never confirmed this, so the first brake was the
    8 m wall. Matching the lead's speed is the stop. Ego does not go to 0.
    """
    for rel in (2.0, 4.0):
        for noise in (0.5, 1.0, 2.0):
            for hz in (10.0, 15.0):
                for seed in range(40):
                    rng = random.Random(seed + int(rel * 10) + int(hz) + int(noise * 10))
                    dt = 1.0 / hz
                    ego = 18.0
                    lead_v = 18.0 - rel
                    gap = 40.0
                    t = 0.0
                    tracker = IoUTracker()
                    first = None
                    max_brake = 0.0
                    min_gap = gap
                    for i in range(int(20.0 * hz)):
                        y = max(0.4, gap + rng.uniform(-noise, noise))
                        plan, cmd, _planner = _drive(tracker, y, t, ego, i)
                        if cmd.brake > 0.02 and first is None:
                            first = gap
                        max_brake = max(max_brake, cmd.brake)
                        a = _decel(cmd.brake)
                        if ego <= 0.05 or (a > 1e-6 and a * dt >= ego):
                            extra = ego * ego / (2.0 * a) if a > 1e-6 else 0.0
                            gap = gap + lead_v * (ego / a if a > 1e-6 else 0.0) - extra
                            ego = 0.0
                            min_gap = min(min_gap, gap)
                            break
                        ego_move = ego * dt - 0.5 * a * dt * dt
                        gap = gap + lead_v * dt - ego_move
                        ego = max(0.0, ego - a * dt)
                        min_gap = min(min_gap, gap)
                        t += dt
                        if abs(ego - lead_v) < 0.5 and i > int(1.5 * hz) and gap > 8.0:
                            break
                    assert first is not None and first > 20.0, (rel, noise, hz, seed, first)
                    assert max_brake < 0.99, (rel, noise, hz, seed, max_brake)
                    assert ego > lead_v - 1.0, (rel, noise, hz, seed, ego, lead_v)
                    assert min_gap > 8.0, (rel, noise, hz, seed, min_gap, ego)


def check_pacing_lead_brake() -> None:
    """A matched lead that then brakes at 6 m/s². Report the remaining gap.

    Perfect knowledge and an 8 m/s² cap still meet this lead late when the
    gap starts near the stopping distance. 36 m has room. Shorter gaps are
    printed and only fail a deep impact.
    """
    found: dict[tuple[float, float, float], float] = {}
    for y0 in (12.5, 16.0, 20.0, 36.0):
        for noise in (0.0, 0.5, 1.0, 2.0):
            for hz in (10.0, 15.0):
                worst = 99.0
                for seed in range(40):
                    rng = random.Random(seed + int(y0 * 10) + int(hz) + int(noise * 10))
                    dt = 1.0 / hz
                    ego = 18.0
                    lead_v = 18.0
                    gap = y0
                    t = 0.0
                    tracker = IoUTracker()
                    for i in range(int(1.0 * hz)):
                        y = max(0.4, gap + rng.uniform(-noise, noise))
                        _drive(tracker, y, t, ego, i)
                        t += dt
                    end = gap
                    for i in range(int(8.0 * hz)):
                        a_lead = 6.0 if lead_v > 0.0 else 0.0
                        y = max(0.4, gap + rng.uniform(-noise, noise))
                        _plan, cmd, _planner = _drive(tracker, y, t, ego, i + 100)
                        a = _decel(cmd.brake)
                        if ego <= 0.05 or (a > 1e-6 and a * dt >= ego):
                            extra = ego * ego / (2.0 * a) if a > 1e-6 else 0.0
                            t_stop = ego / a if a > 1e-6 else 0.0
                            lead_move = max(0.0, lead_v * t_stop - 0.5 * a_lead * t_stop * t_stop)
                            end = gap + lead_move - extra
                            break
                        ego_move = ego * dt - 0.5 * a * dt * dt
                        lead_move = lead_v * dt - 0.5 * a_lead * dt * dt
                        gap = gap + lead_move - ego_move
                        ego = max(0.0, ego - a * dt)
                        lead_v = max(0.0, lead_v - a_lead * dt)
                        t += dt
                        end = gap
                        if gap <= 0.0:
                            break
                    worst = min(worst, end)
                found[(y0, noise, hz)] = worst
    print("lead-brake gaps m:", {k: round(v, 2) for k, v in found.items()})
    for (y0, noise, hz), gap in found.items():
        if y0 == 12.5 and noise <= 0.5:
            # 6 m/s² from 12.5 m, clean and ±0.5 m. The stop has to leave
            # at least a metre when the brake can still make that gap.
            assert gap >= 1.0, (y0, noise, hz, gap)
        elif y0 == 12.5 and noise <= 1.0:
            # ±1 m at 12.5 m can graze. It must not go past half a metre.
            assert gap > -0.5, (y0, noise, hz, gap)
        elif y0 in (16.0, 20.0) and noise == 1.0:
            # ±1 m from 16 m and 20 m used to overlap. The comfort stop
            # inside the cut has to leave a positive gap on every seed.
            assert gap > 0.0, (y0, noise, hz, gap)
        elif noise <= 0.5 or y0 >= 36.0:
            # A quiet pace that breaks gets a full stop in time. ±2 m can
            # hide that break, and the filter then meets a short gap late.
            assert gap > 0.5, (y0, noise, hz, gap)
        else:
            assert gap > -1.5, (y0, noise, hz, gap)


def check_live_tick_publishes_range_brake() -> None:
    """``tick`` itself, not ``follow_path``, puts range_brake on the actuator."""
    import numpy as np

    from python.perception.pipeline import ModularPerception

    cv2 = __import__("cv2")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(img, (80, 470), (300, 280), (220, 220, 220), 4)
    cv2.line(img, (560, 470), (340, 280), (220, 220, 220), 4)

    class _Lead:
        name = "lead"

        def detect(self, bgr):
            return [
                Detection(
                    cls="vehicle", conf=0.9, xyxy=(0, 0, 10, 10),
                    x=0.0, y=22.0, yaw=math.pi / 2,
                )
            ]

    perc = ModularPerception(allow_synthetic=False, detector_id="empty")
    perc.set_detector(_Lead(), [])
    out = perc.tick(img, ego_speed_mps=12.0)
    assert "range_brake" in out.planner, out.planner.keys()
    assert out.planner["range_brake"] > 0.05, out.planner
    assert out.planner["aeb"] != "brake", out.planner
    cmd = plan_command(
        path_ego=out.path_ego, planner=out.planner, ego_speed_mps=12.0, seq=1,
    )
    assert cmd.throttle == 0.0 and cmd.brake > 0.05, cmd
    # The same scene with the key stripped is the bug: speed error does not brake.
    stripped = dict(out.planner)
    stripped.pop("range_brake")
    bare = plan_command(
        path_ego=out.path_ego, planner=stripped, ego_speed_mps=12.0, seq=2,
    )
    assert bare.brake < cmd.brake, (bare.brake, cmd.brake)


def check_nonfinite_pose_does_not_brake() -> None:
    """NaN range is not a miss that spawns a track, and it is not a full stop."""
    from python.perception.detect import pose_ok
    from python.planning.path_predictor import _required_brake

    nan = Detection(
        cls="vehicle", conf=0.9, xyxy=(0, 0, 10, 10),
        x=0.0, y=float("nan"), yaw=math.pi / 2,
    )
    assert pose_ok(nan) is False
    inf = Detection(
        cls="vehicle", conf=0.9, xyxy=(0, 0, 10, 10),
        x=float("inf"), y=12.0, yaw=math.pi / 2,
    )
    assert pose_ok(inf) is False
    fresh = IoUTracker()
    fresh.update([nan, inf], 0.0, ego_speed_mps=12.0)
    assert fresh.tracks == []

    held = IoUTracker()
    good = Detection(
        cls="vehicle", conf=0.9, xyxy=(0, 0, 10, 10),
        x=0.0, y=16.0, yaw=math.pi / 2,
    )
    held.update([good], 0.0, ego_speed_mps=12.0)
    held.update([nan], 0.1, ego_speed_mps=12.0)
    assert len(held.tracks) == 1
    assert held.tracks[0].misses == 1
    # Coast may slide the prediction. The pose itself stays a real range.
    assert math.isfinite(held.tracks[0].y)
    assert abs(held.tracks[0].y - 16.0) < 1.0

    assert _required_brake(float("nan"), -8.0, 0.2) == 0.0
    assert _required_brake(12.0, float("nan"), 0.2) == 0.0
    assert _required_brake(12.0, -8.0, float("inf")) == 0.0
    for bad_y in (float("nan"), float("inf"), float("-inf")):
        tracks = [{
            "class": "vehicle",
            "x": 0.0,
            "y": bad_y,
            "yaw": math.pi / 2,
            "speed_mps": 0.0,
            "range_rate_mps": -18.0,
            "range_rate_sigma": 0.2,
            "range_rate_valid": True,
        }]
        plan = predict_path(
            lanes_bev=_lanes(), lane_conf=0.9, ego_speed_mps=18.0, tracks=tracks,
        )
        _planner, cmd = _live(plan, tracks, 18.0, 0)
        assert plan.aeb != "brake", (bad_y, plan.aeb)
        assert cmd.brake < 0.99, (bad_y, cmd.brake)


def _curve_lanes(radius: float, sign: float) -> list:
    """Lane lines on an arc. ``sign`` +1 turns right, -1 turns left."""
    left, right = [], []
    center = sign * radius
    for i in range(0, 46):
        theta = i * 1.5 / radius
        # Outer line is the left line on a right turn.
        for radius_line, bucket, side in (
            (radius + 1.75, left, -1.0),
            (radius - 1.75, right, 1.0),
        ):
            if sign < 0:
                bucket = right if side < 0 else left
            x = center - sign * radius_line * math.cos(theta)
            y = radius_line * math.sin(theta)
            bucket.append({"x": x, "y": y})
    return [left, right]


def check_curve_cipv_follows_the_path() -> None:
    """A stopped car in the straight tube is not a lead once the path has bent.

    Both directions. On the path, the same stop still brakes. Off the path,
    a forged straight-tube CIPV brake does not override the route.
    """
    ego = 18.0
    for sign in (1.0, -1.0):
        lanes = _curve_lanes(80.0, sign)
        empty = predict_path(lanes_bev=lanes, lane_conf=0.9, ego_speed_mps=ego)
        route = empty.route_ego or empty.path_ego
        assert len(route) >= 8, (sign, len(route))
        on = min(route, key=lambda p: abs(p["y"] - 20.0))
        assert abs(on["y"] - 20.0) < 2.0, on
        off = {
            "class": "vehicle",
            "id": 3,
            "x": 0.0,
            "y": 20.0,
            "yaw": math.pi / 2,
            "speed_mps": 0.0,
        }
        tube = select_cipv([off], ego_speed_mps=ego)
        assert tube.aeb == "brake", (sign, tube.aeb, tube.ttc_lead)
        along = select_cipv([off], ego_speed_mps=ego, path_ego=route)
        assert along.aeb == "off", (sign, along.aeb, on)
        off_plan = predict_path(
            lanes_bev=lanes, lane_conf=0.9, ego_speed_mps=ego, tracks=[off],
        )
        _planner, off_cmd = _live(off_plan, [off], ego, 0)
        assert off_plan.aeb != "brake", (sign, off_plan.aeb, off_plan.range_brake)
        assert off_cmd.brake < 0.99, (sign, off_cmd.brake, off_cmd.throttle)

        class _Forged:
            track = off
            ttc_lead = 20.0 / ego
            aeb = "brake"

        forced = publish_planner(off_plan, _Forged())
        assert forced["aeb"] != "brake", (sign, forced["aeb"])
        forced_cmd = plan_command(
            path_ego=off_plan.path_ego, planner=forced, ego_speed_mps=ego, seq=1,
        )
        assert forced_cmd.brake < 0.99, (sign, forced_cmd)

        lead = {
            "class": "vehicle",
            "id": 4,
            "x": on["x"],
            "y": on["y"],
            "yaw": math.pi / 2,
            "speed_mps": 0.0,
        }
        hit = select_cipv([lead], ego_speed_mps=ego, path_ego=route)
        assert hit.aeb == "brake", (sign, hit.aeb, on)
        on_plan = predict_path(
            lanes_bev=lanes, lane_conf=0.9, ego_speed_mps=ego, tracks=[lead],
        )
        _planner, on_cmd = _live(on_plan, [lead], ego, 2)
        assert on_cmd.brake >= 0.99, (sign, on_cmd.brake, on_plan.aeb, on)


def check_faster_lead_is_not_closing() -> None:
    rows = _approach(12.0, 16.0)
    late = [row for row in rows if row[0] > 45.0]
    assert late
    for y, tr, plan, cmd in late:
        assert abs(tr.speed_mps - 16.0) < 0.5, (y, tr.speed_mps)
        assert tr.range_rate_mps > 3.0, (y, tr.range_rate_mps)
        assert plan.aeb == "off" and cmd.brake == 0.0, (y, plan.aeb, cmd.brake)
        assert plan.stop_reason == "none", (y, plan.stop_reason)


def _roll(gap_fn, seconds: float, hz: float = 10.0, ego0: float = 18.0, xy_fn=None):
    """Brake-only roll through ``plan_command``. Loss is the drop from ego0."""
    dt = 1.0 / hz
    ego = ego0
    t = 0.0
    tracker = IoUTracker()
    min_ego = ego
    tripped = False
    for i in range(int(seconds * hz)):
        gap = float(gap_fn(t))
        if xy_fn is None:
            x, y, lanes = 0.0, gap, None
        else:
            x, y, lanes = xy_fn(t, gap)
        plan, cmd, _planner = _drive(tracker, y, t, ego, i, x=x, lanes=lanes)
        tripped = tripped or plan.aeb == "brake" or cmd.brake >= 0.99
        a = _decel(cmd.brake)
        ego = max(0.0, ego - a * dt)
        min_ego = min(min_ego, ego)
        t += dt
    ids = [tr.id for tr in tracker.tracks]
    rate = tracker.tracks[0].range_rate_mps if tracker.tracks else None
    return {
        "loss": ego0 - min_ego,
        "aeb": tripped,
        "tracks": len(tracker.tracks),
        "ids": ids,
        "rate": rate,
        "brake": cmd.brake,
    }


def _assert_no_phantom(name: str, row: dict) -> None:
    assert row["loss"] <= 1.5, (name, round(row["loss"], 2), row)
    assert row["aeb"] is False, (name, row)


def check_gap_drop_does_not_phantom_stop() -> None:
    """A route-gap wiggle is not a latched full stop.

    The old latch forced brake 1.0 after the filtered rate was already
    flat, and that promoted aeb. These rolls go through ``plan_command``.
    A 1.6 m step, a same-track 2.5 m or 4 m swap, a 1.2 m sine, and 45 s
    of correlated noise each stay inside 1.5 m/s with aeb off.
    """
    def step(time: float) -> float:
        return 27.0 if time < 2.0 else 25.4

    row = _roll(step, 6.0)
    _assert_no_phantom("step-1.6", row)
    assert row["tracks"] == 1 and row["ids"] == [1], row
    assert abs(row["rate"]) < 0.5, row
    assert row["brake"] < 0.2, row
    phantom = {"step": round(row["loss"], 2)}

    for jump in (2.5, 4.0):
        def swap(time: float, jump: float = jump) -> float:
            return 27.0 if time < 2.0 else 27.0 - jump

        row = _roll(swap, 5.0)
        _assert_no_phantom(f"swap-{jump}", row)
        assert row["tracks"] == 1 and row["ids"] == [1], row
        phantom[f"swap-{jump}"] = round(row["loss"], 2)

    def sine(time: float) -> float:
        return 27.0 + 1.2 * math.sin(2.0 * math.pi * time / 4.0)

    row = _roll(sine, 8.0)
    _assert_no_phantom("sine-1.2", row)
    phantom["sine"] = round(row["loss"], 2)

    losses = []
    for seed in range(6):
        rng = random.Random(seed)
        noise = 0.0

        def ar(time: float, rng: random.Random = rng) -> float:
            nonlocal noise
            noise = 0.9 * noise + 0.45 * rng.gauss(0.0, 1.0)
            return 27.0 + noise

        row = _roll(ar, 45.0)
        _assert_no_phantom(f"ar-{seed}", row)
        losses.append(round(row["loss"], 2))
    phantom["ar"] = losses
    print("phantom-drop losses m/s:", phantom)


def _radius_for_forward(y: float, gap: float) -> float:
    """Radius whose centerline forward range is ``y`` at arc gap ``gap``."""
    lo, hi = 40.0, 400.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if mid * math.sin(gap / mid) < y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def check_curve_entry_does_not_phantom_stop() -> None:
    """Entering a 40 m curve shrinks ego-frame y. That is not a stop.

    The curve's own speed limit sheds speed with no lead. The lead, held
    at a constant arc gap, must not add more than 1.5 m/s, and must not
    promote aeb. The radius moves so forward range changes by less than
    0.2 m per frame: a discontinuous jump is a different bug.
    """
    gap = 30.0
    y_far = 400.0 * math.sin(gap / 400.0)
    y_near = 40.0 * math.sin(gap / 40.0)

    def xy(time: float, _gap: float):
        if time < 1.0:
            radius = 400.0
            y = y_far
        else:
            u = min(1.0, (time - 1.0) / 3.0)
            y = y_far + (y_near - y_far) * u
            radius = 400.0 if u <= 0.0 else (40.0 if u >= 1.0 else _radius_for_forward(y, gap))
        lanes = _curve_lanes(radius, 1.0)
        x = radius * (1.0 - math.cos(gap / radius))
        return x, y, lanes

    lead = _roll(lambda _time: gap, 6.0, xy_fn=xy)
    # Same bend, no vehicle. Reuse the roller with a gap the driver never sees.
    empty_loss = 0.0
    dt = 0.1
    ego = 18.0
    min_ego = ego
    for i in range(60):
        time = i * dt
        _x, _y, lanes = xy(time, gap)
        plan = predict_path(
            lanes_bev=lanes, lane_conf=0.9, ego_speed_mps=ego, tracks=[],
        )
        _planner, cmd = _live(plan, [], ego, i)
        assert plan.aeb != "brake" and cmd.brake < 0.99, (time, plan.aeb, cmd.brake)
        a = _decel(cmd.brake)
        ego = max(0.0, ego - a * dt)
        min_ego = min(min_ego, ego)
    empty_loss = 18.0 - min_ego
    excess = lead["loss"] - empty_loss
    print(
        "curve-entry excess m/s:",
        round(excess, 2),
        "lead",
        round(lead["loss"], 2),
        "empty",
        round(empty_loss, 2),
    )
    assert excess <= 1.5, (excess, lead, empty_loss)
    assert lead["aeb"] is False, lead


def check_nonpositive_ttc_is_not_a_brake() -> None:
    """Zero, negative, and non-finite time-to-collision do not full-brake.

    A finite ttc above 0 and under 1.2 still does. 1.2 itself does not.
    """
    path = [{"x": 0.0, "y": float(i)} for i in range(0, 30, 2)]
    base = {"aeb": "off", "target_v": 18.0, "range_brake": 0.0, "blinker": "off"}
    for bad in (-1.0, 0.0, float("-inf"), float("nan"), float("inf")):
        cmd = plan_command(
            path_ego=path,
            planner=dict(base, ttc_lead=bad),
            ego_speed_mps=18.0,
            seq=1,
        )
        assert cmd.brake < 0.99, (bad, cmd.brake)
    full = plan_command(
        path_ego=path,
        planner=dict(base, ttc_lead=1.0),
        ego_speed_mps=18.0,
        seq=2,
    )
    assert full.brake >= 0.99, full.brake
    edge = plan_command(
        path_ego=path,
        planner=dict(base, ttc_lead=1.2),
        ego_speed_mps=18.0,
        seq=3,
    )
    assert edge.brake < 0.99, edge.brake


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
    check_flat_gap_confirms_not_closing()
    check_pacing_lead_holds_speed()
    check_matched_cut_in_no_full_brake()
    check_stopped_lead_at_ego_18()
    check_vru_finish_hold()
    check_stopped_lead_margins()
    check_slow_close_brakes_early()
    check_pacing_lead_brake()
    check_nonfinite_pose_does_not_brake()
    check_curve_cipv_follows_the_path()
    check_live_tick_publishes_range_brake()
    check_faster_lead_is_not_closing()
    check_gap_drop_does_not_phantom_stop()
    check_curve_entry_does_not_phantom_stop()
    check_nonpositive_ttc_is_not_a_brake()
    print("test_lead_road_speed: OK")


if __name__ == "__main__":
    main()
