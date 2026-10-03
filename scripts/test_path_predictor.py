#!/usr/bin/env python3
"""Predicted route from lanes, cars, signs, and lights. The wheel is not an input.

Partial lane lines are continued. A car that is only partly seen, or briefly
unseen, still occupies a predicted footprint the path stops short of. A
roundabout arc is continued around the fitted circle. Engaged, in a drive
gear, steer and throttle come from that path. A camera read on the GE socket
still skips vehicle.control; the next free tick commands again. A pedal pull
still disengages.
"""
from __future__ import annotations

import inspect
import math
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from python.control.actuate import (  # noqa: E402
    TECH_DRIVE_GEAR,
    TECH_SHIFT_MODE,
    BeamNGPyActuator,
    note_soft_esc_engaged,
    soft_esc_sensors_every_tick,
)
from python.control.override import OverrideDetector, load_override_config  # noqa: E402
from python.planning.path_predictor import follow_path, path_length_m, predict_path  # noqa: E402
from python.runtime.shadow import ShadowConfig, shadow_tick  # noqa: E402

CONTROL_YAML = ROOT / "config" / "control.yaml"
YAW_AHEAD = math.pi / 2.0


def _lane(left_x: float, right_x: float, y0: float = 0.0, y1: float = 40.0, step: float = 2.0):
    ys = []
    y = y0
    while y <= y1 + 1e-9:
        ys.append(y)
        y += step
    left = [{"x": left_x, "y": y, "z": 0.0} for y in ys]
    right = [{"x": right_x, "y": y, "z": 0.0} for y in ys]
    return [left, right]


def _max_y(plan) -> float:
    return max(p["y"] for p in plan.path_ego)


def _near(plan, y: float) -> dict:
    return min(plan.path_ego, key=lambda p: abs(p["y"] - y))


def check_wheel_is_not_an_input() -> None:
    params = inspect.signature(predict_path).parameters
    assert not any("steer" in name.lower() for name in params), list(params)
    src = (ROOT / "python" / "planning" / "path_predictor.py").read_text(encoding="utf-8")
    assert "steer_deg" not in src
    assert "steering_input" not in src
    try:
        predict_path(lanes_bev=_lane(-1.75, 1.75), steer_deg=30.0)  # type: ignore[call-arg]
    except TypeError:
        pass
    else:
        raise AssertionError("predict_path accepted a wheel angle")

    plain = _lane(-1.75, 1.75)
    marked = [
        [{**p, "steer_deg": 40.0, "steering_input": -0.8} for p in poly]
        for poly in plain
    ]
    a = predict_path(lanes_bev=plain, lane_conf=0.9, ego_speed_mps=5.0)
    b = predict_path(lanes_bev=marked, lane_conf=0.9, ego_speed_mps=5.0)
    assert a.path_ego == b.path_ego
    assert a.target_v == b.target_v and a.aeb == b.aeb


def check_full_lane_and_commands() -> None:
    plan = predict_path(lanes_bev=_lane(-1.75, 1.75), lane_conf=0.9, ego_speed_mps=5.0)
    assert plan.drivable and plan.prediction == "lanes", plan.prediction
    assert _max_y(plan) >= 30.0
    for p in plan.path_ego:
        assert -1.75 < p["x"] < 1.75, p
    assert abs(_near(plan, 12.0)["x"]) < 0.25
    cmd = follow_path(plan, ego_speed_mps=5.0, seq=3)
    assert cmd.throttle > 0.0 and cmd.brake == 0.0, cmd
    assert abs(cmd.steer) < 0.2, cmd
    # Predicted throttle is the route length in meters, not the pedal command.
    measured = path_length_m(plan.path_ego)
    assert abs(plan.path_length_m - measured) < 1e-6, (plan.path_length_m, measured)
    assert plan.path_length_m > 10.0
    assert plan.planner_dict()["path_length_m"] == plan.path_length_m
    assert abs(plan.pred_brake - cmd.brake) < 1e-6
    assert plan.path_length_m != cmd.throttle


def check_partial_lane_is_continued() -> None:
    # Only the left line, and only the first 10 m. Context is that line's heading
    # and a default lane width. The route continues past what was in frame.
    left = [{"x": -1.75, "y": float(y), "z": 0.0} for y in range(2, 12, 2)]
    plan = predict_path(lanes_bev=[left], lane_conf=0.4, ego_speed_mps=5.0)
    assert plan.drivable and plan.prediction == "extended", plan.prediction
    assert _max_y(plan) >= 30.0, _max_y(plan)
    for p in plan.path_ego:
        assert -1.75 < p["x"] < 1.75, p
    far = _near(plan, 28.0)
    assert abs(far["x"]) < 0.35, far

    # The visible piece is already turning. Continuation keeps that heading,
    # it does not snap back to a straight-ahead guess.
    slope = 0.08
    curved = [{"x": -1.75 + slope * y, "y": float(y)} for y in (2, 4, 6, 8, 10)]
    bent = predict_path(lanes_bev=[curved], lane_conf=0.45, ego_speed_mps=4.0)
    assert bent.prediction == "extended"
    assert _max_y(bent) > 20.0
    sample = _near(bent, 30.0)
    # Centerline sits half a width to the right of the left line, same slope.
    expect = slope * sample["y"]
    assert abs(sample["x"] - expect) < 0.4, (sample, expect)
    left_x = -1.75 + slope * sample["y"]
    assert left_x < sample["x"] < left_x + 3.5

    # Nothing in this frame. The prior lane is the context, and we still drive it.
    missing = predict_path(
        lanes_bev=[],
        prior_lanes=_lane(-1.75, 1.75),
        lane_conf=0.0,
        ego_speed_mps=5.0,
    )
    assert missing.drivable and missing.prediction == "context", missing.prediction
    assert _max_y(missing) >= 30.0
    for p in missing.path_ego:
        assert -1.75 < p["x"] < 1.75, p
    assert predict_path(lanes_bev=[], lane_conf=0.0).drivable is False


def check_car_stop_sign_and_light() -> None:
    lanes = _lane(-1.75, 1.75)
    free = predict_path(lanes_bev=lanes, lane_conf=0.9, ego_speed_mps=8.0)
    car = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=8.0,
        tracks=[{"id": 7, "class": "vehicle", "x": 0.1, "y": 18.0, "yaw": YAW_AHEAD, "speed_mps": 0.0}],
    )
    assert car.stop_reason == "vehicle" and car.cipv_id == 7
    assert _max_y(car) < 18.0 - 1.5
    assert _max_y(car) < _max_y(free) - 5.0
    assert car.path_length_m < free.path_length_m - 4.0
    assert abs(car.path_length_m - path_length_m(car.path_ego)) < 1e-6
    held = follow_path(car, ego_speed_mps=8.0, seq=1)
    # 18 m is not a full stop. Brake scales; it is not 1 while the path is still long.
    assert held.brake < 1.0, held

    # A car in the next lane is not our footprint.
    beside = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=8.0,
        tracks=[{"class": "vehicle", "x": 5.5, "y": 18.0, "yaw": YAW_AHEAD, "speed_mps": 0.0}],
    )
    assert beside.stop_reason == "none"
    assert _max_y(beside) > 30.0

    stop = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=8.0,
        signs=[{"cls": "stop_sign", "x": 2.2, "y": 24.0, "partial": True}],
    )
    assert stop.stop_reason == "stop_sign"
    assert _max_y(stop) > 20.0
    stopped = follow_path(stop, ego_speed_mps=8.0, seq=2)
    assert stopped.brake < 1.0, stopped

    # A sign on the cross street is not this route.
    other = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=8.0,
        signs=[{"cls": "stop_sign", "x": 14.0, "y": 24.0}],
    )
    assert other.stop_reason == "none"
    assert _max_y(other) > 30.0

    red = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=6.0,
        signs=[{"cls": "traffic_light", "x": 0.4, "y": 30.0, "state": "red", "partial": True, "misses": 1}],
    )
    assert red.stop_reason == "red_light"
    assert _max_y(red) > 20.0
    assert follow_path(red, ego_speed_mps=6.0).brake < 1.0

    green = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=6.0,
        signs=[{"cls": "traffic_light", "x": 0.4, "y": 30.0, "state": "green"}],
    )
    unknown = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=6.0,
        signs=[{"cls": "traffic_light", "x": 0.4, "y": 30.0, "state": "unknown"}],
    )
    assert green.stop_reason == "none" and unknown.stop_reason == "none"
    assert _max_y(green) > 30.0 and _max_y(unknown) > 30.0

    limited = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=8.0,
        signs=[{"cls": "speed_limit", "x": 2.0, "y": 16.0, "limit_mps": 5.0}],
    )
    assert limited.stop_reason == "none"
    assert limited.target_v <= 5.0
    assert _max_y(limited) > 30.0


def check_unseen_and_partial_car() -> None:
    lanes = _lane(-1.75, 1.75)
    still = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=8.0,
        tracks=[{
            "id": 3,
            "class": "car",
            "x": 0.0,
            "y": 16.0,
            "yaw": YAW_AHEAD,
            "speed_mps": 0.0,
            "unseen_s": 0.4,
        }],
    )
    # Slower than us, unseen for 0.4 s: the footprint is the coasted point, and the path stops short of it.
    moved = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=8.0,
        tracks=[{
            "id": 3,
            "class": "car",
            "x": 0.0,
            "y": 16.0,
            "yaw": YAW_AHEAD,
            "speed_mps": 2.0,
            "misses": 4,
            "tick_s": 0.1,
        }],
    )
    assert still.occupied and moved.occupied
    assert abs(still.occupied[0]["y"] - 16.0) < 0.2
    # 4 misses * 0.1 s * 2 m/s along the last heading. Not the stale point.
    assert moved.occupied[0]["y"] > still.occupied[0]["y"] + 0.5
    assert _max_y(still) < 16.0
    assert _max_y(moved) > _max_y(still) + 0.4
    assert _max_y(moved) < moved.occupied[0]["y"] - 1.5

    sliver = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=8.0,
        tracks=[{"class": "vehicle", "x": 0.0, "y": 22.0, "yaw": YAW_AHEAD, "length": 0.4, "partial": False}],
    )
    partial = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=8.0,
        tracks=[{
            "class": "vehicle",
            "x": 0.0,
            "y": 22.0,
            "yaw": YAW_AHEAD,
            "length": 0.4,
            "seen_fraction": 0.25,
        }],
    )
    assert partial.occupied[0]["length"] >= 4.5
    assert _max_y(partial) < 22.0 - 2.0
    assert _max_y(partial) < _max_y(sliver) - 1.0


def check_partial_roundabout() -> None:
    cx, cy, radius = 0.0, 16.0, 12.0
    visible = []
    for deg in range(-120, -84, 5):
        ang = math.radians(deg)
        visible.append({
            "x": cx + radius * math.cos(ang),
            "y": cy + radius * math.sin(ang),
            "z": 0.0,
        })
    plan = predict_path(lanes_bev=[visible], lane_conf=0.7, ego_speed_mps=0.0)
    assert plan.drivable and plan.prediction == "roundabout", plan.prediction
    vis_max = max(math.atan2(p["y"] - cy, p["x"] - cx) for p in visible)

    def on_circle(p) -> bool:
        return abs(math.hypot(p["x"] - cx, p["y"] - cy) - radius) < 0.8

    predicted = [p for p in plan.path_ego if on_circle(p) and math.atan2(p["y"] - cy, p["x"] - cx) > vis_max + 0.7]
    assert predicted, "path never left the visible arc"
    # A straight extrapolation of the entry would stay near y=4. The unseen
    # arc climbs away from that tangent.
    assert max(p["y"] for p in predicted) > 8.0

    cmd = follow_path(plan, ego_speed_mps=0.0, seq=4)
    assert cmd.throttle > 0.0 and cmd.brake == 0.0, cmd
    assert cmd.steer > 0.3, cmd

    tick = shadow_tick(
        policy="modular",
        engaged=True,
        heartbeat_ok=True,
        path_debug_preview=not plan.drivable,
        allow_preview_drive=False,
        path_ego=plan.path_ego,
        planner=plan.planner_dict(),
        ego_speed_mps=0.0,
        lane_conf=0.7,
        path_conf=plan.path_conf,
        seq=4,
        e2e_policy=_NoE2E(),
        cfg=ShadowConfig(),
    )
    assert tick.applied.reason == "ok", tick.applied
    assert abs(tick.applied.steer - cmd.steer) < 1e-6
    assert abs(tick.applied.throttle - cmd.throttle) < 1e-6
    return cmd


class _NoE2E:
    backend = "stub"

    def forward(self, *args, **kwargs):
        from python.control.e2e import E2EIntent

        return E2EIntent(steer=0.0, accel=0.0, throttle=0.0, brake=0.0, ok=True, backend="stub", reason="stub")


class _Busy:
    def __init__(self) -> None:
        self.busy = False

    def _socket_io_busy(self) -> bool:
        return self.busy


class _Veh:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.shifts: list[str] = []
        self.sensors = {
            "electrics": {"gear": "D", "wheelspeed": 0.0, "steering_input": -0.45},
        }

    def set_shift_mode(self, mode: str) -> None:
        self.shifts.append(mode)

    def ai_set_mode(self, mode: str) -> None:
        return None

    def control(self, **kw) -> None:
        self.calls.append(kw)

    def queue_lua_command(self, *_a, **_k) -> None:
        return None


def check_drive_gear_socket_and_override(cmd) -> None:
    """Engaged and in drive: vehicle.control gets the path, not the wheel.

    The socket skip and the pedal override stay in front of that command.
    """
    import python.sensors.cameras as cams

    prev_latch = soft_esc_sensors_every_tick()
    prev_docs = os.environ.get("GVD_DOCS_DIR")
    tmp = tempfile.TemporaryDirectory()
    os.environ["GVD_DOCS_DIR"] = tmp.name
    busy = _Busy()
    cams._camera_backends.add(busy)
    veh = _Veh()
    act = BeamNGPyActuator(veh)
    try:
        act.note_engaged(True)
        busy.busy = True
        blocked = act.apply(cmd)
        assert blocked.applied is False and blocked.reason == "camera_io_busy", blocked
        assert veh.calls == []
        assert veh.shifts == []

        busy.busy = False
        fresh = follow_path(
            predict_path(lanes_bev=[_roundabout_arc()], lane_conf=0.7, ego_speed_mps=0.0),
            ego_speed_mps=0.0,
            seq=8,
        )
        assert fresh.throttle > 0.0 and fresh.steer > 0.3
        sent = act.apply(fresh)
        assert sent.applied is True, sent
        assert veh.shifts == [TECH_SHIFT_MODE]
        assert len(veh.calls) == 1, veh.calls
        drove = veh.calls[0]
        assert abs(drove["steering"] - fresh.steer) < 1e-6
        assert abs(drove["throttle"] - fresh.throttle) < 1e-6
        assert int(drove["gear"]) >= TECH_DRIVE_GEAR
        # Electrics still say the wheel is left. The command is the path's right turn.
        assert drove["steering"] != veh.sensors["electrics"]["steering_input"]
        assert drove["steering"] > 0.3 and veh.sensors["electrics"]["steering_input"] < 0.0

        cfg = load_override_config(yaml.safe_load(CONTROL_YAML.read_text(encoding="utf-8")))
        det = OverrideDetector(cfg)
        t = 100.0
        echo = (veh.sensors["electrics"]["steering_input"], fresh.throttle, 0.0)
        armed = False
        for _ in range(80):
            t += 0.05
            det.update(
                engaged=True,
                steering_input=echo[0],
                throttle_input=echo[1],
                brake_input=echo[2],
                now=t,
                own_axes=True,
                player_device=True,
            )
            det.note_command(seq=8, steer=fresh.steer, throttle=fresh.throttle, brake=0.0, now=t)
            if det.armed(t):
                armed = True
                break
        assert armed
        t += 0.05
        pulled = det.update(
            engaged=True,
            steering_input=echo[0],
            throttle_input=echo[1],
            brake_input=0.2,
            now=t,
            own_axes=True,
            player_device=True,
        )
        assert pulled.active and pulled.channel == "brake", pulled
        act.note_engaged(False)
        released = act.stop(seq=9, reason=pulled.reason)
        assert released.throttle == 0.0 and released.brake == 0.0
        assert all(float(c.get("throttle") or 0.0) == 0.0 for c in veh.calls[1:])
        assert not any(float(c.get("throttle") or 0.0) > 0.0 and float(c.get("steering") or 0.0) > 0.3 for c in veh.calls[1:])
    finally:
        cams._camera_backends.discard(busy)
        act.note_engaged(False)
        note_soft_esc_engaged(prev_latch)
        if prev_docs is None:
            os.environ.pop("GVD_DOCS_DIR", None)
        else:
            os.environ["GVD_DOCS_DIR"] = prev_docs
        tmp.cleanup()


def _roundabout_arc() -> list[dict]:
    cx, cy, radius = 0.0, 16.0, 12.0
    pts = []
    for deg in range(-120, -84, 5):
        ang = math.radians(deg)
        pts.append({"x": cx + radius * math.cos(ang), "y": cy + radius * math.sin(ang), "z": 0.0})
    return pts


def _forward_bend(radius: float = 20.0):
    """Two lane lines on a forward arc. Every sample is ahead of the car."""
    left, right = [], []
    y = 2.0
    while y <= 18.0:
        # Circle center (radius, 0): x = R - sqrt(R^2 - y^2), all y > 0.
        center = radius - math.sqrt(max(0.0, radius * radius - y * y))
        left.append({"x": center - 1.75, "y": y, "z": 0.0})
        right.append({"x": center + 1.75, "y": y, "z": 0.0})
        y += 2.0
    return [left, right]


def check_bend_not_roundabout_and_blinkers() -> None:
    bend = predict_path(lanes_bev=_forward_bend(), lane_conf=0.8, ego_speed_mps=8.0)
    assert bend.prediction != "roundabout", bend.prediction
    assert bend.drivable
    assert min(p["y"] for p in bend.path_ego) > -1.0, bend.path_ego
    # Stay with the lane, not a chord off it or a full steer into the circle.
    cmd = follow_path(bend, ego_speed_mps=8.0, seq=11)
    assert abs(cmd.steer) < 0.85, cmd
    for p in bend.path_ego:
        if 2.0 <= p["y"] <= 18.0:
            center = 20.0 - math.sqrt(max(0.0, 400.0 - p["y"] * p["y"]))
            assert abs(p["x"] - center) < 1.2, (p, center)

    # Near samples are still left of the car even after the line crosses x=0.
    bowed = [{"x": -1.75 + 0.2 * y, "y": float(y)} for y in range(0, 32, 2)]
    assert sum(p["x"] for p in bowed) / len(bowed) > 0.0
    sided = predict_path(lanes_bev=[bowed], lane_conf=0.7, ego_speed_mps=5.0)
    assert sided.path_ego[0]["x"] > -1.2, sided.path_ego[0]
    side_cmd = follow_path(sided, ego_speed_mps=5.0)
    assert side_cmd.steer > -0.5, side_cmd

    lanes = _lane(-1.75, 1.75)
    paced = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=8.0,
        tracks=[{"class": "vehicle", "x": 0.0, "y": 30.0, "yaw": YAW_AHEAD, "speed_mps": 8.0}],
    )
    paced_cmd = follow_path(paced, ego_speed_mps=8.0)
    assert paced.path_length_m > 28.0
    assert paced_cmd.brake < 0.35, paced_cmd

    open_road = predict_path(lanes_bev=lanes, lane_conf=0.9, ego_speed_mps=8.0)
    far_stop = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=8.0,
        signs=[{"cls": "stop_sign", "x": 2.0, "y": 24.0}],
    )
    far_cmd = follow_path(far_stop, ego_speed_mps=8.0)
    assert 20.0 < _max_y(far_stop) < _max_y(open_road) - 4.0
    assert far_cmd.brake < 0.35, far_cmd

    close = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=8.0,
        tracks=[{"class": "vehicle", "x": 0.0, "y": 1.2, "yaw": YAW_AHEAD, "speed_mps": 0.0}],
    )
    assert close.stop_reason == "vehicle"
    close_cmd = follow_path(close, ego_speed_mps=8.0)
    assert close_cmd.throttle < 0.2 and close_cmd.brake > 0.5, close_cmd

    person = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=6.0,
        tracks=[{"class": "pedestrian", "x": 0.0, "y": 10.0, "yaw": YAW_AHEAD, "speed_mps": 0.0}],
    )
    rider = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=6.0,
        tracks=[{"class": "bike", "x": 0.2, "y": 11.0, "yaw": YAW_AHEAD, "speed_mps": 0.0}],
    )
    assert person.stop_reason == "pedestrian", person.stop_reason
    assert rider.stop_reason == "bike", rider.stop_reason

    # partial widens the gate. misses above the hold are dropped. Both fields are read.
    outside = {"cls": "stop_sign", "x": 5.6, "y": 16.0}
    ignored = predict_path(lanes_bev=lanes, lane_conf=0.9, ego_speed_mps=8.0, signs=[outside])
    kept = predict_path(
        lanes_bev=lanes, lane_conf=0.9, ego_speed_mps=8.0, signs=[{**outside, "partial": True}],
    )
    assert ignored.stop_reason == "none"
    assert kept.stop_reason == "stop_sign"
    stale = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=8.0,
        signs=[{"cls": "stop_sign", "x": 2.0, "y": 16.0, "misses": 20}],
    )
    fresh = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=8.0,
        signs=[{"cls": "stop_sign", "x": 2.0, "y": 16.0, "misses": 2, "partial": True}],
    )
    assert stale.stop_reason == "none"
    assert fresh.stop_reason == "stop_sign"

    yielded = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=8.0,
        signs=[{"cls": "yield", "x": 2.0, "y": 28.0}],
    )
    ycmd = follow_path(yielded, ego_speed_mps=8.0)
    assert abs(ycmd.brake - 0.35) > 0.05, ycmd

    straight = predict_path(lanes_bev=lanes, lane_conf=0.9, ego_speed_mps=5.0)
    assert straight.blinker == "off"
    assert straight.planner_dict()["blinker"] == "off"
    # A lane change is a lateral shift that straightens. Signal that side.
    left, right = [], []
    for y in range(0, 40, 2):
        center = 0.0 if y < 8 else (3.5 if y > 22 else 3.5 * (y - 8) / 14.0)
        left.append({"x": center - 1.75, "y": float(y)})
        right.append({"x": center + 1.75, "y": float(y)})
    change = predict_path(lanes_bev=[left, right], lane_conf=0.9, ego_speed_mps=8.0)
    assert change.prediction != "roundabout"
    assert change.blinker == "right", change.blinker

    ring = predict_path(lanes_bev=[_roundabout_arc()], lane_conf=0.7, ego_speed_mps=8.0)
    assert ring.prediction == "roundabout"
    assert ring.blinker == "right"
    # 12 m ring is not a 10 m/s secant. Target speed stays near the curve limit.
    assert ring.target_v < 8.0, ring.target_v
    ring_cmd = follow_path(ring, ego_speed_mps=8.0)
    assert ring_cmd.throttle < 0.05, ring_cmd
    assert ring_cmd.blinker == "right"

    from python.planning.corridor import build_path_ego

    wrapped = build_path_ego(
        lanes_bev=lanes,
        lane_conf=0.9,
        curvature=0.0,
        cipv=None,
        steer_deg=25.0,
        path_width=2.4,
    )
    assert wrapped.path_width == 2.4

    from python.control.actuate import BeamNGPyActuator, note_soft_esc_engaged

    class _Lights:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self.lights: list[dict] = []
            self.sensors = {"electrics": {"gear": "D", "wheelspeed": 5.0, "steering_input": 0.4}}

        def set_shift_mode(self, mode: str) -> None:
            return None

        def control(self, **kw) -> None:
            self.calls.append(kw)

        def set_lights(self, **kw) -> None:
            self.lights.append(kw)

    veh = _Lights()
    act = BeamNGPyActuator(veh)
    try:
        act.note_engaged(True)
        sent = act.apply(follow_path(change, ego_speed_mps=8.0, seq=3))
        assert sent.blinker == "right"
        assert veh.lights and veh.lights[-1]["right_signal"] is True and veh.lights[-1]["left_signal"] is False
        straight_sent = act.apply(follow_path(straight, ego_speed_mps=5.0, seq=4))
        assert straight_sent.blinker == "off"
        assert veh.lights[-1]["right_signal"] is False and veh.lights[-1]["left_signal"] is False
    finally:
        note_soft_esc_engaged(False)


def check_noise_yield_gap_and_short_arc() -> None:
    """Locks the second critic pass. These fail if a 5 cm glitch is a circle,
    a bumper car is ignored, a yield does not move the path, a slow close
    slams the brake, or a 2 m tight arc is driven as a straight."""
    lanes = _lane(-1.75, 1.75)
    clean = predict_path(lanes_bev=_forward_bend(), lane_conf=0.8, ego_speed_mps=8.0)
    noisy_lines = _forward_bend()
    noisy_lines[0][3]["y"] = float(noisy_lines[0][3]["y"]) - 0.05
    noisy = predict_path(lanes_bev=noisy_lines, lane_conf=0.8, ego_speed_mps=8.0)
    assert noisy.prediction != "roundabout", noisy.prediction
    assert noisy.prediction == clean.prediction
    assert max(p["x"] for p in noisy.path_ego) < 25.0

    folded = sorted(_roundabout_arc(), key=lambda p: (p["y"], p["x"]))
    ring = predict_path(lanes_bev=[folded], lane_conf=0.7, ego_speed_mps=0.0)
    assert ring.prediction == "roundabout", ring.prediction
    assert len(ring.path_ego) > 10, len(ring.path_ego)
    ring_cmd = follow_path(ring, ego_speed_mps=0.0)
    assert ring.blinker == "right"
    assert ring_cmd.steer > -0.2, ring_cmd

    bumper = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=8.0,
        tracks=[{"class": "vehicle", "x": 0.0, "y": 0.5, "yaw": YAW_AHEAD, "speed_mps": 0.0}],
    )
    bumper_cmd = follow_path(bumper, ego_speed_mps=8.0)
    assert bumper.stop_reason == "vehicle", bumper.stop_reason
    assert bumper_cmd.throttle < 0.2 and bumper_cmd.brake > 0.5, bumper_cmd

    def _yield_at(y: float):
        plan = predict_path(
            lanes_bev=lanes,
            lane_conf=0.9,
            ego_speed_mps=8.0,
            signs=[{"cls": "yield", "x": 2.0, "y": y}],
        )
        return plan, follow_path(plan, ego_speed_mps=8.0)

    free = predict_path(lanes_bev=lanes, lane_conf=0.9, ego_speed_mps=8.0)
    y20, c20 = _yield_at(20.0)
    y6, c6 = _yield_at(6.0)
    y14, c14 = _yield_at(14.0)
    assert y20.path_length_m < free.path_length_m - 8.0
    assert y20.stop_reason == "yield"
    assert c6.brake > c14.brake + 0.05, (c6.brake, c14.brake)
    assert abs(c6.brake - 0.20) > 0.02 or abs(c14.brake - 0.20) > 0.02

    creep = predict_path(
        lanes_bev=lanes,
        lane_conf=0.9,
        ego_speed_mps=10.0,
        tracks=[{"class": "vehicle", "x": 0.0, "y": 10.0, "yaw": YAW_AHEAD, "speed_mps": 9.0}],
    )
    creep_cmd = follow_path(creep, ego_speed_mps=10.0)
    assert creep.aeb != "brake"
    assert 0.05 < creep_cmd.brake < 0.6, creep_cmd

    radius = 7.0
    glimpse = []
    for y in (2.0, 2.5, 3.0, 3.5, 4.0):
        glimpse.append({"x": radius - math.sqrt(radius * radius - y * y), "y": y, "z": 0.0})
    tight = predict_path(lanes_bev=[glimpse], lane_conf=0.6, ego_speed_mps=5.0)
    tight_cmd = follow_path(tight, ego_speed_mps=5.0)
    assert abs(tight.curvature) > 0.04, tight.curvature
    assert tight.target_v < 12.0, tight.target_v
    assert not (tight_cmd.throttle > 0.4 and tight.target_v > 16.0 and tight.blinker == "off")


def check_missed_sign_and_red_light() -> None:
    import numpy as np

    from python.perception.detect import Detection
    from python.perception.pipeline import ModularPerception

    cv2 = __import__("cv2")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(img, (80, 470), (300, 280), (220, 220, 220), 4)
    cv2.line(img, (560, 470), (340, 280), (220, 220, 220), 4)
    blank = np.zeros_like(img)

    class _Seq:
        name = "seq"

        def __init__(self) -> None:
            self.left = 1

        def detect(self, bgr):
            if self.left:
                self.left = 0
                return [
                    Detection(
                        cls="traffic_light", conf=0.8, xyxy=(0, 0, 1, 1),
                        x=0.4, y=18.0, state="red", partial=True,
                    ),
                    Detection(
                        cls="stop_sign", conf=0.8, xyxy=(0, 0, 1, 1),
                        x=2.0, y=16.0, partial=True,
                    ),
                ]
            return []

    perc = ModularPerception(allow_synthetic=False, detector_id="empty")
    perc.set_detector(_Seq(), [])
    first = perc.tick(img, ego_speed_mps=8.0, steer_deg=40.0)
    assert first.planner["stop_reason"] in ("stop_sign", "red_light"), first.planner
    assert first.path_debug_preview is False
    light = [s for s in first.signs if s["cls"] == "traffic_light"][0]
    assert light["state"] == "red"
    second = perc.tick(blank, ego_speed_mps=8.0, steer_deg=-10.0)
    assert second.planner["prediction"] != "none", second.planner
    assert second.planner["stop_reason"] in ("stop_sign", "red_light"), second.planner
    assert second.path_ego and max(p["y"] for p in second.path_ego) < 30.0
    remembered = [s for s in second.signs if s["cls"] == "traffic_light"][0]
    assert remembered["state"] == "red"
    assert int(remembered["misses"]) >= 1
    assert remembered["partial"] is True

    class _RedOnly:
        name = "red"

        def detect(self, bgr):
            return [
                Detection(
                    cls="traffic_light", conf=0.9, xyxy=(0, 0, 1, 1),
                    x=0.3, y=18.0, state="red",
                ),
            ]

    red_perc = ModularPerception(allow_synthetic=False, detector_id="empty")
    red_perc.set_detector(_RedOnly(), [])
    lit = red_perc.tick(img, ego_speed_mps=8.0)
    assert lit.planner["stop_reason"] == "red_light", lit.planner
    assert [s for s in lit.signs if s["cls"] == "traffic_light"][0]["state"] == "red"


def check_perception_ignores_wheel() -> None:
    import numpy as np

    from python.perception.detect import EmptyDetector
    from python.perception.pipeline import ModularPerception

    cv2 = __import__("cv2")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(img, (80, 470), (300, 280), (210, 210, 210), 3)
    cv2.line(img, (560, 470), (340, 280), (210, 210, 210), 3)
    perc = ModularPerception(allow_synthetic=False, detector_id="empty")
    perc.set_detector(EmptyDetector(), ["yolo_weights"])
    right = perc.tick(img, ego_speed_mps=5.0, steer_deg=28.0)
    perc.tracker = perc.tracker.__class__()
    left = perc.tick(img, ego_speed_mps=5.0, steer_deg=-35.0)
    assert right.path_ego == left.path_ego
    assert right.planner["target_v"] == left.planner["target_v"]
    assert right.planner["aeb"] == left.planner["aeb"]
    assert right.path_debug_preview is False, right.path_debug_preview
    assert left.path_debug_preview is False


def main() -> None:
    check_wheel_is_not_an_input()
    check_full_lane_and_commands()
    check_partial_lane_is_continued()
    check_car_stop_sign_and_light()
    check_unseen_and_partial_car()
    cmd = check_partial_roundabout()
    check_drive_gear_socket_and_override(cmd)
    check_bend_not_roundabout_and_blinkers()
    check_noise_yield_gap_and_short_arc()
    check_missed_sign_and_red_light()
    check_perception_ignores_wheel()
    print("test_path_predictor: OK")


if __name__ == "__main__":
    main()
