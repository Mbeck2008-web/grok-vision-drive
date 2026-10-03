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
from python.planning.path_predictor import follow_path, predict_path  # noqa: E402
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
    assert car.aeb == "brake" and car.target_v == 0.0
    assert _max_y(car) < 18.0 - 1.5
    assert _max_y(car) < _max_y(free) - 5.0
    held = follow_path(car, ego_speed_mps=8.0, seq=1)
    assert held.throttle == 0.0 and held.brake == 1.0, held

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
    assert _max_y(stop) < 24.0
    stopped = follow_path(stop, ego_speed_mps=8.0, seq=2)
    assert stopped.throttle == 0.0 and stopped.brake > 0.0

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
    assert _max_y(red) < 30.0
    assert follow_path(red, ego_speed_mps=6.0).throttle == 0.0

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
            "speed_mps": 10.0,
            "misses": 4,
            "tick_s": 0.1,
        }],
    )
    assert still.occupied and moved.occupied
    assert abs(still.occupied[0]["y"] - 16.0) < 0.2
    # 4 misses * 0.1 s * 10 m/s, coasted along the last heading. Not the stale point.
    assert moved.occupied[0]["y"] > still.occupied[0]["y"] + 3.0
    assert _max_y(still) < 16.0
    assert _max_y(moved) > _max_y(still) + 2.0
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
    check_perception_ignores_wheel()
    print("test_path_predictor: OK")


if __name__ == "__main__":
    main()
