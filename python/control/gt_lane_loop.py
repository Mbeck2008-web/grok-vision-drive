"""Offline closed loop with a perfect 8 m lane center.

The plant is a bicycle. ``lane_x`` is ``ego_center_x`` at 8 m, the same
number the ground-truth log stores. The wheel, the launch cap, and the
exit brake are the live functions. A pass is 120 s, at least 300 m, mean
speed at least 4 m/s, and the car still inside a 3.7 m lane.
"""

from __future__ import annotations

import math
from typing import Any

from python.control.actuate import (
    STEER_LOCK_RAD,
    WHEELBASE_M,
    DriveCommand,
    LaneSteerState,
    follow_lane_steer,
    path_steer_from_ego,
)
from python.control.launch_guard import LaunchGuard, apply_lane_exit_brake
from python.debug.gt_geom import ego_center_x
from python.debug.gt_summary import MIN_DRIVEN_M, MIN_MEAN_SPEED_MPS
from python.train.scene_lane_runtime import straight_lane_path

DT_S = 0.1
HOLD_S = 120.0
LANE_HALF_M = 1.85
LOOKAHEAD_M = 8.0
CRUISE_THROTTLE = 0.45


def _wrap(angle: float) -> float:
    return math.remainder(float(angle), 2.0 * math.pi)


def run_gt_lane(
    *,
    seconds: float = HOLD_S,
    dt_s: float = DT_S,
    offset_m: float = 0.4,
    heading_err_rad: float = 0.05,
    curvature: float = 0.0,
    half_width_m: float = LANE_HALF_M,
) -> dict[str, Any]:
    """Drive one road. ``curvature`` is 1/radius, positive for a left bend.

    Offset is metres to the right of the center. Heading error is radians,
    positive when the nose points left of the road.
    """
    offset = float(offset_m)
    heading_err = float(heading_err_rad)
    speed = 0.0
    driven = 0.0
    steer_state = LaneSteerState()
    launch = LaunchGuard()
    speeds: list[float] = []
    offsets: list[float] = []
    steers: list[float] = []
    in_lane_n = 0
    steps = 0
    exit_brakes = 0
    dt = float(dt_s)
    kappa = float(curvature)
    limit = max(1, int(round(float(seconds) / dt)))
    for _ in range(limit):
        lane_x = ego_center_x(offset, heading_err, LOOKAHEAD_M)
        path = straight_lane_path(lane_x)
        geometric = path_steer_from_ego(path)
        cmd = DriveCommand(throttle=CRUISE_THROTTLE, brake=0.0, reason="ok")
        cmd.steer = follow_lane_steer(geometric, lane_x, steer_state, dt)
        # Heading in the launch guard is absolute. The road heading is the
        # reference, so the guard sees the same error the lane formula uses.
        road_heading = 0.0
        vehicle_heading = road_heading + heading_err
        launch.update(
            cmd,
            engaged=True,
            dt_s=dt,
            heading_rad=vehicle_heading,
            speed_mps=speed,
            damage=0.0,
            pos_xy=(driven, offset),
        )
        if apply_lane_exit_brake(cmd, lane_x=lane_x, heading_err_rad=None):
            exit_brakes += 1
        steer = max(-1.0, min(1.0, float(cmd.steer)))
        throttle = max(0.0, min(1.0, float(cmd.throttle)))
        brake = max(0.0, min(1.0, float(cmd.brake)))
        # No-throttle brake is the parking brake on the wire, not a reverse pedal.
        if brake > 0.01 and throttle <= 1e-6:
            speed = 0.0
            accel = 0.0
        else:
            accel = 8.0 * throttle - 0.3 * speed - 16.0 * brake
            speed = max(0.0, speed + accel * dt)
        delta = steer * STEER_LOCK_RAD
        yaw_ccw = 0.0 if speed < 1e-6 else -(speed / WHEELBASE_M) * math.tan(delta)
        along = speed * math.cos(heading_err)
        offset += (-speed * math.sin(heading_err)) * dt
        heading_err = _wrap(heading_err + (yaw_ccw - kappa * along) * dt)
        driven += max(0.0, along) * dt
        inside = abs(offset) <= float(half_width_m)
        if inside:
            in_lane_n += 1
        steps += 1
        speeds.append(speed)
        offsets.append(offset)
        steers.append(steer)
        if not inside and driven > 20.0:
            break
    mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
    in_lane_pct = 100.0 * in_lane_n / steps if steps else 0.0
    saturated = sum(1 for value in steers if abs(value) >= 0.95)
    ok = (
        driven >= MIN_DRIVEN_M
        and mean_speed >= MIN_MEAN_SPEED_MPS
        and in_lane_pct >= 80.0
        and abs(offsets[-1]) <= float(half_width_m)
    )
    return {
        "driven_m": driven,
        "mean_speed_mps": mean_speed,
        "in_lane_pct": in_lane_pct,
        "max_abs_offset_m": max((abs(v) for v in offsets), default=0.0),
        "final_offset_m": offsets[-1] if offsets else 0.0,
        "saturated_share": saturated / steps if steps else 0.0,
        "exit_brakes": exit_brakes,
        "seconds": steps * dt,
        "ok": ok,
    }
