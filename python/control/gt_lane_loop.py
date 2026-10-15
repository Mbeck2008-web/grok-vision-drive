"""Offline closed loop with a perfect lane, on the gt_ctrl1 bicycle.

The plant yaws at ``ACTUATOR_YAW_GAIN`` of the 30 deg bicycle, one
``STEER_LATENCY_S`` after the command. A parking brake sheds speed at
``PARK_DECEL_MPS2`` (gt_ctrl1 fell from 14 m/s to 0 in about 3.2 s), not
in one tick. Instant zero hid the coast off a curve.

``law="legacy"`` is the steer that drove gt_ctrl1: pursuit of the straight
8 m center, and steer 0 plus the parking brake when the measurement drops.
``law="curve"`` is Stanley, a speed-scaled crosstrack lookahead, and
curvature feedforward. Speed is capped so lateral accel stays near
2.5 m/s^2. A dropped measurement brakes on that tick and keeps the
last curve on the wheel.
"""

from __future__ import annotations

import math
from typing import Any

from python.control.actuate import (
    ACTUATOR_YAW_GAIN,
    STEER_LATENCY_S,
    STEER_LOCK_RAD,
    WHEELBASE_M,
    DriveCommand,
    LaneSteerState,
    curve_speed_mps,
    curve_steer,
    follow_lane_steer,
    lane_sample_hold_reason,
    path_steer_from_ego,
)
from python.control.launch_guard import LAUNCH_S, LaunchGuard, apply_lane_exit_brake
from python.debug.gt_geom import ego_center_x
from python.debug.gt_summary import MIN_DRIVEN_M, MIN_MEAN_SPEED_MPS
from python.train.scene_lane_runtime import straight_lane_path

DT_S = 0.1
HOLD_S = 120.0
LANE_HALF_M = 1.85
LOOKAHEAD_M = 8.0
CRUISE_THROTTLE = 0.45
# gt_ctrl1 speed 14.0 at the off_road tick, 0 about 3.2 s later.
PARK_DECEL_MPS2 = 4.5


def _wrap(angle: float) -> float:
    return math.remainder(float(angle), 2.0 * math.pi)


def replay_steer_heading(
    rows: list[dict[str, Any]],
    *,
    yaw_gain: float = ACTUATOR_YAW_GAIN,
    delay_s: float = STEER_LATENCY_S,
    min_speed_mps: float = 4.0,
    until_s: float = 48.0,
) -> dict[str, float]:
    """Integrate commanded steer and compare with logged vehicle heading.

    No road curvature in this integral. Heading rate is the yaw the wheels
    produced. ``until_s`` stops before gt_ctrl1's polyline-end projection.
    """
    engaged = [row for row in rows if row.get("engaged")]
    if len(engaged) < 3:
        return {"n": 0.0, "rms": 0.0, "max_abs": 0.0}
    t0 = float(engaged[0]["t"])
    heading = float(engaged[0]["heading_rad"])
    sent: list[tuple[float, float]] = []
    clock = 0.0
    errors: list[float] = []
    gain = float(yaw_gain)
    delay = float(delay_s)
    for prev, row in zip(engaged, engaged[1:]):
        dt = float(row["t"]) - float(prev["t"])
        if dt <= 0.02 or dt > 0.45:
            continue
        clock += dt
        sent.append((clock, float(prev.get("steer") or 0.0)))
        steer = 0.0
        target = clock - delay
        for stamp, value in sent:
            if stamp <= target:
                steer = value
        speed = max(0.0, float(prev.get("speed_mps") or 0.0))
        steer = max(-1.0, min(1.0, steer))
        yaw = 0.0
        if speed >= 0.5:
            yaw = gain * (-(speed / WHEELBASE_M) * math.tan(steer * STEER_LOCK_RAD))
        heading = _wrap(heading + yaw * dt)
        if speed > min_speed_mps and (float(prev["t"]) - t0) < until_s:
            errors.append(_wrap(heading - float(row["heading_rad"])))
    if not errors:
        return {"n": 0.0, "rms": 0.0, "max_abs": 0.0}
    rms = math.sqrt(sum(err * err for err in errors) / len(errors))
    return {"n": float(len(errors)), "rms": rms, "max_abs": max(abs(err) for err in errors)}


def run_gt_lane(
    *,
    seconds: float = HOLD_S,
    dt_s: float = DT_S,
    offset_m: float = 0.4,
    heading_err_rad: float = 0.05,
    curvature: float = 0.0,
    half_width_m: float = LANE_HALF_M,
    law: str = "curve",
    gap_at_s: float | None = None,
    gap_s: float = 0.0,
    speed_mps: float = 0.0,
    skip_launch: bool = False,
) -> dict[str, Any]:
    """Drive one road. ``curvature`` is 1/radius, positive for a left bend.

    Offset is metres to the right of the center. Heading error is radians,
    positive when the nose points left of the road. ``law="legacy"`` is the
    gt_ctrl1 controller. A gap is a dropped measurement (polyline end):
    legacy zeros the wheel and parks. ``law="curve"`` holds the last
    curvature and keeps steering.
    """
    if law not in ("legacy", "curve"):
        raise ValueError(law)
    offset = float(offset_m)
    heading_err = float(heading_err_rad)
    speed = max(0.0, float(speed_mps))
    driven = 0.0
    steer_state = LaneSteerState()
    launch = LaunchGuard()
    # The gt_ctrl1 drop happened a minute after engage. The launch window
    # must not replace the wheel during that replay.
    if skip_launch:
        launch.active = True
        launch.age_s = LAUNCH_S + 1.0
        launch.heading0 = heading_err
        launch.heading = heading_err
    speeds: list[float] = []
    offsets: list[float] = []
    steers: list[float] = []
    in_lane_n = 0
    steps = 0
    exit_brakes = 0
    dt = float(dt_s)
    kappa = float(curvature)
    limit = max(1, int(round(float(seconds) / dt)))
    sent: list[tuple[float, float]] = []
    clock = 0.0
    held_offset = offset
    held_heading = heading_err
    held_kappa = kappa
    for step in range(limit):
        now = step * dt
        dropped = gap_at_s is not None and float(gap_at_s) <= now < float(gap_at_s) + float(gap_s)
        lane_x: float | None
        if law == "curve":
            if not dropped:
                held_offset = offset
                held_heading = heading_err
                held_kappa = kappa
            lane_x = ego_center_x(held_offset, held_heading, LOOKAHEAD_M)
            geometric = curve_steer(held_offset, held_heading, held_kappa, speed)
            cmd = DriveCommand(throttle=CRUISE_THROTTLE, brake=0.0, reason="ok")
            cmd.steer = follow_lane_steer(geometric, lane_x, steer_state, dt)
            cap = curve_speed_mps(held_kappa)
            if speed > cap:
                cmd.throttle = 0.0
            # A dropped measurement is the polyline ending under the car.
            # gt_ctrl2 waited 1.5 s at 12 m/s. Brake on the first such tick.
            # The wheel stays on the last curve.
            if dropped:
                cmd.throttle = 0.0
                cmd.brake = max(float(cmd.brake), 1.0)
        elif dropped:
            # gt_ctrl1: off_road cleared the path, stop_command set steer 0
            # and brake 1, and the steer state was reset.
            lane_x = None
            cmd = DriveCommand(steer=0.0, throttle=0.0, brake=1.0, reason="gt_lane_missing")
            steer_state.reset()
        else:
            lane_x = ego_center_x(offset, heading_err, LOOKAHEAD_M)
            path = straight_lane_path(lane_x)
            geometric = path_steer_from_ego(path)
            miss = lane_sample_hold_reason(lane_x, LOOKAHEAD_M)
            cmd = DriveCommand(throttle=CRUISE_THROTTLE, brake=0.0, reason="ok")
            if miss is None:
                cmd.steer = follow_lane_steer(geometric, lane_x, steer_state, dt)
            else:
                cmd.steer = follow_lane_steer(0.0, None, steer_state, dt, miss_reason=miss)
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
        if lane_x is not None and apply_lane_exit_brake(cmd, lane_x=lane_x, heading_err_rad=None):
            exit_brakes += 1
        command = max(-1.0, min(1.0, float(cmd.steer)))
        throttle = max(0.0, min(1.0, float(cmd.throttle)))
        brake = max(0.0, min(1.0, float(cmd.brake)))
        clock += dt
        sent.append((clock, command))
        applied = 0.0
        target = clock - STEER_LATENCY_S
        for stamp, value in sent:
            if stamp <= target:
                applied = value
        if len(sent) > 8:
            sent = [(stamp, value) for stamp, value in sent if stamp >= target - dt]
        # No-throttle brake is the parking brake. gt_ctrl1 shed ~4.5 m/s^2,
        # so the car keeps moving for a few seconds after the command.
        if brake > 0.01 and throttle <= 1e-6:
            speed = max(0.0, speed - PARK_DECEL_MPS2 * dt)
        else:
            accel = 8.0 * throttle - 0.3 * speed - 16.0 * brake
            speed = max(0.0, speed + accel * dt)
        delta = applied * STEER_LOCK_RAD
        yaw_ccw = 0.0
        if speed >= 1e-6:
            yaw_ccw = ACTUATOR_YAW_GAIN * (-(speed / WHEELBASE_M) * math.tan(delta))
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
        steers.append(applied)
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
