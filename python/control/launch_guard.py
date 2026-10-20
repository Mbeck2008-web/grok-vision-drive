"""Launch speed cap and crash/stuck stop.

The 47b430e ground-truth run went from 0 to 12.5 m/s in 4 s and yawed about
67 degrees into a building. During the first few seconds the throttle is
capped and a growing heading error (versus the heading at engage) replaces
the lane steer with the command that reduces that error. A damage jump or
a car that stops moving after a hit disengages and brakes. The ground-truth
row is written after ``engaged`` is cleared, so the engaged rows stay at
damage 0 and the jump (the efc25e3 run, 0 → 467) is on the first
disengaged row. That is the crash, not a stuck trigger with no hit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from python.control.actuate import steer_for_heading_error

LAUNCH_S = 6.0
LAUNCH_THROTTLE = 0.15
LAUNCH_SPEED_MPS = 4.0
# Start replacing lane steer once the car has yawed this far off the engage heading.
LAUNCH_YAW_RAD = 0.35
CRASH_DELTA = 400.0
CRASH_TOTAL = 800.0
STUCK_S = 1.5
STUCK_MOVE_M = 0.4
STUCK_DAMAGE = 200.0
# After the launch window a stale or held lane must not be followed at speed.
# The efc25e3 run went 4.4 → 10.8 m/s on a lane frozen for half a second.
STALE_LANE_S = 0.25
STALE_SPEED_MPS = 6.0
# gt_run7 left the lane with steer near 0. lane_x was +2.27 m for the first
# 1.9 s (pursuit of that point is steer +0.31, and the log matched it), which
# yawed the car 0.33 rad right. Then lane_x fell to ~0, the wheel centered,
# and the heading stayed. A lookahead past this, or a nose this far off the
# engage heading, is treated as leaving. The 4 m/s launch cap only braked
# at 0.2 and speed still climbed 4.3 → 6.1 m/s inside the window.
# gt_run8 sent brake 0.8 with throttle 0. Arcade treats that service brake
# as reverse, so the car backed out of the lane at 11 m/s into a building.
# 1.0 is at or above TECH_HOLD_BRAKE, which the tech actuator turns into
# parkingbrake=1 and service brake 0.
EXIT_LANE_X_M = 1.25
EXIT_YAW_RAD = 0.22
EXIT_BRAKE = 1.0
# Nose-forward speed below this is reverse. The guard disengages and parks.
REVERSE_LONG_MPS = 0.3


def apply_stale_speed_cap(
    cmd: Any,
    *,
    speed_mps: float | None,
    lane_age_s: float | None,
    lane_src: str | None,
) -> bool:
    """Hold the car near 6 m/s when the lane picture is old or the center is held.

    Returns True when the cap changed the command. A fresh ``take`` is left
    alone. The launch cap is tighter and still runs first.
    """
    src = str(lane_src or "")
    held = "hold" in src
    stale = lane_age_s is not None and float(lane_age_s) > STALE_LANE_S
    if not held and not stale:
        return False
    if speed_mps is not None and float(speed_mps) > STALE_SPEED_MPS:
        cmd.throttle = 0.0
        cmd.brake = max(float(cmd.brake), 0.3)
    else:
        cmd.throttle = min(float(cmd.throttle), 0.25)
    return True


def longitudinal_mps(vel: Any, forward: Any) -> float | None:
    """Speed along the nose. Negative is backward. ``forward`` need not be a unit vector."""
    if vel is None or forward is None:
        return None
    try:
        vx, vy = float(vel[0]), float(vel[1])
        fx, fy = float(forward[0]), float(forward[1])
    except (TypeError, ValueError, IndexError):
        return None
    if vx != vx or vy != vy or fx != fx or fy != fy:
        return None
    norm = math.hypot(fx, fy)
    if norm < 1e-6:
        return None
    return (vx * fx + vy * fy) / norm


def moving_backward(long_mps: float | None, limit: float = REVERSE_LONG_MPS) -> bool:
    """True when the car is moving backward faster than ``limit`` m/s."""
    if long_mps is None:
        return False
    try:
        speed = float(long_mps)
    except (TypeError, ValueError):
        return False
    if speed != speed:
        return False
    return speed < -float(limit)


def apply_lane_exit_brake(
    cmd: Any,
    *,
    lane_x: float | None,
    heading_err_rad: float | None,
) -> bool:
    """Cut throttle and brake hard while the lookahead is outside the lane.

    ``lane_x`` is the lookahead the wheel is following. The brake is this
    tick only. When ``|lane_x|`` comes back inside ``EXIT_LANE_X_M`` the
    pedals are left alone, so a good lane releases the stop.

    ``heading_err_rad`` is accepted and ignored. Yaw since engage stayed
    above 0.22 rad for about 90 s on gt_run9 after ``lane_x`` was already
    under 0.4 m, and that held the parking brake for the rest of the run.
    """
    del heading_err_rad
    leave = False
    if lane_x is not None:
        try:
            x = float(lane_x)
        except (TypeError, ValueError):
            x = 0.0
        if x == x and abs(x) > EXIT_LANE_X_M:
            leave = True
    if not leave:
        return False
    cmd.throttle = 0.0
    cmd.brake = max(float(getattr(cmd, "brake", 0.0) or 0.0), EXIT_BRAKE)
    return True


def _unwrap(prev: float, heading: float) -> float:
    return prev + (math.remainder(heading - prev, 2.0 * math.pi))


@dataclass
class LaunchGuard:
    age_s: float = 0.0
    heading0: float | None = None
    heading: float | None = None
    damage0: float | None = None
    last_damage: float | None = None
    anchor: tuple[float, float] | None = None
    anchor_age_s: float = 0.0
    active: bool = False

    def reset(self) -> None:
        self.age_s = 0.0
        self.heading0 = None
        self.heading = None
        self.damage0 = None
        self.last_damage = None
        self.anchor = None
        self.anchor_age_s = 0.0
        self.active = False

    def update(
        self,
        cmd: Any,
        *,
        engaged: bool,
        dt_s: float | None,
        heading_rad: float | None,
        speed_mps: float | None,
        damage: float | None,
        pos_xy: tuple[float, float] | None,
    ) -> str | None:
        """Limit ``cmd`` in place. Return a disengage reason, or None.

        ``crash`` is a damage jump. ``stuck`` is no motion after damage.
        The caller brakes and logs that reason. A launch yaw correction
        stays engaged and only changes steer and throttle.
        """
        if not engaged:
            self.reset()
            return None
        dt = 0.0 if dt_s is None or dt_s <= 0.0 else float(dt_s)
        if not self.active:
            self.active = True
            self.age_s = 0.0
            self.heading0 = heading_rad
            self.heading = heading_rad
            self.damage0 = damage
            self.last_damage = damage
            self.anchor = pos_xy
            self.anchor_age_s = 0.0
        self.age_s += dt
        if heading_rad is not None:
            if self.heading is None:
                self.heading = heading_rad
                self.heading0 = heading_rad
            else:
                self.heading = _unwrap(self.heading, heading_rad)
        if self.age_s < LAUNCH_S:
            cmd.throttle = min(float(cmd.throttle), LAUNCH_THROTTLE)
            if speed_mps is not None and float(speed_mps) > LAUNCH_SPEED_MPS:
                cmd.throttle = 0.0
                cmd.brake = max(float(cmd.brake), 0.2)
            if (
                self.heading is not None
                and self.heading0 is not None
                and abs(self.heading - self.heading0) > LAUNCH_YAW_RAD
            ):
                cmd.steer = steer_for_heading_error(self.heading - self.heading0)
                cmd.throttle = 0.0
        reason = self._hit(damage, pos_xy, dt, float(cmd.throttle))
        if damage is not None:
            self.last_damage = float(damage)
        return reason

    def _hit(
        self,
        damage: float | None,
        pos_xy: tuple[float, float] | None,
        dt: float,
        throttle: float,
    ) -> str | None:
        if damage is not None and self.last_damage is not None:
            if float(damage) - float(self.last_damage) >= CRASH_DELTA:
                return "crash"
        if (
            damage is not None
            and self.damage0 is not None
            and float(damage) - float(self.damage0) >= CRASH_TOTAL
        ):
            return "crash"
        if pos_xy is None:
            return None
        if self.anchor is None:
            self.anchor = pos_xy
            self.anchor_age_s = 0.0
            return None
        moved = math.hypot(pos_xy[0] - self.anchor[0], pos_xy[1] - self.anchor[1])
        if moved >= STUCK_MOVE_M:
            self.anchor = pos_xy
            self.anchor_age_s = 0.0
            return None
        self.anchor_age_s += dt
        damaged = damage is not None and float(damage) >= STUCK_DAMAGE
        if self.anchor_age_s >= STUCK_S and damaged and throttle > 0.05:
            return "stuck"
        return None
