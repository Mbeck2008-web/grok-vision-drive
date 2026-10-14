"""Launch speed cap and crash/stuck stop.

The 47b430e ground-truth run went from 0 to 12.5 m/s in 4 s and yawed about
67 degrees into a building. During the first few seconds the throttle is
capped and a growing heading error (versus the heading at engage) replaces
the lane steer with the command that reduces that error. A damage jump or
a car that stops moving after a hit disengages and brakes.
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
