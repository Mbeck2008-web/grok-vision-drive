"""Stop treating a parked or nose-up car as a driving segment.

Excursion phases only advance while the car is moving and upright. A stall
longer than 10 s, a nose pointed at the sky, or a damage jump marks the
span. The validator and the trainer use the same rules on rows already
written.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping

STALL_SPEED_MPS = 1.0
STALL_S = 10.0
TIP_Z = 0.3
DAMAGE_SPIKE = 400.0
SETTLE_S = 2.5
# The live tip check. |dir_z| > 0.3 is only 17 deg of pitch, and a steep
# west-coast mountain road tripped it. With the State sensor's up vector a
# car is tipped when its roof points more than 60 deg from vertical, which a
# road grade never does. Without an up vector, pitch must pass 30 deg.
# The row validator (``bad_row_reasons``) still uses TIP_Z.
TIP_UP_Z = 0.5
TIP_DIR_Z_LIVE = 0.5
# Rows this long after a spawn, teleport, or recover carry ``settle``.
SETTLE_FLAG_S = 3.0
# Longitudinal deceleration from consecutive speeds above this is an impact.
IMPACT_DECEL_MPS2 = 15.0
IMPACT_DT_MIN_S = 0.02
IMPACT_DT_MAX_S = 0.5
IMPACT_DEBOUNCE_S = 2.0
# Rows this long before an impact are the run-up the trainer should mask.
PRE_IMPACT_S = 3.0
# Stopped this long after an impact: respawn instead of waiting for STALL_S.
IMPACT_STALL_S = 2.0
# Moving again this long after an impact closes it without a respawn.
IMPACT_WINDOW_S = 8.0
# Live yield guard (``--min-live-yield``): seconds of moving, non-settle rows
# in the rolling window, the speed that counts as moving, the longest tick
# credited to the window, and cuts in a row before the guard pauses.
LIVE_YIELD_WINDOW_S = 60.0
LIVE_YIELD_MOVING_MPS = 1.0
LIVE_YIELD_DT_MAX_S = 0.5
LIVE_YIELD_MAX_CUTS = 3


def is_tipped(dir_z: float | None, up_z: float | None = None, tip_z: float = TIP_Z) -> bool:
    """Up vector when known (roof > 60 deg from vertical), else ``|dir_z| > tip_z``."""
    if up_z is not None:
        return float(up_z) < TIP_UP_Z
    return dir_z is not None and abs(float(dir_z)) > float(tip_z)


class SettleWindow:
    """``active`` for ``SETTLE_FLAG_S`` of recorded time after ``start``.

    The clock is the collector's driving clock, which only advances on
    written rows, so a car that takes 4 s to roll off still gets 3 s of
    flagged rows.
    """

    def __init__(self, span_s: float = SETTLE_FLAG_S) -> None:
        self.span_s = float(span_s)
        self.until: float | None = None

    def start(self, clock_s: float) -> None:
        self.until = float(clock_s) + self.span_s

    def active(self, clock_s: float) -> bool:
        return self.until is not None and float(clock_s) < self.until


class ImpactWatch:
    """Impacts from the speed trace, and when to give up on the car.

    ``update`` returns the deceleration when this sample is a new impact.
    ``should_respawn`` is True once the car has sat under STALL_SPEED_MPS for
    IMPACT_STALL_S after an impact. ``note_row`` keeps the last rows so the
    collector can name the PRE_IMPACT_S run-up.

    Not every crash is seen: only a longitudinal speed drop counts, so a
    sideswipe that keeps the speed is missed, and a tick gap outside
    [IMPACT_DT_MIN_S, IMPACT_DT_MAX_S] (over 0.5 s) is never scored.
    """

    def __init__(self) -> None:
        self.rows: deque[tuple[float, float, int | None]] = deque()
        self.reset()

    def reset(self) -> None:
        self.clock = 0.0
        self.prev_speed: float | None = None
        self.since_impact: float | None = None
        self.stall_s = 0.0
        self.speed_before: float | None = None
        self.rows.clear()

    def update(self, dt_s: float, speed_mps: float | None) -> float | None:
        dt = max(0.0, float(dt_s or 0.0))
        self.clock += dt
        speed = None if speed_mps is None else float(speed_mps)
        hit: float | None = None
        if speed is not None and self.prev_speed is not None and IMPACT_DT_MIN_S <= dt <= IMPACT_DT_MAX_S:
            decel = (self.prev_speed - speed) / dt
            fresh = self.since_impact is None or self.since_impact >= IMPACT_DEBOUNCE_S
            if decel > IMPACT_DECEL_MPS2 and fresh:
                hit = decel
        before = self.prev_speed
        self.prev_speed = speed
        if hit is not None:
            self.since_impact = 0.0
            self.stall_s = 0.0
            self.speed_before = before
        elif self.since_impact is not None:
            self.since_impact += dt
            if speed is None or speed < STALL_SPEED_MPS:
                self.stall_s += dt
            else:
                self.stall_s = 0.0
        return hit

    def note_row(self, t: float, index: int | None) -> None:
        self.rows.append((self.clock, float(t), index))
        while self.rows and self.rows[0][0] < self.clock - PRE_IMPACT_S - 1.0:
            self.rows.popleft()

    def pre_window(self) -> tuple[float | None, int | None]:
        """``(t, i)`` of the first noted row inside PRE_IMPACT_S before now."""
        for clock, t, index in self.rows:
            if clock >= self.clock - PRE_IMPACT_S - 1e-9:
                return t, index
        return None, None

    @property
    def open(self) -> bool:
        return self.since_impact is not None

    def should_respawn(self) -> bool:
        return self.since_impact is not None and self.stall_s >= IMPACT_STALL_S

    def drove_off(self) -> bool:
        return self.since_impact is not None and self.since_impact >= IMPACT_WINDOW_S and self.stall_s <= 0.0

    def close(self) -> None:
        self.since_impact = None
        self.stall_s = 0.0


class YieldGuard:
    """Rolling share of moving rows with a painted lane 10 m ahead.

    Lateral frame: validity only, so frame-insensitive.

    ``update`` takes one written row. Rows under LIVE_YIELD_MOVING_MPS and
    ``settle`` rows are not counted. It returns True when the last
    ``window_s`` seconds of counted rows are full and their time-weighted
    valid share is under ``min_yield``. The caller then cuts the spawn or
    segment (never its frames) and calls ``cut``.

    A ``min_yield`` of 0 is off. After ``max_cuts`` cuts in a row the guard
    pauses (``paused``): it keeps measuring but stops cutting until one full
    window at or above ``min_yield``, so a map where every road reads low
    cannot respawn forever. A healthy full window also resets the count.
    """

    def __init__(
        self,
        min_yield: float = 0.0,
        window_s: float = LIVE_YIELD_WINDOW_S,
        max_cuts: int = LIVE_YIELD_MAX_CUTS,
    ) -> None:
        self.min_yield = max(0.0, float(min_yield or 0.0))
        self.window_s = max(1.0, float(window_s))
        self.max_cuts = max(1, int(max_cuts))
        self.in_a_row = 0
        self.total_cuts = 0
        self.paused = False
        self.rows: deque[tuple[float, bool]] = deque()
        self.reset()

    @property
    def enabled(self) -> bool:
        return self.min_yield > 0.0

    def reset(self) -> None:
        """Empty the window. Call on every spawn, recover and level load."""
        self.rows.clear()
        self.moving_s = 0.0
        self.valid_s = 0.0

    @property
    def rate(self) -> float | None:
        return None if self.moving_s <= 0.0 else self.valid_s / self.moving_s

    @property
    def full(self) -> bool:
        return self.moving_s >= self.window_s - 1e-6

    def update(self, dt_s: float, speed_mps: float | None, valid: bool, settling: bool) -> bool:
        if not self.enabled:
            return False
        if settling or speed_mps is None or float(speed_mps) < LIVE_YIELD_MOVING_MPS:
            return False
        dt = min(max(0.0, float(dt_s or 0.0)), LIVE_YIELD_DT_MAX_S)
        ok = bool(valid)
        self.rows.append((dt, ok))
        self.moving_s += dt
        self.valid_s += dt if ok else 0.0
        while self.rows and self.moving_s - self.rows[0][0] >= self.window_s - 1e-6:
            old_dt, old_ok = self.rows.popleft()
            self.moving_s -= old_dt
            self.valid_s -= old_dt if old_ok else 0.0
        if not self.full:
            return False
        rate = self.rate or 0.0
        if rate >= self.min_yield:
            self.in_a_row = 0
            self.paused = False
            return False
        return not self.paused

    def cut(self) -> None:
        """Count a cut and empty the window; pause after ``max_cuts`` in a row."""
        self.in_a_row += 1
        self.total_cuts += 1
        if self.in_a_row >= self.max_cuts:
            self.paused = True
        self.reset()

    def summary(self) -> dict[str, Any]:
        return {
            "min_live_yield": self.min_yield,
            "window_s": self.window_s,
            "cuts": self.total_cuts,
            "cuts_in_a_row": self.in_a_row,
            "paused": self.paused,
        }


class Liveness:
    """Watch one segment. ``update`` returns an abort reason, or empty."""

    def __init__(self) -> None:
        self.slow_s = 0.0
        self.prev_damage: float | None = None

    def reset(self) -> None:
        self.slow_s = 0.0
        self.prev_damage = None

    def update(
        self,
        dt_s: float,
        speed_mps: float | None,
        dir_z: float | None,
        damage: float | None,
        *,
        ai_should_move: bool,
        up_z: float | None = None,
        tip_z: float = TIP_Z,
    ) -> str:
        if is_tipped(dir_z, up_z, tip_z):
            return "tipped"
        if damage is not None and self.prev_damage is not None:
            if float(damage) - float(self.prev_damage) >= DAMAGE_SPIKE:
                self.prev_damage = float(damage)
                return "damage"
        if damage is not None:
            self.prev_damage = float(damage)
        if not ai_should_move:
            self.slow_s = 0.0
            return ""
        # A missing speed is a stall while the AI should be moving. The
        # validator does not do this: older rows never stored speed.
        speed = None if speed_mps is None else float(speed_mps)
        if speed is None or speed < STALL_SPEED_MPS:
            self.slow_s += max(0.0, float(dt_s))
            if self.slow_s > STALL_S:
                return "stationary"
            return ""
        self.slow_s = 0.0
        return ""


def car_is_driving(
    speed_mps: float | None,
    dir_z: float | None,
    up_z: float | None = None,
    tip_z: float = TIP_Z,
) -> bool:
    """True when excursion phases may advance."""
    if is_tipped(dir_z, up_z, tip_z):
        return False
    if speed_mps is None:
        return False
    return float(speed_mps) >= STALL_SPEED_MPS


def _ego(row: Mapping[str, Any]) -> Mapping[str, Any]:
    ego = row.get("ego")
    return ego if isinstance(ego, dict) else {}


def _dir_z(ego: Mapping[str, Any]) -> float | None:
    direction = ego.get("dir")
    if not isinstance(direction, (list, tuple)) or len(direction) < 3:
        return None
    try:
        return float(direction[2])
    except (TypeError, ValueError):
        return None


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bad_row_reasons(rows: list[Mapping[str, Any]]) -> dict[int, str]:
    """Index -> stationary, tipped, or damage. A stall must last more than 10 s."""
    reasons: dict[int, str] = {}
    slow_ids: list[int] = []
    slow_s = 0.0
    prev_damage: float | None = None
    prev_segment = ""
    for row in rows:
        try:
            index = int(row.get("i"))
        except (TypeError, ValueError):
            continue
        ego = _ego(row)
        dz = _dir_z(ego)
        if dz is not None and abs(dz) > TIP_Z:
            reasons[index] = "tipped"
        damage = _num(ego.get("damage"))
        if damage is not None and prev_damage is not None and damage - prev_damage >= DAMAGE_SPIKE:
            reasons[index] = "damage"
        if damage is not None:
            prev_damage = damage
        speed = _num(ego.get("speed_mps"))
        segment = str(row.get("segment") or "")
        switched = bool(segment and prev_segment and segment != prev_segment)
        if segment:
            prev_segment = segment
        if switched:
            # The map load sits in the first dt of the new segment.
            slow_ids = []
            slow_s = 0.0
        # Missing speed is not a stall. Rows written before this field
        # existed would otherwise be thrown out.
        stalled = speed is not None and speed < STALL_SPEED_MPS
        if stalled:
            slow_ids.append(index)
            step = max(0.0, _num(row.get("dt_s")) or 0.0)
            if switched and step > 2.0:
                step = 0.0
            slow_s += step
            if slow_s > STALL_S:
                for item in slow_ids:
                    reasons.setdefault(item, "stationary")
        else:
            slow_ids = []
            slow_s = 0.0
    return reasons


def _logged_motion_break(row: Mapping[str, Any] | None) -> bool:
    """True when this row is the recover or hard abort that closed a stall.

    The frames are still a bad training span. Validate treats the gap the
    same way it treats a map change.
    """
    if not row:
        return False
    flag = row.get("recover")
    if flag is True or str(flag).lower() in ("1", "true"):
        return True
    excursion = row.get("excursion")
    if not isinstance(excursion, dict):
        return False
    reason = str(excursion.get("reason") or "")
    return reason in ("stationary", "tipped", "damage")


def motion_issue_lines(rows: list[Mapping[str, Any]]) -> list[str]:
    """One line per contiguous bad span, not one line per frame.

    A stationary span that ends on a logged recover or stall abort is a
    segment break. A stall that runs to the end of the file is still an issue.
    """
    reasons = bad_row_reasons(rows)
    if not reasons:
        return []
    order = []
    for row in rows:
        try:
            order.append(int(row.get("i")))
        except (TypeError, ValueError):
            continue
    lines: list[str] = []
    start = None
    prev = None
    reason = ""
    span_s = 0.0
    by_index = {}
    for row in rows:
        try:
            by_index[int(row.get("i"))] = row
        except (TypeError, ValueError):
            continue

    def close(follower: Mapping[str, Any] | None = None) -> None:
        nonlocal start, prev, reason, span_s
        if start is None or prev is None:
            return
        tail = by_index.get(prev)
        if reason == "stationary" and (_logged_motion_break(tail) or _logged_motion_break(follower)):
            start = None
            prev = None
            reason = ""
            span_s = 0.0
            return
        lines.append(f"{reason} i={start}-{prev} {span_s:.1f}s")
        start = None
        prev = None
        reason = ""
        span_s = 0.0

    for index in order:
        why = reasons.get(index)
        if why is None:
            close(by_index.get(index))
            continue
        row = by_index.get(index) or {}
        step = max(0.0, _num(row.get("dt_s")) or 0.0)
        if start is None or why != reason or (prev is not None and index != prev + 1):
            close()
            start = index
            reason = why
            span_s = 0.0
        prev = index
        span_s += step
    close()
    return lines
