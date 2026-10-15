"""Stop treating a parked or nose-up car as a driving segment.

Excursion phases only advance while the car is moving and upright. A stall
longer than 10 s, a nose pointed at the sky, or a damage jump marks the
span. The validator and the trainer use the same rules on rows already
written.
"""

from __future__ import annotations

from typing import Any, Mapping

STALL_SPEED_MPS = 1.0
STALL_S = 10.0
TIP_Z = 0.3
DAMAGE_SPIKE = 400.0
SETTLE_S = 2.5


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
    ) -> str:
        if dir_z is not None and abs(float(dir_z)) > TIP_Z:
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


def car_is_driving(speed_mps: float | None, dir_z: float | None) -> bool:
    """True when excursion phases may advance."""
    if dir_z is not None and abs(float(dir_z)) > TIP_Z:
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
        # Missing speed is not a stall. Rows written before this field
        # existed would otherwise be thrown out.
        stalled = speed is not None and speed < STALL_SPEED_MPS
        if stalled:
            slow_ids.append(index)
            slow_s += max(0.0, _num(row.get("dt_s")) or 0.0)
            if slow_s > STALL_S:
                for item in slow_ids:
                    reasons.setdefault(item, "stationary")
        else:
            slow_ids = []
            slow_s = 0.0
    return reasons


def motion_issue_lines(rows: list[Mapping[str, Any]]) -> list[str]:
    """One line per contiguous bad span, not one line per frame."""
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

    def close() -> None:
        nonlocal start, prev, reason, span_s
        if start is None or prev is None:
            return
        lines.append(f"{reason} i={start}-{prev} {span_s:.1f}s")
        start = None
        prev = None
        reason = ""
        span_s = 0.0

    for index in order:
        why = reasons.get(index)
        if why is None:
            close()
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
