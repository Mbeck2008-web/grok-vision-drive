"""Longitudinal: target_v + AEB flags from TTC (state only for M2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpeedPlan:
    target_v: float
    ttc_lead: float | None
    aeb: str


def plan_speed(
    *,
    ego_speed_mps: float,
    curvature: float,
    ttc_lead: float | None,
    aeb: str,
    v_cap: float = 18.0,
) -> SpeedPlan:
    # curvature speed limit
    v_curve = v_cap
    if abs(curvature) > 1e-4:
        v_curve = min(v_cap, max(4.0, math_safe_sqrt(1.5 / abs(curvature))))
    target = min(v_cap, v_curve)
    if aeb == "brake":
        target = 0.0
    elif aeb == "warn" and ttc_lead is not None:
        target = min(target, max(2.0, ego_speed_mps * 0.6))
    elif ttc_lead is not None and ttc_lead < 4.0:
        target = min(target, max(3.0, ego_speed_mps * 0.85))
    return SpeedPlan(target_v=float(target), ttc_lead=ttc_lead, aeb=aeb)


def math_safe_sqrt(x: float) -> float:
    import math

    return math.sqrt(max(0.0, x))
