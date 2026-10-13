"""CIPV: closest vehicle inside the planned path corridor."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class CipvResult:
    track: dict[str, Any] | None
    ttc_lead: float | None
    aeb: str  # off | warn | brake


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def select_cipv(
    tracks: list[dict[str, Any]],
    *,
    path_width: float = 2.0,
    ego_speed_mps: float = 0.0,
    margin_m: float = 0.4,
    min_y: float = 4.0,
    path_ego: list[dict[str, Any]] | None = None,
) -> CipvResult:
    """Closest in-path vehicle.

    With ``path_ego``, the corridor is the same lateral test as the range
    brake. Without it, the straight tube ``|x| < path_width/2 + margin``
    remains for callers that have no route yet.
    """
    half = path_width / 2.0 + margin_m
    candidates = []
    for tr in tracks:
        if str(tr.get("class", "")) != "vehicle":
            continue
        x = _finite(tr.get("x", 0))
        y = _finite(tr.get("y", 0))
        if x is None or y is None or y < min_y:
            continue
        if path_ego:
            from python.planning.path_predictor import vehicle_on_route

            width = _finite(tr.get("width"))
            if not vehicle_on_route(x, y, path_ego, width):
                continue
        elif abs(x) > half:
            continue
        # A new track has no range-rate. speed 0 is not a stopped lead.
        if tr.get("range_rate_valid") is False:
            continue
        candidates.append(tr)
    if not candidates:
        return CipvResult(track=None, ttc_lead=None, aeb="off")
    lead = min(candidates, key=lambda t: float(t.get("y", 1e9)))
    y = _finite(lead.get("y", 0))
    # Unknown ego speed → no invented TTC/AEB.
    ego = _finite(ego_speed_mps)
    if y is None or ego is None or ego <= 0.05:
        return CipvResult(track=lead, ttc_lead=None, aeb="off")
    # speed_mps is road speed (ego + signed range-rate), not |Δrange/Δt|.
    v_lead = _finite(lead.get("speed_mps", 0.0))
    if v_lead is None:
        return CipvResult(track=lead, ttc_lead=None, aeb="off")
    v_lead = max(0.0, v_lead)
    closing = ego - v_lead
    if closing <= 0.05:
        return CipvResult(track=lead, ttc_lead=None, aeb="off")
    ttc = y / closing
    aeb = "off"
    if ttc < 1.2:
        aeb = "brake"
    elif ttc < 2.5:
        aeb = "warn"
    return CipvResult(track=lead, ttc_lead=float(ttc), aeb=aeb)
