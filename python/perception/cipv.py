"""CIPV: closest in-path vehicle inside path_ego / corridor tube."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CipvResult:
    track: dict[str, Any] | None
    ttc_lead: float | None
    aeb: str  # off | warn | brake


def select_cipv(
    tracks: list[dict[str, Any]],
    *,
    path_width: float = 2.0,
    ego_speed_mps: float = 0.0,
    margin_m: float = 0.4,
    min_y: float = 4.0,
) -> CipvResult:
    half = path_width / 2.0 + margin_m
    candidates = []
    for tr in tracks:
        if str(tr.get("class", "")) != "vehicle":
            continue
        x, y = float(tr.get("x", 0)), float(tr.get("y", 0))
        if y < min_y:
            continue
        if abs(x) > half:
            continue
        candidates.append(tr)
    if not candidates:
        return CipvResult(track=None, ttc_lead=None, aeb="off")
    lead = min(candidates, key=lambda t: float(t.get("y", 1e9)))
    y = float(lead.get("y", 0))
    v_lead = float(lead.get("speed_mps", 0))
    # closing speed: ego approaching lead
    closing = max(ego_speed_mps - v_lead, 0.1)
    ttc = y / closing
    aeb = "off"
    if ttc < 1.2:
        aeb = "brake"
    elif ttc < 2.5:
        aeb = "warn"
    return CipvResult(track=lead, ttc_lead=float(ttc), aeb=aeb)
