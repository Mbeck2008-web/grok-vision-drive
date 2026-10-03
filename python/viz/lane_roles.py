"""Through, merge, and exit for boundaries the visualizer already has.

The path planner drives the ego pair and does not fold a joining or leaving
line into that pair. Drawing uses the same split. A boundary whose gap to the
pair closes ahead is a merge. One whose gap opens is an exit. A parallel line
stays a through lane, including an outer lane that does not join or leave.
"""

from __future__ import annotations

from typing import Any

# Metres of gap change across the shared y span before a line is not through.
_GAP_M = 0.8
# Shorter than this overlap, the two ends are the same sample.
_OVERLAP_M = 0.5


def _points(ln: dict[str, Any]) -> list[dict[str, float]]:
    raw = ln.get("points") or []
    out: list[dict[str, float]] = []
    for p in raw:
        if isinstance(p, dict):
            out.append({"x": float(p.get("x") or 0.0), "y": float(p.get("y") or 0.0)})
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append({"x": float(p[0]), "y": float(p[1])})
    return out


def _x_at(pts: list[dict[str, float]], y: float) -> float:
    ordered = sorted(pts, key=lambda p: p["y"])
    if y <= ordered[0]["y"]:
        return ordered[0]["x"]
    if y >= ordered[-1]["y"]:
        return ordered[-1]["x"]
    for a, b in zip(ordered, ordered[1:]):
        lo, hi = (a, b) if a["y"] <= b["y"] else (b, a)
        if lo["y"] - 1e-6 <= y <= hi["y"] + 1e-6:
            dy = hi["y"] - lo["y"]
            if abs(dy) < 1e-9:
                return lo["x"]
            t = (y - lo["y"]) / dy
            return lo["x"] + t * (hi["x"] - lo["x"])
    return ordered[-1]["x"]


def _near_x(pts: list[dict[str, float]]) -> float:
    return min(pts, key=lambda p: (p["y"], abs(p["x"])))["x"]


def _role(pts: list[dict[str, float]], ref: list[dict[str, float]]) -> str:
    """Gap change on the y interval both lines actually cover.

    The reference is not extended past its own ends. A short overlap still
    counts. Closing is a merge. Opening is an exit.
    """
    y0 = max(min(p["y"] for p in pts), min(p["y"] for p in ref))
    y1 = min(max(p["y"] for p in pts), max(p["y"] for p in ref))
    if y1 - y0 < _OVERLAP_M:
        return "through"
    gap_near = abs(_x_at(pts, y0) - _x_at(ref, y0))
    gap_far = abs(_x_at(pts, y1) - _x_at(ref, y1))
    if gap_far < gap_near - _GAP_M:
        return "merge"
    if gap_far > gap_near + _GAP_M:
        return "exit"
    return "through"


def classify_lane_roles(lanes: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Copy each boundary and set ``role`` to through, merge, or exit.

    The anchor on each side is the boundary closest to the car at its near
    end, not the mean of the whole line. Every other boundary, including the
    other side of that pair, is judged on the y span both lines share.
    """
    items: list[dict[str, Any]] = []
    for ln in lanes or []:
        if not isinstance(ln, dict):
            continue
        pts = _points(ln)
        copy = dict(ln)
        copy["points"] = pts
        items.append(copy)
    usable = [ln for ln in items if len(ln["points"]) >= 2]
    left = [ln for ln in usable if _near_x(ln["points"]) < 0.0]
    right = [ln for ln in usable if _near_x(ln["points"]) >= 0.0]
    ego_left = min(left, key=lambda ln: abs(_near_x(ln["points"])), default=None)
    ego_right = min(right, key=lambda ln: abs(_near_x(ln["points"])), default=None)
    refs = [ln for ln in (ego_left, ego_right) if ln is not None]
    out: list[dict[str, Any]] = []
    for ln in items:
        role = "through"
        others = [r for r in refs if r is not ln and len(ln["points"]) >= 2]
        if others:
            ref = min(others, key=lambda r: abs(_near_x(ln["points"]) - _near_x(r["points"])))
            role = _role(ln["points"], ref["points"])
        tagged = dict(ln)
        tagged["role"] = role
        out.append(tagged)
    return out
