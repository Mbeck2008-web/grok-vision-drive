"""Path-speed decoupled corridor: lanes + kinematics → path_ego."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class CorridorResult:
    path_ego: list[dict[str, float]]
    path_width: float
    path_conf: float
    curvature: float
    from_planner: bool  # True → path_debug_preview=false


def build_path_ego(
    *,
    lanes_bev: list[list[dict[str, float]]],
    lane_conf: float,
    curvature: float,
    cipv: dict[str, Any] | None,
    steer_deg: float = 0.0,
    length_m: float = 36.0,
    step: float = 1.0,
    path_width: float = 2.0,
) -> CorridorResult:
    # Lane center if we have left+right
    center_xy: list[tuple[float, float]] = []
    if len(lanes_bev) >= 2 and lane_conf >= 0.5:
        # pair by y buckets
        left = sorted(lanes_bev[0], key=lambda p: p["y"])
        right = sorted(lanes_bev[1], key=lambda p: p["y"])
        for i in range(0, min(len(left), len(right), int(length_m))):
            center_xy.append(((left[i]["x"] + right[i]["x"]) * 0.5, (left[i]["y"] + right[i]["y"]) * 0.5))
    if not center_xy:
        # geometric corridor from curvature / steer
        curv = curvature if abs(curvature) > 1e-4 else (steer_deg / 30.0) * 0.03
        x = y = heading = 0.0
        for _ in range(int(length_m / step) + 1):
            center_xy.append((x, y))
            y += step
            heading += curv * step
            x += math.sin(heading) * step * 0.5

    # nudge path away from CIPV if slightly offset
    if cipv is not None:
        lx = float(cipv.get("x", 0))
        if abs(lx) < path_width:
            shift = -0.3 * (1 if lx >= 0 else -1)
            center_xy = [(x + shift, y) for x, y in center_xy]

    path = [{"x": float(x), "y": float(y), "z": 0.0} for x, y in center_xy]
    conf = max(0.35, min(1.0, 0.4 + 0.5 * lane_conf))
    if cipv is not None:
        conf = min(1.0, conf + 0.1)
    # Always a planner path (even geometric) — not steer-preview stub
    return CorridorResult(path_ego=path, path_width=path_width, path_conf=conf, curvature=float(curvature), from_planner=True)
