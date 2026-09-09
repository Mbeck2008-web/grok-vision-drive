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
    from_planner: bool  # True → path_debug_preview=false (lane-derived only)


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
    """Build path_ego.

    `from_planner=True` only when left+right lanes produced a center line.
    Geometric/steer fallback keeps from_planner=False → path_debug_preview stays true.
    """
    center_xy: list[tuple[float, float]] = []
    from_lanes = False
    if len(lanes_bev) >= 2 and lane_conf >= 0.5:
        left = sorted(lanes_bev[0], key=lambda p: p["y"])
        right = sorted(lanes_bev[1], key=lambda p: p["y"])
        for i in range(0, min(len(left), len(right), int(length_m))):
            center_xy.append(
                (
                    (left[i]["x"] + right[i]["x"]) * 0.5,
                    (left[i]["y"] + right[i]["y"]) * 0.5,
                )
            )
        from_lanes = len(center_xy) >= 2

    if not from_lanes:
        curv = curvature if abs(curvature) > 1e-4 else (steer_deg / 30.0) * 0.03
        x = y = heading = 0.0
        center_xy = []
        for _ in range(int(length_m / step) + 1):
            center_xy.append((x, y))
            y += step
            heading += curv * step
            x += math.sin(heading) * step * 0.5

    if from_lanes and cipv is not None:
        lx = float(cipv.get("x", 0))
        if abs(lx) < path_width:
            shift = -0.3 * (1 if lx >= 0 else -1)
            center_xy = [(x + shift, y) for x, y in center_xy]

    path = [{"x": float(x), "y": float(y), "z": 0.0} for x, y in center_xy]
    if from_lanes:
        conf = max(0.45, min(1.0, 0.4 + 0.5 * lane_conf))
        if cipv is not None:
            conf = min(1.0, conf + 0.1)
        curv_out = float(curvature)
    else:
        conf = 0.35
        curv_out = float(curvature if abs(curvature) > 1e-4 else (steer_deg / 30.0) * 0.03)

    return CorridorResult(
        path_ego=path,
        path_width=path_width,
        path_conf=conf,
        curvature=curv_out,
        from_planner=from_lanes,
    )
