"""Legacy corridor wrapper. The drive route is ``predict_path``.

Wheel angle is not read. A caller that still has ``steer_deg`` does not
bend the path with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from python.planning.path_predictor import predict_path


@dataclass
class CorridorResult:
    path_ego: list[dict[str, float]]
    path_width: float
    path_conf: float
    curvature: float
    from_planner: bool


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
    """Lane centerline, continued when a side is missing. ``steer_deg`` is ignored.

    ``path_width`` is the corridor width reported to the caller.
    """
    del steer_deg
    tracks = [cipv] if isinstance(cipv, dict) else []
    plan = predict_path(
        lanes_bev=lanes_bev,
        lane_conf=lane_conf,
        curvature=curvature,
        tracks=tracks,
        length_m=length_m,
        step_m=step,
    )
    return CorridorResult(
        path_ego=plan.path_ego,
        path_width=float(path_width),
        path_conf=plan.path_conf,
        curvature=plan.curvature,
        from_planner=plan.drivable,
    )
