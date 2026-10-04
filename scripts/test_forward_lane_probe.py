#!/usr/bin/env python3
"""Forward lane stays in the route when a nearer-width rear pair is also visible.

No BeamNG. The rear pair at ±1.75 m is exactly 3.5 m wide. The windshield
pair at ±1.8 m is 3.6 m. Width-only scoring used to drop the windshield
pair, stop the route behind the bumper, and steer from a sample still
behind the car.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import plan_command, steer_sample_point  # noqa: E402
from python.planning.path_predictor import predict_path  # noqa: E402


def _line(x_of_y, y0: float, y1: float, step: float = 2.0) -> list[dict]:
    pts = []
    y = y0
    while y <= y1 + 1e-9:
        pts.append({"x": float(x_of_y(y)), "y": float(y), "z": 0.0})
        y += step
    return pts


def _pair(center, half: float, y0: float, y1: float, step: float = 2.0):
    return [
        _line(lambda y, center=center, half=half: center(y) - half, y0, y1, step),
        _line(lambda y, center=center, half=half: center(y) + half, y0, y1, step),
    ]


def check_forward_lane_beats_rear_pair() -> None:
    """Probe: forward x=±1.8, y=2..30 plus rear x=±1.75, y=-16..-4."""
    lanes = _pair(lambda _y: 0.0, 1.8, 2.0, 30.0)
    lanes += _pair(lambda _y: 0.0, 1.75, -16.0, -4.0)
    plan = predict_path(lanes_bev=lanes, lane_conf=0.9, ego_speed_mps=5.0)
    assert plan.drivable and plan.path_ego
    ys = [float(p["y"]) for p in plan.path_ego]
    # The rear pair used to win and set y0=-16. The route now follows the
    # windshield lane, backfilled to the bumper.
    assert min(ys) >= 0.0, (min(ys), ys[:4])
    assert max(ys) >= 30.0, max(ys)
    assert abs(plan.path_width - 3.6) < 0.05, plan.path_width
    assert abs(plan.path_width - 3.5) > 0.05, plan.path_width
    sample = steer_sample_point(plan.path_ego)
    assert sample is not None and float(sample["y"]) > 0.0, sample
    assert float(sample["y"]) >= 8.0, sample
    cmd = plan_command(
        path_ego=plan.path_ego,
        planner=plan.planner_dict(),
        ego_speed_mps=5.0,
        seq=1,
    )
    assert abs(cmd.steer) < 0.05, cmd
    assert abs(cmd.steer - float(sample["x"]) / 2.5) < 1e-6


def check_curved_rear_steers_ahead() -> None:
    """A rear arc used to steer from y=-12 (steer -0.384). Sample must be ahead.

    Points behind the car stay on the route.
    """
    lanes = _pair(lambda y: -0.015 * (y + 4.0) ** 2, 1.75, -20.0, -4.0)
    plan = predict_path(lanes_bev=lanes, lane_conf=0.9, ego_speed_mps=5.0)
    assert plan.drivable and plan.prediction != "roundabout", plan.prediction
    path = plan.path_ego
    assert any(float(p["y"]) < 0.0 for p in path)
    assert any(float(p["y"]) > 0.0 for p in path)
    behind = path[min(8, len(path) - 1)]
    assert abs(float(behind["y"]) + 12.0) < 1e-6, behind
    assert abs(float(behind["x"]) + 0.96) < 1e-6, behind
    sample = steer_sample_point(path)
    assert sample is not None and float(sample["y"]) > 0.0, sample
    cmd = plan_command(
        path_ego=path,
        planner=plan.planner_dict(),
        ego_speed_mps=5.0,
        seq=2,
    )
    behind_steer = max(-1.0, min(1.0, float(behind["x"]) / 2.5))
    assert abs(behind_steer + 0.384) < 1e-6, behind_steer
    assert abs(cmd.steer - behind_steer) > 0.05, (cmd.steer, behind_steer)
    assert abs(cmd.steer - float(sample["x"]) / 2.5) < 1e-6


def check_beside_and_rear_points_stay() -> None:
    side = predict_path(
        lanes_bev=_pair(lambda _y: -4.3, 1.7, 0.0, 12.0),
        lane_conf=0.8,
        ego_speed_mps=5.0,
    )
    assert side.path_ego and side.path_ego[0]["x"] < -3.0, side.path_ego[0]
    back = predict_path(
        lanes_bev=_pair(lambda _y: 0.0, 1.6, -8.0, -2.0),
        lane_conf=0.8,
        ego_speed_mps=5.0,
    )
    assert back.prediction != "roundabout", back.prediction
    assert min(float(p["y"]) for p in back.path_ego) < 0.0
    sample = steer_sample_point(back.path_ego)
    assert sample is not None and float(sample["y"]) > 0.0, sample


def main() -> None:
    check_forward_lane_beats_rear_pair()
    check_curved_rear_steers_ahead()
    check_beside_and_rear_points_stay()
    print("test_forward_lane_probe: OK")


if __name__ == "__main__":
    main()
