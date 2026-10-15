#!/usr/bin/env python3
"""Ego-lane center versus the ede174a ground-truth run.

The log's lane_x stays positive while the car crosses the road, so it does
not track the ground-truth center (correlation well below 0.8). A curb and a
wall on the same side must not become that center. The nearer line, shifted
by half a lane toward the car, does track.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import steer_sample_point  # noqa: E402
from python.debug.gt_geom import ego_center_x  # noqa: E402
from python.debug.gt_summary import pearson  # noqa: E402
from python.planning.path_predictor import predict_path  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "gt_run2_engaged.jsonl"


def _line(x: float) -> list[dict[str, float]]:
    return [{"x": float(x), "y": float(y)} for y in range(0, 18, 2)]


def _center(lanes: list) -> float:
    plan = predict_path(lanes_bev=lanes, lane_conf=0.8, ego_speed_mps=4.0)
    sample = steer_sample_point(plan.path_ego or [])
    assert sample is not None, plan.lane_source
    return float(sample["x"])


def check_log_does_not_track() -> None:
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    xs = []
    truth = []
    for row in rows:
        if row.get("lane_x") is None or row.get("lane_offset_m") is None:
            continue
        xs.append(float(row["lane_x"]))
        truth.append(ego_center_x(float(row["lane_offset_m"]), row.get("road_heading_err_rad"), 8.0))
    assert len(xs) >= 30
    corr = pearson(xs, truth)
    assert corr is not None and corr < 0.5, corr
    # At engage the car is left of center, so the center is to the right.
    assert truth[0] > 0.3 and xs[0] > 0.3


def check_same_side_pair_tracks_the_center() -> None:
    """Right curb plus a wall. Several true offsets. r(detected, truth) > 0.8."""
    offsets = [-1.2, -0.6, 0.0, 0.6, 1.2, 1.6]
    detected: list[float] = []
    truth: list[float] = []
    for offset in offsets:
        right = -offset + 1.75
        wall = right + 1.8
        got = _center([_line(right), _line(wall)])
        detected.append(got)
        truth.append(-offset)
        # The center stays on the same side of the car as the true center.
        if abs(offset) > 0.4:
            assert got * (-offset) > 0.0, (offset, got, right, wall)
    corr = pearson(detected, truth)
    assert corr is not None and corr > 0.8, (corr, list(zip(offsets, detected, truth)))


def check_straddling_pair_is_unchanged() -> None:
    # Car 1 m right of center: lines at -2.75 and +0.75. Center is -1.
    got = _center([_line(-2.75), _line(0.75)])
    assert abs(got - (-1.0)) < 0.35, got


def check_lookahead_sign() -> None:
    # Car left of center, aligned with the road: center is to the right.
    assert ego_center_x(-0.67, 0.0, 8.0) > 0.5
    # Car right of center and yawed clockwise: the center ahead is to the left.
    assert ego_center_x(1.6, -0.35, 8.0) < -2.0


def main() -> None:
    check_lookahead_sign()
    check_log_does_not_track()
    check_straddling_pair_is_unchanged()
    check_same_side_pair_tracks_the_center()
    print("test_lane_perception_bias: OK")


if __name__ == "__main__":
    main()
