#!/usr/bin/env python3
"""Replay the bcb78c3 frames against ground truth.

The flat row map put y=8 m on the bottom of the sector, so lane_x stayed
positive while the center 8 m ahead swung left (r=−0.40, rms 2.85 m).
Pose projection of the same frames has to track gt_center with r>0.6.
A 1 m innovation must not move the wheel in one tick, and a hold must
report a non-zero frame age.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import steer_sample_point  # noqa: E402
from python.perception.lanes import lanes_from_view  # noqa: E402
from python.perception.pipeline import ModularPerception  # noqa: E402
from python.planning.path_predictor import (  # noqa: E402
    LaneTrack,
    follow_lane_center,
    predict_path,
)

FIXTURE = ROOT / "tests" / "fixtures" / "gt_run5"
FRAMES = (
    "frame_000001.jpg",
    "frame_000020.jpg",
    "frame_000040.jpg",
    "frame_000059.jpg",
)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    assert vx > 1e-9 and vy > 1e-9
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _rows() -> list[dict]:
    path = FIXTURE / "gt.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _clean(path: Path):
    image = cv2.imread(str(path))
    assert image is not None, path
    # The saved jpg burns the HUD into the left of the strip.
    image = image.copy()
    image[:28, :420] = (30, 30, 32)
    return image


def main() -> None:
    rows = _rows()
    # File order is the capture order. Match each jpg to the log row with the
    # same index in the periodic saves: tick 1, 20, 40, 59.
    by_tick = {i + 1: row for i, row in enumerate(rows)}
    wanted = (1, 20, 40)
    got_x: list[float] = []
    got_gt: list[float] = []
    for name, tick in zip(FRAMES, (1, 20, 40, 59)):
        image = _clean(FIXTURE / name)
        fit = lanes_from_view(None, image)
        plan = predict_path(lanes_bev=fit.lanes_bev, lane_conf=fit.conf, ego_speed_mps=8.0)
        sample = steer_sample_point(plan.path_ego or [])
        row = by_tick[tick]
        assert sample is not None, (name, plan.lane_source)
        lane_x = float(sample["x"])
        gt = row.get("gt_center_x")
        assert gt is not None, row["t"]
        if tick in wanted and row.get("in_lane") is not False:
            got_x.append(lane_x)
            got_gt.append(float(gt))
            if abs(float(gt)) > 0.4:
                assert lane_x * float(gt) > 0.0, (name, lane_x, gt, plan.lane_source)
    assert len(got_x) >= 3, (got_x, got_gt)
    corr = _pearson(got_x, got_gt)
    assert corr > 0.6, (corr, list(zip(got_x, got_gt)))

    # The hit: last engaged steer was left while the center was to the right,
    # then the car left the road on the left into the fence.
    engaged = [row for row in rows if row.get("engaged")]
    last = engaged[-1]
    assert float(last["lane_x"]) < 0.0 and float(last["gt_center_x"]) > 2.0, last
    assert float(last["steer"]) < -0.15, last["steer"]
    crashed = [row for row in rows if float(row.get("damage") or 0.0) > 4000.0]
    assert crashed, "damage never reached the fence impact"
    assert crashed[0].get("off_road") is True or float(crashed[0]["road_offset_m"]) < -4.0

    track = LaneTrack()
    follow_lane_center(track, 0.2)
    held, mode = follow_lane_center(track, 1.2)
    assert mode == "hold", mode
    assert abs(held - 0.2) < 0.05, held

    _check_hold_age()
    print(
        f"test_gt_run5_replay: OK r={corr:.3f} "
        f"pairs={list(zip([round(x, 2) for x in got_x], [round(g, 2) for g in got_gt]))}"
    )


def _check_hold_age() -> None:
    image = _clean(FIXTURE / "frame_000020.jpg")
    perc = ModularPerception(allow_synthetic=False, detector_id="empty")
    first = perc.tick(None, stitch_bgr=image, ego_speed_mps=6.0, dt_s=0.2)
    assert perc._lane_track.center_x is not None, first.lane_source
    perc._lane_track.center_x = float(perc._lane_track.center_x) + 2.0
    perc._taken_mono = time.monotonic() - 0.5
    nudged = image.copy()
    # Main sector, on the subsampled grid, so this is a new lane frame.
    nudged[48, 520] = (255, 0, 0)
    second = perc.tick(None, stitch_bgr=nudged, ego_speed_mps=6.0, dt_s=0.2)
    assert second.frame_age_s >= 0.4, (second.frame_age_s, second.lane_source)
    assert "hold" in second.lane_source, second.lane_source


if __name__ == "__main__":
    main()
