#!/usr/bin/env python3
"""Replay gt_run6 through the lane-center gate, not a fresh fit per frame.

Live (9563579) stayed engaged 120.2 s and did not crash. in_lane was 92/591
engaged rows (15.6%), mean |offset| 4.43 m, max 5.27 m, 3 departed rows.
lane_x vs gt_center r=0.039, bias -6.78 m, rms 8.11 m. The run5 replay that
reported r=0.98 fit three in-lane frames with a new LaneTrack each time.
It never held a center across the run.

What the log actually did: from t=2.41 to t=4.11 lane_x stayed exactly
+2.0352 (one_right/hold), frame_age 2.08 s to 3.94 s, steer +0.23 to +0.28,
while gt_center went from +0.66 m to -1.01 m. The 0.40 m gate only slews
after three samples that sit within 1.25 m of each other. The raw samples
kept missing that window, so the wrong center never let go. The car was
off the pavement by t=11 and stopped in the brush from about t=40 to t=123
(position stuck near (-834, 22), lane_offset -5.20 m, heading error +0.51 rad).
gt_center there is +9.11 m and matches -offset + 8*sin(heading). It is a
real pose, not a bad road id. The windshield in those frames is leaves, so
a line fit cannot track +9 m. det_center vs gt on the live rows is r=-0.06.

This test does two things with one gate:

* Every saved frame, in order, through one LaneTrack. That is the picture
  replay. Its r is printed. It is not required to exceed 0.6: the brush
  plateau is most of the run, and the paint in those frames is not the lane.
* Every engaged gt_center, with a ±1.7 m sawtooth so consecutive samples
  differ by 3.4 m (past the old 1.25 m same-value window), through the same
  function, starting from the +2.04 lock. That series has to track the
  whole run with r>0.6. The old gate stays near +2 and does not.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import steer_sample_point  # noqa: E402
from python.perception.lanes import lanes_from_view  # noqa: E402
from python.planning.path_predictor import (  # noqa: E402
    CENTER_INNOV_M,
    CENTER_SLEW_M,
    LaneTrack,
    follow_lane_center,
    predict_path,
)

FIXTURE = ROOT / "tests" / "fixtures" / "gt_run6"
# Bigger than CENTER_GATE_M, so two consecutive samples are not "the same
# measurement" under the old counter. The mean of the pair is gt_center.
_JITTER_M = 1.7


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    assert vx > 1e-9 and vy > 1e-9, (vx, vy)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _rows() -> list[dict]:
    path = FIXTURE / "gt.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _frames() -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in FIXTURE.glob("*.jpg"):
        if path.name.startswith(("frame_", "depart_")):
            found.append((int(path.stem.split("_")[-1]), path))
    found.sort()
    return found


def _clean(path: Path):
    image = cv2.imread(str(path))
    assert image is not None, path
    image = image.copy()
    image[:28, :420] = (30, 30, 32)
    return image


def _replay_frames(rows: list[dict]) -> tuple[list[tuple[float, float]], float]:
    """One LaneTrack across every saved frame. Returns pairs and the max step."""
    by_tick = {i + 1: row for i, row in enumerate(rows)}
    track = LaneTrack()
    pairs: list[tuple[float, float]] = []
    steps: list[float] = []
    prev: float | None = None
    for tick, path in _frames():
        row = by_tick[tick]
        image = _clean(path)
        fit = lanes_from_view(None, image)
        plan = predict_path(
            lanes_bev=fit.lanes_bev,
            lane_conf=fit.conf,
            ego_speed_mps=4.0,
            lane_track=track,
        )
        sample = steer_sample_point(plan.path_ego or [])
        gt = row.get("gt_center_x")
        if sample is None or gt is None:
            continue
        lane_x = float(sample["x"])
        pairs.append((lane_x, float(gt)))
        if prev is not None:
            steps.append(abs(lane_x - prev))
        prev = lane_x
    assert steps, "the gate never saw two frames"
    return pairs, max(steps)


def _replay_gate_on_gt(rows: list[dict]) -> tuple[float, list[float], list[float]]:
    """Hold +2.04, then feed this run's gt_center with a 3.4 m sawtooth.

    The mean of each pair of samples is the ground-truth center. Consecutive
    samples are 3.4 m apart, so a counter that demands they match within
    1.25 m never reaches three and the center stays at the lock.
    """
    engaged = [
        row for row in rows
        if row.get("engaged") and row.get("gt_center_x") is not None
    ]
    assert len(engaged) > 500, len(engaged)
    track = LaneTrack()
    locked, mode = follow_lane_center(track, 2.04)
    assert mode == "take" and abs(locked - 2.04) < 1e-9
    got: list[float] = []
    truth: list[float] = []
    for index, row in enumerate(engaged):
        gt = float(row["gt_center_x"])
        measured = gt + (_JITTER_M if index % 2 == 0 else -_JITTER_M)
        center, _mode = follow_lane_center(track, measured)
        got.append(center)
        truth.append(gt)
    # Early in the run the truth is left of +2.04. The lock has to start
    # moving before the car is already off the road.
    assert got[12] < 1.0, got[:13]
    assert truth[12] < 0.0, truth[12]
    return _pearson(got, truth), got, truth


def main() -> None:
    rows = _rows()
    pairs, max_step = _replay_frames(rows)
    assert len(pairs) >= 35, len(pairs)
    # A fresh LaneTrack per frame can jump by a full lane. The shared gate
    # cannot: a take blends at most half of 0.40 m, a slew steps 0.35 m and
    # may then absorb a remainder inside 0.40 m.
    limit = CENTER_SLEW_M + CENTER_INNOV_M + 1e-6
    assert max_step <= limit, (max_step, limit)
    image_r = _pearson([a for a, _b in pairs], [b for _a, b in pairs])
    image_bias = sum(a - b for a, b in pairs) / len(pairs)
    image_rms = math.sqrt(sum((a - b) ** 2 for a, b in pairs) / len(pairs))

    gated_r, got, truth = _replay_gate_on_gt(rows)
    assert gated_r > 0.6, (gated_r, got[0], got[-1], truth[0], truth[-1])

    print(
        f"test_gt_run6_replay: OK frames={len(pairs)} "
        f"image_r={image_r:.3f} image_bias={image_bias:.2f} image_rms={image_rms:.2f} "
        f"max_step={max_step:.3f} gated_r={gated_r:.3f}"
    )


if __name__ == "__main__":
    main()
