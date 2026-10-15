#!/usr/bin/env python3
"""Replay the second PR 82 lane-hold log through the ego-lane center.

The fixture is the engaged seg lines from 196bc2e (west_coast_usa, about
90 s). Logged ``lane_x`` flips sign by more than 2 m from one tick to the
next, and often sits at -16 m to -32 m. Those values are what the old
picker published. ``follow_lane_center`` is the memory the route now uses.
The replay fails if that memory still publishes a sign flip larger than 2 m.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import (  # noqa: E402
    LaneSteerState,
    follow_lane_steer,
    lane_sample_hold_reason,
    path_steer_from_ego,
    steer_sample_point,
)
from python.planning.path_predictor import (  # noqa: E402
    LANE_W_DEFAULT,
    LaneTrack,
    follow_lane_center,
    lane_point_plausible,
    predict_path,
)

FIXTURE = ROOT / "tests" / "fixtures" / "pr82_lane_center" / "segs.jsonl"


def _load() -> list[dict]:
    rows = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line]
    assert len(rows) > 500, len(rows)
    return rows


def _sign_flips(values: list[float | None], jump: float = 2.0) -> int:
    count = 0
    prev: float | None = None
    for value in values:
        if value is None:
            continue
        if prev is not None and prev * value < 0.0 and abs(value - prev) > jump:
            count += 1
        prev = value
    return count


def _path(x: float, y: float) -> list[dict[str, float]]:
    return [{"x": float(x), "y": float(y)}, {"x": float(x), "y": float(y) + 1.0}]


def _replay(rows: list[dict]) -> list[dict]:
    track = LaneTrack()
    steer = LaneSteerState()
    out: list[dict] = []
    for row in rows:
        dt = float(row["dt"])
        raw = row["lane_x"]
        y = 8.0 if row["lane_y"] is None else float(row["lane_y"])
        if raw is None or not lane_point_plausible(float(raw), y):
            tracked = track.center_x
            mode = "outlier" if raw is not None else "none"
        else:
            tracked, mode = follow_lane_center(track, float(raw))
        if tracked is None:
            commanded = follow_lane_steer(0.0, None, steer, dt, miss_reason="lane_hold")
            why = steer.reason
        else:
            miss = lane_sample_hold_reason(float(tracked), y)
            if miss:
                commanded = follow_lane_steer(0.0, None, steer, dt, miss_reason=miss)
            else:
                geom = path_steer_from_ego(_path(float(tracked), y))
                commanded = follow_lane_steer(geom, float(tracked), steer, dt)
            why = steer.reason
        out.append({
            "before_x": raw,
            "after_x": tracked,
            "before_steer": float(row["steer"]),
            "after_steer": float(commanded),
            "mode": mode,
            "why": why,
            "dt": dt,
        })
    return out


def check_same_sign_error_releases_and_one_spike_does_not() -> None:
    """gt_run6 held +2.04 because each new sample missed the last one by >1.25 m.

    Three samples that all sit left of the stored center have to start the
    slew even when they disagree with each other by more than the old gate.
    One sample on the other side must not move the center.
    """
    track = LaneTrack()
    follow_lane_center(track, 2.04)
    for sample in (-1.2, -2.8):
        held, mode = follow_lane_center(track, sample)
        assert mode == "hold", (sample, held, mode)
        assert abs(held - 2.04) < 1e-6, held
    moved, mode = follow_lane_center(track, -0.4)
    assert mode == "slew", (moved, mode)
    assert moved < 2.04 - 0.2, moved
    for sample in (-1.6, -2.2, -0.9, -1.4, -2.0, -1.1, -1.8):
        moved, _mode = follow_lane_center(track, sample)
    assert moved < 0.0, moved

    spiked = LaneTrack()
    follow_lane_center(spiked, 2.04)
    held, mode = follow_lane_center(spiked, -5.0)
    assert mode == "hold" and abs(held - 2.04) < 1e-6, (held, mode)
    # The run5 case: one 1 m step still does not land.
    once = LaneTrack()
    follow_lane_center(once, 0.2)
    held, mode = follow_lane_center(once, 1.2)
    assert mode == "hold" and abs(held - 0.2) < 0.05, (held, mode)


def check_log_flips_and_tracker_does_not(played: list[dict]) -> None:
    raw = [row["before_x"] for row in played]
    tracked = [row["after_x"] for row in played]
    raw_flips = _sign_flips(raw)
    new_flips = _sign_flips(tracked)
    assert raw_flips >= 20, raw_flips
    assert new_flips == 0, new_flips
    far_before = sum(
        1 for row in played
        if row["before_x"] is not None and abs(float(row["before_x"])) > 10.0
    )
    far_after = sum(1 for row in played if row["why"] == "lane_far")
    assert far_before >= 50, far_before
    assert far_after < 10, far_after


def check_one_line_is_half_a_lane_and_does_not_flip() -> None:
    def line(x: float) -> list[dict[str, float]]:
        return [{"x": x, "y": float(y)} for y in range(2, 18, 2)]

    alone = predict_path(lanes_bev=[line(1.2)], lane_conf=0.8, ego_speed_mps=5.0)
    sample = steer_sample_point(alone.path_ego)
    assert sample is not None
    expect = 1.2 - LANE_W_DEFAULT * 0.5
    assert abs(sample["x"] - expect) < 0.05, sample
    assert alone.lane_source == "one_right", alone.lane_source
    # The line itself is not the target.
    assert abs(sample["x"] - 1.2) > 1.0

    track = LaneTrack()
    first = predict_path(
        lanes_bev=[line(0.15)], lane_conf=0.8, ego_speed_mps=5.0, lane_track=track,
    )
    second = predict_path(
        lanes_bev=[line(-0.15)], lane_conf=0.8, ego_speed_mps=5.0, lane_track=track,
    )
    a = steer_sample_point(first.path_ego)
    b = steer_sample_point(second.path_ego)
    assert a is not None and b is not None
    assert abs(b["x"] - a["x"]) < 2.0, (a, b, first.lane_source, second.lane_source)
    assert a["x"] * b["x"] > 0.0, (a, b)


def check_outlier_never_enters_the_fit() -> None:
    lanes = [
        [{"x": -1.7, "y": float(y)} for y in range(2, 18, 2)],
        [
            {"x": 1.7, "y": 2.0},
            {"x": -30.0, "y": 8.0},
            {"x": 1.7, "y": 12.0},
            {"x": 1.7, "y": 16.0},
        ],
    ]
    assert lane_point_plausible(-30.0, 8.0) is False
    plan = predict_path(lanes_bev=lanes, lane_conf=0.8, ego_speed_mps=5.0)
    sample = steer_sample_point(plan.path_ego)
    assert sample is not None and abs(sample["x"]) < 1.0, sample
    assert plan.lane_source == "pair", plan.lane_source
    assert all(abs(p["x"]) < 8.0 for p in plan.path_ego)


def _print_table(played: list[dict]) -> None:
    want: list[int] = []
    prev = None
    line_flip = None
    for i, row in enumerate(played):
        x = row["before_x"]
        if x is None or prev is None:
            if x is not None:
                prev = x
            continue
        jumped = prev * x < 0.0 and abs(x - prev) > 2.0
        # The live pair that sat on the two paint lines, about -1.1 and +1.5.
        if (
            line_flip is None
            and jumped
            and 0.8 < abs(prev) < 2.2
            and 0.8 < abs(x) < 2.2
        ):
            line_flip = i
        if jumped and len(want) < 6:
            want.extend(range(max(0, i - 1), min(len(played), i + 2)))
        prev = x
    if line_flip is not None:
        want.extend(range(max(0, line_flip - 1), min(len(played), line_flip + 4)))
    for i, row in enumerate(played):
        x = row["before_x"]
        if x is not None and float(x) < -15.0:
            want.extend(range(max(0, i - 1), min(len(played), i + 2)))
            break
    seen = sorted(set(want))
    print("replay  lane_x_before  lane_x_after  steer_before  steer_after  mode")
    for i in seen:
        row = played[i]
        bx = "na" if row["before_x"] is None else f"{float(row['before_x']):+.4f}"
        ax = "na" if row["after_x"] is None else f"{float(row['after_x']):+.4f}"
        print(
            f"  {i:4d}  {bx:>13}  {ax:>12}  {row['before_steer']:+.4f}  "
            f"{row['after_steer']:+.4f}  {row['mode']}"
        )
    raw = _sign_flips([row["before_x"] for row in played])
    new = _sign_flips([row["after_x"] for row in played])
    far = sum(1 for row in played if row["why"] == "lane_far")
    print(f"sign_flips_gt2m before={raw} after={new} lane_far_after={far}")


def main() -> None:
    check_one_line_is_half_a_lane_and_does_not_flip()
    check_outlier_never_enters_the_fit()
    check_same_sign_error_releases_and_one_spike_does_not()
    played = _replay(_load())
    check_log_flips_and_tracker_does_not(played)
    _print_table(played)
    print("test_lane_center_replay: OK")


if __name__ == "__main__":
    main()
