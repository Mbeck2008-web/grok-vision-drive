#!/usr/bin/env python3
"""Following the anchored lane's two lines out to 30 m, on synthetic paint.

Each line is walked forward in 1 m steps (``FOLLOW_STEP_M``), so a bend whose
curvature changes is still followed; a fork keeps the branch parallel to the
lane's other line; a line starting inside a lane wide enough for two splits
it. The car is at the origin heading +y.

Plain asserts so ``run_offline_suite.py`` runs it; the functions are also
named ``test_*`` so pytest collects them.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.debug.painted_lanes import LOOKAHEADS_M, PaintedLanes  # noqa: E402

CAR = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
HALF = 1.75


def _line(pts, material: str = "line_white") -> dict:
    return {"material": material, "nodes": [[float(x), float(y), 0.0, 0.15] for x, y in pts]}


def _pieces(fx, y0: float, y1: float, piece_m: float = 6.0, step: float = 1.5) -> list[dict]:
    """A line x = fx(y) as short decal pieces (the map's paint is mostly 2-20 m pieces)."""
    out, y = [], y0
    while y < y1:
        ys = np.arange(y, min(y1, y + piece_m) + 1e-9, step)
        out.append(_line([(fx(v), v) for v in ys]))
        y += piece_m
    return out


def _bend(y: float) -> float:
    """A tight S-bend (city corner pair): 6 m swing, 50 m period, up to 37 deg off the car, radius ~10 m.

    The previous follower jumped 5 -> 10 -> 20 -> 30 m on a slope + slope
    change prediction and lost these lines at 20/30 m.
    """
    return 6.0 * math.sin(2.0 * math.pi * y / 50.0) if y > 0 else 6.0 * 2.0 * math.pi / 50.0 * y


def test_s_bend_is_followed_to_30_m() -> None:
    paint = PaintedLanes(_pieces(lambda y: _bend(y) - HALF, -20, 60) + _pieces(lambda y: _bend(y) + HALF, -20, 60))
    got = paint.lane_at(*CAR)
    for y, g in zip(LOOKAHEADS_M, got):
        assert g is not None, (y, got)
        assert abs(g[0] - (_bend(y) - HALF)) < 0.15 and abs(g[1] - (_bend(y) + HALF)) < 0.15, (y, g, _bend(y))


def test_fork_keeps_the_branch_parallel_to_the_lane() -> None:
    """The right line forks at the car: the lane line runs on, an exit edge leaves at 0.15 m/m."""
    left = _pieces(lambda y: -HALF, -20, 60)
    lane = _pieces(lambda y: HALF, -20, 60)
    edge = [_line([(HALF + 0.15 * max(0.0, y), y) for y in np.arange(-20, 40.1, 1.5)])]
    for decals in (left + lane + edge, left + edge + lane):  # either decal order
        got = PaintedLanes(decals).lane_at(*CAR)
        for y, g in zip(LOOKAHEADS_M, got):
            assert g is not None and abs(g[0] + HALF) < 0.1 and abs(g[1] - HALF) < 0.1, (y, got)


def test_line_starting_inside_a_double_width_lane_splits_it() -> None:
    """A 4.6 m lane with a new line from y = 8 at x = +0.1: the ego lane is the part its centre runs into."""
    paint = PaintedLanes(
        _pieces(lambda y: -2.3, -20, 60) + _pieces(lambda y: 2.3, -20, 60) + _pieces(lambda y: 0.1, 8, 60)
    )
    got = paint.lane_at(*CAR)
    assert got[0] is not None and abs(got[0][0] + 2.3) < 0.05 and abs(got[0][1] - 2.3) < 0.05, got
    for g in got[1:]:
        assert g is not None and abs(g[0] + 2.3) < 0.05 and abs(g[1] - 0.1) < 0.05, got


def test_a_line_crossing_the_lane_does_not_split_it() -> None:
    """A guide line crossing a 4.6 m lane at 0.5 m/m (junction paint) is not a lane line."""
    cross = [_line([(-2.0 + 0.5 * (y - 6.0), y) for y in np.arange(6.0, 14.1, 1.0)])]
    paint = PaintedLanes(_pieces(lambda y: -2.3, -20, 60) + _pieces(lambda y: 2.3, -20, 60) + cross)
    got = paint.lane_at(*CAR)
    for g in got:
        assert g is not None and abs(g[0] + 2.3) < 0.05 and abs(g[1] - 2.3) < 0.05, got


def test_a_line_that_ends_is_lost_after_carry() -> None:
    """The right line ends at y = 12: 20 m and 30 m have no right line (no synthetic edge)."""
    paint = PaintedLanes(_pieces(lambda y: -HALF, -20, 60) + _pieces(lambda y: HALF, -20, 12))
    got = paint.lane_at(*CAR)
    assert got[0] is not None and got[1] is not None and got[2] is None and got[3] is None, got


def main() -> None:
    test_s_bend_is_followed_to_30_m()
    test_fork_keeps_the_branch_parallel_to_the_lane()
    test_line_starting_inside_a_double_width_lane_splits_it()
    test_a_line_crossing_the_lane_does_not_split_it()
    test_a_line_that_ends_is_lost_after_carry()
    print("test_painted_lane_follow: OK")


if __name__ == "__main__":
    main()
