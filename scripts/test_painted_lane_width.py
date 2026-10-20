#!/usr/bin/env python3
"""Minimum painted lane width: the car's own width, not a fixed 2.4 m.

Synthetic pairs, plus the real west_coast_usa paint where a lane ends in a
taper (``tests/fixtures/painted/west_coast_usa_lane_end.json``): the 5 m
lane narrows through 2.34 m and was dropped by the old 2.4 m minimum.

Plain asserts so ``run_offline_suite.py`` runs it; the functions are also
named ``test_*`` so pytest collects them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import python.debug.painted_lanes as PL  # noqa: E402
from python.debug.painted_lanes import DOUBLE_LINE_M, EGO_WIDTH_M, LANE_MIN_M, PaintedLanes  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "painted" / "west_coast_usa_lane_end.json"
NORTH = (0.0, 1.0, 0.0)


def _pair(width: float) -> PaintedLanes:
    return PaintedLanes(
        [{"material": "line_white", "nodes": [[x, y, 0.0, 0.15] for y in range(-50, 101, 10)]} for x in (-width / 2, width / 2)]
    )


def test_minimum_is_the_car_width() -> None:
    assert LANE_MIN_M == EGO_WIDTH_M
    assert DOUBLE_LINE_M < 0.73 < LANE_MIN_M, "a 0.73 m double stays out of the lane range"


def test_synthetic_widths() -> None:
    for width, ok in ((2.34, True), (EGO_WIDTH_M + 0.02, True), (EGO_WIDTH_M - 0.05, False), (1.6, False), (0.73, False)):
        got = _pair(width).lane_at((0.0, 0.0, 0.0), NORTH)
        assert all((g is not None) == ok for g in got), (width, got)


def test_real_lane_end_keeps_the_narrow_lane() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    paint = PaintedLanes(data["decals"])
    poses = data["poses"][:12]  # the car centre is inside the lane at y = 0 for these
    lanes = [paint.lane_at(p["pos"], p["dir"])[0] for p in poses]
    assert all(g is not None for g in lanes), lanes
    widths = [r - l for l, r in lanes]
    assert min(widths) < 2.4 and min(widths) >= LANE_MIN_M, widths
    assert any(abs(w - 2.34) < 0.02 for w in widths), widths
    centers = [0.5 * (l + r) for l, r in lanes]
    assert max(abs(b - a) for a, b in zip(centers, centers[1:])) < 0.5, centers
    # The old fixed minimum drops it.
    old = PL.LANE_MIN_M
    try:
        PL.LANE_MIN_M = 2.4
        dropped = [paint.lane_at(p["pos"], p["dir"])[0] for p in poses]
    finally:
        PL.LANE_MIN_M = old
    assert sum(g is None for g in dropped) >= 3, dropped


def main() -> None:
    test_minimum_is_the_car_width()
    test_synthetic_widths()
    test_real_lane_end_keeps_the_narrow_lane()
    print("test_painted_lane_width: OK")


if __name__ == "__main__":
    main()
