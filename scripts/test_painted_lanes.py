#!/usr/bin/env python3
"""Painted-lane labels on synthetic paint: straight, curve, missing side, z, stop line, per-map relabel.

Plain ``check``-style asserts so ``run_offline_suite.py`` runs it; the
functions are also named ``test_*`` so pytest collects them.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.debug.lane_targets import LANE_GT_DIM, lane_center_is_valid, tech_from_painted  # noqa: E402
from python.debug.painted_lanes import LOOKAHEADS_M, PaintedLanes  # noqa: E402

HALF = 1.75


def _line(points: list[tuple[float, float, float]], material: str = "line_white") -> dict:
    return {"material": material, "nodes": [[x, y, z, 0.2] for x, y, z in points]}


def _straight(x: float, z: float = 0.0, y0: float = -40.0, y1: float = 80.0) -> dict:
    return _line([(x, y, z) for y in np.linspace(y0, y1, 13)])


def _arc(radius: float, center_x: float = -60.0, z: float = 0.0) -> dict:
    """Left-hand bend: circle about (center_x, 0), starting at the car heading +y."""
    pts = []
    for ang in np.linspace(-0.3, 0.9, 25):
        pts.append((center_x + radius * math.cos(ang), radius * math.sin(ang), z))
    return _line(pts)


CAR = ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])


def _close(got, want, tol=0.05) -> bool:
    return got is not None and abs(got[0] - want[0]) <= tol and abs(got[1] - want[1]) <= tol


def test_straight_road() -> None:
    lanes = PaintedLanes([_straight(-HALF), _straight(HALF)])
    got = lanes.lane_at(*CAR)
    assert all(_close(g, (-HALF, HALF)) for g in got), got
    tech = tech_from_painted(got)
    assert lane_center_is_valid(tech["lane_gt_valid"])
    assert abs(tech["lane_gt"][0]) < 0.05 and abs(tech["lane_gt"][-2]) < 0.01


def test_curve_lookahead_follows_the_bend() -> None:
    lanes = PaintedLanes([_arc(60.0 - HALF), _arc(60.0 + HALF)])
    got = lanes.lane_at(*CAR)
    for y, g in zip(LOOKAHEADS_M, got):
        assert g is not None, (y, got)
        want_left = -(60.0 - math.sqrt((60.0 - HALF) ** 2 - y * y))
        want_right = -(60.0 - math.sqrt((60.0 + HALF) ** 2 - y * y))
        assert _close(g, (want_left, want_right), tol=0.15), (y, g, want_left, want_right)
    # At 30 m the lane center is ~8 m left; a straight-line label would say ~0.
    assert 0.5 * (got[3][0] + got[3][1]) < -7.0


def test_one_side_missing_is_nan_and_masked() -> None:
    lanes = PaintedLanes([_straight(-HALF)])
    got = lanes.lane_at(*CAR)
    assert got == [None] * len(LOOKAHEADS_M), got
    tech = tech_from_painted(got)
    vals = np.asarray(tech["lane_gt"], dtype=np.float64)
    mask = np.asarray(tech["lane_gt_valid"], dtype=np.float64)
    assert len(vals) == LANE_GT_DIM
    assert np.all(np.isnan(vals[:12])) and np.all(mask[:13] == 0.0), (vals, mask)
    assert not lane_center_is_valid(mask)
    assert vals[-1] == 0.0  # confidence target says "no lane"


def test_paint_on_another_deck_is_ignored() -> None:
    # A road 8 m overhead with lines that would split the ego lane.
    decals = [_straight(-HALF), _straight(HALF), _straight(-0.5, z=8.0), _straight(3.0, z=8.0)]
    lanes = PaintedLanes(decals)
    assert all(_close(g, (-HALF, HALF)) for g in lanes.lane_at(*CAR))
    unfiltered = lanes.lane_at(*CAR, z_max_m=None)
    assert not all(_close(g, (-HALF, HALF)) for g in unfiltered), unfiltered


def test_stop_line_across_the_lane_is_ignored() -> None:
    # A bar ~50 degrees off the heading crossing the lane at y=10.
    bar = _line([(-1.2, 9.0, 0.0), (1.2, 11.0, 0.0)])
    lanes = PaintedLanes([_straight(-HALF), _straight(HALF), bar])
    got = lanes.lane_at(*CAR)
    assert _close(got[1], (-HALF, HALF)), got
    unfiltered = lanes.lane_at(*CAR, heading_max_deg=None)
    assert unfiltered[1] is None or not _close(unfiltered[1], (-HALF, HALF)), unfiltered


def test_relabel_uses_each_rows_own_map() -> None:
    from scripts.relabel_painted import main as relabel_main

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        zpath = root / "synth.zip"
        with zipfile.ZipFile(zpath, "w") as z:
            body = "\n".join(
                json.dumps({"class": "DecalRoad", **d}) for d in (_straight(-HALF), _straight(HALF))
            )
            z.writestr("levels/synth/main/MissionGroup/roads/items.level.json", body)
        rows = []
        for i in range(6):
            name = "synth" if i < 4 else "other_map"
            rows.append({"i": i, "t": float(i), "map": name, "ego": {"pos": [0.0, 0.0, 0.0], "dir": [0.0, 1.0, 0.0]}})
        rows.append({"i": 6, "t": 6.0, "segment": "synth#0", "ego": {"pos": [0.0, 0.0, 0.0], "dir": [0.0, 1.0, 0.0]}})
        rows.append({"i": 7, "t": 7.0, "ego": {"pos": [0.0, 0.0, 0.0], "dir": [0.0, 1.0, 0.0]}})
        (root / "state.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        assert relabel_main([str(root), "--zip", f"synth={zpath}", "--out", "gt_test.jsonl"]) == 0
        out = [json.loads(x) for x in (root / "gt_test.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [r["i"] for r in out] == [0, 1, 2, 3, 6], out
        assert all(r["tech"]["gt_map"] == "synth" and lane_center_is_valid(r["tech"]["lane_gt_valid"]) for r in out)
        # Never overwrite an existing label file.
        assert relabel_main([str(root), "--zip", f"synth={zpath}", "--out", "gt_test.jsonl"]) == 2


def main() -> None:
    test_straight_road()
    test_curve_lookahead_follows_the_bend()
    test_one_side_missing_is_nan_and_masked()
    test_paint_on_another_deck_is_ignored()
    test_stop_line_across_the_lane_is_ignored()
    test_relabel_uses_each_rows_own_map()
    print("test_painted_lanes: OK")


if __name__ == "__main__":
    main()
