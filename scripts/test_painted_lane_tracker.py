#!/usr/bin/env python3
"""Ego-lane anchor and hysteresis (``LaneTracker``) on synthetic paint.

Three straight lines at x = -3.5, 0, +3.5 (two 3.5 m lanes), the car driving
+y. The label is the lane that contains the car at y = 0; with a tracker it
stays that lane while the car straddles a line, and switches once the car
centre is ``SWITCH_MARGIN_M`` past the line.

Plain asserts so ``run_offline_suite.py`` runs it; the functions are also
named ``test_*`` so pytest collects them.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.debug.lane_targets import tech_from_painted  # noqa: E402
from python.debug.painted_lanes import SWITCH_MARGIN_M, LaneTracker, PaintedLanes  # noqa: E402

W = 3.5
LINES = (-W, 0.0, W)
NORTH = (0.0, 1.0, 0.0)


def _solid(x: float, y0: float = -50.0, y1: float = 400.0) -> dict:
    return {"material": "line_white", "nodes": [[x, y, 0.0, 0.15] for y in range(int(y0), int(y1) + 1, 10)]}


def _gapped(x: float, gap: tuple[float, float]) -> list[dict]:
    """A line in two decal pieces with a gap between them (a piece ending, a missing dash)."""
    return [_solid(x, -50.0, gap[0]), _solid(x, gap[1], 400.0)]


def _paint() -> PaintedLanes:
    return PaintedLanes([_solid(x) for x in LINES])


def _lane_id(car_x: float, got) -> int | None:
    """World lane of a 5 m label: 0 = the left lane (-3.5..0), 1 = the right lane (0..3.5)."""
    if got is None:
        return None
    return int(round((car_x + got[0]) / W)) + 1


def _drive(paint: PaintedLanes, xs: list[float], tracker: LaneTracker | None, step: float = 2.0) -> list[int | None]:
    out = []
    for k, x in enumerate(xs):
        got = paint.lane_at((x, k * step, 0.0), NORTH, tracker=tracker)[0]
        out.append(_lane_id(x, got))
    return out


def _changes(ids: list[int | None]) -> int:
    return sum(1 for a, b in zip(ids, ids[1:]) if a is not None and b is not None and a != b)


def test_stateless_is_the_containing_pair() -> None:
    paint = _paint()
    for x in (-2.5, -1.0, -0.2):
        assert _lane_id(x, paint.lane_at((x, 0.0, 0.0), NORTH)[0]) == 0
    for x in (0.2, 1.0, 2.5):
        assert _lane_id(x, paint.lane_at((x, 0.0, 0.0), NORTH)[0]) == 1
    # Every lookahead follows the same two lines (straight road: same x).
    got = paint.lane_at((0.4, 0.0, 0.0), NORTH)
    assert all(g is not None and abs(g[0] + 0.4) < 0.05 and abs(g[1] - 3.1) < 0.05 for g in got), got


def test_straddling_a_line_does_not_flicker() -> None:
    """The car weaves +-0.4 m about the centre line: under the margin, the lane it came from stays."""
    paint = _paint()
    xs = [-1.75 + 1.75 * min(1.0, k / 10.0) for k in range(11)]  # drift onto the line
    xs += [0.4 * math.sin(k * 0.7) for k in range(60)]  # straddle it
    stateless = _drive(paint, xs, None)
    assert _changes(stateless) >= 10, "the synthetic straddle must flicker without a tracker"
    tracked = _drive(paint, xs, LaneTracker())
    assert set(tracked) == {0}, tracked
    lab = [tech_from_painted(paint.lane_at((x, k * 2.0, 0.0), NORTH, tracker=t))["lane_gt"][0]
           for t in (LaneTracker(),) for k, x in enumerate(xs)]
    assert max(abs(b - a) for a, b in zip(lab, lab[1:])) < 0.5, "no 5 m centre jump while straddling"


def test_real_lane_change_switches_once_past_the_margin() -> None:
    paint = _paint()
    xs = [-1.75 + 0.1 * k for k in range(36)]  # -1.75 -> +1.75, 0.1 m a tick
    ids = _drive(paint, xs, LaneTracker())
    assert None not in ids, ids
    assert _changes(ids) == 1, ids
    first = ids.index(1)
    assert xs[first] > SWITCH_MARGIN_M and xs[first - 1] <= SWITCH_MARGIN_M + 1e-9, (xs[first - 1], xs[first])
    # ... and back again: one more switch, once the centre is the margin past the line the other way.
    back = list(reversed(xs))
    ids2 = _drive(paint, xs + back, LaneTracker())
    assert _changes(ids2) == 2, ids2


def test_respawn_resets() -> None:
    paint = _paint()
    t = LaneTracker()
    ids = _drive(paint, [-1.75, -1.0, -0.4, 0.0, 0.3], t)
    assert ids[-1] == 0, "still held in the left lane at x = +0.3"
    # Respawned at the same spot (or a new segment): no memory of the old lane.
    t.reset()
    assert _lane_id(0.3, paint.lane_at((0.3, 8.0, 0.0), NORTH, tracker=t)[0]) == 1
    # Without a reset, a car moved to another road does not keep the old pair either: it is not there.
    two = PaintedLanes([_solid(x) for x in LINES] + [_solid(100.0 + x) for x in LINES])
    t2 = LaneTracker()
    _drive(two, [-1.75, -0.4, 0.3], t2)
    got = two.lane_at((100.3, 4.0, 0.0), NORTH, tracker=t2)[0]
    assert _lane_id(0.3, got) == 1, got


def test_gap_in_a_line_at_the_car_keeps_the_lane() -> None:
    """The centre line has a 4 m gap. While the gap is at the car, the kept line is carried (CARRY_M)."""
    gap = (100.0, 104.0)
    paint = PaintedLanes([_solid(-W), _solid(W)] + _gapped(0.0, gap))
    ys = [90.0 + k for k in range(25)]
    in_gap = [gap[0] < y < gap[1] for y in ys]
    ahead_in_gap = [gap[0] < y + 5.0 < gap[1] for y in ys]
    stateless = [_lane_id(-1.75, paint.lane_at((-1.75, y, 0.0), NORTH)[0]) for y in ys]
    assert all(g is None for g, gp in zip(stateless, in_gap) if gp), "no containing pair in the gap"
    t = LaneTracker()
    tracked = [_lane_id(-1.75, paint.lane_at((-1.75, y, 0.0), NORTH, tracker=t)[0]) for y in ys]
    for y, g, gp in zip(ys, tracked, ahead_in_gap):
        assert g == (None if gp else 0), (y, g)


def test_kept_pair_must_stay_adjacent() -> None:
    """A line starting between the kept lines (a lane line tapering in): the kept pair is no lane any more.

    The car drives the left lane (-3.5 .. 0) at x = -1.75. From y = 40 a line
    starts 0.7 m inside the right line and runs left at 0.08 m/m. The ego
    lane's right line is that taper line, not the line beyond it.
    """
    taper = {"material": "line_white", "nodes": [[-0.7 - 0.08 * (y - 40.0), y, 0.0, 0.15] for y in range(40, 80, 2)]}
    paint = PaintedLanes([_solid(x) for x in LINES] + [taper])
    t = LaneTracker()
    for k in range(51):
        y = 30.0 + k * 0.5
        got = paint.lane_at((-1.75, y, 0.0), NORTH, tracker=t, ys=(0.0,))[0]
        if y < 40.0:
            assert got is not None and abs(got[0] + 1.75) < 0.05 and abs(got[1] - 1.75) < 0.05, (y, got)
        elif 41.0 <= y <= 49.0:  # until the lane is narrower than the car
            want = -0.7 - 0.08 * (y - 40.0) + 1.75
            assert got is not None and abs(got[1] - want) < 0.05, (y, got, want)


def test_collector_resets_the_tracker_on_every_respawn() -> None:
    """The collector labels with one tracker and resets it at each respawn, recover and new segment.

    Each of those starts a settle window (``settle.start(done_s)``); the reset
    must be in the same block, so the new spot never inherits the old lane.
    """
    src = (ROOT / "scripts" / "collect_scene_lanes.py").read_text(encoding="utf-8").splitlines()
    assert sum("lane_at(pos, direction, tracker=lane_track)" in ln for ln in src) == 1
    starts = [k for k, ln in enumerate(src) if "settle.start(done_s)" in ln]
    assert len(starts) == 3, starts
    for k in starts:
        block = src[max(0, k - 30) : k]
        assert any("lane_track.reset()" in ln for ln in block), f"no lane_track.reset() before line {k + 1}"


def main() -> None:
    test_stateless_is_the_containing_pair()
    test_straddling_a_line_does_not_flicker()
    test_real_lane_change_switches_once_past_the_margin()
    test_respawn_resets()
    test_gap_in_a_line_at_the_car_keeps_the_lane()
    test_kept_pair_must_stay_adjacent()
    test_collector_resets_the_tracker_on_every_respawn()
    print("test_painted_lane_tracker: OK")


if __name__ == "__main__":
    main()
