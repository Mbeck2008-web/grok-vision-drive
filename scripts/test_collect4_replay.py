#!/usr/bin/env python3
"""Replay the 223b16a 6-minute run.

West coast never opened a gate: every ``collect gate`` line was
``weave blocked junction``. The collect3 replay started weaves because its
road was only the driven polyline. Live Tech has a side road at the graph
nodes, and the walk stopped at the first of those and called it a junction
before it had measured 60 m of the highway.

Italy did open a weave, then a lane change. Speed fell from about 10 m/s to
0 at (164.6, -849.4) without a tip. The recover snapped the original spawn
(237.4, -892.2), and the car later nosed down.
"""
from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect.excursion import (  # noqa: E402
    ACTIVE,
    CRUISE_S,
    ExcursionClock,
    gate_reason,
    targets_for,
)
from python.collect.level_session import recover_points  # noqa: E402
from python.debug.gt_geom import RoadFollow, RoadPoly, heading_from_dir, project_follow, project_road  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "collect4" / "trace.jsonl.gz"


def _load() -> list[dict]:
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _with_stubs(rows: list[dict]) -> list[RoadPoly]:
    """Short pieces plus a crossing road at every joint.

    One polyline, which is what the collect3 replay used, has no joints.
    Live Tech breaks the same drive into graph edges, and a side road meets
    many of those ends.
    """
    pts = [(rows[0]["x"], rows[0]["y"])]
    acc = 0.0
    for prev, row in zip(rows, rows[1:]):
        acc += math.hypot(row["x"] - prev["x"], row["y"] - prev["y"])
        if acc >= 12.0:
            pts.append((row["x"], row["y"]))
            acc = 0.0
    if len(pts) < 3:
        pts.append((rows[-1]["x"], rows[-1]["y"]))
    pieces = [RoadPoly(f"p{i}", [pts[i], pts[i + 1]], 7.0) for i in range(len(pts) - 1)]
    stubs = []
    for i, (x, y) in enumerate(pts[1:-1], start=1):
        px, py = pts[i - 1]
        hx, hy = x - px, y - py
        norm = math.hypot(hx, hy) or 1.0
        rx, ry = hy / norm, -hx / norm
        stubs.append(RoadPoly(f"s{i}", [(x, y), (x + rx * 12.0, y + ry * 12.0)], 4.0))
    return pieces + stubs


def check_trace_matches_the_console() -> None:
    rows = _load()
    west = [row for row in rows if row["seg"] == "west_coast_usa#0"]
    italy = [row for row in rows if row["seg"] == "italy#1"]
    assert west and italy
    assert {row["ph"] for row in west} == {"cruise"}
    change = [row for row in italy if row["ph"] == "lane_change"]
    assert change
    stalled = next(row for row in italy if row["i"] > change[-1]["i"] and float(row["spd"] or 0) < 1.0)
    assert abs(stalled["x"] - 164.6) < 1.0 and abs(stalled["y"] + 849.4) < 2.0
    assert abs(stalled["dz"]) < 0.1
    back = next(row for row in italy if row["i"] > stalled["i"] and abs(row["x"] - 237.4) < 0.5)
    assert abs(back["y"] + 892.2) < 0.5
    print(
        f"collect4 west cruise={len(west)} lane_change_rows={len(change)} "
        f"stall=({stalled['x']:.1f},{stalled['y']:.1f}) recover=({back['x']:.1f},{back['y']:.1f})"
    )


def check_side_roads_block_lane_changes_not_weaves() -> None:
    """The collect3 replay had no side roads. Live west coast does."""
    pieces = [
        RoadPoly(f"h{i}", [(0.0, float(i)), (0.0, float(i + 20))], 7.0)
        for i in range(0, 180, 20)
    ]
    stub = RoadPoly("driveway", [(0.0, 20.0), (12.0, 20.0)], 4.0)
    roads = pieces + [stub]
    hit = project_road(roads, -1.75, 5.0, math.pi / 2.0)
    assert hit is not None
    assert gate_reason(roads, hit, -1.75, 5.0, math.pi / 2.0, "weave", 8.0) == "open"
    assert gate_reason(roads, hit, -1.75, 5.0, math.pi / 2.0, "lane_change", 8.0) == "junction"
    assert targets_for("lane_change", -1, 3.5, lane_id=1, road_width_m=7.0) is None
    assert targets_for("lane_change", 1, 3.5, lane_id=0, road_width_m=7.0) is not None
    stall = (164.6, -849.4, 281.7)
    spawn = (237.4, -892.2, 246.3)
    assert recover_points(stall, spawn)[0] == stall
    print("weave ignores a driveway; a leftward lane change does not cross the centerline")


def check_west_coast_still_weaves_with_stubs() -> None:
    rows = [row for row in _load() if row["seg"] == "west_coast_usa#0"]
    roads = _with_stubs(rows)
    clock = ExcursionClock()
    follow = RoadFollow()
    started: list[str] = []
    previous = "cruise"
    for row in rows[1:]:
        dt = row["dt"]
        dt = 0.1 if dt is None or dt <= 0.0 or dt > 2.0 else float(dt)
        heading = heading_from_dir(row["dx"], row["dy"])
        hit = project_follow(roads, row["x"], row["y"], heading, follow, float(row["spd"] or 0.0))
        kind = clock.peek_kind(2)
        chosen = None
        window = True
        if clock.phase == "cruise" and clock.phase_t + dt >= CRUISE_S:
            gate = gate_reason(roads, hit, row["x"], row["y"], heading, kind, float(row["spd"] or 0.0))
            if gate != "open" and kind != "weave":
                gate = gate_reason(roads, hit, row["x"], row["y"], heading, "weave", float(row["spd"] or 0.0))
                kind = "weave"
            chosen = kind
            window = gate == "open"
        move = clock.update(
            dt,
            offset_m=0.0,
            heading_rad=0.0,
            lane_width_m=3.5,
            off_road=False,
            near_junction=False,
            n_lanes=2,
            window_ok=window,
            start_kind=chosen,
        )
        if previous == "cruise" and move.phase in ACTIVE:
            started.append(move.phase)
        previous = move.phase
    print(f"west coast with side roads starts={started}")
    # The 60 s cruise (was 20 s) leaves room for 2 starts in this recording (was >= 4).
    assert len(started) >= 2, started
    assert set(started) == {"weave"}


def main() -> None:
    check_trace_matches_the_console()
    check_side_roads_block_lane_changes_not_weaves()
    check_west_coast_still_weaves_with_stubs()
    print("collect4 replay ok")


if __name__ == "__main__":
    main()
