#!/usr/bin/env python3
"""A weighted 120-minute collect rotates segment-min chunks across every map.

Two segment-end stalls reassign the unspent remainder. They do not burn it
against the 20% lost-time stop, and the run still visits every map.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_scene_lanes import (  # noqa: E402
    continue_after_segment_end,
    segment_at,
    segment_boundary,
    segment_changed,
    segment_fail_action,
    weighted_pieces,
)

MAPS = (
    "west_coast_usa",
    "italy",
    "east_coast_usa",
    "hirochi_raceway",
    "gridmap_v2",
    "small_island",
)
WEIGHTS = {"west_coast_usa": 0.6}
TOTAL_S = 120.0 * 60.0
SEGMENT_S = 12.0 * 60.0
# The 120-minute death charged the unspent rest of a 72-minute west-coast
# block. 39.7 minutes in, that remainder was about 27% of the plan.
DEATH_SKIP_S = 72.0 * 60.0 - 39.7 * 60.0


def check_equal_rotation_is_unchanged() -> None:
    assert segment_at(0.0, ("west_coast_usa", "italy"), SEGMENT_S) == ("west_coast_usa", 0)
    assert segment_at(SEGMENT_S, ("west_coast_usa", "italy"), SEGMENT_S) == ("italy", 1)
    assert segment_boundary(0.0, ("west_coast_usa", "italy"), SEGMENT_S) == SEGMENT_S
    assert segment_boundary(0.0, (), SEGMENT_S) is None
    assert segment_changed("", -1, "west_coast_usa", 0) is True
    assert segment_changed("west_coast_usa", 0, "west_coast_usa", 0) is False
    assert segment_changed("west_coast_usa", 0, "west_coast_usa", 2) is True
    assert segment_changed("", -1, "", 0) is False
    print("equal rotation unchanged")


def check_west_coast_is_twelve_minute_chunks() -> None:
    pieces = weighted_pieces(MAPS, SEGMENT_S, WEIGHTS, TOTAL_S)
    names = [name for name, _dur in pieces]
    assert names == [
        "west_coast_usa",
        "italy",
        "west_coast_usa",
        "east_coast_usa",
        "west_coast_usa",
        "hirochi_raceway",
        "west_coast_usa",
        "gridmap_v2",
        "west_coast_usa",
        "small_island",
        "west_coast_usa",
    ], names
    assert all(dur <= SEGMENT_S + 1e-6 for _name, dur in pieces)
    west = [dur for name, dur in pieces if name == "west_coast_usa"]
    others = [dur for name, dur in pieces if name != "west_coast_usa"]
    assert len(west) == 6 and all(abs(dur - SEGMENT_S) < 1.0 for dur in west)
    assert len(others) == 5 and all(abs(dur - 0.08 * TOTAL_S) < 1.0 for dur in others)
    assert abs(sum(west) - 0.6 * TOTAL_S) < 1.0
    assert abs(sum(dur for _name, dur in pieces) - TOTAL_S) < 1.0

    walked: list[tuple[str, int, float]] = []
    cursor = 0.0
    while cursor < TOTAL_S - 1e-3:
        name, index = segment_at(cursor, MAPS, SEGMENT_S, WEIGHTS, TOTAL_S)
        boundary = segment_boundary(cursor, MAPS, SEGMENT_S, WEIGHTS, TOTAL_S)
        assert boundary is not None and boundary > cursor
        walked.append((name, index, boundary - cursor))
        cursor = boundary
    assert abs(cursor - TOTAL_S) < 1e-3
    assert [name for name, _index, _dur in walked] == names
    assert [index for _name, index, _dur in walked] == list(range(len(names)))
    # Minute 40 is not still west_coast_usa#0.
    name, index = segment_at(40.0 * 60.0, MAPS, SEGMENT_S, WEIGHTS, TOTAL_S)
    assert index != 0 and name != "west_coast_usa", (name, index)
    print("west coast is six 12-minute chunks")


def check_two_segment_ends_cover_every_map() -> None:
    """Stalls at 10 min and 30 min end those chunks. The plan keeps going."""
    stalls = [10.0 * 60.0, 30.0 * 60.0]
    done = 0.0
    cut = 0.0
    lost = 0.0
    failed: dict[str, set[str]] = {}
    visits: list[tuple[str, int, float, float]] = []
    stall_i = 0
    while done < TOTAL_S - 1e-3:
        sched = done + cut
        name, index = segment_at(sched, MAPS, SEGMENT_S, WEIGHTS, TOTAL_S)
        boundary = segment_boundary(sched, MAPS, SEGMENT_S, WEIGHTS, TOTAL_S)
        assert boundary is not None
        piece_end = boundary - cut
        if stall_i < len(stalls) and done + 1e-6 < stalls[stall_i] < piece_end - 1e-6:
            stall_at = stalls[stall_i]
            stall_i += 1
            # One recover (10 s stall + 2.5 s settle) plus the second stall.
            lost += 10.0 + 2.5 + 10.0
            kind, cut, skipped = continue_after_segment_end(
                stall_at,
                cut,
                lost,
                "stationary",
                name,
                failed,
                MAPS,
                SEGMENT_S,
                WEIGHTS,
                TOTAL_S,
            )
            assert kind == "continue", (kind, name, stall_at, lost)
            assert skipped > 1.0
            if visits and visits[-1][1] == index:
                visits[-1] = (name, index, visits[-1][2], stall_at)
            else:
                visits.append((name, index, done, stall_at))
            done = stall_at
            continue
        end = min(TOTAL_S, piece_end)
        if visits and visits[-1][1] == index:
            visits[-1] = (name, index, visits[-1][2], end)
        else:
            visits.append((name, index, done, end))
        done = end

    assert stall_i == 2
    assert abs(done - TOTAL_S) < 1e-3
    covered = {name for name, _index, _t0, _t1 in visits}
    assert covered == set(MAPS), sorted(covered)
    assert lost < 0.20 * TOTAL_S
    assert failed["stationary"] == {"west_coast_usa"}
    for name, _index, t0, t1 in visits:
        assert t1 - t0 <= SEGMENT_S + 1e-3, (name, t1 - t0)
    west_s = sum(t1 - t0 for name, _index, t0, t1 in visits if name == "west_coast_usa")
    assert west_s > 0.55 * TOTAL_S, west_s

    # The death on 903e172: ~32 min skipped inside one west-coast block.
    assert DEATH_SKIP_S / TOTAL_S > 0.20
    fresh: dict[str, set[str]] = {}
    assert (
        segment_fail_action("stationary", "west_coast_usa", fresh, 22.5, DEATH_SKIP_S, TOTAL_S)
        == "continue"
    )
    assert segment_fail_action("stationary", "west_coast_usa", {}, 0.25 * TOTAL_S, 0.0, TOTAL_S) == "exit"
    print(
        "two segment ends still cover "
        + ",".join(name for name, _i, _a, _b in visits)
    )


def main() -> None:
    check_equal_rotation_is_unchanged()
    check_west_coast_is_twelve_minute_chunks()
    check_two_segment_ends_cover_every_map()
    print("collect plan ok")


if __name__ == "__main__":
    main()
