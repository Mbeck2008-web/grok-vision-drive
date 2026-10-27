#!/usr/bin/env python3
"""R1: after a level change the GT logger reads the new level's roads. No BeamNG.

Before the shared road loader, ``GroundTruthLogger`` read the road network
once and kept it for the whole session, so after a level change ``lane_x``
and ``gt_center_x`` were still projected onto the previous level's roads.
This is an intended behaviour change: after the change both must equal what
a logger that only ever saw level B reports for the same poses.

Three triggers are covered: a pose jump (the spawn on B is far from the
last pose on A), sim time going backwards, and the slow level poll. The last
two use a level B whose spawn is within 5 m of the last pose on A, so the
pose jump alone cannot see the change.
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from python.debug.gt_logger import GroundTruthLogger  # noqa: E402
from synthetic_roads import (  # noqa: E402
    SIM0,
    FakeBng,
    FakeClock,
    drive,
    load_graph,
    network_spec,
    patched_logger_env,
    track,
)

KEYS = ("lane_x", "gt_center_x", "road_id")


def check_level_change_reads_new_roads() -> None:
    graph = load_graph()
    plan = graph["level_change"]
    before = track(graph, plan["before"]["level"], plan["before"]["path"])
    after = track(graph, plan["after"]["level"], plan["after"]["path"])
    b_roads = set(network_spec(graph, plan["after"]["level"]))

    bng = FakeBng(graph, plan["before"]["level"])
    clock = FakeClock()
    with tempfile.TemporaryDirectory() as tmp, patched_logger_env(clock):
        logger = GroundTruthLogger(Path(tmp) / "gt.jsonl", Path(tmp) / "gt")
        drive(logger, bng, before, clock)
        reads_before = bng.reads
        bng.switch_level(plan["after"]["level"])
        changed = drive(logger, bng, after, clock)

    ref_bng = FakeBng(graph, plan["after"]["level"])
    ref_clock = FakeClock()
    with tempfile.TemporaryDirectory() as tmp, patched_logger_env(ref_clock):
        ref = GroundTruthLogger(Path(tmp) / "gt.jsonl", Path(tmp) / "gt")
        want = drive(ref, ref_bng, after, ref_clock)

    assert any(row["lane_x"] is not None for row in want), "reference logger found no lane on level B"
    for i, (got_row, want_row) in enumerate(zip(changed, want)):
        for key in KEYS:
            assert got_row[key] == want_row[key], (
                f"tick {i} after the level change: {key}={got_row[key]!r}, level B roads give {want_row[key]!r}"
            )
        assert got_row["road_id"] in b_roads, f"tick {i}: road {got_row['road_id']!r} is not a level B road"
    assert reads_before == 1, reads_before
    assert bng.reads == 2, f"level change should read the network exactly once more, read {bng.reads - 1}"
    print(f"level change: {len(changed)} ticks on level B roads, reads {reads_before} -> {bng.reads}")


NEAR_LEVEL = "synth_near"
POLL_S = 10.0  # the logger asks the game for the level at most this often


def _graph_with_near_level() -> dict:
    """``synth_a`` plus a test-only level whose diagonal road passes 0.4 m from where W->T ends."""
    graph = load_graph()
    graph["levels"][NEAR_LEVEL] = {
        "nodes": {"NA": [-100.0, -100.0], "NB": [200.0, 200.0]},
        "edges": [{"a": "NA", "b": "NB"}],
    }
    return graph


def _near_tracks(graph: dict) -> tuple[list, list]:
    before = track(graph, "synth_a", ["W", "T"])
    full = track(graph, NEAR_LEVEL, ["NA", "NB"])
    ex, ey, _h = before[-1]
    start = min(range(len(full)), key=lambda i: math.hypot(full[i][0] - ex, full[i][1] - ey))
    after = full[start:]
    gap = math.hypot(after[0][0] - ex, after[0][1] - ey)
    assert gap < 5.0, gap
    return before, after


def _drive_ticks(logger, bng, poses, clock) -> tuple[list, list]:
    """Rows, and the network read count after each tick."""
    rows, reads = [], []
    for pose in poses:
        rows.extend(drive(logger, bng, [pose], clock))
        reads.append(bng.reads)
    return rows, reads


def _reference(graph: dict, poses: list) -> list:
    ref_bng = FakeBng(graph, NEAR_LEVEL)
    ref_clock = FakeClock()
    with tempfile.TemporaryDirectory() as tmp, patched_logger_env(ref_clock):
        ref = GroundTruthLogger(Path(tmp) / "gt.jsonl", Path(tmp) / "gt")
        return drive(ref, ref_bng, poses, ref_clock)


def _assert_b_rows(rows: list, want: list, label: str) -> None:
    assert any(row["lane_x"] is not None for row in want), "reference logger found no lane on level B"
    for i, (got_row, want_row) in enumerate(zip(rows, want)):
        for key in KEYS:
            assert got_row[key] == want_row[key], (
                f"{label}: tick {i} after the reload: {key}={got_row[key]!r}, level B roads give {want_row[key]!r}"
            )


def check_near_spawn_sim_time_back() -> None:
    """Level change, spawn 0.4 m from the last pose, sim clock restarts: reload on the next tick."""
    graph = _graph_with_near_level()
    before, after = _near_tracks(graph)
    after = after[:50]  # 5 s: well inside one poll interval, so only the sim clock can see it
    bng = FakeBng(graph, "synth_a")
    clock = FakeClock()
    with tempfile.TemporaryDirectory() as tmp, patched_logger_env(clock):
        logger = GroundTruthLogger(Path(tmp) / "gt.jsonl", Path(tmp) / "gt")
        drive(logger, bng, before, clock)
        assert bng.reads == 1, bng.reads
        bng.switch_level(NEAR_LEVEL)
        bng.sim_t = SIM0  # a new level restarts the simulation clock
        rows, reads = _drive_ticks(logger, bng, after, clock)
    # Tick 0 on B writes the earlier sim time; the next tick reloads.
    assert reads[0] == 1 and reads[1] == 2, f"sim time went back but reads per tick were {reads[:5]}"
    assert reads[-1] == 2, reads[-1]
    _assert_b_rows(rows[1:], _reference(graph, after[1:]), "sim time back")
    print(f"near spawn, sim time back: reload on tick 1 after the change, reads 1 -> {reads[-1]}")


def check_near_spawn_level_poll() -> None:
    """Level change, spawn 0.4 m away, no sim clock at all: the 10 s level poll finds it."""
    graph = _graph_with_near_level()
    before, after = _near_tracks(graph)
    bng = FakeBng(graph, "synth_a")
    bng.get_gamestate = lambda: {"state": "scenario"}  # no sim time: only the poll is left
    clock = FakeClock()
    with tempfile.TemporaryDirectory() as tmp, patched_logger_env(clock):
        logger = GroundTruthLogger(Path(tmp) / "gt.jsonl", Path(tmp) / "gt")
        long_a = track(graph, "synth_a", ["W", "T", "X"])  # 40 s on one level
        lua0 = bng.lua_calls
        drive(logger, bng, long_a, clock)
        polls = bng.lua_calls - lua0
        seconds = len(long_a) * 0.1
        assert bng.reads == 1, bng.reads
        # Cost: one level query when roads load, then at most one per POLL_S.
        assert 1 + int(seconds // POLL_S) - 1 <= polls <= 1 + int(seconds // POLL_S), (polls, seconds)
        drive(logger, bng, before, clock)
        bng.switch_level(NEAR_LEVEL)
        rows, reads = _drive_ticks(logger, bng, after, clock)
    assert reads[-1] == 2, f"level changed with the spawn 0.4 m away; reads stayed {reads[-1]}"
    k = reads.index(2)
    assert k <= int(POLL_S / 0.1) + 1, f"reload took {k} ticks (> {POLL_S} s)"
    _assert_b_rows(rows[k:], _reference(graph, after[k:]), "level poll")
    print(f"near spawn, no sim clock: {polls} level polls in {seconds:.0f} s on A with 1 read; "
          f"reload {k * 0.1:.1f} s after the change, reads 1 -> {reads[-1]}")


def main() -> None:
    check_level_change_reads_new_roads()
    check_near_spawn_sim_time_back()
    check_near_spawn_level_poll()
    print("test_gt_level_change: OK")


if __name__ == "__main__":
    main()
