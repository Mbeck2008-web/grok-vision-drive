#!/usr/bin/env python3
"""R1: after a level change the GT logger reads the new level's roads. No BeamNG.

Before the shared road loader, ``GroundTruthLogger`` read the road network
once and kept it for the whole session, so after a level change ``lane_x``
and ``gt_center_x`` were still projected onto the previous level's roads.
This is an intended behaviour change: after the change both must equal what
a logger that only ever saw level B reports for the same poses.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from python.debug.gt_logger import GroundTruthLogger  # noqa: E402
from synthetic_roads import (  # noqa: E402
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


def main() -> None:
    check_level_change_reads_new_roads()
    print("test_gt_level_change: OK")


if __name__ == "__main__":
    main()
