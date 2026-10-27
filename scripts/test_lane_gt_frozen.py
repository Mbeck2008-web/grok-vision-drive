#!/usr/bin/env python3
"""Steady-state lane_gt rows are unchanged by the shared road loader. No BeamNG.

``tests/fixtures/lane_gt_pre_pr1.jsonl.gz`` was written by this script with
``--write`` on main before the loader existed (commit dd6598d): the real
``GroundTruthLogger`` driven over the synthetic roads in
``tests/fixtures/junctions_synthetic.json`` with a scripted pose track, a
fake sim clock and a fixed wall clock. This check drives the same track
through the current code and compares the rows byte for byte (floats
rounded to 1e-9, see ``synthetic_roads.canonical_bytes``).

    py -3 scripts/test_lane_gt_frozen.py           # compare
    py -3 scripts/test_lane_gt_frozen.py --write   # regenerate (only on purpose)
"""
from __future__ import annotations

import gzip
import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from python.debug.gt_logger import GroundTruthLogger  # noqa: E402
from synthetic_roads import (  # noqa: E402
    FROZEN_PATH,
    FakeBng,
    FakeClock,
    canonical_bytes,
    drive,
    gzip_bytes,
    load_graph,
    patched_logger_env,
    track,
)


def frozen_rows_bytes() -> bytes:
    graph = load_graph()
    spec = graph["frozen_track"]
    poses = track(graph, spec["level"], spec["path"], spec["speed_mps"], spec["dt_s"])
    bng = FakeBng(graph, spec["level"])
    clock = FakeClock()
    with tempfile.TemporaryDirectory() as tmp, patched_logger_env(clock):
        logger = GroundTruthLogger(Path(tmp) / "gt.jsonl", Path(tmp) / "gt")
        rows = drive(logger, bng, poses, clock, speed_mps=spec["speed_mps"], dt_s=spec["dt_s"])
    assert bng.reads == 1, f"steady state read the road network {bng.reads} times"
    return canonical_bytes(rows)


def check_frozen() -> None:
    got = frozen_rows_bytes()
    want = gzip.decompress(FROZEN_PATH.read_bytes())
    if got != want:
        got_lines = got.decode("utf-8").splitlines()
        want_lines = want.decode("utf-8").splitlines()
        for i, (a, b) in enumerate(zip(got_lines, want_lines)):
            if a != b:
                raise AssertionError(f"lane_gt row {i} differs:\n got  {a}\n want {b}")
        raise AssertionError(f"lane_gt row count {len(got_lines)} != {len(want_lines)}")
    rows = want.count(b"\n")
    lanes = want.count(b'"lane_x":null')
    assert rows > 500, rows
    assert lanes < rows // 10, f"{lanes} of {rows} rows have no lane_x; the track is off the roads"
    print(f"lane_gt frozen: {rows} rows identical, sha256 {hashlib.sha256(want).hexdigest()}")


def write() -> None:
    data = frozen_rows_bytes()
    FROZEN_PATH.write_bytes(gzip_bytes(data))
    print(f"wrote {FROZEN_PATH.relative_to(ROOT)}")
    rows = data.count(b"\n")
    print(f"  rows {rows}")
    print(f"  sha256 jsonl {hashlib.sha256(data).hexdigest()}")
    print(f"  sha256 gz    {hashlib.sha256(FROZEN_PATH.read_bytes()).hexdigest()}")


def main() -> None:
    if "--write" in sys.argv[1:]:
        write()
        return
    check_frozen()
    print("test_lane_gt_frozen: OK")


if __name__ == "__main__":
    main()
