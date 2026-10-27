#!/usr/bin/env python3
"""Errors from ``get_road_network`` on the recovery-snap path. No BeamNG.

``snap_to_road`` reads the network through ``RoadLoader.network``. A session
error (``BNGError``, a dead socket) means "no road pose" and returns None, as
before. Any other exception is a bug in the reply or in our call and must not
be hidden as "no road": it propagates out of ``snap_to_road``. The snapshot
note still records it, and the ``polys()`` and GT logger paths stay note-only.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from python.collect import level_session, road_loader  # noqa: E402
from python.debug.gt_logger import GroundTruthLogger  # noqa: E402
from synthetic_roads import FakeBng, FakeClock, drive, load_graph, patched_logger_env, track  # noqa: E402

NEAR = (0.0, -150.0, 0.0)


class BNGError(Exception):
    """Stands in for beamngpy's BNGError (matched by class name)."""


class _RaisingBng(FakeBng):
    def __init__(self, graph: dict[str, Any], exc: BaseException) -> None:
        super().__init__(graph, "synth_a")
        self.exc = exc

    def network(self) -> dict[str, Any]:
        raise self.exc


def check_value_error_propagates() -> None:
    graph = load_graph()
    road_loader.reset_shared()
    bng = _RaisingBng(graph, ValueError("bad road network reply"))
    try:
        level_session.snap_to_road(bng, NEAR)
    except ValueError as exc:
        assert "bad road network reply" in str(exc), exc
    else:
        raise AssertionError("ValueError from get_road_network was swallowed by snap_to_road")
    snap = road_loader.shared().snapshot(bng)
    assert "ValueError: bad road network reply" in snap.note, snap.note
    assert isinstance(snap.error, ValueError), snap.error
    polys, note = road_loader.shared().polys(bng)  # note-only path
    assert polys == [] and "ValueError" in note, (polys, note)
    clock = FakeClock()
    with tempfile.TemporaryDirectory() as tmp, patched_logger_env(clock):
        logger = GroundTruthLogger(Path(tmp) / "gt.jsonl", Path(tmp) / "gt")
        rows = drive(logger, bng, track(graph, "synth_a", ["S", "T"])[:20], clock)  # note-only path
    assert all(row["lane_x"] is None for row in rows)
    print(f"ValueError: raised out of snap_to_road; note kept ({snap.note!r}); polys() and GT logger note-only")


def check_session_error_returns_none() -> None:
    graph = load_graph()
    for exc in (BNGError("road network unavailable"), ConnectionError("socket closed")):
        road_loader.reset_shared()
        bng = _RaisingBng(graph, exc)
        assert level_session.snap_to_road(bng, NEAR) is None, exc
        snap = road_loader.shared().snapshot(bng)
        assert type(exc).__name__ in snap.note, snap.note
        print(f"{type(exc).__name__}: snap_to_road -> None, note {snap.note!r}")


def main() -> None:
    check_value_error_propagates()
    check_session_error_returns_none()
    road_loader.reset_shared()
    print("test_road_loader_errors: OK")


if __name__ == "__main__":
    main()
