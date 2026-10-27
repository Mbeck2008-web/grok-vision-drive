#!/usr/bin/env python3
"""One road loader, one read per level. No BeamNG.

A fake session counts ``get_road_network`` reads through a whole collect
session: start, GT ticks, recoveries, a same-level restart and a level
change. Also checks that the cached network is what a fresh read returns,
that recovery snaps land where the old re-read-every-call path put them,
and that only the loader calls ``get_road_network``.
"""
from __future__ import annotations

import ast
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from python.collect import level_session, liveness, road_loader  # noqa: E402
from python.debug.gt_logger import GroundTruthLogger  # noqa: E402
from synthetic_roads import FakeBng, FakeClock, drive, load_graph, patched_logger_env, track  # noqa: E402

SPAWNS = {"synth_a": (0.0, -150.0, 0.0), "synth_b": (5000.0, 5000.0, 0.0)}


class _Vehicle:
    vid = "ego"


class patched_session:
    """``begin_level`` without a game: spawn, player lookup and teleport are stubs."""

    NAMES = ("resolve_spawn", "spawn_scenario", "player_vehicle", "place_vehicle")

    def __enter__(self) -> "patched_session":
        self._saved = {n: getattr(level_session, n) for n in self.NAMES}
        self._settle = liveness.SETTLE_S
        self.placed: list[tuple[float, float, float]] = []
        level_session.resolve_spawn = lambda bng, level: (SPAWNS[level], (0.0, 0.0, 0.0, 1.0), "known")
        level_session.spawn_scenario = lambda bng, level, pos, rot: ""
        level_session.player_vehicle = lambda bng: (_Vehicle(), "")

        def place(bng: Any, vehicle: Any, pos: Any, rot: Any) -> str:
            self.placed.append(pos)
            return ""

        level_session.place_vehicle = place
        liveness.SETTLE_S = 0.0
        return self

    def __exit__(self, *exc: Any) -> None:
        for name, fn in self._saved.items():
            setattr(level_session, name, fn)
        liveness.SETTLE_S = self._settle


def _legacy_network_spec(bng: Any) -> Any:
    """``level_session._network_spec`` as it was on main: a fresh read on every call."""
    for owner in (getattr(bng, "scenario", None), bng):
        if owner is None:
            continue
        fn = getattr(owner, "get_road_network", None)
        if not callable(fn):
            continue
        try:
            return fn(include_edges=True, drivable_only=True)
        except TypeError:
            return fn()
    return None


def _gt_poses(graph: dict[str, Any], level: str, path: list[str], count: int) -> list[tuple[float, float, float]]:
    forward = track(graph, level, path)
    back = track(graph, level, list(reversed(path)))
    poses: list[tuple[float, float, float]] = []
    while len(poses) < count:
        poses.extend(forward)
        poses.extend(back)
    return poses[:count]


def check_session_read_counts() -> None:
    graph = load_graph()
    loader = road_loader.reset_shared()
    bng = FakeBng(graph, "synth_a")
    counts: list[tuple[str, int]] = []
    with patched_session() as session, tempfile.TemporaryDirectory() as tmp:
        opened = level_session.begin_level(bng, "synth_a")
        assert not opened.error and opened.roads, opened
        assert session.placed, "begin_level did not snap the spawn onto the road"
        counts.append(("session start", bng.reads))
        assert bng.reads == 1, counts

        clock = FakeClock()
        with patched_logger_env(clock):
            logger = GroundTruthLogger(Path(tmp) / "gt.jsonl", Path(tmp) / "gt")
            lua_before = bng.lua_calls
            rows = drive(logger, bng, _gt_poses(graph, "synth_a", ["W", "T", "X", "Y", "YL"], 1000), clock)
        assert len(rows) == 1000 and sum(r["lane_x"] is not None for r in rows) > 900
        counts.append(("1000 GT ticks", bng.reads))
        assert bng.reads == 1, counts
        assert bng.lua_calls - lua_before <= 1, f"level asked {bng.lua_calls - lua_before} times in 1000 ticks"

        stalls = [(10.0, -120.0, 0.3), (-150.0, 2.0, 0.3), (180.0, 30.0, 0.3), (420.0, 15.0, 0.3), (590.0, -95.0, 0.3)]
        snaps = 0
        for stall in stalls:
            for point in level_session.recover_points(stall, SPAWNS["synth_a"]):
                snaps += level_session.snap_to_road(bng, point) is not None
        assert snaps >= 5, snaps
        counts.append(("5 recoveries", bng.reads))
        assert bng.reads == 1, counts

        opened = level_session.begin_level(bng, "synth_a")
        assert not opened.error and opened.roads, opened
        opened = level_session.begin_level(bng, "")
        assert not opened.error and opened.roads, opened
        counts.append(("begin_level same level", bng.reads))
        assert bng.reads == 1, counts

        bng.switch_level("synth_b")
        opened = level_session.begin_level(bng, "synth_b")
        assert not opened.error and opened.roads, opened
        assert {p.road_id for p in opened.roads} == {"BW-BE"}, [p.road_id for p in opened.roads]
        counts.append(("level change", bng.reads))
        assert bng.reads == 2, counts

        # The GT logger follows the car onto level B without another read.
        with patched_logger_env(clock):
            rows = drive(logger, bng, track(graph, "synth_b", ["BW", "BE"]), clock)
        assert any(r["lane_x"] is not None for r in rows)
        assert {r["road_id"] for r in rows if r["road_id"]} == {"BW-BE"}
        counts.append(("GT ticks on level B", bng.reads))
        assert bng.reads == 2, counts
    assert loader.reads == 2 and loader.level == "synth_b", (loader.reads, loader.level)
    assert all(args == (True, True) for args in bng.read_args), bng.read_args
    print("  reads: " + ", ".join(f"{name} {n}" for name, n in counts))


def check_empty_network_retry() -> None:
    """An empty network is never cached: the logger keeps retrying every 2 s, not every tick."""
    graph = load_graph()
    road_loader.reset_shared()
    bng = FakeBng(graph, "synth_a", empty=True)
    poses = track(graph, "synth_a", ["W", "T", "X"])
    clock = FakeClock()
    with tempfile.TemporaryDirectory() as tmp, patched_logger_env(clock):
        logger = GroundTruthLogger(Path(tmp) / "gt.jsonl", Path(tmp) / "gt")
        rows = drive(logger, bng, poses[:100], clock)  # 10 s of ticks
        assert all(r["lane_x"] is None for r in rows)
        empty_reads = bng.reads
        assert 4 <= empty_reads <= 6, f"{empty_reads} reads in 10 s of empty network (want one per 2 s)"
        bng.empty = False
        rows = drive(logger, bng, poses[100:130], clock)
    assert bng.reads == empty_reads + 1, (empty_reads, bng.reads)
    assert any(r["lane_x"] is not None for r in rows), "roads never loaded after the network appeared"
    print(f"  empty network: {empty_reads} reads in 10 s, then 1 read once roads appear")


def check_recovery_equivalence() -> None:
    """3b: cached network == fresh read; recovery snaps match the old re-read path."""
    graph = load_graph()
    loader = road_loader.reset_shared()
    bng = FakeBng(graph, "synth_a")
    loader.enter(bng, "synth_a")
    cached = loader.network(bng)
    fresh = FakeBng(graph, "synth_a").scenario.get_road_network(include_edges=True, drivable_only=True)
    assert cached == fresh, "cached network differs from a fresh read"
    assert bng.read_args == [(True, True)], bng.read_args

    stalls = [(10.0, -120.0, 0.3), (-150.0, 2.0, 0.3), (180.0, 30.0, 0.3), (420.0, 15.0, 0.3),
              (590.0, -95.0, 0.3), (205.0, 150.0, 0.3), (2100.0, 2010.0, 0.3), (900.0, 900.0, 0.3)]
    new_poses = []
    for stall in stalls:
        for point in level_session.recover_points(stall, SPAWNS["synth_a"]):
            new_poses.append(level_session.snap_to_road(bng, point))
    assert bng.reads == 1, bng.reads

    legacy_bng = FakeBng(graph, "synth_a")
    saved = level_session._network_spec
    level_session._network_spec = _legacy_network_spec
    try:
        old_poses = []
        for stall in stalls:
            for point in level_session.recover_points(stall, SPAWNS["synth_a"]):
                old_poses.append(level_session.snap_to_road(legacy_bng, point))
    finally:
        level_session._network_spec = saved
    assert new_poses == old_poses, "recovery snap moved"
    assert any(p is None for p in new_poses) and sum(p is not None for p in new_poses) >= 6
    print(f"  recovery: {len(new_poses)} snaps identical to the re-read path "
          f"({legacy_bng.reads} reads there, 1 here)")


def _source_files(*, skip_tests: bool) -> list[Path]:
    files = []
    for top in ("python", "scripts"):
        for path in sorted((ROOT / top).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if skip_tests and path.parent == ROOT / "scripts" and path.name.startswith("test_"):
                continue
            files.append(path)
    return files


def check_single_read_site() -> None:
    """Acceptance 1, the same scan as ``rg -n "get_road_network\\(" python scripts -g '!scripts/test_*'``."""
    pattern = re.compile(r"get_road_network\(")
    hits = []
    for path in _source_files(skip_tests=True):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(ROOT).as_posix()}:{n}")
    assert hits and all(h.startswith("python/collect/road_loader.py:") for h in hits), hits
    print(f"  get_road_network( only in {', '.join(hits)}")


def check_loader_importers() -> None:
    """``road_loader`` is imported directly only by ``gt_logger`` and ``level_session``."""
    importers = set()
    for path in _source_files(skip_tests=True):
        if path.suffix != ".py" or path.name == "road_loader.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                names = [mod] + [f"{mod}.{a.name}" for a in node.names]
            if any(n.split(".")[-1] == "road_loader" for n in names):
                importers.add(path.relative_to(ROOT).as_posix())
    want = {"python/debug/gt_logger.py", "python/collect/level_session.py"}
    assert importers == want, importers
    print(f"  road_loader imported by {', '.join(sorted(importers))}")


def main() -> None:
    check_session_read_counts()
    check_empty_network_retry()
    check_recovery_equivalence()
    check_single_read_site()
    check_loader_importers()
    road_loader.reset_shared()
    print("test_road_loader_cache: OK")


if __name__ == "__main__":
    main()
