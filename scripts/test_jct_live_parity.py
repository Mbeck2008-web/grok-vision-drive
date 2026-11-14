#!/usr/bin/env python3
"""Live ``jct.jsonl`` from the collector tick path equals the offline relabel, byte for byte.

Spec tests (``python/collect/junction_map.py``):
 4 live = offline parity: the mocked collector drives a scripted track over the synthetic
   roads (3 junctions and a respawn jump) and writes ``state.jsonl`` and ``jct.jsonl``;
   ``label_junctions.py`` on that ``state.jsonl`` gives identical bytes. Three mutants of
   the labeler each break the match: wall-clock (its only change is a fake wall-clock
   reading added to ``next_jct_m``; a clock at a whole second leaves it equal), one-row lag
   (row i gets row i-1's label) and lookahead (row i gets row i+1's label).
 5 schema unchanged: the same run with ``--no-jct`` gives byte-identical ``state.jsonl``
   and ``gt.jsonl``, and no ``jct.jsonl``.
 7 create-only (live half): an existing ``jct.jsonl`` in the out folder is kept.
The collector clocks are fake (``time.time``, ``monotonic``, ``perf_counter``, ``sleep``), so
no assert depends on the wall clock.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import test_collect_spawn_fix as tcs  # noqa: E402
from python.collect import junction_map as jm  # noqa: E402
from python.debug.gt_geom import parse_road_edges  # noqa: E402
from synthetic_roads import WALL0, load_graph, network_spec, track  # noqa: E402

LEVEL = "west_coast_usa"  # the collector's map rotation path; the roads are synthetic


def _poses() -> list[tuple[float, float, float]]:
    data = load_graph()
    # Lane-hugging corners (light smoothing) so the turns drive through the circles.
    first = track(data, "synth_a", ["S", "T", "X", "Y", "YL"], smooth_half=2)
    # Respawn: a jump of well over 30 m back to the west end, then W -> T -> S.
    second = track(data, "synth_a", ["W", "T", "S"], smooth_half=2)
    return first + second


class _TrackVehicle(tcs._CrashVehicle):
    """The mocked car follows ``POSES`` one pose per poll, then writes the stop file."""

    POSES: list[tuple[float, float, float]] = []

    def __init__(self, stop: Path, **kwargs) -> None:
        kwargs["stop_at"] = len(self.POSES)
        kwargs["speed_at"] = lambda n: 10.0
        super().__init__(stop, **kwargs)

    def _set(self, speed: float) -> None:
        i = min(max(self.polls - 1, 0), len(self.POSES) - 1)
        x, y, h = self.POSES[i]
        self.state = {
            "pos": [x, y, 0.5],
            "dir": [round(math.cos(h), 9), round(math.sin(h), 9), 0.0],
            "up": [0.0, 0.0, 1.0],
            "electrics": {"wheelspeed": speed, "steering_input": 0.0, "throttle_input": 0.3, "brake_input": 0.0},
        }


def _polys() -> list:
    roads = network_spec(load_graph(), "synth_a")
    return [parse_road_edges(rid, road["edges"]) for rid, road in roads.items()]


def _run(extra: list[str], maps_dir: Path, poses, pre=None) -> tuple[int, str, Path]:
    """One mocked collector run on fake clocks; returns (code, stdout, out folder)."""
    import scripts.collect_scene_lanes as collect
    import python.collect.spawn_mix as spawn_mix
    from python.collect.level_session import OpenedLevel

    polys = _polys()

    def enter(bng, level, cfg, cams):
        del level, cfg, cams
        return OpenedLevel(vehicle=bng.vehicles.vehicle, roads=list(polys), spawn=(0.0, -198.0, 0.5),
                           cameras={"main": tcs._Cam()})

    pools = {tcs.CITY: [tcs._road("far", [(5000.0, 0.0), (5000.0, 500.0)])], tcs.HIGHWAY: [], tcs.RAMP: []}
    root = Path(tempfile.mkdtemp(prefix="gvd-jct-"))
    if pre is not None:
        pre(root / "scene")
    _TrackVehicle.POSES = list(poses)
    real_time, real_mono, real_perf = time.time, time.monotonic, time.perf_counter
    saved = (collect.enter_map, collect.painted_for, spawn_mix.SpawnDeck.load, tcs._CrashVehicle, collect.JUNCTIONS_DIR)
    collect.enter_map = enter  # type: ignore[assignment]
    collect.painted_for = lambda level: tcs._Paint(lambda x, y: True)  # type: ignore[assignment]
    spawn_mix.SpawnDeck.load = classmethod(lambda cls, path, rng=None, **kw: cls(pools, rng, **kw))  # type: ignore[assignment]
    tcs._CrashVehicle = _TrackVehicle  # type: ignore[misc]
    collect.JUNCTIONS_DIR = maps_dir  # type: ignore[assignment]
    # time.time and perf_counter follow the fake monotonic clock _run_collector installs.
    time.time = lambda: WALL0 + time.monotonic()  # type: ignore[assignment]
    time.perf_counter = lambda: time.monotonic()  # type: ignore[assignment]
    try:
        code, text, out = tcs._run_collector({}, ["--maps", LEVEL, "--no-vary-env", "--no-excursions"] + extra, root)
    finally:
        (collect.enter_map, collect.painted_for, spawn_mix.SpawnDeck.load, tcs._CrashVehicle,
         collect.JUNCTIONS_DIR) = saved  # type: ignore[misc]
        time.time, time.monotonic, time.perf_counter = real_time, real_mono, real_perf  # type: ignore[assignment]
    return code, text, out


def _relabel(state_dir: Path, maps_dir: Path, status: Path) -> tuple[int, str]:
    import label_junctions

    buf = io.StringIO()
    old = label_junctions.now_s
    label_junctions.now_s = lambda: time.time() + 3600.0  # the run's files are "old" for the guard
    try:
        with contextlib.redirect_stdout(buf):
            code = label_junctions.main([str(state_dir), "--junctions", str(maps_dir), "--status-dir", str(status)])
    finally:
        label_junctions.now_s = old
    return code, buf.getvalue()


def _offline(out: Path, maps: Path, tmp: Path, name: str) -> bytes:
    copy_dir = tmp / name
    copy_dir.mkdir()
    shutil.copy2(out / "state.jsonl", copy_dir / "state.jsonl")
    code, text = _relabel(copy_dir, maps, tmp / "status")
    assert code == 0, (code, text)
    return (copy_dir / "jct.jsonl").read_bytes()


def main() -> None:
    poses = _poses()
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        maps = tmp / "maps"
        maps.mkdir()
        doc = jm.build_junction_map(network_spec(load_graph(), "synth_a"), LEVEL, "synthetic")
        jm.write_map(doc, maps)
        (tmp / "status").mkdir()

        # ---- 4 parity
        code, text, out = _run([], maps, poses)
        assert code == 0, (code, text[-3000:])
        live = (out / "jct.jsonl").read_bytes()
        state_lines = (out / "state.jsonl").read_text(encoding="utf-8").splitlines()
        live_lines = live.decode("utf-8").splitlines()
        assert len(live_lines) == len(state_lines) >= 900, (len(live_lines), len(state_lines))
        labels = [json.loads(line) for line in live_lines]
        rows = [json.loads(line) for line in state_lines]
        assert [lab["t"] for lab in labels] == [r["t"] for r in rows]
        ids = {lab["jct_id"] for lab in labels if lab["in_jct"]}
        assert len(ids) >= 3, ids
        jumps = sum(1 for a, b in zip(rows, rows[1:])
                    if math.hypot(b["ego"]["pos"][0] - a["ego"]["pos"][0], b["ego"]["pos"][1] - a["ego"]["pos"][1]) > jm.JUMP_M)
        assert jumps >= 1, "the scripted track has no respawn jump"
        assert "jct_s" in (out / "ticks.jsonl").read_text(encoding="utf-8") or "jct_s" in text, "no jct_s tick cost"
        offline = _offline(out, maps, tmp, "relabel")
        assert offline == live, "offline relabel differs from the live jct.jsonl"
        print(f"  4 parity: {len(live_lines)} rows, {len(ids)} junctions entered, {jumps} respawn jump; live == offline bytes")

        # Mutants of the shared labeler must break the match.
        real_step = jm.JunctionLabeler.step

        def wall_clock(self, row):
            # The only change is the wall-clock reading (fractional second) added to next_jct_m.
            got = real_step(self, row)
            if got.get("next_jct_m") is not None:
                got["next_jct_m"] = round(got["next_jct_m"] + time.time() % 1.0, 2)
            return got

        pending: list = []

        def one_row_lag(self, row):
            # Row i gets row i-1's label (row 0 keeps its own).
            pending.append(real_step(self, row))
            prev = pending.pop(0) if len(pending) > 1 else pending[0]
            return dict(prev, t=row.get("t"))

        by_t = {r["t"]: i for i, r in enumerate(rows)}
        assert len(by_t) == len(rows), "row t values are not unique"

        def lookahead(self, row):
            # Row i gets row i+1's label: the labeler steps on the next (future) row's pose.
            i = by_t[row.get("t")]
            nxt = rows[i + 1] if i + 1 < len(rows) else row
            return dict(real_step(self, nxt), t=row.get("t"))

        def mutant_bytes(name, mutant, clock: float) -> bytes:
            # A fake, fixed wall clock: the mutant's output depends on its value, not on when it runs.
            real_clock = time.time
            jm.JunctionLabeler.step = mutant  # type: ignore[assignment]
            time.time = lambda: clock  # type: ignore[assignment]
            try:
                return _offline(out, maps, tmp, f"mutant_{name}_{clock!r}")
            finally:
                jm.JunctionLabeler.step = real_step  # type: ignore[assignment]
                time.time = real_clock  # type: ignore[assignment]

        clock_whole, clock_frac = WALL0 + 7200.0, WALL0 + 7200.25
        # Control: with the clock on a whole second the wall-clock mutant adds 0 and matches live,
        # so its failure below comes from the clock reading alone.
        assert mutant_bytes("wall_clock", wall_clock, clock_whole) == live, "wall_clock control differs"
        for name, mutant in (("wall_clock", wall_clock), ("one_row_lag", one_row_lag), ("lookahead", lookahead)):
            pending.clear()
            got = mutant_bytes(name, mutant, clock_frac)
            assert got != live, f"the parity check did not catch the {name} mutant"
        print("  4 mutants: wall-clock (clock-only; whole-second control matches), one-row lag and "
              "lookahead labelers all differ from live")

        # ---- 5 schema unchanged
        code, text_off, out_off = _run(["--no-jct"], maps, poses)
        assert code == 0, (code, text_off[-3000:])
        assert not (out_off / "jct.jsonl").exists()
        for name in ("state.jsonl", "gt.jsonl"):
            a, b = out / name, out_off / name
            assert a.is_file() == b.is_file(), name
            if a.is_file():
                assert a.read_bytes() == b.read_bytes(), f"{name} differs with --no-jct"
        assert (out / "state.jsonl").is_file()
        print(f"  5 schema: state.jsonl{' and gt.jsonl' if (out / 'gt.jsonl').is_file() else ''} byte-identical with --no-jct")

        # ---- 7 create-only, live
        def pre(scene: Path) -> None:
            scene.mkdir(parents=True, exist_ok=True)
            (scene / "jct.jsonl").write_bytes(b"keep\n")

        code, text_pre, out_pre = _run([], maps, poses[:80], pre=pre)
        assert code == 0, (code, text_pre[-3000:])
        assert (out_pre / "jct.jsonl").read_bytes() == b"keep\n"
        assert "collect jct" in text_pre and "exists" in text_pre, text_pre[-2000:]
        # A map with no junction file: one log line, no file.
        empty = tmp / "no_maps"
        empty.mkdir()
        code, text_none, out_none = _run([], empty, poses[:80])
        assert code == 0 and not (out_none / "jct.jsonl").exists()
        assert text_none.count(f"collect jct off map={LEVEL}") == 1, text_none[-2000:]
        print("  7 create-only (live): an existing jct.jsonl is kept; a missing map logs once and writes nothing")
    print("test_jct_live_parity: OK")


if __name__ == "__main__":
    main()
