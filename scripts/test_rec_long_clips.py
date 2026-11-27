#!/usr/bin/env python3
"""Long regular-driving clips mixed into a collect_clips batch.

Offline. Plans, argv, tags, and scheduling only; no BeamNG.

    py -3 scripts/test_rec_long_clips.py
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from test_clip_mode import MAPS, _fake_recording  # noqa: E402
from python.collect import clip_mode as _clip_mode  # noqa: E402

_REAL_ROUTE_PLANNER = _clip_mode._route_cached


@contextlib.contextmanager
def fast_routes(fail: bool = False):
    """Plan routes on a small synthetic network instead of the real road dump (about 1 s a route).

    Same planner, tiny map: the clip-mode logic around it is what these checks are about. The real
    dumps are exercised by ``check_long_clip_routes_direction_and_maps`` and ``test_long_route``.
    ``fail`` makes every plan fail, as on a map where no route can be found.
    """
    from python.collect import clip_mode, long_route as lr
    from test_long_route import loop_network

    graphs = {d: lr.build_graph(loop_network(dirt=True, mirror=bool(d))) for d in (0, 1)}
    orig = clip_mode._route_cached

    def fake(map_name, seed, direction, duration_s):
        if fail:
            return None
        return lr.plan_highway_route(graphs[int(direction) % 2], map_name=map_name, seed=int(seed),
                                     direction=int(direction), duration_s=900.0)

    clip_mode._route_cached = fake
    try:
        yield
    finally:
        clip_mode._route_cached = orig


@contextlib.contextmanager
def real_routes():
    from python.collect import clip_mode

    cur = clip_mode._route_cached
    clip_mode._route_cached = _REAL_ROUTE_PLANNER
    try:
        yield
    finally:
        clip_mode._route_cached = cur


def _cli():
    spec = importlib.util.spec_from_file_location("collect_clips_long", ROOT / "scripts" / "collect_clips.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)  # type: ignore[union-attr]
    return cli


def check_default_off_unchanged() -> None:
    """--long-clips defaults to 0; ordinary plans stay 30-60 s with no long tag."""
    from python.collect.clip_mode import plan_clip

    cli = _cli()
    args = cli.parse_args(["--out", "/tmp/x"])
    assert args.long_clips == 0, args
    assert args.long_clip_s == (600.0, 900.0), args.long_clip_s
    assert abs(args.long_clip_highway_frac - 0.7) < 1e-12, args.long_clip_highway_frac

    plans = [plan_clip(i, seed=7, maps=MAPS, max_traffic=8) for i in range(40)]
    for p in plans:
        assert 30.0 <= p.duration_s <= 60.0, p
        assert not getattr(p, "long_highway", False)
        assert "long_highway" not in getattr(p, "tags", [])
    print("default off: --long-clips 0, plans stay 30-60 s")


def check_cli_long_clip_flags() -> None:
    cli = _cli()
    args = cli.parse_args(["--out", "/tmp/x", "--clips", "60", "--long-clips", "8"])
    assert args.long_clips == 8 and args.clips == 60
    assert args.long_clip_s == (600.0, 900.0)
    assert abs(args.long_clip_highway_frac - 0.7) < 1e-12
    args = cli.parse_args(["--out", "/tmp/x", "--clips", "20", "--long-clips", "5", "--long-clip-s", "180,240", "--long-clip-highway-frac", "0.4"])
    assert args.long_clips == 5 and args.long_clip_s == (180.0, 240.0)
    assert abs(args.long_clip_highway_frac - 0.4) < 1e-12
    for bad in (
        ["--long-clips", "-1"],
        ["--clips", "5", "--long-clips", "10"],
        ["--long-clip-s", "300,180"],
        ["--long-clip-s", "0,300"],
        ["--long-clip-s", "abc"],
        ["--max-s", "90"],
        ["--long-clip-highway-frac", "-0.1"],
        ["--long-clip-highway-frac", "1.1"],
    ):
        try:
            cli.parse_args(["--out", "/tmp/x"] + bad)
        except SystemExit:
            continue
        raise AssertionError(f"accepted {bad}")
    # Normal 30-60 bound is unchanged; long range is a separate flag.
    ok = cli.parse_args(["--out", "/tmp/x", "--min-s", "30", "--max-s", "60", "--long-clips", "2", "--clips", "10"])
    assert ok.min_s == 30.0 and ok.max_s == 60.0 and ok.long_clips == 2
    print("CLI: --long-clips / --long-clip-s parse and reject bad values; --max-s cap stays 60")


def check_long_clip_indices_spread_and_seeded() -> None:
    from python.collect.clip_mode import long_clip_indices

    assert long_clip_indices(60, 0, seed=1) == ()
    idx = long_clip_indices(60, 8, seed=1)
    assert idx == long_clip_indices(60, 8, seed=1)
    assert idx != long_clip_indices(60, 8, seed=2)
    assert len(idx) == len(set(idx)) == 8
    assert all(0 <= i < 60 for i in idx)
    # Spread through the batch, not packed at the end (or the start).
    assert min(idx) < 15, idx
    assert max(idx) >= 45, idx
    # One index in each of 8 contiguous buckets covering 0..59.
    for i in range(8):
        lo, hi = i * 60 // 8, (i + 1) * 60 // 8
        assert sum(1 for x in idx if lo <= x < hi) == 1, (idx, lo, hi)
    try:
        long_clip_indices(5, 10, seed=1)
    except ValueError:
        pass
    else:
        raise AssertionError("long_clips > clips was accepted")
    print("long_clip_indices: seeded, unique, spread through the batch")


def check_long_plan_duration_spawn_speed_and_tags() -> None:
    from python.collect.clip_mode import (
        apply_long_highway,
        clip_argv,
        plan_clip,
        write_tags,
    )
    from scripts.gvd_test_env import collect_root

    base = plan_clip(3, seed=11, maps=MAPS, max_traffic=4)
    plan = apply_long_highway(base, min_s=180.0, max_s=300.0, maps=MAPS, highway_frac=1.0)
    again = apply_long_highway(base, min_s=180.0, max_s=300.0, maps=MAPS, highway_frac=1.0)
    assert plan.duration_s == again.duration_s
    assert 180.0 <= plan.duration_s <= 300.0
    assert plan.long_highway is True
    assert plan.spawn_kind == "highway"
    assert plan.speed_min == 8.0 and plan.speed_max == 30.0
    assert "long_highway" in plan.tags
    try:
        clip_argv(plan, Path("/tmp/x/clip"))
    except ValueError:
        pass
    else:
        raise AssertionError("a highway-loop clip got an argv with no route plan (it would roam freely)")
    argv = clip_argv(plan, Path("/tmp/x/clip"), route_plan=Path("/tmp/x/_routes/clip.json"))
    assert argv[argv.index("--route-plan") + 1] == str(Path("/tmp/x/_routes/clip.json"))
    assert "--no-excursions" in argv and "--route" not in argv
    assert float(argv[argv.index("--clip-s") + 1]) == plan.duration_s
    assert argv[argv.index("--spawn-kind") + 1] == "highway"
    assert float(argv[argv.index("--speed-min") + 1]) == 8.0
    assert float(argv[argv.index("--speed-max") + 1]) == 30.0
    assert float(argv[argv.index("--segment-min") + 1]) * 60.0 >= plan.duration_s
    # A normal plan does not pick up the highway speed clamp.
    normal_argv = clip_argv(base, Path("/tmp/x/clip"))
    assert "--speed-min" not in normal_argv and "--speed-max" not in normal_argv
    assert "--route-plan" not in normal_argv and "--no-excursions" not in normal_argv
    assert 30.0 <= base.duration_s <= 60.0

    with collect_root() as root:
        clip = root / "clip"
        clip.mkdir()
        tags = json.loads(write_tags(clip, plan, {"quality": "train", "reasons": []}).read_text(encoding="utf-8"))
    assert "long_highway" in tags["tags"]
    assert tags["duration_s"] == plan.duration_s
    assert tags["spawn_kind"] == "highway"
    assert plan.long is True
    assert "long" in plan.tags
    print("long plan: custom 180-300 s, highway-first route, tagged long + long_highway")


def check_batch_mixes_long_clips() -> None:
    from python.collect.clip_mode import long_clip_indices, run_clips
    from scripts.gvd_test_env import collect_root

    calls = []

    def runner(argv: list[str]) -> int:
        out = Path(argv[argv.index("--out") + 1])
        calls.append(argv)
        _fake_recording(out, rows=50)
        return 0

    with collect_root() as root:
        session = run_clips(
            root, clips=10, seed=5, maps=MAPS, max_traffic=4, runner=runner,
            min_s=30.0, max_s=60.0, long_clips=3, long_min_s=180.0, long_max_s=300.0,
        )
        want = set(long_clip_indices(10, 3, seed=5))
        manifest = json.loads((session / "session.json").read_text(encoding="utf-8"))
        assert manifest["long_clips"] == 3, manifest
        assert len(calls) == 10
        long_seen = []
        for i, argv in enumerate(calls):
            dur = float(argv[argv.index("--clip-s") + 1])
            tags = json.loads((session / Path(argv[argv.index("--out") + 1]).name / "clip_tags.json").read_text(encoding="utf-8"))
            if i in want:
                assert 180.0 <= dur <= 300.0, (i, dur)
                assert "long" in tags["tags"], (i, tags)
                assert tags.get("long") is True
                # Short-clip flags stay off the short clips; long clips stay long.
                long_seen.append(i)
            else:
                assert 30.0 <= dur <= 60.0, (i, dur)
                assert "long" not in tags.get("tags", [])
                assert tags.get("long_highway_planned") is not True
                assert "--segment-min" not in argv
        assert set(long_seen) == want
    print("run_clips: 3 of 10 clips are long regular driving, spread by seed, short clips unchanged")


def check_long_clip_early_end_kept_like_normal() -> None:
    """A crash or short recording is tagged pretrain_only and the batch goes on."""
    from python.collect.clip_mode import long_clip_indices, run_clips
    from scripts.gvd_test_env import collect_root

    def runner(argv: list[str]) -> int:
        out = Path(argv[argv.index("--out") + 1])
        # ~4 s of frames: well under half of a 180-300 s plan.
        _fake_recording(out, rows=40)
        return 4

    with collect_root() as root:
        session = run_clips(
            root, clips=4, seed=9, maps=MAPS, max_traffic=2, runner=runner,
            min_s=30.0, max_s=60.0, long_clips=2, long_min_s=180.0, long_max_s=300.0,
        )
        manifest = json.loads((session / "session.json").read_text(encoding="utf-8"))
        assert len(manifest["results"]) == 4
        want = set(long_clip_indices(4, 2, seed=9))
        long_dirs = 0
        for row in manifest["results"]:
            tags = json.loads((session / row["dir"] / "clip_tags.json").read_text(encoding="utf-8"))
            assert tags["quality"] == "pretrain_only"
            assert (session / row["dir"] / "state.jsonl").is_file()  # kept, not deleted
            if "long_highway" in tags.get("tags", []):
                long_dirs += 1
                assert any("exit" in r or "recorded" in r for r in tags["reasons"]), tags
        assert long_dirs == len(want) == 2
    print("early end: long clips are marked pretrain_only, kept, and the batch continues")


def check_long_dirt_recipe_does_not_fail_highway_footage() -> None:
    """A long clip that inherited a dirt_road edge case is not failed for missing dirt."""
    from python.collect.clip_mode import apply_long_highway, plan_clip, quality_check
    from scripts.gvd_test_env import collect_root

    base = plan_clip(0, seed=1, maps=MAPS, max_traffic=2, edge_cases=("dirt_road",), min_s=4.0, max_s=4.0)
    plan = apply_long_highway(base, min_s=4.0, max_s=4.0, maps=MAPS, highway_frac=1.0)
    assert plan.long and plan.long_highway and plan.spawn_kind == "highway"
    assert plan.map in ("west_coast_usa", "italy")
    with collect_root() as root:
        d = root / "clip"
        _fake_recording(d, rows=50, spawn_type="highway")
        qc = quality_check(d, plan, exit_code=0)
    assert qc["quality"] == "train", qc
    print("long clip: dirt_road edge-case recipe does not fail highway footage")


def check_long_clips_only_on_highway_maps() -> None:
    """Highway-first routes stay on highway maps; maps without highway are not skipped."""
    from python.collect.clip_mode import (
        apply_long_highway,
        highway_maps,
        long_clip_indices,
        map_has_highway_pool,
        plan_clip,
        run_clips,
    )
    from scripts.gvd_test_env import collect_root

    assert map_has_highway_pool("west_coast_usa")
    assert map_has_highway_pool("italy")
    assert not map_has_highway_pool("east_coast_usa")
    assert not map_has_highway_pool("small_island")
    default = ("west_coast_usa", "italy", "east_coast_usa")
    hwy = set(highway_maps(default))
    assert hwy == {"west_coast_usa", "italy"}, hwy

    # highway_frac=1: every long clip on a highway map, tagged long_highway.
    seen_maps = set()
    for seed in range(80):
        for i in long_clip_indices(60, 8, seed):
            base = plan_clip(i, seed=seed, maps=default, max_traffic=4)
            plan = apply_long_highway(base, min_s=180.0, max_s=300.0, maps=default, highway_frac=1.0)
            assert plan.long is True, (seed, i, plan)
            assert plan.long_highway is True, (seed, i, plan)
            assert plan.map in hwy, (seed, i, base.map, plan.map, base.edge_case)
            assert plan.spawn_kind == "highway"
            assert plan.long_route == "highway_loop"
            seen_maps.add(plan.map)
    assert seen_maps <= hwy

    dirt_pool = ("utah", "jungle_rock_island", "italy", "east_coast_usa", "small_island", "west_coast_usa")
    for seed in range(40):
        base = plan_clip(0, seed=seed, maps=dirt_pool, max_traffic=2, edge_cases=("dirt_road",))
        plan = apply_long_highway(base, min_s=180.0, max_s=300.0, maps=dirt_pool, highway_frac=1.0)
        assert plan.long and plan.long_highway, (seed, base.map, plan)
        assert map_has_highway_pool(plan.map), (seed, base.map, plan.map, base.edge_case)

    # A session with no highway map still records a long regular-driving clip
    # (varied seeded route). It is not skipped and is not tagged long_highway.
    base = plan_clip(0, seed=1, maps=("east_coast_usa",), max_traffic=0)
    plan = apply_long_highway(base, min_s=180.0, max_s=300.0, maps=("east_coast_usa",), highway_frac=1.0)
    assert plan.long is True
    assert plan.long_highway is False
    assert "long_highway" not in plan.tags
    assert "long" in plan.tags
    assert plan.map == "east_coast_usa"
    assert plan.long_route == "varied"
    assert 180.0 <= plan.duration_s <= 300.0
    assert plan.duration_s != base.duration_s or base.duration_s >= 180.0

    def runner(argv: list[str]) -> int:
        out = Path(argv[argv.index("--out") + 1])
        _fake_recording(out, rows=50)
        return 0

    with collect_root() as root:
        session = run_clips(
            root, clips=20, seed=3, maps=default, max_traffic=4, runner=runner,
            min_s=30.0, max_s=60.0, long_clips=5, long_min_s=180.0, long_max_s=300.0,
            long_highway_frac=1.0,
        )
        want = set(long_clip_indices(20, 5, seed=3))
        for i, row in enumerate(json.loads((session / "session.json").read_text(encoding="utf-8"))["results"]):
            tags = json.loads((session / row["dir"] / "clip_tags.json").read_text(encoding="utf-8"))
            if i in want:
                assert tags["long"] is True, (i, tags)
                assert tags["long_highway_planned"] is True, (i, tags)
                assert row["map"] in hwy, (i, row)
            else:
                assert tags.get("long") is not True
                assert tags["long_highway_planned"] is False
    print("highway-first routes stay on highway maps; no-highway maps still get varied long clips")


def check_long_clip_boundaries_and_tags() -> None:
    """N==clips and MIN==MAX are valid; MIN>MAX is refused; duration is not pinned; tags/manifest/notes hold."""
    from python.collect.clip_mode import (
        LONG_HIGHWAY_TAG,
        apply_long_highway,
        long_clip_indices,
        plan_clip,
        run_clips,
        write_tags,
    )
    from scripts.gvd_test_env import collect_root

    # N == clips is allowed (planner + CLI). N < 0 is refused in the planner.
    idx = long_clip_indices(8, 8, seed=1)
    assert len(idx) == 8 and set(idx) == set(range(8)), idx
    cli = _cli()
    ok = cli.parse_args(["--out", "/tmp/x", "--clips", "8", "--long-clips", "8"])
    assert ok.long_clips == 8 and ok.clips == 8
    try:
        long_clip_indices(5, -1, seed=1)
    except ValueError:
        pass
    else:
        raise AssertionError("long_clips < 0 was accepted")

    # MIN == MAX is a valid fixed duration; extra comma-separated parts are not.
    same = cli.parse_args(["--out", "/tmp/x", "--long-clip-s", "180,180", "--clips", "4", "--long-clips", "1"])
    assert same.long_clip_s == (180.0, 180.0)
    try:
        cli.parse_args(["--out", "/tmp/x", "--long-clip-s", "180,300,999"])
    except SystemExit:
        pass
    else:
        raise AssertionError("accepted --long-clip-s with 3 parts")

    base = plan_clip(3, seed=11, maps=("west_coast_usa",), max_traffic=4)
    try:
        apply_long_highway(base, min_s=300.0, max_s=180.0, maps=("west_coast_usa",))
    except ValueError:
        pass
    else:
        raise AssertionError("apply_long_highway accepted MIN > MAX")
    pinned = apply_long_highway(base, min_s=200.0, max_s=200.0, maps=("west_coast_usa",), highway_frac=1.0)
    assert pinned.duration_s == 200.0
    assert pinned.min_paint_cov is None

    # Duration uses both ends of the range, and the seed includes the clip index.
    faded = plan_clip(0, seed=4, maps=("west_coast_usa",), max_traffic=2, edge_cases=("faded_paint",))
    assert faded.min_paint_cov == 0.0
    cleared = apply_long_highway(faded, min_s=180.0, max_s=300.0, maps=("west_coast_usa",), highway_frac=1.0)
    assert cleared.min_paint_cov is None
    assert LONG_HIGHWAY_TAG in cleared.notes
    durs = []
    for i in range(40):
        p = plan_clip(i, seed=7, maps=("west_coast_usa",), max_traffic=0)
        durs.append(apply_long_highway(p, min_s=180.0, max_s=300.0, maps=("west_coast_usa",)).duration_s)
    assert any(d > 180.5 for d in durs), durs
    assert any(d < 299.5 for d in durs), durs
    p0 = plan_clip(0, seed=7, maps=("west_coast_usa",), max_traffic=0)
    p1 = plan_clip(1, seed=7, maps=("west_coast_usa",), max_traffic=0)
    # Same seed, different index: duration seed must include the index (not always equal).
    d0 = apply_long_highway(p0, min_s=180.0, max_s=300.0, maps=("west_coast_usa",)).duration_s
    d1 = apply_long_highway(p1, min_s=180.0, max_s=300.0, maps=("west_coast_usa",)).duration_s
    # If a single pair collides, more indices still have to differ.
    more = [
        apply_long_highway(plan_clip(i, seed=7, maps=("west_coast_usa",), max_traffic=0),
                           min_s=180.0, max_s=300.0, maps=("west_coast_usa",)).duration_s
        for i in range(8)
    ]
    assert len(set(more)) > 1, more
    assert d0 == apply_long_highway(p0, min_s=180.0, max_s=300.0, maps=("west_coast_usa",)).duration_s

    def runner(argv: list[str]) -> int:
        out = Path(argv[argv.index("--out") + 1])
        _fake_recording(out, rows=50)
        return 0

    with collect_root() as root:
        clip = root / "clip"
        clip.mkdir()
        plan = apply_long_highway(base, min_s=180.0, max_s=300.0, maps=("west_coast_usa",), highway_frac=1.0)
        tags = json.loads(write_tags(clip, plan, {"quality": "train", "reasons": []}).read_text(encoding="utf-8"))
        assert tags["long_highway_planned"] is True
        assert "long_highway" in tags["tags"]
        session = run_clips(
            root, clips=4, seed=5, maps=("west_coast_usa",), max_traffic=2, runner=runner,
            min_s=30.0, max_s=60.0, long_clips=2, long_min_s=180.0, long_max_s=240.0,
        )
        manifest = json.loads((session / "session.json").read_text(encoding="utf-8"))
        assert manifest["long_min_s"] == 180.0 and manifest["long_max_s"] == 240.0, manifest
        assert manifest["long_clips"] == 2
    print("boundaries: N==clips and MIN==MAX ok; MIN>MAX refused; duration/index/tags/manifest guarded")


def check_collect_clips_main_passes_long_clip_kwargs() -> None:
    """collect_clips.main() forwards --long-clips and --long-clip-s to run_clips (M31-M33)."""
    from scripts.gvd_test_env import collect_root

    cli = _cli()
    captured: dict = {}
    with collect_root() as root:
        orig = cli.run_clips

        def run_with_fake_collector(out, **kwargs):
            captured.update(kwargs)
            def runner(argv: list[str]) -> int:
                path = Path(argv[argv.index("--out") + 1])
                _fake_recording(path, rows=50)
                return 0
            kwargs["runner"] = runner
            return orig(out, **kwargs)

        cli.run_clips = run_with_fake_collector
        try:
            rc = cli.main([
                "--out", str(root), "--clips", "4", "--maps", "west_coast_usa",
                "--long-clips", "2", "--long-clip-s", "180,240",
            ])
            assert rc == 0, rc
        finally:
            cli.run_clips = orig
        assert captured["long_clips"] == 2, captured
        assert captured["long_min_s"] == 180.0, captured
        assert captured["long_max_s"] == 240.0, captured
        assert abs(float(captured["long_highway_frac"]) - 0.7) < 1e-12, captured
        sessions = sorted(p for p in root.iterdir() if p.is_dir() and (p / "session.json").is_file())
        manifest = json.loads((sessions[0] / "session.json").read_text(encoding="utf-8"))
        assert manifest["long_clips"] == 2
        assert manifest["long_min_s"] == 180.0 and manifest["long_max_s"] == 240.0
        long_n = sum(
            1 for row in manifest["results"]
            if json.loads((sessions[0] / row["dir"] / "clip_tags.json").read_text(encoding="utf-8")).get("long")
        )
        assert long_n == 2, manifest["results"]
    print("collect_clips.main: forwards long_clips and long_clip_s to run_clips")



def check_long_clip_default_duration_bounds() -> None:
    """Default long clips are 10-15 min (600-900 s); durations are seeded and not pinned."""
    from python.collect.clip_mode import apply_long_highway, plan_clip

    cli = _cli()
    args = cli.parse_args(["--out", "/tmp/x"])
    assert args.long_clip_s == (600.0, 900.0)
    base = plan_clip(3, seed=11, maps=MAPS, max_traffic=0)
    plan = apply_long_highway(base, min_s=600.0, max_s=900.0, maps=MAPS, highway_frac=1.0)
    assert 600.0 <= plan.duration_s <= 900.0
    durs = [
        apply_long_highway(plan_clip(i, seed=7, maps=("west_coast_usa",), max_traffic=0),
                           min_s=600.0, max_s=900.0, maps=("west_coast_usa",), highway_frac=1.0).duration_s
        for i in range(40)
    ]
    assert any(d > 600.5 for d in durs) and any(d < 899.5 for d in durs), durs
    assert len(set(durs)) > 1, durs
    p0 = apply_long_highway(plan_clip(0, seed=7, maps=("west_coast_usa",), max_traffic=0),
                            min_s=600.0, max_s=900.0, maps=("west_coast_usa",), highway_frac=1.0)
    assert p0.duration_s == apply_long_highway(plan_clip(0, seed=7, maps=("west_coast_usa",), max_traffic=0),
                                               min_s=600.0, max_s=900.0, maps=("west_coast_usa",), highway_frac=1.0).duration_s
    print("default duration: 600-900 s, seeded, not pinned to MIN or MAX")


def check_long_clip_highway_frac_mix() -> None:
    """--long-clip-highway-frac (default 0.7) is a seeded mix, not a hard map skip."""
    from python.collect.clip_mode import apply_long_highway, long_clip_indices, plan_clip

    default = ("west_coast_usa", "italy", "east_coast_usa")
    n_hw = n_var = 0
    for seed in range(40):
        for i in long_clip_indices(60, 8, seed):
            plan = apply_long_highway(
                plan_clip(i, seed=seed, maps=default, max_traffic=0),
                min_s=600.0, max_s=900.0, maps=default, highway_frac=0.7,
            )
            assert plan.long is True
            assert 600.0 <= plan.duration_s <= 900.0
            if plan.long_highway:
                assert plan.long_route == "highway_loop"
                assert plan.map in ("west_coast_usa", "italy")
                n_hw += 1
            else:
                assert plan.long_route == "varied"
                assert plan.map in default
                assert "long_highway" not in plan.tags
                n_var += 1
    total = n_hw + n_var
    frac = n_hw / total
    assert total == 40 * 8, total
    assert 0.60 <= frac <= 0.80, (frac, n_hw, n_var)
    # Same seed+index is deterministic.
    a = apply_long_highway(plan_clip(3, seed=9, maps=default, max_traffic=0),
                           min_s=600.0, max_s=900.0, maps=default, highway_frac=0.7)
    b = apply_long_highway(plan_clip(3, seed=9, maps=default, max_traffic=0),
                           min_s=600.0, max_s=900.0, maps=default, highway_frac=0.7)
    assert a.long_route == b.long_route and a.map == b.map and a.duration_s == b.duration_s
    # frac=0: all varied, still on the pool, including maps with a highway.
    varied = [
        apply_long_highway(plan_clip(i, seed=1, maps=default, max_traffic=0),
                           min_s=600.0, max_s=900.0, maps=default, highway_frac=0.0)
        for i in range(30)
    ]
    assert all(p.long and not p.long_highway and p.long_route == "varied" for p in varied)
    print(f"highway_frac mix: {n_hw}/{total}={frac:.3f} highway-first under default 0.7")


def check_long_clip_routes_direction_and_maps() -> None:
    """Highway-first: spawn a highway end-to-end loop, alternate direction, vary the route.

    Dirt and highway legs are only planned when the map actually has them. A map
    with neither is still a long clip, on a varied seeded route.
    """
    from python.collect.clip_mode import (
        apply_long_highway,
        clip_argv,
        map_has_dirt_roads,
        map_has_highway_pool,
        plan_clip,
    )
    from pathlib import Path as _P

    def apply(i, seed, maps, **kw):
        return apply_long_highway(
            plan_clip(i, seed=seed, maps=maps, max_traffic=0),
            min_s=600.0, max_s=900.0, maps=maps, highway_frac=kw.get("highway_frac", 1.0),
        )

    # italy: highway + city; its road dump has no unpaved material, so no dirt leg is claimed.
    italy = [apply(i, 11, ("italy",)) for i in range(6)]
    assert all(p.long_highway and p.long_route == "highway_loop" for p in italy)
    assert all(p.route_legs[0] == "highway" and p.route_legs[-1] == "highway" for p in italy)
    assert all("city" in p.route_legs for p in italy)
    assert all("dirt" not in p.route_legs and p.route_summary["dirt_m"] == 0.0 for p in italy)
    assert not map_has_dirt_roads("italy")
    # The requested direction alternates; the route's own direction is what its start bearing is.
    assert [p.route_summary["requested_direction"] for p in italy] == [i % 2 for i in range(6)]
    assert set(p.long_direction for p in italy) == {0, 1}, [p.long_direction for p in italy]
    assert all(p.long_direction == p.route_summary["direction"] for p in italy)
    # Route / spawn variety across clips; same index is stable.
    seeds = [p.spawn_seed for p in italy]
    assert len(set(seeds)) > 1, seeds
    assert italy[0].spawn_seed == apply(0, 11, ("italy",)).spawn_seed
    argv = clip_argv(italy[0], _P("/tmp/x/clip"), route_plan="/tmp/r.json")
    assert argv[argv.index("--spawn-kind") + 1] == "highway"
    seg = float(argv[argv.index("--segment-min") + 1])
    assert seg * 60.0 >= italy[0].duration_s, (seg, italy[0].duration_s)

    # west_coast_usa: its dump has gravel/dirt road material, so the dirt leg is real when the return uses it.
    assert map_has_dirt_roads("west_coast_usa")
    wcs = [apply(i, 2, ("west_coast_usa",)) for i in range(6)]
    assert all(p.long_route == "highway_loop" and "city" in p.route_legs for p in wcs)
    assert all(("dirt" in p.route_legs) == (p.route_summary["dirt_m"] > 0.0) for p in wcs)
    assert any("dirt" in p.route_legs for p in wcs)

    # east_coast_usa: no highway (so span roaming), dirt by road material. Still a long clip, no long_highway.
    assert not map_has_highway_pool("east_coast_usa")
    assert map_has_dirt_roads("east_coast_usa")
    ec = [apply(i, 5, ("east_coast_usa",), highway_frac=1.0) for i in range(12)]
    assert all(p.long and not p.long_highway and p.long_route == "varied" and p.map == "east_coast_usa" for p in ec)
    assert all(p.route_legs == ["varied"] and p.route_summary is None and p.long_direction is None for p in ec)
    assert len({p.spawn_seed for p in ec}) > 1
    argv = clip_argv(ec[0], _P("/tmp/x/clip"))
    assert argv[argv.index("--route") + 1] == "span" and "--route-plan" not in argv
    print("routes: direction alternates, spawn/route vary, no-highway maps still counted")


def check_default_pool_lands_on_the_no_highway_map() -> None:
    """With frac < 1 some long clips land on east_coast_usa (varied, free roaming)."""
    from python.collect.clip_mode import apply_long_highway, plan_clip

    default = ("west_coast_usa", "italy", "east_coast_usa")
    landed = {
        apply_long_highway(plan_clip(i, seed=s, maps=default, max_traffic=0),
                           min_s=600.0, max_s=900.0, maps=default, highway_frac=0.7).map
        for s in range(20) for i in range(8)
    }
    assert "east_coast_usa" in landed, landed
    print("default pool: long clips do land on east_coast_usa when frac < 1")


def check_real_map_plan_resolves_to_its_route() -> None:
    """On a real dump a highway-loop plan resolves to the same route the collector is given."""
    from python.collect.clip_mode import apply_long_highway, plan_clip, plan_route

    base = plan_clip(1, seed=3, maps=("west_coast_usa",), max_traffic=0)
    plan = apply_long_highway(base, min_s=800.0, max_s=800.0, maps=("west_coast_usa",), highway_frac=1.0)
    route = plan_route(plan)
    assert plan.long_route == "highway_loop" and route is not None
    assert route.map == "west_coast_usa" and route.direction == plan.long_direction
    assert round(route.highway_m, 1) == plan.route_summary["highway_m"] and list(route.legs) == plan.route_legs
    assert plan_route(base) is None
    print("real map: the plan resolves to its own route")


def _pos_along(route, frac):
    """pos_at(i) for a recording of 100 rows that drives the first ``frac`` of the route, then stops."""
    def pos_at(i):
        s = route.total_m * min(1.0, frac) * min(1.0, i / 99.0)
        k = max(j for j in range(len(route.cum)) if route.cum[j] <= s)
        k = min(k, len(route.points) - 2)
        t = (s - route.cum[k]) / max(1e-9, route.cum[k + 1] - route.cum[k])
        a, b = route.points[k], route.points[k + 1]
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, 0.0)
    return pos_at


def check_highway_loop_clips_are_driven_and_measured() -> None:
    """The route is written, handed to the collector, and scored from the recorded GT pose."""
    from python.collect import long_route as lr
    from python.collect.clip_mode import long_clip_indices, plan_route, run_clips
    from scripts.gvd_test_env import collect_root

    drives = [1.0, 0.2, 0.6]  # one full loop, one stopped early on the highway, one past the far end
    seen: list[dict] = []

    def runner(argv: list[str]) -> int:
        out = Path(argv[argv.index("--out") + 1])
        if "--route-plan" not in argv:
            _fake_recording(out, rows=50)
            return 0
        path = Path(argv[argv.index("--route-plan") + 1])
        route = lr.Route.from_json(json.loads(path.read_text(encoding="utf-8")))
        frac = drives[len([x for x in seen if x["route"]])]
        seen.append({"route": True, "path": path, "argv": argv, "total": route.total_m})
        _fake_recording(out, rows=100, pos_at=_pos_along(route, frac))
        (out / "route_driver.json").write_text(json.dumps({"state": "following", "lines_sent": 7}), encoding="utf-8")
        return 0

    with collect_root() as root:
        session = run_clips(
            root, clips=9, seed=4, maps=("west_coast_usa", "italy"), max_traffic=2, runner=runner,
            min_s=30.0, max_s=60.0, long_clips=3, long_min_s=600.0, long_max_s=900.0, long_highway_frac=1.0,
        )
        manifest = json.loads((session / "session.json").read_text(encoding="utf-8"))
        want = long_clip_indices(9, 3, seed=4)
        assert len(seen) == 3, seen
        assert manifest["long_clips_planned"] == 3 and "long_clips_by_route" not in manifest, manifest
        assert manifest["long_clips_planned_by_route"] == {"highway_loop": 3, "varied": 0}, manifest["long_clips_planned_by_route"]
        measured = []
        verdicts = []
        for i, row in enumerate(manifest["results"]):
            tags = json.loads((session / row["dir"] / "clip_tags.json").read_text(encoding="utf-8"))
            if i not in want:
                assert "route_measure" not in tags and "route_measure" not in row, (i, tags.keys())
                assert "--route-plan" not in tags["argv"]
                continue
            k = len(measured)
            m = tags["route_measure"]
            measured.append(m)
            assert row["route_measure"] == m and row["long_route_planned"] == "highway_loop"
            assert tags["route_verdict"] == row["route_verdict"]
            verdicts.append(tags["route_verdict"])
            path = Path(tags["route_plan_file"])
            assert path.parent == session / "_routes" and path.name == f"{row['dir']}.json" and path.is_file()
            assert tags["argv"][tags["argv"].index("--route-plan") + 1] == str(path)
            assert "--no-excursions" in tags["argv"] and "--route" not in tags["argv"]
            assert tags["route_driver"] == {"state": "following", "lines_sent": 7}
            # What the plan said and what the car did are separate numbers.
            assert tags["route_summary_planned"]["highway_m"] == m["highway_m"] and tags["route_summary_planned"]["total_m"] == m["planned_m"]
            if drives[k] == 1.0:
                assert m["far_end_reached"] and m["loop_closed"] and m["highway_fraction"] >= 0.99, m
                assert m["return_fraction"] >= 0.95 and m["driven_m"] >= 0.97 * m["planned_m"], m
            elif drives[k] == 0.2:
                assert not m["far_end_reached"] and not m["loop_closed"], m
                assert 0.2 * m["planned_m"] - 60 <= m["driven_m"] <= 0.2 * m["planned_m"] + 60, m
                assert 0.0 < m["highway_fraction"] < 1.0 and m["return_fraction"] == 0.0, m
            else:
                assert m["far_end_reached"] and not m["loop_closed"], m
                assert 0.0 < m["return_fraction"] < 1.0, m
        # Delivered is counted after the run, from the GT measure: only the full drive counts.
        assert verdicts == [{1.0: "completed", 0.2: "partial"}.get(d, "far_end_only") for d in drives[:3]], (verdicts, drives)
        assert manifest["long_clip_verdicts"] == {"completed": 1, "partial": 1, "far_end_only": 1}, manifest["long_clip_verdicts"]
        assert manifest["long_clips_delivered"] == 1, manifest["long_clips_delivered"]
        assert manifest["long_clips_delivered_by_route"] == {"highway_loop": 1, "varied": 0}
        tot = manifest["long_route_totals"]
        assert tot["route_clips"] == 3 and tot["far_end_reached"] == 2 and tot["loop_closed"] == 1, tot
        assert abs(tot["driven_m"] - sum(m["driven_m"] for m in measured)) < 0.2, (tot, measured)
        assert abs(tot["mean_highway_fraction"] - sum(m["highway_fraction"] for m in measured) / 3) < 1e-3, tot
        assert tot["dirt_driven_m"] == round(sum(m["dirt_driven_m"] for m in measured), 1)
        # The route file is the route the plan resolves to (same plan, same route).
        plans_dirs = [r["dir"] for i, r in enumerate(manifest["results"]) if i in want]
        for d in plans_dirs:
            rt = lr.Route.from_json(json.loads((session / "_routes" / f"{d}.json").read_text(encoding="utf-8")))
            assert rt.points[0] and rt.far_end_index > 0 and rt.highway_m >= lr.MIN_HIGHWAY_M
    print("highway loop: route file + --route-plan, GT-pose measure per clip and in session.json (full / early stop / far end only)")


def check_requested_direction_alternates_by_long_clip_order() -> None:
    """Direction alternates in long-clip order (not by raw clip index); the seed differs per clip; a failed plan falls back."""
    from python.collect.clip_mode import long_clip_indices, run_clips
    from scripts.gvd_test_env import collect_root

    def runner(argv: list[str]) -> int:
        _fake_recording(Path(argv[argv.index("--out") + 1]), rows=50)
        return 0

    want = long_clip_indices(24, 6, seed=2)
    assert [i % 2 for i in want] != [k % 2 for k in range(6)], want  # raw index parity is not the alternation
    with collect_root() as root:
        session = run_clips(
            root, clips=24, seed=2, maps=("west_coast_usa",), max_traffic=1, runner=runner,
            min_s=30.0, max_s=60.0, long_clips=6, long_min_s=600.0, long_max_s=900.0, long_highway_frac=1.0,
        )
        manifest = json.loads((session / "session.json").read_text(encoding="utf-8"))
        tags = [json.loads((session / manifest["results"][i]["dir"] / "clip_tags.json").read_text(encoding="utf-8")) for i in want]
    assert [t["route_summary_planned"]["requested_direction"] for t in tags] == [k % 2 for k in range(6)], tags
    assert [t["long_direction_planned"] for t in tags] == [k % 2 for k in range(6)]  # the stub network honours the request
    assert len({t["spawn_seed"] for t in tags}) == 6, [t["spawn_seed"] for t in tags]
    assert all(t["long_route_planned"] == "highway_loop" and t["route_legs_planned"][0] == "highway" for t in tags)

    # No route can be planned (here forced): the clip is a varied span clip, says so, and is counted as varied.
    logs: list[str] = []
    with fast_routes(fail=True), collect_root() as root:
        session = run_clips(
            root, clips=6, seed=2, maps=("west_coast_usa",), max_traffic=1, runner=runner,
            min_s=30.0, max_s=60.0, long_clips=2, long_min_s=600.0, long_max_s=900.0, long_highway_frac=1.0,
            log=logs.append,
        )
        manifest = json.loads((session / "session.json").read_text(encoding="utf-8"))
        assert manifest["long_clips_planned_by_route"] == {"highway_loop": 0, "varied": 2}, manifest["long_clips_planned_by_route"]
        assert manifest["long_clip_verdicts"] == {"drive_around": 2} and manifest["long_clips_delivered"] == 2, manifest
        assert manifest["long_route_totals"]["route_clips"] == 0
        for i in long_clip_indices(6, 2, seed=2):
            t = json.loads((session / manifest["results"][i]["dir"] / "clip_tags.json").read_text(encoding="utf-8"))
            assert t["long"] is True and t["long_highway_planned"] is False and t["long_route_planned"] == "varied"
            assert t["route_legs_planned"] == ["varied"] and t["route_summary_planned"] is None and t["long_direction_planned"] is None
            assert "--route" in t["argv"] and "--route-plan" not in t["argv"] and "route_measure" not in t
            assert any("no highway end-to-end route" in n for n in t["plan_notes"]), t["plan_notes"]
        assert not (session / "_routes").exists()
    print("direction alternates in long-clip order, seed differs per clip, failed plan falls back to varied span")


def check_long_plan_guards_and_provenance() -> None:
    """Empty pool and a bad fraction are refused; notes/tags carry 'long'; the spawn seed is re-rolled; argv gets exact segment minutes."""
    import math

    from python.collect.clip_mode import apply_long_highway, clip_argv, plan_clip

    base = plan_clip(2, seed=5, maps=("west_coast_usa",), max_traffic=1)
    for bad_maps in ((), ("",)):
        try:
            apply_long_highway(base, min_s=600.0, max_s=900.0, maps=bad_maps)
        except ValueError as exc:
            assert "at least one map" in str(exc), exc
        else:
            raise AssertionError(f"a long clip was made from an empty pool {bad_maps!r}")
    for bad in (-0.01, 1.01, 7.0):
        try:
            apply_long_highway(base, min_s=600.0, max_s=900.0, maps=("west_coast_usa",), highway_frac=bad)
        except ValueError as exc:
            assert "long_highway_frac" in str(exc), exc
        else:
            raise AssertionError(f"accepted highway_frac {bad}")
    for ok in (0.0, 1.0):
        apply_long_highway(base, min_s=600.0, max_s=900.0, maps=("west_coast_usa",), highway_frac=ok)

    # 'long' is in both the notes and the tags, loop or varied; 'long_highway' only on the loop.
    loop = apply_long_highway(base, min_s=600.0, max_s=900.0, maps=("west_coast_usa",), highway_frac=1.0)
    varied = apply_long_highway(base, min_s=600.0, max_s=900.0, maps=("west_coast_usa",), highway_frac=0.0)
    assert loop.long_route == "highway_loop" and varied.long_route == "varied"
    for p in (loop, varied):
        assert "long" in p.notes and "long" in p.tags, p
    assert "long_highway" in loop.notes and "long_highway" in loop.tags
    assert "long_highway" not in varied.notes and "long_highway" not in varied.tags
    # The spawn seed is a fresh draw for the long clip, not the short plan's.
    seeds = [(plan_clip(i, seed=5, maps=("west_coast_usa",), max_traffic=1), i) for i in range(6)]
    moved = [apply_long_highway(b, min_s=600.0, max_s=900.0, maps=("west_coast_usa",), highway_frac=0.0).spawn_seed != b.spawn_seed for b, _ in seeds]
    assert all(moved), moved

    # --segment-min is the clip's whole minutes, rounded up (a 15 min clip is not cut at 12).
    for secs, minutes in ((200.0, 4.0), (840.0, 14.0), (900.0, 15.0), (901.0, 16.0), (600.0, 10.0)):
        for frac in (0.0, 1.0):
            plan = apply_long_highway(base, min_s=secs, max_s=secs, maps=("west_coast_usa",), highway_frac=frac)
            argv = clip_argv(plan, Path("/tmp/x/clip"), route_plan="/tmp/r.json")
            assert float(argv[argv.index("--segment-min") + 1]) == minutes, (secs, frac, argv)
            assert float(argv[argv.index("--segment-min") + 1]) == math.ceil(plan.duration_s / 60.0)
    print("long plan: empty pool and bad fraction refused, 'long' in notes/tags, spawn seed re-rolled, segment-min is ceil(minutes)")


def check_varied_spawn_kind_and_qc_skip() -> None:
    """A varied long clip spawns on a city road, else a highway, else anywhere; QC does not fail it for an unobserved edge case."""
    from python.collect import clip_mode
    from python.collect.clip_mode import apply_long_highway, plan_clip, quality_check
    from scripts.gvd_test_env import collect_root

    base = plan_clip(0, seed=3, maps=("small_island",), max_traffic=0)
    cases = (
        (True, True, "city"),
        (True, False, "city"),
        (False, True, "highway"),
        (False, False, None),
    )
    city0, hwy0 = clip_mode.map_has_city_pool, clip_mode.map_has_highway_pool
    try:
        for city, hwy, kind in cases:
            clip_mode.map_has_city_pool = lambda m, _c=city: _c
            clip_mode.map_has_highway_pool = lambda m, _h=hwy: _h
            plan = apply_long_highway(base, min_s=600.0, max_s=900.0, maps=("small_island",), highway_frac=0.0)
            assert plan.long_route == "varied" and plan.spawn_kind == kind, (city, hwy, plan.spawn_kind)
    finally:
        clip_mode.map_has_city_pool, clip_mode.map_has_highway_pool = city0, hwy0

    # QC: a varied long clip (not long_highway) that inherited a cut_in recipe is not failed for the cut-in.
    cut = plan_clip(0, seed=1, maps=("west_coast_usa",), max_traffic=4, edge_cases=("cut_in",), min_s=4.0, max_s=4.0)
    plan = apply_long_highway(cut, min_s=4.0, max_s=4.0, maps=("west_coast_usa",), highway_frac=0.0)
    assert plan.long and not plan.long_highway
    with collect_root() as root:
        d = root / "clip"
        _fake_recording(d, rows=50, spawn_type="city")
        qc = quality_check(d, plan, exit_code=0)
        assert qc["quality"] == "train" and qc["edge_case_observed"] is None, qc
        assert qc["edge_case_note"].startswith("long"), qc
        # Same recording against the short cut_in plan is judged on the cut-in (so the skip above is doing the work).
        short = quality_check(d, cut, exit_code=0)
        assert short["edge_case_observed"] is False, short
    print("varied long clip: spawn kind city > highway > none; QC does not judge an inherited edge case")


def check_manifest_frac_warning_and_forwarding() -> None:
    from python.collect.clip_mode import run_clips
    from scripts.gvd_test_env import collect_root

    def runner(argv: list[str]) -> int:
        _fake_recording(Path(argv[argv.index("--out") + 1]), rows=50)
        return 0

    seen = {}
    for frac in (0.0, 1.0):
        logs: list[str] = []
        with collect_root() as root:
            session = run_clips(
                root, clips=8, seed=6, maps=("west_coast_usa", "east_coast_usa"), max_traffic=1, runner=runner,
                min_s=30.0, max_s=60.0, long_clips=4, long_min_s=600.0, long_max_s=900.0,
                long_highway_frac=frac, log=logs.append,
            )
            manifest = json.loads((session / "session.json").read_text(encoding="utf-8"))
        assert manifest["long_highway_frac"] == frac, manifest["long_highway_frac"]
        seen[frac] = manifest["long_clips_planned_by_route"]
        assert not any("no highway map" in x for x in logs), logs  # the pool has a highway map
    assert seen[0.0] == {"highway_loop": 0, "varied": 4}, seen
    assert seen[1.0] == {"highway_loop": 4, "varied": 0}, seen

    # The no-highway warning only fires when long clips were asked for.
    for n, want in ((0, False), (2, True)):
        logs = []
        with collect_root() as root:
            run_clips(
                root, clips=4, seed=6, maps=("east_coast_usa",), max_traffic=1, runner=runner,
                min_s=30.0, max_s=60.0, long_clips=n, long_min_s=600.0, long_max_s=900.0, log=logs.append,
            )
        assert any("no highway map" in x for x in logs) == want, (n, logs)
    print("session.json keeps long_highway_frac, run_clips forwards it, the no-highway warning only fires when long clips were asked for")


def check_route_direction_is_the_routes_own_and_replanned_with_the_request() -> None:
    """A map that cannot give the requested direction: the plan records the route's own direction, and plan_route re-plans with the request."""
    from python.collect import clip_mode, long_route as lr
    from python.collect.clip_mode import apply_long_highway, plan_clip, plan_route
    from test_long_route import loop_network

    graph = lr.build_graph(loop_network(dirt=True))  # one-way east: a westbound request is answered eastbound
    calls: list[tuple] = []
    orig = clip_mode._route_cached

    def eastbound_only(map_name, seed, direction, duration_s):
        calls.append((map_name, int(seed), int(direction), float(duration_s)))
        return lr.plan_highway_route(graph, map_name=map_name, seed=int(seed), direction=int(direction), duration_s=900.0)

    clip_mode._route_cached = eastbound_only
    try:
        base = plan_clip(1, seed=3, maps=("west_coast_usa",), max_traffic=0)
        plan = apply_long_highway(base, min_s=600.0, max_s=900.0, maps=("west_coast_usa",), highway_frac=1.0, ordinal=1)
        assert plan.long_route == "highway_loop"
        assert plan.route_summary["requested_direction"] == 1 and plan.route_summary["direction"] == 0, plan.route_summary
        assert plan.long_direction == 0, "the tag is the route's own direction, not the request"
        n = len(calls)
        route = plan_route(plan)
        assert calls[n] == calls[0] == ("west_coast_usa", plan.spawn_seed, 1, plan.duration_s), calls
        assert route.direction == 0 and route.requested_direction == 1
    finally:
        clip_mode._route_cached = orig
    print("direction: the tag is the route's own direction; plan_route asks the planner for the original request")


def check_disk_guards_unchanged_for_long_clips() -> None:
    """20 GiB free-space floor and 3.5 GiB per-file cap are unchanged at the 43 GiB/900 s ceiling."""
    from python.collect.clip_mode import LONG_CLIP_DISK_GIB_PER_900S
    from python.data.data_root import MIN_FREE_BYTES
    import importlib.util
    from pathlib import Path as _P

    assert MIN_FREE_BYTES == 20 * 1024 * 1024 * 1024
    spec = importlib.util.spec_from_file_location("collect_scene_lanes_disk", _P(__file__).resolve().parents[1] / "scripts" / "collect_scene_lanes.py")
    # Don't exec the collector (it is heavy). Parse the default from the source or clip CLI.
    # The collector flag default is the documented 3.5 GiB cap.
    from scripts import test_control_log  # noqa: F401 — exists in repo
    text = (_P(__file__).resolve().parents[1] / "scripts" / "collect_scene_lanes.py").read_text(encoding="utf-8")
    assert "default=3.5" in text
    assert abs(float(LONG_CLIP_DISK_GIB_PER_900S) - 43.0) < 1e-9
    # 8 long clips ≈ 344 GiB at the ceiling; the floor still stops at 20 GiB free.
    print("disk guards: 20 GiB floor and 3.5 GiB/file unchanged; ceiling 43 GiB per 900 s clip")



def check_verdicts_follow_exit_quality_and_measure() -> None:
    """The delivered count is a measured verdict, not a plan label."""
    from python.collect import long_route as lr
    from python.collect.clip_mode import MIN_DRIVEN_M, VERDICT_DELIVERED, route_verdict

    ok = {"clip_class": "train"}
    full = {"far_end_reached": True, "loop_closed": True, "driven_m": 9000.0}
    far = {"far_end_reached": True, "loop_closed": False, "driven_m": 5000.0}
    short = {"far_end_reached": False, "loop_closed": False, "driven_m": MIN_DRIVEN_M}
    none = {"far_end_reached": False, "loop_closed": False, "driven_m": MIN_DRIVEN_M - 1.0}
    cases = [
        (0, ok, full, "completed"), (0, ok, far, "far_end_only"), (0, ok, short, "partial"), (0, ok, none, "not_driven"),
        (0, ok, {"far_end_reached": False, "loop_closed": True, "driven_m": 9000.0}, "partial"),  # both are required
        (0, ok, None, "drive_around"), (1, ok, full, "failed"), (99, ok, None, "failed"),
        (0, {"clip_class": "reject"}, full, "rejected"), (0, {"clip_class": "pretrain_only"}, full, "completed"),
    ]
    for code, qc, measure, want in cases:
        assert route_verdict(None, code, qc, measure) == want, (code, qc, measure, want)
    assert VERDICT_DELIVERED == ("completed", "drive_around")
    # End to end: a failing collector and a clip that stopped early are planned, never delivered.
    from python.collect.clip_mode import long_clip_indices, run_clips
    from scripts.gvd_test_env import collect_root

    def runner(argv: list[str]) -> int:
        out = Path(argv[argv.index("--out") + 1])
        if "--route-plan" not in argv:
            _fake_recording(out, rows=50)
            return 0
        path = Path(argv[argv.index("--route-plan") + 1])
        route = lr.Route.from_json(json.loads(path.read_text(encoding="utf-8")))
        _fake_recording(out, rows=100, pos_at=_pos_along(route, 1.0))
        seen.append(1)
        return 3 if len(seen) == 1 else 0  # the first route clip's collector fails after recording a full drive

    seen: list[int] = []
    with collect_root() as root:
        session = run_clips(
            root, clips=4, seed=4, maps=("west_coast_usa",), max_traffic=1, runner=runner,
            min_s=30.0, max_s=60.0, long_clips=2, long_min_s=600.0, long_max_s=900.0, long_highway_frac=1.0,
        )
        manifest = json.loads((session / "session.json").read_text(encoding="utf-8"))
        assert manifest["long_clips_planned"] == 2 and manifest["long_clips_planned_by_route"]["highway_loop"] == 2
        first = long_clip_indices(4, 2, seed=4)[0]
        assert manifest["results"][first]["route_verdict"] == "failed", manifest["results"][first]
        assert manifest["long_clips_delivered"] == manifest["long_clip_verdicts"].get("completed", 0) == 1, manifest
        assert manifest["long_clips_delivered_by_route"] == {"highway_loop": 1, "varied": 0}, manifest
    print("long clip verdicts: exit/quality/GT measure decide delivered; a failed collector is planned, not delivered")


def check_no_highway_pool_delivers_drive_around_long_clips() -> None:
    """A pool with no highway map still delivers all requested long clips as varied routes."""
    from python.collect.clip_mode import long_clip_indices, run_clips
    from scripts.gvd_test_env import collect_root

    logs: list[str] = []
    no_hwy = ("east_coast_usa", "small_island")

    def runner(argv: list[str]) -> int:
        out = Path(argv[argv.index("--out") + 1])
        _fake_recording(out, rows=50)
        return 0

    with collect_root() as root:
        session = run_clips(
            root, clips=6, seed=3, maps=no_hwy, max_traffic=2, runner=runner,
            min_s=30.0, max_s=60.0, long_clips=3, long_min_s=600.0, long_max_s=900.0,
            long_highway_frac=1.0, log=logs.append,
        )
        want = set(long_clip_indices(6, 3, seed=3))
        manifest = json.loads((session / "session.json").read_text(encoding="utf-8"))
        assert manifest["long_clips"] == 3
        assert manifest["long_clips_planned"] == 3 and manifest["long_clips_delivered"] == 3, manifest
        assert manifest["long_clip_verdicts"] == {"drive_around": 3}, manifest["long_clip_verdicts"]
        by_route = manifest["long_clips_planned_by_route"]
        assert manifest["long_clips_delivered_by_route"] == by_route, manifest
        assert by_route.get("varied") == 3 and by_route.get("highway_loop", 0) == 0, by_route
        tagged = 0
        for i, row in enumerate(manifest["results"]):
            tags = json.loads((session / row["dir"] / "clip_tags.json").read_text(encoding="utf-8"))
            if i in want:
                assert tags["long"] is True
                assert tags["long_highway_planned"] is False
                assert tags["long_route_planned"] == "varied"
                assert 600.0 <= tags["duration_s"] <= 900.0
                tagged += 1
            else:
                assert tags.get("long") is not True
        assert tagged == 3
        text = "\n".join(str(x) for x in logs)
        assert "no highway" in text.lower() or "falling back" in text.lower(), logs
    print("no-highway pool: all long clips delivered as varied drive-around; warning logged; counts in session.json")


def check_keep_planned_highway_map_and_remap_varies() -> None:
    """N05: do not remap a plan already on a highway map. N06: a remap is not always the first highway map."""
    from python.collect.clip_mode import apply_long_highway, plan_clip

    pool = ("west_coast_usa", "italy", "east_coast_usa")
    # Planned map already has a highway: stay on it (not always remapped).
    for seed in range(30):
        base = plan_clip(0, seed=seed, maps=("west_coast_usa",), max_traffic=0)
        assert base.map == "west_coast_usa"
        plan = apply_long_highway(base, min_s=600.0, max_s=900.0, maps=pool, highway_frac=1.0)
        assert plan.long_highway is True
        assert plan.map == "west_coast_usa", (seed, plan.map)
    # Planned map has no highway: remap, but not always to pool[0] (west_coast_usa).
    remapped = set()
    for seed in range(40):
        base = plan_clip(0, seed=seed, maps=("east_coast_usa",), max_traffic=0)
        plan = apply_long_highway(base, min_s=600.0, max_s=900.0, maps=pool, highway_frac=1.0)
        assert plan.long_highway is True
        assert plan.map in ("west_coast_usa", "italy"), plan.map
        remapped.add(plan.map)
    assert remapped == {"west_coast_usa", "italy"}, remapped
    print("remap: keep a planned highway map; otherwise vary, not always the first highway map")


def check_unreadable_road_dump_is_not_a_highway_map() -> None:
    """N10: a missing or unreadable road dump is not a highway map."""
    import gzip
    from python.collect import clip_mode

    clip_mode.map_has_highway_pool.cache_clear()
    assert clip_mode.map_has_highway_pool("map_that_has_no_dump_zzz") is False

    from scripts.gvd_test_env import collect_root

    real = clip_mode.spawn_roads_path
    try:
        with collect_root() as root:
            bad = root / "bogus.json.gz"
            bad.write_bytes(b"this is not gzip")
            clip_mode.spawn_roads_path = lambda name: bad if name == "bogus_unreadable_map" else real(name)
            clip_mode.map_has_highway_pool.cache_clear()
            assert clip_mode.map_has_highway_pool("bogus_unreadable_map") is False
            # Also a readable gzip with no JSON road network is not a highway map,
            # but the mutant is the exception path: a truncated gzip must stay False.
            trunc = root / "trunc.json.gz"
            with gzip.open(trunc, "wb") as fh:
                fh.write(b"{")
            clip_mode.spawn_roads_path = lambda name: trunc if name == "bogus_truncated_map" else real(name)
            clip_mode.map_has_highway_pool.cache_clear()
            assert clip_mode.map_has_highway_pool("bogus_truncated_map") is False
    finally:
        clip_mode.spawn_roads_path = real
        clip_mode.map_has_highway_pool.cache_clear()
    print("unreadable or missing road dump is not a highway map")


def check_session_id_names_this_launch() -> None:
    """--session-id names the folder and is written to session.json (rec_loop matches on it)."""
    from scripts.gvd_test_env import collect_root

    cli = _cli()
    with collect_root() as root:
        def runner(argv: list[str]) -> int:
            _fake_recording(Path(argv[argv.index("--out") + 1]), rows=50)
            return 0

        from python.collect.clip_mode import run_clips

        session = run_clips(root, clips=1, seed=1, maps=("west_coast_usa",), max_traffic=0,
                            runner=runner, session_id="LAUNCH_7")
        assert session.name == "clips_LAUNCH_7", session.name
        manifest = json.loads((session / "session.json").read_text(encoding="utf-8"))
        assert manifest["session_id"] == "LAUNCH_7", manifest
        # No id: the stamp names it and the manifest still says which id it is.
        auto = run_clips(root, clips=1, seed=1, maps=("west_coast_usa",), max_traffic=0, runner=runner)
        m2 = json.loads((auto / "session.json").read_text(encoding="utf-8"))
        assert m2["session_id"] == auto.name[len("clips_"):], (m2, auto.name)
        # The CLI forwards it.
        args = cli.parse_args(["--out", str(root), "--session-id", "CLI_9"])
        assert args.session_id == "CLI_9"
        assert cli.parse_args(["--out", str(root)]).session_id is None
        captured: dict = {}
        orig = cli.run_clips

        def spy(out, **kw):
            captured.update(kw)
            kw["runner"] = runner
            return orig(out, **kw)

        cli.run_clips = spy
        try:
            assert cli.main(["--out", str(root), "--clips", "1", "--maps", "west_coast_usa", "--session-id", "CLI_9"]) == 0
        finally:
            cli.run_clips = orig
        assert captured["session_id"] == "CLI_9", captured
        assert (root / "clips_CLI_9" / "session.json").is_file()
    print("session id: names the folder, lands in session.json, CLI forwards it")


def main() -> None:
    with fast_routes():
        _checks()
    with real_routes():
        check_long_clip_routes_direction_and_maps()
        check_real_map_plan_resolves_to_its_route()
    print("test_rec_long_clips: OK")


def _checks() -> None:
    check_default_off_unchanged()
    check_cli_long_clip_flags()
    check_long_clip_indices_spread_and_seeded()
    check_long_plan_duration_spawn_speed_and_tags()
    check_batch_mixes_long_clips()
    check_long_clip_early_end_kept_like_normal()
    check_long_dirt_recipe_does_not_fail_highway_footage()
    check_long_clips_only_on_highway_maps()
    check_long_clip_boundaries_and_tags()
    check_collect_clips_main_passes_long_clip_kwargs()
    check_long_clip_default_duration_bounds()
    check_long_clip_highway_frac_mix()
    check_default_pool_lands_on_the_no_highway_map()
    check_highway_loop_clips_are_driven_and_measured()
    check_requested_direction_alternates_by_long_clip_order()
    check_route_direction_is_the_routes_own_and_replanned_with_the_request()
    check_long_plan_guards_and_provenance()
    check_varied_spawn_kind_and_qc_skip()
    check_manifest_frac_warning_and_forwarding()
    check_disk_guards_unchanged_for_long_clips()
    check_verdicts_follow_exit_quality_and_measure()
    check_no_highway_pool_delivers_drive_around_long_clips()
    check_keep_planned_highway_map_and_remap_varies()
    check_unreadable_road_dump_is_not_a_highway_map()
    check_session_id_names_this_launch()


if __name__ == "__main__":
    main()
