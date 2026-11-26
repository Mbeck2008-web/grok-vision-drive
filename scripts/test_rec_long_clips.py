#!/usr/bin/env python3
"""Long highway/arterial clips mixed into a collect_clips batch.

Offline. Plans, argv, tags, and scheduling only; no BeamNG.

    py -3 scripts/test_rec_long_clips.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from test_clip_mode import MAPS, _fake_recording  # noqa: E402


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
    assert args.long_clip_s == (180.0, 300.0), args.long_clip_s

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
    assert args.long_clip_s == (180.0, 300.0)
    args = cli.parse_args(["--out", "/tmp/x", "--clips", "20", "--long-clips", "5", "--long-clip-s", "180,240"])
    assert args.long_clips == 5 and args.long_clip_s == (180.0, 240.0)
    for bad in (
        ["--long-clips", "-1"],
        ["--clips", "5", "--long-clips", "10"],
        ["--long-clip-s", "300,180"],
        ["--long-clip-s", "0,300"],
        ["--long-clip-s", "abc"],
        ["--max-s", "90"],
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
    plan = apply_long_highway(base, min_s=180.0, max_s=300.0)
    again = apply_long_highway(base, min_s=180.0, max_s=300.0)
    assert plan.duration_s == again.duration_s
    assert 180.0 <= plan.duration_s <= 300.0
    assert plan.long_highway is True
    assert plan.spawn_kind == "highway"
    assert plan.speed_min == 20.0 and plan.speed_max == 30.0
    assert "long_highway" in plan.tags
    argv = clip_argv(plan, Path("/tmp/x/clip"))
    assert float(argv[argv.index("--clip-s") + 1]) == plan.duration_s
    assert argv[argv.index("--spawn-kind") + 1] == "highway"
    assert float(argv[argv.index("--speed-min") + 1]) == 20.0
    assert float(argv[argv.index("--speed-max") + 1]) == 30.0
    # A normal plan does not pick up the highway speed clamp.
    normal_argv = clip_argv(base, Path("/tmp/x/clip"))
    assert "--speed-min" not in normal_argv and "--speed-max" not in normal_argv
    assert 30.0 <= base.duration_s <= 60.0

    with collect_root() as root:
        clip = root / "clip"
        clip.mkdir()
        tags = json.loads(write_tags(clip, plan, {"quality": "train", "reasons": []}).read_text(encoding="utf-8"))
    assert "long_highway" in tags["tags"]
    assert tags["duration_s"] == plan.duration_s
    assert tags["spawn_kind"] == "highway"
    print("long plan: 180-300 s, highway spawn, 20-30 m/s, tagged long_highway")


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
                assert argv[argv.index("--spawn-kind") + 1] == "highway"
                assert float(argv[argv.index("--speed-min") + 1]) == 20.0
                assert float(argv[argv.index("--speed-max") + 1]) == 30.0
                assert "long_highway" in tags["tags"]
                long_seen.append(i)
            else:
                assert 30.0 <= dur <= 60.0, (i, dur)
                assert "long_highway" not in tags.get("tags", [])
        assert set(long_seen) == want
    print("run_clips: 3 of 10 clips are long highway, spread by seed, rest unchanged")


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
    plan = apply_long_highway(base, min_s=4.0, max_s=4.0)
    assert plan.long_highway and plan.spawn_kind == "highway"
    with collect_root() as root:
        d = root / "clip"
        _fake_recording(d, rows=50, spawn_type="highway")
        qc = quality_check(d, plan, exit_code=0)
    assert qc["quality"] == "train", qc
    print("long clip: dirt_road edge-case recipe does not fail highway footage")


def main() -> None:
    check_default_off_unchanged()
    check_cli_long_clip_flags()
    check_long_clip_indices_spread_and_seeded()
    check_long_plan_duration_spawn_speed_and_tags()
    check_batch_mixes_long_clips()
    check_long_clip_early_end_kept_like_normal()
    check_long_dirt_recipe_does_not_fail_highway_footage()
    print("test_rec_long_clips: OK")


if __name__ == "__main__":
    main()
