#!/usr/bin/env python3
"""Edge-case clip mode: plans, new folders only, tags, quality check, lossless.

Offline. The collector is replaced by a fake runner that writes a small
recording, so no BeamNG is needed.

    py -3 scripts/test_clip_mode.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from python.sensors.cameras import CAM_IDS  # noqa: E402

MAPS = ("west_coast_usa", "italy", "east_coast_usa", "small_island")


def check_edge_case_list() -> None:
    from python.collect.clip_mode import EDGE_CASES

    assert set(EDGE_CASES) == {
        "sharp_curve", "junction", "night", "rain", "fog", "faded_paint", "dirt_road",
        "merge", "cut_in", "heavy_traffic",
    }, EDGE_CASES
    assert not any("pedestrian" in e for e in EDGE_CASES)
    print("ten edge cases, no pedestrians")


def check_plans_are_seeded_and_bounded() -> None:
    from python.collect.clip_mode import EDGE_CASES, plan_clip

    plans = [plan_clip(i, seed=7, maps=MAPS, max_traffic=8) for i in range(200)]
    again = [plan_clip(i, seed=7, maps=MAPS, max_traffic=8) for i in range(200)]
    assert [p.to_dict() for p in plans] == [p.to_dict() for p in again]
    other = [plan_clip(i, seed=8, maps=MAPS, max_traffic=8) for i in range(200)]
    assert [p.to_dict() for p in plans] != [p.to_dict() for p in other]
    assert {p.edge_case for p in plans} == set(EDGE_CASES)
    assert len({p.map for p in plans}) >= 3
    for p in plans:
        assert 30.0 <= p.duration_s <= 60.0, p
        assert 0 <= p.traffic <= 8, p
        assert p.map in MAPS
        assert p.ego_model == "sbr"
        if p.edge_case == "heavy_traffic":
            assert p.traffic == 8
        if p.edge_case == "cut_in":
            assert p.traffic >= 2
        if p.edge_case in ("heavy_traffic", "cut_in", "merge"):
            assert p.map == "west_coast_usa", p  # the spawn deck and companions only run there today
        if p.edge_case == "night":
            assert p.time_of_day_h >= 21.0 or p.time_of_day_h < 4.5, p
        else:
            assert 8.0 <= p.time_of_day_h <= 17.0, p
        assert p.weather == {"rain": "rain", "fog": "fog"}.get(p.edge_case, "clear"), p
    assert len({p.spawn_seed for p in plans}) == len(plans)
    print("plans: seeded, 30-60 s, traffic <= max, edge-case recipes hold")


def check_collector_argv() -> None:
    from python.collect.clip_mode import clip_argv, plan_clip

    p = plan_clip(3, seed=1, maps=("west_coast_usa",), max_traffic=4, edge_cases=("merge",))
    argv = clip_argv(p, Path("/tmp/x/clip"), chase=True)
    joined = " ".join(argv)
    # Smoke A: "--maps <one map>" (the rotation path), never "--map": only a segment
    # load runs the seeded deck spawn and queues companion traffic.
    assert "--map" not in argv and argv[argv.index("--maps") + 1] == "west_coast_usa", argv
    # Smoke B: the planned length reaches the collector (its --minutes floor is 60 s).
    assert float(argv[argv.index("--clip-s") + 1]) == p.duration_s, argv
    # Smoke C: clips keep full-rate depth/sem; the thinning gate is off.
    assert argv[argv.index("--gt-thin") + 1] == "off", argv
    for want in ("--out /tmp/x/clip", "--ego-model sbr", "--no-vary-env",
                 f"--traffic {p.traffic}", "--weather clear", "--spawn-kind ramp",
                 f"--spawn-seed {p.spawn_seed}", "--chase-cam", "--smoke-s 0"):
        assert want in joined, (want, joined)
    assert f"--minutes {p.duration_s / 60.0:.6f}" in joined, joined
    for banned in ("--strip", "jpeg", "jpg"):
        assert banned not in joined, banned
    # The clock reaches BeamNG as its own 0-1 time (0 noon, 0.5 midnight), so an
    # early-morning night hour (00:30) is never misread as a 0-1 fraction.
    from python.collect.auto_mode import environment_commands

    night = [plan_clip(i, seed=5, maps=("italy",), max_traffic=0, edge_cases=("night",)) for i in range(40)]
    assert any(n.time_of_day_h < 1.0 for n in night), [n.time_of_day_h for n in night]
    for n in night:
        a = clip_argv(n, Path("/tmp/x/clip"))
        tod = a[a.index("--time-of-day") + 1]
        line = environment_commands(tod, "")[0]
        t = float(line.split("time=")[1].rstrip("})"))
        assert 0.375 - 1e-4 <= t <= 0.6875 + 1e-4, (n.time_of_day_h, tod, line)
    day = plan_clip(0, seed=5, maps=("italy",), max_traffic=0, edge_cases=("rain",))
    a = clip_argv(day, Path("/tmp/x/clip"))
    t = float(environment_commands(a[a.index("--time-of-day") + 1], "")[0].split("time=")[1].rstrip("})"))
    assert t <= 0.2084 or t >= 0.8333 - 1e-4, (day.time_of_day_h, t)  # 08:00-17:00
    print("collector argv: one map, sbr, new --out, no lossy option")


def _fake_recording(out: Path, *, rows: int = 40, speed: float = 10.0, yaw_rate: float = 0.0,
                    jpg: bool = False, drop_depth: str | None = None, damage: float = 0.0,
                    traffic_n: int = 2, luma: int = 120, spawn_type: str = "city",
                    traffic_at=None, speed_at=None, impact_at=()) -> None:
    import cv2

    from python.data.raw_frames import RawFrameWriter, camera_sidecar
    from python.sensors.cameras import load_camera_config

    w = RawFrameWriter(out, camera_sidecar(load_camera_config()))
    (out / "sem_classes.json").write_text(json.dumps({"kind": "gvd_sem_classes", "classes": {"0": "unknown", "1": "MUD", "2": "STREET"}}), encoding="utf-8")
    for i in range(rows):
        frames = {}
        stamps = {}
        for cid in CAM_IDS:
            if i % 10 and cid not in ("narrow", "main", "wide"):
                continue
            frames[cid] = np.full((48, 64, 3), luma, np.uint8)
            stamps[cid] = {"wall": 1000.0 + i * 0.1, "mono_before": i * 0.1, "mono_after": i * 0.1}
            if cid != drop_depth:
                frames[f"{cid}_depth"] = np.full((48, 64), 1000, np.uint16)
                frames[f"{cid}_sem"] = np.full((48, 64), 2, np.uint8)
                stamps[f"{cid}_depth"] = stamps[cid]
                stamps[f"{cid}_sem"] = stamps[cid]
        traffic = [{"name": f"gvd_traffic_{k}", "pos": [3.0 * k, 20.0, 0.0], "dir": [0, 1, 0]} for k in range(traffic_n)]
        if traffic_at is not None:
            traffic = traffic_at(i)
        elif i % 5:
            traffic = None
        v = speed if speed_at is None else float(speed_at(i))
        tags = {"traffic": traffic, "spawn_type": spawn_type, "veh": {"wheelspeed": v}}
        if i in impact_at:
            tags["impact"] = {"kind": "contact", "speed_mps": v}
        w.write(
            frames, t=1000.0 + i * 0.1, timestamps=stamps,
            ego={"speed_mps": v, "yaw_rate": yaw_rate, "pos": [0.0, i * 1.0, 0.0], "dir": [0, 1, 0], "damage": damage},
            tech={"lane_gt_valid": [1.0, 1.0, 0.0]},
            tags=tags,
        )
    if jpg:
        cv2.imwrite(str(out / "raw" / "main" / "000" / "000999.jpg"), np.zeros((8, 8, 3), np.uint8))


def check_quality_check_marks_not_deletes() -> None:
    from python.collect.clip_mode import plan_clip, quality_check
    from scripts.gvd_test_env import collect_root

    p = plan_clip(0, seed=1, maps=MAPS, max_traffic=2, edge_cases=("rain",), min_s=4.0, max_s=4.0)
    with collect_root() as root:
        good = root / "good"
        _fake_recording(good)
        qc = quality_check(good, p, exit_code=0)
        assert qc["quality"] == "train", qc
        assert qc["edge_case_observed"] is None and "rain" in qc["edge_case_note"], qc
        cases = {
            "jpg": dict(jpg=True),
            "no_depth": dict(drop_depth="rear"),
            "parked": dict(speed=0.2),
            "damaged": dict(damage=0.5),
            "short": dict(rows=4),
        }
        for name, kw in cases.items():
            d = root / name
            _fake_recording(d, **kw)
            before = sorted(str(x) for x in d.rglob("*"))
            q = quality_check(d, p, exit_code=0)
            assert q["quality"] == "pretrain_only", (name, q)
            assert q["reasons"], (name, q)
            assert sorted(str(x) for x in d.rglob("*")) == before, name  # nothing deleted or rewritten
        d = root / "crashed"
        _fake_recording(d)
        q = quality_check(d, p, exit_code=4)
        assert q["quality"] == "pretrain_only" and any("exit" in r for r in q["reasons"]), q
    print("quality check: bad clips are marked pretrain_only, never deleted")


def check_quality_check_failed_clip_paths() -> None:
    """Signal-killed exit, the 50% span floor, undecodable frames and impact each mark a clip."""
    from python.collect.clip_mode import plan_clip, quality_check
    from scripts.gvd_test_env import collect_root

    p = plan_clip(0, seed=1, maps=MAPS, max_traffic=2, edge_cases=("rain",), min_s=4.0, max_s=4.0)
    assert p.duration_s == 4.0

    def reasons(d: Path, code: int = 0) -> list[str]:
        q = quality_check(d, p, exit_code=code)
        assert (q["quality"] == "pretrain_only") == bool(q["reasons"]), q
        return q["reasons"]

    with collect_root() as root:
        d = root / "ok"
        _fake_recording(d)
        assert reasons(d) == []
        # A collector killed by a signal exits negative (subprocess: -9, -15). Not clean.
        for code in (-9, -15):
            got = reasons(d, code)
            assert got == [f"collector exit {code}"], got
        # Span floor is half the planned length: 1.5 s of a 4 s clip fails, 2.0 s passes.
        d = root / "span15"
        _fake_recording(d, rows=16)
        got = reasons(d)
        assert got == ["recorded 1.5 s of a 4 s clip"], got
        d = root / "span20"
        _fake_recording(d, rows=21)
        assert reasons(d) == [], reasons(d)
        # A PNG that does not decode (first or last of a camera folder) marks the clip.
        for sub in ("main", "rear_depth"):
            d = root / f"bad_{sub}"
            _fake_recording(d)
            folder = sorted(p_ for p_ in (d / "raw" / sub).iterdir() if p_.is_dir())[-1]
            (folder / "999999.png").write_bytes(b"not a png")
            got = reasons(d)
            assert len(got) == 1 and got[0].startswith("undecodable frames") and sub in got[0], got
        # An impact row with zero damage still marks the clip.
        d = root / "impact"
        _fake_recording(d, impact_at={20})
        got = reasons(d)
        assert got == ["ego damaged or impact logged"], got
    print("failed-clip QC: signal kill, half-span floor, undecodable frame, impact")


def check_edge_case_false_positives() -> None:
    """A False observation must not read as True: each case here is a plain clip."""
    from python.collect.clip_mode import plan_clip, quality_check
    from scripts.gvd_test_env import collect_root

    def plan(case: str, traffic: int = 2):
        return plan_clip(0, seed=1, maps=("west_coast_usa",), max_traffic=traffic, edge_cases=(case,), min_s=4.0, max_s=4.0)

    def car(name: str, x: float, y: float) -> dict:
        return {"name": name, "pos": [x, y, 0.0], "dir": [0, 1, 0]}

    with collect_root() as root:
        # cut_in: a lead car in the ego lane and a car that stays in the next lane.
        d = root / "lead"
        _fake_recording(d, traffic_at=lambda i: [car("gvd_traffic_0", 0.0, i + 20.0), car("gvd_traffic_1", 3.5, i + 12.0)])
        q = quality_check(d, plan("cut_in"), exit_code=0)
        assert q["edge_case_observed"] is False and q["quality"] == "pretrain_only", q
        # A real cut-in: next lane, then into the ego lane 1.5 s later.
        d = root / "cut"
        _fake_recording(d, traffic_at=lambda i: [car("gvd_traffic_1", 3.5 if i < 15 else 0.3, i + 15.0)])
        assert quality_check(d, plan("cut_in"), exit_code=0)["edge_case_observed"] is True
        # A lead car on a 25 m-radius bend shows a 3 m offset at 20 m; when the road
        # straightens it is back at 0. That is the road, not a cut-in.
        d = root / "bend"
        _fake_recording(d, yaw_rate=0.4, traffic_at=lambda i: [car("gvd_traffic_0", 3.0 if i < 10 else 0.0, i + 20.0)])
        assert quality_check(d, plan("cut_in"), exit_code=0)["edge_case_observed"] is False
        # Seen out once, then in the ego lane 10 s later: too old to be one manoeuvre.
        d = root / "stale"
        _fake_recording(d, rows=120, traffic_at=lambda i: (
            [car("gvd_traffic_1", 3.5, i + 15.0)] if i == 0 else [car("gvd_traffic_1", 0.2, i + 15.0)] if i >= 100 else []
        ))
        assert quality_check(d, plan("cut_in"), exit_code=0)["edge_case_observed"] is False
        # sharp_curve: a tight ratio at walking speed (3 m/s < 5 m/s) is not a sharp curve.
        d = root / "slow_turn"
        _fake_recording(d, speed=3.0, yaw_rate=0.4)
        q = quality_check(d, plan("sharp_curve"), exit_code=0)
        assert q["edge_case_observed"] is False, q
        # heavy_traffic counts only our companions (gvd_traffic_*), not other cars.
        d = root / "parked"
        _fake_recording(d, traffic_at=lambda i: [car(f"gvd_traffic_{k}", 3.0 * k, 30.0) for k in range(2)]
                        + [car(f"parked_{k}", -6.0, 10.0 * k) for k in range(6)])
        q = quality_check(d, plan("heavy_traffic", 8), exit_code=0)
        assert q["traffic_placed"] == 2 and q["edge_case_observed"] is False, q
        # need = ceil(0.75 * 6) = 5: four placed of six is not heavy.
        p6 = plan("heavy_traffic", 6)
        assert p6.traffic == 6
        d = root / "four_of_six"
        _fake_recording(d, traffic_n=4)
        q = quality_check(d, p6, exit_code=0)
        assert q["traffic_placed"] == 4 and q["edge_case_observed"] is False, q
        d = root / "five_of_six"
        _fake_recording(d, traffic_n=5)
        assert quality_check(d, p6, exit_code=0)["edge_case_observed"] is True
        # Moving check is the median speed: 1.5 s at 10 m/s then parked is a parked clip.
        d = root / "burst"
        _fake_recording(d, speed_at=lambda i: 10.0 if i < 15 else 0.5)
        q = quality_check(d, plan("rain"), exit_code=0)
        assert q["median_speed_mps"] == 0.5 and any("median speed" in r for r in q["reasons"]), q
    print("edge-case QC: lead car, bend, stale sighting, slow turn, parked cars, ceil share, median speed")


def check_edge_case_observation() -> None:
    from python.collect.clip_mode import plan_clip, quality_check
    from scripts.gvd_test_env import collect_root

    def plan(case: str, traffic: int = 2):
        return plan_clip(0, seed=1, maps=("west_coast_usa",), max_traffic=traffic, edge_cases=(case,), min_s=4.0, max_s=4.0)

    with collect_root() as root:
        d = root / "curve"
        _fake_recording(d, speed=10.0, yaw_rate=0.4)  # 40 m radius
        assert quality_check(d, plan("sharp_curve"), exit_code=0)["edge_case_observed"] is True
        d = root / "straight"
        _fake_recording(d, speed=10.0, yaw_rate=0.01)
        q = quality_check(d, plan("sharp_curve"), exit_code=0)
        assert q["edge_case_observed"] is False and q["quality"] == "pretrain_only", q
        d = root / "ramp"
        _fake_recording(d, spawn_type="ramp")
        assert quality_check(d, plan("merge"), exit_code=0)["edge_case_observed"] is True
        d = root / "heavy"
        _fake_recording(d, traffic_n=8)
        q = quality_check(d, plan("heavy_traffic", 8), exit_code=0)
        assert q["edge_case_observed"] is True and q["traffic_placed"] == 8, q
        d = root / "light"
        _fake_recording(d, traffic_n=1)
        q = quality_check(d, plan("heavy_traffic", 8), exit_code=0)
        assert q["edge_case_observed"] is False and q["traffic_placed"] == 1, q
        d = root / "dark"
        _fake_recording(d, luma=20)
        assert quality_check(d, plan("night"), exit_code=0)["edge_case_observed"] is True
        d = root / "bright"
        _fake_recording(d, luma=150)
        assert quality_check(d, plan("night"), exit_code=0)["edge_case_observed"] is False
    print("edge-case observation from the recording (curve radius, ramp spawn, cars placed, luma)")


def check_session_never_overwrites() -> None:
    from python.collect.clip_mode import new_clip_dir, new_session_dir, plan_clip
    from scripts.gvd_test_env import collect_root

    with collect_root() as root:
        s1 = new_session_dir(root, stamp="20261123_190000")
        s2 = new_session_dir(root, stamp="20261123_190000")
        assert s1 != s2 and s1.is_dir() and s2.is_dir()
        p = plan_clip(0, seed=1, maps=MAPS, max_traffic=2)
        a = new_clip_dir(s1, p)
        (a / "keep.txt").write_text("x", encoding="utf-8")
        b = new_clip_dir(s1, p)
        assert a != b and (a / "keep.txt").read_text(encoding="utf-8") == "x"
        assert not any(b.iterdir())
    print("sessions and clips always get a new folder")


def check_tags_never_overwritten() -> None:
    from python.collect.clip_mode import plan_clip, write_tags
    from scripts.gvd_test_env import collect_root

    p = plan_clip(0, seed=1, maps=MAPS, max_traffic=2)
    with collect_root() as root:
        clip = root / "clip"
        clip.mkdir()
        path = write_tags(clip, p, {"quality": "train", "reasons": []})
        first = path.read_bytes()
        try:
            write_tags(clip, p, {"quality": "pretrain_only", "reasons": ["again"]})
        except FileExistsError:
            pass
        else:
            raise AssertionError("clip_tags.json was replaced")
        assert path.read_bytes() == first
        assert json.loads(first)["gfx"] == "vk"
        other = root / "other"
        other.mkdir()
        assert json.loads(write_tags(other, p, {}, gfx="dx11").read_text(encoding="utf-8"))["gfx"] == "dx11"
    print("clip_tags.json is created once and never replaced; gfx is recorded as requested")


def check_run_clips_with_fake_collector() -> None:
    from python.collect.clip_mode import run_clips
    from scripts.gvd_test_env import collect_root

    calls = []

    def runner(argv: list[str]) -> int:
        out = Path(argv[argv.index("--out") + 1])
        calls.append(argv)
        assert not out.exists() or not any(out.iterdir())
        _fake_recording(out, rows=50)
        return 0

    with collect_root() as root:
        session = run_clips(root, clips=3, seed=5, maps=MAPS, max_traffic=4, runner=runner, min_s=4.0, max_s=5.0)
        dirs = sorted(p for p in session.iterdir() if p.is_dir())
        assert len(dirs) == 3 and len(calls) == 3
        manifest = json.loads((session / "session.json").read_text(encoding="utf-8"))
        assert manifest["clips"] == 3 and len(manifest["results"]) == 3
        assert manifest["ego_model"] == "sbr" and manifest["gfx"] == "vk"
        for d in dirs:
            tags = json.loads((d / "clip_tags.json").read_text(encoding="utf-8"))
            for key in ("edge_case", "map", "traffic_requested", "traffic_placed", "weather",
                        "time_of_day_h", "duration_s", "seed", "spawn_seed", "quality", "reasons",
                        "edge_case_observed", "lossless", "ego_model"):
                assert key in tags, (key, tags)
            assert tags["lossless"] is True and tags["ego_model"] == "sbr"
            assert tags["quality"] in ("train", "pretrain_only")
    print("run_clips: 3 clips, 3 new folders, tags + session manifest")


def check_cli() -> None:
    """scripts/collect_clips.py: 30-60 s bounds, one collector subprocess per clip."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("collect_clips", ROOT / "scripts" / "collect_clips.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)  # type: ignore[union-attr]
    for bad in (["--min-s", "20"], ["--max-s", "90"], ["--min-s", "50", "--max-s", "40"], ["--clips", "0"]):
        try:
            cli.parse_args(["--out", "/tmp/x"] + bad)
        except SystemExit:
            continue
        raise AssertionError(f"accepted {bad}")
    args = cli.parse_args(["--out", "/tmp/x", "--clips", "5", "--maps", "west_coast_usa,italy"])
    assert args.min_s == 30.0 and args.max_s == 60.0 and args.clips == 5
    assert args.maps == ("west_coast_usa", "italy")
    assert args.gfx == "vk"  # Michael's rule: Vulkan unless asked otherwise
    assert cli.parse_args(["--out", "/tmp/x", "--gfx", "dx11"]).gfx == "dx11"
    cmd = cli.collector_command(["--map", "italy"], python="py")
    assert cmd[0] == "py" and cmd[1].endswith("collect_scene_lanes.py") and cmd[2:] == ["--map", "italy"]
    print("collect_clips CLI: 30-60 s bounds, one collector process per clip")


def main() -> None:
    check_edge_case_list()
    check_plans_are_seeded_and_bounded()
    check_collector_argv()
    check_quality_check_marks_not_deletes()
    check_quality_check_failed_clip_paths()
    check_edge_case_observation()
    check_edge_case_false_positives()
    check_session_never_overwrites()
    check_tags_never_overwritten()
    check_run_clips_with_fake_collector()
    check_cli()
    print("test_clip_mode: OK")


if __name__ == "__main__":
    main()
