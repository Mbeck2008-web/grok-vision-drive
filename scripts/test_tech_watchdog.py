#!/usr/bin/env python3
"""Recorder Tech watchdog: relaunch hung BeamNG.tech after consecutive attach/load fails.

Offline. Relaunch / Hello / process-tree kill are injected; no live BeamNG,
and this script must not touch a recording host.

    py -3 scripts/test_tech_watchdog.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _write_stop(clip: Path, reason: str) -> None:
    clip.mkdir(parents=True, exist_ok=True)
    (clip / "collect.json").write_text(
        json.dumps({"stop_reason": reason, "stop_code": 2, "done_s": 0.0}) + "\n",
        encoding="utf-8",
    )


def _cli():
    import importlib.util

    spec = importlib.util.spec_from_file_location("collect_clips", ROOT / "scripts" / "collect_clips.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def check_cli_defaults() -> None:
    cli = _cli()
    args = cli.parse_args(["--out", "/tmp/x"])
    assert args.tech_fail_relaunch == 2, args.tech_fail_relaunch
    assert args.tech_relaunch_cap == 3, args.tech_relaunch_cap
    off = cli.parse_args(["--out", "/tmp/x", "--tech-fail-relaunch", "0"])
    assert off.tech_fail_relaunch == 0
    custom = cli.parse_args(["--out", "/tmp/x", "--tech-fail-relaunch", "4", "--tech-relaunch-cap", "1"])
    assert custom.tech_fail_relaunch == 4 and custom.tech_relaunch_cap == 1
    for bad in (["--tech-fail-relaunch", "-1"], ["--tech-relaunch-cap", "0"], ["--tech-relaunch-cap", "-2"]):
        try:
            cli.parse_args(["--out", "/tmp/x"] + bad)
        except SystemExit:
            continue
        raise AssertionError(f"accepted {bad}")
    print("collect_clips CLI: --tech-fail-relaunch default 2, 0=off; cap default 3")


def check_clip_tech_fail_reason() -> None:
    from python.collect.tech_watchdog import clip_tech_fail_reason
    from scripts.gvd_test_env import collect_root

    with collect_root() as root:
        missing = root / "nope"
        missing.mkdir()
        assert clip_tech_fail_reason(missing) is None
        ok = root / "ok"
        _write_stop(ok, "target_reached")
        (ok / "collect.json").write_text(
            json.dumps({"stop_reason": "target_reached", "stop_code": 0}) + "\n", encoding="utf-8"
        )
        assert clip_tech_fail_reason(ok) is None
        recover = root / "recover"
        _write_stop(recover, "segment_fail")
        assert clip_tech_fail_reason(recover) is None
        hello = root / "hello"
        _write_stop(hello, "session_open_failed")
        assert clip_tech_fail_reason(hello) == "session_open_failed"
        load = root / "load"
        _write_stop(load, "level_load_failed")
        assert clip_tech_fail_reason(load) == "level_load_failed"
    print("clip_tech_fail_reason: only session_open_failed and level_load_failed count")


def check_one_fail_does_not_relaunch() -> None:
    from python.collect.tech_watchdog import TechWatchdog
    from scripts.gvd_test_env import collect_root

    calls = []
    with collect_root() as root:
        exe = root / "fake.exe"
        exe.write_text("x", encoding="utf-8")
        clip = root / "c0"
        _write_stop(clip, "session_open_failed")
        dog = TechWatchdog(fail_relaunch=2, relaunch_cap=3, exe=exe, relaunch=lambda: calls.append("go") or True)
        assert dog.after_clip(clip) is None
        assert calls == []
        assert dog.relaunches == 0
    print("watchdog: a single Tech fail does not relaunch")


def check_two_consecutive_tech_fails_relaunch() -> None:
    from python.collect.tech_watchdog import TechWatchdog
    from scripts.gvd_test_env import collect_root

    calls = []
    logs: list[str] = []
    with collect_root() as root:
        exe = root / "BeamNG.tech.exe"
        exe.write_text("x", encoding="utf-8")
        a, b = root / "a", root / "b"
        _write_stop(a, "level_load_failed")
        _write_stop(b, "session_open_failed")
        dog = TechWatchdog(
            fail_relaunch=2,
            relaunch_cap=3,
            exe=exe,
            relaunch=lambda: calls.append("relaunch") or True,
            log=logs.append,
        )
        assert dog.after_clip(a) is None
        assert calls == []
        assert dog.after_clip(b) is None
        assert calls == ["relaunch"]
        assert dog.relaunches == 1
        assert any("relaunch" in line and "1/3" in line for line in logs), logs
    print("watchdog: two consecutive Tech fails trigger one logged relaunch")


def check_non_tech_fail_resets_streak() -> None:
    from python.collect.tech_watchdog import TechWatchdog
    from scripts.gvd_test_env import collect_root

    calls = []
    with collect_root() as root:
        exe = root / "BeamNG.tech.exe"
        exe.write_text("x", encoding="utf-8")
        dog = TechWatchdog(fail_relaunch=2, relaunch_cap=3, exe=exe, relaunch=lambda: calls.append("r") or True)
        a, ok, c = root / "a", root / "ok", root / "c"
        _write_stop(a, "session_open_failed")
        _write_stop(ok, "segment_fail")
        _write_stop(c, "level_load_failed")
        assert dog.after_clip(a) is None
        assert dog.after_clip(ok) is None
        assert dog.after_clip(c) is None
        assert calls == []
        d = root / "d"
        _write_stop(d, "session_open_failed")
        assert dog.after_clip(d) is None
        assert calls == ["r"]
    print("watchdog: a non-Tech fail (segment_fail) resets the consecutive streak")


def check_success_resets_streak() -> None:
    from python.collect.tech_watchdog import TechWatchdog
    from scripts.gvd_test_env import collect_root

    calls = []
    with collect_root() as root:
        exe = root / "BeamNG.tech.exe"
        exe.write_text("x", encoding="utf-8")
        dog = TechWatchdog(fail_relaunch=2, relaunch_cap=3, exe=exe, relaunch=lambda: calls.append("r") or True)
        a, ok = root / "a", root / "ok"
        _write_stop(a, "session_open_failed")
        ok.mkdir()
        (ok / "collect.json").write_text(json.dumps({"stop_reason": "target_reached", "stop_code": 0}) + "\n", encoding="utf-8")
        assert dog.after_clip(a) is None
        assert dog.after_clip(ok) is None
        assert calls == []
    print("watchdog: a successful clip resets the consecutive streak")


def check_off_never_relaunches() -> None:
    from python.collect.tech_watchdog import TechWatchdog
    from scripts.gvd_test_env import collect_root

    calls = []
    with collect_root() as root:
        exe = root / "BeamNG.tech.exe"
        exe.write_text("x", encoding="utf-8")
        dog = TechWatchdog(fail_relaunch=0, relaunch_cap=3, exe=exe, relaunch=lambda: calls.append("r") or True)
        for i in range(5):
            clip = root / f"c{i}"
            _write_stop(clip, "session_open_failed")
            assert dog.after_clip(clip) is None
        assert calls == [] and dog.relaunches == 0
    print("watchdog: --tech-fail-relaunch 0 never relaunches")


def check_missing_exe_halts() -> None:
    from python.collect.tech_watchdog import STOP_RELAUNCH_UNAVAILABLE, TechWatchdog
    from scripts.gvd_test_env import collect_root

    calls = []
    with collect_root() as root:
        dog = TechWatchdog(fail_relaunch=2, relaunch_cap=3, exe=None, relaunch=lambda: calls.append("r") or True)
        a, b = root / "a", root / "b"
        _write_stop(a, "session_open_failed")
        _write_stop(b, "session_open_failed")
        assert dog.after_clip(a) is None
        assert dog.after_clip(b) == STOP_RELAUNCH_UNAVAILABLE
        assert calls == [] and dog.relaunches == 0
    print("watchdog: no BNG_HOME exe halts with tech_relaunch_unavailable (no attach retry)")


def check_hello_failure_halts() -> None:
    from python.collect.tech_watchdog import STOP_RELAUNCH_HELLO_FAILED, TechWatchdog
    from scripts.gvd_test_env import collect_root

    with collect_root() as root:
        exe = root / "BeamNG.tech.exe"
        exe.write_text("x", encoding="utf-8")
        dog = TechWatchdog(fail_relaunch=2, relaunch_cap=3, exe=exe, relaunch=lambda: False)
        a, b = root / "a", root / "b"
        _write_stop(a, "level_load_failed")
        _write_stop(b, "level_load_failed")
        assert dog.after_clip(a) is None
        assert dog.after_clip(b) == STOP_RELAUNCH_HELLO_FAILED
        assert dog.relaunches == 1
    print("watchdog: a relaunch that does not get Hello halts with tech_relaunch_hello_failed")


def check_relaunch_cap_halts() -> None:
    from python.collect.tech_watchdog import STOP_RELAUNCH_LIMIT, TechWatchdog
    from scripts.gvd_test_env import collect_root

    calls = []
    with collect_root() as root:
        exe = root / "BeamNG.tech.exe"
        exe.write_text("x", encoding="utf-8")
        dog = TechWatchdog(fail_relaunch=2, relaunch_cap=3, exe=exe, relaunch=lambda: calls.append("r") or True)
        # 3 successful relaunches from 6 consecutive Tech fails
        for i in range(6):
            clip = root / f"c{i}"
            _write_stop(clip, "session_open_failed")
            halt = dog.after_clip(clip)
            assert halt is None, (i, halt)
        assert calls == ["r", "r", "r"] and dog.relaunches == 3
        # two more consecutive fails would need a 4th relaunch -> halt
        for i, name in enumerate(("x", "y")):
            clip = root / name
            _write_stop(clip, "session_open_failed")
            halt = dog.after_clip(clip)
            if i == 0:
                assert halt is None
            else:
                assert halt == STOP_RELAUNCH_LIMIT
        assert calls == ["r", "r", "r"]
    print("watchdog: a 4th needed relaunch halts with tech_relaunch_limit")


def check_wait_hello_rejects_listen_only() -> None:
    from python.collect.tech_watchdog import wait_tech_hello
    from python.sensors.tech import TechHelloTimeout

    hellos = []

    def listening() -> bool:
        return True

    def hello() -> object:
        hellos.append("hello")
        raise TechHelloTimeout("Hello timeout after 10s on localhost:25252")

    closed = []
    ok = wait_tech_hello(
        listening_fn=listening,
        hello_fn=hello,
        close_fn=lambda bng: closed.append(bng),
        timeout_s=0.2,
        poll_s=0.05,
    )
    assert ok is False
    assert hellos, "must attempt a real Hello, not trust LISTEN"
    assert closed == []
    print("wait_tech_hello: a listening port without Hello is not ready")


def check_wait_hello_requires_hello_ok() -> None:
    from python.collect.tech_watchdog import wait_tech_hello

    def listening() -> bool:
        return True

    closed = []

    def hello() -> object:
        return {"session": True}

    ok = wait_tech_hello(
        listening_fn=listening,
        hello_fn=hello,
        close_fn=lambda bng: closed.append(bng),
        timeout_s=0.2,
        poll_s=0.05,
    )
    assert ok is True
    assert closed == [{"session": True}], "Hello session must be closed (quit_on_close=False) so the collector can attach"
    print("wait_tech_hello: ready only after Hello ok, then the probe session is closed")


def check_relaunch_kills_then_starts_human_argv() -> None:
    from python.collect.tech_watchdog import TECH_RELAUNCH_ARGV, relaunch_tech
    from scripts.gvd_test_env import collect_root

    events: list[tuple] = []
    with collect_root() as root:
        exe = root / "BeamNG.tech.exe"
        exe.write_text("x", encoding="utf-8")

        def kill() -> list[int]:
            events.append(("kill",))
            return [44888]

        def launch(path: Path, argv: tuple[str, ...]) -> int:
            events.append(("launch", str(path), argv))
            return 99

        ok = relaunch_tech(
            exe=exe,
            kill_fn=kill,
            launch_fn=launch,
            wait_fn=lambda: events.append(("hello",)) or True,
        )
        assert ok is True
        assert [e[0] for e in events] == ["kill", "launch", "hello"], events
        assert events[1][1].endswith("BeamNG.tech.exe")
        assert events[1][2] == TECH_RELAUNCH_ARGV
        assert TECH_RELAUNCH_ARGV == ("-tcom", "-console", "-gfx", "vk")
    print("relaunch_tech: kill process tree, then BeamNG.tech.exe -tcom -console -gfx vk, then Hello")


def check_run_clips_halt_records_stop_reason() -> None:
    from python.collect.clip_mode import run_clips
    from python.collect.tech_watchdog import STOP_RELAUNCH_LIMIT, TechWatchdog
    from scripts.gvd_test_env import collect_root
    from scripts.test_clip_mode import MAPS, _fake_recording

    with collect_root() as root:
        exe = root / "BeamNG.tech.exe"
        exe.write_text("x", encoding="utf-8")
        dog = TechWatchdog(fail_relaunch=2, relaunch_cap=1, exe=exe, relaunch=lambda: True)

        def runner(argv: list[str]) -> int:
            out = Path(argv[argv.index("--out") + 1])
            _fake_recording(out, rows=8, speed=10.0)
            _write_stop(out, "session_open_failed")
            return 2

        session = run_clips(
            root,
            clips=8,
            seed=5,
            maps=MAPS,
            max_traffic=0,
            runner=runner,
            min_s=4.0,
            max_s=5.0,
            after_clip=lambda clip, plan, code, qc: dog.after_clip(clip),
        )
        manifest = json.loads((session / "session.json").read_text(encoding="utf-8"))
        assert manifest.get("stop_reason") == STOP_RELAUNCH_LIMIT, manifest
        # 2 fails -> relaunch #1 (cap 1), 2 more fails -> halt. 4 clips, not 8.
        assert len(manifest["results"]) == 4, manifest["results"]
        dirs = sorted(p for p in session.iterdir() if p.is_dir())
        assert len(dirs) == 4
    print("run_clips: halt writes session.json stop_reason and does not start remaining clips")


def check_dirt_road_has_no_unpaved_spawn_kind() -> None:
    """clip_002 italy dirt share 0.00: dirt_road picks a DIRT_MAPS map but not an unpaved pose.

    spawn_mix only classifies city / highway / ramp. There is no dirt/unpaved
    spawn kind, so a legal italy spawn can sit on pavement and the quality
    check correctly reports dirt share 0.00. This is a missing selector, not a
    flipped spawn-kind bug; do not invent a kind the road dump does not have.
    """
    from python.collect.clip_mode import DIRT_MAPS, plan_clip
    from python.collect.spawn_mix import KINDS

    assert KINDS == ("city", "highway", "ramp")
    assert "italy" in DIRT_MAPS
    kinds = set()
    maps = set()
    for i in range(80):
        plan = plan_clip(i, seed=12, maps=("west_coast_usa", "italy", "east_coast_usa"), max_traffic=6)
        if plan.edge_case != "dirt_road":
            continue
        kinds.add(plan.spawn_kind)
        maps.add(plan.map)
    assert kinds, "seed 12 should produce dirt_road plans"
    assert kinds == {None}, kinds
    assert maps <= set(DIRT_MAPS)
    assert "italy" in maps or "east_coast_usa" in maps
    print("dirt_road: no unpaved spawn kind; pavement spawns can yield dirt share 0.00")


def main() -> None:
    check_cli_defaults()
    check_clip_tech_fail_reason()
    check_one_fail_does_not_relaunch()
    check_two_consecutive_tech_fails_relaunch()
    check_non_tech_fail_resets_streak()
    check_success_resets_streak()
    check_off_never_relaunches()
    check_missing_exe_halts()
    check_hello_failure_halts()
    check_relaunch_cap_halts()
    check_wait_hello_rejects_listen_only()
    check_wait_hello_requires_hello_ok()
    check_relaunch_kills_then_starts_human_argv()
    check_run_clips_halt_records_stop_reason()
    check_dirt_road_has_no_unpaved_spawn_kind()
    print("test_tech_watchdog ok")


if __name__ == "__main__":
    main()
