#!/usr/bin/env python3
"""Recorder Tech watchdog: relaunch hung BeamNG.tech after consecutive attach/load fails.

Offline. Relaunch / Hello / process-tree kill are injected; no live BeamNG,
and this script must not touch a recording host.

    py -3 scripts/test_tech_watchdog.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    from python.collect.tech_watchdog import TechWatchdog
    from scripts.gvd_test_env import collect_root

    calls = []
    with collect_root() as root:
        dog = TechWatchdog(fail_relaunch=2, relaunch_cap=3, exe=None, relaunch=lambda: calls.append("r") or True)
        a, b = root / "a", root / "b"
        _write_stop(a, "session_open_failed")
        _write_stop(b, "session_open_failed")
        assert dog.after_clip(a) is None
        assert dog.after_clip(b) == "tech_relaunch_unavailable"
        assert calls == [] and dog.relaunches == 0
    print("watchdog: no BNG_HOME exe halts with tech_relaunch_unavailable (no attach retry)")


def check_hello_failure_halts() -> None:
    from python.collect.tech_watchdog import TechWatchdog
    from scripts.gvd_test_env import collect_root

    with collect_root() as root:
        exe = root / "BeamNG.tech.exe"
        exe.write_text("x", encoding="utf-8")
        dog = TechWatchdog(fail_relaunch=2, relaunch_cap=3, exe=exe, relaunch=lambda: False)
        a, b = root / "a", root / "b"
        _write_stop(a, "level_load_failed")
        _write_stop(b, "level_load_failed")
        assert dog.after_clip(a) is None
        assert dog.after_clip(b) == "tech_relaunch_hello_failed"
        assert dog.relaunches == 1
    print("watchdog: a relaunch that does not get Hello halts with tech_relaunch_hello_failed")


def check_relaunch_cap_halts() -> None:
    from python.collect.tech_watchdog import TechWatchdog
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
                assert halt == "tech_relaunch_limit"
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
            wait_port_fn=lambda: events.append(("port_clear",)) or True,
            wait_fn=lambda: events.append(("hello",)) or True,
        )
        assert ok is None
        assert [e[0] for e in events] == ["kill", "port_clear", "launch", "hello"], events
        assert events[2][1].endswith("BeamNG.tech.exe")
        assert events[2][2] == TECH_RELAUNCH_ARGV
        assert TECH_RELAUNCH_ARGV == ("-tcom", "-console", "-gfx", "vk")
    print("relaunch_tech: kill process tree, then BeamNG.tech.exe -tcom -console -gfx vk, then Hello")


def check_run_clips_halt_records_stop_reason() -> None:
    from python.collect.clip_mode import run_clips
    from python.collect.tech_watchdog import TechWatchdog
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
        assert manifest.get("stop_reason") == "tech_relaunch_limit", manifest
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



def check_wait_hello_retries_exception_then_succeeds() -> None:
    from python.collect.tech_watchdog import wait_tech_hello
    from python.sensors.tech import TechHelloTimeout

    hellos: list[str] = []
    closed: list[object] = []

    def listening() -> bool:
        return True

    def hello() -> object:
        hellos.append("hello")
        if len(hellos) == 1:
            raise TechHelloTimeout("Hello timeout after 10s on localhost:25252")
        return {"session": True}

    logs: list[str] = []
    ok = wait_tech_hello(
        listening_fn=listening,
        hello_fn=hello,
        close_fn=lambda bng: closed.append(bng),
        timeout_s=1.0,
        poll_s=0.05,
        log=logs.append,
    )
    assert ok is True, (ok, hellos, logs)
    assert hellos == ["hello", "hello"], hellos
    assert closed == [{"session": True}]
    assert any("TechHelloTimeout" in line for line in logs), logs
    print("wait_tech_hello: retries hello_fn exceptions until Hello succeeds")


def check_relaunch_does_not_launch_while_port_listening() -> None:
    from python.collect.tech_watchdog import relaunch_tech
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
            listening_fn=lambda: True,
            port_clear_timeout_s=0.15,
            wait_fn=lambda: events.append(("hello",)) or True,
            log=lambda s: events.append(("log", s)),
        )
        assert ok == "tech_relaunch_port_stuck", ok
        assert "launch" not in [e[0] for e in events], events
        assert "hello" not in [e[0] for e in events], events
        assert events[0] == ("kill",)
    print("relaunch_tech: does not launch while the research port stays LISTENING after kill")


def check_wait_live_hello_sets_quit_on_close_false_before_close() -> None:
    from python.collect.tech_watchdog import wait_live_hello
    import python.sensors.tech as tech_mod

    class FakeBng:
        def __init__(self) -> None:
            self.quit_on_close = True
            self.quit_on_close_at_close: bool | None = None

        def close(self) -> None:
            self.quit_on_close_at_close = self.quit_on_close

    bng = FakeBng()
    orig_listen = tech_mod.research_port_listening
    orig_open = tech_mod.open_tech_beamngpy
    orig_load = tech_mod.load_tech_config
    try:
        tech_mod.research_port_listening = lambda *a, **k: True
        tech_mod.open_tech_beamngpy = lambda *a, **k: bng
        tech_mod.load_tech_config = lambda: {"host": "127.0.0.1", "port": 25252, "home": "", "user": ""}
        ok = wait_live_hello(timeout_s=0.5, log=lambda s: None)
        assert ok is True
        assert bng.quit_on_close_at_close is False, bng.quit_on_close_at_close
        assert bng.quit_on_close is False
    finally:
        tech_mod.research_port_listening = orig_listen
        tech_mod.open_tech_beamngpy = orig_open
        tech_mod.load_tech_config = orig_load
    print("wait_live_hello: hold_tech_process sets quit_on_close False before close()")


def check_collect_clips_main_exits_2_on_halt() -> None:
    cli = _cli()
    from scripts.gvd_test_env import collect_root

    with collect_root() as root:
        orig_run = cli.run_clips
        orig_resolve = cli.resolve_tech_exe
        try:
            def fake_halt(out, **kwargs):
                after = kwargs["after_clip"]
                session = out / "clips_halt"
                session.mkdir()
                stops = []
                for name in ("a", "b"):
                    clip = session / name
                    _write_stop(clip, "session_open_failed")
                    stops.append(after(clip, None, 2, None))
                assert stops == [None, "tech_relaunch_unavailable"], stops
                return session

            cli.run_clips = fake_halt
            cli.resolve_tech_exe = lambda: None
            assert cli.main(["--out", str(root), "--clips", "2"]) == 2

            def fake_ok(out, **kwargs):
                return out / "clips_ok"

            cli.run_clips = fake_ok
            assert cli.main(["--out", str(root), "--clips", "1"]) == 0
        finally:
            cli.run_clips = orig_run
            cli.resolve_tech_exe = orig_resolve
    print("collect_clips.main: exits 2 on halt, 0 otherwise")


def check_resolve_tech_exe() -> None:
    from python.collect.tech_watchdog import resolve_tech_exe
    from scripts.gvd_test_env import collect_root
    import python.sensors.tech as tech_mod

    with collect_root() as root:
        exe = root / "BeamNG.tech.exe"
        assert resolve_tech_exe(home=str(root)) is None
        exe.write_text("x", encoding="utf-8")
        assert resolve_tech_exe(home=str(root)) == exe

        saved = {k: os.environ.get(k) for k in ("BNG_HOME", "BEAMNG_HOME")}
        orig_load = tech_mod.load_tech_config
        try:
            os.environ.pop("BEAMNG_HOME", None)
            os.environ["BNG_HOME"] = str(root)
            assert resolve_tech_exe() == exe
            del os.environ["BNG_HOME"]
            os.environ["BEAMNG_HOME"] = str(root)
            assert resolve_tech_exe() == exe
            del os.environ["BEAMNG_HOME"]
            tech_mod.load_tech_config = lambda: {"home": str(root)}
            assert resolve_tech_exe() == exe
            tech_mod.load_tech_config = lambda: {"home": ""}
            assert resolve_tech_exe() is None
        finally:
            tech_mod.load_tech_config = orig_load
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    print("resolve_tech_exe: BNG_HOME / BEAMNG_HOME / yaml home, only when the exe exists")


def check_tech_pids_port_owner_and_exe() -> None:
    from python.collect.tech_watchdog import _tech_pids, kill_tech_process_tree
    from scripts.gvd_test_env import collect_root

    netstat = (
        "Active Connections\n"
        "\n"
        "  Proto  Local Address          Foreign Address        State           PID\n"
        "  TCP    0.0.0.0:25252          0.0.0.0:0              LISTENING       44888\n"
        "  TCP    127.0.0.1:80           0.0.0.0:0              LISTENING       4\n"
    )
    with collect_root() as root:
        exe = root / "BeamNG.tech.exe"
        exe.write_text("x", encoding="utf-8")
        commands: list[list[str]] = []

        def run(cmd, **kwargs):
            commands.append(list(cmd))
            if cmd[0] == "netstat":
                return subprocess.CompletedProcess(cmd, 0, stdout=netstat, stderr="")
            if cmd[0] == "wmic":
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if cmd[0] == "tasklist":
                raise AssertionError(f"tasklist fallback must not run when a port owner exists: {cmd}")
            if cmd[0] == "taskkill":
                return subprocess.CompletedProcess(cmd, 0, stdout="SUCCESS", stderr="")
            raise AssertionError(cmd)

        with patch("python.collect.tech_watchdog.sys.platform", "win32"), patch(
            "python.collect.tech_watchdog.subprocess.run", side_effect=run
        ):
            assert _tech_pids(exe, port=25252) == [44888]
            killed = kill_tech_process_tree(log=lambda s: None, exe=exe, port=25252)
            assert killed == [44888]
            assert any(c[:2] == ["taskkill", "/F"] and "44888" in c for c in commands), commands
    print("_tech_pids: prefers the research-port owner over unscoped tasklist")


def check_tech_pids_install_root_exe() -> None:
    from python.collect.tech_watchdog import _tech_pids
    from scripts.gvd_test_env import collect_root

    with collect_root() as root:
        exe = root / "BeamNG.tech.exe"
        exe.write_text("x", encoding="utf-8")
        wmic = (
            "\nNode,ExecutablePath,ProcessId\n"
            f"BOX,{exe},55111\n"
            f"BOX,{root / 'other' / 'BeamNG.tech.exe'},999\n"
        )

        def run(cmd, **kwargs):
            if cmd[0] == "netstat":
                return subprocess.CompletedProcess(cmd, 0, stdout="Active Connections\n", stderr="")
            if cmd[0] == "wmic":
                return subprocess.CompletedProcess(cmd, 0, stdout=wmic, stderr="")
            if cmd[0] == "tasklist":
                raise AssertionError("tasklist fallback must not run when install-root exe PIDs exist")
            raise AssertionError(cmd)

        with patch("python.collect.tech_watchdog.sys.platform", "win32"), patch(
            "python.collect.tech_watchdog.subprocess.run", side_effect=run
        ):
            assert _tech_pids(exe, port=25252) == [55111]
    print("_tech_pids: scopes fallback-free hits to the install-root exe path")


def check_tech_pids_tasklist_validates_image_name() -> None:
    from python.collect.tech_watchdog import _tech_pids

    tasklist = (
        '"BeamNG.tech.exe","44888","Console","1","1,234 K"\n'
        '"chrome.exe","123","Console","1","80 K"\n'
        "INFO: No tasks are running which match the specified criteria.\n"
        '"","999","Console","1","1 K"\n'
    )

    def run(cmd, **kwargs):
        if cmd[0] in ("netstat", "wmic"):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "tasklist":
            return subprocess.CompletedProcess(cmd, 0, stdout=tasklist, stderr="")
        raise AssertionError(cmd)

    with patch("python.collect.tech_watchdog.sys.platform", "win32"), patch(
        "python.collect.tech_watchdog.subprocess.run", side_effect=run
    ):
        assert _tech_pids(port=25252) == [44888]
    print("_tech_pids: tasklist fallback validates parts[0] is the Tech image name")


def check_kill_tech_logs_taskkill_failure() -> None:
    from python.collect.tech_watchdog import kill_tech_process_tree

    logs: list[str] = []
    netstat = "  TCP    127.0.0.1:25252        0.0.0.0:0              LISTENING       44888\n"

    def run(cmd, **kwargs):
        if cmd[0] == "netstat":
            return subprocess.CompletedProcess(cmd, 0, stdout=netstat, stderr="")
        if cmd[0] == "wmic":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "taskkill":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="ERROR: Access denied")
        raise AssertionError(cmd)

    with patch("python.collect.tech_watchdog.sys.platform", "win32"), patch(
        "python.collect.tech_watchdog.subprocess.run", side_effect=run
    ):
        assert kill_tech_process_tree(log=logs.append, port=25252) == [44888]
    assert any("taskkill" in line and "44888" in line and "rc=1" in line for line in logs), logs
    print("kill_tech_process_tree: logs a failed taskkill returncode")


def check_kill_tech_drops_non_win32_os_kill() -> None:
    from python.collect.tech_watchdog import kill_tech_process_tree

    with patch("python.collect.tech_watchdog._tech_pids", return_value=[44888]), patch(
        "python.collect.tech_watchdog.sys.platform", "linux"
    ), patch("python.collect.tech_watchdog.os.kill") as okill, patch(
        "python.collect.tech_watchdog.subprocess.run"
    ) as run:
        logs: list[str] = []
        pids = kill_tech_process_tree(log=logs.append)
        assert pids == [44888]
        okill.assert_not_called()
        run.assert_not_called()
    src = (ROOT / "python" / "collect" / "tech_watchdog.py").read_text(encoding="utf-8")
    assert "os.kill" not in src
    assert "SIGKILL" not in src
    print("kill_tech_process_tree: no dead non-win32 os.kill branch")


def check_wait_live_hello_applies_env_overrides() -> None:
    from python.collect.tech_watchdog import wait_live_hello
    import python.sensors.tech as tech_mod

    class Bng:
        quit_on_close = True

        def close(self) -> None:
            pass

    calls: list[tuple] = []

    def fake_listen(host, port, **kwargs):
        calls.append(("listen", host, int(port)))
        return True

    def fake_open(host, port, home=None, user=None, launch=False, **kwargs):
        calls.append(("open", host, int(port), home, user, launch))
        return Bng()

    keys = ("GVD_BEAMNG_HOST", "GVD_BEAMNG_PORT", "BNG_HOME", "BEAMNG_HOME", "GVD_BEAMNG_USER")
    saved = {k: os.environ.get(k) for k in keys}
    orig_listen = tech_mod.research_port_listening
    orig_open = tech_mod.open_tech_beamngpy
    orig_load = tech_mod.load_tech_config
    try:
        os.environ.pop("BEAMNG_HOME", None)
        os.environ["GVD_BEAMNG_HOST"] = "10.0.0.9"
        os.environ["GVD_BEAMNG_PORT"] = "34567"
        os.environ["BNG_HOME"] = r"E:\BeamNG"
        os.environ["GVD_BEAMNG_USER"] = r"E:\BeamNG\user"
        tech_mod.research_port_listening = fake_listen
        tech_mod.open_tech_beamngpy = fake_open
        tech_mod.load_tech_config = lambda: {"host": "127.0.0.1", "port": 25252, "home": "", "user": ""}
        assert wait_live_hello(timeout_s=0.5, log=lambda s: None) is True
        assert ("listen", "10.0.0.9", 34567) in calls, calls
        assert ("open", "10.0.0.9", 34567, r"E:\BeamNG", r"E:\BeamNG\user", False) in calls, calls
    finally:
        tech_mod.research_port_listening = orig_listen
        tech_mod.open_tech_beamngpy = orig_open
        tech_mod.load_tech_config = orig_load
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("wait_live_hello: apply_env_overrides honors GVD_BEAMNG_HOST/PORT/USER and BNG_HOME")


def check_launch_tech_popen_argv_devnull() -> None:
    from python.collect.tech_watchdog import TECH_RELAUNCH_ARGV, launch_tech
    from scripts.gvd_test_env import collect_root

    with collect_root() as root:
        exe = root / "BeamNG.tech.exe"
        exe.write_text("x", encoding="utf-8")
        proc = MagicMock()
        with patch("python.collect.tech_watchdog.sys.platform", "win32"), patch(
            "python.collect.tech_watchdog.subprocess.Popen", return_value=proc
        ) as popen:
            got = launch_tech(exe, log=lambda s: None)
            assert got is proc
            args, kwargs = popen.call_args
            cmd = args[0] if args else kwargs.get("args")
            assert cmd == [str(exe), *TECH_RELAUNCH_ARGV] == [str(exe), "-tcom", "-console", "-gfx", "vk"]
            assert kwargs["stdin"] is subprocess.DEVNULL
            assert kwargs["stdout"] is subprocess.DEVNULL
            assert kwargs["stderr"] is subprocess.DEVNULL
            assert kwargs["cwd"] == str(exe.parent)
            flags = kwargs["creationflags"]
            detached = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            new_grp = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
            assert flags & detached, flags
            assert flags & new_grp, flags
    print("launch_tech: Popen argv is human tail and stdin/stdout/stderr=DEVNULL")



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
    check_wait_hello_retries_exception_then_succeeds()
    check_wait_hello_requires_hello_ok()
    check_relaunch_kills_then_starts_human_argv()
    check_relaunch_does_not_launch_while_port_listening()
    check_wait_live_hello_sets_quit_on_close_false_before_close()
    check_wait_live_hello_applies_env_overrides()
    check_collect_clips_main_exits_2_on_halt()
    check_resolve_tech_exe()
    check_tech_pids_port_owner_and_exe()
    check_tech_pids_install_root_exe()
    check_tech_pids_tasklist_validates_image_name()
    check_kill_tech_logs_taskkill_failure()
    check_kill_tech_drops_non_win32_os_kill()
    check_launch_tech_popen_argv_devnull()
    check_run_clips_halt_records_stop_reason()
    check_dirt_road_has_no_unpaved_spawn_kind()
    print("test_tech_watchdog ok")


if __name__ == "__main__":
    main()
