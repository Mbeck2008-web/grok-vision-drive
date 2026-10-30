#!/usr/bin/env python3
"""Offline M6 retail-package checks (no BeamNG, no network)."""
from __future__ import annotations

import ast
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

from typing import Callable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_release_zip as rel  # noqa: E402
from gvd_test_env import isolated_home  # noqa: E402
from python.control.actuate import (  # noqa: E402
    ENGAGE_FRESH_S,
    CmdJsonActuator,
    DriveCommand,
    cmd_path,
    ego_path,
    engage_path,
    make_actuator,
    read_ego_feedback,
    read_engage_flag,
    write_engage_flag,
)
from python.runtime.hw_probe import HwReport, cam_claim  # noqa: E402
from python.runtime.state_io import read_state, state_path  # noqa: E402
from python.sensors.cameras import (  # noqa: E402
    CAM_IDS,
    WindowBackend,
    _pick_best_beamng_rect,
    _skip_beamng_title,
)

CHROME_RE = re.compile(r"tesla|\bfsd\b|full self[- ]driving|autopilot", re.I)


def _copy_repo(dst: Path) -> None:
    ignore = shutil.ignore_patterns(".git", "dist", "__pycache__", ".venv", "venv", "data")
    for item in ROOT.iterdir():
        if item.name in (".git", "dist", ".venv", "venv"):
            continue
        if item.is_dir():
            shutil.copytree(item, dst / item.name, ignore=ignore)
        else:
            shutil.copy2(item, dst / item.name)


def check_release_zip() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "repo"
        tmp.mkdir()
        _copy_repo(tmp)
        # Plant everything that must never ship.
        (tmp / "models").mkdir(exist_ok=True)
        (tmp / "models" / "e2e_current.onnx").write_bytes(b"\x00" * 64)
        (tmp / "models" / "yolov8n.pt").write_bytes(b"\x00" * 64)
        (tmp / "python" / "weights.safetensors").write_bytes(b"\x00" * 16)
        (tmp / "data" / "clips").mkdir(parents=True, exist_ok=True)
        (tmp / "data" / "clips" / "clip_x.mp4").write_bytes(b"\x00" * 16)
        (tmp / "python" / "__pycache__").mkdir()
        (tmp / "python" / "__pycache__" / "run_vision.cpython-312.pyc").write_bytes(b"\x00")
        (tmp / ".git").mkdir()
        (tmp / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        (tmp / "dist").mkdir()
        (tmp / "dist" / "old.zip").write_bytes(b"PK")

        files = rel.collect_files(tmp)
        out = Path(td) / "gvd-retail-test.zip"
        zip_path, names = rel.build(out, version="test", flat=False, root=tmp, files=files)
        problems = rel.verify(zip_path, flat=False, version="test")
        assert not problems, problems

        with zipfile.ZipFile(zip_path) as zf:
            members = zf.namelist()
            assert members and all(m.startswith("gvd-retail-test/") for m in members), members[:5]
            rels = [m.split("/", 1)[1] for m in members]
            for must in rel.MUST_HAVE:
                assert must in rels, f"missing {must}"
            for bad in (".pt", ".safetensors", ".pyc", ".mp4"):
                assert not any(r.endswith(bad) for r in rels), bad
            onnx_members = [r for r in rels if r.endswith(".onnx")]
            assert onnx_members == ["models/yolov8n.onnx"], onnx_members
            assert "models/NOTICE.txt" in rels
            assert "models/e2e_current.onnx" not in rels
            assert "models/yolov8n.pt" not in rels
            assert not any(r.startswith((".git/", "data/", "dist/", "scripts/")) for r in rels)
            assert "requirements-beamng.txt" not in rels
            assert "play_gvd_tech.bat" not in rels  # Tech launcher stays out of the retail zip
            assert "config/tech.yaml" in rels
            assert "python/sensors/tech.py" in rels
            assert not any("clips" in r for r in rels)
            assert "requirements-perception.txt" not in rels
            assert "requirements-viz.txt" not in rels
            # Mod entry point + UI icon survive; tests do not.
            assert "beamng_mod/scripts/gvd/modScript.lua" in rels
            assert "beamng_mod/ui/modules/apps/GVD/app.png" in rels
            assert "docs/media/gvd_cabin_synthetic.png" in rels
            assert "docs/media/gvd_ingame_ui_synthetic.png" in rels
            assert "docs/media/gvd_ingame_ui_drive_synthetic.png" in rels
            assert "docs/media/SOURCE.md" in rels
            assert "AGENTS.md" in rels
            assert "models/.gitkeep" in rels
            assert "models/yolov8n.onnx" in rels
            assert not any(r.startswith("scripts/test_") for r in rels)
            # Windows-friendly: .bat files are CRLF inside the zip.
            for r in ("install.bat", "uninstall.bat", "play_gvd.bat"):
                data = zf.read(f"gvd-retail-test/{r}")
                assert b"\r\n" in data and data.count(b"\n") == data.count(b"\r\n"), r
            ver = zf.read("gvd-retail-test/VERSION.txt").decode("utf-8")
            assert "retail package test" in ver
            assert "1-cam window capture only" in ver
            assert "drives the sim car through Documents/GVD/gvd_cmd.json" in ver

        # Flat variant has no top-level folder.
        flat = Path(td) / "flat.zip"
        rel.build(flat, version="test", flat=True, root=tmp, files=files)
        assert not rel.verify(flat, flat=True, version="test")
        with zipfile.ZipFile(flat) as zf:
            assert "install.bat" in zf.namelist()

    # The real checkout must package cleanly too (manifest only; no dist/ write).
    real = rel.collect_files(ROOT)
    assert "install.bat" in real and "python/run_vision.py" in real
    assert "models/yolov8n.onnx" in real
    assert not any(f.endswith(".pt") for f in real)
    assert not any(f.endswith(".onnx") and f != "models/yolov8n.onnx" for f in real)
    assert "requirements-foxglove.txt" not in real
    assert "config/sensors.yaml" in real
    assert "python/sensors/extras.py" in real
    assert "python/viz/foxglove_bridge.py" in real


def check_boot_line_honesty() -> None:
    def rep(backend: str) -> HwReport:
        return HwReport(cpu="i9-9900K", ram_gb=32, dgpu="GTX 1080 Ti", dgpu_vram_gb=11.0, igpu="UHD630", qsv=True, backend=backend)

    retail = rep("window")
    line = retail.boot_line()
    assert retail.is_retail
    assert "backend=window" in line and "cams=1/8" in line and "retail" in line and "not 8" in line, line
    assert not CHROME_RE.search(line), line

    tech = rep("beamngpy")
    assert not tech.is_retail
    assert "cams<=8" in tech.boot_line() and "tech" in tech.boot_line()
    assert "cams=0/8" in rep("stub").boot_line()
    assert "8" not in cam_claim("window").split("(")[0].replace("1/8", "")  # never "8 cams" on retail


def check_window_backend_one_cam() -> None:
    class _FakeCam:
        def grab(self):
            return np.full((360, 640, 3), 90, dtype=np.uint8)

    wb = WindowBackend(long_side=320)
    wb._cam = _FakeCam()
    wb._region = {"left": 0, "top": 0, "width": 640, "height": 360}
    bundle = wb.grab()
    assert bundle.backend == "window"
    assert bundle.note.startswith("retail: 1 window"), bundle.note
    health = bundle.health_str()
    assert set(health) == set(CAM_IDS)
    ok = [k for k, v in health.items() if v == "ok"]
    assert ok == ["main"], health
    assert set(bundle.frames) <= {"main", "cam_main"}, list(bundle.frames)
    assert bundle.main_bgr() is not None and max(bundle.main_bgr().shape[:2]) <= 320

    # No capture library at all → honest missing, never synthesized frames.
    wb2 = WindowBackend(long_side=320)
    wb2._region = {"left": 0, "top": 0, "width": 10, "height": 10}
    empty = wb2.grab()
    assert empty.main_bgr() is None
    assert all(v == "missing" for v in empty.health_str().values())


def check_window_picker_prefers_drive() -> None:
    """Retail must grab BeamNG.drive, not a crash dialog or a Tech window."""
    assert _skip_beamng_title("Notepad")
    assert _skip_beamng_title("BeamNG.drive Crash Reporter")
    assert not _skip_beamng_title("BeamNG.drive")
    crash = ("BeamNG.drive Crash Reporter", 0, 0, 800, 600)
    tiny = ("BeamNG.drive", 0, 0, 100, 100)
    tech = ("BeamNG.tech", 0, 0, 1920, 1080)
    drive = ("BeamNG.drive", 10, 10, 1280, 720)
    generic = ("BeamNG", 0, 0, 1920, 1080)
    assert _pick_best_beamng_rect([crash, tiny]) is None
    assert _pick_best_beamng_rect([tech, drive]) == (10, 10, 1280, 720)
    assert _pick_best_beamng_rect([generic, tech]) == (0, 0, 1920, 1080)
    assert _pick_best_beamng_rect([tiny, drive]) == (10, 10, 1280, 720)


class _FakeClock:
    """A wall clock only the test moves. ``now`` is what ``time.time()`` returns in GVD's
    actuate / paths / state_io modules while ``fake_clock`` is active."""

    def __init__(self, start: float) -> None:
        self.now = float(start)

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


class _TimeProxy:
    """Stands in for the ``time`` module: ``time()`` reads the fake clock, the rest is real."""

    def __init__(self, clock: _FakeClock) -> None:
        self._clock = clock

    def time(self) -> float:
        return self._clock.now

    def __getattr__(self, name: str):
        return getattr(time, name)


_CLOCK: _FakeClock | None = None
FAKE_CLOCK_START = 1_800_000_000.0


@contextlib.contextmanager
def fake_clock(start: float = FAKE_CLOCK_START):
    """In-process checks judge freshness on a clock the test owns, not on how fast it runs.

    Patches ``time`` in the modules that stamp and judge (engage flag, cmd heartbeat, ego
    ack, lua_bus handshake), and the fake mod's files get their mtimes on the same clock
    (``_stamp_file``). Only for checks that call GVD in-process; a spawned supervisor runs
    on the real clock and is judged on its own stamps instead.
    """
    global _CLOCK
    from python.control import actuate as _actuate
    from python.runtime import paths as _paths
    from python.runtime import state_io as _state_io

    clock = _FakeClock(start)
    saved = [(mod, mod.time) for mod in (_actuate, _paths, _state_io)]
    for mod, _ in saved:
        mod.time = _TimeProxy(clock)
    _CLOCK = clock
    try:
        yield clock
    finally:
        for mod, real in saved:
            mod.time = real
        _CLOCK = None


def _now() -> float:
    return time.time() if _CLOCK is None else _CLOCK.now


def _stamp_file(path: Path, age_s: float = 0.0) -> None:
    """Under a fake clock a file the fake mod writes is stamped `age_s` before its now."""
    if _CLOCK is None and not age_s:
        return
    t = _now() - float(age_s)
    os.utime(path, (t, t))


def _touch_lua_bus() -> Path:
    """Keep a fresh lua_bus handshake at the predicted pin so cmd tests can actuate."""
    from python.runtime import paths as gvd_paths

    pin = gvd_paths.predicted_gvd_docs_path()
    pin.mkdir(parents=True, exist_ok=True)
    abs_pin = pin if pin.is_absolute() else pin.resolve()
    abs_pin.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"lua_bus": str(abs_pin)})
    _write_link_atomic(pin, payload)
    if abs_pin != pin:
        _write_link_atomic(abs_pin, payload)
    return abs_pin


def _write_link_atomic(folder: Path, payload: str) -> None:
    """Write gvd_link.json via a temp file + os.replace. The test rewrites it every ~10 ms;
    an in-place write_text (truncate, then write) let run_vision read it empty and stop one
    tick with bus_mismatch. This fixes the test's writer only, not main.lua's.

    On Windows os.replace raises PermissionError while run_vision has the target open, so
    retry that one error briefly (at most ~50 ms); anything else, or a longer hold, raises."""
    tmp = folder / "gvd_link.json.tmp"
    tmp.write_text(payload, encoding="utf-8")
    _stamp_file(tmp)
    for attempt in range(LINK_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, folder / "gvd_link.json")
            return
        except PermissionError:
            if attempt + 1 >= LINK_REPLACE_ATTEMPTS:
                raise
            time.sleep(0.001)


LINK_REPLACE_ATTEMPTS = 50


# The fake mod plays gvd_main's meaning of echo_seq: its own counter, +1 per echo written
# (`onEgoFeedback`), never the command seq it last applied (that is `applied_seq`).
_FAKE_ECHO = {"k": 0}
# What the current mod stamps as mod_version (M.MOD_VERSION in gvd_main).
_MOD_VERSION = re.search(
    r"M\.MOD_VERSION\s*=\s*'([^']+)'",
    (ROOT / "beamng_mod" / "lua" / "ge" / "extensions" / "gvd" / "main.lua").read_text(encoding="utf-8"),
).group(1)
# Test-only knob: a real sleep per _run_supervisor loop pass (plays a slow host).
STEP_SLEEP_S = float(os.environ.get("GVD_TEST_STEP_SLEEP_S") or 0.0)
# check_ffb_override_live_slow_host: the sleep CI's slow runner amounted to (echo_stale on main).
SLOW_HOST_SLEEP_S = 0.35
FAKE_ECHO_PERIOD_S = 0.1  # gvd_main EGO_POLL_S: the mod echoes at 10 Hz, whatever the loop does
# Most run_vision may take to write its first gvd_cmd.json (imports, model load). Startup only:
# the `seconds` a check asks for, and its timed windows, start when that first command lands.
SUPERVISOR_STARTUP_S = 30.0


def _next_echo_seq() -> int:
    _FAKE_ECHO["k"] += 1
    return _FAKE_ECHO["k"]


def _write_ego(
    seq: int,
    *,
    applying: bool = True,
    speed: float = 5.5,
    steer_in: float = 0.0,
    throttle_in: float = 0.0,
    brake_in: float = 0.0,
    age_s: float = 0.0,
    echo_seq: int | None = None,
    stamp_echo: bool = True,
    mod_version: str | None = None,
) -> None:
    """Play gvd_main's gvd_ego.json. `seq` is the command seq the mod last applied
    (`applied_seq`). `echo_seq` is the mod's own echo counter: by default the next value of it,
    one per call, whatever `seq` is. `stamp_echo=False` is a mod from before the echo_seq
    contract (no echo_seq, no mod_version). `mod_version` defaults to the current mod's."""
    root = _touch_lua_bus()
    payload = {
        "speed_mps": speed,
        "steering_input": steer_in,
        "throttle_input": throttle_in,
        "brake_input": brake_in,
        "applied_seq": seq,
        "applying": applying,
        "mtime": int(_now()),
        "lua_bus": str(root),
    }
    if stamp_echo:
        payload["echo_seq"] = _next_echo_seq() if echo_seq is None else int(echo_seq)
        payload["mod_version"] = _MOD_VERSION if mod_version is None else mod_version
    ego_path().write_text(json.dumps(payload), encoding="utf-8")
    _stamp_file(ego_path(), age_s)


def check_cmd_json_drive_bus() -> None:
    """Retail actuator: engaged → real drive payload; disengaged → idle stop; applied only on Lua ack.

    On the fake clock: every freshness verdict here (ack age, lua_bus handshake, heartbeat
    stamp) is judged at a time the test sets, so a slow runner cannot age a fresh file.
    """
    with fake_clock() as clock:
        _cmd_json_drive_bus(clock)


def _cmd_json_drive_bus(clock: _FakeClock) -> None:
    from python.control.actuate import EGO_FRESH_S

    _touch_lua_bus()
    if ego_path().exists():
        ego_path().unlink()
    act = make_actuator(None, prefer_beamngpy=True)
    assert isinstance(act, CmdJsonActuator) and act.name == "cmd_json"

    # Engaged, no Lua yet → payload carries the real command + engaged=true, but nothing is claimed.
    act.note_engaged(True)
    act.note_ack(read_ego_feedback())
    out = act.apply(DriveCommand(steer=0.3, throttle=0.4, brake=0.0, seq=7, reason="ok"))
    payload = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert payload == {
        "steer": 0.3, "throttle": 0.4, "brake": 0.0, "seq": 7, "engaged": True,
        "heartbeat_mtime": payload["heartbeat_mtime"], "reason": "ok",
    }, payload
    assert payload["heartbeat_mtime"] == clock.now, (payload["heartbeat_mtime"], clock.now)
    assert out.applied is False and out.reason == "cmd_json_pending", out

    # Lua echoes a fresh ack for that seq → applied.
    _write_ego(7, steer_in=0.3)
    fb = read_ego_feedback()
    assert fb is not None and fb.fresh and fb.applying and fb.applied_seq == 7 and fb.speed_mps == 5.5
    act.note_ack(fb)
    out = act.apply(DriveCommand(steer=0.3, throttle=0.4, brake=0.0, seq=8, reason="ok"))
    assert out.applied is True and out.reason == "cmd_json_applied", out

    # Ack too far behind → not applied.
    _write_ego(1)
    act.note_ack(read_ego_feedback())
    out = act.apply(DriveCommand(steer=0.0, throttle=0.1, brake=0.0, seq=9, reason="ok"))
    assert out.applied is False and out.reason == "cmd_json_pending", out

    # Previous session's high applied_seq is not an ack of a restarted supervisor.
    _write_ego(500)
    act.note_ack(read_ego_feedback())
    out = act.apply(DriveCommand(steer=0.0, throttle=0.1, brake=0.0, seq=4, reason="ok"))
    assert out.applied is False and out.reason == "cmd_json_pending", out

    # The ack window's edge, on the fake clock: EGO_FRESH_S old is still fresh, past it is not.
    _write_ego(9)
    clock.advance(EGO_FRESH_S)
    assert read_ego_feedback().fresh is True
    clock.advance(0.01)
    assert read_ego_feedback().fresh is False
    _touch_lua_bus()

    # Stale ack (Lua stopped echoing) → not applied.
    _write_ego(9, age_s=5.0)
    fb = read_ego_feedback()
    assert fb is not None and not fb.fresh
    act.note_ack(fb)
    out = act.apply(DriveCommand(steer=0.0, throttle=0.1, brake=0.0, seq=10, reason="ok"))
    assert out.applied is False and out.reason == "cmd_json_pending", out

    # Disengaged → stop with engaged=false so Lua hands the car back; gate reasons pass through.
    act.note_engaged(False)
    _write_ego(10)
    act.note_ack(read_ego_feedback())
    out = act.stop(seq=11, reason="not_engaged")
    payload = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert payload["engaged"] is False and payload["brake"] == 1.0 and payload["throttle"] == 0.0
    assert payload["reason"] == "not_engaged" and out.applied is False and out.reason == "not_engaged"
    out = act.stop(seq=12, reason="shutdown")
    assert out.applied is False and out.reason == "cmd_json_idle"
    # Engaged gate hold (preview) still rides along as a brake hold Lua applies.
    act.note_engaged(True)
    out = act.stop(seq=13, reason="preview_blocked")
    payload = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert payload["engaged"] is True and payload["brake"] == 1.0 and out.reason == "preview_blocked"

    # Torn / garbage ego file never crashes the loop.
    ego_path().write_text('{"speed_mps": 3.', encoding="utf-8")
    assert read_ego_feedback() is None
    ego_path().unlink()


def _read_json(path: Path) -> dict:
    """Best-effort JSON object. Windows can raise PermissionError while Lua/Python replace the file."""
    for _ in range(4):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            time.sleep(0.002)
    return {}


def _player_named(reason: str, *, engage: dict | None = None, state: dict | None = None) -> bool:
    """True when gvd_engage.json or gvd_state.json names this player_* reason.

    Later ticks rewrite state.disengage_reason to not_engaged and reset override;
    the engage file is the sticky record. Either surface is enough.
    """
    if engage and engage.get("disengage_reason") == reason:
        return True
    if not state:
        return False
    ovr = state.get("override") if isinstance(state.get("override"), dict) else {}
    return state.get("disengage_reason") == reason or ovr.get("reason") == reason


def _run_supervisor(
    extra: list[str],
    seconds: float,
    ack: bool = False,
    steer_echo: Callable[[int], float] | None = None,
    brake_bias: float = 0.0,
    expect_disengage: str | None = None,
    stamp_echo: bool = True,
    freeze_echo_after_s: float | None = None,
    on_engage_drop: Callable[[float], None] | None = None,
    wheel_by_echo: Callable[[int], float] | None = None,
    cmd_blind: tuple[float, float] | None = None,
    mod_version: str | None = None,
    clock: dict | None = None,
    engage_heartbeat: bool = True,
    echo_clock: bool = False,
    loop_sleep_s: float | None = None,
    report: dict | None = None,
) -> dict | None:
    """Run run_vision.py headless for `seconds`; optionally play the mod's ack role meanwhile.

    The run's clock starts at this supervisor's first tick (its first gvd_cmd.json and
    gvd_state.json, `t_ready`), not at spawn: imports (torch through the trainer modules) and
    model load are startup, not the work under test. `seconds`, `cmd_blind` and
    `freeze_echo_after_s` count from `t_ready` on time.monotonic(); wall time is only used to
    compare with the stamps inside the files.
    `clock`, when given, gets `t_spawn` and `t_ready` (wall time), and with
    `freeze_echo_after_s` also `t_frozen`: when the mod wrote its last advancing echo.

    Files from an earlier run never count. gvd_cmd.json and gvd_state.json are removed before
    spawn, the fake mod only takes a command stamped after spawn, and the latch only takes an
    engage file stamped after spawn. The latch prefers the snapshot whose gvd_state.json still
    names `expect_disengage` (the refusal tick); assert on the returned snapshot, not on a later
    read_state().

    The fake mod echoes at 10 Hz (`FAKE_ECHO_PERIOD_S`, gvd_main `EGO_POLL_S`), not per command.
    Each echo reads the newest command, acks its seq as `applied_seq`, and stamps `echo_seq`
    from the mod's own counter, +1 per echo. The two are separate: `echo_seq` moves on with the
    wheel readings, `applied_seq` with the commands.

    Wheel models (pick one): `steer_echo(cmd_seq)` reports a wheel position keyed on the command
    seq being acked; `wheel_by_echo(echo_seq)` keys it on the echo itself (per-sample FFB
    chatter). `stamp_echo=False` plays a mod from before echo_seq. `freeze_echo_after_s` keeps
    rewriting the last echo (fresh mtime, same echo_seq) from that point, a mod whose electrics
    echo stopped. `cmd_blind=(t0, t1)` (seconds after start): commands stop reaching the mod in
    that window (it keeps the last command and its seq) while its echoes keep advancing.
    `on_engage_drop(t)` is called with the wall time the engage file was first seen false.
    `engage_heartbeat=False` plays a mod whose Alt+G heartbeat stopped: gvd_engage.json keeps
    the stamp it had, so it goes stale `ENGAGE_FRESH_S` later.
    `echo_clock=True` runs the supervisor through `scripts/gvd_echo_clock_child.py`: its echo
    gate reads the fake mod's own clock (`FAKE_ECHO_PERIOD_S` per echo written) instead of
    time.monotonic(), so a slow host that delays the fake mod's echoes is not an echo_stale the
    mod would show. Not with `freeze_echo_after_s` (that run tests the stale timing itself).
    `loop_sleep_s` (test-only, default GVD_TEST_STEP_SLEEP_S or 0) adds a real sleep per loop
    pass to play a slow host.
    `report`, when given, gets `hb_age_s` (each loop pass: real wall time minus gvd_cmd.json
    heartbeat_mtime, after the first tick) and `stdout` (the supervisor's stdout text). Timing
    observations only; nothing in this function judges them.

    The fake mod echoes back what it was told to apply, exactly like `gvd_main` does, so the
    override residual is zero unless a test deliberately plays a driver: `steer_echo` /
    `wheel_by_echo` report a wheel position instead of the commanded one, `brake_bias` adds a
    pedal press.

    `expect_disengage` is a `player_*` reason. When set, SIGKILL waits for that reason on
    `gvd_engage.json` plus `gvd_state.json` or this run's refusal line. `cmd_applied` is
    already true a few ticks after engage, before the dwell. Windows PIPE + TerminateProcess
    also drops `[GVD] DISENGAGED:` (Verify FAIL: only boot lines through the preview WARNING),
    so stdout goes to a temp file, never a PIPE; the snapshot's `refusal` is the line from it.

    A caller's `engage:true` written before the call is held off the bus until this
    supervisor's first tick, then written fresh: Alt+G comes after start.
    """
    assert not (echo_clock and freeze_echo_after_s is not None), "echo_clock would hide the frozen echo"
    step_sleep = STEP_SLEEP_S if loop_sleep_s is None else float(loop_sleep_s)
    launcher = ROOT / "scripts" / "gvd_echo_clock_child.py" if echo_clock else ROOT / "python" / "run_vision.py"
    args = [sys.executable, "-u", str(launcher), "--backend", "stub", "--hz", "20", "--encode", "cpu", *extra]
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    clock_dir = tempfile.mkdtemp(prefix="gvd-m6-echo-clock-") if echo_clock else None
    clock_file = Path(clock_dir) / "echo_clock.json" if clock_dir else None
    mod_t = 0.0

    def _write_clock(t: float) -> None:
        tmp = clock_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"t": t}), encoding="utf-8")
        os.replace(tmp, clock_file)

    if clock_file is not None:
        _write_clock(mod_t)
        env["GVD_TEST_ECHO_CLOCK_FILE"] = str(clock_file)
    _touch_lua_bus()
    # A previous run's command / state must not be read as this run's (the mod would hold and
    # echo an old command; the latch would take an old refusal).
    for stale in (cmd_path(), state_path()):
        try:
            stale.unlink()
        except FileNotFoundError:
            pass
    # Alt+G comes after this supervisor is up, never before it: a latch already on at spawn
    # belongs to a previous session. A caller's engage:true is held off the bus until this
    # supervisor's first tick, then written fresh (the player's press).
    engage_after_start = _read_json(engage_path()).get("engaged") is True
    if engage_after_start:
        try:
            engage_path().unlink()
        except FileNotFoundError:
            pass
    t_spawn = time.time()  # wall: compared with the files' own stamps
    m_spawn = time.monotonic()  # monotonic: every deadline and window below
    # Never PIPE. A full pipe freezes the supervisor (engaged, cmd_json_pending), and
    # Windows TerminateProcess discards the child's unflushed stdout anyway. stdout goes to a
    # temp file instead: run_vision prints its refusal / disengage lines with flush=True under
    # -u, so each line is on disk before the tick goes on, and nothing later rewrites it.
    out_dir = tempfile.mkdtemp(prefix="gvd-m6-stdout-")
    out_file = Path(out_dir) / "run_vision.out"
    out_fh = open(out_file, "wb")
    proc = subprocess.Popen(
        args,
        cwd=str(ROOT),
        env=env,
        stdout=out_fh,
        stderr=subprocess.DEVNULL,
    )
    latched: dict | None = None
    loose: dict | None = None  # a matching snapshot whose state no longer names the reason
    t_start = t_spawn  # replaced by t_ready once this supervisor's first tick lands
    m_start = m_spawn
    ready = False
    frozen: dict | None = None
    dropped = False
    next_echo = 0.0
    held_cmd: dict | None = None  # the last command the mod received

    def _ack_once() -> None:
        nonlocal frozen, dropped, next_echo, held_cmd, ready, mod_t
        if on_engage_drop is not None and not dropped:
            if _read_json(engage_path()).get("engaged") is False:
                dropped = True
                on_engage_drop(time.time())
        if not ack:
            return
        now = time.time()
        mnow = time.monotonic()
        if frozen is not None:
            # Electrics echo stopped: the file keeps a fresh mtime, echo_seq does not move.
            _write_ego(**frozen)
            return
        if mnow < next_echo:
            return
        blind = ready and cmd_blind is not None and cmd_blind[0] <= mnow - m_start < cmd_blind[1]
        if not blind or held_cmd is None:
            try:
                got = json.loads(cmd_path().read_text(encoding="utf-8"))
                if int(got.get("seq", -1)) >= 0 and float(got.get("heartbeat_mtime") or 0.0) >= t_spawn - 0.5:
                    held_cmd = got
            except Exception:
                pass
        next_echo = max(next_echo + FAKE_ECHO_PERIOD_S, mnow + FAKE_ECHO_PERIOD_S / 2)
        if clock_file is not None:
            mod_t += FAKE_ECHO_PERIOD_S  # one echo = one mod echo period, however late it is
            _write_clock(mod_t)
        if held_cmd is None:
            # The mod echoes at 10 Hz whether or not a command has arrived yet: the car's
            # speed with nothing applied (applied_seq 0, not applying). The old harness only
            # echoed once *a* gvd_cmd.json existed, so a previous run's file started it.
            _write_ego(0, applying=False, speed=5.5, stamp_echo=stamp_echo, echo_seq=_next_echo_seq(),
                       mod_version=mod_version)
            return
        cmd = held_cmd
        seq = int(cmd.get("seq", -1))
        k = _next_echo_seq()
        if wheel_by_echo is not None:
            steer_in = wheel_by_echo(k)
        elif steer_echo is not None:
            steer_in = steer_echo(seq)
        else:
            steer_in = float(cmd.get("steer", 0.0))
        kw = dict(
            seq=seq,
            applying=True,
            speed=5.5,
            steer_in=steer_in,
            throttle_in=float(cmd.get("throttle", 0.0)),
            brake_in=min(1.0, float(cmd.get("brake", 0.0)) + brake_bias),
            stamp_echo=stamp_echo,
            echo_seq=k,
            mod_version=mod_version,
        )
        _write_ego(**kw)
        if ready and freeze_echo_after_s is not None and mnow - m_start >= freeze_echo_after_s:
            frozen = kw
            if clock is not None:
                clock["t_frozen"] = now

    def _refusal_line() -> dict | None:
        """This run's stdout line naming `expect_disengage`: append-only, never overwritten.

        gvd_state.json names the reason on the refusal tick only; the next tick rewrites it to
        not_engaged. The supervisor's `[GVD] REFUSE ENGAGE:` / `[GVD] DISENGAGED:` line stays.
        """
        try:
            text = out_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        for line in text.splitlines():
            for tag in ("[GVD] REFUSE ENGAGE: ", "[GVD] DISENGAGED: "):
                if not line.startswith(tag):
                    continue
                rest = line[len(tag):]
                if rest == expect_disengage or rest.startswith((f"{expect_disengage}:", f"{expect_disengage} ")):
                    return {"reason": expect_disengage, "line": line}
        return None

    def _try_latch() -> bool:
        """True once a snapshot of this run names the expected disengage on every surface.

        gvd_state.json counts when it still names the reason (the refusal tick), or when this
        run's stdout has the refusal line for it (the tick after, state says not_engaged).
        """
        nonlocal latched, loose
        if not expect_disengage:
            return False
        eng = _read_json(engage_path())
        cmd = _read_json(cmd_path())
        st = _read_json(state_path())
        if eng.get("engaged") is not False or eng.get("disengage_reason") != expect_disengage:
            return False
        try:
            if float(eng.get("mtime") or 0.0) < t_spawn:
                return False  # an engage file from before this run
        except (TypeError, ValueError):
            return False
        if not st or not cmd:
            return False  # removed before spawn: this run has not written them yet
        if st.get("engaged") is not False:
            return False
        if cmd.get("engaged") is not False:
            return False
        snap = {"engage": eng, "cmd": cmd, "state": st, "refusal": _refusal_line()}
        ovr = st.get("override") if isinstance(st.get("override"), dict) else {}
        if st.get("disengage_reason") == expect_disengage or ovr.get("reason") == expect_disengage:
            latched = snap
            return True
        if snap["refusal"] is not None:
            latched = snap
            return True
        # A later tick already rewrote state.disengage_reason to not_engaged. The engage file
        # is the sticky record, so keep this one in case the naming tick is never caught.
        if loose is None:
            loose = snap
        return False

    last_engage_hb = -1e9

    def _touch_engage_heartbeat() -> None:
        """Stand in for Lua's engage-file heartbeat so a multi-second run stays latched."""
        nonlocal last_engage_hb
        if not engage_heartbeat:
            return
        now = time.time()
        if time.monotonic() - last_engage_hb < 1.0:
            return
        try:
            data = json.loads(engage_path().read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict) or data.get("engaged") is not True:
            return
        try:
            again = json.loads(engage_path().read_text(encoding="utf-8"))
        except Exception:
            return
        if again.get("engaged") is not True:
            return
        again["mtime"] = now
        engage_path().write_text(json.dumps(again), encoding="utf-8")
        last_engage_hb = time.monotonic()

    try:
        _touch_lua_bus()
        # Startup: wait for this supervisor's first tick (its gvd_cmd.json and gvd_state.json;
        # both were removed before spawn). Not part of `seconds`.
        startup_end = m_spawn + SUPERVISOR_STARTUP_S
        while not (cmd_path().exists() and state_path().exists()):
            if proc.poll() is not None:
                raise AssertionError(f"run_vision exited during startup ({proc.returncode})")
            if time.monotonic() > startup_end:
                raise AssertionError(f"run_vision wrote no gvd_cmd.json / gvd_state.json within {SUPERVISOR_STARTUP_S:.0f}s of spawn")
            _touch_lua_bus()
            _touch_engage_heartbeat()
            _ack_once()
            time.sleep(0.01)
        t_start = time.time()
        m_start = time.monotonic()
        ready = True
        if engage_after_start:
            write_engage_flag(True)
        if clock is not None:
            clock.update({"t_spawn": t_spawn, "t_ready": t_start})
        m_end = m_start + seconds
        while time.monotonic() < m_end:
            if proc.poll() is not None:
                raise AssertionError(f"run_vision exited early ({proc.returncode})")
            _touch_lua_bus()
            _touch_engage_heartbeat()
            _ack_once()
            if report is not None:
                hb = _read_json(cmd_path()).get("heartbeat_mtime")
                if isinstance(hb, (int, float)):
                    report.setdefault("hb_age_s", []).append(time.time() - float(hb))
            if _try_latch():
                break
            # Tighter than the 20 Hz supervisor loop so the echo always belongs to the newest seq.
            time.sleep(0.01)
            if step_sleep > 0.0:
                time.sleep(step_sleep)
        # Hold SIGKILL until write_state has seen the echo. Killing mid-tick left
        # cmd_applied=False / cmd_json_pending for check_run_vision_e2e.
        # Override runs must not stop on cmd_applied: that flips true before the dwell.
        grace = 1.5 if expect_disengage and latched is None else (0.5 if ack and expect_disengage is None else 0.0)
        until = time.monotonic() + grace
        while time.monotonic() < until:
            if proc.poll() is not None:
                break
            _touch_lua_bus()
            _touch_engage_heartbeat()
            _ack_once()
            if _try_latch():
                break
            if expect_disengage is None and ack:
                try:
                    st = read_state() or {}
                    if st.get("cmd_applied") is True and st.get("cmd_reason") == "cmd_json_applied":
                        break
                except Exception:
                    pass
            time.sleep(0.01)
    finally:
        proc.kill()  # SIGKILL: the `finally` stop never runs, so the last tick's files survive for assertions
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        if clock_dir is not None:
            shutil.rmtree(clock_dir, ignore_errors=True)
        out_fh.close()
        if report is not None:
            report["stdout"] = out_file.read_text(encoding="utf-8", errors="replace") if out_file.exists() else ""
        shutil.rmtree(out_dir, ignore_errors=True)
    return latched if latched is not None else loose


def check_run_vision_e2e() -> None:
    """Supervisor loop end-to-end (stub cameras): engaged writes drive cmds, disengaged writes idle stops."""
    _touch_lua_bus()
    write_engage_flag(False)
    if ego_path().exists():
        ego_path().unlink()

    # Not engaged → every tick is a stop with engaged=false (Lua keeps its hands off).
    _run_supervisor([], seconds=2.0)
    cmd = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert cmd["engaged"] is False and cmd["brake"] == 1.0 and cmd["throttle"] == 0.0 and cmd["seq"] >= 5, cmd
    assert cmd["reason"] == "not_engaged"
    st = read_state()
    assert st and st["engaged"] is False and st["cmd_applied"] is False and st["actuator"] == "cmd_json"
    assert st["cmd_reason"] == "not_engaged" and st["ego_source"] == "none", st
    # Loaded nets keep writing path / tracks / shadow while the car is still yours.
    assert isinstance(st.get("path_ego"), list) and len(st["path_ego"]) >= 2, st.get("path_ego")
    assert isinstance(st.get("tracks"), list)
    assert isinstance(st.get("shadow"), dict) and "steer" in st["shadow"]
    assert st.get("detector")

    # Engaged with no lane path (stub) → preview gate holds the brake, still engaged for Lua.
    # The fake mod echoes (ack=True): engaging needs a fresh session echo (#125).
    _run_supervisor(["--force-engage"], seconds=2.0, ack=True)
    cmd = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert cmd["engaged"] is True and cmd["brake"] == 1.0 and cmd["throttle"] == 0.0 and cmd["reason"] == "preview_blocked", cmd

    # No echo this session: --force-engage refuses with ego_unknown, thr 0 / brk 0 (#125).
    if ego_path().exists():
        ego_path().unlink()
    _run_supervisor(["--force-engage"], seconds=2.0)
    cmd = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert cmd["engaged"] is False and cmd["brake"] == 0.0 and cmd["throttle"] == 0.0 and cmd["reason"] == "ego_unknown", cmd
    st = read_state()
    assert st and st["engaged"] is False and st.get("disengage_reason") == "ego_unknown", st and st.get("disengage_reason")

    # Engaged + preview allowed, still no cameras. Since the lane-fit path
    # replaced the 36 m steer fallback, an empty view is a stop: target_v is 0,
    # the fake echo is 5.5 m/s, and err < -1 selects
    # brake = min(0.6, 0.05 * echo speed). Throttle stays 0. Steer stays 0.
    # The ack path is unchanged.
    echo_speed = 5.5
    expected_brake = min(0.6, 0.05 * echo_speed)
    from python.control.actuate import plan_command

    law = plan_command(
        path_ego=[],
        planner={"target_v": 0.0, "aeb": "off"},
        ego_speed_mps=echo_speed,
        seq=1,
    )
    assert law.throttle == 0.0 and law.steer == 0.0, law
    assert abs(law.brake - expected_brake) < 1e-9, law
    _run_supervisor(["--force-engage", "--allow-preview-drive"], seconds=2.5, ack=True)
    cmd = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert cmd["engaged"] is True and cmd["reason"] == "ok" and cmd["seq"] >= 10, cmd
    assert cmd["throttle"] == 0.0, cmd
    assert abs(float(cmd["brake"]) - expected_brake) < 1e-4, cmd
    assert cmd["steer"] == 0.0, cmd
    st = read_state()
    assert st and st["engaged"] is True and st["cmd_applied"] is True, {k: st.get(k) for k in ("engaged", "cmd_applied", "cmd_reason")}
    assert st["cmd_reason"] == "cmd_json_applied" and st["ego_source"] == "lua" and st["lua_applying"] is True
    assert abs(float(st["ego"]["speed_mps"]) - echo_speed) < 1e-6, st["ego"]
    assert st["cmd_ack_seq"] >= st["cmd_seq"] - 5
    planner = st.get("planner") if isinstance(st.get("planner"), dict) else {}
    assert planner.get("prediction") == "none", planner
    assert float(planner.get("target_v") or 0.0) == 0.0, planner
    assert abs(float(planner.get("pred_brake") or 0.0) - expected_brake) < 1e-4, planner
    assert not st.get("lanes_bev"), st.get("lanes_bev")
    assert float(st.get("lane_conf") or 0.0) == 0.0, st.get("lane_conf")
    assert st.get("path_debug_preview") is True, st.get("path_debug_preview")

    write_engage_flag(False)
    ego_path().unlink()


def check_synthetic_lanes_drive() -> None:
    """Engaged with a real two-lane frame commands throttle. The stub e2e does not.

    The no-camera supervisor check above is a stop. This one feeds the same
    hood view the lane-fit regression uses, through perception and the modular
    policy, and requires throttle with the brake released.
    """
    import test_lanes_hough_int as lanes_fix
    from python.control.e2e import E2EPolicy
    from python.perception.detect import EmptyDetector
    from python.perception.pipeline import ModularPerception
    from python.runtime.shadow import shadow_tick

    road = lanes_fix._yellow_white_road(
        asphalt=(110, 108, 105),
        yellow=(40, 210, 235),
        white=(245, 245, 245),
        width=4,
    )
    perc = ModularPerception(allow_synthetic=False, detector_id="empty")
    perc.set_detector(EmptyDetector(), ["yolo_weights"])
    echo_speed = 5.5
    pout = perc.tick(road, ego_speed_mps=echo_speed)
    assert float(pout.lane_conf) > 0.0 and len(pout.lanes_bev) >= 2, (pout.lane_conf, pout.lanes_bev)
    assert pout.path_debug_preview is False, pout.planner
    assert float(pout.planner.get("target_v") or 0.0) > echo_speed, pout.planner
    held = shadow_tick(
        policy="modular",
        engaged=True,
        heartbeat_ok=True,
        path_debug_preview=bool(pout.path_debug_preview),
        allow_preview_drive=False,
        path_ego=pout.path_ego,
        planner=pout.planner,
        ego_speed_mps=echo_speed,
        lane_conf=float(pout.lane_conf),
        path_conf=float(pout.path_conf),
        seq=1,
        e2e_policy=E2EPolicy(),
        main_bgr=road,
    )
    cmd = held.applied
    assert cmd.reason == "ok", cmd
    assert cmd.throttle > 0.0 and cmd.brake == 0.0, cmd
    assert -1.0 <= cmd.steer <= 1.0, cmd
    assert held.should_disengage is False


def _engage_live_on_last_tick() -> tuple[bool, dict]:
    """Was the Alt+G latch live on the supervisor's last tick?

    Freshness is judged on that tick's own clock: ``heartbeat_mtime`` of the last
    gvd_cmd.json it wrote, not the wall time when the test gets here. SIGKILL, the
    wait and a loaded runner all sit between the two, and none of it is GVD's doing.
    Returns the verdict and what it was judged on.
    """
    cmd = _read_json(cmd_path())
    eng = _read_json(engage_path())
    tick = float(cmd.get("heartbeat_mtime") or 0.0)
    live = tick > 0.0 and read_engage_flag(default=False, now=tick)
    return live, {
        "engage": eng,
        "age_at_last_tick_s": round(tick - float(eng.get("mtime") or 0.0), 3) if tick else None,
        "fresh_s": ENGAGE_FRESH_S,
        "cmd": {k: cmd.get(k) for k in ("seq", "engaged", "reason")},
    }


def _ffb_chatter_keeps_driving(loop_sleep_s: float | None = None) -> None:
    """FFB chatter on the echo (echo gate on the fake mod's clock) must not disengage GVD."""
    _touch_lua_bus()
    if ego_path().exists():
        ego_path().unlink()

    # Force-feedback chatter — alternating kicks, the sign flipping on every 10 Hz echo.
    write_engage_flag(True)
    # Keyed on the echo (one FFB sample each), not on the command seq the echo acks.
    _run_supervisor(
        ["--allow-preview-drive"], seconds=3.0, ack=True, wheel_by_echo=lambda k: 0.9 if k % 2 else -0.9,
        echo_clock=True, loop_sleep_s=loop_sleep_s,
    )
    live, why = _engage_live_on_last_tick()
    assert live is True, ("force-feedback chatter disengaged GVD (or the latch was stale on its last tick)", why)
    assert _read_json(engage_path()).get("disengage_reason") == "none"
    cmd = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert cmd["engaged"] is True and cmd["reason"] == "ok", cmd
    st = read_state()
    assert st and st["engaged"] is True and st["disengage_reason"] == "none", {
        k: st.get(k) for k in ("engaged", "disengage_reason")
    }
    assert st["override"]["channel"] == "none" and st["override"]["armed"] is True, st["override"]
    assert st["override"]["steer_held_ms"] < st["override_cfg"]["steer_hold_ms"], st["override"]
    # No sink regression: the mod's ack path still drives while the wheel chatters.
    assert st["cmd_applied"] is True and st["cmd_reason"] == "cmd_json_applied", st["cmd_reason"]


def check_ffb_override_live() -> None:
    """Whole loop with a wheel on the echo: chatter keeps driving, a real driver gets the car."""
    _ffb_chatter_keeps_driving()

    # A driver actually holding the wheel over. Latch on gvd_engage.json / gvd_state.json,
    # not stdout — Windows PIPE + SIGKILL only kept boot lines through the preview WARNING.
    write_engage_flag(True)
    snap = _run_supervisor(
        ["--allow-preview-drive"],
        seconds=3.0,
        ack=True,
        steer_echo=lambda _seq: 0.25,
        expect_disengage="player_steer",
        echo_clock=True,
    )
    assert snap is not None, "player_steer never appeared on gvd_engage.json / gvd_state.json"
    eng = snap["engage"]
    st = snap["state"]
    assert read_engage_flag(default=True) is False, "a real steer takeover must disengage"
    assert eng.get("disengage_reason") == "player_steer", eng
    assert st.get("engaged") is False, {k: st.get(k) for k in ("engaged", "disengage_reason", "override")}
    assert _player_named("player_steer", engage=eng, state=st), {
        "engage": eng.get("disengage_reason"),
        "state": st.get("disengage_reason"),
        "override": st.get("override"),
    }
    assert json.loads(cmd_path().read_text(encoding="utf-8"))["engaged"] is False

    # Pedals stay tight: a brake press on top of the command is an override with no dwell.
    write_engage_flag(True)
    snap = _run_supervisor(
        ["--allow-preview-drive"],
        seconds=2.0,
        ack=True,
        brake_bias=0.4,
        expect_disengage="player_brake",
        echo_clock=True,
    )
    assert snap is not None, "player_brake never appeared on gvd_engage.json / gvd_state.json"
    eng = snap["engage"]
    st = snap["state"]
    assert read_engage_flag(default=True) is False, "a real brake press must disengage"
    assert eng.get("disengage_reason") == "player_brake", eng
    assert st.get("engaged") is False, {k: st.get(k) for k in ("engaged", "disengage_reason", "override")}
    assert _player_named("player_brake", engage=eng, state=st), {
        "engage": eng.get("disengage_reason"),
        "state": st.get("disengage_reason"),
        "override": st.get("override"),
    }

    # Mirrored thresholds reach the mod through gvd_state.json.
    from python.control.override import config_mirror, load_override_config

    want = config_mirror(load_override_config(yaml.safe_load((ROOT / "config" / "control.yaml").read_text(encoding="utf-8"))))
    assert read_state()["override_cfg"] == want, read_state()["override_cfg"]

    write_engage_flag(False)
    ego_path().unlink()


def check_ffb_override_live_slow_host() -> None:
    """A slow host (SLOW_HOST_SLEEP_S real sleep per loop pass) must not turn chatter into echo_stale.

    On main before the echo clock this disengaged with echo_stale: the fake mod's echoes came
    every ~0.36 s of real time, over ECHO_STALE_MS, although the mod echoes at 10 Hz of its time.
    """
    _ffb_chatter_keeps_driving(loop_sleep_s=SLOW_HOST_SLEEP_S)
    print(f"  slow host ({SLOW_HOST_SLEEP_S:.2f} s real sleep per loop pass): FFB chatter still driving, no echo_stale")


def check_echo_gate_clock_default() -> None:
    """Production keeps the echo gate on time.monotonic() and ECHO_STALE_MS at 300; only tests swap the clock."""
    from python.control import echo_link

    src = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    call = "echo = echo_gate.observe(engaged=bool(engaged) and echo_bus, ego_fb=ego_fb, now=time.monotonic())"
    assert src.count(call) == 1, "run_vision must feed the echo gate time.monotonic()"
    assert src.count("echo_gate.observe(") == 1, "one echo gate call site"
    assert src.count("import time\n") == 1 and "time.monotonic =" not in src, "run_vision's time is the stdlib clock"
    assert echo_link.ECHO_STALE_MS == 300.0 and echo_link.GAP_RUN_MAX == 3
    assert echo_link.EchoGate().stale_s == 0.3, echo_link.EchoGate().stale_s
    # The swap lives only in the test launcher, and nothing in python/ patches the gate.
    for path in (ROOT / "python").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "EchoGate.observe =" not in text and "GVD_TEST_ECHO_CLOCK_FILE" not in text, path
    child = (ROOT / "scripts" / "gvd_echo_clock_child.py").read_text(encoding="utf-8")
    assert "echo_link.EchoGate.observe = _observe" in child


def _clock_branch(tree: ast.AST) -> dict[int, str | None]:
    """id(node) -> "clock" / "default" when it sits under `clock is (not) None`, else None."""
    out: dict[int, str | None] = {}

    def side(test: ast.AST) -> str | None:
        if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "clock"
                and len(test.ops) == 1 and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value is None):
            return "is" if isinstance(test.ops[0], ast.Is) else "isnot" if isinstance(test.ops[0], ast.IsNot) else None
        return None

    def walk(node: ast.AST, where: str | None) -> None:
        out[id(node)] = where
        if isinstance(node, (ast.If, ast.IfExp)) and side(node.test):
            on_body = "default" if side(node.test) == "is" else "clock"
            on_else = "clock" if on_body == "default" else "default"
            walk(node.test, where)
            body = node.body if isinstance(node.body, list) else [node.body]
            orelse = node.orelse if isinstance(node.orelse, list) else [node.orelse]
            for child in body:
                walk(child, on_body)
            for child in orelse:
                walk(child, on_else)
            return
        for child in ast.iter_child_nodes(node):
            walk(child, where)

    walk(tree, None)
    return out


def _call_name(call: ast.Call) -> str:
    fn = call.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return fn.id if isinstance(fn, ast.Name) else ""


def check_supervisor_clock_default() -> None:
    """With no clock, run_vision and the modules it threads keep every stdlib clock call.

    The `clock` seam (main(clock=None), ModularPerception(clock=None)) is test-only; it
    exists for scripts/gvd_clock_child.py. This pins each touched site and probes each
    threaded `now=` default, so a default path cannot drift onto the test clock.
    """
    import inspect

    from python import run_vision
    from python.control import actuate, echo_link, launch_guard, override
    from python.perception import pipeline
    from python.runtime import paths

    src = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    where = _clock_branch(tree)

    # 1. The seam defaults to None and the entry point passes nothing.
    assert inspect.signature(run_vision.main).parameters["clock"].default is None
    assert src.endswith('\n\nif __name__ == "__main__":\n    main()\n'), src[-80:]

    # 2. The real loop sleep is there once, after (not inside) the clock barrier.
    sleep = "            time.sleep(max(0.0, dt - (time.perf_counter() - loop_t0)))\n"
    assert src.count(sleep) == 1
    barrier = (
        "            if clock is not None:\n"
        "                # Test step barrier: report this tick done, block until the test releases the next.\n"
        "                clock.sleep(max(0.0, dt - (clock.perf() - loop_t0)))\n"
        "                continue\n"
    )
    assert src.count(barrier + sleep) == 1, "clock barrier must sit just before the real sleep"

    # 3. Loop timing seeds read time.perf_counter; the clock only overrides them after.
    for pin in (
        "    last_cam_t = time.perf_counter()\n    last_loop_t = time.perf_counter()\n"
        "    if clock is not None:\n        last_cam_t = last_loop_t = clock.perf()\n",
        "            loop_t0 = time.perf_counter()\n"
        "            if clock is not None:\n                loop_t0 = clock.perf()\n",
    ):
        assert src.count(pin) == 1, pin

    # 4. (R7) Both echo gates stay on time.monotonic, outside any clock branch.
    for line in (
        "session_echo.observe(ego_fb=ego_fb, now=time.monotonic())",
        "echo = echo_gate.observe(engaged=bool(engaged) and echo_bus, ego_fb=ego_fb, now=time.monotonic())",
    ):
        assert src.count(line) == 1, line
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "observe":
            assert where[id(node)] is None, ast.get_source_segment(src, node)

    # 5. Perception: clock defaults to None and frame_age_s follows pipeline's time.monotonic.
    assert inspect.signature(pipeline.ModularPerception.__init__).parameters["clock"].default is None
    ticks = iter((100.0, 100.0, 100.5))

    class _Mono:
        def monotonic(self) -> float:
            return next(ticks)

        def __getattr__(self, name: str):
            return getattr(time, name)

    perc = pipeline.ModularPerception(allow_synthetic=False, detector_id="empty")
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    real_time = pipeline.time
    pipeline.time = _Mono()
    try:
        perc.tick(frame, ego_speed_mps=5.0)
        perc.tick(frame, ego_speed_mps=5.0)
        age = perc.tick(frame, ego_speed_mps=5.0).frame_age_s
    finally:
        pipeline.time = real_time
    assert abs(age - 0.5) < 1e-9, age

    # 6. Override: no clock passes now=None, so the warmup dwell follows override's time.monotonic.
    for call in ("override.update(", "override.note_command("):
        assert src.count(call) == 1, call
    assert src.count("now=None if clock is None else clock.mono(),") == 2
    mono = [50.0]

    class _OvMono:
        def monotonic(self) -> float:
            return mono[0]

        def __getattr__(self, name: str):
            return getattr(time, name)

    det = override.OverrideDetector(override.OverrideConfig())
    real_ov = override.time
    override.time = _OvMono()
    try:
        det.update(engaged=False, applied_seq=1)
        last_t = det._last_t
        det.note_command(seq=1, steer=0.0, throttle=0.0, brake=0.0)
    finally:
        override.time = real_ov
    warm = override.OverrideConfig().warmup_s
    assert det._armed_at == 50.0 and last_t == 50.0, (det._armed_at, last_t)
    assert not det.armed(50.0 + warm - 0.01) and det.armed(50.0 + warm + 0.01), warm

    # 7. Each threaded `now=` defaults to time.time(), and RV passes it only on the clock branch.
    threaded = {"read_ego_feedback": 1, "read_engage_flag": 2, "poll": 1, "bus_identity": 2}
    with_now = {name: 0 for name in threaded}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in threaded:
            has_now = any(k.arg == "now" for k in node.keywords)
            seg = ast.get_source_segment(src, node)
            if has_now:
                assert where[id(node)] == "clock", seg
                with_now[_call_name(node)] += 1
            else:
                assert where[id(node)] != "clock", seg
    assert with_now == threaded, with_now
    with fake_clock() as fc:
        t0 = fc.now
        _touch_lua_bus()
        _write_ego(1)
        write_engage_flag(True)
        actuate.write_bot_engage(True, mtime=t0)
        assert read_ego_feedback() is not None and read_ego_feedback().fresh
        assert read_engage_flag(default=False) is True
        assert actuate.BotEngage().poll() is True
        assert paths.bus_identity().note == "ok", paths.bus_identity().note
        fc.advance(max(actuate.EGO_FRESH_S, ENGAGE_FRESH_S, paths.LUA_BUS_FRESH_S) + 0.5)
        fb = read_ego_feedback()
        assert fb is None or not fb.fresh, fb
        assert read_engage_flag(default=False) is False
        assert actuate.BotEngage().poll() is False
        assert paths.bus_identity().note != "ok", paths.bus_identity().note
        actuate.write_bot_engage(False, mtime=fc.now)
        write_engage_flag(False)
        ego_path().unlink()

    # 8. Nothing in python/ reads a test-clock env var or patches `time`.
    for path in (ROOT / "python").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for bad in ("GVD_TEST_SIM_CLOCK", "GVD_TEST_ECHO_CLOCK_FILE",
                    "time.monotonic =", "time.perf_counter =", "time.time =", "time.sleep ="):
            assert bad not in text, (path, bad)

    # 9. Rule constants are unchanged.
    assert (launch_guard.LAUNCH_S, launch_guard.LAUNCH_SPEED_MPS, launch_guard.STUCK_S,
            launch_guard.STALE_LANE_S, launch_guard.STALE_SPEED_MPS) == (6.0, 4.0, 1.5, 0.25, 6.0)
    assert (echo_link.ECHO_STALE_MS, echo_link.GAP_RUN_MAX) == (300.0, 3)
    assert (actuate.EGO_FRESH_S, actuate.ENGAGE_FRESH_S, actuate.HEARTBEAT_STALE_S,
            actuate.SOFT_ESC_FILE_PERIOD_S, paths.LUA_BUS_FRESH_S) == (1.0, 2.5, 1.5, 0.10, 1.0)
    assert override.OverrideConfig() == override.OverrideConfig(
        steer_enter=0.08, steer_exit=0.04, steer_hold_ms=200.0, steer_spike=0.20,
        brake_enter=0.06, throttle_enter=0.10, lpf_tau_ms=80.0)

    # 10. The launcher's bus_identity wrap and actuate.time shim are clock-only: stdlib by default
    # and with no clock.
    from python.runtime import state_io
    from gvd_sim_clock import install_actuate_time_shim, install_bus_identity_wrap

    assert actuate.bus_identity is paths.bus_identity
    assert state_io.bus_identity is paths.bus_identity
    assert actuate.time is time
    install_bus_identity_wrap(None)
    install_actuate_time_shim(None)
    assert actuate.bus_identity is paths.bus_identity
    assert state_io.bus_identity is paths.bus_identity
    assert actuate.time is time
    print("  no clock: every seamed site keeps its stdlib clock; threaded now= defaults to time.time()")


def check_ffb_stale_engage_heartbeat() -> None:
    """The chatter check's latch test is load-bearing: a stale Alt+G heartbeat fails it.

    Same 3 s chatter run as check_ffb_override_live, but the mod's engage heartbeat
    stops, so gvd_engage.json stays engaged:true with the stamp written before spawn.
    Nothing writes engaged:false; only freshness tells it apart from a live latch. The
    supervisor must stop driving on it, and `_engage_live_on_last_tick` must say not live.
    """
    _touch_lua_bus()
    if ego_path().exists():
        ego_path().unlink()

    # The clock the verdict uses is the one passed in: a stamp inside the window is
    # live, one past ENGAGE_FRESH_S is not, whatever the wall time is now.
    write_engage_flag(True)
    stamp = float(_read_json(engage_path())["mtime"])
    assert read_engage_flag(default=False, now=stamp + ENGAGE_FRESH_S - 0.1) is True
    assert read_engage_flag(default=True, now=stamp + ENGAGE_FRESH_S + 0.1) is False
    assert read_engage_flag(default=True, now=stamp - 1.5) is False  # far-future stamp

    write_engage_flag(True)
    _run_supervisor(
        ["--allow-preview-drive"],
        seconds=3.0,
        ack=True,
        wheel_by_echo=lambda k: 0.9 if k % 2 else -0.9,
        engage_heartbeat=False,
    )
    live, why = _engage_live_on_last_tick()
    eng = why["engage"]
    # Still engaged:true on disk, nobody disengaged: only the stamp is old.
    assert eng.get("engaged") is True and eng.get("disengage_reason") == "none", eng
    assert why["age_at_last_tick_s"] is not None and why["age_at_last_tick_s"] > ENGAGE_FRESH_S, why
    assert live is False, ("a stale engage heartbeat read as a live latch", why)
    # The supervisor judged it the same way on that tick: it stopped driving.
    assert why["cmd"]["engaged"] is False and why["cmd"]["reason"] == "not_engaged", why
    print(f"stale engage heartbeat: age {why['age_at_last_tick_s']:.2f}s > {ENGAGE_FRESH_S}s at the last tick -> not live, not driving OK")


def check_echo_outdated_refuses_engage() -> None:
    """A mod without echo_seq cannot tell a repeat from a new sample: engage is refused, by name."""
    _touch_lua_bus()
    if ego_path().exists():
        ego_path().unlink()
    write_engage_flag(True)
    snap = _run_supervisor(
        ["--allow-preview-drive"],
        seconds=3.0,
        ack=True,
        stamp_echo=False,
        expect_disengage="lua_echo_outdated",
    )
    assert snap is not None, "an echo without echo_seq never refused engage"
    assert read_engage_flag(default=True) is False, "an old mod must not keep GVD engaged"
    assert snap["engage"].get("disengage_reason") == "lua_echo_outdated", snap["engage"]
    assert snap["cmd"].get("engaged") is False, snap["cmd"]
    # No silent fallback: nothing was judged on the old filter path while it was refused.
    assert snap["state"].get("disengage_reason") in ("lua_echo_outdated", "not_engaged"), snap["state"]
    write_engage_flag(False)
    ego_path().unlink()


def check_echo_stale_disengages() -> None:
    """A frozen echo while engaged is a link fault (echo_stale), never 'no override'."""
    _touch_lua_bus()
    if ego_path().exists():
        ego_path().unlink()
    drop: list[float] = []
    write_engage_flag(True)
    freeze_at = 1.2
    clock: dict = {}
    snap = _run_supervisor(
        ["--allow-preview-drive"],
        seconds=3.5,
        ack=True,
        freeze_echo_after_s=freeze_at,
        expect_disengage="echo_stale",
        on_engage_drop=drop.append,
        clock=clock,
    )
    assert snap is not None, "a frozen echo never disengaged with echo_stale"
    assert snap["engage"].get("disengage_reason") == "echo_stale", snap["engage"]
    assert snap["cmd"].get("engaged") is False, snap["cmd"]
    assert read_engage_flag(default=True) is False
    # Latency on the two processes' own stamps: the mod's last advancing echo and the
    # supervisor's disengage write (gvd_engage.json mtime). How late the test's loop
    # noticed either one (a loaded runner) is not GVD's latency.
    assert "t_frozen" in clock, clock
    after = float(snap["engage"]["mtime"]) - clock["t_frozen"]
    seen = f", the test saw it {(drop[0] - clock['t_frozen']) * 1000.0:.0f} ms after" if drop else ""
    print(f"  echo_stale {after * 1000.0:.0f} ms after the last advancing echo{seen}")
    assert 0.0 < after < 1.5, f"echo_stale took {after:.2f}s after the echo froze"
    write_engage_flag(False)
    ego_path().unlink()


def check_echo_old_mod_version_refused() -> None:
    """echo_seq present but mod_version below the minimum: refused, and the refusal names it."""
    from python.control.echo_link import MIN_MOD_VERSION, parse_version

    assert parse_version(_MOD_VERSION) >= parse_version(MIN_MOD_VERSION), (_MOD_VERSION, MIN_MOD_VERSION)
    _touch_lua_bus()
    if ego_path().exists():
        ego_path().unlink()
    write_engage_flag(True)
    snap = _run_supervisor(
        ["--allow-preview-drive"],
        seconds=3.0,
        ack=True,
        mod_version="1.7.12",
        expect_disengage="lua_echo_outdated",
    )
    assert snap is not None, "a 1.7.12 mod_version never refused engage"
    assert snap["engage"].get("disengage_reason") == "lua_echo_outdated", snap["engage"]
    # The refusal itself, from the supervisor's own refusal line (stdout, append-only). The
    # refusal tick's gvd_state.json is rewritten to not_engaged one tick later, and the engage now
    # comes after start, so that tick is not reliably caught.
    st = snap["state"]
    refusal = snap.get("refusal") or {}
    assert refusal.get("reason") == "lua_echo_outdated", {"refusal": refusal, "state": st.get("disengage_reason")}
    echo = st.get("echo") or {}
    assert echo.get("min_mod_version") == MIN_MOD_VERSION, echo
    msg = str(refusal.get("line", ""))
    assert "mod 1.7.12" in msg and f"mod {MIN_MOD_VERSION} or newer is required" in msg, msg
    write_engage_flag(False)
    ego_path().unlink()


CMD_BLIND_S = (1.2, 2.4)  # commands stop reaching the mod in this window (seconds after start)
CMD_BLIND_RUN_S = 3.0
CMD_BLIND_ENGAGE_S = 0.5  # Alt+G on the SimClock run, once the echo gate has seen the mod (as check_09)


def check_echo_cmd_blind_not_stale() -> None:
    """Commands stop reaching the mod while its echoes keep advancing: no echo_stale.

    echo_seq is the mod's own counter. Keyed on the command seq it would freeze with the
    commands, and a live wheel would read as a dead link.

    Pass/fail runs on the SimClock step barrier (test_unknown_ego._session(clock=SimClock()),
    #155): one supervisor tick per 50 ms of clock time, the fake mod's 10 Hz echo and Alt+G
    heartbeat on the same clock. The cmd-blind window, the run length and ECHO_STALE_MS are
    unchanged; a loaded runner that descheduled the fake mod for over 300 ms of real time can
    no longer read as echo_stale. Real-clock timing is check_supervisor_tick_latency_report.
    """
    import test_unknown_ego as ue  # noqa: PLC0415 - the SimClock supervisor session and fake mod
    from gvd_sim_clock import SimClock  # noqa: PLC0415

    _touch_lua_bus()
    if ego_path().exists():
        ego_path().unlink()
    write_engage_flag(False)
    mod = ue._Mod()
    mod.echoing = True
    seqs: dict[str, list[int]] = {"cmd": [], "held": []}
    started = {"on": False}

    def step(t: float) -> None:
        # Alt+G after this supervisor is up. The real-clock harness presses on the first tick,
        # after seconds of startup echoes; on the clock startup is a few ticks, so the press
        # waits for the gate to see the mod, still well before the cmd-blind window.
        if not started["on"] and t >= CMD_BLIND_ENGAGE_S:
            started["on"] = True
            write_engage_flag(True)
        mod.blind = CMD_BLIND_S[0] <= t < CMD_BLIND_S[1]
        if mod.blind:
            seqs["cmd"].append(int(_read_json(cmd_path()).get("seq", 0) or 0))
            seqs["held"].append(int((mod.held or {}).get("seq", 0) or 0))

    seen = ue._session(CMD_BLIND_RUN_S, mod, step, clock=SimClock())
    # Load-bearing: the supervisor kept commanding while the mod held one command.
    cmd, held = seqs["cmd"], seqs["held"]
    assert cmd and cmd[-1] > cmd[0], ("no new commands in the blind window", cmd)
    assert len(set(held)) == 1 and held[0] > 0, ("the mod took a command while blind", held)
    assert ue._engaged_snaps(seen), "never engaged"
    eng = _read_json(engage_path())
    st = read_state() or {}
    assert not [s["state"].get("disengage_reason") for s in seen
                if "echo_stale" in (s["state"].get("disengage_reason"), s["engage"].get("disengage_reason"))], \
        "echo_stale while commands were blind"
    assert eng.get("disengage_reason") != "echo_stale", eng
    assert st.get("disengage_reason") != "echo_stale", st.get("disengage_reason")
    assert eng.get("engaged") is True, eng
    assert (st.get("echo") or {}).get("status") in ("fresh", "repeat"), st.get("echo")
    write_engage_flag(False)
    ego_path().unlink()


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]


def check_supervisor_tick_latency_report() -> None:
    """REPORT ONLY, never fails: real-clock supervisor tick time and command heartbeat age.

    The same cmd-blind run as check_echo_cmd_blind_not_stale, on the real clock (stdlib time,
    plain run_vision.py). Prints p50/p99 of the supervisor's own per-tick `heartbeat_ms` (its
    `[GVD] seg` line) and of the gvd_cmd.json heartbeat age the test saw, plus whether this
    host produced echo_stale. Under CI load these numbers move; that is what they are for. Pass
    or fail of the safety behaviour is the SimClock check, not this one.
    """
    rep: dict = {}
    try:
        _touch_lua_bus()
        if ego_path().exists():
            ego_path().unlink()
        write_engage_flag(True)
        _run_supervisor(["--allow-preview-drive"], seconds=CMD_BLIND_RUN_S, ack=True, cmd_blind=CMD_BLIND_S,
                        report=rep)
        tick_ms = [float(m) for m in re.findall(r"^\[GVD\] seg .*?\bheartbeat_ms=([0-9.]+)",
                                                rep.get("stdout", ""), re.M)]
        hb_ms = [a * 1000.0 for a in rep.get("hb_age_s", [])]
        stale = "echo_stale" in (str(_read_json(engage_path()).get("disengage_reason")),
                                 str((read_state() or {}).get("disengage_reason")))

        def ms(v: float | None) -> str:
            return "na" if v is None else f"{v:.1f}"

        print(f"  tick-latency report (real clock, report-only): ticks={len(tick_ms)} "
              f"tick_ms p50={ms(_pct(tick_ms, 0.50))} p99={ms(_pct(tick_ms, 0.99))} "
              f"heartbeat_age_ms p50={ms(_pct(hb_ms, 0.50))} p99={ms(_pct(hb_ms, 0.99))} "
              f"samples={len(hb_ms)} echo_stale={'yes' if stale else 'no'}", flush=True)
    except Exception as exc:  # noqa: BLE001 - report-only: a broken measurement is printed, never a failure
        print(f"  tick-latency report (real clock, report-only): unavailable ({type(exc).__name__}: {exc})"[:400],
              flush=True)
    finally:
        with contextlib.suppress(Exception):
            write_engage_flag(False)
        with contextlib.suppress(FileNotFoundError):
            ego_path().unlink()


def check_lua_harness() -> None:
    """Drive the mod's Lua through the stub-GE harness when a Lua 5.1 interpreter is available."""
    for exe in ("lua5.1", "luajit", "lua"):
        lua = shutil.which(exe)
        if lua:
            res = subprocess.run([lua, "scripts/test_m6_lua_cmd.lua"], cwd=str(ROOT), capture_output=True, text=True, timeout=60)
            assert res.returncode == 0 and "test_m6_lua_cmd: OK" in res.stdout, res.stdout + res.stderr
            bind = subprocess.run([lua, "scripts/test_gvd_bind_reload.lua"], cwd=str(ROOT), capture_output=True, text=True, timeout=60)
            assert bind.returncode == 0 and "test_gvd_bind_reload: OK" in bind.stdout, bind.stdout + bind.stderr
            docs = subprocess.run([lua, "scripts/test_gvd_docs_dir.lua"], cwd=str(ROOT), capture_output=True, text=True, timeout=30)
            assert docs.returncode == 0 and "test_gvd_docs_dir: OK" in docs.stdout, docs.stdout + docs.stderr
            link = subprocess.run([lua, "scripts/test_gvd_state_link.lua"], cwd=str(ROOT), capture_output=True, text=True, timeout=30)
            assert link.returncode == 0 and "test_gvd_state_link: OK" in link.stdout, link.stdout + link.stderr
            return
    print("  (lua interpreter not found — scripts/test_m6_lua_cmd.lua skipped)")


def check_engage_path_contract() -> None:
    p = engage_path()
    assert p.parts[-3:] == ("Documents", "GVD", "gvd_engage.json"), p
    assert p.parent == state_path().parent == cmd_path().parent
    # A leftover true from a crashed session is not a live Alt+G.
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"engaged":true,"mtime":1700000000}', encoding="utf-8")
    assert read_engage_flag(default=False) is False
    p.write_text('{"engaged":true}', encoding="utf-8")
    assert read_engage_flag(default=False) is False, "engaged:true without mtime must not latch"
    # A live latch on the fake clock: fresh through ENGAGE_FRESH_S, stale just past it.
    with fake_clock() as clock:
        p.write_text(json.dumps({"engaged": True, "mtime": clock.now}), encoding="utf-8")
        assert read_engage_flag(default=False) is True
        clock.advance(ENGAGE_FRESH_S)
        assert read_engage_flag(default=False) is True
        clock.advance(0.01)
        assert read_engage_flag(default=True) is False
        # The same file read at the stamp itself through the explicit clock argument.
        assert read_engage_flag(default=False, now=clock.now - ENGAGE_FRESH_S - 0.01) is True
    # Supervisor disengage payload: engaged=false with float mtime + reason (what Lua adopts as OFF and logs).
    write_engage_flag(False, disengage_reason="player_steer")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["engaged"] is False and isinstance(data["mtime"], float)
    assert data["disengage_reason"] == "player_steer"
    assert read_engage_flag(default=True) is False
    write_engage_flag(False)
    assert json.loads(p.read_text(encoding="utf-8"))["disengage_reason"] == "none"


def check_gvd_docs_dir() -> None:
    """Lua relative Documents/GVD VFS reads + Python Drive vs Tech current\\Documents\\GVD writes."""
    import test_gvd_docs_dir as docs_dir

    lua = (ROOT / "beamng_mod" / "lua" / "ge" / "extensions" / "gvd" / "main.lua").read_text(encoding="utf-8")
    docs_dir.check_source_contracts(lua)
    docs_dir.check_python_mirror()
    docs_dir.check_python_known_folder()
    docs_dir.check_writer_lua_folder_agreement()


def _py_string_literals(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                docstrings.add(id(body[0].value))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            out.append(node.value)
    return out


def check_no_chrome() -> None:
    # Player-facing mod files: strict.
    for f in (ROOT / "beamng_mod").rglob("*"):
        if f.suffix.lower() in (".lua", ".js", ".html", ".json"):
            text = f.read_text(encoding="utf-8", errors="ignore")
            assert not CHROME_RE.search(text), f"chrome in {f.relative_to(ROOT)}"
    # Launchers and the release tooling banner.
    for f in ("install.bat", "uninstall.bat", "play_gvd.bat", "play_gvd_tech.bat", "requirements-retail.txt"):
        assert not CHROME_RE.search((ROOT / f).read_text(encoding="utf-8", errors="ignore")), f
    # Runtime strings players see (titles, prints); docstrings/comments may still state the honesty rule.
    for f in [
        ROOT / "python" / "run_vision.py",
        ROOT / "python" / "runtime" / "hw_probe.py",
        ROOT / "python" / "runtime" / "debug_opts.py",
        ROOT / "python" / "runtime" / "models.py",
        ROOT / "python" / "sensors" / "tech.py",
        *(ROOT / "python" / "viz").glob("*.py"),
    ]:
        for s in _py_string_literals(f):
            assert not CHROME_RE.search(s), f"chrome string in {f.relative_to(ROOT)}: {s!r}"
    # Release VERSION.txt banner only names the honesty rule, never a product label.
    ver = rel.version_text("test")
    assert "Not Tesla FSD" in ver and "Full Self" not in ver


def check_player_docs() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    version = (ROOT / "VERSION").read_text(encoding="utf-8").splitlines()[0].strip()
    major, minor = version.split(".")[:2]
    # The alpha warning and the pin open the README, ahead of the quick start.
    assert f"This is an **alpha** (`VERSION` {version}). It may break." in readme
    assert readme.index("This is an **alpha**") < readme.index("## Quick start")
    assert f"Point bumps (`{major}.{minor}.x`) are fixes and small UI." in readme
    assert f"A main alpha bump (`{major}.x.0`) is features, core, or a UI overhaul." in readme
    assert "1.7.13-alpha-<sha>" in readme
    assert "docs/stitch-360.png" in readme
    assert "docs/soft-esc-highway.jpg" not in readme
    assert "Soft Esc parked" in readme
    # The 360 stitch stays in the player-facing part, ahead of the technical reference.
    # The release history left the README for the git log; the README must say so.
    assert readme.index("docs/stitch-360.png") < readme.index("## Technical details")
    assert "The change history is the git log." in readme
    # Retail drives via the cmd JSON bus; the old "cannot drive" wording must be gone everywhere players look.
    assert "beamngpy if importable" not in readme.lower()
    for f in (ROOT / "README.md", ROOT / "play_gvd.bat", ROOT / "play_gvd_tech.bat", ROOT / "scripts" / "make_release_zip.py",
              ROOT / "python" / "run_vision.py", ROOT / "docs" / "gvd_state_schema.md"):
        text = f.read_text(encoding="utf-8", errors="ignore").lower()
        for bad in ("cannot drive", "does not steer", "does not drive", "no-op sink", "never steers"):
            assert bad not in text, f"{f.name}: stale wording {bad!r}"
    assert "1-cam window capture only" in rel.version_text("test") and "gvd_cmd.json" in rel.version_text("test")
    # Soft: retail zip ships base+retail only
    assert "requirements-beamng.txt" not in (ROOT / "scripts" / "make_release_zip.py").read_text()
    assert '"requirements*.txt"' not in (ROOT / "scripts" / "make_release_zip.py").read_text()
    bat = (ROOT / "play_gvd.bat").read_text(encoding="utf-8", errors="ignore")
    assert 'set "GVD_BACKEND=window"' in bat and "--backend %GVD_BACKEND%" in bat
    assert "--backend auto" not in bat
    reqs = (ROOT / "requirements-retail.txt").read_text(encoding="utf-8")
    assert "mss" in reqs and "beamngpy" not in reqs.replace("# No beamngpy here", "")


def main() -> None:
    checks = (
        check_release_zip,
        check_boot_line_honesty,
        check_window_backend_one_cam,
        check_window_picker_prefers_drive,
        check_cmd_json_drive_bus,
        check_engage_path_contract,
        check_gvd_docs_dir,
        check_no_chrome,
        check_player_docs,
        check_lua_harness,
        check_run_vision_e2e,
        check_synthetic_lanes_drive,
        check_ffb_override_live,
        check_ffb_override_live_slow_host,
        check_echo_gate_clock_default,
        check_supervisor_clock_default,
        check_ffb_stale_engage_heartbeat,
        check_echo_outdated_refuses_engage,
        check_echo_stale_disengages,
        check_echo_old_mod_version_refused,
        check_echo_cmd_blind_not_stale,
        check_supervisor_tick_latency_report,
    )
    failed: list[str] = []
    for fn in checks:
        # Own HOME / AppData / bus folder per check: no gvd_state / gvd_cmd / gvd_engage /
        # gvd_strip_record file from another check (or another suite on this PC) is visible.
        try:
            with isolated_home(prefix=f"gvd-m6-{fn.__name__}-"):
                fn()
        except Exception:
            failed.append(fn.__name__)
            traceback.print_exc()
    if failed:
        raise SystemExit("test_m6_retail failed: " + ", ".join(failed))
    print("test_m6_retail: OK")


if __name__ == "__main__":
    main()
