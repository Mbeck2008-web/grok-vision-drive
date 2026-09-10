#!/usr/bin/env python3
"""Offline M6 retail-package checks (no BeamNG, no weights, no network)."""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from typing import Callable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_release_zip as rel  # noqa: E402
from python.control.actuate import (  # noqa: E402
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
from python.sensors.cameras import CAM_IDS, WindowBackend  # noqa: E402

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
            for bad in (".onnx", ".pt", ".safetensors", ".pyc", ".mp4"):
                assert not any(r.endswith(bad) for r in rels), bad
            assert not any(r.startswith((".git/", "data/", "dist/", "scripts/")) for r in rels)
            assert not any("clips" in r for r in rels)
            assert "requirements-beamng.txt" not in rels
            assert "requirements-perception.txt" not in rels
            assert "requirements-viz.txt" not in rels
            # Mod entry point + UI icon survive; tests do not.
            assert "beamng_mod/scripts/gvd/modScript.lua" in rels
            assert "beamng_mod/ui/modules/apps/GVD/app.png" in rels
            assert "models/.gitkeep" in rels
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
    assert not any(f.endswith((".onnx", ".pt")) for f in real)


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


def _write_ego(
    seq: int,
    *,
    applying: bool = True,
    speed: float = 5.5,
    steer_in: float = 0.0,
    throttle_in: float = 0.0,
    brake_in: float = 0.0,
    age_s: float = 0.0,
) -> None:
    ego_path().write_text(
        json.dumps(
            {
                "speed_mps": speed,
                "steering_input": steer_in,
                "throttle_input": throttle_in,
                "brake_input": brake_in,
                "applied_seq": seq,
                "applying": applying,
                "mtime": int(time.time()),
            }
        ),
        encoding="utf-8",
    )
    if age_s:
        old = time.time() - age_s
        os.utime(ego_path(), (old, old))


def check_cmd_json_drive_bus() -> None:
    """Retail actuator: engaged → real drive payload; disengaged → idle stop; applied only on Lua ack."""
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
    assert time.time() - payload["heartbeat_mtime"] < 5.0
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


def _run_supervisor(
    extra: list[str],
    seconds: float,
    ack: bool = False,
    steer_echo: Callable[[int], float] | None = None,
    brake_bias: float = 0.0,
) -> str:
    """Run run_vision.py headless for `seconds`; optionally play the mod's ack role meanwhile.

    The fake mod echoes back what it was told to apply, exactly like `gvd_main` does, so the
    override residual is zero unless a test deliberately plays a driver: `steer_echo(seq)`
    reports a wheel position instead of the commanded one, `brake_bias` adds a pedal press.
    """
    args = [sys.executable, "-u", str(ROOT / "python" / "run_vision.py"), "--backend", "stub", "--hz", "20", "--encode", "cpu", *extra]
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    proc = subprocess.Popen(args, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t_end = time.time() + seconds
    try:
        while time.time() < t_end:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                raise AssertionError(f"run_vision exited early ({proc.returncode}):\n{out}")
            if ack:
                cmd: dict = {}
                try:
                    cmd = json.loads(cmd_path().read_text(encoding="utf-8"))
                    seq = int(cmd.get("seq", -1))
                except Exception:
                    seq = -1
                if seq >= 0:
                    _write_ego(
                        seq,
                        applying=True,
                        speed=5.5,
                        steer_in=steer_echo(seq) if steer_echo else float(cmd.get("steer", 0.0)),
                        throttle_in=float(cmd.get("throttle", 0.0)),
                        brake_in=min(1.0, float(cmd.get("brake", 0.0)) + brake_bias),
                    )
            # Tighter than the 20 Hz supervisor loop so the echo always belongs to the newest seq.
            time.sleep(0.01)
    finally:
        proc.kill()  # SIGKILL: the `finally` stop never runs, so the last tick's files survive for assertions
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
    try:
        return proc.stdout.read() if proc.stdout else ""
    except Exception:
        return ""


def check_run_vision_e2e() -> None:
    """Supervisor loop end-to-end (stub cameras): engaged writes drive cmds, disengaged writes idle stops."""
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
    assert st["cmd_reason"] == "not_engaged" and st["ego_source"] == "none"

    # Engaged with no lane path (stub) → preview gate holds the brake, still engaged for Lua.
    _run_supervisor(["--force-engage"], seconds=2.0)
    cmd = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert cmd["engaged"] is True and cmd["brake"] == 1.0 and cmd["throttle"] == 0.0 and cmd["reason"] == "preview_blocked", cmd

    # Engaged + preview allowed → real drive command every tick; Lua ack flips cmd_applied and ego speed.
    _run_supervisor(["--force-engage", "--allow-preview-drive"], seconds=2.5, ack=True)
    cmd = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert cmd["engaged"] is True and cmd["reason"] == "ok" and cmd["seq"] >= 10, cmd
    assert cmd["throttle"] > 0.0 and cmd["brake"] == 0.0 and -1.0 <= cmd["steer"] <= 1.0, cmd
    st = read_state()
    assert st and st["engaged"] is True and st["cmd_applied"] is True, {k: st.get(k) for k in ("engaged", "cmd_applied", "cmd_reason")}
    assert st["cmd_reason"] == "cmd_json_applied" and st["ego_source"] == "lua" and st["lua_applying"] is True
    assert abs(float(st["ego"]["speed_mps"]) - 5.5) < 1e-6, st["ego"]
    assert st["cmd_ack_seq"] >= st["cmd_seq"] - 5

    write_engage_flag(False)
    ego_path().unlink()


def check_ffb_override_live() -> None:
    """Whole loop with a wheel on the echo: chatter keeps driving, a real driver gets the car."""
    if ego_path().exists():
        ego_path().unlink()

    # Force-feedback chatter — alternating kicks keyed off the seq so the sign flips every tick.
    write_engage_flag(True)
    out = _run_supervisor(["--allow-preview-drive"], seconds=3.0, ack=True, steer_echo=lambda seq: 0.9 if seq % 2 else -0.9)
    assert "DISENGAGED: player_" not in out, out[-2000:]
    assert read_engage_flag(default=False) is True, "force-feedback chatter disengaged GVD"
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

    # A driver actually holding the wheel over. The engage file is the durable record: later ticks
    # only see engaged=false and write the generic not_engaged.
    write_engage_flag(True)
    out = _run_supervisor(["--allow-preview-drive"], seconds=3.0, ack=True, steer_echo=lambda _seq: 0.25)
    assert "DISENGAGED: player_steer" in out, out[-2000:]
    assert read_engage_flag(default=True) is False, "a real steer takeover must disengage"
    assert json.loads(engage_path().read_text(encoding="utf-8"))["disengage_reason"] == "player_steer"
    st = read_state()
    assert st and st["engaged"] is False, {k: st.get(k) for k in ("engaged", "disengage_reason")}
    assert json.loads(cmd_path().read_text(encoding="utf-8"))["engaged"] is False

    # Pedals stay tight: a brake press on top of the command is an override with no dwell.
    write_engage_flag(True)
    out = _run_supervisor(["--allow-preview-drive"], seconds=2.0, ack=True, brake_bias=0.4)
    assert "DISENGAGED: player_brake" in out, out[-2000:]
    assert read_engage_flag(default=True) is False, "a real brake press must disengage"
    assert json.loads(engage_path().read_text(encoding="utf-8"))["disengage_reason"] == "player_brake"

    # Mirrored thresholds reach the mod through gvd_state.json.
    from python.control.override import config_mirror, load_override_config

    want = config_mirror(load_override_config(yaml.safe_load((ROOT / "config" / "control.yaml").read_text(encoding="utf-8"))))
    assert read_state()["override_cfg"] == want, read_state()["override_cfg"]

    write_engage_flag(False)
    ego_path().unlink()


def check_lua_harness() -> None:
    """Drive the mod's Lua through the stub-GE harness when a Lua 5.1 interpreter is available."""
    for exe in ("lua5.1", "luajit", "lua"):
        if shutil.which(exe):
            res = subprocess.run([exe, "scripts/test_m6_lua_cmd.lua"], cwd=str(ROOT), capture_output=True, text=True, timeout=60)
            assert res.returncode == 0 and "test_m6_lua_cmd: OK" in res.stdout, res.stdout + res.stderr
            return
    print("  (lua interpreter not found — scripts/test_m6_lua_cmd.lua skipped)")


def check_engage_path_contract() -> None:
    p = engage_path()
    assert p.parts[-3:] == ("Documents", "GVD", "gvd_engage.json"), p
    assert p.parent == state_path().parent == cmd_path().parent
    # Lua-shaped payload (Alt+A) must read back as engaged.
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"engaged":true,"mtime":1700000000}', encoding="utf-8")
    assert read_engage_flag(default=False) is True
    # Supervisor disengage payload: engaged=false with float mtime + reason (what Lua adopts as OFF and logs).
    write_engage_flag(False, disengage_reason="player_steer")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["engaged"] is False and isinstance(data["mtime"], float)
    assert data["disengage_reason"] == "player_steer"
    assert read_engage_flag(default=True) is False
    write_engage_flag(False)
    assert json.loads(p.read_text(encoding="utf-8"))["disengage_reason"] == "none"


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
    for f in ("install.bat", "uninstall.bat", "play_gvd.bat", "requirements-retail.txt"):
        assert not CHROME_RE.search((ROOT / f).read_text(encoding="utf-8", errors="ignore")), f
    # Runtime strings players see (titles, prints); docstrings/comments may still state the honesty rule.
    for f in [ROOT / "python" / "run_vision.py", ROOT / "python" / "runtime" / "hw_probe.py", *(ROOT / "python" / "viz").glob("*.py")]:
        for s in _py_string_literals(f):
            assert not CHROME_RE.search(s), f"chrome string in {f.relative_to(ROOT)}: {s!r}"
    # Release VERSION.txt banner only names the honesty rule, never a product label.
    ver = rel.version_text("test")
    assert "Not Tesla FSD" in ver and "Full Self" not in ver


def check_player_docs() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "**M6" in readme, "README Status must be M6"
    assert "## Player guide" in readme
    assert "make_release_zip" in readme
    assert "requirements-retail.txt" in readme
    # Retail drives via the cmd JSON bus; the old "cannot drive" wording must be gone everywhere players look.
    assert "gvd_cmd.json" in readme and "gvd_ego.json" in readme
    for f in (ROOT / "README.md", ROOT / "play_gvd.bat", ROOT / "scripts" / "make_release_zip.py",
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
    check_release_zip()
    check_boot_line_honesty()
    check_window_backend_one_cam()
    check_cmd_json_drive_bus()
    check_engage_path_contract()
    check_no_chrome()
    check_player_docs()
    check_lua_harness()
    check_run_vision_e2e()
    check_ffb_override_live()
    print("test_m6_retail: OK")


if __name__ == "__main__":
    main()
