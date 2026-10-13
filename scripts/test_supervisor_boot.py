#!/usr/bin/env python3
"""A new supervisor starts disengaged. A fresh engage after that still counts.

No BeamNG. The subprocess is the stub backend, killed so shutdown cannot
rewrite the engage file.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import test_m6_retail as retail  # noqa: E402
from python.control.actuate import (  # noqa: E402
    BotEngage,
    bot_engage_fresh,
    cmd_path,
    disarm_supervisor_start,
    engage_path,
    read_engage_flag,
    write_bot_engage,
    write_engage_flag,
)


def check_disarm_unit() -> None:
    retail._touch_lua_bus()
    write_engage_flag(True)
    write_bot_engage(True)
    assert read_engage_flag(default=False) is True
    assert bot_engage_fresh() is True
    data = disarm_supervisor_start()
    assert data.get("engaged") is False, data
    assert data.get("disengage_reason") == "supervisor_start", data
    assert read_engage_flag(default=True) is False
    assert bot_engage_fresh() is False
    assert BotEngage().poll() is False
    # A fresh true after the boot write is a real engage for this process.
    write_engage_flag(True)
    assert read_engage_flag(default=False) is True
    assert json.loads(engage_path().read_text(encoding="utf-8"))["disengage_reason"] == "none"


def check_boot_subprocess() -> None:
    """Leftover engaged:true must not drive. The file stays supervisor_start."""
    retail._touch_lua_bus()
    write_engage_flag(True)
    write_bot_engage(True)
    assert read_engage_flag(default=False) is True
    spawned = time.time()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(ROOT / "python" / "run_vision.py"),
            "--backend",
            "stub",
            "--hz",
            "20",
            "--encode",
            "cpu",
        ],
        cwd=str(ROOT),
        env=dict(os.environ, PYTHONPATH=str(ROOT)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    boot = None
    cmd = None
    try:
        deadline = time.time() + 25.0
        while time.time() < deadline:
            if proc.poll() is not None:
                raise AssertionError(f"run_vision exited early ({proc.returncode})")
            retail._touch_lua_bus()
            eng = retail._read_json(engage_path())
            if (
                eng.get("engaged") is False
                and eng.get("disengage_reason") == "supervisor_start"
                and float(eng.get("mtime") or 0) >= spawned - 1.0
            ):
                boot = eng
            if boot is not None and cmd_path().is_file():
                c = retail._read_json(cmd_path())
                if (
                    c.get("engaged") is False
                    and int(c.get("seq") or 0) >= 1
                    and float(c.get("heartbeat_mtime") or 0) >= spawned - 1.0
                ):
                    cmd = c
                    break
            time.sleep(0.02)
    finally:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
    assert boot is not None, "supervisor_start never appeared"
    assert cmd is not None, "disengaged command never appeared"
    assert cmd.get("throttle") == 0.0, cmd
    assert cmd.get("reason") in ("not_engaged", "bus_mismatch"), cmd
    # SIGKILL skips the shutdown rewrite, so the boot reason is what remains.
    left = json.loads(engage_path().read_text(encoding="utf-8"))
    assert left.get("engaged") is False and left.get("disengage_reason") == "supervisor_start", left
    assert bot_engage_fresh() is False


def main() -> None:
    check_disarm_unit()
    check_boot_subprocess()
    print("test_supervisor_boot: OK")


if __name__ == "__main__":
    main()
