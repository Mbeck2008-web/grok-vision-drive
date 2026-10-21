#!/usr/bin/env python3
"""run_vision's default path is unchanged by the cam-age live hold.

A stub supervisor run with no cam-age scene model prints no ``[GVD] cam_age``
line and never writes ``cam_ages`` / ``adhoc_timeouts`` into gvd_state.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gvd_test_env import isolated_home  # noqa: E402

CAM_AGE_KEYS = ("cam_ages", "adhoc_timeouts")


def run_supervisor(extra: list[str], seconds: float = 2.0) -> tuple[str, list[dict]]:
    """Run the stub supervisor for ``seconds`` after its first command; stdout and sampled states."""
    from python.control.actuate import cmd_path
    from python.runtime import paths as gvd_paths
    from python.runtime.state_io import state_path

    states: list[dict] = []
    with isolated_home(prefix="gvd-default-") as home:
        pin = gvd_paths.predicted_gvd_docs_path()
        pin.mkdir(parents=True, exist_ok=True)

        def touch_bus() -> None:
            (pin / "gvd_link.json").write_text(json.dumps({"lua_bus": str(pin.resolve())}), encoding="utf-8")

        touch_bus()
        out_path = Path(home) / "stdout.txt"
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        args = [sys.executable, "-u", str(ROOT / "python" / "run_vision.py"), "--backend", "stub", "--hz", "20", "--encode", "cpu", *extra]
        with out_path.open("w", encoding="utf-8") as out:
            proc = subprocess.Popen(args, cwd=str(ROOT), env=env, stdout=out, stderr=subprocess.STDOUT)
            first = None
            try:
                end = time.time() + 30.0
                while time.time() < end:
                    if proc.poll() is not None:
                        break
                    touch_bus()
                    if first is None and cmd_path().exists():
                        first = time.time()
                        end = first + float(seconds)
                    if first is not None and state_path().exists():
                        try:
                            states.append(json.loads(state_path().read_text(encoding="utf-8")))
                        except (ValueError, OSError):
                            pass
                    time.sleep(0.05)
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=10)
        text = out_path.read_text(encoding="utf-8", errors="replace")
        assert first is not None, text[-2000:]
    return text, states


def check_default_emits_nothing() -> None:
    text, states = run_supervisor([])
    assert "[GVD] seg " in text, text[-1500:]  # the loop ran
    assert "[GVD] cam_age" not in text
    assert states, "no gvd_state.json sampled"
    for st in states:
        for key in CAM_AGE_KEYS:
            assert key not in st, (key, sorted(st)[:20])
    print(f"default run: {text.count('[GVD] seg ')} ticks, no '[GVD] cam_age' line, {len(states)} states without {CAM_AGE_KEYS} OK")


def main() -> None:
    check_default_emits_nothing()
    print("test_live_default_path: all OK")


if __name__ == "__main__":
    main()
