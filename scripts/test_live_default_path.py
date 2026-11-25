#!/usr/bin/env python3
"""run_vision's default path is unchanged by the cam-age live hold.

A stub supervisor run with no cam-age scene model prints no ``[GVD] cam_age``
line and never writes ``cam_ages`` / ``stale_ego_n`` into gvd_state.json.
``--no-engage`` (or GVD_NO_ENGAGE=1) keeps engaged False even with
``--force-engage``; ``--run-seconds`` ends the loop on its own.
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

CAM_AGE_KEYS = ("cam_ages", "stale_ego_n")


def _touch_bus() -> None:
    """Keep a lua_bus handshake at the predicted pin (same write as 8f4a163)."""
    from python.runtime import paths as gvd_paths

    pin = gvd_paths.predicted_gvd_docs_path()
    pin.mkdir(parents=True, exist_ok=True)
    (pin / "gvd_link.json").write_text(json.dumps({"lua_bus": str(pin.resolve())}), encoding="utf-8")


def _observe_supervisor(
    proc: object,
    *,
    seconds: float,
    echo,
    cmd,
    state,
    startup_s: float = 30.0,
) -> tuple[float | None, list[dict]]:
    """Sample cmd/state until ``seconds`` after the first command, or the child exits.

    8f4a163 behavior: break as soon as poll() is not None, before exists().
    """
    states: list[dict] = []
    first: float | None = None
    next_echo, echo_seq = 0.0, 0
    end = time.time() + float(startup_s)
    while time.time() < end:
        if proc.poll() is not None:
            break
        _touch_bus()
        if echo is not None and time.monotonic() >= next_echo:
            next_echo = time.monotonic() + echo.FAKE_ECHO_PERIOD_S
            echo_seq += 1
            echo._write_ego(_cmd_seq(cmd()), speed=0.0, echo_seq=echo_seq)
        if first is None and cmd().exists():
            first = time.time()
            end = first + float(seconds)
        if first is not None and state().exists():
            try:
                states.append(json.loads(state().read_text(encoding="utf-8")))
            except (ValueError, OSError):
                pass
        time.sleep(0.05)
    return first, states


class _ExitedProc:
    """A child that has already exited. Forces the poll-before-observe interleaving."""

    def poll(self):
        return 0

    def kill(self) -> None:
        return None

    def wait(self, timeout=None) -> int:
        return 0


def check_observes_cmd_after_child_exit() -> None:
    """If the child writes gvd_cmd.json and exits before the first poll, first is still set."""
    from python.control.actuate import cmd_path
    from python.runtime import paths as gvd_paths
    from python.runtime.state_io import state_path

    with isolated_home(prefix="gvd-default-"):
        pin = gvd_paths.predicted_gvd_docs_path()
        pin.mkdir(parents=True, exist_ok=True)
        _touch_bus()
        cmd_path().write_text(json.dumps({"seq": 1}), encoding="utf-8")
        first, _states = _observe_supervisor(
            _ExitedProc(), seconds=0.0, echo=None, cmd=cmd_path, state=state_path, startup_s=1.0
        )
        assert first is not None, "missed gvd_cmd.json because the child had already exited"
        assert cmd_path().exists()
    print("observe after child exit: first command still seen OK")


def check_link_writer_atomic() -> None:
    """_touch_bus publishes gvd_link.json via os.replace; a concurrent reader never sees a tear."""
    import threading

    from python.runtime import paths as gvd_paths

    with isolated_home(prefix="gvd-default-"):
        pin = gvd_paths.predicted_gvd_docs_path()
        pin.mkdir(parents=True, exist_ok=True)
        _touch_bus()
        replaces: list[tuple[str, str]] = []
        real_replace = os.replace

        def wrapped_replace(src, dst, *args, **kwargs):
            replaces.append((Path(src).name, Path(dst).name))
            return real_replace(src, dst, *args, **kwargs)

        os.replace = wrapped_replace  # type: ignore[method-assign]
        try:
            _touch_bus()
        finally:
            os.replace = real_replace  # type: ignore[method-assign]
        assert replaces, "gvd_link.json was not published via os.replace"
        assert any(dst == "gvd_link.json" for _src, dst in replaces), replaces

        stop = threading.Event()
        stats = {"ok": 0, "mismatch": 0}

        def writer() -> None:
            while not stop.is_set():
                _touch_bus()

        def reader() -> None:
            while not stop.is_set():
                ident = gvd_paths.bus_identity()
                if ident.matched:
                    stats["ok"] += 1
                else:
                    stats["mismatch"] += 1

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        time.sleep(0.4)
        stop.set()
        for t in threads:
            t.join()
        total = stats["ok"] + stats["mismatch"]
        assert total > 0, "no bus_identity reads"
        assert stats["mismatch"] == 0, (
            f"torn or stale lua_bus during atomic writes: mismatch={stats['mismatch']}/{total}"
        )
    print(f"atomic gvd_link writer: {stats['ok']} matched reads, 0 mismatch OK")


def run_supervisor(extra: list[str], seconds: float = 2.0, echo: bool = False) -> tuple[str, list[dict]]:
    """Run the stub supervisor for ``seconds`` after its first command; stdout and sampled states.

    ``echo=True`` plays the mod's 10 Hz gvd_ego.json echo (a stopped car, echo_seq advancing):
    engaging needs a fresh session echo (#125).
    """
    from python.control.actuate import cmd_path
    from python.runtime import paths as gvd_paths
    from python.runtime.state_io import state_path

    with isolated_home(prefix="gvd-default-") as home:
        pin = gvd_paths.predicted_gvd_docs_path()
        pin.mkdir(parents=True, exist_ok=True)
        _touch_bus()
        m6 = _m6() if echo else None
        out_path = Path(home) / "stdout.txt"
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        args = [sys.executable, "-u", str(ROOT / "python" / "run_vision.py"), "--backend", "stub", "--hz", "20", "--encode", "cpu", *extra]
        with out_path.open("w", encoding="utf-8") as out:
            proc = subprocess.Popen(args, cwd=str(ROOT), env=env, stdout=out, stderr=subprocess.STDOUT)
            try:
                first, states = _observe_supervisor(
                    proc, seconds=seconds, echo=m6, cmd=cmd_path, state=state_path
                )
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=10)
        text = out_path.read_text(encoding="utf-8", errors="replace")
        assert first is not None, text[-2000:]
    return text, states


def _m6():
    import test_m6_retail

    return test_m6_retail


def _cmd_seq(path: Path) -> int:
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("seq", 0) or 0)
    except (OSError, ValueError, AttributeError):
        return 0


def check_default_emits_nothing() -> None:
    text, states = run_supervisor([])
    assert "[GVD] seg " in text, text[-1500:]  # the loop ran
    assert "[GVD] cam_age" not in text
    assert states, "no gvd_state.json sampled"
    for st in states:
        for key in CAM_AGE_KEYS:
            assert key not in st, (key, sorted(st)[:20])
    print(f"default run: {text.count('[GVD] seg ')} ticks, no '[GVD] cam_age' line, {len(states)} states without {CAM_AGE_KEYS} OK")


def check_no_engage() -> None:
    text, states = run_supervisor(["--force-engage"], echo=True)
    assert states and any(st.get("engaged") for st in states), "control: --force-engage never engaged"
    # No echo this session: --force-engage refuses with ego_unknown (#125).
    text, states = run_supervisor(["--force-engage"])
    assert states and not any(st.get("engaged") for st in states), "no echo: --force-engage engaged"
    assert states[-1].get("disengage_reason") == "ego_unknown", states[-1].get("disengage_reason")
    for extra, env in ((["--force-engage", "--no-engage"], {}), (["--force-engage"], {"GVD_NO_ENGAGE": "1"})):
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            text, states = run_supervisor(extra)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        assert "engage HARD-disabled" in text, text[-1500:]
        assert states and not any(st.get("engaged") for st in states), (extra, env)
        assert "[GVD] seg " in text
    print(f"--no-engage / GVD_NO_ENGAGE=1: {len(states)} states never engaged with --force-engage OK")


def check_run_seconds() -> None:
    text, _states = run_supervisor(["--run-seconds", "1.5"], seconds=20.0)
    assert "--run-seconds 1.5 reached; stopping" in text, text[-1500:]
    print("--run-seconds: the supervisor stops on its own OK")


def main() -> None:
    check_observes_cmd_after_child_exit()
    check_link_writer_atomic()
    check_default_emits_nothing()
    check_no_engage()
    check_run_seconds()
    print("test_live_default_path: all OK")


if __name__ == "__main__":
    main()
