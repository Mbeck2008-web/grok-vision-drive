"""Relaunch a hung BeamNG.tech after consecutive recorder attach/load failures.

A listening research port is not a liveness check: a hung Tech can keep
``127.0.0.1:25252`` LISTENING while Hello times out. After N consecutive
``session_open_failed`` or ``level_load_failed`` clips, stop attaching,
kill the process tree, relaunch the install-root exe with
``-tcom -console -gfx vk``, and continue only after a real Hello.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from python.sensors.tech import TECH_HUMAN_TAIL, TECH_RESEARCH_PORT, TECH_ROOT_EXE

TECH_FAIL_REASONS = frozenset({"session_open_failed", "level_load_failed"})
DEFAULT_TECH_FAIL_RELAUNCH = 2
DEFAULT_TECH_RELAUNCH_CAP = 3
TECH_RELAUNCH_ARGV = TECH_HUMAN_TAIL
DEFAULT_HELLO_WAIT_S = 180.0

STOP_RELAUNCH_LIMIT = "tech_relaunch_limit"
STOP_RELAUNCH_UNAVAILABLE = "tech_relaunch_unavailable"
STOP_RELAUNCH_HELLO_FAILED = "tech_relaunch_hello_failed"

TECH_IMAGE_NAMES = ("BeamNG.tech.exe", "BeamNG.tech.x64.exe", "BeamNG.tech")

LogFn = Callable[[str], None]
KillFn = Callable[[], Iterable[int]]
LaunchFn = Callable[[Path, tuple[str, ...]], Any]
WaitFn = Callable[[], bool]
RelaunchFn = Callable[[], bool]


def clip_tech_fail_reason(clip: Path | str) -> str | None:
    """``session_open_failed`` / ``level_load_failed`` from collect.json, else None."""
    path = Path(clip) / "collect.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    reason = data.get("stop_reason")
    if reason in TECH_FAIL_REASONS:
        return str(reason)
    return None


def resolve_tech_exe(home: str | None = None) -> Path | None:
    """Install-root ``BeamNG.tech.exe`` when ``BNG_HOME`` / yaml home exists on disk."""
    if home is None:
        home = os.environ.get("BNG_HOME") or os.environ.get("BEAMNG_HOME") or ""
        if not str(home).strip():
            try:
                from python.sensors.tech import load_tech_config

                home = str(load_tech_config().get("home") or "")
            except Exception:
                home = ""
    if not str(home).strip():
        return None
    exe = Path(home) / TECH_ROOT_EXE
    return exe if exe.is_file() else None


def _tech_pids() -> list[int]:
    pids: list[int] = []
    if sys.platform == "win32":
        for name in TECH_IMAGE_NAMES:
            proc = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {name}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                errors="replace",
            )
            for line in proc.stdout.splitlines():
                # "BeamNG.tech.x64.exe","44888","Session Name",...
                parts = [p.strip().strip('"') for p in line.split(",")]
                if len(parts) >= 2 and parts[1].isdigit():
                    pids.append(int(parts[1]))
    return list(dict.fromkeys(pids))


def kill_tech_process_tree(log: LogFn = print) -> list[int]:
    """Force-kill the BeamNG.tech process tree. No-op when none are running."""
    pids = _tech_pids()
    if not pids:
        log("collect tech relaunch: no BeamNG.tech process to kill")
        return []
    log(f"collect tech relaunch: killing BeamNG.tech pids={pids}")
    if sys.platform == "win32":
        for pid in pids:
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid), "/T"],
                capture_output=True,
                text=True,
                errors="replace",
            )
    else:
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    return pids


def launch_tech(exe: Path, argv: tuple[str, ...] = TECH_RELAUNCH_ARGV, log: LogFn = print) -> subprocess.Popen[Any]:
    """Start install-root ``BeamNG.tech.exe -tcom -console -gfx vk``, detached from this process."""
    cmd = [str(exe), *argv]
    log(f"collect tech relaunch: starting {' '.join(cmd)}")
    kwargs: dict[str, Any] = {"cwd": str(Path(exe).parent)}
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        )
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


def wait_tech_hello(
    *,
    listening_fn: Callable[[], bool],
    hello_fn: Callable[[], Any],
    close_fn: Callable[[Any], None],
    timeout_s: float = DEFAULT_HELLO_WAIT_S,
    poll_s: float = 0.5,
    log: LogFn = print,
) -> bool:
    """True only after a real Hello succeeds. A listening port alone is not ready."""
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() <= deadline:
        if listening_fn():
            try:
                bng = hello_fn()
            except Exception as exc:
                log(f"collect tech relaunch Hello failed: {type(exc).__name__}: {exc}")
                return False
            try:
                close_fn(bng)
            except Exception as exc:
                log(f"collect tech relaunch Hello close: {type(exc).__name__}: {exc}")
            return True
        time.sleep(max(0.01, float(poll_s)))
    log(f"collect tech relaunch: Hello not ready after {timeout_s:g}s")
    return False


def wait_live_hello(*, timeout_s: float = DEFAULT_HELLO_WAIT_S, log: LogFn = print) -> bool:
    """Attach Hello on the configured research host/port; close the probe, leave Tech running."""
    from python.sensors.tech import (
        hold_tech_process,
        load_tech_config,
        open_tech_beamngpy,
        research_port_listening,
    )

    cfg = load_tech_config()
    host = str(cfg.get("host") or "127.0.0.1")
    port = int(cfg.get("port") or TECH_RESEARCH_PORT)
    home = str(cfg.get("home") or "") or None
    user = str(cfg.get("user") or "") or None

    def listening() -> bool:
        return research_port_listening(host, port)

    def hello() -> Any:
        return open_tech_beamngpy(host, port, home=home, user=user, launch=False)

    def close(bng: Any) -> None:
        hold_tech_process(bng)
        closer = getattr(bng, "close", None)
        if callable(closer):
            closer()

    return wait_tech_hello(
        listening_fn=listening,
        hello_fn=hello,
        close_fn=close,
        timeout_s=timeout_s,
        log=log,
    )


def relaunch_tech(
    *,
    exe: Path,
    kill_fn: KillFn = kill_tech_process_tree,
    launch_fn: LaunchFn = launch_tech,
    wait_fn: WaitFn | None = None,
    log: LogFn = print,
) -> bool:
    """Kill hung Tech, start the same human argv, wait for Hello (not just LISTEN)."""
    killed = list(kill_fn())
    log(f"collect tech relaunch: killed pids={killed or 'none'}")
    launch_fn(Path(exe), TECH_RELAUNCH_ARGV)
    waiter = wait_fn if wait_fn is not None else lambda: wait_live_hello(log=log)
    return bool(waiter())


class TechWatchdog:
    """Count consecutive Tech attach/load failures and relaunch, up to a cap."""

    def __init__(
        self,
        *,
        fail_relaunch: int = DEFAULT_TECH_FAIL_RELAUNCH,
        relaunch_cap: int = DEFAULT_TECH_RELAUNCH_CAP,
        exe: Path | None = None,
        relaunch: RelaunchFn | None = None,
        log: LogFn = print,
    ) -> None:
        self.fail_relaunch = int(fail_relaunch)
        self.relaunch_cap = int(relaunch_cap)
        self.exe = Path(exe) if exe is not None else None
        self.log = log
        self._relaunch = relaunch if relaunch is not None else self._default_relaunch
        self._streak = 0
        self._relaunches = 0
        self.halt_reason: str | None = None

    @property
    def relaunches(self) -> int:
        return self._relaunches

    def _default_relaunch(self) -> bool:
        if self.exe is None:
            return False
        return relaunch_tech(exe=self.exe, log=self.log)

    def after_clip(self, clip: Path | str, exit_code: int = 0) -> str | None:
        """Return a session ``stop_reason`` to halt the batch, or None to continue."""
        del exit_code
        if self.fail_relaunch <= 0:
            return None
        reason = clip_tech_fail_reason(clip)
        if reason is None:
            self._streak = 0
            return None
        self._streak += 1
        if self._streak < self.fail_relaunch:
            return None
        if self._relaunches >= self.relaunch_cap:
            return self._halt(STOP_RELAUNCH_LIMIT)
        if self.exe is None or not Path(self.exe).is_file():
            return self._halt(STOP_RELAUNCH_UNAVAILABLE)
        self._relaunches += 1
        self.log(
            f"collect tech relaunch {self._relaunches}/{self.relaunch_cap}: "
            f"reason={reason} exe={self.exe}"
        )
        if not self._relaunch():
            return self._halt(STOP_RELAUNCH_HELLO_FAILED)
        self._streak = 0
        return None

    def _halt(self, reason: str) -> str:
        self.halt_reason = reason
        self.log(f"session stop reason={reason}")
        return reason
