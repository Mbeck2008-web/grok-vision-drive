"""Relaunch a hung BeamNG.tech after consecutive recorder attach/load failures.

A listening research port is not a liveness check: a hung Tech can keep
``127.0.0.1:25252`` LISTENING while Hello times out. After N consecutive
``session_open_failed`` or ``level_load_failed`` clips, stop attaching,
kill the process tree, wait for the research port to close, relaunch the
install-root exe with ``-tcom -console -gfx vk``, and continue only after
a real Hello. Hello exceptions are retried until the wait deadline.
"""
from __future__ import annotations

import json
import os
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
DEFAULT_PORT_CLEAR_S = 20.0
DEFAULT_PORT_CLEAR_POLL_S = 0.2

STOP_RELAUNCH_LIMIT = "tech_relaunch_limit"
STOP_RELAUNCH_UNAVAILABLE = "tech_relaunch_unavailable"
STOP_RELAUNCH_HELLO_FAILED = "tech_relaunch_hello_failed"
STOP_RELAUNCH_PORT_STUCK = "tech_relaunch_port_stuck"

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


def _configured_host_port() -> tuple[str, int]:
    from python.sensors.tech import apply_env_overrides, load_tech_config

    try:
        cfg = apply_env_overrides(load_tech_config())
    except Exception:
        return "127.0.0.1", TECH_RESEARCH_PORT
    host = str(cfg.get("host") or "127.0.0.1")
    try:
        port = int(cfg.get("port") or TECH_RESEARCH_PORT)
    except (TypeError, ValueError):
        port = TECH_RESEARCH_PORT
    return host, port


def _port_owner_pids(port: int) -> list[int]:
    """PIDs with a TCP LISTENING socket on ``port`` (Windows ``netstat``)."""
    proc = subprocess.run(
        ["netstat", "-ano", "-p", "tcp"],
        capture_output=True,
        text=True,
        errors="replace",
    )
    pids: list[int] = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP" or parts[3].upper() != "LISTENING":
            continue
        _head, sep, rest = parts[1].rpartition(":")
        if not sep:
            continue
        try:
            local_port = int(rest)
        except ValueError:
            continue
        if local_port != int(port) or not parts[4].isdigit():
            continue
        pid = int(parts[4])
        if pid > 0:
            pids.append(pid)
    return list(dict.fromkeys(pids))


def _exe_pids(exe: Path) -> list[int]:
    """PIDs whose ExecutablePath is the install-root Tech exe (Windows ``wmic``)."""
    want = os.path.normcase(os.path.normpath(str(exe)))
    proc = subprocess.run(
        ["wmic", "process", "get", "ProcessId,ExecutablePath", "/FORMAT:CSV"],
        capture_output=True,
        text=True,
        errors="replace",
    )
    pids: list[int] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) < 2:
            continue
        pid: int | None = None
        matched = False
        for part in parts:
            if part.isdigit():
                pid = int(part)
            elif part and os.path.normcase(os.path.normpath(part)) == want:
                matched = True
        if matched and pid is not None and pid > 0:
            pids.append(pid)
    return list(dict.fromkeys(pids))


def _tech_pids(exe: Path | str | None = None, *, port: int = TECH_RESEARCH_PORT) -> list[int]:
    """Research-port owner and install-root exe PIDs; tasklist is last-resort fallback."""
    if sys.platform != "win32":
        return []
    pids: list[int] = []
    pids.extend(_port_owner_pids(int(port)))
    if exe is not None and str(exe).strip():
        pids.extend(_exe_pids(Path(exe)))
    if pids:
        return list(dict.fromkeys(pids))
    for name in TECH_IMAGE_NAMES:
        proc = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {name}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            errors="replace",
        )
        for line in proc.stdout.splitlines():
            parts = [p.strip().strip('"') for p in line.split(",")]
            if len(parts) < 2 or parts[0].lower() != name.lower() or not parts[1].isdigit():
                continue
            pid = int(parts[1])
            if pid > 0:
                pids.append(pid)
    return list(dict.fromkeys(pids))


def kill_tech_process_tree(
    log: LogFn = print,
    *,
    exe: Path | None = None,
    port: int = TECH_RESEARCH_PORT,
) -> list[int]:
    """Force-kill the scoped BeamNG.tech process tree. No-op when none are running."""
    pids = _tech_pids(exe, port=port)
    if not pids:
        log("collect tech relaunch: no BeamNG.tech process to kill")
        return []
    log(f"collect tech relaunch: killing BeamNG.tech pids={pids}")
    if sys.platform != "win32":
        return pids
    for pid in pids:
        proc = subprocess.run(
            ["taskkill", "/F", "/PID", str(pid), "/T"],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            log(f"collect tech relaunch: taskkill pid={pid} failed rc={proc.returncode} {err}")
    return pids


def launch_tech(exe: Path, argv: tuple[str, ...] = TECH_RELAUNCH_ARGV, log: LogFn = print) -> subprocess.Popen[Any]:
    """Start install-root ``BeamNG.tech.exe -tcom -console -gfx vk``, detached from this process."""
    cmd = [str(exe), *argv]
    log(f"collect tech relaunch: starting {' '.join(cmd)}")
    kwargs: dict[str, Any] = {
        "cwd": str(Path(exe).parent),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        )
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


def wait_port_closed(
    listening_fn: Callable[[], bool],
    *,
    timeout_s: float = DEFAULT_PORT_CLEAR_S,
    poll_s: float = DEFAULT_PORT_CLEAR_POLL_S,
    log: LogFn = print,
) -> bool:
    """True when the research port is no longer LISTENING."""
    deadline = time.monotonic() + float(timeout_s)
    while True:
        try:
            up = bool(listening_fn())
        except Exception as exc:
            log(f"collect tech relaunch: port-listen check failed: {type(exc).__name__}: {exc}")
            up = True
        if not up:
            return True
        if time.monotonic() >= deadline:
            log(f"collect tech relaunch: research port still LISTENING after {timeout_s:g}s")
            return False
        remaining = deadline - time.monotonic()
        time.sleep(max(0.01, min(float(poll_s), remaining)))


def wait_tech_hello(
    *,
    listening_fn: Callable[[], bool],
    hello_fn: Callable[[], Any],
    close_fn: Callable[[Any], None],
    timeout_s: float = DEFAULT_HELLO_WAIT_S,
    poll_s: float = 0.5,
    log: LogFn = print,
) -> bool:
    """True only after a real Hello succeeds. A listening port alone is not ready.

    ``hello_fn`` exceptions are logged and retried until ``timeout_s``.
    """
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() <= deadline:
        if listening_fn():
            try:
                bng = hello_fn()
            except Exception as exc:
                log(f"collect tech relaunch Hello failed: {type(exc).__name__}: {exc}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(max(0.01, min(float(poll_s), remaining)))
                continue
            try:
                close_fn(bng)
            except Exception as exc:
                log(f"collect tech relaunch Hello close: {type(exc).__name__}: {exc}")
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(max(0.01, min(float(poll_s), remaining)))
    log(f"collect tech relaunch: Hello not ready after {timeout_s:g}s")
    return False


def wait_live_hello(*, timeout_s: float = DEFAULT_HELLO_WAIT_S, log: LogFn = print) -> bool:
    """Attach Hello on the configured research host/port; close the probe, leave Tech running."""
    from python.sensors.tech import (
        apply_env_overrides,
        hold_tech_process,
        load_tech_config,
        open_tech_beamngpy,
        research_port_listening,
    )

    cfg = apply_env_overrides(load_tech_config())
    host = str(cfg.get("host") or "127.0.0.1")
    try:
        port = int(cfg.get("port") or TECH_RESEARCH_PORT)
    except (TypeError, ValueError):
        port = TECH_RESEARCH_PORT
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
    kill_fn: KillFn | None = None,
    launch_fn: LaunchFn = launch_tech,
    wait_fn: WaitFn | None = None,
    listening_fn: Callable[[], bool] | None = None,
    wait_port_fn: Callable[[], bool] | None = None,
    port_clear_timeout_s: float = DEFAULT_PORT_CLEAR_S,
    log: LogFn = print,
) -> str | None:
    """Kill hung Tech, wait for the port to close, start the human argv, wait for Hello.

    Returns None on success, or a ``stop_reason`` if relaunch cannot proceed.
    A listening port after kill aborts without launching a second Tech.
    """
    from python.sensors.tech import research_port_listening

    _host, port = _configured_host_port()
    if kill_fn is None:

        def kill_fn() -> Iterable[int]:
            return kill_tech_process_tree(log=log, exe=Path(exe), port=port)

    killed = list(kill_fn())
    log(f"collect tech relaunch: killed pids={killed or 'none'}")

    def _listening() -> bool:
        if listening_fn is not None:
            return listening_fn()
        return research_port_listening(_host, port)

    def _wait_port() -> bool:
        if wait_port_fn is not None:
            return wait_port_fn()
        return wait_port_closed(_listening, timeout_s=port_clear_timeout_s, log=log)

    if not _wait_port():
        return STOP_RELAUNCH_PORT_STUCK
    launch_fn(Path(exe), TECH_RELAUNCH_ARGV)
    waiter = wait_fn if wait_fn is not None else lambda: wait_live_hello(log=log)
    if not waiter():
        return STOP_RELAUNCH_HELLO_FAILED
    return None


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
        self._fail_reason: str | None = None
        self.halt_reason: str | None = None

    @property
    def relaunches(self) -> int:
        return self._relaunches

    def _default_relaunch(self) -> bool:
        if self.exe is None:
            return False
        reason = relaunch_tech(exe=self.exe, log=self.log)
        if reason:
            self._fail_reason = reason
            return False
        return True

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
            return self._halt(self._fail_reason or STOP_RELAUNCH_HELLO_FAILED)
        self._streak = 0
        return None

    def _halt(self, reason: str) -> str:
        self.halt_reason = reason
        self.log(f"session stop reason={reason}")
        return reason
