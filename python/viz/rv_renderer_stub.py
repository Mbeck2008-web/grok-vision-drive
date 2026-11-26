"""Separate-process Reality View stub entrypoint (Step 1 process boundary).

Runs only as a child process (``python -m python.viz.rv_renderer_stub``).
Contains no renderer, no unprojection, no 3DGS, and no safety-path imports.

Heavy imports (rv_contract / numpy) are deferred so ``--serve`` can arm a
parent-death watch before any slow import. On Windows the parent of a
TerminateProcess'd process does not reparent, so ``os.getppid()`` stays on
the dead pid and is not a valid watch: OpenProcess(SYNCHRONIZE) +
WaitForSingleObject is used instead. Linux uses pidfd_open on ``--parent-pid``
so a parent death during interpreter startup is not missed after reparenting.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import select
import signal
import sys
import threading
import time

_PR_SET_PDEATHSIG = 1
_SYNCHRONIZE = 0x00100000
_WAIT_OBJECT_0 = 0
_POLL_MS = 200

# Test-only hook. Never read from env or config. Production --serve leaves this False.
# Set as a private module attribute from a test process after import, before main().
_TEST_DISABLE_PDEATHSIG = False


def _schema_version() -> int:
    from python.viz.rv_contract import RV_SCHEMA_VERSION

    return int(RV_SCHEMA_VERSION)


def _install_pdeathsig() -> int | None:
    """On Linux, set PR_SET_PDEATHSIG(SIGTERM) via ctypes libc; no-op elsewhere."""
    if _TEST_DISABLE_PDEATHSIG or not sys.platform.startswith("linux"):
        return None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        prctl.restype = ctypes.c_int
        sig = int(signal.SIGTERM)
        if prctl(_PR_SET_PDEATHSIG, sig, 0, 0, 0) == 0:
            return sig
    except Exception:  # noqa: BLE001
        return None
    return None


def _win_kernel32() -> ctypes.WinDLL:
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    k.OpenProcess.restype = ctypes.c_void_p
    k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    k.WaitForSingleObject.restype = ctypes.c_ulong
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    k.CloseHandle.restype = ctypes.c_int
    return k


def _open_parent_watch(ppid: int) -> tuple[str, object] | None:
    """Open a process-object watch on the parent.

    Returns None if the parent is already gone (caller must exit immediately).
    Windows: OpenProcess(SYNCHRONIZE). Linux: pidfd_open. Else a getppid fallback.
    """
    if sys.platform == "win32":
        handle = _win_kernel32().OpenProcess(_SYNCHRONIZE, 0, int(ppid))
        if not handle:
            return None
        return ("winhandle", handle)
    if hasattr(os, "pidfd_open"):
        try:
            return ("pidfd", os.pidfd_open(int(ppid), 0))
        except OSError:
            return None
    return ("ppid", int(ppid))


def _parent_has_exited(watch: tuple[str, object], timeout_s: float = 0.0) -> bool:
    """True if the watched parent process has exited. Optionally wait up to timeout_s."""
    kind, val = watch
    if kind == "winhandle":
        ms = max(0, int(timeout_s * 1000))
        return int(_win_kernel32().WaitForSingleObject(val, ms)) == _WAIT_OBJECT_0
    if kind == "pidfd":
        try:
            ready, _, _ = select.select([val], [], [], timeout_s)
        except (OSError, ValueError):
            return True
        return bool(ready)
    if os.getppid() != int(val):
        return True
    if timeout_s > 0:
        time.sleep(timeout_s)
        return os.getppid() != int(val)
    return False


def _close_parent_watch(watch: tuple[str, object] | None) -> None:
    if watch is None:
        return
    kind, val = watch
    try:
        if kind == "winhandle":
            _win_kernel32().CloseHandle(val)
        elif kind == "pidfd":
            os.close(int(val))
    except Exception:  # noqa: BLE001
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GVD Reality View separate-process stub")
    ap.add_argument("--ping", action="store_true", help="Print schema metadata and exit")
    ap.add_argument("--serve", action="store_true", help="Hold stub boundary process until EOF/SIGTERM/parent-exit")
    ap.add_argument(
        "--parent-pid",
        type=int,
        default=None,
        help="Supervisor pid to watch (set by RvClient.launch). If already gone, the stub exits immediately.",
    )
    args = ap.parse_args(argv)

    if args.ping or not args.serve:
        print(
            json.dumps(
                {
                    "schema_version": _schema_version(),
                    "pid": os.getpid(),
                    "status": "stub",
                }
            ),
            flush=True,
        )
        return 0

    # Watch the supervisor pid, not getppid(): on Windows getppid() stays on a
    # dead parent but we still OpenProcess that pid; on Linux, getppid() is
    # reparented before main() if the parent died during interpreter startup,
    # so pidfd_open(getppid()) would watch the wrong process. If --parent-pid
    # is already gone, OpenProcess / pidfd_open fail and we exit immediately.
    expected_ppid = int(args.parent_pid) if args.parent_pid is not None and int(args.parent_pid) > 0 else os.getppid()
    if expected_ppid == os.getpid():
        return 0
    watch = _open_parent_watch(expected_ppid)
    if watch is None or _parent_has_exited(watch):
        _close_parent_watch(watch)
        return 0

    _install_pdeathsig()
    if _parent_has_exited(watch):
        _close_parent_watch(watch)
        return 0

    stop = False

    def _on_sig(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)

    try:
        stdin_fd: int | None = None
        if sys.platform != "win32":
            try:
                stdin_fd = sys.stdin.buffer.fileno()
            except (AttributeError, OSError, ValueError):
                stdin_fd = None

        if sys.platform == "win32":
            eof_evt = threading.Event()

            def _watch_stdin() -> None:
                try:
                    while not eof_evt.is_set():
                        chunk = sys.stdin.buffer.read(1)
                        if not chunk:
                            break
                except (OSError, ValueError):
                    pass
                finally:
                    eof_evt.set()

            t = threading.Thread(target=_watch_stdin, daemon=True)
            t.start()
            while not stop and not eof_evt.is_set():
                if _parent_has_exited(watch, timeout_s=_POLL_MS / 1000.0):
                    break
        elif stdin_fd is not None:
            wait_fds: list[object] = [stdin_fd]
            if watch[0] == "pidfd":
                wait_fds.append(watch[1])
            while not stop:
                if _parent_has_exited(watch):
                    break
                try:
                    ready, _, _ = select.select(wait_fds, [], [], _POLL_MS / 1000.0)
                except InterruptedError:
                    continue
                except (OSError, ValueError):
                    break
                if _parent_has_exited(watch):
                    break
                if stdin_fd in ready:
                    try:
                        chunk = os.read(stdin_fd, 64)
                    except InterruptedError:
                        continue
                    except OSError:
                        break
                    if not chunk:
                        break
        else:
            eof_evt = threading.Event()

            def _watch_stdin() -> None:
                try:
                    while not eof_evt.is_set():
                        chunk = sys.stdin.buffer.read(1)
                        if not chunk:
                            break
                except (OSError, ValueError):
                    pass
                finally:
                    eof_evt.set()

            t = threading.Thread(target=_watch_stdin, daemon=True)
            t.start()
            while not stop and not eof_evt.is_set():
                if _parent_has_exited(watch):
                    break
                eof_evt.wait(timeout=_POLL_MS / 1000.0)
    finally:
        _close_parent_watch(watch)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
