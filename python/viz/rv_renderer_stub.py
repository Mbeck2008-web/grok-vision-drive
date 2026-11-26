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
import errno
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
_ERROR_INVALID_PARAMETER = 87  # OpenProcess on a pid that no longer exists
_POLL_MS = 200
_WATCH_MODE_LABEL = {"winhandle": "handle", "pidfd": "pidfd", "ppid": "ppid"}

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


def _win_open_process(ppid: int) -> tuple[object | None, int]:
    """OpenProcess(SYNCHRONIZE) on ppid. Returns (handle, 0) or (None, GetLastError())."""
    handle = _win_kernel32().OpenProcess(_SYNCHRONIZE, 0, int(ppid))
    if handle:
        return handle, 0
    return None, int(ctypes.get_last_error())


def _open_win_watch(ppid: int) -> tuple[str, object] | None:
    """Only ERROR_INVALID_PARAMETER means the pid is gone; other errors (e.g. access
    denied) fall back to the ppid poll rather than exiting under a live parent."""
    handle, err = _win_open_process(ppid)
    if handle:
        return ("winhandle", handle)
    if err == _ERROR_INVALID_PARAMETER:
        return None
    return ("ppid", int(ppid))


def _open_pidfd_watch(ppid: int) -> tuple[str, object] | None:
    """Only ESRCH means the pid is gone; ENOSYS/EMFILE/ENFILE/EPERM/other errors fall
    back to the ppid poll rather than exiting under a live parent."""
    try:
        return ("pidfd", os.pidfd_open(int(ppid), 0))
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return None
        return ("ppid", int(ppid))


def _open_parent_watch(ppid: int) -> tuple[str, object] | None:
    """Open a process-object watch on the parent.

    Returns None only if the parent is known to be gone (caller must exit immediately).
    Windows: OpenProcess(SYNCHRONIZE). Linux: pidfd_open. Else (or if the native
    watch cannot be opened for a live parent) a getppid poll.
    """
    if sys.platform == "win32":
        return _open_win_watch(ppid)
    if hasattr(os, "pidfd_open"):
        return _open_pidfd_watch(ppid)
    return ("ppid", int(ppid))


def _emit(line: str) -> None:
    """Best-effort status line on stdout (RvClient sends stdout to DEVNULL)."""
    try:
        out = sys.stdout
        if out is None:
            return
        out.write(line + "\n")
        out.flush()
    except (OSError, ValueError, AttributeError):
        pass


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
        _emit("RV_STUB_EXIT reason=parent_gone_at_start")
        return 0

    _install_pdeathsig()
    if _parent_has_exited(watch):
        _close_parent_watch(watch)
        _emit("RV_STUB_EXIT reason=parent_gone_at_start")
        return 0

    stop = False

    def _on_sig(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)

    # Armed handshake: printed only after the parent watch is open and re-checked.
    _emit(f"RV_WATCH_ARMED mode={_WATCH_MODE_LABEL.get(watch[0], watch[0])} parent_pid={expected_ppid}")
    reason = "signal"
    try:
        stdin_fd: int | None = None
        if sys.platform != "win32":
            try:
                stdin_fd = sys.stdin.buffer.fileno()
            except (AttributeError, OSError, ValueError):
                stdin_fd = None

        if stdin_fd is None:
            # Windows (and POSIX without a usable stdin fd): stdin EOF on a thread,
            # parent watch polled every _POLL_MS.
            eof_evt = threading.Event()

            def _watch_stdin() -> None:
                try:
                    while not eof_evt.is_set():
                        chunk = sys.stdin.buffer.read(1)
                        if not chunk:
                            break
                except (OSError, ValueError, AttributeError):
                    pass
                finally:
                    eof_evt.set()

            t = threading.Thread(target=_watch_stdin, daemon=True)
            t.start()
            while not stop:
                if eof_evt.is_set():
                    reason = "stdin_eof"
                    break
                if watch[0] == "winhandle":
                    if _parent_has_exited(watch, timeout_s=_POLL_MS / 1000.0):
                        reason = "parent_exit"
                        break
                else:
                    if _parent_has_exited(watch):
                        reason = "parent_exit"
                        break
                    eof_evt.wait(timeout=_POLL_MS / 1000.0)
        else:
            wait_fds: list[object] = [stdin_fd]
            if watch[0] == "pidfd":
                wait_fds.append(watch[1])
            while not stop:
                if _parent_has_exited(watch):
                    reason = "parent_exit"
                    break
                try:
                    ready, _, _ = select.select(wait_fds, [], [], _POLL_MS / 1000.0)
                except InterruptedError:
                    continue
                except (OSError, ValueError):
                    reason = "error"
                    break
                if _parent_has_exited(watch):
                    reason = "parent_exit"
                    break
                if stdin_fd in ready:
                    try:
                        chunk = os.read(stdin_fd, 64)
                    except InterruptedError:
                        continue
                    except OSError:
                        reason = "error"
                        break
                    if not chunk:
                        reason = "stdin_eof"
                        break
    finally:
        _close_parent_watch(watch)

    _emit(f"RV_STUB_EXIT reason={reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
