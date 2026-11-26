"""Separate-process Reality View stub entrypoint (Step 1 process boundary).

Runs only as a child process (``python -m python.viz.rv_renderer_stub``).
Contains no renderer, no unprojection, no 3DGS, and no safety-path imports.
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

from python.viz.rv_contract import RV_SCHEMA_VERSION

_PR_SET_PDEATHSIG = 1


def _install_pdeathsig() -> int | None:
    """On Linux, set PR_SET_PDEATHSIG(SIGTERM) via ctypes libc; no-op elsewhere."""
    if not sys.platform.startswith("linux"):
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GVD Reality View separate-process stub")
    ap.add_argument("--ping", action="store_true", help="Print schema metadata and exit")
    ap.add_argument("--serve", action="store_true", help="Hold stub boundary process until EOF/SIGTERM/parent-exit")
    args = ap.parse_args(argv)

    if args.ping or not args.serve:
        print(
            json.dumps(
                {
                    "schema_version": RV_SCHEMA_VERSION,
                    "pid": os.getpid(),
                    "status": "stub",
                }
            ),
            flush=True,
        )
        return 0

    initial_ppid = os.getppid()
    _install_pdeathsig()

    stop = False

    def _on_sig(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)

    stdin_fd: int | None = None
    if sys.platform != "win32":
        try:
            stdin_fd = sys.stdin.buffer.fileno()
        except (AttributeError, OSError, ValueError):
            stdin_fd = None

    if stdin_fd is not None:
        while not stop:
            if os.getppid() != initial_ppid:
                break
            try:
                ready, _, _ = select.select([stdin_fd], [], [], 0.2)
            except InterruptedError:
                continue
            except (OSError, ValueError):
                break
            if ready:
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
            if os.getppid() != initial_ppid:
                break
            eof_evt.wait(timeout=0.2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
