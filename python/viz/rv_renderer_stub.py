"""Separate-process Reality View stub entrypoint (Step 1 process boundary).

Runs only as a child process (``python -m python.viz.rv_renderer_stub``).
Contains no renderer, no unprojection, no 3DGS, and no safety-path imports.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys

from python.viz.rv_contract import RV_SCHEMA_VERSION


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GVD Reality View separate-process stub")
    ap.add_argument("--ping", action="store_true", help="Print schema metadata and exit")
    ap.add_argument("--serve", action="store_true", help="Hold stub boundary process until EOF/SIGTERM")
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

    stop = False

    def _on_sig(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)

    while not stop:
        try:
            chunk = sys.stdin.buffer.read(64)
        except OSError:
            break
        if not chunk:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
