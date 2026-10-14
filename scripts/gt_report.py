#!/usr/bin/env python3
"""Summarize a GVD ground-truth JSONL log.

Prints lane-keep, departures, steer chatter, how lane_x tracks the true
offset, and frame-to-command latency. Writes a PNG beside the log unless
--png is set. Exits 1 when --min-in-lane is set and the engaged in-lane
percentage is missing or below it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.debug.gt_summary import (  # noqa: E402
    distance_failure,
    format_report,
    load_jsonl,
    render_plot,
    summarize,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ground-truth lane-keep report")
    ap.add_argument("log", type=Path, help="logs/gt_<stamp>.jsonl")
    ap.add_argument("--png", type=Path, default=None, help="Plot path. Default: next to the log.")
    ap.add_argument(
        "--min-in-lane",
        type=float,
        default=None,
        help="Exit 1 when engaged in-lane %% is below this (0-100).",
    )
    args = ap.parse_args(argv)
    if not args.log.is_file():
        print(f"gt report: missing {args.log}", file=sys.stderr)
        return 2
    rows = load_jsonl(args.log)
    if not rows:
        print(f"gt report: empty {args.log}", file=sys.stderr)
        return 2
    summary = summarize(rows)
    print(format_report(summary, source=str(args.log)))
    png = args.png if args.png is not None else args.log.with_suffix(".png")
    try:
        render_plot(rows, png)
        print(f"plot={png}")
    except Exception as exc:
        print(f"plot=failed {type(exc).__name__}: {exc}", file=sys.stderr)
    failed = False
    reason = distance_failure(summary)
    if reason is not None:
        print(f"gt report: FAIL {reason}", file=sys.stderr)
        failed = True
    if args.min_in_lane is not None:
        pct = summary.get("in_lane_pct")
        if pct is None or float(pct) < float(args.min_in_lane):
            print(
                f"gt report: FAIL in_lane_pct={pct} need>={args.min_in_lane}",
                file=sys.stderr,
            )
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
