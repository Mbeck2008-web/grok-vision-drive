#!/usr/bin/env python3
"""Check a scene-lane recording and write a report plus montage PNGs.

    py -3 scripts\\validate_scene_lanes.py --strips E:\\GVD_data\\scene_lanes

The report and the PNGs go to ``E:\\GVD_data\\status\\validate`` unless
``--out`` is set. Exit 1 when the report has issues. Exit 2 when the
folder has no state file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.debug.dataset_check import validate_dataset  # noqa: E402
from python.debug.remote_status import status_dir  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a scene-lane recording")
    parser.add_argument("--strips", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Report directory. Default: <data>/status/validate")
    parser.add_argument("--montage", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state = args.strips / "state.jsonl"
    if not state.is_file():
        print(f"validate: no {state}", file=sys.stderr)
        return 2
    out = args.out
    if out is None:
        base = status_dir()
        if base is None:
            print("validate: set GVD_DATA_DIR or pass --out", file=sys.stderr)
            return 2
        out = base / "validate"
    report = validate_dataset(args.strips, out, montage_n=max(1, int(args.montage)))
    per = report.get("per_cam") or {}
    per_text = " ".join(f"{name}={count}" for name, count in sorted(per.items()))
    rows_n = int(report.get("rows") or 0)
    masked_n = report.get("masked_rows")
    masked_text = ""
    if masked_n is not None and rows_n:
        masked_text = f" masked={int(masked_n)}/{rows_n} ({float(report.get('masked_fraction') or 0.0):.2f})"
    print(
        f"validate rows={report['rows']} jpegs={report['jpegs']} "
        f"raw={report.get('raw_frames', 0)}"
        + (f" {per_text}" if per_text else "")
        + masked_text
        + f" issues={report['issue_count']} report={out / 'report.json'}",
        flush=True,
    )
    for issue in report["issues"][:20]:
        print(f"validate {issue}", flush=True)
    return 1 if report["issue_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
