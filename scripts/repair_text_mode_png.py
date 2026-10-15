#!/usr/bin/env python3
"""Repair PNGs that Windows text mode expanded (LF became CRLF).

    py -3 scripts\\repair_text_mode_png.py E:\\GVD_data\\scene_lanes

Writes ``raw_repaired/<cam>/NNNNNN.png`` beside ``raw/``. Originals stay
where they are and are never deleted. The last line is the rescued fraction.
Exit 0 when every PNG decodes. Exit 1 when some do not. Exit 2 when no PNGs
were found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.data.png_repair import repair_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair text-mode PNG captures in place-safe copies")
    parser.add_argument("root", type=Path, help="scene_lanes directory, raw/ directory, or one PNG")
    parser.add_argument("--dry-run", action="store_true", help="Count rescues without writing copies")
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many PNGs (0 = all)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.root.exists():
        print(f"repair: no {args.root}", file=sys.stderr)
        return 2
    report = repair_dataset(args.root, limit=max(0, int(args.limit)), dry_run=bool(args.dry_run))
    total = int(report["total"])
    rescued = int(report["rescued"])
    fraction = float(report["fraction"])
    print(
        f"repair rescued {rescued}/{total} ({fraction:.4f}) "
        f"already_valid={report['already_valid']} repaired={report['repaired']} "
        f"failed={report['failed']}",
        flush=True,
    )
    for name in report["failed_names"]:
        print(f"repair failed {name}", flush=True)
    if total == 0:
        return 2
    return 0 if rescued == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
