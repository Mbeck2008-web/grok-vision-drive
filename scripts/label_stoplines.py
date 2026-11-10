#!/usr/bin/env python3
"""Write a stop-line sidecar from painted decals. Never overwrites gt.jsonl.

Train/eval only. The planner and visualizer must not import this file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.debug.painted_lanes import PaintedLanes  # noqa: E402


def _pose(row: dict[str, Any]) -> tuple[Any, Any] | None:
    pos, direction = row.get("pos"), row.get("dir") or row.get("direction")
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return None
    if not isinstance(direction, (list, tuple)) or len(direction) < 2:
        return None
    return pos, direction


def label_row(paint: PaintedLanes, row: dict[str, Any]) -> dict[str, Any]:
    pose = _pose(row)
    bars: list[dict[str, Any]] = []
    if pose is not None:
        pos, direction = pose
        bars = paint.stop_lines_ahead(pos, direction) + paint.masked_cross_paint(pos, direction)
    return {"t": row.get("t"), "map": row.get("map"), "bars": bars}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Label stop lines from painted decals (sidecar only)")
    ap.add_argument("root", help="Recording folder with state.jsonl")
    ap.add_argument("--decals", help="JSON list of synthetic or cached decals")
    ap.add_argument("--out", default="stoplines.jsonl")
    args = ap.parse_args(argv)
    root = Path(args.root)
    out = root / args.out
    if out.name == "gt.jsonl":
        print(f"refusing to overwrite {out}", flush=True)
        return 2
    if out.exists():
        print(f"refusing to overwrite {out}", flush=True)
        return 2
    state = root / "state.jsonl"
    if not state.is_file():
        print(f"missing {state}", flush=True)
        return 2
    if not args.decals:
        print("label_stoplines: --decals is required (level-zip path is a later hook)", flush=True)
        return 2
    decals = json.loads(Path(args.decals).read_text(encoding="utf-8"))
    paint = PaintedLanes(decals)
    with state.open(encoding="utf-8") as fh, out.open("w", encoding="utf-8") as dest:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            dest.write(json.dumps(label_row(paint, row), sort_keys=True) + "\n")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
