#!/usr/bin/env python3
"""Write signs.jsonl from a signals dump and state.jsonl. Train-only.

Does not talk to BeamNG unless a later caller passes a live session into
``read_map_node_signals``. This CLI reads a JSON dump so it stays offline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.collect.signal_labels import sidecar_row  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export train-only sign/lamp GT sidecar")
    ap.add_argument("root", help="Recording folder with state.jsonl")
    ap.add_argument("--signals", required=True, help="JSON dump of get_map_node_signals (or empty list)")
    ap.add_argument("--out", default="signs.jsonl")
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
    raw = json.loads(Path(args.signals).read_text(encoding="utf-8"))
    with state.open(encoding="utf-8") as fh, out.open("w", encoding="utf-8") as dest:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            dest.write(
                json.dumps(
                    sidecar_row(
                        t=row.get("t"),
                        raw=raw,
                        err=None,
                        ego_pos=row.get("pos"),
                        ego_dir=row.get("dir") or row.get("direction"),
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
