#!/usr/bin/env python3
"""Write signs.jsonl from signals and state.jsonl. Train-only.

``--signals`` as ``.jsonl`` is one ``get_map_node_signals`` payload per tick
(or ``{t, err}`` when the API was missing). A single JSON file is the older
static dump, stamped onto every row; it is not the live lamp.

Does not talk to BeamNG. This CLI stays offline.
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

from python.collect.signal_labels import label_signs, sidecar_row  # noqa: E402
from python.collect.static_signs import (  # noqa: E402
    STATIC_SIGNS_NAME,
    load_static_catalog,
    merge_static_into_rows,
)


def _load_signals(path: Path) -> tuple[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return "ticks", rows
    return "dump", json.loads(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export train-only sign/lamp GT sidecar")
    ap.add_argument("root", help="Recording folder with state.jsonl")
    ap.add_argument(
        "--signals",
        required=True,
        help="Per-tick signals.jsonl, or one JSON dump of get_map_node_signals",
    )
    ap.add_argument("--out", default="signs.jsonl")
    ap.add_argument(
        "--static-signs",
        action="store_true",
        help=(
            "Merge static_signs.json into the sidecar. Off by default; the shape "
            "allowlist is still unverified."
        ),
    )
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
    state_rows = []
    with state.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                state_rows.append(json.loads(line))
    kind, payload = _load_signals(Path(args.signals))
    if kind == "ticks":
        labelled = label_signs(state_rows, payload)
    else:
        labelled = [
            sidecar_row(
                t=row.get("t"),
                raw=payload,
                err=None,
                ego_pos=row.get("pos"),
                ego_dir=row.get("dir") or row.get("direction"),
            )
            for row in state_rows
        ]
    if args.static_signs:
        catalog_path = root / STATIC_SIGNS_NAME
        catalog = load_static_catalog(catalog_path)
        if catalog is not None:
            labelled = merge_static_into_rows(labelled, state_rows, catalog)
        elif catalog_path.is_file():
            print(
                f"[GVD] stop-sign source: absent on {root.name}: unrecognized {STATIC_SIGNS_NAME}",
                flush=True,
            )
    with out.open("w", encoding="utf-8") as dest:
        for row in labelled:
            dest.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
