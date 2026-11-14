#!/usr/bin/env python3
"""Build ``junctions/<map>.json.gz`` from a saved road dump, offline.

    py -3 scripts/build_junction_map.py tests/fixtures/roads/west_coast_usa.json.gz --out junctions/

The dump is the gzip ``python/collect/road_dump.py`` writes (``map`` and
``road_network``). The output is canonical JSON (sorted keys, no spaces) in a
gzip with mtime 0, so two builds of the same dump are the same bytes. It is
create-only: an existing map is kept and the run exits 1. ``src_sha256`` is the
sha256 of the dump file; ``jct_id`` values are stable only for one
``src_sha256`` and one set of ``params``.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.collect import junction_map as jm  # noqa: E402


def build_from_dump(dump: Path | str, map_name: str | None = None) -> dict:
    raw = Path(dump).read_bytes()
    data = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    name = map_name or str(data.get("map") or Path(dump).name.split(".")[0])
    return jm.build_junction_map(data["road_network"], name, hashlib.sha256(raw).hexdigest())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dump", type=Path, help="Road dump (.json.gz) with road_network.")
    parser.add_argument("--out", type=Path, default=ROOT / "junctions", help="Output folder (default junctions/).")
    parser.add_argument("--map", default=None, help="Map name (default: the dump's map field).")
    args = parser.parse_args(argv)
    doc = build_from_dump(args.dump, args.map)
    args.out.mkdir(parents=True, exist_ok=True)
    path = jm.map_path(args.out, doc["map"])
    try:
        jm.write_map(doc, args.out)
    except FileExistsError:
        print(f"build jct {path.name} exists in {args.out}; not written", flush=True)
        return 1
    s = jm.summary(doc)
    print(f"build jct map={doc['map']} junctions={s['junctions']} major={s['major']} minor={s['minor']} "
          f"types={json.dumps(s['types'], sort_keys=True)} sha256={jm.map_sha256(doc)} "
          f"bytes={path.stat().st_size} -> {path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
