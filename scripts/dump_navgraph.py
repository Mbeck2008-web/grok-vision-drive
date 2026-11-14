#!/usr/bin/env python3
"""Save BeamNG's navgraph for the loaded level: one ``NavigraphData`` read (live).

    py -3 scripts/dump_navgraph.py --level west_coast_usa --out logs/navgraph_wcu.json.gz

Freeze first: this refuses to run unless the committed golden
(``tests/fixtures/junctions/<level>.golden.json``) has the sha256 of a fresh
build of ``tests/fixtures/roads/<level>.json.gz``, and the committed
``junctions/<level>.json.gz`` is that same build. The navgraph metric is then
scored against a map that was frozen before the navgraph was seen.

It attaches to a running BeamNG.tech (it never launches one) with the level
already loaded, sends one ``GetRoadGraph`` request and writes ``{"level",
"graph", "coords", "widths"}`` as gzip JSON, create-only. Score it with
``render_junction_overlay.py --navgraph``. Nothing is sent to the vehicle.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.collect import junction_map as jm  # noqa: E402

GOLDEN_DIR = ROOT / "tests" / "fixtures" / "junctions"
DUMP_DIR = ROOT / "tests" / "fixtures" / "roads"
MAPS_DIR = ROOT / "junctions"


def _fresh_build(level: str) -> bytes:
    import build_junction_map  # scripts/ sibling

    return jm.canonical_bytes(build_junction_map.build_from_dump(DUMP_DIR / f"{level}.json.gz", level))


def golden_matches(level: str, build: Callable[[], bytes] | None = None) -> tuple[bool, str]:
    """``(True, "")`` when the committed golden, the committed map and a fresh build agree."""
    golden_path = GOLDEN_DIR / f"{level}.golden.json"
    if not golden_path.is_file():
        return False, f"no golden {golden_path.name}"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    blob = (build or (lambda: _fresh_build(level)))()
    fresh = hashlib.sha256(blob).hexdigest()
    if fresh != golden.get("sha256"):
        return False, f"fresh build sha {fresh[:12]} != golden sha {str(golden.get('sha256'))[:12]}"
    committed = jm.map_path(MAPS_DIR, level)
    if not committed.is_file():
        return False, f"no committed map {committed.name}"
    _doc, sha = jm.load_map_sha(committed)
    if sha != golden["sha256"]:
        return False, f"committed map sha {sha[:12]} != golden sha {golden['sha256'][:12]}"
    return True, ""


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    for attr in ("x", "y", "z"):
        if not hasattr(value, attr):
            break
    else:
        return [float(value.x), float(value.y), float(value.z)]
    return value


def read_navgraph(bng: Any) -> dict[str, Any]:
    from beamngpy.tools import NavigraphData  # type: ignore

    nav = NavigraphData(bng)
    return {"graph": _plain(nav.graph), "coords": _plain(nav.coords3d), "widths": _plain(nav.widths)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--level", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    ok, why = golden_matches(args.level)
    if not ok:
        print(f"navgraph refused: {why}", flush=True)
        return 2
    if args.out.exists():
        print(f"navgraph {args.out.name} exists; not written", flush=True)
        return 1
    from python.sensors.tech import load_tech_config, open_tech_beamngpy, release_tech_beamngpy

    cfg = load_tech_config()
    host = args.host or str(cfg.get("host") or "localhost")
    port = int(args.port if args.port is not None else cfg.get("port") or 25252)
    bng = open_tech_beamngpy(host, port, home=cfg.get("home") or None, user=cfg.get("user") or None, launch=False)
    try:
        doc = {"level": args.level, **read_navgraph(bng)}
    finally:
        release_tech_beamngpy(bng)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "xb") as fh:
        fh.write(gzip.compress(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8"), mtime=0))
    links = sum(1 for nbrs in doc["graph"].values() if len(nbrs) > 2)
    print(f"navgraph level={args.level} nodes={len(doc['graph'])} candidates>2={links} -> {args.out.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
