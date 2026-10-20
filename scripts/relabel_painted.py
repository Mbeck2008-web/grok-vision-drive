"""Offline relabel: write a painted-lane label file beside state.jsonl.

Needs no BeamNG session: ego pos/dir come from state.jsonl and the paint
from each row's own level zip. A row is labelled only against the paint of
the map it was recorded on (``map``, else the ``segment`` prefix). A row
whose map has no paint source is skipped and counted, never labelled
against another map. gt.jsonl and an existing output are never touched.
Train with GVD_GT_FILE=<out>.

    py -3 scripts/relabel_painted.py E:\\GVD_data\\scene_lanes --levels-dir D:\\BeamNG.tech\\...\\content\\levels
    py -3 scripts/relabel_painted.py ROOT --zip west_coast_usa=D:\\...\\west_coast_usa.zip --out gt_painted_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.debug.lane_targets import lane_center_is_valid, tech_from_painted  # noqa: E402
from python.debug.painted_lanes import PaintedLanes, decals_from_level_zip, is_paint  # noqa: E402


def map_of_row(row: Mapping[str, Any]) -> str:
    """Recorded map name, or ``""`` when the row does not say."""
    name = str(row.get("map") or "").strip()
    if name:
        return name
    seg = str(row.get("segment") or "").strip()
    return seg.split("#", 1)[0] if seg else ""


def zip_sources(pairs: list[str], levels_dir: str | None) -> dict[str, Path]:
    """``MAP=PATH`` (or a bare PATH named ``<map>.zip``) plus every zip in ``levels_dir``."""
    found: dict[str, Path] = {}
    if levels_dir:
        for path in sorted(Path(levels_dir).glob("*.zip")):
            found[path.stem] = path
    for item in pairs:
        name, sep, path = item.partition("=")
        if not sep:
            path, name = item, Path(item).stem
        found[name.strip()] = Path(path.strip())
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--zip", action="append", default=[], help="MAP=PATH to content/levels/<map>.zip; repeatable")
    ap.add_argument(
        "--levels-dir",
        default=None,
        help="Directory of level zips. Default $BNG_HOME/content/levels when BNG_HOME is set.",
    )
    ap.add_argument("--out", default="gt_painted.jsonl")
    args = ap.parse_args(argv)
    root = Path(args.root)
    out = root / args.out
    if out.name == "gt.jsonl" or out.exists():
        print(f"refusing to overwrite {out}", flush=True)
        return 2
    levels = args.levels_dir
    home = os.environ.get("BNG_HOME") or os.environ.get("BEAMNG_HOME")
    if levels is None and home:
        levels = str(Path(home) / "content" / "levels")
    sources = zip_sources(list(args.zip), levels)
    painted: dict[str, PaintedLanes | None] = {}

    def lanes_for(name: str) -> PaintedLanes | None:
        if name not in painted:
            path = sources.get(name)
            lanes = None
            if path is not None and path.is_file():
                lanes = PaintedLanes([d for d in decals_from_level_zip(path) if is_paint(d.get("material"))])
                if not lanes:
                    lanes = None
            painted[name] = lanes
            print(f"relabel map={name or '?'} paint_segments={0 if lanes is None else len(lanes.segs)}", flush=True)
        return painted[name]

    n = good = 0
    skipped: Counter[str] = Counter()
    with open(root / "state.jsonl", encoding="utf-8") as src, open(out, "x", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            name = map_of_row(row)
            lanes = lanes_for(name) if name else None
            if lanes is None:
                skipped[name or "<no map>"] += 1
                continue
            ego = row.get("ego") or {}
            pos, direction = ego.get("pos"), ego.get("dir")
            tech = tech_from_painted(lanes.lane_at(pos, direction) if pos and direction else [])
            tech["gt_map"] = name
            good += int(lane_center_is_valid(tech["lane_gt_valid"]))
            dst.write(json.dumps({"i": row.get("i"), "t": row.get("t"), "tech": tech}) + "\n")
    print(
        f"relabel rows={n} labelled={n - sum(skipped.values())} valid_5_10m={good} "
        f"skipped_no_paint_source={sum(skipped.values())} {dict(skipped)} -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
