"""Offline relabel: write gt_painted.jsonl beside state.jsonl from painted line decals.

Needs no BeamNG session: ego pos/dir come from state.jsonl and the paint
from the level zip. gt.jsonl is not touched. Train with GVD_GT_FILE=gt_painted.jsonl.

    py -3 scripts/relabel_painted.py E:\\GVD_data\\scene_lanes --zip D:\\BeamNG.tech\\...\\levels\\west_coast_usa.zip
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.debug.lane_targets import lane_center_is_valid, tech_from_painted  # noqa: E402
from python.debug.painted_lanes import PaintedLanes, decals_from_level_zip, is_paint  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--zip", required=True, help="content/levels/<map>.zip of the recorded map")
    ap.add_argument("--out", default="gt_painted.jsonl")
    args = ap.parse_args(argv)
    root = Path(args.root)
    out = root / args.out
    if out.exists():
        print(f"refusing to overwrite {out}", flush=True)
        return 2
    lanes = PaintedLanes([d for d in decals_from_level_zip(args.zip) if is_paint(d.get("material"))])
    n = good = 0
    with open(root / "state.jsonl", encoding="utf-8") as src, open(out, "w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            ego = row.get("ego") or {}
            pos, direction = ego.get("pos"), ego.get("dir")
            tech = tech_from_painted(lanes.lane_at(pos, direction) if pos and direction else [])
            n += 1
            good += int(lane_center_is_valid(tech["lane_gt_valid"]))
            dst.write(json.dumps({"i": row.get("i"), "t": row.get("t"), "tech": tech}) + "\n")
    print(f"relabel rows={n} valid_5_10m={good} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
