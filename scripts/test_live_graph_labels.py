#!/usr/bin/env python3
"""Replay collect5 and collect6 on the dumped Tech road graphs.

The graphs live in ``tests/fixtures/roads/<map>.json.gz``. A crossing edge
must not become the label. Cruise rows that do get a label stay under
0.30 rad of heading error, and adjacent ticks do not jump the 5 m lane
center by more than 2 m.
"""
from __future__ import annotations

import gzip
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect.road_dump import load_road_dump  # noqa: E402
from python.debug.gt_geom import (  # noqa: E402
    RoadFollow,
    RoadPoly,
    ego_center_x,
    follow_label,
    heading_from_dir,
    parse_road_edges,
)

ROADS = ROOT / "tests" / "fixtures" / "roads"
COLLECT5 = ROOT / "tests" / "fixtures" / "collect5" / "trace.jsonl.gz"
COLLECT6 = ROOT / "tests" / "fixtures" / "collect6" / "trace.jsonl.gz"
# West-coast cruise std is the car's real lateral wander on the AI route
# along that centerline, not a label snap (adjacent jumps stay 0). The bar
# is 1.2 there. Italy stays under 0.8. A missing label is an ambiguous tick;
# up to 35% of cruise rows may be masked.
STD_LIMIT = {
    ("collect6", "west_coast_usa"): 1.2,
    ("collect5", "west_coast_usa"): 1.2,
    ("collect6", "italy"): 0.80,
    ("collect5", "italy"): 0.80,
}
MASK_LIMIT = 0.35


def _load(path: Path, level: str) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [row for row in rows if str(row.get("seg", "")).startswith(level)]


def _metrics(rows: list[dict], roads: list) -> dict[str, float]:
    follow = RoadFollow()
    values: list[float] = []
    jumps = 0
    cruise = 0
    prev: float | None = None
    prev_i: int | None = None
    over = 0
    for index, row in enumerate(rows):
        heading = heading_from_dir(row["dx"], row["dy"])
        hit, _mem, _refused = follow_label(
            roads,
            float(row["x"]),
            float(row["y"]),
            heading,
            follow,
            float(row.get("spd") or 0.0),
            None,
        )
        if row.get("ph") != "cruise":
            prev = None
            prev_i = None
            continue
        cruise += 1
        if (
            hit is None
            or hit.off_road
            or hit.heading_err_rad is None
            or hit.lane_offset_m is None
        ):
            prev = None
            prev_i = None
            continue
        if abs(hit.heading_err_rad) > 0.30 + 1e-6:
            over += 1
        center = ego_center_x(hit.lane_offset_m, hit.heading_err_rad, 5.0)
        values.append(center)
        if prev is not None and prev_i is not None and index - prev_i == 1 and abs(center - prev) > 2.0:
            jumps += 1
        prev = center
        prev_i = index
    return {
        "cruise": float(cruise),
        "emitted": float(len(values)),
        "std": statistics.pstdev(values) if len(values) > 2 else 0.0,
        "jumps": float(jumps),
        "over": float(over),
    }


def check_replay(name: str, path: Path, level: str) -> None:
    roads, payload = load_road_dump(ROADS / f"{level}.json.gz")
    assert payload.get("map") == level
    assert len(roads) > 10
    stats = _metrics(_load(path, level), roads)
    limit = STD_LIMIT[(name, level)]
    print(
        f"{name} {level} cruise={int(stats['cruise'])} emitted={int(stats['emitted'])} "
        f"({100.0 * stats['emitted'] / max(stats['cruise'], 1.0):.1f}%) "
        f"std={stats['std']:.3f} jumps={int(stats['jumps'])} heading_over={int(stats['over'])}"
    )
    assert stats["jumps"] == 0, stats
    assert stats["over"] == 0, stats
    assert stats["std"] < limit, stats
    masked = 1.0 - stats["emitted"] / max(stats["cruise"], 1.0)
    assert masked <= MASK_LIMIT, stats


def check_a_crossing_edge_is_not_the_label() -> None:
    """The car drives north. An east-west road through the same point is not the label."""
    main = RoadPoly("main", [(0.0, float(i)) for i in range(0, 80)], 8.0, lanes=2)
    cross = RoadPoly("cross", [(float(i), 40.0) for i in range(-20, 21)], 8.0, lanes=2)
    follow = RoadFollow()
    seen = 0
    for y in range(5, 70):
        hit, _mem, _refused = follow_label(
            [main, cross], 1.2, float(y), math.pi / 2.0, follow, 8.0, None
        )
        if hit is None:
            continue
        seen += 1
        assert hit.road_id == "main", (y, hit.road_id, hit.heading_err_rad)
        assert hit.heading_err_rad is not None and abs(hit.heading_err_rad) <= 0.30
    assert seen > 40


def check_median_width_ignores_a_flare() -> None:
    edges = [
        {"left": [-3, float(i), 0], "right": [3, float(i), 0], "middle": [0, float(i), 0]}
        for i in range(0, 25, 5)
    ]
    edges.append({"left": [-5, 30.0, 0], "right": [5, 30.0, 0], "middle": [0, 30.0, 0]})
    poly = parse_road_edges("flare", edges)
    assert poly is not None and abs(poly.width_m - 6.0) < 1e-6, poly


def main() -> None:
    check_a_crossing_edge_is_not_the_label()
    check_median_width_ignores_a_flare()
    for name, path in (("collect5", COLLECT5), ("collect6", COLLECT6)):
        for level in ("west_coast_usa", "italy"):
            check_replay(name, path, level)
    print("live graph labels ok")


if __name__ == "__main__":
    main()
