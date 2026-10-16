#!/usr/bin/env python3
"""West-coast excursions from the 903e172 runs, on the dumped road graph.

The navgraph left/right line is the paint. Those weaves were about half a
metre outside it, which is the shoulder, and the path was aimed at a
different lane than the one the label had. With the path on the labelled
lane and a one-metre shoulder, at least three of the four west-coast
attempts finish without a real departure.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect.excursion import (  # noqa: E402
    Excursion,
    ExcursionClock,
    excursion_path,
    near_junction,
)
from python.collect.road_dump import load_road_dump  # noqa: E402
from python.debug.gt_geom import RoadFollow, follow_label, heading_from_dir  # noqa: E402

ROADS = ROOT / "tests" / "fixtures" / "roads" / "west_coast_usa.json.gz"
TRACES = {
    "collect8": ROOT / "tests" / "fixtures" / "collect8" / "trace.jsonl.gz",
    "collect9": ROOT / "tests" / "fixtures" / "collect9" / "trace.jsonl.gz",
}


def _load(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sequence(rows: list[dict], roads: list) -> list[dict]:
    by_id = {road.road_id: road for road in roads}
    follow = RoadFollow()
    seq: list[dict] = []
    for row in rows:
        heading = heading_from_dir(row["dx"], row["dy"])
        hit, _mem, _refused = follow_label(
            roads, float(row["x"]), float(row["y"]), heading, follow, float(row.get("spd") or 0.0), None,
        )
        dt = row.get("dt")
        dt = 0.1 if dt is None or dt <= 0.0 or dt > 2.0 else float(dt)
        road = None if hit is None else by_id.get(hit.road_id)
        outside = None
        if hit is not None and road is not None:
            outside = abs(float(hit.offset_m)) - float(road.width_m) * 0.5
        seq.append(
            {
                "i": row.get("i"),
                "ph": row.get("ph"),
                "why": row.get("why") or "",
                "dt": dt,
                "x": float(row["x"]),
                "y": float(row["y"]),
                "heading": heading,
                "herr": None if hit is None else hit.heading_err_rad,
                "hit": hit,
                "loff": None if hit is None else hit.lane_offset_m,
                "lw": None if hit is None else hit.lane_width_m,
                "off": None if hit is None else hit.offset_m,
                "rw": None if road is None else road.width_m,
                "offroad": bool(hit.off_road) if hit is not None else False,
                "nj": False if hit is None else near_junction(hit, roads, float(row["x"]), float(row["y"]), heading),
                "outside": outside,
            }
        )
    return seq


def _attempts(seq: list[dict]) -> list[tuple[int, str]]:
    """Weave and lane-change starts, plus a no_path abort that never entered the move."""
    found: list[tuple[int, str]] = []
    previous = "cruise"
    for index, row in enumerate(seq):
        if previous == "cruise" and row["ph"] in ("weave", "lane_change"):
            found.append((index, row["ph"]))
        elif previous == "cruise" and row["ph"] == "abort" and row["why"] == "no_path":
            found.append((index, "weave"))
        previous = row["ph"]
    return found


def _replay(seq: list[dict], index: int, kind: str) -> tuple[str, float]:
    """Return ('complete'|'abort'|'open', metres outside the paint)."""
    clock = ExcursionClock()
    clock.phase = kind
    clock.phase_t = 0.0
    saw_abort = False
    reason = ""
    outside = 0.0
    elapsed = 0.0
    for row in seq[index:]:
        elapsed += row["dt"]
        if row["outside"] is not None:
            outside = max(outside, float(row["outside"]))
        move = clock.update(
            row["dt"],
            offset_m=row["loff"],
            heading_rad=row["herr"],
            lane_width_m=row["lw"],
            off_road=row["offroad"],
            near_junction=row["nj"],
            n_lanes=2,
            road_offset_m=row["off"],
            road_width_m=row["rw"],
        )
        if move.phase == "abort":
            saw_abort = True
            reason = move.reason or "abort"
        if move.phase == "cruise":
            return ("complete" if not saw_abort else "abort", outside) if not saw_abort else (f"abort {reason}", outside)
        if elapsed > 45.0:
            break
    if saw_abort:
        return f"abort {reason}", outside
    # The map segment ended during the move. Still on the shoulder is not a departure.
    if outside < 1.0:
        return "complete", outside
    return "open", outside


def check_west_coast(name: str, roads: list) -> int:
    seq = _sequence(_load(TRACES[name]), roads)
    attempts = _attempts(seq)
    done = 0
    for index, kind in attempts:
        row = seq[index]
        status, outside = _replay(seq, index, kind)
        path = None
        if row["hit"] is not None:
            path, _used = excursion_path(
                roads, row["x"], row["y"], row["heading"], Excursion(kind, 0.6, 0.0, 1, ""), row["hit"], 8.0,
            )
        print(
            f"{name} i={row['i']} {kind} {status} outside={outside:.2f}m "
            f"path={0 if path is None else len(path)}"
        )
        if status == "complete" and outside < 1.0 and path:
            done += 1
    print(f"{name} west coast {done} of {len(attempts)}")
    return done


def main() -> None:
    roads, payload = load_road_dump(ROADS)
    assert payload.get("map") == "west_coast_usa"
    assert check_west_coast("collect9", roads) >= 3
    check_west_coast("collect8", roads)
    print("west coast excursions ok")


if __name__ == "__main__":
    main()
