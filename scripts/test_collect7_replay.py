#!/usr/bin/env python3
"""Replay the 72f8dbd 6-minute run on the dumped road graphs.

Five weaves and two lane changes started. All of them aborted, mostly
``road_edge``, while the car was still inside the median paved span.
The paved-edge rule should let at least four of those seven finish,
with a lateral swing of at least half a metre.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect.excursion import ExcursionClock, near_junction  # noqa: E402
from python.collect.road_dump import load_road_dump  # noqa: E402
from python.debug.gt_geom import RoadFollow, follow_label, heading_from_dir  # noqa: E402

TRACE = ROOT / "tests" / "fixtures" / "collect7" / "trace.jsonl.gz"
ROADS = ROOT / "tests" / "fixtures" / "roads"


def _rows(level: str) -> list[dict]:
    with gzip.open(TRACE, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [row for row in rows if str(row.get("seg", "")).startswith(level)]


def _sequence(level: str) -> list[dict]:
    roads, _payload = load_road_dump(ROADS / f"{level}.json.gz")
    by_id = {road.road_id: road for road in roads}
    follow = RoadFollow()
    seq: list[dict] = []
    for row in _rows(level):
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
        dt = row.get("dt")
        dt = 0.1 if dt is None or dt <= 0.0 or dt > 2.0 else float(dt)
        road = None if hit is None else by_id.get(hit.road_id)
        seq.append(
            {
                "ph": row.get("ph"),
                "dt": dt,
                "h": None if hit is None else hit.heading_err_rad,
                "loff": None if hit is None else hit.lane_offset_m,
                "lw": None if hit is None else hit.lane_width_m,
                "off": None if hit is None else hit.offset_m,
                "rw": None if road is None else road.width_m,
                "offroad": bool(hit.off_road) if hit is not None else False,
                "nj": False
                if hit is None
                else near_junction(hit, roads, float(row["x"]), float(row["y"]), heading),
            }
        )
    return seq


def _finish(seq: list[dict], index: int, kind: str) -> tuple[bool, float, str]:
    """Drive the recorded poses with the paved-edge abort rule."""
    clock = ExcursionClock()
    clock.phase = kind
    clock.phase_t = 0.0
    saw_abort = False
    reason = ""
    offsets: list[float] = []
    elapsed = 0.0
    for row in seq[index:]:
        elapsed += row["dt"]
        move = clock.update(
            row["dt"],
            offset_m=row["loff"],
            heading_rad=row["h"],
            lane_width_m=row["lw"],
            off_road=row["offroad"],
            near_junction=row["nj"],
            n_lanes=2,
            road_offset_m=row["off"],
            road_width_m=row["rw"],
        )
        if move.phase in ("weave", "lane_change", "recover") and row["loff"] is not None:
            offsets.append(float(row["loff"]))
        if move.phase == "abort":
            saw_abort = True
            reason = move.reason
        if move.phase == "cruise":
            break
        if elapsed > 45.0:
            break
    swing = (max(offsets) - min(offsets)) if len(offsets) > 1 else 0.0
    return (not saw_abort and clock.phase == "cruise" and swing >= 0.5), swing, reason


def check_four_excursions_finish() -> None:
    finished = 0
    started = 0
    for level in ("west_coast_usa", "italy"):
        seq = _sequence(level)
        previous = "cruise"
        for index, row in enumerate(seq):
            if previous in ("weave", "lane_change", "recover", "abort"):
                previous = row["ph"]
                continue
            if row["ph"] not in ("weave", "lane_change"):
                previous = row["ph"]
                continue
            started += 1
            ok, swing, reason = _finish(seq, index, row["ph"])
            if ok:
                finished += 1
            print(
                f"collect7 {level} {row['ph']} swing={swing:.2f} "
                f"{'complete' if ok else 'abort ' + reason}"
            )
            previous = row["ph"]
    print(f"collect7 finished {finished} of {started}")
    assert started == 7, started
    assert finished >= 4, finished


def main() -> None:
    check_four_excursions_finish()
    print("collect7 replay ok")


if __name__ == "__main__":
    main()
