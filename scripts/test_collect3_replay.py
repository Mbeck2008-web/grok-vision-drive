#!/usr/bin/env python3
"""Replay the 6-minute west_coast + italy collect from 1d664fa.

The raw log never left cruise: every row's excursion reason is ``junction``,
because a road end within a few metres of the lane was treated as a fork and
the cruise timer was wiped every tick. Of the 25 cruise jumps bigger than
2 m, 21 are a lane-index flip (the lane offset swaps sign by about one lane)
and 4 include a polyline heading flip. The car moved under a metre on those
ticks.

Replaying the same positions through the held label and the 60 m window
produces no jump that large, and several in-lane weaves.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from python.collect.excursion import ACTIVE, CRUISE_S, ExcursionClock, gate_reason  # noqa: E402
from python.debug.gt_geom import (  # noqa: E402
    RoadFollow,
    RoadHit,
    RoadPoly,
    heading_from_dir,
    hold_lane_label,
    label_center_x,
    project_follow,
)

FIXTURE = ROOT / "tests" / "fixtures" / "collect3" / "trace.jsonl.gz"


def _load() -> list[dict]:
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _hit(row: dict) -> RoadHit:
    off = not row["m"] or row["off"] or row["gt"] is None or row["h"] is None
    lane_off = 0.0
    if not off:
        lane_off = 5.0 * math.sin(float(row["h"])) - float(row["gt"])
    return RoadHit(
        road_id=str(row["seg"]),
        offset_m=0.0,
        heading_err_rad=None if row["h"] is None else float(row["h"]),
        lane_id=0,
        lane_offset_m=lane_off,
        lane_width_m=3.5,
        off_road=bool(off),
        dist_m=0.0,
    )


def _road(rows: list[dict]) -> RoadPoly:
    pts = [(rows[0]["x"], rows[0]["y"])]
    acc = 0.0
    for prev, row in zip(rows, rows[1:]):
        acc += math.hypot(row["x"] - prev["x"], row["y"] - prev["y"])
        if acc >= 2.0:
            pts.append((row["x"], row["y"]))
            acc = 0.0
    return RoadPoly(str(rows[0]["seg"]), pts, 7.0)


def check_raw_log_is_the_stuck_cruise() -> None:
    rows = _load()
    assert len(rows) > 3000
    assert {row["why"] for row in rows} == {"junction"}
    jumps = 0
    for prev, row in zip(rows, rows[1:]):
        if prev["seg"] != row["seg"] or not prev["m"] or not row["m"]:
            continue
        if prev["gt"] is None or row["gt"] is None:
            continue
        if abs(float(row["gt"]) - float(prev["gt"])) > 2.0:
            jumps += 1
    assert jumps == 25, jumps
    print(f"raw collect3 rows={len(rows)} jumps={jumps} reason=junction")


def check_held_labels_do_not_jump() -> None:
    rows = _load()
    previous: RoadHit | None = None
    jumps = 0
    for prev, row in zip(rows, rows[1:]):
        if prev["seg"] != row["seg"]:
            previous = None
            continue
        proposed = _hit(row)
        if proposed.off_road:
            continue
        chosen = hold_lane_label(previous, proposed)
        if previous is not None and chosen is not None:
            old = label_center_x(previous)
            new = label_center_x(chosen)
            if old is not None and new is not None and abs(new - old) > 2.0:
                jumps += 1
        previous = chosen if chosen is not None and not chosen.off_road else previous
    assert jumps == 0, jumps
    print("held labels have no >2 m cruise jump")


def _starts(rows: list[dict]) -> list[str]:
    road = _road(rows)
    roads = [road]
    clock = ExcursionClock()
    follow = RoadFollow()
    started: list[str] = []
    previous = "cruise"
    for prev, row in zip(rows, rows[1:]):
        dt = row["dt"]
        dt = 0.1 if dt is None or dt <= 0.0 or dt > 2.0 else float(dt)
        heading = heading_from_dir(row["dx"], row["dy"])
        hit = project_follow(roads, row["x"], row["y"], heading, follow, float(row["spd"] or 0.0))
        kind = clock.peek_kind(2)
        chosen = None
        window = True
        if clock.phase == "cruise" and clock.phase_t + dt >= CRUISE_S:
            gate = gate_reason(roads, hit, row["x"], row["y"], heading, kind, float(row["spd"] or 0.0))
            if gate != "open" and kind != "weave":
                gate = gate_reason(roads, hit, row["x"], row["y"], heading, "weave", float(row["spd"] or 0.0))
                kind = "weave"
            chosen = kind
            window = gate == "open"
        move = clock.update(
            dt,
            offset_m=0.0,
            heading_rad=0.0,
            lane_width_m=3.5,
            off_road=False,
            near_junction=False,
            n_lanes=2,
            window_ok=window,
            start_kind=chosen,
        )
        if previous == "cruise" and move.phase in ACTIVE:
            started.append(move.phase)
        previous = move.phase
        del prev
    return started


def check_replay_starts_excursions() -> None:
    rows = _load()
    started: list[str] = []
    for seg in ("west_coast_usa#0", "italy#1"):
        part = _starts([row for row in rows if row["seg"] == seg])
        print(f"{seg} excursions={len(part)} {part}")
        started.extend(part)
    assert len(started) >= 4, started
    assert "weave" in started


def main() -> None:
    check_raw_log_is_the_stuck_cruise()
    check_held_labels_do_not_jump()
    check_replay_starts_excursions()
    print("collect3 replay ok")


if __name__ == "__main__":
    main()
