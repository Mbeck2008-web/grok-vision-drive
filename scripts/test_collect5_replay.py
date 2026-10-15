#!/usr/bin/env python3
"""Replay the 531fa57 6-minute run, and the road-dump format.

The live graph is not in this tree yet. ``--dump-roads`` writes it to
``E:\\GVD_data\\roads\\<map>.json.gz``. Drop those files at
``tests/fixtures/roads/<map>.json.gz`` and this replay drives the collect4
and collect5 trajectories on them. Until then, a graph of short edges plus
a driveway checks the two live failures: a weave aborted as ``junction``
one tick after the gate opened, and a lane center latched at -6.74.
"""
from __future__ import annotations

import gzip
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect.excursion import (  # noqa: E402
    ACTIVE,
    CRUISE_S,
    ExcursionClock,
    gate_reason,
    near_junction,
)
from python.collect.road_dump import load_road_dump, write_road_dump  # noqa: E402
from python.debug.gt_geom import RoadFollow, RoadHit, RoadPoly, follow_label, heading_from_dir, project_follow  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "collect5" / "trace.jsonl.gz"
COLLECT4 = ROOT / "tests" / "fixtures" / "collect4" / "trace.jsonl.gz"
ROADS = ROOT / "tests" / "fixtures" / "roads"


def _load_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load() -> list[dict]:
    return _load_gz(FIXTURE)


def _pieces() -> list[RoadPoly]:
    roads = [
        RoadPoly(f"h{i}", [(0.0, float(i)), (0.0, float(i + 20))], 7.0)
        for i in range(0, 180, 20)
    ]
    roads.append(RoadPoly("driveway", [(0.0, 20.0), (12.0, 20.0)], 4.0))
    return roads


def check_raw_weaves_die_as_junction_and_the_label_latches() -> None:
    rows = _load()
    west = [row for row in rows if row["seg"] == "west_coast_usa#0"]
    weaves = [row for row in west if row["ph"] == "weave"]
    aborts = [row for row in west if row["ph"] == "abort"]
    assert weaves and aborts
    assert {row["why"] for row in aborts} == {"junction"}
    assert max(
        sum(1 for row in west[i : i + 5] if row["ph"] == "weave")
        for i in range(len(west))
    ) <= 2
    latched = [row["gt"] for row in west if row["ph"] in ("weave", "abort") and row["m"]]
    assert latched and max(latched) - min(latched) < 1e-6
    assert abs(latched[0] + 6.74) < 0.02
    italy = [row for row in rows if row["seg"] == "italy#1" and row["ph"] == "cruise" and row["m"]]
    values = [float(row["gt"]) for row in italy]
    assert min(values) < -2.5 and max(values) > 2.5
    print(f"collect5 west weaves abort junction, label latched at {latched[0]:.2f}")


def check_continuation_is_not_a_junction_during_the_move() -> None:
    roads = _pieces()
    hit = RoadHit(
        road_id="h0",
        offset_m=-1.75,
        heading_err_rad=0.0,
        lane_id=0,
        lane_offset_m=0.0,
        lane_width_m=3.5,
        off_road=False,
        dist_m=1.0,
        ahead_m=2.0,
        discontinuous=True,
    )
    # The edge is ending and the follower calls it discontinuous. The
    # highway continues, with a driveway at the joint. That is not a junction.
    assert near_junction(hit, roads, -1.75, 18.0, math.pi / 2.0) is False
    live = project_follow(roads, -1.75, 5.0, math.pi / 2.0, RoadFollow(), 8.0)
    assert gate_reason(roads, live, -1.75, 5.0, math.pi / 2.0, "weave", 8.0) == "open"
    clock = ExcursionClock()
    clock.phase = "weave"
    clock.phase_t = 0.1
    for _ in range(10):
        move = clock.update(
            0.1,
            offset_m=0.1,
            heading_rad=0.0,
            lane_width_m=3.5,
            off_road=False,
            near_junction=near_junction(hit, roads, -1.75, 18.0, math.pi / 2.0),
            n_lanes=2,
        )
        assert move.phase == "weave", move
    print("weave survives a piece end that still has a continuation")


def check_label_follows_the_car() -> None:
    road = RoadPoly("r", [(0.0, 0.0), (0.0, 200.0)], 7.0)
    follow = RoadFollow()
    first, _, _ = follow_label([road], 0.2, 30.0, math.pi / 2.0, follow, 8.0, None)
    second, _, refused = follow_label([road], 3.0, 31.0, math.pi / 2.0, follow, 8.0, first)
    assert first is not None and second is not None
    assert refused is False
    assert abs(second.lane_offset_m - first.lane_offset_m) > 1.0
    assert second is not first
    print(f"label moved {first.lane_offset_m:.2f} -> {second.lane_offset_m:.2f}")


def check_dump_roundtrip() -> None:
    spec = {
        "roads": {
            "main": [
                {"middle": [0, 0, 1], "left": [-3, 0, 1], "right": [3, 0, 1]},
                {"middle": [0, 40, 1], "left": [-3, 40, 1], "right": [3, 40, 1]},
            ]
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = write_road_dump(
            Path(tmp) / "west_coast_usa.json.gz",
            level="west_coast_usa",
            spawn=(-717.121, 101.0, 118.675),
            spawn_source="known",
            road_network=spec,
        )
        polys, payload = load_road_dump(path)
    assert payload["map"] == "west_coast_usa"
    assert abs(payload["spawn"]["pos"][2] - 118.675) < 1e-6
    assert len(polys) == 1 and abs(polys[0].width_m - 6.0) < 1e-6
    print(f"dump roundtrip edges={len(polys)} width={polys[0].width_m:.1f}")


def _replay(rows: list[dict], roads: list[RoadPoly]) -> list[str]:
    clock = ExcursionClock()
    follow = RoadFollow()
    started: list[str] = []
    previous = "cruise"
    alive = 0
    for row in rows:
        dt = row.get("dt")
        dt = 0.1 if dt is None or dt <= 0.0 or dt > 2.0 else float(dt)
        heading = heading_from_dir(row["dx"], row["dy"])
        hit = project_follow(roads, row["x"], row["y"], heading, follow, float(row.get("spd") or 0.0))
        kind = clock.peek_kind(2)
        chosen = None
        window = True
        if clock.phase == "cruise" and clock.phase_t + dt >= CRUISE_S:
            gate = gate_reason(roads, hit, row["x"], row["y"], heading, "weave", float(row.get("spd") or 0.0))
            chosen = "weave"
            window = gate == "open"
        move = clock.update(
            dt,
            offset_m=0.0 if hit is None else hit.lane_offset_m,
            heading_rad=0.0 if hit is None else hit.heading_err_rad,
            lane_width_m=3.5 if hit is None else hit.lane_width_m,
            off_road=bool(hit.off_road) if hit is not None else False,
            near_junction=near_junction(
                hit, roads, row["x"], row["y"], heading,
            ),
            n_lanes=2,
            window_ok=window,
            start_kind=chosen,
        )
        if previous == "cruise" and move.phase == "weave":
            alive = 0
            started.append("weave")
        if move.phase == "weave":
            alive += 1
        if move.phase == "abort" and move.reason == "junction" and alive < 5:
            started.append(f"junction-abort-after-{alive}")
            alive = 0
        previous = move.phase
    return started


def check_live_graph_when_it_is_here() -> None:
    """Drive collect4 and collect5 on dumped Tech graphs, once those files are in the tree."""
    missing: list[str] = []
    for level in ("west_coast_usa", "italy"):
        dump = ROADS / f"{level}.json.gz"
        if not dump.is_file():
            missing.append(level)
            continue
        roads, payload = load_road_dump(dump)
        assert payload.get("map") == level
        assert len(roads) > 10, (level, len(roads))
        for label, path in (("collect5", FIXTURE), ("collect4", COLLECT4)):
            rows = [row for row in _load_gz(path) if str(row["seg"]).startswith(level)]
            started = _replay(rows[::2], roads)
            print(f"live graph {label} {level} starts={started[:12]}")
            assert started, (label, level)
            assert not any(item.startswith("junction-abort-after-") for item in started), (
                label,
                level,
                started,
            )
    if missing:
        print(
            "no tests/fixtures/roads/"
            + ",".join(f"{name}.json.gz" for name in missing)
            + " yet; synthetic graph only"
        )


def main() -> None:
    check_raw_weaves_die_as_junction_and_the_label_latches()
    check_continuation_is_not_a_junction_during_the_move()
    check_label_follows_the_car()
    check_dump_roundtrip()
    check_live_graph_when_it_is_here()
    print("collect5 replay ok")


if __name__ == "__main__":
    main()
