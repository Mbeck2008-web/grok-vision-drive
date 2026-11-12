#!/usr/bin/env python3
"""Legal-direction, right-hand-lane spawns.

Offline. No BeamNG. A pose on a flipped road faces opposite the stored node
order. A one-way pose sits in the rightmost lane. A two-way pose sits in the
right-hand lane (offset at least the centerline deadzone). The deck will not
return a pose the offline wrong-way classifier rejects.
"""
from __future__ import annotations

import io
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect.spawn_mix import SpawnDeck  # noqa: E402
from python.train.frame_quality import (  # noqa: E402
    QualityRoad,
    build_road_grid,
    classify_wrong_way,
)

FIXTURE = ROOT / "tests" / "fixtures" / "roads" / "west_coast_usa.json.gz"


def _heading(rot: tuple[float, float, float, float]) -> tuple[float, float]:
    _qx, _qy, qz, qw = rot
    yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
    return math.cos(yaw), math.sin(yaw)


def _road(
    rid: str,
    pts: list[tuple[float, float]],
    *,
    half: float,
    lanes_left: int,
    lanes_right: int,
    oneway: bool,
    flip: bool,
    z: float = 10.0,
) -> dict:
    """Straight-segment nav road. ``left`` is left of the stored node order."""
    middles = []
    left = []
    right = []
    for i, (x, y) in enumerate(pts):
        if i + 1 < len(pts):
            nx, ny = pts[i + 1][0] - x, pts[i + 1][1] - y
        else:
            nx, ny = x - pts[i - 1][0], y - pts[i - 1][1]
        norm = math.hypot(nx, ny) or 1.0
        nx, ny = nx / norm, ny / norm
        lx, ly = -ny, nx
        middles.append((float(x), float(y), z))
        left.append((float(x) + lx * half, float(y) + ly * half, z))
        right.append((float(x) - lx * half, float(y) - ly * half, z))
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(middles, middles[1:]))
    return {
        "id": rid,
        "pts": middles,
        "left": left,
        "right": right,
        "lanes": lanes_left + lanes_right,
        "lanes_left": lanes_left,
        "lanes_right": lanes_right,
        "length": length,
        "oneway": oneway,
        "flip": flip,
        "flipDirection": flip,
        "width": half * 2.0,
    }


def _quality(road: dict) -> QualityRoad:
    return QualityRoad(
        road_id=str(road["id"]),
        xy=[(p[0], p[1]) for p in road["pts"]],
        width_m=float(road["width"]),
        oneway=bool(road["oneway"]),
        lanes_left=int(road["lanes_left"]),
        lanes_right=int(road["lanes_right"]),
        lanes=int(road["lanes"]),
        drivability=1.0,
        flip=bool(road["flip"]),
        z=[p[2] for p in road["pts"]],
    )


def _classify(road: dict, pos, rot) -> dict:
    roads = [_quality(road)]
    fx, fy = _heading(rot)
    return classify_wrong_way(pos[0], pos[1], math.atan2(fy, fx), roads, build_road_grid(roads), z=pos[2])


def _deck(roads: list[dict], log=None) -> SpawnDeck:
    return SpawnDeck({"city": list(roads), "highway": [], "ramp": []}, log=log)


def check_flip_reverses_node_order() -> None:
    """flipDirection reverses legal travel. Node order is +X; the car faces -X."""
    road = _road("flip-ow", [(0.0, 0.0), (80.0, 0.0)], half=4.0, lanes_left=0, lanes_right=2, oneway=True, flip=True)
    pick = _deck([road]).take()
    assert pick is not None and pick.road_id == "flip-ow"
    fx, fy = _heading(pick.rot)
    assert fx < -0.5, (fx, fy, pick.pos)
    got = _classify(road, pick.pos, pick.rot)
    assert got["verdict"] == "ok", got
    assert got["reason"] == "oneway_with", got
    assert got["road_id"] == "flip-ow"
    print(f"flip heading fx={fx:.2f} offset={got['offset_m']:.2f}")


def check_oneway_rightmost_lane() -> None:
    """Three one-way lanes, width 12 m. Rightmost center is 4 m to the right of +X."""
    road = _road("ow", [(0.0, 0.0), (100.0, 0.0)], half=6.0, lanes_left=0, lanes_right=3, oneway=True, flip=False)
    pick = _deck([road]).take()
    assert pick is not None
    fx, fy = _heading(pick.rot)
    assert fx > 0.5, (fx, fy)
    got = _classify(road, pick.pos, pick.rot)
    assert got["verdict"] == "ok", got
    assert abs(got["offset_m"] - 4.0) < 0.35, got
    assert pick.pos[1] < -3.0, pick.pos
    print(f"oneway rightmost offset={got['offset_m']:.2f} y={pick.pos[1]:.2f}")


def check_twoway_right_of_center() -> None:
    road = _road("tw", [(0.0, 0.0), (100.0, 0.0)], half=4.0, lanes_left=1, lanes_right=1, oneway=False, flip=False)
    pick = _deck([road]).take()
    assert pick is not None
    fx, _fy = _heading(pick.rot)
    assert fx > 0.5
    got = _classify(road, pick.pos, pick.rot)
    assert got["verdict"] == "ok", got
    assert got["reason"] == "twoway_correct_side", got
    assert got["offset_m"] >= 0.75, got
    print(f"twoway offset={got['offset_m']:.2f}")


def check_rejected_pose_is_not_spawned() -> None:
    """A two-way road too narrow to clear the centerline is skipped, not spawned."""
    bad = _road("narrow", [(0.0, 0.0), (40.0, 0.0)], half=0.6, lanes_left=1, lanes_right=1, oneway=False, flip=False)
    good = _road("wide", [(0.0, 30.0), (80.0, 30.0)], half=4.0, lanes_left=1, lanes_right=1, oneway=False, flip=False)
    log: list[str] = []
    deck = _deck([bad, good], log=log.append)
    picks = []
    for _ in range(4):
        pick = deck.take()
        if pick is None:
            break
        picks.append(pick)
    ids = [p.road_id for p in picks]
    assert "narrow" not in ids, ids
    assert ids == ["wide"], ids
    reject = [line for line in log if "narrow" in line and "flipDirection" in line]
    assert reject, log
    assert any(token in reject[0] for token in ("twoway_centerline", "wrong_way", "skipped")), reject
    print(f"rejected narrow road; spawned {ids}")


def check_spawn_log_line() -> None:
    road = _road("logged", [(0.0, 0.0), (60.0, 0.0)], half=4.0, lanes_left=0, lanes_right=1, oneway=True, flip=True)
    log: list[str] = []
    pick = _deck([road], log=log.append).take()
    assert pick is not None
    lines = [line for line in log if "flipDirection" in line and "logged" in line and "offset_m" in line]
    assert len(lines) == 1, log
    line = lines[0]
    for token in ("align=", "verdict=", "kind=", "flipDirection=1"):
        assert token in line, line
    assert "verdict=ok" in line, line
    print(f"log {line}")


def check_traffic_uses_legal_lane() -> None:
    road = _road("tw", [(0.0, 0.0), (0.0, 200.0)], half=4.0, lanes_left=1, lanes_right=1, oneway=False, flip=False)
    deck = _deck([road])
    pick = deck.take()
    assert pick is not None
    slots, notes = deck.traffic_slots(pick, 2)
    assert [slot[0] for slot in slots] == ["ahead", "behind"], (slots, notes)
    ego = _classify(road, pick.pos, pick.rot)
    assert ego["verdict"] == "ok" and ego["offset_m"] >= 0.75, ego
    for where, pos, rot, _along in slots:
        got = _classify(road, pos, rot)
        assert got["verdict"] == "ok", (where, got, notes)
        assert got["reason"] != "twoway_centerline", (where, got)
        assert got["offset_m"] >= 0.75, (where, got)
        sfx, sfy = _heading(rot)
        efx, efy = _heading(pick.rot)
        assert sfx * efx + sfy * efy > 0.5, (where, sfx, sfy)
    print(f"traffic in the right lane offset={ego['offset_m']:.2f}")


def check_road_spawn_flip_and_right_lane() -> None:
    from python.collect.level_session import road_spawn

    class _Road:
        road_id = "flip-x"
        xy = [(0.0, 0.0), (100.0, 0.0)]
        width_m = 12.0
        lanes = 3
        flip = True
        oneway = True
        lanes_left = 0
        lanes_right = 3
        z = [0.0, 0.0]

    buf = io.StringIO()
    with redirect_stdout(buf):
        spawned = road_spawn([_Road()])
    assert spawned is not None
    pos, rot = spawned
    fx, fy = _heading(rot)
    assert fx < -0.5, (fx, fy, pos)
    assert pos[1] > 3.0, pos
    assert pos[2] > 20.0
    text = buf.getvalue()
    assert "flip-x" in text and "flipDirection=1" in text and "verdict=" in text, text
    print(f"road_spawn flip fx={fx:.2f} y={pos[1]:.2f}")


def check_snap_near_west_coast_carriageway() -> None:
    """A point on the centerline of one-way 892682.0 resolves into its right lane, facing north."""
    import gzip
    import json

    from python.collect import road_loader
    from python.collect.level_session import snap_to_road
    from python.train.frame_quality import load_quality_roads

    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        network = json.load(handle)["road_network"]
    spec = {rid: network[rid] for rid in ("892682.0", "892693.0")}

    class _Net:
        def __init__(self) -> None:
            self.scenario = self

        def get_road_network(self, **_kwargs: object) -> dict:
            return spec

    road_loader.reset_shared()
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            placed = snap_to_road(_Net(), (995.5, 186.9, 153.3))
    finally:
        road_loader.reset_shared()
    assert placed is not None
    pos, rot = placed
    roads = load_quality_roads(FIXTURE)
    fx, fy = _heading(rot)
    got = classify_wrong_way(pos[0], pos[1], math.atan2(fy, fx), roads, z=pos[2])
    assert got["road_id"] == "892682.0", got
    assert got["verdict"] == "ok", got
    assert got["offset_m"] > 2.0, got
    assert fy > 0.9 and abs(fx) < 0.15, (fx, fy)
    assert pos[0] > 999.0, pos
    text = buf.getvalue()
    assert "892682.0" in text and "flipDirection=" in text and "offset_m=" in text and "verdict=" in text, text
    print(f"snap 892682.0 ({pos[0]:.2f},{pos[1]:.2f}) offset={got['offset_m']:.2f} fy={fy:.3f}")


def check_fixture_deck_has_no_rejected_poses() -> None:
    from python.train.frame_quality import load_quality_roads, spawn_pose_rejected

    roads = load_quality_roads(FIXTURE)
    grid = build_road_grid(roads)
    deck = SpawnDeck.load(FIXTURE)
    bad = {"wrong_way": 0, "twoway_centerline": 0}
    n = 0
    while True:
        pick = deck.take()
        if pick is None:
            break
        n += 1
        fx, fy = _heading(pick.rot)
        got = classify_wrong_way(pick.pos[0], pick.pos[1], math.atan2(fy, fx), roads, grid, z=pick.pos[2])
        if got.get("verdict") == "wrong_way":
            bad["wrong_way"] += 1
        if got.get("reason") == "twoway_centerline":
            bad["twoway_centerline"] += 1
        assert not spawn_pose_rejected(got), (pick.road_id, got)
    assert n >= 800, n
    assert bad == {"wrong_way": 0, "twoway_centerline": 0}, bad
    print(f"fixture deck picks={n} wrong_way=0 twoway_centerline=0")


def check_spawn_pose_rejected_helper() -> None:
    from python.train.frame_quality import spawn_pose_rejected

    assert spawn_pose_rejected({"verdict": "wrong_way", "reason": "oneway_against"})
    assert spawn_pose_rejected({"verdict": "unknown", "reason": "twoway_centerline"})
    assert spawn_pose_rejected({"verdict": "wrong_way", "reason": "twoway_opposing_lane"})
    assert not spawn_pose_rejected({"verdict": "ok", "reason": "twoway_correct_side"})
    assert not spawn_pose_rejected({"verdict": "unknown", "reason": "off_network"})
    print("spawn_pose_rejected helper ok")


def main() -> None:
    check_flip_reverses_node_order()
    check_oneway_rightmost_lane()
    check_twoway_right_of_center()
    check_rejected_pose_is_not_spawned()
    check_spawn_log_line()
    check_traffic_uses_legal_lane()
    check_road_spawn_flip_and_right_lane()
    check_snap_near_west_coast_carriageway()
    check_fixture_deck_has_no_rejected_poses()
    check_spawn_pose_rejected_helper()
    print("legal spawn ok")


if __name__ == "__main__":
    main()
