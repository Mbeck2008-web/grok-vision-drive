#!/usr/bin/env python3
"""The excursion gate judges the line the start sends, and road heights are per point.

Pure Python: no BeamNG, no torch.

* Direction: on a bend with the car near the right edge, a one-sided weave
  (20 m/s: one half wave, out to one side and back) fits only to the left.
  The gate is given the clock's own next move (``ExcursionClock.start_move``)
  through the same ``sent_line_args`` the start uses, so it opens for the
  safe side, refuses the other, and agrees with the start's path build each
  time. The old gate (direction +1 always) disagreed with the start.
* Inputs: the collector builds the gate's and the start's path inputs with
  one helper (``collect_scene_lanes._line_args``). For a weave and a lane
  change the two are equal field for field, the painted lane included.
* Heights: ``parse_road_edges`` keeps a height per point, so one record
  without a height costs that point, not the road. ``road_heights_line`` logs
  once per map how many roads lack heights; on the west_coast_usa dump none do.
"""
from __future__ import annotations

import math
import re
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.collect_scene_lanes as collect  # noqa: E402
from python.collect import excursion as exc  # noqa: E402
from python.collect.excursion import (  # noqa: E402
    CRUISE_S,
    Excursion,
    ExcursionClock,
    build_sent_line,
    gate_reason,
    sent_line_args,
)
from python.collect.road_dump import load_road_dump  # noqa: E402
from python.debug.gt_geom import (  # noqa: E402
    RoadHit,
    RoadPoly,
    height_coverage,
    join_candidates,
    parse_road_edges,
    road_ahead_split,
    road_heights_line,
)

ROADS_PATH = ROOT / "tests" / "fixtures" / "roads" / "west_coast_usa.json.gz"
HALF_W = 0.9
PAINT = (-1.75, 1.75)  # painted lane at the car, vehicle frame: 3.5 m wide
V = 20.0  # m/s: wave 100 m, one half wave (50 m), so the weave goes to one side only


def _bend(sign: float, radius: float = 400.0, length: float = 200.0, width: float = 7.0) -> RoadPoly:
    """A gentle bend (0.0025 rad/m, under CURVATURE_MAX): left for sign +1, right for -1."""
    pts = []
    for k in range(int(length / 2.0) + 1):
        a = 2.0 * k / radius
        pts.append((sign * (-radius + radius * math.cos(a)), radius * math.sin(a)))
    return RoadPoly("bend", pts, width, lanes=2)


def _car(road: RoadPoly, offset_m: float, s_m: float = 20.0) -> tuple[float, float, float, RoadHit]:
    i = int(s_m / 2.0)
    (ax, ay), (bx, by) = road.xy[i], road.xy[i + 1]
    fx, fy = (bx - ax) / math.hypot(bx - ax, by - ay), (by - ay) / math.hypot(bx - ax, by - ay)
    x, y = ax + fy * offset_m, ay - fx * offset_m  # right normal (fy, -fx)
    lane_id = 1 if offset_m > 0 else 0
    hit = RoadHit(road_id=road.road_id, offset_m=offset_m, heading_err_rad=0.0, lane_id=lane_id,
                  lane_offset_m=offset_m - (1.75 if offset_m > 0 else -1.75), lane_width_m=3.5,
                  off_road=False, dist_m=abs(offset_m))
    return x, y, math.atan2(fy, fx), hit


def _gate(roads, x, y, h, hit, move):
    """The collector's gate call: the clock's next move through the shared helper."""
    line = collect._line_args((x, y, 0.0), h, move, hit, V, PAINT, HALF_W)
    return gate_reason(roads, hit, x, y, h, move.phase, V, weave_lane_w=PAINT[1] - PAINT[0],
                       half_width_m=HALF_W, paint_lane=PAINT, line=line)


def _start(roads, x, y, h, hit, move):
    """The collector's start call: the clock's move through the same helper."""
    info: dict = {}
    pts, used = build_sent_line(roads, collect._line_args((x, y, 0.0), h, move, hit, V, PAINT, HALF_W), info)
    return pts, used, info


def _clock(direction: int, kind_count: int = 0) -> ExcursionClock:
    clock = ExcursionClock()
    clock.direction = direction
    clock._kind = kind_count
    return clock


def _run_start(clock: ExcursionClock, kind: str, n_lanes: int) -> Excursion:
    """Drive the clock through its start, as the collector does after an open gate."""
    move = clock.update(CRUISE_S + 0.1, offset_m=0.0, heading_rad=0.0, lane_width_m=3.5, n_lanes=n_lanes,
                        window_ok=True, start_kind=kind, road_offset_m=0.0, road_width_m=7.0, speed_mps=V,
                        half_width_m=HALF_W, paint_lane=PAINT)
    assert move.phase == kind, (move, kind)
    return move


def check_direction() -> list[str]:
    out = []
    road = _bend(+1.0)
    roads = [road]
    x, y, h, hit = _car(road, 3.0)  # 3.0 m right of center: right edge limit 3.3 m, amplitude 0.42 m
    verdict = {}
    for direction in (1, -1):
        clock = _clock(direction)
        move = clock.start_move(clock.peek_kind(2), 2)
        assert move.phase == "weave" and move.direction == direction, move
        gate = _gate(roads, x, y, h, hit, move)
        started = _run_start(clock, "weave", 2)
        assert started.direction == direction, started
        pts, used, info = _start(roads, x, y, h, hit, started)
        sent = pts is not None
        assert (gate == "open") == sent, (direction, gate, info)
        if not sent:
            assert gate == info.get("refused"), (gate, info)
        verdict[direction] = (gate, "sent" if sent else f"refused {info.get('refused')}")
    assert verdict[-1] == ("open", "sent"), verdict
    assert verdict[1] == ("road_edge", "refused road_edge"), verdict
    out.append(f"direction, left bend, car 3.0 m right: clock dir -1 gate open, start sent; "
               f"dir +1 gate road_edge, start refused road_edge")
    # The old gate built direction +1 whatever the clock held. Mirrored road
    # (car 3.0 m left), clock direction -1: +1 fits, so the old gate opened,
    # and the start, going -1, was refused and aborted no_path.
    mirror = _bend(-1.0)
    mx, my, mh, mhit = _car(mirror, -3.0)
    clock = _clock(-1)
    old = gate_reason([mirror], mhit, mx, my, mh, "weave", V, weave_lane_w=PAINT[1] - PAINT[0],
                      half_width_m=HALF_W, paint_lane=PAINT)
    new = _gate([mirror], mx, my, mh, mhit, clock.start_move("weave", 2))
    pts, _used, info = _start([mirror], mx, my, mh, mhit, _run_start(clock, "weave", 2))
    assert old == "open" and pts is None and info.get("refused") == "road_edge", (old, info)
    assert new == "road_edge", new
    out.append("direction, mirrored: old gate (+1) open but the start (-1) refused road_edge; "
               "new gate road_edge, as the start")
    return out


def check_same_args() -> list[str]:
    road = _bend(+1.0)
    x, y, h, hit = _car(road, 1.75)
    out = []
    for kind, kind_count, direction in (("weave", 0, -1), ("lane_change", 3, 1)):
        clock = _clock(direction, kind_count)
        assert clock.peek_kind(2) == kind
        planned = clock.start_move(clock.peek_kind(2), 2)
        gate_args = collect._line_args((x, y, 0.0), h, planned, hit, V, PAINT, HALF_W)
        started = _run_start(clock, kind, 2)
        start_args = collect._line_args((x, y, 0.0), h, started, hit, V, PAINT, HALF_W)
        assert gate_args == start_args, (gate_args, start_args)
        assert gate_args.paint_lane == PAINT and gate_args.half_width_m == HALF_W, gate_args
        assert gate_args.phase == kind and gate_args.direction == started.direction, gate_args
        if kind == "lane_change":
            # The start flips the last side; the gate sees the flipped side.
            assert started.direction == -direction == gate_args.direction, started
        out.append(f"args, {kind}: gate == start (dir {gate_args.direction:+d}, paint_lane {gate_args.paint_lane}, "
                   f"v {gate_args.speed_mps}, half_w {gate_args.half_width_m})")
    # Lane change asked for on a one-lane road: start_move is the weave the clock really starts.
    clock = _clock(1, 3)
    assert clock.start_move("lane_change", 1) == Excursion("weave", 0.0, 0.0, 1, "")
    # A gate for one phase cannot be handed the other's line.
    try:
        gate_reason([road], hit, x, y, h, "weave", V, line=replace(gate_args, phase="lane_change"))
    except ValueError:
        pass
    else:
        raise AssertionError("gate_reason accepted a lane_change line for a weave")
    # Both call sites in the collector go through _line_args; nothing calls
    # excursion_path with its own arguments.
    src = (ROOT / "scripts" / "collect_scene_lanes.py").read_text(encoding="utf-8")
    assert not re.search(r"\bexcursion_path\(", src), "collector calls excursion_path directly"
    assert len(re.findall(r"line=_line_args\(", src)) == 2, "both gate calls pass line=_line_args(...)"
    assert len(re.findall(r"build_sent_line\(\s*roads, _line_args\(", src)) == 1, "the start builds with _line_args"
    out.append("args: the collector's two gate calls and the start all use _line_args")
    return out


def _edge(x: float, y: float, z: float | None, with_edges_z: bool = True) -> dict:
    mid = [x, y] if z is None else [x, y, z]
    if not with_edges_z:
        return {"middle": mid, "left": [x - 1.75, y], "right": [x + 1.75, y]}
    ez = z if z is not None else None
    left = [x - 1.75, y] if ez is None else [x - 1.75, y, ez]
    right = [x + 1.75, y] if ez is None else [x + 1.75, y, ez]
    return {"middle": mid, "left": left, "right": right}


def check_heights() -> list[str]:
    out = []
    full = parse_road_edges("a", [_edge(0.0, 4.0 * k, 10.0 + k) for k in range(5)])
    assert full is not None and full.z == [10.0, 11.0, 12.0, 13.0, 14.0], full
    # One record without a height: only that point loses it.
    edges = [_edge(0.0, 4.0 * k, 10.0 + k) for k in range(5)]
    edges[2] = _edge(0.0, 8.0, None, with_edges_z=False)
    part = parse_road_edges("b", edges)
    assert part is not None and part.z == [10.0, 11.0, None, 13.0, 14.0], part.z
    # A middle without a height takes the edges' mean height.
    mid_only = [{"middle": [0.0, 4.0 * k], "left": [-1.75, 4.0 * k, 5.0], "right": [1.75, 4.0 * k, 7.0]}
                for k in range(3)]
    got = parse_road_edges("c", mid_only)
    assert got is not None and got.z == [6.0, 6.0, 6.0], got.z
    none = parse_road_edges("d", [_edge(0.0, 4.0 * k, None, with_edges_z=False) for k in range(3)])
    assert none is not None and none.z is None
    out.append("heights: one record without z costs one point ([10, 11, None, 13, 14]); "
               "middle without z uses the edge mean; no z at all -> None")
    cov = height_coverage([full, part, got, none])
    assert (cov.roads, cov.roads_without, cov.roads_partial, cov.points, cov.points_without) == (4, 1, 1, 16, 4), cov
    logged: set[str] = set()
    line = road_heights_line("testmap", [full, part, got, none], logged)
    assert line == ("collect road heights testmap roads=4 without=1 (25.0%) partial=1 (25.0%) "
                    "points_without=4/16 (25.0%)"), line
    assert road_heights_line("testmap", [full], logged) is None, "logged twice for one map"
    assert road_heights_line("othermap", [full], logged) is not None
    out.append(f"heights log once per map: {line}")
    # The collector prints it through _log_road_heights, once per map.
    seen: set[str] = set()
    printed: list[str] = []
    saved = collect.print if hasattr(collect, "print") else None
    collect.print = lambda *a, **k: printed.append(" ".join(str(v) for v in a))  # type: ignore[attr-defined]
    try:
        collect._log_road_heights("testmap", [full, part], seen)
        collect._log_road_heights("testmap", [full, part], seen)
    finally:
        if saved is None:
            del collect.print  # type: ignore[attr-defined]
        else:
            collect.print = saved  # type: ignore[attr-defined]
    assert printed == ["collect road heights testmap roads=2 without=0 (0.0%) partial=1 (50.0%) "
                       "points_without=1/10 (10.0%)"], printed
    # Real dump: every west_coast_usa road carries a height at every point.
    roads, _payload = load_road_dump(ROADS_PATH)
    wc = height_coverage(roads)
    assert wc.roads > 1000 and wc.roads_without == 0 and wc.roads_partial == 0 and wc.points_without == 0, wc
    out.append(f"heights, west_coast_usa dump: {road_heights_line('west_coast_usa', roads, set())}")
    # Per point, not all-or-nothing: a road with a far point missing its
    # height still refuses the stacked parking deck (890809 start, 890846
    # one deck up). All-or-nothing dropped the whole road's z and took the deck.
    byid = {r.road_id: r for r in roads}
    road = byid["890809.0"]
    (ax, ay), (bx, by) = road.xy[1], road.xy[0]
    th = math.atan2(by - ay, bx - ax)
    holed = replace(road, z=list(road.z[:-1]) + [None])
    swapped = [holed if r.road_id == road.road_id else r for r in roads]
    _t, cands = join_candidates(swapped, holed, bx, by, th)
    upper = next(c for c in cands if c.road.road_id == "890846.0" and c.refused != "not_ahead")
    assert upper.refused == "join_dz", upper
    cont, _split = road_ahead_split(swapped, holed, bx, by, th)
    assert cont is not None and cont.road_id == "890798.0", cont
    out.append(f"heights, 890809 with its last point's z missing: 890846 still refused join_dz "
               f"(dz {upper.dz_m:+.2f} m), walk takes 890798")
    return out


def main() -> None:
    lines = check_direction() + check_same_args() + check_heights()
    for line in lines:
        print(line)
    print("collect gate args ok")


if __name__ == "__main__":
    main()
