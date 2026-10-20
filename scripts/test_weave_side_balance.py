#!/usr/bin/env python3
"""Weave sides: each weave attempt aims at the side with fewer completed weaves.

Pure Python: no BeamNG, no torch.

On 1addc938 / 6b59eef every weave started right: ``ExcursionClock.direction``
starts at +1, only a lane change flips it, and the clock is rebuilt at every
startup, recover, segment and respawn (rec5b 46 R / 0 L, rec5c 109 R / 0 L).

* (a) balance: a fresh clock and simulated attempts end with about equal
  completed weaves per side, also when left attempts are refused more often.
* (b) rebuild: a new ``ExcursionClock`` keeps the per-side counts and the
  tie alternation (they live on ``WeaveStats``, the collector's object).
* (c) gate: ``gate_reason`` with -1 checks the left side. A road edge on the
  left only refuses -1 (road_edge) and opens +1, both through the library
  ``direction`` argument and through the collector's ``start_move`` line.
* (d) sign: a -1 weave has a negative ``target_offset_m`` and its
  ``ai.setLine`` points go left of the car (vehicle frame, x right).
* (e) no fallback: a refused -1 attempt does not send, start or count a +1
  weave in the same attempt.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.collect_scene_lanes as collect  # noqa: E402
from python.collect.excursion import (  # noqa: E402
    CRUISE_S,
    Excursion,
    ExcursionClock,
    WeaveStats,
    build_sent_line,
    excursion_commands,
    gate_reason,
    weave_tag,
)
from python.debug.gt_geom import RoadHit, RoadPoly  # noqa: E402

HALF_W = 0.9
PAINT = (-1.75, 1.75)  # painted lane at the car, vehicle frame: 3.5 m wide
V = 20.0  # m/s: wave 100 m, one half wave sent, so a weave goes to one side only


def _straight(width: float = 7.0, length: float = 200.0) -> RoadPoly:
    return RoadPoly("straight", [(0.0, 2.0 * k) for k in range(int(length / 2.0) + 1)], width, lanes=2)


def _car(offset_m: float, s_m: float = 20.0) -> tuple[float, float, float, RoadHit]:
    """Car ``offset_m`` right of the centre line (heading +y; right is +x)."""
    hit = RoadHit(road_id="straight", offset_m=offset_m, heading_err_rad=0.0, lane_id=1 if offset_m > 0 else 0,
                  lane_offset_m=offset_m - (1.75 if offset_m > 0 else -1.75), lane_width_m=3.5,
                  off_road=False, dist_m=abs(offset_m))
    return offset_m, s_m, math.pi / 2.0, hit


def _gate(roads, x, y, h, hit, move):
    line = collect._line_args((x, y, 0.0), h, move, hit, V, PAINT, HALF_W)
    return gate_reason(roads, hit, x, y, h, move.phase, V, weave_lane_w=PAINT[1] - PAINT[0],
                       half_width_m=HALF_W, paint_lane=PAINT, line=line)


def _run_start(clock: ExcursionClock) -> Excursion:
    move = clock.update(CRUISE_S + 0.1, offset_m=0.0, heading_rad=0.0, lane_width_m=3.5, n_lanes=1,
                        window_ok=True, start_kind="weave", road_offset_m=0.0, road_width_m=7.0, speed_mps=V,
                        half_width_m=HALF_W, paint_lane=PAINT)
    assert move.phase == "weave", move
    return move


def _sides(stats: WeaveStats) -> dict:
    return stats.summary()["sides"]  # type: ignore[return-value]


def _simulate(stats: WeaveStats, attempts: int, left_refused_every: int, *, rebuild_every: int = 0) -> None:
    """The collector's attempt loop, without roads: gate, start, end (as collect_scene_lanes does)."""
    clock = ExcursionClock(stats.weave_side)
    left_tries = 0
    for k in range(attempts):
        if rebuild_every and k % rebuild_every == 0:
            clock = ExcursionClock(stats.weave_side)  # respawn / recover / segment
        move = clock.start_move("weave", 1)
        if move.direction < 0:
            left_tries += 1
            if left_refused_every and left_tries % left_refused_every != 0:
                stats.refuse("road_edge", V, direction=move.direction)
                continue
        started = _run_start(clock)
        assert started.direction == move.direction, (started, move)
        stats.start("weave", V, direction=started.direction)
        # Every third weave is aborted (lane_edge), the rest complete.
        stats.weave_end(started.direction, "lane_edge" if k % 3 == 2 else "line_end")
        clock = ExcursionClock(stats.weave_side)


def check_balance() -> list[str]:
    out = []
    for refused_every in (0, 2, 4):  # left refused: never, 1 of 2, 3 of 4 attempts
        stats = WeaveStats()
        _simulate(stats, 400, refused_every)
        s = _sides(stats)
        left, right = s["left"]["completed"], s["right"]["completed"]
        assert left > 0 and right > 0, s
        assert abs(left - right) <= 1, s
        out.append(f"balance, left refused {'never' if not refused_every else f'{refused_every - 1} of {refused_every}'}: "
                   f"completed L {left} / R {right}, started L {s['left']['started']} / R {s['right']['started']}, "
                   f"refused L {s['left']['refused']}")
    return out


def check_rebuild() -> list[str]:
    stats = WeaveStats()
    clock = ExcursionClock(stats.weave_side)
    first = clock.start_move("weave", 1).direction
    assert first == 1, first
    stats.start("weave", V, direction=1)
    stats.weave_end(1, "line_end")
    clock = ExcursionClock(stats.weave_side)  # rebuilt: right is ahead, left is next
    assert clock.start_move("weave", 1).direction == -1
    stats.start("weave", V, direction=-1)
    stats.weave_end(-1, "line_end")
    clock = ExcursionClock(stats.weave_side)  # tie 1-1, last tried left: right next
    assert clock.start_move("weave", 1).direction == 1
    stats.refuse("curve", V, direction=1)
    clock = ExcursionClock(stats.weave_side)  # still a tie, last tried right: left next
    assert clock.start_move("weave", 1).direction == -1
    # Mixed rebuilds in the long run keep the balance.
    stats = WeaveStats()
    _simulate(stats, 300, 3, rebuild_every=2)
    s = _sides(stats)
    assert abs(s["left"]["completed"] - s["right"]["completed"]) <= 1, s
    # The collector builds every clock (startup, recover, segment, respawn) on the one WeaveStats.
    src = (ROOT / "scripts" / "collect_scene_lanes.py").read_text(encoding="utf-8")
    assert "ExcursionClock()" not in src, "a clock built without the side balance"
    assert src.count("ExcursionClock(weave_stats.weave_side)") == 4, src.count("ExcursionClock(weave_stats.weave_side)")
    assert src.count("weave_stats = WeaveStats()") == 1
    return ["rebuild: counts and tie alternation survive new clocks (R, L, R refused, then L); "
            "the collector's 4 clocks all take weave_stats.weave_side; "
            f"300 attempts with rebuilds: completed L {s['left']['completed']} / R {s['right']['completed']}"]


def check_gate_left() -> list[str]:
    roads = [_straight()]
    x, y, h, hit = _car(-2.9)  # 2.9 m left of centre: left edge limit -3.3 m, amplitude ~0.55 m
    lib = {d: gate_reason(roads, hit, x, y, h, "weave", V, weave_lane_w=PAINT[1] - PAINT[0], half_width_m=HALF_W,
                          paint_lane=PAINT, direction=d) for d in (1, -1)}
    assert lib == {1: "open", -1: "road_edge"}, lib
    # The mirror: an edge on the right only refuses +1.
    xr, yr, hr, hitr = _car(2.9)
    mir = {d: gate_reason(roads, hitr, xr, yr, hr, "weave", V, weave_lane_w=PAINT[1] - PAINT[0], half_width_m=HALF_W,
                          paint_lane=PAINT, direction=d) for d in (1, -1)}
    assert mir == {1: "road_edge", -1: "open"}, mir
    # The collector's gate: the clock's chosen side through _line_args.
    stats = WeaveStats()
    stats.start("weave", V, direction=1)
    stats.weave_end(1, "line_end")
    clock = ExcursionClock(stats.weave_side)
    move = clock.start_move("weave", 1)
    assert move.direction == -1, move
    assert _gate(roads, x, y, h, hit, move) == "road_edge"
    assert _gate(roads, x, y, h, hit, Excursion("weave", 0.0, 0.0, 1, "")) == "open"
    return [f"gate, edge on the left: -1 {lib[-1]}, +1 {lib[1]}; edge on the right: -1 {mir[-1]}, +1 {mir[1]}; "
            "collector gate with the chosen -1: road_edge"]


def check_left_sign() -> list[str]:
    roads = [_straight()]
    x, y, h, hit = _car(0.0)
    stats = WeaveStats()
    stats.start("weave", V, direction=1)
    stats.weave_end(1, "line_end")
    clock = ExcursionClock(stats.weave_side)
    move = _run_start(clock)
    assert move.direction == -1, move
    assert move.target_offset_m < 0.0 and move.target_heading_rad > 0.0, move
    info: dict = {}
    pts, used = build_sent_line(roads, collect._line_args((x, y, 0.0), h, move, hit, V, PAINT, HALF_W), info)
    assert pts is not None and used == -1, (used, info)
    off, head = weave_tag(used, info["amp_m"], info["wave_m"])
    assert off < 0.0 and head > 0.0, (off, head)
    # The steer is the sent line: ai.setLine points, vehicle frame x = (p - car) . right, right = (1, 0) here.
    cmds = excursion_commands(V, "weave", used, path=pts)
    line = next(c for c in cmds if c.startswith("ai.setLine"))
    xs = [float(a) - x for a in re.findall(r"pos=\{(-?[\d.]+),", line)]
    assert len(xs) == len(pts), (len(xs), len(pts))
    # Around the quarter wave (the first peak) the line is well left of the car.
    quarter = [vx for (_px, py), vx in zip(pts, xs) if 0.2 * info["wave_m"] <= py - y <= 0.3 * info["wave_m"]]
    assert quarter and max(quarter) < -0.3 and max(xs) <= 0.05, (quarter, max(xs))
    return [f"sign, -1: target_offset_m {off:+.2f} m, heading {head:+.3f} rad (left), "
            f"setLine points left of the car to {min(xs):+.2f} m (vehicle frame x)"]


def check_no_fallback() -> list[str]:
    roads = [_straight()]
    x, y, h, hit = _car(-2.9)  # the left side is refused here, the right side is not
    stats = WeaveStats()
    stats.start("weave", V, direction=1)
    stats.weave_end(1, "line_end")
    clock = ExcursionClock(stats.weave_side)
    move = clock.start_move("weave", 1)
    assert move.direction == -1, move
    gate = _gate(roads, x, y, h, hit, move)
    assert gate == "road_edge", gate
    stats.refuse(gate, V, direction=move.direction)  # the collector's refused attempt
    after = clock.update(CRUISE_S + 0.1, offset_m=0.0, heading_rad=0.0, lane_width_m=3.5, n_lanes=1,
                         window_ok=False, start_kind="weave", road_offset_m=-2.9, road_width_m=7.0, speed_mps=V,
                         half_width_m=HALF_W, paint_lane=PAINT)
    assert after.phase == "cruise", after
    # The start's own builder does not try the other side either.
    info: dict = {}
    pts, used = build_sent_line(roads, collect._line_args((x, y, 0.0), h, move, hit, V, PAINT, HALF_W), info)
    assert pts is None and used == -1 and info.get("refused") == "road_edge", (used, info)
    s = _sides(stats)
    assert s["right"]["started"] == 1 and s["left"]["started"] == 0, s
    assert s["left"]["refused"] == {"road_edge": 1} and s["right"]["refused"] == {}, s
    # Left is still behind: the next attempt aims left again.
    assert ExcursionClock(stats.weave_side).start_move("weave", 1).direction == -1
    return ["no fallback: -1 refused road_edge, clock stays in cruise, no +1 line built or counted; "
            "next attempt still -1"]


def check_stats() -> list[str]:
    stats = WeaveStats()
    stats.start("weave", V, direction=-1)
    stats.weave_end(-1, "timeout")
    stats.start("weave", V, direction=1)
    stats.weave_end(1, "lane_edge")
    stats.refuse("curve", V, direction=1)
    stats.refuse("no_path", V)  # lane change or re-plan: no side
    s = _sides(stats)
    assert s == {"left": {"started": 1, "completed": 1, "refused": {}, "aborted": {}},
                 "right": {"started": 1, "completed": 0, "refused": {"curve": 1}, "aborted": {"lane_edge": 1}}}, s
    return ["weave_stats.sides: started / completed / refused / aborted per side"]


def main() -> None:
    lines = check_balance() + check_rebuild() + check_gate_left() + check_left_sign() + check_no_fallback() + check_stats()
    for line in lines:
        print(line)
    print("weave side balance ok")


if __name__ == "__main__":
    main()
