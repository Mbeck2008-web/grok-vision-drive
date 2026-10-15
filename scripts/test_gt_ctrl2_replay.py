#!/usr/bin/env python3
"""gt_ctrl2: the curve law held the lane, then drove off a polyline end.

Live on 5d8c24b. 57 s engaged, 608 m, mean 10.65 m/s, in_lane 94%.
Two departures. Fresh ``curve`` ticks still have lane_x vs gt_center
r=1. The 26 ``curve_hold`` ticks freeze lane_x while gt_center follows
the rejected polyline, and that pulls the whole-run r down to 0.255.
At 52.5 s the car is past the end of road 119441, steer has fallen to 0,
and 1.5 s later the nearest polyline is 19 m away. Damage 3062 is on
the first disengaged row.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import curve_speed_mps  # noqa: E402
from python.control.gt_lane_loop import run_gt_lane  # noqa: E402
from python.debug.gt_geom import (  # noqa: E402
    RoadFollow,
    RoadPoly,
    project_follow,
    project_road,
)
from python.debug.gt_summary import load_jsonl, pearson, summarize  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "gt_ctrl2" / "gt.jsonl"


def _engaged(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("engaged")]


def check_fixture() -> None:
    rows = load_jsonl(FIXTURE)
    summary = summarize(rows)
    print(
        f"fixture engaged_s={summary['engaged_s']:.1f} in_lane={summary['in_lane_pct']:.1f} "
        f"driven={summary['driven_m']:.0f} mean_v={summary['mean_speed_mps']:.2f} "
        f"mean_off={summary['mean_abs_offset_m']:.2f} max_off={summary['max_abs_offset_m']:.2f} "
        f"r={summary['lane_x_vs_gt_center_r']:.3f} "
        f"departures={len(summary['departures'])} off_road={len(summary['off_road'])}"
    )
    assert summary["engaged_s"] > 50.0
    assert summary["in_lane_pct"] > 90.0
    assert summary["driven_m"] > 500.0
    assert summary["mean_speed_mps"] > 8.0
    assert summary["mean_abs_offset_m"] < 0.8
    assert summary["max_abs_offset_m"] > 15.0
    assert summary["lane_x_vs_gt_center_r"] < 0.4
    assert len(summary["departures"]) >= 2
    assert len(summary["off_road"]) >= 2
    damage = max(float(row.get("damage") or 0.0) for row in rows)
    assert damage > 1000.0
    print(f"damage={damage:.0f}")


def check_hold_is_why_r_falls() -> None:
    """lane_x is the straight 8 m center of the sample the wheel accepted.

    gt_center_x is that same formula on whatever polyline project_road
    picked this tick. They match on a fresh sample and diverge only while
    the wheel is holding the previous one.
    """
    fresh_x: list[float] = []
    fresh_g: list[float] = []
    hold_n = 0
    for row in _engaged(load_jsonl(FIXTURE)):
        if row.get("lane_x") is None or row.get("gt_center_x") is None:
            continue
        if row.get("steer_why") == "curve":
            fresh_x.append(float(row["lane_x"]))
            fresh_g.append(float(row["gt_center_x"]))
        elif row.get("steer_why") == "curve_hold":
            hold_n += 1
    fresh_r = pearson(fresh_x, fresh_g)
    print(f"fresh curve ticks n={len(fresh_x)} r={fresh_r:.3f} holds={hold_n}")
    assert fresh_r is not None and fresh_r > 0.99
    assert hold_n >= 20


def _roads() -> tuple[RoadPoly, RoadPoly, RoadPoly]:
    """A runs north and ends. C crosses at the end. B continues north."""
    ahead = RoadPoly("A", [(0.0, float(i)) for i in range(0, 31)], 7.0)
    cross = RoadPoly("C", [(float(i) - 15.0, 30.0) for i in range(0, 31)], 7.0)
    cont = [(0.0, 30.0)]
    for i in range(1, 31):
        cont.append((0.0, 30.0 + float(i)))
    return ahead, cross, RoadPoly("B", cont, 7.0)


def check_follow_ignores_the_cross_street() -> None:
    ahead, _cross, _cont = _roads()
    # C crosses A at y=20, and the car sits on C so the raw nearest is C.
    cross = RoadPoly("C", [(float(i) - 15.0, 20.0) for i in range(0, 31)], 7.0)
    car = (1.2, 20.0)
    heading = 1.5707963
    nearest = project_road([ahead, cross], car[0], car[1], heading)
    assert nearest is not None and nearest.road_id == "C", nearest
    follow = RoadFollow(road_id="A", forward_heading=heading)
    held = project_follow([ahead, cross], car[0], car[1], heading, follow)
    assert held is not None
    print(f"mid-road nearest={nearest.road_id} follow={held.road_id} disc={held.discontinuous}")
    assert held.road_id == "A"
    assert held.discontinuous is False


def check_follow_chains_and_a_dead_end_stops() -> None:
    ahead, cross, cont = _roads()
    # Past A's tip, closer to the cross street than to the continuation.
    car = (3.0, 32.0)
    heading = 1.5707963
    nearest = project_road([ahead, cross, cont], car[0], car[1], heading)
    assert nearest is not None and nearest.road_id == "C", nearest
    follow = RoadFollow(road_id="A", forward_heading=heading)
    chained = project_follow([ahead, cross, cont], car[0], car[1], heading, follow)
    assert chained is not None
    print(f"junction nearest={nearest.road_id} chained={chained.road_id}")
    assert chained.road_id == "B"
    assert chained.discontinuous is False

    follow = RoadFollow(road_id="A", forward_heading=heading)
    dead = project_follow([ahead, cross], car[0], car[1], heading, follow)
    assert dead is not None
    print(f"dead-end follow={dead.road_id} disc={dead.discontinuous} dist={dead.dist_m:.1f}")
    assert dead.road_id == "A"
    # Ten metres past the tip, still the nearest cross street, nothing continues.
    follow = RoadFollow(road_id="A", forward_heading=heading)
    gone = project_follow([ahead, cross], 1.0, 40.0, heading, follow)
    assert gone is not None and gone.road_id == "A" and gone.discontinuous is True, gone
    # Still on the lane with the tip ahead. gt_ctrl3 parked here for 69 s.
    follow = RoadFollow(road_id="A", forward_heading=heading)
    early = project_follow([ahead], 0.2, 20.0, heading, follow, speed_mps=12.0)
    assert early is not None and early.discontinuous is False, early
    follow = RoadFollow(road_id="A", forward_heading=heading)
    room = project_follow([ahead], 0.2, 5.0, heading, follow, speed_mps=12.0)
    assert room is not None and room.discontinuous is False, room
    # Join the old gates missed: 12 m gap, 0.7 rad bend. A cross street is ~1.57.
    bend = [(0.0, 42.0)]
    aim = heading + 0.7
    for i in range(1, 20):
        bend.append((0.0 + i * math.sin(aim), 42.0 + i * math.cos(aim)))
    follow = RoadFollow(road_id="A", forward_heading=heading)
    joined = project_follow(
        [ahead, cross, RoadPoly("D", bend, 7.0)], 0.2, 28.0, heading, follow, speed_mps=8.0,
    )
    assert joined is not None and joined.road_id == "D" and joined.discontinuous is False, joined


def check_speed_cap_and_immediate_brake() -> None:
    cap = curve_speed_mps(1.0 / 40.0)
    print(f"40 m bend cap={cap:.2f}")
    assert abs(cap - (2.5 * 40.0) ** 0.5) < 1e-6
    assert curve_speed_mps(0.0) == float("inf")
    # gt_ctrl2 was still at 12 m/s a second and a half after the mesh ended.
    # The brake is on the first dropped tick, so a second later the car
    # has already shed several m/s and has not been flung out to 19 m.
    drop = run_gt_lane(
        seconds=4,
        speed_mps=12.0,
        offset_m=0.2,
        heading_err_rad=0.0,
        curvature=1.0 / 80.0,
        gap_at_s=0.2,
        gap_s=3.0,
        half_width_m=1.7,
        skip_launch=True,
        law="curve",
    )
    print(
        f"discontinuity max_off={drop['max_abs_offset_m']:.2f} "
        f"final_v_proxy mean={drop['mean_speed_mps']:.2f}"
    )
    assert drop["max_abs_offset_m"] < 1.5
    assert drop["mean_speed_mps"] < 8.0


def main() -> None:
    check_fixture()
    check_hold_is_why_r_falls()
    check_follow_ignores_the_cross_street()
    check_follow_chains_and_a_dead_end_stops()
    check_speed_cap_and_immediate_brake()
    print("gt_ctrl2 replay: follower stays on the continuous road")


if __name__ == "__main__":
    main()
