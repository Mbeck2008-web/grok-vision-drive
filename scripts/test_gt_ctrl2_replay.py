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

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


def main() -> None:
    check_fixture()
    check_hold_is_why_r_falls()
    print("gt_ctrl2 fixture recorded")


if __name__ == "__main__":
    main()
