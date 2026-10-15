#!/usr/bin/env python3
"""gt_ctrl3 looked like a pass and was a car parked in the lane.

Live on 7c7d208. 120 s, 560 m, in_lane 100%, max offset 1.24 m, no
damage. The moving half peaked at 16 m/s. From 51 s ``curve_hold`` froze
lane_x at -0.323 and steer at -0.025, and the parking brake held speed
at 0 for the remaining 69 s. The car was still in the lane, heading
error 0.05 rad, 3 m from the polyline. That was a dead-end false
positive, which is why the mean speed is 5.3 m/s and not run 2's 10.6.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.debug.gt_summary import load_jsonl, summarize  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "gt_ctrl3" / "gt.jsonl"


def check_fixture() -> None:
    rows = load_jsonl(FIXTURE)
    summary = summarize(rows)
    engaged = [row for row in rows if row.get("engaged")]
    print(
        f"fixture engaged_s={summary['engaged_s']:.1f} in_lane={summary['in_lane_pct']:.1f} "
        f"driven={summary['driven_m']:.0f} mean_v={summary['mean_speed_mps']:.2f} "
        f"max_off={summary['max_abs_offset_m']:.2f} departures={len(summary['departures'])}"
    )
    assert summary["engaged_s"] > 110.0
    assert summary["in_lane_pct"] >= 99.0
    assert summary["driven_m"] > 500.0
    assert summary["max_abs_offset_m"] < 1.5
    assert len(summary["departures"]) == 0
    assert max(float(row.get("damage") or 0.0) for row in rows) < 1.0
    t0 = float(engaged[0]["t"])
    holds = [row for row in engaged if row.get("steer_why") == "curve_hold"]
    assert len(holds) > 300
    hold_s = float(holds[-1]["t"]) - float(holds[0]["t"])
    assert hold_s > 60.0
    stopped = [row for row in engaged if float(row["t"]) - t0 > 60.0]
    assert stopped
    assert max(float(row["speed_mps"]) for row in stopped) < 0.2
    assert abs(float(stopped[-1]["lane_x"]) - (-0.3226)) < 1e-3
    assert abs(float(stopped[-1]["steer"]) - (-0.0251)) < 1e-3
    assert abs(float(stopped[-1]["lane_offset_m"])) < 0.05
    print(f"hold_s={hold_s:.1f} parked_after_60s n={len(stopped)} lane_offset={stopped[-1]['lane_offset_m']:.3f}")


def main() -> None:
    check_fixture()
    print("gt_ctrl3 fixture: pass numbers, parked in curve_hold")


if __name__ == "__main__":
    main()
