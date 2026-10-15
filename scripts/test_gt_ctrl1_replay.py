#!/usr/bin/env python3
"""gt_ctrl1: perfect GT lanes, the car still left.

Live on bf13d4d, ``--lane-source gt``, 120 s. lane_x matched gt_center
(r=1). The wheels also matched the command: heading integrated from the
logged steer at the calibrated gain and delay stays within a few
hundredths of a radian. The departure is the controller. An 8 m straight
lookahead holds a 50 m curve only by sitting ~0.7 m off the center, and
the moment the projection says off-road the wheel is zeroed and the
parking brake comes on while the car is still doing 14 m/s. That coast
is this replay. It has to keep failing on the legacy law.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.gt_lane_loop import replay_steer_heading, run_gt_lane  # noqa: E402
from python.debug.gt_summary import load_jsonl, summarize  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "gt_ctrl1" / "gt.jsonl"


def check_fixture() -> None:
    rows = load_jsonl(FIXTURE)
    summary = summarize(rows)
    print(
        f"fixture engaged_s={summary['engaged_s']:.1f} in_lane={summary['in_lane_pct']:.1f} "
        f"driven={summary['driven_m']:.0f} mean_v={summary['mean_speed_mps']:.2f} "
        f"max_off={summary['max_abs_offset_m']:.2f} "
        f"r={summary['lane_x_vs_gt_center_r']:.3f}"
    )
    assert summary["engaged_s"] > 110.0
    assert summary["in_lane_pct"] < 50.0
    assert summary["driven_m"] > 500.0
    assert summary["mean_speed_mps"] > 4.0
    assert summary["max_abs_offset_m"] > 15.0
    assert summary["lane_x_vs_gt_center_r"] > 0.99
    assert len(summary["departures"]) >= 1
    assert len(summary["off_road"]) >= 1


def check_heading_matches_command() -> None:
    report = replay_steer_heading(load_jsonl(FIXTURE))
    print(
        f"heading replay n={report['n']:.0f} rms={report['rms']:.4f} "
        f"max={report['max_abs']:.3f}"
    )
    assert report["n"] > 200
    # Gain 1.0 at the same 0.12 s delay is ~0.048. This locks the 0.89 fit.
    assert report["rms"] < 0.04
    assert report["max_abs"] < 0.12


def check_legacy_leaves_the_curve() -> None:
    # Steady bias: 8 m pursuit of a straight center on a 50 m bend.
    steady = run_gt_lane(
        seconds=40,
        offset_m=0.15,
        heading_err_rad=0.0,
        curvature=1.0 / 50.0,
        half_width_m=1.7,
    )
    print(
        f"legacy steady50 final_off={steady['final_offset_m']:.2f} "
        f"max={steady['max_abs_offset_m']:.2f}"
    )
    assert steady["final_offset_m"] > 0.5

    # The live drop: centered, 14 m/s, measurement gone, road keeps bending.
    drop = run_gt_lane(
        seconds=6,
        speed_mps=14.0,
        offset_m=0.1,
        heading_err_rad=-0.02,
        curvature=1.0 / 50.0,
        gap_at_s=0.0,
        gap_s=6.0,
        half_width_m=1.7,
        skip_launch=True,
    )
    print(
        f"legacy drop max_off={drop['max_abs_offset_m']:.2f} "
        f"in_lane={drop['in_lane_pct']:.0f} ok={drop['ok']}"
    )
    assert drop["max_abs_offset_m"] > 3.5
    assert drop["in_lane_pct"] < 50.0
    assert drop["ok"] is False


def main() -> None:
    check_fixture()
    check_heading_matches_command()
    check_legacy_leaves_the_curve()
    print("gt_ctrl1 legacy replay fails as the live run did")


if __name__ == "__main__":
    main()
