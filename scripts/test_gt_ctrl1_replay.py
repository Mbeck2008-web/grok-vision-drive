#!/usr/bin/env python3
"""gt_ctrl1: perfect GT lanes, the car still left.

Live on bf13d4d, ``--lane-source gt``, 120 s. lane_x matched gt_center
(r=1). The wheels also matched the command: heading integrated from the
logged steer at the calibrated gain and delay stays within a few
hundredths of a radian. The departure is the controller. An 8 m straight
lookahead holds a 50 m curve only by sitting ~0.7 m off the center, and
the moment the projection says off-road the wheel is zeroed and the
parking brake comes on while the car is still doing 14 m/s. That coast
fails on the legacy law. Stanley plus curvature feedforward, held through
the same gap, stays on the center.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import curve_steer  # noqa: E402
from python.control.gt_lane_loop import replay_steer_heading, run_gt_lane  # noqa: E402
from python.debug.gt_geom import (  # noqa: E402
    PreviewMemory,
    RoadPoly,
    heading_from_dir,
    project_road,
    update_curve_preview,
)
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
        law="legacy",
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
        law="legacy",
    )
    print(
        f"legacy drop max_off={drop['max_abs_offset_m']:.2f} "
        f"in_lane={drop['in_lane_pct']:.0f} ok={drop['ok']}"
    )
    assert drop["max_abs_offset_m"] > 3.5
    assert drop["in_lane_pct"] < 50.0
    assert drop["ok"] is False


def check_curve_holds() -> None:
    # Same 50 m road the legacy law can only hold by sitting 0.7 m off center.
    held = run_gt_lane(
        seconds=40,
        offset_m=0.15,
        heading_err_rad=0.0,
        curvature=1.0 / 50.0,
        half_width_m=1.7,
        law="curve",
    )
    print(
        f"curve steady50 final_off={held['final_offset_m']:.3f} "
        f"max={held['max_abs_offset_m']:.3f} ok={held['ok']}"
    )
    assert held["ok"]
    assert abs(held["final_offset_m"]) < 0.25
    assert held["max_abs_offset_m"] < 0.4

    # Measurement drops after the car is up to speed, the way the polyline
    # ended under gt_ctrl1. The wheel keeps the bend.
    gap = run_gt_lane(
        seconds=40,
        offset_m=0.15,
        heading_err_rad=0.0,
        curvature=1.0 / 50.0,
        gap_at_s=20.0,
        gap_s=3.0,
        half_width_m=1.7,
        law="curve",
    )
    print(
        f"curve gap max_off={gap['max_abs_offset_m']:.3f} "
        f"final={gap['final_offset_m']:.3f} in_lane={gap['in_lane_pct']:.0f}"
    )
    assert gap["max_abs_offset_m"] < 0.5
    assert abs(gap["final_offset_m"]) < 0.25
    assert gap["in_lane_pct"] > 90.0
    legacy = run_gt_lane(
        seconds=40,
        offset_m=0.15,
        heading_err_rad=0.0,
        curvature=1.0 / 50.0,
        gap_at_s=20.0,
        gap_s=3.0,
        half_width_m=1.7,
        law="legacy",
    )
    assert legacy["max_abs_offset_m"] > gap["max_abs_offset_m"] + 1.0


def check_steer_signs() -> None:
    # Left bend, centered and aligned: steer left (negative).
    left = curve_steer(0.0, 0.0, 1.0 / 50.0, 12.0)
    right = curve_steer(0.0, 0.0, -1.0 / 50.0, 12.0)
    nose_left = curve_steer(0.0, 0.2, 0.0, 12.0)
    car_right = curve_steer(0.4, 0.0, 0.0, 12.0)
    print(f"signs left={left:+.3f} right={right:+.3f} nose={nose_left:+.3f} right_of={car_right:+.3f}")
    assert left < -0.05
    assert right > 0.05
    assert nose_left > 0.05
    assert car_right < -0.01
    assert abs(curve_steer(0.0, 0.0, 0.0, 12.0)) < 1e-9


def _left_arc(radius: float, length_m: float = 40.0, step_m: float = 1.0) -> RoadPoly:
    n = int(length_m / step_m) + 1
    pts = []
    for i in range(n):
        phi = (i * step_m) / radius
        pts.append((-radius * (1.0 - math.cos(phi)), radius * math.sin(phi)))
    return RoadPoly("arc", pts, 7.0)


def check_road_curvature() -> None:
    radius = 50.0
    road = _left_arc(radius)
    phi = 0.15
    x = -radius * (1.0 - math.cos(phi))
    y = radius * math.sin(phi)
    heading = math.pi / 2.0 + phi
    hit = project_road([road], x, y, heading)
    assert hit is not None
    print(f"arc kappa={hit.curvature:.4f} ahead={hit.ahead_m:.1f}")
    assert abs(hit.curvature - 1.0 / radius) < 0.004
    assert hit.ahead_m > 10.0
    straight = RoadPoly("hwy", [(0.0, 0.0), (0.0, 100.0)], 7.0)
    flat = project_road([straight], 1.0, 20.0, heading_from_dir(0.0, 1.0))
    assert flat is not None and abs(flat.curvature) < 1e-6


def check_jump_is_held() -> None:
    memory = PreviewMemory()
    first = update_curve_preview(
        memory,
        offset_m=0.15,
        heading_err_rad=-0.02,
        curvature=1.0 / 50.0,
        road_id="63008",
        beside=True,
        dt_s=0.2,
    )
    assert first is not None and not first.held
    # gt_ctrl1 t=46 s: a crossing road, still "in" that lane, heading off by 1.5 rad.
    jumped = update_curve_preview(
        memory,
        offset_m=-0.80,
        heading_err_rad=-1.56,
        curvature=0.0,
        road_id="62913",
        beside=True,
        dt_s=0.2,
    )
    assert jumped is not None and jumped.held
    assert jumped.road_id == "63008"
    assert abs(jumped.curvature - 1.0 / 50.0) < 1e-9
    assert abs(jumped.offset_m - 0.15) < 1e-9
    # Past the hold, the caller slows down. The curve is still the command.
    lost = first
    for _ in range(10):
        lost = update_curve_preview(
            memory,
            offset_m=None,
            heading_err_rad=None,
            curvature=0.0,
            road_id="",
            beside=False,
            dt_s=0.2,
        )
    assert lost is not None and lost.lost
    assert abs(lost.curvature - 1.0 / 50.0) < 1e-9


def main() -> None:
    check_fixture()
    check_heading_matches_command()
    check_legacy_leaves_the_curve()
    check_curve_holds()
    check_steer_signs()
    check_road_curvature()
    check_jump_is_held()
    print("gt_ctrl1 replay: legacy leaves, curve law holds")


if __name__ == "__main__":
    main()
