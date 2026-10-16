#!/usr/bin/env python3
"""Ground-truth report math and road projection. No BeamNG."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.debug.gt_geom import (  # noqa: E402
    RoadPoly,
    heading_from_dir,
    parse_road_edges,
    project_road,
)
from python.debug.gt_logger import GroundTruthLogger, gt_log_enabled  # noqa: E402
from python.debug.gt_summary import (  # noqa: E402
    MIN_DRIVEN_M,
    MIN_MEAN_SPEED_MPS,
    distance_failure,
    format_report,
    load_jsonl,
    pearson,
    render_plot,
    summarize,
)

FIXTURE = ROOT / "tests" / "fixtures" / "gt_lane_sample.jsonl"


def sample_rows() -> list[dict]:
    """Eight seconds on a straight road, one lane departure, one off-road tick."""
    rows = []
    t = 0.0
    while t <= 5.0 + 1e-9:
        offset = 0.40
        off_road = False
        steer = 0.10
        if 2.0 - 1e-9 <= t <= 2.4 + 1e-9:
            offset = 2.20
            steer = -0.30
        elif abs(t - 4.0) < 1e-9:
            offset = 5.0
            off_road = True
            steer = 0.15
        elif t > 2.4:
            steer = 0.15
        rows.append({
            "t": round(t, 2),
            "engaged": True,
            "lane_offset_m": offset,
            "lane_width_m": 3.5,
            "off_road": off_road,
            "road_id": "west_coast",
            "lane_id": 1,
            "lane_x": round(offset * 0.8 + 0.1, 4),
            "steer": steer,
            "steer_why": "pursuit",
            "lane_src": "pair",
            "frame_cmd_ms": 120.0,
            "dist_m": abs(offset),
        })
        t = round(t + 0.2, 2)
    return rows


def check_projection() -> None:
    road = RoadPoly("hwy", [(0.0, 0.0), (0.0, 100.0)], 7.0)
    heading = heading_from_dir(0.0, 1.0)
    hit = project_road([road], 1.0, 20.0, heading)
    assert hit is not None
    assert abs(hit.offset_m - 1.0) < 1e-6, hit
    assert hit.lane_id == 1, hit
    assert abs(hit.lane_offset_m - (1.0 - 1.75)) < 1e-6, hit.lane_offset_m
    assert hit.off_road is False
    assert hit.heading_err_rad is not None and abs(hit.heading_err_rad) < 1e-6
    # Pointing east while the road runs north is a clockwise heading error.
    east = project_road([road], 1.0, 20.0, heading_from_dir(1.0, 0.0))
    assert east is not None and east.heading_err_rad is not None
    assert abs(east.heading_err_rad + 1.5707963) < 1e-3, east.heading_err_rad
    outside = project_road([road], 4.5, 20.0, heading)
    assert outside is not None and outside.off_road is True
    far = project_road([road], 40.0, 20.0, heading)
    assert far is not None and far.off_road is True


def check_parse_edges() -> None:
    edges = [
        {"left": [0, 0, 0], "right": [4, 0, 0], "middle": [2, 0, 0]},
        {"left": [0, 10, 0], "right": [4, 10, 0], "middle": [2, 10, 0]},
    ]
    poly = parse_road_edges("r1", edges)
    assert poly is not None
    assert poly.xy[0] == (2.0, 0.0)
    assert abs(poly.width_m - 4.0) < 1e-6
    implied = parse_road_edges("r2", [
        {"left": [0, 0, 0], "right": [4, 0, 0]},
        {"left": [0, 8, 0], "right": [4, 8, 0]},
    ])
    assert implied is not None and implied.xy[0] == (2.0, 0.0)
    # radius is half-width. A 2 m left/right span with radius 4 is an 8 m road.
    wide = parse_road_edges("r3", [
        {"left": [-1, 0, 0], "right": [1, 0, 0], "middle": [0, 0, 0], "radius": 4.0},
        {"left": [-1, 20, 0], "right": [1, 20, 0], "middle": [0, 20, 0], "radius": 4.0},
    ])
    assert wide is not None and abs(wide.width_m - 8.0) < 1e-6, wide
    # One junction node is 10 m. The road itself is 6 m. The median wins.
    flare = [{"left": [-3, float(i), 0], "right": [3, float(i), 0], "middle": [0, float(i), 0]} for i in range(0, 30, 5)]
    flare.append({"left": [-5, 30, 0], "right": [5, 30, 0], "middle": [0, 30, 0]})
    flared = parse_road_edges("flare", flare)
    assert flared is not None and abs(flared.width_m - 6.0) < 1e-6, flared


def check_report_math() -> None:
    rows = sample_rows()
    summary = summarize(rows)
    # 26 ticks, 0.2 s apart: 5.0 s engaged. Four ticks sit outside the lane
    # (2.0, 2.2, 2.4, 4.0), so 0.8 s is out and 4.2 s is in.
    assert abs(summary["engaged_s"] - 5.0) < 1e-6, summary["engaged_s"]
    assert abs(summary["in_lane_s"] - 4.2) < 1e-6, summary["in_lane_s"]
    assert summary["in_lane_pct"] is not None
    assert abs(summary["in_lane_pct"] - 84.0) < 1e-6, summary["in_lane_pct"]
    assert len(summary["departures"]) == 2, summary["departures"]
    assert abs(summary["departures"][0]["t"] - 2.0) < 1e-6
    assert abs(summary["departures"][1]["t"] - 4.0) < 1e-6
    assert len(summary["off_road"]) == 1
    assert abs(summary["off_road"][0]["t"] - 4.0) < 1e-6
    assert summary["max_abs_offset_m"] == 5.0
    assert summary["steer_sign_changes"] == 2, summary["steer_sign_changes"]
    assert summary["steer_sign_per_s"] is not None
    assert abs(summary["steer_sign_per_s"] - 0.4) < 1e-6
    corr = summary["lane_x_vs_offset_r"]
    assert corr is not None and corr > 0.99, corr
    assert summary["frame_cmd_ms_p50"] == 120.0
    text = format_report(summary, source="sample")
    assert "in_lane_pct=84.0" in text
    assert "departures=2" in text
    assert "driven_m=0.0" in text
    assert "mean_speed_mps=na" in text
    assert distance_failure(summary) is not None
    moving = []
    x = 0.0
    for i in range(40):
        x += 10.0
        moving.append({
            "t": i * 0.5,
            "engaged": True,
            "lane_offset_m": 0.2,
            "lane_width_m": 3.5,
            "off_road": False,
            "lane_x": 0.1,
            "steer": 0.05,
            "gt_center_x": 0.1,
            "speed_mps": 8.0,
            "pos": (x, 0.0, 0.0),
            "frame_cmd_ms": 100.0,
        })
    fast = summarize(moving)
    assert fast["driven_m"] >= MIN_DRIVEN_M, fast["driven_m"]
    assert fast["mean_speed_mps"] >= MIN_MEAN_SPEED_MPS
    assert distance_failure(fast) is None
    ident = pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert ident is not None and abs(ident - 1.0) < 1e-9, ident
    on_disk = load_jsonl(FIXTURE)
    disk = summarize(on_disk)
    assert abs(disk["in_lane_pct"] - summary["in_lane_pct"]) < 1e-6
    print(text)


def check_plot_and_logger() -> None:
    assert gt_log_enabled(False) is False or __import__("os").environ.get("GVD_GT_LOG", "") in (
        "1", "true", "yes", "on",
    )
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        png = render_plot(sample_rows(), folder / "gt.png")
        assert png.is_file() and png.stat().st_size > 100
        logger = GroundTruthLogger(folder / "gt.jsonl", folder / "frames")

        class Pose:
            pos = (1.0, 20.0, 0.0)
            dir = (0.0, 1.0, 0.0)
            speed_mps = 8.0
            damage = 1.0

        logger._roads = [RoadPoly("hwy", [(0.0, 0.0), (0.0, 100.0)], 7.0)]
        logger._road_note = ""
        row = logger.write(
            vdata=Pose(), engaged=True, lane_x=0.4, lane_src="pair",
            steer=0.1, steer_why="pursuit", frame_cmd_ms=115.0,
        )
        assert row["road_id"] == "hwy"
        assert abs(row["lane_offset_m"] - (1.0 - 1.75)) < 1e-6
        assert row["in_lane"] is True
        saved = list((folder / "frames").glob("*.jpg"))
        assert saved, "first tick should save a debug frame"
        text = (folder / "gt.jsonl").read_text(encoding="utf-8")
        assert json.loads(text)["lane_src"] == "pair"


def main() -> None:
    check_projection()
    check_parse_edges()
    check_report_math()
    check_plot_and_logger()
    print("test_gt_report: OK")


if __name__ == "__main__":
    main()
