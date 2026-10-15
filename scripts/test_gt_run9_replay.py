#!/usr/bin/env python3
"""gt_run9 parked on the brake, and frame replay r for runs 5-9.

Engaged 120 s, in_lane 100%, driven 36.7 m, mean speed 0.37 m/s. That is
not a pass. The exit brake tripped on |lane_x|>1.25 for the opening slew
(lane_x started at +4.94) and then on yaw-since-engage > 0.22 for the
remaining ~90 s, after lane_x was already under 0.4 m.

Frame replay uses the shipped route: main-camera lines when main has two
of them, otherwise every camera. r is printed for each run. The parked
log has to fail the distance check.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import steer_sample_point  # noqa: E402
from python.debug.gt_summary import distance_failure, format_report, summarize  # noqa: E402
from python.perception.lanes import lanes_from_view  # noqa: E402
from python.planning.path_predictor import LaneTrack, predict_path  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
RUN9 = FIXTURES / "gt_run9"


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx < 1e-9 or vy < 1e-9:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _rows(folder: Path) -> list[dict]:
    return [json.loads(line) for line in (folder / "gt.jsonl").read_text().splitlines() if line.strip()]


def _frames(folder: Path) -> list[tuple[int, Path]]:
    found = []
    for path in folder.glob("*.jpg"):
        if path.name.startswith(("frame_", "depart_")):
            found.append((int(path.stem.split("_")[-1]), path))
    found.sort()
    return found


def _replay(folder: Path) -> tuple[float, float, float, int]:
    rows = _rows(folder)
    by_tick = {i + 1: row for i, row in enumerate(rows)}
    track = LaneTrack()
    pairs: list[tuple[float, float]] = []
    for tick, path in _frames(folder):
        row = by_tick.get(tick)
        if row is None or row.get("gt_center_x") is None:
            continue
        image = cv2.imread(str(path))
        assert image is not None, path
        image = image.copy()
        image[:28, :420] = (30, 30, 32)
        fit = lanes_from_view(None, image)
        plan = predict_path(
            lanes_bev=fit.lanes_bev,
            lane_conf=fit.conf,
            ego_speed_mps=4.0,
            lane_track=track,
        )
        sample = steer_sample_point(plan.path_ego or [])
        if sample is None:
            continue
        pairs.append((float(sample["x"]), float(row["gt_center_x"])))
    assert len(pairs) >= 3, (folder.name, len(pairs))
    xs = [a for a, _b in pairs]
    ys = [b for _a, b in pairs]
    bias = sum(a - b for a, b in pairs) / len(pairs)
    rms = math.sqrt(sum((a - b) ** 2 for a, b in pairs) / len(pairs))
    return _pearson(xs, ys), bias, rms, len(pairs)


def check_run9_is_parked() -> None:
    rows = _rows(RUN9)
    summary = summarize(rows)
    assert summary["driven_m"] is not None
    assert 30.0 < float(summary["driven_m"]) < 50.0, summary["driven_m"]
    assert float(summary["mean_speed_mps"]) < 1.0, summary["mean_speed_mps"]
    assert summary["in_lane_pct"] == 100.0 or float(summary["in_lane_pct"]) > 99.0
    assert distance_failure(summary) is not None
    text = format_report(summary, source="gt_run9")
    assert "driven_m=" in text and "mean_speed_mps=" in text
    # Opening lookahead is several metres off. Yaw then stays past 0.22
    # after the lookahead has come back.
    engaged = [r for r in rows if r.get("engaged")]
    h0 = float(engaged[0]["heading_rad"])
    x_only = yaw_only = 0
    for row in engaged:
        x = abs(float(row["lane_x"])) if row.get("lane_x") is not None else 0.0
        yaw = abs(math.remainder(float(row["heading_rad"]) - h0, 2.0 * math.pi))
        if x > 1.25 and yaw <= 0.22:
            x_only += 1
        elif yaw > 0.22 and x <= 1.25:
            yaw_only += 1
    assert x_only >= 40, x_only
    assert yaw_only >= 300, yaw_only
    print(
        f"  run9 driven_m={summary['driven_m']:.1f} "
        f"mean_speed_mps={summary['mean_speed_mps']:.2f} "
        f"lane_x_trips={x_only} yaw_only_trips={yaw_only}"
    )


def main() -> None:
    check_run9_is_parked()
    for name in ("gt_run5", "gt_run6", "gt_run7", "gt_run8", "gt_run9"):
        corr, bias, rms, n = _replay(FIXTURES / name)
        print(
            f"  {name} replay r={corr:+.3f} bias={bias:+.2f} rms={rms:.2f} n={n}"
        )
    print("test_gt_run9_replay: OK")


if __name__ == "__main__":
    main()
