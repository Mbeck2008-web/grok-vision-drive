#!/usr/bin/env python3
"""GT lane source, historical steer saturation, and the offline closed loop."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.gt_lane_loop import run_gt_lane  # noqa: E402
from scripts.live_lane_test import build_parser  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
# Run 8 is the arcade reverse into the building. Those full-lock ticks are
# not a forward lane-hold sample.
FORWARD_RUNS = ("gt_run5", "gt_run6", "gt_run7", "gt_run9")


def _engaged(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return [row for row in rows if row.get("engaged")]


def _sat_share(rows: list[dict], limit: float = 0.95) -> tuple[int, int]:
    sat = 0
    for row in rows:
        try:
            steer = abs(float(row.get("steer") or 0.0))
        except (TypeError, ValueError):
            continue
        if steer >= limit:
            sat += 1
    return sat, len(rows)


def check_parser() -> None:
    args = build_parser().parse_args(["--lane-source", "gt", "--seconds", "120"])
    assert args.lane_source == "gt"
    assert args.seconds == 120.0


def check_historical_saturation() -> None:
    pooled_sat = 0
    pooled_n = 0
    for name in FORWARD_RUNS:
        rows = _engaged(FIXTURES / name / "gt.jsonl")
        sat, n = _sat_share(rows)
        assert sat == 0, name
        pooled_sat += sat
        pooled_n += n
        print(f"{name} engaged={n} |steer|>=0.95 {sat}")
    assert pooled_n > 0
    assert pooled_sat / pooled_n == 0.0
    reverse = _engaged(FIXTURES / "gt_run8" / "gt.jsonl")
    sat, n = _sat_share(reverse)
    assert sat > 0 and n > 0
    print(f"gt_run8 reverse engaged={n} |steer|>=0.95 {sat} share={sat / n:.3f} excluded")
    print("gt_run10 log is not in this tree")


def check_closed_loop_holds() -> None:
    straight = run_gt_lane(offset_m=0.4, heading_err_rad=0.05, curvature=0.0)
    curve = run_gt_lane(offset_m=0.2, heading_err_rad=0.0, curvature=1.0 / 200.0)
    for name, result in (("straight", straight), ("curve", curve)):
        print(
            f"{name} driven_m={result['driven_m']:.0f} mean_speed={result['mean_speed_mps']:.2f} "
            f"in_lane={result['in_lane_pct']:.0f} max_off={result['max_abs_offset_m']:.2f} "
            f"sat={result['saturated_share']:.3f} ok={result['ok']}"
        )
        assert result["ok"], result


def main() -> None:
    check_parser()
    check_historical_saturation()
    check_closed_loop_holds()
    print("gt lane source ok")


if __name__ == "__main__":
    main()
