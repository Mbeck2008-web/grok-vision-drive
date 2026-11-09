#!/usr/bin/env python3
"""Holdout matches the README: --holdout-blocks on 12-minute segments, train-only roots.

Collector chunks are 12 minutes, under LONG_SEGMENT_S (15 min). The planner
used to ignore --holdout-blocks and fraction on those short tagged segments
and hold out every fifth segment, including on a root combined_pass_plan
calls with fraction=0 (README: train-only).

    py -3 scripts/test_holdout_honest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.train.continual import pass_plan  # noqa: E402
from python.train.strip_sources import combined_pass_plan  # noqa: E402


def _tagged(names: list[str], duration_s: float = 720.0, hz: float = 1.0, source: str | None = None):
    rows = []
    i = 0
    t = 0.0
    for name in names:
        n = int(duration_s * hz)
        for _k in range(n):
            row = {"i": i, "t": t, "dt_s": 1.0 / hz, "segment": name}
            if source is not None:
                row["source"] = source
            rows.append(row)
            i += 1
            t += 1.0 / hz
        t += 10.0
    return rows


def _hold_segments(plan):
    return {r["segment"] for item in plan if item["kind"] == "holdout" for r in item["rows"]}


def check_holdout_blocks_on_12min_segments() -> None:
    names = [f"map#{i}" for i in range(6)]
    rows = _tagged(names, duration_s=720.0)
    plan = pass_plan(rows, 16, 8, fraction=0.2, blocks=5)
    held = _hold_segments(plan)
    # Old rule: only every fifth name (map#4). Time blocks score a slice of
    # each segment, so more than one segment appears in the holdout.
    assert held != {"map#4"}, held
    assert len(held) > 1, held
    by_i = {int(r["i"]): r["segment"] for r in rows}
    train_segs = {by_i[int(r["i"])] for item in plan if item["kind"] == "train" for r in item["rows"]}
    assert "map#4" in train_segs, train_segs  # no longer dropped wholesale
    print(f"12-minute segments + blocks=5 hold out slices of {sorted(held)}, not only map#4 OK")


def check_fraction_zero_is_train_only() -> None:
    rows = _tagged([f"map#{i}" for i in range(6)], duration_s=720.0)
    plan = pass_plan(rows, 16, 8, fraction=0.0, blocks=1)
    assert not any(item["kind"] == "holdout" for item in plan), "fraction=0 must not hold out short segments"
    print("fraction=0 on 12-minute segments: no holdout OK")


def check_combined_second_root_stays_train_only() -> None:
    prior = _tagged([f"p#{i}" for i in range(6)], duration_s=720.0, source="prior")
    new = _tagged([f"n#{i}" for i in range(6)], duration_s=720.0, source="new")
    plan = combined_pass_plan(prior + new, length=16, stride=8, blocks=5, holdout_key="prior")
    hold_src = {r["source"] for item in plan if item["kind"] == "holdout" for r in item["rows"]}
    assert hold_src == {"prior"}, hold_src
    assert any(r["source"] == "new" and item["kind"] == "train" for item in plan for r in item["rows"])
    print("combined second root is train-only OK")


def main() -> None:
    check_holdout_blocks_on_12min_segments()
    check_fraction_zero_is_train_only()
    check_combined_second_root_stays_train_only()
    print("test_holdout_honest: OK")


if __name__ == "__main__":
    main()
