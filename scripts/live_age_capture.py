#!/usr/bin/env python3
"""Read-only: live per-camera hold ages vs the training ages.

Input: the stdout / log of a live ``run_vision`` session (60 s is enough).
Every tick prints one ``[GVD] cam_age`` line with ``tick_ms`` and the 8
``ages`` (``cam_hold.CAM_AGE_ORDER``, -1 = black). This prints, per side
camera, the live median / p95 of the held (non-black) ages and the black
share next to the training median / p95 from ``cam_hold_report`` on the
same recording the model was trained on, and flags:

* PASS per camera when the live median is within 1 tick (1/15 s, 67 ms) of
  training and the p95 within 2 ticks.
* tick p95 (``tick_ms``) must be <= 67 ms.

Training numbers come either from ``--train-json`` (``cam_hold_report.py
--json`` output) or from ``--strips`` plus the report's own selection
arguments (``--cams``, ``--mask-files``, ``--holdout-blocks`` ...), computed
here read-only. Nothing is written.

    py -3 scripts/live_age_capture.py --log live_60s.log --strips <rec5c> ^
        --holdout-blocks 1 --mask-files mask_v103_hop_delta.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402

from python.data.cam_hold import CAM_AGE_ORDER, HOLD_TICK_S, NEVER_HELD  # noqa: E402
from python.sensors.live_hold import parse_ages_text  # noqa: E402

TICK_P95_MAX_MS = 1000.0 * HOLD_TICK_S  # 66.7 ms: one 15 Hz supervisor tick
MEDIAN_TOL_S = 1.0 * HOLD_TICK_S
P95_TOL_S = 2.0 * HOLD_TICK_S
_LINE = re.compile(r"\[GVD\] cam_age .*?tick_ms=(?P<tick>[0-9.]+).*?ages=(?P<ages>\S+)")


def parse_log(lines: Iterable[str]) -> dict[str, Any]:
    ticks: list[float] = []
    ages: dict[str, list[float]] = {cam: [] for cam in CAM_AGE_ORDER}
    scheduled = 0
    for line in lines:
        got = _LINE.search(line)
        if got is None:
            continue
        ticks.append(float(got.group("tick")))
        if " scheduled=1 " in line:
            scheduled += 1
        parsed = parse_ages_text(got.group("ages"))
        for cam in CAM_AGE_ORDER:
            if cam in parsed:
                ages[cam].append(parsed[cam])
    return {"ticks_ms": ticks, "ages": ages, "scheduled": scheduled}


def _stat(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values, dtype=np.float64), q))


def compare(live: dict[str, Any], train_age_s: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"cams": {}, "ticks": len(live["ticks_ms"]), "scheduled_ticks": live["scheduled"]}
    tick_p95 = _stat(live["ticks_ms"], 95)
    out["tick_p95_ms"] = tick_p95
    out["tick_pass"] = tick_p95 is not None and tick_p95 <= TICK_P95_MAX_MS
    all_pass = bool(out["tick_pass"])
    for cam in CAM_AGE_ORDER:
        if cam in NEVER_HELD:
            continue
        values = live["ages"].get(cam) or []
        held = [v for v in values if v >= 0.0]
        train = train_age_s.get(cam) or {}
        live_med, live_p95 = _stat(held, 50), _stat(held, 95)
        tr_med, tr_p95 = train.get("median"), train.get("p95")
        ok = (
            live_med is not None
            and live_p95 is not None
            and tr_med is not None
            and tr_p95 is not None
            and abs(live_med - float(tr_med)) <= MEDIAN_TOL_S + 1e-9
            and abs(live_p95 - float(tr_p95)) <= P95_TOL_S + 1e-9
        )
        all_pass = all_pass and ok
        out["cams"][cam] = {
            "live_n": len(held),
            "live_black_pct": None if not values else round(100.0 * (len(values) - len(held)) / len(values), 2),
            "live_median": live_med,
            "live_p95": live_p95,
            "train_median": tr_med,
            "train_p95": tr_p95,
            "pass": bool(ok),
        }
    out["pass"] = all_pass
    return out


def table(result: dict[str, Any]) -> str:
    def f(v: Any) -> str:
        return "n/a" if v is None else f"{float(v):.3f}"

    lines = [
        f"ticks={result['ticks']} scheduled_ticks={result['scheduled_ticks']} "
        f"tick_p95_ms={f(result['tick_p95_ms'])} (max {TICK_P95_MAX_MS:.1f}) {'PASS' if result['tick_pass'] else 'FAIL'}",
        f"{'cam':<9}{'live med':>10}{'train med':>11}{'live p95':>10}{'train p95':>11}{'live black %':>14}  verdict",
    ]
    for cam, row in result["cams"].items():
        black = "n/a" if row["live_black_pct"] is None else f"{row['live_black_pct']:.2f}"
        lines.append(
            f"{cam:<9}{f(row['live_median']):>10}{f(row['train_median']):>11}{f(row['live_p95']):>10}"
            f"{f(row['train_p95']):>11}{black:>14}  {'PASS' if row['pass'] else 'FAIL'}"
        )
    lines.append(
        f"overall {'PASS' if result['pass'] else 'FAIL'} (median within {MEDIAN_TOL_S * 1000:.0f} ms, "
        f"p95 within {P95_TOL_S * 1000:.0f} ms, tick p95 <= {TICK_P95_MAX_MS:.1f} ms)"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live cam-age capture vs training ages (read-only)")
    parser.add_argument("--log", type=Path, required=True, help="run_vision stdout / log with '[GVD] cam_age' lines.")
    parser.add_argument("--train-json", type=Path, default=None, help="cam_hold_report.py --json output.")
    parser.add_argument("--strips", type=Path, default=None, help="Recording to compute the training ages from.")
    parser.add_argument("--cams", default="")
    parser.add_argument("--mask-files", default="")
    parser.add_argument("--mask-events", action="store_true")
    parser.add_argument("--holdout-blocks", type=int, default=None)
    parser.add_argument("--holdout-cells-file", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the table.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.train_json is None) == (args.strips is None):
        raise SystemExit("give exactly one of --train-json or --strips")
    if args.train_json is not None:
        train = json.loads(args.train_json.read_text(encoding="utf-8"))
    else:
        from cam_hold_report import report

        from python.train.cam_strip import parse_cams
        from python.train.train_scene import resolve_holdout_blocks

        train = report(
            args.strips,
            cams=parse_cams(args.cams),
            mask_files=args.mask_files,
            mask_events=bool(args.mask_events),
            holdout_blocks=resolve_holdout_blocks(args.holdout_blocks),
            holdout_cells_file=args.holdout_cells_file,
        )
    with args.log.open("r", encoding="utf-8", errors="replace") as handle:
        live = parse_log(handle)
    if not live["ticks_ms"]:
        raise SystemExit(f"{args.log}: no '[GVD] cam_age' lines")
    result = compare(live, train.get("train_age_s") or {})
    print(json.dumps(result, indent=1) if args.json else table(result), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
