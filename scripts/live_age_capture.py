#!/usr/bin/env python3
"""Read-only: grade a live cam-age capture (polling ON) against a polling-OFF baseline.

Two ``run_vision`` logs of ~60 s each, both with the untrained capture model
(``scripts/export_capture_age_model.py``) and engage hard-disabled
(``--no-engage``):

* ``--baseline-log``: ``--cam-age-baseline`` (companion polls OFF);
* ``--log``: the same command without it (scheduled ad-hoc polls ON).

Every tick prints one ``[GVD] cam_age`` line (``tick_ms``, ``would_skip``,
``main_luma`` / ``main_flash``, the 8 ``ages`` in ``cam_hold.CAM_AGE_ORDER``
with -1 = black, cumulative ``polls`` / ``timeouts_by_cam``), and the start
prints one ``[GVD] cam-age model`` line. Bars, each PASS / FAIL:

* model: both logs ran a gvd_cam_age=1, 8-camera, capture_untrained=1 model
  with engage hard-disabled; polling=on / off as labelled.
* engage_off: every tick in both logs has engaged=0, engage_disabled=1.
* mode: ON ticks are scheduled=1 baseline=0; baseline ticks scheduled=0 baseline=1.
* duration: each log spans >= 55 s of ticks.
* ages: per side camera, live median within 1 tick (1/15 s) and p95 within
  2 ticks of the training ages on the recording the model would train on.
* tick_p95: ON-log ``tick_ms`` p95 <= 66.7 ms (one 15 Hz tick).
* would_skip: 0 ON ticks where actuate would skip vehicle.control because a
  camera GE call holds the socket (counted with engage off).
* timeouts: per side camera, timeouts < 1 % of polls (counts reported).
* main_flash / main_luma_jump: the ON rate of flashed main frames and of
  tick-to-tick mean-luminance jumps (> ``--luma-jump``) is not above the
  baseline rate. Any rise fails.

Training numbers come from ``--train-json`` (``cam_hold_report.py --json``)
or from ``--strips`` plus the report's selection arguments, read-only.

    py -3 scripts/live_age_capture.py --log <ON_LOG> --baseline-log <BASELINE_LOG> ^
        --strips <RECORDING> --holdout-blocks 1 --mask-files <MASK_JSONL>
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
TIMEOUT_MAX_FRAC = 0.01
MIN_SPAN_S = 55.0
LUMA_JUMP = 0.05  # mean main luminance (0..1) change between consecutive ticks counted as a jump
SIDE_CAMS = tuple(cam for cam in CAM_AGE_ORDER if cam not in NEVER_HELD)
_LINE = re.compile(r"\[GVD\] cam_age (?P<body>.*)$")
_MODEL = re.compile(r"\[GVD\] cam-age model (?P<body>.*)$")
_KV = re.compile(r"(\w+)=(\S+)")


def _counts(text: str | None) -> dict[str, int] | None:
    if not text or text == "na":
        return None
    out: dict[str, int] = {}
    for part in text.split(","):
        cam, _, value = part.partition(":")
        if cam and value:
            out[cam] = int(value)
    return out


def _num(text: str | None) -> float | None:
    if text is None or text == "na":
        return None
    return float(text)


def parse_log(lines: Iterable[str]) -> dict[str, Any]:
    ticks: list[dict[str, Any]] = []
    model: dict[str, str] | None = None
    respawns = 0
    for line in lines:
        line = line.rstrip("\n")
        if "[GVD] cam-age: respawn" in line:
            respawns += 1
        got = _MODEL.search(line)
        if got is not None and model is None:
            model = dict(_KV.findall(got.group("body")))
            continue
        got = _LINE.search(line)
        if got is None:
            continue
        kv = dict(_KV.findall(got.group("body")))
        if "tick_ms" not in kv or "ages" not in kv:
            continue
        ticks.append(
            {
                "t": _num(kv.get("t")),
                "tick_ms": float(kv["tick_ms"]),
                "scheduled": kv.get("scheduled") == "1",
                "baseline": kv.get("baseline") == "1",
                "engaged": kv.get("engaged"),
                "engage_disabled": kv.get("engage_disabled"),
                "would_skip": kv.get("would_skip"),
                "main_luma": _num(kv.get("main_luma")),
                "main_flash": kv.get("main_flash"),
                "ages": parse_ages_text(kv["ages"]),
                "polls": _counts(kv.get("polls")),
                "timeouts": _counts(kv.get("timeouts_by_cam")),
            }
        )
    ages = {cam: [t["ages"][cam] for t in ticks if cam in t["ages"]] for cam in CAM_AGE_ORDER}
    return {
        "ticks": ticks,
        "ticks_ms": [t["tick_ms"] for t in ticks],
        "ages": ages,
        "scheduled": sum(1 for t in ticks if t["scheduled"]),
        "model": model,
        "respawns": respawns,
    }


def _stat(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values, dtype=np.float64), q))


def compare(live: dict[str, Any], train_age_s: dict[str, Any]) -> dict[str, Any]:
    """Ages per side camera vs training, and the tick p95 bar."""
    out: dict[str, Any] = {"cams": {}, "ticks": len(live["ticks_ms"]), "scheduled_ticks": live["scheduled"]}
    tick_p95 = _stat(live["ticks_ms"], 95)
    out["tick_p95_ms"] = tick_p95
    out["tick_pass"] = tick_p95 is not None and tick_p95 <= TICK_P95_MAX_MS
    ages_pass = True
    for cam in SIDE_CAMS:
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
        ages_pass = ages_pass and ok
        out["cams"][cam] = {
            "live_n": len(held),
            "live_black_pct": None if not values else round(100.0 * (len(values) - len(held)) / len(values), 2),
            "live_median": live_med,
            "live_p95": live_p95,
            "train_median": tr_med,
            "train_p95": tr_p95,
            "pass": bool(ok),
        }
    out["ages_pass"] = ages_pass
    out["pass"] = bool(ages_pass and out["tick_pass"])
    return out


def _summed(ticks: list[dict[str, Any]], key: str) -> dict[str, int]:
    """Total of a cumulative per-camera counter, robust to a reset (respawn) mid-log."""
    total: dict[str, int] = {}
    prev: dict[str, int] = {}
    for tick in ticks:
        now = tick.get(key)
        if now is None:
            continue
        for cam, value in now.items():
            before = prev.get(cam, 0)
            total[cam] = total.get(cam, 0) + (value - before if value >= before else value)
        prev = now
    return total


def exposure(ticks: list[dict[str, Any]], luma_jump: float = LUMA_JUMP) -> dict[str, Any]:
    lumas = [t["main_luma"] for t in ticks]
    have = [t for t in ticks if t["main_luma"] is not None]
    flashes = sum(1 for t in have if t["main_flash"] == "1")
    pairs = [(a, b) for a, b in zip(lumas, lumas[1:]) if a is not None and b is not None]
    jumps = sum(1 for a, b in pairs if abs(b - a) > luma_jump)
    return {
        "frames": len(have),
        "flashes": flashes,
        "flash_rate": None if not have else flashes / len(have),
        "pairs": len(pairs),
        "jumps": jumps,
        "jump_rate": None if not pairs else jumps / len(pairs),
    }


def _span(ticks: list[dict[str, Any]]) -> float:
    ts = [t["t"] for t in ticks if t["t"] is not None]
    return 0.0 if len(ts) < 2 else float(max(ts) - min(ts))


def grade(
    on: dict[str, Any],
    base: dict[str, Any],
    train_age_s: dict[str, Any],
    *,
    luma_jump: float = LUMA_JUMP,
    min_span_s: float = MIN_SPAN_S,
) -> dict[str, Any]:
    bars: dict[str, dict[str, Any]] = {}

    def model_ok(log: dict[str, Any], polling: str) -> bool:
        m = log.get("model") or {}
        return (
            m.get("gvd_cam_age") == "1"
            and m.get("cams") == str(len(CAM_AGE_ORDER))
            and m.get("capture_untrained") == "1"
            and m.get("engage_disabled") == "1"
            and m.get("polling") == polling
        )

    bars["model"] = {
        "pass": model_ok(on, "on") and model_ok(base, "off"),
        "detail": f"on={on.get('model')} baseline={base.get('model')}",
    }
    eng_bad = {
        name: sum(1 for t in log["ticks"] if t["engaged"] != "0" or t["engage_disabled"] != "1")
        for name, log in (("on", on), ("baseline", base))
    }
    bars["engage_off"] = {"pass": not any(eng_bad.values()), "detail": f"ticks not engaged=0/engage_disabled=1: {eng_bad}"}
    mode_bad = {
        "on": sum(1 for t in on["ticks"] if not t["scheduled"] or t["baseline"]),
        "baseline": sum(1 for t in base["ticks"] if t["scheduled"] or not t["baseline"]),
    }
    bars["mode"] = {
        "pass": bool(on["ticks"]) and bool(base["ticks"]) and not any(mode_bad.values()),
        "detail": f"ticks {len(on['ticks'])} on / {len(base['ticks'])} baseline; wrong mode {mode_bad}",
    }
    spans = {"on": _span(on["ticks"]), "baseline": _span(base["ticks"])}
    bars["duration"] = {
        "pass": all(v >= min_span_s for v in spans.values()),
        "detail": f"span s on={spans['on']:.1f} baseline={spans['baseline']:.1f} (min {min_span_s:g})",
    }
    ages = compare(on, train_age_s)
    bars["ages"] = {
        "pass": ages["ages_pass"],
        "detail": f"median within {MEDIAN_TOL_S * 1000:.0f} ms, p95 within {P95_TOL_S * 1000:.0f} ms",
        "cams": ages["cams"],
    }
    base_p95 = _stat(base["ticks_ms"], 95)
    bars["tick_p95"] = {
        "pass": ages["tick_pass"],
        "detail": f"on p95 {ages['tick_p95_ms'] if ages['tick_p95_ms'] is not None else float('nan'):.2f} ms "
        f"(max {TICK_P95_MAX_MS:.1f}); baseline p95 {base_p95 if base_p95 is not None else float('nan'):.2f} ms",
    }
    skips_on = sum(1 for t in on["ticks"] if t["would_skip"] != "0")
    skips_base = sum(1 for t in base["ticks"] if t["would_skip"] != "0")
    bars["would_skip"] = {
        "pass": bool(on["ticks"]) and skips_on == 0,
        "detail": f"would-skip control ticks on={skips_on}/{len(on['ticks'])} baseline={skips_base}/{len(base['ticks'])}",
    }
    polls, touts = _summed(on["ticks"], "polls"), _summed(on["ticks"], "timeouts")
    per_cam = {}
    t_ok = True
    for cam in SIDE_CAMS:
        n, k = int(polls.get(cam, 0)), int(touts.get(cam, 0))
        ok = n > 0 and k < TIMEOUT_MAX_FRAC * n
        t_ok = t_ok and ok
        per_cam[cam] = {"polls": n, "timeouts": k, "pct": None if n == 0 else round(100.0 * k / n, 3), "pass": ok}
    bars["timeouts"] = {"pass": t_ok, "detail": f"timeouts < {TIMEOUT_MAX_FRAC * 100:g}% of polls per cam", "cams": per_cam}
    ex_on, ex_base = exposure(on["ticks"], luma_jump), exposure(base["ticks"], luma_jump)

    def not_above(key: str) -> bool:
        a, b = ex_on[key], ex_base[key]
        return a is not None and b is not None and a <= b + 1e-12

    bars["main_flash"] = {
        "pass": not_above("flash_rate"),
        "detail": f"flashed main frames on={ex_on['flashes']}/{ex_on['frames']} baseline={ex_base['flashes']}/{ex_base['frames']}",
    }
    bars["main_luma_jump"] = {
        "pass": not_above("jump_rate"),
        "detail": f"|d luma| > {luma_jump:g}: on={ex_on['jumps']}/{ex_on['pairs']} baseline={ex_base['jumps']}/{ex_base['pairs']}",
    }
    return {
        "bars": bars,
        "pass": all(bar["pass"] for bar in bars.values()),
        "respawns": {"on": on.get("respawns", 0), "baseline": base.get("respawns", 0)},
        "exposure": {"on": ex_on, "baseline": ex_base},
    }


def table(result: dict[str, Any]) -> str:
    def f(v: Any) -> str:
        return "n/a" if v is None else f"{float(v):.3f}"

    lines = []
    for name, bar in result["bars"].items():
        lines.append(f"{'PASS' if bar['pass'] else 'FAIL'}  {name:<15} {bar['detail']}")
        if name == "ages":
            lines.append(f"      {'cam':<9}{'live med':>10}{'train med':>11}{'live p95':>10}{'train p95':>11}{'live black %':>14}")
            for cam, row in bar["cams"].items():
                black = "n/a" if row["live_black_pct"] is None else f"{row['live_black_pct']:.2f}"
                lines.append(
                    f"      {cam:<9}{f(row['live_median']):>10}{f(row['train_median']):>11}{f(row['live_p95']):>10}"
                    f"{f(row['train_p95']):>11}{black:>14}  {'PASS' if row['pass'] else 'FAIL'}"
                )
        if name == "timeouts":
            for cam, row in bar["cams"].items():
                pct = "n/a" if row["pct"] is None else f"{row['pct']:.3f}%"
                lines.append(f"      {cam:<9} polls {row['polls']:>6} timeouts {row['timeouts']:>5} {pct:>9}  {'PASS' if row['pass'] else 'FAIL'}")
    lines.append(f"respawns seen: {result['respawns']}")
    lines.append(f"OVERALL {'PASS' if result['pass'] else 'FAIL'}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live cam-age capture vs training ages (read-only)")
    parser.add_argument("--log", type=Path, required=True, help="Polling-ON run_vision log ('[GVD] cam_age' lines).")
    parser.add_argument("--baseline-log", type=Path, required=True, help="Polling-OFF (--cam-age-baseline) log.")
    parser.add_argument("--luma-jump", type=float, default=LUMA_JUMP, help="Mean-luminance jump threshold (0..1).")
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
    logs = {}
    for name, path in (("on", args.log), ("baseline", args.baseline_log)):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            logs[name] = parse_log(handle)
        if not logs[name]["ticks"]:
            raise SystemExit(f"{path}: no '[GVD] cam_age' lines")
    result = grade(logs["on"], logs["baseline"], train.get("train_age_s") or {}, luma_jump=float(args.luma_jump))
    print(json.dumps(result, indent=1) if args.json else table(result), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
