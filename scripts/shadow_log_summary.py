#!/usr/bin/env python3
"""Shadow-log summary: a compact, report-only read of one shadow-run GT log.

Debug tool, not the B4 grader: it scores nothing against a bar and refuses
nothing. It prints and writes (``--json``) tick count and rate, the model
sha256 and its label-frame stamp, the masked fraction by reason, the lane
center error at 8 m over unmasked ticks with GT, the longest unscored streaks
and the worst ticks. Exit 0 on any readable log.

Log format. B1 (docs/SHADOW_GATE.md, shadow_run_spec B1) is not in
``gt_logger`` yet; today's rows have ``t``, ``model_lane_x``, ``gt_center_x``
(road network, not paint) and ``engaged`` only. This reads the spec's fields,
with these key names assumed until B1 fixes them:

* header: the first line when it has ``"header": true``; ``model_sha256`` and
  ``onnx_meta`` (with ``gvd_gt_origin`` / ``gvd_gt_labeler``, the E0 keys);
* per tick: ``frame_t`` (else ``t``), ``model_lane_x``, ``mask`` (a reason
  string, a list of them, or null), ``gt_center_x``;
* labels: ``<log stem>.paint_v103.jsonl`` (B2b) or ``--labels``, rows keyed on
  ``frame_t`` with ``tech.lane_gt`` / ``tech.lane_gt_valid``.

GT is the paint label when a labels file exists; otherwise the log's own
``gt_center_x`` and the report says it is not paint. A stamp that is missing,
``unknown`` or comma-joined (mixed frames) is flagged as one B4 would refuse.

    py -3 scripts/shadow_log_summary.py logs/gt_<stamp>.jsonl --json summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

TOP = 5


def _ts(row: dict) -> float | None:
    v = row.get("frame_t", row.get("t"))
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None


def _masks(row: dict) -> list[str]:
    m = row.get("mask")
    if not m:
        return []
    return [str(x) for x in (m if isinstance(m, list) else [m]) if x]


def _stamp(meta: Any) -> dict:
    from python.train.cam_strip import GT_LABELER_META, GT_ORIGIN_META

    meta = meta if isinstance(meta, dict) else {}
    out: dict[str, Any] = {k: meta.get(k) for k in (GT_ORIGIN_META, GT_LABELER_META)}
    flags = []
    for k, v in out.items():
        if not v or str(v) == "unknown":
            flags.append(f"{k} unknown")
        elif "," in str(v):
            flags.append(f"{k} not single ({v})")
    out["flag"] = "; ".join(flags) or None
    return out


def _paint(path: Path) -> dict[float, float]:
    from python.data.strip_writer import read_state_jsonl
    from python.debug.lane_targets import lane_center_is_valid, lane_x_from_vector

    out = {}
    for row in read_state_jsonl(path):
        tech = row.get("tech") or {}
        t = _ts(row)
        if t is None or tech.get("lane_gt") is None or not lane_center_is_valid(tech.get("lane_gt_valid")):
            continue
        x = lane_x_from_vector(tech["lane_gt"])
        if x is not None and math.isfinite(x):
            out[round(t, 6)] = float(x)
    return out


def summarize(log: Path, labels: Path | None = None) -> dict:
    from python.data.strip_writer import read_state_jsonl

    rows = read_state_jsonl(log)
    header = rows[0] if rows and rows[0].get("header") is True else None
    ticks = rows[1:] if header else rows
    labels = labels or log.with_name(log.stem + ".paint_v103.jsonl")
    paint = _paint(labels) if labels.is_file() else None
    ts = [t for t in (_ts(r) for r in ticks) if t is not None]
    span = (ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    masked: dict[str, int] = {}
    errs, why_unscored = [], []
    for n, row in enumerate(ticks):
        reasons = _masks(row)
        for r in reasons:
            masked[r] = masked.get(r, 0) + 1
        pred = row.get("model_lane_x")
        t = _ts(row)
        if paint is not None:
            gt = paint.get(round(t, 6)) if t is not None else None
        else:
            gt = row.get("gt_center_x")
        if not reasons and pred is None:
            reasons = ["no_pred"]
        if not reasons and gt is None:
            reasons = ["no_gt"]
        why_unscored.append(reasons)
        if not reasons:
            errs.append({"tick": n, "frame_t": t, "err_m": abs(float(pred) - float(gt)), "pred": float(pred), "gt": float(gt)})
    streaks, start = [], None
    for n, reasons in enumerate(why_unscored + [[]]):
        if reasons and start is None:
            start = n
        if not reasons and start is not None:
            why = sorted({r for rs in why_unscored[start:n] for r in rs})
            streaks.append({"start": start, "len": n - start, "why": why})
            start = None
    streaks.sort(key=lambda s: (-s["len"], s["start"]))
    a = np.array([e["err_m"] for e in errs], dtype=np.float64)
    nt = len(ticks)
    return {
        "log": log.name,
        "header": header is not None,
        "ticks": nt,
        "hz": (len(ts) - 1) / span if span > 0 else None,
        "model_sha256": (header or {}).get("model_sha256"),
        "stamp": _stamp((header or {}).get("onnx_meta")),
        "engaged_ticks": sum(1 for r in ticks if r.get("engaged")),
        "masked": masked,
        "masked_fraction": {k: v / nt for k, v in masked.items()} if nt else {},
        "lane_err_m": {
            "source": "paint" if paint is not None else "road_network",
            "n": int(a.size),
            "median": float(np.median(a)) if a.size else None,
            "p90": float(np.percentile(a, 90)) if a.size else None,
            "max": float(a.max()) if a.size else None,
        },
        "worst_ticks": sorted(errs, key=lambda e: (-e["err_m"], e["tick"]))[:TOP],
        "unscored_streaks": streaks[:TOP],
    }


def _fmt(v: Any) -> str:
    return "-" if v is None else f"{v:.3f}"


def render(rep: dict) -> str:
    st = rep["stamp"]
    e = rep["lane_err_m"]
    lines = [
        f"[GVD] shadow log {rep['log']}: {rep['ticks']} ticks at {_fmt(rep['hz'])} Hz"
        + ("" if rep["header"] else " (no header: model and stamp unknown)"),
        f"  model sha256={rep['model_sha256'] or 'unknown'} gvd_gt_origin={st.get('gvd_gt_origin') or 'unknown'} "
        f"gvd_gt_labeler={st.get('gvd_gt_labeler') or 'unknown'}",
    ]
    if st["flag"]:
        lines.append(f"  stamp: {st['flag']} (B4 would refuse this model)")
    lines.append(f"  engaged ticks: {rep['engaged_ticks']}")
    lines.append("  masked: " + (", ".join(
        f"{k} {v} ({rep['masked_fraction'][k]:.1%})" for k, v in sorted(rep["masked"].items(), key=lambda kv: -kv[1])) or "none"))
    src = "paint" if e["source"] == "paint" else "road-network gt_center_x, not paint"
    lines.append(f"  lane err 8 m ({src}): n={e['n']} median={_fmt(e['median'])} p90={_fmt(e['p90'])} max={_fmt(e['max'])} m")
    for s in rep["unscored_streaks"]:
        lines.append(f"  unscored streak: ticks {s['start']}..{s['start'] + s['len'] - 1} ({s['len']}) {','.join(s['why'])}")
    for w in rep["worst_ticks"]:
        lines.append(f"  worst: tick {w['tick']} frame_t={_fmt(w['frame_t'])} err={w['err_m']:.3f} pred={w['pred']:.3f} gt={w['gt']:.3f}")
    lines.append("  UNPROVEN (report-only)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("log", help="Shadow-run GT log (gt_<stamp>.jsonl).")
    p.add_argument("--labels", default="", help="Paint labels keyed on frame_t (default: <log stem>.paint_v103.jsonl).")
    p.add_argument("--json", default="", help="Write the report here.")
    args = p.parse_args(argv)
    rep = summarize(Path(args.log), Path(args.labels) if args.labels else None)
    print(render(rep), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
