#!/usr/bin/env python3
"""Learned lane head vs Hough on a held-out set of .npz frames (CPU).

    py -3 scripts/eval_learned_lanes.py --npz-dir <held-out dir> [--onnx <scene onnx>] [--json out.json]

Sample format: python/debug/lane_head_eval.py. The learned head is scored through
the live LearnedLaneSelector (paint gate included; a fallback tick counts as no
prediction). Hough is scored raw (lanes_from_view, today's default) and gated
(the flag-on fallback). ``learned_beats_hough`` needs both verdicts. Exit 0 always on a completed eval; the verdict is the result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from python.debug.lane_head_eval import score, verdict  # noqa: E402
from python.perception.lane_select import LearnedLaneSelector  # noqa: E402
from python.perception.lanes import lanes_from_view  # noqa: E402


def load_samples(npz_dir: Path) -> list[dict]:
    out = []
    for path in sorted(Path(npz_dir).glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            sample = {k: z[k] for k in z.files}
        sample["has_paint"] = bool(np.asarray(sample.get("has_paint", True)).reshape(-1)[0])
        sample["_name"] = path.name
        out.append(sample)
    return out


def run(samples: list[dict], selector: LearnedLaneSelector) -> dict:
    def learned(s):
        sel = selector.select(s.get("main"), s.get("stitch"), strip=s.get("strip"), dt_s=0.05)
        # A fallback tick is not a learned prediction: score it as no output.
        return sel.result if sel.source == "learned" else None

    def hough_raw(s):
        return lanes_from_view(s.get("main"), s.get("stitch"))

    gated = LearnedLaneSelector(None, runtime_factory=lambda _p: None)  # Hough + paint gate

    def hough_gated(s):
        return gated.select(s.get("main"), s.get("stitch")).result

    m_learned = score(samples, learned)
    m_hough = score(samples, hough_raw)
    m_gated = score(samples, hough_gated)
    v_raw = verdict(m_learned, m_hough)
    v_gated = verdict(m_learned, m_gated)
    return {
        "n_samples": len(samples),
        "learned": m_learned,
        "hough": m_hough,
        "hough_gated": m_gated,
        "verdict_vs_hough": v_raw,
        "verdict_vs_hough_gated": v_gated,
        # Flip the --learned-lanes default only when this is true on real held-out data.
        "learned_beats_hough": bool(v_raw["learned_beats_hough"] and v_gated["learned_beats_hough"]),
        "model": selector.describe(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--onnx", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    samples = load_samples(Path(args.npz_dir))
    report = run(samples, LearnedLaneSelector(args.onnx or None))
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
