#!/usr/bin/env python3
"""``shadow_log_summary.py``: a compact, ungraded report of one shadow-run log.

B1's header and per-row fields (``frame_t``, ``mask``) are not in
``gt_logger`` yet, so the synthetic log here follows the shadow-run spec's
documented fields; see the script docstring for the assumed key names.

* ticks and Hz from ``frame_t``; the header line is not a tick.
* stamp: the model's ``gvd_gt_origin`` / ``gvd_gt_labeler`` are printed; a
  missing, ``unknown`` or comma-joined (mixed) value is flagged, never graded.
* masks: masked fraction by reason over all ticks.
* error: |model_lane_x - paint center at 8 m| over unmasked ticks with a valid
  paint label (joined on ``frame_t``): n, median, p90, max; worst ticks ranked.
* streaks: the longest runs of unscored ticks, with why.
* no labels: falls back to the log's road-network ``gt_center_x`` and says so.
* report-only: no pass/fail word, exit 0; the JSON report is written.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

N = 30
MASK = {5: "junction", 6: "junction", 7: "junction", 8: "junction", 20: "respawn", 21: "settle"}
NO_PAINT = {12}
PRED = {i: 0.05 * i for i in range(N)}
TRUTH = 0.9


def _lane(center: float) -> list[float]:
    vals = []
    for _ in range(4):
        vals += [center, center - 1.8, center + 1.8]
    return vals + [0.0, 1.0]


def _log(tmp: Path, meta: dict | None, name: str = "gt_x.jsonl", header: bool = True) -> Path:
    lines = []
    if header:
        lines.append({"header": True, "map": "west_coast_usa", "model": "m.onnx", "model_sha256": "ab" * 32, "onnx_meta": meta or {}})
    for i in range(N):
        lines.append({
            "t": 100.0 + i * 0.1, "frame_t": 5000.0 + i * 0.1, "model_lane_x": PRED[i], "engaged": False,
            "gt_center_x": TRUTH + 1.0, "mask": MASK.get(i),
        })
    log = tmp / name
    log.write_text("\n".join(json.dumps(r) for r in lines) + "\n", encoding="utf-8")
    labels = []
    for i in range(N):
        valid = [0.0] * 14 if i in NO_PAINT else [1.0] * 14
        labels.append({"frame_t": 5000.0 + i * 0.1, "tech": {"lane_gt": _lane(TRUTH), "lane_gt_valid": valid, "gt_origin": "body_centreline"}})
    (tmp / (log.stem + ".paint_v103.jsonl")).write_text("\n".join(json.dumps(r) for r in labels) + "\n", encoding="utf-8")
    return log


def _run(args: list[str]) -> tuple[int, str, dict]:
    from shadow_log_summary import main

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = main(args)
    js = [a for a in args if a.endswith(".json")]
    return rc, out.getvalue(), json.loads(Path(js[0]).read_text(encoding="utf-8")) if js else {}


def check_report(tmp: Path) -> None:
    log = _log(tmp, {"gvd_gt_origin": "body_centreline", "gvd_gt_labeler": "painted_v103"})
    rc, text, rep = _run([str(log), "--json", str(tmp / "r.json")])
    assert rc == 0, text
    assert rep["ticks"] == N and abs(rep["hz"] - 10.0) < 1e-6, rep
    assert rep["model_sha256"] == "ab" * 32 and rep["stamp"]["gvd_gt_origin"] == "body_centreline"
    assert rep["stamp"]["flag"] is None, rep["stamp"]
    assert rep["masked"] == {"junction": 4, "respawn": 1, "settle": 1}, rep["masked"]
    assert abs(rep["masked_fraction"]["junction"] - 4 / N) < 1e-9, rep["masked_fraction"]
    scored = [i for i in range(N) if i not in MASK and i not in NO_PAINT]
    errs = np.array([abs(PRED[i] - TRUTH) for i in scored])
    e = rep["lane_err_m"]
    assert e["source"] == "paint" and e["n"] == len(scored), e
    assert abs(e["median"] - float(np.median(errs))) < 1e-9 and abs(e["p90"] - float(np.percentile(errs, 90))) < 1e-9
    assert abs(e["max"] - float(errs.max())) < 1e-9, e
    worst = rep["worst_ticks"]
    assert [w["tick"] for w in worst[:2]] == [0, 1], worst
    assert rep["unscored_streaks"][0] == {"start": 5, "len": 4, "why": ["junction"]}, rep["unscored_streaks"]
    assert any(s["start"] == 20 and s["len"] == 2 for s in rep["unscored_streaks"]), rep["unscored_streaks"]
    assert "UNPROVEN (report-only)" in text and "PASS" not in text and "FAIL" not in text, text
    print(f"report: {N} ticks at 10 Hz; masks by reason; paint error n={e['n']} median/p90/max; worst and streaks OK")


def check_stamp(tmp: Path) -> None:
    for meta, want in [
        ({}, "unknown"),
        ({"gvd_gt_origin": "unknown", "gvd_gt_labeler": "painted_v103"}, "unknown"),
        ({"gvd_gt_origin": "body_centreline,ref_node", "gvd_gt_labeler": "painted_v103"}, "not single"),
    ]:
        sub = tmp / f"s{len(want)}{len(meta)}"
        sub.mkdir()
        rc, text, rep = _run([str(_log(sub, meta)), "--json", str(sub / "r.json")])
        assert rc == 0 and rep["stamp"]["flag"] and want in rep["stamp"]["flag"], (meta, rep["stamp"])
        assert "B4 would refuse" in text, text
    print("stamp: missing, unknown and mixed origin are flagged (B4 would refuse), still exit 0 OK")


def check_fallback(tmp: Path) -> None:
    sub = tmp / "fb"
    sub.mkdir()
    log = _log(sub, {"gvd_gt_origin": "o", "gvd_gt_labeler": "l"})
    (sub / (log.stem + ".paint_v103.jsonl")).unlink()
    rc, text, rep = _run([str(log), "--json", str(sub / "r.json")])
    e = rep["lane_err_m"]
    assert e["source"] == "road_network" and e["n"] == N - len(MASK), e
    assert "not paint" in text, text
    rc, text, rep = _run([str(_log(sub, None, "nohdr.jsonl", header=False)), "--json", str(sub / "r2.json")])
    assert rep["ticks"] == N and rep["header"] is False and "no header" in text, text
    print("fallback: road-network gt_center_x labelled not paint; a missing header is said OK")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        check_report(t)
        check_stamp(t)
        check_fallback(t)
    print("shadow log summary OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
