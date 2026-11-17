#!/usr/bin/env python3
"""``bench_occ_sparse.py`` runs end to end at a tiny size and reports real fields."""
# gvd-ci: requires torch
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench_occ_sparse  # noqa: E402


def main() -> int:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bench_occ_sparse.main(["--long-side", "64", "--cams", "main,rear", "--queries", "256", "--iters", "1",
                               "--samples", "4", "--steps", "3", "--threads", "2"])
    text = buf.getvalue()
    js, summ = text[: text.rindex("SUMMARY")], text[text.rindex("SUMMARY"):]
    out = json.loads(js)
    assert "fps=" in summ and "infer_vram_mb=n/a (no CUDA)" in summ and "train_vram_mb=" in summ, summ
    assert abs(out["infer_fps"] - 1000.0 / out["infer_ms_median"]) < 1e-6
    assert out["device"] == "cpu" and out["cams"] == ["main", "rear"], out
    assert out["params"] > 10_000 and out["act_bytes_fp32_queries"] > 0
    # every stage is capped: top <= max_coarse, then <= max_points per stage (decode_unseen sweeps the whole top grid)
    bound = (out["max_coarse"] + 2 * out["max_points"]) / out["grid_cells"]
    assert out["infer_ms_median"] > 0 and 0 < out["infer_queried_ratio"] <= bound, (out["infer_queried_ratio"], bound)
    assert sum(out["infer_stage_queries"]) == out["infer_n_coarse"] + out["infer_n_fine"], out["infer_stage_queries"]
    assert out["shard_bytes_per_sample"] > 0 and out["train_compute_ms_median"] > 0 and out["train_peak_rss_mb"] > 0
    assert "infer_cuda_max_allocated_mb" not in out, "no CUDA number without CUDA"
    # peak_vram_mb is UNPROVEN until an nvidia-smi reading is passed; the CPU estimate never fills it
    assert out["peak_vram_mb"] is None and out["vram_status"] == "UNPROVEN", out
    assert out["vram_est_mb_cpu"] == out["mem_estimate_mb_fp32"]["total_mb"] > 0
    assert "peak_vram_mb=UNPROVEN" in summ and "vram_est_mb_cpu=" in summ and "(estimate)" in summ, summ
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bench_occ_sparse.main(["--long-side", "64", "--cams", "main,rear", "--queries", "64", "--iters", "1", "--warmup", "0",
                               "--infer-only", "--threads", "2", "--nvidia-smi-mb", "3071"])
    t2 = buf.getvalue()
    o2 = json.loads(t2[: t2.rindex("SUMMARY")])
    assert o2["peak_vram_mb"] == 3071.0 and o2["vram_status"] == "MEASURED" and o2["vram_est_mb_cpu"] == out["vram_est_mb_cpu"], o2
    assert "peak_vram_mb=3071(nvidia-smi)" in t2
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bench_occ_sparse.main(["--sweep", "--sweep-long", "48", "--queries", "64", "--iters", "1", "--warmup", "0", "--threads", "2"])
    lines = [ln for ln in buf.getvalue().splitlines() if ln.startswith("SUMMARY")]
    rows = json.loads(buf.getvalue().splitlines()[-1])
    assert len(lines) == 6 and len(rows) == 6, lines  # 2 nets x 3 presets x 1 size x fp32 only on CPU
    assert [len(r["cams"]) for r in rows] == [8, 3, 2, 8, 3, 2] and all("train_compute_ms_median" not in r for r in rows)
    cheap, dflt = rows[3]["params"], rows[0]["params"]
    assert cheap < dflt / 2, (cheap, dflt)
    print(f"bench sweep: 6 configs, cheap net {cheap} params vs {dflt} OK")
    print(f"bench: params {out['params']}, infer {out['infer_ms_median']:.1f} ms, queried {out['infer_queried_ratio']:.4f} OK")
    print("test_bench_occ_sparse: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
