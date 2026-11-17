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
    out = json.loads(buf.getvalue())
    assert out["device"] == "cpu" and out["cams"] == ["main", "rear"], out
    assert out["params"] > 10_000 and out["act_bytes_fp32_queries"] > 0
    assert out["infer_ms_median"] > 0 and 0 < out["infer_queried_ratio"] < 0.25, out
    assert out["shard_bytes_per_sample"] > 0 and out["train_compute_ms_median"] > 0 and out["train_peak_rss_mb"] > 0
    assert "infer_cuda_max_allocated_mb" not in out, "no CUDA number without CUDA"
    print(f"bench: params {out['params']}, infer {out['infer_ms_median']:.1f} ms, queried {out['infer_queried_ratio']:.4f} OK")
    print("test_bench_occ_sparse: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
