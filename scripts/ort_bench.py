#!/usr/bin/env python3
"""SceneNet PyTorch vs ONNX Runtime FP32: export, parity, EP/op report, latency bench -> JSON.

Offline tool (docs/ort_bench.md). Never imported by the live supervisor.

    python scripts/ort_bench.py --ep cpu --out bench_cpu.json
    python scripts/ort_bench.py --ep cuda --out bench_cuda.json            # IQU45, onnxruntime-gpu
    python scripts/ort_bench.py --ep cuda --ort-only --onnx X.onnx --out ort_only.json

Full mode exports a seeded default SceneNet (opset 17) through
``scene_net.export_scene_onnx``, checks parity over a recurrent rollout at the
stated tolerance (``ort_bench.PARITY_ATOL`` / ``PARITY_RTOL``) and exits 1 if it
fails. ``--ort-only`` blocks the torch import so RSS/VRAM are the deploy path's.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--ep", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", type=Path, required=True, help="JSON report path.")
    p.add_argument("--workdir", type=Path, default=None, help="Keep the ONNX / optimized ONNX / profile here.")
    p.add_argument("--ticks", type=int, default=6, help="Recurrent rollout length for parity.")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--threads", type=int, default=0, help="intra-op threads for ORT and torch (0 = default).")
    p.add_argument("--ort-only", action="store_true", help="No torch: op report + bench of --onnx.")
    p.add_argument("--onnx", type=Path, default=None, help="Existing ONNX for --ort-only.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ort_only:
        if args.onnx is None:
            print("ort_bench: --ort-only needs --onnx", file=sys.stderr)
            return 2
        sys.modules["torch"] = None  # type: ignore[assignment]  # import torch -> ImportError
    from python.train import ort_bench

    tmp = None
    workdir = args.workdir
    if workdir is None:
        tmp = tempfile.TemporaryDirectory(prefix="gvd_ort_bench_")
        workdir = Path(tmp.name)
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        if args.ort_only:
            report = ort_bench.run_ort_only(args.onnx, ep=args.ep, warmup=args.warmup, iters=args.iters,
                                            threads=args.threads, workdir=workdir)
            ok = True
        else:
            report = ort_bench.run_all(workdir, ep=args.ep, ticks=args.ticks, warmup=args.warmup,
                                       iters=args.iters, seed=args.seed, threads=args.threads)
            ok = bool(report["parity"]["ok"])
    finally:
        if tmp is not None:
            tmp.cleanup()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    b = report["bench"]
    print(f"[GVD] ort_bench ep={args.ep} providers={b['session_providers']} "
          f"ort_p50={b['ort']['p50_ms']:.2f}ms p90={b['ort']['p90_ms']:.2f}ms p99={b['ort']['p99_ms']:.2f}ms "
          f"memcpy={report['ops']['memcpy_total']} fallback={sorted(report['ops']['fallback_ops'])} "
          f"parity={'n/a' if args.ort_only else report['parity']['ok']} out={args.out}", flush=True)
    if not ok:
        print("[GVD] PARITY FAILED: " + "; ".join(report["parity"]["failures"][:10]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
