#!/usr/bin/env python3
"""Opt-in benchmark for the sparse occupancy net (not part of the suite run).

Measures on THIS machine and prints one JSON object (labelled with host, CPU,
device): parameter count, activation bytes (forward hooks, summed outputs:
an upper bound for no-grad inference), inference latency at the live input
long side, peak CUDA memory when CUDA exists, a short training run's data
wait vs compute per step and peak RSS, and shard bytes per sample vs the
loose-PNG layout. Synthetic scene frames only (``tests/synthetic_occ.py``).

    python scripts/bench_occ_sparse.py --device cuda --long-side 640 --batch 4
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def activation_bytes(net, imgs, pts) -> int:
    import torch

    total = [0]

    def hook(_m, _i, out):
        for t in out if isinstance(out, (tuple, list)) else [out]:
            if isinstance(t, torch.Tensor):
                total[0] += t.numel() * t.element_size()

    hs = [m.register_forward_hook(hook) for m in net.modules() if len(list(m.children())) == 0]
    with torch.no_grad():
        net(imgs, pts)
    for h in hs:
        h.remove()
    return int(total[0])


def run(a: argparse.Namespace) -> dict:
    import numpy as np
    import torch
    import yaml

    from python.data.occ_shards import ShardWriter
    from python.perception.occ_geometry import rig_cams
    from python.perception.occ_sparse import SparseOccNet, live_cams
    from python.train.occ_labels import build_occ_labels
    from python.train.occ_train import train
    from synthetic_occ import SEM_CLASSES, render

    torch.set_num_threads(int(a.threads))
    dev = torch.device(a.device)
    cfg = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))
    cams = live_cams(cfg, a.long_side)
    if a.cams:
        cams = [c for c in cams if c.cam_id in set(a.cams.split(","))]
    out: dict = {"host": platform.node(), "cpu": _cpu_name(), "device": str(dev),
                 "cuda_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
                 "torch": torch.__version__, "long_side": a.long_side, "cams": [c.cam_id for c in cams]}
    net = SparseOccNet(cams).to(dev).eval()
    out["params"] = int(sum(p.numel() for p in net.parameters()))
    imgs = [torch.rand(1, 3, c.height, c.width, device=dev) for c in cams]
    pts = torch.rand(1, a.queries, 3, device=dev) * torch.tensor([51.2, 64.0, 6.4], device=dev) + torch.tensor([-25.6, -12.8, -2.0], device=dev)
    out["act_bytes_fp32_queries"] = activation_bytes(net, imgs, pts)
    out["queries"] = a.queries
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    use_half = a.fp16 and dev.type == "cuda"
    times, last = [], None
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_half):
        for i in range(a.warmup + a.iters):
            t = time.perf_counter()
            last = net.predict(imgs)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            if i >= a.warmup:
                times.append((time.perf_counter() - t) * 1e3)
    out["infer_fp16"] = bool(use_half)
    out["infer_ms_median"] = float(np.median(times))
    out["infer_ms_p90"] = float(np.percentile(times, 90))
    out["infer_n_coarse"], out["infer_n_fine"] = int(last.n_coarse), int(last.n_fine)
    out["infer_queried_ratio"] = float(last.queried_ratio())
    if dev.type == "cuda":
        out["infer_cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 20
    # Short training run on synthetic shards.
    full = {c.cam_id: c for c in rig_cams(cfg)}
    with tempfile.TemporaryDirectory() as d:
        sh = Path(d) / "sh"
        t0 = time.perf_counter()
        loose_bytes = 0
        with ShardWriter(sh) as w:
            for i in range(a.samples):
                imgs_np, planes = {}, {}
                for c in cams:
                    cam = full[c.cam_id]
                    dep, sem, bgr = render(cam, box_lo=(-1.0 + 0.3 * i, 10.0, 0.2), box_hi=(1.0 + 0.3 * i, 14.4, 1.8))
                    imgs_np[c.cam_id], planes[c.cam_id] = bgr, (dep, sem)
                    import cv2

                    for arr in (bgr, dep, sem):
                        loose_bytes += len(cv2.imencode(".png", arr, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])[1])
                lab = build_occ_labels(planes, full, sem_classes=SEM_CLASSES, rng=np.random.default_rng(i))
                w.write(f"s{i:05d}", imgs_np, lab.as_arrays(), {"recovery": i % 4 == 0, "t": float(i)})
        out["shard_build_s_per_sample"] = (time.perf_counter() - t0) / a.samples
        shard_bytes = sum(p.stat().st_size for p in w.paths)
        out["shard_bytes_per_sample"] = shard_bytes / a.samples
        out["loose_png_bytes_per_sample_rgb_depth_sem"] = loose_bytes / a.samples
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        res = train(sh, cams, steps=a.steps, batch_size=a.batch, device=a.device, amp=a.amp, workers=a.workers)
        h = res["history"][2:] or res["history"]
        out.update({
            "train_batch": a.batch, "train_amp": res["amp"], "train_steps": a.steps, "workers": a.workers,
            "train_wait_ms_median": float(np.median([x["wait_ms"] for x in h])),
            "train_compute_ms_median": float(np.median([x["compute_ms"] for x in h])),
            "train_peak_rss_mb": res["peak_rss_mb"],
            "train_loss_first_last": [res["history"][0]["loss"], res["history"][-1]["loss"]],
        })
        if "cuda_max_allocated_mb" in res:
            out["train_cuda_max_allocated_mb"] = res["cuda_max_allocated_mb"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--long-side", type=int, default=640)
    ap.add_argument("--cams", default="")
    ap.add_argument("--queries", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--fp16", action="store_true", help="fp16 autocast for inference (CUDA only)")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--amp", default="auto", choices=("auto", "fp16", "off"))
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    a = ap.parse_args(argv)
    print(json.dumps(run(a), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
