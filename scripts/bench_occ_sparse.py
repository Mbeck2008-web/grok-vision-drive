#!/usr/bin/env python3
"""Opt-in benchmark for the sparse occupancy net (not part of the suite run).

Measures on THIS machine and prints one JSON object (labelled with host, CPU,
device): parameter count, activation bytes (forward hooks, summed outputs:
an upper bound for no-grad inference), inference latency at the live input
long side, peak CUDA memory when CUDA exists, a short training run's data
wait vs compute per step and peak RSS, and shard bytes per sample vs the
loose-PNG layout. Synthetic scene frames only (``tests/synthetic_occ.py``).

    python scripts/bench_occ_sparse.py --device cuda --long-side 640 --batch 4

Desktop-IQU45 (GTX 1080 Ti; real VRAM at ~640 px fp16 inference and the AMP
training peak; run with BeamNG idle or note what else holds the GPU):

    py -3 scripts/bench_occ_sparse.py --device cuda --long-side 640 --fp16 --iters 20 --warmup 3 --samples 16 --steps 20 --batch 4 --amp fp16 --workers 4
    py -3 scripts/bench_occ_sparse.py --device cuda --sweep --iters 20 --warmup 3

The sweep is inference only: net (default/cheap) x cameras (all8 / front3 =
main+wide+narrow / min = main+wide) x input long side (640/480/320) x
fp32/fp16, one ``SUMMARY`` line each with FPS next to VRAM. Pascal fp16 math is
~1/64 of fp32: expect fp16 to save memory, not time. Michael's rule: drop the
input resolution before giving up same-tick sync of the cameras.

``infer_cuda_max_allocated_mb`` is the fp16 inference peak (untrained net,
so the coarse-to-fine caps are hit: the worst case), ``train_cuda_max_allocated_mb``
the AMP training peak. Repeat with ``--amp off`` for the fp32 (Pascal default) peak.
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


def _loadavg() -> list[float] | None:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):  # Windows
        return None


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
    from python.perception.occ_sparse import CAM_PRESETS, PROFILES, SparseOccNet, live_cams, memory_estimate, profile_cams

    if a.profile:
        pr = PROFILES[a.profile]
        a.net, a.long_side, a.cam_preset = pr["config"], pr["long_side"], pr["cams"]
        a.fp16 = a.fp16 and pr["compute"] != "fp32"
        a.max_points = pr["max_points"]
        a.max_coarse = pr["max_coarse"]
    from python.train.occ_labels import build_occ_labels
    from python.train.occ_train import train
    from synthetic_occ import SEM_CLASSES, render

    torch.set_num_threads(int(a.threads))
    dev = torch.device(a.device)
    cfg = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))
    cams = live_cams(cfg, a.long_side)
    if a.profile:
        cams = profile_cams(cfg, a.profile)
    elif a.cam_preset:
        cams = [c for c in cams if c.cam_id in set(CAM_PRESETS[a.cam_preset])]
    if a.cams:
        cams = [c for c in cams if c.cam_id in set(a.cams.split(","))]
    out: dict = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "loadavg": _loadavg(),
                 "host": platform.node(), "cpu": _cpu_name(), "device": str(dev),
                 "cuda_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
                 "torch": torch.__version__, "long_side": a.long_side, "cams": [c.cam_id for c in cams],
                 "net_config": a.net, "cam_preset": a.cam_preset or None, "profile": a.profile or None}
    net = SparseOccNet(cams, config=a.net).to(dev).eval()
    out["params"] = int(sum(p.numel() for p in net.parameters()))
    out["cam_res"] = {c.cam_id: [c.width, c.height] for c in cams}
    out["mem_estimate_mb_fp32"] = memory_estimate(cams, config=a.net, max_points=a.max_points)
    out["max_points"], out["max_coarse"] = a.max_points, a.max_coarse  # arithmetic ESTIMATE, not measured
    # peak_vram_mb is only ever an nvidia-smi reading (IQU45); the arithmetic number stays in vram_est_mb_cpu
    out["vram_est_mb_cpu"] = float(out["mem_estimate_mb_fp32"]["total_mb"])
    smi = getattr(a, "nvidia_smi_mb", None)
    out["peak_vram_mb"] = None if smi is None else float(smi)
    out["vram_status"] = "UNPROVEN" if smi is None else "MEASURED"
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
            last = net.predict(imgs, max_points=a.max_points, max_coarse=a.max_coarse)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            if i >= a.warmup:
                times.append((time.perf_counter() - t) * 1e3)
    out["infer_fp16"] = bool(use_half)
    out["infer_ms_median"] = float(np.median(times))
    out["infer_ms_p90"] = float(np.percentile(times, 90))
    out["infer_fps"] = 1000.0 / out["infer_ms_median"]
    out["infer_n_coarse"], out["infer_n_fine"] = int(last.n_coarse), int(last.n_fine)
    out["infer_queried_ratio"] = float(last.queried_ratio())
    out["infer_stage_queries"] = list(last.stage_queries)
    from python.perception.occ_sparse import visible_mask

    fine = net.grid.centers_of_keys(np.arange(net.grid.dense_cells))
    out["frustum_fine_cells"] = int(visible_mask(cams, fine).sum())
    out["grid_cells"] = int(net.grid.dense_cells)
    if dev.type == "cuda":
        out["infer_cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 20
    if a.infer_only:
        out["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        return out
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
        res = train(sh, cams, steps=a.steps, batch_size=a.batch, device=a.device, amp=a.amp, workers=a.workers,
                    net=SparseOccNet(cams, config=a.net), cam_drop=0.3)
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
    out["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    out["loadavg_end"] = _loadavg()
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info(dev)
        out["cuda_mem_free_total_mb_end"] = [free / 2 ** 20, total / 2 ** 20]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--long-side", type=int, default=640)
    ap.add_argument("--cams", default="")
    ap.add_argument("--cam-preset", default="", choices=("", "min", "front3", "all8"),
                    help="min = main+wide, front3 = main+wide+narrow")
    ap.add_argument("--net", default="default", choices=("default", "cheap"))
    ap.add_argument("--profile", default="", choices=("", "1080ti", "1080ti_cheap", "1080ti_min"),
                    help="sets --net, --long-side, --cam-preset; fp32 compute")
    ap.add_argument("--queries", type=int, default=8192)
    ap.add_argument("--max-points", type=int, default=65536, help="per-stage top-k cap in predict")
    ap.add_argument("--max-coarse", type=int, default=2048, help="absolute cap on the top coarse stage")
    ap.add_argument("--nvidia-smi-mb", type=float, default=None,
                    help="peak used MB read off nvidia-smi on Desktop-IQU45 with BeamNG live; only this fills peak_vram_mb")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--fp16", action="store_true", help="fp16 autocast for inference (CUDA only)")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--amp", default="auto", choices=("auto", "fp16", "off"))
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--infer-only", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="inference only over net x cam preset x long side x fp16/fp32; one line each")
    ap.add_argument("--sweep-long", default="640,480,320", help="input long sides for --sweep")
    a = ap.parse_args(argv)
    if not a.sweep:
        res = run(a)
        print(json.dumps(res, indent=1))
        print(summary(res))
        return 0
    rows = []
    for net in ("default", "cheap"):
        for preset in ("all8", "front3", "min"):
            for long_side in [int(x) for x in a.sweep_long.split(",") if x]:
                for fp16 in ((False, True) if a.device.startswith("cuda") else (False,)):
                    b = argparse.Namespace(**vars(a))
                    b.net, b.cam_preset, b.long_side, b.fp16, b.infer_only, b.cams = net, preset, long_side, fp16, True, ""
                    b.profile = ""
                    b.max_points, b.max_coarse = a.max_points, a.max_coarse
                    res = run(b)
                    rows.append(res)
                    print(summary(res), flush=True)
    print(json.dumps(rows))
    return 0


def summary(res: dict) -> str:
    """One line: FPS and VRAM next to each other (Pascal: fp16 saves memory, not time)."""
    vram = res.get("infer_cuda_max_allocated_mb")
    tr = res.get("train_cuda_max_allocated_mb")
    pk = "UNPROVEN" if res.get("peak_vram_mb") is None else f"{res['peak_vram_mb']:.0f}(nvidia-smi)"
    return (f"SUMMARY host={res['host']} device={res['device']} net={res.get('net_config')} cams={len(res['cams'])}"
            f"({','.join(res['cams'])}) long={res['long_side']} fp16={res['infer_fp16']} "
            f"infer_ms={res['infer_ms_median']:.1f} fps={res['infer_fps']:.2f} "
            f"infer_vram_mb={'n/a (no CUDA)' if vram is None else f'{vram:.0f}'} "
            f"stages={res['infer_stage_queries']} frustum={res['frustum_fine_cells']} grid={res['grid_cells']} "
            f"queried/dense={res['infer_queried_ratio']:.3f} vram_est_mb_cpu={res['vram_est_mb_cpu']:.0f}(estimate) "
            f"peak_vram_mb={'UNPROVEN' if res.get('peak_vram_mb') is None else f'{res[chr(112) + "eak_vram_mb"]:.0f}(nvidia-smi)'}"
            + ("" if "train_compute_ms_median" not in res else
               f" train_vram_mb={'n/a (no CUDA)' if tr is None else f'{tr:.0f}'} train_amp={res['train_amp']}"
               f" train_wait_ms={res['train_wait_ms_median']:.0f} train_compute_ms={res['train_compute_ms_median']:.0f}"))


if __name__ == "__main__":
    raise SystemExit(main())
