#!/usr/bin/env python3
"""Bench the Trainer 2.0 48x64 depth decoder against the live budget (CPU / box numbers).

Measures, for the ``tiny`` and ``small`` live presets on the 8-camera strip at
scale 1 (main crop 192x256):

* CPU ms per frame (batch 1, eval, no_grad, median of ``--reps``) of the full
  ``SceneNet.forward_batch`` with the 12x16 head vs with ``depth_hires``, and of
  the decoder alone;
* decoder parameters and fp32 weight + activation bytes (exact tensor sizes);
* peak CUDA allocation when a GPU is present (``torch.cuda.max_memory_allocated``),
  else ``cuda: none`` and the byte count stands as the VRAM estimate.

Budgets it reports against (not enforced here): live frame 100 ms (10 Hz,
``docs/research/gpu_tiers.md``), new heads <= 2 ms total
(``docs/SPEC_intersections.md``), live inference VRAM <= 4 GB
(``config/hardware.yaml``). CPU ms on the box are an upper bound for the
GPU path, and a loaded box inflates them; ``load_avg`` is printed with them.

  python scripts/bench_depth_hires.py [--reps 20] [--threads 4] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LIVE_FRAME_MS = 100.0
NEW_HEADS_MS = 2.0
LIVE_VRAM_GB = 4.0


def _min_median_ms(fn, reps: int) -> tuple[float, float]:
    fn()  # warm-up
    times = []
    for _ in range(max(1, int(reps))):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(min(times)), float(statistics.median(times))


def bench(preset: str, reps: int) -> dict:
    import torch

    from python.train.depth_hires import stem_pyramid
    from python.train.model_size import scene_net_kwargs
    from python.train.scene_net import SceneNet

    out: dict = {"preset": preset}
    nets = {}
    for hires in (False, True):
        torch.manual_seed(0)
        nets[hires] = SceneNet(aux_sensors=True, depth_hires=hires, **scene_net_kwargs(preset)).eval()
    img = torch.rand(1, 3, nets[True].canvas_h, nets[True].canvas_w)
    main = next(s for s in nets[True].sectors if str(s.cam_id) == "main")
    crop = img[:, :, :, main.x0 : main.x1]
    with torch.no_grad():
        # Interleaved A/B so box load drifts hit both arms alike.
        runs = {hires: (lambda n=net, s=net.initial_state(batch=1): n.forward_batch(img, 0.1, s)) for hires, net in nets.items()}
        times: dict[bool, list[float]] = {False: [], True: []}
        for fn in runs.values():
            fn()
        for _ in range(max(1, int(reps))):
            for hires, fn in runs.items():
                t0 = time.perf_counter()
                fn()
                times[hires].append((time.perf_counter() - t0) * 1000.0)
        out["net_ms_12x16"] = float(statistics.median(times[False]))
        out["net_ms_hires"] = float(statistics.median(times[True]))
        # Min is the contention-robust figure on a shared box; median is reported beside it.
        out["net_min_ms_12x16"] = float(min(times[False]))
        out["net_min_ms_hires"] = float(min(times[True]))
        head = nets[True].depth_hires_head
        pyramid = stem_pyramid(nets[True].stem, crop)
        out["decoder_min_ms"], out["decoder_ms"] = _min_median_ms(lambda: head(pyramid), reps * 3)
        depth, far = head(pyramid)
        # Decoder intermediates (fp32 elements): a lateral and a merged map at each of the three
        # FPN levels (width channels), the smoothed map, and the 2-channel output (+ resize).
        width = int(head.smooth.out_channels)
        level = [int(m.shape[-2] * m.shape[-1]) for m in pyramid[-3:]]
        inter = width * (2 * sum(level) + level[0]) + 2 * 2 * int(depth.shape[-2] * depth.shape[-1])
    params = sum(int(p.numel()) for p in head.parameters())
    out["decoder_params"] = params
    out["decoder_weight_bytes"] = params * 4
    out["decoder_activation_bytes"] = inter * 4
    out["net_params_12x16"] = sum(int(p.numel()) for p in nets[False].parameters())
    out["net_params_hires"] = sum(int(p.numel()) for p in nets[True].parameters())
    out["output_shape"] = [int(v) for v in depth.shape]
    out["added_ms"] = out["net_ms_hires"] - out["net_ms_12x16"]
    out["added_min_ms"] = out["net_min_ms_hires"] - out["net_min_ms_12x16"]
    out["est_added_vram_gb"] = (out["decoder_weight_bytes"] + out["decoder_activation_bytes"]) / 1024**3
    from python.train.model_size import PRESETS

    out["preset_est_vram_gb"] = float(PRESETS[preset].est_vram_gb)
    out["est_total_vram_gb"] = out["preset_est_vram_gb"] + out["est_added_vram_gb"]
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        gpu_net = nets[True].to(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        with torch.no_grad():
            gpu_net.forward_batch(img.to(dev), 0.1, gpu_net.initial_state(device=dev, batch=1))
        torch.cuda.synchronize(dev)
        out["cuda_peak_gb"] = torch.cuda.max_memory_allocated(dev) / 1024**3
        out["cuda"] = torch.cuda.get_device_name(dev)
    else:
        out["cuda"] = "none"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--presets", default="tiny,small")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    import torch

    torch.set_num_threads(max(1, int(args.threads)))
    rows = [bench(p.strip(), args.reps) for p in args.presets.split(",") if p.strip()]
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")
    for r in rows:
        print(
            f"{r['preset']}: out {r['output_shape']} decoder {r['decoder_params']} params "
            f"({r['decoder_weight_bytes'] / 1024:.0f} KiB weights, {r['decoder_activation_bytes'] / 1024:.0f} KiB activations); "
            f"CPU net min {r['net_min_ms_12x16']:.2f} -> {r['net_min_ms_hires']:.2f} ms ({r['added_min_ms']:+.2f}), "
            f"median {r['net_ms_12x16']:.2f} -> {r['net_ms_hires']:.2f} ({r['added_ms']:+.2f}); "
            f"decoder alone min {r['decoder_min_ms']:.2f} / median {r['decoder_ms']:.2f} ms; "
            f"live frame {LIVE_FRAME_MS:.0f} ms, new heads {NEW_HEADS_MS:.0f} ms; "
            f"VRAM est {r['preset_est_vram_gb']:.2f} + {r['est_added_vram_gb'] * 1024:.2f} MB = {r['est_total_vram_gb']:.3f} GB <= {LIVE_VRAM_GB:.0f} GB; "
            f"cuda {r['cuda']}" + (f" peak {r['cuda_peak_gb']:.3f} GB" if "cuda_peak_gb" in r else "")
        )
    print(f"threads={int(args.threads)} reps={int(args.reps)} load_avg_1m={load:.1f}")
    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "threads": int(args.threads), "load_avg_1m": load}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
