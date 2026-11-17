"""Front-cluster trade-off (SYNTHETIC, CPU): main+wide+narrow vs front_main_hi (main+wide, main at higher res).

For each rig: input resolution, params, CPU ESTIMATE of inference memory at full res
(batch 1 and offline batch), and occupancy IoU on held-out synthetic clips (start
positions off the training grid) after the same short training at a scaled-down
resolution (128 px base; main at 192 px for front_main_hi, the same 1.5x ratio as
960/640). Prints ``ps`` lines with CPU timings. No FPS / Hz claims; GPU UNPROVEN.

    PYTHONPATH=. python scripts/occ_rig_tradeoff.py [steps]
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from occ_net_fixtures import FULL_CAMS, frame, small_profile, small_rig, train_joint  # noqa: E402

import torch  # noqa: E402

from python.occ.model import OccSparseNet, memory_estimate, param_count, profile_preset  # noqa: E402
from python.occ.runtime import OccRuntime  # noqa: E402


def _ps() -> str:
    import os
    return subprocess.run(["ps", "-o", "pid,%cpu,%mem,rss,etime,time,cmd", "-p", str(os.getpid())], capture_output=True, text=True).stdout.strip()


def heldout_iou(model, cams, starts=(0.37, 4.61), ticks=6, step=0.8):
    from python.occ.grid import DEFAULT_GRID as G
    nx, ny, nz = G.shape
    flat = lambda i: (np.asarray(i, np.int64)[:, 0] * ny + np.asarray(i, np.int64)[:, 1]) * nz + np.asarray(i, np.int64)[:, 2]
    tp = fp = fn = 0
    for y0 in starts:
        rt = OccRuntime(model, cams)
        for k in range(ticks):
            rgb, lab = frame(cams, ego_y=y0 + step * k)
            h = rt.perceive(0.1 * k, {c: (rgb[c], 0.1 * k) for c in cams})
            pred = flat(h["cells"]["idx"]) if len(h["cells"]["idx"]) else np.zeros(0, np.int64)
            occ, free = flat(lab["occ_idx"]), flat(lab["free_idx"])
            t_ = np.intersect1d(pred, occ).size
            tp += t_
            fp += np.intersect1d(pred, free).size
            fn += occ.size - t_
    return tp / max(1, tp + fp + fn)


def main(argv=None) -> int:
    steps = int((argv or sys.argv[1:] or ["300"])[0])
    torch.set_num_threads(1)
    rows = []
    for name, ids, over in (("main_wide_narrow", ("main", "wide", "narrow"), None), ("front_main_hi", ("main", "wide"), {"main": 192})):
        full = profile_preset(name)
        torch.manual_seed(0)
        fm = OccSparseNet(profile=full).eval()
        est = memory_estimate(fm, FULL_CAMS)
        cams = small_rig(ids, 128, over)
        torch.manual_seed(0)
        m = OccSparseNet(profile=small_profile(ids, 128, k_max=32, res_overrides=tuple((over or {}).items())))
        t0 = time.time()
        train_joint(m, cams, steps=steps)
        tt = time.time() - t0
        iou = heldout_iou(m, cams)
        res = {c: full.long_side(c) for c in full.cams}
        rows.append((name, res, param_count(fm), est["infer_gb"], est["infer_gb_batch"], iou, tt))
        print(f"{name}: full-res long sides {res}; params {param_count(fm):,}; CPU ESTIMATE infer {est['infer_gb']:.3f} GB (batch 1), "
              f"{est['infer_gb_batch']:.3f} GB (offline batch {int(est['infer_batch'])}); SYNTHETIC held-out occ IoU {iou:.3f} "
              f"(scaled 128 px, {steps} steps, {tt:.0f} s CPU)\n{_ps()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
