"""One timed inference tick: strip -> SceneNet (+ sparse occ) -> ``occ_out`` v2 for the viz.

Queries = the stored sparse set plus a fixed ground lattice over the whole region (``LATTICE_STRIDE``),
so occluded / never-seen road still decodes (position encoding + scene feature + memory gather;
hit fraction 0 where nothing is stored). ``observed`` = a fresh camera ray wrote that voxel this tick.
Timings are wall clock per stage (CUDA-synchronised on a GPU). Peak VRAM comes from
``torch.cuda.max_memory_allocated`` and is None on CPU: never estimated here.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from python.viz.occ_out import MAX_OCC, OccOutV2, build_occ_out
from python.viz.occ_scene import GRID_SHAPE

LATTICE_STRIDE = 4  # 16 x 28 = 448 ground cells (z layer 2 = 0..0.5 m), grid-size independent cap below
LATTICE_CAP = 512


def unseen_lattice(stride: int = LATTICE_STRIDE) -> np.ndarray:
    ix, iy = np.meshgrid(np.arange(stride // 2, GRID_SHAPE[0], stride), np.arange(stride // 2, GRID_SHAPE[1], stride), indexing="ij")
    ids = np.ravel_multi_index((ix.ravel(), iy.ravel(), np.full(ix.size, 2)), GRID_SHAPE)
    if len(ids) > LATTICE_CAP:
        ids = ids[np.linspace(0, len(ids) - 1, LATTICE_CAP).round().astype(int)]
    return ids.astype(np.int64)


def occ_tick(net: Any, strip: Any, dt: float, state: Any = None, occ_state: Any = None, ages8: Any = None,
             *, ego: Any = None, use_memory: bool = True, poses: Any = None) -> tuple[OccOutV2, Any, Any]:
    """Batch-1 tick. ``dt`` = frame timestamp delta (s). ``ego`` = vehicle (speed, yaw_rate) only with
    ``--ego-from-vehicle``; default None = camera VO. ``use_memory=False`` = no-memory ablation."""
    import torch

    from python.perception.lanes import camera_poses

    cuda = strip.is_cuda
    if cuda:
        torch.cuda.reset_peak_memory_stats(strip.device)

    def now() -> float:
        if cuda:
            torch.cuda.synchronize(strip.device)
        return time.perf_counter()

    core = net.occ_core
    with torch.no_grad():
        t0 = now()
        dts = torch.tensor([float(dt)], device=strip.device)
        kw = {} if ages8 is None else {"ages": ages8}
        pred, state = net.forward_batch(strip, dts, state if state is not None else net.initial_state(), **kw)
        ages = None if ages8 is None else core.ages_for(ages8)
        t1 = now()
        cur = core.fuse(pred.occ_maps, ages)
        t2 = now()
        vox, occ_state = core.step(pred.occ_maps, dt, occ_state if use_memory else None, ego=ego, ages=ages, cur=cur)
        t3 = now()
        stored = vox.ids[0][vox.ids[0] >= 0].cpu().numpy()
        lat = np.setdiff1d(unseen_lattice(), stored)
        q_np = np.concatenate([stored, lat])[:MAX_OCC]
        q = torch.as_tensor(q_np, device=strip.device).unsqueeze(0)
        logits = net.occ_head(pred.occ_feat, q, vox)
        t4 = now()
        samples = core.cam_samples(pred.occ_maps, ages)
    ms = {"backbone_scene_heads": (t1 - t0) * 1e3, "lift_query": (t2 - t1) * 1e3, "memory": (t3 - t2) * 1e3,
          "occ_head": (t4 - t3) * 1e3}
    cur_ids = cur.ids[0]
    seen = torch.unique(cur_ids[cur_ids >= 0]).cpu().numpy()
    counts = {"stored": int(min(len(stored), len(q_np))), "active": int(len(seen)), "frustum": int(core.frustum.numel())}
    live = None
    if ages is not None:
        live = (torch.as_tensor(ages).reshape(-1) <= float(core.stale_s)).tolist()
    d = {k: v[0].detach().float().cpu().numpy() for k, v in pred.as_dict().items() if torch.is_tensor(v)}
    vo = None if occ_state.vo is None else occ_state.vo[0].float().cpu().numpy()
    if ego is not None and vo is not None:  # report the motion actually used for the warp, labelled vehicle
        e = np.asarray(ego, np.float64).reshape(-1)
        vo = np.array([0.0, e[0] * float(dt), e[1] * float(dt)])
    out = build_occ_out(
        cams=list(core.cams), occ_ids=q_np, occ_logits=logits[0].float().cpu().numpy(), seen_ids=seen, counts=counts,
        pred=d, vo=vo, dt_s=float(dt), ego_source="vo" if ego is None else "vehicle",
        carried_ids=None if occ_state.carried is None else occ_state.carried[0].cpu().numpy(),
        cam_samples={k: v[0].float().cpu().numpy() if k != "ids" else v[0].cpu().numpy() for k, v in samples.items()},
        poses=poses if poses is not None else camera_poses(), live_cams=live, timing_ms=ms,
        peak_vram_mb=(torch.cuda.max_memory_allocated(strip.device) / 2**20) if cuda else None)
    return out, state, occ_state
