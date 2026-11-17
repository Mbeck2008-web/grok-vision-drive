"""Shared synthetic fixtures for the sparse occupancy net tests and the rig trade-off script.

Everything here is synthetic and CPU-only: a flat road world with paint, a
stop bar and one parked car, rendered for any subset of the 8 cameras at a
small resolution, plus a short joint occupancy + VO training loop. Labels
come from the synthetic depth/sem/boxes (the same code path as recorder GT).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from python.occ.camera import cameras_from_config  # noqa: E402
from python.occ.labels import CamFrame, voxel_labels  # noqa: E402
from python.occ.model import ALL_CAMS, OccProfile, OccSparseNet, make_queries, occ_loss  # noqa: E402
from python.occ.synthetic import SEM_CLASSES, scene  # noqa: E402

FULL_CAMS = cameras_from_config(ids=ALL_CAMS)
# One intra-op thread: the box is shared and heavily loaded; more threads oversubscribe (measured 13 s vs 0.02 s per forward).
torch.set_num_threads(1)


def small_rig(ids=ALL_CAMS, long_side: int = 128, overrides: dict | None = None):
    """Recorder camera geometry scaled down so CPU tests stay fast."""
    out = {}
    for cid in ids:
        cam = FULL_CAMS[cid]
        ls = (overrides or {}).get(cid, long_side)
        s = ls / max(cam.width, cam.height)
        out[cid] = cam.scaled(max(16, round(cam.width * s)), max(16, round(cam.height * s)))
    return out


def small_profile(ids=ALL_CAMS, long_side: int = 128, **kw) -> OccProfile:
    return OccProfile(cams=tuple(ids), input_long_side=long_side, **kw)


def frame(cams, ego_y: float = 0.0, box: dict | None = None):
    """Render every camera at ego position ``ego_y`` (forward along the road); labels from depth/sem/box."""
    rgb, frames, paint, bars, car = {}, [], [], [], None
    depth = {}
    for cid, cam in cams.items():
        sc = scene(cam, ego_y=ego_y, box=box)
        rgb[cid] = sc["rgb"]
        depth[cid] = sc["depth"]
        frames.append(CamFrame(cam, sc["depth"], sc["sem"]))
        paint, bars, car = sc["paint"], sc["stop_bars"], sc["box"]
    lab = voxel_labels(frames, SEM_CLASSES, [car] if car else [], paint=paint, stop_bars=bars, max_free=1500, seed=0)
    lab["depth"] = depth  # training-only label (recorder depth); never passed to the runtime
    return rgb, lab


def to_tensors(rgbs: list[dict], cams, ages: float = 0.0):
    imgs = {cid: torch.stack([torch.from_numpy(r[cid]).permute(2, 0, 1).float() / 255.0 for r in rgbs]) for cid in cams}
    age = {cid: torch.full((len(rgbs),), float(ages)) for cid in cams}
    return imgs, age


def queries(labs: list[dict], n_pos: int = 384, n_neg: int = 384, vo=None):
    qs = [make_queries(lab, n_pos=n_pos, n_neg=n_neg, seed=k) for k, lab in enumerate(labs)]
    q = {k: torch.from_numpy(np.stack([qq[k] for qq in qs])) for k in qs[0]}
    if vo is not None:
        q["vo_t"] = torch.tensor(vo, dtype=torch.float32)
    for cid in labs[0].get("depth", {}):
        q[f"depth:{cid}"] = torch.from_numpy(np.stack([lab["depth"][cid] for lab in labs]).astype(np.float32))
    return q


def train_joint(model: OccSparseNet, cams, *, steps: int, batch: int = 3, pool_step: float = 0.25, pool_len: float = 24.0,
                max_shift: int = 6, seed: int = 0, lr: float = 1e-3, log=None):
    """Joint occupancy + VO training on synthetic forward-driving pairs (TBPTT 1, VO drives the warp).

    Ego positions come from a pool on a ``pool_step`` grid; pairs advance 0..max_shift
    pool steps (0..1.5 m per tick at the defaults = 0..15 m/s at 10 Hz). Returns the
    loss history (total, vo)."""
    rng = np.random.default_rng(seed)
    n = int(pool_len / pool_step) + 1
    cache: dict[int, tuple] = {}

    def get(i):
        if i not in cache:
            cache[i] = frame(cams, ego_y=i * pool_step)
        return cache[i]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    model.train()
    for it in range(steps):
        i0 = rng.integers(0, n - max_shift, size=batch)
        sh = rng.integers(0, max_shift + 1, size=batch)
        prev = [get(int(i)) for i in i0]
        cur = [get(int(i + s)) for i, s in zip(i0, sh)]
        pim, pag = to_tensors([p[0] for p in prev], cams)
        cim, cag = to_tensors([c[0] for c in cur], cams)
        vo = [[0.0, float(s) * pool_step, 0.0] for s in sh]
        q = queries([c[1] for c in cur], vo=vo)
        with torch.no_grad():
            pout = model(pim, cams, pag, q["pts"][:, :8], memory=None)
        out = model(cim, cams, cag, q["pts"], memory=pout["memory"])
        loss = occ_loss(out, q)
        opt.zero_grad()
        loss["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append((float(loss["total"].detach()), float(torch.as_tensor(loss["vo"]).detach())))
        if log and (it % 20 == 0 or it == steps - 1):
            log(f"step {it} total {hist[-1][0]:.4f} vo {hist[-1][1]:.4f}")
    model.eval()
    return hist


def held_out_clip(speeds_mps, dt: float = 0.1, start: float = 0.13):
    """Ego positions of a held-out forward clip (positions off the training pool grid)."""
    ys = [start]
    for v in speeds_mps:
        ys.append(ys[-1] + v * dt)
    return ys


def math_isfinite_all(xs) -> bool:
    return all(math.isfinite(float(x)) for x in xs)
