"""Train the multi-camera sparse occupancy net from shards (recovery mix enforced by the sampler).

    python -m python.occ.train_occ --shards <dir> --epochs 10 --batch 4 --profile full8 --out occ_sparse.pt

lr stays 1e-3 (repo rule). ``--amp`` uses fp16 autocast on CUDA (memory only on
Pascal). Each step first runs the previous tick (no grad) to build the
temporal memory, then the current tick with the VO head supplying the warp,
so the GRU, the warp and VO are trained (TBPTT length 1).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from python.occ.camera import CameraModel, cameras_from_config
from python.occ.model import ALL_CAMS, OccProfile, OccSparseNet, make_queries, occ_loss, profile_preset, save_checkpoint
from python.occ.sampler import RecoveryMixSampler
from python.occ.shards import ShardDataset

LR = 1e-3


def collate(samples: Sequence[Mapping[str, Any]], profile: OccProfile, cams: Mapping[str, CameraModel], *, seed: int = 0, n_pos: int = 512, n_neg: int = 512):
    """Images per camera (missing -> zeros with age NaN, i.e. masked), ages (B,), queries."""
    B = len(samples)
    images: dict[str, torch.Tensor] = {}
    ages: dict[str, torch.Tensor] = {}
    for cid in profile.cams:
        if cid not in cams:
            continue
        cam = cams[cid]
        s = profile.long_side(cid) / max(cam.width, cam.height)
        h, w = max(16, round(cam.height * s)), max(16, round(cam.width * s))
        stack, age = [], []
        for smp in samples:
            img = smp["rgb"].get(cid)
            meta = smp.get("meta", {})
            if img is None:
                stack.append(torch.zeros(3, h, w))
                age.append(math.nan)
                continue
            x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()[None] / 255.0
            stack.append(F.interpolate(x, size=(h, w), mode="bilinear", antialias=True, align_corners=False)[0])
            try:
                age.append(float(meta.get("t")) - float(meta.get("cam_t", {}).get(cid, meta.get("t"))))
            except (TypeError, ValueError):
                age.append(math.nan)
        images[cid] = torch.stack(stack)
        ages[cid] = torch.tensor(age, dtype=torch.float32)
    qs = [make_queries(s["arrays"], grid=profile.grid, n_pos=n_pos, n_neg=n_neg, seed=seed + k) for k, s in enumerate(samples)]
    q = {k: torch.from_numpy(np.stack([qq[k] for qq in qs])) for k in qs[0]}
    q["vo_t"] = torch.from_numpy(np.stack([np.asarray(s["arrays"].get("vo_t", np.full(3, np.nan)), np.float32) for s in samples]))
    for cid in profile.cams:  # depth-bin head targets (recorder depth, training only)
        ds = [s["arrays"].get(f"depth_{cid}") for s in samples]
        shp = next((d.shape for d in ds if d is not None), None)
        if shp is not None:
            q[f"depth:{cid}"] = torch.from_numpy(np.stack([np.asarray(d, np.float32) if d is not None and d.shape == shp else np.full(shp, np.nan, np.float32) for d in ds]))
    return images, ages, q


def train_step(model: OccSparseNet, cams, opt, scaler, batch, prev_batch=None, *, device: str = "cpu", amp: bool = False, clip: float = 1.0) -> dict[str, float]:
    images, ages, q = batch
    images = {k: v.to(device) for k, v in images.items()}
    ages = {k: v.to(device) for k, v in ages.items()}
    q = {k: v.to(device) for k, v in q.items()}
    memory = None
    if prev_batch is not None:
        pim, pag, pq = prev_batch
        with torch.no_grad():
            memory = model({k: v.to(device) for k, v in pim.items()}, cams, {k: v.to(device) for k, v in pag.items()}, pq["pts"][:, :1].to(device))["memory"]
    model.train()
    cuda = device.startswith("cuda")
    with torch.autocast(device_type="cuda" if cuda else "cpu", enabled=amp and cuda, dtype=torch.float16):
        out = model(images, cams, ages, q["pts"], memory=memory)
    parts = occ_loss(out, q)
    opt.zero_grad(set_to_none=True)
    scaler.scale(parts["total"]).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    scaler.step(opt)
    scaler.update()
    return {k: float(v.detach()) for k, v in parts.items()}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--profile", default="full8")
    ap.add_argument("--long-side", type=int, default=None, help="input resolution knob (first lever)")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="occ_sparse.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    prof = profile_preset(a.profile, **({"input_long_side": a.long_side} if a.long_side else {}))
    ds = ShardDataset(sorted(Path(a.shards).glob("*.tar")))
    sampler = RecoveryMixSampler(ds.recovery_flags(), a.batch, seed=a.seed)
    cams = cameras_from_config(ids=tuple(c for c in prof.cams))
    model = OccSparseNet(profile=prof).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and a.device.startswith("cuda"))
    index_of = {m.get("i"): k for k, m in enumerate(ds.meta)}
    for ep in range(a.epochs):
        losses = []
        for b in sampler.batches(ep):
            cur = [ds[i] for i in b]
            prev_idx = [index_of.get(ds.meta[i].get("i", -10) - 1) for i in b]
            prev = None
            if all(p is not None for p in prev_idx):
                prev = collate([ds[p] for p in prev_idx], prof, cams, seed=a.seed)
            losses.append(train_step(model, cams, opt, scaler, collate(cur, prof, cams, seed=a.seed + ep), prev, device=a.device, amp=a.amp)["total"])
        print(json.dumps({"epoch": ep, "steps": len(losses), "loss_mean": float(np.mean(losses)) if losses else None}), flush=True)
    save_checkpoint(model, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
