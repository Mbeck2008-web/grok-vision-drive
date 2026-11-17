"""Train the sparse occupancy net from shards (recovery mix enforced by the sampler).

    python -m python.occ.train_occ --shards <dir> --epochs 10 --batch 8 --out occ_sparse.pt

lr stays 1e-3 (repo rule). AMP fp16 on CUDA when ``--amp``; CPU runs fp32.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from python.occ.camera import cameras_from_config
from python.occ.model import OccSparseNet, make_queries, occ_loss
from python.occ.sampler import RecoveryMixSampler
from python.occ.shards import ShardDataset

LR = 1e-3


def collate(samples: Sequence[Mapping[str, Any]], *, grid, scale: float = 1.0, seed: int = 0, n_pos: int = 512, n_neg: int = 512) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    import cv2

    imgs: dict[str, list[torch.Tensor]] = {}
    qs = [make_queries(s["arrays"], grid=grid, n_pos=n_pos, n_neg=n_neg, seed=seed + k) for k, s in enumerate(samples)]
    for s in samples:
        for cid, img in s["rgb"].items():
            a = img if scale == 1.0 else cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            imgs.setdefault(cid, []).append(torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0)
    images = {cid: torch.stack(v) for cid, v in imgs.items()}
    q = {k: torch.from_numpy(np.stack([qq[k] for qq in qs])) for k in qs[0]}
    return images, q


def train_steps(model: OccSparseNet, cams, batches: Iterable[tuple[dict, dict]], *, device: str = "cpu", amp: bool = False, clip: float = 1.0) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.startswith("cuda"))
    losses = []
    model.train()
    for images, q in batches:
        images = {k: v.to(device) for k, v in images.items()}
        q = {k: v.to(device) for k, v in q.items()}
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", enabled=amp and device.startswith("cuda"), dtype=torch.float16):
            out = model(images, cams, q["pts"], coarse_pts=q["coarse_pts"])
        parts = occ_loss(out, q)
        opt.zero_grad(set_to_none=True)
        scaler.scale(parts["total"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt)
        scaler.update()
        losses.append(float(parts["total"].detach()))
    return losses


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="occ_sparse.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    paths = sorted(Path(a.shards).glob("*.tar"))
    ds = ShardDataset(paths)
    sampler = RecoveryMixSampler(ds.recovery_flags(), a.batch, seed=a.seed)
    cams = cameras_from_config(ids=tuple(ds.meta[0]["cams"]))
    model = OccSparseNet().to(a.device)
    for ep in range(a.epochs):
        batches = (collate([ds[i] for i in b], grid=model.grid, scale=a.scale, seed=a.seed + ep) for b in sampler.batches(ep))
        losses = train_steps(model, cams, batches, device=a.device, amp=a.amp)
        print(json.dumps({"epoch": ep, "steps": len(losses), "loss_mean": float(np.mean(losses)) if losses else None}), flush=True)
    torch.save({"state_dict": model.state_dict(), "kind": "gvd_occ_sparse_v1"}, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
