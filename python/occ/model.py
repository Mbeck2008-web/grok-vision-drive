"""Sparse occupancy net: tiny CNN + point queries projected into the camera(s).

Camera-only at runtime: inputs are RGB images and the fixed camera mount
geometry from ``config/cameras.yaml``. The head is evaluated only at queried
3-D points, never on a dense grid:

1. coarse: one query column per 2 m BEV pillar (3 heights, averaged), giving
   a pillar occupancy probability;
2. fine: the 0.5 m voxels of the top ``k_max`` pillars above ``coarse_thr``;
   cells above ``occ_thr`` are the output (sparse indices + class + conf).

Training queries are the label's occupied voxels (positives) and its
ray-free voxels (negatives); unknown / ignore voxels are never queried.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from python.occ.camera import CameraModel
from python.occ.classes import N_CLASSES, UNKNOWN
from python.occ.grid import DEFAULT_GRID, VoxelGrid

N_FREQ = 6
# Allowance for the CUDA context + cuDNN workspace (typical, not measured here).
CUDA_CONTEXT_GB = 0.6
COARSE_HEIGHTS = (0.0, 0.75, 1.5)


def _cbr(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def posenc(xyz: torch.Tensor, grid: VoxelGrid = DEFAULT_GRID) -> torch.Tensor:
    lo = torch.tensor([grid.x_min, grid.y_min, grid.z_min], dtype=xyz.dtype, device=xyz.device)
    hi = torch.tensor([grid.x_max, grid.y_max, grid.z_max], dtype=xyz.dtype, device=xyz.device)
    u = (xyz - lo) / (hi - lo) * 2 - 1
    freqs = (2.0 ** torch.arange(N_FREQ, dtype=xyz.dtype, device=xyz.device)) * math.pi
    ang = u[..., None] * freqs
    return torch.cat([u, torch.sin(ang).flatten(-2), torch.cos(ang).flatten(-2)], dim=-1)


POSENC_DIM = 3 + 3 * 2 * N_FREQ


class OccSparseNet(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, widths: Sequence[int] = (16, 32, 64, 96), feat: int = 64, hidden: int = 96, grid: VoxelGrid = DEFAULT_GRID):
        super().__init__()
        w = list(widths)
        self.grid = grid
        self.stem = _cbr(3, w[0], 2)
        self.s4 = nn.Sequential(_cbr(w[0], w[1], 2), _cbr(w[1], w[1], 1))
        self.s8 = nn.Sequential(_cbr(w[1], w[2], 2), _cbr(w[2], w[2], 1))
        self.s16 = nn.Sequential(_cbr(w[2], w[3], 2), _cbr(w[3], w[3], 1))
        self.lat8 = nn.Conv2d(w[2], feat, 1)
        self.lat16 = nn.Conv2d(w[3], feat, 1)
        self.feat = feat
        din = feat + POSENC_DIM + 2
        self.mlp = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.occ_head = nn.Linear(hidden, 1)
        self.cls_head = nn.Linear(hidden, n_classes)
        self.coarse = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))

    def features(self, img: torch.Tensor) -> torch.Tensor:
        x = self.stem(img)
        x = self.s4(x)
        f8 = self.s8(x)
        f16 = self.s16(f8)
        return self.lat8(f8) + F.interpolate(self.lat16(f16), size=f8.shape[-2:], mode="bilinear", align_corners=False)

    @staticmethod
    def _project(cam: CameraModel, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        basis = torch.as_tensor(cam.basis(), dtype=pts.dtype, device=pts.device)
        pos = torch.as_tensor(np.asarray(cam.pos), dtype=pts.dtype, device=pts.device)
        c = (pts - pos) @ basis.T
        z = c[..., 2]
        zs = torch.where(z > 0.1, z, torch.ones_like(z))
        u = cam.f * c[..., 0] / zs + cam.width / 2.0
        v = cam.f * c[..., 1] / zs + cam.height / 2.0
        inview = (z > 0.1) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        g = torch.stack([u / cam.width * 2 - 1, v / cam.height * 2 - 1], dim=-1)
        return g, z, inview

    def point_inputs(self, feats: Mapping[str, torch.Tensor], cams: Mapping[str, CameraModel], pts: torch.Tensor) -> torch.Tensor:
        """Per-point input vector: mean feature over cameras that see it + posenc + depth + seen flag."""
        B, N, _ = pts.shape
        acc = pts.new_zeros(B, N, self.feat)
        cnt = pts.new_zeros(B, N)
        zmin = pts.new_full((B, N), 1e3)
        for cid, fm in feats.items():
            g, z, inview = self._project(cams[cid], pts)
            s = F.grid_sample(fm, g.view(B, N, 1, 2).to(fm.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)
            s = s.view(B, self.feat, N).transpose(1, 2).to(pts.dtype)
            m = inview.to(pts.dtype)
            acc = acc + s * m[..., None]
            cnt = cnt + m
            zmin = torch.where(inview, torch.minimum(zmin, z), zmin)
        mean = acc / cnt.clamp(min=1.0)[..., None]
        seen = (cnt > 0).to(pts.dtype)
        dz = torch.log1p(zmin.clamp(min=0.0, max=1e3)) / math.log1p(1e3) * seen
        return torch.cat([mean, posenc(pts, self.grid), dz[..., None], seen[..., None]], dim=-1)

    def forward(self, images: Mapping[str, torch.Tensor], cams: Mapping[str, CameraModel], pts: torch.Tensor, coarse_pts: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        feats = {cid: self.features(img) for cid, img in images.items()}
        cams_s = {cid: cams[cid].scaled(img.shape[-1], img.shape[-2]) for cid, img in images.items()}
        x = self.point_inputs(feats, cams_s, pts)
        h = self.mlp(x)
        out = {"occ_logit": self.occ_head(h).squeeze(-1), "cls_logit": self.cls_head(h)}
        if coarse_pts is not None:
            B, M, _ = coarse_pts.shape
            cols = torch.cat([coarse_pts.clone() for _ in COARSE_HEIGHTS], dim=1)
            for j, zh in enumerate(COARSE_HEIGHTS):
                cols[:, j * M : (j + 1) * M, 2] = zh
            xc = self.point_inputs(feats, cams_s, cols).view(B, len(COARSE_HEIGHTS), M, -1).mean(dim=1)
            out["coarse_logit"] = self.coarse(xc).squeeze(-1)
        return out

    @torch.no_grad()
    def infer(self, images: Mapping[str, torch.Tensor], cams: Mapping[str, CameraModel], *, k_max: int = 128, coarse_thr: float = 0.3, occ_thr: float = 0.5) -> dict[str, Any]:
        """Sparse inference for one frame (batch 1). Returns cells + query stats."""
        g = self.grid
        for img in images.values():
            if img.dim() != 4 or img.shape[0] != 1:
                raise ValueError("infer takes batch-1 images (1,3,H,W)")
            if img.shape[-1] < 16 or img.shape[-2] < 16:
                raise ValueError(f"image too small: {tuple(img.shape[-2:])}")
        dtype = next(iter(images.values())).dtype
        feats = {cid: self.features(img) for cid, img in images.items()}
        cams_s = {cid: cams[cid].scaled(img.shape[-1], img.shape[-2]) for cid, img in images.items()}
        cpts = torch.as_tensor(g.coarse_centers(), dtype=dtype)[None]
        M = cpts.shape[1]
        cols = cpts.repeat(1, len(COARSE_HEIGHTS), 1)
        for j, zh in enumerate(COARSE_HEIGHTS):
            cols[:, j * M : (j + 1) * M, 2] = zh
        xc = self.point_inputs(feats, cams_s, cols).view(1, len(COARSE_HEIGHTS), M, -1).mean(dim=1)
        cprob = torch.sigmoid(self.coarse(xc).squeeze(-1)[0].float())
        cprob = torch.nan_to_num(cprob, nan=0.0)
        cand = torch.nonzero(cprob > coarse_thr).flatten()
        cand = cand[torch.argsort(cprob[cand], descending=True)][: max(0, int(k_max))]
        cx, cy = g.coarse_shape
        cidx = np.stack([(cand // cy).cpu().numpy(), (cand % cy).cpu().numpy()], axis=1)
        fidx = g.fine_in_coarse(cidx)
        n_q = int(fidx.shape[0])
        empty = {"idx": np.zeros((0, 3), np.int32), "cls": np.zeros(0, np.int8), "conf": np.zeros(0, np.float32)}
        stats = {"coarse_queries": M * len(COARSE_HEIGHTS), "fine_queries": n_q, "n_dense": g.n_dense, "pillars": int(cand.numel())}
        if n_q == 0:
            return {"cells": empty, "stats": stats}
        pts = torch.as_tensor(g.to_center(fidx), dtype=dtype)[None]
        h = self.mlp(self.point_inputs(feats, cams_s, pts))
        prob = torch.nan_to_num(torch.sigmoid(self.occ_head(h).squeeze(-1)[0].float()), nan=0.0)
        cls = torch.argmax(self.cls_head(h)[0].float(), dim=-1)
        keep = (prob > occ_thr).cpu().numpy()
        cells = {"idx": fidx[keep].astype(np.int32), "cls": cls.cpu().numpy()[keep].astype(np.int8), "conf": prob.cpu().numpy()[keep].astype(np.float32)}
        return {"cells": cells, "stats": stats}


def make_queries(labels: Mapping[str, np.ndarray], *, grid: VoxelGrid = DEFAULT_GRID, n_pos: int = 512, n_neg: int = 512, seed: int = 0) -> dict[str, np.ndarray]:
    """Fixed-size training queries for one sample (padded with mask 0)."""
    rng = np.random.default_rng(seed)
    occ, ocls, free = labels["occ_idx"], labels["occ_cls"], labels["free_idx"]
    pi = rng.permutation(len(occ))[:n_pos]
    ni = rng.permutation(len(free))[:n_neg]
    N = n_pos + n_neg
    pts = np.zeros((N, 3), np.float32)
    occ_t = np.zeros(N, np.float32)
    cls_t = np.full(N, -1, np.int64)
    mask = np.zeros(N, np.float32)
    pts[: len(pi)] = grid.to_center(occ[pi])
    occ_t[: len(pi)] = 1.0
    cls_t[: len(pi)] = np.where(ocls[pi] == UNKNOWN, -1, ocls[pi])
    mask[: len(pi)] = 1.0
    pts[n_pos : n_pos + len(ni)] = grid.to_center(free[ni])
    mask[n_pos : n_pos + len(ni)] = 1.0
    cx, cy = grid.coarse_shape
    cpos = set(map(tuple, grid.coarse_of(occ).tolist())) if len(occ) else set()
    cneg = set(map(tuple, grid.coarse_of(free).tolist())) - cpos if len(free) else set()
    centres = grid.coarse_centers()
    coarse_t = np.full(cx * cy, -1.0, np.float32)
    for (a, b) in cpos:
        coarse_t[a * cy + b] = 1.0
    for (a, b) in cneg:
        coarse_t[a * cy + b] = 0.0
    return {"pts": pts, "occ_t": occ_t, "cls_t": cls_t, "mask": mask, "coarse_pts": centres, "coarse_t": coarse_t}


def occ_loss(out: Mapping[str, torch.Tensor], q: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    m = q["mask"]
    occ = F.binary_cross_entropy_with_logits(out["occ_logit"].float(), q["occ_t"], reduction="none")
    l_occ = (occ * m).sum() / m.sum().clamp(min=1.0)
    cls_t = q["cls_t"]
    valid = cls_t >= 0
    if bool(valid.any()):
        l_cls = F.cross_entropy(out["cls_logit"].float()[valid], cls_t[valid])
    else:
        l_cls = out["cls_logit"].sum() * 0.0
    total = l_occ + 0.5 * l_cls
    parts = {"occ": l_occ, "cls": l_cls}
    if "coarse_logit" in out:
        ct = q["coarse_t"]
        cm = (ct >= 0).float()
        lc = F.binary_cross_entropy_with_logits(out["coarse_logit"].float(), ct.clamp(min=0.0), reduction="none")
        l_coarse = (lc * cm).sum() / cm.sum().clamp(min=1.0)
        total = total + l_coarse
        parts["coarse"] = l_coarse
    parts["total"] = total
    return parts


def param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def activation_bytes(model: OccSparseNet, cams: Mapping[str, CameraModel], *, height: int, width: int, n_queries: int, batch: int = 1, bytes_per: int = 2) -> int:
    """Sum of every module output for one forward at this size (an upper bound on live activations).

    Measured by forward hooks on a real CPU forward in fp32, then scaled to
    ``bytes_per`` (2 = fp16). Not a GPU measurement.
    """
    total = 0

    def hook(_m: nn.Module, _i: Any, o: Any) -> None:
        nonlocal total
        if isinstance(o, torch.Tensor):
            total += o.numel()

    hs = [m.register_forward_hook(hook) for m in model.modules() if len(list(m.children())) == 0]
    try:
        model.eval()
        with torch.no_grad():
            imgs = {cid: torch.zeros(batch, 3, height, width) for cid in cams}
            pts = torch.zeros(batch, n_queries, 3)
            pts[..., 1] = 10.0
            model(imgs, cams, pts, coarse_pts=torch.as_tensor(model.grid.coarse_centers())[None].repeat(batch, 1, 1))
    finally:
        for h in hs:
            h.remove()
    return int(total * bytes_per)


def memory_estimate(model: OccSparseNet, cams: Mapping[str, CameraModel], *, height: int = 480, width: int = 640, n_queries: int = 16384, train_batch: int = 8, train_queries: int = 2048) -> dict[str, float]:
    """CPU estimate (GB) of inference fp16 and AMP training memory. Not a GPU number."""
    p = param_count(model)
    gb = 1024**3
    infer_act = activation_bytes(model, cams, height=height, width=width, n_queries=n_queries, batch=1, bytes_per=2)
    train_act = activation_bytes(model, cams, height=height, width=width, n_queries=train_queries, batch=train_batch, bytes_per=2)
    # AMP: fp32 master + fp32 grad + Adam m, v (16 B) + fp16 copy (2 B) per param; all activations kept for backward (x2 for grads).
    return {
        "params": float(p),
        "infer_gb": (p * 2 + infer_act) / gb + CUDA_CONTEXT_GB,
        "train_gb": (p * 18 + 2 * train_act) / gb + CUDA_CONTEXT_GB,
        "cuda_context_gb": CUDA_CONTEXT_GB,
    }
