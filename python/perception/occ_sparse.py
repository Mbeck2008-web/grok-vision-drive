"""Sparse occupancy net: camera images in, occupancy at a sparse point set out.

Runtime is camera-only. ``SparseOccNet.forward(images, points)`` takes the rig
RGB frames and ego-frame query points; it has no depth, semantic, or box
input, and this module never imports the training-only label builder
(``python/train/occ_labels.py``; ``scripts/test_occ_runtime_imports.py``).

Sparse, never dense: features are sampled at query points only. ``predict``
runs coarse-to-fine over ``LEVELS`` (cells 4x, then 2x the fine voxel): every
4x cell some camera sees is queried once for "any fine cell occupied"; only
the 2x children of hot 4x cells are queried next, and only the fine children
of hot 2x cells get an occupancy query. No dense volume tensor exists at any
point. Three levels, not two, because the measured labels are a surface
sheet (15 of 64 fine cells in a hot 4x cell; ``docs/specs/occ_sparse.md``).

Outputs per point: ``occ`` (this fine cell occupied), ``any`` (B,N,levels:
the level-``l`` cell holding the point has an occupied fine cell), ``cls``
logits over ``OCC_CLASS_COUNT`` coarse classes (other / drivable / vehicle).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from python.perception.occ_geometry import DEFAULT_GRID, OccGrid, PinholeCam

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch-free hosts
    torch = None
    nn = None
    F = None

OCC_CLASS_COUNT = 3
FEAT = 64
ENC_CHANNELS = (32, 64, 96)
PE_FREQS = 6
HIDDEN = 128
LEVELS = (4, 2)
COARSE_FACTOR = LEVELS[0]
COARSE_THRESH = 0.3
FINE_THRESH = 0.5
LIVE_LONG_SIDE = 640


def positional_encoding_dim(freqs: int = PE_FREQS) -> int:
    return 3 + 3 * 2 * int(freqs)


def visible_mask(cams: Sequence[PinholeCam], pts: np.ndarray) -> np.ndarray:
    """True where at least one camera's frustum holds the point."""
    from python.perception.occ_geometry import project

    vis = np.zeros((np.asarray(pts).shape[0],), dtype=bool)
    for cam in cams:
        vis |= project(cam, pts)[2]
    return vis


def children_offsets(factor: int, voxel_m: float) -> np.ndarray:
    """Fine-cell centre offsets inside one coarse cell, relative to its centre."""
    f = int(factor)
    o = (np.arange(f) + 0.5) * float(voxel_m) - f * float(voxel_m) / 2.0
    return np.stack(np.meshgrid(o, o, o, indexing="ij"), axis=-1).reshape(-1, 3)


@dataclass
class OccPrediction:
    keys: np.ndarray       # fine cell keys predicted occupied (sorted)
    prob: np.ndarray       # their occupancy probability
    cls: np.ndarray        # their coarse class
    n_coarse: int          # coarse points queried
    n_fine: int            # fine points queried

    @property
    def n_queried(self) -> int:
        return int(self.n_coarse + self.n_fine)

    def queried_ratio(self, grid: OccGrid = DEFAULT_GRID) -> float:
        """Points queried / dense fine cells. Dense would be 1.0."""
        return float(self.n_queried) / float(grid.dense_cells)


if nn is not None:

    class CamEncoder(nn.Module):
        """Shared per-camera CNN, stride 8. GroupNorm: batch 1 is legal."""

        def __init__(self, channels: Sequence[int] = ENC_CHANNELS, feat: int = FEAT) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev = 3
            for ch in channels:
                layers += [nn.Conv2d(prev, ch, 3, stride=2, padding=1), nn.GroupNorm(8, ch), nn.ReLU()]
                prev = ch
            layers += [nn.Conv2d(prev, feat, 3, padding=1), nn.GroupNorm(8, feat), nn.ReLU(), nn.Conv2d(feat, feat, 1)]
            self.net = nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            return self.net(x)

    class SparseOccNet(nn.Module):
        """Camera features sampled at sparse 3-D query points."""

        def __init__(self, cams: Sequence[PinholeCam], *, grid: OccGrid = DEFAULT_GRID, feat: int = FEAT, hidden: int = HIDDEN) -> None:
            super().__init__()
            if not cams:
                raise ValueError("SparseOccNet needs at least one camera")
            self.cam_ids = tuple(c.cam_id for c in cams)
            self.cams = list(cams)
            self.grid = grid
            self.encoder = CamEncoder(feat=feat)
            rot = np.stack([c.rotation() for c in cams]).astype(np.float32)
            pos = np.stack([np.asarray(c.pos) for c in cams]).astype(np.float32)
            intr = np.array([[c.focal_px, c.width, c.height, c.near_m, c.far_m] for c in cams], dtype=np.float32)
            self.register_buffer("cam_rot", torch.from_numpy(rot), persistent=False)
            self.register_buffer("cam_pos", torch.from_numpy(pos), persistent=False)
            self.register_buffer("cam_intr", torch.from_numpy(intr), persistent=False)
            lo = torch.tensor(grid.lo, dtype=torch.float32)
            hi = torch.tensor(grid.hi, dtype=torch.float32)
            self.register_buffer("box_lo", lo, persistent=False)
            self.register_buffer("box_hi", hi, persistent=False)
            pe = positional_encoding_dim()
            self.head = nn.Sequential(
                nn.Linear(feat + pe + 1, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1 + len(LEVELS) + OCC_CLASS_COUNT),
            )

        def _pe(self, pts: Any) -> Any:
            unit = (pts - self.box_lo) / (self.box_hi - self.box_lo) * 2.0 - 1.0
            out = [unit]
            for i in range(PE_FREQS):
                out += [torch.sin((2.0 ** i) * np.pi * unit), torch.cos((2.0 ** i) * np.pi * unit)]
            return torch.cat(out, dim=-1)

        def encode(self, images: Sequence[Any]) -> list[Any]:
            """One feature map per camera, in ``cam_ids`` order. Each image is (B,3,H,W) in [0,1]."""
            if len(images) != len(self.cam_ids):
                raise ValueError(f"expected {len(self.cam_ids)} camera images, got {len(images)}")
            return [self.encoder(img) for img in images]

        def project(self, k: int, pts: Any) -> tuple[Any, Any, Any]:
            """Pixel (u, v) and frustum mask of ego points in camera ``k``."""
            c = (pts - self.cam_pos[k]) @ self.cam_rot[k].T
            f, w, h, near, far = (self.cam_intr[k, i] for i in range(5))
            z = c[..., 2]
            zs = torch.where(z > 1e-6, z, torch.ones_like(z))
            u = w / 2.0 + f * c[..., 0] / zs
            v = h / 2.0 - f * c[..., 1] / zs
            ok = (z > near) & (z < far) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            return u, v, ok

        def query(self, feats: Sequence[Any], pts: Any) -> dict[str, Any]:
            """Head outputs at ego points ``pts`` (B,N,3)."""
            bsz, n, _ = pts.shape
            acc = pts.new_zeros((bsz, n, feats[0].shape[1]))
            cnt = pts.new_zeros((bsz, n, 1))
            for k, fm in enumerate(feats):
                u, v, ok = self.project(k, pts)
                w, h = self.cam_intr[k, 1], self.cam_intr[k, 2]
                gx = u / w * 2.0 - 1.0
                gy = v / h * 2.0 - 1.0
                grid = torch.stack([gx, gy], dim=-1).unsqueeze(1).to(fm.dtype)  # B,1,N,2
                samp = F.grid_sample(fm, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
                samp = samp.squeeze(2).transpose(1, 2).to(acc.dtype)  # B,N,C
                m = ok.unsqueeze(-1).to(acc.dtype)
                acc = acc + samp * m
                cnt = cnt + m
            mean = acc / cnt.clamp(min=1.0)
            seen = (cnt > 0).to(acc.dtype)
            out = self.head(torch.cat([mean, self._pe(pts).to(acc.dtype), seen], dim=-1))
            nl = len(LEVELS)
            return {"occ": out[..., 0], "any": out[..., 1 : 1 + nl], "cls": out[..., 1 + nl :], "seen": seen.squeeze(-1)}

        def forward(self, images: Sequence[Any], pts: Any) -> dict[str, Any]:
            return self.query(self.encode(images), pts)

        @torch.no_grad()
        def predict(
            self,
            images: Sequence[Any],
            *,
            coarse_thresh: float = COARSE_THRESH,
            fine_thresh: float = FINE_THRESH,
            max_points: int = 65536,
        ) -> OccPrediction:
            """Coarse-to-fine sparse occupancy for one frame (batch 1).

            Each stage queries at most ``max_points`` points (the hottest cells
            are kept), so the worst case is bounded too.
            """
            grid = self.grid
            if getattr(self, "_coarse_pts", None) is None:
                top = grid.coarse(LEVELS[0])
                cpts = top.centers_of_keys(np.arange(top.dense_cells))
                self._coarse_pts = cpts[visible_mask(self.cams, cpts)].astype(np.float32)
            dev = self.cam_pos.device
            feats = self.encode(images)
            pts = self._coarse_pts
            n_coarse = 0
            for li, f in enumerate(LEVELS):
                n_coarse += int(pts.shape[0])
                p_any = torch.sigmoid(self.query(feats, torch.from_numpy(pts).to(dev).unsqueeze(0))["any"][0, :, li].float()).cpu().numpy()
                sel = p_any >= float(coarse_thresh)
                hot, p_hot = pts[sel], p_any[sel]
                nxt = LEVELS[li + 1] if li + 1 < len(LEVELS) else 1
                offs = children_offsets(f // nxt, grid.voxel_m * nxt)
                cap = max(1, int(max_points) // offs.shape[0])
                if hot.shape[0] > cap:
                    hot = hot[np.argsort(-p_hot)[:cap]]
                pts = (hot[:, None, :] + offs[None, :, :]).reshape(-1, 3).astype(np.float32)
                if pts.shape[0] == 0:
                    break
            if pts.shape[0] == 0:
                empty = np.zeros((0,), dtype=np.int64)
                return OccPrediction(empty, np.zeros((0,), np.float32), np.zeros((0,), np.uint8), n_coarse, 0)
            out = self.query(feats, torch.from_numpy(pts).to(dev).unsqueeze(0))
            p = torch.sigmoid(out["occ"][0].float()).cpu().numpy()
            c = out["cls"][0].float().argmax(-1).cpu().numpy().astype(np.uint8)
            sel = p >= float(fine_thresh)
            keys = grid.keys_of(pts[sel].astype(np.float64))
            order = np.argsort(keys)
            return OccPrediction(keys[order], p[sel][order], c[sel][order], n_coarse, int(pts.shape[0]))


def live_cams(cfg: Any, long_side: int = LIVE_LONG_SIDE) -> list[PinholeCam]:
    """Rig cameras from cameras.yaml at the live input long side."""
    from python.perception.occ_geometry import rig_cams

    return [c.scaled(long_side) for c in rig_cams(cfg)]
