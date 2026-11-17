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
from typing import Any, Mapping, Sequence

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
# Net sizes. "cheap" aims at the planner tick on a shared Pascal card (fp16 compute
# there is ~1/64 of fp32, so AMP buys memory, not speed: run cheap in fp32).
NET_CONFIGS = {
    "default": {"channels": ENC_CHANNELS, "feat": FEAT, "hidden": HIDDEN, "ctx": True},
    "cheap": {"channels": (16, 32, 48), "feat": 32, "hidden": 64, "ctx": True},
}
# Reduced-camera presets. The live supervisor reads all 8 cams but the 8-cam live
# rate at the planner tick is unproven: main+wide is the minimum.
CAM_PRESETS = {
    "min": ("main", "wide"),
    "front3": ("main", "wide", "narrow"),
    "all8": ("narrow", "main", "wide", "pillarL", "pillarR", "repeatL", "repeatR", "rear"),
}
# Run profiles for the GTX 1080 Ti shared with BeamNG. Compute is fp32: Pascal
# fp16 math is ~1/64 of fp32, so fp16 is a memory option only, never a speed
# claim. ``max_points`` is the per-stage top-k cap on the descent (16,384 = 5% of
# the grid, 5.1% of the 8-camera fine frustum). int8 (TensorRT, DP4A) is the only precision speed-up on this card and is
# NOT implemented: it would need a calibration set and an accuracy delta vs fp32
# (both UNMEASURED) and adds TensorRT as a dependency.
PROFILES = {
    "1080ti": {"config": "default", "long_side": 640, "cams": "all8", "compute": "fp32", "max_points": 16384, "max_coarse": 2048},
    "1080ti_cheap": {"config": "cheap", "long_side": 480, "cams": "front3", "compute": "fp32", "max_points": 16384, "max_coarse": 2048},
    "1080ti_min": {"config": "cheap", "long_side": 320, "cams": "min", "compute": "fp32", "max_points": 16384, "max_coarse": 2048},
    # Michael's front_cluster knob: main+wide only, main at a higher resolution,
    # against front3 (main+wide+narrow, all at 640). Trade-off measured by
    # scripts/compare_front_cluster.py.
    "front_cluster": {"config": "default", "long_side": 640, "main_long_side": 800, "cams": "min", "compute": "fp32", "max_points": 16384, "max_coarse": 2048},
    "front3": {"config": "default", "long_side": 640, "cams": "front3", "compute": "fp32", "max_points": 16384, "max_coarse": 2048},
}
# Absolute cap on the top (4x coarse) stage. 8 cams see 5,044 of the 5,120 top
# cells; the cap keeps that stage from scaling with the grid.
MAX_COARSE = 2048
# Per-camera age weight: exp(-age / tau), 0 past max age (or no frame).
AGE_TAU_S = 0.5
AGE_MAX_S = 1.5


def cam_weights(cam_ids: Sequence[str], *, preset: str | None = None, ages_s: Mapping[str, float] | None = None,
                tau_s: float = AGE_TAU_S, max_age_s: float = AGE_MAX_S) -> np.ndarray:
    """Fusion weight per camera (``cam_ids`` order), float32 in [0, 1].

    ``preset`` masks cameras outside ``CAM_PRESETS[preset]`` (weight 0).
    ``ages_s`` is each camera's own frame age at this tick (now - its frame
    timestamp): a fresh frame weighs 1, a stale one ``exp(-age/tau)``, one older
    than ``max_age_s`` or missing from ``ages_s`` weighs 0. A NaN or negative
    age counts as missing.
    """
    w = np.ones((len(cam_ids),), dtype=np.float32)
    if preset is not None:
        keep = set(CAM_PRESETS[preset])
        w *= np.array([c in keep for c in cam_ids], dtype=np.float32)
    if ages_s is not None:
        for i, c in enumerate(cam_ids):
            a = ages_s.get(c)
            a = float(a) if a is not None else float("nan")
            w[i] *= 0.0 if not (np.isfinite(a) and 0.0 <= a <= max_age_s) else float(np.exp(-a / tau_s))
    return w


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
    stage_queries: tuple = ()  # points queried per stage: (4x coarse, 2x coarse, fine)

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

        def __init__(self, cams: Sequence[PinholeCam], *, grid: OccGrid = DEFAULT_GRID, feat: int | None = None,
                     hidden: int | None = None, channels: Sequence[int] | None = None, config: str = "default",
                     ctx: bool | None = None, decode_unseen: bool = True) -> None:
            super().__init__()
            if not cams:
                raise ValueError("SparseOccNet needs at least one camera")
            cfg = NET_CONFIGS[config]
            feat = int(feat or cfg["feat"])
            hidden = int(hidden or cfg["hidden"])
            self.config = config
            self.cam_ids = tuple(c.cam_id for c in cams)
            self.cams = list(cams)
            self.grid = grid
            self.channels = tuple(int(c) for c in (channels or cfg["channels"]))
            self.hidden = hidden
            # ctx: a per-tick scene vector (cam_w-weighted mean of every camera's
            # pooled feature map, B x feat) fed to every query, so a query no
            # camera ray reaches (occluded / out of view) still decodes from the
            # scene, not from position alone. decode_unseen: predict() also
            # queries top-stage cells outside every frustum.
            self.ctx = bool(cfg.get("ctx", True) if ctx is None else ctx)
            self.decode_unseen = bool(decode_unseen)
            self.encoder = CamEncoder(self.channels, feat=feat)
            self.feat_dim = feat
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
                nn.Linear(feat + pe + 1 + (feat if self.ctx else 0), hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1 + len(LEVELS) + OCC_CLASS_COUNT),
            )

        def _pe(self, pts: Any) -> Any:
            unit = (pts - self.box_lo) / (self.box_hi - self.box_lo) * 2.0 - 1.0
            out = [unit]
            for i in range(PE_FREQS):
                out += [torch.sin((2.0 ** i) * np.pi * unit), torch.cos((2.0 ** i) * np.pi * unit)]
            return torch.cat(out, dim=-1)

        def encode(self, images: Sequence[Any], cam_w: Any = None) -> list[Any]:
            """One feature map per camera, in ``cam_ids`` order. Each image is (B,3,H,W) in [0,1].

            A ``None`` image, or a camera whose weight in ``cam_w`` is 0, is not
            encoded (its feature map is ``None`` and it drops out of fusion).
            """
            if len(images) != len(self.cam_ids):
                raise ValueError(f"expected {len(self.cam_ids)} camera images (None for a missing cam), got {len(images)}")
            w = None if cam_w is None else np.asarray(cam_w, dtype=np.float32).reshape(-1)
            return [None if (img is None or (w is not None and w[k] <= 0.0)) else self.encoder(img)
                    for k, img in enumerate(images)]

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

        def query(self, feats: Sequence[Any], pts: Any, cam_w: Any = None) -> dict[str, Any]:
            """Head outputs at ego points ``pts`` (B,N,3).

            Fusion is the ``cam_w``-weighted mean of the features of the cameras
            that see a point (default weight 1; ``None`` feature map = weight 0).
            ``seen`` is the summed weight, clipped to 1.
            """
            bsz, n, _ = pts.shape
            acc = pts.new_zeros((bsz, n, self.feat_dim))
            cnt = pts.new_zeros((bsz, n, 1))
            if cam_w is not None and not torch.is_tensor(cam_w):
                cam_w = torch.as_tensor(np.asarray(cam_w, dtype=np.float32))
            if cam_w is not None:
                cam_w = cam_w.to(device=pts.device, dtype=pts.dtype)
                cam_w = cam_w.reshape(1, -1).expand(bsz, -1) if cam_w.dim() == 1 else cam_w
            for k, fm in enumerate(feats):
                if fm is None:
                    continue
                u, v, ok = self.project(k, pts)
                w, h = self.cam_intr[k, 1], self.cam_intr[k, 2]
                gx = u / w * 2.0 - 1.0
                gy = v / h * 2.0 - 1.0
                grid = torch.stack([gx, gy], dim=-1).unsqueeze(1).to(fm.dtype)  # B,1,N,2
                samp = F.grid_sample(fm, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
                samp = samp.squeeze(2).transpose(1, 2).to(acc.dtype)  # B,N,C
                m = ok.unsqueeze(-1).to(acc.dtype)
                if cam_w is not None:
                    m = m * cam_w[:, k].reshape(bsz, 1, 1)
                acc = acc + samp * m
                cnt = cnt + m
            mean = acc / cnt.clamp(min=1e-6)
            seen = cnt.clamp(max=1.0)
            parts = [mean, self._pe(pts).to(acc.dtype), seen]
            if self.ctx:
                parts.append(self.scene_ctx(feats, cam_w, bsz, acc).unsqueeze(1).expand(-1, n, -1))
            out = self.head(torch.cat(parts, dim=-1))
            nl = len(LEVELS)
            return {"occ": out[..., 0], "any": out[..., 1 : 1 + nl], "cls": out[..., 1 + nl :], "seen": seen.squeeze(-1)}

        def scene_ctx(self, feats: Sequence[Any], cam_w: Any, bsz: int, like: Any) -> Any:
            """(B, feat): cam_w-weighted mean of the cameras' spatially pooled feature maps."""
            acc = like.new_zeros((bsz, self.feat_dim))
            tot = like.new_zeros((bsz, 1))
            for k, fm in enumerate(feats):
                if fm is None:
                    continue
                w = like.new_ones((bsz, 1)) if cam_w is None else cam_w[:, k].reshape(bsz, 1).to(like.dtype)
                acc = acc + fm.mean(dim=(2, 3)).to(like.dtype) * w
                tot = tot + w
            return acc / tot.clamp(min=1e-6)

        def forward(self, images: Sequence[Any], pts: Any, cam_w: Any = None) -> dict[str, Any]:
            return self.query(self.encode(images), pts, cam_w)

        def reset_temporal(self) -> None:
            """Forget the carried hot top cells and the sweep phase (scene cut, teleport, tests)."""
            self.__dict__.pop("_hot_top", None)
            self.__dict__.pop("_sweep_phase", None)

        @torch.no_grad()
        def predict(
            self,
            images: Sequence[Any],
            *,
            coarse_thresh: float = COARSE_THRESH,
            fine_thresh: float = FINE_THRESH,
            max_points: int = 65536,
            max_coarse: int = MAX_COARSE,
            cam_w: Any = None,
        ) -> OccPrediction:
            """Coarse-to-fine sparse occupancy for one frame (batch 1).

            Every stage has an absolute cap. The top (4x) stage queries at most
            ``max_coarse`` of the visible coarse cells: last frame's hot top
            cells first (sparse temporal carry, <= half the cap), then a
            rotating stride subset of the rest, so every visible coarse cell is
            revisited within ``ceil(visible / cap)`` frames. Each later stage
            keeps the top-``max_points`` children by probability.
            """
            grid = self.grid
            w = np.array([img is not None for img in images], dtype=np.float32)
            if cam_w is not None:
                w = w * np.asarray(cam_w, dtype=np.float32).reshape(-1)
            active = tuple(bool(x > 0.0) for x in w)
            on = [c for c, a in zip(self.cams, active) if a]
            top = grid.coarse(LEVELS[0])
            cap0 = max(1, int(max_coarse))
            # Top-stage candidates are generated arithmetically (never by enumerating the
            # grid): last frame's hot top keys first (<= half the cap), then the keys
            # phase, phase+stride, ... over the top index space, visible ones kept.
            # Every array here is <= cap0 long whatever the grid size.
            carry = self.__dict__.get("_hot_top", {}).get(active, np.zeros((0,), np.int64))[: cap0 // 2]
            n_top = int(top.dense_cells)
            budget = cap0 - int(carry.shape[0])
            stride = max(1, int(np.ceil(n_top / float(max(1, budget)))))
            phase = self.__dict__.setdefault("_sweep_phase", {}).get(active, 0) % stride
            self._sweep_phase[active] = phase + 1
            sweep = np.arange(phase, n_top, stride, dtype=np.int64)[:budget]
            sweep = sweep[~np.isin(sweep, carry)]
            idx = np.concatenate([carry, sweep])[:cap0]
            cpts = top.centers_of_keys(idx)
            if not on:
                vis = np.zeros((idx.shape[0],), bool)
            elif self.decode_unseen:  # occluded / out-of-view cells decode too (from ctx + memory)
                vis = np.ones((idx.shape[0],), bool)
            else:
                vis = visible_mask(on, cpts)
            idx = idx[vis]
            cands = cpts[vis].astype(np.float32)
            self._top_keys = idx
            self._index_bytes = int(idx.nbytes + cpts.nbytes + sweep.nbytes + carry.nbytes)
            self._coarse_pts = cands
            dev = self.cam_pos.device
            feats = self.encode(images, w)
            cw = torch.from_numpy(w).to(dev)
            pts = self._coarse_pts
            stages: list[int] = []
            if pts.shape[0] == 0:
                empty = np.zeros((0,), dtype=np.int64)
                return OccPrediction(empty, np.zeros((0,), np.float32), np.zeros((0,), np.uint8), 0, 0,
                                     (0,) * (len(LEVELS) + 1))
            n_coarse = 0
            for li, f in enumerate(LEVELS):
                n_coarse += int(pts.shape[0])
                stages.append(int(pts.shape[0]))
                p_any = torch.sigmoid(self.query(feats, torch.from_numpy(pts).to(dev).unsqueeze(0), cw)["any"][0, :, li].float()).cpu().numpy()
                sel = p_any >= float(coarse_thresh)
                if li == 0:  # remember this frame's hot top cells (indices into the candidate list)
                    hot_idx = idx[sel][np.argsort(-p_any[sel], kind="stable")]
                    self.__dict__.setdefault("_hot_top", {})[active] = hot_idx.astype(np.int64)
                hot, p_hot = pts[sel], p_any[sel]
                nxt = LEVELS[li + 1] if li + 1 < len(LEVELS) else 1
                offs = children_offsets(f // nxt, grid.voxel_m * nxt)
                cap = max(1, int(max_points) // offs.shape[0])
                if hot.shape[0] > cap:
                    hot = hot[_capped_hot(hot, p_hot, cap, float(grid.z_min), grid.voxel_m * f)]
                pts = (hot[:, None, :] + offs[None, :, :]).reshape(-1, 3).astype(np.float32)
                self._index_bytes = max(self._index_bytes, int(pts.nbytes + hot.nbytes))
                if pts.shape[0] == 0:
                    break
            if pts.shape[0] == 0:
                empty = np.zeros((0,), dtype=np.int64)
                return OccPrediction(empty, np.zeros((0,), np.float32), np.zeros((0,), np.uint8), n_coarse, 0,
                                    tuple(stages + [0] * (len(LEVELS) + 1 - len(stages))))
            out = self.query(feats, torch.from_numpy(pts).to(dev).unsqueeze(0), cw)
            p = torch.sigmoid(out["occ"][0].float()).cpu().numpy()
            c = out["cls"][0].float().argmax(-1).cpu().numpy().astype(np.uint8)
            sel = p >= float(fine_thresh)
            keys = grid.keys_of(pts[sel].astype(np.float64))
            order = np.argsort(keys)
            return OccPrediction(keys[order], p[sel][order], c[sel][order], n_coarse, int(pts.shape[0]),
                                 tuple(stages) + (int(pts.shape[0]),))


def _capped_hot(hot: np.ndarray, p_hot: np.ndarray, cap: int, z_min: float, cell_m: float) -> np.ndarray:
    """Indices of the ``cap`` hot cells to refine.

    At most half the cap goes to the most populous height layer (the ground:
    with whole-grid map labels it alone exceeds the cap and would starve every
    structure above it — VO and occluders need those). Any share the other
    layers do not use goes back to the ground layer.

    Within a height band, slots are spread in xy rather than a global top-k by
    probability. Off-road ground and solid buildings are slightly easier
    occupied cells than the road; a global top-k would never refine the road,
    and camera-VO / memory then have no structure to track.
    """
    lvl = np.floor((hot[:, 2] - z_min) / cell_m + 1e-6).astype(np.int64)
    band = lvl == int(np.bincount(lvl - lvl.min()).argmax() + lvl.min())

    def _spread(idxs: np.ndarray, budget: int) -> np.ndarray:
        if idxs.size <= budget:
            return idxs
        xy = hot[idxs, :2]
        lo = xy.min(0)
        span = np.maximum(xy.max(0) - lo, 1e-6)
        nx = max(1, int(np.ceil(np.sqrt(budget))))
        ny = max(1, int(np.ceil(budget / nx)))
        bx = np.minimum(((xy[:, 0] - lo[0]) / span[0] * nx).astype(np.int64), nx - 1)
        by = np.minimum(((xy[:, 1] - lo[1]) / span[1] * ny).astype(np.int64), ny - 1)
        bin_id = bx * ny + by
        order = np.argsort(-p_hot[idxs], kind="stable")
        _, first = np.unique(bin_id[order], return_index=True)
        winners = order[first]
        if winners.size >= budget:
            return idxs[winners[np.argsort(-p_hot[idxs[winners]], kind="stable")[:budget]]]
        taken = np.zeros(idxs.size, dtype=bool)
        taken[winners] = True
        fill = order[~taken[order]][: budget - winners.size]
        return idxs[np.concatenate([winners, fill])]

    up = np.flatnonzero(~band)
    low = np.flatnonzero(band)
    n_up = max(cap - min(low.size, cap // 2), 0)
    up = _spread(up, n_up)
    return np.concatenate([up, _spread(low, cap - up.size)])


def profile_cams(cfg: Any, profile: str, *, scale: float = 1.0) -> list[PinholeCam]:
    """Cameras of ``PROFILES[profile]`` (preset + input sizes); ``scale`` shrinks every long side."""
    from python.perception.occ_geometry import rig_cams

    pr = PROFILES[profile]
    keep = set(CAM_PRESETS[pr["cams"]])
    out = []
    for c in rig_cams(cfg):
        if c.cam_id in keep:
            ls = pr.get("main_long_side", pr["long_side"]) if c.cam_id == "main" else pr["long_side"]
            out.append(c.scaled(max(8, int(round(ls * scale)))))
    return out


def memory_estimate(cams: Sequence[PinholeCam], *, config: str = "default", max_points: int = 65536,
                    bytes_per: int = 4, temporal_k: int = 1, batch: int = 1) -> dict[str, float]:
    """ESTIMATE (arithmetic, not measured) of inference memory in MB.

    weights + temporal feature cache (K x cams x C x H/8 x W/8 x bytes; K = 1:
    the net keeps only this frame's maps, temporal state is the tracker's few
    floats per track) + encoder peak (largest consecutive stage pair of one
    camera at a time) + query activations (max_points rows through the head,
    x3 for the coarse-to-fine copies in flight).
    """
    cfg = NET_CONFIGS[config]
    ch, feat, hid = tuple(cfg["channels"]), int(cfg["feat"]), int(cfg["hidden"])
    params = 0
    prev = 3
    for c in ch:
        params += prev * c * 9 + c + 2 * c
        prev = c
    params += prev * feat * 9 + feat + 2 * feat + feat * feat + feat
    pe = positional_encoding_dim()
    cx = feat if cfg.get("ctx", True) else 0  # scene context vector into every query
    params += (feat + pe + 1 + cx) * hid + hid + hid * hid + hid + hid * (1 + len(LEVELS) + OCC_CLASS_COUNT) + 1 + len(LEVELS) + OCC_CLASS_COUNT
    mb = 2.0 ** 20
    cache = sum(temporal_k * feat * (c.height // 8) * (c.width // 8) * bytes_per for c in cams)
    stages = []
    for c in cams:
        h, w, maps = c.height, c.width, [3]
        sizes = [3 * h * w]
        for k in ch:
            h, w = (h + 1) // 2, (w + 1) // 2
            sizes.append(k * h * w)
        sizes.append(feat * h * w)
        stages.append(max(a + b for a, b in zip(sizes, sizes[1:])) * bytes_per)
    query = 3 * int(max_points) * (feat + pe + 1 + cx + 2 * hid + 1 + len(LEVELS) + OCC_CLASS_COUNT + 3) * bytes_per
    b = max(1, int(batch))  # everything but the weights scales with the batch
    out = {"weights_mb": params * 4 / mb, "feature_cache_mb": b * cache / mb, "encoder_peak_mb": b * max(stages) / mb,
           "query_mb": b * query / mb, "batch": float(b), "params_formula": float(params), "temporal_k": float(temporal_k)}
    out["total_mb"] = out["weights_mb"] + out["feature_cache_mb"] + out["encoder_peak_mb"] + out["query_mb"]
    return out


def live_cams(cfg: Any, long_side: int = LIVE_LONG_SIDE) -> list[PinholeCam]:
    """Rig cameras from cameras.yaml at the live input long side."""
    from python.perception.occ_geometry import rig_cams

    return [c.scaled(long_side) for c in rig_cams(cfg)]


class CheckpointMismatch(ValueError):
    """A checkpoint's width/depth does not match the net it is loaded into."""


def net_spec(net: Any) -> dict[str, Any]:
    """The knobs that shape the weights. Any change here forces a retrain."""
    return {"channels": list(net.channels), "feat": int(net.feat_dim), "hidden": int(net.hidden),
            "levels": list(LEVELS), "classes": OCC_CLASS_COUNT, "pe_freqs": PE_FREQS, "ctx": bool(getattr(net, "ctx", False))}


def save_checkpoint(net: Any, path: Any, **meta: Any) -> None:
    """Weights + ``net_spec``. Cameras, input size and grid are NOT stored as
    weights (they are non-persistent buffers): those knobs stay loadable."""
    torch.save({"spec": net_spec(net), "state_dict": net.state_dict(), "meta": dict(meta)}, path)


def load_checkpoint(net: Any, path: Any) -> dict[str, Any]:
    """Strict load into ``net``. Raises ``CheckpointMismatch`` naming the knobs
    that differ (width / depth / levels / classes / PE), never a partial load."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "spec" not in ck or "state_dict" not in ck:
        raise CheckpointMismatch(f"{path}: not an occ_sparse checkpoint (no spec/state_dict)")
    want, got = net_spec(net), dict(ck["spec"])
    diff = {k: (got.get(k), want[k]) for k in want if got.get(k) != want[k]}
    if diff:
        raise CheckpointMismatch(f"{path}: checkpoint vs net differ in {diff}; width/depth changes need a retrain")
    try:
        net.load_state_dict(ck["state_dict"], strict=True)
    except RuntimeError as e:  # same spec but different tensors: still refuse
        raise CheckpointMismatch(f"{path}: {e}") from e
    return dict(ck.get("meta") or {})
