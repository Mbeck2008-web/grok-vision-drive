"""Multi-camera fully sparse occupancy net (AI-Day-2022 style, toy scale).

No dense BEV and no dense 3-D volume exist anywhere after the image
backbone. The only dense tensors are the per-camera image feature maps of
the current tick (not cached across ticks). Every later stage works on an
explicit, absolutely capped set of active cells.

Pipeline per tick (camera-only at runtime):

1. **Shared backbone.** Every *fresh* camera image goes through the same tiny
   CNN (one set of weights for all cameras): 1/8 features for pooling, 1/16
   features + a depth-bin head (``depth_bins`` log-spaced bins, 1-80 m) for
   lifting. Camera freshness: ``w = exp(-age / age_tau_s)`` for
   ``0 <= age <= max_age_s`` else 0; a camera with w = 0 is never read (its
   pixels are replaced by zeros and it is skipped).
2. **Sparse depth-binned lifting.** Each 1/16 ray keeps only its top
   ``bins_per_ray`` depth bins whose probability is >= ``depth_conf`` (k = 1:
   one point at the expected depth). A camera keeps at most
   ``q_max // n_fresh`` rays (absolute cap). Lifted points are scatter-meaned
   into 1 m "mid" voxels (2x2x2 fine voxels) by hashed key -> the active set.
   Active count <= cams x rays x bins_per_ray and <= q_max, independent of
   the grid size.
3. **Multi-camera fusion (sparse queries).** Each active cell is projected
   into every fresh camera and pools its 1/8 features with a visibility x
   freshness weight; the lifted and pooled features are fused per cell.
4. **Temporal memory (sparse).** The memory is a list of cells (xyz, feature,
   support weight), at most ``mem_max``. Each tick it is rigidly warped into
   the current ego frame (the cell *positions* move; no map is resampled),
   re-keyed, merged with the current active set by key (a GRU cell fuses
   features where both exist), decayed by ``support_decay`` and pruned.
   A clock jump / repeated / backwards stamp / gap > ``max_gap_s`` resets it
   (runtime). Two submanifold sparse convolutions (hashed 11-neighbour
   gather) mix neighbouring active cells.
5. **Ego motion.** Default: camera-only sparse direct VO on ground points.
   A fixed set of ``VO_GROUND_N`` ground-plane points (ego frame, z=0,
   x in [-3, 3] m, y in [3.5, 30] m) is projected into one forward camera
   (first fresh of main, wide, narrow, ...) and its grey values sampled;
   only those N samples (+ validity) are cached for the next tick, so **no
   dense tensor crosses ticks**. Next tick, candidate motions (dx, dy, dyaw)
   move the points with ``warp_cells`` (the same warp the memory uses) and
   the robust (trimmed) photometric cost is minimised coarse-to-fine.
   Confidence / covariance come from the cost contrast. Flat-ground
   assumption; no learned parameters. (Two learned VO heads -- a sparse
   soft-match + Procrustes one and an image-plane conv regressor -- did not
   track speed on the synthetic clips; see docs/specs/occ_sparse.md.)
   Wheel speed / IMU yaw rate replace it only with ``ego_from_vehicle``
   (``--ego-from-vehicle``, default off). No GT pose ever enters.
6. **Heads.** Per active cell: mid occupancy and object centre/yaw/offset.
   The top ``k_max`` occupied cells expand to their 8 fine 0.5 m voxels;
   the fine head predicts occupancy, semantic class and ground flow. Output
   is a sparse list of occupied voxels.

Lanes, edges, curbs and stop lines are polylines derived from the sparse
cells (``polylines``); motion state comes only from the multi-frame tracker
(``tracks``), never from the flow head.
"""
from __future__ import annotations

import contextlib
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from python.occ.camera import CameraModel
from python.occ.classes import N_CLASSES, UNKNOWN
from python.occ.grid import DEFAULT_GRID, VoxelGrid

N_FREQ = 6
POSENC_DIM = 3 + 3 * 2 * N_FREQ
MID = 2  # mid (active-set) voxel = MID x MID x MID fine voxels (1 m at 0.5 m fine)
D_MIN, D_MAX = 1.0, 80.0
VO_CAM_ORDER = ("main", "wide", "narrow", "pillarL", "pillarR", "repeatL", "repeatR", "rear")
VO_GROUND_X = 16
VO_GROUND_Y = 24
VO_GROUND_N = VO_GROUND_X * VO_GROUND_Y  # absolute cap on VO samples per tick
VO_MIN_VALID = 24
VO_TRIM = 0.8
VO_MIN_CONF = 0.3  # below this the constant-motion prior (last accepted step) is used instead
VO_MIN_CONTRAST = 4.0 / 255.0  # local grey range for a reference point to count as textured  # keep the best 80% of per-point residuals (occluders, moving objects)
VO_DY_RANGE = (-0.6, 3.0)  # coarse forward search per tick (m): -6..30 m/s at 10 Hz


PROPOSAL_X = tuple(-7.5 + 3.0 * i for i in range(6))
PROPOSAL_Y = tuple(2.5 + 3.0 * i for i in range(16))
PROPOSAL_Z = (-0.5, 0.5)
P_MAX = len(PROPOSAL_X) * len(PROPOSAL_Y) * len(PROPOSAL_Z)  # 192: absolute cap on fixed unseen-cell proposals


def proposal_points() -> torch.Tensor:
    """Fixed ego-relative query set (P_MAX,3) for cells no camera ray hits: a coarse lattice around and ahead
    of the ego at road level. A pure function of nothing -- no map, no GT, no scene input."""
    pts = [(x, y, z) for z in PROPOSAL_Z for y in PROPOSAL_Y for x in PROPOSAL_X]
    return torch.tensor(pts, dtype=torch.float32)


def vo_ground_points() -> torch.Tensor:
    """Fixed ego-frame ground points (N,3) used by the direct VO (z = 0 is the road plane)."""
    xs = torch.linspace(-3.0, 3.0, VO_GROUND_X)
    ys = torch.logspace(math.log10(3.5), math.log10(30.0), VO_GROUND_Y)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1), torch.zeros(VO_GROUND_N)], 1)
CUDA_CONTEXT_GB = 0.6
ALL_CAMS = ("narrow", "main", "wide", "pillarL", "pillarR", "repeatL", "repeatR", "rear")
PRESET_MAIN_WIDE = ("main", "wide")
PRESET_MAIN_WIDE_NARROW = ("main", "wide", "narrow")
CKPT_KIND = "gvd_occ_sparse_v3"
MEMORY_VIZ_N = 256  # warped memory points exported for the top-down view
PANEL_N = 256       # projected cells per camera panel
# 11 submanifold neighbours: 3x3 in the ground plane + one cell above and below.
NEIGHBOURS = [(dx, dy, 0) for dx in (-1, 0, 1) for dy in (-1, 0, 1)] + [(0, 0, -1), (0, 0, 1)]


def profile_preset(name: str, **kw: Any) -> "OccProfile":
    """Named rigs. ``full8``: all cameras at 640. ``main_wide``: the minimum live rig.
    ``main_wide_narrow``: next step up. ``front_main_hi``: front cluster without narrow,
    main at 960 long side (Michael's knob), wide at 640."""
    presets = {
        "full8": dict(cams=ALL_CAMS),
        "main_wide": dict(cams=PRESET_MAIN_WIDE),
        "main_wide_narrow": dict(cams=PRESET_MAIN_WIDE_NARROW),
        "front_main_hi": dict(cams=PRESET_MAIN_WIDE, res_overrides=(("main", 960),)),
    }
    if name not in presets:
        raise KeyError(f"unknown profile {name!r}; known: {sorted(presets)}")
    return OccProfile(**{**presets[name], **kw})


class CheckpointError(ValueError):
    pass


@dataclass(frozen=True)
class OccArch:
    """Knobs that change weight shapes: a change forces a retrain."""

    widths: tuple[int, ...] = (16, 32, 64, 96)
    feat: int = 32
    hidden: int = 32
    n_classes: int = N_CLASSES
    depth_bins: int = 32


@dataclass(frozen=True)
class OccProfile:
    """Run-time knobs: any value keeps a checkpoint loadable (same OccArch).

    ``input_long_side`` is the first lever when the rig cannot keep same-tick
    sync (Michael's rule: drop resolution before giving up sync). ``infer_dtype``
    fp16 saves memory only on Pascal (fp16 math there is ~1/64 rate); fp32 is
    the default. Every stage cap is an absolute count, never a grid fraction.
    """

    cams: tuple[str, ...] = ALL_CAMS
    input_long_side: int = 640
    fallback_long_sides: tuple[int, ...] = (480, 320)
    infer_dtype: str = "fp32"
    max_age_s: float = 0.35
    age_tau_s: float = 0.25
    max_gap_s: float = 0.5
    support_decay: float = 0.7
    bins_per_ray: int = 1        # top-k depth bins kept per ray
    depth_conf: float = 0.05     # a kept bin needs at least this probability
    q_max: int = 2048            # active cells after lifting and after the temporal merge
    mem_max: int = 1024          # temporal memory cells
    k_max: int = 320             # active cells expanded to fine queries (x8)
    mid_thr: float = 0.3
    occ_thr: float = 0.5
    grid: VoxelGrid = field(default_factory=VoxelGrid)
    res_overrides: tuple[tuple[str, int], ...] = ()
    ego_from_vehicle: bool = False

    def long_side(self, cam_id: str) -> int:
        return int(dict(self.res_overrides).get(cam_id, self.input_long_side))

    def __post_init__(self) -> None:
        if self.infer_dtype not in ("fp32", "fp16"):
            raise ValueError("infer_dtype must be fp32 or fp16")
        if not (16 <= int(self.input_long_side) <= 4096):
            raise ValueError("input_long_side out of range")
        if not set(self.cams) <= set(ALL_CAMS):
            raise ValueError(f"unknown cameras: {sorted(set(self.cams) - set(ALL_CAMS))}")
        if not (self.max_age_s >= 0 and self.age_tau_s > 0 and self.max_gap_s > 0):
            raise ValueError("ages must be non-negative / positive")
        if not (self.bins_per_ray >= 1 and self.q_max >= 1 and self.mem_max >= 0 and self.k_max >= 0):
            raise ValueError("stage caps must be positive")


def depth_bin_centers(n: int) -> torch.Tensor:
    """Log-spaced z-depth bin centres (metres)."""
    return torch.exp(torch.linspace(math.log(D_MIN), math.log(D_MAX), n))


def camera_weight(age: torch.Tensor, max_age_s: float, tau_s: float) -> torch.Tensor:
    """Per-camera freshness weight in [0, 1]; 0 for stale, future-stamped or non-finite ages."""
    a = torch.nan_to_num(age.float(), nan=-1.0, posinf=-1.0, neginf=-1.0)
    ok = (a >= 0) & (a <= max_age_s)
    return torch.where(ok, torch.exp(-a.clamp(min=0.0) / tau_s), torch.zeros_like(a))


MAX_STEP_M = 6.0
MAX_STEP_RAD = 0.5


def sanitize_motion(motion: torch.Tensor) -> torch.Tensor:
    """Non-finite -> 0, then clamp to the per-tick bound, so a bad estimate cannot NaN or fling the memory."""
    m = torch.nan_to_num(motion, nan=0.0, posinf=0.0, neginf=0.0)
    lim = torch.tensor([MAX_STEP_M, MAX_STEP_M, MAX_STEP_RAD], dtype=m.dtype, device=m.device)
    return torch.maximum(torch.minimum(m, lim), -lim)


def ego_motion(speed_mps: float, yaw_rate: float, dt: float) -> tuple[float, float, float]:
    """Ego displacement (dx, dy) in the previous ego frame and the heading change, for one tick."""
    dyaw = yaw_rate * dt
    d = speed_mps * dt
    return -d * math.sin(dyaw / 2.0), d * math.cos(dyaw / 2.0), dyaw


def warp_cells(xyz: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
    """Move previous-frame cell positions (N,3) into the current ego frame.

    ``motion`` (3,) = (dx, dy, dyaw): a static point at current p sits at
    ``R(dyaw) p + (dx, dy)`` in the previous frame, so p = R^T (q - t).
    Only positions move; nothing is resampled.
    """
    dx, dy, dyaw = motion[0], motion[1], motion[2]
    c, s = torch.cos(dyaw), torch.sin(dyaw)
    qx, qy = xyz[:, 0] - dx, xyz[:, 1] - dy
    return torch.stack([c * qx + s * qy, -s * qx + c * qy, xyz[:, 2]], dim=1)


def procrustes_2d(p: torch.Tensor, q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted rigid fit q ~ R(yaw) p + t in 2-D; returns (tx, ty, yaw)."""
    w = w / w.sum().clamp(min=1e-9)
    mp, mq = (w[:, None] * p).sum(0), (w[:, None] * q).sum(0)
    P, Q = p - mp, q - mq
    s = (w * (P[:, 0] * Q[:, 1] - P[:, 1] * Q[:, 0])).sum()
    c = (w * (P[:, 0] * Q[:, 0] + P[:, 1] * Q[:, 1])).sum()
    yaw = torch.atan2(s, c + 1e-12)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    t = mq - torch.stack([cy * mp[0] - sy * mp[1], sy * mp[0] + cy * mp[1]])
    return torch.stack([t[0], t[1], yaw])


def _cbr(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def posenc(xyz: torch.Tensor, grid: VoxelGrid = DEFAULT_GRID) -> torch.Tensor:
    lo = torch.tensor([grid.x_min, grid.y_min, grid.z_min], dtype=xyz.dtype, device=xyz.device)
    hi = torch.tensor([grid.x_max, grid.y_max, grid.z_max], dtype=xyz.dtype, device=xyz.device)
    u = (xyz - lo) / (hi - lo) * 2 - 1
    freqs = (2.0 ** torch.arange(N_FREQ, dtype=xyz.dtype, device=xyz.device)) * math.pi
    ang = u[..., None] * freqs
    return torch.cat([u, torch.sin(ang).flatten(-2), torch.cos(ang).flatten(-2)], dim=-1)


def _rows(n: int, width: int, budget: int) -> int:
    """Row chunk so a (rows x width) temporary stays under ``budget`` elements."""
    return max(1, min(n, budget // max(1, width) - 1))


class SubMConv(nn.Module):
    """Submanifold sparse conv on hashed cell keys: out_i = sum_o W_o x_{nbr(i,o)}; inactive neighbours are 0."""

    def __init__(self, c: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(len(NEIGHBOURS), c, c) / math.sqrt(c * len(NEIGHBOURS)))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x: torch.Tensor, nbr: torch.Tensor) -> torch.Tensor:
        out = x.new_zeros(x.shape) + self.bias.to(x.dtype)
        for o in range(nbr.shape[1]):
            idx = nbr[:, o]
            g = torch.where((idx >= 0)[:, None], x[idx.clamp(min=0)], torch.zeros_like(x))
            out = out + g @ self.weight[o].to(x.dtype)
        return out


class OccSparseNet(nn.Module):
    def __init__(self, arch: OccArch | None = None, profile: OccProfile | None = None):
        super().__init__()
        self.arch = arch or OccArch()
        self.profile = profile or OccProfile()
        a = self.arch
        w = list(a.widths)
        if len(w) != 4:
            raise ValueError("widths must have 4 stages")
        C = a.feat
        self.feat = C
        self.stem = _cbr(3, w[0], 2)
        self.s4 = nn.Sequential(_cbr(w[0], w[1], 2), _cbr(w[1], w[1], 1))
        self.s8 = nn.Sequential(_cbr(w[1], w[2], 2), _cbr(w[2], w[2], 1))
        self.s16 = nn.Sequential(_cbr(w[2], w[3], 2), _cbr(w[3], w[3], 1))
        self.lat8 = nn.Conv2d(w[2], C, 1)
        self.lat16 = nn.Conv2d(w[3], C, 1)
        self.depth_head = nn.Conv2d(C, a.depth_bins, 1)
        self.register_buffer("bins", depth_bin_centers(a.depth_bins), persistent=False)
        self.cell_in = nn.Linear(2 * C + POSENC_DIM + 1, C)
        self.gru_zr = nn.Linear(2 * C + 2, 2 * C)
        self.gru_h = nn.Linear(2 * C + 2, C)
        self.sconv1 = SubMConv(C)
        self.sconv2 = SubMConv(C)
        # Camera-only VO on [previous, current] 1/8 maps of one forward camera -> (dx, dy, dyaw, 3 x log-variance).
        self.mid_head = nn.Linear(C, 1)
        self.obj_head = nn.Linear(C, 5)  # centre logit, cos yaw, sin yaw, dx, dy
        din = 2 * C + POSENC_DIM + 1
        self.mlp = nn.Sequential(nn.Linear(din, a.hidden), nn.ReLU(inplace=True), nn.Linear(a.hidden, a.hidden), nn.ReLU(inplace=True))
        self.occ_head = nn.Linear(a.hidden, 1)
        self.cls_head = nn.Linear(a.hidden, a.n_classes)
        self.flow_head = nn.Linear(a.hidden, 2)
        self._in_backbone = False  # True only while dense image tensors are being computed

    @contextlib.contextmanager
    def image_stage(self):
        """Marks code that handles dense per-camera image tensors (input prep + backbone)."""
        prev = self._in_backbone
        self._in_backbone = True
        try:
            yield
        finally:
            self._in_backbone = prev

    @property
    def grid(self) -> VoxelGrid:
        return self.profile.grid

    @property
    def budget(self) -> int:
        """Element budget for any post-backbone temporary: half the full fine grid."""
        return max(1024, self.grid.n_dense // 2)

    # --------------------------------------------------------------- keys
    def mid_shape(self) -> tuple[int, int, int]:
        nx, ny, nz = self.grid.shape
        return -(-nx // MID), -(-ny // MID), -(-nz // MID)

    def unkey_center(self, keys: torch.Tensor) -> torch.Tensor:
        """Centre (N,3) float32 of the mid cells with these keys."""
        g = self.grid
        lo = torch.tensor([g.x_min, g.y_min, g.z_min], dtype=torch.float32, device=keys.device)
        return lo + (self.unkey(keys).float() + 0.5) * (g.voxel * MID)

    def mid_index(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.grid
        lo = torch.tensor([g.x_min, g.y_min, g.z_min], dtype=xyz.dtype, device=xyz.device)
        hi = torch.tensor([g.x_max, g.y_max, g.z_max], dtype=xyz.dtype, device=xyz.device)
        ok = torch.isfinite(xyz).all(dim=1) & (xyz >= lo).all(dim=1) & (xyz < hi).all(dim=1)
        idx = torch.floor((torch.where(ok[:, None], xyz, lo) - lo) / (g.voxel * MID)).long()
        mx, my, mz = self.mid_shape()
        idx = torch.minimum(idx.clamp(min=0), torch.tensor([mx - 1, my - 1, mz - 1], device=xyz.device))
        return idx, ok

    def key(self, idx: torch.Tensor) -> torch.Tensor:
        mx, my, mz = self.mid_shape()
        return (idx[:, 0] * my + idx[:, 1]) * mz + idx[:, 2]

    def unkey(self, k: torch.Tensor) -> torch.Tensor:
        mx, my, mz = self.mid_shape()
        return torch.stack([k // (my * mz), (k // mz) % my, k % mz], dim=1)

    @staticmethod
    def lookup(keys: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """Index of each query key in ``keys`` (-1 if absent). Sorted-key hash; no dense table."""
        if keys.numel() == 0 or query.numel() == 0:
            return torch.full(query.shape, -1, dtype=torch.long, device=query.device)
        sk, order = torch.sort(keys)
        pos = torch.searchsorted(sk, query).clamp(max=sk.numel() - 1)
        hit = sk[pos] == query
        return torch.where(hit, order[pos], torch.full_like(pos, -1))

    def neighbours(self, keys: torch.Tensor) -> torch.Tensor:
        idx = self.unkey(keys)
        mx, my, mz = self.mid_shape()
        lim = torch.tensor([mx, my, mz], device=keys.device)
        cols = []
        for off in NEIGHBOURS:
            n = idx + torch.tensor(off, device=keys.device)
            ok = ((n >= 0) & (n < lim)).all(dim=1)
            j = self.lookup(keys, self.key(n.clamp(min=0)))
            cols.append(torch.where(ok, j, torch.full_like(j, -1)))
        return torch.stack(cols, dim=1)

    # ----------------------------------------------------------- backbone
    def backbone(self, img: torch.Tensor):
        x = self.stem(img)
        x = self.s4(x)
        f8 = self.s8(x)
        f16 = self.s16(f8)
        l16 = self.lat16(f16)
        p8 = self.lat8(f8) + F.interpolate(l16, size=f8.shape[-2:], mode="bilinear", align_corners=False)
        return p8, self.depth_head(torch.relu(p8))

    def encode(self, images: Mapping[str, torch.Tensor], cams: Mapping[str, CameraModel], weights: Mapping[str, float]):
        """Per fresh camera (one sample): 1/8 pooling map, 1/16 lift features, depth logits.

        These per-camera image-plane maps are the only dense tensors in the
        net; they live for this tick only. A camera with weight 0 is skipped
        entirely (its pixels are never read)."""
        out = {}
        with self.image_stage():
            for cid, img in images.items():
                if cid not in self.profile.cams or cid not in cams or not (weights.get(cid, 0.0) > 0.0):
                    continue
                x = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)[None]
                p8, dl = self.backbone(x)
                out[cid] = {"p8": p8, "lf": p8[0], "dl": dl[0], "cam": cams[cid].scaled(img.shape[-1], img.shape[-2]), "w": float(weights[cid]),
                            "gray": x.float().mean(1, keepdim=True).clamp(0.0, 1.0)}
        return out

    # ------------------------------------------------------------- lifting
    def lift(self, enc) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Sparse depth-binned lifting: per ray keep the top ``bins_per_ray`` bins above ``depth_conf``.

        Rays are processed in chunks and each camera keeps a running top
        ``q_max // n_fresh`` by bin probability, so no temporary ever holds
        every ray x every bin x features."""
        p = self.profile
        C = self.feat
        if not enc:
            z = next(self.parameters())
            return z.new_zeros(0, 3), z.new_zeros(0, C), z.new_zeros(0), {"rays": 0, "lifted": 0}
        per_cam = max(1, p.q_max // len(enc))
        pts, feats, ws = [], [], []
        rays = 0
        for e in enc.values():
            cam, dl, lf = e["cam"], e["dl"], e["lf"]
            D, h, w = dl.shape
            R = h * w
            rays += R
            k = min(p.bins_per_ray, D)
            bins = self.bins.float()
            basis = torch.as_tensor(cam.basis(), dtype=torch.float32, device=dl.device)
            pos = torch.as_tensor(np.asarray(cam.pos), dtype=torch.float32, device=dl.device)
            with self.image_stage():  # flat views of this camera's dense image-plane maps (backbone outputs)
                dlf = dl.reshape(D, R)
                cf_all = lf.reshape(C, R)
            best_x = dl.new_zeros(0, 3, dtype=torch.float32)
            best_f = dl.new_zeros(0, C, dtype=torch.float32)
            best_p = dl.new_zeros(0, dtype=torch.float32)
            step = _rows(R, D + C + 8, self.budget)
            for r0 in range(0, R, step):
                r = torch.arange(r0, min(R, r0 + step), device=dl.device)
                prob = torch.softmax(dlf[:, r].float(), dim=0)
                top_p, top_i = torch.topk(prob, k, dim=0)  # (k, r)
                u = ((r % w).float() + 0.5) * (cam.width / w)
                v = ((r // w).float() + 0.5) * (cam.height / h)
                for j in range(k):
                    z = (prob * bins[:, None]).sum(0) if k == 1 else bins[top_i[j]]  # k=1: expected depth, one point per ray
                    keep = top_p[j] >= p.depth_conf
                    if not bool(keep.any()):
                        continue
                    zk = z[keep]
                    cxyz = torch.stack([(u[keep] - cam.width / 2.0) / cam.f * zk, (v[keep] - cam.height / 2.0) / cam.f * zk, zk], dim=1)
                    xyz = cxyz @ basis + pos
                    _, ok = self.mid_index(xyz.detach())
                    rk = r[keep][ok]
                    pk = top_p[j][keep][ok]
                    best_x = torch.cat([best_x, xyz[ok]])
                    best_f = torch.cat([best_f, cf_all[:, rk].t().float() * pk[:, None]])
                    best_p = torch.cat([best_p, pk])
                    if best_p.numel() > per_cam:
                        sel = torch.topk(best_p, per_cam).indices
                        best_x, best_f, best_p = best_x[sel], best_f[sel], best_p[sel]
            pts.append(best_x)
            feats.append(best_f)
            ws.append(torch.full((best_p.numel(),), e["w"], dtype=torch.float32, device=dl.device))
        pts, feats, ws = torch.cat(pts), torch.cat(feats), torch.cat(ws)
        if pts.shape[0] > p.q_max:
            sel = torch.topk(ws, p.q_max).indices
            pts, feats, ws = pts[sel], feats[sel], ws[sel]
        return pts, feats, ws, {"rays": rays, "lifted": int(pts.shape[0])}

    def pool_cams(self, enc, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Visibility x freshness weighted mean of 1/8 camera features at sparse points (N,C), and weight sum (N,)."""
        N = pts.shape[0]
        acc = pts.new_zeros(N, self.feat)
        wsum = pts.new_zeros(N)
        for e in enc.values():
            cam, fm = e["cam"], e["p8"]
            basis = torch.as_tensor(cam.basis(), dtype=pts.dtype, device=pts.device)
            pos = torch.as_tensor(np.asarray(cam.pos), dtype=pts.dtype, device=pts.device)
            c = (pts - pos) @ basis.T
            z = c[:, 2]
            zs = torch.where(z > 0.1, z, torch.ones_like(z))
            u = cam.f * c[:, 0] / zs + cam.width / 2.0
            v = cam.f * c[:, 1] / zs + cam.height / 2.0
            vis = (z > 0.1) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
            g = torch.stack([u / cam.width * 2 - 1, v / cam.height * 2 - 1], dim=-1).view(1, N, 1, 2)
            s = F.grid_sample(fm, g.to(fm.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)
            s = s.view(self.feat, N).t().to(pts.dtype)
            ww = vis.to(pts.dtype) * e["w"]
            acc = acc + torch.where(ww[:, None] > 0, s * ww[:, None], torch.zeros_like(s))
            wsum = wsum + ww
        return acc / wsum.clamp(min=1e-6)[:, None], wsum

    def voxelize(self, pts, feats, ws):
        """Scatter-mean lifted points into hashed mid cells: keys, mean xyz, mean feature, max weight."""
        idx, _ = self.mid_index(pts.detach())
        keys = self.key(idx)
        uk, inv = torch.unique(keys, return_inverse=True)
        U = uk.numel()
        cnt = pts.new_zeros(U).index_add_(0, inv, torch.ones_like(ws))
        xyz = pts.new_zeros(U, 3).index_add_(0, inv, pts) / cnt.clamp(min=1)[:, None]
        f = feats.new_zeros(U, self.feat).index_add_(0, inv, feats) / cnt.clamp(min=1)[:, None]
        w = pts.new_zeros(U).scatter_reduce_(0, inv, ws, reduce="amax", include_self=False)
        return uk, xyz, f, w

    # ------------------------------------------------------------- one tick
    def sparse_step(self, images, cams, ages: Mapping[str, float], motion=None, memory=None) -> dict[str, Any]:
        """One sample, one tick. ``ages``: seconds per camera. ``memory``: previous tick's sparse memory."""
        p = self.profile
        weights = {cid: float(camera_weight(torch.tensor([float(ages[cid]) if cid in ages else math.nan]), p.max_age_s, p.age_tau_s)[0]) for cid in images}
        tm: dict[str, float] = {}
        t0 = time.perf_counter()
        enc = self.encode(images, cams, weights)
        t1 = time.perf_counter()
        tm["backbone"] = (t1 - t0) * 1e3
        pts, feats, ws, lstat = self.lift(enc)
        keys, xyz, f, w = self.voxelize(pts, feats, ws)
        pooled, seen = self.pool_cams(enc, xyz) if keys.numel() else (f, w)
        wd = self.cell_in.weight.dtype
        if keys.numel():
            cur = torch.cat([torch.relu(self.cell_in(torch.cat([f[i:i + n].to(wd), pooled[i:i + n].to(wd), posenc(xyz[i:i + n], self.grid).to(wd),
                                                                (seen[i:i + n] > 0).to(wd)[:, None]], 1)))
                             for i, n in self._chunks(keys.numel(), 3 * self.feat + POSENC_DIM + 1)])
        else:
            cur = f.to(wd)
        # ---- camera-only VO: soft match against the previous tick's cells
        vo, vo_conf, vo_cov = None, 0.0, None
        t2 = time.perf_counter()
        tm["lift_query"] = (t2 - t1) * 1e3
        vo_cid = next((c for c in VO_CAM_ORDER if c in enc), None)
        if memory is not None and vo_cid is not None and int(memory["vo_cam"]) == VO_CAM_ORDER.index(vo_cid) \
                and memory["vo_ref"].shape[0] == VO_GROUND_N:
            vo, vo_conf, vo_cov = self.vo_direct(memory["vo_ref"], memory["vo_ok"], enc[vo_cid])
        vo_fallback = False
        if memory is not None and (vo is None or vo_conf < VO_MIN_CONF):
            last = memory.get("vo_last")
            if last is not None and last.numel() == 3 and bool(torch.isfinite(last).all()):
                # weak / no photometric fix: constant-motion prior from the last accepted VO step, flagged + low confidence
                vo, vo_conf, vo_cov, vo_fallback = last.float().clone(), 0.5 * float(vo_conf), [1.0, 1.0, 0.01], True
        t3 = time.perf_counter()
        tm["vo_head"] = (t3 - t2) * 1e3
        if memory is not None and motion is None and not p.ego_from_vehicle:
            motion = vo
        mem_keys = keys.new_zeros(0)
        mot = sanitize_motion(motion.to(xyz.dtype).reshape(3)) if memory is not None and motion is not None else None
        if mot is not None and memory["xyz"].shape[0]:
            mxyz = warp_cells(memory["xyz"].to(xyz.dtype), mot)
            midx, mok = self.mid_index(mxyz.detach())
            mw = memory["w"].to(xyz.dtype) * p.support_decay
            keep = mok & (mw >= 0.05)
            mxyz, mfeat, mw, mem_keys = mxyz[keep], memory["feat"].to(cur.dtype)[keep], mw[keep], self.key(midx[keep])
        else:
            mfeat = cur.new_zeros(0, self.feat)
            mxyz = xyz.new_zeros(0, 3)
            mw = xyz.new_zeros(0)
        # ---- unseen-cell proposals: fixed ego-relative lattice (cap P_MAX); features from camera pooling only
        ppts = proposal_points().to(device=xyz.device, dtype=xyz.dtype)
        pidx, pok = self.mid_index(ppts)
        pkeys = torch.unique(self.key(pidx[pok]))
        pxyz = self.unkey_center(pkeys).to(xyz.dtype)
        if pkeys.numel():
            ppool, pseen = self.pool_cams(enc, pxyz)
            pcur = torch.relu(self.cell_in(torch.cat([torch.zeros(pkeys.numel(), self.feat, dtype=wd, device=xyz.device), ppool.to(wd),
                                                      posenc(pxyz, self.grid).to(wd), (pseen > 0).to(wd)[:, None]], 1)))
        else:
            pcur = cur.new_zeros(0, self.feat)
        # ---- sparse temporal merge by key (active this tick | carried by memory | proposed)
        allk = torch.cat([keys, mem_keys, pkeys])
        uk, inv = torch.unique(allk, return_inverse=True)
        U = uk.numel()
        ic, im, ip = inv[: keys.numel()], inv[keys.numel(): keys.numel() + mem_keys.numel()], inv[keys.numel() + mem_keys.numel():]
        fc = cur.new_zeros(U, self.feat)
        if pkeys.numel():
            fc = fc.index_copy(0, ip, pcur.to(fc.dtype))
        if keys.numel():
            fc = fc.index_copy(0, ic, cur)
        ac = xyz.new_zeros(U).index_fill_(0, ic, 1.0) if keys.numel() else xyz.new_zeros(U)
        mc = cur.new_zeros(U).index_add_(0, im, torch.ones_like(mw, dtype=cur.dtype))
        fm = cur.new_zeros(U, self.feat).index_add_(0, im, mfeat) / mc.clamp(min=1)[:, None]
        am = (mc > 0).to(xyz.dtype)
        wm = xyz.new_zeros(U).scatter_reduce_(0, im, mw.to(xyz.dtype), reduce="amax", include_self=True) if mw.numel() else xyz.new_zeros(U)
        mc32 = xyz.new_zeros(U).index_add_(0, im, torch.ones_like(mw, dtype=xyz.dtype))
        xm = xyz.new_zeros(U, 3).index_add_(0, im, mxyz.to(xyz.dtype)) / mc32.clamp(min=1)[:, None]
        xc = xyz.new_zeros(U, 3).index_copy_(0, ic, xyz) if keys.numel() else xyz.new_zeros(U, 3)
        xp = xyz.new_zeros(U, 3).index_copy_(0, ip, pxyz) if pkeys.numel() else xyz.new_zeros(U, 3)
        cxyz = torch.where(ac[:, None] > 0, xc, torch.where(am[:, None] > 0, xm, xp))
        state = torch.cat([self._gru(fc[i:i + n], fm[i:i + n], ac[i:i + n], am[i:i + n]) for i, n in self._chunks(U, 4 * self.feat + 4)]) if U else fc
        cw = torch.maximum(ac, wm)
        if U > p.q_max:
            sel = torch.topk(cw + 1e-3 * ac + 1e-4 * am, p.q_max).indices
            uk, state, cw, cxyz, ac, am = uk[sel], state[sel], cw[sel], cxyz[sel], ac[sel], am[sel]
            U = p.q_max
        if U:
            nbr = self.neighbours(uk)
            # proposal-only cells (no lift, no memory) read their neighbours but never feed observed / carried cells
            pon = (ac == 0) & (am == 0)
            nbr = torch.where((nbr >= 0) & pon[nbr.clamp(min=0)], torch.full_like(nbr, -1), nbr)
            h = torch.relu(self.sconv1(state, nbr)) + state
            h = torch.relu(self.sconv2(h, nbr)) + h
        else:
            h = state
        t4 = time.perf_counter()
        tm["memory"] = (t4 - t3) * 1e3
        mid_logit = self.mid_head(h).squeeze(-1)
        obj = self.obj_head(h)
        tm["mid_obj_heads"] = (time.perf_counter() - t4) * 1e3
        # ---- next tick's memory (sparse, capped)
        mk = torch.topk(cw * torch.sigmoid(mid_logit.detach().float()).to(cw.dtype) + 1e-3 * cw, min(p.mem_max, U)).indices if U else uk[:0]
        if vo_cid is not None:
            vo_ref, vo_ok = self.vo_sample(enc[vo_cid], vo_ground_points(), contrast=True)
            vo_cam = torch.tensor(VO_CAM_ORDER.index(vo_cid))
        else:
            vo_ref, vo_ok, vo_cam = torch.zeros(0), torch.zeros(0, dtype=torch.bool), torch.tensor(-1)
        vo_last = vo.detach().float() if vo is not None else torch.zeros(0)
        new_mem = {"xyz": cxyz[mk], "feat": state[mk], "w": cw[mk], "vo_ref": vo_ref, "vo_ok": vo_ok, "vo_cam": vo_cam, "vo_last": vo_last}
        stats = {"rays": lstat["rays"], "lifted_points": lstat["lifted"], "active_cells": int(keys.numel()), "memory_cells_in": int(mem_keys.numel()),
                 "merged_cells": int(U), "memory_cells_out": int(mk.numel()), "vo_fallback": int(vo_fallback),
                 "proposal_cells": int(pkeys.numel()), "observed_cells": int((ac > 0).sum())}
        return {"enc": enc, "keys": uk, "xyz": cxyz, "h": h, "observed": ac, "carried": am * (ac == 0).to(am.dtype), "w": cw, "mid_logit": mid_logit, "obj": obj, "memory": new_mem,
                "vo": vo, "vo_logvar": None, "vo_conf": vo_conf, "vo_cov": vo_cov, "motion_used": mot, "stats": stats, "weights": weights, "timing_ms": tm}

    def _chunks(self, n: int, width: int):
        r = _rows(n, width, self.budget)
        return [(i, min(r, n - i)) for i in range(0, n, r)]

    def _gru(self, fc, fm, ac, am):
        ac, am = ac.to(fc.dtype), am.to(fc.dtype)
        x = torch.cat([fc, fm, ac[:, None], am[:, None]], 1)
        z, r = torch.sigmoid(self.gru_zr(x)).chunk(2, dim=1)
        hh = torch.tanh(self.gru_h(torch.cat([fc, r * fm, ac[:, None], am[:, None]], 1)))
        return (1 - z) * fm + z * hh

    @staticmethod
    def vo_sample(e: Mapping[str, Any], pts: torch.Tensor, contrast: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Grey values of one camera at ego-frame points (N,3) (bilinear) and their in-image validity (N,).

        ``contrast``: validity also requires local texture along v (max-min of samples at v +-1.5 / +-3 px >= VO_MIN_CONTRAST),
        so flat regions (a uniform car face, sky, blank road) never anchor the match."""
        cam, g = e["cam"], e["gray"]
        pts = pts.to(torch.float32)
        basis = torch.as_tensor(cam.basis(), dtype=pts.dtype)
        pos = torch.as_tensor(np.asarray(cam.pos), dtype=pts.dtype)
        c = (pts - pos) @ basis.T
        z = c[:, 2]
        zs = torch.where(z > 0.1, z, torch.ones_like(z))
        u = cam.f * c[:, 0] / zs + cam.width / 2.0
        v = cam.f * c[:, 1] / zs + cam.height / 2.0
        ok = (z > 0.1) & (u >= 1) & (u < cam.width - 1) & (v >= 1) & (v < cam.height - 1)
        gr = torch.stack([u / cam.width * 2 - 1, v / cam.height * 2 - 1], dim=-1).view(1, -1, 1, 2)
        val = F.grid_sample(g.to(pts.dtype), gr, mode="bilinear", padding_mode="border", align_corners=False).view(-1)
        if contrast:
            d = 1.5
            nb = []
            # vertical neighbours only: forward motion moves ground points mostly along v, so edges of
            # longitudinal lines / road borders (gradient along u, invariant to forward motion) must not
            # anchor the match (counterexample V9: they pinned dy at 0)
            for du, dv in ((0.0, d), (0.0, -d), (0.0, 2 * d), (0.0, -2 * d)):
                gg = torch.stack([(u + du) / cam.width * 2 - 1, (v + dv) / cam.height * 2 - 1], dim=-1).view(1, -1, 1, 2)
                nb.append(F.grid_sample(g.to(pts.dtype), gg, mode="bilinear", padding_mode="border", align_corners=False).view(-1))
            nb = torch.stack(nb)
            ok = ok & ((nb.max(0).values - nb.min(0).values) >= VO_MIN_CONTRAST)
        return val.detach(), ok.detach()

    def _vo_cost(self, ref, ok, e, cands: torch.Tensor) -> torch.Tensor:
        """Trimmed mean |grey residual| per candidate motion (K,3); inf when too few points stay visible."""
        pts = vo_ground_points()
        out = []
        for i, n in self._chunks(cands.shape[0], 4 * pts.shape[0]):
            cs = []
            for m in cands[i:i + n]:
                val, okc = self.vo_sample(e, warp_cells(pts, m))
                good = ok & okc
                r = (val - ref).abs()[good]
                if r.numel() < VO_MIN_VALID:
                    cs.append(torch.tensor(float("inf")))
                    continue
                k = max(VO_MIN_VALID, int(VO_TRIM * r.numel()))
                cs.append(torch.topk(r, k, largest=False).values.mean())
            out.append(torch.stack(cs))
        return torch.cat(out) if out else torch.zeros(0)

    def vo_direct(self, ref: torch.Tensor, ok: torch.Tensor, e: Mapping[str, Any]):
        """Camera-only sparse direct VO: (dx, dy, dyaw) minimising the photometric cost of the cached ground samples.

        Coarse 1-D forward search, then three 3x3x3 local refinements with halving steps.
        Returns motion (3,) float32 (no gradient), confidence in [0, 1], diagonal covariance."""
        with torch.no_grad():
            ref, ok = ref.float(), ok.bool()
            if int(ok.sum()) < VO_MIN_VALID:
                return None, 0.0, None
            lo, hi = VO_DY_RANGE
            dys = torch.arange(lo, hi + 1e-6, 0.1)
            coarse = torch.stack([torch.zeros_like(dys), dys, torch.zeros_like(dys)], 1)
            cc = self._vo_cost(ref, ok, e, coarse)
            if not torch.isfinite(cc).any():
                return None, 0.0, None
            b = int(torch.argmin(cc))
            best, bc = coarse[b].clone(), float(cc[b])
            step = torch.tensor([0.1, 0.05, 0.02])
            offs = torch.stack(torch.meshgrid(*(torch.tensor([-1.0, 0.0, 1.0]),) * 3, indexing="ij"), -1).reshape(-1, 3)
            for _ in range(3):
                cands = best[None] + offs * step[None]
                c = self._vo_cost(ref, ok, e, cands)
                k = int(torch.argmin(c))
                if float(c[k]) < bc:
                    best, bc = cands[k].clone(), float(c[k])
                step = step / 2
            fin = cc[torch.isfinite(cc)]
            med = float(fin.median())
            contrast = max(0.0, (med - bc) / (med + 1e-6))
            frac = float(ok.float().mean())
            # local sharpness: the minimum must stand out against +-0.15 m forward (counterexample V9: a flat
            # basin at small steps gave a wrong answer with conf 1.0 from the global contrast alone)
            nb = self._vo_cost(ref, ok, e, torch.stack([best + torch.tensor([0.0, 0.15, 0.0]), best - torch.tensor([0.0, 0.15, 0.0])]))
            nbf = nb[torch.isfinite(nb)]
            sharp = max(0.0, (float(nbf.min()) - bc) / (float(nbf.min()) + 1e-6)) if nbf.numel() else 0.0
            conf = float(min(1.0, 2.0 * contrast) * min(1.0, sharp / 0.25) * min(1.0, frac / 0.25))
            sxy = 0.05 / max(contrast, 0.02)
            cov = [sxy ** 2, sxy ** 2, (0.01 / max(contrast, 0.02)) ** 2]
            return best.float(), conf, cov

    # ---------------------------------------------------------- fine head
    def point_features(self, st: Mapping[str, Any], pts: torch.Tensor) -> torch.Tensor:
        """Fine-head features at sparse query points (N,3): containing active cell + pooled camera features."""
        N = pts.shape[0]
        out = []
        for i, n in self._chunks(N, 2 * self.feat + POSENC_DIM + 1 + self.arch.hidden):
            pp = pts[i:i + n]
            idx, ok = self.mid_index(pp.detach())
            j = self.lookup(st["keys"], self.key(idx))
            j = torch.where(ok, j, torch.full_like(j, -1))
            if "carried" in st and st["keys"].numel():  # proposal-only cells (no lift, no memory) lend no cell state to fine queries
                jj = j.clamp(min=0)
                backed = (st["observed"][jj] > 0) | (st["carried"][jj] > 0)
                j = torch.where(backed, j, torch.full_like(j, -1))
            hc = torch.where((j >= 0)[:, None], st["h"][j.clamp(min=0)], torch.zeros(n, self.feat, dtype=pp.dtype, device=pp.device)) if st["keys"].numel() else pp.new_zeros(n, self.feat)
            fc, wsum = self.pool_cams(st["enc"], pp)
            wd = self.occ_head.weight.dtype
            out.append(self.mlp(torch.cat([hc.to(wd), fc.to(wd), posenc(pp, self.grid).to(wd), (wsum > 0).to(wd)[:, None]], 1)))
        return torch.cat(out) if out else pts.new_zeros(0, self.arch.hidden)

    # ------------------------------------------------------------ training
    def forward(self, images, cams, ages, pts, *, motion=None, memory=None) -> dict[str, Any]:
        """Batch forward (per-sample sparse sets). ``images`` cid -> (B,3,H,W); ``ages`` cid -> (B,);
        ``pts`` (B,N,3) training queries; ``memory`` list of per-sample memories or None."""
        B = pts.shape[0]
        occ, cls, flow, vos, mems, cells, mots, lvs = [], [], [], [], [], [], [], []
        for b in range(B):
            with self.image_stage():
                im = {c: v[b] for c, v in images.items()}
            ag = {c: float(v[b]) for c, v in ages.items()}
            mo = None if motion is None else motion[b]
            me = None if memory is None else memory[b]
            st = self.sparse_step(im, cams, ag, mo, me)
            h = self.point_features(st, pts[b])
            occ.append(self.occ_head(h).squeeze(-1))
            cls.append(self.cls_head(h))
            flow.append(self.flow_head(h))
            vos.append(st["vo"])
            lvs.append(st["vo_logvar"])
            mots.append(st["motion_used"])
            mems.append(st["memory"])
            cells.append({k: st[k] for k in ("keys", "xyz", "mid_logit", "obj", "observed")} | {"dl": {c: e["dl"] for c, e in st["enc"].items()}})
        vo = torch.stack(vos) if all(v is not None for v in vos) else None
        vo_logvar = torch.stack(lvs) if vo is not None and all(v is not None for v in lvs) else None
        return {"occ_logit": torch.stack(occ), "cls_logit": torch.stack(cls), "flow": torch.stack(flow), "vo": vo, "vo_logvar": vo_logvar, "memory": mems, "cells": cells,
                "motion_used": mots}

    # --------------------------------------------------------------- infer
    @torch.no_grad()
    def infer(self, images, cams, ages, *, motion=None, memory=None) -> dict[str, Any]:
        """One tick, batch 1. Sparse readout; returns cells, objects, memory and per-stage counts."""
        p, g = self.profile, self.grid
        dtype = torch.float16 if p.infer_dtype == "fp16" else torch.float32
        dev = next(self.parameters()).device
        for cid, img in images.items():
            if img.dim() != 4 or img.shape[0] != 1 or img.shape[1] != 3:
                raise ValueError(f"{cid}: infer takes (1,3,H,W), got {tuple(img.shape)}")
            if img.shape[-1] < 16 or img.shape[-2] < 16:
                raise ValueError(f"{cid}: image too small {tuple(img.shape[-2:])}")
        if motion is not None and not p.ego_from_vehicle:
            raise ValueError("vehicle ego motion passed but profile.ego_from_vehicle is off (default: camera-only VO)")
        with self.image_stage():
            imgs = {cid: img.to(dev, dtype)[0] for cid, img in images.items()}
        mot = None if motion is None else torch.tensor(list(motion), dtype=dtype, device=dev)
        st = self.sparse_step(imgs, cams, {c: float(a) for c, a in ages.items()}, mot, memory)
        fresh = sorted(c for c, wt in st["weights"].items() if wt > 0 and c in p.cams and c in cams)
        stats = dict(st["stats"], fine_grid_dense=g.n_dense, mid_grid_dense=int(np.prod(self.mid_shape())), cams_fresh=fresh,
                     caps={"q_max": p.q_max, "mem_max": p.mem_max, "k_max": p.k_max, "bins_per_ray": p.bins_per_ray, "p_max": P_MAX})
        cells = {"idx": np.zeros((0, 3), np.int32), "cls": np.zeros(0, np.int8), "conf": np.zeros(0, np.float32), "flow": np.zeros((0, 2), np.float32),
                 "observed": np.zeros(0, bool)}
        objects: list[dict[str, Any]] = []
        prob = torch.nan_to_num(torch.sigmoid(st["mid_logit"].float()), nan=0.0)
        # with no fresh camera only memory-carried cells may decode (observed = False); no memory -> empty
        eligible = (prob > p.mid_thr) if fresh else (prob > p.mid_thr) & (st["carried"] > 0)
        cand = torch.nonzero(eligible).flatten()
        cand = cand[torch.argsort(prob[cand], descending=True)][: p.k_max]
        stats["mid_cells_expanded"] = int(cand.numel())
        tf = time.perf_counter()
        if cand.numel():
            midx = self.unkey(st["keys"][cand])
            offs = torch.tensor([(a, b, c) for a in range(MID) for b in range(MID) for c in range(MID)], device=dev)
            fidx = (midx[:, None, :] * MID + offs[None]).reshape(-1, 3)
            fobs = (st["observed"][cand] > 0)[:, None].expand(-1, offs.shape[0]).reshape(-1)  # child inherits: hit by this tick's lift
            nx, ny, nz = g.shape
            inb = (fidx[:, 0] < nx) & (fidx[:, 1] < ny) & (fidx[:, 2] < nz)
            fidx, fobs = fidx[inb], fobs[inb]
            fnp = fidx.cpu().numpy()
            pts = torch.as_tensor(g.to_center(fnp), dtype=dtype, device=dev)
            h = self.point_features(st, pts)
            pr = torch.nan_to_num(torch.sigmoid(self.occ_head(h).squeeze(-1).float()), nan=0.0)
            cl = torch.argmax(torch.nan_to_num(self.cls_head(h).float(), nan=0.0), dim=-1)
            fl = torch.nan_to_num(self.flow_head(h).float(), nan=0.0, posinf=0.0, neginf=0.0)
            keep = (pr > p.occ_thr).cpu().numpy()
            cells = {"idx": fnp[keep].astype(np.int32), "cls": cl.cpu().numpy()[keep].astype(np.int8),
                     "conf": pr.cpu().numpy()[keep].astype(np.float32), "flow": fl.cpu().numpy()[keep].astype(np.float32),
                     "observed": fobs.cpu().numpy()[keep].astype(bool)}
            stats["fine_queries"] = int(fnp.shape[0])
        else:
            stats["fine_queries"] = 0
        stats["cells_queried"] = stats["fine_queries"]
        stats["cells_stored"] = int(cells["idx"].shape[0])
        timing = dict(st["timing_ms"])
        timing["fine_head"] = (time.perf_counter() - tf) * 1e3
        to = time.perf_counter()
        if fresh and st["keys"].numel():
            objects = self._objects(st)
        timing["object_head"] = (time.perf_counter() - to) * 1e3
        tp = time.perf_counter()
        panels = self.camera_panels(cells, cams, fresh, st["weights"], ages)
        timing["panels"] = (time.perf_counter() - tp) * 1e3
        mem = st["memory"]
        topn = torch.topk(mem["w"].float(), min(MEMORY_VIZ_N, mem["w"].numel())).indices if mem["w"].numel() else mem["w"][:0].long()
        vo = st["vo"]
        return {"cells": cells, "objects": objects, "memory": {k: v.detach() for k, v in mem.items()}, "stats": stats,
                "motion_used": None if st["motion_used"] is None else [float(v) for v in st["motion_used"]],
                "vo": None if vo is None else [float(v) for v in sanitize_motion(vo.float())], "vo_conf": float(st["vo_conf"]),
                "vo_cov": None if st["vo_cov"] is None else [float(v) for v in st["vo_cov"]],
                "memory_pts": {"xyz": mem["xyz"][topn].float().cpu().numpy().astype(np.float32), "w": mem["w"][topn].float().clamp(0, 1).cpu().numpy().astype(np.float32),
                               "n_total": int(mem["w"].numel())},
                "cam_panels": panels, "timing_ms": timing}

    def camera_panels(self, cells, cams, fresh, weights, ages) -> dict[str, dict[str, Any]]:
        """Per fresh camera: the stored sparse cells projected into it (u, v normalised to [0,1], z-depth,
        class, conf), top ``PANEL_N`` by conf. Masked / stale cameras are absent (not zero-filled)."""
        out = {}
        if not len(cells["idx"]):
            ctr = np.zeros((0, 3), np.float32)
        else:
            ctr = self.grid.to_center(cells["idx"]).astype(np.float64)
        for cid in fresh:
            cam = cams[cid]
            uv, z, ok = cam.project(ctr) if len(ctr) else (np.zeros((0, 2)), np.zeros(0), np.zeros(0, bool))
            sel = np.nonzero(ok)[0]
            if len(sel) > PANEL_N:
                sel = sel[np.argsort(-cells["conf"][sel], kind="stable")[:PANEL_N]]
            z = np.asarray(z)[sel]
            out[cid] = {"uv": (np.asarray(uv)[sel] / np.array([cam.width, cam.height])).astype(np.float32).reshape(-1, 2).clip(0, 1),
                        "depth": z.astype(np.float32), "cls": cells["cls"][sel].astype(np.int8), "conf": cells["conf"][sel].astype(np.float32),
                        "age_s": float(ages.get(cid, 0.0)), "weight": float(weights.get(cid, 0.0))}
        return out

    def frustum_voxels(self, cams: Mapping[str, CameraModel], ids: Sequence[str]) -> int:
        """Number of fine voxels whose centre lies in at least one listed camera's frustum (numpy, reporting only)."""
        g = self.grid
        nx, ny, nz = g.shape
        hit = np.zeros(g.n_dense, bool)
        ii = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1).reshape(-1, 3)
        ctr = g.to_center(ii)
        for cid in ids:
            if cid in cams:
                _, _, ok = cams[cid].project(ctr)
                hit |= ok
        return int(hit.sum())

    def _objects(self, st) -> list[dict[str, Any]]:
        o = st["obj"].float()
        prob = torch.nan_to_num(torch.sigmoid(o[:, 0]), nan=0.0) * st["observed"].float().clamp(min=0.5)
        order = torch.argsort(prob, descending=True)
        xyz = st["xyz"].float()
        out: list[dict[str, Any]] = []
        for i in order.tolist():
            if float(prob[i]) <= 0.5 or len(out) >= 64:
                break
            x = float(xyz[i, 0] + torch.nan_to_num(o[i, 3]).clamp(-2, 2))
            y = float(xyz[i, 1] + torch.nan_to_num(o[i, 4]).clamp(-2, 2))
            if any(math.hypot(x - q["x"], y - q["y"]) < 2.0 for q in out):
                continue
            c, s = float(torch.nan_to_num(o[i, 1])), float(torch.nan_to_num(o[i, 2]))
            ang = math.atan2(s, c) if abs(c) + abs(s) > 0 else math.pi / 2
            out.append({"cls": "vehicle", "x": x, "y": y, "yaw": ang, "conf": float(prob[i]), "src": "boxes:traffic"})
        return out


# ------------------------------------------------------------------ training
KEY_PAD = 4096
BOX_PAD = 16


def make_queries(labels: Mapping[str, np.ndarray], *, grid: VoxelGrid = DEFAULT_GRID, n_pos: int = 512, n_neg: int = 512, seed: int = 0) -> dict[str, np.ndarray]:
    """Fixed-size training targets for one sample (padded): fine queries, mid-cell label keys, boxes."""
    rng = np.random.default_rng(seed)
    occ, ocls, free = labels["occ_idx"], labels["occ_cls"], labels["free_idx"]
    oflow = labels.get("occ_flow", np.full((len(occ), 2), np.nan, np.float32))
    pi = rng.permutation(len(occ))[:n_pos]
    ni = rng.permutation(len(free))[:n_neg]
    N = n_pos + n_neg
    pts = np.zeros((N, 3), np.float32)
    occ_t = np.zeros(N, np.float32)
    cls_t = np.full(N, -1, np.int64)
    flow_t = np.full((N, 2), np.nan, np.float32)
    mask = np.zeros(N, np.float32)
    pts[: len(pi)] = grid.to_center(occ[pi])
    occ_t[: len(pi)] = 1.0
    cls_t[: len(pi)] = np.where(ocls[pi] == UNKNOWN, -1, ocls[pi])
    flow_t[: len(pi)] = oflow[pi]
    mask[: len(pi)] = 1.0
    pts[n_pos : n_pos + len(ni)] = grid.to_center(free[ni])
    mask[n_pos : n_pos + len(ni)] = 1.0
    nx, ny, nz = grid.shape
    mx, my, mz = -(-nx // MID), -(-ny // MID), -(-nz // MID)

    def mkeys(idx):
        m = np.asarray(idx, np.int64).reshape(-1, 3) // MID
        k = np.unique((m[:, 0] * my + m[:, 1]) * mz + m[:, 2])
        out = np.full(KEY_PAD, -1, np.int64)
        k = k[rng.permutation(len(k))[:KEY_PAD]] if len(k) > KEY_PAD else k
        out[: len(k)] = k
        return out

    boxes = np.full((BOX_PAD, 5), np.nan, np.float32)
    bx = np.asarray(labels.get("boxes", np.zeros((0, 5), np.float32)), np.float32).reshape(-1, 5)[:BOX_PAD]
    boxes[: len(bx)] = bx
    return {"pts": pts, "occ_t": occ_t, "cls_t": cls_t, "flow_t": flow_t, "mask": mask, "mid_occ": mkeys(occ), "mid_free": mkeys(free), "boxes": boxes}


def depth_target_bins(depth: torch.Tensor, hw: tuple[int, int], n_bins: int) -> torch.Tensor:
    """(H,W) metres (NaN/inf invalid) -> (h,w) long bin index at the 1/16 grid (-1 invalid)."""
    valid = torch.isfinite(depth) & (depth > 0)
    d = torch.where(valid, depth, torch.zeros_like(depth))[None, None].float()
    vs = F.adaptive_avg_pool2d(valid[None, None].float(), hw)[0, 0]
    ds = F.adaptive_avg_pool2d(d, hw)[0, 0] / vs.clamp(min=1e-6)
    b = torch.round((torch.log(ds.clamp(min=D_MIN, max=D_MAX)) - math.log(D_MIN)) / (math.log(D_MAX) - math.log(D_MIN)) * (n_bins - 1)).long()
    return torch.where(vs > 0.5, b, torch.full_like(b, -1))


def occ_loss(out: Mapping[str, Any], q: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    m = q["mask"]
    occ = F.binary_cross_entropy_with_logits(out["occ_logit"].float(), q["occ_t"], reduction="none")
    l_occ = (occ * m).sum() / m.sum().clamp(min=1.0)
    zero = out["occ_logit"].sum() * 0.0
    cls_t = q["cls_t"]
    valid = cls_t >= 0
    l_cls = F.cross_entropy(out["cls_logit"].float()[valid], cls_t[valid]) if bool(valid.any()) else zero
    ft = q["flow_t"]
    fmask = torch.isfinite(ft).all(dim=-1)
    l_flow = F.smooth_l1_loss(out["flow"].float()[fmask], ft[fmask]) if bool(fmask.any()) else zero
    l_mid, l_obj, l_yaw, l_depth = zero, zero, zero, zero
    nb = 0
    for b, cell in enumerate(out.get("cells", [])):
        keys = cell["keys"]
        if keys.numel():
            pos = torch.isin(keys, q["mid_occ"][b])
            neg = torch.isin(keys, q["mid_free"][b]) & ~pos
            lab = pos | neg
            if bool(lab.any()):
                l_mid = l_mid + F.binary_cross_entropy_with_logits(cell["mid_logit"].float()[lab], pos.float()[lab])
            bx = q["boxes"][b]
            bx = bx[torch.isfinite(bx[:, :3]).all(dim=1)]
            tgt = torch.zeros(keys.numel())
            if bx.shape[0]:
                d = torch.cdist(bx[:, :2].float(), cell["xyz"][:, :2].float().detach())
                j = d.argmin(dim=1)
                near = d.gather(1, j[:, None])[:, 0] < 2.0
                if bool(near.any()):
                    jj = j[near]
                    tgt[jj] = 1.0
                    o = cell["obj"][jj].float()
                    yaw = F.normalize(o[:, 1:3], dim=1)
                    by = bx[near]
                    l_yaw = l_yaw + (1 - (yaw[:, 0] * torch.cos(by[:, 2]) + yaw[:, 1] * torch.sin(by[:, 2]))).mean() \
                        + F.smooth_l1_loss(o[:, 3:5], (by[:, :2] - cell["xyz"][jj, :2].detach()).float())
            l_obj = l_obj + F.binary_cross_entropy_with_logits(cell["obj"][:, 0].float(), tgt, pos_weight=torch.tensor(10.0))
        for cid, dl in cell["dl"].items():
            dk = f"depth:{cid}"
            if dk in q:
                t = depth_target_bins(q[dk][b], dl.shape[-2:], dl.shape[0])
                vm = t >= 0
                if bool(vm.any()):
                    l_depth = l_depth + F.cross_entropy(dl.float().permute(1, 2, 0)[vm], t[vm]) / max(1, len(cell["dl"]))
        nb += 1
    nb = max(1, nb)
    total = l_occ + 0.5 * l_cls + 0.1 * l_flow + (l_mid + 0.5 * l_obj + 0.2 * l_yaw + 0.5 * l_depth) / nb
    l_vo = zero
    if out.get("vo") is not None and "vo_t" in q:
        vt = q["vo_t"]
        vm = torch.isfinite(vt).all(dim=-1)
        if bool(vm.any()):
            l_vo = F.smooth_l1_loss(out["vo"].float()[vm], vt[vm])
            total = total + 2.0 * l_vo
            lv = out.get("vo_logvar")
            if lv is not None:  # calibrates the reported VO variance (does not pull the motion itself)
                err2 = (out["vo"].float()[vm].detach() - vt[vm]) ** 2
                total = total + 0.05 * (0.5 * (lv[vm] + err2 * torch.exp(-lv[vm]))).mean()
    return {"vo": l_vo, "occ": l_occ, "cls": l_cls, "flow": l_flow, "mid": l_mid / nb, "obj": l_obj / nb, "yaw": l_yaw / nb, "depth": l_depth / nb, "total": total}


# --------------------------------------------------------------- checkpoints
def save_checkpoint(model: OccSparseNet, path: Path | str) -> None:
    torch.save({"kind": CKPT_KIND, "arch": asdict(model.arch), "profile": asdict(model.profile), "state_dict": model.state_dict()}, path)


def load_checkpoint(path: Path | str, profile: OccProfile | None = None, arch: OccArch | None = None) -> OccSparseNet:
    """Load into ``profile`` (any run-time knobs). Refuses a different arch, never partially loads."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - any unreadable file is a checkpoint error
        raise CheckpointError(f"{Path(path).name}: unreadable checkpoint: {exc}") from exc
    if not isinstance(ck, dict) or ck.get("kind") != CKPT_KIND:
        raise CheckpointError(f"{Path(path).name}: not a {CKPT_KIND} checkpoint")
    saved = ck.get("arch") or {}
    norm = {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in saved.items()}
    want = asdict(arch) if arch is not None else saved
    wnorm = {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in want.items()}
    if norm != wnorm:
        diff = sorted(k for k in set(norm) | set(wnorm) if norm.get(k) != wnorm.get(k))
        raise CheckpointError(f"architecture differs ({', '.join(diff)}): saved {norm}, wanted {wnorm}; retrain required")
    try:
        model = OccSparseNet(OccArch(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved.items()}), profile)
        model.load_state_dict(ck["state_dict"], strict=True)
    except (RuntimeError, TypeError, KeyError, ValueError) as exc:
        raise CheckpointError(f"state_dict does not match the architecture: {exc}") from exc
    return model


# ----------------------------------------------------------------- estimates
def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _rig(model: OccSparseNet, cams: Mapping[str, CameraModel], long_side: int | None, batch: int = 1):
    imgs, ages = {}, {}
    for cid, cam in cams.items():
        if cid not in model.profile.cams:
            continue
        ls = model.profile.long_side(cid) if long_side is None else long_side
        s = ls / max(cam.width, cam.height)
        w, h = max(16, int(round(cam.width * s))), max(16, int(round(cam.height * s)))
        g = torch.Generator().manual_seed(0)
        imgs[cid] = torch.rand(batch, 3, h, w, generator=g)
        ages[cid] = torch.zeros(batch)
    return imgs, ages


def cost_estimate(model: OccSparseNet, cams: Mapping[str, CameraModel], *, long_side: int | None = None, n_queries: int | None = None, batch: int = 1) -> dict[str, float]:
    """FLOPs (multiply-add = 2) and summed module outputs for one forward at the profile's caps (worst case:
    every stage filled to its absolute cap). CPU hooks; not a GPU measurement."""
    p = model.profile
    n_q = (p.k_max * MID**3) if n_queries is None else n_queries
    flops = 0
    acts = 0

    def hook(m: nn.Module, inp: Any, o: Any) -> None:
        nonlocal flops, acts
        if isinstance(o, torch.Tensor):
            acts += o.numel()
            if isinstance(m, nn.Conv2d):
                k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
                flops += 2 * k * o.numel()
            elif isinstance(m, nn.Linear):
                flops += 2 * m.in_features * o.numel()
            elif isinstance(m, SubMConv):
                flops += 2 * len(NEIGHBOURS) * m.weight.shape[1] * o.numel()

    hs = [m.register_forward_hook(hook) for m in model.modules() if len(list(m.children())) == 0 or isinstance(m, SubMConv)]
    try:
        model.eval()
        imgs, ages = _rig(model, cams, long_side, batch)
        with torch.no_grad():
            enc = {}
            for b in range(batch):
                enc = model.encode({c: v[b] for c, v in imgs.items()}, cams, {c: 1.0 for c in imgs})
            # sparse stages at their caps (independent of the actual image content)
            C = model.feat
            Q, M = p.q_max, p.mem_max
            for _ in range(batch):
                xyz = torch.zeros(Q, 3)
                xyz[:, 1] = torch.linspace(0, 60, Q)
                f = torch.zeros(Q, C)
                model.cell_in(torch.zeros(Q, 2 * C + POSENC_DIM + 1))
                model._gru(torch.zeros(Q, C), torch.zeros(Q, C), torch.zeros(Q), torch.zeros(Q))
                nbr = torch.full((Q, len(NEIGHBOURS)), -1, dtype=torch.long)
                model.sconv1(f, nbr)
                model.sconv2(f, nbr)
                model.mid_head(f)
                model.obj_head(f)
                # direct VO: (37 coarse + 81 refine) candidates x VO_GROUND_N bilinear samples, no parameters
                flops += 118 * VO_GROUND_N * 20
                h = model.mlp(torch.zeros(n_q, 2 * C + POSENC_DIM + 1))
                model.occ_head(h)
                model.cls_head(h)
                model.flow_head(h)
                acts += Q * C * 4 + M * (C + 4)
                # camera pooling: one bilinear gather per camera per point
                flops += 2 * 4 * C * (Q + n_q) * len(enc)
    finally:
        for h_ in hs:
            h_.remove()
    return {"flops": float(flops), "activations": float(acts)}


def temporal_cache_bytes(model: OccSparseNet, bytes_per: int = 2) -> int:
    """Temporal cache: sparse memory mem_max x (C feature + 3 xyz + 1 weight) plus the direct-VO reference:
    VO_GROUND_N grey samples + validity. Nothing dense (no image or feature map) is kept across ticks."""
    p = model.profile
    return int((p.mem_max * (model.feat + 4) + 2 * VO_GROUND_N) * bytes_per)


OFFLINE_BATCH = 8  # offline runner: 8 independent clips per forward (each clip strictly causal)


def memory_estimate(model: OccSparseNet, cams: Mapping[str, CameraModel], *, long_side: int | None = None, train_batch: int = 4, train_queries: int = 2048,
                    infer_bytes: int = 2, infer_batch: int = OFFLINE_BATCH) -> dict[str, float]:
    """CPU ESTIMATE (GB) of inference (batch 1 and the offline runner's batch) and AMP-training memory for this rig.
    Not a GPU measurement: GPU peak is UNPROVEN until nvidia-smi output exists."""
    pc = param_count(model)
    gb = 1024**3
    inf = cost_estimate(model, cams, long_side=long_side, batch=1)
    tr = cost_estimate(model, cams, long_side=long_side, n_queries=train_queries, batch=train_batch)
    infb = cost_estimate(model, cams, long_side=long_side, batch=infer_batch) if infer_batch > 1 else inf
    cache = temporal_cache_bytes(model, infer_bytes)
    return {
        "label": "CPU ESTIMATE (forward hooks, stages at their caps), not a GPU measurement",
        "params": float(pc),
        "gflops_per_tick": inf["flops"] / 1e9,
        "temporal_cache_mb": cache / 1024**2,
        "infer_gb": (pc * infer_bytes + inf["activations"] * infer_bytes + cache) / gb + CUDA_CONTEXT_GB,
        "infer_batch": float(infer_batch),
        "infer_gb_batch": (pc * infer_bytes + infb["activations"] * infer_bytes + infer_batch * cache) / gb + CUDA_CONTEXT_GB,
        "gpu_peak": "UNPROVEN",
        "train_gb": (pc * 18 + 2 * tr["activations"] * 2) / gb + CUDA_CONTEXT_GB,
        "cuda_context_gb": CUDA_CONTEXT_GB,
    }
