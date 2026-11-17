"""Sparse occupancy net: camera-only runtime half (docs/specs/occ_sparse.md).

First-principles design. The net never builds a dense 3-D volume. It answers
*queries*: a list of ego-frame points (x right, y forward, z up, metres).
Each query is projected into every camera with the static rig calibration,
the shared 2-D feature map is sampled there (bilinear), cameras that cannot
see the point are masked out, and a small MLP turns the pooled feature plus
a Fourier position code into an occupancy logit, semantic logits and an
optional ground-plane flow. Memory scales with the query count, not with the
grid volume.

Live inference is coarse-to-fine (:func:`infer_sparse`): every coarse cell
is asked once (capped at ``max_coarse``, nearest first), then only the children of the most likely coarse cells are
asked at full resolution, capped at ``max_fine``.

This module imports no ground truth. Depth, semantic ids and vehicle boxes
live in the training-only label builder (``python/train``). Inputs here are RGB
images and the camera poses from ``config/cameras.yaml``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Semantic classes of an occupied cell. Order is the sem-logit order and part
# of the ONNX contract. Classes without a proven label source train on nothing
# (their label stays CLS_IGNORE; see docs/specs/occ_sparse.md "label sources").
OCC_CLASSES = ("drivable", "vehicle", "lane_white", "lane_yellow", "crosswalk", "stop_line", "curb")
CLS_DRIVABLE = 0
CLS_VEHICLE = 1
CLS_LANE_WHITE = 2
CLS_LANE_YELLOW = 3
CLS_CROSSWALK = 4
CLS_STOP_LINE = 5
CLS_CURB = 6
CLS_IGNORE = 255
INFER_VRAM_CAP_BYTES = 4 * 2**30
TRAIN_VRAM_CAP_BYTES = 11 * 2**30
DEFAULT_MAX_FINE = 8192
DEFAULT_MAX_COARSE = 8192  # coarse queries per tick; nearest-first, never grows with the grid
MAX_MEM_LEN = 8
REF_CAMS = 8.0  # "seen" feature scale; fixed so cam count is a loadable knob
CAM_AGE_TAU_S = 0.3   # a camera frame 0.3 s old counts 1/e; GVD's rig refreshes some cams at 1-2 Hz
CAM_MAX_AGE_S = 1.0   # older than this: masked out of fusion


@dataclass(frozen=True)
class OccGrid:
    """Ego-frame voxel grid. Only its index space is used; nothing dense is stored."""

    x_min: float = -25.6
    x_max: float = 25.6
    y_min: float = -12.8
    y_max: float = 51.2
    z_min: float = -1.6
    z_max: float = 3.2
    voxel_m: float = 0.4
    coarse: tuple[int, int, int] = (4, 4, 2)

    @property
    def lo(self) -> np.ndarray:
        return np.array([self.x_min, self.y_min, self.z_min], np.float64)

    @property
    def hi(self) -> np.ndarray:
        return np.array([self.x_max, self.y_max, self.z_max], np.float64)

    @property
    def dims(self) -> tuple[int, int, int]:
        span = (self.hi - self.lo) / self.voxel_m
        return tuple(int(round(s)) for s in span)  # type: ignore[return-value]

    @property
    def coarse_dims(self) -> tuple[int, int, int]:
        return tuple(int(d // c) for d, c in zip(self.dims, self.coarse))  # type: ignore[return-value]

    def to_index(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Integer voxel index per point and the inside mask."""
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        idx = np.floor((pts - self.lo) / self.voxel_m).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < np.array(self.dims)), axis=1)
        return idx, inside

    def centers(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, np.float64).reshape(-1, 3)
        return self.lo + (idx + 0.5) * self.voxel_m

    def coarse_centers(self, cidx: np.ndarray) -> np.ndarray:
        cidx = np.asarray(cidx, np.float64).reshape(-1, 3)
        size = self.voxel_m * np.array(self.coarse, np.float64)
        return self.lo + (cidx + 0.5) * size

    def linear(self, idx: np.ndarray) -> np.ndarray:
        nx, ny, nz = self.dims
        idx = np.asarray(idx, np.int64).reshape(-1, 3)
        return (idx[:, 0] * ny + idx[:, 1]) * nz + idx[:, 2]

    def unlinear(self, lin: np.ndarray) -> np.ndarray:
        nx, ny, nz = self.dims
        lin = np.asarray(lin, np.int64)
        return np.stack([lin // (ny * nz), (lin // nz) % ny, lin % nz], axis=1)


def camera_basis(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Rows: camera right, up, forward in the ego frame (same math as lanes.py)."""
    from python.sensors.cameras import yaw_pitch_to_dir_up

    fwd, up = (np.array(v, np.float64) for v in yaw_pitch_to_dir_up(yaw_deg, pitch_deg))
    right = np.cross(fwd, up)
    return np.stack([right, up, fwd])


def pixel_rays(pose: Any, width: int, height: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Ego-frame ray per sampled pixel, scaled so the forward component is 1.

    Same pinhole as ``python.perception.lanes.project_pose_pixel``: pixel
    centre ``u`` maps to ``nx = (u - (w-1)/2) / (w/2)``. Returns rays
    (H', W', 3) and the (rows, cols) sampled.
    """
    w, h = int(width), int(height)
    fov_v = math.radians(float(pose.fov_v_deg))
    tan_v = math.tan(fov_v * 0.5)
    tan_h = tan_v * (w / h)
    s = max(1, int(stride))
    rows = np.arange(s // 2, h, s)
    cols = np.arange(s // 2, w, s)
    nx = (cols - (w - 1.0) * 0.5) / (w * 0.5)
    ny = (rows - (h - 1.0) * 0.5) / (h * 0.5)
    basis = camera_basis(pose.yaw_deg, pose.pitch_deg)
    rx = nx[None, :] * tan_h
    ry = ny[:, None] * tan_v
    rays = (
        basis[2][None, None, :]
        + rx[..., None] * basis[0][None, None, :]
        - ry[..., None] * basis[1][None, None, :]
    )
    return rays, (rows, cols)


def rig_poses(cam_ids: Sequence[str] | None = None) -> list[Any]:
    """Poses for the runtime rig from config/cameras.yaml, in CAM_IDS order."""
    from python.perception.lanes import camera_poses
    from python.sensors.cameras import CAM_IDS

    found = camera_poses()
    ids = list(cam_ids) if cam_ids else list(CAM_IDS)
    missing = [c for c in ids if c not in found]
    if missing:
        raise ValueError(f"occ rig: no pose in cameras.yaml for {missing}")
    return [found[c] for c in ids]


# ---------------------------------------------------------------- torch part


def _torch():
    import torch

    return torch


def _build_net_class():
    torch = _torch()
    nn = torch.nn
    F = torch.nn.functional

    def cam_weight(valid, age):
        """(B,N) camera weight: valid * exp(-age/tau), 0 past CAM_MAX_AGE_S. None, None -> None."""
        if valid is None and age is None:
            return None
        w = valid if valid is not None else torch.ones_like(age)
        if age is not None:
            w = w * torch.exp(-torch.clamp(age, min=0.0) / CAM_AGE_TAU_S) * (age <= CAM_MAX_AGE_S).to(w.dtype)
        return w

    class SparseOccNet(nn.Module):
        """Camera-only query decoder. ``forward(images, queries)``.

        images: (B, N, 3, H, W) float in 0..1, N = len(poses), any H, W with
        the rig aspect. queries: (B, Q, 3) ego metres, or (B, Q, 4) with a
        level column (0 fine voxel, 1 coarse cell).
        Returns ``{"occ": (B,Q) logit, "sem": (B,Q,K), "flow": (B,Q,2), "box": (B,Q,6)}``.
        """

        def __init__(
            self,
            poses: Sequence[Any] | None,
            grid: OccGrid | None = None,
            *,
            width: int = 64,
            hidden: int = 128,
            n_freq: int = 4,
            mem_len: int = 2,
            depth: int = 1,
        ) -> None:
            super().__init__()
            self.mem_len = int(mem_len)
            self.width = int(width)
            poses = list(poses) if poses is not None else rig_poses()
            self.grid = grid or OccGrid()
            self.cam_ids = tuple(p.cam_id for p in poses)
            pos = np.array([[p.x, p.y, p.z] for p in poses], np.float32)
            rot = np.stack([camera_basis(p.yaw_deg, p.pitch_deg) for p in poses]).astype(np.float32)
            tan_v = np.array([math.tan(math.radians(p.fov_v_deg) * 0.5) for p in poses], np.float32)
            tan_h = tan_v * np.array([p.width / p.height for p in poses], np.float32)
            self.register_buffer("cam_pos", torch.from_numpy(pos), persistent=False)
            self.register_buffer("cam_rot", torch.from_numpy(rot), persistent=False)
            self.register_buffer("tan_hv", torch.from_numpy(np.stack([tan_h, tan_v], 1)), persistent=False)
            lo = self.grid.lo.astype(np.float32)
            hi = self.grid.hi.astype(np.float32)
            self.register_buffer("g_mid", torch.from_numpy((lo + hi) * 0.5), persistent=False)
            self.register_buffer("g_half", torch.from_numpy((hi - lo) * 0.5), persistent=False)
            self.register_buffer("freqs", torch.tensor([math.pi * 2.0**k for k in range(n_freq)]), persistent=False)
            c1, c2, c3 = max(8, width // 3), max(8, width // 2), max(8, (3 * width) // 4)
            self.backbone = nn.Sequential(
                nn.Conv2d(3, c1, 3, 2, 1), nn.ReLU(inplace=True),
                nn.Conv2d(c1, c2, 3, 2, 1), nn.ReLU(inplace=True),
                nn.Conv2d(c2, c3, 3, 2, 1), nn.ReLU(inplace=True),
                nn.Conv2d(c3, width, 3, 1, 1), nn.ReLU(inplace=True),
                *[m for _ in range(max(0, int(depth) - 1)) for m in (nn.Conv2d(width, width, 3, 1, 1),
                                                                     nn.ReLU(inplace=True))],
                nn.Conv2d(width, width, 1),
            )
            # current pooled + temporal pooled + (current - temporal) + has_memory + seen + pe + level
            d_in = 3 * width + 1 + 1 + 3 + 6 * n_freq + 1
            # learned weight per memory slot; fixed size so mem_len can change without a retrain
            self.age_w = nn.Parameter(torch.zeros(MAX_MEM_LEN))
            if self.mem_len > MAX_MEM_LEN:
                raise ValueError(f"mem_len {self.mem_len} > {MAX_MEM_LEN}")
            # camera-only visual odometry: ego motion between the newest past frame and now, from the
            # same cached backbone maps. Predicted per camera in that camera's axes, rotated to ego.
            self.vo_conv = nn.Sequential(
                nn.Conv2d(2 * width, width, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(width, width, 3, 2, 1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((3, 4)),
            )
            self.vo_fc = nn.Sequential(nn.Linear(width * 12, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 3))
            self.mlp = nn.Sequential(
                nn.Linear(d_in, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            )
            self.occ_head = nn.Linear(hidden, 1)
            self.sem_head = nn.Linear(hidden, len(OCC_CLASSES))
            self.flow_head = nn.Linear(hidden, 2)
            # objects: per active cell, offset to its box centre (dx, dy), log length/width, yaw sin/cos.
            # Trained from recorded boxes only; cells of one box vote for the same centre at runtime.
            self.box_head = nn.Linear(hidden, 6)

        def project(self, xyz):
            """Normalised image coords (B,N,Q,2) and visibility (B,N,Q)."""
            d = xyz[:, None, :, :] - self.cam_pos[None, :, None, :]
            cam = torch.einsum("bnqk,njk->bnqj", d, self.cam_rot)
            z = cam[..., 2]
            zs = torch.clamp(z, min=1e-3)
            nx = cam[..., 0] / (zs * self.tan_hv[None, :, None, 0])
            ny = -cam[..., 1] / (zs * self.tan_hv[None, :, None, 1])
            vis = (z > 0.1) & (nx.abs() <= 1.0) & (ny.abs() <= 1.0)
            uv = torch.stack([nx, ny], -1)
            uv = torch.where(vis[..., None], uv, torch.full_like(uv, 2.0))
            return uv, vis

        def encode(self, images):
            """Shared backbone over every camera: (B,N,3,H,W) -> (B,N,C,h,w)."""
            b, n, _, h, w = images.shape
            f = self.backbone(images.reshape(b * n, 3, h, w))
            return f.reshape(b, n, f.shape[1], f.shape[2], f.shape[3])

        def _pool(self, feat, xyz, cam_w=None):
            """Multi-camera fusion at 3D points: weighted mean of bilinear samples over the cameras
            that see the point. cam_w (B,N) in [0,1] down-weights stale or missing cameras (0 = masked).
            Returns pooled (B,Q,C) and the number of contributing cameras (B,Q)."""
            b, n, c, fh, fw = feat.shape
            q = xyz.shape[1]
            uv, vis = self.project(xyz)
            samp = F.grid_sample(feat.reshape(b * n, c, fh, fw), uv.reshape(b * n, q, 1, 2), mode="bilinear",
                                 padding_mode="zeros", align_corners=False).reshape(b, n, c, q)
            m = vis.to(samp.dtype)
            if cam_w is not None:
                m = m * cam_w[:, :, None].to(samp.dtype)
            wsum = m.sum(1)
            pooled = (samp * m[:, :, None, :]).sum(1) / torch.clamp(wsum, min=1e-6)[:, None, :]
            return pooled.transpose(1, 2), (m > 0).to(samp.dtype).sum(1)

        def forward(self, images, queries, mem_feat=None, mem_xf=None, mem_valid=None, cam_valid=None,
                    cam_age=None, mem_age=None):
            """images (B,N,3,H,W) now. Temporal memory (optional, K = mem_len past frames):
            mem_feat (B,K,N,C,h,w) cached ``encode`` outputs, mem_xf (B,K,2,3) affine map from
            current-ego xy to that frame's ego xy, mem_valid (B,K) or (B,K,N) 1/0.
            Per-camera freshness: cam_valid (B,N) 1/0, cam_age (B,N) and mem_age (B,K,N) seconds
            since that camera's frame; weight exp(-age / CAM_AGE_TAU_S), masked past CAM_MAX_AGE_S."""
            return self.decode(self.encode(images), queries, mem_feat, mem_xf, mem_valid, cam_valid, cam_age,
                               mem_age)

        def vo(self, feat, past, cam_valid=None, cam_age=None):
            """(B,3) = (dx right, dy forward, dyaw ccw rad): the current ego in the past ego's frame.
            Camera-only: inputs are this tick's and the newest cached backbone maps."""
            b, n, c, fh, fw = feat.shape
            z = self.vo_conv(torch.cat([feat, past], 2).reshape(b * n, 2 * c, fh, fw)).reshape(b * n, -1)
            o = self.vo_fc(z).reshape(b, n, 3)
            ax_r = F.normalize(self.cam_rot[:, 0, :2], dim=-1)
            ax_f = F.normalize(self.cam_rot[:, 2, :2], dim=-1)
            t = o[..., 0:1] * ax_r[None] + o[..., 1:2] * ax_f[None]
            d = torch.cat([t, o[..., 2:3]], -1)
            w = cam_weight(cam_valid, cam_age)
            w = torch.ones(b, n, device=d.device, dtype=d.dtype) if w is None else w.to(d.dtype)
            return (d * w[..., None]).sum(1) / torch.clamp(w.sum(1, keepdim=True), min=1e-6)

        @staticmethod
        def cam_weight_for(valid, age):
            return cam_weight(valid, age)

        HEADS = ("occ", "sem", "flow", "box")

        def head(self, name, hid):
            if name == "occ":
                return self.occ_head(hid)[..., 0]
            return getattr(self, f"{name}_head")(hid)

        def decode(self, feat, queries, mem_feat=None, mem_xf=None, mem_valid=None, cam_valid=None, cam_age=None,
                   mem_age=None):
            hid = self.hidden(feat, queries, mem_feat, mem_xf, mem_valid, cam_valid, cam_age, mem_age)
            return {k: self.head(k, hid) for k in self.HEADS}

        def hidden(self, feat, queries, mem_feat=None, mem_xf=None, mem_valid=None, cam_valid=None, cam_age=None,
                   mem_age=None):
            """Shared per-query trunk (B,Q,hidden): multi-cam fusion + temporal memory + position."""
            b, n = feat.shape[:2]
            xyz = queries[..., :3]
            level = queries[..., 3:4] if queries.shape[-1] > 3 else torch.zeros_like(xyz[..., :1])
            pooled, count = self._pool(feat, xyz, cam_weight(cam_valid, cam_age))
            seen = (count / REF_CAMS)[..., None]  # rig-independent: weights load on 2/3/8-cam rigs
            tnum = torch.zeros_like(pooled)
            tden = torch.zeros_like(seen)
            if mem_feat is not None and self.mem_len > 0:
                wts = F.softplus(self.age_w)
                for k in range(min(self.mem_len, mem_feat.shape[1])):
                    A = mem_xf[:, k, :, :2]
                    t = mem_xf[:, k, :, 2]
                    xy = torch.einsum("bij,bqj->bqi", A, xyz[..., :2]) + t[:, None, :]
                    mv = mem_valid[:, k]
                    mv = mv[:, None].expand(-1, n) if mv.dim() == 1 else mv
                    cw = cam_weight(mv, None if mem_age is None else mem_age[:, k])
                    pk, ck = self._pool(mem_feat[:, k], torch.cat([xy, xyz[..., 2:3]], -1), cw)
                    mk = (ck[..., None] > 0).to(pk.dtype) * wts[k]
                    tnum = tnum + pk * mk
                    tden = tden + mk
            has = (tden > 0).to(pooled.dtype)
            temporal = tnum / torch.clamp(tden, min=1e-6)
            p = (xyz - self.g_mid) / self.g_half
            ang = p[..., None] * self.freqs
            pe = torch.cat([p, torch.sin(ang).flatten(-2), torch.cos(ang).flatten(-2)], -1)
            return self.mlp(torch.cat([pooled, temporal, (pooled - temporal) * has, has, seen, pe, level], -1))

    return SparseOccNet


_NET_CLS = None


def __getattr__(name: str):  # lazy torch import: numpy-only callers stay torch-free
    global _NET_CLS
    if name == "SparseOccNet":
        if _NET_CLS is None:
            _NET_CLS = _build_net_class()
        return _NET_CLS
    raise AttributeError(name)


def all_coarse_queries(grid: OccGrid) -> np.ndarray:
    cx, cy, cz = grid.coarse_dims
    cidx = np.stack(np.meshgrid(np.arange(cx), np.arange(cy), np.arange(cz), indexing="ij"), -1).reshape(-1, 3)
    return cidx


def nearest_coarse_queries(grid: OccGrid, max_coarse: int = DEFAULT_MAX_COARSE) -> np.ndarray:
    """At most ``max_coarse`` coarse cells, nearest to ego (xy) first. Work and memory are
    O(max_coarse) regardless of grid range/resolution: only a window around ego is enumerated."""
    cdims = np.array(grid.coarse_dims, np.int64)
    cap = max(1, int(max_coarse))
    if int(np.prod(cdims)) <= cap:
        return all_coarse_queries(grid)
    size = grid.voxel_m * np.array(grid.coarse, np.float64)
    ego = np.clip(np.floor(-grid.lo / size).astype(np.int64), 0, cdims - 1)
    h = max(1, int(math.ceil((math.sqrt(cap / max(1, cdims[2])) - 1) / 2)))
    while True:
        lo = np.maximum(ego[:2] - h, 0)
        hi = np.minimum(ego[:2] + h + 1, cdims[:2])
        cnt = int(np.prod(hi - lo)) * int(cdims[2])
        if cnt >= cap or (np.all(lo == 0) and np.all(hi == cdims[:2])):
            break
        h = h + max(1, h // 4)
    ix, iy, iz = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), np.arange(cdims[2]), indexing="ij")
    cidx = np.stack([ix, iy, iz], -1).reshape(-1, 3)
    d = np.hypot(*grid.coarse_centers(cidx)[:, :2].T)
    return cidx[np.argsort(d, kind="stable")[:cap]]


def coarse_children(grid: OccGrid, cidx: np.ndarray) -> np.ndarray:
    fx, fy, fz = grid.coarse
    off = np.stack(np.meshgrid(np.arange(fx), np.arange(fy), np.arange(fz), indexing="ij"), -1).reshape(-1, 3)
    base = np.asarray(cidx, np.int64).reshape(-1, 1, 3) * np.array(grid.coarse)
    return (base + off[None]).reshape(-1, 3)


def relative_xf(cur: Sequence[float], past: Sequence[float]) -> np.ndarray:
    """2x3 affine: current-ego xy -> past-ego xy. Poses are world (x, y, dir_x, dir_y);
    ego frame is x right, y forward (``signal_labels.world_to_ego`` basis)."""
    def basis(pz):
        f = np.array(pz[2:4], np.float64)
        f = f / max(1e-9, float(np.hypot(*f)))
        return np.array([f[1], -f[0]]), f
    rc, fc = basis(cur)
    rp, fp = basis(past)
    d = np.array(cur[:2], np.float64) - np.array(past[:2], np.float64)
    return np.array([[rc @ rp, fc @ rp, d @ rp], [rc @ fp, fc @ fp, d @ fp]], np.float32)


def pose_step(pose: Sequence[float], delta: Sequence[float]) -> tuple[float, float, float, float]:
    """Advance an (x, y, dir_x, dir_y) pose by (dx right, dy forward, dyaw ccw) in its own frame."""
    x, y, fx, fy = map(float, pose)
    dx, dy, dyaw = map(float, delta)
    nx, ny = x + dx * fy + dy * fx, y - dx * fx + dy * fy
    c, s_ = math.cos(dyaw), math.sin(dyaw)
    return (nx, ny, fx * c - fy * s_, fx * s_ + fy * c)


def vo_speed_yaw(delta: Sequence[float] | None, dt: float) -> tuple[float, float]:
    """VO delta over dt -> (speed m/s, yaw rate rad/s) for OccTracker; (0, 0) before the first delta."""
    if delta is None or dt <= 0:
        return 0.0, 0.0
    return float(math.hypot(delta[0], delta[1]) / dt), float(delta[2] / dt)


def delta_from_xf(xf: Any) -> Any:
    """Inverse of relative_xf for training targets: (...,2,3) -> (...,3) (dx, dy, dyaw) of the current
    ego in the past frame. Works on numpy or torch."""
    t = xf[..., :, 2]
    if hasattr(xf, "new_zeros"):
        yaw = _torch().atan2(-xf[..., 0, 1], xf[..., 1, 1])
        return _torch().cat([t, yaw[..., None]], -1)
    yaw = np.arctan2(-xf[..., 0, 1], xf[..., 1, 1])
    return np.concatenate([t, yaw[..., None]], -1)


class FeatureMemory:
    """Runtime temporal memory: the last ``mem_len`` backbone outputs, each frame's ego pose and
    each camera's own capture time (cameras refresh at different rates)."""

    def __init__(self, mem_len: int = 2) -> None:
        self.mem_len = int(mem_len)
        self.odom: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # VO-integrated ego pose
        self.frames: list[tuple[Any, tuple[float, float, float, float] | None, Any]] = []

    def tensors(self, ego_pose: Sequence[float] | None, like: Any, now: float | None = None):
        torch = _torch()
        b, n, c, fh, fw = like.shape
        k_mem = max(1, self.mem_len)
        feat = like.new_zeros((b, k_mem, n, c, fh, fw))
        xf = like.new_zeros((b, k_mem, 2, 3))
        valid = like.new_zeros((b, k_mem, n))
        age = like.new_zeros((b, k_mem, n))
        for k, (f, pose, cam_t) in enumerate(reversed(self.frames[-self.mem_len:])):  # k=0 newest past
            if pose is None or ego_pose is None or tuple(f.shape) != tuple(like.shape):
                continue
            feat[:, k] = f
            xf[:, k] = torch.from_numpy(relative_xf(ego_pose, pose)).to(like)
            valid[:, k] = 1.0
            if cam_t is not None and now is not None:
                age[:, k] = torch.as_tensor(float(now) - np.asarray(cam_t, np.float64), dtype=like.dtype)
        return feat, xf, valid, age

    def push(self, feat: Any, ego_pose: Sequence[float] | None, cam_t: Sequence[float] | None = None) -> None:
        pose = None if ego_pose is None else tuple(map(float, ego_pose))
        self.frames = (self.frames + [(feat.detach(), pose, None if cam_t is None else np.asarray(cam_t))])
        self.frames = self.frames[-self.mem_len:] if self.mem_len > 0 else []


def infer_sparse(net: Any, images: Any, *, tau: float = 0.5, max_fine: int = DEFAULT_MAX_FINE,
                 max_coarse: int = DEFAULT_MAX_COARSE,
                 memory: FeatureMemory | None = None, ego_pose: Sequence[float] | None = None,
                 cam_valid: Sequence[float] | None = None, cam_t: Sequence[float] | None = None,
                 now: float | None = None, ego_from_vehicle: bool = False) -> dict[str, Any]:
    """Coarse-to-fine sparse inference. Returns fine voxel idx, prob, sem, flow, box.

    The backbone runs once per call; coarse and fine queries reuse its
    features. With ``memory`` the last frames' features (warped by ego
    pose) feed the temporal branch, then this frame is pushed.
    ``evaluated`` counts every query sent through the decoder:
    at most ``max_coarse`` coarse cells (nearest first) plus at most ``max_fine`` fine voxels,
    so per-tick work is capped no matter how large the grid is.
    ``stage_ms`` holds per-stage wall time (CUDA-synchronised on GPU).
    """
    import time

    torch = _torch()
    grid = net.grid
    dev = next(net.parameters()).device
    cuda = dev.type == "cuda"
    ms: dict[str, float] = {}

    def lap(name: str, t0: float) -> float:
        if cuda:
            torch.cuda.synchronize(dev)
        t1 = time.perf_counter()
        ms[name] = ms.get(name, 0.0) + (t1 - t0) * 1e3
        return t1

    cidx = nearest_coarse_queries(grid, max_coarse)
    cq = np.concatenate([grid.coarse_centers(cidx), np.ones((len(cidx), 1))], 1)
    with torch.no_grad():
        t = time.perf_counter()
        feat = net.encode(images)
        t = lap("backbone", t)
        cv = None if cam_valid is None else torch.as_tensor(np.asarray(cam_valid, np.float32))[None].to(feat)
        ca = None
        if cam_t is not None and now is not None:
            ca = torch.as_tensor(float(now) - np.asarray(cam_t, np.float64), dtype=feat.dtype)[None].to(feat)
        ego_delta = None
        if memory is not None:
            if ego_pose is not None and not ego_from_vehicle:
                raise ValueError("ego_pose given but ego_from_vehicle is off: runtime ego motion is camera-only (VO)")
            if ego_pose is None:  # default: camera-only visual odometry from cached features
                if memory.frames and tuple(memory.frames[-1][0].shape) == tuple(feat.shape):
                    ego_delta = net.vo(feat, memory.frames[-1][0], cv, ca)[0].tolist()
                    memory.odom = pose_step(memory.odom, ego_delta)
                ego_pose = memory.odom
            t = lap("vo", t)
            mf, mx, mv, ma = memory.tensors(ego_pose, feat, now)
            mem = (mf, mx, mv, cv, ca, ma if (cam_t is not None and now is not None) else None)
            memory.push(feat, ego_pose, cam_t)
            t = lap("memory", t)
        else:
            mem = (None, None, None, cv, ca, None)
        cp = torch.sigmoid(net.decode(feat, torch.tensor(cq, dtype=torch.float32, device=dev)[None], *mem)["occ"][0])
        t = lap("query_coarse", t)
        per = int(np.prod(grid.coarse))
        k = max(0, int(max_fine) // per)
        # over budget: keep the NEAREST occupied coarse cells (planner-relevant first; far and
        # occluded space is the least supervised). Ties by distance never depend on prob ordering.
        cand = np.nonzero((cp > tau).cpu().numpy())[0]
        dist = np.hypot(*grid.coarse_centers(cidx[cand])[:, :2].T) if len(cand) else np.zeros(0)
        keep = cand[np.argsort(dist, kind="stable")][:k]
        evaluated = len(cidx)
        base = {"evaluated": evaluated, "coarse_prob": cp.cpu(), "ego_delta": ego_delta, "stage_ms": ms,
                "ego_pose": None if memory is None else tuple(map(float, ego_pose)),
                "ego_source": "vehicle" if (ego_from_vehicle and memory is not None) else "vo"}
        if len(keep) == 0:
            z = torch.zeros((0, 3), dtype=torch.long)
            return {"idx": z, "prob": torch.zeros(0), "sem": torch.zeros((0, len(OCC_CLASSES))),
                    "flow": torch.zeros((0, 2)), "box": torch.zeros((0, 6)), **base}
        fidx = coarse_children(grid, cidx[keep])
        fq = np.concatenate([grid.centers(fidx), np.zeros((len(fidx), 1))], 1)
        hid = net.hidden(feat, torch.tensor(fq, dtype=torch.float32, device=dev)[None], *mem)
        t = lap("query_fine", t)
        out = {}
        for name in net.HEADS:
            out[name] = net.head(name, hid)[0]
            t = lap(f"head_{name}", t)
        evaluated += len(fidx)
        fp = torch.sigmoid(out["occ"]).cpu()
        sel = fp > tau
        base["evaluated"] = evaluated
        return {"idx": torch.from_numpy(fidx)[sel], "prob": fp[sel], "sem": out["sem"].cpu()[sel],
                "flow": out["flow"].cpu()[sel], "box": out["box"].cpu()[sel], **base}


def export_onnx(net: Any, path: Path | str, *, n_queries: int = 1024, image_hw: tuple[int, int] = (480, 640)) -> Path:
    """FP32 ONNX of the full forward incl. temporal memory and per-camera freshness; Q is dynamic."""
    torch = _torch()
    net = net.eval()
    n = len(net.cam_ids)
    images = torch.zeros(1, n, 3, int(image_hw[0]), int(image_hw[1]))
    queries = torch.zeros(1, int(n_queries), 3)
    with torch.no_grad():
        f = net.encode(images)
    k = max(1, net.mem_len)
    mem_feat = torch.zeros((1, k) + tuple(f.shape[1:]))
    mem_xf = torch.zeros(1, k, 2, 3)
    mem_valid = torch.zeros(1, k, n)
    cam_valid = torch.ones(1, n)
    cam_age = torch.zeros(1, n)
    mem_age = torch.zeros(1, k, n)

    class _Wrap(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, images, queries, mem_feat, mem_xf, mem_valid, cam_valid, cam_age, mem_age):
            out = self.inner(images, queries, mem_feat, mem_xf, mem_valid, cam_valid, cam_age, mem_age)
            return out["occ"], out["sem"], out["flow"], out["box"]

    names = ["images", "queries", "mem_feat", "mem_xf", "mem_valid", "cam_valid", "cam_age", "mem_age"]
    path = Path(path)
    torch.onnx.export(
        _Wrap(net), (images, queries, mem_feat, mem_xf, mem_valid, cam_valid, cam_age, mem_age), str(path),
        dynamo=False, opset_version=17, input_names=names, output_names=["occ", "sem", "flow", "box"],
        dynamic_axes={"queries": {1: "q"}, "occ": {1: "q"}, "sem": {1: "q"}, "flow": {1: "q"}, "box": {1: "q"}},
    )
    return path


def memory_budget(net: Any, *, n_cams: int = 8, hw: tuple[int, int] = (480, 640), n_queries: int | None = None,
                  batch: int = 2, train_queries: int = 12288) -> dict[str, Any]:
    """Upper-bound memory from a real CPU forward with hooks (an estimate, not a GPU reading).

    infer_bytes = params + inputs + the sum of every module output in one
    no-grad forward (no reuse assumed, so it over-counts). train bytes =
    fp32 params x4 (weights, grads, Adam m and v) + activations x2 (kept for
    backward plus their grads) x batch; fp16 halves the activations.
    """
    torch = _torch()
    grid = net.grid
    if n_queries is None:
        n_queries = min(int(np.prod(grid.coarse_dims)), DEFAULT_MAX_COARSE) + DEFAULT_MAX_FINE
    sizes: list[int] = []

    def hook(_m, _i, out):
        outs = out if isinstance(out, (tuple, list)) else [out]
        for o in outs:
            if hasattr(o, "element_size"):
                sizes.append(o.numel() * o.element_size())

    leaves = [m for m in net.modules() if not list(m.children())]
    handles = [m.register_forward_hook(hook) for m in leaves]
    n = len(net.cam_ids)
    try:
        with torch.no_grad():
            img = torch.zeros(1, n, 3, hw[0], hw[1])
            f = net.encode(img)
            k = max(1, net.mem_len)
            mem = (torch.zeros((1, k) + tuple(f.shape[1:])), torch.zeros(1, k, 2, 3), torch.ones(1, k, n))
            mem_bytes = mem[0].numel() * 4
            sizes.clear()
            net(img, torch.zeros(1, n_queries, 3), *mem)  # memory valid: temporal branch counted
            act_infer = sum(sizes)
            sizes.clear()
            net(img, torch.zeros(1, train_queries, 3), *mem)
            act_train = sum(sizes)
    finally:
        for hnd in handles:
            hnd.remove()
    params = sum(p.numel() for p in net.parameters()) * 4
    scale = float(n_cams) / float(n)
    in_bytes = n_cams * 3 * hw[0] * hw[1] * 4
    mem_b = int(mem_bytes * scale)  # cached past features (no grad, also in training)
    infer = params + in_bytes + mem_b + int(act_infer * scale) + n_queries * 4 * 4
    past_in = in_bytes * max(1, net.mem_len)  # past frames' images, encoded under no_grad
    train32 = params * 4 + int(act_train * scale) * 2 * batch + (in_bytes + past_in + mem_b) * batch
    train16 = params * 4 + params // 2 + int(act_train * scale) * batch + (in_bytes + past_in) * batch // 2 \
        + mem_b * batch
    return {
        "params": params // 4,
        "infer_bytes": int(infer),
        "mem_cache_bytes": int(mem_b),  # K x cams x C x h x w x 4 B of cached past backbone maps
        "train_fp32_bytes": int(train32),
        "train_fp16_bytes": int(train16),
        "batch": batch,
        "n_queries": n_queries,
        "train_queries": train_queries,
        "kind": "cpu_hook_upper_bound_estimate",
    }


# ---------------------------------------------------------------------------
# Hardware profiles: one architecture, sized by config, never forked code.
# Knobs that keep a checkpoint loadable: long_side, cams, grid range/voxel, mem_len, max_fine,
# precision (rig, grid and position-scale buffers are rebuilt, not saved; age_w is MAX_MEM_LEN wide).
# A changed voxel range rescales the position encoding, so it loads but needs a fine-tune.
# Knobs that force a retrain: width, hidden, depth (weight shapes change; load_compatible refuses).
PROFILES: dict[str, dict[str, Any]] = {
    "1080ti": {"width": 48, "hidden": 128, "depth": 1, "long_side": 640, "cams": None, "voxel_m": 0.4,
               "mem_len": 2, "max_fine": 8192, "precision": "fp32"},
    "1080ti_2cam": {"width": 48, "hidden": 128, "depth": 1, "long_side": 640, "cams": ["main", "wide"],
                    "voxel_m": 0.4, "mem_len": 2, "max_fine": 8192, "precision": "fp32"},
    "1080ti_3cam": {"width": 48, "hidden": 128, "depth": 1, "long_side": 640, "cams": ["main", "wide", "narrow"],
                    "voxel_m": 0.4, "mem_len": 2, "max_fine": 8192, "precision": "fp32"},
    # Michael 12:00 AM option: drop narrow, raise main's resolution (same weights; trade-off in the spec)
    "1080ti_mainwide_hi": {"width": 48, "hidden": 128, "depth": 1, "long_side": 960, "cams": ["main", "wide"],
                           "voxel_m": 0.4, "mem_len": 2, "max_fine": 8192, "precision": "fp32"},
    "high": {"width": 128, "hidden": 256, "depth": 3, "long_side": 960, "cams": None, "voxel_m": 0.2,
             "mem_len": 4, "max_fine": 32768, "precision": "fp16"},
}
PRECISIONS = ("fp32", "fp16", "int8")  # int8 = TensorRT export path: NOT implemented here (UNPROVEN)


def build_profile(name: str, *, poses: Sequence[Any] | None = None) -> tuple[Any, OccGrid, dict[str, Any]]:
    """(net, grid, cfg) for a named profile. ``poses`` overrides the rig (tests)."""
    cfg = dict(PROFILES[name])
    if cfg["precision"] not in PRECISIONS:
        raise ValueError(f"precision {cfg['precision']}")
    grid = OccGrid(voxel_m=float(cfg["voxel_m"]))
    if poses is None:
        poses = rig_poses(cfg["cams"])
    elif cfg["cams"]:
        poses = [p for p in poses if p.cam_id in cfg["cams"]]
    net = _build_net_class()(poses, grid, width=cfg["width"], hidden=cfg["hidden"], depth=cfg["depth"],
                             mem_len=cfg["mem_len"])
    return net, grid, cfg


def load_compatible(net: Any, state: dict[str, Any]) -> None:
    """Load a checkpoint from any profile; raise ValueError naming every shape mismatch (= retrain)."""
    own = net.state_dict()
    missing = sorted(set(own) - set(state))
    extra = sorted(set(state) - set(own))
    shape = sorted(k for k in set(own) & set(state) if tuple(own[k].shape) != tuple(state[k].shape))
    if missing or extra or shape:
        raise ValueError(f"checkpoint not loadable in this profile (retrain): shape={shape[:6]} "
                         f"missing={missing[:6]} extra={extra[:6]}")
    net.load_state_dict(state, strict=True)
