"""LSS BEV head on the SharedStem s4/s8/s16 taps (GVD 2.0 spec, Phase 5 prototype).

Off by default. Nothing in the driving path imports this module: SceneNet,
its forward and its ONNX export are unchanged. ``build_bev_head`` returns
None unless the caller sets ``bev_lss.enabled``.

Guarantees (each one has a check in ``scripts/test_bev_lss.py``):

G1  Output is ``B x len(BEV_CHANNELS) x 128 x 128``; cells are 0.5 m.
G2  Grid is ego-centric in the GVD vehicle frame (x right, y forward, z up,
    ``config/cameras.yaml``). Row 0 is the far-forward edge (y = +32 m),
    column 0 is the left edge (x = -32 m). Cell (row, col) covers
    y in [32 - 0.5 (row + 1), 32 - 0.5 row), x in [-32 + 0.5 col, -32 + 0.5 (col + 1)).
G3  Points outside the grid or the z band are dropped, never clamped onto
    an edge cell.
G4  Per-camera intrinsics K and cam-to-ego extrinsics are explicit inputs.
    The recordings carry no K, so ``camera_intrinsics`` derives a pinhole
    from ``fov_v`` and the letterboxed 192x256 crop (square pixels, centred
    principal point, no distortion; UNVERIFIED against a recorded K).
G5  Unknown is the default. A cell no valid ray reaches is ``unknown``
    (logit forced high) and never ``observed_free`` (logit forced low).
    ``observed_free`` is allowed only where a minimum share of the cell's
    depth mass passed through it (``pass_mass / (hit_mass + pass_mass)
    >= PASS_FRAC_TAU``). Softmax residual mass is not enough: cells that
    only lie behind a confident depth hit (prior cumulative probability
    already ``OCCLUDE_HIT_TAU``) have free forced low and ``unknown``
    plus ``occluded`` forced high. This is structural, not a loss weight.
G6  The output is differentiable with respect to the stem taps and the
    depth head.
G7  No NaN or Inf, including a batch where no camera is valid.
G8  Deterministic for fixed inputs on CPU (CPU ``scatter_add`` is ordered;
    CUDA float atomics are not bit-exact, UNVERIFIED on GPU).
G9  Optional ego compensation per camera: a static point seen ``dt`` seconds
    after the reference time moves by ``+v * dt`` along +y (forward).
    The speed is a plain argument, default None = no compensation. It is
    not connected to any live input.
G10 Size: parameter count and fp32 activation memory for 8 cameras at batch 1
    are bounded (CPU-derived estimate, not a GPU measurement).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

# Torch-only module. Nothing on the torch-free path imports it (SceneNet does not).
import torch
import torch.nn.functional as F
from torch import nn

# Grid (spec: about 128x128 at about 0.5 m, ego-centric).
BEV_N = 128
BEV_CELL_M = 0.5
BEV_HALF_M = BEV_N * BEV_CELL_M / 2.0  # 32 m
BEV_Z_MIN_M = -2.0
BEV_Z_MAX_M = 4.0

# Output channels, in spec order (GVD 2.0 spec, ARCHITECTURE / BEV).
BEV_CHANNELS: tuple[str, ...] = (
    "road",
    "markings",
    "curbs",
    "sidewalks",
    "boundaries",
    "intersections",
    "vehicles",
    "static_obstacles",
    "occupied",
    "observed_free",
    "unknown",
    "occluded",
    "confidence",
    "elevation",
)
CH_FREE = BEV_CHANNELS.index("observed_free")
CH_UNKNOWN = BEV_CHANNELS.index("unknown")
CH_OCCLUDED = BEV_CHANNELS.index("occluded")
# Finite, so a forced logit never produces NaN in a loss.
FORCE_LOGIT = 30.0
# Free is allowed only when this share of a cell's depth mass passed through
# it: pass_mass / (hit_mass + pass_mass) >= PASS_FRAC_TAU. A raw `passed > 0`
# test cannot tell "seen through" from softmax residual behind a wall.
PASS_FRAC_TAU = 0.5
# A ray has a confident hit once cumulative depth probability reaches this.
# Bins past that hit are occluded. 0.95 is below the weakest wall case the
# tests pin (p_hit = 0.987 at softmax margin 8).
OCCLUDE_HIT_TAU = 0.95

# Categorical depth (audit item 9: 40 log bins, 1-48 m).
DEPTH_BINS = 40
DEPTH_MIN_M = 1.0
DEPTH_MAX_M = 48.0

# SharedStem crop and the tap the frustum lives on.
CROP_H = 192
CROP_W = 256
FEAT_STRIDE = 8
FPN_CH = 64
CTX_CH = 32
CAM_FEAT = 4  # fx / FX_REF, sin yaw, cos yaw, mount height
FX_REF_PX = 300.0
STEM_TAP_STRIDES = (4, 8, 16)

DEFAULT_ENABLED = False


def depth_bin_centers(bins: int = DEPTH_BINS, lo: float = DEPTH_MIN_M, hi: float = DEPTH_MAX_M) -> np.ndarray:
    """Log-spaced bin centres in metres, strictly increasing, inside [lo, hi]."""
    if bins < 1 or not (0.0 < lo < hi):
        raise ValueError(f"bad depth bins: bins={bins} lo={lo} hi={hi}")
    edges = np.geomspace(lo, hi, bins + 1)
    return (0.5 * (edges[1:] + edges[:-1])).astype(np.float32)


def content_rect(res_w: int, res_h: int, crop_w: int = CROP_W, crop_h: int = CROP_H) -> tuple[int, int, int, int]:
    """Letterbox of a ``res_w x res_h`` camera into the crop: (nw, nh, x0, y0).

    Same arithmetic as ``scene_net.letterbox_rect``.
    """
    scale = min(crop_w / float(res_w), crop_h / float(res_h))
    nw = max(1, min(crop_w, int(round(res_w * scale))))
    nh = max(1, min(crop_h, int(round(res_h * scale))))
    return nw, nh, (crop_w - nw) // 2, (crop_h - nh) // 2


def camera_intrinsics(fov_v_deg: float, res_w: int, res_h: int, crop_w: int = CROP_W, crop_h: int = CROP_H) -> np.ndarray:
    """Pinhole K (3x3) on the letterboxed crop, from the vertical fov.

    fy = (content_h / 2) / tan(fov_v / 2), fx = fy, principal point at the
    content centre. UNVERIFIED: square pixels, no distortion.
    """
    if not (0.0 < float(fov_v_deg) < 180.0):
        raise ValueError(f"fov_v must be in (0, 180), got {fov_v_deg}")
    nw, nh, x0, y0 = content_rect(res_w, res_h, crop_w, crop_h)
    fy = (nh / 2.0) / math.tan(math.radians(float(fov_v_deg)) / 2.0)
    return np.array(
        [[fy, 0.0, x0 + nw / 2.0], [0.0, fy, y0 + nh / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def cam_to_ego(pos_m: Sequence[float], yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """4x4 camera (x right, y down, z forward) to ego (x right, y forward, z up).

    Yaw is positive to the right, pitch positive up, roll 0, matching
    ``sensors.cameras.yaw_pitch_to_dir_up``.
    """
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    fwd = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    down = np.cross(fwd, right)
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = np.stack([right, down, fwd], axis=1)
    out[:3, 3] = np.asarray(pos_m, dtype=np.float32)
    return out


@dataclass(frozen=True)
class CamGeom:
    cam_id: str
    K: np.ndarray  # 3x3 on the crop
    T: np.ndarray  # 4x4 cam to ego
    content: tuple[int, int, int, int]  # nw, nh, x0, y0 on the crop
    yaw_deg: float


def rig_geometry(
    cam_ids: Sequence[str], cameras_cfg: Mapping[str, Any], model: str | None = None
) -> list[CamGeom]:
    """Per-camera geometry from a parsed ``cameras.yaml``. A missing field raises.

    When ``cameras_cfg`` contains ``vehicle_overrides``, ``model`` is required
    and the rig is resolved through ``python.sensors.rig_overrides.rig_for_model``
    (the default collection ego is the eSBR, whose repeater mounts differ).
    """
    from python.sensors.rig_overrides import OVERRIDES_KEY, rig_for_model

    by_id = {str(c["id"]): c for c in cameras_cfg.get("cameras") or [] if isinstance(c, Mapping) and c.get("id")}
    for cid in cam_ids:
        if str(cid) not in by_id:
            raise KeyError(f"camera {cid!r} is not in cameras.yaml")
    if cameras_cfg.get(OVERRIDES_KEY) is not None and model is None:
        raise ValueError(
            "cameras.yaml has vehicle_overrides; pass model= (e.g. 'sbr' or 'etk800') "
            "so rig_geometry can apply them via rig_for_model"
        )
    cameras_cfg = rig_for_model(cameras_cfg, model)
    by_id = {str(c["id"]): c for c in cameras_cfg.get("cameras") or [] if isinstance(c, Mapping) and c.get("id")}
    out: list[CamGeom] = []
    for cid in cam_ids:
        spec = by_id.get(str(cid))
        if spec is None:
            raise KeyError(f"camera {cid!r} is not in cameras.yaml")
        res_w, res_h = (int(v) for v in spec["live_res"])
        out.append(
            CamGeom(
                cam_id=str(cid),
                K=camera_intrinsics(float(spec["fov_v"]), res_w, res_h),
                T=cam_to_ego(spec["pos_m"], float(spec["yaw_deg"]), float(spec["pitch_deg"])),
                content=content_rect(res_w, res_h),
                yaw_deg=float(spec["yaw_deg"]),
            )
        )
    return out


def pixel_valid_mask(geoms: Sequence[CamGeom], stride: int = FEAT_STRIDE) -> np.ndarray:
    """``N x h x w`` float mask: 1 where the feature-pixel centre is inside the letterbox content."""
    h, w = CROP_H // stride, CROP_W // stride
    u = (np.arange(w) + 0.5) * stride
    v = (np.arange(h) + 0.5) * stride
    out = np.zeros((len(geoms), h, w), dtype=np.float32)
    for i, g in enumerate(geoms):
        nw, nh, x0, y0 = g.content
        rows = (v >= y0) & (v < y0 + nh)
        cols = (u >= x0) & (u < x0 + nw)
        out[i] = rows[:, None] & cols[None, :]
    return out


def bev_cell(x_m: float, y_m: float) -> tuple[int, int] | None:
    """(row, col) of an ego-frame ground point, or None outside the grid (G2)."""
    row = math.floor((BEV_HALF_M - float(y_m)) / BEV_CELL_M)
    col = math.floor((float(x_m) + BEV_HALF_M) / BEV_CELL_M)
    if 0 <= row < BEV_N and 0 <= col < BEV_N:
        return row, col
    return None


def stem_taps(stem: Any, x: Any) -> tuple[Any, Any, Any]:
    """s4, s8, s16 ReLU outputs of ``SharedStem.conv`` (read only, no new stem layers).

    The final tap equals ``stem.features(x)``.
    """
    relu_outs = []
    h = x
    for layer in stem.conv:
        h = layer(h)
        if isinstance(layer, nn.ReLU):
            relu_outs.append(h)
    if len(relu_outs) != 4:
        raise RuntimeError(f"SharedStem has {len(relu_outs)} stages, the BEV taps need 4")
    return relu_outs[1], relu_outs[2], relu_outs[3]

def frustum_ego_points(
    K: Any,
    T: Any,
    depths: Any,
    feat_hw: tuple[int, int],
    stride: int = FEAT_STRIDE,
    time_offset_s: Any = None,
    ego_speed_mps: Any = None,
) -> Any:
    """Ego-frame xyz of every (camera, depth bin, feature pixel): ``B x N x D x h x w x 3``.

    ``K``: B x N x 3 x 3; ``T``: B x N x 4 x 4 (cam to ego); ``depths``: D metres
    along the optical axis. Optional ego compensation (G9): ``time_offset_s``
    (B x N, camera time minus reference time) and ``ego_speed_mps`` (B) both set
    shift every point by ``+v * dt`` along +y. Either one None means no shift.
    """
    h, w = feat_hw
    dev, dt = K.device, K.dtype
    u = (torch.arange(w, device=dev, dtype=dt) + 0.5) * stride
    v = (torch.arange(h, device=dev, dtype=dt) + 0.5) * stride
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    fx = K[..., 0, 0][..., None, None]
    fy = K[..., 1, 1][..., None, None]
    cx = K[..., 0, 2][..., None, None]
    cy = K[..., 1, 2][..., None, None]
    rays = torch.stack([(uu - cx) / fx, (vv - cy) / fy, torch.ones_like(uu).expand_as((uu - cx) / fx)], dim=-1)
    pts = rays.unsqueeze(2) * depths.view(1, 1, -1, 1, 1, 1)  # B N D h w 3
    rot = T[..., :3, :3]
    trans = T[..., :3, 3]
    ego = torch.einsum("bnij,bndhwj->bndhwi", rot, pts) + trans[:, :, None, None, None, :]
    return ego_compensate(ego, time_offset_s, ego_speed_mps)

def ego_compensate(points: Any, time_offset_s: Any = None, ego_speed_mps: Any = None) -> Any:
    """Shift camera points into the reference time (G9). Pure; default is identity.

    ``points``: B x N x ... x 3 ego xyz. ``time_offset_s``: B x N seconds
    (camera time minus reference). ``ego_speed_mps``: B forward speed. A
    static point seen dt after the reference is at y + v * dt at the reference.
    """
    if time_offset_s is None or ego_speed_mps is None:
        return points
    shift = ego_speed_mps.to(points.dtype).view(-1, 1) * time_offset_s.to(points.dtype)  # B N
    view = list(shift.shape) + [1] * (points.dim() - 3)
    dy = shift.view(*view)
    return torch.stack([points[..., 0], points[..., 1] + dy, points[..., 2]], dim=-1)

def depth_survival(prob: Any, dim: int = 1) -> Any:
    """Depth mass strictly beyond each bin along ``dim``: sum_{j > i} p_j (the ray passed bin i)."""
    return (prob.flip(dim).cumsum(dim).flip(dim) - prob).clamp(min=0.0)


def depth_already_hit(prob: Any, tau: float = OCCLUDE_HIT_TAU, dim: int = 2) -> Any:
    """True at bins already behind a confident hit: prior cumulative mass >= ``tau``."""
    prior = prob.cumsum(dim) - prob
    return prior >= tau


def bev_flat_index(points: Any) -> tuple[Any, Any]:
    """Flat cell index (row * N + col) and a validity mask for ego xyz points (G2, G3)."""
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    row = torch.floor((BEV_HALF_M - y) / BEV_CELL_M)
    col = torch.floor((x + BEV_HALF_M) / BEV_CELL_M)
    ok = (row >= 0) & (row < BEV_N) & (col >= 0) & (col < BEV_N) & (z >= BEV_Z_MIN_M) & (z < BEV_Z_MAX_M)
    idx = torch.where(ok, row * BEV_N + col, torch.zeros_like(row)).long()
    return idx, ok

def splat(values: Any, idx: Any, ok: Any) -> Any:
    """Sum-pool ``values`` (B x P x C) into ``B x C x BEV_N x BEV_N`` (scatter_add).

    Points with ``ok`` False go to a discard cell that is sliced off, so the
    graph has static shapes and no boolean gather (ONNX friendly).
    """
    b, p, c = values.shape
    cells = BEV_N * BEV_N
    dump = torch.full_like(idx, cells)
    local = torch.where(ok, idx, dump)
    base = torch.arange(b, device=values.device).view(b, 1) * (cells + 1)
    flat = (local + base).reshape(-1)
    acc = values.new_zeros(b * (cells + 1), c)
    # scatter_add, not index_add: the legacy ONNX exporter turns index_add into a
    # ScatterElements with no reduction, which is wrong for repeated indices.
    # scatter_add exports as ScatterElements(reduction="add") (opset >= 16).
    acc = acc.scatter_add(0, flat.view(-1, 1).expand(b * p, c), values.reshape(b * p, c))
    acc = acc.view(b, cells + 1, c)[:, :cells]
    return acc.permute(0, 2, 1).reshape(b, c, BEV_N, BEV_N)

def _dws(cin: int, cout: int, stride: int = 1, dilation: int = 1) -> Any:
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride, dilation, dilation=dilation, groups=cin, bias=False),
        nn.GroupNorm(8, cin),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.GroupNorm(8, cout),
        nn.ReLU(),
    )

class _Res(nn.Module):
    def __init__(self, ch: int, dilation: int = 1) -> None:
        super().__init__()
        self.block = _dws(ch, ch, 1, dilation)

    def forward(self, x: Any) -> Any:
        return x + self.block(x)

class SpatialPath(nn.Module):
    """s16 context (dilated), light FPN to s8 (s4 pooled in), camera-aware depth and context heads."""

    def __init__(self, c4: int = 64, c8: int = 128, c16: int = 128, depth_bins: int = DEPTH_BINS) -> None:
        super().__init__()
        self.ctx16 = nn.Sequential(_Res(c16, 2), _Res(c16, 4))
        self.lat16 = nn.Conv2d(c16, FPN_CH, 1)
        self.lat8 = nn.Conv2d(c8, FPN_CH, 1)
        self.lat4 = nn.Conv2d(c4, FPN_CH, 1)
        self.film = nn.Linear(CAM_FEAT, 2 * FPN_CH)  # camera-aware (focal) conditioning
        self.fuse8 = _dws(FPN_CH, FPN_CH)
        self.depth = nn.Conv2d(FPN_CH, depth_bins, 1)
        self.ctx = nn.Conv2d(FPN_CH, CTX_CH, 1)

    def forward(self, f4: Any, f8: Any, f16: Any, cam_feat: Any) -> tuple[Any, Any]:
        p16 = self.lat16(self.ctx16(f16))
        p8 = self.lat8(f8) + F.interpolate(p16, size=f8.shape[-2:], mode="nearest")
        p8 = p8 + F.avg_pool2d(self.lat4(f4), 2)
        gamma, beta = self.film(cam_feat)[:, :, None, None].chunk(2, dim=1)
        p8 = self.fuse8(p8 * (1.0 + gamma) + beta)
        return self.depth(p8).softmax(dim=1), self.ctx(p8)

class BevEncoder(nn.Module):
    """Small residual U-Net at 128x128 -> 64x64 -> 128x128."""

    def __init__(self, cin: int, cout: int = len(BEV_CHANNELS)) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Conv2d(cin, 64, 3, 1, 1, bias=False), nn.GroupNorm(8, 64), nn.ReLU())
        self.down = nn.Sequential(_dws(64, 128, 2), _Res(128, 1), _Res(128, 2))
        self.up = _dws(128 + 64, 64)
        self.head = nn.Conv2d(64, cout, 1)

    def forward(self, x: Any) -> Any:
        a = self.inp(x)
        b = F.interpolate(self.down(a), size=a.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.up(torch.cat([a, b], dim=1)))

class BevLssHead(nn.Module):
    """Lift (depth x context), splat (scatter_add), BEV encoder. Taps in, ``B x 14 x 128 x 128`` out."""

    def __init__(
        self,
        depth_bins: int = DEPTH_BINS,
        pass_frac_tau: float = PASS_FRAC_TAU,
        occlude_hit_tau: float = OCCLUDE_HIT_TAU,
    ) -> None:
        super().__init__()
        self.spatial = SpatialPath(depth_bins=depth_bins)
        self.encoder = BevEncoder(CTX_CH + 2)
        self.pass_frac_tau = float(pass_frac_tau)
        self.occlude_hit_tau = float(occlude_hit_tau)
        self.register_buffer("depths", torch.from_numpy(depth_bin_centers(depth_bins)), persistent=False)

    @staticmethod
    def camera_features(K: Any, T: Any) -> Any:
        """B x N x CAM_FEAT: fx / FX_REF (focal conditioning), sin / cos yaw, mount height."""
        fwd = T[..., :3, 2]
        yaw = torch.atan2(fwd[..., 0], fwd[..., 1])
        return torch.stack([K[..., 0, 0] / FX_REF_PX, torch.sin(yaw), torch.cos(yaw), T[..., 2, 3]], dim=-1)

    def forward(
        self,
        f4: Any,
        f8: Any,
        f16: Any,
        K: Any,
        T: Any,
        cam_valid: Any = None,
        pixel_valid: Any = None,
        time_offset_s: Any = None,
        ego_speed_mps: Any = None,
    ) -> dict[str, Any]:
        """Taps are ``B x N x C x H x W``. ``K`` B x N x 3 x 3, ``T`` B x N x 4 x 4.

        ``cam_valid`` (B x N, 1 = use this camera) and ``pixel_valid``
        (N x h x w letterbox mask) are optional. Returns ``logits`` (B x 14 x 128 x 128),
        ``hit_mass``, ``pass_mass`` (B x 1 x 128 x 128) and ``depth_prob``.
        """
        b, n = f8.shape[:2]
        flat = lambda t: t.reshape(b * n, *t.shape[2:])  # noqa: E731
        cam_feat = self.camera_features(K, T).reshape(b * n, CAM_FEAT)
        prob, ctx = self.spatial(flat(f4), flat(f8), flat(f16), cam_feat)
        d, h, w = prob.shape[1:]
        weight = torch.ones(b, n, 1, 1, 1, dtype=prob.dtype, device=prob.device)
        if cam_valid is not None:
            weight = weight * cam_valid.to(prob.dtype).view(b, n, 1, 1, 1)
        if pixel_valid is not None:
            weight = weight * pixel_valid.to(prob.dtype).view(-1 if pixel_valid.dim() == 4 else 1, n, 1, h, w)
        prob = prob.view(b, n, d, h, w)
        beyond = depth_survival(prob, dim=2)
        prob_w = prob * weight
        beyond_w = beyond * weight
        blocked = depth_already_hit(prob, self.occlude_hit_tau, dim=2).to(dtype=prob.dtype)
        unblocked = 1.0 - blocked
        occ_w = blocked * weight
        clear_w = (prob_w + beyond_w) * unblocked
        ctx = ctx.view(b, n, 1, CTX_CH, h, w)
        lifted = prob_w.unsqueeze(3) * ctx  # B N D C h w
        pts = frustum_ego_points(K, T, self.depths.to(K.dtype), (h, w), FEAT_STRIDE, time_offset_s, ego_speed_mps)
        idx, ok = bev_flat_index(pts)  # B N D h w
        values = torch.cat(
            [
                lifted.permute(0, 1, 2, 4, 5, 3),
                prob_w.unsqueeze(-1),
                beyond_w.unsqueeze(-1),
                occ_w.unsqueeze(-1),
                clear_w.unsqueeze(-1),
            ],
            dim=-1,
        ).reshape(b, n * d * h * w, CTX_CH + 4)
        bev = splat(values, idx.reshape(b, -1), ok.reshape(b, -1))
        feat = bev[:, :CTX_CH]
        hit = bev[:, CTX_CH : CTX_CH + 1]
        passed = bev[:, CTX_CH + 1 : CTX_CH + 2]
        occ = bev[:, CTX_CH + 2 : CTX_CH + 3]
        clear = bev[:, CTX_CH + 3 : CTX_CH + 4]
        logits = self.encoder(torch.cat([feat, torch.log1p(hit), torch.log1p(passed)], dim=1))
        logits = apply_visibility(
            logits, hit, passed, occluded_mass=occ, clear_mass=clear, pass_frac_tau=self.pass_frac_tau
        )
        return {
            "logits": logits,
            "hit_mass": hit,
            "pass_mass": passed,
            "occluded_mass": occ,
            "clear_mass": clear,
            "depth_prob": prob,
            "bev_feat": feat,
        }

def apply_visibility(
    logits: Any,
    hit: Any,
    passed: Any,
    occluded_mass: Any = None,
    clear_mass: Any = None,
    pass_frac_tau: float = PASS_FRAC_TAU,
) -> Any:
    """G5: unobserved -> unknown, never free; free only where pass share >= ``pass_frac_tau``.

    Cells with ``occluded_mass > 0`` and no unblocked (``clear_mass``) observation
    are behind a confident hit: free is forced low, unknown and occluded high.
    """
    total = hit + passed
    pass_frac = passed / (total + 1e-12)
    free_ok = (total > 0) & (pass_frac >= pass_frac_tau)
    seen = total > 0
    behind = None
    if occluded_mass is not None:
        behind = occluded_mass > 0
        if clear_mass is not None:
            behind = behind & (clear_mass <= 0)
            seen = clear_mass > 0
        else:
            seen = seen & ~behind
        free_ok = free_ok & ~behind
    free = torch.where(free_ok, logits[:, CH_FREE : CH_FREE + 1], torch.full_like(hit, -FORCE_LOGIT))
    unknown = torch.where(seen, logits[:, CH_UNKNOWN : CH_UNKNOWN + 1], torch.full_like(hit, FORCE_LOGIT))
    if behind is None:
        return torch.cat(
            [logits[:, :CH_FREE], free, logits[:, CH_FREE + 1 : CH_UNKNOWN], unknown, logits[:, CH_UNKNOWN + 1 :]],
            dim=1,
        )
    occ = torch.where(behind, torch.full_like(hit, FORCE_LOGIT), logits[:, CH_OCCLUDED : CH_OCCLUDED + 1])
    return torch.cat(
        [
            logits[:, :CH_FREE],
            free,
            logits[:, CH_FREE + 1 : CH_UNKNOWN],
            unknown,
            occ,
            logits[:, CH_OCCLUDED + 1 :],
        ],
        dim=1,
    )

def rig_tensors(geoms: Sequence[CamGeom], batch: int = 1) -> dict[str, Any]:
    """K (B x N x 3 x 3), T (B x N x 4 x 4) and pixel_valid (N x h x w) tensors."""
    K = torch.from_numpy(np.stack([g.K for g in geoms])).unsqueeze(0).expand(batch, -1, -1, -1).contiguous()
    T = torch.from_numpy(np.stack([g.T for g in geoms])).unsqueeze(0).expand(batch, -1, -1, -1).contiguous()
    return {"K": K, "T": T, "pixel_valid": torch.from_numpy(pixel_valid_mask(geoms))}


def build_bev_head(cfg: Mapping[str, Any] | None = None) -> Any:
    """``BevLssHead`` when ``cfg['bev_lss']['enabled']`` is true, else None (default off)."""
    section = (cfg or {}).get("bev_lss") or {}
    if not bool(section.get("enabled", DEFAULT_ENABLED)):
        return None
    return BevLssHead(
        depth_bins=int(section.get("depth_bins", DEPTH_BINS)),
        pass_frac_tau=float(section.get("pass_frac_tau", PASS_FRAC_TAU)),
        occlude_hit_tau=float(section.get("occlude_hit_tau", OCCLUDE_HIT_TAU)),
    )
