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
G5  Unknown is the default, decided per ray by depth transmittance.
    For a ray (valid camera and pixel) the probability that it reached
    depth bin i is ``T_i = sum_{j >= i} p_j`` and that it passed through
    bin i is ``S_i = sum_{j > i} p_j``. A cell is seen only if some valid
    ray reached it (``T_i >= REACH_TAU``); otherwise ``unknown`` is forced
    high and ``observed_free`` forced low. ``observed_free`` is allowed only
    where some valid ray passed through the cell (``S_i >= FREE_PASS_TAU``)
    and the cell is seen. A cell valid rays landed in but none reached is
    behind a wall: ``occluded`` is forced high as well. Cells no valid ray
    landed in are unknown, not occluded. A partial wall (p < 1) or a second
    depth guess beyond it does not open the cells in between unless the
    ray more likely than not got there. Only ray points near the local
    ground count for seen / free / occluded. Each ray's surface is its
    median-depth point (the last bin it more likely than not reached); a
    ray with no surface in range (median in the last bin: sky) counts
    nowhere. The ground height of a cell is the lowest surface height seen
    within ``GROUND_WINDOW_CELLS`` (a soft minimum over every surface point
    in the window). A ray point counts if it is
    less than ``GROUND_BAND_M`` above the ground height of its own cell;
    a cell with no surface in its window has no ground height and no ray
    point counts there. Heights are ego-frame differences, so ego pitch,
    a grade and the (unverified) ego-origin height do not move the band.
    A ray that skims a car roof passes far above the road seen beside the
    car, so the hidden ground behind it and the car's own cells stay
    unknown; a ray that lands on the roof is above that ground too.
    Limits: objects lower than the band (curbs, debris under 0.5 m) can be
    passed over; where no ground is visible within the window (a wide
    shadow, a lot full of cars) the lowest visible surface stands in for
    the ground. A surface in the local window counts toward that ground
    only if it sits within ``GROUND_BAND_M`` of a wider-scale soft-min
    ground (``GROUND_WIDE_WINDOW_CELLS``); a barrier wider than the local
    window cannot lift hidden cells into the free band. A ray that has
    already passed a cell whose surface sits at least ``GROUND_BAND_M``
    above that wider-scale ground does not count later points as seen or
    free. A camera's ray also does not count seen or free in a cell whose
    ground is behind a surface at least ``RAISED_OCC_M`` above the local
    ground on that camera's sight line, so a 0.6 m barrier that is too
    low to be "raised" for the band still hides the ground in its umbra.
    This is structural, not a loss weight.
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
# Visibility cutoffs on per-ray transmittance (G5). Both are the majority
# decision: a ray counts as having reached (passed through) a bin only when
# the depth head says that is more likely than not. Not tuned to a fixture:
# any wall the head is more than half sure of hides what is behind it, and
# a wall it is less than half sure of does not.
REACH_TAU = 0.5  # seen: T_i = sum_{j >= i} p_j >= REACH_TAU
FREE_PASS_TAU = 0.5  # may be free: S_i = sum_{j > i} p_j >= FREE_PASS_TAU
# Ground band (G5): a ray point counts for seen / free / occluded only if it is
# less than GROUND_BAND_M above the local ground height (the lowest surface
# seen within GROUND_WINDOW_CELLS of its cell). 0.5 m is below a car's sill
# and roof, so a ray passing over (or landing on) a vehicle never opens it.
GROUND_BAND_M = 0.5
# Window for the local ground height: 13 cells = +-3 m, wider than the gap
# between neighbouring ground hits along a ray column out to ~25 m and than a
# car's half width, so road beside a vehicle is in every window over it.
GROUND_WINDOW_CELLS = 13
# Wider-scale ground: soft-min surface height over this many cells (21 = +-5 m).
# 6 % over 5 m is 0.30 m, still inside the 0.5 m band, so a grade keeps its
# road surfaces; a 0.6 m barrier sits outside the band and is rejected.
# The wall is only 0.3 m thick, so the road in front is in this window.
GROUND_WIDE_WINDOW_CELLS = 21
# Soft minimum of the surface heights in a window: -log(mean(exp(-k z))) / k,
# with exp taken relative to BEV_Z_MIN_M. With k = 10 / m a surface 1.5 m
# above the road (a roof) weighs exp(-15) as much as the road; the exponent
# stays in [-60, 0] over the z band, well inside float32.
GROUND_SOFTMIN_K = 10.0
# A surface this far above the local ground is a 2.5D occluder for the
# camera-to-ground sight line. 0.3 m is below GROUND_BAND_M (so a 0.6 m
# wall's front-face hits, which sit at ~0.35-0.45 m and never become
# "raised" for the band, still close the umbra) and above a 6 % grade's
# residual versus the local soft-min.
RAISED_OCC_M = 0.3
# Smooth max exponent (per metre) for per-cell occluder top via scatter_add.
OCC_SOFTMAX_K = 50.0
# Dilate those occluders by one cell (+-0.5 m) so a sparsely sampled wall
# still closes, without eating the open ground just in front of a car.
# Sight line uses 12 steps (~2.5 m max spacing across 32 m, still under a
# 1-cell-dilated front after extend) with 0 clearance under the estimated
# occluder top bound. 14 steps exceeded the 25% activation-memory cap.
OCC_WINDOW_CELLS = 3
SIGHTLINE_STEPS = 12
SIGHTLINE_CLEARANCE_M = 0.0
# Occluders closer to the destination than this are ignored, so a 1-cell
# dilated wall does not close the last open cell in front of it, and a
# constant-depth fixture wall does not self-occlude, while close barriers
# (y = 3-5 m) still hide their near umbra. 1.2 m is tight enough for those
# near umbras and still leaves check_occlusion_branches a shared wide view.
SIGHTLINE_MARGIN_M = 1.2
# Laterally extend a raised front through unobserved cells in the same row
# (up to 4 m). A 24x32 feature grid can miss a low barrier outside a narrow
# camera's FOV while a wide camera still lands on the ground behind it.
# A pole or short wall with visible road beside it stops immediately.
OCC_FRONT_EXTEND_CELLS = 8
# Only extend a row that already has a wide raised front (3 m). A pole or
# a diagonal stair step must not grow a 4 m bar of phantom wall.
OCC_MIN_FRONT_CELLS = 6
# When a pixel row above a raised hit clears, the top sits at most this
# far above the hit. Raw z_up at close range can be >1 m and would hide
# ground that is geometrically visible over a low close wall.
OCC_GRAZE_CAP_M = 0.35

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


def depth_bin_inner_edges(bins: int = DEPTH_BINS, lo: float = DEPTH_MIN_M, hi: float = DEPTH_MAX_M) -> np.ndarray:
    """The ``bins - 1`` log-spaced edges between neighbouring depth bins (edge j sits between bins j and j + 1)."""
    depth_bin_centers(bins, lo, hi)  # same validation
    return np.geomspace(lo, hi, bins + 1)[1:-1].astype(np.float32)


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


def stem_taps(stem: Any, x: Any) -> tuple[Any, Any, Any, Any]:
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


def depth_transmittance(prob: Any, dim: int = 1) -> Any:
    """Probability the ray reached each bin along ``dim``: sum_{j >= i} p_j (1 - mass before the bin)."""
    return prob.flip(dim).cumsum(dim).flip(dim)


def visibility_gates(reach: Any, passed: Any, reach_tau: float = REACH_TAU, free_pass_tau: float = FREE_PASS_TAU) -> tuple[Any, Any]:
    """Per ray and bin: (reached, passed through) as 0/1 in ``reach``'s dtype. Both cutoffs are inclusive."""
    return (reach >= reach_tau).to(reach.dtype), (passed >= free_pass_tau).to(reach.dtype)


def ray_surface(reached: Any, dim: int = 2) -> tuple[Any, Any]:
    """Per ray: (one-hot of its surface bin along ``dim``, 1 if it has a surface in range else 0).

    ``reached`` is the 0/1 reach gate (1 up to the last bin the ray more likely than not reached, then 0).
    The surface is that last reached bin. A ray still reached at the last bin (sky, nothing within range)
    or reached nowhere has none.
    """
    d = reached.shape[dim]
    nxt = torch.cat([reached.narrow(dim, 1, d - 1), torch.zeros_like(reached.narrow(dim, 0, 1))], dim=dim)
    has = reached.narrow(dim, 0, 1) * (1.0 - reached.narrow(dim, d - 1, 1))
    return (reached - nxt) * has, has


def ground_height(z_sum: Any, w_sum: Any, window: int = GROUND_WINDOW_CELLS, k: float = GROUND_SOFTMIN_K) -> tuple[Any, Any]:
    """Local ground height per cell and a 0/1 "has ground" mask (both B x 1 x N x N).

    ``z_sum`` = splatted ``w * exp(-k (z - BEV_Z_MIN_M))`` of the ray surface points, ``w_sum`` = splatted
    ``w``. Both are summed over a ``window`` x ``window`` neighbourhood and the ground height is the soft
    minimum ``BEV_Z_MIN_M - log(z_sum / w_sum) / k`` of every surface point in it: about the mean height on
    an even grade, about the lowest height where raised surfaces (car faces) mix with the road. Cells with
    no surface in the window have none.
    """
    pad = window // 2
    zp = F.avg_pool2d(z_sum, window, stride=1, padding=pad, count_include_pad=True)
    wp = F.avg_pool2d(w_sum, window, stride=1, padding=pad, count_include_pad=True)
    has = (wp > 0) & (zp > 0)
    ratio = torch.where(has, zp / torch.where(has, wp, torch.ones_like(wp)), torch.ones_like(zp))
    ground = torch.where(has, BEV_Z_MIN_M - torch.log(ratio) / k, torch.zeros_like(zp))
    return ground, has.to(z_sum.dtype)


def two_scale_ground(
    z_sum: Any,
    w_sum: Any,
    local_window: int = GROUND_WINDOW_CELLS,
    wide_window: int = GROUND_WIDE_WINDOW_CELLS,
    k: float = GROUND_SOFTMIN_K,
    near_m: float = GROUND_BAND_M,
) -> tuple[Any, Any, Any, Any]:
    """Local ground from surfaces that sit near a wider-scale soft-min ground.

    The wide-scale ground is a cell-normalized soft-min over ``wide_window``
    cells (each occupied cell has equal weight 1, so dense near-range wall
    hits cannot outvote sparser road hits). A per-cell surface is kept only
    when it is less than ``near_m`` above that wide-scale height (and some
    wide-scale mass exists). The local soft-min then runs on the kept
    surfaces only, so a barrier wider than the local window cannot become
    the ground behind it. Surfaces at least ``near_m`` above the wide-scale
    ground are returned as a raised-surface mask.
    """
    cell_has = w_sum > 0
    safe_w = torch.where(cell_has, w_sum, torch.ones_like(w_sum))
    cell_z = torch.where(
        cell_has,
        BEV_Z_MIN_M - torch.log((z_sum / safe_w).clamp(min=1e-30)) / k,
        torch.zeros_like(z_sum),
    )
    cw_w = cell_has.to(z_sum.dtype)
    cw_z = torch.where(cell_has, torch.exp((-k * (cell_z - BEV_Z_MIN_M)).clamp(min=-60.0, max=0.0)), torch.zeros_like(z_sum))
    g_wide, has_wide = ground_height(cw_z, cw_w, window=wide_window, k=k)
    has_w = has_wide > 0
    near = cell_has & has_w & ((cell_z - g_wide) < near_m)
    raised = (cell_has & has_w & ((cell_z - g_wide) >= near_m)).to(z_sum.dtype)
    ground, has = ground_height(
        torch.where(near, z_sum, torch.zeros_like(z_sum)),
        torch.where(near, w_sum, torch.zeros_like(w_sum)),
        window=local_window,
        k=k,
    )
    return ground, has, raised, g_wide


def ray_crossed_raised(
    idx: Any,
    ok: Any,
    raised: Any,
    z: Any = None,
    z_top: Any = None,
    clearance: float = 0.0,
) -> Any:
    """Per-sample mask: an earlier bin of the same ray sits in a raised cell.

    ``idx`` / ``ok`` are B x N x D x h x w (or D-1 at the bin edges). ``raised``
    is B x 1 x N x N. Cumulative along the depth dimension, so an overflight
    that later re-enters the ground band stays blocked. When ``z`` / ``z_top``
    are given, a sample in a raised cell does not count if it is already at or
    above the occluder top (the ray is going over, not through).
    """
    b = raised.shape[0]
    ids = idx.reshape(b, -1).clamp(min=0, max=BEV_N * BEV_N - 1)
    at = raised.reshape(b, BEV_N * BEV_N).gather(1, ids).view_as(idx) * ok.to(raised.dtype)
    if z is not None and z_top is not None:
        top = z_top.reshape(b, BEV_N * BEV_N).gather(1, ids).view_as(idx)
        at = at * (z < top + clearance).to(at.dtype)
    # Exclusive prefix: any earlier bin was raised. cumsum of a 0/1 mask, not
    # cummax, so the ONNX opset-17 export stays legal.
    prior = (at.cumsum(dim=2) - at) > 0
    return prior.to(raised.dtype)


def extend_occluder_front(
    has_occ: Any,
    z_occ: Any,
    low_surf: Any,
    max_cells: int = OCC_FRONT_EXTEND_CELLS,
) -> tuple[Any, Any]:
    """Copy a raised front sideways through unobserved cells in the same row.

    ``low_surf`` marks cells with a near-ground surface (an opening). Walking
    stops there, so a 0.2 m pole does not hide the road beside it. Each step
    is a 1-cell pad, legal for ONNX opset-17.
    """
    occ = has_occ
    z = z_occ
    wide_row = (has_occ > 0).sum(dim=-1, keepdim=True) >= int(OCC_MIN_FRONT_CELLS)
    for _ in range(int(max_cells)):
        occ_l = F.pad(occ[:, :, :, :-1], (1, 0))
        occ_r = F.pad(occ[:, :, :, 1:], (0, 1))
        z_l = F.pad(z[:, :, :, :-1], (1, 0))
        z_r = F.pad(z[:, :, :, 1:], (0, 1))
        can = (occ <= 0) & (low_surf <= 0) & wide_row
        take_l = can & (occ_l > 0)
        take_r = can & (occ_r > 0) & (~take_l)
        z = torch.where(take_l, z_l, torch.where(take_r, z_r, z))
        # Cast before Max: opset 17 rejects Max on bool (take_l/take_r are masks).
        occ = torch.maximum(occ, (take_l | take_r).to(occ.dtype))
    return occ, z


def sightline_blocked(
    z_occ: Any,
    has_occ: Any,
    ground: Any,
    T: Any,
    n_steps: int = SIGHTLINE_STEPS,
    clearance: float = SIGHTLINE_CLEARANCE_M,
) -> Any:
    """Per-camera mask (B x Ncam x N x N): the camera-to-ground sight line hits a 2.5D occluder.

    ``z_occ`` / ``has_occ`` are B x 1 x N x N absolute heights of raised surfaces
    (dilated). ``ground`` is B x 1 x N x N, ``T`` is B x Ncam x 4 x 4 (cam-to-ego).
    Sampled along the open segment from the camera origin to the cell centre at
    the cell's ground height. A hit is an occluder whose top is more than
    ``clearance`` above the sight line.
    """
    b, ncam = T.shape[:2]
    dtype, device = z_occ.dtype, z_occ.device
    ys = 32.0 - 0.5 * (torch.arange(BEV_N, device=device, dtype=dtype) + 0.5)
    xs = -32.0 + 0.5 * (torch.arange(BEV_N, device=device, dtype=dtype) + 0.5)
    grid_y = ys.view(1, 1, BEV_N, 1).expand(b, ncam, BEV_N, BEV_N)
    grid_x = xs.view(1, 1, 1, BEV_N).expand(b, ncam, BEV_N, BEV_N)
    cam_x = T[..., 0, 3].to(dtype).reshape(b, ncam, 1, 1)
    cam_y = T[..., 1, 3].to(dtype).reshape(b, ncam, 1, 1)
    cam_z = T[..., 2, 3].to(dtype).reshape(b, ncam, 1, 1)
    dest_z = ground.to(dtype)
    t = torch.linspace(0, 1, n_steps + 2, device=device, dtype=dtype)[1:-1].view(1, 1, n_steps, 1, 1)
    sx = cam_x.unsqueeze(2) + t * (grid_x.unsqueeze(2) - cam_x.unsqueeze(2))
    sy = cam_y.unsqueeze(2) + t * (grid_y.unsqueeze(2) - cam_y.unsqueeze(2))
    sz = cam_z.unsqueeze(2) + t * (dest_z.unsqueeze(2) - cam_z.unsqueeze(2))
    grid = torch.stack([sx / 32.0, -sy / 32.0], dim=-1).reshape(b * ncam, n_steps, BEV_N * BEV_N, 2)
    occ_both = torch.cat([z_occ, has_occ], dim=1).unsqueeze(1).expand(b, ncam, 2, BEV_N, BEV_N).reshape(
        b * ncam, 2, BEV_N, BEV_N
    )
    sampled = F.grid_sample(occ_both, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    z_s = sampled[:, 0]
    h_s = sampled[:, 1]
    margin = float(SIGHTLINE_MARGIN_M)
    dx_d, dy_d = grid_x - cam_x, grid_y - cam_y
    r_d = torch.sqrt(dx_d * dx_d + dy_d * dy_d).unsqueeze(2)
    # Since s = cam + t * (dest - cam), radial distance r_s = t * r_d, so r_s < r_d - margin iff (1 - t) * r_d > margin.
    in_front = ((1.0 - t) * r_d > margin).reshape(b * ncam, n_steps, BEV_N * BEV_N)
    hit = (h_s > 0) & (z_s > sz.reshape(b * ncam, n_steps, BEV_N * BEV_N) + clearance) & in_front
    return hit.any(dim=1).view(b, ncam, BEV_N, BEV_N).to(dtype)


def near_ground(z: Any, ground: Any, band_m: float = GROUND_BAND_M) -> Any:
    """True where a height is less than ``band_m`` above the local ground height (exclusive)."""
    return (z - ground) < band_m


def _check_band(value: float) -> float:
    value = float(value)
    if not (0.0 < value <= BEV_Z_MAX_M - BEV_Z_MIN_M):
        raise ValueError(f"ground_band_m must be in (0, {BEV_Z_MAX_M - BEV_Z_MIN_M}], got {value}")
    return value


def _check_tau(name: str, value: float) -> float:
    value = float(value)
    if not (0.0 < value <= 1.0):
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


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
        reach_tau: float = REACH_TAU,
        free_pass_tau: float = FREE_PASS_TAU,
        ground_band_m: float = GROUND_BAND_M,
    ) -> None:
        super().__init__()
        self.spatial = SpatialPath(depth_bins=depth_bins)
        self.encoder = BevEncoder(CTX_CH + 2)
        self.reach_tau = _check_tau("reach_tau", reach_tau)
        self.free_pass_tau = _check_tau("free_pass_tau", free_pass_tau)
        self.ground_band_m = _check_band(ground_band_m)
        self.register_buffer("depths", torch.from_numpy(depth_bin_centers(depth_bins)), persistent=False)
        self.register_buffer("edges", torch.from_numpy(depth_bin_inner_edges(depth_bins)), persistent=False)

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
        ``hit_mass``, ``pass_mass``, ``clear_mass`` (sum of T over valid rays that reached the
        cell), ``through_mass`` (sum of S over valid rays that passed through it),
        ``occluded_mass`` (sum of 1 - T over valid rays that landed there but did not reach it),
        all B x 1 x 128 x 128, and ``depth_prob``. The three visibility masses count only ray
        points less than ``ground_band_m`` above the two-scale local ground height (G5),
        and only before the ray has passed a raised surface.
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
        pts = frustum_ego_points(K, T, self.depths.to(K.dtype), (h, w), FEAT_STRIDE, time_offset_s, ego_speed_mps)
        reach = depth_transmittance(prob, dim=2)
        reached, through = visibility_gates(reach, beyond, self.reach_tau, self.free_pass_tau)
        idx, ok = bev_flat_index(pts)  # B N D h w
        z = pts[..., 2].to(prob.dtype)
        surf, has_surf = ray_surface(reached, dim=2)  # B N D h w one-hot, B N 1 h w
        ray_w = has_surf * weight
        g_vals = torch.stack([surf * ray_w * torch.exp(-GROUND_SOFTMIN_K * (z - BEV_Z_MIN_M)), surf * ray_w], dim=-1)
        g_sum = splat(g_vals.reshape(b, n * d * h * w, 2), idx.reshape(b, -1), ok.reshape(b, -1))
        ground, has_ground, raised, g_wide = two_scale_ground(g_sum[:, 0:1], g_sum[:, 1:2])
        flat_g = (ground * has_ground).reshape(b, BEV_N * BEV_N)
        flat_has = has_ground.reshape(b, BEV_N * BEV_N)
        g_at = flat_g.gather(1, idx.reshape(b, -1)).view_as(z)
        # Threshold tracks the configured band: default 0.5 m -> 0.3 m, so a
        # 0.6 m wall's 0.35-0.45 m front-face hits still close the umbra; a
        # 2.0 m band (objects under the band may be passed over) raises it
        # so a 1.5 m car roof is not an occluder.
        occ_m = self.ground_band_m - (GROUND_BAND_M - RAISED_OCC_M)
        high = surf * ray_w * ((z - g_at) >= occ_m) * ok.to(prob.dtype)
        # When a ray hits a raised obstacle and the valid pixel row just above it
        # passes beyond that depth bin, the true top edge can sit up to that
        # upper ray's height at the hit depth. Bound z_occ by that height too.
        z_up = torch.cat([z[:, :, :, :1, :], z[:, :, :, :-1, :]], dim=3)
        pv_up = torch.cat([torch.zeros_like(weight[:, :, :, :1, :]), weight[:, :, :, :-1, :]], dim=3)
        above_cleared = torch.cat([torch.zeros_like(through[:, :, :, :1, :]), through[:, :, :, :-1, :]], dim=3) * pv_up
        # Bound the top by the clearing ray, but never more than GRAZE_CAP above the hit
        # (close-range row spacing is coarse). Hits with no clear-above bound are tall
        # of unknown height: if a cell has only those, use BEV_Z_MAX_M so a thin pole
        # whose pixels miss the true top still closes its far umbra.
        z_bound_hit = torch.minimum(torch.maximum(z, z_up), z + OCC_GRAZE_CAP_M)
        z_eff = torch.where(above_cleared > 0, z_bound_hit, z)
        bound = high * (above_cleared > 0).to(prob.dtype)
        unbound = high * (above_cleared <= 0).to(prob.dtype)
        bound_vals = torch.stack([bound * z_eff, bound], dim=-1).reshape(b, n * d * h * w, 2)
        bound_sum = splat(bound_vals, idx.reshape(b, -1), ok.reshape(b, -1))
        has_bound = (bound_sum[:, 1:2] > 0).to(prob.dtype)
        has_unbound = splat(unbound.reshape(b, n * d * h * w, 1), idx.reshape(b, -1), ok.reshape(b, -1))
        has_occ = ((bound_sum[:, 1:2] + has_unbound) > 0).to(prob.dtype)
        z_mean = torch.where(
            has_bound > 0,
            bound_sum[:, 0:1] / torch.where(has_bound > 0, bound_sum[:, 1:2], torch.ones_like(has_bound)),
            torch.zeros_like(has_bound),
        )
        z_mean_at = z_mean.reshape(b, BEV_N * BEV_N).gather(1, idx.reshape(b, -1)).view_as(z)
        dz = (z_eff - z_mean_at).clamp(min=-1.4, max=1.4)
        exp_sum = splat((bound * torch.exp(OCC_SOFTMAX_K * dz)).reshape(b, n * d * h * w, 1), idx.reshape(b, -1), ok.reshape(b, -1))
        z_bound = torch.where(
            has_bound > 0,
            z_mean + torch.log(exp_sum.clamp(min=1e-30)) / OCC_SOFTMAX_K,
            torch.full_like(has_occ, BEV_Z_MIN_M - 1.0),
        )
        z_occ = torch.where(has_bound > 0, z_bound, torch.where(has_occ > 0, torch.full_like(has_occ, BEV_Z_MAX_M), z_bound))
        safe_w = torch.where(g_sum[:, 1:2] > 0, g_sum[:, 1:2], torch.ones_like(g_sum[:, 1:2]))
        cell_z = torch.where(
            g_sum[:, 1:2] > 0,
            BEV_Z_MIN_M - torch.log((g_sum[:, 0:1] / safe_w).clamp(min=1e-30)) / GROUND_SOFTMIN_K,
            torch.zeros_like(g_sum[:, 0:1]),
        )
        low_surf = ((g_sum[:, 1:2] > 0) & ((cell_z - g_wide) < RAISED_OCC_M)).to(prob.dtype)
        # Hits on a wall front often splat into the cell just in front of the
        # wall. Seed the next farther cell (larger y) when it has no road
        # surface, then extend laterally along that true front.
        occ_far = F.pad(has_occ[:, :, 1:, :], (0, 0, 0, 1))
        z_far = F.pad(z_occ[:, :, 1:, :], (0, 0, 0, 1))
        n_occ = (has_occ > 0).sum(dim=-1, keepdim=True)
        thin_row = (n_occ >= 1) & (n_occ <= 2)
        seed = (has_occ <= 0) & (low_surf <= 0) & (occ_far > 0) & (~thin_row)
        has_occ = torch.maximum(has_occ, seed.to(has_occ.dtype))
        z_occ = torch.where(seed, torch.maximum(z_occ, z_far), z_occ)
        has_occ, z_occ = extend_occluder_front(has_occ, z_occ, low_surf)
        # A 1-2 cell front (a pole) has no reliable top bound: treat it as tall,
        # and do not dilate it (dilation would hide the visible road beside it).
        n_occ = (has_occ > 0).sum(dim=-1, keepdim=True)
        thin_row = (n_occ >= 1) & (n_occ <= 2)
        z_occ = torch.where((has_occ > 0) & thin_row, torch.full_like(z_occ, BEV_Z_MAX_M), z_occ)
        occ_d = F.max_pool2d(has_occ, OCC_WINDOW_CELLS, stride=1, padding=OCC_WINDOW_CELLS // 2)
        z_d = F.max_pool2d(z_occ, OCC_WINDOW_CELLS, stride=1, padding=OCC_WINDOW_CELLS // 2)
        has_occ = torch.where(thin_row, has_occ, occ_d)
        z_occ = torch.where(thin_row, z_occ, z_d)
        z_occ = torch.where(has_occ > 0, z_occ, torch.zeros_like(z_occ))
        crossed = 1.0 - ray_crossed_raised(idx, ok, raised, z, z_occ)
        blocked = sightline_blocked(z_occ, has_occ, g_wide, T)
        ids_c = idx.reshape(b, n, -1).clamp(min=0, max=BEV_N * BEV_N - 1)
        visible = (1.0 - blocked.reshape(b, n, BEV_N * BEV_N).gather(2, ids_c).view_as(idx)) * ok.to(prob.dtype)

        def _band(zs: Any, ids: Any) -> Any:
            g = flat_g.gather(1, ids.reshape(b, -1)).view_as(zs)
            ok_g = flat_has.gather(1, ids.reshape(b, -1)).view_as(zs)
            return near_ground(zs, g, self.ground_band_m).to(prob.dtype) * ok_g * ray_w

        band_w = _band(z, idx)  # B N D h w
        clear_w = reach * reached * band_w * crossed * visible
        through_w = beyond * through * band_w * crossed * visible
        occ_w = (1.0 - reach) * (1.0 - reached) * band_w
        # Visibility is also sampled at the edge between bins j and j + 1. In the surface band only
        # near-surface points count, and the bin centres alone leave whole cell rows unsampled at range
        # (bins ~1.4 m apart at 15 m, cells 0.5 m). An edge counts only the mass beyond bin j + 1
        # (S_{j+1}) for both reach and pass, so a surface anywhere in bin j + 1, or snapped into it
        # from just before the edge, never opens the edge point.
        pts_e = frustum_ego_points(K, T, self.edges.to(K.dtype), (h, w), FEAT_STRIDE, time_offset_s, ego_speed_mps)
        idx_e, ok_e = bev_flat_index(pts_e)
        z_e = pts_e[..., 2].to(prob.dtype)
        band_e = _band(z_e, idx_e)
        crossed_e = 1.0 - ray_crossed_raised(idx_e, ok_e, raised, z_e, z_occ)
        ids_e = idx_e.reshape(b, n, -1).clamp(min=0, max=BEV_N * BEV_N - 1)
        visible_e = (1.0 - blocked.reshape(b, n, BEV_N * BEV_N).gather(2, ids_e).view_as(idx_e)) * ok_e.to(prob.dtype)
        s_e = beyond[:, :, 1:]
        reached_e, through_e = visibility_gates(s_e, s_e, self.reach_tau, self.free_pass_tau)
        vis_e = torch.stack(
            [
                (1.0 - s_e) * (1.0 - reached_e) * band_e,
                s_e * reached_e * band_e * crossed_e * visible_e,
                s_e * through_e * band_e * crossed_e * visible_e,
            ],
            dim=-1,
        ).reshape(b, n * (d - 1) * h * w, 3)
        vis_e = splat(vis_e, idx_e.reshape(b, -1), ok_e.reshape(b, -1))
        ctx = ctx.view(b, n, 1, CTX_CH, h, w)
        lifted = prob_w.unsqueeze(3) * ctx  # B N D C h w
        values = torch.cat(
            [
                lifted.permute(0, 1, 2, 4, 5, 3),
                prob_w.unsqueeze(-1),
                beyond_w.unsqueeze(-1),
                occ_w.unsqueeze(-1),
                clear_w.unsqueeze(-1),
                through_w.unsqueeze(-1),
            ],
            dim=-1,
        ).reshape(b, n * d * h * w, CTX_CH + 5)
        bev = splat(values, idx.reshape(b, -1), ok.reshape(b, -1))
        feat = bev[:, :CTX_CH]
        hit = bev[:, CTX_CH : CTX_CH + 1]
        passed = bev[:, CTX_CH + 1 : CTX_CH + 2]
        occ = bev[:, CTX_CH + 2 : CTX_CH + 3]
        clear = bev[:, CTX_CH + 3 : CTX_CH + 4]
        thru = bev[:, CTX_CH + 4 : CTX_CH + 5] + vis_e[:, 2:3]
        occ = occ + vis_e[:, 0:1]
        clear = clear + vis_e[:, 1:2]
        logits = self.encoder(torch.cat([feat, torch.log1p(hit), torch.log1p(passed)], dim=1))
        logits = apply_visibility(logits, clear, thru, occluded_mass=occ)
        return {
            "logits": logits,
            "hit_mass": hit,
            "pass_mass": passed,
            "occluded_mass": occ,
            "clear_mass": clear,
            "through_mass": thru,
            "ground_height": ground * has_ground,
            "has_ground": has_ground,
            "depth_prob": prob,
            "bev_feat": feat,
        }

def apply_visibility(logits: Any, clear_mass: Any, through_mass: Any, occluded_mass: Any = None) -> Any:
    """G5 on per-cell transmittance masses (see ``BevLssHead.forward``).

    Seen = some valid ray reached the cell (``clear_mass > 0``); unseen cells
    have unknown forced high and free forced low. Free is allowed only where
    the cell is seen and some valid ray passed through it (``through_mass > 0``).
    Cells with ``occluded_mass > 0`` that no ray reached have occluded forced high.
    """
    seen = clear_mass > 0
    free_ok = seen & (through_mass > 0)
    behind = None
    if occluded_mass is not None:
        behind = (occluded_mass > 0) & ~seen
    force = torch.full_like(clear_mass, FORCE_LOGIT)
    free = torch.where(free_ok, logits[:, CH_FREE : CH_FREE + 1], -force)
    unknown = torch.where(seen, logits[:, CH_UNKNOWN : CH_UNKNOWN + 1], force)
    if behind is None:
        return torch.cat(
            [logits[:, :CH_FREE], free, logits[:, CH_FREE + 1 : CH_UNKNOWN], unknown, logits[:, CH_UNKNOWN + 1 :]],
            dim=1,
        )
    occ = torch.where(behind, force, logits[:, CH_OCCLUDED : CH_OCCLUDED + 1])
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
        reach_tau=float(section.get("reach_tau", REACH_TAU)),
        free_pass_tau=float(section.get("free_pass_tau", FREE_PASS_TAU)),
        ground_band_m=float(section.get("ground_band_m", GROUND_BAND_M)),
    )
