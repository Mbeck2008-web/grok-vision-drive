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
# A ray surface this far above the local ground is a viewshed occluder (G5).
# 0.3 m is below GROUND_BAND_M (so a 0.6 m wall's face hits close its
# umbra even though they never become "raised" for the band) and above the
# ground estimate's residual on a 6 % grade. Surfaces lower than this are
# the band's stated "may be passed over" limit.
RAISED_OCC_M = 0.3
# The ground cannot rise faster than this grade away from the ego's tyre
# contact patch (|x| <= EGO_CONTACT_HALF_X_M, |y| <= EGO_CONTACT_HALF_Y_M,
# at or below the ego origin, so z <= 0 there). The local ground estimate
# is clamped under that bound: a wall close ahead with no road visible in
# front of it cannot drag the ground up its own face. Lowering a ground
# estimate only ever shrinks seen / free (G5), so the clamp is conservative.
GROUND_SLOPE_CAP = 0.15
EGO_CONTACT_HALF_X_M = 0.8
EGO_CONTACT_HALF_Y_M = 1.4
# Trust range per camera (G5). A wall of DESIGN_WALL_M at range r has
# DESIGN_WALL_M - occ_m of face above the occluder cutoff; a camera whose
# pixel rows are FEAT_STRIDE / fy rad apart is sure to put a ray on that
# part of the face only while r * FEAT_STRIDE / fy <= DESIGN_WALL_M - occ_m.
# Past that range a wall can fall between two of its rows unseen, so its
# ray points count as seen only within SEEN_TIGHT_M of the ground (a ray
# that clears any occluder and is that close to the ground cannot be over
# ground hidden by more than that). Main: 11.8 m, narrow 19 m, wide 2.8 m.
DESIGN_WALL_M = 0.6
SEEN_TIGHT_M = 0.1
# Lowest ego ground taken under a mount (z0 is -0.23 m on the etk800 rig).
EGO_GROUND_DROP_M = 0.5
# Viewshed (G5): per camera, occluder tops are swept outward along
# VIEWSHED_AZ_BINS azimuths in VIEWSHED_RANGE_STEP_M steps out to
# VIEWSHED_RANGE_M (past the farthest grid corner from any mount). 0.5 deg
# is 0.4 m of arc at 46 m, under the one-cell (0.5 m) lateral dilation.
VIEWSHED_AZ_BINS = 720
VIEWSHED_RANGE_STEP_M = 0.25
VIEWSHED_RANGE_M = 48.0
# Occluder tops are dilated by one cell laterally (OCC_WINDOW_CELLS = 3), so
# a wall end or a hit that landed in the neighbouring cell still closes.
OCC_WINDOW_CELLS = 3
# Occluders closer than this to the destination along the sight line are
# ignored, so a cell is not shadowed by its own (dilated) top. 0.5 m is one
# cell: the first ground row behind a 0.3 m wall is 0.75 m past its face,
# and the face's top is spread over the whole depth bin it was seen in.
SIGHTLINE_MARGIN_M = 0.5
# Upper bound on a cell's occluder top from sums only (ONNX scatter-add):
# two log-sum-exp passes, coarse then sharp; the result is never below the
# true maximum and at most log(hits) / TOP_SHARP_K above it.
TOP_COARSE_K = 10.0
TOP_SHARP_K = 60.0

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

    The wide-scale ground is the same soft-min as ``ground_height``, over
    ``wide_window`` cells. A per-cell surface is kept only when it is less
    than ``near_m`` above that wide-scale height (and some wide-scale mass
    exists). The local soft-min then runs on the kept surfaces only, so a
    barrier wider than the local window cannot become the ground behind it.
    Surfaces at least ``near_m`` above the wide-scale ground are returned
    as a raised-surface mask. ``near_m`` is the height cutoff.
    """
    g_wide, has_wide = ground_height(z_sum, w_sum, window=wide_window, k=k)
    cell_has = w_sum > 0
    safe_w = torch.where(cell_has, w_sum, torch.ones_like(w_sum))
    cell_z = torch.where(
        cell_has,
        BEV_Z_MIN_M - torch.log((z_sum / safe_w).clamp(min=1e-30)) / k,
        torch.zeros_like(z_sum),
    )
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


def ray_crossed_raised(idx: Any, ok: Any, raised: Any) -> Any:
    """Per-sample mask: an earlier bin of the same ray sits in a raised cell.

    ``idx`` / ``ok`` are B x N x D x h x w (or D-1 at the bin edges). ``raised``
    is B x 1 x N x N. Cumulative along the depth dimension, so an overflight
    that later re-enters the ground band stays blocked.
    """
    b = raised.shape[0]
    ids = idx.reshape(b, -1).clamp(min=0, max=BEV_N * BEV_N - 1)
    at = raised.reshape(b, BEV_N * BEV_N).gather(1, ids).view_as(idx) * ok.to(raised.dtype)
    # Exclusive prefix: any earlier bin was raised. cumsum of a 0/1 mask, not
    # cummax, so the ONNX opset-17 export stays legal.
    prior = (at.cumsum(dim=2) - at) > 0
    return prior.to(raised.dtype)


def ray_top_bounds(pts: Any, pts_e: Any, surf_w: Any, ground: Any, occ_m: float) -> tuple[Any, Any, Any]:
    """Upper bounds on occluder tops from ray surfaces (G5). Returns (top, weight, flat cell index), B x P.

    ``pts`` B x N x D x h x w x 3 (bin centres), ``pts_e`` the same at the D - 1 inner bin edges,
    ``surf_w`` the weighted one-hot surface bin, ``ground`` B x 1 x N x N. A surface is an occluder when it
    is at least ``occ_m`` above the ground at its cell (anywhere in its bin). Its true point lies somewhere
    in its depth bin, and the object's top lies somewhere under the next pixel row up, which did not stop
    there (top-edge grazing). So the bound is the height of the next-up row's ray at the bin's near and
    far edges (the larger), and it is placed at the near edge, centre and far edge of the bin.
    """
    b, n, d, h, w = surf_w.shape
    # bin edges per bin: near = edge j-1 (centre for j = 0), far = edge j (centre for j = D-1)
    near = torch.cat([pts[:, :, :1], pts_e], dim=2)
    far = torch.cat([pts_e, pts[:, :, -1:]], dim=2)
    # next pixel row up (smaller v) at the same z-depth; row 0 mirrors row 1's step
    def up(p: Any) -> Any:
        if h < 2:
            return p
        above = torch.cat([2.0 * p[:, :, :, :1] - p[:, :, :, 1:2], p[:, :, :, :-1]], dim=3)
        return above
    top = torch.maximum(up(near)[..., 2], up(far)[..., 2])
    top = torch.maximum(top, torch.maximum(near[..., 2], far[..., 2]))
    g_flat = ground.reshape(b, BEV_N * BEV_N)
    tops, ws, ids = [], [], []
    raised = None
    for p in (near, pts, far):
        i, ok = bev_flat_index(p)
        g = g_flat.gather(1, i.reshape(b, -1)).view_as(top)
        r = (p[..., 2] - g >= occ_m).to(surf_w.dtype)
        raised = r if raised is None else torch.maximum(raised, r)
    for p in (near, pts, far):
        i, ok = bev_flat_index(p)
        ids.append(i.reshape(b, -1))
        ws.append((surf_w * raised * ok.to(surf_w.dtype)).reshape(b, -1))
        tops.append(top.reshape(b, -1))
    return torch.cat(tops, dim=1), torch.cat(ws, dim=1), torch.cat(ids, dim=1)


def occluder_tops(top: Any, idx: Any, w: Any) -> tuple[Any, Any]:
    """Per-cell upper bound on the occluder top (B x 1 x N x N) and a 0/1 "has occluder" mask.

    ``top`` / ``w`` / ``idx`` are B x P (absolute top bound, 0/1 weight, flat cell index). The bound is a
    two-pass log-sum-exp of the tops in the cell, so it is never below their maximum (and at most
    ``log(n) / TOP_SHARP_K`` above it), built from ``splat`` sums only. Then dilated by
    ``OCC_WINDOW_CELLS``. Cells without an occluder get -100 m.
    """
    b = top.shape[0]
    ok = w > 0
    t = top.clamp(min=BEV_Z_MIN_M, max=BEV_Z_MAX_M + 4.0)
    e1 = torch.exp(TOP_COARSE_K * (t - BEV_Z_MIN_M)) * w
    s1 = splat(torch.stack([e1, w], dim=-1), idx, ok)
    has = s1[:, 1:2] > 0
    t1 = torch.where(has, BEV_Z_MIN_M + torch.log(torch.where(has, s1[:, 0:1], torch.ones_like(s1[:, 0:1]))) / TOP_COARSE_K, torch.zeros_like(s1[:, 0:1]))
    t1_at = t1.reshape(b, BEV_N * BEV_N).gather(1, idx.clamp(min=0, max=BEV_N * BEV_N - 1))
    e2 = torch.exp(TOP_SHARP_K * (t - t1_at).clamp(max=0.0)) * w
    s2 = splat(e2.unsqueeze(-1), idx, ok)
    tops = torch.where(has, t1 + torch.log(torch.where(has, s2, torch.ones_like(s2))) / TOP_SHARP_K, torch.full_like(s2, -100.0))
    pad = OCC_WINDOW_CELLS // 2
    tops = F.max_pool2d(tops, OCC_WINDOW_CELLS, stride=1, padding=pad)
    has_d = F.max_pool2d(has.to(top.dtype), OCC_WINDOW_CELLS, stride=1, padding=pad)
    return tops, has_d


def _polar_grid(T: Any, dtype: Any, device: Any) -> tuple[Any, Any, Any]:
    """Per camera polar sample grid: (grid for ``grid_sample`` B*Ncam x A x R x 2, range centres R, camera
    xyz B*Ncam x 3). Azimuth bin a is centred on ``-pi + (a + 0.5) 2 pi / A`` from +y towards +x."""
    b, ncam = T.shape[:2]
    na = VIEWSHED_AZ_BINS
    step = float(VIEWSHED_RANGE_STEP_M)
    nr = int(math.ceil(VIEWSHED_RANGE_M / step))
    cam = T[..., :3, 3].to(dtype).reshape(b * ncam, 3)
    az = -math.pi + (torch.arange(na, device=device, dtype=dtype) + 0.5) * (2.0 * math.pi / na)
    rr = (torch.arange(nr, device=device, dtype=dtype) + 0.5) * step
    px = cam[:, 0].view(-1, 1, 1) + torch.sin(az).view(1, na, 1) * rr.view(1, 1, nr)
    py = cam[:, 1].view(-1, 1, 1) + torch.cos(az).view(1, na, 1) * rr.view(1, 1, nr)
    return torch.stack([px / BEV_HALF_M, -py / BEV_HALF_M], dim=-1), rr, cam


def _polar_index(dx: Any, dy: Any, r: Any) -> Any:
    """Flat (azimuth, range) index into an A x R polar map for offsets ``dx``, ``dy`` from the camera at
    horizontal range ``r`` (the range bin is clamped to the map)."""
    na = VIEWSHED_AZ_BINS
    nr = int(math.ceil(VIEWSHED_RANGE_M / float(VIEWSHED_RANGE_STEP_M)))
    a_idx = torch.floor((torch.atan2(dx, dy) + math.pi) * (na / (2.0 * math.pi))).clamp(min=0, max=na - 1)
    k = torch.floor(r / float(VIEWSHED_RANGE_STEP_M)).clamp(min=0, max=nr - 1)
    return (a_idx * nr + k).long()


def viewshed_blocked(tops: Any, dest_z: Any, T: Any, margin: float = SIGHTLINE_MARGIN_M) -> Any:
    """Per-camera mask (B x Ncam x N x N): the sight line from the camera to the cell centre at ``dest_z``
    passes at or under some occluder top more than ``margin`` before the cell.

    A horizon sweep: for each camera, ``tops`` (B x 1 x N x N, -100 where none) is sampled along
    ``VIEWSHED_AZ_BINS`` azimuths every ``VIEWSHED_RANGE_STEP_M`` m, turned into the slope
    ``(top - cam_z) / r`` and swept outward with a running maximum (log-step doubling, ONNX friendly).
    The cell is blocked if that running maximum, up to ``r_d - margin``, is at least the slope of its own
    sight line ``(dest_z - cam_z) / r_d``.
    """
    b, ncam = T.shape[:2]
    dtype, device = tops.dtype, tops.device
    na = VIEWSHED_AZ_BINS
    step = float(VIEWSHED_RANGE_STEP_M)
    nr = int(math.ceil(VIEWSHED_RANGE_M / step))
    grid, rr, cam = _polar_grid(T, dtype, device)
    cx, cy, cz = cam[:, 0].view(-1, 1, 1), cam[:, 1].view(-1, 1, 1), cam[:, 2].view(-1, 1, 1)
    src = (tops + 100.0).expand(b, ncam, BEV_N, BEV_N).reshape(b * ncam, 1, BEV_N, BEV_N)
    top_p = F.grid_sample(src, grid, mode="nearest", padding_mode="zeros", align_corners=False) - 100.0
    slope = (top_p.view(b * ncam, na, nr) - cz) / rr.view(1, 1, nr)
    run = slope
    shift = 1
    while shift < nr:
        prev = F.pad(run[..., :-shift], (shift, 0), value=-1.0e4)
        run = torch.maximum(run, prev)
        shift *= 2
    ys = BEV_HALF_M - BEV_CELL_M * (torch.arange(BEV_N, device=device, dtype=dtype) + 0.5)
    xs = -BEV_HALF_M + BEV_CELL_M * (torch.arange(BEV_N, device=device, dtype=dtype) + 0.5)
    dx = xs.view(1, 1, BEV_N) - cx
    dy = ys.view(1, BEV_N, 1) - cy
    r_d = torch.sqrt(dx * dx + dy * dy)
    a_idx = torch.floor((torch.atan2(dx, dy) + math.pi) * (na / (2.0 * math.pi))).clamp(min=0, max=na - 1)
    # last range bin whose centre is closer than r_d - margin
    k = torch.ceil((r_d - margin) / step - 0.5) - 1.0
    has_k = k >= 0
    flat = (a_idx * nr + k.clamp(min=0, max=nr - 1)).long().reshape(b * ncam, BEV_N * BEV_N)
    occ_slope = run.reshape(b * ncam, na * nr).gather(1, flat).view(b * ncam, BEV_N, BEV_N)
    dest_slope = (dest_z.to(dtype).expand(b, ncam, BEV_N, BEV_N).reshape(b * ncam, BEV_N, BEV_N) - cz) / r_d.clamp(min=1e-3)
    blocked = has_k & (occ_slope >= dest_slope)
    return blocked.view(b, ncam, BEV_N, BEV_N).to(dtype)


def ground_ceiling(dtype: Any = torch.float32, device: Any = None) -> Any:
    """Upper bound on the ground height per cell (1 x 1 x N x N): ``GROUND_SLOPE_CAP`` times the distance
    from the ego tyre contact patch (0 inside it)."""
    ys = BEV_HALF_M - BEV_CELL_M * (torch.arange(BEV_N, device=device, dtype=dtype) + 0.5)
    xs = -BEV_HALF_M + BEV_CELL_M * (torch.arange(BEV_N, device=device, dtype=dtype) + 0.5)
    dx = (xs.abs() - EGO_CONTACT_HALF_X_M).clamp(min=0.0).view(1, BEV_N)
    dy = (ys.abs() - EGO_CONTACT_HALF_Y_M).clamp(min=0.0).view(BEV_N, 1)
    return (GROUND_SLOPE_CAP * torch.sqrt(dx * dx + dy * dy)).view(1, 1, BEV_N, BEV_N)


def _row_gain(T: Any, elev: Any) -> Any:
    """How much farther apart than ``r * FEAT_STRIDE / fy`` two neighbouring pixel rows are, measured
    vertically at horizontal range r, for a ray at elevation ``elev`` (rad, broadcast against B x N x ...)
    of a camera pitched ``p``: ``cos^2(elev - p) / cos^2(elev)`` (pinhole rows are evenly spaced in
    tan(elev - p); off-centre columns are closer, so the centre column is the worst case)."""
    fwd_z = T[..., 2, 2].clamp(min=-1.0, max=1.0)
    pitch = torch.asin(fwd_z).view(*T.shape[:2], *([1] * (elev.dim() - 2)))
    return torch.cos(elev - pitch) ** 2 / torch.cos(elev).clamp(min=1e-3) ** 2


def _trust_m(K: Any, T: Any, dtype: Any) -> Any:
    """B x N trust range per camera: the range out to which neighbouring pixel rows are less than
    ``DESIGN_WALL_M - RAISED_OCC_M`` apart on a wall face (``(DESIGN_WALL_M - RAISED_OCC_M) * fy /
    FEAT_STRIDE`` shrunk by ``_row_gain`` at the elevation of the face bottom, the mount taken
    ``EGO_GROUND_DROP_M`` above the ground for the steepest view)."""
    base = (DESIGN_WALL_M - RAISED_OCC_M) * K[..., 1, 1].to(dtype) / FEAT_STRIDE
    drop = T[..., 2, 3].to(dtype) + EGO_GROUND_DROP_M - RAISED_OCC_M
    r = base
    for _ in range(4):  # shrinking r steepens the view and raises the gain: iterate to the fixed point
        r = base / _row_gain(T.to(dtype), -torch.atan(drop.clamp(min=0.0) / r))
    return r


def detection_cover(T: Any, K: Any, ground: Any, blocked: Any, occ_m: float, cam_w: Any, pixel_valid: Any = None) -> tuple[Any, Any]:
    """(cover, cap), each B x 1 x N x N. ``cap``: a wall on the cell that no valid camera puts a pixel
    row on above the occluder cutoff has its top under ``cap`` (the finest row step over the cameras
    that see the cell). ``cover`` is 1 where some valid camera is sure to put a pixel row on the face of a design wall
    standing anywhere on the cell: its ground is not behind an occluder for that camera (``blocked``),
    the face span (``occ_m`` to ``occ_m + DESIGN_WALL_M - RAISED_OCC_M`` above ``ground``) projects
    inside its image (and ``pixel_valid``) from all four cell corners, and either the cell is within the
    camera's trust range or a pixel row centre lies between the lowest projected top and the highest
    projected bottom of the span over the four corners (a horizontal plane maps to the image by a
    homography, so the corners bound the cell). Not eroded: ``uncovered_prefix`` samples the 2 x 2
    cells around every polar sample."""
    b, n = T.shape[:2]
    dtype, device = ground.dtype, ground.device
    ys = BEV_HALF_M - BEV_CELL_M * (torch.arange(BEV_N, device=device, dtype=dtype) + 0.5)
    xs = -BEV_HALF_M + BEV_CELL_M * (torch.arange(BEV_N, device=device, dtype=dtype) + 0.5)
    R = T[..., :3, :3].to(dtype)
    t = T[..., :3, 3].to(dtype)
    gx = xs.view(1, 1, 1, BEV_N).expand(b, n, BEV_N, BEV_N)
    gy = ys.view(1, 1, BEV_N, 1).expand(b, n, BEV_N, BEV_N)
    dxy = torch.sqrt((gx - t[..., 0, None, None]) ** 2 + (gy - t[..., 1, None, None]) ** 2)
    ok = blocked < 0.5
    h_img, w_img = (pixel_valid.shape[-2], pixel_valid.shape[-1]) if pixel_valid is not None else (CROP_H // FEAT_STRIDE, CROP_W // FEAT_STRIDE)
    if pixel_valid is not None:
        pv = pixel_valid.to(dtype).reshape(-1, n, h_img * w_img)
        pv = pv.expand(b, n, h_img * w_img) if pv.shape[0] == 1 else pv
    half = 0.5 * BEV_CELL_M
    v_lo = None  # lowest (largest v) projected top over the corners
    v_hi = None  # highest (smallest v) projected bottom over the corners
    for hz, is_top in ((occ_m, False), (occ_m + (DESIGN_WALL_M - RAISED_OCC_M), True)):
        gz = (ground + hz).expand(b, n, BEV_N, BEV_N)
        for ox, oy in ((-half, -half), (-half, half), (half, -half), (half, half)):
            d = torch.stack([gx + ox - t[..., 0, None, None], gy + oy - t[..., 1, None, None], gz - t[..., 2, None, None]], dim=-1)
            pc = torch.einsum("bnji,bnyxj->bnyxi", R, d)  # R^T (p - t): camera frame
            zc = pc[..., 2]
            zs = torch.where(zc > 0.1, zc, torch.ones_like(zc))
            u = K[..., 0, 0, None, None].to(dtype) * pc[..., 0] / zs + K[..., 0, 2, None, None].to(dtype)
            v = K[..., 1, 1, None, None].to(dtype) * pc[..., 1] / zs + K[..., 1, 2, None, None].to(dtype)
            inside = (zc > 0.1) & (u >= 0) & (u < CROP_W) & (v >= 0) & (v < CROP_H)
            if pixel_valid is not None:
                col = torch.floor(u / FEAT_STRIDE).clamp(min=0, max=w_img - 1)
                row = torch.floor(v / FEAT_STRIDE).clamp(min=0, max=h_img - 1)
                flat = (row * w_img + col).long().reshape(b, n, -1)
                inside = inside & (pv.gather(2, flat).view_as(inside) > 0)
            ok = ok & inside
            if is_top:
                v_lo = v if v_lo is None else torch.maximum(v_lo, v)
            else:
                v_hi = v if v_hi is None else torch.minimum(v_hi, v)
    # a row centre (i + 0.5) * FEAT_STRIDE in [v_lo, v_hi]
    row_hit = torch.floor(v_hi / FEAT_STRIDE - 0.5) >= torch.ceil(v_lo / FEAT_STRIDE - 0.5)
    sure = (dxy <= _trust_m(K, T, dtype)[..., None, None]) | row_hit
    # otherwise its rows are one row step apart on the face (pitch gain at the face bottom), so a wall
    # it misses is under occ_m + that step
    elev = -torch.atan((t[..., 2, None, None] - ground - occ_m).clamp(min=0.0) / dxy.clamp(min=1e-3))
    step = dxy * FEAT_STRIDE / K[..., 1, 1, None, None].to(dtype) * _row_gain(T.to(dtype), elev)
    face = DESIGN_WALL_M - RAISED_OCC_M
    gap = torch.where(sure, torch.full_like(step, face), torch.maximum(step, torch.full_like(step, face)))
    use = ok & (cam_w.to(dtype).view(b, n, 1, 1) > 0)
    gap = torch.where(use, gap, torch.full_like(gap, 1.0e3)).amin(dim=1, keepdim=True)
    cover = (gap <= face).to(dtype)
    return cover, ground + occ_m + gap


def uncovered_prefix(cover: Any, T: Any, cap: Any) -> Any:
    """Per camera B*Ncam x (A*R): number of uncovered polar samples (``cover`` < 0.5) along each azimuth
    up to and including each range bin."""
    b, ncam = T.shape[:2]
    grid, _, _ = _polar_grid(T, cover.dtype, cover.device)
    src = cover.expand(b, ncam, BEV_N, BEV_N).reshape(b * ncam, 1, BEV_N, BEV_N)
    # The sight line to any point lies within half an azimuth bin (<= 0.21 m at 48 m) of its bin's
    # centre line, so each of its points is within a quarter cell (Chebyshev) of a sample (0.125 m
    # along, plus the lateral offset). The four cells a quarter cell off a sample in x and y are the
    # 2 x 2 block holding every such point: a sample counts as covered only if all four are.
    q = 0.5 * BEV_CELL_M / BEV_HALF_M
    cap_src = (cap - 100.0).expand(b, ncam, BEV_N, BEV_N).reshape(b * ncam, 1, BEV_N, BEV_N)
    cov_p = None
    cap_p = None
    for ox, oy in ((-q, -q), (-q, q), (q, -q), (q, q)):
        off = torch.tensor([ox, oy], dtype=cover.dtype, device=cover.device)
        c = F.grid_sample(src, grid + off, mode="nearest", padding_mode="zeros", align_corners=False)
        cov_p = c if cov_p is None else torch.minimum(cov_p, c)
        # off the grid: zeros -> +100 m, no bound
        h = F.grid_sample(cap_src, grid + off, mode="nearest", padding_mode="zeros", align_corners=False) + 100.0
        cap_p = h if cap_p is None else torch.maximum(cap_p, h)
    unc = (cov_p < 0.5).to(cover.dtype).squeeze(1)  # B*Ncam x A x R
    cap_run = cap_p.squeeze(1)
    shift = 1
    while shift < cap_run.shape[-1]:
        cap_run = torch.maximum(cap_run, F.pad(cap_run[..., :-shift], (shift, 0), value=-1.0e4))
        shift *= 2
    nr = unc.shape[-1]
    # far edge (m) of the last uncovered sample up to each range bin (0 if none), running max by doubling
    far = unc * (torch.arange(nr, device=cover.device, dtype=cover.dtype) + 1.0).view(1, 1, nr) * float(VIEWSHED_RANGE_STEP_M)
    shift = 1
    while shift < nr:
        far = torch.maximum(far, F.pad(far[..., :-shift], (shift, 0), value=0.0))
        shift *= 2
    return unc.cumsum(dim=-1).reshape(b * ncam, -1), unc.reshape(b * ncam, -1), far.reshape(b * ncam, -1), cap_run.reshape(b * ncam, -1)


def trusted_band(pts: Any, T: Any, K: Any, z_rel: Any, band: float, occ_m: float, ground: Any, unc: tuple[Any, Any]) -> Any:
    """0/1 per ray point: may count as seen (G5). ``pts`` B x N x D' x h x w x 3, ``z_rel`` its height
    above the local ground, ``ground`` B x 1 x N x N, ``unc`` from ``uncovered_prefix``.

    A design wall (``wall = DESIGN_WALL_M + (occ_m - RAISED_OCC_M)`` tall) has ``DESIGN_WALL_M -
    RAISED_OCC_M`` of face above the occluder cutoff; out to its trust range a camera's pixel rows are
    closer than that, so one of them lands on it and the viewshed has it. Past the trust range a wall can
    fall between two rows unseen. A ray can only clear a design wall where it is more than ``wall`` above
    the ground, so its points get the full ``band`` if it is under ``wall`` where it leaves the trust
    range, or if every cell it crosses from there until it drops under ``wall`` is covered by some
    camera (``detection_cover``). Otherwise a point counts only within SEEN_TIGHT_M of the ground: a ray
    that clears every occluder and is that close to the ground is not over ground hidden by more.
    """
    b, n = T.shape[:2]
    cam = T[..., :3, 3].to(pts.dtype).view(b, n, 1, 1, 1, 3)
    vec = pts - cam
    r_h = torch.sqrt(vec[..., 0] * vec[..., 0] + vec[..., 1] * vec[..., 1])
    trust = _trust_m(K, T, pts.dtype).view(b, n, 1, 1, 1)
    frac = (trust / r_h.clamp(min=1e-6)).clamp(max=1.0)
    q = cam + vec * frac.unsqueeze(-1)
    idx_q, ok_q = bev_flat_index(q)
    g_q = ground.reshape(b, BEV_N * BEV_N).gather(1, idx_q.reshape(b, -1)).view_as(r_h)
    h_q = q[..., 2] - g_q
    wall = DESIGN_WALL_M + (occ_m - RAISED_OCC_M)
    own = (r_h <= trust) | (ok_q & (h_q <= wall))
    # where the ray drops under ``wall`` (heights linear between q and the point)
    r_w = trust + (h_q - wall) / (h_q - z_rel).clamp(min=1e-6) * (r_h - trust)
    r_end = torch.minimum(r_w, r_h)
    cum, unc_bin, far, cap_run = unc
    i_q = _polar_index(vec[..., 0], vec[..., 1], trust.expand_as(r_h)).reshape(b * n, -1)
    i_w = _polar_index(vec[..., 0], vec[..., 1], r_end).reshape(b * n, -1)
    count = cum.gather(1, i_w) - cum.gather(1, i_q) + unc_bin.gather(1, i_q)
    covered = ok_q & (count.view_as(r_h) < 0.5)
    # an undetected design wall can only stand on an uncovered cell where the ray is above ``wall``
    # (range <= r_end, and no farther than the far edge of the last uncovered sample before it, plus
    # one cell for the sampling); the ray passes z_rel * r / r_h above the sight line to this point's
    # ground at range r, so the ground under the point is hidden by less than z_rel * r_o / r_h
    r_unc = far.gather(1, i_w).view_as(r_h) + BEV_CELL_M
    r_o = torch.minimum(r_end, r_unc).clamp(min=0.0)
    shadow = z_rel * r_o / r_h.clamp(min=1e-6)
    # The wall is also missed by this camera's own rows: its top is under occ_m + one row step above
    # its ground, else a row lands on its face above the cutoff (both bounds grow with range, so the
    # worst case is the farthest range r_o). Hidden depth is that top minus the sight line there.
    o = cam + vec * (r_o / r_h.clamp(min=1e-6)).unsqueeze(-1)
    idx_o, ok_o = bev_flat_index(o)
    g_o = ground.reshape(b, BEV_N * BEV_N).gather(1, idx_o.reshape(b, -1)).view_as(r_h)
    cz = cam[..., 2]
    elev = -torch.atan((cz - g_o - occ_m).clamp(min=0.0) / r_o.clamp(min=1e-3))
    step = r_o * FEAT_STRIDE / K[..., 1, 1].to(pts.dtype).view(b, n, 1, 1, 1) * _row_gain(T.to(pts.dtype), elev)
    sight_o = cz + (pts[..., 2] - z_rel - cz) * r_o / r_h.clamp(min=1e-6)
    comb = g_o + occ_m + step - sight_o
    shadow = torch.where(ok_o, torch.minimum(shadow, comb), shadow)
    # and by every other camera: under the largest ``cap`` sampled up to r_o (tops only rise with range
    # for the bound, so the farthest range is the worst case for the sight line too)
    i_o = _polar_index(vec[..., 0], vec[..., 1], r_o).reshape(b * n, -1)
    shadow = torch.minimum(shadow, cap_run.gather(1, i_o).view_as(r_h) - sight_o)
    # hidden is judged on the sight line to the cell centre, up to half a cell diagonal nearer than the
    # point's foot: that line is lower at r_o by at most (cz - ground) * r_o * d / (r_h (r_h - d))
    d_c = 0.5 * math.sqrt(2.0) * BEV_CELL_M
    shadow = shadow + (cz - (pts[..., 2] - z_rel)).clamp(min=0.0) * r_o * d_c / (r_h * (r_h - d_c)).clamp(min=1e-3)
    tight = min(float(band), SEEN_TIGHT_M)
    return torch.where(own | covered, z_rel < band, (z_rel < band) & (shadow < tight)).to(z_rel.dtype)


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
        pts_e = frustum_ego_points(K, T, self.edges.to(K.dtype), (h, w), FEAT_STRIDE, time_offset_s, ego_speed_mps)
        z = pts[..., 2].to(prob.dtype)
        surf, has_surf = ray_surface(reached, dim=2)  # B N D h w one-hot, B N 1 h w
        ray_w = has_surf * weight
        g_vals = torch.stack([surf * ray_w * torch.exp(-GROUND_SOFTMIN_K * (z - BEV_Z_MIN_M)), surf * ray_w], dim=-1)
        g_sum = splat(g_vals.reshape(b, n * d * h * w, 2), idx.reshape(b, -1), ok.reshape(b, -1))
        ground, has_ground, raised, g_wide = two_scale_ground(g_sum[:, 0:1], g_sum[:, 1:2])
        ground = torch.minimum(ground, ground_ceiling(ground.dtype, ground.device))
        flat_g = (ground * has_ground).reshape(b, BEV_N * BEV_N)
        flat_has = has_ground.reshape(b, BEV_N * BEV_N)
        crossed = 1.0 - ray_crossed_raised(idx, ok, raised)
        # Viewshed occluders (G5). Threshold tracks the configured band: default 0.5 m -> 0.3 m, so a 0.6 m
        # wall's face hits are occluders; a 2.0 m band raises it so a 1.5 m car roof is not one.
        occ_m = self.ground_band_m - (GROUND_BAND_M - RAISED_OCC_M)
        g_low = torch.where(has_ground > 0, torch.minimum(ground, g_wide), g_wide)
        top_bound, hit_w, hit_idx = ray_top_bounds(pts, pts_e, surf * ray_w, ground * has_ground, occ_m)
        tops, _ = occluder_tops(top_bound, hit_idx, hit_w)
        blocked = viewshed_blocked(tops, g_low, T)
        ids_c = idx.reshape(b, n, -1).clamp(min=0, max=BEV_N * BEV_N - 1)
        visible = (1.0 - blocked.reshape(b, n, BEV_N * BEV_N).gather(2, ids_c).view_as(idx)) * ok.to(prob.dtype)

        def _rel(zs: Any, ids: Any) -> tuple[Any, Any]:
            g = flat_g.gather(1, ids.reshape(b, -1)).view_as(zs)
            ok_g = flat_has.gather(1, ids.reshape(b, -1)).view_as(zs)
            return zs - g, ok_g * ray_w

        cam_w = cam_valid.to(prob.dtype) if cam_valid is not None else torch.ones(b, n, dtype=prob.dtype, device=prob.device)
        cover, cap = detection_cover(T, K, ground * has_ground, blocked, occ_m, cam_w, pixel_valid)
        unc = uncovered_prefix(cover, T, cap)
        z_rel, ok_rel = _rel(z, idx)
        band_w = near_ground(z_rel, 0.0, self.ground_band_m).to(prob.dtype) * ok_rel  # B N D h w
        seen_w = trusted_band(pts, T, K, z_rel, self.ground_band_m, occ_m, ground * has_ground, unc) * ok_rel
        clear_w = reach * reached * seen_w * crossed * visible
        through_w = beyond * through * band_w * crossed * visible
        occ_w = (1.0 - reach) * (1.0 - reached) * band_w
        # Visibility is also sampled at the edge between bins j and j + 1. In the surface band only
        # near-surface points count, and the bin centres alone leave whole cell rows unsampled at range
        # (bins ~1.4 m apart at 15 m, cells 0.5 m). An edge counts only the mass beyond bin j + 1
        # (S_{j+1}) for both reach and pass, so a surface anywhere in bin j + 1, or snapped into it
        # from just before the edge, never opens the edge point.
        idx_e, ok_e = bev_flat_index(pts_e)
        z_rel_e, ok_rel_e = _rel(pts_e[..., 2].to(prob.dtype), idx_e)
        band_e = near_ground(z_rel_e, 0.0, self.ground_band_m).to(prob.dtype) * ok_rel_e
        seen_e = trusted_band(pts_e, T, K, z_rel_e, self.ground_band_m, occ_m, ground * has_ground, unc) * ok_rel_e
        crossed_e = 1.0 - ray_crossed_raised(idx_e, ok_e, raised)
        ids_e = idx_e.reshape(b, n, -1).clamp(min=0, max=BEV_N * BEV_N - 1)
        visible_e = (1.0 - blocked.reshape(b, n, BEV_N * BEV_N).gather(2, ids_e).view_as(idx_e)) * ok_e.to(prob.dtype)
        s_e = beyond[:, :, 1:]
        reached_e, through_e = visibility_gates(s_e, s_e, self.reach_tau, self.free_pass_tau)
        vis_e = torch.stack(
            [
                (1.0 - s_e) * (1.0 - reached_e) * band_e,
                s_e * reached_e * seen_e * crossed_e * visible_e,
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
