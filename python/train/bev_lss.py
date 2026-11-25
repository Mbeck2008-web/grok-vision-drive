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
    A hit that climbs from the hit of the row below it in the same column
    faster than ``GROUND_SLOPE_CAP`` (least steep reading of the two depth
    bins) is a face, an occluder too. Virtual rays half way between
    neighbouring pixel columns add seen / free only in bins both real rays
    passed through. A point "within SEEN_TIGHT_M of the ground" keeps its
    cell seen behind a bounded top only if it is also that close to its own
    ray's landing.
    Named limitation (observable vs unobservable): a barrier that no valid
    feature ray of any camera touches (it falls between two rows, e.g. a
    0.3 m wall 15 m ahead on main alone) is invisible to the rig, and the
    ground behind it can read as seen. That count is pinned by
    ``check_unobservable_leak_pinned``; every barrier some ray touches is
    held to zero leaks by the probe battery.
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
# A one-row face hit more than this above both neighbouring rows' hits in its column is on an object (bump).
BUMP_M = 0.1
EGO_CONTACT_HALF_X_M = 0.8
EGO_CONTACT_HALF_Y_M = 1.4
# A ray point within SEEN_TIGHT_M of the ground, close to its own landing, cannot be over ground hidden
# by more than that whatever stands before it (``tight_clear``).
SEEN_TIGHT_M = 0.1
# Viewshed (G5): per camera, occluder tops are swept outward along
# VIEWSHED_AZ_BINS azimuths in VIEWSHED_RANGE_STEP_M steps out to
# VIEWSHED_RANGE_M (past the farthest grid corner from any mount). 0.5 deg
# is 0.4 m of arc at 46 m; each sample takes the highest top of the 2 x 2
# cells a quarter cell around it, which covers that arc and the step.
# Occluder tops are not dilated by a fixed window: each hit is spread over
# its own pixel column's azimuth span (half way to the neighbouring columns)
# and its depth bin (near edge to far edge), so the shadow of a thin pole or
# sign is the span its rays actually cover (over-hide check).
VIEWSHED_AZ_BINS = 360
# Places per hit along its depth bin (near edge to far edge) and across its
# pixel column: 13 x 13 keeps neighbouring places under 0.37 m apart along a
# 48 m bin (~4.4 m) and under 0.3 m across the wide camera's ~3.5 m column
# at 32 m (the viewshed's 3 x 3 sampling bridges a one-cell gap).
TOP_RADIAL_SAMPLES = 7
TOP_LATERAL_SAMPLES = 7
# Occluder tops live on their own finer grid (OCC_N x OCC_N over the same
# 64 m square, 0.25 m cells): a 0.2 m pole fills 1-2 cells, not a 0.5 m BEV
# cell (or two when it straddles a cell edge).
OCC_N = 2 * BEV_N
VIEWSHED_RANGE_STEP_M = 0.5
VIEWSHED_RANGE_M = 48.0
# Occluders closer than this to the destination along the sight line are
# ignored, so a cell is not shadowed by its own top. 0.5 m is one
# cell: the first ground row behind a 0.3 m wall is 0.75 m past its face,
# and the face's top is spread over the whole depth bin it was seen in.
SIGHTLINE_MARGIN_M = 0.5
# Upper bound on a cell's occluder top from sums only (ONNX scatter-add):
# a log-sum-exp; the result is never below the true maximum and at most log(hits) / k above it, in passes of rising k
# (each shifted by the last, so exp stays in float32 range: k * slack < 87).
TOP_LSE_K = (5.0, 60.0)

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


def ray_top_bounds(pts: Any, depths: Any, edges: Any, surf_w: Any, ground: Any, occ_m: float, reached: Any = None, cam_xyz: Any = None) -> tuple[Any, Any, Any, Any]:
    """Bounds on occluder extents from ray surfaces (G5). Returns (top, bottom, weight, flat cell index), B x P.

    ``pts`` B x N x D x h x w x 3 (bin centres at ``depths``; ``edges`` the D - 1 inner bin edges),
    ``surf_w`` the weighted one-hot surface bin, ``ground`` B x 1 x N x N. A surface is an occluder when
    some point of its ray in its bin (near edge, centre, far edge) is at least ``occ_m`` above the ground
    at its cell. Its true point lies somewhere in the bin, and the object's top lies under the next pixel
    row up, which did not stop there (top-edge grazing): at depth t in the bin the top is under that row's
    ray at t. So the bound at each place is the next-up row's height there (the hit's own height if
    higher; a hit in the top feature row has no row above it and gets no bound, ``BEV_Z_MAX_M + 4``).
    Seen from the hitting camera, every such bound has the next-up ray's own elevation angle, so its
    shadow ends where that ray lands. Each hit is placed along its bin (``TOP_RADIAL_SAMPLES``, near to
    far edge) and across its own pixel column (``TOP_LATERAL_SAMPLES``, half way to each neighbouring
    column's ray at the same depth), not over a fixed dilated window: adjacent hit columns tile a wall
    and a thin pole or sign shadows only its own column's azimuth span (over-hide check).

    ``bottom`` is a lower bound on where the occluder starts: if the next pixel row down (``reached``,
    B x N x D x h x w) got past the hit's bin, the object is above that row's ray there (a sign panel or
    an overpass deck with open road under it); otherwise -100 (a solid column down to the ground).
    """
    b, n, d, h, w = surf_w.shape
    dt = pts.dtype
    # Every point of a ray is affine in its depth (``frustum_ego_points``): p(t) = o + t * v, so the hit's
    # bin ends and its neighbouring rays' points at the same depths are per-ray arithmetic, not
    # B x N x D x h x w x 3 products (activation budget).
    dep = depths.to(dt)
    v = (pts[:, :, 1] - pts[:, :, 0]) / (dep[1] - dep[0])
    o = pts[:, :, 0] - v * dep[0]
    one = (surf_w > 0).to(dt)
    wr = surf_w.sum(dim=2)  # B N h w (one surface bin per ray)
    jd = torch.arange(d, device=pts.device, dtype=dt).view(1, 1, d, 1, 1)
    j_hit = (one * jd).sum(dim=2)
    has_hit = one.sum(dim=2) > 0.5
    jl = j_hit.long()
    e_all = torch.cat([dep[:1], edges.to(dt), dep[-1:]])  # near edge of bin j is e_all[j], far edge e_all[j + 1]
    t_n, t_f, t_c = e_all[jl], e_all[jl + 1], dep[jl]

    def at(oo: Any, vv: Any, t: Any) -> Any:
        return oo + vv * t.unsqueeze(-1)

    def up(t: Any) -> Any:  # row r gets row r - 1 (row 0 keeps its own)
        return torch.cat([t[:, :, :1], t[:, :, :-1]], dim=2) if h >= 2 else t

    def down(t: Any) -> Any:  # row r gets row r + 1 (last row keeps its own)
        return torch.cat([t[:, :, 1:], t[:, :, -1:]], dim=2) if h >= 2 else t

    def left(t: Any) -> Any:  # column c gets column c - 1 (column 0 mirrors column 1)
        return torch.cat([2.0 * t[:, :, :, :1] - t[:, :, :, 1:2], t[:, :, :, :-1]], dim=3)

    def right(t: Any) -> Any:  # column c gets column c + 1 (last column mirrors)
        return torch.cat([t[:, :, :, 1:], 2.0 * t[:, :, :, -1:] - t[:, :, :, -2:-1]], dim=3)

    n_h, f_h, c_h = at(o, v, t_n), at(o, v, t_f), at(o, v, t_c)
    o_u, v_u = up(o), up(v)
    n_u, f_u = at(o_u, v_u, t_n), at(o_u, v_u, t_f)
    if w >= 2:
        o_l, v_l, o_r, v_r = left(o), left(v), right(o), right(v)
        l_n, r_n = at(o_l, v_l, t_n) - n_h, at(o_r, v_r, t_n) - n_h
        l_f, r_f = at(o_l, v_l, t_f) - f_h, at(o_r, v_r, t_f) - f_h
    else:
        l_n = r_n = l_f = r_f = torch.zeros_like(n_h)
    # top feature row: no row above it (a comparison, not a slice assignment: ONNX would export that as a
    # ScatterND without reduction)
    no_up = (torch.arange(h, device=pts.device) == 0).view(1, 1, h, 1).expand(b, n, h, w)
    # Bottom bound, per column from the bottom row up: a hit whose next row down hit the same object
    # (surface within one bin) takes that row's bound; else, if the row down got past the hit's bin
    # (its surface is farther, or it is sky), the object starts above that row's ray in the bin; else
    # (the row down stopped nearer, or there is none) it may reach the ground: -100.
    if reached is not None:
        sky = reached[:, :, -1] > 0.5
    else:
        sky = torch.zeros_like(has_hit)
    pass_bin = torch.where(has_hit, j_hit, torch.where(sky, torch.full_like(j_hit, float(d)), torch.full_like(j_hit, -1.0)))
    o_d, v_d = down(o), down(v)
    d_low = torch.minimum(at(o_d, v_d, t_n)[..., 2], at(o_d, v_d, t_f)[..., 2])
    bot_rows: list[Any] = [None] * h
    pb_rows: list[Any] = [None] * h
    neg = torch.full_like(j_hit[:, :, 0], -100.0)
    for r in range(h - 1, -1, -1):
        if r == h - 1 or reached is None:
            bot_rows[r], pb_rows[r] = neg, torch.full_like(neg, -1.0)
            continue
        same = has_hit[:, :, r + 1] & ((j_hit[:, :, r + 1] - j_hit[:, :, r]).abs() <= 1.0)
        bot_rows[r] = torch.where(same, bot_rows[r + 1], d_low[:, :, r])
        pb_rows[r] = torch.where(same, pb_rows[r + 1], pass_bin[:, :, r + 1])
    bot_ray = torch.stack(bot_rows, dim=2)
    pb_ray = torch.stack(pb_rows, dim=2)
    bot_ray = torch.where(has_hit & (pb_ray > j_hit), torch.minimum(bot_ray, c_h[..., 2]), torch.full_like(bot_ray, -100.0))
    g_flat = ground.reshape(b, BEV_N * BEV_N)
    raised = None
    lift = None  # highest reading of the hit above its local ground
    for p in (n_h, c_h, f_h):
        i, ok = _flat_index(p)
        g = g_flat.gather(1, i.reshape(b, -1)).view_as(wr)
        r = (p[..., 2] - g >= occ_m).to(surf_w.dtype)
        raised = r if raised is None else torch.maximum(raised, r)
        lift = p[..., 2] - g if lift is None else torch.maximum(lift, p[..., 2] - g)
    # Riser and bump occluders track the configured band like ``occ_m`` does: at the default band
    # (occ_m = RAISED_OCC_M) any height counts; a 2.0 m band (occ_m 1.8 m) makes a 1.5 m car a ground-like
    # object again, so its faces are not occluders either.
    tall = lift >= (occ_m - RAISED_OCC_M)
    # Riser (G5): a face too low to clear ``occ_m`` above the local ground (a 0.45 m wall at 12 m whose
    # only hits are under 0.3 m) still stands out from the row below it in the same column. The ground
    # rises at most GROUND_SLOPE_CAP per metre of range, so if even the least steep reading of the two
    # hits (this hit at its bin's far edge, the row below's at its near edge, where each could be in its
    # bin) climbs faster than that, this hit is on a face, not on the road: it is an occluder. Both
    # points are ego-frame, measured per column of one camera; nothing is tuned to a wall.
    if h >= 2 and cam_xyz is not None:
        cam = cam_xyz.to(pts.dtype).view(b, n, 1, 1, 3)
        def _shift_down(t: Any) -> Any:  # row r gets row r + 1 (B x N x h x w [x 3])
            return torch.cat([t[:, :, 1:], t[:, :, -1:]], dim=2)
        a_n = _shift_down(n_h)
        not_last = (torch.arange(h, device=pts.device) < h - 1).view(1, 1, h, 1)
        has_a = (_shift_down(has_hit.to(pts.dtype)) > 0.5) & not_last
        rng = lambda q: torch.sqrt((q[..., 0] - cam[..., 0]) ** 2 + (q[..., 1] - cam[..., 1]) ** 2)  # noqa: E731
        d_r = (rng(f_h) - rng(a_n)).clamp(min=0.0)
        riser = has_hit & has_a & ((f_h[..., 2] - a_n[..., 2]) > GROUND_SLOPE_CAP * d_r)
        # Bump (G5): a face so low and far that only one row lands on it (a 0.3 m wall at 10-15 m: one row
        # hits it 0.2 m up, the row below lands on the road short of it, the row above clears it and lands
        # on the road behind). Its hit stands above both neighbouring rows' hits in the column; a road,
        # whatever its grade, rises or falls monotonically between them and a crest bulges by
        # centimetres over a row gap. So a hit more than BUMP_M above both neighbours is on an object.
        def _shift_up(t: Any) -> Any:  # row r gets row r - 1 (the next row up)
            return torch.cat([t[:, :, :1], t[:, :, :-1]], dim=2)
        not_first = (torch.arange(h, device=pts.device) > 0).view(1, 1, h, 1)
        has_u = (_shift_up(has_hit.to(pts.dtype)) > 0.5) & not_first
        bump = has_hit & has_a & has_u & (c_h[..., 2] - torch.maximum(_shift_down(c_h)[..., 2], _shift_up(c_h)[..., 2]) > BUMP_M)
        raised = torch.maximum(raised, ((riser | bump) & tall).to(raised.dtype))
    weight = wr * raised
    m_r, m_l = TOP_RADIAL_SAMPLES, TOP_LATERAL_SAMPLES
    tr = torch.linspace(0.0, 1.0, m_r, device=pts.device, dtype=dt) if m_r > 1 else torch.full((1,), 0.5, device=pts.device, dtype=dt)
    ul = torch.linspace(-0.5, 0.5, m_l, device=pts.device, dtype=dt) if m_l > 1 else torch.zeros(1, device=pts.device, dtype=dt)
    trv = tr.view(m_r, 1)
    base = n_h.unsqueeze(-2) + (f_h - n_h).unsqueeze(-2) * trv  # B N h w R 3
    bound = torch.maximum(n_u[..., 2:3] + (f_u[..., 2:3] - n_u[..., 2:3]) * tr, base[..., 2])  # B N h w R
    bound = torch.where(no_up.unsqueeze(-1), torch.full_like(bound, BEV_Z_MAX_M + 4.0), bound)
    low = torch.minimum(bot_ray.unsqueeze(-1), base[..., 2])
    lft = l_n.unsqueeze(-2) + (l_f - l_n).unsqueeze(-2) * trv
    rgt = r_n.unsqueeze(-2) + (r_f - r_n).unsqueeze(-2) * trv
    # across the column: half way to the left neighbour (u < 0) ... half way to the right one (u > 0)
    q = (base.unsqueeze(-2) + lft.unsqueeze(-2) * (-ul).clamp(min=0.0).view(m_l, 1)
         + rgt.unsqueeze(-2) * ul.clamp(min=0.0).view(m_l, 1))  # B N h w R L 3
    i, ok = _flat_index(q, OCC_N)
    shape = (b, n, h, w, m_r, m_l)
    ws = (weight.view(b, n, h, w, 1, 1) * ok.to(surf_w.dtype)).reshape(b, -1)
    return (bound.unsqueeze(-1).expand(shape).reshape(b, -1), low.unsqueeze(-1).expand(shape).reshape(b, -1),
            ws, i.reshape(b, -1))


def occluder_tops(top: Any, idx: Any, w: Any) -> tuple[Any, Any]:
    """Per-cell upper bounds on occluder values (B x C x OCC_N x OCC_N, the 0.25 m occluder grid) and a
    0/1 "has occluder" mask (B x 1 x OCC_N x OCC_N).

    ``top`` B x P x C (C bounds per hit, e.g. top and -bottom), ``w`` B x P 0/1 weight, ``idx`` B x P flat
    cell index. Each bound is a log-sum-exp ``t* = log(sum exp(k t)) / k`` of the values in the cell,
    never below their maximum and at most ``log(n) / k`` above it, built from ``splat`` sums only (ONNX
    scatter-add). Passes of rising sharpness ``TOP_LSE_K``, each shifted by the previous pass's result so
    no exponent leaves float32 range (k_next * log(n) / k_prev < 87 for n <= 1000 places per cell; past
    that a cell keeps the looser first-pass bound); the last pass is at most log(n) / 60 (11.5 cm for
    1000 places, 6.5 cm for 50) above the maximum, an over-estimate of the top (more shadow, never less). No dilation:
    ``ray_top_bounds`` already spreads each hit over its column's azimuth span and its depth bin. Cells
    without an occluder get -100.
    """
    b, p, c = top.shape
    ok = w > 0
    t = top.clamp(min=-(BEV_Z_MAX_M + 4.0), max=BEV_Z_MAX_M + 4.0)
    wz = torch.where(ok, w, torch.zeros_like(w)).unsqueeze(-1)
    flat = _splat_index(idx, ok, OCC_N)
    flat_i = idx.clamp(min=0, max=OCC_N * OCC_N - 1).unsqueeze(-1).expand(b, p, c)
    # first shift: the batch-wide highest value per channel (every cell's maximum is within 16 m of it, so
    # the first pass, k = 5, cannot underflow: a cell has a place iff its first sum is positive)
    ref = torch.where(ok.unsqueeze(-1), t, torch.full_like(t, -(BEV_Z_MAX_M + 4.0))).amax(dim=1, keepdim=True)
    ref_at = ref.expand(b, p, c)
    cur = ref.transpose(1, 2).reshape(b, c, 1, 1).expand(b, c, OCC_N, OCC_N)
    has = None
    for k in TOP_LSE_K:
        s_k = _splat_flat((t - ref_at).clamp_(max=0.0).mul_(k).exp_().mul_(wz), flat, OCC_N)
        if has is None:
            has = s_k[:, :1] > 0
        safe = has & (s_k > 0)
        cur = torch.where(safe, cur + torch.log(torch.where(safe, s_k, torch.ones_like(s_k))) / k, cur)
        ref_at = cur.reshape(b, c, OCC_N * OCC_N).transpose(1, 2).gather(1, flat_i)
    tops = torch.where(has, cur, torch.full_like(cur, -100.0))
    return tops, has.to(top.dtype)


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


def viewshed_blocked(tops: Any, dest_z: Any, T: Any, margin: float = SIGHTLINE_MARGIN_M, bottoms: Any = None) -> Any:
    """Per-camera mask (B x Ncam x N x N): the sight line from the camera to the cell centre at ``dest_z``
    passes at or under some occluder top more than ``margin`` before the cell.

    A horizon sweep: for each camera, ``tops`` (B x 1 x N x N, -100 where none) is sampled along
    ``VIEWSHED_AZ_BINS`` azimuths every ``VIEWSHED_RANGE_STEP_M`` m, turned into the slope
    ``(top - cam_z) / r`` and swept outward with a running maximum (log-step doubling, ONNX friendly).
    The cell is blocked if that running maximum, up to ``r_d - margin``, is at least the slope of its own
    sight line ``(dest_z - cam_z) / r_d``.

    ``bottoms`` (B x 1 x N x N lower bounds on where each cell's occluder starts, +100 where none) drops
    floating occluders: a sight line from the camera to a cell at range r_d <= VIEWSHED_RANGE_M whose
    ground is no higher than ``g_max`` (the highest ``dest_z``) is at range r no higher than
    ``cz - (cz - g_max) r / VIEWSHED_RANGE_M`` if g_max <= cz (else g_max), so an occluder that starts
    above that at its range passes over every such sight line.
    """
    b, ncam = T.shape[:2]
    dtype, device = tops.dtype, tops.device
    na = VIEWSHED_AZ_BINS
    step = float(VIEWSHED_RANGE_STEP_M)
    nr = int(math.ceil(VIEWSHED_RANGE_M / step))
    grid, rr, cam = _polar_grid(T, dtype, device)
    cx, cy, cz = cam[:, 0].view(-1, 1, 1), cam[:, 1].view(-1, 1, 1), cam[:, 2].view(-1, 1, 1)
    no = tops.shape[-1]
    # The sight line to a cell centre lies within half an azimuth bin (<= 0.21 m at 48 m) of its bin's
    # centre line and every point of it within 0.125 m (along) of a range sample: the tops are first
    # max-pooled over 3 x 3 occluder cells (0.25 m), so a cell the line clips by a corner, or a one-cell
    # gap between two hit placements, is not skipped; then one nearest sample per polar bin.
    gv = grid.view(b, ncam * na, nr, 2)  # every camera's polar samples in one call (stacked along rows)
    src = F.max_pool2d(tops + 100.0, 3, stride=1, padding=1)
    top_p = F.grid_sample(src, gv, mode="nearest", padding_mode="zeros", align_corners=False)
    slope = (top_p.view(b * ncam, na, nr) - 100.0 - cz) / rr.view(1, 1, nr)
    if bottoms is not None:
        # lowest bottom over the same 3 x 3 block (off-grid: 100, no occluder)
        bsrc = F.max_pool2d(100.0 - bottoms.clamp(min=-100.0, max=100.0), 3, stride=1, padding=1)
        bot_p = 100.0 - F.grid_sample(bsrc, gv, mode="nearest", padding_mode="zeros", align_corners=False).view(b * ncam, na, nr)
        g_max = dest_z.to(dtype).reshape(b, -1).amax(dim=1).view(b, 1).expand(b, ncam).reshape(b * ncam, 1, 1)
        h_lim = torch.where(g_max <= cz, cz - (cz - g_max) * rr.view(1, 1, nr) / float(VIEWSHED_RANGE_M), g_max.expand_as(cz))
        slope = torch.where(bot_p > h_lim, torch.full_like(slope, -1.0e4), slope)
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


def tight_clear(pts: Any, T: Any, z_rel: Any, tight: float = SEEN_TIGHT_M, z_land: Any = None) -> Any:
    """0/1 per ray point: the point is so close to its ground that no occluder can hide that ground (G5).

    The ray reached the point, so it passed over every occluder before it. At an occluder at range
    r_o the sight line to the ground under the point is ``z_rel * r_o / r_h`` lower than the ray, and
    the sight line to the cell centre (up to half a cell diagonal d nearer) at most
    ``(cz - g) * r_o * d / (r_h (r_h - d))`` lower again. Taking r_o -> r_h (the worst case, an occluder
    anywhere before the point), the cell centre is hidden by less than ``tight`` under any top, which
    is the hidden test, whatever the viewshed found. Used to keep a ray's own landing cells (and its
    last few cm above the road) seen behind a conservatively bounded top.

    ``z_land`` (the height of the ray's own surface, broadcast against ``z_rel``): the point must also be
    less than ``tight`` above it. The local ground is a stand-in where no road is visible in the window
    (only a wall face seen behind a wall close ahead), so "near the ground" alone can hold for a ray
    point in mid-air behind the wall; a point near its own ray's landing is over measured ground.
    """
    b, n = T.shape[:2]
    cam = T[..., :3, 3].to(pts.dtype).view(b, n, *([1] * (pts.dim() - 3)), 3)
    d_c = 0.5 * math.sqrt(2.0) * BEV_CELL_M
    # in place where a temporary is ours (activation budget)
    dy = pts[..., 1] - cam[..., 1]
    r_h = (pts[..., 0] - cam[..., 0]).pow_(2).addcmul_(dy, dy).sqrt_()
    # camera height over the point's ground g = z - z_rel
    worst = (cam[..., 2] - pts[..., 2]).add_(z_rel).clamp_(min=0.0).mul_(d_c).div_((r_h - d_c).clamp_(min=1e-3)).add_(z_rel.clamp(min=0.0))
    ok = (r_h > 2.0 * d_c) & (worst < tight)
    if z_land is not None:
        ok = ok & ((pts[..., 2].to(z_rel.dtype) - z_land) < tight)
    return ok.to(z_rel.dtype)


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


def bev_flat_index(points: Any, n: int = BEV_N) -> tuple[Any, Any]:
    """Flat cell index (row * n + col) and a validity mask for ego xyz points (G2, G3). ``n`` cells span
    the same 64 m square (``OCC_N`` for the finer occluder grid). The lift calls this for the bin centres
    and the bin edges; the visibility gate's derived points (hit placements, sight-line samples) use the
    same arithmetic through ``_flat_index``."""
    return _flat_index(points, n)


def _flat_index(points: Any, n: int = BEV_N) -> tuple[Any, Any]:
    cell = 2.0 * BEV_HALF_M / n
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    row = (BEV_HALF_M - y).div_(cell).floor_()  # in place on our own temporaries (activation budget)
    col = (x + BEV_HALF_M).div_(cell).floor_()
    ok = (row >= 0) & (row < n) & (col >= 0) & (col < n) & (z >= BEV_Z_MIN_M) & (z < BEV_Z_MAX_M)
    idx = torch.where(ok, row.mul_(n).add_(col), torch.zeros_like(col)).long()
    return idx, ok

def splat(values: Any, idx: Any, ok: Any, n: int = BEV_N) -> Any:
    """Sum-pool ``values`` (B x P x C) into ``B x C x n x n`` (scatter_add; default the BEV grid).

    Points with ``ok`` False go to a discard cell that is sliced off, so the
    graph has static shapes and no boolean gather (ONNX friendly).
    """
    return _splat_flat(values, _splat_index(idx, ok, n), n)


def _splat_index(idx: Any, ok: Any, n: int = BEV_N) -> Any:
    """Flat scatter index for ``splat`` (B * P): cell + batch offset, invalid points to the discard cell."""
    b = idx.shape[0]
    cells = n * n
    local = torch.where(ok, idx, torch.full_like(idx, cells))
    base = torch.arange(b, device=idx.device).view(b, 1) * (cells + 1)
    return local.add_(base).reshape(-1)


def _splat_flat(values: Any, flat: Any, n: int = BEV_N) -> Any:
    b, p, c = values.shape
    cells = n * n
    acc = values.new_zeros(b * (cells + 1), c)
    # scatter_add, not index_add: the legacy ONNX exporter turns index_add into a
    # ScatterElements with no reduction, which is wrong for repeated indices.
    # scatter_add exports as ScatterElements(reduction="add") (opset >= 16).
    acc = acc.scatter_add(0, flat.view(-1, 1).expand(b * p, c), values.reshape(b * p, c))
    acc = acc.view(b, cells + 1, c)[:, :cells]
    return acc.permute(0, 2, 1).reshape(b, c, n, n)

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
        z_land = (surf * pts[..., 2].to(prob.dtype)).sum(dim=2, keepdim=True)  # B N 1 h w: own surface height
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
        top_bound, bot_bound, hit_w, hit_idx = ray_top_bounds(pts, self.depths, self.edges, surf * ray_w, ground * has_ground, occ_m, reached, T[..., :3, 3])
        # highest top and (as an upper bound on -bottom, never above the minimum) lowest bottom per cell
        tb, has_occ = occluder_tops(torch.stack([top_bound, -bot_bound], dim=-1), hit_idx, hit_w)
        tops = tb[:, 0:1]
        bottoms = torch.where(has_occ > 0, -tb[:, 1:2], torch.full_like(has_occ, 100.0))
        blocked = viewshed_blocked(tops, g_low, T, bottoms=bottoms)
        ids_c = idx.reshape(b, n, -1).clamp(min=0, max=BEV_N * BEV_N - 1)
        visible = (1.0 - blocked.reshape(b, n, BEV_N * BEV_N).gather(2, ids_c).view_as(idx)) * ok.to(prob.dtype)

        def _rel(zs: Any, ids: Any) -> tuple[Any, Any]:
            g = flat_g.gather(1, ids.reshape(b, -1)).view_as(zs)
            ok_g = flat_has.gather(1, ids.reshape(b, -1)).view_as(zs)
            return zs - g, ok_g * ray_w

        z_rel, ok_rel = _rel(z, idx)
        band_w = near_ground(z_rel, 0.0, self.ground_band_m).to(prob.dtype) * ok_rel  # B N D h w
        # only where the ray passed the point (not in its surface bin: a binned surface can sit up to half
        # a bin past the true one, e.g. behind a wall's foot)
        tight_c = tight_clear(pts, T, z_rel, min(float(self.ground_band_m), SEEN_TIGHT_M), z_land) * ok_rel * through
        clear_w = reach * reached * band_w * crossed * torch.maximum(visible, tight_c)
        through_w = beyond * through * band_w * crossed * torch.maximum(visible, tight_c)
        occ_w = (1.0 - reach) * (1.0 - reached) * band_w
        # Visibility is also sampled at the edge between bins j and j + 1. In the surface band only
        # near-surface points count, and the bin centres alone leave whole cell rows unsampled at range
        # (bins ~1.4 m apart at 15 m, cells 0.5 m). An edge counts only the mass beyond bin j + 1
        # (S_{j+1}) for both reach and pass, so a surface anywhere in bin j + 1, or snapped into it
        # from just before the edge, never opens the edge point.
        idx_e, ok_e = bev_flat_index(pts_e)
        z_rel_e, ok_rel_e = _rel(pts_e[..., 2].to(prob.dtype), idx_e)
        band_e = near_ground(z_rel_e, 0.0, self.ground_band_m).to(prob.dtype) * ok_rel_e
        crossed_e = 1.0 - ray_crossed_raised(idx_e, ok_e, raised)
        ids_e = idx_e.reshape(b, n, -1).clamp(min=0, max=BEV_N * BEV_N - 1)
        visible_e = (1.0 - blocked.reshape(b, n, BEV_N * BEV_N).gather(2, ids_e).view_as(idx_e)) * ok_e.to(prob.dtype)

        s_e = beyond[:, :, 1:]
        reached_e, through_e = visibility_gates(s_e, s_e, self.reach_tau, self.free_pass_tau)
        visible_e = torch.maximum(visible_e, tight_clear(pts_e, T, z_rel_e, min(float(self.ground_band_m), SEEN_TIGHT_M), z_land) * ok_rel_e * through_e)
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
        occ = occ + vis_e[:, 0:1]
        thru = bev[:, CTX_CH + 4 : CTX_CH + 5] + vis_e[:, 2:3]
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
