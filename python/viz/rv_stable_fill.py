"""Reality View stable fill: steady chase view and no unexplained black, camera-only, CPU.

Guarantees, from the problem rather than from the v2 preview code:

G1 view steadiness: the chase camera does not pass the car's own small yaw wobble or
   the camera-only yaw noise straight into the picture. It follows the ego heading
   through a critically damped second-order filter (``DampedHeading``), so steady
   turns are followed and frame-to-frame wobble is filtered out.
G2 depth steadiness: one camera's depth for the same surface does not jump between
   frames. ``CamDepthFilter`` keeps a per-camera, confidence-weighted EMA of log depth
   in that camera's image after warping the previous estimate by ego motion. A
   pixel whose new depth disagrees by more than ``reset_dlog`` (a car moved,
   disocclusion) restarts from the new value, so moving things are not smeared.
G3 persistence: ground seen a moment ago is still there. ``GroundGrid`` keeps a
   world-frame (odometry frame) grid of ground colour with a weight that decays
   with time; a cell is drawn while its weight is above ``min_weight``.
G4 every chase pixel is explained: observed (camera data), filled (a small splat
   gap closed from its painted neighbours), guessed ground (the ground-plane prior
   for road near the ego that no camera currently sees; it is replaced as soon as a
   camera sees it), sky (a vertical gradient from the sky colour the cameras see) or
   far (ground beyond ``far_m``, drawn as horizon haze). ``StableRender.masks``
   says which; only ``unknown`` may stay ``UNKNOWN_RGB``.

Measurement helpers (``temporal_jitter``, ``black_fraction``, ``classify_unknown``)
are what the tests and the before/after numbers use.

Runtime inputs: recorded RGB cameras, the net's predicted depth and sky labels,
camera-only ego motion. No sim depth or sim pose. numpy (+ cv2 for resize) only,
no GPU. Flag: off by default, used only by the offline preview tool.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from python.viz.reality_view_c import UNKNOWN_RGB, CameraPose, camera_from_angles

STABLE_FILL_DEFAULT_ON: bool = False

# measurement
BLACK_TOL: int = 6          # L-inf distance to UNKNOWN_RGB that counts as "unknown dark"
BLACK_MAX_RGB: int = 8      # a pixel this dark in every channel also counts as black
# fusion / view
STATIC_MAX_HEIGHT_M: float = 0.35
GRID_CELL_M: float = 0.25
GRID_HALF_M: float = 60.0
FAR_M: float = 120.0
PRIOR_MAX_M: float = 30.0
# Farthest depth worth drawing per camera: the GT depth of that camera stops at its far plane
# (cameras.json far_m; a pixel at ~far_m is "no hit"), and the depth head's metric range stops at
# FAR_CLASS_M (python/train/depth_hires.py: GT >= 250 m is the "far" class, not a distance).
HEAD_FAR_CLASS_M: float = 250.0
GT_NO_HIT_FRAC: float = 0.995

MASK_NAMES = ("observed", "filled", "ground_prior", "sky", "far", "unknown")
CAUSES = ("sky", "far_plane", "never_seen", "occluded", "invalid_depth", "splat_gap")


# ------------------------------------------------------------------ measurement
def _luma(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"need (H, W, 3) RGB, got {a.shape}")
    a = a.astype(np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def temporal_jitter(frames: Sequence[np.ndarray]) -> float:
    """High-frequency temporal energy: mean over t of mean_px |Y_t - (Y_{t-1} + Y_{t+1}) / 2|.

    Luma 0..255. Smooth motion (constant image velocity) scores near 0; a frame that
    jumps away and back scores high. Needs >= 3 equal-shape RGB frames.
    """
    if len(frames) < 3:
        raise ValueError("temporal_jitter needs at least 3 frames")
    ys = [_luma(f) for f in frames]
    if any(y.shape != ys[0].shape for y in ys):
        raise ValueError("temporal_jitter needs equal-shape frames")
    vals = [float(np.mean(np.abs(ys[t] - 0.5 * (ys[t - 1] + ys[t + 1])))) for t in range(1, len(ys) - 1)]
    return float(np.mean(vals))


def black_mask(rgb: np.ndarray, unknown_rgb: Sequence[int] = UNKNOWN_RGB, tol: int = BLACK_TOL) -> np.ndarray:
    """Pixels that read as black: within ``tol`` (L-inf) of UNKNOWN_RGB, or <= BLACK_MAX_RGB in every channel."""
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"need (H, W, 3) RGB, got {a.shape}")
    a = a.astype(np.int32)
    near_unk = np.max(np.abs(a - np.asarray(unknown_rgb, np.int32)[None, None, :]), axis=2) <= int(tol)
    return near_unk | (np.max(a, axis=2) <= BLACK_MAX_RGB)


def black_fraction(rgb: np.ndarray, **kw) -> float:
    return float(np.mean(black_mask(rgb, **kw)))


def _pixel_rays(cam: CameraPose) -> np.ndarray:
    """Unit-free ego-frame ray directions (H, W, 3) through pixel centres."""
    ys, xs = np.mgrid[0:cam.height, 0:cam.width].astype(np.float64)
    d = np.stack([(xs + 0.5 - cam.cx) / cam.fx, -(ys + 0.5 - cam.cy) / cam.fy, np.ones_like(xs)], -1)
    return d @ cam.rotation_matrix()


def project(cam: CameraPose, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ego-frame points -> (u, v, z) in ``cam`` (z = planar depth along the optical axis)."""
    p = np.asarray(pts, np.float64).reshape(-1, 3)
    cp = (p - np.asarray(cam.pos)[None, :]) @ cam.rotation_matrix().T
    z = cp[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.cx + cam.fx * cp[:, 0] / z
        v = cam.cy - cam.fy * cp[:, 1] / z
    return u, v, z


@dataclass(frozen=True)
class CamEvidence:
    """What one camera contributed this frame (for black-cause classification)."""

    cam: CameraPose
    depth: np.ndarray          # planar depth (H, W), metres
    valid: np.ndarray          # pixel became a point
    sky: np.ndarray | None = None


def classify_unknown(chase: CameraPose, unknown: np.ndarray, evidence: Sequence[CamEvidence], *,
                     far_m: float = FAR_M, seen_ratio: float = 0.85) -> dict[str, float]:
    """Split unknown chase pixels by cause, as fractions of ALL chase pixels.

    Each unknown pixel's ray is cast onto the ground plane z = 0:
      sky          ray never reaches the ground (at or above the horizon)
      far_plane    reaches the ground beyond ``far_m``
      never_seen   the ground point is outside every camera's image
      occluded     inside a camera's image, but that camera saw a nearer surface there
      invalid_depth inside a camera's image, but no camera kept a point there (cut, sky
                   label, ego mask, non-finite depth)
      splat_gap    a camera saw it with valid depth at about the right range, yet the
                   render left it unpainted
    The fractions sum to the unknown fraction.
    """
    unk = np.asarray(unknown, bool)
    if unk.shape != (chase.height, chase.width):
        raise ValueError(f"unknown mask {unk.shape} is not the chase shape {(chase.height, chase.width)}")
    out = {c: 0.0 for c in CAUSES}
    n_all = float(unk.size)
    if not unk.any():
        return out
    rays = _pixel_rays(chase)[unk]
    pos = np.asarray(chase.pos, np.float64)
    dz = rays[:, 2]
    cause = np.full(rays.shape[0], -1, np.int64)
    sky = dz >= -1e-9
    cause[sky] = 0
    t = np.where(sky, np.nan, -pos[2] / np.where(sky, -1.0, dz))
    X = pos[None, :] + t[:, None] * rays
    rng = np.hypot(X[:, 0], X[:, 1])
    far = ~sky & (rng > far_m)
    cause[far] = 1
    todo = cause < 0
    inside_any = np.zeros(rays.shape[0], bool)
    seen_ok = np.zeros(rays.shape[0], bool)
    seen_near = np.zeros(rays.shape[0], bool)
    for ev in evidence:
        if not todo.any():
            break
        u, v, z = project(ev.cam, X)
        ui, vi = np.floor(u), np.floor(v)
        ins = todo & np.isfinite(z) & (z > ev.cam.near_m) & (ui >= 0) & (ui < ev.cam.width) & (vi >= 0) & (vi < ev.cam.height)
        if not ins.any():
            continue
        inside_any |= ins
        ii = np.nonzero(ins)[0]
        yy, xx = vi[ii].astype(np.int64), ui[ii].astype(np.int64)
        val = np.asarray(ev.valid, bool)[yy, xx]
        dep = np.asarray(ev.depth, np.float64)[yy, xx]
        ok = val & np.isfinite(dep) & (dep >= seen_ratio * z[ii])
        near = val & np.isfinite(dep) & (dep < seen_ratio * z[ii])
        seen_ok[ii[ok]] = True
        seen_near[ii[near]] = True
    c = np.where(~inside_any, 2, np.where(seen_ok, 5, np.where(seen_near, 3, 4)))
    cause[todo] = c[todo]
    for k, name in enumerate(CAUSES):
        out[name] = float(np.sum(cause == k)) / n_all
    return out


# ------------------------------------------------------------------ view pose
@dataclass
class DampedHeading:
    """Critically damped 2nd-order follower of the ego heading (rad).

    ``omega`` (rad/s) sets the bandwidth: a step in heading is followed with time
    constant 1/omega and no overshoot; wobble well above omega is attenuated.
    """

    omega: float = 1.5
    h: float | None = None
    rate: float = 0.0

    def step(self, target: float, dt: float) -> float:
        if not (math.isfinite(target) and math.isfinite(dt)) or dt < 0:
            raise ValueError("non-finite heading input")
        if self.h is None:
            self.h, self.rate = float(target), 0.0
            return self.h
        # integrate in small substeps for stability at large dt
        n = max(1, int(math.ceil(dt / 0.02)))
        hstep = dt / n
        w = float(self.omega)
        for _ in range(n):
            err = math.atan2(math.sin(target - self.h), math.cos(target - self.h))
            acc = w * w * err - 2.0 * w * self.rate
            self.rate += acc * hstep
            self.h += self.rate * hstep
        return self.h


def chase_camera(base_pos: Sequence[float], pitch_deg: float, fov_v: float, width: int, height: int,
                 yaw_offset_rad: float, *, far_m: float = 400.0) -> CameraPose:
    """Chase camera in the CURRENT ego frame, swung by ``yaw_offset_rad`` about the ego origin.

    ``yaw_offset_rad`` = damped heading - ego heading (CCW +). 0 gives the rigid camera.
    """
    c, s = math.cos(yaw_offset_rad), math.sin(yaw_offset_rad)
    x, y, z = (float(v) for v in base_pos)
    pos = (c * x - s * y, s * x + c * y, z)
    # camera_from_angles yaw is clockwise-positive from +Y (fwd = (sin yaw, cos yaw)); CCW offset -> negative yaw
    return camera_from_angles("chase", pos, -math.degrees(yaw_offset_rad), pitch_deg, fov_v, width, height,
                              near_m=0.05, far_m=far_m)


# ------------------------------------------------------------------ depth
@dataclass
class CamDepthFilter:
    """Per-camera confidence-weighted EMA of log depth, carried between frames by ego motion.

    Backward warp, so there are no holes when the scene expands as the car drives on: each
    current pixel is unprojected with its new depth, moved into the previous frame's ego pose and
    looked up in the previous filtered depth image. If the previous surface there agrees in log
    depth within ``reset_dlog``, the previous surface point is carried into the current frame
    and blended with the new value by weight; otherwise the pixel restarts from the new value.
    """

    reset_dlog: float = 0.25
    half_life_s: float = 1.0
    max_weight: float = 6.0
    depth: np.ndarray | None = None    # previous filtered planar depth (H, W)
    w: np.ndarray | None = None
    valid: np.ndarray | None = None
    cam: CameraPose | None = None
    pose: tuple[float, float, float] | None = None
    t: float | None = None

    def update(self, cam: CameraPose, depth: np.ndarray, valid: np.ndarray, pose: tuple[float, float, float],
               t: float, conf: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (filtered planar depth, weight) for this frame and keep the state."""
        d = np.asarray(depth, np.float64)
        h, w_ = d.shape
        val = np.asarray(valid, bool) & np.isfinite(d) & (d > 0)
        c = np.ones_like(d) if conf is None else np.asarray(conf, np.float64)
        cur = np.log(np.where(val, d, 1.0))
        out_l = cur.copy()
        out_w = np.where(val, c, 0.0)
        if (self.depth is not None and self.cam is not None and self.pose is not None and self.t is not None
                and self.depth.shape == d.shape):
            decay = 0.5 ** (max(0.0, t - self.t) / self.half_life_s)
            ys, xs = np.nonzero(val)
            P = unproject(cam, d, ys, xs)                      # current ego frame
            Pp = transform_points(P, pose, self.pose)          # previous ego frame
            u, v, z = project(self.cam, Pp)
            ui, vi = np.floor(u), np.floor(v)
            ok = np.isfinite(z) & (z > self.cam.near_m) & (ui >= 0) & (ui < w_) & (vi >= 0) & (vi < h)
            if ok.any():
                k = np.nonzero(ok)[0]
                yi, xi = vi[k].astype(np.int64), ui[k].astype(np.int64)
                pv = self.valid[yi, xi]
                k, yi, xi = k[pv], yi[pv], xi[pv]
                dp = self.depth[yi, xi]
                agree_prev = np.abs(np.log(dp) - np.log(z[k])) < self.reset_dlog
                k, yi, xi, dp = k[agree_prev], yi[agree_prev], xi[agree_prev], dp[agree_prev]
                if k.size:
                    # the previous surface point along this pixel's line of sight, back in the current frame
                    Q = Pp[k] * (dp / z[k])[:, None] + np.asarray(self.cam.pos)[None, :] * (1.0 - dp / z[k])[:, None]
                    Qc = transform_points(Q, self.pose, pose)
                    _, _, zc = project(cam, Qc)
                    ww = self.w[yi, xi] * decay
                    cy, cx = ys[k], xs[k]
                    good = np.isfinite(zc) & (zc > 0)
                    cy, cx, zc, ww = cy[good], cx[good], zc[good], ww[good]
                    lw = np.log(zc)
                    cc = c[cy, cx]
                    out_l[cy, cx] = (ww * lw + cc * cur[cy, cx]) / (ww + cc)
                    out_w[cy, cx] = np.minimum(ww + cc, self.max_weight)
        depth_f = np.where(val, np.exp(out_l), d)
        # the blend uses a neighbouring ray's carried point (floor lookup): never let it push a pixel below the
        # road plane that its own input had on or above it (snapped/clamped road stayed flat before the filter)
        rz = _pixel_rays(cam)[..., 2]
        hcam = float(cam.pos[2])
        if hcam > 0:
            dn = rz < -1e-6
            hit = np.where(dn, hcam / np.where(dn, -rz, 1.0), np.inf)
            keep = val & dn & (d <= hit * (1 + 1e-9)) & (depth_f > hit)
            depth_f = np.where(keep, hit, depth_f)
        self.depth = np.where(val, depth_f, 1.0)
        self.w = out_w
        self.valid = val
        self.cam = cam
        self.pose = tuple(pose)
        self.t = float(t)
        return depth_f, out_w


def depth_cap_m(cam_far_m: float, head_far_m: float = HEAD_FAR_CLASS_M) -> float:
    """Farthest metric depth to draw for one camera: min(GT far plane x 0.995, head far class)."""
    f = float(cam_far_m)
    if not (math.isfinite(f) and f > 0):
        raise ValueError(f"camera far_m must be finite and > 0, got {cam_far_m!r}")
    return min(f * GT_NO_HIT_FRAC, float(head_far_m))


def far_to_cap(depth: np.ndarray, sky: np.ndarray, cap_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Far-flagged pixels are drawn at the cap, never dropped.

    A non-sky pixel whose depth is beyond ``cap_m`` (or +inf: the net / GT said "far") is placed at
    ``cap_m`` along its own ray, so distant buildings stay in the picture at the farthest range the
    depth source supports. Sky stays sky. Returns (depth, far mask).
    """
    d = np.asarray(depth, np.float64)
    sk = np.asarray(sky, bool)
    if d.shape != sk.shape:
        raise ValueError(f"depth {d.shape} and sky {sk.shape} differ")
    far = ~sk & ((d > cap_m) | np.isposinf(d))
    return np.where(far, float(cap_m), d), far


def ground_clamp(cam: CameraPose, depth: np.ndarray, *, ground_z: float = 0.0, margin: float = 1.1) -> tuple[np.ndarray, np.ndarray]:
    """No surface below the road: cap planar depth at the ray's ground-plane hit (times ``margin``).

    A downward pixel ray from a camera ``h`` metres above the ground meets the road at planar depth
    ``h / (-dz) * fwd_z``; anything the net puts farther than that would sit under the road and
    renders as road bending down away from the car (seen on clip015 in the tunnel, where the wide
    cam's bottom rows came out 2.5x too far). Rays at or above the horizon are untouched.
    Returns (clamped depth, clamped mask). Geometry only: camera mount + flat-ground assumption.
    """
    d = np.asarray(depth, np.float64)
    if d.shape != (cam.height, cam.width):
        raise ValueError(f"depth {d.shape} is not the camera shape {(cam.height, cam.width)}")
    h = float(cam.pos[2]) - float(ground_z)
    if not (h > 0 and math.isfinite(h)):
        return d.copy(), np.zeros(d.shape, bool)
    rays = _pixel_rays(cam)                         # ego rays scaled to planar depth 1 (H, W, 3)
    dz = rays[..., 2]
    down = dz < -1e-6
    zmax = np.where(down, h / np.where(down, -dz, 1.0), np.inf) * float(margin)  # planar depth of the ground hit
    clamp = down & ~np.isnan(d) & (d > zmax)        # +inf on a downward ray is under the road too
    return np.where(clamp, zmax, d), clamp


ROAD_CLASS_IDS: tuple[int, ...] = (2,)   # net semantic id for drivable road (clip015: GT/ground-hit 0.86-1.04)
ROAD_SNAP_MAX_M: float = 25.0           # GT road rises 0.4-1.1 deg off ego z=0 (clip015); beyond ~25 m pinning to z=0 is worse than net depth (f40 15-25 m: -0.55 m at 60 vs +0.15 m at 25)
ROAD_SNAP_ERODE_PX: int = 4             # stay this many px inside the road class (12x16 class grid edges are blocky)
# Runtime snap rejection (no GT): a road-labelled px whose net depth already puts it this far ABOVE the road
# plane (net depth much shorter than the ground-ray hit) is a wall/car/pole/curb top under the coarse 12x16
# label, not road; snapping it would throw it down and outward onto the plane (Tracer: 37% of snapped px on
# clip015 f40 pillarR had GT z > 0.3 m, landing +5-7 m outward). Px below the plane (the dips) still snap.
ROAD_SNAP_MAX_ABOVE_M: float = 0.30
# Cams whose road is snapped (switchable; GVD direction 2026-11-26: restore snap on the non-wide cams, keep the 0.30 m
# rejection; wide stays as it was until Michael picks). narrow stays unsnapped. NOTE the frozen bins (scratch/stable/bins6.py,
# median |dz| m, off -> snap + rejection) say main/narrow lose to snap at 5-15 m even with the rejection: f160 main 5-10 m
# 0.08 -> 0.13, 10-15 m 0.12 -> 0.19; f100 main 10-15 m 0.12 -> 0.18; narrow 10-15 m 0.09 -> 0.21. Wide with the rejection
# gains at 0-5 m (f160 0.11 -> 0.07, held-out f100 0.20 -> 0.06) and is unchanged at 5-15 m. Side cams: see gate (b) in
# tests/props (pillarL/R, repeatR snap curb/car px).
ROAD_SNAP_CAMS: tuple[str, ...] = ("wide", "main", "pillarL", "pillarR", "repeatL", "repeatR", "rear")


def _erode(mask: np.ndarray, r: int) -> np.ndarray:
    """Square binary erosion (radius ``r`` px), numpy only; outside the image counts as False."""
    m = np.asarray(mask, bool)
    if r <= 0:
        return m.copy()
    h, w = m.shape
    pad = np.zeros((h + 2 * r, w + 2 * r), bool)
    pad[r:r + h, r:r + w] = m
    rows = np.ones((h + 2 * r, w), bool)
    for k in range(2 * r + 1):
        rows &= pad[:, k:k + w]
    out = np.ones((h, w), bool)
    for k in range(2 * r + 1):
        out &= rows[k:k + h]
    return out


def road_snap(cam: CameraPose, depth: np.ndarray, road: np.ndarray, *, ground_z: float = 0.0,
              max_hit_m: float = ROAD_SNAP_MAX_M, erode_px: int = ROAD_SNAP_ERODE_PX,
              max_above_m: float = ROAD_SNAP_MAX_ABOVE_M) -> tuple[np.ndarray, np.ndarray]:
    """Road pixels lie on the road: replace their depth with the exact ground-plane hit.

    The net's depth is a 12x16 grid per crop, upsampled; across the near road the true depth changes
    by ~10x over a few grid rows, so the upsampled road bends (below the road near the car, above it
    farther out) and the main/wide seams on the road do not meet. For pixels the net labels road, the
    camera mount already gives the answer at full resolution. Only pixels ``erode_px`` inside the
    road label, with a downward ray hitting within ``max_hit_m``, and not already more than ``max_above_m``
    above the plane by the net's own depth (wall/car/pole under a coarse label), are snapped. Returns (depth, mask).
    """
    d = np.asarray(depth, np.float64)
    rd = np.asarray(road, bool)
    if d.shape != (cam.height, cam.width) or rd.shape != d.shape:
        raise ValueError(f"depth {d.shape} / road {rd.shape} must be the camera shape {(cam.height, cam.width)}")
    h = float(cam.pos[2]) - float(ground_z)
    if not (h > 0 and math.isfinite(h)):
        return d.copy(), np.zeros(d.shape, bool)
    dz = _pixel_rays(cam)[..., 2]
    down = dz < -1e-6
    hit = np.where(down, h / np.where(down, -dz, 1.0), np.inf)
    with np.errstate(invalid="ignore"):
        above = np.isfinite(d) & (h + d * dz > float(max_above_m))   # net puts it > max_above_m over the plane
    snap = _erode(rd, int(erode_px)) & down & (hit <= float(max_hit_m)) & ~above
    return np.where(snap, hit, d), snap


def snap_road_for_cam(cam_id: str, cam: CameraPose, depth: np.ndarray, road: np.ndarray, **kw) -> tuple[np.ndarray, np.ndarray]:
    """``road_snap`` for ROAD_SNAP_CAMS; other cams come back unchanged with an empty mask."""
    d = np.asarray(depth, np.float64)
    if cam_id in ROAD_SNAP_CAMS:
        return road_snap(cam, d, road, **kw)
    if d.shape != (cam.height, cam.width) or np.shape(road) != d.shape:
        raise ValueError(f"depth {d.shape} / road {np.shape(road)} must be the camera shape {(cam.height, cam.width)}")
    return d.copy(), np.zeros(d.shape, bool)


def yield_mask(cam: CameraPose, better: Sequence[CameraPose], *, margin_px: float = 4.0, probe_m: float = 200.0) -> np.ndarray:
    """Pixels of ``cam`` whose viewing direction a better camera also covers (static, per rig).

    Two cameras drawing the same surface with different depth errors leave a doubled, stepped
    surface, and the nearer (usually wrong) copy wins the z-buffer: on clip015 the wide cam's too-near
    points painted over almost all of main's view. Each direction is drawn by one camera only: a
    pixel yields when its ray, probed ``probe_m`` out (the rig baseline is cm, so parallax is
    negligible), lands at least ``margin_px`` inside a better camera's image.
    """
    rays = _pixel_rays(cam).reshape(-1, 3)
    pts = np.asarray(cam.pos)[None, :] + rays * (probe_m / np.linalg.norm(rays, axis=1, keepdims=True))
    out = np.zeros(rays.shape[0], bool)
    for b in better:
        u, v, z = project(b, pts)
        m = float(margin_px)
        out |= np.isfinite(u) & (z > 0) & (u >= m) & (u <= b.width - m) & (v >= m) & (v <= b.height - m)
    return out.reshape(cam.height, cam.width)


def unproject(cam: CameraPose, depth: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    d = np.asarray(depth, np.float64)[ys, xs]
    xc = ((xs + 0.5 - cam.cx) / cam.fx) * d
    yc = -((ys + 0.5 - cam.cy) / cam.fy) * d
    return np.asarray(cam.pos)[None, :] + np.stack([xc, yc, d], -1) @ cam.rotation_matrix()


def _rot(h: float) -> np.ndarray:
    c, s = math.cos(h), math.sin(h)
    return np.array([[c, -s], [s, c]])


def transform_points(pts: np.ndarray, pose_from: Sequence[float], pose_to: Sequence[float]) -> np.ndarray:
    """Ego-frame points at ``pose_from`` (x, y, heading CCW) -> ego frame at ``pose_to``. z unchanged."""
    p = np.asarray(pts, np.float64).reshape(-1, 3)
    if p.shape[0] == 0:
        return p
    xy_w = p[:, :2] @ _rot(pose_from[2]).T + np.asarray(pose_from[:2])
    xy_t = (xy_w - np.asarray(pose_to[:2])) @ _rot(pose_to[2])
    return np.concatenate([xy_t, p[:, 2:3]], 1)


# ------------------------------------------------------------------ world ground grid
@dataclass
class GroundGrid:
    """World-frame ground colour memory with time decay. Cells are GRID_CELL_M square."""

    cell: float = GRID_CELL_M
    half_life_s: float = 4.0
    min_weight: float = 0.15
    cells: dict = field(default_factory=dict)   # (ix, iy) -> [r, g, b, z, w, t]
    _arr: tuple | None = None

    def add(self, pts_world: np.ndarray, cols: np.ndarray, conf: np.ndarray, t: float) -> None:
        p = np.asarray(pts_world, np.float64).reshape(-1, 3)
        if p.shape[0] == 0:
            return
        ix = np.floor(p[:, 0] / self.cell).astype(np.int64)
        iy = np.floor(p[:, 1] / self.cell).astype(np.int64)
        key = (ix << 32) ^ (iy & 0xFFFFFFFF)
        uk, inv = np.unique(key, return_inverse=True)
        cw = np.bincount(inv, weights=conf)
        col = np.stack([np.bincount(inv, weights=conf * cols[:, k]) for k in range(3)], 1) / cw[:, None]
        zz = np.bincount(inv, weights=conf * p[:, 2]) / cw
        first = np.zeros(uk.size, np.int64)
        first[inv[::-1]] = np.arange(inv.size)[::-1]
        for j in range(uk.size):
            k = (int(ix[first[j]]), int(iy[first[j]]))
            old = self.cells.get(k)
            if old is None:
                self.cells[k] = [col[j, 0], col[j, 1], col[j, 2], zz[j], min(cw[j], 6.0), t]
            else:
                ow = old[4] * 0.5 ** (max(0.0, t - old[5]) / self.half_life_s)
                nw = ow + cw[j]
                a = cw[j] / nw
                self.cells[k] = [old[0] + a * (col[j, 0] - old[0]), old[1] + a * (col[j, 1] - old[1]),
                                 old[2] + a * (col[j, 2] - old[2]), old[3] + a * (zz[j] - old[3]), min(nw, 6.0), t]
        self._arr = None

    def live(self, t: float, centre_xy: Sequence[float], radius: float = GRID_HALF_M) -> tuple[np.ndarray, np.ndarray]:
        """(world xyz of cell centres, rgb uint8) of cells still above ``min_weight`` near ``centre_xy``."""
        if not self.cells:
            return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
        keys = np.array(list(self.cells.keys()), np.int64)
        vals = np.array(list(self.cells.values()), np.float64)
        w = vals[:, 4] * 0.5 ** (np.maximum(0.0, t - vals[:, 5]) / self.half_life_s)
        xy = (keys + 0.5) * self.cell
        near = np.hypot(xy[:, 0] - centre_xy[0], xy[:, 1] - centre_xy[1]) <= radius
        keep = (w >= self.min_weight) & near
        # forget cells that decayed away or are far behind
        drop = (w < self.min_weight * 0.5) | ~near
        if drop.any():
            for k in keys[drop]:
                self.cells.pop((int(k[0]), int(k[1])), None)
        pts = np.concatenate([xy[keep], vals[keep, 3:4]], 1)
        return pts, np.clip(np.round(vals[keep, :3]), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ render
@dataclass(frozen=True)
class StableRender:
    rgb: np.ndarray
    depth: np.ndarray
    masks: Mapping[str, np.ndarray]

    def fraction(self, name: str) -> float:
        return float(np.mean(self.masks[name]))


def splat(chase: CameraPose, pts: np.ndarray, cols: np.ndarray, foot: np.ndarray, *,
          splat_scale: float = 1.0, max_radius_px: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-buffered square splats, radius = splat_scale * fy * footprint / (2 z). Nearest wins, ties to lower index."""
    H, W = chase.height, chase.width
    rgb = np.empty((H, W, 3), np.uint8)
    rgb[:] = UNKNOWN_RGB
    dep = np.zeros((H, W), np.float32)
    obs = np.zeros((H, W), bool)
    p = np.asarray(pts, np.float64).reshape(-1, 3)
    if p.shape[0] == 0:
        return rgb, dep, obs
    u, v, z = project(chase, p)
    keep = np.isfinite(z) & (z > chase.near_m) & (z <= chase.far_m)
    u, v, z = u[keep], v[keep], z[keep]
    cols = np.asarray(cols)[keep]
    f = np.asarray(foot, np.float64)[keep]
    r = np.clip(np.floor(splat_scale * 0.5 * chase.fy * f / z + 0.5), 0, max_radius_px).astype(np.int64)
    px, py = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    near = (px >= -r) & (px < W + r) & (py >= -r) & (py < H + r)
    px, py, z, cols, r = px[near], py[near], z[near], cols[near], r[near]
    if px.size == 0:
        return rgb, dep, obs
    if px.size >= (1 << 24):
        raise ValueError("too many points for the packed z-buffer key")
    ids = np.arange(px.size, dtype=np.int64)
    zkey = (np.clip(np.round(z * 1000.0), 0, (1 << 38) - 1).astype(np.int64) << 24) | ids
    zbuf = np.full(H * W, np.iinfo(np.int64).max, np.int64)
    for rad in np.unique(r):
        sel = np.nonzero(r == rad)[0]
        offs = np.arange(-rad, rad + 1)
        oy, ox = np.meshgrid(offs, offs, indexing="ij")
        oy, ox = oy.ravel(), ox.ravel()
        step = max(1, 2_000_000 // oy.size)
        for c0 in range(0, sel.size, step):
            s_ = sel[c0:c0 + step]
            yy = (py[s_][:, None] + oy[None, :]).ravel()
            xx = (px[s_][:, None] + ox[None, :]).ravel()
            kk = np.repeat(zkey[s_], oy.size)
            ins = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            np.minimum.at(zbuf, yy[ins] * W + xx[ins], kk[ins])
    hit = np.nonzero(zbuf != np.iinfo(np.int64).max)[0]
    if hit.size:
        win = zbuf[hit] & ((1 << 24) - 1)
        rgb.reshape(-1, 3)[hit] = cols[win]
        dep.reshape(-1)[hit] = z[win]
        obs.reshape(-1)[hit] = True
    return rgb, dep, obs


def fill_gaps(rgb: np.ndarray, painted: np.ndarray, *, win: int = 5, min_frac: float = 0.35, iters: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Close small splat gaps: an unpainted pixel with >= ``min_frac`` painted pixels in its
    ``win`` x ``win`` window takes their mean colour. Large holes are left alone."""
    import cv2

    out = rgb.astype(np.float32).copy()
    pm = np.asarray(painted, bool).copy()
    filled = np.zeros_like(pm)
    k = (int(win), int(win))
    for _ in range(int(iters)):
        m = pm.astype(np.float32)
        cnt = cv2.boxFilter(m, -1, k, normalize=True, borderType=cv2.BORDER_CONSTANT)
        acc = cv2.boxFilter(out * m[..., None], -1, k, normalize=True, borderType=cv2.BORDER_CONSTANT)
        new = (~pm) & (cnt >= min_frac)
        if not new.any():
            break
        out[new] = acc[new] / cnt[new][:, None]
        pm |= new
        filled |= new
    return np.clip(np.round(out), 0, 255).astype(np.uint8), filled


def sky_gradient(height: int, width: int, horizon_v: np.ndarray, sky_rgb: Sequence[float], haze_rgb: Sequence[float]) -> np.ndarray:
    """Vertical gradient: ``sky_rgb`` at the top, ``haze_rgb`` at the horizon row (per column)."""
    v = np.arange(height, dtype=np.float64)[:, None]
    hv = np.asarray(horizon_v, np.float64).reshape(1, width)
    a = np.clip(1.0 - (hv - v) / np.maximum(hv, 1.0), 0.0, 1.0)  # 0 at top, 1 at horizon
    a = a ** 1.5
    s, hz = np.asarray(sky_rgb, np.float64), np.asarray(haze_rgb, np.float64)
    return np.clip(np.round((1 - a)[..., None] * s + a[..., None] * hz), 0, 255).astype(np.uint8)


def compose(chase: CameraPose, pts: np.ndarray, cols: np.ndarray, foot: np.ndarray, *,
            sky_rgb: Sequence[float] | None, ground_rgb: Sequence[float] | None,
            far_m: float = FAR_M, prior_max_m: float = PRIOR_MAX_M, splat_scale: float = 1.0,
            max_radius_px: int = 4, fill: bool = True) -> StableRender:
    """Splat, close small gaps, then explain every remaining pixel (sky / far / ground prior / unknown)."""
    rgb, dep, obs = splat(chase, pts, cols, foot, splat_scale=splat_scale, max_radius_px=max_radius_px)
    filled = np.zeros_like(obs)
    if fill:
        rgb, filled = fill_gaps(rgb, obs)
    painted = obs | filled
    rays = _pixel_rays(chase)
    dz = rays[..., 2]
    up = dz >= -1e-9
    t = np.where(up, np.inf, -chase.pos[2] / np.where(up, -1.0, dz))
    gx = chase.pos[0] + t * rays[..., 0]
    gy = chase.pos[1] + t * rays[..., 1]
    rng = np.where(up, np.inf, np.hypot(gx, gy))
    sky_m = ~painted & up
    far_m_ = ~painted & ~up & (rng > far_m)
    prior = ~painted & ~up & (rng <= prior_max_m)
    unknown = ~painted & ~sky_m & ~far_m_ & ~prior
    haze = None
    if sky_rgb is not None and (sky_m.any() or far_m_.any()):
        s = np.asarray(sky_rgb, np.float64)
        haze = 0.6 * s + 0.4 * np.array([225.0, 225.0, 225.0])
        # horizon row per column: where the ray elevation crosses 0
        cols_h = np.full(chase.width, float(chase.height))
        crossing = up[:-1] & ~up[1:]
        for x in range(chase.width):
            r_ = np.nonzero(crossing[:, x])[0]
            if r_.size:
                cols_h[x] = float(r_[0] + 1)
            elif up[:, x].all():
                cols_h[x] = float(chase.height)
            else:
                cols_h[x] = 0.0
        g = sky_gradient(chase.height, chase.width, cols_h, s, haze)
        rgb[sky_m] = g[sky_m]
        rgb[far_m_] = np.clip(np.round(haze), 0, 255).astype(np.uint8)
    else:
        unknown |= sky_m | far_m_
        sky_m = np.zeros_like(sky_m)
        far_m_ = np.zeros_like(far_m_)
    if ground_rgb is not None and prior.any():
        # guessed clear road: flat colour, slightly darker with range so it reads as ground
        shade = np.clip(1.0 - 0.25 * (rng / max(prior_max_m, 1e-6)), 0.7, 1.0)
        g = np.asarray(ground_rgb, np.float64)[None, None, :] * shade[..., None]
        rgb[prior] = np.clip(np.round(g[prior]), 0, 255).astype(np.uint8)
    else:
        unknown |= prior
        prior = np.zeros_like(prior)
    masks = {"observed": obs, "filled": filled, "ground_prior": prior, "sky": sky_m, "far": far_m_, "unknown": unknown}
    return StableRender(rgb, dep, masks)
