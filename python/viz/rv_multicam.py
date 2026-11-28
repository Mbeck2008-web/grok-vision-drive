"""Reality View multi-camera fusion (offline, CPU, NumPy only).

Spec: ``docs/specs/reality_view_3dgs.md`` (C path, offline). Extends the step-2
C prototype (``reality_view_c``) from one camera to the whole rig:

* every camera with a known extrinsic is unprojected into ONE ego frame per
  timestep, from a *predicted* depth map per camera (never sim GT);
* edge-aware upsampling of low-res predicted depth (joint bilateral upsampling
  of inverse depth, guided by the camera's own image);
* depth-discontinuity cuts, so a foreground edge never stretches into a sheet;
* far / sky pixels go onto a distant backdrop dome instead of being dropped, so
  the whole camera view is painted;
* per-pixel splat size from depth (the pixel's world footprint), z-buffered,
  supersampled and box-downsampled (anti-aliasing);
* short multi-frame accumulation using ego motion from wheel speed plus a
  camera-only yaw estimate (no sim pose). Past frames contribute only
  low, static-looking points (ground), so moving cars do not ghost.

Honesty: a chase pixel that no camera covers stays ``UNKNOWN_RGB`` and is
reported as not painted. Offline / test-process only; never imported by the
supervisor or the live loop. No new dependencies (NumPy + stdlib).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from python.viz.reality_view_c import (
    UNKNOWN_RGB,
    CameraPose,
    _check_cam_id,
    _validate_pose,
    camera_from_angles,
)

MAX_FUSED_POINTS: int = 1_200_000
MAX_SPLAT_RADIUS_PX: int = 6
DEFAULT_DOME_M: float = 150.0
DEFAULT_FAR_M: float = 120.0
DEFAULT_CUT_REL: float = 0.12
STATIC_MAX_HEIGHT_M: float = 0.35
# Rig-specific image regions where a camera sees the ego car itself (fractions: row_from, col_from, col_to).
# sbr rig: the wide camera sees the green hood in its bottom ~20 %.
EGO_IMAGE_MASKS_SBR: dict[str, tuple[float, float, float]] = {"wide": (0.80, 0.10, 0.90)}
EGO_BOX_SBR: tuple[float, float, float, float, float] = (1.0, -2.45, 2.4, -0.5, 1.6)


@dataclass(frozen=True)
class FusedPoints:
    """Ego-frame points (N,3) float32, colours (N,3) uint8, world footprint (N,) metres per pixel."""

    pts: np.ndarray
    cols: np.ndarray
    foot: np.ndarray
    static: np.ndarray

    @property
    def n(self) -> int:
        return int(self.pts.shape[0])


@dataclass(frozen=True)
class SplatRender:
    rgb: np.ndarray
    depth_m: np.ndarray
    observed: np.ndarray
    point_count: int

    @property
    def painted_fraction(self) -> float:
        return float(self.observed.mean()) if self.observed.size else 0.0


def empty_points() -> FusedPoints:
    return FusedPoints(
        pts=np.zeros((0, 3), np.float32),
        cols=np.zeros((0, 3), np.uint8),
        foot=np.zeros((0,), np.float32),
        static=np.zeros((0,), bool),
    )


def concat_points(parts: Sequence[FusedPoints], cap: int = MAX_FUSED_POINTS) -> FusedPoints:
    parts = [p for p in parts if p.n]
    if not parts:
        return empty_points()
    out = FusedPoints(
        pts=np.concatenate([p.pts for p in parts]),
        cols=np.concatenate([p.cols for p in parts]),
        foot=np.concatenate([p.foot for p in parts]),
        static=np.concatenate([p.static for p in parts]),
    )
    if out.n > cap:
        keep = np.linspace(0, out.n - 1, cap, dtype=np.int64)
        out = FusedPoints(out.pts[keep], out.cols[keep], out.foot[keep], out.static[keep])
    return out


def camera_from_record(rec: Mapping[str, Any], width: int, height: int, *, far_m: float = 400.0) -> CameraPose:
    """Pinhole pose from a ``cameras.json`` / ``state.jsonl`` camera record, rescaled to ``width`` x ``height``.

    Uses ``pos_m``, ``yaw_deg``, ``pitch_deg``, ``fov_v`` only (no roll in the rig).
    A train-only / GT camera id is refused.
    """
    cid = _check_cam_id(str(rec.get("id") or rec.get("cam_id") or ""))
    if rec.get("train_only") is True or cid.endswith(("_depth", "_sem")):
        raise ValueError(f"refusing train-only camera {cid!r}")
    return camera_from_angles(
        cid,
        tuple(float(v) for v in rec["pos_m"]),
        float(rec.get("yaw_deg") or 0.0),
        float(rec.get("pitch_deg") or 0.0),
        float(rec["fov_v"]),
        int(width),
        int(height),
        near_m=0.05,
        far_m=float(far_m),
    )


def focal_depth_scale(fy_eff_cam: float, fy_eff_ref: float) -> float:
    """Metric-depth correction for a net trained on one focal length (canonical-camera rule).

    An object of size S at depth Z spans f*S/Z pixels, so a net that learned
    depth for focal ``f_ref`` reads ``Z_ref = f_ref * S / px``; the same pixels
    under focal ``f_cam`` mean ``Z = Z_ref * f_cam / f_ref``.
    """
    a = float(fy_eff_cam)
    b = float(fy_eff_ref)
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 0.0 or b <= 0.0:
        raise ValueError("effective focal lengths must be finite and > 0")
    return a / b


def _resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Pixel-centre bilinear resize (NumPy; matches cv2.INTER_LINEAR sampling)."""
    h, w = img.shape
    ys = (np.arange(out_h) + 0.5) * (h / out_h) - 0.5
    xs = (np.arange(out_w) + 0.5) * (w / out_w) - 0.5
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    a = img[y0][:, x0]
    b = img[y0][:, x1]
    c = img[y1][:, x0]
    d = img[y1][:, x1]
    return (a * (1 - wy) * (1 - wx) + b * (1 - wy) * wx + c * wy * (1 - wx) + d * wy * wx).astype(np.float64)


def _cell_means(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Mean of ``img`` over each of the h x w low-res cells (cell = pixel-centre bins)."""
    H, W = img.shape
    ry = np.minimum((np.arange(H) * h) // H, h - 1)
    rx = np.minimum((np.arange(W) * w) // W, w - 1)
    lab = ry[:, None] * w + rx[None, :]
    s = np.bincount(lab.ravel(), weights=img.ravel().astype(np.float64), minlength=h * w)
    n = np.bincount(lab.ravel(), minlength=h * w)
    return (s / np.maximum(n, 1)).reshape(h, w)


def joint_bilateral_upsample(lo: np.ndarray, guide: np.ndarray, *, sigma_s: float = 0.6, sigma_r: float = 0.08) -> np.ndarray:
    """Joint bilateral upsampling (Kopf et al. 2007) of ``lo`` to ``guide``'s size.

    Each output pixel averages the 3x3 nearest low-res cells, weighted by
    distance to the cell centre and by how close the pixel's guide value is to
    the cell's mean guide value, so depth edges snap to image edges.
    """
    lo = np.asarray(lo, np.float64)
    g = np.asarray(guide, np.float64)
    h, w = lo.shape
    H, W = g.shape
    gm = _cell_means(g, h, w)
    v = (np.arange(H) + 0.5) * (h / H) - 0.5
    u = (np.arange(W) + 0.5) * (w / W) - 0.5
    jc = np.clip(np.round(v).astype(np.int64), 0, h - 1)
    ic = np.clip(np.round(u).astype(np.int64), 0, w - 1)
    num = np.zeros((H, W))
    den = np.zeros((H, W))
    for dj in (-1, 0, 1):
        j = np.clip(jc + dj, 0, h - 1)
        wy = np.exp(-((v - j) ** 2) / (2 * sigma_s**2))
        for di in (-1, 0, 1):
            i = np.clip(ic + di, 0, w - 1)
            wx = np.exp(-((u - i) ** 2) / (2 * sigma_s**2))
            cell = lo[j][:, i]
            gcell = gm[j][:, i]
            wr = np.exp(-((g - gcell) ** 2) / (2 * sigma_r**2))
            wt = wy[:, None] * wx[None, :] * wr + 1e-12
            num += wt * cell
            den += wt
    return num / den


def upsample_depth(
    depth_lo: np.ndarray,
    out_h: int,
    out_w: int,
    guide_gray: np.ndarray | None = None,
    *,
    min_m: float = 0.3,
    sigma_s: float = 0.6,
    sigma_r: float = 0.08,
) -> np.ndarray:
    """Low-res metric depth -> (out_h, out_w), interpolated in inverse depth, optionally edge-aware.

    1/Z is linear in the image for any plane (a road up to the horizon), so
    interpolating inverse depth keeps planes flat. With ``guide_gray`` (the
    camera's own image, 0..1, output size) joint bilateral upsampling snaps
    depth edges to image edges.
    """
    d = np.asarray(depth_lo, dtype=np.float64)
    if d.ndim != 2 or d.size == 0:
        raise ValueError("depth_lo must be a non-empty 2-D array")
    inv = 1.0 / np.maximum(np.where(np.isfinite(d), d, 1e6), min_m)
    if guide_gray is not None:
        g = np.asarray(guide_gray, dtype=np.float64)
        if g.shape != (out_h, out_w):
            raise ValueError(f"guide shape {g.shape} != {(out_h, out_w)}")
        up = joint_bilateral_upsample(inv, g, sigma_s=sigma_s, sigma_r=sigma_r)
    else:
        up = _resize_bilinear(inv, int(out_h), int(out_w))
    up = np.maximum(up, 1e-6)
    return (1.0 / up).astype(np.float32)


def ego_image_mask(cam_id: str, height: int, width: int,
                   masks: Mapping[str, tuple[float, float, float]] = EGO_IMAGE_MASKS_SBR) -> np.ndarray:
    """True where ``cam_id`` sees the ego car's own body (never unprojected)."""
    out = np.zeros((int(height), int(width)), bool)
    spec = masks.get(str(cam_id))
    if spec is None:
        return out
    r0, c0, c1 = spec
    out[int(round(r0 * height)):, int(round(c0 * width)):int(round(c1 * width))] = True
    return out


def _nonplanar_jumps(inv: np.ndarray, axis: int, rel: float, ratio: float) -> np.ndarray:
    """Steps along ``axis`` that are big (relative) AND much bigger than the neighbouring steps.

    On any plane (a road, a wall) inverse depth is linear in the image, so its
    steps are constant and are never flagged; a silhouette is a lone big step.
    """
    D = np.abs(np.diff(inv, axis=axis))
    n = D.shape[axis]
    if n == 0:
        return np.zeros(D.shape, bool)
    prev = np.concatenate([np.take(D, [0], axis=axis), np.take(D, np.arange(n - 1), axis=axis)], axis=axis)
    nxt = np.concatenate([np.take(D, np.arange(1, n), axis=axis), np.take(D, [n - 1], axis=axis)], axis=axis)
    ref = np.minimum(prev, nxt)
    if n == 1:
        ref = np.zeros_like(D)
    a = np.take(inv, np.arange(n), axis=axis)
    b = np.take(inv, np.arange(1, n + 1), axis=axis)
    base = np.maximum(np.minimum(a, b), 1e-9)
    return (D / base > float(rel)) & (D > float(ratio) * ref)


def discontinuity_mask(depth: np.ndarray, rel: float = DEFAULT_CUT_REL, ratio: float = 3.0) -> np.ndarray:
    """True on both sides of a sharp, non-planar depth step (a silhouette) at full resolution."""
    d = np.asarray(depth, dtype=np.float64)
    inv = 1.0 / np.maximum(d, 1e-6)
    cut = np.zeros(d.shape, bool)
    jy = _nonplanar_jumps(inv, 0, rel, ratio)
    cut[:-1] |= jy
    cut[1:] |= jy
    jx = _nonplanar_jumps(inv, 1, rel, ratio)
    cut[:, :-1] |= jx
    cut[:, 1:] |= jx
    return cut


def sheet_mask(depth_hi: np.ndarray, depth_lo: np.ndarray, *, rel: float = 0.25, ratio: float = 2.0,
               keep_frac: float = 0.2) -> np.ndarray:
    """Cut the stretched sheet that upsampling draws between two low-res cells across a silhouette.

    For every low-res neighbour pair with a non-planar jump, high-res pixels in
    the span between the two cell centres whose inverse depth is *between* the
    two cells' values (more than ``keep_frac`` of the gap away from both) are
    cut. Pixels the edge-aware filter already snapped to either side are kept.
    """
    hi = np.asarray(depth_hi, np.float64)
    lo = np.asarray(depth_lo, np.float64)
    H, W = hi.shape
    h, w = lo.shape
    inv_hi = 1.0 / np.maximum(hi, 1e-6)
    inv_lo = 1.0 / np.maximum(lo, 1e-6)
    sy, sx = H / h, W / w
    out = np.zeros((H, W), bool)
    for axis in (0, 1):
        jumps = _nonplanar_jumps(inv_lo, axis, rel, ratio)
        for j, i in zip(*np.nonzero(jumps)):
            if axis == 0:
                a, b = inv_lo[j, i], inv_lo[j + 1, i]
                r0, r1 = int(math.floor((j + 0.5) * sy)), int(math.ceil((j + 1.5) * sy))
                c0, c1 = int(math.floor(i * sx)), int(math.ceil((i + 1) * sx))
            else:
                a, b = inv_lo[j, i], inv_lo[j, i + 1]
                r0, r1 = int(math.floor(j * sy)), int(math.ceil((j + 1) * sy))
                c0, c1 = int(math.floor((i + 0.5) * sx)), int(math.ceil((i + 1.5) * sx))
            lo_v, hi_v = min(a, b), max(a, b)
            gap = hi_v - lo_v
            blk = inv_hi[r0:r1, c0:c1]
            mid = (blk > lo_v + keep_frac * gap) & (blk < hi_v - keep_frac * gap)
            out[r0:r1, c0:c1] |= mid
    return out


def apply_backdrop(
    cam: CameraPose,
    depth: np.ndarray,
    far_mask: np.ndarray,
    *,
    far_m: float = DEFAULT_FAR_M,
    dome_m: float = DEFAULT_DOME_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Put sky / beyond-``far_m`` pixels on a dome of radius ``dome_m`` around the camera.

    Returns (planar depth with backdrop applied, backdrop mask). Planar depth
    of a ray at radius R is R * cos(angle to the optical axis).
    """
    d = np.asarray(depth, dtype=np.float64).copy()
    h, w = d.shape
    ys, xs = np.mgrid[0:h, 0:w]
    xn = (xs + 0.5 - cam.cx) / cam.fx
    yn = (ys + 0.5 - cam.cy) / cam.fy
    cosang = 1.0 / np.sqrt(1.0 + xn * xn + yn * yn)
    back = np.asarray(far_mask, bool) | ~np.isfinite(d) | (d > float(far_m))
    d[back] = float(dome_m) * cosang[back]
    return d.astype(np.float32), back


def unproject_camera(
    cam: CameraPose,
    rgb: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    backdrop: np.ndarray | None = None,
    foot_max_ratio: float = 6.0,
    ego_box: tuple[float, float, float, float, float] | None = None,
) -> FusedPoints:
    """Unproject a camera's RGB + planar depth into ego-frame points with per-point footprint.

    ``ego_box`` = (half_width, y_min, y_max, z_min, z_max) in the ego frame: points inside it
    (the car's own hood / body seen by a camera) are dropped.
    """
    _validate_pose(cam)
    rgb = np.asarray(rgb)
    d = np.asarray(depth, dtype=np.float64)
    h, w = d.shape
    if rgb.shape[:2] != (h, w) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb {rgb.shape} does not match depth {d.shape}")
    if (h, w) != (cam.height, cam.width):
        raise ValueError(f"depth {d.shape} does not match camera {(cam.height, cam.width)}")
    ok = np.isfinite(d) & (d > 0.0)
    if valid is not None:
        ok &= np.asarray(valid, bool)
    ys, xs = np.nonzero(ok)
    if ys.size == 0:
        return empty_points()
    ds = d[ys, xs]
    gy, gx = np.mgrid[0:h, 0:w]
    dsafe = np.where(np.isfinite(d) & (d > 0), d, 0.0)
    grid = np.stack([((gx + 0.5 - cam.cx) / cam.fx) * dsafe, -((gy + 0.5 - cam.cy) / cam.fy) * dsafe, dsafe], -1)
    # Footprint = spacing to the right / lower neighbour in 3-D (a ground pixel seen at a grazing
    # angle covers much more than d/f along the road). Clamped to [d/f, foot_max_ratio * d/f].
    base = dsafe / cam.fy
    sp = np.zeros((h, w))
    if w > 1:
        sx = np.linalg.norm(np.diff(grid, axis=1), axis=-1)
        sp[:, :-1] = np.maximum(sp[:, :-1], sx)
    if h > 1:
        sy = np.linalg.norm(np.diff(grid, axis=0), axis=-1)
        sp[:-1] = np.maximum(sp[:-1], sy)
    foot_full = np.clip(sp, base, float(foot_max_ratio) * base)
    xc = ((xs + 0.5 - cam.cx) / cam.fx) * ds
    yc = -((ys + 0.5 - cam.cy) / cam.fy) * ds
    pts = np.asarray(cam.pos)[None, :] + np.stack([xc, yc, ds], -1) @ cam.rotation_matrix()
    foot = foot_full[ys, xs]
    static = pts[:, 2] <= STATIC_MAX_HEIGHT_M
    if backdrop is not None:
        static &= ~np.asarray(backdrop, bool)[ys, xs]
    cols = rgb[ys, xs].astype(np.uint8)
    if ego_box is not None:
        hw, y0, y1, z0, z1 = (float(v) for v in ego_box)
        inside = (np.abs(pts[:, 0]) <= hw) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1) & (pts[:, 2] >= z0) & (pts[:, 2] <= z1)
        keep = ~inside
        pts, cols, foot, static = pts[keep], cols[keep], foot[keep], static[keep]
    return FusedPoints(pts.astype(np.float32), cols, foot.astype(np.float32), static)


# ---------------------------------------------------------------- ego motion


def phase_correlation_dx(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """Horizontal shift (px) of ``gray`` relative to ``prev_gray`` (positive = content moved right)."""
    a = np.asarray(prev_gray, np.float64)
    b = np.asarray(gray, np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("phase correlation needs two equal 2-D images")
    win = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    fa = np.fft.fft2((a - a.mean()) * win)
    fb = np.fft.fft2((b - b.mean()) * win)
    r = fb * np.conj(fa)
    r /= np.maximum(np.abs(r), 1e-12)
    c = np.fft.ifft2(r).real
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    w = c.shape[1]
    # sub-pixel parabola on x
    l = c[iy, (ix - 1) % w]
    m = c[iy, ix]
    rr = c[iy, (ix + 1) % w]
    den = l - 2 * m + rr
    sub = 0.5 * (l - rr) / den if abs(den) > 1e-12 else 0.0
    dx = ix + sub
    if dx > w / 2:
        dx -= w
    return float(dx)


def camera_yaw_delta(prev_gray: np.ndarray, gray: np.ndarray, fx: float) -> float:
    """Yaw change (rad, + = left / CCW about +Z) from the far-field image shift of a forward camera."""
    dx = phase_correlation_dx(prev_gray, gray)
    return float(math.atan2(dx, float(fx)))


@dataclass
class EgoOdometry:
    """Planar dead reckoning from wheel speed + camera yaw. World frame = ego frame at t0.

    ``step`` takes ``dyaw_rad`` in the ego convention (left / CCW about +Z is
    positive). BeamNG ``imu.yaw_rate`` has the opposite sign, so anyone wiring
    IMU into ``step`` later must flip it and add a test for that flip. This
    class does not read the IMU.
    """

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0  # rad, CCW, 0 = +Y forward

    def step(self, speed_mps: float, dt_s: float, dyaw_rad: float) -> None:
        if not (math.isfinite(speed_mps) and math.isfinite(dt_s) and math.isfinite(dyaw_rad)) or dt_s < 0:
            raise ValueError("non-finite odometry input")
        mid = self.heading + 0.5 * dyaw_rad
        dist = float(speed_mps) * float(dt_s)
        self.x += -math.sin(mid) * dist
        self.y += math.cos(mid) * dist
        self.heading += dyaw_rad

    def pose(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.heading)


def _rot(h: float) -> np.ndarray:
    c, s = math.cos(h), math.sin(h)
    return np.array([[c, -s], [s, c]])


def transform_ego_points(pts: np.ndarray, pose_from: tuple[float, float, float], pose_to: tuple[float, float, float]) -> np.ndarray:
    """Move ego-frame points captured at ``pose_from`` into the ego frame at ``pose_to`` (z unchanged)."""
    p = np.asarray(pts, np.float64)
    if p.size == 0:
        return p.astype(np.float32).reshape(0, 3)
    xy_w = p[:, :2] @ _rot(pose_from[2]).T + np.array(pose_from[:2])
    xy_t = (xy_w - np.array(pose_to[:2])) @ _rot(pose_to[2])
    return np.concatenate([xy_t, p[:, 2:3]], 1).astype(np.float32)


def accumulate(current: FusedPoints, history: Sequence[tuple[FusedPoints, tuple[float, float, float]]],
               pose_now: tuple[float, float, float]) -> FusedPoints:
    """Current points plus the *static* points of past frames moved into the current ego frame.

    Past non-static points (anything above ``STATIC_MAX_HEIGHT_M`` or on the
    backdrop) are dropped, so a car that drove away never leaves a ghost.
    """
    parts = [current]
    for past, pose in history:
        if not past.n:
            continue
        m = past.static
        if not np.any(m):
            continue
        parts.append(FusedPoints(transform_ego_points(past.pts[m], pose, pose_now), past.cols[m], past.foot[m], past.static[m]))
    return concat_points(parts)


# ---------------------------------------------------------------- render


def render_splats(
    pts: FusedPoints,
    chase: CameraPose,
    *,
    supersample: int = 2,
    splat_scale: float = 1.2,
    max_radius_px: int = MAX_SPLAT_RADIUS_PX,
) -> SplatRender:
    """Z-buffered square splats sized by each point's world footprint, supersampled then box-filtered.

    Deterministic: the nearest point wins each pixel; ties go to the lower
    point index. Unpainted pixels stay ``UNKNOWN_RGB`` with depth 0.
    """
    _validate_pose(chase)
    ss = int(supersample)
    if ss < 1 or ss > 4:
        raise ValueError("supersample must be 1..4")
    W, H = chase.width * ss, chase.height * ss
    fx, fy, cx, cy = chase.fx * ss, chase.fy * ss, chase.cx * ss, chase.cy * ss
    rgb = np.empty((H, W, 3), np.uint8)
    rgb[:] = UNKNOWN_RGB
    depth = np.zeros((H, W), np.float32)
    obs = np.zeros((H, W), bool)
    n_used = 0
    if pts.n:
        rel = pts.pts.astype(np.float64) - np.asarray(chase.pos)[None, :]
        cp = rel @ chase.rotation_matrix().T
        z = cp[:, 2]
        keep = np.isfinite(z) & (z > chase.near_m) & (z <= chase.far_m)
        cp, z = cp[keep], z[keep]
        cols, foot = pts.cols[keep], pts.foot[keep].astype(np.float64)
        u = cx + fx * cp[:, 0] / z
        v = cy - fy * cp[:, 1] / z
        r = np.clip(np.floor(splat_scale * 0.5 * fy * foot / z + 0.5), 0, max_radius_px * ss).astype(np.int64)
        px = np.floor(u).astype(np.int64)
        py = np.floor(v).astype(np.int64)
        near = (px >= -r) & (px < W + r) & (py >= -r) & (py < H + r)
        px, py, z, cols, r = px[near], py[near], z[near], cols[near], r[near]
        ids = np.arange(px.size, dtype=np.int64)
        n_used = int(px.size)
        # Z-buffer with one packed int64 key per splat pixel: (depth in mm) << 24 | point index.
        # np.minimum.at keeps the nearest point per pixel; ties go to the lower index (deterministic).
        if px.size >= (1 << 24):
            raise ValueError("too many points for the packed z-buffer key")
        zkey = np.clip(np.round(z * 1000.0), 0, (1 << 38) - 1).astype(np.int64) << 24 | ids
        zbuf = np.full(H * W, np.iinfo(np.int64).max, np.int64)
        for rad in np.unique(r):
            sel = np.nonzero(r == rad)[0]
            offs = np.arange(-rad, rad + 1)
            oy, ox = np.meshgrid(offs, offs, indexing="ij")
            oy, ox = oy.ravel(), ox.ravel()
            # chunk to bound memory
            step = max(1, 4_000_000 // oy.size)
            for c0 in range(0, sel.size, step):
                ss_ = sel[c0 : c0 + step]
                yy = (py[ss_][:, None] + oy[None, :]).ravel()
                xx = (px[ss_][:, None] + ox[None, :]).ravel()
                kk = np.repeat(zkey[ss_], oy.size)
                ins = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
                np.minimum.at(zbuf, yy[ins] * W + xx[ins], kk[ins])
        hit = np.nonzero(zbuf != np.iinfo(np.int64).max)[0]
        if hit.size:
            win = (zbuf[hit] & ((1 << 24) - 1)).astype(np.int64)
            rgb.reshape(-1, 3)[hit] = cols[win]
            depth.reshape(-1)[hit] = z[win].astype(np.float32)
            obs.reshape(-1)[hit] = True
    if ss > 1:
        h, w = chase.height, chase.width
        o = obs.reshape(h, ss, w, ss)
        cnt = o.sum((1, 3))
        acc = (rgb.astype(np.float64) * obs[..., None]).reshape(h, ss, w, ss, 3).sum((1, 3))
        unk = np.asarray(UNKNOWN_RGB, np.float64)
        frac = (cnt / float(ss * ss))[..., None]
        mean = np.where(cnt[..., None] > 0, acc / np.maximum(cnt[..., None], 1), unk)
        rgb_out = np.clip(np.round(frac * mean + (1 - frac) * unk), 0, 255).astype(np.uint8)
        dsum = (depth * obs).reshape(h, ss, w, ss).sum((1, 3))
        depth_out = np.where(cnt > 0, dsum / np.maximum(cnt, 1), 0.0).astype(np.float32)
        obs_out = cnt * 2 >= ss * ss
        rgb_out[cnt == 0] = UNKNOWN_RGB
        return SplatRender(rgb_out, depth_out, obs_out, n_used)
    return SplatRender(rgb, depth, obs, n_used)


def project_wire_box(chase: CameraPose, half_w: float = 0.95, y0: float = -2.35, y1: float = 2.3, h: float = 1.45) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Ego-car outline segments in chase pixels (a vector overlay, not camera data)."""
    c = [(x, y, z) for x in (-half_w, half_w) for y in (y0, y1) for z in (0.0, h)]
    P = np.asarray(c) - np.asarray(chase.pos)
    cp = P @ chase.rotation_matrix().T
    if np.any(cp[:, 2] <= chase.near_m):
        return []
    u = chase.cx + chase.fx * cp[:, 0] / cp[:, 2]
    v = chase.cy - chase.fy * cp[:, 1] / cp[:, 2]
    pts = [(int(round(a)), int(round(b))) for a, b in zip(u, v)]
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
    return [(pts[i], pts[j]) for i, j in edges]
