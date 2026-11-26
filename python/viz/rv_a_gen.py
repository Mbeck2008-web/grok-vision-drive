"""Reality View step 5: approach A generator (SH0 Gaussians, CPU, offline).

Spec: ``docs/specs/reality_view_3dgs.md`` (Feasibility plan, step 5).

Predicted metric planar depth (H, W) + RGB (H, W, 3) uint8 + pinhole K ->
one SH0 Gaussian per valid pixel (mean, anisotropic scale, quaternion,
opacity, SH0 colour), capped at ``n_max`` (hard ceiling ``N_MAX_HARD``).
``render_sh0`` is a small deterministic CPU EWA splatter used for the
offline canned-pose render and honesty checks only; the live render target
remains ``gvd_viz`` (GL 3.3), not this function.

Rules:
* Offline / test-process only. Not imported by the supervisor, the live
  vision loop, or the safety path (heartbeat / AEB / engage).
* No new net, no RNG, no clock, no sim GT. NumPy + stdlib only.
* Unknown depth (NaN, +/-inf, <= 0, < near, > far, masked) never becomes a
  Gaussian; unrendered pixels are ``UNKNOWN_RGB`` with depth 0, never an
  opaque surface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

N_MAX_HARD: int = 300_000          # working cap from the spec (step 6), not raisable by callers
SH_C0: float = 0.28209479177387814  # Y_0^0
UNKNOWN_RGB: tuple[int, int, int] = (24, 20, 18)
DEFAULT_OPACITY: float = 0.95
THIN_RATIO: float = 0.1             # splat thickness along the camera axis vs footprint
MAX_RADIUS_PX: int = 16
OBSERVED_ALPHA: float = 0.5
MIN_ALPHA: float = 1.0 / 255.0
MAX_INPUT_PIXELS: int = 4_000_000


@dataclass(frozen=True)
class Sh0Gaussians:
    """N SH0 Gaussians in the output frame (camera frame unless cam_to_ego was given)."""

    means: np.ndarray    # (N, 3) float32
    scales: np.ndarray   # (N, 3) float32, std-dev in metres
    quats: np.ndarray    # (N, 4) float32, (w, x, y, z), unit
    opacity: np.ndarray  # (N,) float32 in (0, 1]
    sh0: np.ndarray      # (N, 3) float32, colour = 0.5 + SH_C0 * sh0

    @property
    def n(self) -> int:
        return int(self.means.shape[0])


@dataclass(frozen=True)
class ChaseCam:
    """Pinhole render camera: K (3x3), world->cam 4x4 (OpenCV axes), size, clip."""

    K: np.ndarray
    w2c: np.ndarray
    width: int
    height: int
    near_m: float = 0.1
    far_m: float = 200.0


@dataclass(frozen=True)
class RenderResult:
    rgb: np.ndarray       # (H, W, 3) uint8; UNKNOWN_RGB where not observed
    depth_m: np.ndarray   # (H, W) float32; 0 where not observed
    alpha: np.ndarray     # (H, W) float32 accumulated opacity
    observed: np.ndarray  # (H, W) bool, alpha >= OBSERVED_ALPHA


def _empty() -> Sh0Gaussians:
    z3 = np.zeros((0, 3), np.float32)
    return Sh0Gaussians(z3, z3.copy(), np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), z3.copy())


def _check_n_max(n_max: object) -> int:
    if isinstance(n_max, bool) or not isinstance(n_max, (int, np.integer)):
        raise TypeError(f"n_max must be an int, got {type(n_max).__name__}")
    n = int(n_max)
    if n < 1 or n > N_MAX_HARD:
        raise ValueError(f"n_max must be in [1, {N_MAX_HARD}], got {n}")
    return n


def _check_k(K: np.ndarray) -> np.ndarray:
    k = np.asarray(K, dtype=np.float64)
    if k.shape != (3, 3) or not np.all(np.isfinite(k)):
        raise ValueError("K must be a finite 3x3 matrix")
    if not (k[0, 0] > 0 and k[1, 1] > 0):
        raise ValueError("K focal lengths must be > 0")
    return k


def _rot_to_quat(r: np.ndarray) -> np.ndarray:
    """3x3 rotation -> unit (w, x, y, z), deterministic branch order."""
    t = float(np.trace(r))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    qa = np.asarray(q, dtype=np.float64)
    return qa / np.linalg.norm(qa)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(N, 4) (w, x, y, z) -> (N, 3, 3)."""
    q = q.astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], 1)


def generate_sh0(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    *,
    n_max: int,
    valid_mask: np.ndarray | None = None,
    near_m: float = 0.2,
    far_m: float = 80.0,
    cam_to_ego: np.ndarray | None = None,
    opacity: float = DEFAULT_OPACITY,
) -> Sh0Gaussians:
    """Unproject valid predicted depth x RGB to at most ``n_max`` SH0 Gaussians."""
    n_cap = _check_n_max(n_max)
    k = _check_k(K)
    d = np.asarray(depth_m)
    if d.ndim != 2 or d.size == 0 or d.size > MAX_INPUT_PIXELS:
        raise ValueError(f"depth must be (H, W) with <= {MAX_INPUT_PIXELS} px, got {d.shape}")
    h, w = d.shape
    c = np.asarray(rgb)
    if c.shape != (h, w, 3) or c.dtype != np.uint8:
        raise ValueError(f"rgb must be uint8 {(h, w, 3)}, got {c.dtype} {c.shape}")
    if not (math.isfinite(near_m) and math.isfinite(far_m) and 0.0 < near_m < far_m):
        raise ValueError("need 0 < near_m < far_m, both finite")
    if not (0.0 < opacity <= 1.0):
        raise ValueError("opacity must be in (0, 1]")

    d64 = d.astype(np.float64)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(d64) & (d64 >= near_m) & (d64 <= far_m)
    if valid_mask is not None:
        vm = np.asarray(valid_mask)
        if vm.shape != (h, w):
            raise ValueError("valid_mask must match depth shape")
        valid &= vm.astype(bool)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return _empty()

    # N cap: deterministic even pick over valid pixels (raster order); splats grow to keep coverage.
    vs, us = np.nonzero(valid)
    grow = 1.0
    if vs.size > n_cap:
        pick = np.unique(np.linspace(0, vs.size - 1, n_cap).round().astype(np.int64))
        grow = math.sqrt(vs.size / pick.size)
        vs, us = vs[pick], us[pick]
    assert vs.size <= n_cap

    z = d64[vs, us]
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    pts = np.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z], axis=1)
    sx = 0.5 * z / fx * grow
    sy = 0.5 * z / fy * grow
    sz = THIN_RATIO * np.minimum(sx, sy)
    scales = np.stack([sx, sy, sz], axis=1)

    r = np.eye(3)
    if cam_to_ego is not None:
        m = np.asarray(cam_to_ego, dtype=np.float64)
        if m.shape != (4, 4) or not np.all(np.isfinite(m)):
            raise ValueError("cam_to_ego must be a finite 4x4")
        r = m[:3, :3]
        if not np.allclose(r @ r.T, np.eye(3), atol=1e-6) or np.linalg.det(r) <= 0:
            raise ValueError("cam_to_ego rotation must be proper orthonormal")
        pts = pts @ r.T + m[:3, 3]
    quat = np.broadcast_to(_rot_to_quat(r), (vs.size, 4))

    col = c[vs, us].astype(np.float64) / 255.0
    out = Sh0Gaussians(
        means=pts.astype(np.float32),
        scales=scales.astype(np.float32),
        quats=np.ascontiguousarray(quat, dtype=np.float32),
        opacity=np.full((vs.size,), opacity, dtype=np.float32),
        sh0=((col - 0.5) / SH_C0).astype(np.float32),
    )
    if not all(np.all(np.isfinite(a)) for a in (out.means, out.scales, out.quats, out.sh0)):
        raise ValueError("non-finite Gaussian produced")  # unreachable with the mask above
    return out


def concat_capped(parts: Sequence[Sh0Gaussians], *, n_max: int) -> Sh0Gaussians:
    """Concatenate generations, keeping at most ``n_max`` (newest parts win, evenly thinned)."""
    n_cap = _check_n_max(n_max)
    parts = [p for p in parts if p.n]
    if not parts:
        return _empty()
    cat = {f: np.concatenate([getattr(p, f) for p in parts]) for f in ("means", "scales", "quats", "opacity", "sh0")}
    n = cat["means"].shape[0]
    if n > n_cap:
        pick = np.unique(np.linspace(n - 1, 0, n_cap).round().astype(np.int64))
        cat = {f: v[pick] for f, v in cat.items()}
    return Sh0Gaussians(**cat)


def look_at_w2c(eye: np.ndarray, target: np.ndarray, down: np.ndarray | None = None) -> np.ndarray:
    """OpenCV-axis world->cam 4x4 looking from eye to target (+y down)."""
    eye = np.asarray(eye, dtype=np.float64)
    f = np.asarray(target, dtype=np.float64) - eye
    f /= np.linalg.norm(f)
    dn = np.array([0.0, 1.0, 0.0]) if down is None else np.asarray(down, dtype=np.float64)
    x = np.cross(dn, f)
    x /= np.linalg.norm(x)
    y = np.cross(f, x)
    r = np.stack([x, y, f])
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = -r @ eye
    return m


def render_sh0(g: Sh0Gaussians, cam: ChaseCam) -> RenderResult:
    """Deterministic CPU EWA splat render (front-to-back). Offline checks only."""
    hh, ww = int(cam.height), int(cam.width)
    k = _check_k(cam.K)
    w2c = np.asarray(cam.w2c, dtype=np.float64)
    acc = np.zeros((hh, ww, 3))
    dep = np.zeros((hh, ww))
    trans = np.ones((hh, ww))
    if g.n:
        pc = g.means.astype(np.float64) @ w2c[:3, :3].T + w2c[:3, 3]
        keep = (pc[:, 2] > cam.near_m) & (pc[:, 2] < cam.far_m)
        idx = np.nonzero(keep)[0]
        pc = pc[idx]
        rg = _quat_to_rot(g.quats[idx])
        s2 = g.scales[idx].astype(np.float64) ** 2
        cov = np.einsum("nij,nj,nkj->nik", rg, s2, rg)
        rc = w2c[:3, :3]
        cov = np.einsum("ij,njk,lk->nil", rc, cov, rc)
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        jac = np.zeros((idx.size, 2, 3))
        jac[:, 0, 0] = fx / z
        jac[:, 0, 2] = -fx * x / z ** 2
        jac[:, 1, 1] = fy / z
        jac[:, 1, 2] = -fy * y / z ** 2
        c2 = np.einsum("nij,njk,nlk->nil", jac, cov, jac) + 0.3 * np.eye(2)
        mu_u = fx * x / z + cx
        mu_v = fy * y / z + cy
        colour = np.clip(0.5 + SH_C0 * g.sh0[idx].astype(np.float64), 0.0, 1.0)
        op = g.opacity[idx].astype(np.float64)
        order = np.argsort(z, kind="stable")
        for i in order:
            a, b, cc = c2[i, 0, 0], c2[i, 0, 1], c2[i, 1, 1]
            det = a * cc - b * b
            if not det > 0:
                continue
            lam = 0.5 * (a + cc) + math.sqrt(max(0.0, 0.25 * (a - cc) ** 2 + b * b))
            rad = min(MAX_RADIUS_PX, int(math.ceil(3.0 * math.sqrt(lam))))
            u0, v0 = mu_u[i], mu_v[i]
            ul, ur = max(0, int(math.floor(u0 - rad))), min(ww - 1, int(math.ceil(u0 + rad)))
            vt, vb = max(0, int(math.floor(v0 - rad))), min(hh - 1, int(math.ceil(v0 + rad)))
            if ul > ur or vt > vb:
                continue
            du = np.arange(ul, ur + 1) - u0
            dv = (np.arange(vt, vb + 1) - v0)[:, None]
            power = -0.5 * (cc * du * du - 2.0 * b * du * dv + a * dv * dv) / det
            al = np.minimum(0.99, op[i] * np.exp(power))
            al = np.where(al >= MIN_ALPHA, al, 0.0)
            t = trans[vt:vb + 1, ul:ur + 1]
            wgt = t * al
            acc[vt:vb + 1, ul:ur + 1] += wgt[..., None] * colour[i]
            dep[vt:vb + 1, ul:ur + 1] += wgt * z[i]
            trans[vt:vb + 1, ul:ur + 1] = t * (1.0 - al)
    alpha = 1.0 - trans
    observed = alpha >= OBSERVED_ALPHA
    safe = np.where(observed, alpha, 1.0)
    rgb = np.empty((hh, ww, 3), dtype=np.uint8)
    rgb[...] = np.asarray(UNKNOWN_RGB, dtype=np.uint8)
    rgb[observed] = np.clip(np.rint(acc[observed] / safe[observed][:, None] * 255.0), 0, 255).astype(np.uint8)
    depth = np.where(observed, dep / safe, 0.0).astype(np.float32)
    return RenderResult(rgb=rgb, depth_m=depth, alpha=np.where(observed, alpha, 0.0).astype(np.float32),
                        observed=observed)
