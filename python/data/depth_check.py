"""Depth-plane geometry: planar z vs ray length, back-projection, ground fit.

The collector stores each training-only depth plane as uint16 cm
(``raw_frames.DEPTH_KIND``). This module holds the conversions a trainer
needs and the offline check that tells the two readings apart on real
frames: back-project the road pixels under each reading and fit a plane.
The right reading gives a flat road (sub-cm residual, almost every road
pixel an inlier); the wrong one bends it. Nothing here scales the depth.

Camera frame here: x right, y down, z forward (OpenCV). Pixel ``(u, v)``
has its centre at ``(u + 0.5, v + 0.5)``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

KINDS = ("planar_z", "ray")
GROUND_CLASSES = ("STREET", "ASPHALT")
# Plane fit: RANSAC inlier band, then the reported inlier band (metres).
RANSAC_BAND_M = 0.03
INLIER_BAND_M = 0.05
# Only road pixels nearer than this and below this image fraction are fitted:
# far road is a few pixels tall and its cm quantisation dominates.
FIT_MAX_M = 60.0
FIT_MIN_M = 1.0
FIT_ROW_FRAC = 0.55
# Verdict: the better reading must beat the other by this rms factor.
VERDICT_RATIO = 3.0


def _rays(intr: Mapping[str, Any], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = int(shape[0]), int(shape[1])
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    xn = (u + 0.5 - float(intr["cx"])) / float(intr["fx"])
    yn = (v + 0.5 - float(intr["cy"])) / float(intr["fy"])
    return xn, yn


def depth_z_to_ray(depth_m: np.ndarray, intr: Mapping[str, Any]) -> np.ndarray:
    """Ray length through each pixel from planar z (same units in and out)."""
    z = np.asarray(depth_m, dtype=np.float64)
    xn, yn = _rays(intr, z.shape[:2])
    return z * np.sqrt(1.0 + xn * xn + yn * yn)


def backproject(depth_m: np.ndarray, intr: Mapping[str, Any], kind: str) -> np.ndarray:
    """(H, W, 3) camera-frame points from a depth plane read as ``kind``."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    d = np.asarray(depth_m, dtype=np.float64)
    xn, yn = _rays(intr, d.shape[:2])
    z = d if kind == "planar_z" else d / np.sqrt(1.0 + xn * xn + yn * yn)
    return np.stack([xn * z, yn * z, z], axis=-1)


def synthetic_ground_depth(
    intr: Mapping[str, Any], resolution: Any, *, height_m: float, pitch_deg: float
) -> np.ndarray:
    """Planar z (m) of a flat ground seen by a camera ``height_m`` up, pitched down. inf above the horizon."""
    w, h = int(resolution[0]), int(resolution[1])
    xn, yn = _rays(intr, (h, w))
    p = math.radians(float(pitch_deg))
    # Ground plane in camera coords: n . X = height, n = (0, cos p, sin p) for a camera pitched down by p.
    denom = yn * math.cos(p) + math.sin(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(denom > 1e-9, float(height_m) / denom, np.inf)
    return z


def fit_ground_plane(
    depth_cm: np.ndarray,
    mask: np.ndarray,
    intr: Mapping[str, Any],
    kind: str,
    *,
    seed: int = 0,
    iters: int = 400,
) -> dict[str, Any]:
    """RANSAC + least-squares plane through the masked pixels.

    Returns camera height above the plane (m), camera pitch relative to the
    plane (deg, positive = looking down), rms of the inlier residual (m),
    inlier fraction (within ``INLIER_BAND_M``), and the pixel count.
    """
    pts_all = backproject(np.asarray(depth_cm, dtype=np.float64) / 100.0, intr, kind)
    pts = pts_all[np.asarray(mask, dtype=bool)]
    n_px = int(pts.shape[0])
    if n_px < 50:
        return {"kind": kind, "n": n_px, "height_m": None, "pitch_deg": None, "rms_m": None, "inlier_frac": None}
    rng = np.random.default_rng(int(seed))
    best: tuple[int, np.ndarray, float] | None = None
    for _ in range(int(iters)):
        s = pts[rng.choice(n_px, 3, replace=False)]
        normal = np.cross(s[1] - s[0], s[2] - s[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-12:
            continue
        normal = normal / norm
        off = -float(normal @ s[0])
        count = int((np.abs(pts @ normal + off) < RANSAC_BAND_M).sum())
        if best is None or count > best[0]:
            best = (count, normal, off)
    assert best is not None
    _count, normal, off = best
    inl = np.abs(pts @ normal + off) < INLIER_BAND_M
    q = pts[inl]
    centre = q.mean(axis=0)
    _u, _s, vt = np.linalg.svd(q - centre, full_matrices=False)
    normal = vt[-1]
    if normal[1] < 0:  # ground normal points down the image (+y) from the camera
        normal = -normal
    off = -float(normal @ centre)
    res = q @ normal + off
    return {
        "kind": kind,
        "n": n_px,
        "height_m": round(abs(off), 6),
        "pitch_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, float(normal[2]))))), 4),
        "rms_m": round(float(res.std()), 6),
        "inlier_frac": round(float(inl.mean()), 6),
    }


def ground_mask(depth_cm: np.ndarray, sem: np.ndarray, classes: Mapping[str, Any]) -> np.ndarray:
    ids = [int(k) for k, v in dict(classes or {}).items() if str(v) in GROUND_CLASSES]
    d = np.asarray(depth_cm, dtype=np.float64) / 100.0
    rows = np.arange(d.shape[0])[:, None] >= d.shape[0] * FIT_ROW_FRAC
    return np.isin(np.asarray(sem), ids) & (d > FIT_MIN_M) & (d < FIT_MAX_M) & rows


def depth_kind_verdict(
    depth_cm: np.ndarray, sem: np.ndarray, classes: Mapping[str, Any], intr: Mapping[str, Any]
) -> dict[str, Any]:
    """Fit the road under both readings; the verdict is the flatter one by ``VERDICT_RATIO``."""
    mask = ground_mask(depth_cm, sem, classes)
    fits = {kind: fit_ground_plane(depth_cm, mask, intr, kind) for kind in KINDS}
    z, ray = fits["planar_z"]["rms_m"], fits["ray"]["rms_m"]
    verdict = "unclear"
    if z is not None and ray is not None:
        if ray > VERDICT_RATIO * max(z, 1e-9):
            verdict = "planar_z"
        elif z > VERDICT_RATIO * max(ray, 1e-9):
            verdict = "ray"
    return {"verdict": verdict, "ground_px": int(mask.sum()), **fits}
