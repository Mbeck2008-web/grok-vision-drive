"""Ray-cast scenes for the BEV LSS visibility gate (G5): oriented boxes on a tilted ground plane.

Shared by ``scripts/test_bev_lss.py`` (fixed counterexamples, offline suite) and
``scripts/prop_bev_lss_gate.py`` (hypothesis property tests, dev only). numpy + torch only.

Scene: ground plane ``z = grade * y + z0`` in the ego frame (a grade is also ego pitch on a flat
road; ``z0`` is the road height under the ego origin), plus boxes ``(cx, cy, yaw_deg, length,
thickness, height)`` standing on the plane at their centre. ``length`` runs along the box's x
axis before the yaw (a wall across the road at yaw 0). The depth head is replaced by a fixed one
that puts ``p_true`` on the bin of each feature pixel's true z-depth and spreads the rest evenly.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from python.train import bev_lss as bl

Box = tuple  # (cx, cy, yaw_deg, length, thickness, height)


def box_frame(box: Box, grade: float, z0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centre xyz of the base, rotation box->ego 3x3, half extents) of an upright box on the plane."""
    cx, cy, yaw, length, thick, height = (float(v) for v in box)
    base = grade * cy + z0
    a = math.radians(yaw)
    R = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    centre = np.array([cx, cy, base + 0.5 * height])
    half = np.array([0.5 * length, 0.5 * thick, 0.5 * height])
    return centre, R, half


def segment_hits_box(o: np.ndarray, d: np.ndarray, box: Box, grade: float, z0: float, shrink_xy: float = 0.0, shrink_top: float = 0.0, t_max: float = 1.0, shrink_thick: float | None = None) -> np.ndarray:
    """For rays ``o + t d`` (``o`` 3 or ...x3, ``d`` ...x3): first entry t in (0, t_max) of the box, else inf.

    ``shrink_xy`` insets the four side faces, ``shrink_top`` lowers the top (a "robustly inside" box).
    """
    centre, R, half = box_frame(box, grade, z0)
    half = half.copy()
    half[0] -= shrink_xy
    half[1] -= shrink_xy if shrink_thick is None else shrink_thick
    lo_z = -half[2]
    hi_z = half[2] - shrink_top
    o = np.broadcast_to(np.asarray(o, dtype=np.float64), np.shape(d))
    lo_l = (o - centre) @ R  # ego -> box frame (R orthonormal: v @ R == R.T v)
    dl = np.asarray(d, dtype=np.float64) @ R
    lo = np.array([-half[0], -half[1], lo_z])
    hi = np.array([half[0], half[1], hi_z])
    if np.any(hi <= lo):
        return np.full(dl.shape[:-1], np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        ta = (lo - lo_l) / dl
        tb = (hi - lo_l) / dl
    t0 = np.minimum(ta, tb)
    t1 = np.maximum(ta, tb)
    par = np.abs(dl) < 1e-12
    inside = (lo_l >= lo) & (lo_l <= hi)
    t0 = np.where(par, np.where(inside, -np.inf, np.inf), t0)
    t1 = np.where(par, np.where(inside, np.inf, -np.inf), t1)
    tn = t0.max(axis=-1)
    tf = t1.min(axis=-1)
    hit = (tf >= tn) & (tf > 0) & (tn < t_max)
    return np.where(hit, np.maximum(tn, 0.0), np.inf)


def ray_dirs(g) -> np.ndarray:
    """24 x 32 x 3 ego displacement per metre of z-depth, one per feature pixel (bin-centre rays)."""
    K, T = g.K.astype(np.float64), g.T.astype(np.float64)
    u = (np.arange(32) + 0.5) * bl.FEAT_STRIDE
    v = (np.arange(24) + 0.5) * bl.FEAT_STRIDE
    uu, vv = np.meshgrid(u, v, indexing="xy")
    rc = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)
    return rc @ T[:3, :3].T


def scene_zdepth(g, boxes: Sequence[Box], grade: float = 0.0, z0: float = 0.0) -> np.ndarray:
    """24 x 32 true z-depth of the first surface per feature pixel (60 = nothing within range)."""
    o = g.T[:3, 3].astype(np.float64)
    d = ray_dirs(g)
    den = d[..., 2] - grade * d[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = (grade * o[1] + z0 - o[2]) / den
    t = np.where((np.abs(den) > 1e-12) & (tg > 0), tg, np.inf)
    for b in boxes:
        t = np.minimum(t, segment_hits_box(o, d, b, grade, z0, t_max=np.inf))
    return np.where(np.isfinite(t), t, 60.0)


def depth_prob(geoms, boxes: Sequence[Box], grade: float = 0.0, z0: float = 0.0, p_true: float = 0.95) -> torch.Tensor:
    """N x D x 24 x 32 fixed depth head for the scene (nearest log bin; beyond DEPTH_MAX_M -> last bin)."""
    log_c = np.log(bl.depth_bin_centers())
    P = torch.full((len(geoms), bl.DEPTH_BINS, 24, 32), (1.0 - p_true) / (bl.DEPTH_BINS - 1))
    for n, g in enumerate(geoms):
        z = scene_zdepth(g, boxes, grade, z0)
        k = np.argmin(np.abs(log_c[None, None] - np.log(np.clip(z, 1.0, None))[..., None]), -1)
        k = np.where(z > bl.DEPTH_MAX_M, bl.DEPTH_BINS - 1, k)
        P[n].scatter_(0, torch.from_numpy(k)[None].long(), p_true)
    return P


class FixedDepth(torch.nn.Module):
    """Spatial stand-in with a per-camera N x D x h x w depth distribution and unit context."""

    def __init__(self, P: torch.Tensor) -> None:
        super().__init__()
        self.P = P

    def forward(self, f4, f8, f16, cam):
        n, _, h, w = f8.shape
        return self.P[:n].contiguous(), torch.ones(n, bl.CTX_CH, h, w)


def run_scene(geoms, boxes: Sequence[Box], grade: float = 0.0, z0: float = 0.0, p_true: float = 0.95, head=None) -> dict:
    """Head output dict for the scene (B = 1)."""
    rig = bl.rig_tensors(geoms)
    head = (head or bl.BevLssHead()).eval()
    head.spatial = FixedDepth(depth_prob(geoms, boxes, grade, z0, p_true))
    nc = len(geoms)
    taps = (torch.zeros(1, nc, 64, 48, 64), torch.zeros(1, nc, 128, 24, 32), torch.zeros(1, nc, 128, 12, 16))
    with torch.no_grad():
        return head(*taps, rig["K"], rig["T"], pixel_valid=rig["pixel_valid"])


def cell_centres() -> tuple[np.ndarray, np.ndarray]:
    """(x, y) of every BEV cell centre, BEV_N x BEV_N each (row 0 = +32 m ahead)."""
    i = np.arange(bl.BEV_N) + 0.5
    ys = bl.BEV_HALF_M - bl.BEV_CELL_M * i
    xs = -bl.BEV_HALF_M + bl.BEV_CELL_M * i
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return X, Y


def in_footprint(box: Box, X: np.ndarray, Y: np.ndarray, pad: float = 0.0) -> np.ndarray:
    cx, cy, yaw, length, thick, _ = (float(v) for v in box)
    a = math.radians(yaw)
    lx = (X - cx) * math.cos(a) + (Y - cy) * math.sin(a)
    ly = -(X - cx) * math.sin(a) + (Y - cy) * math.cos(a)
    return (np.abs(lx) <= 0.5 * length + pad) & (np.abs(ly) <= 0.5 * thick + pad)


def ground_truth(geoms, boxes: Sequence[Box], grade: float = 0.0, z0: float = 0.0, below_top: float = 0.02, inset: float = 0.02, foot_pad: float = 0.75, inset_thick: float | None = None, resolve: bool = False) -> dict:
    """Per cell: ``hidden`` (in view of >= 1 camera, and every in-view camera's sight line to the
    ground centre passes through some box shrunk by ``inset`` on the sides and ``below_top`` at the
    top; ``inset_thick`` overrides the inset of the two long faces), ``inview`` (count of cameras whose content rect contains the ground centre) and ``near_box``
    (centre within ``foot_pad`` of a footprint, where 0.5 m cells straddle a face; excluded from hidden).
    With ``resolve``, hidden cells that some in-view camera cannot resolve (RESOLVE_MARGIN_M) move to
    ``sub_pitch``.
    """
    X, Y = cell_centres()
    G = np.stack([X, Y, grade * Y + z0], -1)  # N x N x 3
    inview = np.zeros(X.shape, dtype=np.int64)
    unresolved = np.zeros(X.shape, dtype=bool)
    blocked_all = np.ones(X.shape, dtype=bool)
    for g in geoms:
        Ti = np.linalg.inv(g.T.astype(np.float64))
        cam = G @ Ti[:3, :3].T + Ti[:3, 3]
        zc = cam[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = g.K[0, 0] * cam[..., 0] / zc + g.K[0, 2]
            v = g.K[1, 1] * cam[..., 1] / zc + g.K[1, 2]
        nw, nh, x0, y0 = g.content
        vis = (zc > 0) & (u >= x0) & (u < x0 + nw) & (v >= y0) & (v < y0 + nh)
        o = g.T[:3, 3].astype(np.float64)
        t = np.full(X.shape, np.inf)
        res_ok = np.zeros(X.shape, dtype=bool)
        pitch = bl.FEAT_STRIDE / float(g.K[1, 1])
        for b in boxes:
            tb = segment_hits_box(o, G - o, b, grade, z0, shrink_xy=inset, shrink_top=below_top, t_max=1.0, shrink_thick=inset_thick)
            first = tb < t
            res_ok = np.where(first, pitch * tb * zc <= float(b[5]) - RESOLVE_MARGIN_M, res_ok)
            t = np.minimum(t, tb)
        blk = np.isfinite(t)
        inview += vis
        blocked_all &= ~vis | blk
        unresolved |= vis & blk & ~res_ok
    near = np.zeros(X.shape, dtype=bool)
    for b in boxes:
        near |= in_footprint(b, X, Y, pad=foot_pad)
    hidden = (inview > 0) & blocked_all & ~near
    if resolve:
        # cells some in-view camera cannot resolve are reported apart ("sub_pitch"), not asserted
        return {"hidden": hidden & ~unresolved, "sub_pitch": hidden & unresolved, "inview": inview, "near_box": near}
    return {"hidden": hidden, "sub_pitch": np.zeros_like(hidden), "inview": inview, "near_box": near}


# Resolution clause of the stated invariant: a camera resolves an occluder when its vertical ray
# pitch (one feature row, stride / fy radians) times the z-depth at which the sight line enters the
# occluder is at most the occluder height minus RESOLVE_MARGIN_M, so at least one of its ray rows
# crosses the face between 0.3 m above the base and the top. 0.3 m = the 0.15 m occluder threshold
# on a flat road + Skeptic's 0.1 m under-top tolerance + 0.05 m of ground-estimate error. A wall
# that falls between two rows of every camera that sees the cell leaves a depth map identical to
# open road (scripts/prop_bev_lss_gate.py --limit-demo); no gate can tell those cells apart.
RESOLVE_MARGIN_M = 0.3


def labels(out: dict) -> dict:
    lg = out["logits"][0]
    return {
        "free": (lg[bl.CH_FREE] != -bl.FORCE_LOGIT).numpy(),
        "seen": (lg[bl.CH_UNKNOWN] != bl.FORCE_LOGIT).numpy(),
        "through": (out["through_mass"][0, 0] > 0).numpy(),
        "clear": (out["clear_mass"][0, 0] > 0).numpy(),
        "finite": bool(torch.isfinite(lg).all()),
    }


def leaks(geoms, boxes, grade=0.0, z0=0.0, p_true=0.95, head=None, **gt_kw) -> dict:
    """Counts of hidden cells that the gate opens (free-eligible / seen / through / clear)."""
    out = run_scene(geoms, boxes, grade, z0, p_true, head)
    lab = labels(out)
    gt = ground_truth(geoms, boxes, grade, z0, **gt_kw)
    h = gt["hidden"]
    return {
        "hidden": int(h.sum()),
        "free": int((lab["free"] & h).sum()),
        "seen": int((lab["seen"] & h).sum()),
        "through": int((lab["through"] & h).sum()),
        "clear": int((lab["clear"] & h).sum()),
        "free_total": int(lab["free"].sum()),
        "finite": lab["finite"],
        "free_cells": np.argwhere(lab["free"] & h),
        "sub_pitch": int(gt["sub_pitch"].sum()),
        "sub_pitch_free": int((lab["free"] & gt["sub_pitch"]).sum()),
        "out": out,
    }
