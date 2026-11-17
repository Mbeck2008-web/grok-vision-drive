"""Sparse occupancy ground truth. TRAINING ONLY (docs/specs/occ_sparse.md).

Inputs are the recorded ``<cam>_depth`` (uint16 cm, view-Z) and optional
``<cam>_sem`` (uint8 ids + ``sem_classes.json`` names) of any camera that
has them, plus traffic boxes from ``state.jsonl``. Nothing here is imported
by the runtime net (``python.train.occ_sparse``).

Rules (each one is a test in scripts/test_occ_sparse.py):

* Output is sparse: int16 voxel index lists, never a dense grid.
* A depth hit marks its voxel occupied. Voxels the ray crossed before the
  hit (minus one voxel of margin) are observed free. Everything else is
  unknown and is never written, so unknown can never train as free.
* Depth 0 and the clamp value 65535 are invalid: no hit, no free space.
* Occupied wins: a voxel that is both is occupied.
* Semantics: a palette name maps to a class only by exact list
  (``class_for_name``): drivable = ``DRIVABLE_CLASS_ALLOWLIST``; lane paint =
  ``LANE_LINE_MATERIALS`` (yellow when the name says yellow); crosswalk,
  stop line and curb lists are empty until a palette dump proves a name.
  Class 0 and every other name is ignore (255). No name is guessed.
* Vehicle boxes: with recorded length and width, the oriented footprint;
  without them, only a 0.8 m core disk around the box centre (inside any
  car), never a 4.5 x 1.8 m default. Class ``vehicle``, flow = speed x
  heading in the ego frame. Flow is NaN everywhere else.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from python.debug.painted_lanes import LANE_LINE_MATERIALS
from python.train.occ_sparse import (
    CLS_CROSSWALK, CLS_CURB, CLS_DRIVABLE, CLS_IGNORE, CLS_LANE_WHITE, CLS_LANE_YELLOW, CLS_STOP_LINE, CLS_VEHICLE,
    OCC_CLASSES, OccGrid, pixel_rays,
)
from python.train.sensor_aux import DRIVABLE_CLASS_ALLOWLIST

# Exact palette names per class. Empty = no proven source yet: that class gets
# no label (CLS_IGNORE) until a sem_classes.json dump shows the name.
# line_reflectors are raised markers, not paint, so they are not lane paint.
LANE_PAINT_NAMES = frozenset(n for n in LANE_LINE_MATERIALS if n != "line_reflectors")
CROSSWALK_NAMES: frozenset[str] = frozenset()
STOP_LINE_NAMES: frozenset[str] = frozenset()
CURB_NAMES: frozenset[str] = frozenset()
# A thin painted class wins its voxel at this share of the voxel's votes
# (0.2 m paint in a 0.4 m voxel never holds a majority against the road).
PAINT_VOTE_SHARE = 0.25
_THIN = (CLS_LANE_WHITE, CLS_LANE_YELLOW, CLS_CROSSWALK, CLS_STOP_LINE)

DEPTH_CM_MAX = 65535
VEHICLE_CORE_M = 0.8
VEHICLE_Z_M = (0.2, 1.4)
FREE_STEP_FRAC = 0.5
__all__ = [
    "CLS_DRIVABLE", "CLS_IGNORE", "CLS_VEHICLE", "DEPTH_CM_MAX", "VEHICLE_CORE_M",
    "SparseOccLabels", "build_occ_labels", "sample_queries", "occ_loss",
]


@dataclass
class SparseOccLabels:
    occ_idx: np.ndarray   # (M, 3) int16
    occ_cls: np.ndarray   # (M,) uint8: CLS_DRIVABLE, CLS_VEHICLE or CLS_IGNORE
    flow: np.ndarray      # (M, 2) float32 ego-frame m/s, NaN = no flow label
    free_idx: np.ndarray  # (F, 3) int16
    box: np.ndarray | None = None  # (M, 6) float32: dx, dy to recorded box centre, length, width, sin, cos yaw; NaN = none


def _empty() -> SparseOccLabels:
    return SparseOccLabels(np.zeros((0, 3), np.int16), np.zeros(0, np.uint8),
                           np.zeros((0, 2), np.float32), np.zeros((0, 3), np.int16))


def class_for_name(name: Any) -> int:
    """Occupancy class for one palette name; CLS_IGNORE unless the name is on a list."""
    raw = str(name or "")
    low = raw.strip().lower()
    if raw in DRIVABLE_CLASS_ALLOWLIST:
        return CLS_DRIVABLE
    if low in LANE_PAINT_NAMES:
        return CLS_LANE_YELLOW if "yellow" in low else CLS_LANE_WHITE
    if low in CROSSWALK_NAMES:
        return CLS_CROSSWALK
    if low in STOP_LINE_NAMES:
        return CLS_STOP_LINE
    if low in CURB_NAMES:
        return CLS_CURB
    return CLS_IGNORE


def _palette_lut(sem_names: Mapping[str, Any] | None) -> np.ndarray | None:
    """uint8 lut: sem id -> class. None when no name maps to a class."""
    lut = np.full(65536, CLS_IGNORE, np.uint8)
    hit = False
    for key, name in dict(sem_names or {}).items():
        try:
            cid = int(key)
        except (TypeError, ValueError):
            continue
        if cid <= 0 or cid > 65535:
            continue
        c = class_for_name(name)
        if c != CLS_IGNORE:
            lut[cid] = c
            hit = True
    return lut if hit else None


def _vote(lin: np.ndarray, cls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One class per voxel: thin paint at >= PAINT_VOTE_SHARE, else the majority (ignore votes count)."""
    if not len(lin):
        return np.zeros(0, np.int64), np.zeros(0, np.uint8)
    vox, inv = np.unique(lin, return_inverse=True)
    n_cls = len(OCC_CLASSES) + 1  # last column = ignore
    col = np.where(cls == CLS_IGNORE, n_cls - 1, cls).astype(np.int64)
    counts = np.zeros((len(vox), n_cls), np.int64)
    np.add.at(counts, (inv, col), 1)
    best = counts.argmax(1)
    total = counts.sum(1)
    thin = counts[:, list(_THIN)]
    t_best = thin.argmax(1)
    t_ok = thin[np.arange(len(vox)), t_best] >= PAINT_VOTE_SHARE * total
    best = np.where(t_ok, np.array(_THIN)[t_best], best)
    out = np.where(best == n_cls - 1, CLS_IGNORE, best).astype(np.uint8)
    return vox, out


def _ray_box_exit(origin: np.ndarray, unit: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Distance along each unit ray to where it leaves the grid box (0 if it never enters)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / unit
        t1 = (lo - origin) * inv
        t2 = (hi - origin) * inv
    tmax = np.nanmin(np.where(np.isnan(np.maximum(t1, t2)), np.inf, np.maximum(t1, t2)), axis=1)
    return np.clip(tmax, 0.0, None)


def _camera_points(depth_cm: np.ndarray, pose: Any, stride: int):
    h, w = depth_cm.shape
    rays, (rows, cols) = pixel_rays(pose, w, h, stride)
    d = depth_cm[np.ix_(rows, cols)].astype(np.float64)
    valid = (d > 0) & (d < DEPTH_CM_MAX)
    r = d / 100.0
    return rays, r, valid, (rows, cols)


def _free_voxels(origin: np.ndarray, rays: np.ndarray, r: np.ndarray, grid: OccGrid,
                 chunk: int = 4096) -> np.ndarray:
    """Linear ids of voxels crossed before each hit, margin one voxel."""
    norm = np.linalg.norm(rays, axis=1)
    unit = rays / norm[:, None]
    length = r * norm - grid.voxel_m
    t_end = np.minimum(length, _ray_box_exit(origin, unit, grid.lo, grid.hi))
    step = grid.voxel_m * FREE_STEP_FRAC
    out: list[np.ndarray] = []
    for s in range(0, len(unit), chunk):
        te = t_end[s:s + chunk]
        if not np.any(te > 0):
            continue
        m = int(math.ceil(float(te.max()) / step))
        t = (np.arange(m) + 0.5) * step
        keep = t[None, :] < te[:, None]
        pts = origin[None, None, :] + unit[s:s + chunk, None, :] * t[None, :, None]
        pts = pts[keep]
        idx, inside = grid.to_index(pts)
        if inside.any():
            out.append(np.unique(grid.linear(idx[inside])))
    return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)


def _box_cells(box: Mapping[str, Any], grid: OccGrid) -> np.ndarray | None:
    try:
        x, y, yaw = float(box["x"]), float(box["y"]), float(box["yaw"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, yaw)):
        return None
    length = box.get("length")
    width = box.get("width")
    reach = VEHICLE_CORE_M
    sized = isinstance(length, (int, float)) and isinstance(width, (int, float)) and length > 0 and width > 0
    if sized:
        reach = 0.5 * math.hypot(float(length), float(width))
    lo = np.array([x - reach, y - reach, VEHICLE_Z_M[0]])
    hi = np.array([x + reach, y + reach, VEHICLE_Z_M[1]])
    a, _ = grid.to_index(lo[None])
    b, _ = grid.to_index(hi[None])
    a = np.clip(a[0], 0, np.array(grid.dims) - 1)
    b = np.clip(b[0], 0, np.array(grid.dims) - 1)
    rng = [np.arange(a[k], b[k] + 1) for k in range(3)]
    idx = np.stack(np.meshgrid(*rng, indexing="ij"), -1).reshape(-1, 3)
    c = grid.centers(idx)
    zin = (c[:, 2] >= VEHICLE_Z_M[0]) & (c[:, 2] <= VEHICLE_Z_M[1])
    dx, dy = c[:, 0] - x, c[:, 1] - y
    if sized:
        hx, hy = math.cos(yaw), math.sin(yaw)
        along = dx * hx + dy * hy
        across = -dx * hy + dy * hx
        inside = (np.abs(along) <= 0.5 * float(length)) & (np.abs(across) <= 0.5 * float(width))
    else:
        inside = np.hypot(dx, dy) <= VEHICLE_CORE_M
    return grid.linear(idx[inside & zin])


def build_occ_labels(
    depth_by_cam: Mapping[str, np.ndarray],
    poses: Mapping[str, Any],
    grid: OccGrid,
    *,
    stride: int = 4,
    sem_by_cam: Mapping[str, np.ndarray] | None = None,
    sem_names: Mapping[str, Any] | None = None,
    boxes: Sequence[Mapping[str, Any]] | None = None,
) -> SparseOccLabels:
    """Sparse labels for one tick. Any camera with depth and a pose contributes."""
    lut = _palette_lut(sem_names)
    occ_lin: list[np.ndarray] = []
    occ_c: list[np.ndarray] = []
    free_lin: list[np.ndarray] = []
    for cam, depth in dict(depth_by_cam or {}).items():
        pose = poses.get(cam)
        if pose is None or depth is None or np.asarray(depth).ndim != 2:
            continue
        depth = np.asarray(depth, np.uint16)
        rays, r, valid, (rows, cols) = _camera_points(depth, pose, stride)
        if not valid.any():
            continue
        origin = np.array([pose.x, pose.y, pose.z], np.float64)
        rv = rays[valid]
        pts = origin + rv * r[valid][:, None]
        idx, inside = grid.to_index(pts)
        cls = np.full(len(pts), CLS_IGNORE, np.uint8)
        sem = None if sem_by_cam is None else sem_by_cam.get(cam)
        if sem is not None and np.asarray(sem).shape == depth.shape and lut is not None:
            sid = np.asarray(sem)[np.ix_(rows, cols)][valid].astype(np.int64)
            cls = lut[np.clip(sid, 0, 65535)]
        occ_lin.append(grid.linear(idx[inside]))
        occ_c.append(cls[inside])
        free_lin.append(_free_voxels(origin, rv, r[valid], grid))

    lin = np.concatenate(occ_lin) if occ_lin else np.zeros(0, np.int64)
    cls = np.concatenate(occ_c) if occ_c else np.zeros(0, np.uint8)
    u_lin, u_cls = _vote(lin, cls)
    flow = np.full((len(u_lin), 2), np.nan, np.float32)
    box_t = np.full((len(u_lin), 6), np.nan, np.float32)

    for box in boxes or []:
        cells = _box_cells(box, grid)
        if cells is None or len(cells) == 0:
            continue
        speed = box.get("speed", box.get("speed_mps"))
        yaw = float(box["yaw"])
        fv = (np.array([math.cos(yaw), math.sin(yaw)], np.float32) * float(speed)
              if isinstance(speed, (int, float)) and math.isfinite(float(speed)) else np.full(2, np.nan, np.float32))
        new = np.setdiff1d(cells, u_lin, assume_unique=False)
        u_lin = np.concatenate([u_lin, new])
        u_cls = np.concatenate([u_cls, np.full(len(new), CLS_VEHICLE, np.uint8)])
        flow = np.concatenate([flow, np.full((len(new), 2), np.nan, np.float32)])
        box_t = np.concatenate([box_t, np.full((len(new), 6), np.nan, np.float32)])
        hit = np.isin(u_lin, cells)
        u_cls[hit] = CLS_VEHICLE
        flow[hit] = fv
        ln, wd = box.get("length"), box.get("width")
        if isinstance(ln, (int, float)) and isinstance(wd, (int, float)) and math.isfinite(float(ln)) and math.isfinite(float(wd)):
            cen = grid.centers(grid.unlinear(u_lin[hit]))[:, :2]
            box_t[hit, 0] = float(box["x"]) - cen[:, 0]
            box_t[hit, 1] = float(box["y"]) - cen[:, 1]
            box_t[hit, 2:6] = [float(ln), float(wd), math.sin(yaw), math.cos(yaw)]

    free = np.concatenate(free_lin) if free_lin else np.zeros(0, np.int64)
    free = np.setdiff1d(np.unique(free), u_lin)
    if not len(u_lin) and not len(free):
        return _empty()
    order = np.argsort(u_lin)
    return SparseOccLabels(
        occ_idx=grid.unlinear(u_lin[order]).astype(np.int16),
        occ_cls=u_cls[order],
        flow=flow[order],
        free_idx=grid.unlinear(free).astype(np.int16),
        box=box_t[order],
    )


def _coarse_sets(lab: SparseOccLabels, grid: OccGrid) -> tuple[np.ndarray, np.ndarray]:
    """Coarse cells with any occupied child; coarse cells with free children and no occupied child."""
    f = np.array(grid.coarse)
    cdims = np.array(grid.coarse_dims)

    def lin(c):
        c = c[np.all(c < cdims, axis=1)]
        return np.unique((c[:, 0] * cdims[1] + c[:, 1]) * cdims[2] + c[:, 2]) if len(c) else np.zeros(0, np.int64)

    occ_c = lin(lab.occ_idx.astype(np.int64) // f)
    free_c = np.setdiff1d(lin(lab.free_idx.astype(np.int64) // f), occ_c)

    def unlin(v):
        return np.stack([v // (cdims[1] * cdims[2]), (v // cdims[2]) % cdims[1], v % cdims[2]], 1)

    return unlin(occ_c), unlin(free_c)


def sample_queries(lab: SparseOccLabels, grid: OccGrid, *, n_occ: int, n_free: int,
                   rng: np.random.Generator, n_coarse: int | None = None) -> dict[str, Any]:
    """Fixed-size query batch from sparse labels: occupied, free, then coarse cells.

    Rows with no label source are padded with mask 0 (they add no loss).
    """
    import torch

    n_coarse = (n_occ + n_free) // 4 if n_coarse is None else n_coarse
    q_rows: list[np.ndarray] = []
    occ_t: list[np.ndarray] = []
    mask: list[np.ndarray] = []
    cls_t: list[np.ndarray] = []
    flow_t: list[np.ndarray] = []
    box_l: list[np.ndarray] = []

    def take(n_avail: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        if n_avail == 0:
            return np.zeros(n, np.int64), np.zeros(n, np.float32)
        return rng.integers(0, n_avail, n), np.ones(n, np.float32)

    sel, m = take(len(lab.occ_idx), n_occ)
    cen = grid.centers(lab.occ_idx[sel]) if len(lab.occ_idx) else np.zeros((n_occ, 3))
    q_rows.append(np.concatenate([cen, np.zeros((n_occ, 1))], 1))
    occ_t.append(np.ones(n_occ, np.float32))
    mask.append(m)
    c = lab.occ_cls[sel].astype(np.int64) if len(lab.occ_idx) else np.full(n_occ, CLS_IGNORE)
    cls_t.append(np.where((c == CLS_IGNORE) | (m == 0), -100, c))
    fl = lab.flow[sel] if len(lab.occ_idx) else np.full((n_occ, 2), np.nan, np.float32)
    flow_t.append(fl)
    bx = lab.box[sel] if (lab.box is not None and len(lab.occ_idx)) else np.full((n_occ, 6), np.nan, np.float32)
    box_l.append(bx)

    sel, m = take(len(lab.free_idx), n_free)
    cen = grid.centers(lab.free_idx[sel]) if len(lab.free_idx) else np.zeros((n_free, 3))
    q_rows.append(np.concatenate([cen, np.zeros((n_free, 1))], 1))
    occ_t.append(np.zeros(n_free, np.float32))
    mask.append(m)
    cls_t.append(np.full(n_free, -100))
    flow_t.append(np.full((n_free, 2), np.nan, np.float32))
    box_l.append(np.full((n_free, 6), np.nan, np.float32))

    if n_coarse > 0:
        c_occ, c_free = _coarse_sets(lab, grid)
        for cells, target, n in ((c_occ, 1.0, n_coarse // 2), (c_free, 0.0, n_coarse - n_coarse // 2)):
            sel, m = take(len(cells), n)
            cen = grid.coarse_centers(cells[sel]) if len(cells) else np.zeros((n, 3))
            q_rows.append(np.concatenate([cen, np.ones((n, 1))], 1))
            occ_t.append(np.full(n, target, np.float32))
            mask.append(m)
            cls_t.append(np.full(n, -100))
            flow_t.append(np.full((n, 2), np.nan, np.float32))
            box_l.append(np.full((n, 6), np.nan, np.float32))

    flow = np.concatenate(flow_t).astype(np.float32)
    fmask = np.all(np.isfinite(flow), axis=1).astype(np.float32)
    boxv = np.concatenate(box_l).astype(np.float32)
    bmask = np.all(np.isfinite(boxv), axis=1).astype(np.float32)
    return {
        "box": torch.tensor(np.nan_to_num(boxv)),
        "box_mask": torch.tensor(bmask),
        "queries": torch.tensor(np.concatenate(q_rows), dtype=torch.float32),
        "occ": torch.tensor(np.concatenate(occ_t)),
        "occ_mask": torch.tensor(np.concatenate(mask)),
        "cls": torch.tensor(np.concatenate(cls_t), dtype=torch.long),
        "flow": torch.tensor(np.nan_to_num(flow)),
        "flow_mask": torch.tensor(fmask),
    }


def occ_loss(pred: Mapping[str, Any], batch: Mapping[str, Any], *, w_sem: float = 0.5, w_flow: float = 0.1, w_box: float = 0.2):
    """Masked BCE on occupancy + CE on supervised classes + L1 on box flow + smooth-L1 on recorded boxes."""
    import torch
    import torch.nn.functional as F

    m = batch["occ_mask"].float()
    bce = F.binary_cross_entropy_with_logits(pred["occ"].float(), batch["occ"].float(), reduction="none")
    loss = (bce * m).sum() / torch.clamp(m.sum(), min=1.0)
    cls = batch["cls"]
    if bool((cls >= 0).any()):
        loss = loss + w_sem * F.cross_entropy(pred["sem"].float().reshape(-1, pred["sem"].shape[-1]),
                                              cls.reshape(-1), ignore_index=-100)
    fm = batch["flow_mask"].float()
    if bool((fm > 0).any()):
        l1 = (pred["flow"].float() - batch["flow"].float()).abs().sum(-1)
        loss = loss + w_flow * (l1 * fm).sum() / torch.clamp(fm.sum(), min=1.0)
    bm = batch.get("box_mask")
    if bm is not None and "box" in pred and bool((bm > 0).any()):
        t = batch["box"].float()
        tgt = torch.cat([t[..., :2], torch.log(torch.clamp(t[..., 2:4], min=0.1)), t[..., 4:6]], -1)
        l1 = F.smooth_l1_loss(pred["box"].float(), tgt, reduction="none").sum(-1)
        loss = loss + w_box * (l1 * bm.float()).sum() / torch.clamp(bm.float().sum(), min=1.0)
    return loss
