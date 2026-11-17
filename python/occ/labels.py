"""Training-only sparse occupancy labels from recorded depth, semantics and boxes.

Never imported by runtime modules. Inputs per tick:

* ``raw/<cam>_depth`` uint16 centimetres (0 and 65535 are invalid) for each
  ``train_gt`` camera (``main``, ``wide``).
* ``raw/<cam>_sem`` uint8 class ids with names in ``sem_classes.json``.
* the tick's ``traffic`` rows (world pos/dir) turned into ego boxes.

Output is sparse: unique voxel indices plus a class per occupied voxel,
a sample of known-free voxels (in front of a depth hit), and an ignore band
around each box (the box extent is not measured; see ``vehicle_labels``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from python.occ.camera import CameraModel
from python.occ.classes import BLOCKED, CURB, ROAD, UNKNOWN, VEHICLE
from python.occ.grid import DEFAULT_GRID, VoxelGrid

DEPTH_CM_INVALID = (0, 65535)
DRIVABLE_NAMES = frozenset({"ROAD", "Asphalt", "ASPHALT", "Highway", "HIGHWAY"})  # sensor_aux allowlist
BLOCK_H_M = 0.30          # above ground: blocked (geometric, no fine class needed)
CURB_H_M = (0.05, 0.30)   # curb band next to road
FREE_MARGIN_M = 0.75      # a voxel this far in front of the depth hit is free
# Conservative core car box (smaller than any companion) and the outer ignore box.
BOX_CORE = (3.6, 1.6, 1.3)
BOX_OUTER = (5.2, 2.2, 1.9)
PRIORITY = {UNKNOWN: 0, ROAD: 1, CURB: 2, BLOCKED: 3, VEHICLE: 4}


@dataclass
class CamFrame:
    cam: CameraModel
    depth: np.ndarray            # (H, W) uint16 cm or float metres
    sem: np.ndarray | None = None  # (H, W) uint8 ids


def depth_to_m(depth: np.ndarray) -> np.ndarray:
    """uint16 cm -> metres with 0/65535 as NaN; float input is metres, non-finite/<=0 -> NaN."""
    arr = np.asarray(depth)
    if arr.dtype == np.uint16:
        m = arr.astype(np.float64) / 100.0
        bad = (arr == DEPTH_CM_INVALID[0]) | (arr == DEPTH_CM_INVALID[1])
        return np.where(bad, np.nan, m)
    m = arr.astype(np.float64)
    return np.where(np.isfinite(m) & (m > 0.0), m, np.nan)


def road_ids(sem_classes: Mapping[str, Any] | None) -> frozenset[int]:
    """Sem ids whose name is in the drivable allowlist. Accepts the file payload or its ``classes``."""
    payload = dict(sem_classes or {})
    classes = payload.get("classes", payload)
    out = set()
    for k, name in dict(classes).items():
        try:
            cid = int(k)
        except (TypeError, ValueError):
            continue
        if cid > 0 and str(name) in DRIVABLE_NAMES:
            out.add(cid)
    return frozenset(out)


def _box_mask(centres: np.ndarray, box: Mapping[str, float], dims: tuple[float, float, float]) -> np.ndarray:
    yaw = float(box["yaw"])
    # Box heading is the forward axis (length); yaw from +x.
    fx, fy = math.cos(yaw), math.sin(yaw)
    dx = centres[:, 0] - float(box["x"])
    dy = centres[:, 1] - float(box["y"])
    along = dx * fx + dy * fy
    across = -dx * fy + dy * fx
    z0 = float(box.get("z", 0.0))
    return (np.abs(along) <= dims[0] / 2) & (np.abs(across) <= dims[1] / 2) & (centres[:, 2] >= z0) & (centres[:, 2] <= z0 + dims[2])


def _box_voxels(grid: VoxelGrid, box: Mapping[str, float], dims: tuple[float, float, float]) -> np.ndarray:
    r = math.hypot(dims[0], dims[1]) / 2 + grid.voxel
    x, y = float(box["x"]), float(box["y"])
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(float(box["yaw"]))):
        return np.zeros((0, 3), dtype=np.int32)
    lo, ok_lo = grid.to_index(np.array([[x - r, y - r, grid.z_min]]))
    nx, ny, nz = grid.shape
    i0 = int(np.clip(np.floor((x - r - grid.x_min) / grid.voxel), 0, nx))
    i1 = int(np.clip(np.ceil((x + r - grid.x_min) / grid.voxel), 0, nx))
    j0 = int(np.clip(np.floor((y - r - grid.y_min) / grid.voxel), 0, ny))
    j1 = int(np.clip(np.ceil((y + r - grid.y_min) / grid.voxel), 0, ny))
    if i1 <= i0 or j1 <= j0:
        return np.zeros((0, 3), dtype=np.int32)
    gi, gj, gk = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), np.arange(nz), indexing="ij")
    idx = np.stack([gi.reshape(-1), gj.reshape(-1), gk.reshape(-1)], axis=1).astype(np.int32)
    return idx[_box_mask(grid.to_center(idx), box, dims)]


def boxes_from_row(row: Mapping[str, Any]) -> list[dict[str, float]]:
    """Ego-frame boxes (x, y, yaw) for each traffic row with a finite pos and dir."""
    from python.collect.signal_labels import world_to_ego

    ego = row.get("ego") if isinstance(row.get("ego"), Mapping) else row
    tags = row.get("tags") if isinstance(row.get("tags"), Mapping) else {}
    traffic = row.get("traffic", tags.get("traffic"))
    out: list[dict[str, float]] = []
    for item in traffic or []:
        if not isinstance(item, Mapping):
            continue
        pos, d = item.get("pos"), item.get("dir")
        try:
            px, py = float(pos[0]), float(pos[1])
            dx, dy = float(d[0]), float(d[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if not all(math.isfinite(v) for v in (px, py, dx, dy)) or math.hypot(dx, dy) < 1e-8:
            continue
        a = world_to_ego((px, py), ego.get("pos"), ego.get("dir"))
        b = world_to_ego((px + dx, py + dy), ego.get("pos"), ego.get("dir"))
        if a is None or b is None:
            continue
        out.append({"x": a[0], "y": a[1], "yaw": math.atan2(b[1] - a[1], b[0] - a[0])})
    return out


def voxel_labels(
    frames: Sequence[CamFrame],
    sem_classes: Mapping[str, Any] | None = None,
    boxes: Iterable[Mapping[str, float]] = (),
    *,
    grid: VoxelGrid = DEFAULT_GRID,
    ground_z: float = 0.0,
    max_free: int = 4096,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Sparse labels for one tick.

    Returns ``occ_idx`` (N,3) int32 unique, ``occ_cls`` (N,) int8,
    ``free_idx`` (M,3) int32 unique and disjoint from ``occ_idx`` and from
    every box's outer band, ``ignore_idx`` (K,3) int32 (box band).
    """
    rids = road_ids(sem_classes)
    pts_all: list[np.ndarray] = []
    cls_all: list[np.ndarray] = []
    for fr in frames:
        d = depth_to_m(fr.depth)
        ego, valid = fr.cam.backproject(d)
        if fr.sem is not None:
            sem = np.asarray(fr.sem)
            if sem.shape != d.shape:
                raise ValueError(f"sem shape {sem.shape} != depth {d.shape}")
            is_road = np.isin(sem.reshape(-1), list(rids)) if rids else np.zeros(sem.size, dtype=bool)
        else:
            is_road = np.zeros(d.size, dtype=bool)
        h = ego[:, 2] - ground_z
        cls = np.full(d.size, UNKNOWN, dtype=np.int8)
        cls[is_road] = ROAD
        cls[(~is_road) & (h > BLOCK_H_M)] = BLOCKED
        pts_all.append(ego[valid])
        cls_all.append(cls[valid])
        # curb candidates resolved after road cells are known
    pts = np.concatenate(pts_all) if pts_all else np.zeros((0, 3))
    cls = np.concatenate(cls_all) if cls_all else np.zeros((0,), dtype=np.int8)
    idx, inb = grid.to_index(pts)
    idx, cls, hh = idx[inb], cls[inb], (pts[inb, 2] - ground_z)

    nx, ny, nz = grid.shape
    # Curb: low non-road point whose BEV cell touches a road BEV cell laterally.
    road_bev = set(map(tuple, idx[cls == ROAD][:, :2].tolist()))
    cand = (cls == UNKNOWN) & (hh >= CURB_H_M[0]) & (hh <= CURB_H_M[1])
    if road_bev and cand.any():
        ci = np.nonzero(cand)[0]
        for k in ci:
            ix, iy = int(idx[k, 0]), int(idx[k, 1])
            if (ix - 1, iy) in road_bev or (ix + 1, iy) in road_bev:
                cls[k] = CURB

    # Per-voxel class: highest priority among its points.
    flat = grid.flat(idx)
    pri = np.vectorize(PRIORITY.get, otypes=[np.int64])(cls) if cls.size else np.zeros(0, dtype=np.int64)
    order = np.lexsort((-pri, flat))
    flat_s, cls_s = flat[order], cls[order]
    keep = np.ones(flat_s.size, dtype=bool)
    keep[1:] = flat_s[1:] != flat_s[:-1]
    occ = {int(f): int(c) for f, c in zip(flat_s[keep], cls_s[keep])}

    ignore: set[int] = set()
    for box in boxes:
        core = grid.flat(_box_voxels(grid, box, BOX_CORE))
        outer = grid.flat(_box_voxels(grid, box, BOX_OUTER))
        for f in core.tolist():
            occ[int(f)] = VEHICLE
        ignore.update(int(f) for f in outer.tolist())
    ignore -= set(occ)
    # A non-vehicle depth hit inside the outer band is still kept as occupied.

    occ_flat = np.array(sorted(occ), dtype=np.int64)
    occ_cls = np.array([occ[int(f)] for f in occ_flat], dtype=np.int8)

    # Free space: random voxels seen in front of a valid depth hit.
    rng = np.random.default_rng(seed)
    n_try = max_free * 4
    free: set[int] = set()
    if frames and max_free > 0:
        cand_flat = rng.integers(0, grid.n_dense, size=n_try)
        centres = grid.to_center(grid.unflat(cand_flat))
        is_free = np.zeros(n_try, dtype=bool)
        for fr in frames:
            d = depth_to_m(fr.depth)
            uv, z, ok = fr.cam.project(centres)
            u = np.clip(uv[:, 0].astype(np.int64, copy=False) if ok.any() else np.zeros(n_try, dtype=np.int64), 0, fr.cam.width - 1)
            v = np.clip(uv[:, 1].astype(np.int64, copy=False) if ok.any() else np.zeros(n_try, dtype=np.int64), 0, fr.cam.height - 1)
            hit = d[v, u]
            is_free |= ok & np.isfinite(hit) & (z < hit - FREE_MARGIN_M)
        blocked = set(occ) | ignore
        for f in cand_flat[is_free].tolist():
            if f not in blocked:
                free.add(int(f))
            if len(free) >= max_free:
                break
    free_flat = np.array(sorted(free), dtype=np.int64)
    ign_flat = np.array(sorted(ignore), dtype=np.int64)
    return {
        "occ_idx": grid.unflat(occ_flat),
        "occ_cls": occ_cls,
        "free_idx": grid.unflat(free_flat),
        "ignore_idx": grid.unflat(ign_flat),
    }
