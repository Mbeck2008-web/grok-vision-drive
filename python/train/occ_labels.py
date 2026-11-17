"""Sparse occupancy labels from recorded depth + semantics (TRAINING ONLY).

Inputs are the training-only planes the recorder writes for ``train_gt``
cameras (``raw/<cam>_depth`` uint16 cm, ``raw/<cam>_sem`` uint8 ids,
``python/data/raw_frames.py``) for any subset of the eight rig cameras, plus
the 2 Hz traffic rows (``tags.traffic``). The live path never imports this
module (``scripts/test_occ_runtime_imports.py``).

Output is sparse: a sorted vector of occupied cell keys of ``OccGrid`` with one
coarse class per key, and a set of free-space points sampled on the observed
rays in front of each hit. No dense grid is allocated.

Rules (``docs/specs/labels_v4.md``):
* depth 0 is invalid; depth at the far plane is sky / no hit (free ray, no cell).
* coarse class names come from fixed allowlists only. Drivable reuses
  ``sensor_aux.DRIVABLE_CLASS_ALLOWLIST`` unchanged.
* a traffic row without a recorded size never invents a box. It only relabels
  depth-observed cells near its centre as vehicle. A row with an explicit
  ``length``/``width``/``height`` fills that box.
* cells inside the ego clear box are dropped (the ego body in the repeaters).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from python.perception.occ_geometry import DEFAULT_GRID, OccGrid, PinholeCam, backproject, pixel_rays
from python.train.sensor_aux import DRIVABLE_CLASS_ALLOWLIST

# Coarse classes of an occupied cell. 0 is "other" (occupied, class unknown).
OCC_CLASSES = ("other", "drivable", "vehicle")
OCC_OTHER, OCC_DRIVABLE, OCC_VEHICLE = 0, 1, 2
# Annotation names seen in a real recording's sem_classes.json
# (sgt_2h_20261105: CAR, TRUCK, TRUCK_TRAILERS). Exact names only.
VEHICLE_CLASS_ALLOWLIST = frozenset({"CAR", "TRUCK", "TRUCK_TRAILERS"})
# Pixel stride for back-projection (each camera). Stride 2 keeps 68% of the
# stride-1 cells at 11% of the time on a real frame (spec, measurements).
PIXEL_STRIDE = 2
# A hit this close to the far plane is the far plane (sky): ray is free, no cell.
FAR_FRAC = 0.995
# Free samples stop this far before the hit so they never sit in the hit cell.
FREE_MARGIN_M = 0.6
FREE_PER_CAM = 2048
# Traffic rows without a size: depth cells within this xy radius become vehicle.
VEHICLE_RADIUS_M = 2.0
VEHICLE_MAX_Z_M = 2.5
# Ego clear box (drop, never label): |x| <= 1.1, -2.6 <= y <= 2.6, z <= 2.0.
EGO_CLEAR = (1.1, -2.6, 2.6, 2.0)


@dataclass
class OccLabels:
    keys: np.ndarray        # int64 (K,) sorted unique occupied cell keys
    cls: np.ndarray         # uint8 (K,) coarse class per key
    free: np.ndarray        # float32 (M,3) free-space points (ego frame)
    cams: tuple[str, ...]   # cameras that contributed

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {"keys": self.keys, "cls": self.cls, "free": self.free}


def class_lut(sem_classes: Mapping[str, Any] | None) -> np.ndarray:
    """uint8[256] id -> coarse class from a ``sem_classes.json`` payload."""
    lut = np.zeros((256,), dtype=np.uint8)
    classes = dict((sem_classes or {}).get("classes") or {})
    for sid, name in classes.items():
        try:
            i = int(sid)
        except (TypeError, ValueError):
            continue
        if not 0 < i < 256:
            continue
        if name in DRIVABLE_CLASS_ALLOWLIST:
            lut[i] = OCC_DRIVABLE
        elif name in VEHICLE_CLASS_ALLOWLIST:
            lut[i] = OCC_VEHICLE
    return lut


def _ego_clear(pts: np.ndarray) -> np.ndarray:
    ax, y0, y1, zt = EGO_CLEAR
    return (np.abs(pts[:, 0]) <= ax) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1) & (pts[:, 2] <= zt)


def _box_exit(origin: np.ndarray, rays: np.ndarray, grid: OccGrid) -> np.ndarray:
    """Ray parameter where each ray leaves the grid box (0 when it starts outside)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / rays
        t0 = (grid.lo - origin) * inv
        t1 = (grid.hi - origin) * inv
    t_out = np.nanmin(np.maximum(t0, t1), axis=-1)
    inside = bool(np.all((origin >= grid.lo) & (origin < grid.hi)))
    return np.maximum(0.0, t_out) if inside else np.zeros(rays.shape[0])


def _majority(keys: np.ndarray, cls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique keys and the most common class per key (ties: higher class id)."""
    if keys.size == 0:
        return keys.astype(np.int64), cls.astype(np.uint8)
    n = len(OCC_CLASSES)
    combo = keys.astype(np.int64) * n + cls.astype(np.int64)
    uc, counts = np.unique(combo, return_counts=True)
    uk = uc // n
    ucls = uc % n
    # Sort by key, then count, then class; the last row per key wins.
    order = np.lexsort((ucls, counts, uk))
    uk, ucls = uk[order], ucls[order]
    last = np.r_[uk[1:] != uk[:-1], True]
    return uk[last], ucls[last].astype(np.uint8)


def world_to_ego(pts: np.ndarray, ego_pos: Sequence[float], ego_dir: Sequence[float], ego_up: Sequence[float] | None) -> np.ndarray:
    """World points -> ego frame (+X right = dir x up, +Y = dir, +Z = up)."""
    fwd = np.asarray(ego_dir, dtype=np.float64)
    fwd = fwd / np.linalg.norm(fwd)
    up = np.asarray(ego_up if ego_up is not None else (0.0, 0.0, 1.0), dtype=np.float64)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rot = np.stack([right, fwd, up])
    return (np.asarray(pts, dtype=np.float64) - np.asarray(ego_pos, dtype=np.float64)) @ rot.T


def vehicles_in_ego(traffic: Any, ego: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Traffic rows (world pos/dir) -> ego-frame centres, heading, optional size."""
    if not isinstance(ego, Mapping) or ego.get("pos") is None or ego.get("dir") is None:
        return []
    out: list[dict[str, Any]] = []
    for car in list(traffic or []):
        if not isinstance(car, Mapping) or car.get("pos") is None:
            continue
        try:
            c = world_to_ego(np.asarray([car["pos"]], dtype=np.float64), ego["pos"], ego["dir"], ego.get("up"))[0]
        except Exception:
            continue
        row: dict[str, Any] = {"center": c, "name": str(car.get("name") or "")}
        if car.get("dir") is not None:
            d = world_to_ego(np.asarray([car["dir"]], dtype=np.float64), (0, 0, 0), ego["dir"], ego.get("up"))[0]
            row["yaw"] = float(np.arctan2(d[0], d[1]))
        size = [car.get(k) for k in ("length", "width", "height")]
        if all(isinstance(s, (int, float)) and s > 0 for s in size):
            row["size"] = tuple(float(s) for s in size)
        out.append(row)
    return out


def _box_cells(grid: OccGrid, v: Mapping[str, Any]) -> np.ndarray:
    """Keys of cells whose centre lies in a vehicle box with an explicit size."""
    length, width, height = v["size"]
    yaw = float(v.get("yaw", 0.0))
    c = np.asarray(v["center"], dtype=np.float64)
    r = 0.5 * float(np.hypot(length, width)) + grid.voxel_m
    lo = np.array([c[0] - r, c[1] - r, c[2] - 0.5 * grid.voxel_m])
    hi = np.array([c[0] + r, c[1] + r, c[2] + height])
    step = grid.voxel_m
    xs = np.arange(grid.x_min + step / 2, grid.x_max, step)
    ys = np.arange(grid.y_min + step / 2, grid.y_max, step)
    zs = np.arange(grid.z_min + step / 2, grid.z_max, step)
    xs, ys, zs = xs[(xs >= lo[0]) & (xs <= hi[0])], ys[(ys >= lo[1]) & (ys <= hi[1])], zs[(zs >= lo[2]) & (zs <= hi[2])]
    if not (xs.size and ys.size and zs.size):
        return np.zeros((0,), dtype=np.int64)
    g = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    # Box frame: forward along yaw (yaw 0 = +Y).
    dx, dy = g[:, 0] - c[0], g[:, 1] - c[1]
    fwd = dx * np.sin(yaw) + dy * np.cos(yaw)
    lat = dx * np.cos(yaw) - dy * np.sin(yaw)
    inside = (np.abs(fwd) <= length / 2) & (np.abs(lat) <= width / 2) & (g[:, 2] >= c[2]) & (g[:, 2] <= c[2] + height)
    return grid.keys_of(g[inside])


def build_occ_labels(
    planes: Mapping[str, tuple[np.ndarray, np.ndarray | None]],
    cams: Mapping[str, PinholeCam],
    *,
    sem_classes: Mapping[str, Any] | None = None,
    vehicles: Sequence[Mapping[str, Any]] = (),
    grid: OccGrid = DEFAULT_GRID,
    stride: int = PIXEL_STRIDE,
    free_per_cam: int = FREE_PER_CAM,
    rng: np.random.Generator | None = None,
    radial: bool = False,
) -> OccLabels:
    """Fuse every camera that has a depth plane this tick into sparse labels.

    ``planes[cam_id] = (depth_cm uint16 HxW, sem uint8 HxW or None)``. A camera
    listed in ``planes`` but not in ``cams`` is refused (no silent drop).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    lut = class_lut(sem_classes)
    all_keys: list[np.ndarray] = []
    all_cls: list[np.ndarray] = []
    free: list[np.ndarray] = []
    used: list[str] = []
    s = max(1, int(stride))
    for cid, (depth_cm, sem) in planes.items():
        if cid not in cams:
            raise KeyError(f"no camera geometry for {cid}")
        cam = cams[cid]
        d = np.asarray(depth_cm)
        if d.ndim != 2:
            raise ValueError(f"{cid} depth must be 2-D")
        if (d.shape[1], d.shape[0]) != (cam.width, cam.height):
            cam = cam.scaled(max(d.shape))
            if (d.shape[1], d.shape[0]) != (cam.width, cam.height):
                raise ValueError(f"{cid} depth {d.shape} does not match camera {cam.width}x{cam.height}")
        vs, us = np.mgrid[s // 2 : d.shape[0] : s, s // 2 : d.shape[1] : s]
        us, vs = us.reshape(-1), vs.reshape(-1)
        dm = d[vs, us].astype(np.float64) / 100.0
        valid = dm > 0.0
        hit = valid & (dm < FAR_FRAC * cam.far_m)
        used.append(cid)
        # Occupied cells from hits.
        if np.any(hit):
            pts = backproject(cam, dm[hit], us[hit], vs[hit], radial=radial)
            keep = grid.inside(pts) & ~_ego_clear(pts)
            pts = pts[keep]
            if sem is not None:
                c = lut[np.asarray(sem)[vs[hit], us[hit]][keep]]
            else:
                c = np.zeros((pts.shape[0],), dtype=np.uint8)
            all_keys.append(grid.keys_of(pts))
            all_cls.append(c.astype(np.uint8))
        # Free samples along valid rays (sky rays run to the far plane).
        if np.any(valid) and free_per_cam > 0:
            idx = np.flatnonzero(valid)
            pick = idx[rng.integers(0, idx.size, size=int(free_per_cam))]
            rays = pixel_rays(cam, us[pick], vs[pick])
            dist = np.where(hit[pick], dm[pick], cam.far_m)
            if radial:
                dist = dist / np.linalg.norm(rays, axis=-1)
            scale = np.linalg.norm(rays, axis=-1)
            stop = np.maximum(0.0, dist - FREE_MARGIN_M / scale)
            # Free samples past the grid are wasted: stop where the ray leaves the box.
            stop = np.minimum(stop, _box_exit(np.asarray(cam.pos, dtype=np.float64), rays, grid))
            frac = rng.random(pick.size)
            # near-surface bias: half the samples in the last 30% of the free run
            frac = np.where(rng.random(pick.size) < 0.5, 0.7 + 0.3 * frac, frac)
            t = cam.near_m + frac * np.maximum(0.0, stop - cam.near_m)
            fp = np.asarray(cam.pos, dtype=np.float64) + rays * t[:, None]
            fp = fp[grid.inside(fp) & ~_ego_clear(fp) & (stop > cam.near_m)]
            free.append(fp)
    keys = np.concatenate(all_keys) if all_keys else np.zeros((0,), dtype=np.int64)
    cls = np.concatenate(all_cls) if all_cls else np.zeros((0,), dtype=np.uint8)
    keys, cls = _majority(keys, cls)
    # Vehicle rows: sized boxes add cells; unsized rows relabel nearby observed cells.
    for v in vehicles:
        if "size" in v:
            bk = np.unique(_box_cells(grid, v))
            bk = bk[bk >= 0]
            merged = np.union1d(keys, bk)
            mcls = np.zeros(merged.shape, dtype=np.uint8)
            mcls[np.searchsorted(merged, keys)] = cls
            mcls[np.searchsorted(merged, bk)] = OCC_VEHICLE
            keys, cls = merged, mcls
        elif keys.size:
            ctr = grid.centers_of_keys(keys)
            c = np.asarray(v["center"], dtype=np.float64)
            near = (np.hypot(ctr[:, 0] - c[0], ctr[:, 1] - c[1]) <= VEHICLE_RADIUS_M) & (ctr[:, 2] <= c[2] + VEHICLE_MAX_Z_M) & (cls != OCC_DRIVABLE)
            cls = cls.copy()
            cls[near] = OCC_VEHICLE
    fr = np.concatenate(free) if free else np.zeros((0, 3))
    if fr.size and keys.size:
        fk = grid.keys_of(fr)
        pos = np.searchsorted(keys, fk)
        pos = np.clip(pos, 0, keys.size - 1)
        fr = fr[keys[pos] != fk]
    return OccLabels(keys.astype(np.int64), cls.astype(np.uint8), fr.astype(np.float32), tuple(used))


def sparse_ratio(labels: OccLabels, grid: OccGrid = DEFAULT_GRID) -> float:
    """Stored occupied cells / dense cells."""
    return float(labels.keys.size) / float(grid.dense_cells)
