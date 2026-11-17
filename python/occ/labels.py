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
from python.occ.classes import BLOCKED, CURB, LANE_WHITE, LANE_YELLOW, PAINT_CLASSES, ROAD, STOP_LINE, UNKNOWN, VEHICLE
from python.occ.grid import DEFAULT_GRID, VoxelGrid

DEPTH_CM_INVALID = (0, 65535)
DRIVABLE_NAMES = frozenset({"ROAD", "Asphalt", "ASPHALT", "Highway", "HIGHWAY"})  # sensor_aux allowlist
BLOCK_H_M = 0.30          # above ground: blocked (geometric, no fine class needed)
CURB_H_M = (0.05, 0.30)   # curb band next to road
FREE_MARGIN_M = 0.75      # a voxel this far in front of the depth hit is free
# Conservative core car box (smaller than any companion) and the outer ignore box.
BOX_CORE = (3.6, 1.6, 1.3)
BOX_OUTER = (5.2, 2.2, 1.9)
PRIORITY = {UNKNOWN: 0, ROAD: 1, LANE_WHITE: 2, LANE_YELLOW: 2, STOP_LINE: 2, CURB: 3, BLOCKED: 4, VEHICLE: 5}
STATIC_CLASSES = frozenset({ROAD, CURB, BLOCKED}) | PAINT_CLASSES
PAINT_VIS_MARGIN_M = 1.0


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
        box = {"x": a[0], "y": a[1], "yaw": math.atan2(b[1] - a[1], b[0] - a[0])}
        vel = item.get("vel")
        try:
            vx, vy = float(vel[0]), float(vel[1])
        except (TypeError, ValueError, IndexError, KeyError):
            vx = vy = math.nan
        o = world_to_ego((0.0, 0.0), (0.0, 0.0), ego.get("dir"))
        e = world_to_ego((vx, vy), (0.0, 0.0), ego.get("dir")) if math.isfinite(vx) and math.isfinite(vy) else None
        if o is not None and e is not None:
            box["vx"], box["vy"] = e[0] - o[0], e[1] - o[1]  # ground velocity in ego axes
        out.append(box)
    return out


def paint_segments_from_decals(decals: Iterable[Mapping[str, Any]], pos: Any, direction: Any, *, reach_m: float = 90.0) -> list[dict[str, Any]]:
    """Ego-frame lane-paint segments ``{x0, y0, x1, y1, yellow}`` from DecalRoads (painted_lanes materials)."""
    from python.debug.painted_lanes import _catmull_rom, is_paint

    try:
        px, py = float(pos[0]), float(pos[1])
        fx, fy = float(direction[0]), float(direction[1])
    except (TypeError, ValueError, IndexError):
        return []
    n = math.hypot(fx, fy)
    if not (math.isfinite(px) and math.isfinite(py)) or not math.isfinite(n) or n < 1e-6:
        return []
    fx, fy = fx / n, fy / n
    rx, ry = fy, -fx
    out = []
    for d in decals:
        mat = str(d.get("material") or "")
        nodes = d.get("nodes")
        if not is_paint(mat) or not nodes or len(nodes) < 2:
            continue
        p = np.asarray(_catmull_rom(nodes), dtype=np.float64)[:, :2] - (px, py)
        ex, ey = p @ (rx, ry), p @ (fx, fy)
        yellow = "yellow" in mat.lower()
        for k in range(len(ex) - 1):
            if min(math.hypot(ex[k], ey[k]), math.hypot(ex[k + 1], ey[k + 1])) > reach_m:
                continue
            out.append({"x0": float(ex[k]), "y0": float(ey[k]), "x1": float(ex[k + 1]), "y1": float(ey[k + 1]), "yellow": yellow})
    return out


def _seg_points(x0: float, y0: float, x1: float, y1: float, step: float) -> np.ndarray:
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        return np.zeros((0, 2))
    q = math.hypot(x1 - x0, y1 - y0) / step  # length or length/step can overflow to inf for huge finite endpoints (fuzz F1)
    n = (int(min(10000, max(1, math.ceil(q)))) if math.isfinite(q) else 10000) + 1
    t = np.linspace(0.0, 1.0, n)
    return np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], axis=1)


def voxel_labels(
    frames: Sequence[CamFrame],
    sem_classes: Mapping[str, Any] | None = None,
    boxes: Iterable[Mapping[str, float]] = (),
    *,
    paint: Iterable[Mapping[str, Any]] = (),
    stop_bars: Iterable[Mapping[str, Any]] = (),
    grid: VoxelGrid = DEFAULT_GRID,
    ground_z: float = 0.0,
    max_free: int = 4096,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Sparse labels for one tick.

    Returns ``occ_idx`` (N,3) int32 unique, ``occ_cls`` (N,) int8,
    ``free_idx`` (M,3) int32 unique and disjoint from ``occ_idx`` and from
    every box's outer band, ``ignore_idx`` (K,3) int32 (box band),
    ``occ_flow`` (N,2) float32 ground velocity in ego axes (box ``vx/vy`` for
    VEHICLE, 0 for static classes, NaN = unlabeled).
    """
    boxes = list(boxes)
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

    # Painted lanes / stop bars: ground cells along each stroke that a camera sees.
    iz_g = int(math.floor((ground_z + 0.01 - grid.z_min) / grid.voxel))
    pxy: list[np.ndarray] = []
    pcls: list[np.ndarray] = []
    for seg in paint:
        try:
            pts2 = _seg_points(float(seg["x0"]), float(seg["y0"]), float(seg["x1"]), float(seg["y1"]), grid.voxel / 2)
        except (KeyError, TypeError, ValueError):
            continue
        pxy.append(pts2)
        pcls.append(np.full(len(pts2), LANE_YELLOW if seg.get("yellow") is True else LANE_WHITE, np.int8))
    for bar in stop_bars:
        if bar.get("valid") is not True:
            continue
        try:
            pts2 = _seg_points(float(bar["x0"]), float(bar["y_m"]), float(bar["x1"]), float(bar["y_m"]), grid.voxel / 2)
        except (KeyError, TypeError, ValueError):
            continue
        pxy.append(pts2)
        pcls.append(np.full(len(pts2), STOP_LINE, np.int8))
    if pxy and 0 <= iz_g < nz and frames:
        p2 = np.concatenate(pxy)
        p3 = np.concatenate([p2, np.full((len(p2), 1), ground_z)], axis=1)
        seen = np.zeros(len(p3), dtype=bool)
        for fr in frames:
            d = depth_to_m(fr.depth)
            uv, z, ok = fr.cam.project(p3)
            u = np.clip(np.nan_to_num(uv[:, 0], nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64), 0, fr.cam.width - 1)
            v = np.clip(np.nan_to_num(uv[:, 1], nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64), 0, fr.cam.height - 1)
            hit = d[v, u]
            seen |= ok & np.isfinite(hit) & (hit >= z - PAINT_VIS_MARGIN_M)
        pi, pin = grid.to_index(p3)
        keep_p = pin & seen
        if keep_p.any():
            pi = pi[keep_p]
            pi[:, 2] = iz_g
            idx = np.concatenate([idx, pi])
            cls = np.concatenate([cls, np.concatenate(pcls)[keep_p]])

    # Per-voxel class: highest priority among its points.
    flat = grid.flat(idx)
    pri = np.vectorize(PRIORITY.get, otypes=[np.int64])(cls) if cls.size else np.zeros(0, dtype=np.int64)
    order = np.lexsort((-pri, flat))
    flat_s, cls_s = flat[order], cls[order]
    keep = np.ones(flat_s.size, dtype=bool)
    keep[1:] = flat_s[1:] != flat_s[:-1]
    occ = {int(f): int(c) for f, c in zip(flat_s[keep], cls_s[keep])}

    ignore: set[int] = set()
    vflow: dict[int, tuple[float, float]] = {}
    for box in boxes:
        core = grid.flat(_box_voxels(grid, box, BOX_CORE))
        outer = grid.flat(_box_voxels(grid, box, BOX_OUTER))
        try:
            bv = (float(box.get("vx", math.nan)), float(box.get("vy", math.nan)))
        except (TypeError, ValueError):
            bv = (math.nan, math.nan)
        if not all(math.isfinite(v) and abs(v) < 100.0 for v in bv):
            bv = (math.nan, math.nan)
        for f in core.tolist():
            occ[int(f)] = VEHICLE
            vflow[int(f)] = bv
        ignore.update(int(f) for f in outer.tolist())
    ignore -= set(occ)
    # A non-vehicle depth hit inside the outer band is still kept as occupied.

    occ_flat = np.array(sorted(occ), dtype=np.int64)
    occ_cls = np.array([occ[int(f)] for f in occ_flat], dtype=np.int8)
    occ_flow = np.full((len(occ_flat), 2), np.nan, dtype=np.float32)
    for k, (f, c) in enumerate(zip(occ_flat.tolist(), occ_cls.tolist())):
        if c == VEHICLE:
            occ_flow[k] = vflow.get(int(f), (math.nan, math.nan))
        elif c in STATIC_CLASSES:
            occ_flow[k] = 0.0

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
        "occ_flow": occ_flow,
        "boxes": np.array([[float(b.get("x", np.nan)), float(b.get("y", np.nan)), float(b.get("yaw", np.nan)),
                            float(b.get("vx", np.nan)), float(b.get("vy", np.nan))] for b in boxes], dtype=np.float32).reshape(-1, 5),
        "free_idx": grid.unflat(free_flat),
        "ignore_idx": grid.unflat(ign_flat),
    }


def vo_target(prev_row: Mapping[str, Any] | None, row: Mapping[str, Any]) -> np.ndarray:
    """Training label for the VO head: ego motion (dx, dy, dyaw) from the previous tick's pose,
    in the previous ego frame. GT pose is used here (training) only. NaN when unknown."""
    nan = np.full(3, np.nan, np.float32)
    if not isinstance(prev_row, Mapping):
        return nan

    def pose(r: Mapping[str, Any]):
        e = r.get("ego") if isinstance(r.get("ego"), Mapping) else r
        try:
            return float(e["pos"][0]), float(e["pos"][1]), math.atan2(float(e["dir"][1]), float(e["dir"][0]))
        except (KeyError, TypeError, ValueError, IndexError):
            return None

    a, b = pose(prev_row), pose(row)
    if a is None or b is None or not all(math.isfinite(v) for v in (*a, *b)):
        return nan
    th = a[2]
    fx, fy = math.cos(th), math.sin(th)
    rx, ry = fy, -fx
    dxw, dyw = b[0] - a[0], b[1] - a[1]
    dyaw = (b[2] - a[2] + math.pi) % (2 * math.pi) - math.pi
    return np.array([dxw * rx + dyw * ry, dxw * fx + dyw * fy, dyaw], np.float32)
