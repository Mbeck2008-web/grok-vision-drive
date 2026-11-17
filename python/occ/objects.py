"""Object boxes from predicted VEHICLE cells (BEV connected components)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from python.occ.classes import SRC_VEHICLE, VEHICLE
from python.occ.grid import DEFAULT_GRID, VoxelGrid

MAX_OBJECTS = 64
MIN_CELLS = 2


def objects_from_cells(idx: np.ndarray, cls: np.ndarray, conf: np.ndarray | None = None, *, grid: VoxelGrid = DEFAULT_GRID) -> list[dict[str, Any]]:
    idx = np.asarray(idx, dtype=np.int64).reshape(-1, 3)
    cls = np.asarray(cls).reshape(-1)
    conf = np.ones(cls.shape, dtype=np.float32) if conf is None else np.asarray(conf, dtype=np.float32).reshape(-1)
    sel = cls == VEHICLE
    veh, vconf = idx[sel], conf[sel]
    if veh.shape[0] == 0:
        return []
    bev: dict[tuple[int, int], list[int]] = {}
    for k, (ix, iy, _) in enumerate(veh.tolist()):
        bev.setdefault((ix, iy), []).append(k)
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for cell in sorted(bev):
        if cell in seen:
            continue
        stack, comp = [cell], []
        seen.add(cell)
        while stack:
            c = stack.pop()
            comp.append(c)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dy)
                    if n in bev and n not in seen:
                        seen.add(n)
                        stack.append(n)
        comps.append(comp)
    v = grid.voxel
    out = []
    for comp in comps:
        if len(comp) < MIN_CELLS:
            continue
        ks = [k for c in comp for k in bev[c]]
        cells = np.array(comp, dtype=np.float64)
        xy = np.stack([grid.x_min + (cells[:, 0] + 0.5) * v, grid.y_min + (cells[:, 1] + 0.5) * v], axis=1)
        mu = xy.mean(axis=0)
        cov = np.cov((xy - mu).T) if len(comp) > 1 else np.eye(2)
        w_, vecs = np.linalg.eigh(cov)
        major = vecs[:, int(np.argmax(w_))]
        yaw = math.atan2(major[1], major[0])
        # Box heading is ambiguous by pi; keep it in (0, pi] (forward-ish half).
        if yaw <= 0.0:
            yaw += math.pi
        a = (xy - mu) @ major
        b = (xy - mu) @ np.array([-major[1], major[0]])
        zs = veh[ks, 2]
        out.append(
            {
                "cls": "vehicle",
                "x": float(mu[0]),
                "y": float(mu[1]),
                "yaw": float(yaw),
                "l": float(a.max() - a.min() + v),
                "w": float(b.max() - b.min() + v),
                "h": float((zs.max() + 1) * v + grid.z_min - max(0.0, grid.z_min + zs.min() * v)),
                "conf": float(np.clip(vconf[ks].mean(), 0.0, 1.0)),
                "src": SRC_VEHICLE,
            }
        )
    out.sort(key=lambda o: math.hypot(o["x"], o["y"]))
    return out[:MAX_OBJECTS]
