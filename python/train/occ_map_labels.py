"""Full-region road targets from map GT + camera-ray visibility (training only).

Camera depth+sem labels (``occ_labels``) only cover what a camera ray hit.
For predicting unseen road structure (occluded intersection legs, the back of
a roundabout, corners behind cars and buildings) the ground layer is labelled
from the map instead, over the WHOLE grid, not masked to visible pixels:

* map source: the recorder's road dump (``E:\\GVD_data\\roads\\<map>.json.gz``
  via ``road_dump.load_road_dump`` -> ``RoadPoly`` centerline + width + z)
  and the ego pose row of the tick (``state.jsonl``), both sim truth, so this
  module is a training-label module and is unreachable from the runtime
  (``test_occ_runtime_imports``);
* ``drivable``: ground-layer cells whose centre is within width/2 of a road
  centerline; ``other``: every other ground-layer cell of the grid;
* ``edges``: road outlines (centerline offset by +-width/2, 0.4 m spacing),
  points that fall inside another road's surface dropped (junction mouths);
* ``visible``: sparse keys of every cell a camera ray crosses up to and
  including its depth hit (8 cams, pixel stride), used ONLY to split the
  metrics into visible / occluded, never to mask the loss.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from python.perception.occ_geometry import DEFAULT_GRID, OccGrid, PinholeCam, backproject
from python.train.occ_labels import FAR_FRAC, world_to_ego

DRIVABLE, OTHER = 1, 0
EDGE_STEP_M = 0.4
VIS_STRIDE = 4


@dataclass
class MapRoad:
    """One road in the ego frame: centerline (N,2), width, surface z (N,) or None."""
    xy: np.ndarray
    width_m: float
    z: np.ndarray | None = None
    closed: bool = False


@dataclass
class MapLabels:
    keys: np.ndarray            # int64 sorted ground-layer keys (whole grid)
    cls: np.ndarray             # uint8 per key: 1 drivable, 0 other
    edges: list[np.ndarray] = field(default_factory=list)   # (M,2) ego-frame edge points per outline
    ground_k: int = 0

    @property
    def drivable_keys(self) -> np.ndarray:
        return self.keys[self.cls == DRIVABLE]


def roads_to_ego(polys: Sequence[Any], ego_pos: Sequence[float], ego_dir: Sequence[float],
                 ego_up: Sequence[float] | None = None) -> list[MapRoad]:
    """``RoadPoly`` (world) -> ``MapRoad`` (ego frame). Missing z -> None."""
    out = []
    for p in polys:
        xy = np.asarray(p.xy, dtype=np.float64).reshape(-1, 2)
        if xy.shape[0] < 2:
            continue
        zs = getattr(p, "z", None)
        z = np.array([np.nan if v is None else float(v) for v in zs]) if zs is not None else np.full(len(xy), np.nan)
        e = world_to_ego(np.concatenate([xy, np.nan_to_num(z, nan=float(ego_pos[2]))[:, None]], 1), ego_pos, ego_dir, ego_up)
        out.append(MapRoad(e[:, :2], float(p.width_m), None if np.isnan(z).all() else e[:, 2]))
    return out


def _segments(r: MapRoad) -> tuple[np.ndarray, np.ndarray]:
    a = r.xy
    if r.closed:
        return a, np.roll(a, -1, axis=0)
    return a[:-1], a[1:]


def _seg_dist(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,) distance from pts (N,2) to the polyline segments a->b (S,2), min over segments (chunked)."""
    best = np.full(pts.shape[0], np.inf)
    for i in range(0, a.shape[0], 64):
        aa, bb = a[i:i + 64], b[i:i + 64]
        d = bb - aa
        L2 = np.maximum((d ** 2).sum(-1), 1e-12)
        t = np.clip(((pts[:, None, :] - aa[None]) * d[None]).sum(-1) / L2[None], 0.0, 1.0)
        q = aa[None] + t[..., None] * d[None]
        best = np.minimum(best, np.sqrt(((pts[:, None, :] - q) ** 2).sum(-1)).min(1))
    return best


def _ground_layer(grid: OccGrid) -> np.ndarray:
    v = float(grid.voxel_m)
    nx, ny, _ = grid.shape
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    return np.stack([grid.x_min + (ix.ravel() + 0.5) * v, grid.y_min + (iy.ravel() + 0.5) * v], 1), ix.ravel(), iy.ravel()


def build_map_labels(roads: Sequence[MapRoad], *, grid: OccGrid = DEFAULT_GRID, ground_z: float = 0.2) -> MapLabels:
    """Ground-layer drivable / other over the whole grid + edge points (ego frame)."""
    xy, ix, iy = _ground_layer(grid)
    gk = int(np.clip(np.floor((ground_z - grid.z_min) / grid.voxel_m), 0, grid.shape[2] - 1))
    drv = np.zeros(xy.shape[0], bool)
    for r in roads:
        a, b = _segments(r)
        lo = np.minimum(a, b).min(0) - r.width_m
        hi = np.maximum(a, b).max(0) + r.width_m
        near = (xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1])
        if near.any():
            idx = np.flatnonzero(near)
            drv[idx[_seg_dist(xy[idx], a, b) <= r.width_m / 2.0]] = True
    keys = grid.key_of_cell(np.stack([ix, iy, np.full(ix.shape, gk)], 1))
    order = np.argsort(keys)
    return MapLabels(keys[order], np.where(drv, DRIVABLE, OTHER).astype(np.uint8)[order], map_edges(roads, grid=grid), gk)


def map_edges(roads: Sequence[MapRoad], *, grid: OccGrid = DEFAULT_GRID, step: float = EDGE_STEP_M) -> list[np.ndarray]:
    """Road outlines in the grid, points inside another road's surface removed."""
    out = []
    for i, r in enumerate(roads):
        a, b = _segments(r)
        for side in (-1.0, 1.0):
            pts = []
            for p, q in zip(a, b):
                d = q - p
                L = float(np.hypot(*d))
                if L < 1e-6:
                    continue
                n = np.array([d[1], -d[0]]) / L  # right normal
                for t in np.arange(0.0, L, step):
                    pts.append(p + d * (t / L) + side * n * r.width_m / 2.0)
            if not pts:
                continue
            P = np.asarray(pts)
            keep = (P[:, 0] > grid.x_min) & (P[:, 0] < grid.x_max) & (P[:, 1] > grid.y_min) & (P[:, 1] < grid.y_max)
            for j, o in enumerate(roads):
                if j != i and keep.any():
                    oa, ob = _segments(o)
                    keep[keep] &= _seg_dist(P[keep], oa, ob) > o.width_m / 2.0 - 0.2
            if keep.sum() >= 2:
                out.append(P[keep])
    return out


def visible_keys(planes: Mapping[str, tuple[np.ndarray, Any]], cams: Mapping[str, PinholeCam], *,
                 grid: OccGrid = DEFAULT_GRID, stride: int = VIS_STRIDE) -> np.ndarray:
    """Sorted keys of every cell a camera ray crosses, up to and including its depth hit.

    Sky / far-plane pixels count as free rays to the grid edge. Only for the
    visible-vs-occluded metric split; the loss never sees this mask.
    """
    out: list[np.ndarray] = []
    step = float(grid.voxel_m) * 0.5
    for cid, (depth_cm, _sem) in planes.items():
        cam = cams[cid]
        d = np.asarray(depth_cm, dtype=np.float64)[::stride, ::stride] / 100.0
        vs, us = np.mgrid[0:depth_cm.shape[0]:stride, 0:depth_cm.shape[1]:stride]
        vs, us, d = vs.ravel(), us.ravel(), d.ravel()
        ok = d > 0.0
        d = np.where(d >= FAR_FRAC * cam.far_m, cam.far_m, d)
        hit = backproject(cam, d[ok], us[ok], vs[ok])
        o = np.asarray(cam.pos, dtype=np.float64)
        vec = hit - o
        dist = np.linalg.norm(vec, axis=1)
        unit = vec / np.maximum(dist, 1e-9)[:, None]
        n_max = int(np.ceil(dist.max() / step)) + 1 if dist.size else 0
        for c0 in range(0, unit.shape[0], 2048):
            u, dd = unit[c0:c0 + 2048], dist[c0:c0 + 2048]
            ts = np.arange(n_max) * step
            m = ts[None, :] <= dd[:, None] + step * 0.5
            pts = o + u[:, None, :] * np.minimum(ts[None, :], dd[:, None])[..., None]
            k = grid.keys_of(pts[m])
            out.append(np.unique(k[k >= 0]))
    return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)


def merge_targets(cam_keys: np.ndarray, cam_cls: np.ndarray, ml: MapLabels) -> tuple[np.ndarray, np.ndarray]:
    """Camera hits + map ground layer over the whole grid. On the ground layer the
    map class wins, except a vehicle cell (traffic box / vehicle hit) stays vehicle."""
    from python.train.occ_labels import OCC_CLASSES

    veh = OCC_CLASSES.index("vehicle")
    cam_keys = np.asarray(cam_keys, np.int64)
    cam_cls = np.asarray(cam_cls, np.uint8)
    keep = ~np.isin(cam_keys, ml.keys) | (cam_cls == veh)
    mk = ~np.isin(ml.keys, cam_keys[cam_cls == veh])
    keys = np.concatenate([cam_keys[keep], ml.keys[mk]])
    cls = np.concatenate([cam_cls[keep], ml.cls[mk]])
    o = np.argsort(keys)
    return keys[o], cls[o]
