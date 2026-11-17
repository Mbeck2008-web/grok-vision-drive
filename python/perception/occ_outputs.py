"""Runtime perception outputs derived from the sparse occupancy prediction.

Camera-only: every function here takes the net's ``OccPrediction`` (sparse
fine cell keys + class) and ego kinematics. No depth, semantic, or box input.

* ``road_edges``: left/right drivable -> non-drivable transition of the
  drivable run nearest ``x = 0``, one point per 0.4 m row ahead and behind,
  median-smoothed over 5 rows (see ``road_edges``).
* ``objects``: 8-connected BEV clusters of vehicle cells -> centre, footprint
  (PCA extent), height, heading (PCA major axis, +-pi ambiguous, and only the
  visible face when the cameras see one side: ``yaw_src = "pca"``). A moving
  track replaces it with the velocity heading (``yaw_src = "track"``).
* ``OccTracker``: nearest-neighbour association over frames with ego-motion
  compensation; velocity is the least-squares slope over the last
  ``MIN_FRAMES``+ positions. Motion state needs at least ``MIN_FRAMES``
  frames and is ``unknown`` before that: a single frame never sets it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from python.perception.occ_geometry import DEFAULT_GRID, OccGrid

DRIVABLE, VEHICLE = 1, 2
EDGE_MIN_RUN = 3          # cells: a drivable run narrower than 1.2 m is not road
EDGE_SMOOTH = 5           # rows in the median filter
EDGE_GAP = 6              # unobserved cells bridged inside a drivable run (ego clear box is 5.5)
OBJ_MIN_CELLS = 4
MIN_FRAMES = 3
GATE_M = 3.0
MAX_MISSES = 2
PARKED_MPS = 0.7
MOTION_STATES = ("unknown", "same_way", "oncoming", "crossing", "alongside", "parked")


def _bev(keys: np.ndarray, grid: OccGrid) -> np.ndarray:
    c = grid.cell_of_key(np.asarray(keys, dtype=np.int64))
    return np.unique(c[:, :2], axis=0) if c.size else np.zeros((0, 2), np.int64)


def _runs(xs: np.ndarray, blocked: set[int], gap: int) -> list[tuple[int, int]]:
    """Contiguous runs of sorted cells; gaps up to ``gap`` with no blocked cell are bridged."""
    runs: list[list[int]] = []
    for x in xs.tolist():
        if runs and x - runs[-1][1] - 1 <= gap and not any(g in blocked for g in range(runs[-1][1] + 1, x)):
            runs[-1][1] = x
        else:
            runs.append([x, x])
    return [(a, b) for a, b in runs]


def road_edges(keys: np.ndarray, cls: np.ndarray, *, grid: OccGrid = DEFAULT_GRID) -> list[dict[str, Any]]:
    """Left and right road-edge polylines (ego x right, y forward, metres).

    Per 0.4 m row: drivable runs (unobserved gaps up to ``EDGE_GAP`` cells
    bridged), the run nearest ``x = 0``; its left/right boundary is an edge
    point only when the next cell outward is an observed non-drivable cell
    (a real drivable -> other transition, not the end of what the cameras
    saw). Then a median filter and a 1 m outlier cut per side.
    """
    keys = np.asarray(keys, dtype=np.int64)
    cls = np.asarray(cls)
    drv = _bev(keys[cls == DRIVABLE], grid)
    if drv.shape[0] == 0:
        return []
    oth = _bev(keys[cls != DRIVABLE], grid)
    v = float(grid.voxel_m)
    x0 = grid.x_min
    ego_ix = (0.0 - x0) / v
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    for iy in np.unique(drv[:, 1]):
        xs = np.sort(drv[drv[:, 1] == iy, 0])
        blocked = set(oth[oth[:, 1] == iy, 0].tolist())
        runs = [(a, b) for a, b in _runs(xs, blocked, EDGE_GAP) if b - a + 1 >= EDGE_MIN_RUN]
        if not runs:
            continue
        dist = [0.0 if a <= ego_ix <= b + 1 else min(abs(a - ego_ix), abs(b + 1 - ego_ix)) for a, b in runs]
        a, b = runs[int(np.argmin(dist))]
        y = grid.y_min + (iy + 0.5) * v
        if (a - 1) in blocked or (a - 2) in blocked:
            left.append((x0 + a * v, y))
        if (b + 1) in blocked or (b + 2) in blocked:
            right.append((x0 + (b + 1) * v, y))
    out = []
    for side, pts in (("left", left), ("right", right)):
        if len(pts) < 2:
            continue
        a = np.asarray(pts)
        k = EDGE_SMOOTH // 2
        med = np.array([float(np.median(a[max(0, i - k) : i + k + 1, 0])) for i in range(a.shape[0])])
        keep = np.abs(a[:, 0] - med) <= 1.0
        a, med = a[keep], med[keep]
        if a.shape[0] < 2:
            continue
        out.append({"side": side, "kind": "road_edge", "source": "occ_drivable",
                    "pts": [[round(float(x), 2), round(float(y), 2)] for x, y in zip(med, a[:, 1])]})
    return out


def _clusters(cells: np.ndarray) -> list[np.ndarray]:
    """8-connected components of BEV cells (N,2) -> index arrays."""
    if cells.shape[0] == 0:
        return []
    lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(cells)}
    seen = np.zeros(cells.shape[0], dtype=bool)
    out = []
    for s in range(cells.shape[0]):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            i = stack.pop()
            comp.append(i)
            a, b = int(cells[i, 0]), int(cells[i, 1])
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    j = lookup.get((a + da, b + db))
                    if j is not None and not seen[j]:
                        seen[j] = True
                        stack.append(j)
        out.append(np.asarray(comp))
    return out


def objects(keys: np.ndarray, cls: np.ndarray, *, grid: OccGrid = DEFAULT_GRID) -> list[dict[str, Any]]:
    """Vehicle clusters: centre x/y, z0, length/width/height, heading (rad from +x)."""
    keys = np.asarray(keys, dtype=np.int64)
    vk = keys[np.asarray(cls) == VEHICLE]
    if vk.size == 0:
        return []
    c3 = grid.cell_of_key(vk)
    bev = _bev(vk, grid)
    v = float(grid.voxel_m)
    out = []
    for comp in _clusters(bev):
        if comp.size < OBJ_MIN_CELLS:
            continue
        cells = bev[comp]
        xy = grid.lo[:2] + (cells + 0.5) * v
        ctr = xy.mean(axis=0)
        cov = np.cov((xy - ctr).T) if comp.size > 1 else np.eye(2)
        w_, vecs = np.linalg.eigh(cov)
        major = vecs[:, int(np.argmax(w_))]
        yaw = math.atan2(major[1], major[0])
        if yaw <= -math.pi / 2:
            yaw += math.pi
        elif yaw > math.pi / 2:
            yaw -= math.pi
        proj_l = (xy - ctr) @ major
        proj_w = (xy - ctr) @ np.array([-major[1], major[0]])
        inside = np.isin(c3[:, 0] * 100000 + c3[:, 1], cells[:, 0] * 100000 + cells[:, 1])
        zs = grid.z_min + (c3[inside, 2] + 0.5) * v
        out.append({"cls": "vehicle", "x": float(ctr[0]), "y": float(ctr[1]), "yaw": float(yaw), "yaw_src": "pca",
                    "length": float(proj_l.max() - proj_l.min() + v), "width": float(proj_w.max() - proj_w.min() + v),
                    "height": float(zs.max() - zs.min() + v), "cells": int(comp.size)})
    return out


def motion_state(vx: float, vy: float, x: float, y: float) -> str:
    """World velocity in the ego frame (+y = ego heading) -> legend state."""
    speed = math.hypot(vx, vy)
    if speed < PARKED_MPS:
        return "parked"
    ang = abs(math.degrees(math.atan2(vx, vy)))  # 0 = same way as ego, 180 = toward ego
    if ang <= 35.0:
        return "alongside" if abs(x) > 1.8 and abs(y) < 6.0 else "same_way"
    if ang >= 145.0:
        return "oncoming"
    if 55.0 <= ang <= 125.0:
        return "crossing"
    return "same_way" if ang < 90.0 else "oncoming"


@dataclass
class _Track:
    tid: int
    hist: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y) in the current ego frame
    obj: dict[str, Any] = field(default_factory=dict)
    misses: int = 0


class OccTracker:
    """Multi-frame tracks of occupancy objects. Motion state only after ``MIN_FRAMES``."""

    def __init__(self, *, min_frames: int = MIN_FRAMES, gate_m: float = GATE_M) -> None:
        self.min_frames = int(min_frames)
        self.gate_m = float(gate_m)
        self.tracks: list[_Track] = []
        self._next = 1
        self._t: float | None = None

    def _compensate(self, dt: float, speed: float, yaw_rate: float) -> None:
        """Re-express stored positions in the new ego frame after the ego moved."""
        dyaw = yaw_rate * dt
        fwd = speed * dt
        c, s = math.cos(dyaw), math.sin(dyaw)
        for tr in self.tracks:
            new = []
            for t, x, y in tr.hist:
                y2 = y - fwd
                # ego turned left by dyaw (CCW): world points rotate clockwise in the ego frame
                new.append((t, c * x + s * y2, -s * x + c * y2))
            tr.hist = new

    def update(self, objs: Sequence[Mapping[str, Any]], t: float, *, ego_speed_mps: float = 0.0, ego_yaw_rate: float = 0.0) -> list[dict[str, Any]]:
        if self._t is not None:
            self._compensate(max(0.0, float(t) - self._t), float(ego_speed_mps), float(ego_yaw_rate))
        self._t = float(t)
        free = list(range(len(objs)))
        pairs = []
        for ti, tr in enumerate(self.tracks):
            _, px, py = tr.hist[-1]
            for oi in free:
                d = math.hypot(objs[oi]["x"] - px, objs[oi]["y"] - py)
                if d <= self.gate_m:
                    pairs.append((d, ti, oi))
        used_t, used_o = set(), set()
        for d, ti, oi in sorted(pairs):
            if ti in used_t or oi in used_o:
                continue
            used_t.add(ti); used_o.add(oi)
            tr = self.tracks[ti]
            tr.hist = (tr.hist + [(float(t), float(objs[oi]["x"]), float(objs[oi]["y"]))])[-10:]
            tr.obj = dict(objs[oi]); tr.misses = 0
        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.misses += 1
        self.tracks = [tr for tr in self.tracks if tr.misses <= MAX_MISSES]
        for oi in free:
            if oi not in used_o:
                self.tracks.append(_Track(self._next, [(float(t), float(objs[oi]["x"]), float(objs[oi]["y"]))], dict(objs[oi])))
                self._next += 1
        return self.report()

    def report(self) -> list[dict[str, Any]]:
        out = []
        for tr in self.tracks:
            if tr.misses:
                continue
            row = dict(tr.obj)
            row["id"] = tr.tid
            row["frames"] = len(tr.hist)
            row["motion"] = "unknown"
            if len(tr.hist) >= self.min_frames:
                h = np.asarray(tr.hist)
                tt = h[:, 0] - h[:, 0].mean()
                den = float((tt ** 2).sum())
                if den > 0:
                    vx = float((tt * (h[:, 1] - h[:, 1].mean())).sum() / den)
                    vy = float((tt * (h[:, 2] - h[:, 2].mean())).sum() / den)
                    row["vx"], row["vy"] = vx, vy
                    row["speed_mps"] = math.hypot(vx, vy)
                    row["motion"] = motion_state(vx, vy, row["x"], row["y"])
                    if row["speed_mps"] >= PARKED_MPS:  # moving: heading follows velocity, not PCA
                        row["yaw"] = math.atan2(vy, vx)
                        row["yaw_src"] = "track"
            out.append(row)
        return out


def occ_outputs(pred: Any, tracker: OccTracker, t: float, *, ego_speed_mps: float = 0.0, ego_yaw_rate: float = 0.0,
                grid: OccGrid = DEFAULT_GRID) -> dict[str, Any]:
    """Everything the viz adapter needs from one ``OccPrediction``."""
    objs = objects(pred.keys, pred.cls, grid=grid)
    return {
        "t": float(t),
        "grid": grid,
        "cells": {"keys": np.asarray(pred.keys), "cls": np.asarray(pred.cls)},
        "edges": road_edges(pred.keys, pred.cls, grid=grid),
        "objects": tracker.update(objs, t, ego_speed_mps=ego_speed_mps, ego_yaw_rate=ego_yaw_rate),
        "ego": {"speed_mps": float(ego_speed_mps)},
    }
