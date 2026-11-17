"""Road-edge and curb polylines derived from predicted occupied cells.

Road edges: for each forward row of the BEV, the ROAD run that contains (or
is nearest to) the ego's lateral position; its outer cell boundaries are the
left and right edge points. Curbs: CURB cells per side, median x per row.
Rows are chained into a polyline while consecutive rows are at most
``MAX_ROW_GAP`` apart and the lateral jump is under ``MAX_JUMP_M``. A chain
shorter than ``MIN_ROWS`` rows is dropped (no one-cell edges).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from python.occ.classes import CURB, LANE_WHITE, LANE_YELLOW, PAINT_CLASSES, ROAD, SRC_CURB_LINE, SRC_EDGE, SRC_LANE_LINE, SRC_STOP_LINE, STOP_LINE
from python.occ.grid import DEFAULT_GRID, VoxelGrid

MIN_ROWS = 3
MAX_ROW_GAP = 2
MAX_JUMP_M = 1.5
MAX_EDGES = 8
MAX_POINTS = 120


def _runs(xs: np.ndarray) -> list[tuple[int, int]]:
    xs = np.unique(xs)
    if xs.size == 0:
        return []
    cuts = np.nonzero(np.diff(xs) > 1)[0]
    starts = np.concatenate([[0], cuts + 1])
    ends = np.concatenate([cuts, [xs.size - 1]])
    return [(int(xs[a]), int(xs[b])) for a, b in zip(starts, ends)]


def _chain(rows: list[tuple[int, float]], grid: VoxelGrid) -> list[list[list[float]]]:
    out: list[list[list[float]]] = []
    cur: list[tuple[int, float]] = []
    for iy, x in rows:
        if cur and (iy - cur[-1][0] > MAX_ROW_GAP or abs(x - cur[-1][1]) > MAX_JUMP_M):
            out.append(cur)
            cur = []
        cur.append((iy, x))
    if cur:
        out.append(cur)
    lines = []
    for c in out:
        if len(c) < MIN_ROWS:
            continue
        pts = [[round(float(x), 2), round(float(grid.y_min + (iy + 0.5) * grid.voxel), 2)] for iy, x in c]
        if len(pts) > MAX_POINTS:
            step = len(pts) / float(MAX_POINTS)
            pts = [pts[int(i * step)] for i in range(MAX_POINTS - 1)] + [pts[-1]]
        lines.append(pts)
    return lines


def edges_from_cells(idx: np.ndarray, cls: np.ndarray, *, grid: VoxelGrid = DEFAULT_GRID, ego_x: float = 0.0) -> list[dict[str, Any]]:
    idx = np.asarray(idx, dtype=np.int64).reshape(-1, 3)
    cls = np.asarray(cls).reshape(-1)
    if idx.shape[0] != cls.shape[0]:
        raise ValueError("idx and cls lengths differ")
    inb = (idx >= 0).all(axis=1) & (idx < np.array(grid.shape)).all(axis=1)
    idx, cls = idx[inb], cls[inb]
    v = grid.voxel
    ego_ix = (ego_x - grid.x_min) / v - 0.5
    left: list[tuple[int, float]] = []
    right: list[tuple[int, float]] = []
    road = idx[(cls == ROAD) | np.isin(cls, list(PAINT_CLASSES))]  # paint lies on the road surface
    for iy in np.unique(road[:, 1]) if road.size else []:
        runs = _runs(road[road[:, 1] == iy][:, 0])
        a, b = min(runs, key=lambda r: 0.0 if r[0] <= ego_ix <= r[1] else min(abs(r[0] - ego_ix), abs(r[1] - ego_ix)))
        left.append((int(iy), grid.x_min + a * v))
        right.append((int(iy), grid.x_min + (b + 1) * v))
    out: list[dict[str, Any]] = []
    for side in (left, right):
        for pts in _chain(side, grid):
            out.append({"pts": pts, "kind": "predicted", "type": "road_edge", "src": SRC_EDGE})
    curb = idx[cls == CURB]
    centre_x = grid.x_min + (curb[:, 0] + 0.5) * v if curb.size else np.zeros(0)
    for sel in (centre_x < ego_x, centre_x >= ego_x):
        c, cx = curb[sel], centre_x[sel]
        rows = [(int(iy), float(np.median(cx[c[:, 1] == iy]))) for iy in np.unique(c[:, 1])] if c.size else []
        for pts in _chain(rows, grid):
            out.append({"pts": pts, "kind": "predicted", "type": "curb", "src": SRC_CURB_LINE})
    out.sort(key=lambda e: -len(e["pts"]))
    return out[:MAX_EDGES]


MAX_LANES = 16
LANE_JOIN_M = 0.75


def lanes_from_cells(idx: np.ndarray, cls: np.ndarray, *, grid: VoxelGrid = DEFAULT_GRID) -> list[dict[str, Any]]:
    """Lane-line polylines from LANE_WHITE / LANE_YELLOW cells.

    Per forward row, each run of paint cells gives one point at its centre.
    Points are chained row to row to the nearest open line within
    ``LANE_JOIN_M``; lines shorter than ``MIN_ROWS`` are dropped. Paint is the
    majority colour of the line's cells (yellow only when most are yellow).
    """
    idx = np.asarray(idx, dtype=np.int64).reshape(-1, 3)
    cls = np.asarray(cls).reshape(-1)
    if idx.shape[0] != cls.shape[0]:
        raise ValueError("idx and cls lengths differ")
    inb = (idx >= 0).all(axis=1) & (idx < np.array(grid.shape)).all(axis=1)
    sel = inb & ((cls == LANE_WHITE) | (cls == LANE_YELLOW))
    p, c = idx[sel], cls[sel]
    v = grid.voxel
    open_lines: list[dict[str, Any]] = []
    done: list[dict[str, Any]] = []
    for iy in (np.unique(p[:, 1]) if p.size else []):
        row = p[:, 1] == iy
        pts_here = []
        for a, b in _runs(p[row][:, 0]):
            in_run = row & (p[:, 0] >= a) & (p[:, 0] <= b)
            pts_here.append((grid.x_min + (a + b + 1) / 2 * v, int((c[in_run] == LANE_YELLOW).sum()), int(in_run.sum())))
        used = set()
        for ln in open_lines:
            if iy - ln["rows"][-1][0] > MAX_ROW_GAP:
                continue
            best = None
            for k, (x, _, _) in enumerate(pts_here):
                if k in used or abs(x - ln["rows"][-1][1]) > LANE_JOIN_M:
                    continue
                if best is None or abs(x - ln["rows"][-1][1]) < abs(pts_here[best][0] - ln["rows"][-1][1]):
                    best = k
            if best is not None:
                used.add(best)
                x, ny, n = pts_here[best]
                ln["rows"].append((int(iy), x))
                ln["yellow"] += ny
                ln["n"] += n
        still = []
        for ln in open_lines:
            (still if iy - ln["rows"][-1][0] <= MAX_ROW_GAP else done).append(ln)
        open_lines = still
        for k, (x, ny, n) in enumerate(pts_here):
            if k not in used:
                open_lines.append({"rows": [(int(iy), x)], "yellow": ny, "n": n})
    out = []
    for ln in done + open_lines:
        for pts in _chain(ln["rows"], grid):
            out.append({"pts": pts, "paint": "yellow" if ln["yellow"] * 2 > ln["n"] else "white", "kind": "predicted", "src": SRC_LANE_LINE})
    out.sort(key=lambda e: -len(e["pts"]))
    return out[:MAX_LANES]


def stop_lines_from_cells(idx: np.ndarray, cls: np.ndarray, *, grid: VoxelGrid = DEFAULT_GRID) -> list[dict[str, Any]]:
    """Stop bars ``{x0, x1, y, src}`` from STOP_LINE cells: one bar per run of >= 3 cells in a row."""
    idx = np.asarray(idx, dtype=np.int64).reshape(-1, 3)
    cls = np.asarray(cls).reshape(-1)
    inb = (idx >= 0).all(axis=1) & (idx < np.array(grid.shape)).all(axis=1)
    p = idx[inb & (cls == STOP_LINE)]
    v = grid.voxel
    out = []
    for iy in (np.unique(p[:, 1]) if p.size else []):
        for a, b in _runs(p[p[:, 1] == iy][:, 0]):
            if b - a + 1 >= MIN_ROWS:
                out.append({"x0": round(grid.x_min + a * v, 2), "x1": round(grid.x_min + (b + 1) * v, 2),
                            "y": round(grid.y_min + (iy + 0.5) * v, 2), "src": SRC_STOP_LINE})
    # merge bars in adjacent rows (a 0.5 m bar can span two rows)
    merged: list[dict[str, Any]] = []
    for bar in sorted(out, key=lambda r: (r["y"], r["x0"])):
        if merged and abs(bar["y"] - merged[-1]["y"]) <= v + 1e-6 and bar["x0"] <= merged[-1]["x1"] and bar["x1"] >= merged[-1]["x0"]:
            m = merged[-1]
            m["x0"], m["x1"], m["y"] = min(m["x0"], bar["x0"]), max(m["x1"], bar["x1"]), round((m["y"] + bar["y"]) / 2, 2)
            continue
        merged.append(bar)
    return merged[:16]
