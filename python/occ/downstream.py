"""Everything after the heads: tracker, polylines, occ_out v1, viz message. Head outputs only.

This module (and everything it imports) never sees a camera image, a torch
tensor or a ground-truth pose: its input is the ``heads`` dict from
``OccRuntime.perceive`` (numpy cells, object list, the VO estimate, sparse
camera panels, timers). Camera *geometry* (``python.occ.camera``, numpy) is
used only to project the 3-D boxes into the panels. The viz message is built
from ``occ_out`` (``occ_out_to_outputs`` -> ``occ_to_viz``), so every debug
view is drawn from model outputs. Tests poison images and pose fields.
"""
from __future__ import annotations

import math
import time
from typing import Any, Mapping

from python.occ.grid import VoxelGrid
from python.occ.objects import objects_from_cells
from python.occ.occ_out import build_occ_out, check_occ_out, occ_out_to_outputs
from python.occ.polylines import edges_from_cells, lanes_from_cells, stop_lines_from_cells
from python.occ.tracks import MotionTracker
from python.occ.viz_adapter import occ_to_viz

HEAD_KEYS = frozenset({"t", "cells", "objects", "motion_used", "stats", "engaged", "speed_mps", "vo", "vo_conf", "vo_cov", "memory_pts", "cam_panels",
                       "timing_ms", "ego_from_vehicle"})


class Downstream:
    def __init__(self, grid: VoxelGrid, cams: Mapping[str, Any] | None = None, *, cpu_mem_est_mb: float | None = None):
        self.grid = grid
        self.cams = dict(cams or {})
        self.cpu_mem_est_mb = cpu_mem_est_mb
        self.tracker = MotionTracker()
        self.seq = 0
        self.t_prev: float | None = None

    def __call__(self, heads: Mapping[str, Any]) -> dict[str, Any]:
        extra = set(heads) - HEAD_KEYS
        if extra:
            raise KeyError(f"downstream takes head outputs only, got extra keys {sorted(extra)}")
        t = float(heads["t"])
        cells = heads["cells"]
        motion = heads.get("motion_used")
        lat = dict(heads.get("timing_ms") or {})
        t0 = time.perf_counter()
        dets = _with_extents(heads.get("objects") or [], objects_from_cells(cells["idx"], cells["cls"], cells.get("conf"), grid=self.grid))
        tracks = self.tracker.update(t, dets, motion=motion if motion is not None else (float("nan"),) * 3)
        t1 = time.perf_counter()
        edges = edges_from_cells(cells["idx"], cells["cls"], grid=self.grid)
        lanes = lanes_from_cells(cells["idx"], cells["cls"], grid=self.grid)
        stops = stop_lines_from_cells(cells["idx"], cells["cls"], grid=self.grid)
        t2 = time.perf_counter()
        lat["tracker"] = (t1 - t0) * 1e3
        lat["polylines"] = (t2 - t1) * 1e3
        dt = None if self.t_prev is None or not math.isfinite(t) else t - self.t_prev  # frame stamps, never a constant
        occ_out = build_occ_out(t=t, seq=self.seq, heads=heads, tracks=tracks, lanes=lanes, edges=edges, stop_lines=stops, grid=self.grid, cams=self.cams,
                                dt=dt, ego_from_vehicle=bool(heads.get("ego_from_vehicle", False)), latency_ms=lat, cpu_mem_est_mb=self.cpu_mem_est_mb)
        occ_out["debug"]["latency_ms"]["build"] = (time.perf_counter() - t2) * 1e3
        check_occ_out(occ_out)
        outputs = {"t": t, "cells": cells, "tracks": tracks, "edges": edges, "lanes": lanes, "stop_lines": stops}
        sp = occ_out["ego"]["speed"]
        ego = {"engaged": bool(heads.get("engaged", False)), "speed_mps": sp["value"] if sp["valid"] else 0.0}
        viz = occ_to_viz(occ_out_to_outputs(occ_out), seq=self.seq, ego=ego, grid=self.grid)
        self.seq += 1
        self.t_prev = t if math.isfinite(t) else None
        return {"outputs": outputs, "viz": viz, "occ_out": occ_out}


def _with_extents(objs, clusters, max_d: float = 2.5):
    """Object-head detections with l/w/h from the nearest VEHICLE cell cluster (if any within max_d)."""
    out = []
    for o in objs:
        row = dict(o)
        best = min(clusters, key=lambda c: math.hypot(c["x"] - o["x"], c["y"] - o["y"]), default=None)
        if best is not None and math.hypot(best["x"] - o["x"], best["y"] - o["y"]) <= max_d:
            row.update({"l": best["l"], "w": best["w"], "h": best["h"]})
        out.append(row)
    return out
