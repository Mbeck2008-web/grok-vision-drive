"""Frozen occ_out v2 field set (gvd/occ-sparse-s2 b939818).

The canonical module is python/perception/occ_out.py on the occupancy swarm
branch and is not in this tree. This snapshot is the field set the C++ viz
publisher adapter reads. scripts/test_viz_cpp_publisher.py checks the frozen
keys against a copy of the real module when one is on PYTHONPATH or at
GVD_OCC_OUT_CONTRACT (it must not edit that tree).
"""
from __future__ import annotations

from typing import Any, Mapping

# Source of the freeze: origin/gvd/occ-sparse-s2
OCC_OUT_CONTRACT_SHA = "b939818"
OCC_OUT_VERSION = 2
OCC_OUT_VERSIONS = (1, 2)
OCC_CLASS_NAMES = ("other", "drivable", "vehicle")
LANE_PAINTS = ("white", "yellow")
EDGE_KINDS = ("road_edge", "curb")
EDGE_SIDES = ("left", "right")
CAM_IDS = ("narrow", "main", "wide", "pillarL", "pillarR", "repeatL", "repeatR", "rear")
EGO_SOURCES = ("vo", "vehicle")

TOP_KEYS = (
    "v", "type", "seq", "t", "grid", "counts", "cells", "edges", "lanes",
    "corridor", "objects", "boxes2d", "cams", "ego", "timing_ms",
    "peak_vram_mb", "vram_status", "vram_est_mb_cpu",
)
GRID_KEYS = ("voxel_m", "origin", "shape")
COUNT_KEYS = ("full", "frustum", "queried", "stored", "stage_queries", "cells_out", "objects", "edges")
CELL_KEYS = ("ijk", "cls", "prob", "observed")
EDGE_KEYS = ("side", "kind", "pts", "conf", "observed")
LANE_KEYS = ("paint", "pts", "conf", "observed")
EGO_KEYS = ("speed_mps", "yaw_rate_rps", "source", "ok", "dt_s")
CAM_KEYS = ("status", "age_s", "weight", "res")
TIMING_KEYS = ("predict", "outputs", "total")
CORRIDOR_KEYS = ("pts", "conf")

# Default OccGrid on b939818 (python/perception/occ_geometry.py).
DEFAULT_VOXEL_M = 0.4
DEFAULT_ORIGIN = (-25.6, -12.8, -2.0)
DEFAULT_SHAPE = (128, 160, 16)  # prod = 327_680


def cell_center(ijk: list[int] | tuple[int, ...], grid: Mapping[str, Any]) -> list[float] | None:
    """Ego-frame centre of a cell: origin + (ijk + 0.5) * voxel_m."""
    try:
        origin = list(grid["origin"])
        voxel = float(grid["voxel_m"])
        if len(ijk) < 2 or len(origin) < 2 or voxel <= 0:
            return None
        n = min(len(ijk), len(origin), 3)
        return [float(origin[a]) + (int(ijk[a]) + 0.5) * voxel for a in range(n)]
    except (TypeError, ValueError, KeyError):
        return None


def frozen_field_set() -> dict[str, tuple[str, ...]]:
    return {
        "TOP_KEYS": TOP_KEYS,
        "GRID_KEYS": GRID_KEYS,
        "COUNT_KEYS": COUNT_KEYS,
        "CELL_KEYS": CELL_KEYS,
        "EDGE_KEYS": EDGE_KEYS,
        "LANE_KEYS": LANE_KEYS,
        "EGO_KEYS": EGO_KEYS,
        "OCC_CLASS_NAMES": OCC_CLASS_NAMES,
        "LANE_PAINTS": LANE_PAINTS,
        "EDGE_KINDS": EDGE_KINDS,
        "CAM_IDS": CAM_IDS,
    }


def load_live_field_set(mod: Any) -> dict[str, tuple[str, ...]]:
    return {
        "TOP_KEYS": tuple(mod.TOP_KEYS),
        "GRID_KEYS": tuple(mod.GRID_KEYS),
        "COUNT_KEYS": tuple(mod.COUNT_KEYS),
        "CELL_KEYS": tuple(mod.CELL_KEYS),
        "EDGE_KEYS": tuple(mod.EDGE_KEYS),
        "LANE_KEYS": tuple(mod.LANE_KEYS),
        "EGO_KEYS": tuple(mod.EGO_KEYS),
        "OCC_CLASS_NAMES": tuple(mod.OCC_CLASS_NAMES),
        "LANE_PAINTS": tuple(mod.LANE_PAINTS),
        "EDGE_KINDS": tuple(mod.EDGE_KINDS),
        "CAM_IDS": tuple(mod.CAM_IDS),
    }


def make_occ_out_v2(
    *,
    cells: dict[str, Any] | None = None,
    edges: list[dict[str, Any]] | None = None,
    lanes: list[dict[str, Any]] | None = None,
    seq: int = 1,
    t: float = 0.0,
) -> dict[str, Any]:
    """A fully-keyed v2 message the real validator would accept (no objects)."""
    edges = list(edges or [])
    lanes = list(lanes or [])
    if cells is None:
        cells = {"ijk": [], "cls": [], "prob": [], "observed": []}
    n = len(cells.get("ijk") or [])
    shape = list(DEFAULT_SHAPE)
    full = int(shape[0] * shape[1] * shape[2])
    queried = max(n, 1)
    stored = max(n, 1)
    cams = {cid: {"status": "ABSENT", "age_s": None, "weight": 0.0, "res": None} for cid in CAM_IDS}
    return {
        "v": OCC_OUT_VERSION,
        "type": "occ_out",
        "seq": int(seq),
        "t": float(t),
        "grid": {"voxel_m": DEFAULT_VOXEL_M, "origin": list(DEFAULT_ORIGIN), "shape": shape},
        "counts": {
            "full": full,
            "frustum": 0,
            "queried": queried,
            "stored": stored,
            "stage_queries": [queried],
            "cells_out": n,
            "objects": 0,
            "edges": len(edges),
        },
        "cells": cells,
        "edges": edges,
        "lanes": lanes,
        "corridor": {"pts": [], "conf": 0.0},
        "objects": [],
        "boxes2d": {},
        "cams": cams,
        "ego": {"speed_mps": 0.0, "yaw_rate_rps": 0.0, "source": "vo", "ok": True, "dt_s": 0.1},
        "timing_ms": {"predict": 0.0, "outputs": 0.0, "total": 0.0},
        "peak_vram_mb": None,
        "vram_status": "UNPROVEN",
        "vram_est_mb_cpu": None,
    }
