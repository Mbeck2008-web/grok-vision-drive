"""``occ_to_viz``: sparse occupancy outputs -> visualizer message schema v1.

v1 is frozen at ``f3555bf:docs/specs/viz_cpp.md`` (``python/viz/cpp_publisher.py``
on ``gvd/cpp-viz``). The constants below mirror it; a test compares them with
that commit when git can read it. v1 fields are filled from perception
outputs only. Extras the v1 parser ignores (it reads only its own keys):

* ``occ``: ``{"cells": [[x, y, z, cls], ...], "classes": [...], "rgb": {...}, "src": {...}, "n_total": N}``
* per edge ``type`` (``road_edge`` / ``curb``) and ``src``
* per track ``motion``, ``motion_frames``, ``src``, ``motion_src``
* ``unlabeled``: outputs the viz can draw that this net does not produce
  because no recorder label backs them (lane paint, crosswalks, stop lines, signs).
"""
from __future__ import annotations

import json
import math
from typing import Any, Mapping

import numpy as np

from python.occ.classes import CLASS_NAMES, CLASS_RGB, CLASS_SRC, KNOWN_SOURCES, N_CLASSES, RED_POLYLINE_KINDS, SRC_EDGE, SRC_MOTION, SRC_VEHICLE, UNLABELED, UNLABELED_OUTPUTS
from python.occ.grid import DEFAULT_GRID, VoxelGrid

SCHEMA_VERSION = 1
MAX_DATAGRAM = 60000
MAX_TRACKS = 64
MAX_SIGNS = 16
MAX_LANES = 16
MAX_EDGES = 8
MAX_POINTS = 120
MESSAGE_KEYS = frozenset(
    {"v", "type", "seq", "t", "ego", "lead_id", "aeb", "lanes", "edges", "tracks", "signs", "path", "path_conf"}
)
MAX_OCC_CELLS = 1500
# C++ reads seq as long long and ids as int.
INT64 = (-(2**63), 2**63)
INT32 = (-(2**31), 2**31)
MOTION_STATES = ("unknown", "parked", "same_way", "oncoming", "crossing", "alongside")


def _int_in(v: Any, rng: tuple[int, int]) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and rng[0] <= v < rng[1]


def _src(v: Any) -> str:
    return v if isinstance(v, str) and v in KNOWN_SOURCES else UNLABELED


class VizSchemaError(ValueError):
    pass


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _r(v: Any, nd: int = 2) -> float:
    return round(float(v), nd)


def _pts_ok(pts: Any) -> bool:
    return isinstance(pts, list) and all(isinstance(p, list) and len(p) == 2 and _num(p[0]) and _num(p[1]) for p in pts)


def validate_v1(msg: Any) -> list[str]:
    """Problems with ``msg`` as a v1 scene message (empty list = valid). Unknown keys are ignored."""
    errs: list[str] = []
    if not isinstance(msg, Mapping):
        return ["message is not an object"]
    if msg.get("type") != "gvd_scene":
        errs.append("type != gvd_scene")
    if not (isinstance(msg.get("v"), int) and not isinstance(msg.get("v"), bool) and msg.get("v") == 1):
        errs.append("v != 1")
    if not _int_in(msg.get("seq"), INT64):
        errs.append("seq not an int64")
    if not _num(msg.get("t")):
        errs.append("t not a finite number")
    ego = msg.get("ego")
    if not isinstance(ego, Mapping) or not isinstance(ego.get("engaged"), bool) or not _num(ego.get("speed_mps")):
        errs.append("ego.engaged/speed_mps")
    elif "set_mps" in ego and not _num(ego["set_mps"]):
        errs.append("ego.set_mps")
    if "lead_id" in msg and not _int_in(msg["lead_id"], INT32):
        errs.append("lead_id not an int32")
    if not isinstance(msg.get("aeb"), str):
        errs.append("aeb not a string")
    if not _num(msg.get("path_conf")):
        errs.append("path_conf")
    for key, cap in (("lanes", MAX_LANES), ("edges", MAX_EDGES), ("tracks", MAX_TRACKS), ("signs", MAX_SIGNS)):
        arr = msg.get(key)
        if not isinstance(arr, list):
            errs.append(f"{key} not a list")
            continue
        if len(arr) > cap:
            errs.append(f"{key} over cap {cap}")
    for i, ln in enumerate(msg.get("lanes") or []):
        if not isinstance(ln, Mapping) or not _pts_ok(ln.get("pts")) or not (2 <= len(ln["pts"]) <= MAX_POINTS):
            errs.append(f"lanes[{i}].pts")
        elif ln.get("paint") not in ("white", "yellow"):
            errs.append(f"lanes[{i}].paint")
    for i, ed in enumerate(msg.get("edges") or []):
        if not isinstance(ed, Mapping) or not _pts_ok(ed.get("pts")) or not (2 <= len(ed["pts"]) <= MAX_POINTS):
            errs.append(f"edges[{i}].pts")
    for i, tr in enumerate(msg.get("tracks") or []):
        if not isinstance(tr, Mapping) or not _num(tr.get("x")) or not _num(tr.get("y")) or not isinstance(tr.get("cls"), str):
            errs.append(f"tracks[{i}] x/y/cls")
            continue
        if "id" in tr and not _int_in(tr["id"], INT32):
            errs.append(f"tracks[{i}].id")
        for k in ("yaw", "l", "w", "h", "v"):
            if k in tr and not _num(tr[k]):
                errs.append(f"tracks[{i}].{k}")
    for i, sg in enumerate(msg.get("signs") or []):
        if not isinstance(sg, Mapping) or not _num(sg.get("x")) or not _num(sg.get("y")) or not sg.get("cls"):
            errs.append(f"signs[{i}]")
    path = msg.get("path")
    if not isinstance(path, Mapping) or not _pts_ok(path.get("pts")):
        errs.append("path.pts")
    try:
        size = len(json.dumps(msg, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        if size > MAX_DATAGRAM:
            errs.append(f"{size} bytes > {MAX_DATAGRAM}")
    except (TypeError, ValueError) as exc:
        errs.append(f"not JSON: {exc}")
    return errs


def strip_to_v1(msg: Mapping[str, Any]) -> dict[str, Any]:
    """Only the keys a v1 reader looks at (what ``cpp_publisher`` would send)."""
    return {k: v for k, v in msg.items() if k in MESSAGE_KEYS}


def _track_row(tr: Mapping[str, Any]) -> dict[str, Any] | None:
    if not (_num(tr.get("x")) and _num(tr.get("y"))):
        return None
    row: dict[str, Any] = {"cls": str(tr.get("cls") or "vehicle"), "x": _r(tr["x"]), "y": _r(tr["y"])}
    if _int_in(tr.get("id"), INT32):
        row["id"] = int(tr["id"])
    if _num(tr.get("yaw")):
        row["yaw"] = _r(tr["yaw"], 3)
    for k in ("l", "w", "h", "v"):
        if _num(tr.get(k)):
            row[k] = _r(tr[k])
    frames = tr.get("motion_frames")
    frames = int(frames) if _int_in(frames, INT32) and frames > 0 else 0
    motion = tr.get("motion")
    # A motion state needs a track over >= 2 frames; anything else is unknown.
    if motion not in MOTION_STATES or frames < 2:
        motion = "unknown"
    row["motion"] = motion
    row["motion_frames"] = frames
    row["motion_src"] = SRC_MOTION
    row["src"] = _src(tr.get("src", SRC_VEHICLE))
    return row


def occ_to_viz(outputs: Mapping[str, Any], *, seq: int = 0, ego: Mapping[str, Any] | None = None, grid: VoxelGrid = DEFAULT_GRID) -> dict[str, Any]:
    """Build a v1 message (plus documented extras) from one tick of occupancy outputs.

    ``outputs``: ``t`` (s), ``cells`` (``idx`` (N,3), ``cls`` (N,), ``conf`` (N,)),
    ``tracks`` (tracker rows), ``edges`` (``polylines.edges_from_cells``).
    """
    t = outputs.get("t", 0.0)
    if not _num(t):
        raise VizSchemaError("t must be a finite number")
    eg = dict(ego or {})
    speed = float(eg.get("speed_mps", 0.0)) if _num(eg.get("speed_mps")) else 0.0
    msg: dict[str, Any] = {
        "v": SCHEMA_VERSION,
        "type": "gvd_scene",
        "seq": int(seq) % INT64[1],
        "t": round(float(t), 4),
        "ego": {"engaged": bool(eg.get("engaged", False)), "speed_mps": _r(speed)},
        "aeb": "off",
        "lanes": [],
        "edges": [],
        "tracks": [],
        "signs": [],
        "path": {"pts": []},
        "path_conf": 1.0,
    }
    for ed in list(outputs.get("edges") or [])[:MAX_EDGES]:
        pts = [[_r(p[0]), _r(p[1])] for p in ed.get("pts", []) if _num(p[0]) and _num(p[1])][:MAX_POINTS]
        ty = ed.get("type", "road_edge")
        if len(pts) >= 2 and ty in RED_POLYLINE_KINDS:  # v1 edges draw red: only edges and curbs
            msg["edges"].append({"pts": pts, "kind": "predicted", "type": ty, "src": _src(ed.get("src", SRC_EDGE))})
    for tr in outputs.get("tracks") or []:
        row = _track_row(tr) if isinstance(tr, Mapping) else None
        if row is not None:
            msg["tracks"].append(row)
        if len(msg["tracks"]) >= MAX_TRACKS:
            break
    cells = outputs.get("cells") or {}
    idx = np.asarray(cells.get("idx", np.zeros((0, 3))), dtype=np.int64).reshape(-1, 3)
    cls = np.asarray(cells.get("cls", np.zeros(0)), dtype=np.int64).reshape(-1)
    conf = np.asarray(cells.get("conf", np.ones(cls.shape)), dtype=np.float64).reshape(-1)
    if not (idx.shape[0] == cls.shape[0] == conf.shape[0]):
        raise VizSchemaError("cells idx/cls/conf lengths differ")
    good = (idx >= 0).all(axis=1) & (idx < np.array(grid.shape)).all(axis=1) & (cls >= 0) & (cls < N_CLASSES)
    idx, cls = idx[good], cls[good]
    conf = np.nan_to_num(conf[good], nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    order = np.argsort(-conf, kind="stable")[:MAX_OCC_CELLS]
    xyz = grid.to_center(idx[order]) if order.size else np.zeros((0, 3))
    occ_cells = [[_r(p[0], 1), _r(p[1], 1), _r(p[2], 1), int(c)] for p, c in zip(xyz.tolist(), cls[order].tolist())]
    msg["occ"] = {
        "cells": occ_cells,
        "n_total": int(cls.shape[0]),
        "voxel": grid.voxel,
        "classes": list(CLASS_NAMES),
        "rgb": {str(k): list(v) for k, v in CLASS_RGB.items()},
        "src": {CLASS_NAMES[k]: v for k, v in CLASS_SRC.items()},
    }
    msg["unlabeled"] = {name: UNLABELED for name in UNLABELED_OUTPUTS}
    # Fit the datagram: thin occ cells first (they are extras), never v1 fields.
    while len(json.dumps(msg, separators=(",", ":"), allow_nan=False)) > MAX_DATAGRAM and msg["occ"]["cells"]:
        msg["occ"]["cells"] = msg["occ"]["cells"][: len(msg["occ"]["cells"]) // 2]
    return msg
