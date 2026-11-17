"""Adapter: sparse-occupancy perception outputs -> ``gvd_scene`` v1 message.

Schema v1 is frozen at ``docs/specs/viz_cpp.md`` @ f3555bf (branch
``gvd/cpp-viz``, PR #232): required keys ``v, type, seq, t, ego, aeb, lanes,
edges, tracks, signs, path, path_conf``. Every v1 field is filled with the
v1 shape. Things v1 has no field for ride as OPTIONAL keys the C++ parser
ignores (it reads only the v1 fields):

* ``occ``: ``{"voxel", "origin", "cells": [[ix, iy, iz, cls], ...]}`` sparse
  occupied cells with ``OCC_CLASS_NAMES`` classes;
* ``edges[].side`` / ``edges[].source`` / ``edges[].color``;
* ``lanes[].color`` (always equal to ``paint``: white or yellow);
* ``tracks[].motion`` (``same_way|oncoming|crossing|alongside|parked|unknown``)
  and ``tracks[].color`` from ``MOTION_COLORS``;
* ``crosswalks`` / ``stop_lines``: ``[{"pts": ...}]`` polylines, white.

Colour rule (Michael): white paint white, yellow paint yellow, RED ONLY for
road edges and curbs. A parked car is orange, never red.

Reads only the keys named in ``occ_to_viz``. Ground-truth keys in the input
(``gt_*``, ``*_gt``, labels) are never read.
"""
from __future__ import annotations

import json
import math
from typing import Any, Mapping

import numpy as np

SCHEMA_VERSION = 1
MAX_DATAGRAM = 60000
MAX_TRACKS = 64
MAX_LANES = 16
MAX_EDGES = 8
MAX_POINTS = 120
MAX_OCC_CELLS = 2500
V1_KEYS = ("v", "type", "seq", "t", "ego", "aeb", "lanes", "edges", "tracks", "signs", "path", "path_conf")
OCC_CLASS_NAMES = ("other", "drivable", "vehicle")
EDGE_COLOR = "red"
LANE_PAINTS = ("white", "yellow")
MOTION_COLORS = {
    "same_way": "green",     # legend "Longitudinal"
    "oncoming": "yellow",    # legend "Oncoming" (amber)
    "crossing": "purple",    # legend "Crossing"
    "alongside": "blue",     # legend "Lateral Forward" (inferred mapping)
    "parked": "orange",      # legend says red; Michael's rule: red is edges/curbs only
    "unknown": "blue",
}
MARK_COLOR = "white"         # crosswalks and stop lines


def _f(v: Any, nd: int = 2) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def _pts(src: Any) -> list[list[float]]:
    if isinstance(src, Mapping):
        src = src.get("pts") or src.get("points") or []
    out = []
    for p in src if isinstance(src, (list, tuple)) else []:
        if isinstance(p, Mapping):
            x, y = _f(p.get("x")), _f(p.get("y"))
        elif isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2:
            x, y = _f(p[0]), _f(p[1])
        else:
            continue
        if x is not None and y is not None:
            out.append([x, y])
    if len(out) > MAX_POINTS:
        step = len(out) / float(MAX_POINTS)
        out = [out[int(i * step)] for i in range(MAX_POINTS - 1)] + [out[-1]]
    return out


def _paint(rec: Any) -> str:
    """Yellow only when the record says yellow paint; otherwise white (v1 rule)."""
    if isinstance(rec, Mapping):
        for key in ("paint", "paint_color", "color", "colour", "material", "line_color"):
            raw = rec.get(key)
            if isinstance(raw, str) and "yellow" in raw.lower():
                return "yellow"
        if rec.get("yellow") is True:
            return "yellow"
    return "white"


def _occ_block(outputs: Mapping[str, Any]) -> dict[str, Any] | None:
    cells = outputs.get("cells")
    grid = outputs.get("grid")
    if not isinstance(cells, Mapping) or grid is None:
        return None
    keys = np.asarray(cells.get("keys", []), dtype=np.int64)
    cls = np.asarray(cells.get("cls", []), dtype=np.int64)
    if keys.size == 0:
        return {"voxel": float(grid.voxel_m), "origin": [float(grid.x_min), float(grid.y_min), float(grid.z_min)], "cells": []}
    # Nearest the ego first, so any cut (here or in ``encode``) drops far cells.
    ctr = grid.centers_of_keys(keys)
    order = np.argsort(np.hypot(ctr[:, 0], ctr[:, 1]), kind="stable")[:MAX_OCC_CELLS]
    keys, cls = keys[order], cls[order]
    c = grid.cell_of_key(keys)
    return {"voxel": float(grid.voxel_m), "origin": [float(grid.x_min), float(grid.y_min), float(grid.z_min)],
            "classes": list(OCC_CLASS_NAMES),
            "cells": [[int(a), int(b), int(z), int(k)] for (a, b, z), k in zip(c.tolist(), cls.tolist())]}


def occ_to_viz(outputs: Mapping[str, Any], *, seq: int = 0) -> dict[str, Any]:
    """v1 ``gvd_scene`` from ``occ_outputs`` (+ optional lanes / marks / path).

    Reads: ``t``, ``ego.{speed_mps,engaged,set_mps}``, ``cells``, ``grid``,
    ``edges``, ``curbs``, ``objects``, ``lanes``, ``crosswalks``,
    ``stop_lines``, ``path``, ``path_conf``, ``lead_id``. Nothing else.
    """
    o = outputs if isinstance(outputs, Mapping) else {}
    ego = o.get("ego") if isinstance(o.get("ego"), Mapping) else {}
    msg: dict[str, Any] = {
        "v": SCHEMA_VERSION, "type": "gvd_scene", "seq": int(seq), "t": _f(o.get("t"), 4) or 0.0,
        "ego": {"engaged": bool(ego.get("engaged")), "speed_mps": _f(ego.get("speed_mps")) or 0.0},
        "aeb": "off", "lanes": [], "edges": [], "tracks": [], "signs": [],
        "path": {"pts": _pts(o.get("path"))},
        "path_conf": _f(o.get("path_conf")) if _f(o.get("path_conf")) is not None else 1.0,
    }
    if _f(ego.get("set_mps")) is not None:
        msg["ego"]["set_mps"] = _f(ego.get("set_mps"))
    if isinstance(o.get("lead_id"), int) and not isinstance(o.get("lead_id"), bool):
        msg["lead_id"] = int(o["lead_id"])
    for group, kind in (("edges", "road_edge"), ("curbs", "curb")):
        for ed in list(o.get(group) or []):
            pts = _pts(ed)
            if len(pts) < 2 or len(msg["edges"]) >= MAX_EDGES:
                continue
            rec = {"pts": pts, "kind": "predicted", "class": kind, "color": EDGE_COLOR}
            if isinstance(ed, Mapping):
                for k in ("side", "source"):
                    if isinstance(ed.get(k), str):
                        rec[k] = ed[k]
            msg["edges"].append(rec)
    for ln in list(o.get("lanes") or [])[:MAX_LANES]:
        pts = _pts(ln)
        if len(pts) < 2:
            continue
        paint = _paint(ln)
        msg["lanes"].append({"pts": pts, "paint": paint, "color": paint, "kind": "detected"})
    for tr in list(o.get("objects") or []):
        if not isinstance(tr, Mapping) or len(msg["tracks"]) >= MAX_TRACKS:
            continue
        x, y = _f(tr.get("x")), _f(tr.get("y"))
        if x is None or y is None:
            continue
        motion = str(tr.get("motion") or "unknown")
        motion = motion if motion in MOTION_COLORS else "unknown"
        row: dict[str, Any] = {"cls": str(tr.get("cls") or "vehicle"), "x": x, "y": y, "motion": motion,
                               "color": MOTION_COLORS[motion]}
        if isinstance(tr.get("id"), int):
            row["id"] = int(tr["id"])
        for src, key, nd in (("yaw", "yaw", 3), ("length", "l", 2), ("width", "w", 2), ("height", "h", 2), ("speed_mps", "v", 2)):
            val = _f(tr.get(src), nd)
            if val is not None:
                row[key] = val
        msg["tracks"].append(row)
    for group in ("crosswalks", "stop_lines"):
        marks = [{"pts": p, "color": MARK_COLOR} for p in (_pts(m) for m in list(o.get(group) or [])) if len(p) >= 2]
        if marks:
            msg[group] = marks
    occ = _occ_block(o)
    if occ is not None:
        msg["occ"] = occ
    return msg


def encode(msg: Mapping[str, Any], max_bytes: int = MAX_DATAGRAM) -> bytes:
    """Compact JSON under one datagram: drops far occupancy cells first."""
    m = json.loads(json.dumps(msg))
    while True:
        data = json.dumps(m, separators=(",", ":"), allow_nan=False).encode("utf-8")
        cells = (m.get("occ") or {}).get("cells") or []
        if len(data) <= max_bytes or not cells:
            break
        m["occ"]["cells"] = cells[: len(cells) // 2]
    if len(data) > max_bytes:
        raise ValueError(f"scene message is {len(data)} bytes, over {max_bytes}")
    return data
