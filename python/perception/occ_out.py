"""``occ_out`` v2: the occupancy runtime's output contract (camera-only).

v2 = v1 plus exactly one field per element type: ``observed`` (bool) per
occupancy cell (``cells.observed``) and per polyline point
(``edges[].observed``, ``lanes[].observed``): true when a LIVE camera's ray
reaches it this tick with no predicted occluder in front; false for cells the
net decoded without a ray (occluded / out of view) or carried by the VO-warped
memory. Everything else is v1. The validator is version-gated: a v1 message
must not carry ``observed``, a v2 message must.

One dict per tick, built by ``OccRuntime.step``. Strict: ``validate_occ_out``
raises ``OccOutError`` on a missing field, an unknown field (any level), a
wrong type or shape, NaN/inf, a probability or confidence outside [0, 1], an
unknown class / lane paint / motion / camera status, a count field that
disagrees with its array, ``stored > queried > full``, a heading that is not a
unit vector, a polyline with < 2 points or a ``conf`` list of another length,
or any array over its cap. New fields need a version bump.

Fields and the debug-view panel that uses them
(``gvd-viz-refs/label-video/debug_view_multipanel.png``):

* 3-D grid panel: ``cells`` (sparse ijk + class + prob), ``edges`` (red dots),
  ``lanes`` (lane-dot chains; always empty here: paint is UNLABELED);
* camera panels: ``cams`` (all 8, ``LIVE``/``STALE``/``ABSENT``; a masked or
  missing camera is ``ABSENT``, never zeros), ``boxes2d`` (projections of the
  3-D object boxes into each LIVE camera, clipped to the image, same track id
  and class), HUD: ``ego``, ``timing_ms``, ``counts``;
* top-down map: ``corridor`` (drivable polygon, ego frame, implicitly closed,
  <= ``MAX_CORRIDOR_VERTS``; empty when confidence is low), ``objects`` (id,
  centre, size, ``heading``), ego car at the origin;
* graphs: ``ego.speed_mps`` and ``ego.yaw_rate_rps`` per tick.

Units: metres, seconds, radians; ego frame +x right, +y forward, +z up; yaw
rate left (CCW) positive. ``heading`` is ``[cos, sin]`` of the object's
direction from +x: the box (PCA) axis while ``heading_src == "box"``
(stationary or too few frames; +-pi ambiguous), the track velocity once moving
(``"velocity"``). There is no separate ``yaw`` field.

``peak_vram_mb`` is ``None`` (``vram_status = "UNPROVEN"``) until GVD posts
``nvidia-smi`` from Desktop-IQU45; the arithmetic CPU-side estimate is the
separate ``vram_est_mb_cpu`` and never fills ``peak_vram_mb``.
"""
from __future__ import annotations

import json
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from python.perception.occ_geometry import DEFAULT_GRID, OccGrid, PinholeCam, project

OCC_OUT_VERSION = 2          # v2 = v1 + ``observed`` per cell and per polyline point
OCC_OUT_VERSIONS = (1, 2)
MAX_MEMORY = 16384
CAM_IDS = ("narrow", "main", "wide", "pillarL", "pillarR", "repeatL", "repeatR", "rear")
OCC_CLASS_NAMES = ("other", "drivable", "vehicle")
OBJECT_CLASSES = ("vehicle",)
LANE_PAINTS = ("white", "yellow")
EDGE_KINDS = ("road_edge", "curb")
EDGE_SIDES = ("left", "right")
MOTION_STATES = ("unknown", "same_way", "oncoming", "crossing", "alongside", "parked")
CAM_STATUS = ("LIVE", "STALE", "ABSENT")
EGO_SOURCES = ("vo", "vehicle")
HEADING_SRC = ("box", "velocity")
MAX_CELLS = 4096
MAX_EDGES = 8
MAX_LANES = 16
MAX_POINTS = 120
MAX_OBJECTS = 64
MAX_BOXES_PER_CAM = 32
MAX_CORRIDOR_VERTS = 64
CORRIDOR_MIN_CONF = 0.5
STALE_WEIGHT = 0.5
MAX_BYTES = 200_000

TOP_KEYS = ("v", "type", "seq", "t", "grid", "counts", "cells", "edges", "lanes", "corridor", "objects", "boxes2d",
            "cams", "ego", "timing_ms", "peak_vram_mb", "vram_status", "vram_est_mb_cpu")
GRID_KEYS = ("voxel_m", "origin", "shape")
COUNT_KEYS = ("full", "frustum", "queried", "stored", "stage_queries", "cells_out", "objects", "edges")
CELL_KEYS_V1 = ("ijk", "cls", "prob")
EDGE_KEYS_V1 = ("side", "kind", "pts", "conf")
LANE_KEYS_V1 = ("paint", "pts", "conf")
# v2 adds exactly one field per element type: observed (true = a LIVE camera ray reaches it this tick)
CELL_KEYS = CELL_KEYS_V1 + ("observed",)
EDGE_KEYS = EDGE_KEYS_V1 + ("observed",)
LANE_KEYS = LANE_KEYS_V1 + ("observed",)
KEYS_BY_VERSION = {1: (CELL_KEYS_V1, EDGE_KEYS_V1, LANE_KEYS_V1), 2: (CELL_KEYS, EDGE_KEYS, LANE_KEYS)}
CORRIDOR_KEYS = ("pts", "conf")
OBJECT_KEYS = ("id", "cls", "x", "y", "z0", "l", "w", "h", "heading", "heading_src", "motion", "speed_mps", "frames")
BOX_KEYS = ("id", "cls", "box")
CAM_KEYS = ("status", "age_s", "weight", "res")
EGO_KEYS = ("speed_mps", "yaw_rate_rps", "source", "ok", "dt_s")
TIMING_KEYS = ("predict", "outputs", "total")


PANEL_FIELDS = {  # debug_view_multipanel.png panel -> occ_out fields it draws from
    "grid3d": ("cells.ijk", "cells.cls", "cells.prob", "grid.voxel_m", "grid.origin", "edges.pts", "lanes.pts", "lanes.paint"),
    "cameras": ("cams.status", "cams.age_s", "cams.res", "boxes2d.id", "boxes2d.cls", "boxes2d.box",
                "ego.speed_mps", "ego.source", "timing_ms.total", "counts.stored", "counts.full"),
    "topdown": ("corridor.pts", "corridor.conf", "objects.id", "objects.x", "objects.y", "objects.l", "objects.w",
                "objects.heading", "objects.motion"),
    "graphs": ("t", "ego.speed_mps", "ego.yaw_rate_rps", "ego.dt_s"),
}


def field_present(msg: Mapping[str, Any], path: str) -> bool:
    """``a.b`` resolves in msg; through a list or a per-camera map, in every element."""
    def walk(node: Any, parts: list[str]) -> bool:
        if not parts:
            return True
        if isinstance(node, list):
            return all(walk(n, parts) for n in node)
        if not isinstance(node, Mapping):
            return False
        if parts[0] in node:
            return walk(node[parts[0]], parts[1:])
        if node and all(k in CAM_IDS for k in node):  # cams / boxes2d keyed by camera id
            return all(walk(v, parts) for v in node.values())
        return False
    return walk(msg, path.split("."))


class OccOutError(ValueError):
    """``occ_out`` violates the v1/v2 contract."""


def _keys(d: Any, want: Sequence[str], where: str) -> None:
    if not isinstance(d, Mapping):
        raise OccOutError(f"{where}: not an object")
    missing = [k for k in want if k not in d]
    extra = [k for k in d if k not in want]
    if missing:
        raise OccOutError(f"{where}: missing {missing}")
    if extra:
        raise OccOutError(f"{where}: unknown fields {extra} (new fields need a version bump)")


def _num(x: Any, where: str, lo: float | None = None, hi: float | None = None) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise OccOutError(f"{where}: not a number ({type(x).__name__})")
    if not math.isfinite(float(x)):
        raise OccOutError(f"{where}: not finite")
    if (lo is not None and x < lo) or (hi is not None and x > hi):
        raise OccOutError(f"{where}: {x} outside [{lo}, {hi}]")
    return float(x)


def _int(x: Any, where: str, lo: int = 0) -> int:
    if isinstance(x, bool) or not isinstance(x, int):
        raise OccOutError(f"{where}: not an int ({type(x).__name__})")
    if x < lo:
        raise OccOutError(f"{where}: {x} < {lo}")
    return x


def _enum(x: Any, allowed: Sequence[str], where: str) -> str:
    if x not in allowed:
        raise OccOutError(f"{where}: {x!r} not in {list(allowed)}")
    return x


def _list(x: Any, where: str, cap: int, min_len: int = 0) -> list:
    if not isinstance(x, list):
        raise OccOutError(f"{where}: not a list")
    if len(x) > cap:
        raise OccOutError(f"{where}: {len(x)} entries over cap {cap}")
    if len(x) < min_len:
        raise OccOutError(f"{where}: {len(x)} entries, needs >= {min_len}")
    return x


def _bools(x: Any, where: str, n: int) -> None:
    if not isinstance(x, list) or len(x) != n:
        raise OccOutError(f"{where}: want a list of {n} bools")
    for i, b in enumerate(x):
        if not isinstance(b, bool):
            raise OccOutError(f"{where}[{i}]: not a bool")


def _polyline(rec: Mapping[str, Any], where: str, cap: int = MAX_POINTS, closed: bool = False) -> None:
    pts = _list(rec["pts"], f"{where}.pts", cap, 3 if closed else 2)
    for i, p in enumerate(pts):
        if not isinstance(p, list) or len(p) != 2:
            raise OccOutError(f"{where}.pts[{i}]: not [x, y]")
        _num(p[0], f"{where}.pts[{i}].x"); _num(p[1], f"{where}.pts[{i}].y")
    if "conf" in rec and isinstance(rec["conf"], list):
        conf = rec["conf"]
        if len(conf) != len(pts):
            raise OccOutError(f"{where}.conf: {len(conf)} values for {len(pts)} points")
        for i, c in enumerate(conf):
            _num(c, f"{where}.conf[{i}]", 0.0, 1.0)


def _segments_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def polygon_simple(pts: Sequence[Sequence[float]]) -> bool:
    """True when the closed polygon has no self-intersection (O(n^2), n <= MAX_CORRIDOR_VERTS)."""
    n = len(pts)
    segs = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            if _segments_cross(*segs[i], *segs[j]):
                return False
    return True


def validate_occ_out(msg: Any) -> dict[str, Any]:
    """Raise ``OccOutError`` unless ``msg`` is a valid occ_out v1 or v2. Returns ``msg``."""
    _keys(msg, TOP_KEYS, "occ_out")
    if msg["v"] not in OCC_OUT_VERSIONS or isinstance(msg["v"], bool) or msg["type"] != "occ_out":
        raise OccOutError(f"occ_out: v/type {msg['v']!r}/{msg['type']!r}, want one of {OCC_OUT_VERSIONS}/occ_out")
    ver = int(msg["v"])
    cell_keys, edge_keys, lane_keys = KEYS_BY_VERSION[ver]
    _int(msg["seq"], "seq"); _num(msg["t"], "t")
    g = msg["grid"]; _keys(g, GRID_KEYS, "grid")
    _num(g["voxel_m"], "grid.voxel_m", 1e-3)
    for k, n in (("origin", 3), ("shape", 3)):
        if not isinstance(g[k], list) or len(g[k]) != n:
            raise OccOutError(f"grid.{k}: want {n} values")
    for i, x in enumerate(g["origin"]):
        _num(x, f"grid.origin[{i}]")
    for i, x in enumerate(g["shape"]):
        _int(x, f"grid.shape[{i}]", 1)
    c = msg["counts"]; _keys(c, COUNT_KEYS, "counts")
    full = _int(c["full"], "counts.full", 1)
    frustum = _int(c["frustum"], "counts.frustum")
    queried = _int(c["queried"], "counts.queried")
    stored = _int(c["stored"], "counts.stored")
    sq = _list(c["stage_queries"], "counts.stage_queries", 8, 1)
    for i, q in enumerate(sq):
        _int(q, f"counts.stage_queries[{i}]")
    if full != int(np.prod(g["shape"])):
        raise OccOutError(f"counts.full {full} != prod(grid.shape)")
    # v2: stored = this tick's predicted cells + VO-warped memory cells (<= MAX_MEMORY)
    mem = MAX_MEMORY if ver >= 2 else 0
    if not (stored <= min(full, queried + mem) and queried <= full and frustum <= full):
        raise OccOutError(f"counts: need stored {stored} <= queried {queried} (+ memory {mem}) <= full {full}, frustum {frustum} <= full")
    if sum(sq) != queried:
        raise OccOutError(f"counts.stage_queries sum {sum(sq)} != queried {queried}")
    cells = msg["cells"]; _keys(cells, cell_keys, "cells")
    ijk = _list(cells["ijk"], "cells.ijk", MAX_CELLS)
    cls = _list(cells["cls"], "cells.cls", MAX_CELLS)
    prob = _list(cells["prob"], "cells.prob", MAX_CELLS)
    if ver >= 2:
        _bools(_list(cells["observed"], "cells.observed", MAX_CELLS), "cells.observed", len(ijk))
    if not (len(ijk) == len(cls) == len(prob) == c["cells_out"]):
        raise OccOutError(f"cells: lengths {len(ijk)}/{len(cls)}/{len(prob)} vs counts.cells_out {c['cells_out']}")
    if c["cells_out"] > stored:
        raise OccOutError("counts.cells_out > counts.stored")
    shape = g["shape"]
    for i, (q, k, p) in enumerate(zip(ijk, cls, prob)):
        if not isinstance(q, list) or len(q) != 3:
            raise OccOutError(f"cells.ijk[{i}]: not [i, j, k]")
        for a, (x, n) in enumerate(zip(q, shape)):
            if _int(x, f"cells.ijk[{i}][{a}]") >= n:
                raise OccOutError(f"cells.ijk[{i}][{a}]: {x} outside the grid")
        if _int(k, f"cells.cls[{i}]") >= len(OCC_CLASS_NAMES):
            raise OccOutError(f"cells.cls[{i}]: unknown class {k}")
        _num(p, f"cells.prob[{i}]", 0.0, 1.0)
    edges = _list(msg["edges"], "edges", MAX_EDGES)
    if len(edges) != _int(c["edges"], "counts.edges"):
        raise OccOutError("counts.edges != len(edges)")
    for i, e in enumerate(edges):
        _keys(e, edge_keys, f"edges[{i}]")
        if ver >= 2:
            _bools(e["observed"], f"edges[{i}].observed", len(e["pts"]) if isinstance(e["pts"], list) else -1)
        _enum(e["side"], EDGE_SIDES, f"edges[{i}].side"); _enum(e["kind"], EDGE_KINDS, f"edges[{i}].kind")
        _list(e["conf"], f"edges[{i}].conf", MAX_POINTS)
        _polyline(e, f"edges[{i}]")
    for i, ln in enumerate(_list(msg["lanes"], "lanes", MAX_LANES)):
        _keys(ln, lane_keys, f"lanes[{i}]")
        if ver >= 2:
            _bools(ln["observed"], f"lanes[{i}].observed", len(ln["pts"]) if isinstance(ln["pts"], list) else -1)
        _enum(ln["paint"], LANE_PAINTS, f"lanes[{i}].paint")
        _list(ln["conf"], f"lanes[{i}].conf", MAX_POINTS)
        _polyline(ln, f"lanes[{i}]")
    cor = msg["corridor"]; _keys(cor, CORRIDOR_KEYS, "corridor")
    cconf = _num(cor["conf"], "corridor.conf", 0.0, 1.0)
    if cor["pts"]:
        _polyline({"pts": cor["pts"]}, "corridor", MAX_CORRIDOR_VERTS, closed=True)
        if cor["pts"][0] == cor["pts"][-1]:
            raise OccOutError("corridor: closed implicitly; do not repeat the first vertex")
        if not polygon_simple(cor["pts"]):
            raise OccOutError("corridor: self-intersecting")
        if cconf < CORRIDOR_MIN_CONF:
            raise OccOutError("corridor: polygon given below the confidence floor")
    elif not isinstance(cor["pts"], list):
        raise OccOutError("corridor.pts: not a list")
    objs = _list(msg["objects"], "objects", MAX_OBJECTS)
    if len(objs) != _int(c["objects"], "counts.objects"):
        raise OccOutError("counts.objects != len(objects)")
    ids: dict[int, str] = {}
    for i, o in enumerate(objs):
        w = f"objects[{i}]"
        _keys(o, OBJECT_KEYS, w)
        oid = _int(o["id"], f"{w}.id")
        if oid in ids:
            raise OccOutError(f"{w}.id: duplicate {oid}")
        ids[oid] = _enum(o["cls"], OBJECT_CLASSES, f"{w}.cls")
        for k in ("x", "y", "z0"):
            _num(o[k], f"{w}.{k}")
        for k in ("l", "w", "h"):
            _num(o[k], f"{w}.{k}", 0.0, 50.0)
        hd = o["heading"]
        if not isinstance(hd, list) or len(hd) != 2:
            raise OccOutError(f"{w}.heading: want [cos, sin]")
        cs, sn = _num(hd[0], f"{w}.heading[0]", -1.0, 1.0), _num(hd[1], f"{w}.heading[1]", -1.0, 1.0)
        if abs(cs * cs + sn * sn - 1.0) > 1e-3:
            raise OccOutError(f"{w}.heading: cos^2 + sin^2 = {cs * cs + sn * sn:.4f}")
        _enum(o["heading_src"], HEADING_SRC, f"{w}.heading_src")
        _enum(o["motion"], MOTION_STATES, f"{w}.motion")
        _num(o["speed_mps"], f"{w}.speed_mps", 0.0, 100.0)
        _int(o["frames"], f"{w}.frames", 1)
    cams = msg["cams"]; _keys(cams, CAM_IDS, "cams")
    for cid in CAM_IDS:
        cm = cams[cid]; _keys(cm, CAM_KEYS, f"cams.{cid}")
        st = _enum(cm["status"], CAM_STATUS, f"cams.{cid}.status")
        wgt = _num(cm["weight"], f"cams.{cid}.weight", 0.0, 1.0)
        if st == "ABSENT":
            if cm["age_s"] is not None or cm["res"] is not None or wgt != 0.0:
                raise OccOutError(f"cams.{cid}: ABSENT carries no age, res or weight")
        else:
            _num(cm["age_s"], f"cams.{cid}.age_s", 0.0)
            if not isinstance(cm["res"], list) or len(cm["res"]) != 2:
                raise OccOutError(f"cams.{cid}.res: want [w, h]")
            for a in cm["res"]:
                _int(a, f"cams.{cid}.res", 1)
    b2 = msg["boxes2d"]
    if not isinstance(b2, Mapping):
        raise OccOutError("boxes2d: not an object")
    for cid, boxes in b2.items():
        if cid not in CAM_IDS:
            raise OccOutError(f"boxes2d: unknown camera {cid!r}")
        if cams[cid]["status"] != "LIVE":
            raise OccOutError(f"boxes2d.{cid}: boxes for a camera that is {cams[cid]['status']}")
        wdt, hgt = cams[cid]["res"]
        for j, bx in enumerate(_list(boxes, f"boxes2d.{cid}", MAX_BOXES_PER_CAM)):
            _keys(bx, BOX_KEYS, f"boxes2d.{cid}[{j}]")
            if _int(bx["id"], f"boxes2d.{cid}[{j}].id") not in ids:
                raise OccOutError(f"boxes2d.{cid}[{j}]: id {bx['id']} is not an object")
            if bx["cls"] != ids[bx["id"]]:
                raise OccOutError(f"boxes2d.{cid}[{j}]: class differs from the object")
            box = bx["box"]
            if not isinstance(box, list) or len(box) != 4:
                raise OccOutError(f"boxes2d.{cid}[{j}].box: want [u0, v0, u1, v1]")
            u0, v0, u1, v1 = (_num(x, f"boxes2d.{cid}[{j}].box") for x in box)
            if not (0 <= u0 < u1 <= wdt and 0 <= v0 < v1 <= hgt):
                raise OccOutError(f"boxes2d.{cid}[{j}].box {box} not inside {wdt}x{hgt}")
    ego = msg["ego"]; _keys(ego, EGO_KEYS, "ego")
    _num(ego["speed_mps"], "ego.speed_mps", -20.0, 100.0)
    _num(ego["yaw_rate_rps"], "ego.yaw_rate_rps", -5.0, 5.0)
    _enum(ego["source"], EGO_SOURCES, "ego.source")
    if not isinstance(ego["ok"], bool):
        raise OccOutError("ego.ok: not a bool")
    if ego["dt_s"] is not None:
        _num(ego["dt_s"], "ego.dt_s", 0.0)
    tm = msg["timing_ms"]; _keys(tm, TIMING_KEYS, "timing_ms")
    for k in TIMING_KEYS:
        _num(tm[k], f"timing_ms.{k}", 0.0)
    if msg["peak_vram_mb"] is not None:
        _num(msg["peak_vram_mb"], "peak_vram_mb", 0.0)
        if msg["vram_status"] != "MEASURED":
            raise OccOutError("peak_vram_mb set without vram_status MEASURED")
    elif msg["vram_status"] != "UNPROVEN":
        raise OccOutError("vram_status must be UNPROVEN while peak_vram_mb is None")
    if msg["vram_est_mb_cpu"] is not None:
        _num(msg["vram_est_mb_cpu"], "vram_est_mb_cpu", 0.0)
    if len(encode_occ_out(msg, check=False)) > MAX_BYTES:
        raise OccOutError(f"occ_out over {MAX_BYTES} bytes")
    return msg


def encode_occ_out(msg: Mapping[str, Any], *, check: bool = True) -> bytes:
    if check:
        validate_occ_out(msg)
    return json.dumps(msg, separators=(",", ":"), allow_nan=False).encode("utf-8")


# ---------------------------------------------------------------- builders


def _r(x: float, nd: int = 3) -> float:
    return round(float(x), nd)


def edge_conf(edges: Sequence[Mapping[str, Any]], keys: np.ndarray, cls: np.ndarray, prob: np.ndarray,
              grid: OccGrid) -> list[list[float]]:
    """Per edge point: occupancy probability of the drivable boundary cell (max over z)."""
    from python.perception.occ_outputs import DRIVABLE

    m = np.asarray(cls) == DRIVABLE
    c = grid.cell_of_key(np.asarray(keys, dtype=np.int64)[m])
    p = np.asarray(prob, dtype=np.float64)[m]
    best: dict[tuple[int, int], float] = {}
    for (ix, iy, _), pv in zip(c.tolist(), p.tolist()):
        best[(ix, iy)] = max(best.get((ix, iy), 0.0), pv)
    v = float(grid.voxel_m)
    out = []
    for e in edges:
        confs = []
        for x, y in e["pts"]:
            iy = int(math.floor((y - grid.y_min) / v))
            ix = int(math.floor((x - grid.x_min) / v))
            near = [best.get((ix + d, iy), 0.0) for d in (-1, 0, 1)]
            confs.append(_r(min(1.0, max(near)), 3))
        out.append(confs)
    return out


def corridor(edges: Sequence[Mapping[str, Any]], *, max_verts: int = MAX_CORRIDOR_VERTS) -> dict[str, Any]:
    """Drivable corridor polygon from the left/right road edges (ego frame).

    Rows where both edges exist are resampled to <= max_verts/2 rows; the
    polygon runs up the left edge and back down the right one (implicitly
    closed). conf = share of the shared y span where both edges exist with
    left < right. Below ``CORRIDOR_MIN_CONF`` (or < 2 shared rows) it is empty.
    """
    side = {e["side"]: np.asarray(e["pts"], dtype=np.float64) for e in edges if e.get("kind", "road_edge") == "road_edge"}
    if "left" not in side or "right" not in side:
        return {"pts": [], "conf": 0.0}
    L, R = side["left"], side["right"]
    L, R = L[np.argsort(L[:, 1])], R[np.argsort(R[:, 1])]
    y0, y1 = max(L[0, 1], R[0, 1]), min(L[-1, 1], R[-1, 1])
    if y1 - y0 < 1.0:
        return {"pts": [], "conf": 0.0}
    n = max(2, min(max_verts // 2, int((y1 - y0) / 0.4) + 1))
    ys = np.linspace(y0, y1, n)
    xl, xr = np.interp(ys, L[:, 1], L[:, 0]), np.interp(ys, R[:, 1], R[:, 0])
    ok = xl < xr - 0.4
    rows_l = ((L[:, 1] >= y0) & (L[:, 1] <= y1)).sum()
    rows_r = ((R[:, 1] >= y0) & (R[:, 1] <= y1)).sum()
    span_rows = (y1 - y0) / 0.4 + 1.0
    conf = float(min(1.0, min(rows_l, rows_r) / span_rows) * ok.mean())
    if conf < CORRIDOR_MIN_CONF or ok.sum() < 2:
        return {"pts": [], "conf": _r(conf)}
    ys, xl, xr = ys[ok], xl[ok], xr[ok]
    pts = [[_r(x, 2), _r(y, 2)] for x, y in zip(xl, ys)] + [[_r(x, 2), _r(y, 2)] for x, y in zip(xr[::-1], ys[::-1])]
    return {"pts": pts, "conf": _r(conf)}


def box_corners(o: Mapping[str, Any], heading: Sequence[float]) -> np.ndarray:
    cs, sn = float(heading[0]), float(heading[1])
    hl, hw = float(o["l"]) / 2.0, float(o["w"]) / 2.0
    out = []
    for a in (-hl, hl):
        for b in (-hw, hw):
            x = float(o["x"]) + a * cs - b * sn
            y = float(o["y"]) + a * sn + b * cs
            for z in (float(o["z0"]), float(o["z0"]) + float(o["h"])):
                out.append((x, y, z))
    return np.asarray(out)


def project_box(cam: PinholeCam, corners: np.ndarray) -> list[float] | None:
    """Image box of a 3-D box in ``cam``, clipped to the image; None if not in view."""
    uv, z, _ = project(cam, corners)
    front = z > cam.near_m
    if front.sum() < 1:
        return None
    u, v = uv[front, 0], uv[front, 1]
    u0, u1 = max(0.0, float(u.min())), min(float(cam.width), float(u.max()))
    v0, v1 = max(0.0, float(v.min())), min(float(cam.height), float(v.max()))
    if u1 - u0 < 1.0 or v1 - v0 < 1.0:
        return None
    return [_r(u0, 1), _r(v0, 1), _r(u1, 1), _r(v1, 1)]


def cam_panels(cams: Sequence[PinholeCam], weights: np.ndarray, ages: Mapping[str, float] | None,
               present: Sequence[bool], preset: str | None = None) -> dict[str, dict[str, Any]]:
    """Per camera: ABSENT (not in the rig / preset, no frame, no age), else LIVE or STALE by fusion weight."""
    from python.perception.occ_sparse import CAM_PRESETS

    keep = set(CAM_PRESETS[preset]) if preset is not None else set(CAM_IDS)
    by = {c.cam_id: (c, float(w), bool(p)) for c, w, p in zip(cams, weights, present)}
    out = {}
    for cid in CAM_IDS:
        if cid not in by or cid not in keep or not by[cid][2]:
            out[cid] = {"status": "ABSENT", "age_s": None, "weight": 0.0, "res": None}
            continue
        cam, w, _ = by[cid]
        age = 0.0 if ages is None else ages.get(cid)
        if age is None or not math.isfinite(float(age)) or float(age) < 0:
            out[cid] = {"status": "ABSENT", "age_s": None, "weight": 0.0, "res": None}
            continue
        out[cid] = {"status": "LIVE" if w >= STALE_WEIGHT else "STALE", "age_s": _r(age), "weight": _r(w),
                    "res": [int(cam.width), int(cam.height)]}
    return out


def ground_cell(keys: np.ndarray, cls: np.ndarray, grid: OccGrid) -> int | None:
    """Median z cell of the drivable cells (road surface), None without any."""
    from python.perception.occ_outputs import DRIVABLE

    m = np.asarray(cls) == DRIVABLE
    if not m.any():
        return None
    return int(np.median(grid.cell_of_key(np.asarray(keys, np.int64)[m])[:, 2]))


def occluders(keys: np.ndarray, cls: np.ndarray, grid: OccGrid) -> np.ndarray:
    """Sorted keys that block a camera ray: predicted non-drivable cells >= 1 voxel above the road surface."""
    from python.perception.occ_outputs import DRIVABLE

    keys = np.asarray(keys, np.int64)
    cls = np.asarray(cls)
    if not keys.size:
        return keys
    g = ground_cell(keys, cls, grid)
    z = grid.cell_of_key(keys)[:, 2]
    m = (cls != DRIVABLE) & (z >= (g + 1 if g is not None else -1))
    return np.sort(keys[m])


def observed_mask(pts: np.ndarray, cams: Sequence[PinholeCam], occl: np.ndarray, grid: OccGrid,
                  chunk: int = 1024) -> np.ndarray:
    """(N,) True where some camera in ``cams`` sees the point with no occluder cell
    on the ray before it (the last voxel before the point is not tested).
    Sparse: ray samples are looked up in the sorted occluder keys."""
    pts = np.asarray(pts, np.float64).reshape(-1, 3)
    obs = np.zeros(pts.shape[0], bool)
    v = float(grid.voxel_m)
    occl = np.asarray(occl, np.int64)
    for cam in cams:
        _, _, ok = project(cam, pts)
        idx = np.flatnonzero(ok & ~obs)
        if not idx.size:
            continue
        o = np.asarray(cam.pos, np.float64)
        if not occl.size:
            obs[idx] = True
            continue
        for c0 in range(0, idx.size, chunk):
            ii = idx[c0:c0 + chunk]
            vec = pts[ii] - o
            dist = np.linalg.norm(vec, axis=1)
            unit = vec / np.maximum(dist, 1e-9)[:, None]
            ts = np.arange(0.0, float(dist.max()), v * 0.5)
            m = ts[None, :] < (dist[:, None] - v)
            samp = o + unit[:, None, :] * ts[None, :, None]
            k = grid.keys_of(samp.reshape(-1, 3)).reshape(m.shape)
            pos = np.clip(np.searchsorted(occl, k), 0, occl.size - 1)
            hit = (occl[pos] == k) & (k >= 0) & m
            obs[ii[~hit.any(axis=1)]] = True
    return obs


class OccMemory:
    """VO-warped temporal memory of static cells (causal: past ticks only).

    Each tick: last tick's static cells (not vehicles, which move) are warped
    into the current ego frame with this tick's camera-VO motion, their
    probability decays by exp(-dt/tau). A warped cell a camera sees now is
    dropped (the current prediction is the truth there); an unseen one is
    kept with ``observed = False`` if this tick did not predict it. VO not
    ok -> memory cleared (no blind carry). Capped at ``MAX_MEMORY`` cells.
    """

    def __init__(self, grid: OccGrid, *, tau_s: float = 2.0, min_prob: float = 0.35, cap: int = MAX_MEMORY) -> None:
        self.grid, self.tau_s, self.min_prob, self.cap = grid, float(tau_s), float(min_prob), int(cap)
        self.keys = np.zeros(0, np.int64); self.cls = np.zeros(0, np.uint8); self.prob = np.zeros(0, np.float32)

    def reset(self) -> None:
        self.__init__(self.grid, tau_s=self.tau_s, min_prob=self.min_prob, cap=self.cap)

    def step(self, keys, cls, prob, obs, *, ok: bool, fwd: float, dyaw: float, dt: float | None,
             cams: Sequence[PinholeCam], occl: np.ndarray):
        """-> merged (keys, cls, prob, observed), sorted by key."""
        from python.perception.occ_ego import _warp
        from python.perception.occ_outputs import VEHICLE

        g = self.grid
        keys = np.asarray(keys, np.int64); cls = np.asarray(cls, np.uint8); prob = np.asarray(prob, np.float32)
        obs = np.asarray(obs, bool)
        mk = np.zeros(0, np.int64); mc = np.zeros(0, np.uint8); mp = np.zeros(0, np.float32)
        if self.keys.size and ok and dt is not None and dt > 0:
            c = g.centers_of_keys(self.keys)
            xy = _warp(c[:, :2], float(fwd), float(dyaw))
            nk = g.keys_of(np.concatenate([xy, c[:, 2:3]], 1))
            p = self.prob * np.float32(math.exp(-dt / self.tau_s))
            keep = (nk >= 0) & (p >= self.min_prob) & ~np.isin(nk, keys)
            mk, mc, mp = nk[keep], self.cls[keep], p[keep]
            if mk.size:
                mk, first = np.unique(mk, return_index=True)
                mc, mp = mc[first], mp[first]
                seen = observed_mask(g.centers_of_keys(mk), cams, occl, g)
                mk, mc, mp = mk[~seen], mc[~seen], mp[~seen]
        K = np.concatenate([keys, mk]); C = np.concatenate([cls, mc]); P = np.concatenate([prob, mp])
        O = np.concatenate([obs, np.zeros(mk.size, bool)])
        o = np.argsort(K, kind="stable")
        K, C, P, O = K[o], C[o], P[o], O[o]
        st = (C != VEHICLE) & (P >= self.min_prob)
        if ok or dt is None:  # first tick: nothing to warp yet, start remembering
            gk = ground_cell(K, C, g)
            flat = (C == 0) & (g.cell_of_key(K)[:, 2] <= (gk if gk is not None else -1)) if K.size else np.zeros(0, bool)
            top = np.lexsort((-P[st], flat[st]))[: self.cap]  # plain ground last, then by probability
            self.keys, self.cls, self.prob = K[st][top], C[st][top], P[st][top]
        else:  # VO lost: no blind carry
            self.reset()
        return K, C, P, O


class OccRuntime:
    """Camera-only tick: images (+ per-camera ages) -> validated ``occ_out`` v2.

    Causal: a tick reads only this tick's images and state carried from earlier
    ticks (VO, tracker, memory); nothing is revised retroactively.

    Ego motion is camera VO unless ``from_vehicle`` (``--ego-from-vehicle``).
    Latency comes from in-code ``perf_counter`` timers.
    """

    def __init__(self, net: Any, *, from_vehicle: bool = False, max_points: int = 16384, max_coarse: int = 2048,
                 preset: str | None = None, vram_est_mb_cpu: float | None = None,
                 frustum_cells: int | None = None, memory: bool = True, timer: Any = None) -> None:
        from python.perception.occ_ego import EgoMotion
        from python.perception.occ_outputs import OccTracker

        self.net = net
        self.grid: OccGrid = net.grid
        self.tracker = OccTracker()
        self.ego = EgoMotion(from_vehicle=from_vehicle, grid=net.grid)
        self.max_points, self.max_coarse, self.preset = int(max_points), int(max_coarse), preset
        self.vram_est_mb_cpu = vram_est_mb_cpu
        self.seq = 0
        self._prev_t: float | None = None
        self._frustum: dict[tuple, int] = {}
        self._frustum_fixed = frustum_cells  # precomputed (e.g. by bench_occ_sparse) for huge grids
        self.memory = OccMemory(net.grid) if memory else None
        self._timer = timer or time.perf_counter  # injectable: a fixed clock makes occ_out byte-reproducible

    def _frustum_count(self, active: tuple) -> int:
        """Fine cells visible to the active cameras, counted in fixed-size chunks (no grid-sized array)."""
        if self._frustum_fixed is not None:
            return int(self._frustum_fixed)
        if active not in self._frustum:
            from python.perception.occ_sparse import visible_mask

            on = [c for c, a in zip(self.net.cams, active) if a]
            n, total, step = int(self.grid.dense_cells), 0, 65536
            for a in range(0, n, step):
                ks = np.arange(a, min(n, a + step), dtype=np.int64)
                total += int(visible_mask(on, self.grid.centers_of_keys(ks)).sum()) if on else 0
            self._frustum[active] = total
        return self._frustum[active]

    def step(self, images: Sequence[Any], t: float, *, ages_s: Mapping[str, float] | None = None,
             vehicle: Mapping[str, Any] | None = None) -> dict[str, Any]:
        from python.perception.occ_outputs import occ_outputs
        from python.perception.occ_sparse import OccPrediction, cam_weights

        ids = self.net.cam_ids
        w = cam_weights(ids, preset=self.preset, ages_s=ages_s) if (ages_s is not None or self.preset) else np.ones(len(ids), np.float32)
        present = [img is not None for img in images]
        w = w * np.asarray(present, dtype=np.float32)
        self._frustum_count(tuple(bool(x > 0) for x in w))  # once per camera set, cached, outside the timers
        clk = self._timer
        t0 = clk()
        raw = self.net.predict(images, max_points=self.max_points, max_coarse=self.max_coarse, cam_w=w)
        t1 = clk()
        g = self.grid
        est = self.ego.update(raw.keys, raw.cls, t, vehicle)  # VO on this tick's own cells
        dt = None if self._prev_t is None else float(t) - self._prev_t
        live = [c for c, x in zip(self.net.cams, w) if x >= STALE_WEIGHT]
        occl = occluders(raw.keys, raw.cls, g)
        obs = observed_mask(g.centers_of_keys(np.asarray(raw.keys, np.int64)), live, occl, g) if len(raw.keys) else np.zeros(0, bool)
        if self.memory is not None:
            keys, cls, prob, obs = self.memory.step(raw.keys, raw.cls, raw.prob, obs, ok=bool(est.ok),
                                                    fwd=est.speed_mps * (dt or 0.0), dyaw=est.yaw_rate * (dt or 0.0),
                                                    dt=dt, cams=live, occl=occl)
        else:
            keys, cls, prob = np.asarray(raw.keys), np.asarray(raw.cls), np.asarray(raw.prob)
        pred = OccPrediction(keys, prob, cls, raw.n_coarse, raw.n_fine, raw.stage_queries)
        out = occ_outputs(pred, self.tracker, t, ego_speed_mps=est.speed_mps, ego_yaw_rate=est.yaw_rate, grid=g)
        out["ego"].update({"source": est.source, "motion_ok": bool(est.ok), "yaw_rate": float(est.yaw_rate)})
        self._obs_ctx = (obs, live, occl)
        self.last_cells = (np.asarray(keys, np.int64), np.asarray(cls), np.asarray(prob), np.asarray(obs, bool))  # pre-cap, for eval
        t2 = clk()
        msg = self._build(pred, out, t, w, ages_s, present, (t1 - t0) * 1e3, (t2 - t1) * 1e3)
        msg["timing_ms"]["total"] = _r((clk() - t0) * 1e3, 2)
        return validate_occ_out(msg)

    def _build(self, pred, out, t, w, ages, present, ms_pred, ms_out) -> dict[str, Any]:
        g = self.grid
        keys, cls, prob = np.asarray(pred.keys), np.asarray(pred.cls), np.asarray(pred.prob)
        ctr = g.centers_of_keys(keys) if keys.size else np.zeros((0, 3))
        # cap order: plain ground ("other" at road-surface height: grass / terrain) last, else nearest first
        gk0 = ground_cell(keys, cls, g)
        flat = (cls == 0) & (g.cell_of_key(keys)[:, 2] <= (gk0 if gk0 is not None else -1)) if keys.size else np.zeros(0, bool)
        order = np.lexsort((np.hypot(ctr[:, 0], ctr[:, 1]), flat))[:MAX_CELLS] if keys.size else np.zeros(0, int)
        ijk = g.cell_of_key(keys[order]) if keys.size else np.zeros((0, 3), int)
        edges = [e for e in out["edges"] if len(e["pts"]) >= 2][:MAX_EDGES]
        for e in edges:
            e["pts"] = e["pts"][:MAX_POINTS]
        confs = edge_conf(edges, keys, cls, prob, g)
        obs, live, occl = self._obs_ctx
        gk = ground_cell(keys, cls, g)
        gz = g.z_min + ((gk if gk is not None else 0) + 0.5) * g.voxel_m
        e_obs = []
        for e in edges:
            P = np.asarray(e["pts"], np.float64).reshape(-1, 2)
            e_obs.append(observed_mask(np.concatenate([P, np.full((len(P), 1), gz)], 1), live, occl, g).tolist())
        objs = []
        for o in out["objects"][:MAX_OBJECTS]:
            moving = o.get("yaw_src") == "track"  # tracker: speed >= PARKED_MPS over >= MIN_FRAMES
            yaw = math.atan2(o["vy"], o["vx"]) if moving else float(o["yaw"])
            objs.append({"id": int(o["id"]), "cls": "vehicle", "x": _r(o["x"]), "y": _r(o["y"]), "z0": _r(o.get("z0", 0.0)),
                         "l": _r(o["length"]), "w": _r(o["width"]), "h": _r(o["height"]),
                         "heading": [math.cos(yaw), math.sin(yaw)], "heading_src": "velocity" if moving else "box",
                         "motion": o["motion"] if o["motion"] in MOTION_STATES else "unknown",
                         "speed_mps": _r(o.get("speed_mps") or 0.0), "frames": int(o.get("frames", 1))})
        cams = cam_panels(self.net.cams, w, ages, present, self.preset)
        boxes2d: dict[str, list] = {}
        for cam in self.net.cams:
            if cams[cam.cam_id]["status"] != "LIVE":
                continue
            bl = []
            for o in objs:
                bx = project_box(cam, box_corners(o, o["heading"]))
                if bx is not None and len(bl) < MAX_BOXES_PER_CAM:
                    bl.append({"id": o["id"], "cls": o["cls"], "box": bx})
            if bl:
                boxes2d[cam.cam_id] = bl
        active = tuple(bool(x > 0) for x in w)
        dt = None if self._prev_t is None else _r(float(t) - self._prev_t, 6)
        self._prev_t = float(t)
        msg = {
            "v": OCC_OUT_VERSION, "type": "occ_out", "seq": self.seq, "t": _r(t, 6),
            "grid": {"voxel_m": float(g.voxel_m), "origin": [float(g.x_min), float(g.y_min), float(g.z_min)],
                     "shape": [int(x) for x in g.shape]},
            "counts": {"full": int(g.dense_cells), "frustum": self._frustum_count(active), "queried": int(pred.n_queried),
                       "stored": int(keys.size), "stage_queries": [int(x) for x in pred.stage_queries],
                       "cells_out": int(order.size), "objects": len(objs), "edges": len(edges)},
            "cells": {"ijk": ijk.astype(int).tolist(), "cls": cls[order].astype(int).tolist(),
                      "prob": [_r(p) for p in prob[order].tolist()],
                      "observed": [bool(b) for b in np.asarray(obs, bool)[order].tolist()]},
            "edges": [{"side": e["side"], "kind": "road_edge", "pts": e["pts"], "conf": c, "observed": [bool(b) for b in ob]}
                      for e, c, ob in zip(edges, confs, e_obs)],
            "lanes": [],
            "corridor": corridor(edges),
            "objects": objs,
            "boxes2d": boxes2d,
            "cams": cams,
            "ego": {"speed_mps": _r(out["ego"]["speed_mps"]), "yaw_rate_rps": _r(out["ego"]["yaw_rate"], 4),
                    "source": "vehicle" if out["ego"]["source"] == "vehicle_signals" else "vo",
                    "ok": bool(out["ego"]["motion_ok"]), "dt_s": dt},
            "timing_ms": {"predict": _r(ms_pred, 2), "outputs": _r(ms_out, 2), "total": 0.0},
            "peak_vram_mb": None, "vram_status": "UNPROVEN",
            "vram_est_mb_cpu": None if self.vram_est_mb_cpu is None else _r(self.vram_est_mb_cpu, 1),
        }
        self.seq += 1
        return msg


VIZ_DELTA = (  # occ_out field -> gvd_scene v1
    ("cells.ijk/cls", "occ.cells [[ix,iy,iz,cls]] (optional block)"),
    ("cells.prob, cells.observed", "not in viz v1: dropped"),
    ("edges[].pts/side", "edges[] class road_edge, color red"),
    ("edges[].conf, edges[].observed", "not in viz v1: dropped"),
    ("objects[].heading", "tracks[].yaw = atan2(sin, cos)"),
    ("objects[].l/w/h/speed_mps/motion", "tracks[].l/w/h/v/motion"),
    ("corridor, boxes2d, cams, counts, timing_ms, ego.yaw_rate_rps", "not in viz v1: debug-view only"),
)


def occ_out_to_viz_inputs(msg: Mapping[str, Any]) -> dict[str, Any]:
    """occ_out (v1/v2) -> the input dict ``occ_viz.occ_to_viz`` reads (known keys only).

    Mapping / delta to viz v1 (kept here; ``occ_viz.py`` and the C++ viz are
    GVD's): see ``VIZ_DELTA``."""
    from python.perception.occ_geometry import OccGrid as _G

    validate_occ_out(msg)
    g = msg["grid"]
    v = float(g["voxel_m"]); ox, oy, oz = g["origin"]; nx, ny, nz = g["shape"]
    grid = _G(ox, ox + nx * v, oy, oy + ny * v, oz, oz + nz * v, v)
    ijk = np.asarray(msg["cells"]["ijk"], dtype=np.int64).reshape(-1, 3)
    return {
        "t": msg["t"],
        "ego": {"speed_mps": msg["ego"]["speed_mps"], "engaged": False},
        "grid": grid,
        "cells": {"keys": grid.key_of_cell(ijk) if ijk.size else np.zeros(0, np.int64),
                  "cls": np.asarray(msg["cells"]["cls"], dtype=np.int64)},
        "edges": [{"side": e["side"], "pts": e["pts"], "source": "occ"} for e in msg["edges"]],
        "objects": [{"id": o["id"], "cls": o["cls"], "x": o["x"], "y": o["y"], "yaw": math.atan2(o["heading"][1], o["heading"][0]),
                     "length": o["l"], "width": o["w"], "height": o["h"], "speed_mps": o["speed_mps"], "motion": o["motion"]}
                    for o in msg["objects"]],
    }
