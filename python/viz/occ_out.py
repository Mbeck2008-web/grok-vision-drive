"""``occ_out`` v2: one tick of model outputs for every C++ viz debug tab (torch-free, GT-free).

Built only from network outputs (sparse occ set + occ head, SceneNet lane/curb/object heads,
camera-only VO, per-ray top-k depth samples) plus wall-clock timers. Never images, labels or
sim pose. Every array has a fixed cap; nothing scales with the voxel grid.

v2 = v1 + ``observed`` per occ cell and per polyline point (True = a camera ray reached that
voxel this tick; False = inferred from priors/memory). Unknown keys anywhere are rejected.

Conventions: ego frame x right, y FORWARD, z up; yaw/dyaw + = left. Object arrow heading is the
box yaw (``sin_yaw``/``cos_yaw``), not a velocity direction, so it is defined at standstill.
Not predicted -> explicit None/"unknown", never a made-up default: object ``z``/``h`` (2-D boxes),
lane ``style`` (no solid/dashed head), VO ``conf``. ``peak_vram_mb`` is None (UNPROVEN) unless
measured by ``torch.cuda`` on a GPU; ``cpu_mem_est_mb`` is a separate, labelled CPU estimate.
Per-cam 2-D boxes are projections of the 3-D box footprints (same ``slot``), not a separate head.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from python.viz.occ_scene import FLOW_DIM, GRID_MIN, GRID_SHAPE, OBJECT_CLASSES, OCC_CLASSES, VOXEL_M, _sigmoid, voxel_centers

OCC_OUT_VERSION = 2
FULL_GRID = int(np.prod(GRID_SHAPE))
MAX_OCC = 4096 + 512  # stored set (1080ti keep) + fixed unseen-region lattice
MAX_MEM = 1024
MAX_CAM_SAMPLES = 512  # per cam (12x16 px x top-2 bins = 384 at the 192 px strip)
MAX_LANES = 16
MAX_OBJECTS = 32
MAX_CORRIDOR = 64
LANE_TYPES = ("white", "yellow", "curb", "road_edge")
LANE_STYLES = ("solid", "dashed", "unknown")
EGO_SOURCES = ("vo", "vehicle")
STAGES = ("backbone_scene_heads", "lift_query", "memory", "occ_head")

KEYS = {
    "": {"v", "cams", "occ", "counts", "lanes", "objects", "ego", "corridor", "cam_samples", "cam_boxes", "debug"},
    "occ": {"ids", "xyz", "prob", "cls", "flow", "observed"},
    "counts": {"queried", "stored", "active", "frustum", "full_grid"},
    "lane": {"type", "style", "pts", "conf", "observed"},
    "object": {"slot", "x", "y", "z", "l", "w", "h", "sin_yaw", "cos_yaw", "speed_mps", "cls", "conf"},
    "ego": {"dx", "dy", "dyaw", "dt_s", "speed_mps", "yaw_rate_rps", "source", "conf", "memory_xyz"},
    "cam_samples": {"row", "col", "depth_m", "prob", "cls"},
    "cam_box": {"slot", "cls", "x0", "y0", "x1", "y1"},
    "debug": {"ms", "peak_vram_mb", "peak_vram_src", "cpu_mem_est_mb"},
}
# What gvd_scene viz v1 (docs/specs/viz_cpp.md @ f3555bf) has no field for: dropped by occ_out_to_viz, v2 viz delta.
VIZ_V2_DELTA = ("occ.prob", "occ.observed", "lanes.conf", "lanes.observed", "objects.l/w/conf", "ego", "corridor",
                "cam_samples", "cam_boxes", "debug", "counts")


@dataclass
class OccOutV2:
    v: int = OCC_OUT_VERSION
    cams: list = field(default_factory=list)
    occ: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    lanes: list = field(default_factory=list)
    objects: list = field(default_factory=list)
    ego: dict = field(default_factory=dict)
    corridor: list = field(default_factory=list)
    cam_samples: dict = field(default_factory=dict)
    cam_boxes: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _r(x: Any, n: int = 3) -> Any:
    return np.round(np.asarray(x, np.float64), n).tolist()


def ids_of_xy(pts: np.ndarray, z: float = 0.25) -> np.ndarray:
    """Voxel id of ego-frame ground points (``z`` default = the ground layer centre), -1 outside the grid."""
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    ijk = np.floor((np.c_[p, np.full(len(p), z)] - np.asarray(GRID_MIN)) / VOXEL_M).astype(np.int64)
    ok = ((ijk >= 0) & (ijk < np.asarray(GRID_SHAPE))).all(1)
    out = np.full(len(p), -1, np.int64)
    out[ok] = np.ravel_multi_index(tuple(ijk[ok].T), GRID_SHAPE)
    return out


def _lanes(pred: Mapping[str, Any], seen: np.ndarray) -> list[dict[str, Any]]:
    out = []
    paint = pred.get("lane_paint")
    for key, conf_key, kind in (("lanes", "lane_valid", None), ("curbs", "curb_valid", "curb")):
        polys = pred.get(key)
        if polys is None:
            continue
        polys = np.asarray(polys, np.float32)
        conf = np.clip(np.asarray(pred.get(conf_key, np.ones(len(polys))), np.float32).reshape(-1), 0.0, 1.0)
        for i, pts in enumerate(polys):
            if len(pts) < 2 or not np.isfinite(pts).all():
                continue
            if kind is None:
                typ = "yellow" if paint is not None and float(np.asarray(paint).reshape(-1)[i]) > 0.5 else "white"
            else:
                typ = kind
            obs = np.isin(ids_of_xy(pts), seen)
            # One confidence per polyline from the head; repeated per point (stated, not invented).
            out.append({"type": typ, "style": "unknown", "pts": _r(pts, 2), "conf": [round(float(conf[i]), 3)] * len(pts),
                        "observed": [bool(b) for b in obs]})
    return out[:MAX_LANES]


def _objects(pred: Mapping[str, Any]) -> list[dict[str, Any]]:
    objs = pred.get("objects")
    if objs is None:
        return []
    objs = np.asarray(objs, np.float32)
    valid = np.clip(np.asarray(pred.get("object_valid", np.ones(len(objs))), np.float32).reshape(-1), 0.0, 1.0)
    cls = pred.get("object_class")
    out = []
    for i, o in enumerate(objs[:MAX_OBJECTS]):
        if not np.isfinite(o).all():
            continue
        name = OBJECT_CLASSES[int(np.asarray(cls)[i].argmax())] if cls is not None else "vehicle"
        out.append({"slot": i, "x": round(float(o[0]), 2), "y": round(float(o[1]), 2), "z": None,
                    "l": round(float(o[3]), 2), "w": round(float(o[4]), 2), "h": None,
                    "sin_yaw": round(float(np.sin(o[2])), 4), "cos_yaw": round(float(np.cos(o[2])), 4),
                    "speed_mps": round(float(o[5]), 2), "cls": name, "conf": round(float(valid[i]), 3)})
    return out


def _hull(pts: np.ndarray) -> list[list[float]]:
    """Closed convex hull (monotone chain) of ego-frame xy points; [] under 3 distinct points."""
    p = sorted(set(map(tuple, np.round(np.asarray(pts, np.float64), 2).tolist())))
    if len(p) < 3:
        return []

    def cross(o: Any, a: Any, b: Any) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo: list = []
    for q in p:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    up: list = []
    for q in reversed(p):
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0:
            up.pop()
        up.append(q)
    h = lo[:-1] + up[:-1]
    if len(h) < 3:
        return []
    if len(h) > MAX_CORRIDOR - 1:
        h = [h[int(i)] for i in np.linspace(0, len(h) - 1, MAX_CORRIDOR - 1).round().astype(int)]
    return [list(v) for v in h] + [list(h[0])]


def project_xyz(pose: Any, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ego-frame points -> (u, v) pixels of a ``cameras.yaml`` pose (inverse of ``lift_voxels``) and an in-front mask."""
    d = np.asarray(xyz, np.float64).reshape(-1, 3) - np.array([float(pose.x), float(pose.y), float(pose.z)])
    tv = math.tan(math.radians(float(pose.fov_v_deg)) / 2.0)
    th = tv * float(pose.width) / float(pose.height)
    az = np.arctan2(d[:, 0], d[:, 1]) - math.radians(float(pose.yaw_deg))
    az = (az + np.pi) % (2 * np.pi) - np.pi
    el = np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])) - math.radians(float(pose.pitch_deg))
    front = (np.abs(az) < np.pi / 2 - 1e-3) & (np.abs(el) < np.pi / 2 - 1e-3)
    u = (np.tan(np.where(front, az, 0.0)) / th + 1.0) / 2.0 * float(pose.width)
    v = (-np.tan(np.where(front, el, 0.0)) / tv + 1.0) / 2.0 * float(pose.height)
    return np.c_[u, v], front


def _cam_boxes(objects: Sequence[Mapping[str, Any]], cams: Sequence[str], poses: Mapping[str, Any]) -> dict[str, list]:
    out: dict[str, list] = {}
    for cam in cams:
        pose = poses.get(cam)
        if pose is None:
            continue
        boxes = []
        for o in objects:
            c, s, hl, hw = o["cos_yaw"], o["sin_yaw"], o["l"] / 2.0, o["w"] / 2.0
            # Footprint corners at z=0 (no height head). Yaw + = left, 0 = facing +y.
            corners = [(o["x"] + a * hw * c - b * hl * s, o["y"] + a * hw * s + b * hl * c) for a in (-1, 1) for b in (-1, 1)]
            uv, front = project_xyz(pose, np.c_[np.asarray(corners), np.zeros(4)])
            if not front.all():
                continue
            x0, y0 = np.clip(uv.min(0), 0, [pose.width, pose.height])
            x1, y1 = np.clip(uv.max(0), 0, [pose.width, pose.height])
            if x1 - x0 >= 1 and y1 - y0 >= 1:
                boxes.append({"slot": o["slot"], "cls": o["cls"], "x0": round(float(x0), 1), "y0": round(float(y0), 1),
                              "x1": round(float(x1), 1), "y1": round(float(y1), 1)})
        out[cam] = boxes[:MAX_OBJECTS]
    return out


def build_occ_out(
    *,
    cams: Sequence[str],
    occ_ids: Any,
    occ_logits: Any,
    seen_ids: Any,
    counts: Mapping[str, int],
    pred: Mapping[str, Any] | None = None,
    vo: Any = None,
    dt_s: float | None = None,
    ego_source: str = "vo",
    carried_ids: Any = None,
    cam_samples: Mapping[str, Any] | None = None,
    poses: Mapping[str, Any] | None = None,
    live_cams: Sequence[bool] | None = None,
    timing_ms: Mapping[str, float] | None = None,
    peak_vram_mb: float | None = None,
    cpu_mem_est_mb: float | None = None,
) -> OccOutV2:
    """Numpy-only assembly. ``occ_logits`` ``(K, 1 + classes + FLOW_DIM)`` on ``occ_ids``; ``seen_ids`` =
    voxels a fresh camera ray wrote this tick (``observed``). ``live_cams[i]`` False = stale cam, left out."""
    if ego_source not in EGO_SOURCES:
        raise ValueError(f"ego source {ego_source!r}")
    pred = pred or {}
    ids = np.asarray(occ_ids, np.int64).reshape(-1)
    lg = np.asarray(occ_logits, np.float32).reshape(len(ids), -1)
    keep = (ids >= 0) & np.isfinite(lg).all(1)
    ids, lg = ids[keep][:MAX_OCC], lg[keep][:MAX_OCC]
    seen = np.unique(np.asarray(seen_ids, np.int64).reshape(-1))
    seen = seen[seen >= 0]
    n = len(OCC_CLASSES)
    prob = _sigmoid(lg[:, 0])
    cls = lg[:, 1 : 1 + n].argmax(1)
    occ = {"ids": ids.tolist(), "xyz": _r(voxel_centers(ids), 2), "prob": _r(prob), "cls": [OCC_CLASSES[int(k)] for k in cls],
           "flow": _r(lg[:, 1 + n : 1 + n + FLOW_DIM], 2), "observed": [bool(b) for b in np.isin(ids, seen)]}
    cls_of = dict(zip(occ["ids"], occ["cls"]))
    drive = voxel_centers(ids[(cls == OCC_CLASSES.index("drivable")) & (prob > 0.5)])[:, :2]
    ego: dict[str, Any] = {"dx": None, "dy": None, "dyaw": None, "dt_s": None, "speed_mps": None, "yaw_rate_rps": None,
                           "source": ego_source, "conf": None, "memory_xyz": []}
    if vo is not None:
        v = np.asarray(vo, np.float64).reshape(-1)
        ego.update(dx=round(float(v[0]), 3), dy=round(float(v[1]), 3), dyaw=round(float(v[2]), 4))
        if dt_s is not None and float(dt_s) > 0:
            ego.update(dt_s=round(float(dt_s), 4), speed_mps=round(float(math.hypot(v[0], v[1]) / float(dt_s)), 3),
                       yaw_rate_rps=round(float(v[2] / float(dt_s)), 4))
    if carried_ids is not None:
        c = np.asarray(carried_ids, np.int64).reshape(-1)
        ego["memory_xyz"] = _r(voxel_centers(c[c >= 0][:MAX_MEM]), 2)
    live = [True] * len(cams) if live_cams is None else [bool(x) for x in live_cams]
    samples: dict[str, Any] = {}
    if cam_samples is not None:
        d = np.asarray(cam_samples["depth_m"], np.float32)  # (n, H, W, K)
        p = np.asarray(cam_samples["prob"], np.float32)
        sid = np.asarray(cam_samples["ids"], np.int64)
        for ci, cam in enumerate(cams):
            if not live[ci]:
                continue  # masked cam: absent, not zeros
            r, col, k = (a[:MAX_CAM_SAMPLES] for a in np.nonzero(sid[ci] >= 0))
            v_ids = sid[ci][r, col, k]
            samples[cam] = {"row": r.tolist(), "col": col.tolist(), "depth_m": _r(d[ci][r, col, k], 2),
                            "prob": _r(np.clip(p[ci][r, col, k], 0, 1)), "cls": [cls_of.get(int(x), "unknown") for x in v_ids]}
    objects = _objects(pred)
    live_names = [c for c, ok in zip(cams, live) if ok]
    t = {s: (None if timing_ms is None or s not in timing_ms else round(float(timing_ms[s]), 2)) for s in STAGES}
    cnt = {k: int(counts[k]) for k in ("stored", "active", "frustum")}
    cnt.update(queried=len(occ["ids"]), full_grid=FULL_GRID)
    return OccOutV2(cams=list(cams), occ=occ, counts=cnt, lanes=_lanes(pred, seen), objects=objects, ego=ego,
                    corridor=_hull(drive), cam_samples=samples,
                    cam_boxes=_cam_boxes(objects, live_names, poses or {}),
                    debug={"ms": t, "peak_vram_mb": None if peak_vram_mb is None else round(float(peak_vram_mb), 1),
                           "peak_vram_src": "torch.cuda" if peak_vram_mb is not None else "UNPROVEN",
                           "cpu_mem_est_mb": None if cpu_mem_est_mb is None else round(float(cpu_mem_est_mb), 1)})


class OccOutError(ValueError):
    pass


def _need(ok: bool, msg: str) -> None:
    if not ok:
        raise OccOutError(msg)


def _keys(d: Any, kind: str, where: str) -> None:
    _need(isinstance(d, Mapping), f"{where}: not a mapping")
    got, want = set(d), KEYS[kind]
    _need(not (got - want), f"{where}: unknown keys {sorted(got - want)} (new fields need a version bump)")
    _need(not (want - got), f"{where}: missing keys {sorted(want - got)}")


def _num(x: Any, where: str, lo: float | None = None, hi: float | None = None, none_ok: bool = False) -> None:
    if x is None and none_ok:
        return
    _need(isinstance(x, (int, float)) and not isinstance(x, bool), f"{where}: not a number ({type(x).__name__})")
    _need(math.isfinite(float(x)), f"{where}: NaN/inf")
    _need(lo is None or float(x) >= lo, f"{where}: {x} < {lo}")
    _need(hi is None or float(x) <= hi, f"{where}: {x} > {hi}")


def _segs_cross(a: Any, b: Any, c: Any, d: Any) -> bool:
    def o(p: Any, q: Any, r: Any) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return o(a, b, c) * o(a, b, d) < 0 and o(c, d, a) * o(c, d, b) < 0


def validate_occ_out(d: Mapping[str, Any]) -> None:
    """Strict v2 schema. Raises ``OccOutError``: unknown/missing keys at any level, wrong types or lengths,
    NaN/inf, prob/conf outside [0, 1], unknown class/type/source, caps, counts != array lengths."""
    _keys(d, "", "occ_out")
    _need(d["v"] == OCC_OUT_VERSION and not isinstance(d["v"], bool), f"version {d['v']!r} != {OCC_OUT_VERSION}")
    cams = d["cams"]
    _need(isinstance(cams, list) and all(isinstance(c, str) for c in cams) and len(set(cams)) == len(cams), "cams")
    o = d["occ"]
    _keys(o, "occ", "occ")
    k = len(o["ids"])
    _need(k <= MAX_OCC, f"occ {k} > cap {MAX_OCC}")
    for name in KEYS["occ"]:
        _need(isinstance(o[name], list) and len(o[name]) == k, f"occ.{name}: length != {k}")
    _need(all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < FULL_GRID for i in o["ids"]), "occ.ids: not grid ids")
    _need(len(set(o["ids"])) == k, "occ.ids: duplicates")
    for i in range(k):
        _need(isinstance(o["xyz"][i], list) and len(o["xyz"][i]) == 3, "occ.xyz: shape")
        _need(isinstance(o["flow"][i], list) and len(o["flow"][i]) == FLOW_DIM, "occ.flow: shape")
        for x in o["xyz"][i] + o["flow"][i]:
            _num(x, "occ.xyz/flow")
        _num(o["prob"][i], "occ.prob", 0.0, 1.0)
        _need(o["cls"][i] in OCC_CLASSES, f"occ.cls {o['cls'][i]!r}")
        _need(isinstance(o["observed"][i], bool), "occ.observed: not bool")
    c = d["counts"]
    _keys(c, "counts", "counts")
    for name in KEYS["counts"]:
        _need(isinstance(c[name], int) and not isinstance(c[name], bool) and c[name] >= 0, f"counts.{name}")
    _need(c["queried"] == k, f"counts.queried {c['queried']} != len(occ.ids) {k}")
    _need(c["full_grid"] == FULL_GRID, "counts.full_grid")
    _need(c["stored"] <= c["queried"] <= c["full_grid"], f"counts: stored {c['stored']} <= queried {c['queried']} <= full")
    _need(c["active"] <= c["frustum"] <= c["full_grid"], "counts: active <= frustum <= full")
    _need(isinstance(d["lanes"], list) and len(d["lanes"]) <= MAX_LANES, "lanes cap")
    for ln in d["lanes"]:
        _keys(ln, "lane", "lane")
        _need(ln["type"] in LANE_TYPES and ln["style"] in LANE_STYLES, f"lane type/style {ln['type']!r}/{ln['style']!r}")
        m = len(ln["pts"])
        _need(m >= 2, "lane: < 2 points")
        _need(isinstance(ln["conf"], list) and len(ln["conf"]) == m, "lane.conf: one per point")
        _need(isinstance(ln["observed"], list) and len(ln["observed"]) == m, "lane.observed: one per point")
        for p, cf, ob in zip(ln["pts"], ln["conf"], ln["observed"]):
            _need(isinstance(p, list) and len(p) == 2, "lane.pts shape")
            _num(p[0], "lane.x"), _num(p[1], "lane.y"), _num(cf, "lane.conf", 0.0, 1.0)
            _need(isinstance(ob, bool), "lane.observed: not bool")
    _need(isinstance(d["objects"], list) and len(d["objects"]) <= MAX_OBJECTS, "objects cap")
    slots = set()
    for ob in d["objects"]:
        _keys(ob, "object", "object")
        _need(isinstance(ob["slot"], int) and ob["slot"] not in slots, "object.slot: unique int")
        slots.add(ob["slot"])
        for f in ("x", "y", "speed_mps", "sin_yaw", "cos_yaw"):
            _num(ob[f], f"object.{f}")
        _num(ob["l"], "object.l", 0.0), _num(ob["w"], "object.w", 0.0), _num(ob["conf"], "object.conf", 0.0, 1.0)
        _num(ob["z"], "object.z", none_ok=True), _num(ob["h"], "object.h", 0.0, none_ok=True)
        _need(abs(ob["sin_yaw"] ** 2 + ob["cos_yaw"] ** 2 - 1.0) < 1e-2, "object: sin^2+cos^2 != 1")
        _need(ob["cls"] in OBJECT_CLASSES, f"object.cls {ob['cls']!r}")
    e = d["ego"]
    _keys(e, "ego", "ego")
    _need(e["source"] in EGO_SOURCES, f"ego.source {e['source']!r}")
    for f in ("dx", "dy", "dyaw", "speed_mps", "yaw_rate_rps"):
        _num(e[f], f"ego.{f}", none_ok=True)
    _num(e["dt_s"], "ego.dt_s", 1e-6, none_ok=True)
    _num(e["conf"], "ego.conf", 0.0, 1.0, none_ok=True)
    _need(isinstance(e["memory_xyz"], list) and len(e["memory_xyz"]) <= MAX_MEM, "ego.memory_xyz cap")
    for p in e["memory_xyz"]:
        _need(isinstance(p, list) and len(p) == 3, "ego.memory_xyz shape")
        for x in p:
            _num(x, "ego.memory_xyz")
    poly = d["corridor"]
    _need(isinstance(poly, list) and len(poly) <= MAX_CORRIDOR, "corridor cap")
    if poly:
        _need(len(poly) >= 4 and poly[0] == poly[-1], "corridor: not a closed polygon (>= 3 vertices)")
        for p in poly:
            _need(isinstance(p, list) and len(p) == 2, "corridor shape")
            _num(p[0], "corridor.x"), _num(p[1], "corridor.y")
        segs = list(zip(poly[:-1], poly[1:]))
        for i in range(len(segs)):
            for j in range(i + 2, len(segs)):
                if i == 0 and j == len(segs) - 1:
                    continue
                _need(not _segs_cross(*segs[i], *segs[j]), "corridor: self-intersecting")
    for tab, kind, cap in (("cam_samples", "cam_samples", MAX_CAM_SAMPLES), ("cam_boxes", "cam_box", MAX_OBJECTS)):
        _need(isinstance(d[tab], Mapping) and set(d[tab]) <= set(cams), f"{tab}: cam not in cams")
    for cam, s in d["cam_samples"].items():
        _keys(s, "cam_samples", f"cam_samples.{cam}")
        m = len(s["row"])
        _need(m <= MAX_CAM_SAMPLES and all(isinstance(s[f], list) and len(s[f]) == m for f in KEYS["cam_samples"]), f"cam_samples.{cam}: lengths/cap")
        for i in range(m):
            _need(isinstance(s["row"][i], int) and isinstance(s["col"][i], int) and s["row"][i] >= 0 and s["col"][i] >= 0, "cam_samples px")
            _num(s["depth_m"][i], "cam_samples.depth_m", 0.0), _num(s["prob"][i], "cam_samples.prob", 0.0, 1.0)
            _need(s["cls"][i] in OCC_CLASSES, f"cam_samples.cls {s['cls'][i]!r}")
    for cam, boxes in d["cam_boxes"].items():
        _need(isinstance(boxes, list) and len(boxes) <= MAX_OBJECTS, "cam_boxes cap")
        for b in boxes:
            _keys(b, "cam_box", f"cam_boxes.{cam}")
            _need(b["slot"] in slots, f"cam_boxes.{cam}: slot {b['slot']} has no 3-D object")
            _need(b["cls"] == next(o["cls"] for o in d["objects"] if o["slot"] == b["slot"]), "cam_box cls != 3-D cls")
            for f in ("x0", "y0", "x1", "y1"):
                _num(b[f], f"cam_box.{f}", 0.0)
            _need(b["x0"] < b["x1"] and b["y0"] < b["y1"], "cam_box: empty")
    g = d["debug"]
    _keys(g, "debug", "debug")
    _need(isinstance(g["ms"], Mapping) and set(g["ms"]) == set(STAGES), "debug.ms stages")
    for s, v in g["ms"].items():
        _num(v, f"debug.ms.{s}", 0.0, none_ok=True)
    _num(g["peak_vram_mb"], "debug.peak_vram_mb", 0.0, none_ok=True)
    _num(g["cpu_mem_est_mb"], "debug.cpu_mem_est_mb", 0.0, none_ok=True)
    _need(g["peak_vram_src"] == ("UNPROVEN" if g["peak_vram_mb"] is None else "torch.cuda"), "debug.peak_vram_src")


def occ_out_to_viz(d: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """occ_out v2 -> gvd_scene viz v1 message (validated) + the list of dropped v2-only fields (``VIZ_V2_DELTA``)."""
    from python.viz.occ_scene import build_message, validate_message

    validate_occ_out(d)
    o = d["occ"]
    cells = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "cls": c, "vx": float(f[0]), "vy": float(f[1])}
             for p, c, f, pr in zip(o["xyz"], o["cls"], o["flow"], o["prob"]) if pr >= 0.5 and c != "unknown"]
    lanes = [{"points": ln["pts"], "paint": ln["type"], "kind": "detected"} for ln in d["lanes"] if ln["type"] in ("white", "yellow")]
    edges = [{"points": ln["pts"], "kind": "detected", "src": "curb" if ln["type"] == "curb" else "occ"} for ln in d["lanes"] if ln["type"] in ("curb", "road_edge")]
    tracks = [{"id": ob["slot"], "class": ob["cls"], "x": ob["x"], "y": ob["y"],
               "yaw": math.atan2(ob["sin_yaw"], ob["cos_yaw"]), "speed_mps": ob["speed_mps"], "motion": "unknown"} for ob in d["objects"]]
    msg = build_message({"lanes_ext": lanes, "road_edges": edges, "tracks": tracks, "signs": [], "occ_cells": cells})
    bad = validate_message(msg)
    if bad:
        raise OccOutError(f"viz v1 rejects the feed: {bad}")
    return msg, list(VIZ_V2_DELTA)
