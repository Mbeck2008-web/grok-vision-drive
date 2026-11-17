"""``occ_out`` v1: one versioned per-tick record carrying every debug-viz tab, built ONLY from
sparse net outputs (``occ_sparse.infer_sparse``). No pixels, no ground truth, no dense tensors.

Tabs (docs/specs/occ_sparse.md, "occ_out v1"):

* ``occ``      active cells: voxel idx, xyz, prob, sem class + conf, flow (vx, vy); counts.
* ``lanes``    polylines with type in LANE_TYPES and per-point conf. Colour and curb/edge come from
               the sem head; solid vs dashed is geometric (segment length and gaps along the
               chained polyline), NOT a trained output.
* ``objects``  box (xyz, lwh, yaw sin/cos), velocity, class, conf, track id. Cells of one vehicle
               vote with the box head for a shared centre; height is the cells' z-extent (no
               recorded height exists to train it).
* ``ego``      VO delta (dx right, dy forward, dyaw ccw), a weight-based conf (heuristic, not a
               calibrated covariance), and the memory point set: earlier ticks' active cells
               re-expressed in the current ego frame through the runtime ego pose, top-N.
* ``cams``     per camera: active cells projected to normalised uv, depth along the optical axis,
               sem class. Capped per camera; masked cameras are skipped.
* ``debug``    per-stage ms (code timers), ``peak_vram_mb`` (torch.cuda reading on GPU, else None;
               UNPROVEN until IQU45 nvidia-smi), ``cpu_mem_est_mb`` (a labelled estimate, never merged), counts.
* ``ego.speed_mps`` / ``ego.yaw_rate`` = |(dx, dy)| / dt and dyaw / dt with dt from tick timestamps;
  ``ego.source`` is "vo" unless ``--ego-from-vehicle``. ``ego.corridor``: closed drivable polygon
  (ego frame) from the left/right drivable boundary, empty when too few confident cells.
* Object heading for top-down arrows IS the box yaw (``yaw_sin``/``yaw_cos``); no second field.
  ``cams[].boxes`` are projections of the 3D boxes (same ``track_id``), clipped to the image.
Unknown keys anywhere are rejected by :func:`validate_occ_out`; new fields mean v2.

numpy only at build time (inputs may be torch tensors; they are converted).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from python.perception.occ_viz import CLASS_NAMES, OccTracker, _axis, _edges, bev_components

OCC_OUT_VERSION = 1
LANE_TYPES = ("white_solid", "white_dashed", "yellow_solid", "yellow_dashed", "curb", "road_edge")
OCC_OUT_CAPS = {"cells": 16384, "lanes": 16, "lane_pts": 120, "objects": 64, "mem_pts": 2048, "cam_pts": 1024,
                "cams": 8, "corridor": 240}
SCHEMA = {  # exact key sets; anything else is rejected
    "occ": {"idx", "xyz", "prob", "sem", "sem_conf", "flow", "counts"},
    "counts": {"stored", "queried", "full"},
    "lane": {"type", "pts", "conf"},
    "object": {"xyz", "lwh", "yaw_sin", "yaw_cos", "vel", "cls", "conf", "n_cells", "track_id"},
    "ego": {"delta", "conf", "cov", "speed_mps", "yaw_rate", "dt", "source", "corridor", "memory_pts"},
    "cam": {"cam_id", "uv", "depth", "sem", "boxes"},
    "box2d": {"track_id", "cls", "conf", "uv_min", "uv_max"},
    "debug": {"stage_ms", "peak_vram_mb", "cpu_mem_est_mb", "counts"},
}
OBJ_CLASSES = ("vehicle",)
DASH_MAX_M = 4.5      # a paint segment shorter than this is a dash
DASH_LINK_M = 13.0    # dashes closer than this end-to-start (and aligned) chain into one polyline
DASH_LINK_COS = 0.95
OBJ_HASH_M = 1.0
CORRIDOR_MIN_ROWS = 3   # fewer confident drivable rows than this -> empty corridor
CORRIDOR_MIN_CONF = 0.5      # predicted box centres within one hash cell (8-connected) are one object
TOP_KEYS = ("version", "t", "seq", "occ", "lanes", "objects", "ego", "cams", "debug", "provenance")
FORBIDDEN = ("image", "frame", "pixel", "gt", "ground_truth", "label", "rgb")
PROVENANCE = {
    "occ": "heads occ/sem/flow",
    "lanes": "sem head (white/yellow/curb/drivable); solid vs dashed geometric, untrained",
    "objects": "sem vehicle cells + box head (trained on recorded boxes); h from cell z-extent; ids OccTracker",
    "ego": "VO head; conf = weighted live-cam fraction (heuristic); cov UNLABELED",
    "cams": "active cells projected through rig poses",
    "debug": "code timers; peak_vram_mb torch.cuda (None on CPU, UNPROVEN until IQU45 nvidia-smi); cpu_mem_est_mb estimate",
}


def _np(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _components(ij: np.ndarray) -> np.ndarray:
    """8-connected components that also join duplicate cells (bev_components keys one row per cell)."""
    if len(ij) == 0:
        return np.zeros(0, np.int64)
    u, inv = np.unique(ij, axis=0, return_inverse=True)
    return bev_components(u)[inv.reshape(-1)]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _binned(xy: np.ndarray, conf: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Ordered polyline (bin along the principal axis) with per-point mean conf and its length."""
    c, a = _axis(xy)
    n = np.array([-a[1], a[0]])
    s, t = (xy - c) @ a, (xy - c) @ n
    b = np.floor(s / step).astype(np.int64)
    pts, cf = [], []
    for k in np.unique(b):
        m = b == k
        pts.append(c + a * float(s[m].mean()) + n * float(t[m].mean()))
        cf.append(float(conf[m].mean()))
    return np.array(pts, np.float32).reshape(-1, 2), np.array(cf, np.float32), float(s.max() - s.min())


def _paint_lanes(xy: np.ndarray, conf: np.ndarray, color: str, voxel: float) -> list[dict[str, Any]]:
    if len(xy) < 3:
        return []
    lab = _components(np.floor(xy / voxel).astype(np.int64))
    segs = []
    for k in np.unique(lab):
        m = lab == k
        if m.sum() < 3:
            continue
        p, cf, ln = _binned(xy[m], conf[m], voxel)
        if len(p) >= 2:
            segs.append([p, cf, ln])
    segs.sort(key=lambda sg: float(sg[0][0, 1]))
    out: list[dict[str, Any]] = []
    used = [False] * len(segs)
    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        p, cf, ln = segs[i]
        if ln >= DASH_MAX_M:
            out.append({"type": f"{color}_solid", "pts": p, "conf": cf})
            continue
        chain_p, chain_c, n_dash = [p], [cf], 1
        for j in range(i + 1, len(segs)):
            q, qc, qln = segs[j]
            if used[j] or qln >= DASH_MAX_M:
                continue
            end, start = chain_p[-1][-1], q[0]
            gap = start - end
            d_prev = chain_p[-1][-1] - chain_p[-1][0]
            nrm = float(np.linalg.norm(gap)) * max(1e-6, float(np.linalg.norm(d_prev)))
            if 0 < float(np.linalg.norm(gap)) < DASH_LINK_M and float(gap @ d_prev) / nrm > DASH_LINK_COS:
                chain_p.append(q)
                chain_c.append(qc)
                used[j] = True
                n_dash += 1
        kind = "dashed" if n_dash >= 2 else "solid"  # one short isolated segment: no evidence of a gap pattern
        out.append({"type": f"{color}_{kind}", "pts": np.concatenate(chain_p), "conf": np.concatenate(chain_c)})
    return out


def _cap(p: np.ndarray, c: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    if len(p) <= n:
        return p, c
    k = np.linspace(0, len(p) - 1, n).round().astype(int)
    return p[k], c[k]


def _ego_to_world(xy: np.ndarray, pose: Sequence[float]) -> np.ndarray:
    x, y, fx, fy = map(float, pose)
    r, f = np.array([fy, -fx]), np.array([fx, fy])
    return np.array([x, y]) + xy[:, :1] * r + xy[:, 1:2] * f


def _world_to_ego(w: np.ndarray, pose: Sequence[float]) -> np.ndarray:
    x, y, fx, fy = map(float, pose)
    d = w - np.array([x, y])
    return np.stack([d @ np.array([fy, -fx]), d @ np.array([fx, fy])], 1)


@dataclass
class OccOutV1:
    t: float
    seq: int
    occ: dict[str, Any]
    lanes: list[dict[str, Any]]
    objects: list[dict[str, Any]]
    ego: dict[str, Any]
    cams: list[dict[str, Any]]
    debug: dict[str, Any]
    provenance: dict[str, str] = field(default_factory=lambda: dict(PROVENANCE))
    version: int = OCC_OUT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready (lists, floats rounded to mm / 1e-3)."""
        def conv(v: Any) -> Any:
            if isinstance(v, np.ndarray):
                return np.round(v, 3).tolist() if v.dtype.kind == "f" else v.tolist()
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [conv(x) for x in v]
            if isinstance(v, (np.floating, float)):
                return round(float(v), 3)
            if isinstance(v, np.integer):
                return int(v)
            return v
        return {k: conv(getattr(self, k)) for k in TOP_KEYS}


class OccOutBuilder:
    """Stateful per-stream builder (track ids, memory points). One per camera rig/session."""

    def __init__(self, net: Any, *, max_mem_ticks: int = 2) -> None:
        self.grid = net.grid
        self.cam_ids = tuple(net.cam_ids)
        self.cam_pos = _np(net.cam_pos).astype(np.float64)
        self.cam_rot = _np(net.cam_rot).astype(np.float64)
        self.tan_hv = _np(net.tan_hv).astype(np.float64)
        self.tracker = OccTracker()
        self.mem: list[np.ndarray] = []  # world-frame (x, y, z, prob) of earlier ticks, newest last
        self.max_mem_ticks = int(max_mem_ticks)
        self.seq = 0
        self.t_prev: float | None = None
        self.device_cuda = next(net.parameters()).device.type == "cuda"

    def build(self, res: Mapping[str, Any], *, t: float, cam_valid: Sequence[float] | None = None,
              cpu_mem_est_mb: float | None = None) -> OccOutV1:
        g = self.grid
        idx = _np(res["idx"]).astype(np.int64).reshape(-1, 3)
        prob = _np(res["prob"]).astype(np.float32).reshape(-1)
        semp = _softmax(_np(res["sem"]).astype(np.float32).reshape(len(idx), -1)) if len(idx) else np.zeros((0, len(CLASS_NAMES)), np.float32)
        sem = semp.argmax(-1).astype(np.uint8) if len(idx) else np.zeros(0, np.uint8)
        semc = semp.max(-1).astype(np.float32) if len(idx) else np.zeros(0, np.float32)
        flow = _np(res["flow"]).astype(np.float32).reshape(-1, 2)
        box = _np(res.get("box", np.zeros((len(idx), 6)))).astype(np.float32).reshape(-1, 6)
        xyz = g.centers(idx).astype(np.float32)
        order = np.argsort(-prob, kind="stable")[:OCC_OUT_CAPS["cells"]]
        idx, prob, sem, semc, flow, box, xyz = (a[order] for a in (idx, prob, sem, semc, flow, box, xyz))
        full = int(np.prod(g.dims))
        counts = {"stored": int(len(idx)), "queried": int(res.get("evaluated", 0)), "full": full}
        occ = {"idx": idx.astype(np.int16), "xyz": xyz, "prob": prob, "sem": sem, "sem_conf": semc, "flow": flow,
               "counts": counts}

        # lanes
        v = g.voxel_m
        conf = prob * semc
        lanes: list[dict[str, Any]] = []
        for cname, color in (("lane_white", "white"), ("lane_yellow", "yellow")):
            m = sem == CLASS_NAMES.index(cname)
            lanes += _paint_lanes(xyz[m, :2].astype(np.float64), conf[m], color, v)
        m = sem == CLASS_NAMES.index("curb")
        if m.sum() >= 3:
            lab = _components(np.floor(xyz[m, :2] / v).astype(np.int64))
            for k in np.unique(lab):
                mk = lab == k
                if mk.sum() >= 3:
                    p, cf, _ = _binned(xyz[m][mk, :2].astype(np.float64), conf[m][mk], v)
                    lanes.append({"type": "curb", "pts": p, "conf": cf})
        m = sem == CLASS_NAMES.index("drivable")
        if m.sum() >= 2:
            dxy, dcf = xyz[m, :2].astype(np.float64), conf[m]
            row = np.floor(dxy[:, 1] / v).astype(np.int64)
            for ed in _edges(dxy, v):
                p = np.array(ed["pts"], np.float32)
                cf = np.array([float(dcf[row == int(math.floor(q[1] / v))].max(initial=0.0)) for q in p], np.float32)
                lanes.append({"type": "road_edge", "pts": p, "conf": cf})
        lanes = [dict(zip(("type", "pts", "conf"), (ln["type"], *_cap(ln["pts"], ln["conf"], OCC_OUT_CAPS["lane_pts"]))))
                 for ln in lanes[:OCC_OUT_CAPS["lanes"]]]

        # objects
        objects: list[dict[str, Any]] = []
        m = sem == CLASS_NAMES.index("vehicle")
        if m.any():
            cxy = xyz[m, :2] + box[m, :2]
            lab = _components(np.floor(cxy / OBJ_HASH_M).astype(np.int64))
            dets = []
            for k in np.unique(lab):
                mk = lab == k
                w = conf[m][mk] + 1e-6
                bx = box[m][mk]
                z = xyz[m][mk, 2]
                yaw_v = (bx[:, 4:6] * w[:, None]).sum(0)
                yaw_v = yaw_v / max(1e-6, float(np.linalg.norm(yaw_v)))
                c = (cxy[mk] * w[:, None]).sum(0) / w.sum()
                h = float(z.max() - z.min() + v)
                objects.append({"xyz": np.array([c[0], c[1], (z.max() + z.min()) / 2], np.float32),
                                "lwh": np.array([float(np.exp((bx[:, 2] * w).sum() / w.sum())),
                                                 float(np.exp((bx[:, 3] * w).sum() / w.sum())), h], np.float32),
                                "yaw_sin": float(yaw_v[0]), "yaw_cos": float(yaw_v[1]),
                                "vel": (flow[m][mk] * w[:, None]).sum(0) / w.sum(), "cls": "vehicle",
                                "conf": float(conf[m][mk].max()), "n_cells": int(mk.sum()), "track_id": None})
                dets.append({"x": float(c[0]), "y": float(c[1]), "class": "vehicle",
                             "axis_yaw": math.atan2(float(yaw_v[0]), float(yaw_v[1]))})
            objects = objects[:OCC_OUT_CAPS["objects"]]
            dets = dets[:OCC_OUT_CAPS["objects"]]
        else:
            dets = []
        dt = 0.0 if self.t_prev is None else max(0.0, float(t) - self.t_prev)
        d = res.get("ego_delta")
        spd, yr = (0.0, 0.0) if d is None or dt <= 0 else (math.hypot(d[0], d[1]) / dt, d[2] / dt)
        for o, row in zip(objects, self.tracker.update(dets, float(t), ego_speed_mps=spd, yaw_rate=yr)):
            o["track_id"] = int(row["id"])

        # ego + memory points
        pose = res.get("ego_pose")
        mem_pts = np.zeros((0, 4), np.float32)
        if pose is not None:
            if self.mem:
                w = np.concatenate(self.mem[::-1])
                e = _world_to_ego(w[:, :2], pose)
                mem_pts = np.concatenate([e, w[:, 2:4]], 1).astype(np.float32)[:OCC_OUT_CAPS["mem_pts"]]
            top = np.argsort(-prob, kind="stable")[:OCC_OUT_CAPS["mem_pts"]]
            cur = np.concatenate([_ego_to_world(xyz[top, :2].astype(np.float64), pose), xyz[top, 2:3], prob[top, None]], 1)
            self.mem = (self.mem + [cur])[-self.max_mem_ticks:]
        cv = np.ones(len(self.cam_ids)) if cam_valid is None else np.asarray(cam_valid, np.float64)
        corridor = np.zeros((0, 2), np.float32)
        m = (sem == CLASS_NAMES.index("drivable")) & (conf >= CORRIDOR_MIN_CONF)
        if m.sum() >= 2:
            dxy = xyz[m, :2].astype(np.float64)
            row = np.floor(dxy[:, 1] / v).astype(np.int64)
            rows_u = np.unique(row)
            if len(rows_u) >= CORRIDOR_MIN_ROWS:
                k = np.linspace(0, len(rows_u) - 1, min(len(rows_u), OCC_OUT_CAPS["corridor"] // 2)).round().astype(int)
                left = [[float(dxy[row == r, 0].min()) - v / 2, (r + 0.5) * v] for r in rows_u[k]]
                right = [[float(dxy[row == r, 0].max()) + v / 2, (r + 0.5) * v] for r in rows_u[k]]
                corridor = np.array(left + right[::-1], np.float32)  # closed ring (last joins first)
        ego = {"delta": None if d is None else np.asarray(d, np.float32), "conf": None if d is None else float(cv.mean()),
               "cov": None, "speed_mps": None if d is None or dt <= 0 else float(spd),
               "yaw_rate": None if d is None or dt <= 0 else float(yr), "dt": float(dt),
               "source": str(res.get("ego_source", "vo")), "corridor": corridor, "memory_pts": mem_pts}

        # camera panels
        cams = []
        for n, cid in enumerate(self.cam_ids[:OCC_OUT_CAPS["cams"]]):
            if cv[n] <= 0:
                continue
            c = (xyz.astype(np.float64) - self.cam_pos[n]) @ self.cam_rot[n].T
            zc = c[:, 2]
            zs = np.maximum(zc, 1e-3)
            u, vv = c[:, 0] / (zs * self.tan_hv[n, 0]), -c[:, 1] / (zs * self.tan_hv[n, 1])
            vis = np.nonzero((zc > 0.1) & (np.abs(u) <= 1) & (np.abs(vv) <= 1))[0][:OCC_OUT_CAPS["cam_pts"]]
            boxes = []
            for o in objects:
                cs, sn = o["yaw_cos"], o["yaw_sin"]
                ln, wd, ht = (float(a) for a in o["lwh"])
                corners = np.array([[sx * ln / 2, sy * wd / 2, sz * ht / 2] for sx in (-1, 1) for sy in (-1, 1)
                                    for sz in (-1, 1)])
                # box yaw is from +x (pi/2 = ego heading); length runs along the heading
                rx = np.stack([corners[:, 0] * cs - corners[:, 1] * sn, corners[:, 0] * sn + corners[:, 1] * cs,
                               corners[:, 2]], 1) + np.asarray(o["xyz"], np.float64)
                cc = (rx - self.cam_pos[n]) @ self.cam_rot[n].T
                if (cc[:, 2] <= 0.1).any():
                    continue
                pu = cc[:, 0] / (cc[:, 2] * self.tan_hv[n, 0])
                pv = -cc[:, 1] / (cc[:, 2] * self.tan_hv[n, 1])
                lo_uv = np.clip([pu.min(), pv.min()], -1, 1)
                hi_uv = np.clip([pu.max(), pv.max()], -1, 1)
                if (hi_uv > lo_uv).all():
                    boxes.append({"track_id": o["track_id"], "cls": o["cls"], "conf": o["conf"],
                                  "uv_min": lo_uv.astype(np.float32), "uv_max": hi_uv.astype(np.float32)})
            cams.append({"cam_id": cid, "uv": np.stack([u[vis], vv[vis]], 1).astype(np.float32),
                         "depth": zc[vis].astype(np.float32), "sem": sem[vis], "boxes": boxes[:OCC_OUT_CAPS["objects"]]})

        vram = None
        if self.device_cuda:
            import torch
            vram = float(torch.cuda.max_memory_allocated() / 2**20)
        debug = {"stage_ms": {k: float(x) for k, x in dict(res.get("stage_ms", {})).items()}, "peak_vram_mb": vram,
                 "cpu_mem_est_mb": None if cpu_mem_est_mb is None else float(cpu_mem_est_mb), "counts": dict(counts)}
        self.t_prev = float(t)
        self.seq += 1
        return OccOutV1(t=float(t), seq=self.seq - 1, occ=occ, lanes=lanes, objects=objects, ego=ego, cams=cams,
                        debug=debug)


def validate_occ_out(d: Mapping[str, Any]) -> list[str]:
    """Schema check on ``OccOutV1.to_dict()``: exact keys (unknown -> error), types, shapes, finite
    values, ranges, caps, count consistency, no image/GT fields."""
    errs: list[str] = []

    def num(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

    def unit(v: Any) -> bool:
        return num(v) and 0.0 <= v <= 1.0

    def rows(v: Any, w: int, cap: int) -> bool:
        return isinstance(v, list) and len(v) <= cap and all(
            isinstance(r, list) and len(r) == w and all(num(x) for x in r) for r in v)

    def keys(obj: Any, name: str, where: str) -> bool:
        if not isinstance(obj, Mapping):
            errs.append(f"{where}: not a mapping")
            return False
        extra, missing = set(obj) - SCHEMA[name], SCHEMA[name] - set(obj)
        if extra or missing:
            errs.append(f"{where}: unknown {sorted(map(str, extra))} missing {sorted(missing)}")
        for k in obj:
            if any(tok in FORBIDDEN for tok in str(k).lower().split("_")):
                errs.append(f"forbidden key {where}.{k}")
        return not missing

    if not isinstance(d, Mapping) or set(d) != set(TOP_KEYS):
        errs.append(f"top keys {sorted(set(d) ^ set(TOP_KEYS)) if isinstance(d, Mapping) else d!r}")
        return errs
    if d["version"] != OCC_OUT_VERSION or not num(d["t"]) or not isinstance(d["seq"], int) or isinstance(d["seq"], bool):
        errs.append("version/t/seq")
    if d["provenance"] != PROVENANCE:
        errs.append("provenance")
    o = d["occ"]
    if keys(o, "occ", "occ"):
        n = len(o["prob"]) if isinstance(o["prob"], list) else -1
        cap = OCC_OUT_CAPS["cells"]
        if not (rows(o["idx"], 3, cap) and rows(o["xyz"], 3, cap) and rows(o["flow"], 2, cap)):
            errs.append("occ arrays")
        if not all(isinstance(o[k], list) and len(o[k]) == n for k in ("idx", "xyz", "sem", "sem_conf", "flow")):
            errs.append("occ lengths")
        if not (all(unit(p) for p in o["prob"]) and all(unit(p) for p in o["sem_conf"])):
            errs.append("occ prob/conf range")
        if not all(isinstance(s, int) and 0 <= s < len(CLASS_NAMES) for s in o["sem"]):
            errs.append("occ sem class")
        c = o["counts"]
        if keys(c, "counts", "occ.counts"):
            if not (all(isinstance(c[k], int) for k in SCHEMA["counts"]) and c["stored"] <= c["queried"] <= c["full"]
                    and c["stored"] == n):
                errs.append("occ counts")
    else:
        c = None
    if not isinstance(d["lanes"], list) or len(d["lanes"]) > OCC_OUT_CAPS["lanes"]:
        errs.append("lanes cap")
    for ln in d["lanes"] if isinstance(d["lanes"], list) else []:
        if keys(ln, "lane", "lane") and (ln["type"] not in LANE_TYPES or not rows(ln["pts"], 2, OCC_OUT_CAPS["lane_pts"])
                                        or len(ln["pts"]) < 2 or not isinstance(ln["conf"], list)
                                        or len(ln["conf"]) != len(ln["pts"]) or not all(unit(x) for x in ln["conf"])):
            errs.append(f"lane {ln.get('type')}")
    objs = d["objects"] if isinstance(d["objects"], list) else []
    if not isinstance(d["objects"], list) or len(objs) > OCC_OUT_CAPS["objects"]:
        errs.append("objects cap")
    for ob in objs:
        if keys(ob, "object", "object") and not (
                rows([ob["xyz"]], 3, 1) and rows([ob["lwh"]], 3, 1) and rows([ob["vel"]], 2, 1)
                and all(x > 0 for x in ob["lwh"]) and num(ob["yaw_sin"]) and num(ob["yaw_cos"]) and unit(ob["conf"])
                and ob["cls"] in OBJ_CLASSES and isinstance(ob["n_cells"], int)
                and (ob["track_id"] is None or isinstance(ob["track_id"], int))
                and abs(ob["yaw_sin"] ** 2 + ob["yaw_cos"] ** 2 - 1) < 0.01):
            errs.append("object fields")
    e = d["ego"]
    if keys(e, "ego", "ego"):
        ok = ((e["delta"] is None or rows([e["delta"]], 3, 1)) and rows(e["memory_pts"], 4, OCC_OUT_CAPS["mem_pts"])
              and e["cov"] is None and (e["conf"] is None or unit(e["conf"])) and num(e["dt"]) and e["dt"] >= 0
              and e["source"] in ("vo", "vehicle") and rows(e["corridor"], 2, OCC_OUT_CAPS["corridor"])
              and (len(e["corridor"]) == 0 or len(e["corridor"]) >= 3)
              and all(x is None or num(x) for x in (e["speed_mps"], e["yaw_rate"])))
        if not ok:
            errs.append("ego")
    cams = d["cams"] if isinstance(d["cams"], list) else []
    if not isinstance(d["cams"], list) or len(cams) > OCC_OUT_CAPS["cams"]:
        errs.append("cams cap")
    for cm in cams:
        if not keys(cm, "cam", "cam"):
            continue
        k = len(cm["depth"]) if isinstance(cm["depth"], list) else -1
        if not (isinstance(cm["cam_id"], str) and rows(cm["uv"], 2, OCC_OUT_CAPS["cam_pts"])
                and len(cm["uv"]) == k == len(cm["sem"]) and all(num(z) and z > 0 for z in cm["depth"])
                and all(-1 <= x <= 1 for r in cm["uv"] for x in r)
                and all(isinstance(s, int) and 0 <= s < len(CLASS_NAMES) for s in cm["sem"])
                and isinstance(cm["boxes"], list) and len(cm["boxes"]) <= OCC_OUT_CAPS["objects"]):
            errs.append(f"cam {cm.get('cam_id')}")
            continue
        for bx in cm["boxes"]:
            if keys(bx, "box2d", "box2d") and not (
                    rows([bx["uv_min"]], 2, 1) and rows([bx["uv_max"]], 2, 1) and unit(bx["conf"])
                    and bx["cls"] in OBJ_CLASSES and (bx["track_id"] is None or isinstance(bx["track_id"], int))
                    and all(-1 <= x <= 1 for x in bx["uv_min"] + bx["uv_max"])
                    and all(a < b for a, b in zip(bx["uv_min"], bx["uv_max"]))):
                errs.append("box2d fields")
    dbg = d["debug"]
    if keys(dbg, "debug", "debug"):
        if not (isinstance(dbg["stage_ms"], Mapping) and all(num(x) and x >= 0 for x in dbg["stage_ms"].values())
                and (dbg["peak_vram_mb"] is None or num(dbg["peak_vram_mb"]))
                and (dbg["cpu_mem_est_mb"] is None or num(dbg["cpu_mem_est_mb"])) and dbg["counts"] == c):
            errs.append("debug")
    return errs


# occ_out v1 -> gvd_scene v1. Fields v1 has no slot for (the v2 delta), listed for the viz track:
VIZ_V2_DELTA = ("occ.cells(prob,sem,flow)", "lanes[].conf per point", "lanes[].type dashed/solid", "objects[].lwh.h",
                "objects[].vel", "objects[].conf", "ego.delta/conf/yaw_rate/source", "ego.corridor", "ego.memory_pts",
                "cams[] (uv, depth, sem, 2D boxes)", "debug (stage_ms, vram, counts)")


def occ_out_to_viz(d: Mapping[str, Any]) -> dict[str, Any]:
    """Map a validated occ_out v1 dict to a gvd_scene v1 message (v1 keys only)."""
    msg: dict[str, Any] = {
        "v": 1, "type": "gvd_scene", "seq": int(d["seq"]), "t": round(float(d["t"]), 4),
        "ego": {"engaged": False, "speed_mps": round(float(d["ego"]["speed_mps"] or 0.0), 2)},
        "aeb": "off", "lanes": [], "edges": [], "tracks": [], "signs": [], "path": {"pts": []}, "path_conf": 0.0,
    }
    for ln in d["lanes"]:
        if ln["type"] in ("curb", "road_edge"):
            if len(msg["edges"]) < 8:
                msg["edges"].append({"pts": ln["pts"], "kind": "detected"})
        elif len(msg["lanes"]) < 16:
            msg["lanes"].append({"pts": ln["pts"], "paint": ln["type"].split("_")[0], "kind": "detected"})
    for ob in d["objects"]:
        msg["tracks"].append({"id": -1 if ob["track_id"] is None else int(ob["track_id"]), "cls": ob["cls"],
                              "x": ob["xyz"][0], "y": ob["xyz"][1],
                              "yaw": round(math.atan2(ob["yaw_sin"], ob["yaw_cos"]), 3),
                              "l": ob["lwh"][0], "w": ob["lwh"][1]})
    return msg
