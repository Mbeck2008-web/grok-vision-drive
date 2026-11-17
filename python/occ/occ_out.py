"""``occ_out`` v2: the one per-tick debug/viz record, built from model head outputs only.

Numpy only. Sparse and capped: every array has an absolute cap that does not
depend on the voxel grid, so the record size is O(caps) under grid growth.
No ground truth, scenetree or sim pose is read anywhere on this path; the VO
delta is the VO head's output, and ego speed / yaw rate come from it (source
``"vo"``) unless ``--ego-from-vehicle`` is on (source ``"vehicle"``).

Layout (strict: missing fields, unknown fields, wrong dtype/shape, NaN/inf,
out-of-range probabilities, unknown classes / lane types are all rejected;
any new field is a version bump; v1 records, or v1-shaped records labelled v2,
are rejected):

    schema "gvd.occ_out", version 2, t (s), seq (int)
    occ:      idx int32 (N,3) fine voxel ids, xyz float32 (N,3), prob float32 (N,),
              cls int8 (N,), flow float32 (N,2) (vx, vy m/s),
              n_stored == N <= n_queried <= n_full, n_active (active sparse cells),
              observed bool (N,) [v2]: True only if this tick's own lift (a camera ray's
              kept depth bin) hit the cell's 1 m parent; memory-carried / proposed -> False
    lanes:    type [L] in LANE_TYPES, pts [L] float32 (P,2) P>=2, conf [L] float32 (P,),
              observed [L] bool (P,) [v2]: nearest supporting cell within 1 m is observed
    objects:  xyz/lwh float32 (M,3), yaw_sin/yaw_cos float32 (M,) (heading = box yaw;
              arrows use it; no separate heading field), vel float32 (M,2),
              cls [M] in OBJECT_CLASSES, conf float32 (M,), track_id int32 (M,) (-1 none),
              motion [M] in MOTION_STATES, motion_frames int32 (M,)
    ego:      dt (s, from frame stamps), vo_delta float32 (3,) (dx, dy, dyaw), vo_valid,
              vo_conf [0,1], vo_cov float32 (3,) diag variances,
              speed {value m/s >= 0, source vo|vehicle, valid},
              yaw_rate {value rad/s left-positive, source vo|vehicle, valid}
    corridor: pts float32 (K,2) closed ego-frame polygon (first == last, K-1 >= 3
              distinct, simple) or empty (K=0) when confidence is low; conf [0,1]
    memory:   xyz float32 (K,3) warped top-N memory cells, w float32 (K,), n_total
    cams:     {cam: {uv float32 (S,2) in [0,1], depth float32 (S,), cls int8 (S,),
              conf float32 (S,), age_s, boxes: {xyxy float32 (B,4) in [0,1], track_id int32 (B,),
              cls [B]}}}; masked / stale cameras are ABSENT
    debug:    latency_ms {stage: ms from in-code timers}, peak_vram_mb "UNPROVEN" or None
              (until IQU45 nvidia-smi output exists), cpu_mem_est_mb float or None,
              cpu_mem_est_label str, counts {name: int}
"""
from __future__ import annotations

import base64
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from python.occ.classes import CURB, LANE_WHITE, LANE_YELLOW, N_CLASSES, ROAD, SRC_CURB_LINE, SRC_EDGE, SRC_LANE_LINE, SRC_OBJ_HEAD, SRC_STOP_LINE, STOP_LINE
from python.occ.grid import VoxelGrid

SCHEMA = "gvd.occ_out"
VERSION = 2  # v1 -> v2: occ.observed, lanes.observed (nothing else)
ALL_CAMS = ("narrow", "main", "wide", "pillarL", "pillarR", "repeatL", "repeatR", "rear")
LANE_TYPES = ("white_solid", "white_dashed", "yellow_solid", "yellow_dashed", "curb", "road_edge", "stop_line")
OBJECT_CLASSES = ("vehicle",)
MOTION_STATES = ("unknown", "parked", "same_way", "oncoming", "crossing", "alongside")
EGO_SOURCES = ("vo", "vehicle")
STAGES = ("backbone", "lift_query", "vo_head", "memory", "mid_obj_heads", "fine_head", "object_head", "panels", "tracker", "polylines", "build")
COUNT_KEYS = ("rays", "lifted_points", "active_cells", "memory_cells_in", "merged_cells", "memory_cells_out", "fine_queries", "cells_stored")
# Absolute caps (independent of the grid).
OCC_CAP = 4096
LANE_CAP = 32
LANE_PTS_CAP = 160
OBJ_CAP = 64
MEM_CAP = 256
PANEL_CAP = 256
BOX2D_CAP = 64
CORRIDOR_CAP = 65  # vertices incl. the closing repeat
CORRIDOR_MIN_CONF = 0.5
VRAM_UNPROVEN = "UNPROVEN"
CPU_EST_LABEL = "CPU ESTIMATE (forward hooks), not a GPU measurement"

# Fields occ_out carries that viz v1 (f3555bf) has no slot for: the v2 delta for the viz schema.
V2_DELTA = (
    "occ.observed", "lanes.observed (per point)", "occ.prob", "occ.flow", "occ.n_stored", "occ.n_queried", "occ.n_full", "occ.n_active",
    "lanes.conf (per point)", "lanes.type dashed/solid", "objects.vel (vx, vy)", "objects.conf", "objects.yaw_sin/yaw_cos (v1 has yaw)",
    "ego.vo_delta", "ego.vo_conf", "ego.vo_cov", "ego.yaw_rate", "ego.speed.source", "corridor", "memory", "cams (panels + 2D boxes)",
    "debug.latency_ms", "debug.peak_vram_mb", "debug.cpu_mem_est_mb", "debug.counts",
)

KEYS = {
    "": {"schema", "version", "t", "seq", "occ", "lanes", "objects", "ego", "corridor", "memory", "cams", "debug"},
    "occ": {"idx", "xyz", "prob", "cls", "flow", "n_stored", "n_queried", "n_full", "n_active", "observed"},
    "lanes": {"type", "pts", "conf", "observed"},
    "objects": {"xyz", "lwh", "yaw_sin", "yaw_cos", "vel", "cls", "conf", "track_id", "motion", "motion_frames"},
    "ego": {"dt", "vo_delta", "vo_valid", "vo_conf", "vo_cov", "speed", "yaw_rate"},
    "speed": {"value", "source", "valid"},
    "corridor": {"pts", "conf"},
    "memory": {"xyz", "w", "n_total"},
    "cam": {"uv", "depth", "cls", "conf", "age_s", "boxes"},
    "boxes": {"xyxy", "track_id", "cls"},
    "debug": {"latency_ms", "peak_vram_mb", "cpu_mem_est_mb", "cpu_mem_est_label", "counts"},
}


class OccOutError(ValueError):
    pass


# ------------------------------------------------------------------ validate
def _isnum(v: Any) -> bool:
    return isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, (bool, np.bool_)) and math.isfinite(float(v))


def _isint(v: Any) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


class _V:
    def __init__(self) -> None:
        self.errs: list[str] = []

    def keys(self, path: str, d: Any, kind: str) -> bool:
        if not isinstance(d, Mapping):
            self.errs.append(f"{path}: not a mapping")
            return False
        want = KEYS[kind]
        miss = sorted(want - set(d))
        extra = sorted(str(k) for k in set(d) - want)
        if miss:
            self.errs.append(f"{path}: missing {miss}")
        if extra:
            self.errs.append(f"{path}: unknown {extra} (new fields need occ_out v2)")
        return not miss

    def arr(self, path: str, a: Any, dtype: Any, shape: tuple, lo: float | None = None, hi: float | None = None) -> bool:
        if not isinstance(a, np.ndarray):
            self.errs.append(f"{path}: not an ndarray")
            return False
        if a.dtype != np.dtype(dtype):
            self.errs.append(f"{path}: dtype {a.dtype} != {np.dtype(dtype)}")
            return False
        if a.ndim != len(shape) or any(s is not None and a.shape[i] != s for i, s in enumerate(shape)):
            self.errs.append(f"{path}: shape {a.shape} != {shape}")
            return False
        if a.dtype.kind == "f" and not np.isfinite(a).all():
            self.errs.append(f"{path}: NaN/inf")
            return False
        if a.size and lo is not None and float(a.min()) < lo:
            self.errs.append(f"{path}: below {lo}")
            return False
        if a.size and hi is not None and float(a.max()) > hi:
            self.errs.append(f"{path}: above {hi}")
            return False
        return True

    def strs(self, path: str, v: Any, n: int, allowed: Sequence[str]) -> None:
        if not isinstance(v, list) or len(v) != n:
            self.errs.append(f"{path}: not a list of length {n}")
            return
        bad = [x for x in v if not isinstance(x, str) or x not in allowed]
        if bad:
            self.errs.append(f"{path}: unknown {sorted(set(map(str, bad)))[:4]}")

    def num(self, path: str, v: Any, lo: float | None = None, hi: float | None = None) -> None:
        if not _isnum(v):
            self.errs.append(f"{path}: not a finite number")
        elif (lo is not None and float(v) < lo) or (hi is not None and float(v) > hi):
            self.errs.append(f"{path}: out of range [{lo}, {hi}]")

    def cap(self, path: str, n: int, cap: int) -> None:
        if n > cap:
            self.errs.append(f"{path}: {n} > cap {cap}")


def _segments_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def polygon_is_simple(pts: np.ndarray) -> bool:
    """Closed polygon (pts[0] == pts[-1]) with no two non-adjacent edges properly crossing."""
    n = len(pts) - 1
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _segments_cross(pts[i], pts[i + 1], pts[j], pts[j + 1]):
                return False
    return True


def validate_occ_out(o: Any) -> list[str]:
    """Problems with ``o`` as occ_out v2 (empty = valid). Strict: unknown fields are errors."""
    v = _V()
    if not v.keys("occ_out", o, ""):
        return v.errs
    if o["schema"] != SCHEMA:
        v.errs.append("schema")
    if not (_isint(o["version"]) and o["version"] == VERSION):
        v.errs.append(f"version != {VERSION}")
    v.num("t", o["t"])
    if not (_isint(o["seq"]) and 0 <= o["seq"] < 2**63):
        v.errs.append("seq")
    occ = o["occ"]
    if v.keys("occ", occ, "occ"):
        n = occ["idx"].shape[0] if isinstance(occ["idx"], np.ndarray) and occ["idx"].ndim == 2 else -1
        v.arr("occ.idx", occ["idx"], np.int32, (None, 3), lo=0)
        v.arr("occ.xyz", occ["xyz"], np.float32, (n, 3))
        v.arr("occ.prob", occ["prob"], np.float32, (n,), 0.0, 1.0)
        if v.arr("occ.cls", occ["cls"], np.int8, (n,), 0, N_CLASSES - 1):
            pass
        v.arr("occ.flow", occ["flow"], np.float32, (n, 2), -100.0, 100.0)
        v.arr("occ.observed", occ["observed"], np.bool_, (n,))
        for k in ("n_stored", "n_queried", "n_full", "n_active"):
            if not (_isint(occ[k]) and occ[k] >= 0):
                v.errs.append(f"occ.{k}: not a non-negative int")
        if all(_isint(occ[k]) for k in ("n_stored", "n_queried", "n_full")):
            if occ["n_stored"] != n:
                v.errs.append(f"occ.n_stored {occ['n_stored']} != len {n}")
            if not (occ["n_stored"] <= occ["n_queried"] <= occ["n_full"]):
                v.errs.append("occ: need n_stored <= n_queried <= n_full")
        v.cap("occ", max(n, 0), OCC_CAP)
    ln = o["lanes"]
    if v.keys("lanes", ln, "lanes"):
        L = len(ln["type"]) if isinstance(ln["type"], list) else -1
        v.strs("lanes.type", ln["type"], L, LANE_TYPES)
        v.cap("lanes", max(L, 0), LANE_CAP)
        if not (isinstance(ln["pts"], list) and isinstance(ln["conf"], list) and isinstance(ln["observed"], list)
                and len(ln["pts"]) == L == len(ln["conf"]) == len(ln["observed"])):
            v.errs.append("lanes: type/pts/conf/observed lengths differ")
        else:
            for i, (p, c, ob_) in enumerate(zip(ln["pts"], ln["conf"], ln["observed"])):
                P = p.shape[0] if isinstance(p, np.ndarray) and p.ndim == 2 else -1
                if v.arr(f"lanes.pts[{i}]", p, np.float32, (None, 2)) and not (2 <= P <= LANE_PTS_CAP):
                    v.errs.append(f"lanes.pts[{i}]: {P} points not in [2, {LANE_PTS_CAP}]")
                v.arr(f"lanes.conf[{i}]", c, np.float32, (P,), 0.0, 1.0)
                v.arr(f"lanes.observed[{i}]", ob_, np.bool_, (P,))
    ob = o["objects"]
    if v.keys("objects", ob, "objects"):
        M = ob["xyz"].shape[0] if isinstance(ob["xyz"], np.ndarray) and ob["xyz"].ndim == 2 else -1
        v.arr("objects.xyz", ob["xyz"], np.float32, (None, 3))
        v.arr("objects.lwh", ob["lwh"], np.float32, (M, 3), 0.0, 50.0)
        ok_s = v.arr("objects.yaw_sin", ob["yaw_sin"], np.float32, (M,), -1.0, 1.0)
        ok_c = v.arr("objects.yaw_cos", ob["yaw_cos"], np.float32, (M,), -1.0, 1.0)
        if ok_s and ok_c and M > 0 and not np.allclose(ob["yaw_sin"].astype(np.float64) ** 2 + ob["yaw_cos"].astype(np.float64) ** 2, 1.0, atol=1e-3):
            v.errs.append("objects: sin^2 + cos^2 != 1")
        v.arr("objects.vel", ob["vel"], np.float32, (M, 2), -100.0, 100.0)
        v.strs("objects.cls", ob["cls"], M, OBJECT_CLASSES)
        v.arr("objects.conf", ob["conf"], np.float32, (M,), 0.0, 1.0)
        v.arr("objects.track_id", ob["track_id"], np.int32, (M,), -1)
        v.strs("objects.motion", ob["motion"], M, MOTION_STATES)
        if v.arr("objects.motion_frames", ob["motion_frames"], np.int32, (M,), 0) and M > 0 and isinstance(ob["motion"], list) and len(ob["motion"]) == M:
            if any(m != "unknown" and f < 3 for m, f in zip(ob["motion"], ob["motion_frames"].tolist())):
                v.errs.append("objects: motion state with < 3 frames")
        v.cap("objects", max(M, 0), OBJ_CAP)
    eg = o["ego"]
    if v.keys("ego", eg, "ego"):
        v.num("ego.dt", eg["dt"], 0.0)
        v.arr("ego.vo_delta", eg["vo_delta"], np.float32, (3,))
        if not isinstance(eg["vo_valid"], bool):
            v.errs.append("ego.vo_valid: not a bool")
        v.num("ego.vo_conf", eg["vo_conf"], 0.0, 1.0)
        v.arr("ego.vo_cov", eg["vo_cov"], np.float32, (3,), 0.0)
        for k in ("speed", "yaw_rate"):
            sp = eg[k]
            if v.keys(f"ego.{k}", sp, "speed"):
                v.num(f"ego.{k}.value", sp["value"], 0.0 if k == "speed" else None)
                if sp["source"] not in EGO_SOURCES:
                    v.errs.append(f"ego.{k}.source")
                if not isinstance(sp["valid"], bool):
                    v.errs.append(f"ego.{k}.valid")
    co = o["corridor"]
    if v.keys("corridor", co, "corridor"):
        v.num("corridor.conf", co["conf"], 0.0, 1.0)
        if v.arr("corridor.pts", co["pts"], np.float32, (None, 2)):
            K = co["pts"].shape[0]
            v.cap("corridor", K, CORRIDOR_CAP)
            if K:
                if K < 4 or not np.array_equal(co["pts"][0], co["pts"][-1]):
                    v.errs.append("corridor: not closed with >= 3 distinct vertices")
                elif not polygon_is_simple(co["pts"].astype(np.float64)):
                    v.errs.append("corridor: self-intersecting")
                if _isnum(co["conf"]) and float(co["conf"]) < CORRIDOR_MIN_CONF:
                    v.errs.append("corridor: low confidence must be empty")
    me = o["memory"]
    if v.keys("memory", me, "memory"):
        K = me["xyz"].shape[0] if isinstance(me["xyz"], np.ndarray) and me["xyz"].ndim == 2 else -1
        v.arr("memory.xyz", me["xyz"], np.float32, (None, 3))
        v.arr("memory.w", me["w"], np.float32, (K,), 0.0, 1.0)
        if not (_isint(me["n_total"]) and me["n_total"] >= max(K, 0)):
            v.errs.append("memory.n_total")
        v.cap("memory", max(K, 0), MEM_CAP)
    cams = o["cams"]
    if not isinstance(cams, Mapping):
        v.errs.append("cams: not a mapping")
    else:
        for cid, pan in cams.items():
            if cid not in ALL_CAMS:
                v.errs.append(f"cams: unknown camera {cid!r}")
                continue
            if not v.keys(f"cams.{cid}", pan, "cam"):
                continue
            S = pan["uv"].shape[0] if isinstance(pan["uv"], np.ndarray) and pan["uv"].ndim == 2 else -1
            v.arr(f"cams.{cid}.uv", pan["uv"], np.float32, (None, 2), 0.0, 1.0)
            v.arr(f"cams.{cid}.depth", pan["depth"], np.float32, (S,), 0.0)
            v.arr(f"cams.{cid}.cls", pan["cls"], np.int8, (S,), 0, N_CLASSES - 1)
            v.arr(f"cams.{cid}.conf", pan["conf"], np.float32, (S,), 0.0, 1.0)
            v.num(f"cams.{cid}.age_s", pan["age_s"], 0.0)
            v.cap(f"cams.{cid}", max(S, 0), PANEL_CAP)
            bx = pan["boxes"]
            if v.keys(f"cams.{cid}.boxes", bx, "boxes"):
                B = bx["xyxy"].shape[0] if isinstance(bx["xyxy"], np.ndarray) and bx["xyxy"].ndim == 2 else -1
                if v.arr(f"cams.{cid}.boxes.xyxy", bx["xyxy"], np.float32, (None, 4), 0.0, 1.0) and B > 0:
                    if not ((bx["xyxy"][:, 2] > bx["xyxy"][:, 0]) & (bx["xyxy"][:, 3] > bx["xyxy"][:, 1])).all():
                        v.errs.append(f"cams.{cid}.boxes: empty box")
                v.arr(f"cams.{cid}.boxes.track_id", bx["track_id"], np.int32, (B,), -1)
                v.strs(f"cams.{cid}.boxes.cls", bx["cls"], B, OBJECT_CLASSES)
                v.cap(f"cams.{cid}.boxes", max(B, 0), BOX2D_CAP)
    db = o["debug"]
    if v.keys("debug", db, "debug"):
        lat = db["latency_ms"]
        if not isinstance(lat, Mapping) or set(lat) - set(STAGES):
            v.errs.append("debug.latency_ms: unknown stage")
        else:
            for k, val in lat.items():
                v.num(f"debug.latency_ms.{k}", val, 0.0)
        if db["peak_vram_mb"] not in (VRAM_UNPROVEN, None):
            v.errs.append("debug.peak_vram_mb must be 'UNPROVEN' or None until IQU45 nvidia-smi output exists")
        if db["cpu_mem_est_mb"] is not None:
            v.num("debug.cpu_mem_est_mb", db["cpu_mem_est_mb"], 0.0)
        if db["cpu_mem_est_label"] != CPU_EST_LABEL:
            v.errs.append("debug.cpu_mem_est_label")
        cn = db["counts"]
        if not isinstance(cn, Mapping) or set(cn) != set(COUNT_KEYS) or not all(_isint(x) and x >= 0 for x in cn.values()):
            v.errs.append("debug.counts")
        elif isinstance(occ, Mapping) and _isint(occ.get("n_stored")) and cn["cells_stored"] != occ["n_stored"]:
            v.errs.append("debug.counts.cells_stored != occ.n_stored")
    return v.errs


def check_occ_out(o: Any) -> None:
    errs = validate_occ_out(o)
    if errs:
        raise OccOutError("; ".join(errs[:10]))


# ------------------------------------------------------------ serialization
def _enc(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return {"__nd__": [x.dtype.str, list(x.shape), base64.b64encode(np.ascontiguousarray(x).tobytes()).decode("ascii")]}
    if isinstance(x, Mapping):
        return {str(k): _enc(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_enc(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def _dec(x: Any) -> Any:
    if isinstance(x, dict):
        if set(x) == {"__nd__"}:
            dt, shape, data = x["__nd__"]
            return np.frombuffer(base64.b64decode(data), dtype=np.dtype(dt)).reshape(shape).copy()
        return {k: _dec(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_dec(v) for v in x]
    return x


def to_bytes(o: Mapping[str, Any]) -> bytes:
    """Lossless (bit-exact arrays) JSON encoding."""
    return json.dumps(_enc(o), separators=(",", ":"), allow_nan=False, sort_keys=True).encode("utf-8")


def from_bytes(b: bytes) -> dict[str, Any]:
    return _dec(json.loads(b.decode("utf-8")))


# ------------------------------------------------------------------- build
def _f32(a, shape) -> np.ndarray:
    return np.asarray(a, dtype=np.float32).reshape(shape)


def _point_observed(pts: np.ndarray, cxy: np.ndarray, obs: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Per polyline point: observed flag of the nearest supporting cell within ``radius`` (False if none)."""
    out = np.zeros(len(pts), bool)
    if not len(cxy):
        return out
    for i in range(len(pts)):
        di = np.linalg.norm(cxy - pts[i], axis=1)
        j = int(np.argmin(di))
        out[i] = bool(obs[j]) if di[j] <= radius else False
    return out


def _point_conf(pts: np.ndarray, cxy: np.ndarray, conf: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Per polyline point: conf of the nearest supporting cell within ``radius`` (0 if none)."""
    if not len(cxy):
        return np.zeros(len(pts), np.float32)
    d = np.linalg.norm(pts[:, None, :] - cxy[None, :, :], axis=2) if len(pts) * len(cxy) <= 400_000 else None
    out = np.zeros(len(pts), np.float32)
    for i in range(len(pts)):
        di = d[i] if d is not None else np.linalg.norm(cxy - pts[i], axis=1)
        j = int(np.argmin(di))
        out[i] = conf[j] if di[j] <= radius else 0.0
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0).astype(np.float32)


def merge_dashes(lanes: Sequence[Mapping[str, Any]], *, max_gap_m: float = 8.0, max_dx_m: float = 0.75) -> list[dict[str, Any]]:
    """Join collinear same-paint lane segments; a joined line with a gap > 1.5 m is ``dashed``."""
    segs = []
    for ln in lanes:
        pts = np.asarray(ln.get("pts", []), np.float64).reshape(-1, 2)
        if len(pts) >= 2 and np.isfinite(pts).all():
            segs.append({"paint": "yellow" if ln.get("paint") == "yellow" else "white", "pts": pts[np.argsort(pts[:, 1], kind="stable")]})
    segs.sort(key=lambda s: (s["paint"], s["pts"][0, 1]))
    out: list[dict[str, Any]] = []
    for s in segs:
        for o in out:
            if o["paint"] == s["paint"] and 0.0 <= s["pts"][0, 1] - o["pts"][-1, 1] <= max_gap_m and abs(s["pts"][0, 0] - o["pts"][-1, 0]) <= max_dx_m:
                o["pts"] = np.concatenate([o["pts"], s["pts"]])
                break
        else:
            out.append(dict(s))
    for o in out:
        gaps = np.diff(o["pts"][:, 1])
        o["style"] = "dashed" if len(gaps) and gaps.max() > 1.5 else "solid"
    return out


def corridor_from_cells(idx: np.ndarray, cls: np.ndarray, conf: np.ndarray, grid: VoxelGrid, *, max_rows: int = (CORRIDOR_CAP - 1) // 2) -> tuple[np.ndarray, float]:
    """Drivable corridor polygon (closed, y-monotone so simple) from ROAD/paint cells ahead of the ego.

    Per 1 m row from y=0 forward: the run of drivable cells containing (or nearest) the previous
    row's centre (x=0 first). Stops at the first row with no run. Empty when < 2 rows or the mean
    cell confidence is below CORRIDOR_MIN_CONF."""
    empty = np.zeros((0, 2), np.float32)
    idx = np.asarray(idx, np.int64).reshape(-1, 3)
    cls = np.asarray(cls).reshape(-1)
    conf = np.asarray(conf, np.float64).reshape(-1)
    sel = np.isin(cls, [ROAD, LANE_WHITE, LANE_YELLOW, STOP_LINE])
    if not sel.any():
        return empty, 0.0
    p, c = idx[sel], conf[sel]
    v = grid.voxel
    row_of = (p[:, 1] // 2)  # 1 m rows
    start = int(math.floor((0.0 - grid.y_min) / v)) // 2
    centre = (0.0 - grid.x_min) / v
    left, right, confs = [], [], []
    r = start
    while len(left) < max_rows:
        m = row_of == r
        if not m.any():
            break
        xs = np.unique(p[m, 0])
        runs, a = [], xs[0]
        for i in range(1, len(xs) + 1):
            if i == len(xs) or xs[i] != xs[i - 1] + 1:
                runs.append((a, xs[i - 1]))
                if i < len(xs):
                    a = xs[i]
        best = min(runs, key=lambda ab: 0.0 if ab[0] <= centre <= ab[1] + 1 else min(abs(ab[0] - centre), abs(ab[1] + 1 - centre)))
        if not (best[0] - 2 <= centre <= best[1] + 3):
            break
        y = grid.y_min + (r * 2 + 1) * v
        left.append((grid.x_min + best[0] * v, y))
        right.append((grid.x_min + (best[1] + 1) * v, y))
        confs.append(float(np.mean(c[m & (p[:, 0] >= best[0]) & (p[:, 0] <= best[1])])))
        centre = (best[0] + best[1] + 1) / 2.0
        r += 1
    cf = float(np.clip(np.mean(confs), 0.0, 1.0)) if confs else 0.0
    if len(left) < 2 or cf < CORRIDOR_MIN_CONF:
        return empty, cf
    ring = left + right[::-1]
    ring.append(ring[0])
    return np.asarray(ring, np.float32), cf


def box_corners(x: float, y: float, z: float, l: float, w: float, h: float, yaw: float) -> np.ndarray:
    """8 corners (ego frame) of a box whose length axis points along ``yaw`` (radians from +x)."""
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for dl in (-0.5, 0.5):
        for dw in (-0.5, 0.5):
            for dh in (-0.5, 0.5):
                out.append((x + dl * l * c - dw * w * s, y + dl * l * s + dw * w * c, z + dh * h))
    return np.asarray(out, np.float64)


def boxes_2d(objects: Mapping[str, np.ndarray], cam) -> dict[str, Any]:
    """Project 3-D object boxes into one camera; clipped to the image, normalised to [0,1]. Same track ids."""
    xy, tid, cl = [], [], []
    for i in range(objects["xyz"].shape[0]):
        x, y, z = (float(v) for v in objects["xyz"][i])
        l, w, h = (float(v) for v in objects["lwh"][i])
        yaw = math.atan2(float(objects["yaw_sin"][i]), float(objects["yaw_cos"][i]))
        uv, depth, _ = cam.project(box_corners(x, y, z, l, w, h, yaw))
        if not (np.asarray(depth) > 0.1).all():
            continue  # partly behind the camera: no honest 2-D box
        u0, v0 = np.clip(uv.min(axis=0), 0, [cam.width, cam.height])
        u1, v1 = np.clip(uv.max(axis=0), 0, [cam.width, cam.height])
        if u1 - u0 < 1.0 or v1 - v0 < 1.0:
            continue
        xy.append([u0 / cam.width, v0 / cam.height, u1 / cam.width, v1 / cam.height])
        tid.append(int(objects["track_id"][i]))
        cl.append(objects["cls"][i])
        if len(xy) >= BOX2D_CAP:
            break
    return {"xyxy": np.clip(np.asarray(xy, np.float32).reshape(-1, 4), 0.0, 1.0), "track_id": np.asarray(tid, np.int32), "cls": cl}


def build_occ_out(*, t: float, seq: int, heads: Mapping[str, Any], tracks: Sequence[Mapping[str, Any]], lanes: Sequence[Mapping[str, Any]],
                  edges: Sequence[Mapping[str, Any]], stop_lines: Sequence[Mapping[str, Any]], grid: VoxelGrid, cams: Mapping[str, Any],
                  dt: float | None, ego_from_vehicle: bool, latency_ms: Mapping[str, float], cpu_mem_est_mb: float | None = None) -> dict[str, Any]:
    """Assemble occ_out v2 from head outputs (+ tracker / polylines, which only consume head outputs)."""
    cells = heads["cells"]
    st = heads.get("stats") or {}
    idx = np.asarray(cells["idx"], np.int32).reshape(-1, 3)
    cls = np.asarray(cells["cls"], np.int8).reshape(-1)
    conf = np.clip(np.nan_to_num(np.asarray(cells["conf"], np.float32).reshape(-1), nan=0.0), 0.0, 1.0)
    flow = np.clip(np.nan_to_num(np.asarray(cells.get("flow", np.zeros((len(idx), 2))), np.float32).reshape(-1, 2), nan=0.0, posinf=0.0, neginf=0.0), -100, 100)
    obs = np.asarray(cells.get("observed", np.zeros(len(idx), bool)), bool).reshape(-1)
    if obs.shape[0] != len(idx):
        raise OccOutError(f"cells.observed length {obs.shape[0]} != {len(idx)} cells")
    good = (idx >= 0).all(1) & (idx < np.array(grid.shape)).all(1) & (cls >= 0) & (cls < N_CLASSES)
    idx, cls, conf, flow, obs = idx[good], cls[good], conf[good], flow[good], obs[good]
    if len(idx) > OCC_CAP:
        o = np.argsort(-conf, kind="stable")[:OCC_CAP]
        idx, cls, conf, flow, obs = idx[o], cls[o], conf[o], flow[o], obs[o]
    xyz = grid.to_center(idx).astype(np.float32) if len(idx) else np.zeros((0, 3), np.float32)
    n_full = int(grid.n_dense)
    n_q = int(min(max(int(st.get("fine_queries", len(idx))), len(idx)), n_full))
    # ---- lanes / edges / stop lines with per-point confidence
    cxy = xyz[:, :2].astype(np.float64)
    types, pts_l, conf_l, obs_l = [], [], [], []

    def add(kind: str, pts: np.ndarray, support: np.ndarray) -> None:
        if len(types) >= LANE_CAP or len(pts) < 2 or not np.isfinite(pts).all():
            return
        pts = pts[:LANE_PTS_CAP].astype(np.float32)
        types.append(kind)
        pts_l.append(pts)
        conf_l.append(_point_conf(pts.astype(np.float64), cxy[support], conf[support]))
        obs_l.append(_point_observed(pts.astype(np.float64), cxy[support], obs[support]))

    paint_m = np.isin(cls, [LANE_WHITE, LANE_YELLOW])
    for ln in merge_dashes(lanes):
        add(f"{ln['paint']}_{ln['style']}", ln["pts"], paint_m)
    for ed in edges:
        ty = ed.get("type")
        if ty in ("curb", "road_edge"):
            add(ty, np.asarray(ed.get("pts", []), np.float64).reshape(-1, 2), cls == (CURB if ty == "curb" else ROAD))
    for bar in stop_lines:
        try:
            add("stop_line", np.array([[bar["x0"], bar["y"]], [bar["x1"], bar["y"]]], np.float64), cls == STOP_LINE)
        except (KeyError, TypeError, ValueError):
            continue
    # ---- objects (heading = box yaw)
    rows = [r for r in tracks if all(_isnum(r.get(k)) for k in ("x", "y"))][:OBJ_CAP]
    M = len(rows)
    yaw = np.array([float(r["yaw"]) if _isnum(r.get("yaw")) else math.pi / 2 for r in rows], np.float64)
    objects = {
        "xyz": _f32([[r["x"], r["y"], (float(r["h"]) if _isnum(r.get("h")) else 1.5) / 2] for r in rows], (M, 3)),
        "lwh": np.clip(_f32([[float(r.get(k)) if _isnum(r.get(k)) else d for k, d in (("l", 4.5), ("w", 1.9), ("h", 1.5))] for r in rows], (M, 3)), 0.0, 50.0),
        "yaw_sin": np.sin(yaw).astype(np.float32), "yaw_cos": np.cos(yaw).astype(np.float32),
        "vel": np.clip(_f32([[float(r["vx"]) if _isnum(r.get("vx")) else 0.0, float(r["vy"]) if _isnum(r.get("vy")) else 0.0] for r in rows], (M, 2)), -100, 100),
        "cls": ["vehicle" for _ in rows],
        "conf": np.clip(_f32([float(r["conf"]) if _isnum(r.get("conf")) else 0.0 for r in rows], (M,)), 0.0, 1.0),
        "track_id": np.asarray([int(r["id"]) if _isint(r.get("id")) and 0 <= r["id"] < 2**31 else -1 for r in rows], np.int32),
        "motion": [r.get("motion") if r.get("motion") in MOTION_STATES and _isint(r.get("motion_frames")) and r["motion_frames"] >= 3 else "unknown" for r in rows],
        "motion_frames": np.asarray([int(r["motion_frames"]) if _isint(r.get("motion_frames")) and 0 <= r["motion_frames"] < 2**31 else 0 for r in rows], np.int32),
    }
    # ---- ego: VO delta from the VO head; speed / yaw rate over the real frame dt
    vo = heads.get("vo")
    vo_ok = vo is not None and len(vo) == 3 and all(_isnum(x) for x in vo)
    vo_delta = np.asarray(vo if vo_ok else (0.0, 0.0, 0.0), np.float32)
    cov = heads.get("vo_cov")
    vo_cov = np.clip(np.nan_to_num(np.asarray(cov if (vo_ok and cov is not None) else (0.0, 0.0, 0.0), np.float32), nan=1e6, posinf=1e6), 0.0, 1e6)
    src = "vehicle" if ego_from_vehicle else "vo"
    mo = heads.get("motion_used") if ego_from_vehicle else (list(vo_delta) if vo_ok else None)
    dt_ok = dt is not None and _isnum(dt) and dt > 0
    mo_ok = mo is not None and len(mo) == 3 and all(_isnum(x) for x in mo)
    valid = bool(dt_ok and mo_ok)
    spd = float(math.hypot(float(mo[0]), float(mo[1])) / dt) if valid else 0.0
    yr = float(float(mo[2]) / dt) if valid else 0.0
    ego = {"dt": float(dt) if dt_ok else 0.0, "vo_delta": vo_delta, "vo_valid": bool(vo_ok), "vo_conf": float(np.clip(heads.get("vo_conf") or 0.0, 0, 1)) if vo_ok else 0.0,
           "vo_cov": vo_cov, "speed": {"value": spd if math.isfinite(spd) else 0.0, "source": src, "valid": valid and math.isfinite(spd)},
           "yaw_rate": {"value": yr if math.isfinite(yr) else 0.0, "source": src, "valid": valid and math.isfinite(yr)}}
    # ---- corridor
    cpts, ccf = corridor_from_cells(idx, cls, conf, grid)
    # ---- memory
    mp = heads.get("memory_pts") or {}
    mxyz = np.nan_to_num(np.asarray(mp.get("xyz", np.zeros((0, 3))), np.float32).reshape(-1, 3))[:MEM_CAP]
    mw = np.clip(np.nan_to_num(np.asarray(mp.get("w", np.zeros(0)), np.float32).reshape(-1)), 0, 1)[: len(mxyz)]
    memory = {"xyz": mxyz, "w": mw, "n_total": int(max(int(mp.get("n_total", len(mxyz))), len(mxyz)))}
    # ---- camera panels (fresh cameras only) + projected 2-D boxes
    panels = {}
    for cid, pan in (heads.get("cam_panels") or {}).items():
        if cid not in ALL_CAMS or cid not in cams:
            continue
        uv = np.clip(np.nan_to_num(np.asarray(pan["uv"], np.float32).reshape(-1, 2)), 0, 1)[:PANEL_CAP]
        S = len(uv)
        panels[cid] = {"uv": uv, "depth": np.clip(np.nan_to_num(np.asarray(pan["depth"], np.float32).reshape(-1)[:S]), 0, None),
                       "cls": np.asarray(pan["cls"], np.int8).reshape(-1)[:S], "conf": np.clip(np.nan_to_num(np.asarray(pan["conf"], np.float32).reshape(-1)[:S]), 0, 1),
                       "age_s": float(max(0.0, pan.get("age_s", 0.0))) if _isnum(pan.get("age_s")) else 0.0, "boxes": boxes_2d(objects, cams[cid])}
    counts = {k: int(max(0, st.get(k, 0))) for k in COUNT_KEYS}
    counts["cells_stored"] = int(len(idx))
    out = {
        "schema": SCHEMA, "version": VERSION, "t": float(t), "seq": int(seq) % 2**63,
        "occ": {"idx": idx, "xyz": xyz, "prob": conf.astype(np.float32), "cls": cls, "flow": flow.astype(np.float32), "n_stored": int(len(idx)),
                "n_queried": n_q, "n_full": n_full, "n_active": int(max(0, st.get("merged_cells", 0))), "observed": obs},
        "lanes": {"type": types, "pts": pts_l, "conf": conf_l, "observed": obs_l},
        "objects": objects, "ego": ego, "corridor": {"pts": cpts, "conf": float(ccf)}, "memory": memory, "cams": panels,
        "debug": {"latency_ms": {k: float(max(0.0, v)) for k, v in latency_ms.items() if k in STAGES and _isnum(v)}, "peak_vram_mb": VRAM_UNPROVEN,
                  "cpu_mem_est_mb": None if cpu_mem_est_mb is None else float(cpu_mem_est_mb), "cpu_mem_est_label": CPU_EST_LABEL, "counts": counts},
    }
    return out


def occ_out_to_outputs(o: Mapping[str, Any]) -> dict[str, Any]:
    """occ_out v2 -> the ``outputs`` dict the (unchanged, GVD-owned) ``occ_to_viz`` v1 reads, so viz is drawn from
    occ_out. v2-only fields (``observed``) are dropped here and listed in V2_DELTA; viz moves to v2 on GVD's track."""
    if o.get("version") != VERSION:
        raise OccOutError(f"occ_out_to_outputs needs occ_out v{VERSION}, got {o.get('version')!r}")
    ob = o["objects"]
    tracks = []
    for i in range(ob["xyz"].shape[0]):
        row = {"x": float(ob["xyz"][i, 0]), "y": float(ob["xyz"][i, 1]), "cls": ob["cls"][i], "yaw": math.atan2(float(ob["yaw_sin"][i]), float(ob["yaw_cos"][i])),
               "l": float(ob["lwh"][i, 0]), "w": float(ob["lwh"][i, 1]), "h": float(ob["lwh"][i, 2]), "motion": ob["motion"][i],
               "motion_frames": int(ob["motion_frames"][i]), "src": SRC_OBJ_HEAD, "conf": float(ob["conf"][i])}
        if int(ob["track_id"][i]) >= 0:
            row["id"] = int(ob["track_id"][i])
        tracks.append(row)
    lanes, edges, bars = [], [], []
    for ty, pts in zip(o["lanes"]["type"], o["lanes"]["pts"]):
        p = pts.astype(np.float64).tolist()
        if ty in ("curb", "road_edge"):
            edges.append({"pts": p, "type": ty, "src": SRC_EDGE if ty == "road_edge" else SRC_CURB_LINE, "kind": "predicted"})
        elif ty == "stop_line":
            bars.append({"x0": p[0][0], "x1": p[-1][0], "y": p[0][1], "src": SRC_STOP_LINE})
        else:
            lanes.append({"pts": p, "paint": ty.split("_")[0], "src": SRC_LANE_LINE, "kind": "predicted"})
    occ = o["occ"]
    return {"t": float(o["t"]), "cells": {"idx": occ["idx"], "cls": occ["cls"], "conf": occ["prob"]}, "tracks": tracks, "edges": edges, "lanes": lanes, "stop_lines": bars}
