#!/usr/bin/env python3
"""occ_out v2 contract: strict schema, properties, serialization, viz mapping (numpy only, no torch).

INVARIANTS
 O1  every occ_out the builder emits validates; arrays are length-consistent,
     probs/confs in [0,1], no NaN/inf, sin^2+cos^2 = 1, n_stored == len <=
     n_queried <= n_full, counts equal array lengths, polylines >= 2 points
     with one conf per point, corridor empty or closed + simple + capped.
 O2  strict: a missing field, an unknown field (new fields need v2), wrong
     dtype/shape, NaN/inf, out-of-range prob/conf, unknown class / lane type /
     camera, count mismatch, numeric peak_vram_mb are all rejected.
 O3  to_bytes/from_bytes round-trips bit-exact.
 O4  byte size is O(caps): flat under 64x grid growth (incl. numpy index arrays).
 O5  ego: speed = |(dx,dy)|/dt and yaw_rate = dyaw/dt with dt from the frame
     stamps (never a constant), speed >= 0, source "vo" unless
     --ego-from-vehicle; the VO delta is the VO head's output only.
 O6  2-D boxes are projections of the 3-D boxes: same track ids, clipped to
     the image; masked cameras are absent.
 O7  peak_vram_mb is "UNPROVEN"/None, separate from cpu_mem_est_mb.
 O8  occ_to_viz never silently ignores unknown keys (counted drop or reject);
     viz drawn from occ_out matches it; track ids stable across ticks.
 O9  the builder's import trace has no torch / GT / sim modules.

    py -3 scripts/test_occ_out.py
"""
from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from python.occ import classes as C  # noqa: E402
from python.occ import occ_out as O  # noqa: E402
from python.occ import viz_adapter as V  # noqa: E402
from python.occ.camera import CameraModel  # noqa: E402
from python.occ.downstream import Downstream  # noqa: E402
from python.occ.grid import VoxelGrid  # noqa: E402

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    HAVE_HYP = True
    settings.register_profile("ci", max_examples=60, deadline=None, derandomize=True, suppress_health_check=list(HealthCheck))
    settings.register_profile("fuzz", max_examples=1500, deadline=None, suppress_health_check=list(HealthCheck))
    settings.load_profile(os.environ.get("GVD_HYPOTHESIS_PROFILE", "ci"))
except ImportError:
    HAVE_HYP = False

G = VoxelGrid()
CAMS = {
    "main": CameraModel("main", (0.0, 1.2, 1.26), 0.0, -6.0, 34.0, 640, 480),
    "wide": CameraModel("wide", (-0.03, 1.2, 1.24), 0.0, -8.0, 88.7, 640, 360),
    "rear": CameraModel("rear", (0.0, -2.25, 0.78), 180.0, -10.0, 70.0, 640, 360),
}
NAN = float("nan")


# ------------------------------------------------------------------ fixtures
def road_scene(width=7.0, y1=40.0, conf=0.9, paint=True):
    """Synthetic predicted cells: road strip ahead, two lane lines (one dashed), a curb, a stop bar."""
    v = G.voxel
    iz = int((0.0 - G.z_min) / v)
    cells, cls = [], []
    for iy in range(int((0 - G.y_min) / v), int((y1 - G.y_min) / v)):
        for ix in range(int((-width / 2 - G.x_min) / v), int((width / 2 - G.x_min) / v)):
            x = G.x_min + (ix + 0.5) * v
            y = G.y_min + (iy + 0.5) * v
            k = C.ROAD
            if paint and abs(x - 1.75) < 0.26 and (y % 9.0) < 3.0:
                k = C.LANE_WHITE
            elif paint and abs(x + 1.75) < 0.26:
                k = C.LANE_YELLOW
            elif paint and abs(y - 30.0) < 0.26 and -1.75 < x < 1.75:
                k = C.STOP_LINE
            cells.append([ix, iy, iz])
            cls.append(k)
        cells.append([int((width / 2 - G.x_min) / v), iy, iz + 1])
        cls.append(C.CURB)
    idx = np.asarray(cells, np.int32)
    return {"idx": idx, "cls": np.asarray(cls, np.int8), "conf": np.full(len(idx), conf, np.float32), "flow": np.zeros((len(idx), 2), np.float32),
            "observed": idx[:, 1] < int((25.0 - G.y_min) / v)}  # far cells: memory-carried / proposed (not hit this tick)


def heads_for(cells, *, t=0.0, vo=(0.0, 1.0, 0.0), objects=None, panels=("main", "wide"), ego_from_vehicle=False, motion_used=None):
    st_ = {k: 0 for k in O.COUNT_KEYS}
    st_.update({"fine_queries": int(len(cells["idx"]) + 10), "merged_cells": 100, "rays": 400, "lifted_points": 300, "active_cells": 120})
    return {"t": t, "cells": cells, "objects": objects or [], "motion_used": list(motion_used) if motion_used is not None else (list(vo) if vo is not None else None),
            "stats": st_, "vo": None if vo is None else list(vo), "vo_conf": 0.7, "vo_cov": [0.01, 0.01, 0.0001],
            "memory_pts": {"xyz": np.zeros((5, 3), np.float32), "w": np.full(5, 0.5, np.float32), "n_total": 40},
            "cam_panels": {c: {"uv": np.full((3, 2), 0.5, np.float32), "depth": np.full(3, 10.0, np.float32), "cls": np.ones(3, np.int8),
                               "conf": np.full(3, 0.8, np.float32), "age_s": 0.01, "weight": 1.0} for c in panels},
            "timing_ms": {"backbone": 3.0, "lift_query": 1.0}, "engaged": False, "ego_from_vehicle": ego_from_vehicle}


def make_valid(t=0.0, **kw):
    ds = Downstream(G, CAMS, cpu_mem_est_mb=744.0)
    ds(heads_for(road_scene(), t=t - 0.1, **kw))
    return ds(heads_for(road_scene(), t=t, **kw))["occ_out"]


def _deep_equal(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a.view(np.uint8) if a.size else a, b.view(np.uint8) if b.size else b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def check_properties(o):
    assert O.validate_occ_out(o) == [], O.validate_occ_out(o)
    occ = o["occ"]
    n = len(occ["idx"])
    assert all(len(occ[k]) == n for k in ("xyz", "prob", "cls", "flow")) and occ["n_stored"] == n
    assert occ["n_stored"] <= occ["n_queried"] <= occ["n_full"]
    assert o["debug"]["counts"]["cells_stored"] == n
    assert o["version"] == 2 and occ["observed"].dtype == np.bool_ and len(occ["observed"]) == n
    un = ~occ["observed"]  # observed=False cells: finite prob in [0,1] and a valid class
    assert np.isfinite(occ["prob"][un]).all() and ((occ["prob"][un] >= 0) & (occ["prob"][un] <= 1)).all()
    assert ((occ["cls"][un] >= 0) & (occ["cls"][un] < C.N_CLASSES)).all()
    assert len(o["lanes"]["observed"]) == len(o["lanes"]["pts"])
    for p, c, ob_ in zip(o["lanes"]["pts"], o["lanes"]["conf"], o["lanes"]["observed"]):
        assert len(p) >= 2 and len(c) == len(p) and ((c >= 0) & (c <= 1)).all()
        assert ob_.dtype == np.bool_ and len(ob_) == len(p)
    ob = o["objects"]
    if len(ob["xyz"]):
        assert np.allclose(ob["yaw_sin"].astype(np.float64) ** 2 + ob["yaw_cos"].astype(np.float64) ** 2, 1, atol=1e-3)
    for pan in o["cams"].values():
        b = pan["boxes"]["xyxy"]
        assert ((b >= 0) & (b <= 1)).all()
        assert set(pan["boxes"]["track_id"].tolist()) <= set(ob["track_id"].tolist())
    cp = o["corridor"]["pts"]
    assert len(cp) == 0 or (len(cp) >= 4 and np.array_equal(cp[0], cp[-1]) and O.polygon_is_simple(cp.astype(np.float64)) and len(cp) <= O.CORRIDOR_CAP)
    sp, yr = o["ego"]["speed"], o["ego"]["yaw_rate"]
    assert sp["value"] >= 0 and math.isfinite(sp["value"]) and math.isfinite(yr["value"])
    back = O.from_bytes(O.to_bytes(o))
    assert _deep_equal(o, back)
    assert O.validate_occ_out(back) == []


# ------------------------------------------------------------------ O1 / O3
def test_builder_scene_valid():
    o = make_valid(t=1.0, objects=[{"cls": "vehicle", "x": 0.0, "y": 15.0, "yaw": math.pi / 2, "conf": 0.9, "src": "boxes:traffic"}])
    check_properties(o)
    types_ = o["lanes"]["type"]
    assert "white_dashed" in types_ and "yellow_solid" in types_ and "stop_line" in types_ and "curb" in types_, types_
    cp = o["corridor"]["pts"]
    assert len(cp) >= 4 and o["corridor"]["conf"] >= O.CORRIDOR_MIN_CONF
    assert cp[:, 0].min() >= -3.6 and cp[:, 0].max() <= 3.6 and cp[:, 1].min() >= 0  # ego frame, the road strip
    assert o["objects"]["xyz"].shape == (1, 3)
    # 2-D boxes: projected from the 3-D box, clipped, same track id
    assert o["cams"]["main"]["boxes"]["xyxy"].shape[0] == 1
    print(f"O1 example occ_out: {len(O.to_bytes(o))} bytes, lanes {types_}, corridor {len(cp)} vertices, cams {sorted(o['cams'])}")


def test_low_confidence_corridor_empty():
    o = make_valid(t=1.0)
    ds = Downstream(G, CAMS)
    o = ds(heads_for(road_scene(conf=0.4), t=0.0))["occ_out"]
    assert o["corridor"]["pts"].shape == (0, 2)
    pts, cf = O.corridor_from_cells(np.zeros((0, 3), np.int32), np.zeros(0, np.int8), np.zeros(0, np.float32), G)
    assert pts.shape == (0, 2) and cf == 0.0


if HAVE_HYP:
    cell_st = st.lists(st.tuples(st.integers(-3, 90), st.integers(-3, 170), st.integers(-2, 9), st.integers(-2, 10),
                                 st.one_of(st.floats(-1, 2), st.just(NAN))), max_size=400)
    track_st = st.lists(st.fixed_dictionaries({"x": st.one_of(st.floats(-30, 30), st.just(NAN)), "y": st.floats(-15, 80)},
                                              optional={"yaw": st.one_of(st.floats(-7, 7), st.just(NAN)), "l": st.floats(-1, 20), "w": st.floats(0, 5),
                                                        "h": st.floats(0, 5), "conf": st.floats(-1, 2), "id": st.integers(-5, 2**33),
                                                        "motion": st.sampled_from(["parked", "oncoming", "bogus", "unknown"]), "motion_frames": st.integers(-1, 9),
                                                        "vx": st.floats(-300, 300), "vy": st.one_of(st.floats(-50, 50), st.just(NAN))}), max_size=80)

    def _cells(raw):
        a = np.array([r[:4] for r in raw], np.int64).reshape(-1, 4)
        return {"idx": a[:, :3].astype(np.int32), "cls": a[:, 3].astype(np.int8), "conf": np.array([r[4] for r in raw], np.float32).reshape(-1),
                "flow": np.zeros((len(a), 2), np.float32), "observed": np.array([bool(r[4] > 0.5) for r in raw], bool).reshape(-1)}

    @given(cell_st, track_st, st.floats(0.001, 2.0), st.one_of(st.none(), st.tuples(st.floats(-6, 6), st.floats(-6, 6), st.floats(-0.5, 0.5))),
           st.sets(st.sampled_from(sorted(CAMS))), st.booleans())
    def test_builder_properties(raw, tracks, dt, vo, panels, efv):
        cells = _cells(raw)
        ds = Downstream(G, CAMS)
        ds.tracker.update = lambda t, dets, **k: [dict(r) for r in tracks]  # inject arbitrary tracker rows
        ds(heads_for(cells, t=5.0, vo=vo, panels=sorted(panels), ego_from_vehicle=efv))
        o = ds(heads_for(cells, t=5.0 + dt, vo=vo, panels=sorted(panels), ego_from_vehicle=efv))["occ_out"]
        check_properties(o)
        assert set(o["cams"]) == set(panels)
        eg = o["ego"]
        assert abs(eg["dt"] - dt) < 1e-9  # from the frame stamps
        src = "vehicle" if efv else "vo"
        assert eg["speed"]["source"] == src and eg["yaw_rate"]["source"] == src
        if vo is not None:
            assert eg["vo_valid"] and np.array_equal(eg["vo_delta"], np.asarray(vo, np.float32))
            sp = math.hypot(float(np.float32(vo[0])), float(np.float32(vo[1]))) / dt if not efv else math.hypot(vo[0], vo[1]) / dt
            assert abs(eg["speed"]["value"] - sp) <= 1e-4 * max(1.0, sp)
            yr = (float(np.float32(vo[2])) if not efv else vo[2]) / dt
            assert abs(eg["yaw_rate"]["value"] - yr) <= 1e-4 * max(1.0, abs(yr))
        elif not efv:
            assert not eg["vo_valid"] and not eg["speed"]["valid"] and eg["speed"]["value"] == 0.0

    @given(cell_st)
    def test_corridor_property(raw):
        cells = _cells(raw)
        good = (cells["idx"] >= 0).all(1) & (cells["idx"] < np.array(G.shape)).all(1)
        pts, cf = O.corridor_from_cells(cells["idx"][good], cells["cls"][good], np.nan_to_num(cells["conf"][good]).clip(0, 1), G)
        assert 0.0 <= cf <= 1.0
        if len(pts):
            assert cf >= O.CORRIDOR_MIN_CONF and len(pts) >= 4 and len(pts) <= O.CORRIDOR_CAP and np.array_equal(pts[0], pts[-1])
            assert O.polygon_is_simple(pts.astype(np.float64))
            assert (pts[:, 0] >= G.x_min).all() and (pts[:, 0] <= G.x_max).all() and (pts[:, 1] >= G.y_min).all() and (pts[:, 1] <= G.y_max).all()

    @given(st.lists(st.tuples(st.floats(-15, 15), st.floats(-5, 60), st.floats(-math.pi, math.pi), st.floats(1, 8), st.floats(1, 3), st.integers(0, 1000)),
                    max_size=20), st.sampled_from(sorted(CAMS)))
    def test_boxes_2d_match_3d(objs, cid):
        ob = {"xyz": np.array([[o[0], o[1], 0.75] for o in objs], np.float32).reshape(-1, 3), "lwh": np.array([[o[3], o[4], 1.5] for o in objs], np.float32).reshape(-1, 3),
              "yaw_sin": np.sin([o[2] for o in objs]).astype(np.float32), "yaw_cos": np.cos([o[2] for o in objs]).astype(np.float32),
              "track_id": np.arange(len(objs), dtype=np.int32) * 7 + 1, "cls": ["vehicle"] * len(objs)}  # unique ids
        cam = CAMS[cid]
        bx = O.boxes_2d(ob, cam)
        xy = bx["xyxy"]
        assert ((xy >= 0) & (xy <= 1)).all() and ((xy[:, 2] > xy[:, 0]) & (xy[:, 3] > xy[:, 1])).all()
        assert len(bx["track_id"]) == len(bx["cls"]) == len(xy) <= O.BOX2D_CAP
        for k, tid in enumerate(bx["track_id"].tolist()):
            i = (tid - 1) // 7
            uv, z, ok = cam.project(np.array([[objs[i][0], objs[i][1], 0.75]]))
            if ok[0]:
                u, v = uv[0, 0] / cam.width, uv[0, 1] / cam.height
                assert xy[k, 0] - 1e-4 <= u <= xy[k, 2] + 1e-4 and xy[k, 1] - 1e-4 <= v <= xy[k, 3] + 1e-4


# ------------------------------------------------------------------ O2 strict
def _paths(o, prefix=()):
    for k, v in o.items():
        yield prefix + (k,)
        if isinstance(v, dict) and k not in ("latency_ms", "counts"):
            yield from _paths(v, prefix + (k,))


def _get(o, path):
    for k in path[:-1]:
        o = o[k]
    return o


CORRUPTIONS = [
    ("missing", None), ("unknown", None), ("dtype", ("occ", "prob")), ("dtype", ("occ", "idx")), ("shape", ("occ", "xyz")), ("nan", ("occ", "prob")),
    ("inf", ("objects", "xyz")), ("range", ("occ", "prob")), ("range", ("objects", "conf")), ("class", ("occ", "cls")), ("lane_type", None),
    ("count", None), ("one_point", None), ("conf_len", None), ("vram", None), ("corridor_open", None), ("sincos", None), ("cam", None),
    ("speed_neg", None), ("source", None), ("version", None), ("box_range", None), ("motion_frames", None),
    ("v1_record", None), ("v1_keys_in_v2", None), ("obs_len", None), ("obs_dtype", None), ("lane_obs_len", None), ("lane_obs_missing", None),
]


def corrupt(o, kind, target, pick=0):
    o = copy.deepcopy(o)
    if kind in ("missing", "unknown"):
        paths = [p for p in _paths(o)]
        p = paths[pick % len(paths)]
        parent = _get(o, p)
        if kind == "missing":
            if len(p) == 2 and p[0] == "cams":  # a whole camera may legitimately be absent; drop a required field instead
                o["occ"].pop("n_full")
                return o
            parent.pop(p[-1])
        else:
            tgt = parent[p[-1]] if isinstance(parent[p[-1]], dict) and p[-1] not in ("latency_ms", "counts", "cams") else parent
            tgt["zz_new_field"] = 1
        return o
    if kind == "dtype":
        d = _get(o, target)
        d[target[-1]] = d[target[-1]].astype(np.float64 if d[target[-1]].dtype.kind == "f" else np.int64)
    elif kind == "shape":
        d = _get(o, target)
        d[target[-1]] = d[target[-1]][:, :2]
    elif kind in ("nan", "inf"):
        d = _get(o, target)
        a = d[target[-1]].copy()
        a.reshape(-1)[0] = np.nan if kind == "nan" else np.inf
        d[target[-1]] = a
    elif kind == "range":
        d = _get(o, target)
        a = d[target[-1]].copy()
        a.reshape(-1)[0] = 1.5
        d[target[-1]] = a
    elif kind == "class":
        o["occ"]["cls"] = o["occ"]["cls"].copy()
        o["occ"]["cls"][0] = C.N_CLASSES
    elif kind == "lane_type":
        o["lanes"]["type"][0] = "purple_solid"
    elif kind == "count":
        o["occ"]["n_stored"] += 1
    elif kind == "one_point":
        o["lanes"]["pts"][0] = o["lanes"]["pts"][0][:1]
        o["lanes"]["conf"][0] = o["lanes"]["conf"][0][:1]
    elif kind == "conf_len":
        o["lanes"]["conf"][0] = o["lanes"]["conf"][0][:-1]
    elif kind == "vram":
        o["debug"]["peak_vram_mb"] = o["debug"]["cpu_mem_est_mb"]  # aliasing the CPU estimate into the VRAM field
    elif kind == "corridor_open":
        o["corridor"]["pts"] = o["corridor"]["pts"][:-1]
    elif kind == "sincos":
        o["objects"]["yaw_sin"] = o["objects"]["yaw_sin"] * 0.5
    elif kind == "cam":
        o["cams"]["front_left"] = o["cams"]["main"]
    elif kind == "speed_neg":
        o["ego"]["speed"]["value"] = -1.0
    elif kind == "source":
        o["ego"]["speed"]["source"] = "gt"
    elif kind == "version":
        o["version"] = 3
    elif kind == "v1_record":  # a genuine v1 record (no observed fields, version 1)
        o["version"] = 1
        o["occ"].pop("observed")
        o["lanes"].pop("observed")
    elif kind == "v1_keys_in_v2":  # v2 fields under a v1 label
        o["version"] = 1
    elif kind == "obs_len":
        o["occ"]["observed"] = o["occ"]["observed"][:-1]
    elif kind == "obs_dtype":
        o["occ"]["observed"] = o["occ"]["observed"].astype(np.uint8)
    elif kind == "lane_obs_len":
        o["lanes"]["observed"][0] = o["lanes"]["observed"][0][:-1]
    elif kind == "lane_obs_missing":  # v1-shaped lanes inside a v2 record
        o["lanes"].pop("observed")
    elif kind == "box_range":
        o["cams"]["main"]["boxes"]["xyxy"] = o["cams"]["main"]["boxes"]["xyxy"] + 0.9
    elif kind == "motion_frames":
        o["objects"]["motion"][0] = "parked"
        o["objects"]["motion_frames"] = np.zeros_like(o["objects"]["motion_frames"])
    return o


def _base():
    return make_valid(t=1.0, objects=[{"cls": "vehicle", "x": 0.0, "y": 15.0, "yaw": math.pi / 2, "conf": 0.9, "src": "boxes:traffic"}])


def test_rejects_every_corruption():
    base = _base()
    assert O.validate_occ_out(base) == []
    for kind, target in CORRUPTIONS:
        bad = corrupt(base, kind, target)
        errs = O.validate_occ_out(bad)
        assert errs, f"{kind} {target} accepted"
        try:
            O.check_occ_out(bad)
        except O.OccOutError:
            pass
        else:
            raise AssertionError(kind)
    # every single required field, removed one at a time; every level gets an unknown key
    n = len(list(_paths(base)))
    for i in range(n):
        assert O.validate_occ_out(corrupt(base, "missing", None, i)), list(_paths(base))[i]
        assert O.validate_occ_out(corrupt(base, "unknown", None, i)), ("unknown", list(_paths(base))[i])


if HAVE_HYP:

    @given(st.sampled_from(CORRUPTIONS), st.integers(0, 10_000))
    def test_rejects_property(c, pick):
        bad = corrupt(_CACHE_BASE(), c[0], c[1], pick)
        assert O.validate_occ_out(bad), c


_BASE: list = []


def _CACHE_BASE():
    if not _BASE:
        _BASE.append(_base())
    return _BASE[0]


def test_ignore_path_mutant_killed():
    """A mutant validator that ignores unknown keys (the 'ignore' path) is caught by the strict tests."""
    src = (ROOT / "python" / "occ" / "occ_out.py").read_text(encoding="utf-8")
    needle = "        if extra:\n            self.errs.append"
    assert needle in src
    mod = types.ModuleType("occ_out_mutant")
    mod.__dict__["__name__"] = "occ_out_mutant"
    exec(compile(src.replace(needle, "        if False:\n            self.errs.append"), "occ_out_mutant", "exec"), mod.__dict__)  # noqa: S102
    base = _base()
    bad = corrupt(base, "unknown", None, 0)
    assert O.validate_occ_out(bad), "real validator must reject"
    assert mod.validate_occ_out(bad) == [], "mutant should accept (otherwise this check proves nothing)"
    try:
        saved = O.validate_occ_out
        O.validate_occ_out = mod.validate_occ_out
        try:
            test_rejects_every_corruption()
        except AssertionError:
            killed = True
        else:
            killed = False
    finally:
        O.validate_occ_out = saved
    assert killed, "ignore-path mutant survived the strict schema tests"


# ------------------------------------------------------------------ O4 size
def test_size_flat_under_64x_grid_growth():
    cells = road_scene()
    small = Downstream(G, CAMS)
    big_grid = VoxelGrid(x_min=-160.0, x_max=160.0, y_min=-80.0, y_max=560.0)
    assert big_grid.n_dense == 64 * G.n_dense
    big = Downstream(big_grid, CAMS)
    # the same physical cells, re-indexed into the big grid
    xyz = G.to_center(cells["idx"])
    bidx, ok = big_grid.to_index(xyz)
    bcells = dict(cells, idx=bidx.astype(np.int32))
    o1 = small(heads_for(cells, t=0.0))["occ_out"]
    o2 = big(heads_for(bcells, t=0.0))["occ_out"]
    b1, b2 = len(O.to_bytes(o1)), len(O.to_bytes(o2))

    def int_numel(o):
        tot = 0
        for v in _walk_arrays(o):
            if v.dtype.kind in "iu":
                tot += v.size
        return tot

    print(f"O4 occ_out bytes: {b1} (1x grid, {G.n_dense} cells) vs {b2} (64x grid, {big_grid.n_dense} cells); int index elements {int_numel(o1)} vs {int_numel(o2)}")
    assert abs(b2 - b1) <= 0.02 * b1, (b1, b2)
    assert int_numel(o1) == int_numel(o2)
    # worst case: every array at its cap, independent of the grid
    cap = (O.OCC_CAP * (12 + 12 + 4 + 1 + 8) + O.LANE_CAP * O.LANE_PTS_CAP * 12 + O.OBJ_CAP * 60 + O.MEM_CAP * 16 + 8 * (O.PANEL_CAP * 17 + O.BOX2D_CAP * 24)) * 4 // 3 + 20000
    assert b1 < cap and b2 < cap


def _walk_arrays(o):
    if isinstance(o, np.ndarray):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _walk_arrays(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk_arrays(v)


# ------------------------------------------------------------------ O5 ego
def test_speed_from_frame_stamps_not_constant():
    ds = Downstream(G, CAMS)
    stamps = [0.0, 0.07, 0.19, 0.24, 0.5]
    prev = None
    for t in stamps:
        o = ds(heads_for(road_scene(), t=t, vo=(0.02, 0.9, 0.01)))["occ_out"]
        if prev is not None:
            dt = t - prev
            assert abs(o["ego"]["dt"] - dt) < 1e-12
            assert abs(o["ego"]["speed"]["value"] - math.hypot(np.float32(0.02), np.float32(0.9)) / dt) < 1e-4
            assert abs(o["ego"]["yaw_rate"]["value"] - float(np.float32(0.01)) / dt) < 1e-4  # left-positive rad/s
            assert o["ego"]["speed"]["source"] == "vo"
        else:
            assert not o["ego"]["speed"]["valid"]
        prev = t


def test_vo_delta_only_from_vo_head():
    """Flag off: motion_used is ignored for the VO fields; flag on: speed source is 'vehicle', VO delta still the head's."""
    ds = Downstream(G, CAMS)
    ds(heads_for(road_scene(), t=0.0, vo=None, motion_used=(0.0, 3.0, 0.0)))
    o = ds(heads_for(road_scene(), t=0.1, vo=None, motion_used=(0.0, 3.0, 0.0)))["occ_out"]
    assert not o["ego"]["vo_valid"] and not o["ego"]["speed"]["valid"] and np.array_equal(o["ego"]["vo_delta"], np.zeros(3, np.float32))
    ds2 = Downstream(G, CAMS)
    ds2(heads_for(road_scene(), t=0.0, vo=(0.0, 0.5, 0.0), motion_used=(0.0, 2.0, 0.0), ego_from_vehicle=True))
    o2 = ds2(heads_for(road_scene(), t=0.1, vo=(0.0, 0.5, 0.0), motion_used=(0.0, 2.0, 0.0), ego_from_vehicle=True))["occ_out"]
    assert o2["ego"]["speed"]["source"] == "vehicle" and abs(o2["ego"]["speed"]["value"] - 20.0) < 1e-6
    assert np.array_equal(o2["ego"]["vo_delta"], np.array([0.0, 0.5, 0.0], np.float32))


# ------------------------------------------------------------------ O7 vram
def test_vram_never_aliases_cpu_estimate():
    o = make_valid(t=1.0)
    assert o["debug"]["peak_vram_mb"] == O.VRAM_UNPROVEN and o["debug"]["cpu_mem_est_mb"] == 744.0
    assert o["debug"]["cpu_mem_est_label"] == O.CPU_EST_LABEL
    for bad in (744.0, 0, "744", o["debug"]["cpu_mem_est_mb"]):
        b = copy.deepcopy(o)
        b["debug"]["peak_vram_mb"] = bad
        assert O.validate_occ_out(b), bad
    b = copy.deepcopy(o)
    b["debug"]["peak_vram_mb"] = None
    assert O.validate_occ_out(b) == []


# ------------------------------------------------------------------ O8 viz
def test_v1_rejected_and_viz_bridge_drops_v2_fields():
    """occ_out v2 only: a v1 record, or v1 keys mixed into v2, is rejected; the bridge to the (GVD-owned,
    unchanged) v1 occ_to_viz drops exactly the v2-only fields and refuses a non-v2 record."""
    o = make_valid(t=1.0)
    assert o["version"] == 2 and O.validate_occ_out(o) == []
    for kind in ("v1_record", "v1_keys_in_v2", "lane_obs_missing"):
        assert O.validate_occ_out(corrupt(o, kind, None)), kind
    out = O.occ_out_to_outputs(o)
    assert "observed" not in out["cells"] and set(out["cells"]) == {"idx", "cls", "conf"}
    assert V.validate_v1(V.occ_to_viz(out)) == []
    b = copy.deepcopy(o)
    b["version"] = 1
    try:
        O.occ_out_to_outputs(b)
    except O.OccOutError:
        pass
    else:
        raise AssertionError("bridge accepted a v1 record")
    assert {"occ.observed", "lanes.observed (per point)"} <= set(O.V2_DELTA)


def test_viz_matches_occ_out_and_ids_stable():
    ds = Downstream(G, CAMS)
    car = [{"cls": "vehicle", "x": 2.0, "y": 20.0, "yaw": math.pi / 2, "conf": 0.9, "src": "boxes:traffic"}]
    ids = []
    for k in range(5):
        r = ds(heads_for(road_scene(), t=0.1 * k, vo=(0.0, 0.0, 0.0), objects=car))
        o, msg = r["occ_out"], r["viz"]
        assert V.validate_v1(msg) == []
        # lanes / stop lines live in occ_out.lanes; the committed v1 occ_to_viz (GVD-owned, unchanged here) does not draw them
        assert sum(t.startswith(("white", "yellow")) for t in o["lanes"]["type"]) >= 2
        assert len(msg["edges"]) == min(sum(t in ("curb", "road_edge") for t in o["lanes"]["type"]), V.MAX_EDGES)
        assert sorted(t["id"] for t in msg["tracks"] if "id" in t) == sorted(i for i in o["objects"]["track_id"].tolist() if i >= 0)
        ids.append(o["objects"]["track_id"].tolist())
    assert all(i == ids[0] for i in ids) and ids[0][0] >= 0, ids


# ------------------------------------------------------------------ O11 occluded eval rules
def test_occluded_eval_rules():
    """Occluded GT = full labels minus runtime-visible labels; a score that does not beat the best baseline is 0;
    no occluded number without a disjoint map/region split."""
    import tempfile
    from python.occ import offline_eval as E

    full = {"occ_idx": np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], np.int32), "free_idx": np.array([[4, 4, 4], [5, 5, 5]], np.int32)}
    vis = {"occ_idx": np.array([[1, 1, 1]], np.int32), "free_idx": np.array([[4, 4, 4]], np.int32)}
    occ, free = E.occluded_sets(full, vis, G)
    assert occ.size == 2 and free.size == 1
    pred = E._flat(np.array([[2, 2, 2], [5, 5, 5]], np.int32), G)
    r = E.occluded_iou(pred, occ, free)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1) and abs(r["iou"] - 1 / 3) < 1e-9
    t = E.score_table(0.4, {"prior": 0.5, "nomem": 0.1}, {"train": "a", "eval": "b"}, "SYNTHETIC")
    assert t["occluded_score"] == 0.0 and not t["beats_baseline"]
    t = E.score_table(0.6, {"prior": 0.5, "nomem": 0.1}, {"train": "a", "eval": "b"}, "SYNTHETIC")
    assert t["occluded_score"] == 0.6 and t["beats_baseline"]
    for bad in (None, {"train": {"m1": "x"}, "eval": {"m1": "y"}}, {"train": {"m1": "x"}, "eval": {}}):
        try:
            if bad is None:
                E.load_split(None)
            else:
                with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                    json.dump(bad, f)
                E.load_split(f.name)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"split accepted: {bad}")


# ------------------------------------------------------------------ O9 imports
def test_builder_import_trace_clean():
    for mod in ("python.occ.occ_out", "python.occ.downstream"):
        code = f"import sys; sys.path.insert(0, {str(ROOT)!r}); import {mod}; print('\\n'.join(sorted(sys.modules)))"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, cwd=ROOT)
        assert out.returncode == 0, out.stderr
        mods = set(out.stdout.split())
        bad = sorted(m for m in mods if m.split(".")[0] in ("torch", "cv2", "PIL", "beamngpy")
                     or m.startswith(("python.collect", "python.debug", "python.train", "python.data", "python.occ.labels", "python.occ.model",
                                      "python.occ.runtime", "python.occ.synthetic", "python.occ.shards", "python.occ.train_occ")))
        assert not bad, (mod, bad)
    src = (ROOT / "python" / "occ" / "occ_out.py").read_text(encoding="utf-8").split('"""', 2)[2].lower()  # code, not the docstring
    for word in ("scenetree", "gt_pose", "state.jsonl", "world_pos"):
        assert word not in src, word


PLAIN_TESTS = ["test_builder_scene_valid", "test_low_confidence_corridor_empty", "test_rejects_every_corruption", "test_ignore_path_mutant_killed",
               "test_size_flat_under_64x_grid_growth", "test_speed_from_frame_stamps_not_constant", "test_vo_delta_only_from_vo_head",
               "test_vram_never_aliases_cpu_estimate", "test_v1_rejected_and_viz_bridge_drops_v2_fields", "test_occluded_eval_rules", "test_viz_matches_occ_out_and_ids_stable",
               "test_builder_import_trace_clean"]
PROPERTY_TESTS = ["test_builder_properties", "test_corridor_property", "test_boxes_2d_match_3d", "test_rejects_property"]


def main() -> int:
    from gvd_test_env import skip_check

    for name in PLAIN_TESTS:
        globals()[name]()
        print(f"{name} OK", flush=True)
    for name in PROPERTY_TESTS:
        if not HAVE_HYP:
            skip_check(name, "hypothesis not installed (pip install -r requirements-dev.txt)")
            continue
        globals()[name]()
        print(f"{name} OK (hypothesis)", flush=True)
    print("occ_out v2 invariants O1-O11 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
