#!/usr/bin/env python3
"""Sparse occupancy: invariants, attacked with property-based tests (hypothesis).

Breaker method (adversarial): the invariants are written down first, then
each one is attacked with generated inputs and named edge cases. Minimal
counterexamples hypothesis found before the fixes are recorded in
``docs/specs/occ_sparse.md`` (section "Adversarial evidence").

INVARIANTS
 I1  grid: in-bounds indices are within ``shape``; non-finite or out-of-range
     points are never in-bounds; centre(index) maps back to index;
     flat/unflat is a bijection; fine voxels of a pillar are unique, in
     bounds and map back to that pillar.
 I2  camera: project(backproject(pixel, d)) == pixel and z == d.
 I3  labels: occ_idx int32, unique, in bounds; occ_cls is a known class;
     free_idx unique, in bounds, disjoint from occ and from the box ignore
     band; ROAD only when a sem id names a drivable class; VEHICLE only when
     a box was given; never raises on NaN / zero / 65535 depth, empty
     frame lists, NaN or far-away boxes, many boxes.
 I4  depth_to_m: every output is NaN or a finite metre value > 0.
 I5  sampler: every index valid; every batch full; batch mode keeps each
     batch's recoveries in [ceil(.2B), floor(.3B)]; window mode keeps every
     aligned 10-window at 2 or 3; the epoch share is in [0.2, 0.3];
     deterministic per (seed, epoch); epoch length >= dataset; a dataset with
     no recoveries or only recoveries raises.
 I6  shards: write -> read is lossless (arrays bit-equal incl. NaN, RGB
     equal, meta equal, key order kept, recover flag kept) across shard
     rolls; the reader opens only ``.tar`` files.
 I7  tracker: motion in MOTION_STATES; a state other than ``unknown`` needs
     >= 2 frames; the first frame after any reset (start, backwards or
     repeated stamp, gap, NaN kinematics) is all ``unknown``; <= 64 rows;
     ids unique; a world-fixed car is ``parked`` whatever the ego does; a car
     driving the ego's way is ``same_way``/``alongside``; opposite is
     ``oncoming``; perpendicular is ``crossing``.
 I8  polylines: edges of a straight synthetic road are within one voxel of
     the true edges; every point finite and inside the grid; <= 8 edges;
     each has >= MIN_ROWS points, a ``type`` in {road_edge, curb} and a src.
 I9  viz: ``occ_to_viz`` output validates against v1 (also after a JSON
     round trip and after stripping to v1 keys), stays <= 60000 bytes, keeps
     v1 caps, ignores unknown keys; every track/edge carries a src; motion
     is a known state; outputs with no label source are listed UNLABELED;
     red only for curb cells and edge/curb polylines; no lane paint is
     invented (lanes empty).

    py -3 scripts/test_occ_sparse_props.py
"""
from __future__ import annotations

import builtins
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from python.occ import classes as C  # noqa: E402
from python.occ.camera import CameraModel  # noqa: E402
from python.occ.grid import DEFAULT_GRID, VoxelGrid  # noqa: E402
from python.occ.labels import CamFrame, depth_to_m, road_ids, voxel_labels  # noqa: E402
from python.occ.polylines import MIN_ROWS, edges_from_cells  # noqa: E402
from python.occ.sampler import WINDOW, RecoveryMixError, RecoveryMixSampler, batch_bounds  # noqa: E402
from python.occ.shards import ShardDataset, ShardError, ShardWriter, is_recovery_row  # noqa: E402
from python.occ.tracks import MAX_TRACKS, MOTION_STATES, MotionTracker  # noqa: E402
from python.occ.synthetic import ground_depth  # noqa: E402
from python.occ.viz_adapter import MAX_DATAGRAM, MESSAGE_KEYS, occ_to_viz, strip_to_v1, validate_v1  # noqa: E402

try:
    from hypothesis import HealthCheck, assume, given, settings
    from hypothesis import strategies as st
    from hypothesis.extra import numpy as hnp

    HAVE_HYP = True
    settings.register_profile("ci", max_examples=60, deadline=None, derandomize=True, suppress_health_check=list(HealthCheck))
    settings.register_profile("fuzz", max_examples=1500, deadline=None, suppress_health_check=list(HealthCheck))
    settings.load_profile(os.environ.get("GVD_HYPOTHESIS_PROFILE", "ci"))
except ImportError:  # dev-only dependency (requirements-dev.txt)
    HAVE_HYP = False

G = DEFAULT_GRID
import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in cast")
SEM = {"kind": "gvd_sem_classes", "train_only": True, "classes": {"0": "unknown", "1": "ASPHALT", "2": "Grass", "3": "PROP"}}
MAIN = CameraModel("main", (0.0, 1.2, 1.26), 0.0, -6.0, 34.0, 640, 480)
WIDE = CameraModel("wide", (-0.03, 1.2, 1.24), 0.0, -8.0, 88.7, 640, 360)


# ---------------------------------------------------------------- strategies
if HAVE_HYP:
    finite = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)
    anyfloat = st.floats(allow_nan=True, allow_infinity=True, width=64)
    small_cam = st.builds(
        lambda w, h, yaw, pitch, fov: CameraModel("main", (0.0, 1.2, 1.26), yaw, pitch, fov, w, h),
        st.integers(1, 24), st.integers(1, 24), st.floats(-90, 90), st.floats(-30, 10), st.floats(5, 120),
    )

    @st.composite
    def cam_frames(draw):
        cams = draw(st.lists(small_cam, min_size=0, max_size=2))
        frames = []
        for cam in cams:
            kind = draw(st.sampled_from(["u16", "f32"]))
            if kind == "u16":
                d = draw(hnp.arrays(np.uint16, (cam.height, cam.width), elements=st.sampled_from([0, 1, 50, 120, 400, 2500, 9000, 65534, 65535])))
            else:
                d = draw(hnp.arrays(np.float32, (cam.height, cam.width), elements=st.one_of(st.floats(-5, 400, width=32), st.just(float("nan")), st.just(float("inf")))))
            sem = draw(st.one_of(st.none(), hnp.arrays(np.uint8, (cam.height, cam.width), elements=st.integers(0, 5))))
            frames.append(CamFrame(cam, d, sem))
        return frames

    box_st = st.fixed_dictionaries({"x": st.one_of(finite, anyfloat), "y": st.one_of(finite, anyfloat), "yaw": st.one_of(st.floats(-7, 7), anyfloat)})


# ---------------------------------------------------------------- I1 grid
def _check_grid_index(pts):
    with np.errstate(all="raise"):
        idx, ok = G.to_index(pts)
    shape = np.array(G.shape)
    assert idx.dtype == np.int32
    assert ((idx[ok] >= 0) & (idx[ok] < shape)).all()
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    assert not ok[~np.isfinite(p).all(axis=1)].any()
    back, ok2 = G.to_index(G.to_center(idx[ok]))
    assert ok2.all() and (back == idx[ok]).all()
    assert (G.unflat(G.flat(idx[ok])) == idx[ok]).all()


if HAVE_HYP:

    @given(hnp.arrays(np.float64, st.tuples(st.integers(0, 40), st.just(3)), elements=st.one_of(st.floats(-100, 100), anyfloat)))
    def test_grid_index(pts):
        _check_grid_index(pts)

    @given(st.lists(st.tuples(st.integers(-2, 25), st.integers(-2, 45)), max_size=12))
    def test_fine_in_coarse(cells):
        c = np.array(cells, dtype=np.int64).reshape(-1, 2)
        cx, cy = G.coarse_shape
        valid = c[(c[:, 0] >= 0) & (c[:, 0] < cx) & (c[:, 1] >= 0) & (c[:, 1] < cy)]
        f = G.fine_in_coarse(valid)
        assert np.unique(G.flat(f)).size == f.shape[0]
        assert ((f >= 0) & (f < np.array(G.shape))).all()
        if f.size:
            back = {tuple(r) for r in G.coarse_of(f).tolist()}
            assert back <= {tuple(r) for r in valid.tolist()}

    # ------------------------------------------------------------ I2 camera
    @given(small_cam, st.floats(0.2, 250.0), st.data())
    def test_camera_roundtrip(cam, d, data):
        u = data.draw(st.integers(0, cam.width - 1))
        v = data.draw(st.integers(0, cam.height - 1))
        depth = np.full((cam.height, cam.width), np.nan)
        depth[v, u] = d
        ego, valid = cam.backproject(depth)
        assert valid.sum() == 1
        uv, z, ok = cam.project(ego[valid])
        assert ok[0]
        assert abs(uv[0, 0] - (u + 0.5)) < 1e-6 * max(1.0, d) + 1e-6 and abs(uv[0, 1] - (v + 0.5)) < 1e-6 * max(1.0, d) + 1e-6
        assert abs(z[0] - d) < 1e-9 * max(1.0, d) + 1e-9

    # ------------------------------------------------------------ I3/I4 labels
    @given(cam_frames(), st.lists(box_st, max_size=4), st.integers(0, 50))
    def test_labels_invariants(frames, boxes, max_free):
        lab = voxel_labels(frames, SEM, boxes, max_free=max_free, seed=1)
        _check_labels(lab, frames, boxes, max_free)

    @given(st.one_of(hnp.arrays(np.uint16, st.tuples(st.integers(0, 6), st.integers(0, 6))), hnp.arrays(np.float32, st.tuples(st.integers(0, 6), st.integers(0, 6)), elements=st.floats(width=32))))
    def test_depth_to_m(arr):
        m = depth_to_m(arr)
        ok = ~np.isnan(m)
        assert np.isfinite(m[ok]).all() and (m[ok] > 0).all()
        if arr.dtype == np.uint16:
            assert np.isnan(m[(arr == 0) | (arr == 65535)]).all()


def _check_labels(lab, frames, boxes, max_free):
    occ, cls, free, ign = lab["occ_idx"], lab["occ_cls"], lab["free_idx"], lab["ignore_idx"]
    for a in (occ, free, ign):
        assert a.dtype == np.int32 and a.ndim == 2 and a.shape[1] == 3
        assert ((a >= 0) & (a < np.array(G.shape))).all()
        assert np.unique(G.flat(a)).size == a.shape[0]
    assert cls.shape == (occ.shape[0],) and set(np.unique(cls).tolist()) <= set(range(C.N_CLASSES))
    fo, ff, fi = set(G.flat(occ).tolist()), set(G.flat(free).tolist()), set(G.flat(ign).tolist())
    assert not (fo & ff) and not (ff & fi) and not (fo & fi)
    assert len(ff) <= max_free
    if not any(f.sem is not None and np.isin(f.sem, list(road_ids(SEM))).any() for f in frames):
        assert not (cls == C.ROAD).any() and not (cls == C.CURB).any()
    if not boxes:
        assert not (cls == C.VEHICLE).any()


def test_labels_edge_cases():
    """Named edge cases: no frames, all-invalid depth, NaN everywhere, box far outside, many boxes."""
    _check_labels(voxel_labels([], SEM, []), [], [], 4096)
    cam = MAIN.scaled(16, 12)
    for d in (np.zeros((12, 16), np.uint16), np.full((12, 16), 65535, np.uint16), np.full((12, 16), np.nan, np.float32)):
        lab = voxel_labels([CamFrame(cam, d, None)], SEM, [])
        assert lab["occ_idx"].shape[0] == 0 and lab["free_idx"].shape[0] == 0
    boxes = [{"x": 1e9, "y": -1e9, "yaw": 0.0}, {"x": float("nan"), "y": 5.0, "yaw": 0.0}]
    boxes += [{"x": float(x), "y": float(y), "yaw": 0.3} for x in range(-16, 17, 4) for y in range(0, 64, 8)]
    lab = voxel_labels([CamFrame(cam, np.full((12, 16), 1000, np.uint16), None)], SEM, boxes)
    _check_labels(lab, [CamFrame(cam, np.full((12, 16), 1000, np.uint16), None)], boxes, 4096)
    assert (lab["occ_cls"] == C.VEHICLE).sum() > 0
    try:
        voxel_labels([CamFrame(cam, np.zeros((5, 5), np.uint16), None)], SEM, [])
    except ValueError:
        pass
    else:
        raise AssertionError("wrong-shape depth must raise")


def test_labels_synthetic_scene():
    """Flat ASPHALT ground + one box: road cells near z=0, vehicle cells at the box, free cells above ground."""
    cam = MAIN.scaled(160, 120)
    d = ground_depth(cam, box={"x": 0.0, "y0": 15.0, "w": 1.8, "h": 1.4})
    ground = ground_depth(cam)
    on_box = np.isfinite(d) & ~(np.isclose(d, ground, rtol=1e-6) | ~np.isfinite(ground))
    sem = np.where(on_box, 3, np.where(np.isfinite(d), 1, 0)).astype(np.uint8)  # box face is PROP, ground ASPHALT
    lab = voxel_labels([CamFrame(cam, d.astype(np.float32), sem)], SEM, [{"x": 0.0, "y": 17.0, "yaw": math.pi / 2}], max_free=2000)
    occ, cls = lab["occ_idx"], lab["occ_cls"]
    zc = G.to_center(occ)[:, 2]
    assert (cls == C.ROAD).sum() > 50
    assert (np.abs(zc[cls == C.ROAD]) <= G.voxel).all()
    veh = G.to_center(occ[cls == C.VEHICLE])
    assert len(veh) > 0 and (np.abs(veh[:, 1] - 17.0) <= 2.0).all()
    fz = G.to_center(lab["free_idx"])[:, 2]
    assert lab["free_idx"].shape[0] > 100 and (fz > -G.voxel).all()


# ---------------------------------------------------------------- I5 sampler
def _check_sampler(flags, batch, seed, epoch):
    flags = list(flags)
    s = RecoveryMixSampler(flags, batch, seed=seed)
    bs = s.batches(epoch)
    assert bs == s.batches(epoch)
    assert len(bs) == len(s)
    flat = [i for b in bs for i in b]
    assert len(flat) >= len(flags) and all(len(b) == batch for b in bs)
    assert all(0 <= i < len(flags) for i in flat)
    rec = [flags[i] for i in flat]
    share = sum(rec) / len(rec)
    assert 0.2 - 1e-9 <= share <= 0.3 + 1e-9, share
    bounds = batch_bounds(batch)
    if bounds is not None:
        for b in bs:
            k = sum(flags[i] for i in b)
            assert bounds[0] <= k <= bounds[1], (batch, k)
    else:
        for w in range(0, len(rec), WINDOW):
            assert sum(rec[w : w + WINDOW]) in (2, 3)
    return s


if HAVE_HYP:

    @given(st.lists(st.booleans(), min_size=2, max_size=300), st.integers(1, 64), st.integers(0, 2**31), st.integers(0, 5))
    def test_sampler_mix(flags, batch, seed, epoch):
        assume(any(flags) and not all(flags))
        _check_sampler(flags, batch, seed, epoch)

    @given(st.integers(1, 200), st.integers(1, 64), st.booleans())
    def test_sampler_pure_raises(n, batch, value):
        try:
            RecoveryMixSampler([value] * n, batch)
        except RecoveryMixError:
            return
        raise AssertionError("pure dataset must raise")


def test_sampler_edge_cases():
    for bad in ([], [True, False]):
        for b in (0, -1, 2.5, True):
            try:
                RecoveryMixSampler(bad, b)
            except RecoveryMixError:
                continue
            raise AssertionError(f"bad input accepted: {bad} {b}")
    _check_sampler([True] + [False] * 999, 8, 0, 0)       # 0.1% recoveries -> still 25%
    _check_sampler([False] + [True] * 999, 8, 0, 0)       # 99.9% recoveries -> still 25%
    for b in (1, 2, 3, 6):
        assert _check_sampler([True, False, False, False], b, 3, 1).mode == "window"
    # every recovery shows up when the quota covers the pool
    s = RecoveryMixSampler([True] * 5 + [False] * 95, 10, seed=4)
    assert set(range(5)) <= set(i for b in s.batches(0) for i in b)


# ---------------------------------------------------------------- I6 shards
def _roundtrip(samples, max_bytes, tmp):
    opened: list[str] = []
    with ShardWriter(tmp, "t", max_bytes=max_bytes, max_count=3) as w:
        for key, meta, rgb, arrays in samples:
            w.write(key, meta=meta, rgb=rgb, arrays=arrays)
    real_open = builtins.open

    def spy(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    builtins.open = spy
    try:
        ds = ShardDataset(sorted(Path(tmp).glob("*.tar")))
        got = list(ds)
    finally:
        builtins.open = real_open
    assert all(p.endswith(".tar") for p in opened), opened
    assert not list(Path(tmp).glob("*.part"))
    assert ds.keys() == [s[0] for s in samples]
    assert ds.recovery_flags() == [bool(s[1].get("recover") is True) for s in samples]
    for (key, meta, rgb, arrays), g in zip(samples, got):
        assert g["key"] == key and g["meta"] == meta
        assert set(g["rgb"]) == set(rgb) and all(np.array_equal(g["rgb"][c], rgb[c]) for c in rgb)
        assert set(g["arrays"]) == set(arrays)
        for n, a in arrays.items():
            b = g["arrays"][n]
            assert b.dtype == a.dtype and b.shape == a.shape
            assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    return ds


if HAVE_HYP:
    arr_st = st.one_of(
        hnp.arrays(st.sampled_from([np.int8, np.int32, np.uint16]), hnp.array_shapes(min_dims=1, max_dims=2, min_side=0, max_side=6)),
        hnp.arrays(np.float32, hnp.array_shapes(min_dims=1, max_dims=2, min_side=0, max_side=6), elements=st.floats(width=32)),
    )
    sample_st = st.tuples(
        st.fixed_dictionaries({"recover": st.booleans(), "t": st.floats(-1e6, 1e6), "i": st.integers(0, 10**7)}),
        st.dictionaries(st.sampled_from(["main", "wide"]), hnp.arrays(np.uint8, st.tuples(st.integers(1, 9), st.integers(1, 9), st.just(3))), min_size=1),
        st.dictionaries(st.sampled_from(["occ_idx", "occ_cls", "free_idx", "x_1"]), arr_st, max_size=3),
    )

    @given(st.lists(sample_st, min_size=1, max_size=7), st.integers(2048, 200000))
    def test_shard_roundtrip(samples, max_bytes):
        keyed = [(f"k{i:04d}", *s) for i, s in enumerate(samples)]
        with tempfile.TemporaryDirectory() as tmp:
            try:
                _roundtrip(keyed, max_bytes, tmp)
            except ShardError as exc:
                assert "over the shard cap" in str(exc)


def test_shard_no_stale_mix():
    """A second build into the same folder must not leave the first build's extra shards readable as one set."""
    img = np.zeros((2, 2, 3), np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        with ShardWriter(tmp, "r", max_count=1) as w:
            for i in range(4):
                w.write(f"a{i}", meta={"recover": False}, rgb={"main": img}, arrays={})
        try:
            with ShardWriter(tmp, "r", max_count=1) as w:
                w.write("b0", meta={"recover": True}, rgb={"main": img}, arrays={})
        except ShardError:
            return
        keys = ShardDataset(sorted(Path(tmp).glob("*.tar"))).keys()
        assert keys == ["b0"], f"stale shards mixed into a new build: {keys}"


def test_shard_edge_cases():
    img = np.zeros((2, 2, 3), np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        w = ShardWriter(tmp, "e")
        for key in ("", "a.b", "a/b", "../x", "k k"):
            try:
                w.write(key, meta={}, rgb={"main": img}, arrays={})
            except ShardError:
                continue
            raise AssertionError(f"bad key accepted: {key!r}")
        w.write("ok", meta={}, rgb={"main": img}, arrays={})
        for bad in (dict(meta={}, rgb={}, arrays={}), dict(meta={}, rgb={"main": np.zeros((0, 2, 3), np.uint8)}, arrays={}), dict(meta={"t": float("nan")}, rgb={"main": img}, arrays={})):
            try:
                w.write("ok2", **bad)
            except (ShardError, ValueError):
                continue
            raise AssertionError(f"bad sample accepted: {bad}")
        try:
            w.write("ok", meta={}, rgb={"main": img}, arrays={})
        except ShardError:
            pass
        else:
            raise AssertionError("duplicate key accepted")
        w.close()
        (Path(tmp) / "loose.png").write_bytes(b"x")
        try:
            ShardDataset([Path(tmp) / "loose.png"])
        except ShardError:
            pass
        else:
            raise AssertionError("loose png accepted as a shard")
    assert is_recovery_row({"tags": {"recover": True}}) and is_recovery_row({"tags": {"excursion": {"phase": "recover"}}})
    assert not is_recovery_row({"tags": {"excursion": {"phase": "weave"}}}) and not is_recovery_row(None) and not is_recovery_row({"tags": {"recover": 1}})


# ---------------------------------------------------------------- I7 tracker
def _check_frame(rows, fresh):
    assert len(rows) <= MAX_TRACKS
    ids = [r["id"] for r in rows if r["id"] is not None]
    assert len(ids) == len(set(ids))
    for r in rows:
        assert r["motion"] in MOTION_STATES and r["motion_src"] == C.SRC_MOTION
        if r["motion"] != "unknown":
            assert r["motion_frames"] >= 2
        if fresh:
            assert r["motion"] == "unknown"


if HAVE_HYP:
    det_st = st.fixed_dictionaries({"x": st.one_of(st.floats(-30, 30), anyfloat), "y": st.one_of(st.floats(-10, 80), anyfloat), "cls": st.sampled_from(["vehicle", "pedestrian"])})
    frame_st = st.tuples(st.one_of(st.floats(-0.3, 1.0), anyfloat), st.lists(det_st, max_size=80), st.one_of(st.floats(0, 40), anyfloat), st.one_of(st.floats(-1, 1), anyfloat))

    @given(st.lists(frame_st, min_size=1, max_size=25))
    def test_tracker_fuzz(frames):
        tr = MotionTracker()
        t = 0.0
        for dt, dets, v, w in frames:
            t_prev = tr.t
            resets = tr.resets
            t = t + dt if math.isfinite(dt) else dt
            rows = tr.update(t, dets, speed_mps=v, yaw_rate=w)
            fresh = t_prev is None or tr.resets > resets
            _check_frame(rows, fresh)
            t = t if math.isfinite(t) else 0.0

    @given(st.floats(0, 30), st.floats(-0.3, 0.3), st.floats(-15, 15), st.floats(5, 60), st.floats(0.05, 0.1), st.sampled_from(["parked", "same", "oncoming", "crossing"]), st.floats(3, 40))
    def test_tracker_physics(ego_v, ego_w, ox, oy, dt, kind, obj_v):
        """Simulate ego + one object in the world; the state must match the object's true motion."""
        tr = MotionTracker()
        ex, ey, th = 0.0, 0.0, math.pi / 2
        heading = {"parked": None, "same": math.pi / 2, "oncoming": -math.pi / 2, "crossing": 0.0}[kind]
        wx, wy = ox, oy
        last = None
        for k in range(8):
            if k:
                th += ego_w * dt
                ex += ego_v * dt * math.cos(th)
                ey += ego_v * dt * math.sin(th)
                if heading is not None:
                    wx += obj_v * dt * math.cos(heading)
                    wy += obj_v * dt * math.sin(heading)
            dx, dy = wx - ex, wy - ey
            x = dx * math.sin(th) - dy * math.cos(th)
            y = dx * math.cos(th) + dy * math.sin(th)
            rows = tr.update(k * dt, [{"x": x, "y": y, "cls": "vehicle"}], speed_mps=ego_v, yaw_rate=ego_w)
            if k == 0:
                assert rows[0]["motion"] == "unknown"
            if rows:
                last = rows[0]
        assert last is not None and last["motion_frames"] >= 2
        rel = (heading - th) if heading is not None else 0.0
        rel = math.degrees((rel + math.pi) % (2 * math.pi) - math.pi)
        if kind == "parked":
            assert last["motion"] == "parked", last
        elif abs(rel) < 30:
            assert last["motion"] in ("same_way", "alongside"), (last, rel)
        elif abs(rel) > 150:
            assert last["motion"] == "oncoming", (last, rel)
        elif 60 < abs(rel) < 120:
            assert last["motion"] == "crossing", (last, rel)


def test_tracker_edge_cases():
    tr = MotionTracker()
    assert tr.update(0.0, [], speed_mps=10) == []
    car = [{"x": 0.0, "y": 20.0, "cls": "vehicle"}]
    assert tr.update(0.1, car, speed_mps=0.0)[0]["motion"] == "unknown"   # first sighting
    assert tr.update(0.15, car, speed_mps=0.0)[0]["motion"] == "unknown"  # span < MIN_SPAN_S
    assert tr.update(0.3, car, speed_mps=0.0)[0]["motion"] == "parked"
    assert tr.update(0.3, car, speed_mps=0.0)[0]["motion"] == "unknown"   # repeated stamp resets
    assert tr.update(0.1, car, speed_mps=0.0)[0]["motion"] == "unknown"   # clock went back
    tr.update(0.2, car, speed_mps=0.0)
    assert tr.update(5.0, car, speed_mps=0.0)[0]["motion"] == "unknown"   # dropped frames / jump
    assert tr.update(5.1, car, speed_mps=float("nan"))[0]["motion"] == "unknown"
    many = [{"x": float(i % 9), "y": float(i), "cls": "vehicle"} for i in range(200)]
    assert len(tr.update(6.0, many, speed_mps=1.0)) <= MAX_TRACKS
    # a stationary car straight ahead, ego driving at it: parked, not same_way
    tr = MotionTracker()
    rows = []
    for k in range(6):
        rows = tr.update(k * 0.1, [{"x": 0.0, "y": 30.0 - 1.5 * k, "cls": "vehicle"}], speed_mps=15.0)
    assert rows[0]["motion"] == "parked", rows


# ---------------------------------------------------------------- I8 polylines
def _road_cells(width_m: float, centre_x: float, y0: float = 0.0, y1: float = 40.0):
    xs = np.arange(G.shape[0])
    xc = G.x_min + (xs + 0.5) * G.voxel
    ys = np.arange(G.shape[1])
    yc = G.y_min + (ys + 0.5) * G.voxel
    sel_x = xs[np.abs(xc - centre_x) <= width_m / 2]
    sel_y = ys[(yc >= y0) & (yc <= y1)]
    iz = int((0.0 - G.z_min) / G.voxel)
    gx, gy = np.meshgrid(sel_x, sel_y, indexing="ij")
    idx = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, iz)], axis=1)
    return idx, np.full(idx.shape[0], C.ROAD)


def _check_edges(edges):
    assert len(edges) <= 8
    for e in edges:
        assert e["type"] in C.RED_POLYLINE_KINDS and e["src"] in C.KNOWN_SOURCES and e["src"] != C.UNLABELED
        assert len(e["pts"]) >= MIN_ROWS and len(e["pts"]) <= 120
        for x, y in e["pts"]:
            assert math.isfinite(x) and math.isfinite(y)
            assert G.x_min <= x <= G.x_max and G.y_min <= y <= G.y_max


if HAVE_HYP:

    @given(st.floats(3.0, 14.0), st.floats(-4.0, 4.0))
    def test_edges_straight_road(width, cx):
        idx, cls = _road_cells(width, cx)
        assume(idx.shape[0] > 0)
        edges = edges_from_cells(idx, cls)
        _check_edges(edges)
        road = [e for e in edges if e["type"] == "road_edge"]
        assert len(road) == 2
        xs = sorted(np.mean([p[0] for p in e["pts"]]) for e in road)
        assert abs(xs[0] - (cx - width / 2)) <= G.voxel + 1e-6 and abs(xs[1] - (cx + width / 2)) <= G.voxel + 1e-6

    @given(st.one_of(st.lists(st.tuples(st.integers(-5, 170), st.integers(-5, 170), st.integers(-3, 10), st.integers(-1, 6)), max_size=400),
                     st.lists(st.tuples(st.integers(-6, 3), st.integers(-5, 170), st.integers(-3, 10), st.sampled_from([C.ROAD, C.CURB])), min_size=0, max_size=400)))
    def test_edges_random_cells(cells):
        a = np.array(cells, dtype=np.int64).reshape(-1, 4)
        _check_edges(edges_from_cells(a[:, :3], a[:, 3]))

    @st.composite
    def clustered_cells(draw):
        """Cells around a few centres, so components of >= 2 VEHICLE cells actually occur."""
        out = []
        for _ in range(draw(st.integers(0, 6))):
            cx, cy = draw(st.integers(-3, 82)), draw(st.integers(-3, 162))
            for _ in range(draw(st.integers(1, 30))):
                out.append((cx + draw(st.integers(-3, 3)), cy + draw(st.integers(-4, 4)), draw(st.integers(-2, 9)),
                            draw(st.sampled_from([C.VEHICLE, C.VEHICLE, C.ROAD, C.CURB])), draw(st.one_of(st.floats(0, 1), anyfloat))))
        return out

    @given(st.one_of(clustered_cells(), st.lists(st.tuples(st.integers(-3, 82), st.integers(-3, 162), st.integers(-2, 9), st.integers(0, 4), st.one_of(st.floats(0, 1), anyfloat)), max_size=300)))
    def test_objects_from_cells(cells):
        from python.occ.objects import MAX_OBJECTS, objects_from_cells

        a = np.array([c[:4] for c in cells], dtype=np.int64).reshape(-1, 4)
        conf = np.array([c[4] for c in cells], dtype=np.float32)
        objs = objects_from_cells(a[:, :3], a[:, 3], conf)
        assert len(objs) <= MAX_OBJECTS
        if not (a[:, 3] == C.VEHICLE).any():
            assert objs == []
        for o in objs:
            for k in ("x", "y", "yaw", "l", "w", "h", "conf"):
                assert math.isfinite(o[k]), (k, o)
            assert G.x_min <= o["x"] <= G.x_max and G.y_min <= o["y"] <= G.y_max
            assert o["l"] > 0 and o["w"] > 0 and o["h"] > 0, o
            assert 0.0 <= o["conf"] <= 1.0 and 0.0 < o["yaw"] <= math.pi
            assert o["src"] == C.SRC_VEHICLE


def test_curb_line():
    idx, cls = _road_cells(7.0, 0.0)
    iz = int((0.0 - G.z_min) / G.voxel)
    cx = int((3.75 - G.x_min) / G.voxel)
    cy = np.arange(int((0 - G.y_min) / G.voxel), int((30 - G.y_min) / G.voxel))
    curb = np.stack([np.full(cy.size, cx), cy, np.full(cy.size, iz)], axis=1)
    edges = edges_from_cells(np.concatenate([idx, curb]), np.concatenate([cls, np.full(cy.size, C.CURB)]))
    _check_edges(edges)
    curbs = [e for e in edges if e["type"] == "curb"]
    assert len(curbs) == 1 and abs(np.mean([p[0] for p in curbs[0]["pts"]]) - 3.75) <= G.voxel


# ---------------------------------------------------------------- I9 viz
def _check_viz(msg):
    errs = validate_v1(msg)
    assert not errs, errs
    data = json.dumps(msg, separators=(",", ":"), allow_nan=False)
    assert len(data.encode()) <= MAX_DATAGRAM
    rt = json.loads(data)
    assert rt == msg and not validate_v1(rt) and not validate_v1(strip_to_v1(rt))
    assert set(strip_to_v1(rt)) <= MESSAGE_KEYS
    assert msg["lanes"] == []  # no paint label source: no invented lanes
    assert -(2**63) <= msg["seq"] < 2**63
    for tr in msg["tracks"]:
        assert tr["src"] in C.KNOWN_SOURCES and tr["motion"] in MOTION_STATES and tr["motion_src"] == C.SRC_MOTION
        assert tr["motion"] == "unknown" or tr["motion_frames"] >= 2
        assert "id" not in tr or -(2**31) <= tr["id"] < 2**31
    for ed in msg["edges"]:
        assert ed["src"] in C.KNOWN_SOURCES and ed["type"] in C.RED_POLYLINE_KINDS
    assert all(v == C.UNLABELED for v in msg["unlabeled"].values())
    rgb = {int(k): tuple(v) for k, v in msg["occ"]["rgb"].items()}
    for cell in msg["occ"]["cells"]:
        assert 0 <= cell[3] < C.N_CLASSES
        assert G.x_min <= cell[0] <= G.x_max and G.y_min <= cell[1] <= G.y_max and G.z_min <= cell[2] <= G.z_max
        assert C.is_red(rgb[cell[3]]) == (cell[3] in C.RED_CLASSES)
    noisy = dict(msg, zzz_unknown={"a": [1, 2]}, occ2=[1])
    assert not validate_v1(noisy)


if HAVE_HYP:
    track_st = st.fixed_dictionaries(
        {"x": st.one_of(st.floats(-50, 50), anyfloat), "y": st.one_of(st.floats(-50, 100), anyfloat), "id": st.one_of(st.none(), st.integers(-2**40, 2**40), st.booleans()),
         "yaw": st.one_of(st.floats(-4, 4), anyfloat), "motion": st.one_of(st.sampled_from(MOTION_STATES), st.text(max_size=6)), "motion_frames": st.integers(-1, 9),
         "l": anyfloat, "v": st.floats(0, 40), "src": st.one_of(st.sampled_from(sorted(C.KNOWN_SOURCES)), st.text(max_size=8))}
    )

    @given(
        st.floats(0, 1e6), st.integers(0, 2**70),
        st.lists(st.tuples(st.integers(-3, 170), st.integers(-3, 170), st.integers(-3, 10), st.integers(-2, 9), st.one_of(st.floats(0, 1), anyfloat)), max_size=300),
        st.lists(track_st, max_size=90),
        st.lists(st.tuples(st.lists(st.tuples(anyfloat, anyfloat), max_size=150), st.one_of(st.sampled_from(["road_edge", "curb", "lane", "stop_line"]), st.text(max_size=5)), st.one_of(st.sampled_from(sorted(C.KNOWN_SOURCES)), st.text(max_size=5))), max_size=12),
        st.floats(0, 60), st.booleans(),
    )
    def test_occ_to_viz(t, seq, cells, tracks, edges, speed, engaged):
        a = np.array([c[:4] for c in cells], dtype=np.int64).reshape(-1, 4)
        conf = np.array([c[4] for c in cells], dtype=np.float32)
        outputs = {"t": t, "cells": {"idx": a[:, :3], "cls": a[:, 3], "conf": conf}, "tracks": tracks,
                   "edges": [{"pts": [list(p) for p in e], "type": ty, "src": sr} for e, ty, sr in edges]}
        _check_viz(occ_to_viz(outputs, seq=seq, ego={"speed_mps": speed, "engaged": engaged}))


def test_viz_edge_cases():
    n = 100000
    rng = np.random.default_rng(0)
    idx = np.stack([rng.integers(0, 80, n), rng.integers(0, 160, n), rng.integers(0, 8, n)], axis=1)
    out = {"t": 1.0, "cells": {"idx": idx, "cls": rng.integers(0, 5, n), "conf": rng.random(n)}, "tracks": [{"x": 1.0, "y": float(i), "motion": "parked", "motion_frames": 3} for i in range(200)], "edges": []}
    msg = occ_to_viz(out)
    _check_viz(msg)
    assert len(msg["tracks"]) == 64 and msg["occ"]["n_total"] == n
    _check_viz(occ_to_viz({"t": 0.0}))
    for bad_t in (float("nan"), float("inf"), None, "x"):
        try:
            occ_to_viz({"t": bad_t})
        except Exception:
            continue
        raise AssertionError(f"t={bad_t!r} accepted")
    base = occ_to_viz({"t": 0.0})
    for key, val in (("v", 2), ("type", "x"), ("t", float("nan")), ("tracks", [{"x": 0, "y": 0, "cls": "vehicle"}] * 65), ("lanes", [{"pts": [[0, 0], [0, 1]], "paint": "purple"}]), ("seq", True), ("edges", [{"pts": [[0, 0]]}])):
        assert validate_v1(dict(base, **{key: val})), key
    for k, rgb in C.CLASS_RGB.items():
        assert C.is_red(rgb) == (k in C.RED_CLASSES), (k, rgb)
    assert C.RED_POLYLINE_KINDS == {"road_edge", "curb"}
    for k in range(C.N_CLASSES):
        assert C.class_src(k) in C.KNOWN_SOURCES and C.class_src(k) != C.UNLABELED
    assert C.class_src(99) == C.UNLABELED


def test_v1_constants_match_f3555bf():
    """The mirrored v1 caps/keys equal ``python/viz/cpp_publisher.py`` at f3555bf when git can read it."""
    from gvd_test_env import skip_check

    try:
        src = subprocess.run(["git", "show", "f3555bf:python/viz/cpp_publisher.py"], cwd=ROOT, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        skip_check("v1_constants", f"git unavailable: {exc}")
        return
    if src.returncode != 0:
        skip_check("v1_constants", "f3555bf not in this clone")
        return
    ns: dict = {}
    for line in src.stdout.splitlines():
        if line.startswith(("MAX_", "SCHEMA_VERSION")) and "=" in line:
            exec(line, ns)  # noqa: S102 - constants only
    from python.occ import viz_adapter as V

    for k in ("MAX_DATAGRAM", "MAX_TRACKS", "MAX_SIGNS", "MAX_LANES", "MAX_EDGES", "MAX_POINTS", "SCHEMA_VERSION"):
        assert ns[k] == getattr(V, k), k
    text = src.stdout
    body = text[text.index("MESSAGE_KEYS"):]
    keys = set(json.loads("[" + body[body.index("{") + 1 : body.index("}")].replace("\n", "").strip().rstrip(",") + "]"))
    assert keys == set(V.MESSAGE_KEYS), keys ^ set(V.MESSAGE_KEYS)


# Minimal counterexamples hypothesis shrank to before the fixes (fuzz profile, 1500 examples).
# Each of these failed on the first cut of the code (see docs/specs/occ_sparse.md, Adversarial evidence).
def test_reg_r1_grid_huge_coordinate():
    """A huge finite coordinate went through an undefined float->int64 cast."""
    with np.errstate(all="raise"):
        _, ok = G.to_index(np.array([[4.61168602e18] * 3]))
    assert not ok.any()


def test_reg_r2_shards_stale_build():
    """A second build into the same folder left the first build's shards readable as one set."""
    test_shard_no_stale_mix()


def test_reg_r3_tracker_fast_car():
    """A same-way car at 33 m/s, dt 0.094 s, moved 3.1 m > the 3 m gate and never reached 2 frames."""
    tr = MotionTracker()
    rows = []
    for k in range(8):
        rows = tr.update(k * 0.09375, [{"x": 0.0, "y": 5.0 + 33.0 * 0.09375 * k, "cls": "vehicle"}], speed_mps=0.0)
    assert rows[0]["motion_frames"] >= 2 and rows[0]["motion"] in ("same_way", "alongside"), rows


def test_reg_r4_viz_track_id_int32():
    msg = occ_to_viz({"t": 0.0, "tracks": [{"x": 0.0, "y": 0.0, "id": 2**31, "motion": "parked", "motion_frames": 3}]})
    _check_viz(msg)


def test_reg_r5_viz_unknown_motion_string():
    msg = occ_to_viz({"t": 0.0, "tracks": [{"x": 0.0, "y": 1.0, "motion": "0", "motion_frames": 3}]})
    _check_viz(msg)


def test_reg_r6_viz_single_frame_motion():
    msg = occ_to_viz({"t": 0.0, "tracks": [{"x": 0.0, "y": 2.0, "motion": "parked", "motion_frames": 1}]})
    _check_viz(msg)
    assert msg["tracks"][0]["motion"] == "unknown"


def test_reg_r7_viz_edge_src_and_type():
    msg = occ_to_viz({"t": 0.0, "edges": [{"pts": [[0, 0], [0, 1]], "type": "road_edge", "src": ""},
                                          {"pts": [[0, 0], [0, 1]], "type": "lane", "src": C.SRC_EDGE}]})
    _check_viz(msg)
    assert [e["src"] for e in msg["edges"]] == [C.UNLABELED]


def test_reg_r8_viz_cells_out_of_grid_or_class():
    msg = occ_to_viz({"t": 0.0, "cells": {"idx": np.array([[1, 1, -1], [1, 1, 1]]), "cls": np.array([1, -1]), "conf": np.array([0.5, 0.5])}})
    _check_viz(msg)
    assert msg["occ"]["cells"] == []


def test_reg_r9_viz_seq_int64():
    _check_viz(occ_to_viz({"t": 0.0}, seq=2**63))


def test_reg_r10_objects_height_bounds_conf():
    from python.occ.objects import objects_from_cells

    o = objects_from_cells(np.array([[0, 0, 0], [0, 1, 0]]), np.array([3, 3]), np.array([np.nan, np.nan]))
    assert o and o[0]["h"] > 0 and 0.0 <= o[0]["conf"] <= 1.0, o
    assert objects_from_cells(np.array([[-1, 0, 0], [-1, 1, 0]]), np.array([3, 3])) == []


REGRESSION_TESTS = [n for n in list(globals()) if n.startswith("test_reg_r")]


PROPERTY_TESTS = [
    "test_grid_index", "test_fine_in_coarse", "test_camera_roundtrip", "test_labels_invariants", "test_depth_to_m",
    "test_sampler_mix", "test_sampler_pure_raises", "test_shard_roundtrip", "test_tracker_fuzz", "test_tracker_physics",
    "test_edges_straight_road", "test_edges_random_cells", "test_objects_from_cells", "test_occ_to_viz",
]
PLAIN_TESTS = [
    "test_labels_edge_cases", "test_labels_synthetic_scene", "test_sampler_edge_cases", "test_shard_no_stale_mix", "test_shard_edge_cases",
    "test_tracker_edge_cases", "test_curb_line", "test_viz_edge_cases", "test_v1_constants_match_f3555bf",
]


def main() -> int:
    from gvd_test_env import skip_check

    for name in PLAIN_TESTS + REGRESSION_TESTS:
        globals()[name]()
        print(f"{name} OK", flush=True)
    for name in PROPERTY_TESTS:
        if not HAVE_HYP:
            skip_check(name, "hypothesis not installed (pip install -r requirements-dev.txt)")
            continue
        globals()[name]()
        print(f"{name} OK (hypothesis)", flush=True)
    print("occ_sparse invariants I1-I9 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
