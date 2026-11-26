#!/usr/bin/env python3
"""Offline CPU checks for Reality View step 6 (rolling merge: voxel-hash EMA static map).

Spec: docs/specs/reality_view_3dgs.md step 6 and ``check_no_ghost_rolling_map``.
Module under test: python/viz/rv_rolling_map.py (pure numpy; no GL / CUDA / device code).

Invariants (R = ``RollingMap``; every one is asserted by the examples, the seeded fuzz and
the hypothesis properties):
I1  Shapes & dtypes: snapshot xyz (n,3) float64, rgb (n,3) uint8, opacity (n,) float32,
    last_t_s (n,) float64, hits (n,) int64; all finite; opacity in (0, 1]; hits >= 1.
I2  Count: len(map) <= n_max after every merge (n_max <= 300_000 = rv_contract.RV_MAX_SPLATS).
I3  Window: every stored centre is within +/- radius_m (<= 64 m) of the ego of the newest merge,
    per axis.
I4  Cells: at most one record per 0.25 m voxel; the stored centre lies inside its own voxel.
I5  EMA: centre/colour move toward the frame mean by alpha; opacity w' = w + beta*o*(1-w) is
    bounded by 1 and never decreases on re-observation; a new cell starts at W_NEW*o < 0.1.
I6  Eviction: age > max_age_s goes; outside the window goes; over n_max the oldest go first,
    ties farthest first, ties lowest key.
I7  Dynamic: a point inside a dynamic box (+ margin) never enters the map; cells already
    inside such a box are purged; an all-dynamic frame adds nothing; no stored centre is inside
    a current box.
I8  Determinism: same call sequence -> byte-identical snapshots; no wall clock, no RNG.
I9  Empty input at the same (t, ego) is a no-op; contract violations raise ValueError with the
    state unchanged.
I10 Clock: a timestamp > backward_tol_s behind the newest is refused (state unchanged); a smaller
    step back is clamped; a forward jump simply ages cells out.
I11 Model equivalence: RollingMap == an independent dict-based reference model (tolerance 1e-9).
I12 CPU-only: the module imports only numpy/stdlib and carries no GL / CUDA / device-selection
    code or config.

``check_no_ghost_rolling_map`` is the spec test with the spec thresholds (unmeasured, proposed;
not tuned): (a) duplicate-surface <= 5% of occupied 0.25 m cells; (b) median |z_render - z_gt|
<= 0.50 m under sigma=0.15 m / 1 deg pose noise and 80 ms inter-cam skew; (c) no persistent
static splat (opacity > 0.1) in a vehicle's old cells >= 1.0 s after it is gone.
Stricter readings chosen where the spec is loose are listed in docs/specs/rv_rolling_map_notes.md.
"""
from __future__ import annotations

import ast
import hashlib
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from gvd_test_env import skip_check  # noqa: E402

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
except ImportError:  # dev-only dep (/workspace/swarm/swarm-4/devdeps)
    given = None

from python.viz import rv_contract  # noqa: E402
from python.viz import rv_rolling_map as rm  # noqa: E402
from python.viz.rv_rolling_map import DynamicBox, MapSnapshot, MergeStats, RollingMap  # noqa: E402

# ---- spec thresholds (docs/specs/reality_view_3dgs.md step 6), copied verbatim, never tuned ----
SPEC_CELL_M = 0.25
SPEC_DUP_RANGE_M = 0.5
SPEC_DUP_OPACITY = 0.1
SPEC_DUP_MAX_RATE = 0.05
SPEC_NOISE_T_M = 0.15
SPEC_NOISE_ROT_DEG = 1.0
SPEC_SKEW_S = 0.080
SPEC_DEPTH_RANGE_M = (0.5, 40.0)
SPEC_MEDIAN_MAX_M = 0.50
SPEC_GONE_S = 1.0
SPEC_LEFTOVER_OPACITY = 0.1


# =====================================================================================
# Reference model (independent dict implementation of the same contract, for I11)
# =====================================================================================
def _dist3(d) -> float:
    """Euclidean norm in the exact operation order the module uses (so cap tie-breaks agree bit for bit)."""
    return math.sqrt(float(d[0]) * float(d[0]) + float(d[1]) * float(d[1]) + float(d[2]) * float(d[2]))


class ModelMap:
    def __init__(self, **kw):
        d = dict(voxel_m=0.25, radius_m=64.0, n_max=300_000, max_age_s=30.0, alpha=0.3, beta=0.35,
                 dyn_margin_m=0.5, backward_tol_s=0.25)
        d.update(kw)
        self.__dict__.update(d)
        self.cells: dict[tuple[int, int, int], dict] = {}
        self.t = -math.inf

    @staticmethod
    def _in_box(p, b: DynamicBox, margin: float) -> bool:
        dx, dy, dz = p[0] - b.cx, p[1] - b.cy, p[2] - b.cz
        c, s = math.cos(b.yaw_rad), math.sin(b.yaw_rad)
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        return abs(lx) <= b.length / 2 + margin and abs(ly) <= b.width / 2 + margin and abs(dz) <= b.height / 2 + margin

    def merge(self, pts, rgb, t, ego, boxes=(), opacity=None):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        ego = np.asarray(ego, dtype=np.float64)
        if t < self.t - self.backward_tol_s:
            return "rejected_clock_backwards"
        te = max(t, self.t)
        for key in list(self.cells):
            c = self.cells[key]
            if te - c["t"] > self.max_age_s or np.max(np.abs(c["p"] - ego)) > self.radius_m or any(
                    self._in_box(c["p"], b, self.dyn_margin_m) for b in boxes):
                del self.cells[key]
        groups: dict[tuple[int, int, int], list] = {}
        for i in range(pts.shape[0]):
            p = pts[i]
            o = 1.0 if opacity is None else float(opacity[i])
            if not (np.all(np.isfinite(p)) and np.isfinite(o) and o > 0.0):
                continue
            if np.max(np.abs(p - ego)) > self.radius_m:
                continue
            if any(self._in_box(p, b, self.dyn_margin_m) for b in boxes):
                continue
            key = tuple(int(x) for x in np.floor(p / self.voxel_m))
            groups.setdefault(key, []).append((p, np.asarray(rgb[i], dtype=np.float64), o))
        for key, items in groups.items():
            mp = sum(x[0] for x in items) / len(items)
            mc = sum(x[1] for x in items) / len(items)
            mo = sum(x[2] for x in items) / len(items)
            if key in self.cells:
                c = self.cells[key]
                c["p"] = c["p"] + self.alpha * (mp - c["p"])
                c["c"] = c["c"] + self.alpha * (mc - c["c"])
                c["w"] = min(1.0, c["w"] + self.beta * mo * (1.0 - c["w"]))
                c["t"] = te
                c["h"] += 1
            else:
                self.cells[key] = dict(p=mp, c=mc, w=0.05 * mo, t=te, h=1)
        # final purge: no centre may sit inside a current box
        for key in list(self.cells):
            if any(self._in_box(self.cells[key]["p"], b, self.dyn_margin_m) for b in boxes):
                del self.cells[key]
        if len(self.cells) > self.n_max:
            order = sorted(self.cells, key=lambda k: (-(te - self.cells[k]["t"]),
                                                      -_dist3(self.cells[k]["p"] - ego), k))
            for k in order[: len(self.cells) - self.n_max]:
                del self.cells[k]
        self.t = te
        return "ok"

    def arrays(self):
        keys = sorted(self.cells)
        if not keys:
            return (np.empty((0, 3)), np.empty((0, 3)), np.empty((0,)), np.empty((0,)), np.empty((0,), np.int64))
        return (np.array([self.cells[k]["p"] for k in keys]), np.array([self.cells[k]["c"] for k in keys]),
                np.array([self.cells[k]["w"] for k in keys]), np.array([self.cells[k]["t"] for k in keys]),
                np.array([self.cells[k]["h"] for k in keys], dtype=np.int64))


def _snap_bytes(s: MapSnapshot) -> bytes:
    return b"".join(a.tobytes() for a in (s.xyz, s.rgb, s.opacity, s.last_t_s, s.hits))


def _assert_matches_model(m: RollingMap, model: ModelMap, ctx: str = "") -> None:
    s = m.snapshot()
    p, c, w, t, h = model.arrays()
    assert s.xyz.shape == p.shape, f"{ctx}: cell count {s.xyz.shape} vs model {p.shape}"
    if p.shape[0]:
        assert np.allclose(s.xyz, p, atol=1e-9, rtol=0), f"{ctx}: xyz differs from model"
        assert np.array_equal(s.rgb, np.clip(np.rint(c), 0, 255).astype(np.uint8)) or np.max(
            np.abs(s.rgb.astype(np.float64) - np.clip(np.rint(c), 0, 255))) <= 1, f"{ctx}: rgb differs from model"
        assert np.allclose(s.opacity, w.astype(np.float32), atol=1e-6, rtol=0), f"{ctx}: opacity differs from model"
        assert np.array_equal(s.last_t_s, t) and np.array_equal(s.hits, h), f"{ctx}: age/hits differ from model"


# =====================================================================================
# Invariant checker used by examples, fuzz and properties
# =====================================================================================
def _assert_invariants(m: RollingMap, ego, boxes=(), ctx: str = "") -> None:
    s = m.snapshot()
    n = len(m)
    assert s.xyz.shape == (n, 3) and s.xyz.dtype == np.float64, ctx
    assert s.rgb.shape == (n, 3) and s.rgb.dtype == np.uint8, ctx
    assert s.opacity.shape == (n,) and s.opacity.dtype == np.float32, ctx
    assert s.last_t_s.shape == (n,) and s.last_t_s.dtype == np.float64, ctx
    assert s.hits.shape == (n,) and s.hits.dtype == np.int64, ctx
    assert n <= m.n_max <= rv_contract.RV_MAX_SPLATS, f"{ctx}: count {n} > n_max {m.n_max}"
    if n == 0:
        return
    assert bool(np.all(np.isfinite(s.xyz))) and bool(np.all(np.isfinite(s.last_t_s))), ctx
    assert bool(np.all(s.opacity > 0.0)) and bool(np.all(s.opacity <= 1.0)), f"{ctx}: opacity range"
    assert bool(np.all(s.hits >= 1)), ctx
    ego = np.asarray(ego, dtype=np.float64)
    assert float(np.max(np.abs(s.xyz - ego))) <= m.radius_m + 1e-9, f"{ctx}: point outside +/-{m.radius_m} m window"
    ijk = np.floor(s.xyz / m.voxel_m)
    assert np.unique(ijk, axis=0).shape[0] == n, f"{ctx}: two records in one voxel"
    lo, hi = ijk * m.voxel_m, (ijk + 1) * m.voxel_m
    assert bool(np.all(s.xyz >= lo - 1e-9)) and bool(np.all(s.xyz <= hi + 1e-9)), f"{ctx}: centre left its voxel"
    assert float(np.max(s.last_t_s)) <= m.t_s, ctx
    for b in boxes:
        d = s.xyz - np.array([b.cx, b.cy, b.cz])
        cy, sy = math.cos(b.yaw_rad), math.sin(b.yaw_rad)
        lx, ly = cy * d[:, 0] + sy * d[:, 1], -sy * d[:, 0] + cy * d[:, 1]
        inside = (np.abs(lx) <= b.length / 2 + m.dyn_margin_m) & (np.abs(ly) <= b.width / 2 + m.dyn_margin_m) & (
            np.abs(d[:, 2]) <= b.height / 2 + m.dyn_margin_m)
        assert not bool(inside.any()), f"{ctx}: stored centre inside a dynamic box"


def _expect_value_error(fn, needle: str | None = None) -> None:
    try:
        fn()
    except ValueError as exc:
        if needle is not None:
            assert str(exc) == needle, f"ValueError message {exc!s} != {needle!r}"
        return
    raise AssertionError("expected ValueError")


def _pts(*rows):
    return np.array(rows, dtype=np.float64).reshape(-1, 3)


def _rgb(n, v=100):
    return np.full((n, 3), v, dtype=np.uint8)


# =====================================================================================
# Examples
# =====================================================================================
def check_spec_constants_and_cpu_only() -> None:
    """Spec constants, cap == RV_MAX_SPLATS, and the module is CPU-only numpy (I12, I8)."""
    assert rm.N_MAX_DEFAULT == 300_000 == rv_contract.RV_MAX_SPLATS
    assert rm.RADIUS_M_DEFAULT == 64.0 and rm.VOXEL_M_DEFAULT == 0.25
    m = RollingMap()
    assert (m.n_max, m.radius_m, m.voxel_m) == (300_000, 64.0, 0.25) and len(m) == 0 and m.t_s == -math.inf
    src_path = ROOT / "python" / "viz" / "rv_rolling_map.py"
    src = src_path.read_text(encoding="utf-8")
    if "mutmut" in src:  # mutmut's trampolined copy imports mutmut and wraps every function: the source scan is meaningless there
        print("  (source scan skipped: mutmut trampoline copy)")
    else:
        tree = ast.parse(src, filename=str(src_path))
        allowed = {"__future__", "math", "dataclasses", "typing", "numpy"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert a.name.split(".")[0] in allowed, f"forbidden import {a.name}"
            elif isinstance(node, ast.ImportFrom):
                assert (node.level == 0) and (node.module or "").split(".")[0] in allowed, f"forbidden import from {node.module}"
        low = src.lower()
        for token in ("cuda", "gpu", "torch", "opengl", "device", "glfw", "moderngl", "vulkan", "cupy", "opencl",
                      "igpu", "random", "time.time", "perf_counter", "datetime", "beamng", "heartbeat", "aeb", "engage"):
            assert token not in low, f"module must not mention {token!r} (CPU-only, deterministic, off the safety path)"
    # defaults of the RV flag are untouched by this module (flag lives in rv_contract, default OFF)
    assert rv_contract.reality_view_enabled(None) is False
    print("check_spec_constants_and_cpu_only: OK")


def check_contract_errors_leave_state() -> None:
    """Bad shapes / dtypes / scalars raise ValueError and leave the map untouched (I9)."""
    m = RollingMap(voxel_m=1.0, radius_m=10.0)
    ego = (0.0, 0.0, 0.0)
    m.merge(_pts((1, 1, 1), (2, 2, 2)), _rgb(2), t_s=1.0, ego_xyz=ego)
    before = _snap_bytes(m.snapshot())
    good = dict(t_s=2.0, ego_xyz=ego)
    p, c = _pts((1, 1, 1)), _rgb(1)
    bad_calls = [
        (lambda: m.merge(np.zeros((2, 2)), _rgb(2)[:, :2], **good), "points_xyz must be an (N, 3) real array"),
        (lambda: m.merge(np.zeros(3), np.zeros(3, np.uint8), **good), "points_xyz must be an (N, 3) real array"),
        (lambda: m.merge(np.zeros((1, 3), dtype=bool), c, **good), "points_xyz must be an (N, 3) real array"),
        (lambda: m.merge(np.zeros((1, 3), dtype=complex), c, **good), "points_xyz must be an (N, 3) real array"),
        (lambda: m.merge(p, c.astype(np.float32), **good), "rgb must be a uint8 array shaped like points_xyz"),
        (lambda: m.merge(p, _rgb(2), **good), "rgb must be a uint8 array shaped like points_xyz"),
        (lambda: m.merge(p, c, t_s=float("nan"), ego_xyz=ego), "t_s must be finite"),
        (lambda: m.merge(p, c, t_s=float("inf"), ego_xyz=ego), "t_s must be finite"),
        (lambda: m.merge(p, c, t_s=True, ego_xyz=ego), "t_s must be a real number"),
        (lambda: m.merge(p, c, t_s="x", ego_xyz=ego), "t_s must be a real number"),
        (lambda: m.merge(p, c, t_s=2.0, ego_xyz=(0.0, 0.0)), "ego_xyz must be 3 finite floats"),
        (lambda: m.merge(p, c, t_s=2.0, ego_xyz=(0.0, float("nan"), 0.0)), "ego_xyz must be 3 finite floats"),
        (lambda: m.merge(p, c, t_s=2.0, ego_xyz=(1e12, 0.0, 0.0)), "ego_xyz outside the supported world range"),
        (lambda: m.merge(p, c, t_s=2.0, ego_xyz=(0.0, -1e12, 0.0)), "ego_xyz outside the supported world range"),
        (lambda: m.merge(p, c, opacity=np.array([1.5]), **good), "opacity must not exceed 1"),
        (lambda: m.merge(p, c, opacity=np.array([0.5, 0.5]), **good), "opacity must be an (N,) real array"),
        (lambda: m.merge(p, c, opacity=np.array(["a"]), **good), "opacity must be an (N,) real array"),
        (lambda: m.merge(p, c, dynamic_boxes=[object()], **good), "dynamic_boxes must be DynamicBox instances"),
        (lambda: m.merge(p, c, dynamic_boxes=[DynamicBox(0, 0, 0, 0.0, 1, 1)], **good), "dynamic box dimensions must be > 0"),
        (lambda: m.merge(p, c, dynamic_boxes=[DynamicBox(0, 0, 0, 1, -1, 1)], **good), "dynamic box dimensions must be > 0"),
        (lambda: m.merge(p, c, dynamic_boxes=[DynamicBox(0, 0, 0, 1, 1, 0.0)], **good), "dynamic box dimensions must be > 0"),
        (lambda: m.merge(p, c, dynamic_boxes=[DynamicBox(float("nan"), 0, 0, 1, 1, 1)], **good), "cx must be finite"),
        (lambda: m.merge(p, c, dynamic_boxes=[DynamicBox(0, 0, 0, 1, 1, 1, float("inf"))], **good), "yaw_rad must be finite"),
        (lambda: m.merge(p, c, dynamic_boxes=[DynamicBox(0, 0, 0, 1, 1, 1)] * (rm.MAX_DYNAMIC_BOXES + 1), **good), "too many dynamic boxes"),
        (lambda: m.merge(p, c, dynamic_boxes=[DynamicBox(0, 0, 0, 1, 0.0, 1)], **good), "dynamic box dimensions must be > 0"),
        (lambda: m.merge(p, c, dynamic_boxes=[DynamicBox(0, 0, 0, -1, 1, 1)], **good), "dynamic box dimensions must be > 0"),
        (lambda: m.merge(p, c, dynamic_boxes=[DynamicBox(0, 0, 0, 1, 1, -1)], **good), "dynamic box dimensions must be > 0"),
        (lambda: m.merge(p, c, dynamic_boxes=[DynamicBox("a", 0, 0, 1, 1, 1)], **good), "cx must be a real number"),
        (lambda: m.merge(p, c, dynamic_boxes=[DynamicBox(0, 0, 0, 1, 1, True)], **good), "height must be a real number"),
        (lambda: m.merge(p, c, dynamic_boxes=[DynamicBox(0, 0, 0, 1, 1, float("inf"))], **good), "height must be finite"),
        (lambda: m.merge(p, c, dynamic_boxes=[DynamicBox(0, 0, float("nan"), 1, 1, 1)], **good), "cz must be finite"),
        (lambda: m.merge(p, c, dynamic_boxes=[DynamicBox(0, float("nan"), 0, 1, 1, 1)], **good), "cy must be finite"),
        (lambda: m.merge(p, c, dynamic_boxes=[DynamicBox(0, 0, 0, float("nan"), 1, 1)], **good), "length must be finite"),
        (lambda: m.merge(p, c, dynamic_boxes=[DynamicBox(0, 0, 0, 1, float("nan"), 1)], **good), "width must be finite"),
        (lambda: m.merge(p, c, t_s=2.0, ego_xyz=(10**30, 0, 0)), "ego_xyz outside the supported world range"),
        (lambda: m.merge(p, c, t_s=2.0, ego_xyz=np.array([0.0, 0.0, 0.0, 0.0])), "ego_xyz must be 3 finite floats"),
        (lambda: m.merge(p, c, t_s=2.0, ego_xyz=(0.0, 0.0, float("inf"))), "ego_xyz must be 3 finite floats"),
    ]
    for fn, needle in bad_calls:
        _expect_value_error(fn, needle)
        assert _snap_bytes(m.snapshot()) == before and m.t_s == 1.0, "contract error mutated state"
    # exactly MAX_DYNAMIC_BOXES boxes is accepted
    m.merge(p, c, dynamic_boxes=[DynamicBox(50, 50, 50, 1, 1, 1)] * rm.MAX_DYNAMIC_BOXES, **good)
    # constructor validation (exact messages)
    nan_ = float("nan")
    for kw, msg in (
        (dict(voxel_m=0.04), "voxel_m must be in [0.05, 8]"), (dict(voxel_m=8.1), "voxel_m must be in [0.05, 8]"),
        (dict(voxel_m=nan_), "voxel_m must be finite"), (dict(voxel_m="x"), "voxel_m must be a real number"),
        (dict(radius_m=64.1), "radius_m must be in [voxel_m, 64]"), (dict(radius_m=0.2), "radius_m must be in [voxel_m, 64]"),
        (dict(radius_m=nan_), "radius_m must be finite"), (dict(radius_m=None), "radius_m must be a real number"),
        (dict(n_max=0), "n_max must be in [1, 300000]"), (dict(n_max=300_001), "n_max must be in [1, 300000]"),
        (dict(n_max=True), "n_max must be an int"), (dict(n_max=1.5), "n_max must be an int"),
        (dict(n_max=np.bool_(True)), "n_max must be an int"),
        (dict(max_age_s=0.0), "max_age_s must be > 0"), (dict(max_age_s=-1.0), "max_age_s must be > 0"),
        (dict(max_age_s=float("inf")), "max_age_s must be finite"), (dict(max_age_s=nan_), "max_age_s must be finite"),
        (dict(alpha=0.0), "alpha and beta must be in (0, 1]"), (dict(alpha=1.1), "alpha and beta must be in (0, 1]"),
        (dict(beta=0.0), "alpha and beta must be in (0, 1]"), (dict(beta=1.1), "alpha and beta must be in (0, 1]"),
        (dict(alpha=nan_), "alpha must be finite"), (dict(beta=nan_), "beta must be finite"),
        (dict(alpha=True), "alpha must be a real number"), (dict(beta=False), "beta must be a real number"),
        (dict(dyn_margin_m=-0.1), "dyn_margin_m and backward_tol_s must be >= 0"),
        (dict(backward_tol_s=-1.0), "dyn_margin_m and backward_tol_s must be >= 0"),
        (dict(dyn_margin_m=nan_), "dyn_margin_m must be finite"), (dict(backward_tol_s=nan_), "backward_tol_s must be finite"),
    ):
        _expect_value_error(lambda kw=kw: RollingMap(**kw), msg)
    RollingMap(voxel_m=0.05, radius_m=0.05, n_max=1, alpha=1.0, beta=1.0, dyn_margin_m=0.0, backward_tol_s=0.0)
    RollingMap(voxel_m=8.0, radius_m=64.0, n_max=300_000)
    RollingMap(n_max=np.int64(5))
    # too many points in one merge: exactly the limit is fine, one more is not (limit patched down to keep this fast)
    old_limit = rm.MAX_MERGE_POINTS
    rm.MAX_MERGE_POINTS = 10
    try:
        m.merge(np.zeros((10, 3)) + 50.0, _rgb(10), t_s=3.0, ego_xyz=ego)  # all outside the window: accepted, dropped
        _expect_value_error(lambda: m.merge(np.zeros((11, 3)), _rgb(11), t_s=3.0, ego_xyz=ego), "too many points in one merge")
    finally:
        rm.MAX_MERGE_POINTS = old_limit
    assert old_limit == 4_000_000
    # world-range limit is exact: (2**20 - 2) * voxel - radius; ego just inside works, just outside is refused
    mw = RollingMap(voxel_m=0.25, radius_m=64.0)
    lim = (2 ** 20 - 2) * 0.25 - 64.0
    assert mw._world_limit_m == lim == 262079.5
    mw.merge(_pts((lim + 63.9, 0.0, 0.0)), _rgb(1), t_s=0.0, ego_xyz=(lim, 0.0, 0.0))
    assert len(mw) == 1
    _expect_value_error(lambda: mw.merge(_pts((0, 0, 0)), _rgb(1), t_s=1.0, ego_xyz=(lim + 0.01, 0.0, 0.0)), "ego_xyz outside the supported world range")
    mw.merge(_pts((-lim - 63.9, 0.0, 0.0)), _rgb(1), t_s=2.0, ego_xyz=(-lim, 0.0, 0.0))
    assert len(mw) == 1
    _expect_value_error(lambda: mw.merge(_pts((0, 0, 0)), _rgb(1), t_s=3.0, ego_xyz=(0.0, -lim - 0.01, 0.0)), "ego_xyz outside the supported world range")
    _expect_value_error(lambda: mw.merge(_pts((0, 0, 0)), _rgb(1), t_s=3.0, ego_xyz=(0.0, 0.0, lim + 0.01)), "ego_xyz outside the supported world range")
    mx = RollingMap(voxel_m=8.0, radius_m=64.0)
    assert mx._world_limit_m == (2 ** 20 - 2) * 8.0 - 64.0
    print("check_contract_errors_leave_state: OK")


def check_empty_and_garbage_points() -> None:
    """Empty input is a no-op at the same (t, ego); NaN/inf/<=0 opacity points are dropped, never stored (I3, I9)."""
    m = RollingMap(voxel_m=1.0, radius_m=10.0)
    ego = (0.0, 0.0, 0.0)
    st0 = m.merge(np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), t_s=0.0, ego_xyz=ego)
    assert st0.status == "ok" and st0.count == 0 and st0.n_in == 0 and len(m) == 0
    m.merge(_pts((1.5, 1.5, 1.5)), _rgb(1), t_s=1.0, ego_xyz=ego)
    snap = _snap_bytes(m.snapshot())
    stz = m.merge(np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), t_s=1.0, ego_xyz=ego)
    assert _snap_bytes(m.snapshot()) == snap and stz.count == 1 and stz.n_new == 0 and stz.n_updated == 0
    nan, inf = float("nan"), float("inf")
    garbage = _pts((nan, 0, 0), (0, inf, 0), (0, 0, -inf), (3.5, 3.5, 3.5), (4.5, 4.5, 4.5), (5.5, 5.5, 5.5), (6.5, 6.5, 6.5))
    op = np.array([1.0, 1.0, 1.0, nan, inf, 0.0, -0.5])
    s1 = m.merge(garbage, _rgb(7), t_s=1.0, ego_xyz=ego, opacity=op)
    assert (s1.n_in, s1.n_nonfinite, s1.n_new, s1.count) == (7, 7, 0, 1), s1
    mix = m.merge(_pts((nan, 0, 0), (2.5, 2.5, 2.5), (0, inf, 0), (3.5, 3.5, 3.5)), _rgb(4), t_s=1.0, ego_xyz=ego)
    assert (mix.n_in, mix.n_nonfinite, mix.n_out_of_window, mix.n_dynamic, mix.n_new, mix.n_updated) == (4, 2, 0, 0, 2, 0), mix
    m.merge(np.empty((0, 3)), np.empty((0, 3), np.uint8), t_s=1.0, ego_xyz=(0.0, 0.0, 0.0))
    # out of window: exactly on the boundary is kept, just outside is dropped (per axis, inclusive)
    s2 = m.merge(_pts((10.0, 0.0, 0.0), (10.0001, 0.0, 0.0), (0.0, -10.0, 0.0), (0.0, -10.0001, 0.0), (0.0, 0.0, 10.0001)),
                 _rgb(5), t_s=1.0, ego_xyz=ego)
    assert (s2.n_out_of_window, s2.n_new) == (3, 2), s2
    # huge finite coordinates are out of window, not errors
    s3 = m.merge(_pts((1e300, 0, 0), (-1e300, 0, 0)), _rgb(2), t_s=1.0, ego_xyz=ego)
    assert (s3.n_out_of_window, s3.n_new) == (2, 0)
    # integer / float32 inputs are accepted
    s4 = m.merge(np.array([[1, 1, 1]], dtype=np.int32), _rgb(1), t_s=1.0, ego_xyz=ego)
    assert s4.n_in == 1 and s4.n_updated == 1
    m.merge(np.array([[2.5, 2.5, 2.5]], dtype=np.float32), _rgb(1), t_s=1.0, ego_xyz=ego)
    _assert_invariants(m, ego, ctx="garbage")
    print("check_empty_and_garbage_points: OK")


def check_ema_and_opacity_numbers() -> None:
    """Exact EMA / opacity arithmetic and the one-frame-ghost-is-invisible rule (I5)."""
    m = RollingMap(voxel_m=1.0, radius_m=10.0, alpha=0.25, beta=0.5)
    ego = (0.0, 0.0, 0.0)
    m.merge(_pts((1.2, 1.2, 1.2), (1.4, 1.6, 1.8)), _rgb(2, 100), t_s=0.0, ego_xyz=ego, opacity=np.array([1.0, 0.5]))
    s = m.snapshot()
    assert len(m) == 1 and int(s.hits[0]) == 1
    assert np.allclose(s.xyz[0], (1.3, 1.4, 1.5)) and int(s.rgb[0, 0]) == 100
    assert abs(float(s.opacity[0]) - 0.05 * 0.75) < 1e-7 and float(s.opacity[0]) < rm.VISIBLE_OPACITY
    m.merge(_pts((1.9, 1.9, 1.9)), np.array([[200, 100, 0]], dtype=np.uint8), t_s=1.0, ego_xyz=ego)
    s = m.snapshot()
    assert np.allclose(s.xyz[0], 0.75 * np.array([1.3, 1.4, 1.5]) + 0.25 * 1.9)
    assert tuple(int(x) for x in s.rgb[0]) == (125, 100, 75)
    w1 = 0.0375 + 0.5 * 1.0 * (1 - 0.0375)
    assert abs(float(s.opacity[0]) - w1) < 1e-6 and float(s.opacity[0]) >= rm.VISIBLE_OPACITY
    assert int(s.hits[0]) == 2 and float(s.last_t_s[0]) == 1.0
    # saturation: w stays <= 1 and never drops over many hits
    prev = float(s.opacity[0])
    for i in range(60):
        m.merge(_pts((1.5, 1.5, 1.5)), _rgb(1), t_s=2.0 + i, ego_xyz=ego)
        cur = float(m.snapshot().opacity[0])
        assert prev <= cur <= 1.0
        prev = cur
    assert prev > 0.999
    assert int(m.snapshot().hits[0]) == 62
    # alpha == 1 replaces, beta == 1 saturates in one extra hit
    m2 = RollingMap(voxel_m=1.0, radius_m=10.0, alpha=1.0, beta=1.0)
    m2.merge(_pts((1.1, 1.1, 1.1)), _rgb(1, 10), t_s=0.0, ego_xyz=ego)
    m2.merge(_pts((1.9, 1.8, 1.7)), _rgb(1, 250), t_s=1.0, ego_xyz=ego)
    s2 = m2.snapshot()
    assert np.allclose(s2.xyz[0], (1.9, 1.8, 1.7)) and int(s2.rgb[0, 0]) == 250 and float(s2.opacity[0]) == 1.0
    # opacity weighting of the frame mean: low-opacity hits raise w less than full-opacity hits
    ma, mb = (RollingMap(voxel_m=1.0, radius_m=10.0) for _ in range(2))
    for mm, o in ((ma, 1.0), (mb, 0.2)):
        for t in (0.0, 1.0, 2.0):
            mm.merge(_pts((1.5, 1.5, 1.5)), _rgb(1), t_s=t, ego_xyz=ego, opacity=np.array([o]))
    assert float(mb.snapshot().opacity[0]) < float(ma.snapshot().opacity[0])
    # colour rounds (not truncates) and clips to uint8
    m3 = RollingMap(voxel_m=1.0, radius_m=10.0, alpha=0.5)
    m3.merge(_pts((1.5, 1.5, 1.5)), np.array([[10, 11, 255]], dtype=np.uint8), t_s=0.0, ego_xyz=ego)
    m3.merge(_pts((1.5, 1.5, 1.5)), np.array([[11, 11, 255]], dtype=np.uint8), t_s=1.0, ego_xyz=ego)
    assert tuple(int(x) for x in m3.snapshot().rgb[0]) == (10, 11, 255)  # 10.5 -> 10 (round half even)
    m3.merge(_pts((1.5, 1.5, 1.5)), np.array([[14, 11, 255]], dtype=np.uint8), t_s=2.0, ego_xyz=ego)
    assert tuple(int(x) for x in m3.snapshot().rgb[0]) == (12, 11, 255)  # 12.25 -> 12
    m5 = RollingMap(voxel_m=1.0, radius_m=10.0)
    m5.merge(_pts((1.5, 1.5, 1.5)), _rgb(1, 0), t_s=0.0, ego_xyz=ego)
    m5.merge(_pts((1.5, 1.5, 1.5)), _rgb(1, 0), t_s=1.0, ego_xyz=ego)
    assert tuple(int(x) for x in m5.snapshot().rgb[0]) == (0, 0, 0)  # black stays black
    # voxel assignment is floor (negative coordinates, exact boundaries)
    m4 = RollingMap(voxel_m=1.0, radius_m=10.0)
    m4.merge(_pts((-0.0001, 0.0, 0.0), (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.9999, 0.0, 0.0), (1.0, 0.0, 0.0)), _rgb(5), t_s=0.0, ego_xyz=ego)
    assert len(m4) == 3  # cells -1 (-0.0001, -1.0), 0 (0.0, 0.9999), 1 (1.0)
    print("check_ema_and_opacity_numbers: OK")


def check_eviction_examples() -> None:
    """Age, window, cap eviction exactly as specified (I6, I2, I3)."""
    ego = (0.0, 0.0, 0.0)
    # age: > max_age evicted, == max_age kept
    m = RollingMap(voxel_m=1.0, radius_m=10.0, max_age_s=5.0)
    m.merge(_pts((1.5, 1.5, 1.5)), _rgb(1), t_s=0.0, ego_xyz=ego)
    s = m.merge(np.empty((0, 3)), np.empty((0, 3), np.uint8), t_s=5.0, ego_xyz=ego)
    assert len(m) == 1 and s.n_evicted_age == 0
    s = m.merge(np.empty((0, 3)), np.empty((0, 3), np.uint8), t_s=5.0001, ego_xyz=ego)
    assert len(m) == 0 and s.n_evicted_age == 1 and s.n_evicted_window == 0
    # a re-hit resets the age
    m.merge(_pts((1.5, 1.5, 1.5)), _rgb(1), t_s=6.0, ego_xyz=ego)
    m.merge(_pts((1.5, 1.5, 1.5)), _rgb(1), t_s=10.0, ego_xyz=ego)
    m.merge(np.empty((0, 3)), np.empty((0, 3), np.uint8), t_s=14.0, ego_xyz=ego)
    assert len(m) == 1
    # window: ego moves, far cells go (inclusive boundary), counted once
    m = RollingMap(voxel_m=1.0, radius_m=10.0)
    m.merge(_pts((1.5, 1.5, 1.5), (9.5, 1.5, 1.5), (-8.5, 1.5, 1.5)), _rgb(3), t_s=0.0, ego_xyz=ego)
    s = m.merge(np.empty((0, 3)), np.empty((0, 3), np.uint8), t_s=0.0, ego_xyz=(-0.5, 0.0, 0.0))
    assert len(m) == 3 and s.n_evicted_window == 0  # 9.5 - (-0.5) = 10.0 inclusive
    s = m.merge(np.empty((0, 3)), np.empty((0, 3), np.uint8), t_s=0.0, ego_xyz=(-0.6, 0.0, 0.0))
    assert len(m) == 2 and s.n_evicted_window == 1
    s = m.merge(np.empty((0, 3)), np.empty((0, 3), np.uint8), t_s=0.0, ego_xyz=(0.0, 0.0, 11.6))
    assert len(m) == 0 and s.n_evicted_window == 2  # z axis counts too
    # teleport: everything old is gone, new frame survives
    m = RollingMap(voxel_m=1.0, radius_m=10.0)
    m.merge(_pts((1.5, 1.5, 1.5), (2.5, 2.5, 2.5)), _rgb(2), t_s=0.0, ego_xyz=ego)
    m.merge(_pts((1001.5, 5.5, 5.5)), _rgb(1), t_s=0.1, ego_xyz=(1000.0, 5.0, 5.0))
    assert len(m) == 1 and float(m.snapshot().xyz[0, 0]) == 1001.5
    # age and window counters are exclusive (a cell both old and far counts as age once)
    m = RollingMap(voxel_m=1.0, radius_m=10.0, max_age_s=1.0)
    m.merge(_pts((1.5, 1.5, 1.5)), _rgb(1), t_s=0.0, ego_xyz=ego)
    s = m.merge(np.empty((0, 3)), np.empty((0, 3), np.uint8), t_s=9.0, ego_xyz=(100.0, 0.0, 0.0))
    assert (s.n_evicted_age, s.n_evicted_window, s.n_purged_dynamic) == (1, 0, 0)
    # cap: oldest first, ties farthest first, ties lowest key; exact count
    m = RollingMap(voxel_m=1.0, radius_m=60.0, n_max=3)
    m.merge(_pts((5.5, 0.5, 0.5)), _rgb(1), t_s=0.0, ego_xyz=ego)       # oldest, near
    m.merge(_pts((30.5, 0.5, 0.5)), _rgb(1), t_s=1.0, ego_xyz=ego)      # newer, far
    s = m.merge(_pts((2.5, 0.5, 0.5), (20.5, 0.5, 0.5), (-20.5, 0.5, 0.5), (1.5, 0.5, 0.5)), _rgb(4), t_s=2.0, ego_xyz=ego)
    xs = sorted(float(x) for x in m.snapshot().xyz[:, 0])
    # 6 candidates, drop 3: age 2.0 -> (5.5), age 1.0 -> (30.5), then age 0 ties: farthest 20.5 vs -20.5 tie -> lowest key (-20.5)
    assert len(m) == 3 and s.n_evicted_cap == 3 and xs == [1.5, 2.5, 20.5], xs
    # cap with all-equal age and distance falls to lowest key; cap == n_max exactly keeps everything
    m = RollingMap(voxel_m=1.0, radius_m=60.0, n_max=2)
    m.merge(_pts((10.5, 0.5, 0.5), (-10.5, 0.5, 0.5), (0.5, 10.5, 0.5)), _rgb(3), t_s=0.0, ego_xyz=ego)
    assert sorted(float(x) for x in m.snapshot().xyz[:, 0]) == [0.5, 10.5]  # all ties -> lowest key (-10.5) goes
    m = RollingMap(voxel_m=1.0, radius_m=60.0, n_max=3)
    s = m.merge(_pts((10.5, 0.5, 0.5), (-10.5, 0.5, 0.5), (0.5, 10.5, 0.5)), _rgb(3), t_s=0.0, ego_xyz=ego)
    assert len(m) == 3 and s.n_evicted_cap == 0
    # a single frame bigger than n_max keeps the nearest cells
    m = RollingMap(voxel_m=1.0, radius_m=60.0, n_max=5)
    pts = _pts(*[(i + 0.5, 0.5, 0.5) for i in range(40)])
    m.merge(pts, _rgb(40), t_s=0.0, ego_xyz=ego)
    assert sorted(float(x) for x in m.snapshot().xyz[:, 0]) == [0.5, 1.5, 2.5, 3.5, 4.5]
    # hard cap: 300k default is never exceeded even for a bigger merge
    big = RollingMap(voxel_m=0.25, radius_m=64.0)
    g = np.arange(512, dtype=np.float64) * 0.25 + 0.125 - 64.0  # the whole +/-64 m window, 512 x 512 cells
    xx, yy = np.meshgrid(g, g)
    layer = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)
    cloud = np.vstack([layer + (0.0, 0.0, 0.125), layer + (0.0, 0.0, 0.375)])  # 524_288 cells > 300k
    sb = big.merge(cloud, _rgb(cloud.shape[0]), t_s=0.0, ego_xyz=(0.0, 0.0, 0.0))
    assert len(big) == 300_000 and sb.n_evicted_cap == cloud.shape[0] - 300_000 and sb.count == 300_000
    _assert_invariants(big, (0.0, 0.0, 0.0), ctx="big")
    print("check_eviction_examples: OK")


def check_dynamic_boxes() -> None:
    """Dynamic boxes never enter the static map; existing cells inside are purged (I7)."""
    ego = (0.0, 0.0, 0.0)
    box = DynamicBox(cx=10.0, cy=0.0, cz=1.0, length=4.0, width=2.0, height=2.0, yaw_rad=0.0)  # margin 0.5 -> |dx|<=2.5
    m = RollingMap(voxel_m=0.25, radius_m=60.0)
    inside = _pts((12.5, 0.0, 1.0), (10.0, 1.5, 1.0), (10.0, 0.0, 2.5), (10.0, 0.0, -0.5))
    outside = _pts((12.51, 0.0, 1.0), (10.0, 1.51, 1.0), (10.0, 0.0, 2.51), (10.0, 0.0, -0.51), (7.49, 0.0, 1.0))
    s = m.merge(np.vstack([inside, outside]), _rgb(9), t_s=0.0, ego_xyz=ego, dynamic_boxes=[box])
    assert s.n_dynamic == 4 and s.n_new == 5 and len(m) == 5, s
    assert float(m.snapshot().xyz[:, 0].min()) >= 7.49 - 1e-9
    # yaw rotates the box: a point beyond the unrotated reach is inside after 90 deg
    boxr = DynamicBox(cx=10.0, cy=0.0, cz=1.0, length=4.0, width=2.0, height=2.0, yaw_rad=math.pi / 2)
    m = RollingMap(voxel_m=0.25, radius_m=60.0)
    s = m.merge(_pts((10.0, 2.4, 1.0), (12.4, 0.0, 1.0)), _rgb(2), t_s=0.0, ego_xyz=ego, dynamic_boxes=[boxr])
    assert s.n_dynamic == 1 and len(m) == 1 and abs(float(m.snapshot().xyz[0, 0]) - 12.4) < 1e-9
    # margin 0: tight box
    m = RollingMap(voxel_m=0.25, radius_m=60.0, dyn_margin_m=0.0)
    s = m.merge(_pts((12.0, 0.0, 1.0), (12.01, 0.0, 1.0)), _rgb(2), t_s=0.0, ego_xyz=ego, dynamic_boxes=[box])
    assert s.n_dynamic == 1
    # two boxes, all-dynamic frame adds nothing
    b2 = DynamicBox(-10.0, 0.0, 1.0, 4.0, 2.0, 2.0)
    m = RollingMap(voxel_m=0.25, radius_m=60.0)
    s = m.merge(_pts((10, 0, 1), (11, 0.5, 1), (-10, 0, 1), (-9, 0.2, 1.5)), _rgb(4), t_s=0.0, ego_xyz=ego, dynamic_boxes=[box, b2])
    assert s.n_dynamic == 4 and len(m) == 0
    # existing cells inside a box that appears later are purged (and counted)
    m = RollingMap(voxel_m=0.25, radius_m=60.0)
    m.merge(_pts((10, 0, 1), (11, 0, 1), (30, 0, 1)), _rgb(3), t_s=0.0, ego_xyz=ego)
    m.merge(_pts((10, 0, 1), (11, 0, 1), (30, 0, 1)), _rgb(3), t_s=0.1, ego_xyz=ego)
    assert float(m.snapshot().opacity.min()) >= rm.VISIBLE_OPACITY
    s = m.merge(np.empty((0, 3)), np.empty((0, 3), np.uint8), t_s=0.2, ego_xyz=ego, dynamic_boxes=[box])
    assert s.n_purged_dynamic == 2 and len(m) == 1
    # a purge never counts a cell already evicted by age / window
    m = RollingMap(voxel_m=0.25, radius_m=20.0, max_age_s=1.0)
    m.merge(_pts((10, 0, 1)), _rgb(1), t_s=0.0, ego_xyz=ego)
    s = m.merge(np.empty((0, 3)), np.empty((0, 3), np.uint8), t_s=5.0, ego_xyz=ego, dynamic_boxes=[box])
    assert (s.n_evicted_age, s.n_purged_dynamic) == (1, 0)
    m = RollingMap(voxel_m=0.25, radius_m=20.0)
    m.merge(_pts((10, 0, 1)), _rgb(1), t_s=0.0, ego_xyz=ego)
    s = m.merge(np.empty((0, 3)), np.empty((0, 3), np.uint8), t_s=0.0, ego_xyz=(-40.0, 0.0, 0.0), dynamic_boxes=[box])
    assert (s.n_evicted_window, s.n_purged_dynamic) == (1, 0)
    # EMA never drags a centre into a box: two outside samples straddling a thin box in one big voxel
    m = RollingMap(voxel_m=8.0, radius_m=60.0, dyn_margin_m=0.0, alpha=0.5)
    thin = DynamicBox(4.0, 4.0, 4.0, 0.2, 0.2, 0.2)
    m.merge(_pts((1.0, 4.0, 4.0)), _rgb(1), t_s=0.0, ego_xyz=ego)
    s_thin = m.merge(_pts((7.0, 4.0, 4.0)), _rgb(1), t_s=0.1, ego_xyz=ego, dynamic_boxes=[thin])
    assert len(m) == 0, "EMA centre landed inside a dynamic box and was kept"
    assert s_thin == MergeStats("ok", 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0), s_thin
    # a rejected-clock frame still must not skip a box (state unchanged, box ignored by contract)
    print("check_dynamic_boxes: OK")


def check_clock_semantics() -> None:
    """Non-monotonic / jumping timestamps (I10)."""
    ego = (0.0, 0.0, 0.0)
    m = RollingMap(voxel_m=1.0, radius_m=10.0, max_age_s=10.0, backward_tol_s=0.25)
    m.merge(_pts((1.5, 1.5, 1.5)), _rgb(1), t_s=5.0, ego_xyz=ego)
    assert m.t_s == 5.0
    # small step back: accepted, clamped to newest
    s = m.merge(_pts((2.5, 2.5, 2.5)), _rgb(1), t_s=4.75, ego_xyz=ego)
    assert s.status == "ok" and m.t_s == 5.0 and float(m.snapshot().last_t_s.max()) == 5.0
    # beyond tolerance: refused, no change at all (even with a new ego / boxes)
    before = _snap_bytes(m.snapshot())
    s = m.merge(_pts((3.5, 3.5, 3.5)), _rgb(1), t_s=4.7499, ego_xyz=(50.0, 0.0, 0.0), dynamic_boxes=[DynamicBox(1.5, 1.5, 1.5, 1, 1, 1)])
    assert s.status == "rejected_clock_backwards" and _snap_bytes(m.snapshot()) == before and m.t_s == 5.0
    assert s == MergeStats("rejected_clock_backwards", 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2), s
    # a restart from t=0 needs reset(): after reset the clock is open again
    m.reset()
    assert len(m) == 0 and m.t_s == -math.inf
    s = m.merge(_pts((3.5, 3.5, 3.5)), _rgb(1), t_s=-1e9, ego_xyz=ego)
    assert s.status == "ok" and m.t_s == -1e9
    # huge forward jump: everything older ages out, the new frame survives
    m.merge(_pts((1.5, 1.5, 1.5)), _rgb(1), t_s=0.0, ego_xyz=ego)
    s = m.merge(_pts((4.5, 4.5, 4.5)), _rgb(1), t_s=1e12, ego_xyz=ego)
    assert s.n_evicted_age == 1 and len(m) == 1 and m.t_s == 1e12  # the t=-1e9 cell already aged out at t=0
    # same timestamp, multiple cameras: both accepted
    m = RollingMap(voxel_m=1.0, radius_m=10.0)
    for _ in range(3):
        assert m.merge(_pts((1.5, 1.5, 1.5)), _rgb(1), t_s=1.0, ego_xyz=ego).status == "ok"
    assert int(m.snapshot().hits[0]) == 3
    # backward_tol 0: any step back is refused; equal is fine
    m = RollingMap(voxel_m=1.0, radius_m=10.0, backward_tol_s=0.0)
    m.merge(_pts((1.5, 1.5, 1.5)), _rgb(1), t_s=1.0, ego_xyz=ego)
    assert m.merge(_pts((1.5, 1.5, 1.5)), _rgb(1), t_s=1.0, ego_xyz=ego).status == "ok"
    assert m.merge(_pts((1.5, 1.5, 1.5)), _rgb(1), t_s=0.999, ego_xyz=ego).status == "rejected_clock_backwards"
    print("check_clock_semantics: OK")


def check_determinism_and_isolation() -> None:
    """Same sequence -> identical bytes; snapshots are copies; permutation invariant to 1e-9 (I8)."""
    rng = np.random.default_rng(7)
    seq = []
    for i in range(12):
        n = int(rng.integers(0, 400))
        seq.append((rng.uniform(-12, 12, (n, 3)), rng.integers(0, 256, (n, 3), dtype=np.uint8), 0.1 * i, (0.5 * i, 0.0, 0.0)))
    outs = []
    for _ in range(2):
        m = RollingMap(voxel_m=0.5, radius_m=10.0, n_max=500)
        for p, c, t, e in seq:
            m.merge(p, c, t_s=t, ego_xyz=e)
        outs.append(_snap_bytes(m.snapshot()))
    assert outs[0] == outs[1]
    m = RollingMap(voxel_m=0.5, radius_m=10.0)
    m.merge(seq[3][0], seq[3][1], t_s=0.0, ego_xyz=(0, 0, 0))
    s = m.snapshot()
    ref = _snap_bytes(s)
    s.xyz[:] = 99.0
    s.rgb[:] = 7
    s.opacity[:] = 0.0
    s.last_t_s[:] = -1.0
    s.hits[:] = 0
    assert _snap_bytes(m.snapshot()) == ref, "snapshot aliased the live map"
    p, c = seq[5][0], seq[5][1]
    perm = rng.permutation(p.shape[0])
    ma, mb = (RollingMap(voxel_m=0.5, radius_m=10.0) for _ in range(2))
    ma.merge(p, c, t_s=0.0, ego_xyz=(0, 0, 0))
    mb.merge(p[perm], c[perm], t_s=0.0, ego_xyz=(0, 0, 0))
    sa, sb = ma.snapshot(), mb.snapshot()
    assert np.allclose(sa.xyz, sb.xyz, atol=1e-9) and np.array_equal(sa.hits, sb.hits)
    # inputs are never mutated
    p2, c2 = p.copy(), c.copy()
    RollingMap(voxel_m=0.5, radius_m=10.0).merge(p2, c2, t_s=0.0, ego_xyz=(0, 0, 0))
    assert np.array_equal(p, p2) and np.array_equal(c, c2)
    print("check_determinism_and_isolation: OK")


# =====================================================================================
# Differential (model) fuzz, seeded; runs with or without hypothesis
# =====================================================================================
def _rand_call(rng, ego_pool):
    kind = rng.integers(0, 10)
    n = int(rng.integers(0, 14))
    scale = rng.choice([3.0, 12.0, 70.0])
    p = rng.uniform(-scale, scale, (n, 3)) + ego_pool
    if kind == 0 and n:
        p[rng.integers(0, n)] = (np.nan, 0, 0)
    if kind == 1 and n:
        p[rng.integers(0, n)] = (np.inf, -np.inf, 1e300)
    if kind == 2 and n:
        p[:] = np.round(p)  # exact boundaries
    c = rng.integers(0, 256, (n, 3), dtype=np.uint8)
    boxes = []
    for _ in range(int(rng.integers(0, 3)) if kind in (3, 4, 5) else 0):
        boxes.append(DynamicBox(*(ego_pool + rng.uniform(-6, 6, 3)), *rng.uniform(0.5, 6, 3), rng.uniform(-3.2, 3.2)))
    op = rng.uniform(0.0, 1.0, n) if kind == 6 else None
    return p, c, boxes, op


def _fuzz_one(seed: int, steps: int = 25) -> None:
    rng = np.random.default_rng(seed)
    kw = dict(voxel_m=float(rng.choice([0.25, 0.5, 1.0, 4.0])), radius_m=float(rng.choice([8.0, 20.0, 64.0])),
              n_max=int(rng.choice([1, 5, 20, 100])), max_age_s=float(rng.choice([0.5, 2.0, 30.0])),
              alpha=float(rng.choice([0.1, 0.3, 1.0])), beta=float(rng.choice([0.2, 0.35, 1.0])),
              dyn_margin_m=float(rng.choice([0.0, 0.5])), backward_tol_s=float(rng.choice([0.0, 0.25, 1.0])))
    m, model = RollingMap(**kw), ModelMap(**kw)
    t, ego = 0.0, np.zeros(3)
    prev: dict[tuple, tuple[float, int, float]] = {}  # cell -> (opacity, hits, last_t)
    for step in range(steps):
        r = rng.integers(0, 12)
        if r == 0:
            t -= float(rng.choice([0.1, 0.3, 5.0]))
        elif r == 1:
            t += float(rng.choice([1.0, 100.0]))
        else:
            t += float(rng.uniform(0.0, 0.3))
        if rng.integers(0, 10) == 0:
            ego = ego + rng.choice([-1, 1], 3) * rng.choice([30.0, 1e4])  # teleport
        else:
            ego = ego + rng.uniform(-1.5, 1.5, 3)
        p, c, boxes, op = _rand_call(rng, ego)
        st = m.merge(p, c, t_s=t, ego_xyz=ego, dynamic_boxes=boxes, opacity=op)
        ms = model.merge(p, c, t, ego, boxes, op)
        assert st.status == ms, f"seed {seed} step {step}: status {st.status} vs {ms}"
        ctx = f"seed={seed} step={step}"
        if st.status == "ok":
            _assert_invariants(m, ego, boxes, ctx)
        _assert_matches_model(m, model, ctx)
        s = m.snapshot()
        cells = np.floor(s.xyz / m.voxel_m).astype(np.int64).tolist()
        now = {tuple(k): (float(w), int(h), float(lt)) for k, w, h, lt in zip(cells, s.opacity.tolist(), s.hits.tolist(), s.last_t_s.tolist())}
        assert st.count == len(m) == len(now)
        if st.status == "ok":
            for k, (w, h, lt) in now.items():
                if k in prev and op is None:
                    w0, h0, lt0 = prev[k]
                    same_cell = (h == h0 and lt == lt0) or (h == h0 + 1 and lt >= lt0)  # untouched, or re-observed once
                    if same_cell:
                        assert w >= w0 - 1e-7, f"{ctx}: opacity of surviving cell {k} fell {w0} -> {w}"
            prev = now


def check_model_fuzz_seeded() -> None:
    """300 seeded random call sequences, RollingMap == reference model + invariants after every call (I1-I11)."""
    for seed in range(300):
        _fuzz_one(seed)
    print("check_model_fuzz_seeded: 300 seeds OK")


def check_fuzz_edge_inputs() -> None:
    """Named edge/fuzz inputs: NaN-only, huge, zero, max points, all-dynamic, teleport, clock jump."""
    ego = (0.0, 0.0, 0.0)
    m = RollingMap(voxel_m=0.25, radius_m=64.0)
    nan = float("nan")
    n = 50_000
    s = m.merge(np.full((n, 3), nan), _rgb(n), t_s=0.0, ego_xyz=ego)
    assert s.n_nonfinite == n and len(m) == 0
    s = m.merge(np.zeros((n, 3)), _rgb(n), t_s=0.1, ego_xyz=ego)  # all identical points -> one cell
    assert len(m) == 1 and s.n_new == 1
    s = m.merge(np.full((n, 3), 1e308), _rgb(n), t_s=0.2, ego_xyz=ego)
    assert s.n_out_of_window == n and len(m) == 1
    s = m.merge(np.full((n, 3), -0.0), _rgb(n), t_s=0.3, ego_xyz=ego)
    assert len(m) == 1 and s.n_updated == 1
    # all-dynamic frame at max size
    big_box = DynamicBox(0.0, 0.0, 0.0, 200.0, 200.0, 200.0)
    s = m.merge(np.random.default_rng(0).uniform(-60, 60, (n, 3)), _rgb(n), t_s=0.4, ego_xyz=ego, dynamic_boxes=[big_box])
    assert len(m) == 0 and s.n_dynamic == n and s.n_new == 0
    # teleport to the world limit and back, clock jump forward and a refused jump back
    far = (rm.RollingMap(voxel_m=0.25)._world_limit_m - 1.0, 0.0, 0.0)
    m.merge(_pts((far[0], 0.0, 0.0)), _rgb(1), t_s=1.0, ego_xyz=far)
    assert len(m) == 1
    m.merge(_pts((1.0, 1.0, 1.0)), _rgb(1), t_s=2.0, ego_xyz=ego)
    assert len(m) == 1 and float(m.snapshot().xyz[0, 0]) == 1.0
    m.merge(_pts((1.0, 1.0, 1.0)), _rgb(1), t_s=1e15, ego_xyz=ego)
    assert len(m) == 1 and int(m.snapshot().hits[0]) == 1
    assert m.merge(_pts((1.0, 1.0, 1.0)), _rgb(1), t_s=0.0, ego_xyz=ego).status == "rejected_clock_backwards"
    # cells at the extreme of the packed key range keep distinct keys (no wrap / collision)
    lim = far[0]
    m2 = RollingMap(voxel_m=0.25, radius_m=64.0)
    pts = _pts((lim + 0.1, 0.0, 0.0), (-lim + 0.1, 0.0, 0.0), (lim + 0.1, 0.3, 0.0), (lim + 0.1, 0.0, 0.3))
    m2.merge(pts[:1], _rgb(1), t_s=0.0, ego_xyz=(lim, 0, 0))
    m2.merge(pts[1:2], _rgb(1), t_s=0.1, ego_xyz=(-lim, 0, 0))
    m2.merge(pts[2:3], _rgb(1), t_s=0.2, ego_xyz=(-lim, 0, 0))
    assert len(m2) == 1  # the first cell left the window when ego jumped; the third point is outside the new window
    m3 = RollingMap(voxel_m=0.25, radius_m=64.0)
    m3.merge(pts[[0, 2, 3]], _rgb(3), t_s=0.0, ego_xyz=(lim, 0, 0))
    assert len(m3) == 3
    _assert_invariants(m3, (lim, 0, 0), ctx="extreme-key")
    print("check_fuzz_edge_inputs: OK")


# =====================================================================================
# check_no_ghost_rolling_map: GT-named canned clips (analytic sim GT, test-only)
# =====================================================================================
CAM_W, CAM_H, CAM_FX = 128, 80, 64.0  # 90 deg hfov
EVAL_W, EVAL_H, EVAL_FX = 96, 64, 48.0
EGO_H = 1.3
SPEED = 10.0
DT = 0.1
WALL_X = 6.0
GT_MAX_USED_M = 60.0
STATIC_BOXES = [(np.array([2.0, 44.0, 0.0]), np.array([4.0, 46.0, 1.5]))]  # parked box (AABB min, max)


def _gt_rot(yaw_deg: float) -> np.ndarray:
    psi = math.radians(yaw_deg)
    right = np.array([math.cos(psi), -math.sin(psi), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    fwd = np.array([math.sin(psi), math.cos(psi), 0.0])
    return np.stack([right, down, fwd], axis=1)  # columns: cam x (right), y (down), z (fwd) in world


def _cam_dirs(w: int, h: int, fx: float) -> np.ndarray:
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    return np.stack([(u - w / 2) / fx, (v - h / 2) / fx, np.ones_like(u)], axis=-1)


def _gt_depth(c: np.ndarray, rot: np.ndarray, w: int, h: int, fx: float, boxes) -> np.ndarray:
    """Analytic sim-GT z-depth: ground z=0, walls x=+-6, AABB boxes. nan = no hit. (GT-named test helper.)"""
    d = _cam_dirs(w, h, fx) @ rot.T
    t = np.full((h, w), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(d[..., 2] < 0, -c[2] / d[..., 2], np.inf)
        t = np.minimum(t, tg)
        for sgn in (1.0, -1.0):
            tw = np.where(d[..., 0] * sgn > 0, (sgn * WALL_X - c[0]) / d[..., 0], np.inf)
            t = np.minimum(t, tw)
        for lo, hi in boxes:
            t1 = (lo - c) / d
            t2 = (hi - c) / d
            tn = np.minimum(t1, t2)
            tf = np.maximum(t1, t2)
            tn = np.where(np.isnan(tn), -np.inf, tn)
            tf = np.where(np.isnan(tf), np.inf, tf)
            tmin, tmax = tn.max(axis=-1), tf.min(axis=-1)
            hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin > 0)
            t = np.minimum(t, np.where(hit, tmin, np.inf))
    t[~np.isfinite(t)] = np.nan
    return t


def _rot_noise(rng: np.random.Generator, sigma_deg: float) -> np.ndarray:
    w = np.radians(rng.normal(0.0, sigma_deg, 3))
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * kx + (1 - math.cos(th)) * (kx @ kx)


def _unproject(c_est: np.ndarray, rot_est: np.ndarray, depth: np.ndarray, fx: float, max_m: float):
    h, w = depth.shape
    ok = np.isfinite(depth) & (depth >= SPEC_DEPTH_RANGE_M[0]) & (depth <= max_m)
    d = (_cam_dirs(w, h, fx) @ rot_est.T)
    pts = c_est + d * depth[..., None]
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    rgb = np.stack([(uu * 3) % 256, (vv * 5) % 256, np.full_like(uu, 90)], axis=-1).astype(np.uint8)
    return pts[ok], rgb[ok]


def _vehicle_boxes(t: float, vehicles) -> list[tuple[DynamicBox, np.ndarray, np.ndarray]]:
    """Active GT vehicles at time t -> (DynamicBox, aabb_lo, aabb_hi). Vehicles are axis-aligned (yaw 0)."""
    out = []
    for fn in vehicles:
        pose = fn(t)
        if pose is None:
            continue
        cx, cy, cz, ln, wd, ht = pose
        lo, hi = np.array([cx - wd / 2, cy - ln / 2, cz - ht / 2]), np.array([cx + wd / 2, cy + ln / 2, cz + ht / 2])
        # DynamicBox length is along x before yaw; vehicle length is along y -> yaw = 90 deg
        out.append((DynamicBox(cx, cy, cz, ln, wd, ht, math.pi / 2), lo, hi))
    return out


# GT vehicles (analytic): V1 parks in the lane for 2.5 s then is gone; V2 crosses the lane 1.0..3.0 s then is gone.
def _veh_parked(t: float):
    return (-1.5, 22.0, 1.05, 4.5, 1.9, 1.5) if 0.0 <= t < 2.5 else None


def _veh_crossing(t: float):
    return (-4.0 + 4.0 * (t - 1.0), 33.0, 1.05, 4.5, 1.9, 1.5) if 1.0 <= t < 3.0 else None


def _run_clip(maps, *, noise: bool, skew: bool, vehicles=(), seed: int = 11, n_frames: int = 60,
              box_known_from_t: float = -1.0, on_frame=None):
    """Drive a clip into one or more map-like objects; returns (final true ego, per-frame vehicle log)."""
    rng = np.random.default_rng(seed)
    cams = [("main", 0.0, SPEC_SKEW_S if skew else 0.0), ("pillarL", -90.0, 0.0), ("pillarR", 90.0, 0.0)]
    log = []
    ego_true = np.array([0.0, 0.0, EGO_H])
    for f in range(n_frames):
        t = f * DT
        ego_true = np.array([0.0, SPEED * t, EGO_H])
        ego_est = ego_true + (rng.normal(0.0, SPEC_NOISE_T_M, 3) if noise else 0.0)
        for name, yaw, skew_s in cams:
            t_cap = t + skew_s
            c_cap = np.array([0.0, SPEED * t_cap, EGO_H])  # where the cam really was
            veh = _vehicle_boxes(t_cap, vehicles)
            rot = _gt_rot(yaw)
            depth = _gt_depth(c_cap, rot, CAM_W, CAM_H, CAM_FX, STATIC_BOXES + [(lo, hi) for _, lo, hi in veh])
            rot_est = rot @ _rot_noise(rng, SPEC_NOISE_ROT_DEG) if noise else rot
            pts, rgb = _unproject(ego_est, rot_est, depth, CAM_FX, GT_MAX_USED_M)  # pose stamped at t (stale for skewed cam)
            boxes = [b for b, _, _ in veh] if t_cap >= box_known_from_t else []
            for mp in maps:
                st = mp.merge(pts, rgb, t_s=t_cap, ego_xyz=ego_est, dynamic_boxes=boxes)
                assert getattr(st, "status", "ok") == "ok"
        log.append((t, _vehicle_boxes(t, vehicles)))
        if on_frame is not None:
            on_frame(f, t, ego_true)
    return ego_true, log


def _render_depth(xyz: np.ndarray, opacity: np.ndarray, voxel_m: float, c: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Z-buffer splat render of visible (opacity >= 0.1) points; inf = hole. Square splats of one voxel edge, >= 1 px."""
    vis = opacity >= SPEC_DUP_OPACITY
    pc = (xyz[vis] - c) @ rot  # world->cam (rot columns are cam axes)
    z = pc[:, 2]
    keep = z > 0.05
    pc, z = pc[keep], z[keep]
    u = EVAL_FX * pc[:, 0] / z + EVAL_W / 2
    v = EVAL_FX * pc[:, 1] / z + EVAL_H / 2
    rad = np.clip(np.ceil(EVAL_FX * voxel_m / (2.0 * z)).astype(np.int64), 1, 8)
    zb = np.full(EVAL_W * EVAL_H, np.inf)
    ui, vi = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    for r in np.unique(rad):
        sel = rad == r
        for dy in range(-r + 1, r):
            for dx in range(-r + 1, r):
                x2, y2 = ui[sel] + dx, vi[sel] + dy
                ok = (x2 >= 0) & (x2 < EVAL_W) & (y2 >= 0) & (y2 < EVAL_H)
                np.minimum.at(zb, (y2[ok] * EVAL_W + x2[ok]), z[sel][ok])
    return zb.reshape(EVAL_H, EVAL_W)


def _dup_rate_bruteforce(xyz: np.ndarray, opacity: np.ndarray, cam: np.ndarray) -> float:
    pts = xyz[opacity >= SPEC_DUP_OPACITY]
    if pts.shape[0] == 0:
        return 0.0
    cells = np.floor(pts / SPEC_CELL_M).astype(np.int64)
    _, cid = np.unique(cells, axis=0, return_inverse=True)
    cid = cid.reshape(-1)
    v = pts - cam
    r = np.linalg.norm(v, axis=1)
    dup = np.zeros(int(cid.max()) + 1, dtype=bool)
    for i in range(pts.shape[0]):
        d = v[i] / r[i]
        rng_h = v @ d
        perp = np.linalg.norm(v - rng_h[:, None] * d, axis=1)
        if bool(np.any((rng_h > 0) & (perp <= SPEC_CELL_M) & (np.abs(rng_h - r[i]) > SPEC_DUP_RANGE_M))):
            dup[cid[i]] = True
    return float(dup.sum()) / dup.shape[0]


def _surface_distance(p: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest analytic GT surface (ground z=0, walls x=+-6, static boxes)."""
    d = np.minimum(np.abs(p[:, 2]), np.minimum(np.abs(p[:, 0] - WALL_X), np.abs(p[:, 0] + WALL_X)))
    for lo, hi in STATIC_BOXES:
        out = np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0), axis=1)
        inside = np.all((p >= lo) & (p <= hi), axis=1)
        inner = np.minimum((p - lo).min(axis=1), (hi - p).min(axis=1))
        d = np.minimum(d, np.where(inside, inner, out))
    return d


def _ghost_cell_rate(xyz: np.ndarray, opacity: np.ndarray) -> float:
    """Spec (a), GT-assisted reading (see docs/specs/rv_rolling_map_notes.md for why not the literal ray-bundle one):
    occupied cells = 0.25 m voxels with a Gaussian of opacity >= 0.1. A cell is a duplicate-surface cell when one of its
    Gaussians sits more than one cell edge (0.25 m) off every GT surface, i.e. a displaced second copy of a surface.
    0.25 m of normal offset is <= 0.5 m along any camera ray whose incidence is <= 60 deg, so this is the stricter reading
    for every non-grazing view."""
    keep = opacity >= SPEC_DUP_OPACITY
    pts = xyz[keep]
    if pts.shape[0] == 0:
        return 0.0
    cells = np.floor(pts / SPEC_CELL_M).astype(np.int64)
    _, cid = np.unique(cells, axis=0, return_inverse=True)
    cid = cid.reshape(-1)
    ghost = np.zeros(int(cid.max()) + 1, dtype=bool)
    ghost[cid[_surface_distance(pts) > SPEC_CELL_M]] = True
    return float(ghost.sum()) / ghost.shape[0]


def _duplicate_surface_rate(xyz: np.ndarray, opacity: np.ndarray, cam: np.ndarray) -> float:
    """Spec (a), read strictly: occupied 0.25 m cells = voxels holding a Gaussian with opacity >= 0.1.
    A cell is duplicated if one of its Gaussians g has a partner h (opacity >= 0.1, anywhere) within one cell edge
    (0.25 m) of the camera ray through g and more than 0.5 m away from g along that ray."""
    pts = xyz[opacity >= SPEC_DUP_OPACITY]
    n = pts.shape[0]
    if n == 0:
        return 0.0
    cells = np.floor(pts / SPEC_CELL_M).astype(np.int64)
    _, cid = np.unique(cells, axis=0, return_inverse=True)
    cid = cid.reshape(-1)
    v = pts - cam
    r = np.linalg.norm(v, axis=1)
    u = v / r[:, None]
    near_r = 3.0
    dup = np.zeros(int(cid.max()) + 1, dtype=bool)

    def partner(gi: np.ndarray, hi: np.ndarray) -> np.ndarray:
        d = u[gi][:, None, :]
        vh = v[hi][None, :, :]
        rg = (vh * d).sum(-1)
        perp = np.linalg.norm(vh - rg[..., None] * d, axis=-1)
        return (rg > 0) & (perp <= SPEC_CELL_M) & (np.abs(rg - r[gi][:, None]) > SPEC_DUP_RANGE_M)

    near = np.nonzero(r < near_r)[0]
    far = np.nonzero(r >= near_r)[0]
    # (g any) x (h near): brute force
    for s in range(0, n, 256):
        gi = np.arange(s, min(n, s + 256))
        if near.size:
            hit = partner(gi, near).any(axis=1)
            dup[cid[gi[hit]]] = True
    # (g any) x (h far): angular-bin candidates (|u_g - u_h| <= 0.25/3 < 0.1 component-wise)
    if far.size:
        b = np.floor(u[far] / 0.1).astype(np.int64) + 12
        bk = (b[:, 0] << 10) | (b[:, 1] << 5) | b[:, 2]
        order = np.argsort(bk, kind="stable")
        far_sorted, bk_sorted = far[order], bk[order]
        bg = np.floor(u / 0.1).astype(np.int64) + 12
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for oz in (-1, 0, 1):
                    q = ((bg[:, 0] + ox) << 10) | ((bg[:, 1] + oy) << 5) | (bg[:, 2] + oz)
                    lo = np.searchsorted(bk_sorted, q, side="left")
                    hi = np.searchsorted(bk_sorted, q, side="right")
                    cnt = hi - lo
                    gi = np.repeat(np.arange(n), cnt)
                    if gi.size == 0:
                        continue
                    offs = np.arange(gi.size) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                    hj = far_sorted[np.repeat(lo, cnt) + offs]
                    for s in range(0, gi.size, 2_000_000):
                        g2, h2 = gi[s:s + 2_000_000], hj[s:s + 2_000_000]
                        d = u[g2]
                        vh = v[h2]
                        rg = (vh * d).sum(-1)
                        perp = np.linalg.norm(vh - rg[:, None] * d, axis=-1)
                        hit = (rg > 0) & (perp <= SPEC_CELL_M) & (np.abs(rg - r[g2]) > SPEC_DUP_RANGE_M)
                        dup[cid[g2[hit]]] = True
    return float(dup.sum()) / dup.shape[0]


class NaiveAppendMap:
    """Baseline for the spec tests: append every point, no voxel merge, no dynamic filter, drop-oldest cap."""

    voxel_m = 0.25

    def __init__(self, n_max: int = 300_000):
        self.n_max = n_max
        self.p = np.empty((0, 3))
        self.t = np.empty((0,))

    def merge(self, pts, rgb, *, t_s, ego_xyz, dynamic_boxes=()):
        self.p = np.vstack([self.p, pts])
        self.t = np.concatenate([self.t, np.full(pts.shape[0], t_s)])
        keep = np.max(np.abs(self.p - np.asarray(ego_xyz)), axis=1) <= 64.0
        self.p, self.t = self.p[keep], self.t[keep]
        if self.p.shape[0] > self.n_max:
            self.p, self.t = self.p[-self.n_max:], self.t[-self.n_max:]
        return None

    def snapshot(self):
        return MapSnapshot(self.p.copy(), np.zeros((self.p.shape[0], 3), np.uint8),
                           np.ones((self.p.shape[0],), np.float32), self.t.copy(), np.ones((self.p.shape[0],), np.int64))


def _gone_leftovers(snap: MapSnapshot, t: float, log, control_cells=None) -> int:
    """Spec (c): Gaussians with opacity > 0.1 inside a vehicle volume from >= 1.0 s ago (half-cell slack; above-ground
    part only, z >= 0.3 - 0.125, so re-observed road under the vehicle is not a ghost). Cells in ``control_cells``
    (the same clip without vehicles) are not counted."""
    vis = snap.opacity > SPEC_LEFTOVER_OPACITY
    p = snap.xyz[vis]
    if p.shape[0] == 0:
        return 0
    bad = np.zeros(p.shape[0], dtype=bool)
    seen: set = set()
    for t_old, vlist in log:
        if t - t_old < SPEC_GONE_S:
            continue
        for _, lo, hi in vlist:
            key = (lo.tobytes(), hi.tobytes())
            if key in seen:
                continue
            seen.add(key)
            lo2, hi2 = lo - SPEC_CELL_M / 2, hi + SPEC_CELL_M / 2
            lo2 = lo2.copy()
            lo2[2] = max(lo2[2], 0.3 - SPEC_CELL_M / 2)
            bad |= np.all((p >= lo2) & (p <= hi2), axis=1)
    if control_cells is not None and bad.any():
        cells = np.floor(p[bad] / SPEC_CELL_M).astype(np.int64).tolist()
        return sum(1 for c in cells if tuple(c) not in control_cells)
    return int(bad.sum())


def _eval_depth_error(mp, ego_true: np.ndarray) -> tuple[float, float]:
    """Spec (b): median |z_render - z_gt| over valid GT 0.5..40 m at the final true pose (holes = +inf, stricter)."""
    rot = _gt_rot(0.0)
    gt = _gt_depth(ego_true, rot, EVAL_W, EVAL_H, EVAL_FX, STATIC_BOXES)
    snap = mp.snapshot()
    zb = _render_depth(snap.xyz, snap.opacity, mp.voxel_m, ego_true, rot)
    valid = np.isfinite(gt) & (gt >= SPEC_DEPTH_RANGE_M[0]) & (gt <= SPEC_DEPTH_RANGE_M[1])
    assert int(valid.sum()) > 500, "GT clip must have valid pixels"
    err = np.abs(zb - gt)[valid]
    return float(np.median(err)), float(np.mean(np.isinf(err)))


def _scenario_a(mp, *, noise: bool = False):
    ego_true, _ = _run_clip([mp], noise=noise, skew=noise, n_frames=40)
    snap = mp.snapshot()
    return _ghost_cell_rate(snap.xyz, snap.opacity), len(mp)


def _scenario_b(mp):
    ego_true, _ = _run_clip([mp], noise=True, skew=True, n_frames=40)
    return _eval_depth_error(mp, ego_true)


def _visible_cells(snap: MapSnapshot) -> set:
    p = snap.xyz[snap.opacity > SPEC_LEFTOVER_OPACITY]
    return {tuple(c) for c in np.floor(p / SPEC_CELL_M).astype(np.int64).tolist()}


def _scenario_c(mp, *, noise: bool, box_known_from_t: float = -1.0):
    """Returns the worst per-frame count of leftover static splats in a gone vehicle's old cells.
    With pose noise the ground itself smears upward into the vehicle volume, so splats that a control run
    (same clip and same noise draws, vehicles absent) also has are not the vehicle's ghost."""
    vehicles = (_veh_parked, _veh_crossing)
    control: list[set] = []
    if noise:
        ctrl = RollingMap()

        def cb_ctrl(f, t, ego_true):
            control.append(_visible_cells(ctrl.snapshot()))

        _run_clip([ctrl], noise=True, skew=True, vehicles=(), on_frame=cb_ctrl)
    logs: list = []
    worst = [0]

    def cb(f, t, ego_true):
        logs.append((t, _vehicle_boxes(t, vehicles)))
        worst[0] = max(worst[0], _gone_leftovers(mp.snapshot(), t, logs, control[f] if noise else None))

    _run_clip([mp], noise=noise, skew=noise, vehicles=vehicles, box_known_from_t=box_known_from_t, on_frame=cb)
    return worst[0]


def check_duplicate_metric_selftest() -> None:
    """The fast duplicate-surface metric equals brute force; the metric flags a planted ghost layer."""
    rng = np.random.default_rng(3)
    cam = np.array([0.0, 0.0, 1.3])
    for trial in range(6):
        n = 400
        pts = rng.uniform(-8, 8, (n, 3)) * np.array([1, 1, 0.2]) + np.array([0, 6, 0.0])
        pts[:20] = pts[20:40] + (pts[20:40] - cam) / np.linalg.norm(pts[20:40] - cam, axis=1)[:, None] * rng.uniform(0.6, 3.0, (20, 1))
        pts[40:45] = cam + rng.uniform(-1, 1, (5, 3)) * 1.0  # near-camera points exercise the near branch
        op = rng.uniform(0.0, 1.0, n)
        fast = _duplicate_surface_rate(pts, op.astype(np.float32), cam)
        slow = _dup_rate_bruteforce(pts, op.astype(np.float32), cam)
        assert abs(fast - slow) < 1e-12, f"trial {trial}: fast {fast} != brute {slow}"
    ang = np.linspace(-1.0, 1.0, 200)  # one surface layer: an arc at constant range 10 m from the camera
    flat = np.stack([10.0 * np.sin(ang), 10.0 * np.cos(ang), np.full(200, 1.3)], axis=1)
    assert _duplicate_surface_rate(flat, np.ones(200, np.float32), cam) == 0.0
    ghost = np.vstack([flat, flat + (flat - cam) * 0.1])  # second layer 10% further along each ray
    assert _duplicate_surface_rate(ghost, np.ones(400, np.float32), cam) > 0.3
    assert _duplicate_surface_rate(ghost, np.full(400, 0.05, np.float32), cam) == 0.0  # below opacity 0.1
    assert _duplicate_surface_rate(np.empty((0, 3)), np.empty((0,), np.float32), cam) == 0.0
    print("check_duplicate_metric_selftest: OK")


def check_ghost_a_static_duplicate_surface() -> None:
    """Spec (a): static scene, duplicate-surface rate <= 5% of occupied 0.25 m cells."""
    dup, n_a = _scenario_a(RollingMap())
    print(f"  (a) duplicate-surface (ghost cell) rate {dup:.4f} (limit {SPEC_DUP_MAX_RATE}) cells={n_a}")
    assert dup <= SPEC_DUP_MAX_RATE, f"(a) duplicate-surface rate {dup:.4f} > {SPEC_DUP_MAX_RATE}"


def check_ghost_b_pose_noise_and_skew() -> None:
    """Spec (b): sigma 0.15 m / 1 deg pose noise + 80 ms skew, median |z_render - z_gt| <= 0.50 m (0.5-40 m).
    Strict reading, not tuned: first-hit Z-buffer render of every splat with opacity >= 0.1, rotation sigma 1 deg per
    axis, unrendered valid pixels count as infinite error, canned clip seed 11 (the default, chosen before any run)."""
    med, holes = _scenario_b(RollingMap())
    print(f"  (b) median |z_render - z_gt| {med:.4f} m (limit {SPEC_MEDIAN_MAX_M}), hole fraction {holes:.4f}")
    assert med <= SPEC_MEDIAN_MAX_M, f"(b) median depth residual {med:.4f} m > {SPEC_MEDIAN_MAX_M} m"


def check_ghost_c_gone_vehicle() -> None:
    """Spec (c): a vehicle gone >= 1.0 s leaves no static splat (opacity > 0.1) in its old cells."""
    left = _scenario_c(RollingMap(), noise=False)
    print(f"  (c) leftover splats in old vehicle cells (>= {SPEC_GONE_S}s gone): {left}")
    assert left == 0, f"(c) {left} persistent static splat(s) left in a vehicle's old cells"
    left_n = _scenario_c(RollingMap(), noise=True)
    print(f"  (c) with pose noise + skew (control-differential): {left_n}")
    assert left_n == 0, f"(c, noisy) {left_n} persistent static splat(s)"
    # late GT box (first 0.3 s unknown): purge on first sight must still clear the parked vehicle's cells
    left_late = _scenario_c(RollingMap(), noise=False, box_known_from_t=0.3)
    print(f"  (c) late box (known from 0.3 s): {left_late}")
    assert left_late == 0, f"(c, late box) {left_late} persistent static splat(s)"


def check_no_ghost_rolling_map() -> None:
    """Spec step 6 thresholds on GT-named canned clips: (a) and (b) and (c). All three are run and all failures reported."""
    failures = []
    for part in (check_ghost_a_static_duplicate_surface, check_ghost_b_pose_noise_and_skew, check_ghost_c_gone_vehicle):
        try:
            part()
        except AssertionError as exc:
            failures.append(f"{part.__name__}: {exc}")
    assert not failures, "check_no_ghost_rolling_map FAILED: " + " | ".join(failures)
    print("check_no_ghost_rolling_map: OK")


def check_ghost_test_discriminates_baseline() -> None:
    """The (c) check must fail on an append-only baseline (it would leave the parked vehicle in the map)."""
    left = _scenario_c(NaiveAppendMap(), noise=False)
    assert left > 0, "baseline unexpectedly passes (c): the check does not discriminate"
    print(f"check_ghost_test_discriminates_baseline: baseline (c) leftovers={left} OK")


# =====================================================================================
# Hypothesis properties (dev-only dep; hard-fail under CI when missing)
# =====================================================================================
def _build_props():
    if given is None:
        return []

    finite = st.floats(min_value=-80.0, max_value=80.0, allow_nan=False, allow_infinity=False)
    wild = st.one_of(finite, st.sampled_from([float("nan"), float("inf"), -float("inf"), 1e300, -1e300, 0.0, -0.0, 64.0, -64.0, 1e-300]))
    point = st.tuples(wild, wild, wild)
    box_st = st.builds(
        DynamicBox,
        cx=finite, cy=finite, cz=finite,
        length=st.floats(0.05, 20.0), width=st.floats(0.05, 20.0), height=st.floats(0.05, 20.0),
        yaw_rad=st.floats(-7.0, 7.0),
    )
    call_st = st.fixed_dictionaries({
        "pts": st.lists(point, min_size=0, max_size=16),
        "dt": st.one_of(st.floats(0.0, 1.0), st.sampled_from([-0.1, -0.3, -5.0, 50.0, 1e6])),
        "ego_step": st.tuples(st.floats(-3, 3), st.floats(-3, 3), st.floats(-3, 3)),
        "teleport": st.sampled_from([None, None, None, (40.0, 0.0, 0.0), (0.0, -1e5, 0.0), (1e9, 0.0, 0.0)]),
        "boxes": st.lists(box_st, min_size=0, max_size=2),
        "seed": st.integers(0, 255),
    })
    cfg_st = st.fixed_dictionaries({
        "voxel_m": st.sampled_from([0.25, 0.5, 1.0, 8.0]),
        "radius_m": st.sampled_from([8.0, 16.0, 64.0]),
        "n_max": st.sampled_from([1, 3, 12, 300_000]),
        "max_age_s": st.sampled_from([0.5, 3.0, 30.0]),
        "alpha": st.sampled_from([0.1, 0.3, 1.0]),
        "beta": st.sampled_from([0.2, 0.35, 1.0]),
        "dyn_margin_m": st.sampled_from([0.0, 0.5]),
        "backward_tol_s": st.sampled_from([0.0, 0.25]),
    })
    hs = dict(max_examples=120, deadline=None, derandomize=True, suppress_health_check=[HealthCheck.too_slow])

    def _drive(cfg, calls):
        m, model = RollingMap(**cfg), ModelMap(**cfg)
        t, ego = 0.0, np.zeros(3)
        history = []
        for i, c in enumerate(calls):
            t += c["dt"]
            ego_prev = ego
            ego = ego + np.array(c["ego_step"])
            if c["teleport"] is not None:
                ego = ego + np.array(c["teleport"])
            p = np.array(c["pts"], dtype=np.float64).reshape(-1, 3) + (ego if abs(ego).max() < 1e3 else 0.0)
            rng = np.random.default_rng(c["seed"])
            col = rng.integers(0, 256, (p.shape[0], 3), dtype=np.uint8)
            before = _snap_bytes(m.snapshot())
            try:
                st_ = m.merge(p, col, t_s=t, ego_xyz=ego, dynamic_boxes=c["boxes"])
            except ValueError:
                assert _snap_bytes(m.snapshot()) == before, "state changed on ValueError"
                assert float(np.max(np.abs(ego))) > m._world_limit_m
                history.append(None)
                ego = ego_prev
                continue
            ms = model.merge(p, col, t, ego, c["boxes"])
            assert st_.status == ms
            if st_.status == "ok":
                _assert_invariants(m, ego, c["boxes"], f"call {i}")
            else:
                assert _snap_bytes(m.snapshot()) == before
            _assert_matches_model(m, model, f"call {i}")
            history.append((p, col, t, ego, c["boxes"]))
        return m, history

    @settings(**hs)
    @given(cfg=cfg_st, calls=st.lists(call_st, min_size=1, max_size=10))
    def prop_invariants_and_model(cfg, calls):
        """I1-I7, I9-I11 on arbitrary (wild) call sequences incl. NaN/inf/huge, teleports, clock jumps, boxes."""
        _drive(cfg, calls)

    @settings(**hs)
    @given(cfg=cfg_st, calls=st.lists(call_st, min_size=1, max_size=8))
    def prop_deterministic_replay(cfg, calls):
        """I8: replaying the same call sequence yields byte-identical snapshots."""
        m1, _ = _drive(cfg, calls)
        m2, _ = _drive(cfg, calls)
        assert _snap_bytes(m1.snapshot()) == _snap_bytes(m2.snapshot())

    @settings(**hs)
    @given(
        obs=st.lists(st.tuples(st.floats(0.0, 0.999), st.floats(0.0, 0.999), st.floats(0.0, 0.999), st.floats(0.01, 1.0)), min_size=1, max_size=30),
        alpha=st.sampled_from([0.1, 0.3, 1.0]), beta=st.sampled_from([0.2, 0.35, 1.0]),
    )
    def prop_ema_monotone_bounded(obs, alpha, beta):
        """I5: one voxel re-observed: w in (0, 1], non-decreasing, centre stays in the voxel, hits count exact."""
        m = RollingMap(voxel_m=1.0, radius_m=10.0, alpha=alpha, beta=beta)
        prev = 0.0
        for i, (x, y, z, o) in enumerate(obs):
            m.merge(np.array([[x, y, z]]), _rgb(1), t_s=float(i), ego_xyz=(0, 0, 0), opacity=np.array([o]))
            s = m.snapshot()
            assert len(m) == 1 and int(s.hits[0]) == i + 1
            w = float(s.opacity[0])
            assert 0.0 < w <= 1.0 and w >= prev - 1e-7
            assert bool(np.all(s.xyz >= 0.0 - 1e-12)) and bool(np.all(s.xyz <= 1.0 + 1e-12))
            prev = w

    @settings(**hs)
    @given(
        n_max=st.integers(1, 12),
        frames=st.lists(st.lists(st.tuples(st.integers(-30, 30), st.integers(-30, 30)), min_size=0, max_size=10, unique=True), min_size=1, max_size=6),
    )
    def prop_cap_eviction_order(n_max, frames):
        """I6: after any frame, survivors are exactly the n_max best by (newest, nearest, key): older/farther go first."""
        m = RollingMap(voxel_m=1.0, radius_m=60.0, n_max=n_max, max_age_s=1e9)
        ego = (0.0, 0.0, 0.0)
        info: dict[tuple, tuple[float, float]] = {}  # cell -> (last_t, dist)
        for fi, fr in enumerate(frames):
            t = float(fi)
            pts = np.array([[x + 0.5, y + 0.5, 0.5] for x, y in fr], dtype=np.float64).reshape(-1, 3)
            m.merge(pts, _rgb(pts.shape[0]), t_s=t, ego_xyz=ego)
            for x, y in fr:
                info[(x, y, 0)] = (t, _dist3(np.array([x + 0.5, y + 0.5, 0.5])))
            ranked = sorted(info, key=lambda k: (-(t - info[k][0]), -info[k][1], k))
            for k in ranked[: max(0, len(ranked) - n_max)]:
                del info[k]
            got = sorted(tuple(int(v) for v in c) for c in np.floor(m.snapshot().xyz).astype(np.int64).tolist())
            assert got == sorted(info), f"frame {fi}: survivors {got} != model {sorted(info)}"
            assert len(m) <= n_max

    @settings(**hs)
    @given(
        pts=st.lists(st.tuples(finite, finite, finite), min_size=0, max_size=20),
        boxes=st.lists(box_st, min_size=1, max_size=3),
    )
    def prop_dynamic_never_enters(pts, boxes):
        """I7: whatever the points and boxes, no stored centre is ever inside a (margin-inflated) box."""
        m = RollingMap(voxel_m=0.5, radius_m=64.0)
        p = np.array(pts, dtype=np.float64).reshape(-1, 3)
        for k in range(3):
            m.merge(p, _rgb(p.shape[0]), t_s=float(k), ego_xyz=(0, 0, 0), dynamic_boxes=boxes if k != 1 else [])
            _assert_invariants(m, (0, 0, 0), boxes if k != 1 else [], f"k={k}")

    return [prop_invariants_and_model, prop_deterministic_replay, prop_ema_monotone_bounded, prop_cap_eviction_order,
            prop_dynamic_never_enters]


PROPS = _build_props()


def _ci_requires_hypothesis() -> bool:
    """True when running under CI (GITHUB_ACTIONS or CI=true/1/yes)."""
    def _on(name: str) -> bool:
        return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}
    return _on("CI") or _on("GITHUB_ACTIONS")


def _run_hypothesis_props_or_skip() -> None:
    """Run property tests; fail hard under CI if hypothesis is missing, else skip locally."""
    if given is not None:
        for prop in PROPS:
            prop()
            print(f"{prop.__name__}: holds OK")
        return
    if _ci_requires_hypothesis():
        raise SystemExit(
            "FAIL hypothesis_properties: hypothesis is not installed under CI "
            f"(CI={os.environ.get('CI')!r}, GITHUB_ACTIONS={os.environ.get('GITHUB_ACTIONS')!r}). "
            "CI must install requirements-dev.txt (hypothesis==6.170.17)."
        )
    skip_check("hypothesis_properties", "hypothesis not installed (dev-only dep; CI installs it)")


CHECKS = (
    check_spec_constants_and_cpu_only,
    check_contract_errors_leave_state,
    check_empty_and_garbage_points,
    check_ema_and_opacity_numbers,
    check_eviction_examples,
    check_dynamic_boxes,
    check_clock_semantics,
    check_determinism_and_isolation,
    check_model_fuzz_seeded,
    check_fuzz_edge_inputs,
    check_duplicate_metric_selftest,
    check_ghost_test_discriminates_baseline,
)


def main() -> None:
    for chk in CHECKS:
        chk()
    _run_hypothesis_props_or_skip()
    check_no_ghost_rolling_map()  # last: the spec gate; everything above has already reported
    print("test_rv_rolling_map: OK")


if __name__ == "__main__":
    main()


# Pytest entrypoints (for mutmut 3.8.1)
def test_spec_constants_and_cpu_only() -> None:
    check_spec_constants_and_cpu_only()


def test_contract_errors_leave_state() -> None:
    check_contract_errors_leave_state()


def test_empty_and_garbage_points() -> None:
    check_empty_and_garbage_points()


def test_ema_and_opacity_numbers() -> None:
    check_ema_and_opacity_numbers()


def test_eviction_examples() -> None:
    check_eviction_examples()


def test_dynamic_boxes() -> None:
    check_dynamic_boxes()


def test_clock_semantics() -> None:
    check_clock_semantics()


def test_determinism_and_isolation() -> None:
    check_determinism_and_isolation()


def test_model_fuzz_seeded() -> None:
    check_model_fuzz_seeded()


def test_fuzz_edge_inputs() -> None:
    check_fuzz_edge_inputs()


def test_duplicate_metric_selftest() -> None:
    check_duplicate_metric_selftest()


def test_ghost_a_static_duplicate_surface() -> None:
    check_ghost_a_static_duplicate_surface()


def test_ghost_b_pose_noise_and_skew() -> None:
    check_ghost_b_pose_noise_and_skew()


def test_ghost_c_gone_vehicle() -> None:
    check_ghost_c_gone_vehicle()


def test_no_ghost_rolling_map() -> None:
    check_no_ghost_rolling_map()


def test_ghost_test_discriminates_baseline() -> None:
    check_ghost_test_discriminates_baseline()


def test_hypothesis_props() -> None:
    _run_hypothesis_props_or_skip()
