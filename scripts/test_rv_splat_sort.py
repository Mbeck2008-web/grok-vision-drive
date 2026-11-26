#!/usr/bin/env python3
"""Offline CPU checks for the Reality View splat depth sort (python/viz/rv_splat_sort.py).

Invariants (docs/specs/rv_splat_sort_notes.md):
I1  Output is a permutation of 0..N-1, dtype intp, input length preserved.
I2  Quantization: float64 clip to [near, far], t = (c-near)/(far-near),
    q = floor(t*65535 + 0.5) as uint16; monotone, q(near)=0, q(far)=65535.
I3  Order == stable argsort on the quantized keys (descending: larger q first, equal q by
    ascending input index; ascending: plain stable argsort). NaN last in input order in both.
I4  +/-inf clip to the end bins; empty -> empty intp; single -> [0]; all-equal -> identity.
I5  The input is never mutated; no RNG / clock; invalid near/far/dtype/shape raise ValueError.
I6  Behavior change vs float argsort(-d): the order differs only between splats whose float
    depths are in the same quantum (<= one quantum apart).
I7  Perf proxy: 300k keys, p95 <= 15 ms, gated only behind an A/A (float argsort vs float
    argsort) precondition <= 1.03 (official number is the IQU45 bench run by GVD).
"""
from __future__ import annotations

import ast
import math
import os
import sys
import time
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

from python.viz import rv_splat_sort as rs  # noqa: E402
from python.viz.rv_splat_sort import quantize_depth, quantum_m, sort_splats  # noqa: E402

NEAR, FAR = 0.05, 200.0  # bench range (rv_gl33_splat_bench: proj near 0.05, far 200)
N_PERF = 300_000
PERF_P95_MS = 15.0  # fixed target; not tuned after a run
AA_MAX = 1.03
PERF_RUNS = 50
PERF_PAIRS = 7


def _q_ref(x: float, near: float, far: float) -> int:
    """Independent scalar quantizer (pure Python float64, same IEEE ops as the spec)."""
    c = min(max(float(x), near), far)
    t = (c - near) / (far - near)
    return int(math.floor(t * 65535.0 + 0.5))


def _ref_order(depth, near, far, descending=True) -> np.ndarray:
    """Reference: stable argsort on quantized keys; NaN last in input order."""
    d = np.asarray(depth, dtype=np.float64)
    nan_idx = np.flatnonzero(np.isnan(d))
    valid = np.flatnonzero(~np.isnan(d))
    q = np.array([_q_ref(x, near, far) for x in d[valid]], dtype=np.int64)
    key = -q if descending else q
    return np.concatenate([valid[np.argsort(key, kind="stable")], nan_idx]).astype(np.intp)


def _float_ref(d: np.ndarray) -> np.ndarray:
    """Today's bench sort: np.argsort(-d, kind='stable') on float32 depth."""
    return np.argsort(-d, kind="stable")


def _assert_perm(order: np.ndarray, n: int) -> None:
    assert order.dtype == np.intp, order.dtype
    assert order.shape == (n,), order.shape
    assert np.array_equal(np.sort(order), np.arange(n))


def _raises(exc, fn, *a, **k) -> bool:
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


# ---------------------------------------------------------------- examples

def check_quantizer() -> None:
    q = quantize_depth(np.array([NEAR, FAR, (NEAR + FAR) / 2, -5.0, 1e9, -np.inf, np.inf], np.float32), NEAR, FAR)
    assert q.dtype == np.uint16
    assert q[0] == 0 and q[1] == 65535, q
    assert q[3] == 0 and q[5] == 0 and q[4] == 65535 and q[6] == 65535, q
    # Exact reference, incl. bin edges (half-quantum rounds up) on a clean range.
    near, far = 0.0, 65535.0  # quantum exactly 1.0 m
    xs = np.array([0.0, 0.25, 0.49, 0.5, 0.51, 1.0, 1.5, 65534.5, 65535.0, 70000.0], np.float64)
    got = quantize_depth(xs, near, far)
    want = [_q_ref(x, near, far) for x in xs]
    assert got.tolist() == want == [0, 0, 0, 1, 1, 1, 2, 65535, 65535, 65535], (got.tolist(), want)
    assert quantize_depth(np.array([0.4999999], np.float64), near, far)[0] == 0
    assert quantize_depth(np.array([0.5], np.float64), near, far)[0] == 1
    # Monotone on a sweep, matches the scalar reference everywhere.
    sweep = np.linspace(-1.0, 210.0, 5001).astype(np.float32)
    qs = quantize_depth(sweep, NEAR, FAR)
    assert np.all(np.diff(qs.astype(np.int64)) >= 0)
    assert qs.tolist() == [_q_ref(x, NEAR, FAR) for x in sweep]
    assert quantize_depth(np.zeros(0, np.float32), NEAR, FAR).shape == (0,)
    assert quantize_depth(np.zeros(0, np.float32), NEAR, FAR).dtype == np.uint16
    assert _raises(ValueError, quantize_depth, np.array([np.nan]), NEAR, FAR)
    assert math.isclose(quantum_m(NEAR, FAR), (FAR - NEAR) / 65535.0, rel_tol=0, abs_tol=0)
    assert 0.00304 < quantum_m(NEAR, FAR) < 0.00306  # 3.05 mm at the bench range
    assert quantum_m(0.0, 65535.0) == 1.0
    print("check_quantizer: OK")


def check_examples() -> None:
    d = np.array([5.0, 1.0, 9.0, 3.0], np.float32)
    assert sort_splats(d, NEAR, FAR).tolist() == [2, 0, 3, 1]  # farthest first
    assert sort_splats(d, NEAR, FAR, descending=False).tolist() == [1, 3, 0, 2]
    assert sort_splats(d, NEAR, FAR, True).tolist() == sort_splats(d, NEAR, FAR).tolist()  # default descending
    # empty / single / all-equal
    e = sort_splats(np.zeros(0, np.float32), NEAR, FAR)
    assert e.dtype == np.intp and e.shape == (0,)
    e = sort_splats(np.zeros(0, np.float32), NEAR, FAR, descending=False)
    assert e.dtype == np.intp and e.shape == (0,)
    assert sort_splats(np.array([7.0], np.float32), NEAR, FAR).tolist() == [0]
    assert sort_splats(np.array([np.nan], np.float32), NEAR, FAR).tolist() == [0]
    for desc in (True, False):
        eq = sort_splats(np.full(1000, 12.5, np.float32), NEAR, FAR, desc)
        assert np.array_equal(eq, np.arange(1000)), desc
    # already sorted / reverse sorted
    srt = np.linspace(1.0, 100.0, 777).astype(np.float32)
    assert np.array_equal(sort_splats(srt, NEAR, FAR, descending=False), np.arange(777))
    assert np.array_equal(sort_splats(srt, NEAR, FAR, descending=True), np.arange(777)[::-1])
    assert np.array_equal(sort_splats(srt[::-1].copy(), NEAR, FAR, descending=True), np.arange(777))
    # same-quantum ties: stable by input index (not by float depth)
    q = quantum_m(NEAR, FAR)
    mid = NEAR + 30000 * q  # centre of bin 30000
    tie = np.array([mid + q * 0.10, mid + q * 0.05, mid - q * 0.10], np.float32)
    assert len({int(x) for x in quantize_depth(tie, NEAR, FAR)}) == 1
    assert sort_splats(tie, NEAR, FAR).tolist() == [0, 1, 2]  # float argsort would give [0, 1, 2]
    assert sort_splats(tie[::-1].copy(), NEAR, FAR).tolist() == [0, 1, 2]  # float argsort would give [2, 1, 0]
    assert _float_ref(tie[::-1].copy()).tolist() == [2, 1, 0]
    # 300k random vs reference-by-construction (vectorized float64 formula)
    rng = np.random.default_rng(7)
    big = rng.uniform(0.0, 64.0, N_PERF).astype(np.float32)
    c = np.clip(big.astype(np.float64), NEAR, FAR)
    qk = np.floor((c - NEAR) / (FAR - NEAR) * 65535.0 + 0.5).astype(np.int64)
    assert np.array_equal(sort_splats(big, NEAR, FAR), np.argsort(-qk, kind="stable"))
    assert np.array_equal(sort_splats(big, NEAR, FAR, False), np.argsort(qk, kind="stable"))
    _assert_perm(sort_splats(big, NEAR, FAR), N_PERF)
    print("check_examples: OK")


def check_nan_inf() -> None:
    d = np.array([np.nan, 3.0, np.inf, np.nan, -np.inf, 1.0, np.inf, -np.inf, 3.0], np.float32)
    # descending: +inf (bin 65535) first in index order, then 3.0s, 1.0, -inf (bin 0), NaN last
    got = sort_splats(d, NEAR, FAR, True)
    assert got.tolist() == [2, 6, 1, 8, 5, 4, 7, 0, 3], got.tolist()
    got = sort_splats(d, NEAR, FAR, False)
    assert got.tolist() == [4, 7, 5, 1, 8, 2, 6, 0, 3], got.tolist()
    # same place as today's float argsort(-d) for the non-NaN, NaN last (the finite 3.0 pair ties)
    f = _float_ref(d).tolist()
    assert f[-2:] == [0, 3] and sort_splats(d, NEAR, FAR, True).tolist()[-2:] == [0, 3]
    assert f[:2] == [2, 6] and f[-4:-2] == [4, 7]
    allnan = np.full(50, np.nan, np.float32)
    assert np.array_equal(sort_splats(allnan, NEAR, FAR), np.arange(50))
    assert np.array_equal(sort_splats(allnan, NEAR, FAR, False), np.arange(50))
    allinf = np.full(40, np.inf, np.float32)
    assert np.array_equal(sort_splats(allinf, NEAR, FAR), np.arange(40))
    mixed = np.array([np.inf, -np.inf], np.float32)
    assert sort_splats(mixed, NEAR, FAR).tolist() == [0, 1]
    assert sort_splats(mixed, NEAR, FAR, False).tolist() == [1, 0]
    # out-of-range finite depths clip to the end bins
    oor = np.array([-3.0, 1e6, 0.0, 250.0, 100.0], np.float32)
    assert sort_splats(oor, NEAR, FAR).tolist() == [1, 3, 4, 0, 2]
    print("check_nan_inf: OK")


def check_validation_and_purity() -> None:
    d = np.array([3.0, 1.0, 2.0], np.float32)
    for near, far in [(1.0, 1.0), (2.0, 1.0), (np.nan, 1.0), (0.0, np.inf), (-np.inf, 1.0), (0.0, np.nan),
                      (True, 5.0), (0.0, True), (-1e308, 1e308), ("a", 1.0), (None, 1.0)]:
        assert _raises(ValueError, sort_splats, d, near, far), (near, far)
        assert _raises(ValueError, quantize_depth, d, near, far), (near, far)
    for bad in (np.zeros((2, 2), np.float32), np.float32(3.0), np.array([True, False]), np.array([1j, 2j]),
                np.array(["a", "b"])):
        assert _raises(ValueError, sort_splats, bad, NEAR, FAR), repr(bad)
    for bad_desc in (1, 0, None, "yes"):
        assert _raises(ValueError, sort_splats, d, NEAR, FAR, bad_desc), repr(bad_desc)
    assert sort_splats(d, NEAR, FAR, descending=np.bool_(False)).tolist() == [1, 2, 0]
    # integer and float64 / float16 inputs are accepted and agree with the float64 reference
    assert sort_splats(np.array([3, 1, 2], np.int32), NEAR, FAR).tolist() == [0, 2, 1]
    assert sort_splats(d.astype(np.float64), NEAR, FAR).tolist() == [0, 2, 1]
    assert sort_splats(d.astype(np.float16), NEAR, FAR).tolist() == [0, 2, 1]
    # no mutation (all dtypes, incl. NaN payload), non-contiguous view, determinism
    rng = np.random.default_rng(3)
    for dt in (np.float32, np.float64):
        a = rng.uniform(-5, 300, 4000).astype(dt)
        a[::97] = np.nan
        a[5::193] = np.inf
        before = a.tobytes()
        o1 = sort_splats(a, NEAR, FAR)
        assert a.tobytes() == before
        o2 = sort_splats(a, NEAR, FAR)
        assert np.array_equal(o1, o2)
        view = a[::2]
        vb = view.copy().tobytes()
        sort_splats(view, NEAR, FAR)
        assert view.tobytes() == vb and a.tobytes() == before
        assert np.array_equal(sort_splats(view, NEAR, FAR), sort_splats(np.ascontiguousarray(view), NEAR, FAR))
    ro = rng.uniform(1, 9, 100).astype(np.float32)
    ro.setflags(write=False)
    sort_splats(ro, NEAR, FAR)  # read-only input must work
    # the output does not alias the input
    out = sort_splats(d, NEAR, FAR)
    out[:] = 0
    assert d.tolist() == [3.0, 1.0, 2.0]
    print("check_validation_and_purity: OK")


def check_module_is_numpy_only_and_safe() -> None:
    src = Path(rs.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import"
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"__future__", "math", "numpy"}, mods
    low = src.lower()
    for word in ("random", "time.", "perf_counter", "heartbeat", "aeb", "engage", "cuda", "torch", "moderngl", "numba"):
        assert word not in low, word
    print("check_module_is_numpy_only_and_safe: OK")


# ------------------------------------------------- behavior change vs float

def _assert_float_order_within_quantum(d32: np.ndarray, near: float, far: float) -> int:
    """New order differs from float order only within one quantum. Returns #pairs that flipped (<=)."""
    order = sort_splats(d32, near, far, True)
    quantum = quantum_m(near, far)
    d = d32.astype(np.float64)
    ok = ~np.isnan(d) & (d >= near) & (d <= far)  # clipped depths are culled by the renderer
    seq = d[order]
    okp = ok[order]
    # every later in-range element that is deeper than an earlier one must be within one quantum
    sv = np.where(okp, seq, -np.inf)
    rev_max = np.maximum.accumulate(sv[::-1])[::-1]
    later_max = np.concatenate([rev_max[1:], [-np.inf]])
    gap = np.full(seq.shape, -np.inf)
    sel = okp & np.isfinite(later_max)
    gap[sel] = later_max[sel] - seq[sel]
    worst = gap.max() if gap.size else -np.inf
    assert worst <= quantum * (1 + 1e-9), (worst, quantum)
    # pairs in different bins keep float order: key sequence is non-increasing for valid entries
    valid_pos = order[~np.isnan(d)[order]]
    qq = quantize_depth(d32[valid_pos], near, far).astype(np.int64)
    assert np.all(np.diff(qq) <= 0)
    return int(worst > 0)


def check_float_order_differs_only_within_quantum() -> None:
    rng = np.random.default_rng(11)
    q = quantum_m(NEAR, FAR)
    # random scene + a dense cluster of near-ties so that real flips happen
    d = rng.uniform(0.1, 64.0, 50_000).astype(np.float32)
    cl = (30.0 + rng.uniform(0.0, 3.0 * q, 5_000)).astype(np.float32)
    d = np.concatenate([d, cl])
    rng.shuffle(d)
    new = sort_splats(d, NEAR, FAR)
    old = _float_ref(d)
    assert not np.array_equal(new, old), "scene should contain same-quantum ties that reorder"
    _assert_float_order_within_quantum(d, NEAR, FAR)
    # a scene with all depths > 1 quantum apart is identical to the float argsort
    spaced = (1.0 + np.arange(2000) * 5.0 * q).astype(np.float32)
    rng.shuffle(spaced)
    assert np.array_equal(sort_splats(spaced, NEAR, FAR), _float_ref(spaced))
    # a 64 m viz range has a ~1 mm quantum
    q64 = quantum_m(NEAR, 64.0)
    assert 0.00097 < q64 < 0.00099
    _assert_float_order_within_quantum(d, NEAR, 64.0)
    # worst-case misorder is real (>0) but bounded by one quantum: two splats 0.8 quantum apart
    mid = NEAR + 30000 * q  # centre of bin 30000
    two = np.array([mid - 0.4 * q, mid + 0.4 * q], np.float32)
    assert len(set(quantize_depth(two, NEAR, FAR).tolist())) == 1
    assert _float_ref(two).tolist() == [1, 0]  # float order: deeper splat (index 1) first
    assert sort_splats(two, NEAR, FAR).tolist() == [0, 1]  # key tie: input order
    # ... but one bin apart is never flipped (centres of adjacent bins, 1 quantum apart)
    adj = np.array([mid, mid + 1.0 * q], np.float32)
    assert sort_splats(adj, NEAR, FAR).tolist() == [1, 0]
    print("check_float_order_differs_only_within_quantum: OK")


# ---------------------------------------------------- render instrument

def _composite(order, pos_uv, z, col, alpha, w, h, rad=2):
    """CPU mimic of the bench GL pass: premultiplied alpha over, depth channel z*a blended
    the same way (blend ONE, ONE_MINUS_SRC_ALPHA), final zbuf = sum(z*a)/sum(a)."""
    C = np.zeros((h, w, 3))
    D = np.zeros((h, w))
    A = np.zeros((h, w))
    ys, xs = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    r2 = (xs ** 2 + ys ** 2) * (9.0 / (rad * rad))
    foot = np.exp(-0.5 * r2) * (r2 <= 9.0)
    for i in order:
        u, v = pos_uv[i]
        x0, y0 = u - rad, v - rad
        xa, xb = max(x0, 0), min(x0 + 2 * rad + 1, w)
        ya, yb = max(y0, 0), min(y0 + 2 * rad + 1, h)
        if xa >= xb or ya >= yb:
            continue
        a = alpha[i] * foot[ya - y0:yb - y0, xa - x0:xb - x0]
        om = 1.0 - a
        C[ya:yb, xa:xb] = col[i] * a[..., None] + C[ya:yb, xa:xb] * om[..., None]
        D[ya:yb, xa:xb] = z[i] * a + D[ya:yb, xa:xb] * om
        A[ya:yb, xa:xb] = a + A[ya:yb, xa:xb] * om
    zbuf = np.where(A > 1e-3, D / np.maximum(A, 1e-6), 0.0)
    return C, zbuf, A


def _scene(n, n_cluster, seed):
    rng = np.random.default_rng(seed)
    w, h = 96, 72
    uv = np.stack([rng.integers(0, w, n), rng.integers(0, h, n)], 1)
    z = rng.uniform(1.0, 40.0, n).astype(np.float32)
    col = rng.uniform(0.0, 1.0, (n, 3))
    alpha = rng.uniform(0.3, 0.9, n)
    if n_cluster:
        q = quantum_m(NEAR, FAR)
        k = n_cluster
        uv[:k] = np.stack([rng.integers(30, 40, k), rng.integers(20, 30, k)], 1)
        z[:k] = (20.0 + rng.uniform(0.0, 0.9 * q, k)).astype(np.float32)  # one quantum
    return w, h, uv, z, col, alpha


def render_instrument_stats(n=5000, n_cluster=0, seed=5):
    """Render one fixed scene with float argsort order and with the new order; max per-pixel diffs."""
    w, h, uv, z, col, alpha = _scene(n, n_cluster, seed)
    old = _float_ref(z)
    new = sort_splats(z, NEAR, FAR)
    c0, z0, a0 = _composite(old, uv, z, col, alpha, w, h)
    c1, z1, a1 = _composite(new, uv, z, col, alpha, w, h)
    return {
        "splats": n, "cluster": n_cluster, "order_differs": bool(not np.array_equal(old, new)),
        "n_pos_differ": int((old != new).sum()),
        "max_depth_diff_m": float(np.abs(z0 - z1).max()),
        "max_colour_diff_u8": float(np.abs(c0 - c1).max() * 255.0),
        "max_alpha_diff": float(np.abs(a0 - a1).max()),
        "quantum_m": quantum_m(NEAR, FAR),
    }


def check_render_instrument() -> None:
    # (1) natural scene: random float depths, same-quantum collisions are rare and far apart in x/y
    nat = render_instrument_stats(n=5000, n_cluster=0)
    q = nat["quantum_m"]
    print("render_instrument natural:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in nat.items()})
    assert nat["max_depth_diff_m"] <= q * (1 + 1e-9), nat
    assert nat["max_alpha_diff"] < 1e-9, nat  # total transmittance is order-independent within a bin
    # (2) adversarial: 400 splats in ONE quantum, overlapping footprints, random colours
    adv = render_instrument_stats(n=5000, n_cluster=400)
    print("render_instrument adversarial:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in adv.items()})
    assert adv["order_differs"], adv
    assert adv["max_depth_diff_m"] <= q * (1 + 1e-9), adv  # z-weights only move within a quantum
    assert adv["max_alpha_diff"] < 1e-9, adv
    # colour can differ for coincident splats in the same quantum; that is the accepted cost, bounded by 255
    assert adv["max_colour_diff_u8"] <= 255.0
    # (3) scene whose depths are all >1 quantum apart: render is bit-identical
    w, h, uv, z, col, alpha = _scene(3000, 0, 9)
    z = (1.0 + np.random.default_rng(1).permutation(3000) * 4.0 * quantum_m(NEAR, FAR)).astype(np.float32)
    old, new = _float_ref(z), sort_splats(z, NEAR, FAR)
    assert np.array_equal(old, new)
    ra, rb = _composite(old, uv, z, col, alpha, w, h), _composite(new, uv, z, col, alpha, w, h)
    assert all(np.array_equal(x, y) for x, y in zip(ra, rb))
    print("check_render_instrument: OK")


# ------------------------------------------------------------------ perf

def _p95(x) -> float:
    return float(np.percentile(np.asarray(x, np.float64), 95))


def _time_ms(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1e3


def perf_numbers(n=N_PERF, runs=PERF_RUNS, pairs=PERF_PAIRS) -> dict:
    """A/A first (float argsort vs float argsort), then new sort vs float argsort, both as
    interleaved pairs with the same statistic (p95 over `runs` iterations per side)."""
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.0, 64.0, n).astype(np.float32)
    neg = -depth

    def base():
        np.argsort(neg, kind="stable")  # the bench sort; -d is precomputed for both A and A'

    def new():
        sort_splats(depth, NEAR, FAR)

    for _ in range(3):  # warm up allocator / caches
        base()
        new()

    def one_pair(fa, fb):
        ta, tb = [], []
        for _ in range(runs):
            ta.append(_time_ms(fa))
            tb.append(_time_ms(fb))
        return _p95(ta), _p95(tb)

    aa = [one_pair(base, base) for _ in range(pairs)]
    ab = [one_pair(base, new) for _ in range(pairs)]
    aa_ratios = [b / a for a, b in aa]
    # symmetric A/A: report the worst deviation from 1 in either direction
    aa_worst = max(max(r, 1.0 / r) for r in aa_ratios)
    return {
        "n": n, "runs_per_side": runs, "pairs": pairs,
        "aa_ratios": [round(r, 4) for r in aa_ratios], "aa_worst": round(aa_worst, 4),
        "base_p95_ms": [round(a, 3) for a, _ in ab], "new_p95_ms": [round(b, 3) for _, b in ab],
        "pair_ratios_new_over_base": [round(b / a, 4) for a, b in ab],
        "new_p95_median_ms": round(float(np.median([b for _, b in ab])), 3),
        "base_p95_median_ms": round(float(np.median([a for a, _ in ab])), 3),
        "load1": os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
    }


def check_perf_proxy() -> None:
    r = perf_numbers()
    print("perf_proxy (box, PROXY ONLY; official = GVD's IQU45 bench with A/A<=1.03):", r)
    if r["aa_worst"] > AA_MAX:
        skip_check("perf_proxy_15ms", f"A/A worst ratio {r['aa_worst']} > {AA_MAX}: box too noisy to gate; "
                   f"numbers above are not a pass. Official gate is the IQU45 bench.")
        return
    assert r["new_p95_median_ms"] <= PERF_P95_MS, r
    assert max(r["new_p95_ms"]) <= PERF_P95_MS, r
    assert max(r["pair_ratios_new_over_base"]) < 1.0, r  # never slower than float argsort
    print("check_perf_proxy: OK")


# ----------------------------------------------------------- hypothesis

def _build_props():
    if given is None:
        return []
    JIT = float(np.float32(0.004))
    f32 = st.floats(width=32, allow_nan=True, allow_infinity=True)
    ranges = st.tuples(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.001, max_value=500.0, allow_nan=False, allow_infinity=False),
    ).map(lambda t: (t[0], t[0] + t[1]))
    # depth lists biased toward collisions: a few base values each with sub-quantum jitter
    base = st.lists(f32, min_size=0, max_size=60)
    clustered = st.tuples(
        st.lists(st.floats(min_value=0.0, max_value=70.0, width=32), min_size=1, max_size=6),
        st.lists(st.tuples(st.integers(0, 5), st.floats(min_value=-JIT, max_value=JIT, width=32)), min_size=0, max_size=60),
    ).map(lambda t: [np.float32(t[0][i % len(t[0])] + j) for i, j in t[1]])
    arrays = st.one_of(base, clustered).map(lambda xs: np.array(xs, dtype=np.float32))
    cfg = settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])

    @cfg
    @given(arrays, ranges, st.booleans())
    def prop_matches_stable_argsort_on_quantized_keys(d, nf, desc):
        near, far = nf
        got = sort_splats(d, near, far, desc)
        assert np.array_equal(got, _ref_order(d, near, far, desc))

    @cfg
    @given(arrays, ranges, st.booleans())
    def prop_permutation(d, nf, desc):
        _assert_perm(sort_splats(d, nf[0], nf[1], desc), d.size)

    @cfg
    @given(arrays, ranges, st.booleans())
    def prop_idempotent_resort(d, nf, desc):
        near, far = nf
        o = sort_splats(d, near, far, desc)
        assert np.array_equal(sort_splats(d[o], near, far, desc), np.arange(d.size))

    @cfg
    @given(arrays, ranges, st.booleans())
    def prop_input_not_mutated(d, nf, desc):
        before = d.tobytes()
        sort_splats(d, nf[0], nf[1], desc)
        assert d.tobytes() == before

    @cfg
    @given(arrays, ranges)
    def prop_quantizer_matches_scalar_and_is_monotone(d, nf):
        near, far = nf
        v = d[~np.isnan(d)]
        q = quantize_depth(v, near, far)
        assert q.dtype == np.uint16
        assert q.tolist() == [_q_ref(x, near, far) for x in v]
        s = np.sort(v)
        assert np.all(np.diff(quantize_depth(s, near, far).astype(np.int64)) >= 0)

    @cfg
    @given(arrays, ranges)
    def prop_float_order_differs_only_within_quantum(d, nf):
        near, far = nf
        _assert_float_order_within_quantum(d, near, far)

    @cfg
    @given(arrays, ranges)
    def prop_descending_is_reverse_key_order_of_ascending(d, nf):
        near, far = nf
        a = sort_splats(d, near, far, False)
        b = sort_splats(d, near, far, True)
        nn = int((~np.isnan(d)).sum())
        qa = quantize_depth(d[a[:nn]], near, far).astype(np.int64)
        qb = quantize_depth(d[b[:nn]], near, far).astype(np.int64)
        assert np.array_equal(qa, qb[::-1])
        assert np.array_equal(a[nn:], b[nn:])  # NaN tail identical

    return [
        prop_matches_stable_argsort_on_quantized_keys,
        prop_permutation,
        prop_idempotent_resort,
        prop_input_not_mutated,
        prop_quantizer_matches_scalar_and_is_monotone,
        prop_float_order_differs_only_within_quantum,
        prop_descending_is_reverse_key_order_of_ascending,
    ]


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


def main() -> None:
    check_quantizer()
    check_examples()
    check_nan_inf()
    check_validation_and_purity()
    check_module_is_numpy_only_and_safe()
    check_float_order_differs_only_within_quantum()
    check_render_instrument()
    _run_hypothesis_props_or_skip()
    check_perf_proxy()
    print("test_rv_splat_sort: OK")


if __name__ == "__main__":
    main()


# Pytest entrypoints (for mutmut 3.8.1); perf is excluded from mutation runs with -k.
def test_quantizer() -> None:
    check_quantizer()


def test_examples() -> None:
    check_examples()


def test_nan_inf() -> None:
    check_nan_inf()


def test_validation_and_purity() -> None:
    check_validation_and_purity()


def test_module_is_numpy_only_and_safe() -> None:
    check_module_is_numpy_only_and_safe()


def test_float_order_within_quantum() -> None:
    check_float_order_differs_only_within_quantum()


def test_render_instrument() -> None:
    check_render_instrument()


def test_hypothesis_props() -> None:
    _run_hypothesis_props_or_skip()


def test_perf_proxy() -> None:
    check_perf_proxy()
