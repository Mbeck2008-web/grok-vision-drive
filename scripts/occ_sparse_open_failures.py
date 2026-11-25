#!/usr/bin/env python3
"""Open PR #237 fixtures that the current band picker still fails.

This file is intentionally NOT named test_*.py, so run_offline_suite does
not run it. The default suite has no xfail. After the swarm fixes Majors 2
and 3 and Minor 6, these asserts should be moved into test_occ_net.py.

Run:  PYTHONPATH=. python scripts/occ_sparse_open_failures.py
Current head (d2ce025 / later test-only commits) is expected to exit 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from python.perception.occ_geometry import DEFAULT_GRID  # noqa: E402
from python.perception.occ_sparse import _capped_hot, _column_ground_lvl  # noqa: E402

G = DEFAULT_GRID


def _levels(n, cell, layers):
    ix, iy = [a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), indexing="ij")]
    col, lv, pp, gt = [], [], [], np.full(n * n, -999)
    for mask, lvl, p, is_gnd in layers:
        m = mask(ix, iy)
        c = ix[m] * n + iy[m]
        col.append(c)
        lv.append(np.full(int(m.sum()), lvl) if np.isscalar(lvl) else lvl(ix[m], iy[m]))
        pp.append(np.full(int(m.sum()), p) if np.isscalar(p) else p(ix[m], iy[m]))
        if is_gnd:
            gt[c] = lv[-1]
    col = np.concatenate(col)
    lvl = np.concatenate(lv).astype(np.int64)
    p = np.concatenate(pp).astype(np.float32)
    band = _column_ground_lvl(col, lvl, p, col // n, col % n, cell)
    return band[np.arange(n * n)], gt, ix, iy, n


def _kept_walls(f, g, pg, pf, sig=0.0):
    cm = G.voxel_m * f
    cap = 2048 if f == 2 else 512
    xs = G.x_min + cm * (np.arange(int(51.2 / cm)) + 0.5)
    ys = G.y_min + cm * (np.arange(int(64 / cm)) + 0.5)
    X, Y = [a.ravel() for a in np.meshgrid(xs, ys, indexing="ij")]
    zc = lambda z: G.z_min + cm * (np.floor((z - G.z_min) / cm) + 0.5)
    bld = (np.abs(X) > 8) & (np.abs(X) < 14) & (Y > 0) & (Y < 45)
    gz = lambda x, y, g=g: 0.2 + g * np.maximum(y, 0)
    parts = [
        (np.stack([X, Y, zc(gz(X, Y))], 1), pg, "ground"),
        (np.stack([X, Y, zc(gz(X, Y) - cm)], 1), pf, "below"),
    ]
    parts += [
        (np.stack([X[bld], Y[bld], zc(gz(X[bld], Y[bld]) + k * cm)], 1), pg, "wall")
        for k in (1, 2, 3)
    ]
    hot = np.concatenate([p[0] for p in parts])
    ph = np.concatenate([np.full(len(p[0]), p[1]) for p in parts]).astype(np.float32)
    lab = np.concatenate([np.full(len(p[0]), p[2]) for p in parts])
    keep = (hot[:, 2] >= G.z_min) & (hot[:, 2] < G.z_max)
    hot, ph, lab = hot[keep], ph[keep], lab[keep]
    idx = _capped_hot(hot, ph, cap, G.z_min, cm)
    n_wall = int((lab[idx] == "wall").sum())
    n_tot = int((lab == "wall").sum())
    n_bel = int((lab[idx] == "below").sum())
    return n_wall, n_tot, n_bel, cap


def _f4_bld_high(g, sig):
    f, cap, cm = 4, 512, G.voxel_m * 4
    xs = G.x_min + cm * (np.arange(int(51.2 / cm)) + 0.5)
    ys = G.y_min + cm * (np.arange(int(64 / cm)) + 0.5)
    X, Y = [a.ravel() for a in np.meshgrid(xs, ys, indexing="ij")]
    zc = lambda z: G.z_min + cm * (np.floor((z - G.z_min) / cm) + 0.5)
    bld = (np.abs(X) > 8) & (np.abs(X) < 14) & (Y > 0) & (Y < 45)
    gz = lambda x, y, g=g: 0.2 + g * np.maximum(y, 0)
    parts = [
        (np.stack([X, Y, zc(gz(X, Y))], 1), 0.95, "ground"),
        (np.stack([X, Y, zc(gz(X, Y) - cm)], 1), 0.9, "below"),
    ]
    parts += [
        (np.stack([X[bld], Y[bld], zc(gz(X[bld], Y[bld]) + k * cm)], 1), 0.95, "wall")
        for k in (1, 2, 3)
    ]
    rng = np.random.default_rng(1)
    hot = np.concatenate([p[0] for p in parts])
    ph = np.concatenate([np.full(len(p[0]), p[1]) for p in parts]).astype(np.float64)
    ph = np.clip(ph + rng.normal(0, sig, len(ph)), 0, 1).astype(np.float32)
    lab = np.concatenate([np.full(len(p[0]), p[2]) for p in parts])
    keep = (hot[:, 2] >= G.z_min) & (hot[:, 2] < G.z_max)
    hot, ph, lab = hot[keep], ph[keep], lab[keep]
    lvl = np.floor((hot[:, 2] - G.z_min) / cm + 1e-6).astype(np.int64)
    ix = np.floor(hot[:, 0] / cm + 1e-6).astype(np.int64)
    iy = np.floor(hot[:, 1] / cm + 1e-6).astype(np.int64)
    ix0, iy0 = ix - int(ix.min()), iy - int(iy.min())
    col = ix0 * (int(iy0.max()) + 1) + iy0
    gl = _column_ground_lvl(col, lvl, ph, ix0, iy0, cm)[col]
    in_bld = (np.abs(hot[:, 0]) > 8) & (np.abs(hot[:, 0]) < 14) & (hot[:, 1] > 0) & (hot[:, 1] < 45)
    tg, is_bld = {}, {}
    for c, l, lb, b in zip(col, lvl, lab, in_bld):
        c = int(c)
        if lb == "ground":
            tg[c] = int(l)
        if b:
            is_bld[c] = True
    gl_c = {int(c): int(g_) for c, g_ in zip(col, gl)}
    bld_cs = [c for c in tg if is_bld.get(c, False)]
    high = sum(gl_c[c] > tg[c] for c in bld_cs)
    return high, len(bld_cs)


def main() -> int:
    failed = []

    # --- Major 2: low-p ground more than 12 m from a strong voter ---
    n, cell = 64, 0.8
    # Corner of strong road, the rest a p .60 road one layer up.
    band, gt, ix, iy, n = _levels(n, cell, [
        (lambda x, y: (x < 6) & (y < 6), 4, 0.90, True),
        (lambda x, y: ~((x < 6) & (y < 6)), 5, 0.60, True),
        (lambda x, y: ~((x < 6) & (y < 6)), 4, 0.55, False),
    ])
    dx = np.where(ix < 6, 0, ix - 5)
    dy = np.where(iy < 6, 0, iy - 5)
    far = (np.maximum(dx, dy) > 15)  # more than 12 m (15 * 0.8 m) from the corner
    n_far = int(far.sum())
    n_ok = int((band[far] == 5).sum())
    if n_ok != n_far:
        failed.append(
            f"Major 2 corner +1 step p.60: far cols at own seed 5: {n_ok}/{n_far} "
            f"(v9 inherits the corner height; want own seed)"
        )
    else:
        print(f"Major 2 corner +1 step p.60: far cols {n_ok}/{n_far} at own seed OK")

    # +10% grade, road .95 for x<32 and .62 beyond (fill .90/.57).
    band, gt, ix, iy, n = _levels(n, cell, [
        (lambda x, y: x >= 0, lambda x, y: 4 + (0.10 * y).astype(int),
         lambda x, y: np.where(x < 32, 0.95, 0.62), True),
        (lambda x, y: x >= 0, lambda x, y: 3 + (0.10 * y).astype(int),
         lambda x, y: np.where(x < 32, 0.90, 0.57), False),
    ])
    wrong = int((band != gt).sum())
    if wrong != 0:
        failed.append(
            f"Major 2 +10% grade, half p.62: {wrong}/4096 columns off the grade "
            f"(v9 ~568, inherited flat height; want 0)"
        )
    else:
        print("Major 2 +10% grade, half p.62: 0/4096 wrong OK")

    # Road p .63, fill .58, one-layer cars p .66 every 9th row.
    band, gt, ix, iy, n = _levels(n, cell, [
        (lambda x, y: x >= 0, 4, 0.63, True),
        (lambda x, y: x >= 0, 3, 0.58, False),
        (lambda x, y: (y % 9) == 0, 5, 0.66, False),
    ])
    at_car = int((band == 5).sum())
    if at_car != 0:
        failed.append(
            f"Major 2 road p.63 + cars p.66: band at car in {at_car}/4096 columns "
            f"(v9: 4096; cars are the only voters; want 0)"
        )
    else:
        print("Major 2 road p.63 + cars p.66: no column at the car layer OK")

    # --- Major 3: fill within 0.01 of ground, or saturated, on a grade ---
    w, tot, bel, cap = _kept_walls(2, 0.05, 0.95, 0.94)
    # v7 held 1108/2198; v9 keeps 48.
    if w < 1000:
        failed.append(
            f"Major 3 +5% ground .95 / fill .94: walls {w}/{tot} (v7: 1108; want >= 1000)"
        )
    else:
        print(f"Major 3 +5% .95/.94: walls {w}/{tot} OK")

    w, tot, bel, cap = _kept_walls(2, 0.10, 1.0, 1.0)
    # Saturated +10% keeps 8/1344 on v9 / 6cdfb8d / cfb2a42; walls must stay.
    if w < 1000:
        failed.append(
            f"Major 3 +10% saturated ground=fill=1.0: walls {w}/{tot} "
            f"(v9: 8; want >= 1000)"
        )
    else:
        print(f"Major 3 +10% saturated 1.0/1.0: walls {w}/{tot} OK")

    # --- Minor 6: band must not land on the first wall layer ---
    for g, sig, tag in ((0.10, 0.01, "+10% σ.01"), (0.15, 0.01, "+15% σ.01"),
                        (0.10, 0.02, "+10% σ.02"), (0.15, 0.02, "+15% σ.02")):
        high, n_bld = _f4_bld_high(g, sig)
        if high != 0:
            failed.append(
                f"Minor 6 f=4 {tag}: {high}/{n_bld} building columns have the band "
                f"on the first wall layer (b91b957: 0; want 0)"
            )
        else:
            print(f"Minor 6 f=4 {tag}: 0/{n_bld} building cols high OK")

    print()
    if failed:
        print(f"occ_sparse_open_failures: {len(failed)} failed (expected on current code):")
        for line in failed:
            print("  -", line)
        return 1
    print("occ_sparse_open_failures: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
