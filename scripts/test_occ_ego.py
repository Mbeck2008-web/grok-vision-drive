#!/usr/bin/env python3
"""Camera-only ego motion and ego-aligned temporal memory.

A static synthetic world (posts + a parked car + road) is seen by an ego that
drives forward at 15 m/s and turns at 0.2 rad/s, at 10 Hz. Per frame only the
ego-frame occupied cells exist (what the net outputs).

* ``occ_odometry`` recovers the per-frame forward travel and yaw from the cells
  alone (camera VO, no pose input);
* warping the previous frame's static cells by that estimate lands them on the
  current frame's cells, and the cells mapped back to the world with the
  integrated VO pose stay fixed in world space across frames;
* the tracker fed with that VO motion keeps the parked car ``parked``, and its
  compensated history stays put; with no compensation it would not;
* a featureless frame (road only) holds the last estimate and reports not ok.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from python.perception.occ_ego import (  # noqa: E402
    EgoMotion, VO_MIN_CELLS, VO_MIN_SCORE, _static_bev, _warp, occ_odometry,
)
from python.perception.occ_geometry import DEFAULT_GRID, PinholeCam  # noqa: E402
try:
    from python.perception.occ_geometry import observed_mask  # noqa: E402
except ImportError:
    from python.perception.occ_out import observed_mask  # noqa: E402
from python.perception.occ_outputs import OccTracker, objects  # noqa: E402

G = DEFAULT_GRID
FWD, DYAW, DT = 1.5, 0.02, 0.1


def world():
    rng = np.random.default_rng(3)
    posts = np.stack([rng.uniform(-22, 22, 150), rng.uniform(-10, 70, 150)], 1)
    posts = posts[np.abs(posts[:, 0]) > 4.5]  # not on the road
    car = np.array([[x, y] for x in np.arange(3.0, 4.81, 0.2) for y in np.arange(30.0, 34.41, 0.2)])  # parked, right side
    road = np.array([[x, y] for x in np.arange(-4.0, 4.0, 0.4) for y in np.arange(-10, 70, 0.4)])
    return posts, car, road


def pose(i: int):
    """Ego pose after i frames: move forward FWD then turn DYAW (CCW), repeated."""
    px, py, h = 0.0, 0.0, 0.0
    for _ in range(i):
        px += -math.sin(h) * FWD
        py += math.cos(h) * FWD
        h += DYAW
    return px, py, h


def to_ego(q, p):
    px, py, h = p
    d = q - np.array([px, py])
    return np.stack([d @ np.array([math.cos(h), math.sin(h)]), d @ np.array([-math.sin(h), math.cos(h)])], 1)


def frame_cells(i: int):
    posts, car, road = world()
    p = pose(i)
    keys, cls = [], []
    for pts, c, zs in ((posts, 0, (0.2, 0.6, 1.0, 1.4, 1.8)), (car, 2, (0.2, 0.6, 1.0, 1.4)), (road, 1, (-0.2,))):
        e = to_ego(pts, p)
        for z in zs:
            xyz = np.concatenate([e, np.full((len(e), 1), z)], 1)
            ins = G.inside(xyz)
            keys.append(G.keys_of(xyz[ins])); cls.append(np.full(int(ins.sum()), c, np.uint8))
    # a lead car pacing the ego 15 m ahead: fixed in the ego frame, it must not drag VO to zero motion
    lead = np.array([[x, y] for x in np.arange(-0.8, 0.81, 0.2) for y in np.arange(15.0, 19.41, 0.2)])
    for z in (0.2, 0.6, 1.0, 1.4):
        xyz = np.concatenate([lead, np.full((len(lead), 1), z)], 1)
        keys.append(G.keys_of(xyz)); cls.append(np.full(len(xyz), 2, np.uint8))
    k = np.concatenate(keys); c = np.concatenate(cls)
    k, idx = np.unique(k, return_index=True)
    return k, c[idx]


def check_vo() -> None:
    errs_f, errs_y, overlaps = [], [], []
    for i in range(1, 8):
        k0, c0 = frame_cells(i - 1)
        k1, c1 = frame_cells(i)
        f, y, score = occ_odometry(k0, c0, k1, c1)
        errs_f.append(f - FWD); errs_y.append(y - DYAW)
        prev, cur = _static_bev(k0, c0, G), _static_bev(k1, c1, G)
        cur_set = {tuple(v) for v in np.floor(cur / G.voxel_m).astype(int)}
        def ov(ff, yy):
            w = np.floor(_warp(prev, ff, yy) / G.voxel_m).astype(int)
            return float(np.mean([tuple(v) in cur_set for v in w]))
        overlaps.append((ov(f, y), ov(FWD, DYAW), ov(0.0, 0.0)))
        assert score > 0.4, (i, score)
    ef, ey = np.abs(errs_f), np.abs(errs_y)
    assert ef.max() < 0.25 and abs(np.mean(errs_f)) < 0.1, errs_f
    assert ey.max() < 0.01 and abs(np.mean(errs_y)) < 0.004, errs_y
    # cells are quantized independently each frame, so even the true motion overlaps < 1
    assert all(vo >= true - 0.02 and vo > still + 0.3 for vo, true, still in overlaps), overlaps
    o = np.array(overlaps)
    print(f"VO: forward err max {ef.max():.3f} m/frame, yaw err max {ey.max():.4f} rad/frame; warped static-cell overlap "
          f"VO {o[:, 0].mean():.2f} vs true motion {o[:, 1].mean():.2f} vs no warp {o[:, 2].mean():.2f} OK")


def check_world_fixed() -> None:
    import argparse

    from python.perception.occ_ego import add_ego_args

    em = EgoMotion.from_args(add_ego_args(argparse.ArgumentParser()).parse_args([]))  # the default: flag OFF
    assert em.source == "camera_vo" and not em.from_vehicle
    px, py, h = 0.0, 0.0, 0.0
    posts_w = None
    drift = []
    tr = OccTracker()
    reps = []
    pose_err: list[float] = []
    for i in range(10):
        k, c = frame_cells(i)
        est = em.update(k, c, i * DT, vehicle=None)
        if i > 0:
            # flag off (default) still warps: camera-VO motion, non-identity and right
            assert est.ok and est.source == "camera_vo", (i, est)
            assert abs(est.speed_mps - FWD / DT) < 1.5 and abs(est.yaw_rate - DYAW / DT) < 0.1, (i, est)
            f, y = est.speed_mps * DT, est.yaw_rate * DT
            px += -math.sin(h) * f; py += math.cos(h) * f; h += y
            tx, ty, _ = pose(i)
            pose_err.append(math.hypot(px - tx, py - ty))
        bev = _static_bev(k, c, G)
        # back to world with the VO pose
        wx = px + bev[:, 0] * math.cos(h) - bev[:, 1] * math.sin(h)
        wy = py + bev[:, 0] * math.sin(h) + bev[:, 1] * math.cos(h)
        def frac(xs, ys):
            cc = np.floor(np.stack([xs, ys], 1) / G.voxel_m).astype(int)
            return float(np.mean([tuple(v) in posts_w for v in cc]))
        if posts_w is None:  # true world post + car cells, dilated by one cell (quantization)
            posts, car, _ = world()
            base = np.floor(np.concatenate([posts, car]) / G.voxel_m).astype(int)
            posts_w = {(int(a) + dx, int(b) + dy) for a, b in base for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        drift.append((frac(wx, wy), frac(bev[:, 0], bev[:, 1])))
        objs = objects(k, c)
        reps.append(tr.update(objs, i * DT, ego_speed_mps=est.speed_mps if i else 0.0, ego_yaw_rate=est.yaw_rate if i else 0.0))
    tp, tpx, tpy, th = pose(9), *pose(9)
    assert math.hypot(px - tpx, py - tpy) < 1.0 and abs(h - th) < 0.03, ((px, py, h), (tpx, tpy, th))
    vo_frac = [a for a, _ in drift]
    still_frac = [b for _, b in drift]
    assert min(vo_frac) > 0.9 and still_frac[-1] < 0.5, drift
    cars = sorted((r for r in reps[-1] if r["frames"] >= 5), key=lambda r: r["y"])
    assert [r["motion"] for r in cars] == ["same_way", "parked"], reps[-1]
    print(f"world-fixed: VO pose after 9 frames within {math.hypot(px - tpx, py - tpy):.2f} m / {abs(h - th):.4f} rad, "
          f"static cells on their true world cell (+-1) every frame >= {min(vo_frac):.2f} (no warp: {still_frac[-1]:.2f} by frame 9), "
          f"parked car stays parked, pacing lead car same_way OK")
    print(f"world-fixed: camera-VO (flag off) pose drift per frame, m: {[round(e, 3) for e in pose_err]} "
          f"over {9 * FWD:.1f} m driven ({pose_err[-1] / (9 * FWD) * 100:.1f}%)")



def check_vo_clutter() -> None:
    """A dense ego-fixed 'other' sheet must not lock VO at zero when posts move."""
    rng = np.random.default_rng(0)
    posts = np.stack([rng.uniform(-20, -6, 80), rng.uniform(5, 40, 80)], 1)
    clutter = np.stack([rng.uniform(-24, 24, 3000), rng.uniform(-12, 50, 3000)], 1)
    fwd = 2.5
    def cells(shift):
        # posts are observed; ego-fixed decode_unseen clutter is not, so it
        # cannot vote. The residual path is gone.
        post_xy = posts - np.array([0.0, shift])
        xy = np.concatenate([post_xy, clutter])
        obs = np.concatenate([np.ones(len(post_xy), bool), np.zeros(len(clutter), bool)])
        z = 1.2
        pts = np.concatenate([xy, np.full((len(xy), 1), z)], 1)
        keys = G.keys_of(pts)
        ok = keys >= 0
        return keys[ok], np.zeros(ok.sum(), np.uint8), obs[ok]
    k0, c0, o0 = cells(0.0)
    k1, c1, o1 = cells(fwd)
    est_fwd, dyaw, score = occ_odometry(k0, c0, k1, c1, grid=G, prev_obs=o0, cur_obs=o1)
    assert abs(est_fwd - fwd) < 0.3, (est_fwd, dyaw, score)
    assert score >= VO_MIN_SCORE, (est_fwd, score)
    print(f"VO clutter: fwd {est_fwd:.3f} m (true {fwd}) score {score:.3f} despite {3000} ego-fixed cells OK")


def check_vo_parked_flicker() -> None:
    """A parked ego plus >= 2% flickering cells must stay at rest with ok=True."""
    rng = np.random.default_rng(1)
    n_static = 600
    base = np.stack([rng.uniform(-20, 20, n_static), rng.uniform(0, 45, n_static)], 1)

    def keys(xy, z=1.2):
        pts = np.concatenate([xy, np.full((len(xy), 1), z)], 1)
        k = G.keys_of(pts)
        k = k[k >= 0]
        return k, np.zeros(k.size, np.uint8)

    for nfl in (12, 30, 60):  # 2%, 5%, 10%
        # one pair is enough for the odometry check; run a few ticks
        fwds, scores = [], []
        for t in range(5):
            a = np.concatenate([base, np.stack([rng.uniform(-20, 20, nfl), rng.uniform(0, 45, nfl)], 1)])
            b = np.concatenate([base, np.stack([rng.uniform(-20, 20, nfl), rng.uniform(0, 45, nfl)], 1)])
            f, y, sc = occ_odometry(*keys(a), *keys(b), grid=G)
            fwds.append(f)
            scores.append(sc)
            assert abs(f) < 0.1, (nfl, t, f, y, sc)
            assert sc >= VO_MIN_SCORE, (nfl, t, f, sc)  # ok=True
        print(f"VO parked+flicker {nfl}/{n_static} ({100 * nfl / n_static:.0f}%): "
              f"|fwd| max {max(abs(x) for x in fwds):.3f} m, score min {min(scores):.2f} OK")


def check_vo_parked_range_flicker() -> None:
    """Parked ego plus one-voxel range flicker must stay at rest with ok=True.

    Uniform-random flicker is already covered. Boundary smear sits on the
    surface, one voxel along the camera ray (forward y), and is not random:
    a residual path that accepts a sharp 1-voxel peak reports 0.4 m/frame
    (4 m/s at 10 Hz) with ok=True. One-sided and +-1 jitter at 5, 10, 17
    and 25% must keep |fwd| < 0.1.
    """
    rng = np.random.default_rng(7)
    n_static = 600
    v = float(G.voxel_m)

    def snap(xy):
        return (np.floor(np.asarray(xy, np.float64) / v) + 0.5) * v

    base = snap(np.stack([rng.uniform(-20, 20, n_static), rng.uniform(0, 45, n_static)], 1))

    def keys(xy, z=1.2):
        pts = np.concatenate([xy, np.full((len(xy), 1), z)], 1)
        k = G.keys_of(pts)
        k = np.unique(k[k >= 0])
        return k, np.zeros(k.size, np.uint8)

    def one_sided(n, r):
        idx = r.choice(len(base), n, replace=False)
        return base[idx] + np.array([0.0, v])

    def plus_minus(n, r):
        idx = r.choice(len(base), n, replace=False)
        sgn = r.choice([-1.0, 1.0], n)
        return base[idx] + np.stack([np.zeros(n), sgn * v], 1)

    for name, flick in (("one-sided +1 y", one_sided), ("+-1 y", plus_minus)):
        for frac in (0.05, 0.10, 0.17, 0.25):
            n = int(round(frac * n_static))
            fwds, scores = [], []
            for t in range(6):
                r = np.random.default_rng(70 + t + n)
                a = np.concatenate([base, flick(n, r)])
                b = np.concatenate([base, flick(n, r)])
                f, y, sc = occ_odometry(*keys(a), *keys(b), grid=G)
                fwds.append(f)
                scores.append(sc)
                assert abs(f) < 0.1, (name, frac, t, f, y, sc)
                assert sc >= VO_MIN_SCORE, (name, frac, t, f, sc)
            print(f"VO parked+{name} {n}/{n_static} ({100 * frac:.0f}%): "
                  f"|fwd| max {max(abs(x) for x in fwds):.3f} m, score min {min(scores):.2f} OK")



def check_vo_parked_structured() -> None:
    """Parked-ego structured disturbances must stay at rest with ok=True.

    Ports Skeptic's vo_probe3 / vo_lat probes: 2- and 3-voxel range smear,
    whole-wall depth jumps, a moving non-vehicle object, and lateral smear.
    The residual path accepted these as motion; full-set (observed) VO must
    not. |fwd| < 0.1 and |yaw| < 0.0025, score >= VO_MIN_SCORE, 6 ticks.
    """
    v = float(G.voxel_m)
    snap = lambda xy: (np.floor(np.asarray(xy, np.float64) / v) + 0.5) * v

    def keys(xy, z=1.2):
        pts = np.concatenate([xy, np.full((len(xy), 1), z)], 1)
        k = G.keys_of(pts)
        k = np.unique(k[k >= 0])
        return k, np.zeros(k.size, np.uint8)

    def run(tag, gen, n=6, seed=7):
        rng = np.random.default_rng(seed)
        fwds, yaws, scores = [], [], []
        for t in range(n):
            a, b = gen(rng, t)
            f, y, sc = occ_odometry(*keys(a), *keys(b), grid=G)
            fwds.append(f); yaws.append(y); scores.append(sc)
            assert abs(f) < 0.1, (tag, t, f, y, sc)
            assert abs(y) < 0.0025, (tag, t, f, y, sc)
            assert sc >= VO_MIN_SCORE, (tag, t, f, sc)
        print(f"VO parked+{tag}: |fwd| max {max(abs(x) for x in fwds):.3f} m, "
              f"|yaw| max {max(abs(x) for x in yaws):.4f} rad, score min {min(scores):.2f} OK")

    r0 = np.random.default_rng(1)
    base = snap(np.stack([r0.uniform(-20, 20, 600), r0.uniform(0, 45, 600)], 1))

    def smear(k, sided, nfl):
        def g(rng, t):
            fl = []
            for _ in range(2):
                idx = rng.choice(len(base), nfl, replace=False)
                sg = np.ones(nfl) if sided == 1 else rng.choice([-1.0, 1.0], nfl)
                fl.append(base[idx] + np.stack([np.zeros(nfl), sg * k * v], 1))
            return np.concatenate([base, fl[0]]), np.concatenate([base, fl[1]])
        return g

    for k in (1, 2, 3):
        for nfl in (30, 100, 200):
            run(f"one-sided {k}-voxel range smear n={nfl}", smear(k, 1, nfl))
            run(f"+-{k}-voxel range jitter n={nfl}", smear(k, 2, nfl))

    wall = snap(np.stack([np.linspace(-6, 6, 30).repeat(2), np.tile([20.0, 20.4], 30)], 1))
    for k in (1, 2, 3):
        def g(rng, t, k=k):
            return np.concatenate([base, wall]), np.concatenate([base, wall + [0, k * v]])
        run(f"whole wall depth jumps {k} voxel(s)", g)

    def gc(rng, t):
        c = snap(np.stack([rng.uniform(-4, 4, 40), rng.uniform(10, 14, 40)], 1))
        return np.concatenate([base, c]), base.copy()
    run("40-cell cluster vanishes", gc)

    obj = snap(np.stack([np.linspace(2, 4, 6).repeat(8), np.tile(np.linspace(5, 7.8, 8), 6)], 1))
    for vel in (0.6, 1.0, 2.0):
        def gm(rng, t, vel=vel):
            return np.concatenate([base, obj + [0, vel * t]]), np.concatenate([base, obj + [0, vel * (t + 1)]])
        run(f"48-cell other object moving {vel} m/frame", gm)

    def radial(n, r, k=1):
        idx = r.choice(len(base), n, replace=False)
        pt = base[idx]
        d = pt / np.linalg.norm(pt, axis=1, keepdims=True)
        return snap(pt + d * k * v)

    def lateral(n, r, k=1):
        idx = r.choice(len(base), n, replace=False)
        return base[idx] + np.array([k * v * np.sign(r.standard_normal()), 0])

    for name, fn in (("radial 1-voxel smear", radial), ("lateral 1-voxel smear", lateral)):
        for n in (30, 60, 100, 150):
            def g(rng, t, n=n, fn=fn):
                return np.concatenate([base, fn(n, rng)]), np.concatenate([base, fn(n, rng)])
            run(f"{name} n={n}", g)


def check_vo_slow_motion() -> None:
    """Slow real motion stays within a voxel of truth, with and without clutter.

    Ports Skeptic's vo_slow.py. Clean continuous-world points must match base
    (quantization, max |err| <= 0.2 m). With 3000 ego-fixed unobserved clutter
    cells, observed posts still track; no residual path is used.
    """
    r = np.random.default_rng(3)
    world = np.stack([r.uniform(-20, 20, 4000), r.uniform(-5, 50, 4000)], 1)
    clutter = (np.floor(np.stack([r.uniform(-6, 6, 3000), r.uniform(0, 30, 3000)], 1) / 0.4) + 0.5) * 0.4

    def keys_obs(xy, obs=None, z=1.2):
        pts = np.concatenate([xy, np.full((len(xy), 1), z)], 1)
        k = G.keys_of(pts)
        ok = k >= 0
        k = k[ok]
        c = np.zeros(k.size, np.uint8)
        if obs is None:
            return k, c, None
        return k, c, np.asarray(obs, bool)[ok]

    for clut in (False, True):
        for fwd in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.5):
            errs, oks = [], 0
            for t in range(8):
                y0 = t * fwd
                a = world - [0, y0]
                b = world - [0, y0 + fwd]
                if clut:
                    a_xy = np.concatenate([a[:300], clutter])
                    b_xy = np.concatenate([b[:300], clutter])
                    oa = np.concatenate([np.ones(300, bool), np.zeros(len(clutter), bool)])
                    ob = np.concatenate([np.ones(300, bool), np.zeros(len(clutter), bool)])
                    k0, c0, o0 = keys_obs(a_xy, oa)
                    k1, c1, o1 = keys_obs(b_xy, ob)
                    f, yw, sc = occ_odometry(k0, c0, k1, c1, grid=G, prev_obs=o0, cur_obs=o1)
                else:
                    k0, c0, _ = keys_obs(a)
                    k1, c1, _ = keys_obs(b)
                    f, yw, sc = occ_odometry(k0, c0, k1, c1, grid=G)
                errs.append(f - fwd)
                oks += int(sc >= VO_MIN_SCORE)
            err = np.asarray(errs)
            assert oks == 8, (clut, fwd, oks, err)
            assert float(np.max(np.abs(err))) <= 0.2 + 1e-9, (clut, fwd, err)
            print(f"VO slow {'clutter' if clut else 'clean  '} fwd {fwd:.2f}: "
                  f"mean err {float(np.mean(err)):+.3f} max|err| {float(np.max(np.abs(err))):.3f} ok {oks}/8 OK")


def check_featureless() -> None:
    em = EgoMotion()
    k, c = frame_cells(0)
    em.update(k, c, 0.0)
    k1, c1 = frame_cells(1)
    good = em.update(k1, c1, 0.1)
    road = c1 == 1
    held = em.update(k1[road], c1[road], 0.2)
    assert good.ok and not held.ok and held.speed_mps == good.speed_mps, (good, held)
    print("featureless frame: estimate held, ok=False OK")



def _xy_keys(xy, z=1.0):
    pts = np.column_stack([np.asarray(xy, np.float64), np.full(len(xy), z)])
    k = G.keys_of(pts)
    ok = k >= 0
    k, idx = np.unique(k[ok], return_index=True)
    return k, np.zeros(k.size, np.uint8)


def _fwd_cam() -> PinholeCam:
    return PinholeCam(
        "main", (0.0, 1.20, 1.26),
        (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0),
        640, 400, 50.0, 0.3, 80.0,
    )


def check_vo_score_gate() -> None:
    """Pin VO_MIN_SCORE from both sides (kills H1 0.43 and H2 0.3).

    A true-motion tick whose best overlap is in [0.40, 0.43) must be
    ok; a wrong or ambiguous tick in [0.3, 0.4) must hold.
    """
    # 100 unique static cells in a 10x10 grid ahead of the ego.
    xs, ys = np.meshgrid(np.linspace(-8, 8, 10), np.linspace(8, 24, 10), indexing="ij")
    base = np.column_stack([xs.ravel(), ys.ravel()])
    assert len(base) == 100
    k0, c0 = _xy_keys(base)

    def run(n_hit: int) -> tuple[object, float]:
        hit = base[:n_hit]
        miss = base[n_hit:] + np.array([0.0, 25.0])  # far ahead, unique match
        k1, c1 = _xy_keys(np.concatenate([hit, miss]))
        em = EgoMotion()
        em.update(k0, c0, 0.0)
        est = em.update(k1, c1, 0.1)
        _fwd, _yw, sc = occ_odometry(k0, c0, k1, c1, grid=G)
        return est, sc

    est_hi, sc_hi = run(40)
    assert 0.40 <= sc_hi < 0.43, sc_hi
    assert est_hi.ok, (est_hi, sc_hi, "score in [0.40, 0.43) must be ok (H1)")
    est_lo, sc_lo = run(35)
    assert 0.30 <= sc_lo < 0.40, sc_lo
    assert not est_lo.ok, (est_lo, sc_lo, "score in [0.3, 0.4) must hold (H2)")
    print(f"VO score gate: true-motion {sc_hi:.3f} ok={est_hi.ok}; "
          f"ambiguous {sc_lo:.3f} ok={est_lo.ok} OK")


def check_vo_min_cells_visible() -> None:
    """A visible region of 1-11 cells must hold, even if they all match (D4)."""
    cam = _fwd_cam()
    # 8 posts in the front frustum, 80 more behind the camera (out of view).
    front = np.column_stack([np.linspace(-2, 2, 8), np.full(8, 12.0)])
    rear = np.column_stack([np.linspace(-10, 10, 80), np.full(80, -6.0)])
    xy = np.concatenate([front, rear])
    k, c = _xy_keys(xy, z=1.0)
    # Occupied cells at z=1.0 also act as occluders (none in front of the 8).
    em = EgoMotion()
    em.update(k, c, 0.0, cams=[cam], occl=k)
    est = em.update(k, c, 0.1, cams=[cam], occl=k)
    _fwd, _yw, sc = occ_odometry(k, c, k, c, grid=G, cams=[cam], occl=k)
    assert sc == 0.0 and not est.ok, (sc, est, "1-11 visible cells must hold (D4)")
    # And the un-gated overlap on those 8 cells would have been perfect.
    from python.perception.occ_ego import _score, _bev_keys, _static_bev
    prev = _static_bev(k, c, G)
    cur_k = np.unique(_bev_keys(prev, G))
    cur_k = cur_k[cur_k >= 0]
    ungated = _score(prev, cur_k, G, 0.0, 0.0)
    assert ungated == 1.0, ungated
    print(f"VO min-cells: 8 visible cells score={sc} ok={est.ok} "
          f"(ungated overlap {ungated:.2f}) OK")


def check_vo_visibility_z() -> None:
    """Visibility height is 0.6 m, not the ground (kills D5 z=0.0)."""
    cam = _fwd_cam()
    # Posts behind a low wall: a ray to z=0.0 hits the wall, a ray to z=0.6
    # goes over it. Enough posts that 0.6 is ok and 0.0 drops under 12 cells.
    wall_xy = np.column_stack([np.linspace(-3, 3, 16), np.full(16, 18.0)])
    wall = G.keys_of(np.column_stack([wall_xy, np.full(16, 0.2)]))
    wall = np.unique(wall[wall >= 0])
    posts_xy = np.column_stack([np.linspace(-2.5, 2.5, 20), np.full(20, 20.0)])
    posts, pcls = _xy_keys(posts_xy, z=0.6)
    # Also a handful of posts with a clear line of sight (no wall), not enough
    # alone to pass VO_MIN_CELLS. The wall is occluder-only: it must not
    # join the VO cell set, or those cells stay visible at z=0.0 too.
    clear_xy = np.column_stack([np.linspace(8, 10, 5), np.full(5, 12.0)])
    clear, ccls = _xy_keys(clear_xy, z=0.6)
    k = np.unique(np.concatenate([posts, clear]))
    c = np.zeros(k.size, np.uint8)
    pts06 = np.column_stack([np.concatenate([posts_xy, clear_xy]),
                             np.full(25, 0.6)])
    pts00 = np.column_stack([np.concatenate([posts_xy, clear_xy]),
                             np.full(25, 0.0)])
    n06 = int(observed_mask(pts06, [cam], wall, G).sum())
    n00 = int(observed_mask(pts00, [cam], wall, G).sum())
    assert n06 >= VO_MIN_CELLS > n00, (n06, n00, VO_MIN_CELLS)
    em = EgoMotion()
    em.update(k, c, 0.0, cams=[cam], occl=wall)
    est = em.update(k, c, 0.1, cams=[cam], occl=wall)
    assert est.ok, (est, n06, n00, "default z=0.6 must see the posts over the wall (D5)")
    print(f"VO vis z: observed at 0.6 m={n06}, at 0.0 m={n00}; default ok={est.ok} OK")


def check_vo_observed_region_cost() -> None:
    """The current observed region is computed once per tick, not per hypothesis."""
    import time
    import python.perception.occ_ego as ego_mod
    import python.perception.occ_geometry as geom

    posts, _car, _road = world()
    # Two frames 1.5 m apart, with occluders = the static keys.
    def cells(i):
        p = pose(i)
        e = to_ego(posts, p)
        keys = []
        for z in (0.2, 0.6, 1.0, 1.4, 1.8):
            xyz = np.column_stack([e, np.full(len(e), z)])
            ins = G.inside(xyz)
            keys.append(G.keys_of(xyz[ins]))
        k = np.concatenate(keys)
        k = np.unique(k[k >= 0])
        return k, np.zeros(k.size, np.uint8)

    k0, c0 = cells(0)
    k1, c1 = cells(1)
    cam = _fwd_cam()
    occl = k1

    calls = {"n": 0}
    orig = ego_mod._observed_bev_keys

    def wrap(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    ego_mod._observed_bev_keys = wrap
    try:
        t0 = time.perf_counter()
        occ_odometry(k0, c0, k1, c1, grid=G, cams=[cam], occl=occl)
        ms_cams = (time.perf_counter() - t0) * 1e3
    finally:
        ego_mod._observed_bev_keys = orig
    assert calls["n"] == 1, (calls["n"], "observed region must be built once per tick")
    t0 = time.perf_counter()
    times = []
    for _ in range(3):
        t1 = time.perf_counter()
        occ_odometry(k0, c0, k1, c1, grid=G)
        times.append((time.perf_counter() - t1) * 1e3)
    ms_nocams = float(np.median(times))
    # Generous wall-clock bound: the old per-hypothesis path was ~80x
    # (~1.9 s vs ~24 ms). Stay within 15x of the no-cams path, and under 200 ms.
    assert ms_cams <= max(200.0, 15.0 * ms_nocams), (ms_cams, ms_nocams)
    print(f"VO cost: observed-region builds={calls['n']}, "
          f"with cams {ms_cams:.0f} ms, no cams {ms_nocams:.0f} ms OK")


def main() -> int:
    check_vo()
    check_vo_clutter()
    check_vo_parked_flicker()
    check_vo_parked_range_flicker()
    check_vo_parked_structured()
    check_vo_slow_motion()
    check_world_fixed()
    check_featureless()
    check_vo_score_gate()
    check_vo_min_cells_visible()
    check_vo_visibility_z()
    check_vo_observed_region_cost()
    print("test_occ_ego: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
