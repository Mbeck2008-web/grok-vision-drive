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

from python.perception.occ_ego import EgoMotion, _static_bev, _warp, occ_odometry  # noqa: E402
from python.perception.occ_geometry import DEFAULT_GRID  # noqa: E402
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
    em = EgoMotion()
    assert em.source == "camera_vo" and not em.from_vehicle
    px, py, h = 0.0, 0.0, 0.0
    posts_w = None
    drift = []
    tr = OccTracker()
    reps = []
    for i in range(10):
        k, c = frame_cells(i)
        est = em.update(k, c, i * DT, vehicle=None)
        if i > 0:
            assert est.ok, i
            f, y = est.speed_mps * DT, est.yaw_rate * DT
            px += -math.sin(h) * f; py += math.cos(h) * f; h += y
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
    cars = [r for r in reps[-1] if r["frames"] >= 5]
    assert len(cars) == 1 and cars[0]["motion"] == "parked", reps[-1]
    print(f"world-fixed: VO pose after 9 frames within {math.hypot(px - tpx, py - tpy):.2f} m / {abs(h - th):.4f} rad, "
          f"static cells on their true world cell (+-1) every frame >= {min(vo_frac):.2f} (no warp: {still_frac[-1]:.2f} by frame 9), "
          f"parked car stays parked OK")


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


def main() -> int:
    check_vo()
    check_world_fixed()
    check_featureless()
    print("test_occ_ego: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
