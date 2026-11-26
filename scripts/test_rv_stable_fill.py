#!/usr/bin/env python3
"""Offline CPU checks for Reality View stable fill (python/viz/rv_stable_fill.py).

Spec (GVD ask, Reality View v2 "jumps around, black everywhere"):
S1 metrics: temporal_jitter is 0 on a static or linearly changing sequence and grows with
   frame-to-frame flicker; black_fraction counts UNKNOWN_RGB-ish and near-black pixels only.
S2 view steadiness: on the same wobbling ego heading, the damped chase view has clearly less
   temporal jitter than the rigid chase view, and still follows a steady turn (no lag past 3 s,
   no overshoot).
S3 depth steadiness: CamDepthFilter cuts per-pixel log-depth flicker on a static scene seen by a
   moving camera, and restarts (no smear) where the depth jumps by more than reset_dlog.
S4 every pixel explained: compose with sky + ground prior leaves no unknown pixel on a scene
   with gaps, sky and unseen near road; the same scene without them leaves black.
S5 classify_unknown: fractions sum to the unknown fraction and each cause lands where the
   geometry says (sky above horizon, never_seen outside every camera, splat_gap where a camera
   saw valid depth); a wrong mask shape raises.
S7 ground clamp: a camera cannot see below the road. Depth farther than the downward ray's
   ground-plane hit (wide cam 88.7 deg vFOV, pitch -8, 1.24 m up) is capped there (x margin);
   true road, nearer surfaces (hood, walls) and rays at/above the horizon are untouched.
S9 road is flat: net-labelled road pixels (>= 4 px inside the label, downward ray hitting within
   25 m, not > 0.30 m above the plane by the net's own depth, S11) are placed exactly on the ground plane, so the near road
   cannot bend down/up and two cameras' road meets; road label edges, road beyond 25 m, non-road and
   upward rays are left alone.
S11 snap rejection (runtime, no GT): a road-labelled px whose net depth puts it > 0.30 m above the road
   plane is not snapped (wall labelled road stays put; pole/car under a 12x16 road label keeps height).
S12 road snap runs only on ROAD_SNAP_CAMS (wide).
S7b ground_clamp caps +inf on a downward ray at the ground hit; NaN stays NaN.
S3b CamDepthFilter does not push on/above-road input below the road plane.
S8 draw distance: per-cam cap = min(GT far plane x 0.995, depth-head far class 250 m)
   (narrow 250, main/wide 250, 100 m cams 99.5); non-sky pixels beyond the cap or +inf are
   drawn AT the cap (kept, not dropped); sky stays sky; nearer depth is untouched.
S6 flag default OFF; numpy/cv2 only (no torch); DampedHeading rejects non-finite input.
"""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from python.viz import rv_stable_fill as sf  # noqa: E402
from python.viz.reality_view_c import UNKNOWN_RGB, camera_from_angles  # noqa: E402

MOD = ROOT / "python" / "viz" / "rv_stable_fill.py"


def _ground_scene(seed: int = 0, half: float = 40.0, step: float = 0.25):
    """Textured ground points (world frame) with a strong checker so view motion shows in luma."""
    xs = np.arange(-half, half, step) + step / 2
    gx, gy = np.meshgrid(xs, xs)
    pts = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], -1)
    chk = ((np.floor(gx / 1.0) + np.floor(gy / 1.0)) % 2).ravel()
    cols = np.where(chk[:, None] > 0, [200, 200, 200], [40, 60, 40]).astype(np.uint8)
    return pts, cols


def _world_to_ego(pts: np.ndarray, heading: float) -> np.ndarray:
    return sf.transform_points(pts, (0.0, 0.0, 0.0), (0.0, 0.0, heading))


def check_metrics() -> None:
    a = np.full((20, 30, 3), 100, np.uint8)
    assert sf.temporal_jitter([a, a, a, a]) == 0.0
    ramp = [np.full((20, 30, 3), v, np.uint8) for v in (10, 20, 30, 40)]
    assert sf.temporal_jitter(ramp) < 1e-9, "linear change is motion, not jitter"
    flick = [np.full((20, 30, 3), v, np.uint8) for v in (100, 110, 100, 110)]
    j = sf.temporal_jitter(flick)
    assert abs(j - 10.0) < 0.2, j
    try:
        sf.temporal_jitter([a, a])
        raise AssertionError("fewer than 3 frames must raise")
    except ValueError:
        pass
    img = np.full((10, 10, 3), 150, np.uint8)
    img[:2] = UNKNOWN_RGB
    img[2, :5] = (5, 5, 5)
    img[3, :5] = (UNKNOWN_RGB[0] + 7, UNKNOWN_RGB[1], UNKNOWN_RGB[2])  # outside tol, not near-black
    assert abs(sf.black_fraction(img) - 0.25) < 1e-9, sf.black_fraction(img)


def _render_seq(pts_w, cols, headings, damped: bool, w=160, h=90):
    dh = sf.DampedHeading(omega=1.5)
    frames, offs = [], []
    foot = np.full(len(pts_w), 0.25 * 1.5)
    for hd in headings:
        tgt = dh.step(hd, 0.2)
        off = (tgt - hd) if damped else 0.0
        offs.append(off)
        cam = sf.chase_camera((0.0, -4.0, 2.6), -12.0, 60.0, w, h, off)
        r = sf.compose(cam, _world_to_ego(pts_w, hd), cols, foot, sky_rgb=(120, 150, 190), ground_rgb=(90, 90, 90))
        frames.append(r.rgb)
    return frames, offs


def check_damped_view_less_jitter() -> None:
    pts, cols = _ground_scene()
    n = 24
    # car wobble measured on clip015: ~0.5 deg per frame alternating, on a slow steady turn
    headings = [0.002 * i + math.radians(0.5) * (1 if i % 2 else -1) for i in range(n)]
    rig, _ = _render_seq(pts, cols, headings, damped=False)
    dmp, offs = _render_seq(pts, cols, headings, damped=True)
    jr, jd = sf.temporal_jitter(rig[4:]), sf.temporal_jitter(dmp[4:])
    assert jr > 1.0, f"scene too flat to measure (rigid jitter {jr})"
    assert jd < 0.5 * jr, f"damped jitter {jd:.3f} not < half rigid {jr:.3f}"
    assert max(abs(o) for o in offs) < math.radians(3.0)


def check_damped_follows_turn() -> None:
    dh = sf.DampedHeading(omega=1.5)
    dh.step(0.0, 0.2)
    hs = [dh.step(math.radians(30.0), 0.2) for _ in range(15)]  # 3 s after a 30 deg step
    assert all(h <= math.radians(30.0) + 1e-9 for h in hs), "critically damped: no overshoot"
    # analytic critically damped step residual: (1 + w t) exp(-w t); w=1.5 -> 6.1% at 3 s, 0.6% at 5 s
    res3 = (1 + 1.5 * 3.0) * math.exp(-1.5 * 3.0)
    assert abs((math.radians(30.0) - hs[-1]) / math.radians(30.0) - res3) < 0.01, math.degrees(hs[-1])
    hs += [dh.step(math.radians(30.0), 0.2) for _ in range(10)]
    assert (math.radians(30.0) - hs[-1]) / math.radians(30.0) < 0.01, math.degrees(hs[-1])
    # wraps the short way across +-pi
    dw = sf.DampedHeading(omega=1.5)
    dw.step(math.pi - 0.05, 0.2)
    for _ in range(20):
        x = dw.step(-math.pi + 0.05, 0.2)
    assert abs(math.atan2(math.sin(x - (-math.pi + 0.05)), math.cos(x - (-math.pi + 0.05)))) < 0.01
    for bad in (float("nan"), float("inf")):
        try:
            sf.DampedHeading().step(bad, 0.2)
            raise AssertionError("non-finite heading must raise")
        except ValueError:
            pass


def check_depth_filter_flicker_and_reset() -> None:
    rng = np.random.default_rng(3)
    cam = camera_from_angles("front", (0.0, 1.5, 1.4), 0.0, 0.0, 50.0, 64, 48, near_m=0.1, far_m=200.0)
    # static wall 40 m ahead in world; the car drives forward 1 m per frame (5 m/s at 0.2 s)
    f = sf.CamDepthFilter()
    raw_d, filt_d = [], []
    for k in range(12):
        true = 40.0 - k * 1.0
        d = np.full((48, 64), true) * np.exp(rng.normal(0, 0.08, (48, 64)))
        val = np.ones((48, 64), bool)
        out, w = f.update(cam, d, val, (0.0, float(k), 0.0), 0.2 * k)
        raw_d.append(np.log(d) - math.log(true))
        filt_d.append(np.log(out) - math.log(true))
    raw_err = float(np.mean(np.abs(np.stack(raw_d[4:]))))
    filt_err = float(np.mean(np.abs(np.stack(filt_d[4:]))))
    assert filt_err < 0.6 * raw_err, (filt_err, raw_err)
    # a pedestrian steps in: depth halves in the left half -> must take the new value now
    jump = np.full((48, 64), 40.0 - 12 * 1.0)
    jump[:, :32] *= 0.4
    out, w = f.update(cam, jump, np.ones((48, 64), bool), (0.0, 12.0, 0.0), 2.4)
    assert np.allclose(out[:, :32], jump[:, :32]), "a jump > reset_dlog must not be smeared"
    assert float(w[:, :32].max()) <= 1.0 + 1e-9


def check_compose_explains_every_pixel() -> None:
    cam = sf.chase_camera((0.0, -4.0, 2.6), -12.0, 60.0, 160, 90, 0.0)
    pts, cols = _ground_scene(half=130.0, step=0.5)
    keep = (pts[:, 1] > 6.0) & (np.arange(len(pts)) % 7 != 0)  # unseen near road + sparse gaps
    pts, cols = pts[keep], cols[keep]
    foot = np.full(len(pts), 0.5)
    bare = sf.compose(cam, pts, cols, foot, sky_rgb=None, ground_rgb=None, fill=False)
    full = sf.compose(cam, pts, cols, foot, sky_rgb=(120, 150, 190), ground_rgb=(90, 90, 90), fill=True)
    b0, b1 = sf.black_fraction(bare.rgb), sf.black_fraction(full.rgb)
    assert b0 > 0.3, f"before: scene should show black ({b0})"
    assert b1 < 0.01, f"after: black {b1}"
    assert full.fraction("unknown") < 0.01
    tot = sum(full.masks[k].astype(np.int64) for k in sf.MASK_NAMES)
    assert int(tot.max()) == 1 and int(tot.min()) == 1, "masks must partition the image"
    assert full.fraction("sky") > 0.05 and full.fraction("ground_prior") > 0.05


def check_classifier() -> None:
    chase = sf.chase_camera((0.0, -4.0, 2.6), -12.0, 60.0, 80, 45, 0.0)
    front = camera_from_angles("front", (0.0, 1.5, 1.4), 0.0, 0.0, 50.0, 64, 48, near_m=0.1, far_m=200.0)
    dep = np.full((48, 64), 50.0)
    val = np.ones((48, 64), bool)
    unk = np.ones((45, 80), bool)
    ev = [sf.CamEvidence(front, dep, val)]
    c = sf.classify_unknown(chase, unk, ev)
    assert abs(sum(c.values()) - 1.0) < 1e-9
    assert c["sky"] > 0.05 and c["never_seen"] > 0.05, c
    # a camera that sees ground at the right range everywhere -> visible ground is splat_gap
    big = sf.classify_unknown(chase, unk, [sf.CamEvidence(front, np.full((48, 64), 1e4), val)])
    assert big["splat_gap"] > 0.05 and big["occluded"] == 0.0, big
    none = sf.classify_unknown(chase, unk, [sf.CamEvidence(front, dep, np.zeros_like(val))])
    assert none["invalid_depth"] > 0.05 and none["splat_gap"] == 0.0, none
    near = sf.classify_unknown(chase, unk, [sf.CamEvidence(front, np.full((48, 64), 0.5), val)])
    assert near["occluded"] > 0.05, near
    assert sum(sf.classify_unknown(chase, np.zeros_like(unk), ev).values()) == 0.0
    try:
        sf.classify_unknown(chase, np.ones((44, 80), bool), ev)
        raise AssertionError("wrong mask shape must raise")
    except ValueError:
        pass


def check_ground_clamp() -> None:
    cam = camera_from_angles("wide", (-0.03, 1.2, 1.24), 0.0, -8.0, 88.7, 320, 180, near_m=0.05, far_m=400.0)
    hit, _ = sf.ground_clamp(cam, np.full((180, 320), 1e9), margin=1.0)
    # hand geometry, bottom-centre pixel: elevation = pitch - atan((v - cy)/fy)
    v = 179
    fy = 90.0 / math.tan(math.radians(88.7 / 2))
    a = math.atan((v + 0.5 - 90.0) / fy)
    el = math.radians(8.0) + a
    want = 1.24 / math.sin(el) * math.cos(a)
    assert abs(hit[v, 160] - want) / want < 1e-3, (hit[v, 160], want)
    hor = hit >= 1e9  # untouched input
    assert hor[0].all() and not hor[-1].any(), "rays at/above the horizon must stay unclamped"
    # true road stays, 2.5x-too-far road is capped, nearer things (hood/walls) stay
    road = np.where(hor, 50.0, hit)
    out, m = sf.ground_clamp(cam, road)
    assert not m.any() and np.allclose(out, road)
    far = road * 2.5
    out, m = sf.ground_clamp(cam, far)
    assert m[~hor].all() and np.allclose(out[~hor], hit[~hor] * 1.1)
    near = np.where(hor, 50.0, hit * 0.4)
    out, m = sf.ground_clamp(cam, near)
    assert not m.any()
    try:
        sf.ground_clamp(cam, np.ones((179, 320)))
        raise AssertionError("wrong depth shape must raise")
    except ValueError:
        pass


def check_far_drawn_at_cap() -> None:
    assert sf.depth_cap_m(800.0) == 250.0 and sf.depth_cap_m(300.0) == 250.0
    assert abs(sf.depth_cap_m(100.0) - 99.5) < 1e-9
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        try:
            sf.depth_cap_m(bad)
            raise AssertionError(f"far_m {bad} must raise")
        except ValueError:
            pass
    d = np.array([[10.0, 249.0, 260.0, np.inf], [400.0, 655.35, 5.0, np.inf]])
    sky = np.array([[False, False, False, False], [False, False, False, True]])
    out, far = sf.far_to_cap(d, sky, 250.0)
    assert far.tolist() == [[False, False, True, True], [True, True, False, False]]
    assert np.allclose(out[far], 250.0), "far pixels must be drawn at the cap, not dropped"
    assert out[0, 0] == 10.0 and out[0, 1] == 249.0 and out[1, 2] == 5.0
    assert np.isinf(out[1, 3]), "sky stays sky"
    # end to end: a building 400 m out (beyond the old 120 m cut) survives into the chase render
    cam = camera_from_angles("main", (0.0, 1.2, 1.26), 0.0, 0.0, 34.0, 64, 48, near_m=0.05, far_m=400.0)
    dep, _ = sf.far_to_cap(np.full((48, 64), 400.0), np.zeros((48, 64), bool), sf.depth_cap_m(300.0))
    ys, xs = np.nonzero(np.ones((48, 64), bool))
    pts = sf.unproject(cam, dep, ys, xs)
    chase = sf.chase_camera((0.0, -4.0, 2.6), -12.0, 60.0, 160, 90, 0.0)
    r = sf.compose(chase, pts, np.full((len(pts), 3), 200, np.uint8), np.full(len(pts), 8.0),
                   sky_rgb=None, ground_rgb=None, fill=False)
    assert r.masks["observed"].sum() > 20, "far building must be drawn"
    try:
        sf.far_to_cap(d, sky[:, :3], 250.0)
        raise AssertionError("shape mismatch must raise")
    except ValueError:
        pass


def check_road_snap() -> None:
    cam = camera_from_angles("wide", (0.0, 1.5, 1.24), 0.0, -8.0, 88.7, 160, 90, near_m=0.05, far_m=300.0)
    ys, xs = np.mgrid[0:90, 0:160]
    rays = sf._pixel_rays(cam); dz = rays[..., 2]
    hit = np.where(dz < -1e-6, 1.24 / np.where(dz < -1e-6, -dz, 1.0), np.inf)
    # net-like bent road: 2.5x too far at the bottom, 0.5x too near mid-way (the clip015 tunnel shape)
    bend = np.interp(ys, [0, 45, 89], [1.0, 0.5, 2.5])
    raw = np.where(np.isfinite(hit), hit * bend, 80.0)
    road = (ys >= 20) & (xs >= 30) & (xs < 130)  # reaches past ROAD_SNAP_MAX_M near the horizon
    assert sf.ROAD_SNAP_MAX_M == 25.0
    R = 20.0  # this small fixture's horizon is only ~60 m out, so test the range gate at 20 m
    out, m = sf.road_snap(cam, raw, road, max_hit_m=R)
    above = (cam.pos[2] + raw * dz) > sf.ROAD_SNAP_MAX_ABOVE_M   # net already puts it > 0.30 m up: not road (S11)
    core = sf._erode(road, sf.ROAD_SNAP_ERODE_PX) & (dz < -1e-6) & (hit <= R) & ~above
    assert m.tolist() == core.tolist() and core.sum() > 1000
    assert (sf._erode(road, sf.ROAD_SNAP_ERODE_PX) & (hit <= R) & above).any(), "fixture must have a too-near band"
    z = cam.pos[2] + out * dz
    assert np.max(np.abs(z[m])) < 1e-9, "snapped road must lie on the ground plane"
    zr = cam.pos[2] + raw * dz
    assert np.max(np.abs(zr[core])) > 0.5, "fixture must actually bend"
    assert np.array_equal(out[~m], raw[~m]), "non-road / label edge / far road untouched"
    edge = road & ~sf._erode(road, sf.ROAD_SNAP_ERODE_PX)
    assert edge.any() and not m[edge].any(), "label edge is not snapped"
    far_road = road & (hit > R) & np.isfinite(hit)
    assert far_road.any() and not m[far_road].any(), "road beyond the range gate is not snapped"
    try:
        sf.road_snap(cam, raw, road[:, :10])
        raise AssertionError("shape mismatch must raise")
    except ValueError:
        pass


def check_ownership() -> None:
    # each direction is drawn by one camera: a wide pixel yields exactly when its ray lands >= margin
    # inside the better (main) image; outside main's view the wide keeps its pixels
    wide = camera_from_angles("wide", (0.0, 1.5, 1.24), 0.0, 0.0, 120.0, 120, 60, near_m=0.05, far_m=300.0)
    main = camera_from_angles("main", (0.0, 1.5, 1.26), 0.0, 0.0, 50.0, 80, 40, near_m=0.05, far_m=300.0)
    y = sf.yield_mask(wide, [main], margin_px=2.0)
    assert y.shape == (60, 120) and 0.05 < y.mean() < 0.5
    assert y[30, 60] and not y[30, 2] and not y[30, 117], "centre yields, edges keep"
    rays = sf._pixel_rays(wide).reshape(-1, 3)
    u, v, z = sf.project(main, np.asarray(wide.pos) + rays * 200.0 / np.linalg.norm(rays, axis=1, keepdims=True))
    inside = (z > 0) & (u >= 2) & (u <= 78) & (v >= 2) & (v <= 38)
    assert np.array_equal(y.ravel(), inside), "yield must equal 'ray covered by main'"
    assert not sf.yield_mask(wide, []).any(), "no better camera, nothing yields"
    # end to end: with ownership the overlap draws main's (correct) surface, not wide's too-near copy
    S = 12
    ys, xs = np.nonzero(y)
    wpts = sf.unproject(wide, np.full((60, 120), 4.0), ys, xs)   # wide's too-near copy of a wall at 10 m
    my, mx = np.nonzero(np.ones((40, 80), bool))
    mpts = sf.unproject(main, np.full((40, 80), 10.0), my, mx)
    chase = sf.chase_camera((0.0, -4.0, 2.6), -12.0, 60.0, 160, 90, 0.0)
    def draw(own):
        pts = mpts if own else np.vstack([mpts, wpts])
        col = np.vstack([np.tile([0, 255, 0], (len(mpts), 1)), np.tile([255, 0, 0], (0 if own else len(wpts), 1))]).astype(np.uint8)
        return sf.compose(chase, pts, col, np.full(len(pts), 3.0), sky_rgb=None, ground_rgb=None, fill=False).rgb
    red = lambda im: int(((im[..., 0] > 200) & (im[..., 1] < 50)).sum())
    assert red(draw(False)) > S and red(draw(True)) == 0


def _flat_scene(cid, pos, pitch, fov, w=160, h=90):
    cam = camera_from_angles(cid, pos, 0.0, pitch, fov, w, h, near_m=0.05, far_m=300.0)
    ys, xs = np.mgrid[0:h, 0:w]
    dz = sf._pixel_rays(cam)[..., 2]
    down = dz < -1e-6
    hit = np.where(down, cam.pos[2] / np.where(down, -dz, 1.0), np.inf)
    return cam, ys, xs, dz, down, hit


def check_snap_rejects_above_road() -> None:
    """S11 (Skeptic/GVD): runtime-only snap rejection, no GT. A road-labelled px whose net depth puts it
    more than ROAD_SNAP_MAX_ABOVE_M (0.30 m) above the road plane (net depth much shorter than its
    ground-ray hit) is not snapped: a wall labelled road stays put, a pole/car labelled road at the
    12x16 class grid keeps its height through road_snap + ground_clamp; true road still snaps."""
    assert sf.ROAD_SNAP_MAX_ABOVE_M == 0.30
    # wall labelled road: a vertical face 6 m ahead filling the lower image, the whole frame labelled road
    cam, ys, xs, dz, down, hit = _flat_scene("wide", (0.0, 1.2, 1.24), -8.0, 88.7)
    wall = np.full(hit.shape, 6.0)
    z = cam.pos[2] + wall * dz
    face = down & (z > 0.31)
    assert face.sum() > 500
    out, m = sf.road_snap(cam, wall, np.ones(hit.shape, bool))
    assert not m[face].any() and np.array_equal(out[face], wall[face]), "wall labelled road was snapped"
    road_part = down & (z < 0.0)             # wall depth lies past the ground there: true under-road dip, snap it
    assert m[road_part & sf._erode(np.ones(hit.shape, bool), sf.ROAD_SNAP_ERODE_PX) & (hit <= sf.ROAD_SNAP_MAX_M)].all()
    # pole / car on exact road, road label at the 12x16 class grid (majority per cell)
    for cid, pos, pitch, fov in (("wide", (0.0, 1.2, 1.24), -8.0, 88.7), ("main", (0.0, 1.2, 1.26), -6.0, 34.0)):
        cam, ys, xs, dz, down, hit = _flat_scene(cid, pos, pitch, fov)
        H, W = hit.shape
        D = 8.0
        zobj = cam.pos[2] + D * dz
        obj = (xs >= 70) & (xs < 82) & down & (hit > D) & (zobj < 1.5)
        assert obj.sum() > 30, (cid, obj.sum())
        depth = np.where(obj, D, np.where(down, hit, 80.0))
        cell = np.zeros((12, 16)); cnt = np.zeros((12, 16))
        cy, cx = ys * 12 // H, xs * 16 // W
        np.add.at(cell, (cy, cx), (~obj).astype(float)); np.add.at(cnt, (cy, cx), 1.0)
        road = (cell / cnt > 0.5)[cy, cx] & down
        assert road[obj].any(), "fixture: the coarse label must cover part of the object"
        out, m = sf.road_snap(cam, depth, road)
        out, cl = sf.ground_clamp(cam, out, margin=1.0)
        hi = obj & (zobj > sf.ROAD_SNAP_MAX_ABOVE_M)
        zo = cam.pos[2] + out * dz
        assert np.max(np.abs(zo[hi] - zobj[hi])) < 1e-9, f"{cid}: object px above 0.30 m lost height"
        assert m[road & ~obj & sf._erode(road, sf.ROAD_SNAP_ERODE_PX)].any(), "true road must still snap"


def check_snap_cams() -> None:
    """S12: road snap runs only on ROAD_SNAP_CAMS (wide): main/narrow net depth beats the flat z=0
    plane at 10-15 m on clip015 (f40/f160 gate, f100/f200 held-out)."""
    assert tuple(sf.ROAD_SNAP_CAMS) == ("wide",), sf.ROAD_SNAP_CAMS
    cam, ys, xs, dz, down, hit = _flat_scene("wide", (0.0, 1.2, 1.24), -8.0, 88.7)
    raw = np.where(down, hit * 1.5, 80.0)
    road = down.copy()
    o, m = sf.snap_road_for_cam("wide", cam, raw, road)
    assert m.any()
    for cid in ("main", "narrow", "pillarL", "pillarR", "repeatL", "repeatR", "rear"):
        o, m = sf.snap_road_for_cam(cid, cam, raw, road)
        assert not m.any() and np.array_equal(o, raw), cid


def check_clamp_inf() -> None:
    """S7b: +inf on a downward ray is under the road too: ground_clamp caps it at the ground hit
    (Breaker: far_to_cap otherwise drew it at the cap, z=-274 m). Up rays keep +inf (far_to_cap's job)."""
    cam, ys, xs, dz, down, hit = _flat_scene("wide", (0.0, 1.2, 1.24), -8.0, 88.7)
    d = np.full(hit.shape, np.inf)
    out, m = sf.ground_clamp(cam, d, margin=1.0)
    assert m[down].all() and np.allclose(out[down], hit[down])
    assert np.isposinf(out[~down]).all() and not m[~down].any()
    nan = np.full(hit.shape, np.nan)
    out, m = sf.ground_clamp(cam, nan, margin=1.0)
    assert np.isnan(out).all() and not m.any(), "NaN stays invalid"


def check_filter_keeps_ground() -> None:
    """S3b (Breaker F): CamDepthFilter never puts a pixel below the road plane that its input had on or
    above it (the floor-lookup blend of a neighbouring ray's carried point dipped snapped road 5 cm)."""
    cam, ys, xs, dz, down, hit = _flat_scene("wide", (-0.03, 1.2, 1.24), -8.0, 88.7, 320, 180)
    depth = np.where(down, hit, 300.0)
    valid = down & (hit < 60.0)
    F = sf.CamDepthFilter()
    for k in range(3):
        d, _ = F.update(cam, depth, valid, (0.0, 1.0 * k, 0.0), 0.1 * k)
    z = cam.pos[2] + d * dz
    assert np.min(z[valid]) >= -1e-6, float(np.min(z[valid]))


def check_flag_and_deps() -> None:
    assert sf.STABLE_FILL_DEFAULT_ON is False
    tree = ast.parse(MOD.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    assert not (mods & {"torch", "tensorflow", "onnxruntime", "cupy"}), mods


def test_metrics() -> None:
    check_metrics()


def test_damped_view_less_jitter() -> None:
    check_damped_view_less_jitter()


def test_damped_follows_turn() -> None:
    check_damped_follows_turn()


def test_depth_filter() -> None:
    check_depth_filter_flicker_and_reset()


def test_compose_explains_every_pixel() -> None:
    check_compose_explains_every_pixel()


def test_classifier() -> None:
    check_classifier()


def test_ground_clamp() -> None:
    check_ground_clamp()


def test_far_drawn_at_cap() -> None:
    check_far_drawn_at_cap()


def test_road_snap() -> None:
    check_road_snap()
    check_ownership()


def test_snap_rejects_above_road() -> None:
    check_snap_rejects_above_road()


def test_snap_cams() -> None:
    check_snap_cams()


def test_clamp_inf() -> None:
    check_clamp_inf()


def test_filter_keeps_ground() -> None:
    check_filter_keeps_ground()


def test_flag_and_deps() -> None:
    check_flag_and_deps()


if __name__ == "__main__":
    for fn in (check_metrics, check_damped_view_less_jitter, check_damped_follows_turn,
               check_depth_filter_flicker_and_reset, check_compose_explains_every_pixel,
               check_classifier, check_ground_clamp, check_far_drawn_at_cap, check_road_snap, check_ownership,
               check_snap_rejects_above_road, check_snap_cams, check_clamp_inf, check_filter_keeps_ground, check_flag_and_deps):
        fn()
        print("ok", fn.__name__, flush=True)
    print("test_rv_stable_fill: all checks passed")
