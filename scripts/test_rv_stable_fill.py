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


def test_flag_and_deps() -> None:
    check_flag_and_deps()


if __name__ == "__main__":
    for fn in (check_metrics, check_damped_view_less_jitter, check_damped_follows_turn,
               check_depth_filter_flicker_and_reset, check_compose_explains_every_pixel,
               check_classifier, check_ground_clamp, check_flag_and_deps):
        fn()
        print("ok", fn.__name__, flush=True)
    print("test_rv_stable_fill: all checks passed")
