#!/usr/bin/env python3
"""Offline tests for Reality View step 5 (approach A generator, CPU).

Spec: docs/specs/reality_view_3dgs.md (Feasibility plan, step 5).
Fixture predicted-depth tensor + RGB + K -> SH0 Gaussians -> offline render
from a canned chase pose. No sim GT (the depth is an analytic fixture, not
BeamNG output), no GPU, no new deps.

Tests:
- check_n_cap
- check_unknown_not_free
- check_deterministic_cpu
- check_canned_pose_honesty_pin
- check_offline_only_no_new_deps
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / "tests" / "fixtures" / "rv_a" / "canned_render.npz"

# fixture source camera (OpenCV cam frame: +x right, +y down, +z forward)
W, H = 64, 48
K = np.array([[48.0, 0.0, 31.5], [0.0, 48.0, 23.5], [0.0, 0.0, 1.0]])
CAM_H = 1.26          # camera height above ground (m)
WALL_Z = 12.0         # wall plane depth (m)
WALL_HALF_W = 2.0
WALL_TOP = 1.0        # wall top is 1.0 m above the camera (y = -1.0)
GROUND_RGB = (90, 90, 90)
LANE_RGB = (235, 235, 235)
WALL_RGB = (200, 40, 30)


def fixture_scene() -> tuple[np.ndarray, np.ndarray]:
    """Analytic predicted-depth fixture: ground plane + one wall; sky has no depth (inf)."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    xn = (u - K[0, 2]) / K[0, 0]
    yn = (v - K[1, 2]) / K[1, 1]
    z = np.full((H, W), np.inf)
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    with np.errstate(divide="ignore", invalid="ignore"):
        zg = np.where(yn > 0, CAM_H / yn, np.inf)
    ground = np.isfinite(zg)
    z[ground] = zg[ground]
    rgb[ground] = GROUND_RGB
    lane = ground & (np.abs(xn * zg - 1.5) < 0.15)
    rgb[lane] = LANE_RGB
    wall = (np.abs(xn * WALL_Z) < WALL_HALF_W) & (yn * WALL_Z > -WALL_TOP) & (WALL_Z < z)
    z[wall] = WALL_Z
    rgb[wall] = WALL_RGB
    return z.astype(np.float32), rgb


def canned_chase():
    from python.viz.rv_a_gen import ChaseCam, look_at_w2c

    k = np.array([[40.0, 0.0, 39.5], [0.0, 40.0, 29.5], [0.0, 0.0, 1.0]])
    eye = np.array([0.0, -4.0, -6.0])      # 4 m above, 6 m behind the source cam
    target = np.array([0.0, CAM_H, 30.0])  # ground point 30 m ahead
    return ChaseCam(K=k, w2c=look_at_w2c(eye, target), width=80, height=60, near_m=0.1, far_m=120.0)


def check_n_cap() -> None:
    """A generator cannot exceed N_max; the cap is hard and validated."""
    from python.viz import rv_a_gen as g

    # 640x480 all-valid: the default cap must hold and never exceed the hard cap.
    big_d = np.full((480, 640), 10.0, dtype=np.float32)
    big_rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    big_k = np.array([[480.0, 0.0, 319.5], [0.0, 480.0, 239.5], [0.0, 0.0, 1.0]])
    full = g.generate_sh0(big_d, big_rgb, big_k, n_max=g.N_MAX_HARD)
    assert 0 < full.n <= g.N_MAX_HARD, full.n
    assert full.n <= 640 * 480

    for n_max in (1, 7, 1000, 4096, 50_000):
        s = g.generate_sh0(big_d, big_rgb, big_k, n_max=n_max)
        assert 0 < s.n <= n_max, (n_max, s.n)
        for name in ("means", "scales", "quats", "opacity", "sh0"):
            arr = getattr(s, name)
            assert arr.shape[0] == s.n, (name, arr.shape, s.n)
            assert arr.dtype == np.float32, (name, arr.dtype)
            assert np.all(np.isfinite(arr)), name
        # subsampling keeps coverage: summed splat footprint ~ the full image footprint
        area_full = float(np.sum(full.scales[:, 0].astype(np.float64) * full.scales[:, 1]))
        area_s = float(np.sum(s.scales[:, 0].astype(np.float64) * s.scales[:, 1]))
        if n_max >= 1000:
            assert 0.5 <= area_s / area_full <= 2.0, (n_max, area_s / area_full)
            # and stays spatially even: every 32x32 px tile of the source image keeps a splat
            pu = np.floor(s.means[:, 0] / s.means[:, 2] * big_k[0, 0] + big_k[0, 2] + 0.5).astype(int)
            pv = np.floor(s.means[:, 1] / s.means[:, 2] * big_k[1, 1] + big_k[1, 2] + 0.5).astype(int)
            tiles = np.zeros((480 // 32, 640 // 32), dtype=bool)
            tiles[pv // 32, pu // 32] = True
            assert tiles.all(), (n_max, int((~tiles).sum()), "empty 32px tiles after cap")

    # the hard cap cannot be raised by a caller, and a bad cap fails loudly
    for bad in (g.N_MAX_HARD + 1, 0, -5, True, 3.5, None):
        try:
            g.generate_sh0(big_d, big_rgb, big_k, n_max=bad)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"n_max={bad!r} must be rejected")

    # a map/merge of two generations is still capped by the caller's N_max
    m = g.concat_capped([full, full], n_max=g.N_MAX_HARD)
    assert m.n <= g.N_MAX_HARD, m.n
    m2 = g.concat_capped([s, s, s], n_max=1000)
    assert m2.n <= 1000, m2.n


def check_unknown_not_free() -> None:
    """Unknown / invalid depth never becomes a Gaussian or an opaque rendered surface."""
    from python.viz import rv_a_gen as g

    k = np.array([[12.0, 0.0, 7.5], [0.0, 12.0, 5.5], [0.0, 0.0, 1.0]])
    rgb = np.full((12, 16, 3), 255, dtype=np.uint8)
    chase = g.ChaseCam(K=np.array([[20.0, 0, 15.5], [0, 20.0, 11.5], [0, 0, 1.0]]),
                       w2c=np.eye(4), width=32, height=24, near_m=0.05, far_m=100.0)
    unknown = np.asarray(g.UNKNOWN_RGB, dtype=np.uint8)
    bad_depths = [
        ("nan", np.full((12, 16), np.nan, dtype=np.float32), None),
        ("pos_inf", np.full((12, 16), np.inf, dtype=np.float32), None),
        ("neg_inf", np.full((12, 16), -np.inf, dtype=np.float32), None),
        ("zero", np.zeros((12, 16), dtype=np.float32), None),
        ("negative", np.full((12, 16), -3.0, dtype=np.float32), None),
        ("below_near", np.full((12, 16), 0.10, dtype=np.float32), None),
        ("beyond_far", np.full((12, 16), 50.5, dtype=np.float32), None),
        ("masked_out", np.full((12, 16), 10.0, dtype=np.float32), np.zeros((12, 16), dtype=bool)),
    ]
    for label, d, mask in bad_depths:
        s = g.generate_sh0(d, rgb, k, valid_mask=mask, near_m=0.2, far_m=50.0, n_max=1000)
        assert s.n == 0, f"{label} leaked {s.n} Gaussians"
        r = g.render_sh0(s, chase)
        assert not r.observed.any(), f"{label} produced observed pixels"
        assert np.all(r.alpha == 0.0), f"{label} produced opacity"
        assert np.all(r.depth_m == 0.0), f"{label} produced depth"
        assert np.all(r.rgb == unknown), f"{label} painted non-unknown color"

    # mixed: only the one valid pixel becomes a Gaussian, at the right place
    d = np.full((12, 16), np.nan, dtype=np.float32)
    d[6, 8] = 10.0
    s = g.generate_sh0(d, rgb, k, near_m=0.2, far_m=50.0, n_max=1000)
    assert s.n == 1, s.n
    want = np.array([(8 - 7.5) / 12.0 * 10.0, (6 - 5.5) / 12.0 * 10.0, 10.0])
    assert np.allclose(s.means[0], want, atol=1e-5), (s.means[0], want)

    # left half unknown: nothing generated from it, and the chase view of that half stays unknown
    d = np.full((12, 16), 10.0, dtype=np.float32)
    d[:, :8] = np.nan
    s = g.generate_sh0(d, rgb, k, near_m=0.2, far_m=50.0, n_max=1000)
    assert s.n == 12 * 8, s.n
    assert float(s.means[:, 0].min()) > 0.0, "a Gaussian came from the unknown half"
    r = g.render_sh0(s, chase)
    assert r.observed[:, 16:].mean() > 0.5, "valid half not rendered"
    assert not r.observed[:, :14].any(), "unknown half rendered as surface"
    assert np.all(r.rgb[:, :14] == unknown)
    assert np.all(r.depth_m[~r.observed] == 0.0)

    # no Gaussian is ever more than fully opaque or non-finite
    assert np.all((s.opacity > 0.0) & (s.opacity <= 1.0))

    # bad intrinsics / shapes fail loudly instead of producing garbage
    good = np.full((12, 16), 10.0, dtype=np.float32)
    for bad_k in (np.zeros((3, 3)), np.full((3, 3), np.nan), np.eye(2),
                  np.array([[-12.0, 0, 7.5], [0, 12.0, 5.5], [0, 0, 1.0]])):
        try:
            g.generate_sh0(good, rgb, bad_k, n_max=10)
        except ValueError:
            continue
        raise AssertionError(f"bad K accepted: {bad_k}")
    for bad_rgb in (np.zeros((12, 15, 3), np.uint8), np.zeros((12, 16), np.uint8),
                    np.zeros((12, 16, 3), np.float32)):
        try:
            g.generate_sh0(good, bad_rgb, k, n_max=10)
        except ValueError:
            continue
        raise AssertionError(f"bad rgb accepted: {bad_rgb.shape} {bad_rgb.dtype}")
    # sim GT keys are not part of this API
    try:
        g.generate_sh0(good, rgb, k, n_max=10, sim_depth=good)  # type: ignore[call-arg]
        raise AssertionError("sim_depth kwarg accepted")
    except TypeError:
        pass


def check_deterministic_cpu() -> None:
    """Same fixture -> bit-identical Gaussians and render (no RNG, no clock)."""
    from python.viz import rv_a_gen as g

    z, rgb = fixture_scene()
    a = g.generate_sh0(z, rgb, K, n_max=2000)
    b = g.generate_sh0(z.copy(), rgb.copy(), K.copy(), n_max=2000)
    for name in ("means", "scales", "quats", "opacity", "sh0"):
        assert getattr(a, name).tobytes() == getattr(b, name).tobytes(), name
    ra = g.render_sh0(a, canned_chase())
    rb = g.render_sh0(b, canned_chase())
    assert ra.rgb.tobytes() == rb.rgb.tobytes()
    assert ra.depth_m.tobytes() == rb.depth_m.tobytes()
    assert ra.observed.tobytes() == rb.observed.tobytes()


def _project(k: np.ndarray, w2c: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    pc = w2c[:3, :3] @ p + w2c[:3, 3]
    return float(k[0, 0] * pc[0] / pc[2] + k[0, 2]), float(k[1, 1] * pc[1] / pc[2] + k[1, 2]), float(pc[2])


def check_canned_pose_honesty_pin() -> None:
    """Offline render from a canned pose: surfaces drawn, unseen space stays unknown, pinned."""
    from python.viz import rv_a_gen as g

    z, rgb = fixture_scene()
    s = g.generate_sh0(z, rgb, K, n_max=g.N_MAX_HARD, near_m=0.2, far_m=80.0)
    n_valid = int((np.isfinite(z) & (z >= 0.2) & (z <= 80.0)).sum())
    assert s.n == n_valid, (s.n, n_valid)  # sky (inf) and beyond-far generated nothing
    chase = canned_chase()
    r = g.render_sh0(s, chase)
    unknown = np.asarray(g.UNKNOWN_RGB, dtype=np.uint8)

    def px(p):
        u, v, _ = _project(chase.K, chase.w2c, np.asarray(p, dtype=np.float64))
        return int(round(v)), int(round(u))

    # surfaces the source cam saw are surfaces in the chase view
    v, u = px((0.0, -0.2, WALL_Z))          # wall face centre
    assert r.observed[v, u] and tuple(r.rgb[v, u]) == WALL_RGB, (v, u, r.rgb[v, u])
    v, u = px((-1.0, CAM_H, 8.0))           # ground in front of the wall
    assert r.observed[v, u], (v, u)
    assert np.abs(r.rgb[v, u].astype(int) - GROUND_RGB).max() <= 10, r.rgb[v, u]
    assert abs(float(r.depth_m[v, u]) - _project(chase.K, chase.w2c, np.array([-1.0, CAM_H, 8.0]))[2]) < 0.5

    # honesty: ground directly behind the wall was never seen -> must stay unknown, not filled
    # (chase ray over the wall top lands at z~25.6 m; 3-sigma splat spill reaches ~z 31, so probe past it)
    for zz in (35.0, 40.0, 50.0):
        v, u = px((0.0, CAM_H, zz))
        assert 0 <= v < chase.height and 0 <= u < chase.width, (zz, v, u)
        assert not r.observed[v, u], f"occluded ground at z={zz} painted as surface"
        assert tuple(r.rgb[v, u]) == tuple(unknown) and r.depth_m[v, u] == 0.0
    # honesty: the sky (no depth) is unknown
    assert not r.observed[0].any() and np.all(r.rgb[0] == unknown)
    assert np.all(r.depth_m[~r.observed] == 0.0)
    assert np.all(np.isfinite(r.depth_m))

    # cross-host pin (tolerant to libm last-ulp drift, strict on structure)
    if os.environ.get("GVD_RV_A_REPIN") == "1":
        np.savez_compressed(FIXTURE, rgb=r.rgb, observed=r.observed)
        print(f"repinned {FIXTURE}")
    pin = np.load(FIXTURE)
    mism = float(np.mean(pin["observed"] != r.observed))
    assert mism <= 0.005, f"observed mask drifted from pin: {mism:.4f}"
    both = pin["observed"] & r.observed
    diff = np.abs(pin["rgb"].astype(int) - r.rgb.astype(int))[both]
    assert diff.size == 0 or int(diff.max()) <= 3, int(diff.max())
    assert 0.08 <= float(r.observed.mean()) <= 0.20, float(r.observed.mean())  # 0.128 at pin: source FOV trapezoid


def check_offline_only_no_new_deps() -> None:
    """Module imports numpy + stdlib only and is not imported by supervisor / safety / live code."""
    src = (ROOT / "python" / "viz" / "rv_a_gen.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    allowed = {"__future__", "math", "dataclasses", "typing", "numpy"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] in allowed, a.name
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] in allowed, node.module
    users = []
    for p in (ROOT / "python").rglob("*.py"):
        if p.name == "rv_a_gen.py":
            continue
        if "rv_a_gen" in p.read_text(encoding="utf-8", errors="replace"):
            users.append(str(p.relative_to(ROOT)))
    assert not users, f"rv_a_gen imported on a live path: {users}"


def main() -> None:
    check_n_cap()
    check_unknown_not_free()
    check_deterministic_cpu()
    check_canned_pose_honesty_pin()
    check_offline_only_no_new_deps()
    print("test_rv_a_gen: ok")


if __name__ == "__main__":
    main()
