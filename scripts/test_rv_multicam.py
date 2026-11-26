#!/usr/bin/env python3
"""Offline CPU checks for Reality View multi-camera fusion (python/viz/rv_multicam.py).

M1  Unknown stays unknown: no points -> every chase pixel UNKNOWN_RGB, depth 0, painted 0.
M2  Multi-camera fusion: two cameras pointing opposite ways paint more of a surround
    chase view than either alone, and both land in one ego frame (round-trip of a
    known point through unproject -> chase render).
M3  Depth-discontinuity cut: a near box in front of a far wall is cut at its silhouette,
    so no point is created between the two depths (no stretched sheet).
M4  Backdrop: sky / beyond-far pixels are placed on the dome (range == dome_m) and are
    painted, never dropped.
M5  Footprint splats: a grazing ground plane rendered with neighbour-spacing footprints
    paints more of the chase view than with d/f footprints (no stripes).
M6  Ego motion: straight odometry advances +Y; a camera yaw estimate recovers a known
    horizontal shift with the right sign; transform_ego_points round-trips.
M7  No ghosts: accumulation keeps only static (low) points from past frames.
M8  Determinism + no GT: identical inputs give byte-identical renders; GT / train-only
    camera ids are refused; the module imports no forbidden modules.
M9  Edge-aware upsampling: guided upsampling keeps a sharp depth edge sharper than
    plain bilinear; inverse-depth upsampling keeps a flat ground plane flat.
M10 Ego mask: the wide camera's hood region is masked; other cameras are untouched.
M11 Focal scale: Z correction is f_cam/f_ref (not inverted); zero, negative, and NaN focals are refused.
M12 Ego-box cull: points inside EGO_BOX_SBR are dropped; points outside (X, Y, or Z) are kept.
M13 Fused-point cap: MAX_FUSED_POINTS is 1_200_000; over-cap concat downsamples deterministically to the cap.
"""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from python.viz import rv_multicam as M  # noqa: E402
from python.viz.reality_view_c import UNKNOWN_RGB, camera_from_angles  # noqa: E402


def _cam(cid, yaw, w=64, h=48, pitch=-6.0, fov=60.0, pos=(0.0, 0.0, 1.3)):
    return camera_from_angles(cid, pos, yaw, pitch, fov, w, h, near_m=0.05, far_m=400.0)


def _chase(w=96, h=54, pos=(0.0, -10.0, 12.0), pitch=-50.0, fov=80.0):
    return camera_from_angles("chase", pos, 0.0, pitch, fov, w, h, near_m=0.05, far_m=400.0)


def _ground_depth(cam):
    """Planar depth of the z=0 ground for every pixel of ``cam`` (inf above the horizon)."""
    ys, xs = np.mgrid[0 : cam.height, 0 : cam.width]
    xn = (xs + 0.5 - cam.cx) / cam.fx
    yn = -(ys + 0.5 - cam.cy) / cam.fy
    R = cam.rotation_matrix()
    dirs = np.stack([xn, yn, np.ones_like(xn)], -1) @ R  # ego-frame direction per unit planar depth
    dz = dirs[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dz < -1e-6, -cam.pos[2] / dz, np.inf)
    return t.astype(np.float32)


def check_unknown_stays_unknown():
    r = M.render_splats(M.empty_points(), _chase())
    assert r.painted_fraction == 0.0
    assert np.all(r.rgb == np.asarray(UNKNOWN_RGB, np.uint8))
    assert np.all(r.depth_m == 0.0)


def _ground_points(cam, color):
    d = _ground_depth(cam)
    valid = np.isfinite(d) & (d < 60)
    d = np.where(valid, d, 1.0)
    rgb = np.zeros((cam.height, cam.width, 3), np.uint8)
    rgb[:] = color
    return M.unproject_camera(cam, rgb, d, valid)


def check_multicam_fusion():
    front = _cam("main", 0.0)
    rear = _cam("rear", 180.0)
    ch = _chase()
    a = M.render_splats(_ground_points(front, (200, 0, 0)), ch)
    b = M.render_splats(_ground_points(rear, (0, 0, 200)), ch)
    both = M.render_splats(M.concat_points([_ground_points(front, (200, 0, 0)), _ground_points(rear, (0, 0, 200))]), ch)
    assert both.painted_fraction > max(a.painted_fraction, b.painted_fraction) + 0.05, (a.painted_fraction, b.painted_fraction, both.painted_fraction)
    # Round trip: a single pixel at known depth lands where geometry says.
    # A level forward camera at (0, 0, 1.3) puts every pixel of a constant
    # 10 m depth map at ego-Y = 10, so asserting Y alone cannot catch a
    # camera-Y sign flip (median X and Z also stay 0 and 1.3 by symmetry).
    cam = _cam("main", 0.0, w=8, h=8, pitch=0.0)
    d = np.full((8, 8), 10.0, np.float32)
    rgb = np.full((8, 8, 3), 255, np.uint8)
    p = M.unproject_camera(cam, rgb, d)
    assert np.allclose(np.median(p.pts, axis=0), [0.0, 10.0, 1.3], atol=1e-4)
    # Pixel (row=2, col=5): sample (5.5, 2.5) is 1.5 px right of and 1.5 px
    # above the principal point (cx=cy=4). Image-y is down, so a row above
    # the principal point has positive camera-Y (up) and must land at higher
    # ego-Z. A Y-sign flip in unproject inverts Z around the camera height.
    col, row, depth = 5, 2, 10.0
    u, v = col + 0.5, row + 0.5
    xc = (u - cam.cx) / cam.fx * depth
    yc = -(v - cam.cy) / cam.fy * depth
    expected = (
        np.asarray(cam.pos)
        + xc * np.asarray(cam.right)
        + yc * np.asarray(cam.up)
        + depth * np.asarray(cam.fwd)
    )
    got = p.pts[row * cam.width + col]
    assert np.allclose(got, expected, atol=1e-5), (got, expected)
    assert expected[0] > 0.0 and abs(expected[1] - 10.0) < 1e-6 and expected[2] > cam.pos[2]


def check_discontinuity_cut():
    d = np.full((40, 40), 30.0, np.float32)
    d[10:30, 10:30] = 5.0
    cut = M.discontinuity_mask(d)
    assert cut[9, 20] and cut[10, 20] and cut[29, 20] and cut[30, 20]
    assert not cut[20, 20] and not cut[2, 2]
    # Any plane (inverse depth linear in the image, e.g. a road up to the horizon) is never cut.
    cam = _cam("main", 0.0, w=64, h=48, pitch=-10.0)
    g = _ground_depth(cam)[26:]
    assert np.isfinite(g).all() and not M.discontinuity_mask(g).any()
    # Upsampling a low-res silhouette draws a sheet of in-between depths; sheet_mask cuts it.
    lo = d[4::8, 4::8]
    up = M.upsample_depth(lo, 40, 40)
    mid = (1 / up > 1 / 30 + 0.25 * (1 / 5 - 1 / 30)) & (1 / up < 1 / 5 - 0.25 * (1 / 5 - 1 / 30))
    assert mid.any()
    sm = M.sheet_mask(up, lo)
    assert sm[mid].all(), "every in-between pixel must be cut"
    assert not sm[(np.abs(up - 30) < 0.5)].any() and not sm[(np.abs(up - 5) < 0.05)].any()
    # A smooth low-res ground is not cut either.
    glo = _ground_depth(cam)[30::4, ::4]
    assert not M.sheet_mask(M.upsample_depth(glo, glo.shape[0] * 4, glo.shape[1] * 4), glo).any()


def check_backdrop():
    cam = _cam("main", 0.0, w=32, h=24, pitch=0.0)
    d = np.full((24, 32), 10.0, np.float32)
    sky = np.zeros((24, 32), bool)
    sky[:6] = True
    d[6:8] = 500.0  # beyond far
    dd, back = M.apply_backdrop(cam, d, sky, far_m=120.0, dome_m=150.0)
    assert back[:8].all() and not back[8:].any()
    ys, xs = np.nonzero(back)
    xn = (xs + 0.5 - cam.cx) / cam.fx
    yn = (ys + 0.5 - cam.cy) / cam.fy
    rng = dd[ys, xs] * np.sqrt(1 + xn**2 + yn**2)
    assert np.allclose(rng, 150.0, atol=1e-3)
    pts = M.unproject_camera(cam, np.full((24, 32, 3), 9, np.uint8), dd, backdrop=back)
    assert pts.n == 24 * 32, "backdrop pixels must be painted, not dropped"
    assert not pts.static[: 8 * 32].any()
    assert pts.static.sum() <= (~back).sum()
    # The top-row fixture sits above the horizon, so those dome points are
    # already tall and would pass the static assert even without the backdrop
    # flag. A below-horizon far band lands at ego-Z <= STATIC_MAX_HEIGHT_M
    # and would be labelled static if the flag were ignored.
    d2 = np.full((24, 32), 10.0, np.float32)
    sky2 = np.zeros((24, 32), bool)
    sky2[-6:] = True
    dd2, back2 = M.apply_backdrop(cam, d2, sky2, far_m=120.0, dome_m=150.0)
    pts2 = M.unproject_camera(cam, np.full((24, 32, 3), 9, np.uint8), dd2, backdrop=back2)
    n_back = 6 * 32
    assert back2[-6:].all() and not back2[:-6].any()
    assert pts2.n == 24 * 32
    assert (pts2.pts[-n_back:, 2] <= M.STATIC_MAX_HEIGHT_M).all(), "fixture: backdrop pts would otherwise be static"
    assert not pts2.static[-n_back:].any()


def check_footprint_splats():
    cam = _cam("main", 0.0, w=64, h=48, pitch=-10.0)
    d = _ground_depth(cam)
    valid = np.isfinite(d) & (d < 50)
    d = np.where(valid, d, 1.0)
    rgb = np.full((48, 64, 3), 128, np.uint8)
    ch = camera_from_angles("chase", (0.0, -6.0, 3.0), 0.0, -20.0, 60.0, 160, 90, near_m=0.05, far_m=400.0)
    wide = M.render_splats(M.unproject_camera(cam, rgb, d, valid, foot_max_ratio=6.0), ch, max_radius_px=6)
    thin = M.render_splats(M.unproject_camera(cam, rgb, d, valid, foot_max_ratio=1.0), ch, max_radius_px=6)
    assert wide.painted_fraction > thin.painted_fraction + 0.02, (wide.painted_fraction, thin.painted_fraction)


def check_ego_motion():
    o = M.EgoOdometry()
    for _ in range(10):
        o.step(10.0, 0.1, 0.0)
    x, y, h = o.pose()
    assert abs(x) < 1e-9 and abs(y - 10.0) < 1e-9 and h == 0.0
    o2 = M.EgoOdometry()
    o2.step(10.0, 1.0, math.radians(90))
    assert o2.pose()[0] < 0, "a left (CCW) turn moves the car to -X"
    rng = np.random.default_rng(0)
    img = rng.random((64, 128))
    img = np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-0.01 * np.add.outer(np.arange(64) ** 2, np.arange(128) ** 2))))
    assert abs(M.phase_correlation_dx(img, np.roll(img, 4, axis=1)) - 4.0) < 0.5
    # Content moving right means the car yawed left: positive yaw.
    assert M.camera_yaw_delta(img, np.roll(img, 4, axis=1), fx=100.0) > 0
    p = np.array([[1.0, 2.0, 0.5], [-3.0, 7.0, 0.0]], np.float32)
    a, b = (1.0, 2.0, 0.3), (4.0, -1.0, -0.7)
    back = M.transform_ego_points(M.transform_ego_points(p, a, b), b, a)
    assert np.allclose(back, p, atol=1e-4)
    # A point fixed in the world appears 1 m closer after driving 1 m forward.
    q = M.transform_ego_points(np.array([[0.0, 5.0, 0.0]]), (0, 0, 0), (0, 1.0, 0))
    assert np.allclose(q, [[0.0, 4.0, 0.0]], atol=1e-6)


def check_no_ghosts():
    pts = np.array([[0.0, 10.0, 0.0], [0.0, 10.0, 1.2]], np.float32)  # ground + car body
    past = M.FusedPoints(pts, np.full((2, 3), 7, np.uint8), np.full(2, 0.1, np.float32), pts[:, 2] <= M.STATIC_MAX_HEIGHT_M)
    out = M.accumulate(M.empty_points(), [(past, (0.0, 0.0, 0.0))], (0.0, 2.0, 0.0))
    assert out.n == 1 and abs(float(out.pts[0, 2])) < 1e-6 and abs(float(out.pts[0, 1]) - 8.0) < 1e-5


def check_determinism_and_no_gt():
    front = _cam("main", 0.0)
    pts = M.concat_points([_ground_points(front, (10, 200, 30)), _ground_points(_cam("rear", 180.0), (1, 2, 3))])
    a = M.render_splats(pts, _chase())
    b = M.render_splats(pts, _chase())
    assert a.rgb.tobytes() == b.rgb.tobytes() and a.depth_m.tobytes() == b.depth_m.tobytes()
    for bad in ("main_depth", "gt_main", "sim_depth", "aux_depth"):
        try:
            M.camera_from_record({"id": bad, "pos_m": [0, 0, 1], "fov_v": 40}, 8, 6)
        except ValueError:
            continue
        raise AssertionError(f"{bad} accepted")
    try:
        M.camera_from_record({"id": "main", "pos_m": [0, 0, 1], "fov_v": 40, "train_only": True}, 8, 6)
        raise AssertionError("train_only accepted")
    except ValueError:
        pass
    tree = ast.parse((ROOT / "python" / "viz" / "rv_multicam.py").read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    allowed = {"__future__", "math", "dataclasses", "typing", "numpy", "python.viz.reality_view_c"}
    assert mods <= allowed, mods - allowed


def check_upsampling():
    lo = np.full((6, 8), 40.0)
    lo[:, 4:] = 5.0
    g = np.zeros((48, 64))
    g[:, 32:] = 1.0
    plain = M.upsample_depth(lo, 48, 64)
    guided = M.upsample_depth(lo, 48, 64, g)
    def blur_width(d):
        row = 1.0 / d[24]
        return int(np.sum((row > 1 / 40 + 0.1 * (1 / 5 - 1 / 40)) & (row < 1 / 5 - 0.1 * (1 / 5 - 1 / 40))))
    assert blur_width(guided) < blur_width(plain), (blur_width(guided), blur_width(plain))
    cam = _cam("main", 0.0, w=64, h=48, pitch=-10.0)
    gd = _ground_depth(cam)
    rows = gd[24:]
    inv_blocks = (1.0 / rows).reshape(6, 4, 16, 4).mean((1, 3))  # cell means of inverse depth
    up = M.upsample_depth(1.0 / inv_blocks, 24, 64)
    assert np.allclose(1 / up[2:-2], 1 / rows[2:-2], rtol=0.05), "inverse-depth upsampling keeps the ground flat"


def check_ego_mask():
    m = M.ego_image_mask("wide", 180, 320)
    assert m[-1, 160] and not m[0, 160] and not m[-1, 0]
    assert not M.ego_image_mask("main", 48, 64).any()


def check_focal_depth_scale():
    # Canonical-camera rule: Z = Z_ref * f_cam / f_ref. An inverted ratio
    # (f_ref / f_cam) is the mutant this locks out.
    assert M.focal_depth_scale(200.0, 100.0) == 2.0
    assert M.focal_depth_scale(100.0, 200.0) == 0.5
    assert M.focal_depth_scale(47.0, 47.0) == 1.0
    for a, b in (
        (0.0, 100.0),
        (100.0, 0.0),
        (-10.0, 100.0),
        (100.0, -10.0),
        (float("nan"), 100.0),
        (100.0, float("nan")),
        (float("inf"), 100.0),
        (100.0, float("inf")),
    ):
        try:
            M.focal_depth_scale(a, b)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid focals {(a, b)}")


def check_ego_box_cull():
    assert M.EGO_BOX_SBR == (1.0, -2.45, 2.4, -0.5, 1.6)
    cam = _cam("main", 0.0, w=8, h=8, pitch=0.0)
    d = np.ones((8, 8), np.float32)
    rgb = np.zeros((8, 8, 3), np.uint8)
    valid = np.zeros((8, 8), bool)
    # Inside: near centre pixel, depth 1 m -> ~(0.07, 1.0, 1.23)
    valid[4, 4] = True
    d[4, 4] = 1.0
    rgb[4, 4] = (255, 0, 0)
    # Outside in +Y: depth 10 m -> Y = 10
    valid[4, 3] = True
    d[4, 3] = 10.0
    rgb[4, 3] = (0, 255, 0)
    # Outside in +Z only: top row, depth 2 m -> Z > 1.6, Y = 2 in range
    valid[0, 4] = True
    d[0, 4] = 2.0
    rgb[0, 4] = (0, 0, 255)
    # Outside in |X| only: left edge, depth 2.2 m -> |X| > 1, Y = 2.2 in range
    valid[4, 0] = True
    d[4, 0] = 2.2
    rgb[4, 0] = (255, 255, 0)
    raw = M.unproject_camera(cam, rgb, d, valid)
    kept = M.unproject_camera(cam, rgb, d, valid, ego_box=M.EGO_BOX_SBR)
    assert raw.n == 4 and kept.n == 3
    assert not np.any(np.all(kept.cols == (255, 0, 0), axis=1))
    for col in ((0, 255, 0), (0, 0, 255), (255, 255, 0)):
        assert np.any(np.all(kept.cols == col, axis=1)), col
    hw, y0, y1, z0, z1 = M.EGO_BOX_SBR

    def _inside(pts):
        return (
            (np.abs(pts[:, 0]) <= hw)
            & (pts[:, 1] >= y0)
            & (pts[:, 1] <= y1)
            & (pts[:, 2] >= z0)
            & (pts[:, 2] <= z1)
        )

    assert int(_inside(raw.pts).sum()) == 1
    assert not _inside(kept.pts).any()


def check_fused_point_cap():
    assert M.MAX_FUSED_POINTS == 1_200_000
    assert M.concat_points.__defaults__ == (M.MAX_FUSED_POINTS,)

    def _blob(n, seed):
        rng = np.random.default_rng(seed)
        return M.FusedPoints(
            pts=rng.normal(size=(n, 3)).astype(np.float32),
            cols=rng.integers(0, 256, size=(n, 3), dtype=np.uint8),
            foot=rng.random(n).astype(np.float32),
            static=rng.random(n) > 0.5,
        )

    a, b = _blob(80, 0), _blob(50, 1)
    raw = M.concat_points([a, b], cap=10**9)
    assert raw.n == 130
    cap = 100
    out = M.concat_points([a, b], cap=cap)
    out2 = M.concat_points([a, b], cap=cap)
    assert out.n == cap
    assert out.pts.tobytes() == out2.pts.tobytes()
    assert out.cols.tobytes() == out2.cols.tobytes()
    assert out.foot.tobytes() == out2.foot.tobytes()
    assert out.static.tobytes() == out2.static.tobytes()
    keep = np.linspace(0, raw.n - 1, cap, dtype=np.int64)
    assert np.array_equal(out.pts, raw.pts[keep])
    assert np.array_equal(out.cols, raw.cols[keep])
    assert np.array_equal(out.foot, raw.foot[keep])
    assert np.array_equal(out.static, raw.static[keep])
    small = M.concat_points([a], cap=cap)
    assert small.n == 80
    assert np.array_equal(small.pts, a.pts)
    # Equal-to-cap is kept as-is (the downsample is strict-greater).
    exact = M.concat_points([_blob(cap, 2)], cap=cap)
    assert exact.n == cap


def main() -> None:
    for name, fn in list(globals().items()):
        if name.startswith("check_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("test_rv_multicam: OK")


if __name__ == "__main__":
    main()
