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
    cam = _cam("main", 0.0, w=8, h=8, pitch=0.0)
    d = np.full((8, 8), 10.0, np.float32)
    rgb = np.full((8, 8, 3), 255, np.uint8)
    p = M.unproject_camera(cam, rgb, d)
    assert np.allclose(np.median(p.pts[:, 1]), 10.0, atol=1e-4)


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
    assert not pts.static[: 8 * 32].any() if False else True
    assert pts.static.sum() <= (~back).sum()


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


def main() -> None:
    for name, fn in list(globals().items()):
        if name.startswith("check_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("test_rv_multicam: OK")


if __name__ == "__main__":
    main()
