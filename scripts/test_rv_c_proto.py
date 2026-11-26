#!/usr/bin/env python3
"""Offline CPU checks for Reality View step 2 (C prototype: depth unprojection + chase render).

Invariants (python/viz/reality_view_c.py):
I1  Shapes & dtypes: rgb is (out_h, out_w, 3) uint8, depth_m is (out_h, out_w) float32,
    observed is (out_h, out_w) bool.
I2  Finite outputs: depth_m is strictly finite everywhere; unobserved pixels have
    depth_m == 0.0 and rgb == UNKNOWN_RGB; observed pixels have near_m < depth_m <= far_m.
I3  Unknown not free: NaN, +inf, -inf, <= 0, < near_m, > far_m, masked-out pixels, and
    empty view lists never unproject or produce observed/free pixels.
I4  Behind-camera culling: points with z_chase <= chase_cam.near_m (behind or at origin
    of chase camera) are strictly culled before billboard splatting.
I5  Pose round-trip: rendering a valid synthetic depth frame from the source camera's own
    pose (billboard_radius_px=0) maps every valid pixel back to itself with
    |depth_out[v, u] - depth_in[v, u]| < 1e-4 and exact RGB match.
I6  Boundary sizes: empty views -> empty render; 1x1 input and 1x1 output render cleanly;
    zero/negative/huge output sizes, huge inputs, and non-finite/non-positive intrinsics
    raise ValueError cleanly.
I7  Deterministic CPU: identical inputs yield byte-identical rgb, depth_m, observed, and
    PNG bytes across repeated calls and fresh processes.
"""
from __future__ import annotations

import ast
import hashlib
import math
import subprocess
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

from python.viz.reality_view_c import (  # noqa: E402
    DEFAULT_CHASE_HEIGHT_M,
    MAX_INPUT_PIXELS,
    MAX_RENDER_H,
    MAX_RENDER_W,
    REALITY_VIEW_ENABLED_DEFAULT,
    UNKNOWN_RGB,
    CameraPose,
    DepthColorView,
    RenderResult,
    camera_from_angles,
    chase_pose_default,
    encode_png_bytes,
    is_reality_view_enabled,
    render_chase_depth_billboards,
)

# Pinned sha256 of the deterministic 320x180 synthetic depth+pose PNG fixture.
PINNED_PNG_SHA256 = "b8f6cf977313198cb7cc317cb6d66db9648948086793e3d3fa4fafd5b6a83316"


def _synthetic_fixture() -> tuple[list[DepthColorView], CameraPose]:
    """Deterministic multi-camera synthetic road + obstacle box (no sim GT)."""
    cam_main = camera_from_angles(
        cam_id="main",
        pos_m=(0.0, 1.20, 1.26),
        yaw_deg=0.0,
        pitch_deg=-6.0,
        fov_v_deg=34.0,
        width=64,
        height=48,
        near_m=0.05,
        far_m=100.0,
    )
    cam_pillar_l = camera_from_angles(
        cam_id="pillarL",
        pos_m=(-0.88, 0.40, 1.18),
        yaw_deg=-78.0,
        pitch_deg=-6.0,
        fov_v_deg=65.3,
        width=64,
        height=48,
        near_m=0.05,
        far_m=100.0,
    )
    views: list[DepthColorView] = []
    for idx, cam in enumerate((cam_main, cam_pillar_l)):
        h, w = cam.height, cam.width
        uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        depth = np.full((h, w), np.nan, dtype=np.float32)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        road = vv >= (h // 2)
        depth[road] = (4.0 + 36.0 * (1.0 - (vv[road] - h // 2) / max(1, h // 2))).astype(np.float32)
        rgb[road, 0] = 45 + idx * 15
        rgb[road, 1] = 55
        rgb[road, 2] = 65
        stripe = road & (np.abs(uu - w / 2.0) <= 2.0)
        rgb[stripe] = (230, 220, 180)
        box = (vv >= int(h * 0.40)) & (vv <= int(h * 0.65)) & (uu >= int(w * 0.35)) & (uu <= int(w * 0.60))
        depth[box] = np.float32(9.5)
        rgb[box] = (200, 90, 40) if idx == 0 else (180, 100, 50)
        views.append(DepthColorView(cam=cam, rgb=rgb, depth_m=depth))
    chase = chase_pose_default(width=320, height=180)
    return views, chase


def check_deterministic_cpu() -> None:
    """Same inputs give byte-identical output across repeated calls and a fresh process."""
    views, chase = _synthetic_fixture()
    r1 = render_chase_depth_billboards(views, chase, billboard_radius_px=1)
    r2 = render_chase_depth_billboards(views, chase, billboard_radius_px=1)
    assert isinstance(r1, RenderResult)
    assert r1.rgb.tobytes() == r2.rgb.tobytes()
    assert r1.depth_m.tobytes() == r2.depth_m.tobytes()
    assert r1.observed.tobytes() == r2.observed.tobytes()
    assert r1.point_count == r2.point_count and r1.point_count > 0
    p1 = encode_png_bytes(r1.rgb)
    p2 = encode_png_bytes(r2.rgb)
    assert p1 == p2

    code = (
        "import hashlib, sys\n"
        "sys.path.insert(0, '.')\n"
        "sys.path.insert(0, 'scripts')\n"
        "import test_rv_c_proto as t\n"
        "views, chase = t._synthetic_fixture()\n"
        "res = t.render_chase_depth_billboards(views, chase, billboard_radius_px=1)\n"
        "print(hashlib.sha256(t.encode_png_bytes(res.rgb)).hexdigest())\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, check=True)
    child_sha = proc.stdout.strip().splitlines()[-1]
    assert child_sha == hashlib.sha256(p1).hexdigest()
    print(f"check_deterministic_cpu: sha256={child_sha[:16]}... OK")


def check_png_pin() -> None:
    """Deterministic synthetic depth+pose fixture rendered to PNG matches pinned sha256."""
    views, chase = _synthetic_fixture()
    res = render_chase_depth_billboards(views, chase, billboard_radius_px=1)
    png = encode_png_bytes(res.rgb)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    got = hashlib.sha256(png).hexdigest()
    assert got == PINNED_PNG_SHA256, f"PNG sha mismatch: got={got} expected={PINNED_PNG_SHA256}"
    print(f"check_png_pin: sha256={got} OK")


def check_unknown_not_free() -> None:
    """Unknown/invalid depth never produces rendered observed/free points."""
    cam = camera_from_angles("main", (0.0, 1.2, 1.26), 0.0, -6.0, 34.0, 16, 12, near_m=0.2, far_m=50.0)
    chase = chase_pose_default(width=64, height=48)
    rgb = np.full((12, 16, 3), 255, dtype=np.uint8)

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
    for label, d_arr, mask in bad_depths:
        res = render_chase_depth_billboards([DepthColorView(cam=cam, rgb=rgb, depth_m=d_arr, valid_mask=mask)], chase)
        assert res.point_count == 0, f"{label} leaked points: {res.point_count}"
        assert not bool(res.observed.any()), f"{label} produced observed pixels"
        assert np.all(res.depth_m == 0.0), f"{label} produced non-zero depth"
        assert np.all(res.rgb == np.asarray(UNKNOWN_RGB, dtype=np.uint8)), f"{label} painted non-unknown color"

    mixed_d = np.full((12, 16), np.nan, dtype=np.float32)
    mixed_d[6, 8] = 10.0
    res_mix = render_chase_depth_billboards([DepthColorView(cam=cam, rgb=rgb, depth_m=mixed_d)], chase, billboard_radius_px=0)
    assert res_mix.point_count == 1
    assert int(res_mix.observed.sum()) == 1
    assert int((res_mix.depth_m > 0).sum()) == 1
    assert np.all(res_mix.rgb[~res_mix.observed] == np.asarray(UNKNOWN_RGB, dtype=np.uint8))
    print("check_unknown_not_free: OK")


def check_no_sim_gt_on_live_path() -> None:
    """Static/AST/import check: no sim GT in prototype or live path; supervisor does not import prototype; flag defaults OFF."""
    proto_path = ROOT / "python" / "viz" / "reality_view_c.py"
    proto_src = proto_path.read_text(encoding="utf-8")
    tree = ast.parse(proto_src, filename=str(proto_path))

    forbidden_modules = (
        "python.control",
        "python.runtime",
        "python.run_vision",
        "python.data.raw_frames",
        "python.debug",
        "beamngpy",
        "torch",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for bad in forbidden_modules:
                    assert not alias.name.startswith(bad), f"prototype imports forbidden {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for bad in forbidden_modules:
                assert not mod.startswith(bad), f"prototype imports forbidden {mod}"

    lowered = proto_src.lower()
    for gt_token in ("_depth_gt", "sim_gt", "train_gt", "gt_logger", "beamng"):
        assert gt_token not in lowered, f"forbidden GT token {gt_token!r} in {proto_path}"

    live_paths = [
        ROOT / "python" / "run_vision.py",
        ROOT / "python" / "viz" / "view_select.py",
        ROOT / "python" / "viz" / "cpp_publisher.py",
        *sorted((ROOT / "python" / "control").glob("*.py")),
        *sorted((ROOT / "python" / "runtime").glob("*.py")),
    ]
    for p in live_paths:
        src = p.read_text(encoding="utf-8")
        ptree = ast.parse(src, filename=str(p))
        for node in ast.walk(ptree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "reality_view" not in alias.name, f"{p} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "reality_view" not in mod, f"{p} imports from {mod}"

    import yaml
    from python.viz.view_select import render_reality_placeholder

    assert REALITY_VIEW_ENABLED_DEFAULT is False
    cams_cfg = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))
    assert is_reality_view_enabled(cams_cfg) is False
    assert is_reality_view_enabled(None) is False
    assert is_reality_view_enabled({"viz": {"reality_view_enabled": True}}) is True
    ph1 = render_reality_placeholder({"engaged": False}, 320, 180)
    ph2 = render_reality_placeholder({"engaged": False}, 320, 180)
    assert np.array_equal(ph1, ph2)
    lazy_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, python.viz.reality_view_c as rv; "
            "heavy = [m for m in ('torch','cv2','beamngpy','PIL','scipy','python.runtime','python.control') if m in sys.modules]; "
            "assert not heavy, heavy; "
            "assert rv.is_reality_view_enabled(None) is False",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert lazy_proc.returncode == 0, lazy_proc.stderr
    print("check_no_sim_gt_on_live_path: OK")


def check_invariants_examples() -> None:
    """Explicit example-based checks for invariants I1..I6 (behind-camera cull, pose round-trip, 1x1, zero/huge)."""
    chase = chase_pose_default(width=64, height=48)
    empty_res = render_chase_depth_billboards([], chase)
    assert empty_res.rgb.shape == (48, 64, 3) and empty_res.rgb.dtype == np.uint8
    assert empty_res.depth_m.shape == (48, 64) and empty_res.depth_m.dtype == np.float32
    assert empty_res.observed.shape == (48, 64) and empty_res.observed.dtype == bool
    assert empty_res.point_count == 0 and not empty_res.observed.any()

    src_cam = camera_from_angles("main", (0.0, 1.2, 1.26), 0.0, -6.0, 40.0, 32, 24, near_m=0.1, far_m=80.0)
    d_in = np.linspace(2.0, 30.0, 24 * 32, dtype=np.float32).reshape(24, 32)
    rgb_in = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb_in[..., 0] = np.arange(24, dtype=np.uint8)[:, None] * 10
    rgb_in[..., 1] = np.arange(32, dtype=np.uint8)[None, :] * 7
    rgb_in[..., 2] = 123
    rt = render_chase_depth_billboards([DepthColorView(cam=src_cam, rgb=rgb_in, depth_m=d_in)], src_cam, billboard_radius_px=0)
    assert rt.point_count == 24 * 32
    assert bool(rt.observed.all())
    assert np.allclose(rt.depth_m, d_in, atol=1e-4)
    assert np.array_equal(rt.rgb, rgb_in)

    rear_cam = camera_from_angles("rear", (0.0, -2.25, 0.78), 180.0, 0.0, 60.0, 8, 6, near_m=0.1, far_m=50.0)
    d_rear = np.full((6, 8), 10.0, dtype=np.float32)
    r_cull = render_chase_depth_billboards([DepthColorView(cam=rear_cam, rgb=np.full((6, 8, 3), 200, np.uint8), depth_m=d_rear)], chase, billboard_radius_px=3)
    assert r_cull.point_count == 0
    assert not bool(r_cull.observed.any())

    c1 = camera_from_angles("one", (0.0, 0.0, 1.0), 0.0, 0.0, 45.0, 1, 1, near_m=0.1, far_m=50.0)
    r1 = render_chase_depth_billboards(
        [DepthColorView(cam=c1, rgb=np.array([[[10, 20, 30]]], dtype=np.uint8), depth_m=np.array([[5.0]], dtype=np.float32))],
        c1,
        billboard_radius_px=0,
    )
    assert r1.rgb.shape == (1, 1, 3) and tuple(int(x) for x in r1.rgb[0, 0]) == (10, 20, 30)
    assert abs(float(r1.depth_m[0, 0]) - 5.0) < 1e-4

    for bad_w, bad_h in [(0, 10), (10, 0), (-1, 10), (MAX_RENDER_W + 1, 10), (10, MAX_RENDER_H + 1)]:
        try:
            camera_from_angles("bad", (0.0, 0.0, 1.0), 0.0, 0.0, 45.0, bad_w, bad_h)
            raise AssertionError(f"expected ValueError for ({bad_w}, {bad_h})")
        except ValueError:
            pass

    big_n = MAX_INPUT_PIXELS + 1
    big_cam = camera_from_angles("big", (0.0, 0.0, 1.0), 0.0, 0.0, 45.0, 1920, 1080)
    try:
        render_chase_depth_billboards(
            [DepthColorView(cam=big_cam, rgb=np.zeros((1, big_n, 3), np.uint8), depth_m=np.ones((1, big_n), np.float32))],
            chase,
        )
        raise AssertionError("expected ValueError for huge input")
    except ValueError:
        pass
    assert DEFAULT_CHASE_HEIGHT_M > 0
    # Max-legal-frame (360x640, matching RV_MAX_LONG_SIDE=640) buffer sizes + camera-agnostic views
    chase_640 = chase_pose_default(width=640, height=360)
    cam_a = camera_from_angles("cam_a", (0.0, 1.2, 1.26), 0.0, -6.0, 40.0, 640, 360, near_m=0.05, far_m=100.0)
    cam_b = camera_from_angles("cam_b", (-0.88, 0.4, 1.18), -78.0, -6.0, 65.0, 640, 360, near_m=0.05, far_m=100.0)
    d_640 = np.full((360, 640), 12.0, dtype=np.float32)
    rgb_640 = np.full((360, 640, 3), 120, dtype=np.uint8)
    r_640 = render_chase_depth_billboards(
        [DepthColorView(cam_a, rgb_640, d_640), DepthColorView(cam_b, rgb_640, d_640)],
        chase_640,
        billboard_radius_px=0,
    )
    assert r_640.rgb.shape == (360, 640, 3) and r_640.rgb.dtype == np.uint8 and r_640.rgb.nbytes == 360 * 640 * 3
    assert r_640.depth_m.shape == (360, 640) and r_640.depth_m.dtype == np.float32 and r_640.depth_m.nbytes == 360 * 640 * 4
    assert r_640.observed.shape == (360, 640) and r_640.observed.dtype == bool and r_640.observed.nbytes == 360 * 640
    assert np.all(np.isfinite(r_640.depth_m))
    assert np.all(r_640.depth_m[~r_640.observed] == np.float32(0.0))
    assert 0 < r_640.point_count <= 300_000
    assert encode_png_bytes(r_640.rgb)[:8] == b"\x89PNG\r\n\x1a\n"
    # Camera-agnostic: accepts any arbitrary cam_id sequence (1, 2, or 3 cams; never hardcodes wide in/out)
    for cids in (["main"], ["main", "pillarL"], ["main", "pillarL", "arbitrary_aux"]):
        v_list = [
            DepthColorView(
                camera_from_angles(cid, (0.0, 1.2, 1.26), 0.0, -6.0, 40.0, 8, 8),
                np.full((8, 8, 3), 80, dtype=np.uint8),
                np.full((8, 8), 8.0, dtype=np.float32),
            )
            for cid in cids
        ]
        rr = render_chase_depth_billboards(v_list, chase, billboard_radius_px=0)
        assert rr.point_count > 0
    print("check_invariants_examples: OK")


def _build_props():
    if given is None:
        return []

    @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        in_w=st.integers(min_value=1, max_value=24),
        in_h=st.integers(min_value=1, max_value=24),
        out_w=st.integers(min_value=1, max_value=64),
        out_h=st.integers(min_value=1, max_value=64),
        yaw=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        pitch=st.floats(min_value=-45.0, max_value=45.0, allow_nan=False, allow_infinity=False),
        cx=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        cy=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        cz=st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
        radius=st.integers(min_value=0, max_value=2),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def prop_finite_unknown_determinism(in_w, in_h, out_w, out_h, yaw, pitch, cx, cy, cz, radius, seed):
        rng = np.random.default_rng(seed)
        src_cam = camera_from_angles("fuzz", (0.0, 1.0, 1.3), 0.0, -5.0, 50.0, in_w, in_h, near_m=0.2, far_m=60.0)
        chase = camera_from_angles("chase", (cx, cy, cz), yaw, pitch, 60.0, out_w, out_h, near_m=0.2, far_m=100.0)
        depth = rng.uniform(-5.0, 80.0, size=(in_h, in_w)).astype(np.float32)
        kind = rng.integers(0, 6, size=(in_h, in_w))
        depth[kind == 0] = np.nan
        depth[kind == 1] = np.inf
        depth[kind == 2] = -np.inf
        depth[kind == 3] = 0.0
        rgb = rng.integers(0, 256, size=(in_h, in_w, 3), dtype=np.uint8)
        mask = rng.integers(0, 4, size=(in_h, in_w)) > 0

        r1 = render_chase_depth_billboards([DepthColorView(cam=src_cam, rgb=rgb, depth_m=depth, valid_mask=mask)], chase, billboard_radius_px=radius)
        r2 = render_chase_depth_billboards([DepthColorView(cam=src_cam, rgb=rgb, depth_m=depth, valid_mask=mask)], chase, billboard_radius_px=radius)
        assert r1.rgb.shape == (out_h, out_w, 3) and r1.rgb.dtype == np.uint8
        assert r1.depth_m.shape == (out_h, out_w) and r1.depth_m.dtype == np.float32
        assert r1.observed.shape == (out_h, out_w) and r1.observed.dtype == bool
        assert bool(np.isfinite(r1.depth_m).all())
        assert bool(np.all(r1.depth_m[~r1.observed] == 0.0))
        assert bool(np.all(r1.rgb[~r1.observed] == np.asarray(UNKNOWN_RGB, dtype=np.uint8)))
        if r1.observed.any():
            assert float(r1.depth_m[r1.observed].min()) > chase.near_m - 1e-5
            assert float(r1.depth_m[r1.observed].max()) <= chase.far_m + 1e-4
        assert r1.rgb.tobytes() == r2.rgb.tobytes()
        assert r1.depth_m.tobytes() == r2.depth_m.tobytes()

        all_bad = np.where(np.isfinite(depth), -1.0, depth).astype(np.float32)
        rb = render_chase_depth_billboards([DepthColorView(cam=src_cam, rgb=rgb, depth_m=all_bad)], chase, billboard_radius_px=radius)
        assert rb.point_count == 0 and not bool(rb.observed.any())

    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        w=st.integers(min_value=1, max_value=24),
        h=st.integers(min_value=1, max_value=24),
        yaw=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        pitch=st.floats(min_value=-60.0, max_value=60.0, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def prop_pose_round_trip(w, h, yaw, pitch, seed):
        rng = np.random.default_rng(seed)
        cam = camera_from_angles("rt", (1.5, -2.0, 2.5), yaw, pitch, 55.0, w, h, near_m=0.1, far_m=80.0)
        d_in = rng.uniform(0.5, 60.0, size=(h, w)).astype(np.float32)
        rgb_in = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        res = render_chase_depth_billboards([DepthColorView(cam=cam, rgb=rgb_in, depth_m=d_in)], cam, billboard_radius_px=0)
        assert res.point_count == w * h
        assert bool(res.observed.all())
        assert np.allclose(res.depth_m, d_in, atol=1e-4)
        assert np.array_equal(res.rgb, rgb_in)

    return [prop_finite_unknown_determinism, prop_pose_round_trip]


PROPS = _build_props()



def check_mutation_and_edge_cases() -> None:
    """Targeted boundary, validation, downsampling, and PNG edge cases."""
    # Flag CLI override and chase_pose_default defaults
    assert is_reality_view_enabled(None, cli_enabled=True) is True
    assert is_reality_view_enabled({"viz": {}}, cli_enabled=False) is False
    dflt_chase = chase_pose_default()
    assert dflt_chase.cam_id == "chase" and dflt_chase.width == 320 and dflt_chase.height == 180

    # Bool, non-finite, zero-length, and clip-plane validation
    for bad_kw in [
        dict(width=True),
        dict(height=False),
        dict(fov_v_deg=True),
        dict(fov_v_deg=0.0),
        dict(fov_v_deg=180.0),
        dict(fov_v_deg=float("nan")),
        dict(near_m=0.0),
        dict(near_m=-1.0),
        dict(near_m=5.0, far_m=5.0),
        dict(pos_m=(0.0, 0.0)),
        dict(pos_m=(0.0, float("inf"), 1.0)),
        dict(yaw_deg=float("nan")),
        dict(pitch_deg=float("inf")),
    ]:
        kw = dict(cam_id="x", pos_m=(0.0, 0.0, 1.0), yaw_deg=0.0, pitch_deg=0.0, fov_v_deg=45.0, width=8, height=8, near_m=0.1, far_m=50.0)
        kw.update(bad_kw)
        try:
            camera_from_angles(**kw)
            raise AssertionError(f"expected ValueError for {bad_kw}")
        except ValueError:
            pass

    # _validate_pose checks bad CameraPose fields directly
    good = camera_from_angles("g", (0.0, 0.0, 1.0), 0.0, 0.0, 45.0, 8, 8, near_m=0.1, far_m=50.0)
    for bad_pose in [
        CameraPose("b", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), 8, 8, 0.0, 8.0, 4.0, 4.0, 0.1, 50.0),
        CameraPose("b", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), 8, 8, 8.0, 0.0, 4.0, 4.0, 0.1, 50.0),
        CameraPose("b", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), 8, 8, 8.0, -1.0, 4.0, 4.0, 0.1, 50.0),
        CameraPose("b", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), 8, 8, 8.0, 8.0, 4.0, 4.0, 10.0, 10.0),
        CameraPose("b", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), 8, 8, 8.0, 8.0, 4.0, 4.0, 0.1, 50.0),
        CameraPose("b", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), 8, 8, 8.0, 8.0, 4.0, 4.0, 0.0, 50.0),
        CameraPose("b", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), 8, 8, 8.0, 8.0, 4.0, 4.0, 10.0, 5.0),
    ]:
        try:
            render_chase_depth_billboards([], bad_pose)
            raise AssertionError(f"expected ValueError for bad pose {bad_pose}")
        except ValueError:
            pass

    # Input view shape / mask / dtype validation
    rgb_ok = np.zeros((4, 4, 3), dtype=np.uint8)
    d_ok = np.ones((4, 4), dtype=np.float32)
    c4 = camera_from_angles("c4", (0.0, 0.0, 1.0), 0.0, 0.0, 45.0, 4, 4, near_m=0.5, far_m=20.0)
    for bad_view in [
        DepthColorView(c4, np.zeros((4, 4), dtype=np.uint8), d_ok),
        DepthColorView(c4, np.zeros((4, 4, 4), dtype=np.uint8), d_ok),
        DepthColorView(c4, rgb_ok, np.ones((4, 4, 1), dtype=np.float32)),
        DepthColorView(c4, np.zeros((0, 4, 3), dtype=np.uint8), np.ones((0, 4), dtype=np.float32)),
        DepthColorView(c4, np.zeros((4, 0, 3), dtype=np.uint8), np.ones((4, 0), dtype=np.float32)),
        DepthColorView(c4, np.zeros((2, 4, 3), dtype=np.uint8), d_ok),
        DepthColorView(c4, rgb_ok, d_ok, valid_mask=np.ones((2, 4), dtype=bool)),
    ]:
        try:
            render_chase_depth_billboards([bad_view], c4)
            raise AssertionError("expected ValueError for bad input view")
        except ValueError:
            pass

    # Float RGB clipping & NaN/inf sanitization
    rgb_f = np.array([[[np.nan, np.inf, -np.inf], [-10.0, 300.0, 128.0]]], dtype=np.float32)
    d_1x2 = np.array([[2.0, 2.0]], dtype=np.float32)
    c1x2 = camera_from_angles("f", (0.0, 0.0, 1.0), 0.0, 0.0, 45.0, 2, 1, near_m=0.5, far_m=20.0)
    rf = render_chase_depth_billboards([DepthColorView(c1x2, rgb_f, d_1x2)], c1x2, billboard_radius_px=0)
    assert tuple(int(v) for v in rf.rgb[0, 0]) == (0, 255, 0)
    assert tuple(int(v) for v in rf.rgb[0, 1]) == (0, 255, 128)

    # Exact near_m and far_m boundaries (inclusive in source view; chase near_m exclusive, far_m inclusive)
    c_loose_chase = camera_from_angles("lc", (0.0, 0.0, 1.0), 0.0, 0.0, 45.0, 2, 1, near_m=0.4, far_m=20.0)
    d_bound = np.array([[0.5, 20.0]], dtype=np.float32)
    rb = render_chase_depth_billboards([DepthColorView(c1x2, np.full((1, 2, 3), 99, np.uint8), d_bound)], c_loose_chase, billboard_radius_px=0)
    assert rb.point_count == 2 and bool(rb.observed.all())
    # If chase near_m == 0.5, z == 0.5 is NOT > near_m so it is culled while z == 20.0 <= 20.0 stays
    c_tight = camera_from_angles("t", (0.0, 0.0, 1.0), 0.0, 0.0, 45.0, 2, 1, near_m=0.5, far_m=20.0)
    assert rb.depth_m[0, 1] == np.float32(20.0)
    # Exactly at z == near_m on a different source near_m:
    c_src_loose = camera_from_angles("sl", (0.0, 0.0, 1.0), 0.0, 0.0, 45.0, 2, 1, near_m=0.1, far_m=25.0)
    rb2 = render_chase_depth_billboards([DepthColorView(c_src_loose, np.full((1, 2, 3), 99, np.uint8), np.array([[0.5, 20.0]], np.float32))], c_tight, billboard_radius_px=0)
    assert rb2.point_count == 1 and not bool(rb2.observed[0, 0]) and bool(rb2.observed[0, 1])

    # Top-row hy == 0 billboard coverage & default radius == 1
    d_top = np.full((4, 4), np.nan, dtype=np.float32)
    d_top[0, 0] = 5.0
    r_def = render_chase_depth_billboards([DepthColorView(c4, np.full((4, 4, 3), 77, np.uint8), d_top)], c4)
    assert bool(r_def.observed[0, 0]) and bool(r_def.observed[1, 1]) and int(r_def.observed.sum()) == 4

    # max_points uniform subsampling
    d_16 = np.full((4, 4), 5.0, dtype=np.float32)
    rgb_16 = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    r_cap = render_chase_depth_billboards([DepthColorView(c4, rgb_16, d_16)], c4, billboard_radius_px=0, max_points=5)
    assert r_cap.point_count == 5 and int(r_cap.observed.sum()) == 5
    # Verify first and last pixels were kept by linspace(0, n-1, cap)
    assert bool(r_cap.observed[0, 0]) and bool(r_cap.observed[3, 3])
    for bad_mp in (0, -1, MAX_INPUT_PIXELS):
        try:
            render_chase_depth_billboards([], c4, max_points=bad_mp)
            raise AssertionError(f"expected ValueError for max_points={bad_mp}")
        except ValueError:
            pass

    # PNG encoder edge cases (1x1, boundary dims, invalid shape/dtype)
    p1x1 = encode_png_bytes(np.array([[[1, 2, 3]]], dtype=np.uint8))
    assert encode_png_bytes(np.zeros((1, MAX_RENDER_W, 3), dtype=np.uint8))[:8] == b"\x89PNG\r\n\x1a\n"
    assert encode_png_bytes(np.zeros((MAX_RENDER_H, 1, 3), dtype=np.uint8))[:8] == b"\x89PNG\r\n\x1a\n" 
    assert p1x1[:8] == b"\x89PNG\r\n\x1a\n"
    for bad_png in [
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
        np.zeros((0, 2, 3), dtype=np.uint8),
        np.zeros((2, 0, 3), dtype=np.uint8),
        np.zeros((1, MAX_RENDER_W + 1, 3), dtype=np.uint8),
        np.zeros((MAX_RENDER_H + 1, 1, 3), dtype=np.uint8),
    ]:
        try:
            encode_png_bytes(bad_png)
            raise AssertionError("expected ValueError for bad PNG input")
        except ValueError:
            pass

    # Behind-camera / off-frustum point_count == 0
    c_off = camera_from_angles("off", (0.0, 0.0, 1.0), 0.0, 0.0, 20.0, 4, 4, near_m=0.1, far_m=50.0)
    c_side_src = camera_from_angles("ws", (0.0, 0.0, 1.0), 90.0, 0.0, 45.0, 2, 2, near_m=0.1, far_m=50.0)
    r_off = render_chase_depth_billboards([DepthColorView(c_side_src, np.full((2, 2, 3), 50, np.uint8), np.full((2, 2), 10.0, np.float32))], c_off, billboard_radius_px=0)
    assert r_off.point_count == 0 and not bool(r_off.observed.any())

    # Non-zero yaw+pitch defaults, small valid fov, and max_points=1
    cam_yp = camera_from_angles("yp", (0.0, 0.0, 0.0), 35.0, -15.0, 0.5, 4, 4)
    assert cam_yp.near_m == 0.05 and cam_yp.far_m == 100.0
    exp_fwd = np.array([
        math.sin(math.radians(35.0)) * math.cos(math.radians(-15.0)),
        math.cos(math.radians(35.0)) * math.cos(math.radians(-15.0)),
        math.sin(math.radians(-15.0)),
    ], dtype=np.float64)
    exp_up = np.array([
        -math.sin(math.radians(35.0)) * math.sin(math.radians(-15.0)),
        -math.cos(math.radians(35.0)) * math.sin(math.radians(-15.0)),
        math.cos(math.radians(-15.0)),
    ], dtype=np.float64)
    assert np.allclose(cam_yp.fwd, exp_fwd, atol=1e-12)
    assert np.allclose(cam_yp.up, exp_up, atol=1e-12)
    assert abs(np.linalg.norm(cam_yp.right) - 1.0) < 1e-12
    assert abs(np.linalg.norm(cam_yp.up) - 1.0) < 1e-12

    r_mp1 = render_chase_depth_billboards(
        [DepthColorView(c1x2, np.full((1, 2, 3), 77, np.uint8), np.array([[2.0, 3.0]], dtype=np.float32))],
        c_loose_chase,
        billboard_radius_px=0,
        max_points=1,
    )
    assert r_mp1.point_count == 1
    # Mixed empty-valid view + non-empty view exercises empty unproject branch in concatenation
    r_mix = render_chase_depth_billboards([
        DepthColorView(c1x2, np.full((1, 2, 3), 10, np.uint8), np.full((1, 2), np.nan, np.float32)),
        DepthColorView(c1x2, np.full((1, 2, 3), 77, np.uint8), np.array([[2.0, 3.0]], dtype=np.float32)),
    ], c_loose_chase, billboard_radius_px=0)
    assert r_mix.point_count == 2

    # Single-fault rejection tests for forbidden GT/aux cam_ids and NaN/inf K & T fields
    for bad_cid in (
        "", "   ", 123, "depth_gt", "gt_depth", "sim_depth", "aux_depth",
        "sem_gt", "gt_sem", "path_world", "debug_gt", "labels", "foo_gt", "gt_bar", "cam_sim_depth_0",
    ):
        try:
            camera_from_angles(bad_cid, (0.0, 0.0, 1.0), 0.0, 0.0, 45.0, 8, 8)
            raise AssertionError(f"expected ValueError for bad cam_id={bad_cid!r}")
        except ValueError:
            pass
        if isinstance(bad_cid, str):
            p_bad_cid = CameraPose(bad_cid, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), 8, 8, 8.0, 8.0, 4.0, 4.0, 0.1, 50.0)
            try:
                render_chase_depth_billboards([], p_bad_cid)
                raise AssertionError(f"expected ValueError in _validate_pose for cam_id={bad_cid!r}")
            except ValueError:
                pass

    base_pose_kw = dict(
        cam_id="ok", pos=(0.0, 0.0, 1.0), right=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0),
        fwd=(0.0, 1.0, 0.0), width=8, height=8, fx=8.0, fy=8.0, cx=4.0, cy=4.0, near_m=0.1, far_m=50.0,
    )
    for bad_val in (float("nan"), float("inf"), float("-inf")):
        for fld in ("fx", "fy", "cx", "cy", "near_m", "far_m"):
            kw_p = dict(base_pose_kw, **{fld: bad_val})
            try:
                render_chase_depth_billboards([], CameraPose(**kw_p))
                raise AssertionError(f"expected ValueError for {fld}={bad_val}")
            except ValueError:
                pass
        for vec_fld, vec_base in (
            ("pos", (0.0, 0.0, 1.0)),
            ("right", (1.0, 0.0, 0.0)),
            ("up", (0.0, 0.0, 1.0)),
            ("fwd", (0.0, 1.0, 0.0)),
        ):
            for idx in range(3):
                v_mut = list(vec_base)
                v_mut[idx] = bad_val
                kw_p = dict(base_pose_kw, **{vec_fld: tuple(v_mut)})
                try:
                    render_chase_depth_billboards([], CameraPose(**kw_p))
                    raise AssertionError(f"expected ValueError for {vec_fld}[{idx}]={bad_val}")
                except ValueError:
                    pass
    for vec_fld in ("up", "fwd"):
        kw_p = dict(base_pose_kw, **{vec_fld: (0.0, 0.0, 0.0)})
        try:
            render_chase_depth_billboards([], CameraPose(**kw_p))
            raise AssertionError(f"expected ValueError for zero {vec_fld}")
        except ValueError:
            pass
    print("check_mutation_and_edge_cases: OK")


def main() -> None:
    check_deterministic_cpu()
    check_unknown_not_free()
    check_no_sim_gt_on_live_path()
    check_invariants_examples()
    check_mutation_and_edge_cases()
    check_png_pin()
    if given is None:
        skip_check("hypothesis_properties", "hypothesis not installed (dev-only dep; CI installs it)")
    else:
        for prop in PROPS:
            prop()
            print(f"{prop.__name__}: holds OK")
    print("test_rv_c_proto: OK")


if __name__ == "__main__":
    main()


# Pytest entrypoints (for mutmut 3.8.1)
def test_deterministic_cpu() -> None:
    check_deterministic_cpu()


def test_unknown_not_free() -> None:
    check_unknown_not_free()


def test_no_sim_gt_on_live_path() -> None:
    check_no_sim_gt_on_live_path()


def test_invariants_examples() -> None:
    check_invariants_examples()


def test_png_pin() -> None:
    check_png_pin()


def test_hypothesis_props() -> None:
    for prop in PROPS:
        prop()


def test_mutation_and_edge_cases() -> None:
    check_mutation_and_edge_cases()
