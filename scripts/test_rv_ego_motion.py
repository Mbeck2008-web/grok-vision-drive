#!/usr/bin/env python3
"""Offline CPU checks for Reality View ego motion (python/viz/rv_multicam.py).

Covers phase_correlation_dx, camera_yaw_delta, EgoOdometry, transform_ego_points
and accumulate. Property / fuzz checks use seeded numpy RNG only (no new deps).

E1  phase_correlation_dx sign: content rolled right gives +dx, left gives -dx.
E2  Integer shifts are recovered (seeded fuzz over sizes and shifts, |err| < 0.5 px).
E3  Wrap: a roll of w-k reads as -k; a roll of a full width reads as 0.
E4  Zero motion and textureless (constant / zero) frames give exactly 0.0, finite.
E5  Bad frames are refused: shape mismatch, non-2-D, empty, and non-finite pixels.
E6  camera_yaw_delta: + = left / CCW; magnitude atan(dx/fx); fx <= 0 / non-finite refused.
E7  EgoOdometry: zero motion stays at origin; straight drive advances +Y; a closed
    constant-curvature loop returns near the origin; non-finite / dt<0 refused; the
    pose stays finite.
E8  transform_ego_points round-trips (seeded fuzz); a left turn moves a point ahead to +X.
E9  accumulate: identity pose keeps static points exactly; non-static past points dropped.
E10 Drift scorer guard: a clip covering < 1 m is skipped with a reason, never divided by 0.
GT_E11 IMU yaw-sign flip (GT, test-only): BeamNG imu.yaw_rate has the opposite sign of
    the ego yaw (from EVAL_ONLY GT heading) on 100% of turning frames.
GT_E12 Drift (GT, test-only): EgoOdometry(wheel speed, camera yaw) vs EVAL_ONLY GT pose on
    the unseen clip clip_028_sharp_curve_west_coast_usa. Bar (set by GVD before any run,
    not tunable): end-point drift <= 2.0 m per 100 m and |yaw error| <= 2 deg per 100 m.

Honesty notes for GT_E12:
- The scored clip was named by GVD in the GVD Swarm room (2026-11-26 1:52 AM CT) before
  any odometry ran on it. No code under python/ was changed after looking at it.
- clip_015_heavy_traffic_west_coast_usa is a dev clip only (the v2 preview ran the
  accumulator on it); no drift was measured on clip_015 before 2026-11-26 and it is not
  scored here.
- GT pose is read only here (gt_eval_only/gt_pose_EVAL_ONLY.jsonl); nothing under
  python/ imports it.

Clip data lives outside the repo (default /workspace/rv-preview, override with
GVD_RV_DATA_DIR). When it is missing (e.g. CI), the GT checks are skipped visibly.
"""
from __future__ import annotations

import json
import math
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from gvd_test_env import skip_check  # noqa: E402
from python.viz import rv_multicam as M  # noqa: E402

DATA = Path(os.environ.get("GVD_RV_DATA_DIR", "/workspace/rv-preview"))
DRIFT_CLIP = "clip_028_sharp_curve_west_coast_usa"  # named before the first run; do not swap
DRIFT_DIR = DATA / "drift_eval" / DRIFT_CLIP
DRIFT_MAX_M_PER_100M = 2.0     # GVD bar, fixed before any run
YAW_MAX_DEG_PER_100M = 2.0     # GVD bar, fixed before any run
MIN_TRAVEL_M = 1.0             # below this the drift score is skipped, not divided
TURN_MIN_DEG = 0.5             # per-frame GT yaw change that counts as "turning"


def _texture(h: int, w: int, seed: int) -> np.ndarray:
    """Smooth periodic random texture (low-pass noise)."""
    r = np.random.default_rng(seed).random((h, w))
    fy = np.fft.fftfreq(h)[:, None] * h
    fx = np.fft.fftfreq(w)[None, :] * w
    return np.real(np.fft.ifft2(np.fft.fft2(r) * np.exp(-0.002 * (fy ** 2 + fx ** 2))))


def _raises(fn, *a) -> bool:
    try:
        fn(*a)
    except (ValueError, TypeError):
        return True
    return False


# ---------------------------------------------------------------- phase correlation


def check_e1_sign():
    img = _texture(48, 64, 0)
    for k in (1, 3, 7, 20):
        assert M.phase_correlation_dx(img, np.roll(img, k, axis=1)) > 0, k
        assert M.phase_correlation_dx(img, np.roll(img, -k, axis=1)) < 0, k


def check_e2_int_shift_fuzz():
    rng = np.random.default_rng(20261126)
    for t in range(200):
        h, w = int(rng.integers(16, 96)), int(rng.integers(16, 128))
        k = int(rng.integers(-(w // 2) + 1, w // 2))
        im = _texture(h, w, t)
        d = M.phase_correlation_dx(im, np.roll(im, k, axis=1))
        assert abs(d - k) < 0.5, (h, w, k, d)


def check_e3_wrap():
    img = _texture(40, 64, 3)
    for k in (1, 5, 20):
        assert abs(M.phase_correlation_dx(img, np.roll(img, 64 - k, axis=1)) + k) < 0.5, k
    assert abs(M.phase_correlation_dx(img, np.roll(img, 64, axis=1))) < 1e-6
    # vertical-only motion is not read as yaw
    assert abs(M.phase_correlation_dx(img, np.roll(img, 6, axis=0))) < 0.5


def check_e4_zero_and_textureless():
    img = _texture(32, 48, 4)
    assert abs(M.phase_correlation_dx(img, img)) < 1e-9
    for a in (np.ones((32, 48)), np.zeros((32, 48)), np.full((32, 48), 0.37)):
        d = M.phase_correlation_dx(a, a)
        assert d == 0.0 and math.isfinite(d), d
    assert M.phase_correlation_dx(np.ones((32, 48)), np.full((32, 48), 5.0)) == 0.0


def check_e5_bad_frames_refused():
    img = _texture(16, 16, 5)
    assert _raises(M.phase_correlation_dx, img, img[:, :8])
    assert _raises(M.phase_correlation_dx, np.ones((4, 4, 3)), np.ones((4, 4, 3)))
    assert _raises(M.phase_correlation_dx, np.ones((0, 0)), np.ones((0, 0)))
    for bad in (np.nan, np.inf):
        a = img.copy()
        a[3, 3] = bad
        # FINDING: today this silently returns 0.0 ("no motion") instead of refusing.
        assert _raises(M.phase_correlation_dx, a, img), f"non-finite pixel {bad} not refused"


def check_e6_camera_yaw_delta():
    img = _texture(48, 64, 6)
    shifted = np.roll(img, 4, axis=1)
    y = M.camera_yaw_delta(img, shifted, fx=100.0)
    assert y > 0, "content moving right = car yawed left = +yaw"
    assert abs(y - math.atan2(M.phase_correlation_dx(img, shifted), 100.0)) < 1e-12
    assert M.camera_yaw_delta(img, np.roll(img, -4, axis=1), fx=100.0) < 0
    assert abs(M.camera_yaw_delta(img, img, fx=785.0)) < 1e-12, "zero motion"
    for fx in (0.0, -100.0, float("nan"), float("inf")):
        # FINDING: today fx=0 gives +90 deg, fx<0 flips the sign, inf gives 0.0.
        assert _raises(M.camera_yaw_delta, img, shifted, fx), f"fx={fx} not refused"


# ---------------------------------------------------------------- odometry


def check_e7_odometry_properties():
    o = M.EgoOdometry()
    for _ in range(50):
        o.step(0.0, 0.1, 0.0)
    assert o.pose() == (0.0, 0.0, 0.0)
    o = M.EgoOdometry()
    for _ in range(10):
        o.step(10.0, 0.1, 0.0)
    x, y, h = o.pose()
    assert abs(x) < 1e-9 and abs(y - 10.0) < 1e-9 and h == 0.0
    for sign in (1.0, -1.0):  # left and right loops, R = 20 m
        o = M.EgoOdometry()
        n, r = 360, 20.0
        for _ in range(n):
            o.step(2 * math.pi * r / n, 1.0, sign * 2 * math.pi / n)
        x, y, h = o.pose()
        assert math.hypot(x, y) < 1e-6 * r and abs(h - sign * 2 * math.pi) < 1e-9, (x, y, h)
    o = M.EgoOdometry()
    o.step(5.0, 1.0, math.radians(90))
    assert o.pose()[0] < 0, "left turn drifts to -X"
    for args in ((1.0, float("nan"), 0.0), (float("inf"), 0.1, 0.0), (1.0, 0.1, float("nan")), (1.0, -0.1, 0.0)):
        assert _raises(M.EgoOdometry().step, *args), args
    o = M.EgoOdometry()
    try:
        o.step(1e308, 1e308, 0.0)
    except (ValueError, OverflowError):
        pass
    # FINDING: finite-but-huge inputs overflow to a non-finite pose without an error.
    assert all(math.isfinite(v) for v in o.pose()), o.pose()


def check_e8_transform_fuzz():
    rng = np.random.default_rng(8)
    for _ in range(100):
        p = rng.normal(0, 30, (17, 3)).astype(np.float32)
        a = (*rng.normal(0, 50, 2), float(rng.uniform(-7, 7)))
        b = (*rng.normal(0, 50, 2), float(rng.uniform(-7, 7)))
        back = M.transform_ego_points(M.transform_ego_points(p, a, b), b, a)
        assert np.allclose(back, p, atol=2e-3), np.abs(back - p).max()
        assert np.array_equal(M.transform_ego_points(p, a, b)[:, 2], p[:, 2])
    q = M.transform_ego_points(np.array([[0.0, 10.0, 0.0]]), (0, 0, 0), (0, 0, math.pi / 2))
    assert np.allclose(q, [[10.0, 0.0, 0.0]], atol=1e-5), q
    assert M.transform_ego_points(np.zeros((0, 3)), (0, 0, 0), (1, 1, 1)).shape == (0, 3)


def check_e9_accumulate():
    pts = np.array([[1.0, 5.0, 0.0], [2.0, 9.0, 0.1], [0.0, 10.0, 1.2]], np.float32)
    static = pts[:, 2] <= M.STATIC_MAX_HEIGHT_M
    past = M.FusedPoints(pts, np.full((3, 3), 9, np.uint8), np.full(3, 0.1, np.float32), static)
    pose = (3.0, -2.0, 0.4)
    out = M.accumulate(M.empty_points(), [(past, pose)], pose)
    assert out.n == int(static.sum()) and bool(np.all(out.static))
    assert np.allclose(out.pts, pts[static], atol=1e-4)
    empty_hist = M.accumulate(past, [], pose)
    assert empty_hist.n == past.n
    nothing_static = M.FusedPoints(pts, past.cols, past.foot, np.zeros(3, bool))
    assert M.accumulate(M.empty_points(), [(nothing_static, pose)], pose).n == 0


# ---------------------------------------------------------------- GT drift (test-only)


def drift_per_100m(end_err_m: float, yaw_err_rad: float, travel_m: float):
    """(m per 100 m, deg per 100 m), or None when travel < MIN_TRAVEL_M (skip, never /0)."""
    if not (math.isfinite(travel_m) and travel_m >= MIN_TRAVEL_M):
        return None
    return 100.0 * end_err_m / travel_m, 100.0 * math.degrees(abs(yaw_err_rad)) / travel_m


def check_e10_drift_guard():
    assert drift_per_100m(0.0, 0.0, 0.0) is None
    assert drift_per_100m(0.3, 0.1, 0.999) is None
    assert drift_per_100m(1.0, 0.0, float("nan")) is None
    m, d = drift_per_100m(1.0, math.radians(1.0), 50.0)
    assert abs(m - 2.0) < 1e-12 and abs(d - 2.0) < 1e-9


def _load_clip(d: Path):
    state = [json.loads(line) for line in (d / "state.jsonl").read_text().splitlines() if line.strip()]
    gt = [json.loads(line) for line in (d / "gt_eval_only" / "gt_pose_EVAL_ONLY.jsonl").read_text().splitlines() if line.strip()]
    assert len(state) == len(gt) and all(g["eval_only"] for g in gt)
    return state, gt


def _gt_heading(g) -> float:
    return math.atan2(g["dir"][1], g["dir"][0])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def check_gt_e11_imu_yaw_sign_flip():
    found = False
    for d in (DRIFT_DIR, DATA / "v2src"):
        if not (d / "gt_eval_only" / "gt_pose_EVAL_ONLY.jsonl").exists():
            continue
        found = True
        state, gt = _load_clip(d)
        turning = flipped = 0
        for i in range(1, len(state)):
            dyaw = _wrap(_gt_heading(gt[i]) - _gt_heading(gt[i - 1]))
            if abs(dyaw) < math.radians(TURN_MIN_DEG):
                continue
            turning += 1
            flipped += int(math.copysign(1, state[i]["imu"]["yaw_rate"]) == -math.copysign(1, dyaw))
        print(f"  {d.name}: imu.yaw_rate opposite to ego yaw on {flipped}/{turning} turning frames")
        assert turning > 20 and flipped == turning, (d.name, flipped, turning)
    if not found:
        skip_check("check_gt_e11_imu_yaw_sign_flip", f"no EVAL_ONLY GT clip under {DATA}")


def check_gt_e12_drift_clip028():
    if not (DRIFT_DIR / "gt_eval_only" / "gt_pose_EVAL_ONLY.jsonl").exists():
        skip_check("check_gt_e12_drift_clip028", f"{DRIFT_CLIP} not under {DATA / 'drift_eval'}")
        return
    from PIL import Image  # test-only

    state, gt = _load_clip(DRIFT_DIR)
    cams = json.loads((DRIFT_DIR / "cameras.json").read_text())["cameras"]
    fx = float(next(c for c in cams if c["id"] == "main")["intrinsics"]["fx"])

    def gray(rec):
        p = DRIFT_DIR / "clip028" / rec["cams"]["main"]["file"]
        return np.asarray(Image.open(p).convert("L"), np.float64) / 255.0

    o = M.EgoOdometry()
    travel = 0.0
    prev = gray(state[0])
    for i in range(1, len(state)):
        cur = gray(state[i])
        dt = float(state[i]["ego"]["sim_t"] - state[i - 1]["ego"]["sim_t"])
        o.step(float(state[i]["veh"]["wheelspeed"]), dt, M.camera_yaw_delta(prev, cur, fx))
        travel += math.dist(gt[i]["pos"][:2], gt[i - 1]["pos"][:2])
        prev = cur
    h0 = _gt_heading(gt[0])
    dx, dy = (gt[-1]["pos"][0] - gt[0]["pos"][0], gt[-1]["pos"][1] - gt[0]["pos"][1])
    gx = math.sin(h0) * dx - math.cos(h0) * dy   # ego0 +X (right)
    gy = math.cos(h0) * dx + math.sin(h0) * dy   # ego0 +Y (forward)
    gh = _wrap(_gt_heading(gt[-1]) - h0)
    x, y, h = o.pose()
    err_m = math.hypot(x - gx, y - gy)
    yaw_err = _wrap(h - gh)
    score = drift_per_100m(err_m, yaw_err, travel)
    if score is None:
        skip_check("check_gt_e12_drift_clip028", f"travel {travel:.3f} m < {MIN_TRAVEL_M} m")
        return
    m100, d100 = score
    print(f"  {DRIFT_CLIP}: travel {travel:.2f} m, end err {err_m:.2f} m ({m100:.2f} m/100m), "
          f"yaw err {math.degrees(yaw_err):.2f} deg ({d100:.2f} deg/100m)")
    assert m100 <= DRIFT_MAX_M_PER_100M and d100 <= YAW_MAX_DEG_PER_100M, (m100, d100)


def main() -> None:
    failed = []
    for name, fn in list(globals().items()):
        if name.startswith("check_") and callable(fn):
            try:
                fn()
                print(f"ok {name}")
            except AssertionError as e:
                failed.append(name)
                print(f"FAIL {name}: {e!r}")
                traceback.print_exc(limit=1)
    if failed:
        print(f"test_rv_ego_motion: {len(failed)} FAILED: {', '.join(failed)}")
        raise SystemExit(1)
    print("test_rv_ego_motion: all OK")


if __name__ == "__main__":
    main()
