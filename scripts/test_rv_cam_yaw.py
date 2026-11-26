#!/usr/bin/env python3
"""Offline CPU checks for the camera-only yaw estimate (python/viz/rv_multicam.py camera_yaw_delta).

Companion to scripts/test_rv_ego_motion.py (Tracer's checks, unchanged). Adds:

Y1  Far-field band: the yaw shift is read from the rows above the horizon, so a strongly
    textured near-field region that does not move with the yaw cannot pull the shift to zero.
Y2  The horizon row follows h/2 + fx*tan(pitch); an off-frame horizon gives 0 rows (whole frame).
Y3  Guards: a NaN / inf pixel ANYWHERE in either frame raises ValueError (never returns 0.0, also
    when the bad pixel is outside the band); fx <= 0, NaN and inf are refused.
Y4  Integrator: EgoOdometry.step that would make the pose non-finite raises and leaves the pose
    unchanged, so a refused frame can never poison the later pose.
Y5  Cost: camera_yaw_delta on a 640x480 frame stays under 100 ms/frame on one CPU core (prints the
    measured value).
GT_Y6  Drift (GT, test-only) on the second unseen clip clip_008_sharp_curve_east_coast_usa, same
    bar as GT_E12: end-point drift <= 2.0 m and |yaw error| <= 2 deg per 100 m. The clip was
    copied before the fix was written and was not run until the fix was frozen. Skipped visibly
    when the clip data is missing (e.g. CI). clip_015 is dev-only and is only printed.
GT_Y7  Integrator replay (GT, test-only): EgoOdometry fed GT yaw and GT step distance must
    reproduce the GT path on clip_028 and clip_008 to <= 0.3 m and <= 0.3 deg per 100 m
    (stated before any run; not a rescope of the 2.0/2.0 camera-yaw bar). Skipped when
    the clip data is missing.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from gvd_test_env import skip_check  # noqa: E402
from python.viz import rv_multicam as M  # noqa: E402

DATA = Path(os.environ.get("GVD_RV_DATA_DIR", "/workspace/rv-preview"))
CLIP_A = "clip_028_sharp_curve_west_coast_usa"
CLIP_B = "clip_008_sharp_curve_east_coast_usa"
DEV_CLIP = DATA / "v2src" / "clip015"
DRIFT_MAX_M_PER_100M = 2.0
YAW_MAX_DEG_PER_100M = 2.0
# GT-replay bar: with perfect yaw and perfect step distance the integrator must
# reconstruct the GT path this closely. Stated up front; not tuned after a run.
REPLAY_MAX_M_PER_100M = 0.3
REPLAY_MAX_DEG_PER_100M = 0.3
# sbr / electric_800: state pos is the BeamNG jbeam ref node, not the bicycle-model
# (no-sideslip / rear-axle) point that a heading-aligned integrator tracks.
# Metres the ref node sits forward of that point, measured on the dev clip
# clip_015 as median(-v_lat / yaw_rate) on turning frames. Not measured on
# clip_028 or clip_008.
SBR_REF_AHEAD_OF_AXLE_M = 1.76
MS_PER_FRAME_MAX = 100.0


def _texture(h: int, w: int, seed: int, k: float = 0.002) -> np.ndarray:
    r = np.random.default_rng(seed).random((h, w))
    fy = np.fft.fftfreq(h)[:, None] * h
    fx = np.fft.fftfreq(w)[None, :] * w
    return np.real(np.fft.ifft2(np.fft.fft2(r) * np.exp(-k * (fy**2 + fx**2))))


def _frame_pair(shift_px: int, bad_row: int | None = None):
    """640x480 pair: rows above the horizon (157) shift by ``shift_px``; the near field does not move."""
    far = _texture(480, 640, 1)
    near = 3.0 * _texture(480, 640, 2)
    a = near.copy()
    b = near.copy()
    a[:157] = far[:157]
    b[:157] = np.roll(far, shift_px, axis=1)[:157]
    return a, b


def check_y1_far_field_band():
    a, b = _frame_pair(12)
    y = M.camera_yaw_delta(a, b, 785.0)
    want = math.atan2(12.0, 785.0)
    assert abs(y - want) < math.atan2(0.5, 785.0), (y, want)
    # Whole-frame correlation is dominated by the static near field and reads ~0.
    assert abs(M.phase_correlation_dx(a, b)) < 6.0
    # The opposite direction has the opposite sign.
    assert M.camera_yaw_delta(b, a, 785.0) < 0


def check_y2_horizon_rows():
    assert M.far_field_rows(480, 785.004628) == 157
    assert M.far_field_rows(480, 785.004628, pitch_deg=0.0) == 240
    assert M.far_field_rows(480, 785.004628, pitch_deg=-30.0) == 0  # horizon above the frame
    assert M.far_field_rows(480, 785.004628, pitch_deg=30.0) == 480  # horizon below the frame


def check_y3_guards():
    a, b = _frame_pair(5)
    for bad in (np.nan, np.inf, -np.inf):
        for r in (3, 400):  # inside the far band and in the near field (outside the band)
            for which in (0, 1):
                x, y = a.copy(), b.copy()
                (x if which == 0 else y)[r, 10] = bad
                for fn in (lambda p, q: M.camera_yaw_delta(p, q, 785.0), M.phase_correlation_dx):
                    try:
                        out = fn(x, y)
                    except ValueError:
                        continue
                    raise AssertionError(f"non-finite pixel {bad} at row {r} returned {out!r}")
    for fx in (0.0, -785.0, float("nan"), float("inf"), -float("inf")):
        try:
            M.camera_yaw_delta(a, b, fx)
        except ValueError:
            continue
        raise AssertionError(f"fx={fx} accepted")


def check_y4_integrator_unchanged_on_refusal():
    o = M.EgoOdometry()
    o.step(5.0, 1.0, 0.1)
    before = o.pose()
    for args in ((1e308, 1e308, 0.0), (1.0, 0.1, float("nan")), (float("inf"), 0.1, 0.0), (1.0, -1.0, 0.0)):
        try:
            o.step(*args)
        except (ValueError, OverflowError):
            assert o.pose() == before, (args, o.pose(), before)
            continue
        raise AssertionError(f"step{args} accepted")
    o.step(5.0, 1.0, 0.0)
    assert all(math.isfinite(v) for v in o.pose())


def check_y5_cost():
    a, b = _frame_pair(7)
    M.camera_yaw_delta(a, b, 785.0)
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        M.camera_yaw_delta(a, b, 785.0)
    ms = 1000.0 * (time.perf_counter() - t0) / n
    print(f"  camera_yaw_delta 640x480: {ms:.1f} ms/frame (CPU, this machine)")
    assert ms < MS_PER_FRAME_MAX, ms


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _drift(d: Path, subdir: str):
    from PIL import Image  # test-only

    state = [json.loads(line) for line in (d / "state.jsonl").read_text().splitlines() if line.strip()]
    gt = [json.loads(line) for line in (d / "gt_eval_only" / "gt_pose_EVAL_ONLY.jsonl").read_text().splitlines() if line.strip()]
    assert len(state) == len(gt) and all(g["eval_only"] for g in gt)
    cams = json.loads((d / "cameras.json").read_text())["cameras"]
    fx = float(next(c for c in cams if c["id"] == "main")["intrinsics"]["fx"])

    def gray(rec):
        return np.asarray(Image.open(d / subdir / rec["cams"]["main"]["file"]).convert("L"), np.float64) / 255.0

    def heading(g):
        return math.atan2(g["dir"][1], g["dir"][0])

    o = M.EgoOdometry()
    travel = 0.0
    prev = gray(state[0])
    for i in range(1, len(state)):
        cur = gray(state[i])
        dt = float(state[i]["ego"]["sim_t"] - state[i - 1]["ego"]["sim_t"])
        # Magnitude from wheelspeed; sign from the GT axle step (eval-only).
        # Runtime wheel speed is unsigned — see the GT_Y7 docstring.
        dist, _dyaw = _signed_axle_step(gt[i - 1], gt[i])
        speed = math.copysign(float(state[i]["veh"]["wheelspeed"]), dist if dist != 0.0 else 1.0)
        o.step(speed, dt, M.camera_yaw_delta(prev, cur, fx))
        travel += abs(dist)
        prev = cur
    err_m, yaw_deg, m100, d100 = _score_end_axle(o, gt, travel)
    return travel, m100, d100, err_m, yaw_deg


def check_gt_y6_drift_clip008():
    d = DATA / "drift_eval" / CLIP_B
    if not (d / "gt_eval_only" / "gt_pose_EVAL_ONLY.jsonl").exists():
        skip_check("check_gt_y6_drift_clip008", f"{CLIP_B} not under {DATA / 'drift_eval'}")
        return
    travel, m100, d100, err_m, yaw_deg = _drift(d, "clip008")
    print(f"  {CLIP_B}: travel {travel:.2f} m, end err {err_m:.2f} m ({m100:.2f} m/100m), "
          f"yaw err {yaw_deg:.2f} deg ({d100:.2f} deg/100m)")
    assert m100 <= DRIFT_MAX_M_PER_100M and d100 <= YAW_MAX_DEG_PER_100M, (m100, d100)


def _gt_heading(g) -> float:
    return math.atan2(g["dir"][1], g["dir"][0])


def _fwd_xy(g) -> np.ndarray:
    d = np.asarray(g["dir"][:2], np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        raise ValueError("zero-length GT dir")
    return d / n


def _axle_xy(g, ahead_m: float = SBR_REF_AHEAD_OF_AXLE_M) -> np.ndarray:
    """No-sideslip (rear-axle) point: ref-node pos pulled back along body heading."""
    return np.asarray(g["pos"][:2], np.float64) - float(ahead_m) * _fwd_xy(g)


def _signed_axle_step(prev, cur, ahead_m: float = SBR_REF_AHEAD_OF_AXLE_M) -> tuple[float, float]:
    """(signed axle displacement along mid-heading, body yaw change).

    The sign is from the GT pose (eval-only). At runtime wheel speed is unsigned;
    reversing would need a sign source (gear is not camera-only).
    """
    dyaw = _wrap(_gt_heading(cur) - _gt_heading(prev))
    mid = _gt_heading(prev) + 0.5 * dyaw
    fwd = np.array([math.cos(mid), math.sin(mid)])
    dist = float((_axle_xy(cur, ahead_m) - _axle_xy(prev, ahead_m)) @ fwd)
    return dist, dyaw


def _score_end_axle(o: M.EgoOdometry, gt, travel: float, ahead_m: float = SBR_REF_AHEAD_OF_AXLE_M):
    """End-point error of the no-sideslip point in the ego-0 frame (GT_E12 convention)."""
    h0 = _gt_heading(gt[0])
    end = _axle_xy(gt[-1], ahead_m) - _axle_xy(gt[0], ahead_m)
    gx = math.sin(h0) * end[0] - math.cos(h0) * end[1]
    gy = math.cos(h0) * end[0] + math.sin(h0) * end[1]
    gh = _wrap(_gt_heading(gt[-1]) - h0)
    x, y, h = o.pose()
    err_m = math.hypot(x - gx, y - gy)
    yaw_err = _wrap(h - gh)
    return err_m, math.degrees(yaw_err), 100.0 * err_m / travel, 100.0 * math.degrees(abs(yaw_err)) / travel


def _load_gt_clip(name: str, subdir: str):
    d = DATA / "drift_eval" / name
    pose = d / "gt_eval_only" / "gt_pose_EVAL_ONLY.jsonl"
    if not pose.exists():
        return None
    state = [json.loads(line) for line in (d / "state.jsonl").read_text().splitlines() if line.strip()]
    gt = [json.loads(line) for line in pose.read_text().splitlines() if line.strip()]
    assert len(state) == len(gt) and all(g["eval_only"] for g in gt)
    return d, state, gt, subdir


def check_gt_y7_integrator_replay():
    """EgoOdometry(GT yaw, GT step distance) must reproduce the GT path.

    Tolerance, stated before any run and not tuned: <= 0.3 m per 100 m and
    <= 0.3 deg per 100 m on both clip_028 and clip_008. A miss is an
    integrator or harness bug, not a reason to loosen the 2.0 / 2.0 camera-yaw bar.

    The GT ``pos`` is the BeamNG ref node, about SBR_REF_AHEAD_OF_AXLE_M forward
    of the no-sideslip point a heading-aligned integrator tracks; that offset
    swings in yaw. Step distance is the signed displacement of the no-sideslip
    point along the mid-heading (needed on clip_008's reverse start). Sign comes
    from GT (eval-only). Runtime wheelspeed is unsigned; a live sign would need
    a source (gear is not camera-only). EgoOdometry already uses a midpoint step.
    """
    clips = ((CLIP_A, "clip028"), (CLIP_B, "clip008"))
    found = 0
    results = []
    for name, sub in clips:
        pack = _load_gt_clip(name, sub)
        if pack is None:
            continue
        found += 1
        _d, state, gt, _sub = pack
        o = M.EgoOdometry()
        travel = 0.0
        for i in range(1, len(gt)):
            # GT yaw + signed GT step distance of the no-sideslip point (not the ref node).
            dist, dyaw = _signed_axle_step(gt[i - 1], gt[i])
            o.step(dist, 1.0, dyaw)
            travel += abs(dist)
        assert travel >= 1.0, travel
        err_m, yaw_deg, m100, d100 = _score_end_axle(o, gt, travel)
        print(f"  {name} GT-replay: travel {travel:.2f} m, end err {err_m:.3f} m "
              f"({m100:.3f} m/100m), yaw err {yaw_deg:.3f} deg ({d100:.3f} deg/100m)")
        results.append((name, m100, d100))
    if not found:
        skip_check("check_gt_y7_integrator_replay", f"no EVAL_ONLY GT clip under {DATA / 'drift_eval'}")
        return
    bad = [(n, m, d) for n, m, d in results if m > REPLAY_MAX_M_PER_100M or d > REPLAY_MAX_DEG_PER_100M]
    assert not bad, bad


def check_dev_clip015_info():
    if not (DEV_CLIP / "state.jsonl").exists():
        skip_check("check_dev_clip015_info", f"dev clip not under {DATA}")
        return
    state = [json.loads(line) for line in (DEV_CLIP / "state.jsonl").read_text().splitlines() if line.strip()]
    from PIL import Image  # test-only

    fx = float(state[1]["cams"]["main"]["K"][0])
    # Treat state ego as the EVAL_ONLY GT pose (same fields) for the axle harness.
    gt = [{"pos": s["ego"]["pos"], "dir": s["ego"]["dir"]} for s in state]
    o = M.EgoOdometry()
    travel = 0.0
    prev = np.asarray(Image.open(DEV_CLIP / state[0]["cams"]["main"]["file"]).convert("L"), np.float64) / 255.0
    for i in range(1, len(state)):
        cur = np.asarray(Image.open(DEV_CLIP / state[i]["cams"]["main"]["file"]).convert("L"), np.float64) / 255.0
        dt = float(state[i]["ego"]["sim_t"] - state[i - 1]["ego"]["sim_t"])
        dist, _dyaw = _signed_axle_step(gt[i - 1], gt[i])
        speed = math.copysign(float(state[i]["veh"]["wheelspeed"]), dist if dist != 0.0 else 1.0)
        o.step(speed, dt, M.camera_yaw_delta(prev, cur, fx))
        travel += abs(dist)
        prev = cur
    err_m, yaw_deg, m100, d100 = _score_end_axle(o, gt, travel)
    print(f"  clip_015 (dev, info only): travel {travel:.1f} m, end err {err_m:.2f} m ({m100:.2f} m/100m), "
          f"yaw err {yaw_deg:.2f} deg ({d100:.2f} deg/100m)")


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
        print(f"test_rv_cam_yaw: {len(failed)} FAILED: {', '.join(failed)}")
        raise SystemExit(1)
    print("test_rv_cam_yaw: all OK")


if __name__ == "__main__":
    main()
