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
CLIP_B = "clip_008_sharp_curve_east_coast_usa"
DEV_CLIP = DATA / "v2src" / "clip015"
DRIFT_MAX_M_PER_100M = 2.0
YAW_MAX_DEG_PER_100M = 2.0
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
        o.step(float(state[i]["veh"]["wheelspeed"]), dt, M.camera_yaw_delta(prev, cur, fx))
        travel += math.dist(gt[i]["pos"][:2], gt[i - 1]["pos"][:2])
        prev = cur
    h0 = heading(gt[0])
    dx, dy = gt[-1]["pos"][0] - gt[0]["pos"][0], gt[-1]["pos"][1] - gt[0]["pos"][1]
    gx = math.sin(h0) * dx - math.cos(h0) * dy
    gy = math.cos(h0) * dx + math.sin(h0) * dy
    gh = _wrap(heading(gt[-1]) - h0)
    x, y, h = o.pose()
    err_m = math.hypot(x - gx, y - gy)
    yaw_err = _wrap(h - gh)
    return travel, 100.0 * err_m / travel, 100.0 * math.degrees(abs(yaw_err)) / travel, err_m, math.degrees(yaw_err)


def check_gt_y6_drift_clip008():
    d = DATA / "drift_eval" / CLIP_B
    if not (d / "gt_eval_only" / "gt_pose_EVAL_ONLY.jsonl").exists():
        skip_check("check_gt_y6_drift_clip008", f"{CLIP_B} not under {DATA / 'drift_eval'}")
        return
    travel, m100, d100, err_m, yaw_deg = _drift(d, "clip008")
    print(f"  {CLIP_B}: travel {travel:.2f} m, end err {err_m:.2f} m ({m100:.2f} m/100m), "
          f"yaw err {yaw_deg:.2f} deg ({d100:.2f} deg/100m)")
    assert m100 <= DRIFT_MAX_M_PER_100M and d100 <= YAW_MAX_DEG_PER_100M, (m100, d100)


def check_dev_clip015_info():
    if not (DEV_CLIP / "state.jsonl").exists():
        skip_check("check_dev_clip015_info", f"dev clip not under {DATA}")
        return
    state = [json.loads(line) for line in (DEV_CLIP / "state.jsonl").read_text().splitlines() if line.strip()]
    from PIL import Image  # test-only

    fx = float(state[1]["cams"]["main"]["K"][0])
    o = M.EgoOdometry()
    travel = 0.0
    prev = np.asarray(Image.open(DEV_CLIP / state[0]["cams"]["main"]["file"]).convert("L"), np.float64) / 255.0
    for i in range(1, len(state)):
        cur = np.asarray(Image.open(DEV_CLIP / state[i]["cams"]["main"]["file"]).convert("L"), np.float64) / 255.0
        o.step(float(state[i]["veh"]["wheelspeed"]), float(state[i]["ego"]["sim_t"] - state[i - 1]["ego"]["sim_t"]),
               M.camera_yaw_delta(prev, cur, fx))
        travel += math.dist(state[i]["ego"]["pos"][:2], state[i - 1]["ego"]["pos"][:2])
        prev = cur
    h0 = math.atan2(state[0]["ego"]["dir"][1], state[0]["ego"]["dir"][0])
    dx = state[-1]["ego"]["pos"][0] - state[0]["ego"]["pos"][0]
    dy = state[-1]["ego"]["pos"][1] - state[0]["ego"]["pos"][1]
    gx, gy = math.sin(h0) * dx - math.cos(h0) * dy, math.cos(h0) * dx + math.sin(h0) * dy
    gh = _wrap(math.atan2(state[-1]["ego"]["dir"][1], state[-1]["ego"]["dir"][0]) - h0)
    x, y, h = o.pose()
    em = math.hypot(x - gx, y - gy)
    print(f"  clip_015 (dev, info only): travel {travel:.1f} m, end err {em:.2f} m ({100*em/travel:.2f} m/100m), "
          f"yaw err {math.degrees(_wrap(h-gh)):.2f} deg ({100*abs(math.degrees(_wrap(h-gh)))/travel:.2f} deg/100m)")


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
