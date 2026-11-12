#!/usr/bin/env python3
"""Offline checks for the perceived-lane ribbon, windshield projector, clip hold, and GT-off viz."""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _lane_gt(centers: list[float], conf: float) -> np.ndarray:
    from python.debug.lane_targets import LANE_GT_DIM, LOOKAHEADS_M

    values = np.full((LANE_GT_DIM,), np.nan, dtype=np.float32)
    for i, x in enumerate(centers):
        if i >= len(LOOKAHEADS_M):
            break
        values[i * 3] = np.float32(x)
        if np.isfinite(x):
            values[i * 3 + 1] = np.float32(x - 1.8)
            values[i * 3 + 2] = np.float32(x + 1.8)
    values[-1] = np.float32(conf)
    return values


def check_projection_monotonic() -> None:
    from python.debug.lane_targets import LOOKAHEADS_M
    from python.train.scene_ckpt_runtime import project_main_ground

    xs = [0.0, 0.4, 1.2, 2.0]
    pts = []
    for x, y in zip(xs, LOOKAHEADS_M):
        uv = project_main_ground(x, y, 640, 480)
        assert uv is not None, (x, y)
        pts.append(uv)
    # Farther ahead sits higher in the image (smaller OpenCV v).
    for (u0, v0), (u1, v1) in zip(pts, pts[1:]):
        assert v1 < v0, f"expected farther point higher in the image, got {pts}"
    assert pts[0][1] > pts[-1][1]
    # A right-bending lane moves right on the screen.
    assert pts[0][0] < pts[-1][0]
    print("projection: 5/10/20/30 m farther=higher and right bend moves right OK")


def check_spline_through_knots() -> None:
    """The curve hits every finite center and stops at the last one."""
    from python.viz.scene_ribbon import SCENE_RIBBON_STEP_M, scene_ckpt_centers, scene_ckpt_ribbon_polyline

    centers = scene_ckpt_centers(_lane_gt([0.0, 0.15, 1.1, 2.6], 0.92))
    poly = scene_ckpt_ribbon_polyline(centers, 0.92)
    assert len(poly) >= 2
    for knot in centers:
        assert any(abs(p["y"] - knot["y"]) <= 1e-6 and abs(p["x"] - knot["x"]) <= 1e-5 for p in poly), knot
    last_y = centers[-1]["y"]
    assert max(p["y"] for p in poly) <= last_y + 1e-6
    assert abs(poly[-1]["y"] - last_y) <= 1e-6
    assert abs(poly[0]["y"] - centers[0]["y"]) <= 1e-6
    ys = [p["y"] for p in poly]
    gaps = [b - a for a, b in zip(ys, ys[1:])]
    assert gaps and max(gaps) <= SCENE_RIBBON_STEP_M + 1e-6, max(gaps)
    assert all(g > 1e-9 for g in gaps)

    # 30 m missing: the stroke ends at 20 m. No sample is invented out to 30.
    short = scene_ckpt_centers(_lane_gt([0.0, 0.4, 1.2, float("nan")], 0.9))
    short_poly = scene_ckpt_ribbon_polyline(short, 0.9)
    assert short_poly
    assert abs(short_poly[-1]["y"] - 20.0) <= 1e-6
    assert all(p["y"] <= 20.0 + 1e-6 for p in short_poly)
    for knot in short:
        assert any(abs(p["y"] - knot["y"]) <= 1e-6 and abs(p["x"] - knot["x"]) <= 1e-5 for p in short_poly)

    # A hole at 10 m is two separate runs. Nothing is drawn across 5 m to 20 m.
    hole = scene_ckpt_centers(_lane_gt([0.2, float("nan"), 1.5, 2.2], 0.9))
    hole_poly = scene_ckpt_ribbon_polyline(hole, 0.9)
    hole_ys = [p["y"] for p in hole_poly]
    assert hole_ys
    assert all(y <= 5.0 + 1e-3 or y >= 20.0 - 1e-3 for y in hole_ys), hole_ys
    assert max(hole_ys) <= 30.0 + 1e-6
    print(
        f"spline: {len(poly)} samples, step<={SCENE_RIBBON_STEP_M} m, "
        "through every knot, stop at last finite OK"
    )


def check_no_overshoot() -> None:
    """Polyline samples stay inside each successive knot pair's x-span (PCHIP).

    Catmull-Rom overshoots the plateau [0,0,1,1] to about [-0.074, 1.074] and
    must fail this check. Fritsch-Carlson PCHIP must pass both fixtures.
    """
    from python.viz.scene_ribbon import scene_ckpt_centers, scene_ckpt_ribbon_polyline

    fixtures = (
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 2.0, -1.0, 3.0],
    )
    for xs in fixtures:
        centers = scene_ckpt_centers(_lane_gt(xs, 0.9))
        assert [c["y"] for c in centers] == [5.0, 10.0, 20.0, 30.0], centers
        assert [c["x"] for c in centers] == xs, centers
        poly = scene_ckpt_ribbon_polyline(centers, 0.9)
        assert poly, xs
        for a, b in zip(centers, centers[1:]):
            lo = min(a["x"], b["x"]) - 1e-6
            hi = max(a["x"], b["x"]) + 1e-6
            y0, y1 = a["y"], b["y"]
            mid = [p for p in poly if y0 - 1e-9 <= p["y"] <= y1 + 1e-9]
            assert mid, (xs, a, b)
            for p in mid:
                assert lo <= p["x"] <= hi, (
                    f"overshoot on {xs}: sample {p} outside knot span "
                    f"[{a['x']}, {b['x']}] at y in [{y0}, {y1}]"
                )
    print("no-overshoot: plateau [0,0,1,1] and zigzag [0,2,-1,3] stay in-span OK")


def check_ribbon_honesty() -> None:
    from python.train.scene_ckpt_runtime import SceneCkptPred
    from python.viz.scene_ribbon import scene_ckpt_centers, scene_ckpt_ribbon_polyline

    centers = scene_ckpt_centers(_lane_gt([0.0, 0.4, 1.2, 2.0], 0.9))
    assert [c["y"] for c in centers] == [5.0, 10.0, 20.0, 30.0]
    poly = scene_ckpt_ribbon_polyline(centers, 0.9)
    assert poly, "confident centers must draw a ribbon"
    assert max(p["y"] for p in poly) <= 30.0 + 1e-6
    # Screen-space chase x should follow the increasing sample x.
    from python.viz.stage import Cam

    cam = Cam()
    us = [cam.project(p["x"], p["y"], 0.0)[0] for p in poly]
    assert us[-1] > us[0], (us[0], us[-1])

    nan_vec = _lane_gt([0.0, 0.4, float("nan"), 2.0], 0.9)
    nan_c = scene_ckpt_centers(nan_vec)
    assert [c["y"] for c in nan_c] == [5.0, 10.0, 30.0]
    nan_poly = scene_ckpt_ribbon_polyline(nan_c, 0.9)
    ys = [p["y"] for p in nan_poly]
    assert ys, nan_poly
    assert all(y <= 10.0 + 1e-3 or y >= 30.0 - 1e-3 for y in ys), ys
    # Isolated 30 m sample cannot form a second stroke, so the ribbon ends at 10 m.
    assert max(ys) <= 10.0 + 1e-3, ys

    assert scene_ckpt_ribbon_polyline(centers, 0.2) == []
    assert scene_ckpt_ribbon_polyline(centers, 0.39999) == []
    pred = SceneCkptPred(lane_gt=nan_vec, lane_x=0.2, confidence=0.2)
    assert pred.as_state()["caption"] == "LOW"
    assert pred.as_state()["centers"][0]["y"] == 5.0
    print("ribbon: no stroke past 30 m, no interpolate across NaN, fade under 0.40 OK")


def check_overlay_and_command() -> None:
    from python.control.actuate import DriveCommand
    from python.train.scene_ckpt_runtime import (
        SceneCkptPred,
        apply_scene_ckpt_to_command,
        draw_scene_ckpt_overlay,
    )

    cmd = DriveCommand(steer=0.2, throttle=0.3, brake=0.0, seq=1, reason="lane")
    pred = SceneCkptPred(lane_gt=_lane_gt([0.0, 0.4, 1.2, 2.0], 0.88), lane_x=0.5, confidence=0.88)
    after = apply_scene_ckpt_to_command(cmd, pred)
    assert after.steer == cmd.steer and after.throttle == cmd.throttle and after.brake == cmd.brake

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    high = draw_scene_ckpt_overlay(frame, pred)
    low = draw_scene_ckpt_overlay(frame, SceneCkptPred(lane_gt=pred.lane_gt, lane_x=0.5, confidence=0.2))
    none = draw_scene_ckpt_overlay(frame, None)
    assert none is not None and none.shape == frame.shape
    # High-confidence overlay paints a full ribbon; low-confidence dims it (display only).
    blue = np.array([234, 126, 61], dtype=np.int16)
    high_blue = int(np.any(np.all(np.abs(high.astype(np.int16) - blue) <= 4, axis=-1)))
    low_full = int(np.any(np.all(np.abs(low.astype(np.int16) - blue) <= 4, axis=-1)))
    none_blue = int(np.any(np.all(np.abs(none.astype(np.int16) - blue) <= 4, axis=-1)))
    assert high_blue, "windshield overlay should paint the perceived ribbon"
    assert not low_full, "confidence 0.2 must dim the ribbon, not paint it full-strength"
    assert int(np.abs(low.astype(np.int16) - frame.astype(np.int16)).sum()) > 0
    assert not none_blue, "no scene-ckpt output must not paint a ribbon"
    print("overlay + command: ribbon dims on low confidence, command unchanged OK")


def _stage_ui():
    from python.viz.stage import VizUI

    ui = VizUI()
    ui.show_nerd = False
    ui.layers = {0}
    ui.debug.viz_forecast = False
    ui.debug.viz_lanes = False
    ui.debug.viz_signs = False
    ui.debug.viz_pip = True
    return ui


def check_stage_ribbon_and_no_gt() -> None:
    from python.train.scene_ckpt_runtime import SceneCkptPred
    from python.viz.scene_ribbon import scene_ckpt_centers
    from python.viz.stage import Cam, render_stage, debug_gt_enabled

    pred = SceneCkptPred(lane_gt=_lane_gt([0.0, 0.4, 1.2, 2.0], 0.9), lane_x=0.5, confidence=0.9)
    base = {
        "engaged": False,
        "loop_hz": 3.0,
        "policy": "modular",
        "e2e_backend": "stub",
        "veto_reason": "none",
        "path_conf": 0.95,
        "path_width": 2.0,
        "path_debug_preview": False,
        "viz_smoke": False,
        "ego": {"speed_mps": 8.0, "brake": 0.0},
        "planner": {"aeb": "off", "target_v": 8.0, "ttc_lead": 4.0, "corridor_width": 2.0},
        "path_ego": [],
        "tracks": [],
        "lanes_ext": [],
        "road_edges": [],
        "signs": [],
        "path_world": [{"x": float(i), "y": float(i)} for i in range(0, 40)],
        "debug_gt": False,
    }
    ui = _stage_ui()
    ui.top_down = True
    assert debug_gt_enabled(base) is False
    off = render_stage(dict(base), ui=ui)
    empty_pw = dict(base)
    empty_pw["path_world"] = []
    bare = render_stage(empty_pw, ui=ui)
    assert int(np.abs(off.astype(np.int16) - bare.astype(np.int16)).sum()) == 0, (
        "default viz must not draw path_world / game-pose overlays"
    )
    on_st = dict(base)
    on_st["debug_gt"] = True
    shown = render_stage(on_st, ui=ui)
    assert int(np.abs(shown.astype(np.int16) - off.astype(np.int16)).sum()) > 0
    src = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    assert "--debug-gt" in src and "DEBUG ONLY" in src

    ui2 = _stage_ui()
    with_ck = dict(base)
    with_ck["scene_ckpt"] = pred.as_state()
    assert with_ck["scene_ckpt"]["centers"] == scene_ckpt_centers(pred.lane_gt)
    painted = render_stage(with_ck, ui=ui2)
    missing = render_stage(dict(base), ui=ui2)
    assert int(np.abs(painted.astype(np.int16) - missing.astype(np.int16)).sum()) > 0
    low_st = dict(base)
    low_st["scene_ckpt"] = SceneCkptPred(lane_gt=pred.lane_gt, lane_x=0.5, confidence=0.2).as_state()
    low = render_stage(low_st, ui=ui2)
    # Low confidence still draws, but dimmer than the confident ribbon. No LOW text.
    high_delta = np.abs(painted.astype(np.int16) - missing.astype(np.int16))
    low_delta = np.abs(low.astype(np.int16) - missing.astype(np.int16))
    below_high = int(high_delta[30:, :, :].sum())
    below_low = int(low_delta[30:, :, :].sum())
    assert below_low > 0, "LOW confidence must still paint a dimmed ribbon on the stage"
    assert below_high > below_low, (below_high, below_low)
    # Title bar must not spell LOW; the mode tag is tested in check_mode_tags_always_drawn.
    src = (ROOT / "python" / "viz" / "stage.py").read_text(encoding="utf-8")
    assert "scene_ckpt_mode_tag" in src
    assert 'if cap != "LOW"' not in src

    # PIP stays up when the loop is under 8 Hz.
    main = np.full((180, 320, 3), (20, 220, 20), dtype=np.uint8)
    slow = dict(base)
    slow["loop_hz"] = 3.0
    ui3 = _stage_ui()
    pip_frame = render_stage(slow, ui=ui3, main_frame=main)
    pip = pip_frame[12:192, 12:332]
    assert float(pip.mean()) > 80, "camera PIP must stay visible at low loop_hz"
    print("stage: no default GT overlay, ribbon gated, PIP stays at 3 Hz OK")


def check_clip_hold() -> None:
    from gvd_test_env import skip_check
    from python.viz.clip_render import (
        count_mp4_frames,
        encode_held_mp4,
        ffmpeg_bin,
        held_repeats,
        load_clip_ticks,
        render_clip_folder,
        timestamps_of,
    )

    reps = held_repeats([0.0, 0.1, 0.2], fps=30.0)
    assert reps == [3, 3, 3], reps
    assert sum(reps) == 9
    cut = held_repeats([0.0, 0.1, 0.8], fps=30.0, max_gap_s=0.5)
    assert cut[0] == 3
    assert sum(cut) == 9, cut

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "clip"
        frames = folder / "frames"
        frames.mkdir(parents=True)
        rows = []
        ts = [0.0, 0.1, 0.2]
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        import cv2

        paths = []
        for i, (t, col) in enumerate(zip(ts, colors)):
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            img[:] = col
            fp = frames / f"{i:06d}.jpg"
            cv2.imwrite(str(fp), img)
            paths.append(fp)
            rows.append({"t": t, "engaged": False})
        (folder / "state.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        loaded_rows, loaded_files = load_clip_ticks(folder)
        assert timestamps_of(loaded_rows) == ts
        assert len(loaded_files) == 3
        # Frame-plan / timing only: CI has no ffmpeg, so keep the hold+cut
        # contract covered without calling encode.
        plan = held_repeats(timestamps_of(loaded_rows), fps=30.0)
        assert plan == [3, 3, 3], plan
        assert sum(plan) == 9
        if ffmpeg_bin() is None:
            skip_check("clip_mp4_encode", "ffmpeg / imageio-ffmpeg not installed")
            print("clip renderer: hold/cut plan 0/0.1/0.2 s -> 9 frames at 30 fps OK (encode SKIP)")
            return
        dest = folder / "out.mp4"
        info = render_clip_folder(folder, dest, fps=30.0)
        assert info["n_out"] == 9
        n = count_mp4_frames(dest)
        assert n == 9, (n, info)
        # Also exercise the lower helper with the same repeats.
        dest2 = folder / "out2.mp4"
        encode_held_mp4(loaded_files, held_repeats(ts, fps=30.0), dest2, fps=30.0)
        assert count_mp4_frames(dest2) == 9
    print("clip renderer: 0/0.1/0.2 s -> 9 frames at 30 fps OK")


def check_bench_reports_cpu_ms() -> None:
    from python.viz.stage import VizUI, render_stage

    st = {
        "engaged": False,
        "loop_hz": 12.0,
        "policy": "modular",
        "path_conf": 0.95,
        "path_width": 2.0,
        "ego": {"speed_mps": 8.0, "brake": 0.0},
        "planner": {"aeb": "off", "target_v": 8.0, "ttc_lead": 4.0},
        "path_ego": [{"x": 0.1 * (i / 10.0), "y": float(i)} for i in range(0, 37)],
        "tracks": [],
        "lanes_ext": [],
        "road_edges": [],
        "signs": [],
        "scene_ckpt": {
            "shadow": True,
            "source": "scene_ckpt",
            "confidence": 0.9,
            "centers": [
                {"x": 0.0, "y": 5.0},
                {"x": 0.4, "y": 10.0},
                {"x": 1.2, "y": 20.0},
                {"x": 2.0, "y": 30.0},
            ],
        },
    }
    ui = _stage_ui()
    # Warmup
    render_stage(dict(st), ui=ui, width=640, height=400)
    times = []
    for _ in range(6):
        t0 = time.perf_counter()
        img = render_stage(dict(st), ui=ui, width=640, height=400)
        times.append((time.perf_counter() - t0) * 1000.0)
    after = float(np.median(times))
    before = _legacy_per_quad_ms((800, 1280, 3), nseg=36)
    once = _once_blend_ms((800, 1280, 3), nseg=36)
    print(
        f"bench (CPU, labeled): legacy per-quad copies {before:.2f} ms; "
        f"single blend {once:.2f} ms; render_stage 640x400 after {after:.2f} ms "
        f"(median of {len(times)})"
    )
    assert img.shape == (400, 640, 3)
    assert once < before, (once, before)
    print("bench: single blend faster than legacy per-quad copies OK")


def _legacy_per_quad_ms(shape: tuple[int, ...], nseg: int = 36) -> float:
    """Reproduce the old per-quad img.copy()+addWeighted cost (CPU)."""
    import cv2

    img = np.zeros(shape, dtype=np.uint8)
    h, w = shape[:2]
    times = []
    for _ in range(4):
        t0 = time.perf_counter()
        work = img
        for i in range(nseg):
            overlay = work.copy()
            x0 = 20 + i
            cv2.rectangle(overlay, (x0, 10), (x0 + 8, h - 10), (234, 126, 61), -1)
            cv2.addWeighted(overlay, 0.45, work, 0.55, 0, work)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def _once_blend_ms(shape: tuple[int, ...], nseg: int = 36) -> float:
    """New cost model: draw every quad into one overlay, blend once (CPU)."""
    import cv2

    img = np.zeros(shape, dtype=np.uint8)
    h, w = shape[:2]
    times = []
    for _ in range(4):
        t0 = time.perf_counter()
        overlay = np.zeros_like(img)
        for i in range(nseg):
            x0 = 20 + i
            cv2.rectangle(overlay, (x0, 10), (x0 + 8, h - 10), (234, 126, 61), -1)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))



def _heading_from_plus_y_deg(x0: float, y0: float, x1: float, y1: float) -> float:
    import math

    return math.degrees(math.atan2(x1 - x0, y1 - y0))


def _screen_heading_deg(p0: tuple[int, int], p1: tuple[int, int]) -> float:
    import math

    return math.degrees(math.atan2(p1[0] - p0[0], -(p1[1] - p0[1])))


def _corridor_state(path: list[dict], *, path_width: float = 2.0) -> dict:
    return {
        "engaged": False,
        "loop_hz": 12.0,
        "policy": "modular",
        "e2e_backend": "stub",
        "veto_reason": "none",
        "path_conf": 0.95,
        "path_width": float(path_width),
        "path_debug_preview": False,
        "viz_smoke": False,
        "ego": {"speed_mps": 8.0, "brake": 0.0},
        "planner": {"aeb": "off", "target_v": 8.0, "ttc_lead": 4.0, "corridor_width": float(path_width)},
        "path_ego": path,
        "tracks": [],
        "lanes_ext": [],
        "road_edges": [],
        "signs": [],
        "debug_gt": False,
    }


def _painted_xs(delta, cam, y_m: float, thresh: int = 8) -> list[float]:
    xs: list[float] = []
    for x in np.linspace(-3.0, 3.0, 601):
        px, py = cam.project(float(x), y_m, 0.03)
        if 0 <= px < delta.shape[1] and 0 <= py < delta.shape[0] and int(delta[py, px].sum()) > thresh:
            xs.append(float(x))
    return xs


def check_no_dashed_ribbon_center() -> None:
    """Fill and edge outline stay; nothing strokes a dashed line down the middle."""
    import cv2

    from python.viz.stage import Cam, render_stage

    src = (ROOT / "python" / "viz" / "stage.py").read_text(encoding="utf-8")
    func = src.split("def _draw_filled_corridor", 1)[1].split("\ndef ", 1)[0]
    assert "_stroke_poly(img, center" not in func, "filled corridor must not stroke a center dashed line"

    path = [{"x": 0.0, "y": float(i)} for i in range(0, 31)]
    ui = _stage_ui()
    ui.top_down = True
    ui.debug.viz_pip = False
    cam = Cam(top_down=True)
    img = render_stage(_corridor_state(path), ui=ui)
    cx, cy = cam.project(0.0, 12.0, 0.03)
    ox, oy = cam.project(0.35, 12.0, 0.03)
    center_c = img[cy, cx].astype(np.int16)
    inner_c = img[oy, ox].astype(np.int16)
    dist = int(np.abs(center_c - inner_c).sum())
    assert dist <= 30, f"center should match ribbon fill, not a dash ({center_c} vs {inner_c}, dist={dist})"
    print("no-dash: center matches fill, no center stroke OK")


def check_straight_ribbon_aligned_with_ego() -> None:
    """Straight model centers stay on ego +Y; left/right curves bend the correct way."""
    from python.viz.stage import EGO_LENGTH_M, Cam, _ribbon_centerline

    straight = [{"x": 0.0, "y": float(i)} for i in range(0, 31)]
    samples = _ribbon_centerline(straight)
    assert samples, "straight path must produce a ribbon centerline"
    assert all(abs(p[0]) <= 1e-9 for p in samples), samples[:3]
    headings = [
        _heading_from_plus_y_deg(a[0], a[1], b[0], b[1])
        for a, b in zip(samples, samples[1:])
    ]
    assert headings and max(abs(h) for h in headings) < 1.0, headings

    cam = Cam()
    rear = cam.project(0.0, -EGO_LENGTH_M * 0.5, 0.0)
    front = cam.project(0.0, EGO_LENGTH_M * 0.5, 0.0)
    box_h = _screen_heading_deg(rear, front)
    rib_a = cam.project(0.0, 5.0, 0.03)
    rib_b = cam.project(0.0, 25.0, 0.03)
    rib_h = _screen_heading_deg(rib_a, rib_b)
    assert abs(rib_h - box_h) < 1.0, (rib_h, box_h)

    left = [{"x": -0.15 * float(i), "y": float(i)} for i in range(0, 21)]
    right = [{"x": 0.15 * float(i), "y": float(i)} for i in range(0, 21)]
    ls = _ribbon_centerline(left)
    rs = _ribbon_centerline(right)
    assert ls[-1][0] < -1.0 and rs[-1][0] > 1.0, (ls[-1], rs[-1])
    h_left = _heading_from_plus_y_deg(0.0, 0.0, ls[-1][0], ls[-1][1])
    h_right = _heading_from_plus_y_deg(0.0, 0.0, rs[-1][0], rs[-1][1])
    assert h_left < -1.0, h_left
    assert h_right > 1.0, h_right
    print(
        f"align: straight max|heading|={max(abs(h) for h in headings):.3f} deg, "
        f"screen vs box {rib_h - box_h:.3f} deg, left {h_left:.2f} right {h_right:.2f} OK"
    )


def check_ribbon_width_matches_ego() -> None:
    """Painted ribbon width is the ego box width, not the planner corridor."""
    import cv2

    from python.viz.stage import EGO_WIDTH_M, Cam, paint_half_m, render_stage

    car_half = EGO_WIDTH_M * 0.5
    for raw in (0.4, 1.0, 1.75):
        got = paint_half_m(raw)
        assert abs(got - car_half) <= 0.02 * car_half, (raw, got, car_half)

    ui = _stage_ui()
    ui.top_down = True
    ui.debug.viz_pip = False
    cam = Cam(top_down=True)
    path = [{"x": 0.0, "y": float(i)} for i in range(0, 31)]
    widths = {}
    for pw in (0.8, 2.0, 3.5):
        st = _corridor_state(path, path_width=pw)
        painted = render_stage(st, ui=ui)
        bare = dict(st)
        bare["path_ego"] = []
        empty = render_stage(bare, ui=ui)
        delta = cv2.absdiff(painted, empty)
        xs = _painted_xs(delta, cam, 12.0)
        assert xs, pw
        widths[pw] = max(xs) - min(xs)
    ref = widths[2.0]
    for pw, w in widths.items():
        rel = abs(w - ref) / ref
        assert rel <= 0.02, f"pw={pw} width={w:.3f} vs car-cap {ref:.3f} rel={rel:.3f}"
    # A corridor narrower than the car must still paint about the ego width,
    # not shrink to the lane/corridor number (0.8 m would be a clear miss).
    assert widths[0.8] > 1.5, widths
    print(f"width: paint_half_m always {car_half:.3f} m; rendered {widths} OK")


def check_low_conf_dims_ribbon_no_text() -> None:
    """Low confidence dims the ribbon; high stays full; no conf text either way.

    Display only: apply_scene_ckpt_lane still returns stop below the live floor.
    """
    import cv2
    from types import SimpleNamespace

    from python.train.scene_ckpt_runtime import (
        SceneCkptPred,
        apply_scene_ckpt_lane,
        draw_scene_ckpt_overlay,
        scene_ckpt_overlay_caption,
    )
    from python.viz.scene_ribbon import RIBBON_BGR, SCENE_RIBBON_MIN_CONF

    high_pred = SceneCkptPred(lane_gt=_lane_gt([0.0, 0.0, 0.0, 0.0], 0.88), lane_x=0.0, confidence=0.88)
    low_pred = SceneCkptPred(lane_gt=_lane_gt([0.0, 0.0, 0.0, 0.0], 0.12), lane_x=0.0, confidence=0.12)
    assert float(low_pred.confidence) < SCENE_RIBBON_MIN_CONF <= float(high_pred.confidence)

    pout = SimpleNamespace(
        path_ego=[{"x": 0.0, "y": 1.0}],
        lane_conf=0.9,
        path_conf=0.9,
        lane_source="hough",
        path_debug_preview=False,
        planner={"target_v": 10.0},
    )
    assert apply_scene_ckpt_lane(pout, low_pred, drive=True) == "stop"
    assert apply_scene_ckpt_lane(
        SimpleNamespace(
            path_ego=[], lane_conf=0.0, path_conf=0.0, lane_source="", path_debug_preview=False, planner={},
        ),
        high_pred,
        drive=True,
    ) == "lane"

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    drawn: list[str] = []
    real_put = cv2.putText

    def rec_put(img, text, *args, **kwargs):
        drawn.append(str(text))
        return real_put(img, text, *args, **kwargs)

    cv2.putText = rec_put  # type: ignore[assignment]
    try:
        high = draw_scene_ckpt_overlay(frame, high_pred)
        low = draw_scene_ckpt_overlay(frame, low_pred)
    finally:
        cv2.putText = real_put  # type: ignore[assignment]

    for text in drawn:
        assert "conf=" not in text, text
        assert not str(text).startswith("LOW"), text
        assert "SHADOW" in text or "DRIVE" in text or "STALE" in text
    # Helper is mode-only too: no numeric or LOW confidence text.
    assert scene_ckpt_overlay_caption(low_pred) == "SHADOW"
    assert "conf=" not in scene_ckpt_overlay_caption(low_pred)
    assert high is not None and low is not None
    blue = np.array(RIBBON_BGR, dtype=np.int16)
    high_full = int(np.any(np.all(np.abs(high.astype(np.int16) - blue) <= 8, axis=-1)))
    low_full = int(np.any(np.all(np.abs(low.astype(np.int16) - blue) <= 8, axis=-1)))
    assert high_full, "normal confidence should paint a full-strength ribbon"
    assert not low_full, "low confidence should not paint a full-strength ribbon"
    # Ignore the top-left swatch so a color chip alone cannot pass as a ribbon.
    high_body = high.copy(); high_body[:70, :60] = frame[:70, :60]
    low_body = low.copy(); low_body[:70, :60] = frame[:70, :60]
    low_delta = int(np.abs(low_body.astype(np.int16) - frame.astype(np.int16)).sum())
    high_delta = int(np.abs(high_body.astype(np.int16) - frame.astype(np.int16)).sum())
    assert low_delta > 0, "low confidence must still draw a dimmed ribbon"
    assert high_delta > low_delta, (high_delta, low_delta)
    print("low-conf display: dimmed ribbon, full at 0.88, no conf text, gate still stops OK")


def check_scene_ribbon_is_sole_blue_band() -> None:
    """When the perceived ribbon is shown: one ego-width blue band, no cyan corridor, nothing behind the bumper."""
    import cv2

    from python.train.scene_ckpt_runtime import SceneCkptPred
    from python.viz.stage import (
        CORRIDOR,
        EGO_LENGTH_M,
        EGO_WIDTH_M,
        PATH_BLUE,
        Cam,
        render_stage,
    )

    bumper = EGO_LENGTH_M * 0.5
    pred = SceneCkptPred(lane_gt=_lane_gt([0.0, 0.0, 0.0, 0.0], 0.9), lane_x=0.0, confidence=0.9)
    st = _corridor_state([{"x": 2.0, "y": float(i)} for i in range(0, 31)], path_width=3.5)
    st["scene_ckpt"] = pred.as_state()
    ui = _stage_ui()
    ui.top_down = True
    ui.debug.viz_pip = False
    painted = render_stage(st, ui=ui)
    bare = dict(st)
    bare["path_ego"] = []
    bare.pop("scene_ckpt", None)
    empty = render_stage(bare, ui=ui)
    delta = cv2.absdiff(painted, empty)
    cam = Cam(top_down=True)

    def painted_at(x: float, y: float, thresh: int = 12) -> bool:
        px, py = cam.project(float(x), float(y), 0.03)
        return int(delta[py, px].sum()) > thresh

    assert not painted_at(0.0, 0.0), "ribbon/corridor must not draw at the vehicle center"
    assert not painted_at(0.0, 1.0), "nothing should be drawn behind the ego front bumper"
    assert not painted_at(2.0, 12.0), "path_ego corridor layer must not draw when the scene ribbon is shown"
    assert painted_at(0.0, 12.0), "the perceived ribbon should paint ahead of the car"

    # Colour: PATH_BLUE, not the ice-cyan corridor fill.
    px, py = cam.project(0.0, 12.0, 0.03)
    pix = painted[py, px].astype(np.int16)
    d_blue = int(np.abs(pix - np.array(PATH_BLUE, dtype=np.int16)).sum())
    d_cyan = int(np.abs(pix - np.array(CORRIDOR, dtype=np.int16)).sum())
    assert d_blue < d_cyan, (tuple(pix), d_blue, d_cyan)

    xs = _painted_xs(delta, cam, 12.0)
    assert xs, "expected a ribbon cross-section past the nose"
    width = max(xs) - min(xs)
    assert abs(width - EGO_WIDTH_M) / EGO_WIDTH_M <= 0.35, (width, EGO_WIDTH_M)
    assert min(xs) > -1.5 and max(xs) < 1.5, xs[:3]
    # Starts at the bumper, not at the cabin origin.
    assert not painted_at(0.0, bumper - 0.6)
    assert painted_at(0.0, bumper + 0.8)
    print(
        f"sole-ribbon: no cabin wedge, no offset corridor, blue@12m width={width:.2f} m OK"
    )



def _is_conf_text(text: str) -> bool:
    t = str(text)
    return "conf=" in t or t.startswith("LOW") or t.startswith("LOW/")


def check_mode_tags_always_drawn() -> None:
    """SHADOW at conf 0.12 and 0.88; STALE when stale. A drop-SHADOW mutation dies."""
    import cv2

    from python.runtime.frame_gate import STALE_CAPTION
    from python.train.scene_ckpt_runtime import (
        SceneCkptPred,
        draw_scene_ckpt_overlay,
        scene_ckpt_overlay_caption,
    )
    from python.viz.stage import render_stage

    high_pred = SceneCkptPred(lane_gt=_lane_gt([0.0, 0.0, 0.0, 0.0], 0.88), lane_x=0.0, confidence=0.88)
    low_pred = SceneCkptPred(lane_gt=_lane_gt([0.0, 0.0, 0.0, 0.0], 0.12), lane_x=0.0, confidence=0.12)

    assert scene_ckpt_overlay_caption(high_pred) == "SHADOW"
    assert scene_ckpt_overlay_caption(low_pred) == "SHADOW"
    assert scene_ckpt_overlay_caption(high_pred, drive=True, frame_stale=True) == STALE_CAPTION
    assert scene_ckpt_overlay_caption(None, frame_stale=True) == STALE_CAPTION

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    drawn: list[str] = []
    real_put = cv2.putText

    def rec_put(img, text, *args, **kwargs):
        drawn.append(str(text))
        return real_put(img, text, *args, **kwargs)

    cv2.putText = rec_put  # type: ignore[assignment]
    try:
        drawn.clear()
        high = draw_scene_ckpt_overlay(frame, high_pred)
        assert drawn == ["SHADOW"], drawn
        drawn.clear()
        low = draw_scene_ckpt_overlay(frame, low_pred)
        assert drawn == ["SHADOW"], drawn
        drawn.clear()
        stale = draw_scene_ckpt_overlay(frame, high_pred, frame_stale=True)
        assert drawn == [STALE_CAPTION], drawn
        assert high is not None and low is not None and stale is not None
        # Fail-closed: a stale frame must not paint the ribbon body.
        high_body = high.copy(); high_body[:70, :200] = frame[:70, :200]
        stale_body = stale.copy(); stale_body[:70, :200] = frame[:70, :200]
        assert int(np.abs(high_body.astype(np.int16) - frame.astype(np.int16)).sum()) > 0
        assert int(np.abs(stale_body.astype(np.int16) - frame.astype(np.int16)).sum()) == 0

        ui = _stage_ui()
        path = [{"x": 0.0, "y": float(i)} for i in range(0, 31)]
        for pred, conf in ((high_pred, 0.88), (low_pred, 0.12)):
            st = _corridor_state(path)
            st["path_ego"] = [{"x": 2.0, "y": float(i)} for i in range(0, 31)]
            st["scene_ckpt"] = pred.as_state()
            drawn.clear()
            render_stage(st, ui=ui)
            assert "SHADOW" in drawn, (conf, drawn)
            assert STALE_CAPTION not in drawn, (conf, drawn)
            assert not any(_is_conf_text(t) for t in drawn), drawn

        stale_st = _corridor_state(path)
        stale_st["path_ego"] = [{"x": 2.0, "y": float(i)} for i in range(0, 31)]
        stale_st["scene_ckpt"] = high_pred.as_state()
        stale_st["scene_ckpt"]["frame_stale"] = True
        drawn.clear()
        stale_img = render_stage(stale_st, ui=ui)
        assert STALE_CAPTION in drawn, drawn
        assert "SHADOW" not in drawn, drawn
        assert not any(_is_conf_text(t) for t in drawn), drawn

        # Stage fail-closed: no ribbon body when stale (HUD tag still present).
        # Bare has no scene_ckpt and no path_ego, so no corridor either.
        fresh_st = dict(stale_st)
        fresh_st["scene_ckpt"] = high_pred.as_state()
        fresh_img = render_stage(fresh_st, ui=ui)
        bare = dict(stale_st)
        bare.pop("scene_ckpt", None)
        bare["path_ego"] = []
        missing = render_stage(bare, ui=ui)
        assert stale_img is not None and fresh_img is not None
        assert int(np.abs(fresh_img[30:].astype(np.int16) - missing[30:].astype(np.int16)).sum()) > 0
        assert int(np.abs(stale_img[30:].astype(np.int16) - missing[30:].astype(np.int16)).sum()) == 0
    finally:
        cv2.putText = real_put  # type: ignore[assignment]
    print("mode tags: SHADOW at 0.12 and 0.88, STALE when stale, no conf text OK")


def check_stop_bar_anchors_to_visible_ribbon() -> None:
    """When the scene ribbon is the only band, the AEB stop bar follows it, not hidden path_ego."""
    import cv2

    from python.train.scene_ckpt_runtime import SceneCkptPred
    from python.viz.stage import Cam, EGO_WIDTH_M, render_stage

    pred = SceneCkptPred(lane_gt=_lane_gt([0.0, 0.0, 0.0, 0.0], 0.9), lane_x=0.0, confidence=0.9)
    st = _corridor_state([{"x": 2.0, "y": float(i)} for i in range(0, 31)], path_width=3.5)
    st["scene_ckpt"] = pred.as_state()
    st["engaged"] = True
    st["ego"] = {"speed_mps": 0.0, "brake": 1.0}
    st["planner"] = {
        **st["planner"],
        "aeb": "brake",
        "target_v": 0.0,
        "cipv_id": 1,
        "ttc_lead": 0.8,
    }
    st["tracks"] = [{"id": 1, "x": 0.1, "y": 14.0, "class": "vehicle", "lead": True}]

    ui = _stage_ui()
    ui.top_down = True
    ui.debug.viz_pip = False
    # Hide the lead box so it cannot cover the bar we are measuring.
    ui.debug.viz_ghosts = False
    painted = render_stage(st, ui=ui)
    no_lead = dict(st)
    no_lead["tracks"] = []
    no_lead["planner"] = {**st["planner"], "cipv_id": None}
    empty = render_stage(no_lead, ui=ui)
    delta = cv2.absdiff(painted, empty)
    cam = Cam(top_down=True)
    y_bar = 14.0 - 2.2
    white = np.array([252, 248, 244], dtype=np.int16)
    xs: list[float] = []
    for x in np.linspace(-4.0, 4.0, 801):
        px, py = cam.project(float(x), y_bar, 0.05)
        if not (0 <= px < painted.shape[1] and 0 <= py < painted.shape[0]):
            continue
        if int(delta[py, px].sum()) <= 8:
            continue
        if int(np.abs(painted[py, px].astype(np.int16) - white).sum()) <= 24:
            xs.append(float(x))
    assert xs, "expected a white AEB stop bar at the lead's y-2.2 m"
    mid = 0.5 * (min(xs) + max(xs))
    half = EGO_WIDTH_M * 0.5
    assert abs(mid) <= 0.35, f"bar mid x={mid:.3f} should sit on the ribbon at x=0, not path_ego at x=2"
    assert max(xs) < 1.25, xs[-3:]
    assert min(xs) > -1.25, xs[:3]
    width = max(xs) - min(xs)
    assert abs(width - EGO_WIDTH_M) / EGO_WIDTH_M <= 0.45, (width, EGO_WIDTH_M, mid)
    assert mid < 1.0
    print(f"stop-bar: mid x={mid:.3f} m width={width:.2f} m (ego half={half:.3f}) OK")


def _ego_center_column(img: np.ndarray) -> float:
    """Screen column through the rendered ego box, ignoring the HUD and the caption."""
    road = np.array([15, 12, 10], dtype=np.int16)
    view = img[24:-24]
    diff = np.abs(view.astype(np.int16) - road).sum(axis=2)
    ys, xs = np.where(diff > 25)
    assert ys.size > 200, "ego box should paint on the chase view"
    height = int(ys.max() - ys.min())
    width = int(xs.max() - xs.min())
    assert height < 220 and width < 260, (height, width, "mask should be the car, not the road")
    return 0.5 * (float(xs.min()) + float(xs.max()))


def _ribbon_row_mid(delta: np.ndarray, py: int, *, min_span: int, thresh: int = 12) -> float | None:
    if py < 0 or py >= delta.shape[0]:
        return None
    # The tail fades, so the 30 m row is only a few counts above the road.
    xs = np.flatnonzero(delta[py].sum(axis=1) > thresh)
    if xs.size < 2:
        return None
    span = int(xs[-1] - xs[0])
    if span < min_span:
        return None
    return 0.5 * (float(xs[0]) + float(xs[-1]))


def check_chase_straight_path_on_ego_center() -> None:
    """A straight detection (lane_x=0 at 8 m) stays on the ego box center column.

    The chase lens sits on the ego centerline. Near and far ribbon points share
    that column within 2 px. lane_x=+1 m paints to the right, lane_x=-1 m to
    the left, by the same screen offset.
    """
    import cv2

    from python.train.scene_ckpt_runtime import SceneCkptPred
    from python.viz.stage import Cam, render_stage

    cam = Cam()
    assert abs(cam.eye[0]) < 1e-9, cam.eye
    # Look target shares the lens x, so the basis has no yaw.
    forward = cam._basis[1]
    assert abs(float(forward[0])) < 1e-6, forward

    def frame(lane_x: float):
        st = _corridor_state([])
        st["scene_ckpt"] = SceneCkptPred(
            lane_gt=_lane_gt([lane_x, lane_x, lane_x, lane_x], 0.92),
            lane_x=lane_x,
            confidence=0.92,
        ).as_state()
        ui = _stage_ui()
        ui.top_down = False
        ui.debug.viz_pip = False
        ui.debug.viz_lanes = False
        return render_stage(st, ui=ui)

    ui_blank = _stage_ui()
    ui_blank.top_down = False
    ui_blank.debug.viz_pip = False
    ui_blank.debug.viz_lanes = False
    ego = render_stage(_corridor_state([]), ui=ui_blank)
    column = _ego_center_column(ego)
    optical = ego.shape[1] * 0.5
    assert abs(column - optical) <= 2.0, (
        f"ego box column {column:.1f} is off the camera center {optical:.1f}"
    )

    straight = frame(0.0)
    delta = cv2.absdiff(straight, ego)
    # lane_x is the 8 m sample. 30 m is the far published center.
    for y_m, label in ((8.0, "near"), (30.0, "far")):
        u, py = cam.project(0.0, y_m, 0.03)
        assert abs(u - column) <= 2.0, f"{label} point y={y_m} u={u} column={column:.1f}"
        if label == "far":
            mid = _ribbon_row_mid(delta, py, min_span=15, thresh=4)
            assert mid is not None, f"far ribbon missing at row {py}"
            assert abs(mid - column) <= 2.0, f"far ribbon mid {mid:.1f} column={column:.1f}"

    _, py8 = cam.project(0.0, 8.0, 0.03)
    _, py_far = cam.project(0.0, 30.0, 0.03)
    near_mid = None
    step = -1 if py_far < py8 else 1
    for py in range(py8, py_far + step, step):
        near_mid = _ribbon_row_mid(delta, py, min_span=15)
        if near_mid is not None:
            break
    assert near_mid is not None, "straight ribbon should be visible ahead of the car"
    assert abs(near_mid - column) <= 2.0, f"near ribbon mid {near_mid:.1f} column={column:.1f}"

    def side_at(lane_x: float, y_m: float) -> tuple[float, float]:
        painted = frame(lane_x)
        d = cv2.absdiff(painted, ego)
        u, py = cam.project(lane_x, y_m, 0.03)
        mid = _ribbon_row_mid(d, py, min_span=15)
        assert mid is not None, f"lane_x={lane_x} ribbon missing at y={y_m} row={py}"
        return float(u), float(mid)

    # 20 m is ahead of the box, so both offsets are fully painted.
    u_r, mid_r = side_at(1.0, 20.0)
    u_l, mid_l = side_at(-1.0, 20.0)
    assert u_r > column + 2.0, (u_r, column)
    assert u_l < column - 2.0, (u_l, column)
    assert abs((u_r - column) - (column - u_l)) <= 2.0, (u_r, u_l, column)
    assert mid_r > column + 2.0 and mid_l < column - 2.0, (mid_r, mid_l, column)
    assert abs((mid_r - column) - (column - mid_l)) <= 2.0, (mid_r, mid_l, column)
    print(
        f"chase center: ego column {column:.1f}, "
        f"+1 m u {u_r - column:.1f}px / -1 m u {column - u_l:.1f}px OK"
    )


def main() -> None:
    check_projection_monotonic()
    check_spline_through_knots()
    check_no_overshoot()
    check_ribbon_honesty()
    check_overlay_and_command()
    check_stage_ribbon_and_no_gt()
    check_clip_hold()
    check_bench_reports_cpu_ms()
    check_no_dashed_ribbon_center()
    check_straight_ribbon_aligned_with_ego()
    check_chase_straight_path_on_ego_center()
    check_ribbon_width_matches_ego()
    check_low_conf_dims_ribbon_no_text()
    check_scene_ribbon_is_sole_blue_band()
    check_mode_tags_always_drawn()
    check_stop_bar_anchors_to_visible_ribbon()
    print("test_viz_fsd_ribbon: all OK")


if __name__ == "__main__":
    main()
