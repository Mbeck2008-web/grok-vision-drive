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
    # High-confidence overlay paints a blue ribbon; low-confidence does not.
    blue = np.array([234, 126, 61], dtype=np.int16)
    high_blue = int(np.any(np.all(np.abs(high.astype(np.int16) - blue) <= 4, axis=-1)))
    low_blue = int(np.any(np.all(np.abs(low.astype(np.int16) - blue) <= 4, axis=-1)))
    none_blue = int(np.any(np.all(np.abs(none.astype(np.int16) - blue) <= 4, axis=-1)))
    assert high_blue, "windshield overlay should paint the perceived ribbon"
    assert not low_blue, "confidence 0.2 must not paint a ribbon"
    assert not none_blue, "no scene-ckpt output must not paint a ribbon"
    print("overlay + command: ribbon gated on confidence, command unchanged OK")


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
    # LOW may add a caption but must not add a long ribbon; allow caption pixels.
    delta = np.abs(low.astype(np.int16) - missing.astype(np.int16))
    # Caption sits on the 22 px bar.
    below = delta[30:, :, :]
    assert int(below.sum()) == 0, "LOW confidence must not paint a ribbon on the stage"

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


def main() -> None:
    check_projection_monotonic()
    check_ribbon_honesty()
    check_overlay_and_command()
    check_stage_ribbon_and_no_gt()
    check_clip_hold()
    check_bench_reports_cpu_ms()
    print("test_viz_fsd_ribbon: all OK")


if __name__ == "__main__":
    main()
