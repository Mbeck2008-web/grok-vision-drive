#!/usr/bin/env python3
"""Vector stage: black field, dotted lanes, wire boxes, overlay, legacy still selectable."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _ui(style="early", overlay=False, top_down=False):
    from python.viz.stage import VizUI

    ui = VizUI()
    ui.show_nerd = False
    ui.style = style
    ui.overlay = overlay
    ui.top_down = top_down
    return ui


def _lane(x, index, y0=8.0, y1=40.0, kind="detected"):
    ys = []
    y = float(y0)
    while y <= y1 + 1e-6:
        ys.append(round(y, 3))
        y += 2.0
    return {
        "points": [{"x": float(x), "y": y} for y in ys],
        "kind": kind,
        "index": index,
        "side": "left" if x < 0 else "right",
        "style": "dashed",
    }


def _straight(x, y0, y1, step):
    pts = []
    y = float(y0)
    while y <= y1 + 1e-6:
        pts.append({"x": float(x), "y": round(y, 3)})
        y += step
    return pts


def _base(**extra):
    st = {
        "engaged": True,
        "loop_hz": 12.0,
        "path_conf": 0.92,
        "path_width": 2.0,
        "ego": {"speed_mps": 12.4, "brake": 0.0},
        "planner": {"cipv_id": 1, "aeb": "off", "target_v": 11.0, "ttc_lead": 3.0},
        "cmd_reason": "cmd_json_applied",
        "cmd_applied": True,
        "lua_applying": True,
        "path_ego": _straight(2.8, 8.0, 40.0, 2.0),
        "lanes_ext": [_lane(-3.2, -1), _lane(1.8, 1), _lane(-6.2, -2)],
        "road_edges": [{"points": _straight(-8.8, 8.0, 40.0, 2.0), "kind": "detected", "side": "left"}],
        "tracks": [
            {"id": 1, "class": "vehicle", "x": 0.3, "y": 18.0, "yaw": 1.57},
            {"id": 2, "class": "vehicle", "x": 4.4, "y": 22.0, "yaw": 1.57},
            {"id": 3, "class": "truck", "x": -4.8, "y": 16.0, "yaw": 1.57},
            {"id": 4, "class": "pedestrian", "x": 6.2, "y": 14.0, "yaw": 1.57},
            {"id": 5, "class": "bike", "x": -7.0, "y": 12.0, "yaw": 1.57},
        ],
        "signs": [
            {"cls": "traffic_light", "x": 4.0, "y": 26.0, "state": "green"},
            {"cls": "stop_sign", "x": -2.4, "y": 24.0},
        ],
    }
    st.update(extra)
    return st


def _render(state, ui, w=1280, h=720, frame=None):
    from python.viz.stage import render_stage

    return render_stage(state, ui=ui, width=w, height=h, main_frame=frame)


def _pix(frame, u, v):
    h, w = frame.shape[:2]
    x = min(w - 1, max(0, int(u)))
    y = min(h - 1, max(0, int(v)))
    return frame[y, x]


def _is_white(p):
    return int(p[0]) > 200 and int(p[1]) > 200 and int(p[2]) > 200


def _is_red(p):
    return int(p[2]) > 180 and int(p[2]) > int(p[1]) + 60 and int(p[2]) > int(p[0]) + 60


def _is_purple(p):
    return int(p[0]) > 160 and int(p[2]) > 160 and int(p[1]) < 90


def _is_blue(p):
    return int(p[0]) > 180 and int(p[0]) > int(p[2]) + 40 and int(p[1]) > 80


def _is_green(p):
    return int(p[1]) > 180 and int(p[1]) > int(p[0]) + 40 and int(p[1]) > int(p[2]) + 40


def _is_orange(p):
    return int(p[2]) > 180 and int(p[1]) > 80 and int(p[2]) > int(p[0]) + 40 and int(p[1]) > int(p[0]) + 20


def _is_magenta(p):
    return int(p[0]) > 180 and int(p[2]) > 150 and int(p[1]) < 90


def _is_yellow(p):
    return int(p[1]) > 180 and int(p[2]) > 180 and int(p[0]) < 80


def test_black_and_dots() -> None:
    from python.viz.early_vector import GROUND_Z, make_early_cam

    ui = _ui()
    # Lane only, so the midpoint is not a box or a path dot.
    st = {
        "path_conf": 0.9,
        "path_ego": [],
        "lanes_ext": [_lane(-3.2, -1)],
        "road_edges": [],
        "tracks": [],
        "signs": [],
        "ego": {"speed_mps": 5.0},
    }
    frame = _render(st, ui)
    assert frame.shape == (720, 1280, 3)
    assert int(frame[4, -4].max()) == 0, "corner stays black"
    cam = make_early_cam(1280, 720)
    u, v = cam.project(-3.2, 8.0, GROUND_Z)
    assert _is_white(_pix(frame, u, v)), _pix(frame, u, v)
    mu, mv = cam.project(-3.2, 9.0, GROUND_Z)
    assert int(_pix(frame, mu, mv).max()) < 25, "dotted lane keeps a gap"
    print("black field and dotted lane gap OK")


def test_colors() -> None:
    from python.viz.early_vector import (
        CENTER_Z_FRAC,
        GROUND_Z,
        LIGHT_Z,
        SIGN_Z,
        class_dims,
        make_early_cam,
    )

    frame = _render(_base(), _ui())
    cam = make_early_cam(1280, 720)
    assert _is_white(_pix(frame, *cam.project(-3.2, 8.0, GROUND_Z)))
    assert _is_purple(_pix(frame, *cam.project(-6.2, 8.0, GROUND_Z))), "outer lane is purple"
    assert _is_red(_pix(frame, *cam.project(-8.8, 8.0, GROUND_Z))), "curb is red"
    assert _is_blue(_pix(frame, *cam.project(2.8, 8.0, GROUND_Z))), "path is blue"
    assert int(_pix(frame, *cam.project(2.8, 9.0, GROUND_Z)).max()) < 30, "path dots have gaps"

    def center(tr):
        _l, _w, h = class_dims(tr)
        return cam.project(float(tr["x"]), float(tr["y"]), h * CENTER_Z_FRAC)

    tracks = {t["id"]: t for t in _base()["tracks"]}
    assert _is_green(_pix(frame, *center(tracks[1]))), "lead is green"
    assert _is_blue(_pix(frame, *center(tracks[2]))), "car is blue"
    assert _is_orange(_pix(frame, *center(tracks[3]))), "truck is orange"
    assert _is_magenta(_pix(frame, *center(tracks[4]))), "ped is magenta"
    assert _is_yellow(_pix(frame, *center(tracks[5]))), "bike is yellow"
    light = cam.project(4.0, 26.0, LIGHT_Z)
    assert _is_green(_pix(frame, *light)), "lit lamp"
    stop = cam.project(-2.4, 24.0, SIGN_Z)
    assert _is_red(_pix(frame, *stop)), "stop icon"
    # Glow sits beside the body, under the car, brighter than the black field.
    gu, gv = cam.project(0.0, 0.25, GROUND_Z)
    glow = _pix(frame, gu + 120, gv)
    assert int(glow.max()) >= 18, glow
    assert int(glow.max()) < 160, glow
    print("role colors, lamp, stop icon, ego glow OK")


def test_curve_follows_polyline() -> None:
    from python.viz.early_vector import GROUND_Z, make_early_cam

    pts = []
    for y in range(8, 42, 2):
        x = 3.0 * math_sin((y - 8) / 10.0)
        pts.append({"x": x, "y": float(y)})
    st = {"lanes_ext": [{"points": pts, "kind": "detected", "index": -1}], "path_ego": [], "tracks": []}
    frame = _render(st, _ui())
    cam = make_early_cam(1280, 720)
    bulge_y = 24.0
    bulge_x = 3.0 * math_sin((bulge_y - 8) / 10.0)
    # Chord from the first sample to the last is not the bulge.
    u, v = cam.project(bulge_x, bulge_y, GROUND_Z)
    assert int(_pix(frame, u, v).max()) > 150, "dot sits on the bend"
    cu, cv = cam.project(0.0, bulge_y, GROUND_Z)
    assert int(_pix(frame, cu, cv).max()) < 25, "bend is not the straight chord"
    print("lane dots follow the bend OK")


def math_sin(v):
    import math
    return math.sin(v)


def test_zero_and_legacy_and_gt() -> None:
    empty = _render({}, _ui())
    assert empty.shape == (720, 1280, 3)
    assert int(empty[4, -4].max()) == 0
    assert int(empty[430:680, 500:780].max()) > 140, "ego still draws with no detections"

    st = _base()
    legacy = _render(st, _ui("legacy"))
    early = _render(st, _ui("early"))
    default = _render(st, _ui("legacy"))
    bare = _ui()
    bare.style = "legacy"
    same = _render(st, bare)
    assert np.array_equal(legacy, same)
    assert int(np.abs(legacy.astype(np.int16) - early.astype(np.int16)).sum()) > 1000

    marked = dict(st)
    marked["debug_gt"] = True
    marked["path_world"] = [{"x": 80.0, "y": float(i), "z": 0.0} for i in range(40)]
    a = _render(st, _ui(top_down=True), w=640, h=480)
    b = _render(marked, _ui(top_down=True), w=640, h=480)
    assert np.array_equal(a, b), "map pose must not change the vector frame"
    print("empty frame, legacy selectable, ground-truth ignored OK")


def test_overlay_and_ground_lock() -> None:
    from python.train.scene_ckpt_runtime import project_main_ground
    from python.viz.early_vector import CENTER_Z_FRAC, GROUND_Z, class_dims, project_main_xyz

    locked = project_main_ground(1.0, 15.0, 1280, 720)
    got = project_main_xyz(1.0, 15.0, 0.0, 1280, 720)
    assert locked == got, (locked, got)

    cam_color = (40, 70, 110)
    src = np.zeros((720, 1280, 3), dtype=np.uint8)
    src[:, :] = cam_color
    st = _base()
    st["tracks"] = [{"id": 7, "class": "truck", "x": -2.2, "y": 16.0, "yaw": 1.57}]
    st["path_ego"] = _straight(0.0, 8.0, 36.0, 2.0)
    st["road_edges"] = [{"points": _straight(4.2, 12.0, 36.0, 2.0), "kind": "detected"}]
    frame = _render(st, _ui(overlay=True), frame=src)
    assert tuple(int(c) for c in frame[12, 12]) == cam_color, "sky of the camera stays"
    assert tuple(int(c) for c in frame[-4, 20]) != cam_color, "speed bar replaces the bottom"
    assert int(frame[-16].max()) > 180, "speed readout is on the bar"
    assert _is_blue(frame[-22, 40]), frame[-22, 40]
    length, _w, height = class_dims(st["tracks"][0])
    box = project_main_xyz(-2.2, 16.0, height * CENTER_Z_FRAC, 1280, 720)
    assert box is not None and _is_orange(_pix(frame, *box)), _pix(frame, *box)
    path = project_main_xyz(0.0, 12.0, GROUND_Z, 1280, 720)
    assert path is not None and _is_blue(_pix(frame, *path))
    curb = project_main_xyz(4.2, 12.0, GROUND_Z, 1280, 720)
    assert curb is not None and _is_red(_pix(frame, *curb))
    bare = _render({}, _ui(overlay=True))
    assert bare.shape == (720, 1280, 3)
    assert int(bare[-4].max()) > 0
    print("overlay camera, speed bar, ground lock OK")


def test_parser_and_time() -> None:
    from python.run_vision import build_parser

    args = build_parser().parse_args([])
    assert args.viz_style == "early"
    assert args.viz_overlay is False
    assert args.viz_debug is False
    assert args.viz_showcase is False
    legacy = build_parser().parse_args(["--viz-style", "legacy", "--viz-overlay"])
    assert legacy.viz_style == "legacy" and legacy.viz_overlay is True

    st = _base()
    ui = _ui()
    _render(st, ui)
    times = []
    for _ in range(12):
        t0 = time.perf_counter()
        _render(st, ui)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    med = times[len(times) // 2]
    print(f"early 1280x720 median {med:.2f} ms  min {times[0]:.2f}  max {times[-1]:.2f}")
    assert med < 8.0, med


def main() -> None:
    test_black_and_dots()
    test_colors()
    test_curve_follows_polyline()
    test_zero_and_legacy_and_gt()
    test_overlay_and_ground_lock()
    test_parser_and_time()
    print("test_viz_early_fsd: OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"test_viz_early_fsd: FAIL — {exc}")
        sys.exit(1)
