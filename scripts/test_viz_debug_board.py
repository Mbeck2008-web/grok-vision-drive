#!/usr/bin/env python3
"""Debug board: off is free, on shows grid, cameras, map, and graphs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _state():
    return {
        "engaged": True,
        "loop_hz": 12.5,
        "t": 3.5,
        "path_conf": 0.8,
        "ego": {"speed_mps": 9.0, "yaw_rate": 0.02, "steer": 0.1, "throttle": 0.3, "brake": 0.0},
        "planner": {"cipv_id": 1, "aeb": "off", "target_v": 10.0},
        "cmd_reason": "cmd_json_applied",
        "cmd_applied": True,
        "lua_applying": True,
        "path_ego": [{"x": 0.0, "y": float(i)} for i in range(2, 30, 2)],
        "lanes_ext": [
            {"points": [{"x": -1.8, "y": float(y)} for y in range(4, 40, 2)], "kind": "detected", "index": -1},
            {"points": [{"x": 1.8, "y": float(y)} for y in range(4, 40, 2)], "kind": "detected", "index": 1},
        ],
        "tracks": [
            {"id": 1, "class": "vehicle", "x": 0.2, "y": 16.0, "yaw": 1.57},
            {"id": 2, "class": "truck", "x": 4.0, "y": 20.0, "yaw": 1.57},
        ],
        "flags": {"WET_ROAD": 0.8, "RAINING": 0.1},
        "calib": {"pitch": 0.1, "yaw": 1.2, "roll": 0.0},
    }


def _ui(board=False):
    from python.viz.stage import VizUI

    ui = VizUI()
    ui.show_nerd = False
    ui.style = "early"
    ui.debug_board = board
    return ui


def test_off_is_free() -> None:
    from python.viz.debug_board import calls, reset_calls
    from python.viz.stage import VizUI, render_stage

    reset_calls()
    st = _state()
    ui = _ui(False)
    a = render_stage(st, ui=ui, width=640, height=360)
    b = render_stage(dict(st), ui=_ui(False), width=640, height=360)
    assert np.array_equal(a, b)
    assert calls() == 0, calls()
    assert ui.handle_key(ord("b")) is True
    assert ui.debug_board is True
    assert ui.handle_key(ord("d")) is True
    assert ui.nerd_tab == "drive"
    plain = VizUI()
    assert plain.debug_board is False
    print("debug off does not draw the board; b toggles, d stays DRIVE OK")


def test_panels() -> None:
    from python.viz.debug_board import calls, panel_rects, reset_calls
    from python.viz.stage import render_stage

    reset_calls()
    ui = _ui(True)
    main = np.full((240, 320, 3), (210, 40, 30), dtype=np.uint8)
    side = np.full((120, 160, 3), (20, 20, 200), dtype=np.uint8)
    frames = {"main": main, "pillarL": side}
    st = _state()
    frame = render_stage(st, ui=ui, width=1280, height=720, main_frame=main, cam_frames=frames)
    assert frame.shape == (720, 1280, 3)
    assert calls() == 1
    rects = panel_rects(1280, 720)
    for name in ("grid", "cams", "map", "speed", "yaw", "pedals"):
        x0, y0, x1, y1 = rects[name]
        assert x1 > x0 and y1 > y0
        patch = frame[y0:y1, x0:x1]
        assert int(patch.max()) > 20, name
    gx0, gy0, gx1, gy1 = rects["grid"]
    grid = frame[gy0:gy1, gx0:gx1]
    green = (grid[:, :, 1] > 50) & (grid[:, :, 1] > grid[:, :, 0] + 10) & (grid[:, :, 1] > grid[:, :, 2] + 10)
    assert int(green.sum()) > 200, "grid or lead should light green pixels"
    cx0, cy0, cx1, cy1 = rects["cams"]
    cams = frame[cy0:cy1, cx0:cx1]
    # Main tile is the top third and keeps the blue-heavy camera.
    top = cams[: cams.shape[0] // 3]
    assert int(top[:, :, 0].mean()) > 80, top[:, :, 0].mean()
    # A hot flag paints green text. A cold flag does not paint the whole tile.
    hot = (top[:, :, 1] > 180) & (top[:, :, 1] > top[:, :, 2] + 40) & (top[:, :, 0] < 120)
    assert int(hot.sum()) > 10, "flag above 0.5 is green"
    # Second render with a new speed moves the speed graph.
    st["ego"] = dict(st["ego"], speed_mps=20.0)
    frame2 = render_stage(st, ui=ui, width=1280, height=720, main_frame=main, cam_frames=frames)
    sx0, sy0, sx1, sy1 = rects["speed"]
    assert int(np.abs(frame[sy0:sy1, sx0:sx1].astype(int) - frame2[sy0:sy1, sx0:sx1].astype(int)).sum()) > 0
    # Missing cameras and an empty state still return a frame.
    bare = render_stage({}, ui=_ui(True), width=800, height=450)
    assert bare.shape == (450, 800, 3)
    assert calls() == 3
    print("debug panels, readout, graphs, empty state OK")


def main() -> None:
    test_off_is_free()
    test_panels()
    print("test_viz_debug_board: OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"test_viz_debug_board: FAIL — {exc}")
        sys.exit(1)
