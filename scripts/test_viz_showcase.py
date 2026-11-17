#!/usr/bin/env python3
"""Showcase presets, layouts, and offline interpolation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _state(x=0.0, speed=8.0):
    return {
        "engaged": True,
        "path_conf": 0.9,
        "ego": {"speed_mps": speed},
        "planner": {"cipv_id": 1, "aeb": "off", "target_v": 10.0},
        "cmd_reason": "cmd_json_applied",
        "cmd_applied": True,
        "lua_applying": True,
        "path_ego": [{"x": 0.1 * i, "y": float(i)} for i in range(2, 30, 2)],
        "lanes_ext": [
            {"points": [{"x": -1.8, "y": float(y)} for y in range(4, 40, 2)], "kind": "detected", "index": -1},
            {"points": [{"x": 1.8, "y": float(y)} for y in range(4, 40, 2)], "kind": "detected", "index": 1},
        ],
        "tracks": [{"id": 1, "class": "vehicle", "x": x, "y": 16.0, "yaw": 1.57}],
    }


def test_presets_and_layouts() -> None:
    from python.viz.showcase import PRESETS, preset_size, render_showcase
    from python.viz.stage import VizUI, render_stage

    ui = VizUI()
    ui.show_nerd = False
    ui.showcase = True
    ui.style = "early"
    cam = np.full((180, 320, 3), (20, 40, 180), dtype=np.uint8)
    for name in ("square", "story", "wide"):
        w, h = preset_size(name)
        assert (w, h) == PRESETS[name]
        frame = render_showcase(_state(), ui, width=w, height=h, main_frame=cam)
        assert frame.shape == (h, w, 3), (name, frame.shape)
        # Mark sits in the bottom-right and is brighter than the black field.
        assert int(frame[-20, -20].max()) > 80 or int(frame[-22, -40].max()) > 80
    ui.showcase_layout = "split"
    split = render_showcase(_state(), ui, width=640, height=360, main_frame=cam)
    left = split[:, :320]
    assert float(left[:, :, 2].mean()) > float(left[:, :, 0].mean()) + 40
    ui.showcase_layout = "pip"
    pip = render_showcase(_state(), ui, width=640, height=360, main_frame=cam)
    inset = pip[20:40, 20:40]
    assert float(inset[:, :, 2].mean()) > 100
    # Live path uses the same painter and stays off unless asked.
    off = VizUI()
    off.show_nerd = False
    off.style = "early"
    a = render_stage(_state(), ui=off, width=320, height=180)
    off.showcase = False
    assert a.shape == (180, 320, 3)
    print("showcase sizes and layouts OK")


def test_lerp() -> None:
    from python.viz.showcase import lerp_state

    a = _state(0.0, 4.0)
    b = _state(4.0, 12.0)
    mid = lerp_state(a, b, 0.5)
    assert abs(mid["ego"]["speed_mps"] - 8.0) < 1e-6
    assert abs(mid["tracks"][0]["x"] - 2.0) < 1e-6
    assert abs(lerp_state(a, b, 0.0)["tracks"][0]["x"] - 0.0) < 1e-6
    assert abs(lerp_state(a, b, 1.0)["tracks"][0]["x"] - 4.0) < 1e-6
    # A track id that exists on only one side is kept, not invented.
    a["tracks"].append({"id": 9, "class": "pedestrian", "x": 1.0, "y": 8.0})
    mixed = lerp_state(a, b, 0.5)
    assert any(tr.get("id") == 9 for tr in mixed["tracks"])
    print("showcase interpolation OK")


def main() -> None:
    test_presets_and_layouts()
    test_lerp()
    print("test_viz_showcase: OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"test_viz_showcase: FAIL — {exc}")
        sys.exit(1)
