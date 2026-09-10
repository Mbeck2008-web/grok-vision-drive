#!/usr/bin/env python3
"""Offline checks for the OpenCV GVD VISION lexicon (no display required)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BANNED = [
    (r"tesla", "Tesla marks"),
    (r"\bfsd\b", "FSD label"),
    (r"full[\s-]*self[\s-]*driv", "Full Self-Driving label"),
    (r"autopilot", "Autopilot label"),
]


def main() -> None:
    stage_src = (ROOT / "python" / "viz" / "stage.py").read_text(encoding="utf-8")
    nerd_src = (ROOT / "python" / "viz" / "nerd.py").read_text(encoding="utf-8")
    for name, text in (("stage.py", stage_src), ("nerd.py", nerd_src)):
        low = text.lower()
        for pattern, label in BANNED:
            hit = re.search(pattern, low)
            assert hit is None, f"{name}: {label} found ({hit.group(0)!r})"
    assert "GVD" in stage_src and "VISION" in stage_src

    from python.viz.nerd import _extras_line, _nav_line, scene_note
    from python.viz.stage import (
        VizUI,
        in_path,
        is_halted,
        is_hazard,
        is_slowing,
        lane_draw_mode,
        pace_scale,
        render_stage,
    )

    assert lane_draw_mode("detected", smoke=False) == "solid"
    assert lane_draw_mode("predicted", smoke=False) == "dashed"
    assert lane_draw_mode("stub", smoke=False) is None, "live must skip stub paint"
    assert lane_draw_mode("stub", smoke=True) == "dashed"

    coast = {"planner": {"aeb": "off", "target_v": 12.0, "ttc_lead": 3.0, "cipv_id": 1},
             "ego": {"speed_mps": 12.0, "brake": 0.0}}
    assert not is_slowing(coast) and not is_halted(coast) and pace_scale(coast) == 1.0
    slow = {"planner": {"aeb": "off", "target_v": 8.0, "ttc_lead": 2.4, "cipv_id": 1},
            "ego": {"speed_mps": 12.0, "brake": 0.0}}
    assert is_slowing(slow) and not is_halted(slow) and pace_scale(slow) == 0.78
    brake = {"planner": {"aeb": "brake", "target_v": 0.0, "ttc_lead": 0.8, "cipv_id": 1},
             "ego": {"speed_mps": 12.0, "brake": 0.4}}
    assert is_halted(brake) and pace_scale(brake) == 0.55
    lead = {"id": 1, "class": "vehicle", "x": 0.2, "y": 16}
    other = {"id": 2, "class": "vehicle", "x": 3.5, "y": 22}
    assert is_hazard(lead, brake, 1) and not is_hazard(other, brake, 1)
    path = [{"x": 0.0, "y": float(i)} for i in range(0, 30)]
    assert in_path(lead, path, 2.0) and not in_path(other, path, 2.0)
    assert not in_path(lead, [], 2.0)

    note = scene_note({
        "lanes_ext": [
            {"kind": "detected", "index": -1},
            {"kind": "detected", "index": 1},
            {"kind": "predicted", "index": -2},
            {"kind": "stub", "index": 3},
        ],
        "road_edges": [{"kind": "predicted"}],
        "signs": [{"cls": "stop_sign"}, {"cls": "traffic_light"}, {"cls": "pole"}],
    })
    assert "2 seen" in note and "1 pred" in note and "1 stub" in note
    assert "edges pred" in note and "2 signs" in note and "1 pole" in note
    assert _nav_line({"capture_backend": "window", "nav": {"mode": "missing"}}) == ""
    hint = _nav_line(
        {
            "capture_backend": "beamngpy",
            "nav": {
                "mode": "hint",
                "gps": {"lat": 53.09, "lon": 8.81, "ok": True},
                "pin": {"lat": 53.10, "lon": 8.82, "name": "west gate"},
                "range_m": 1234.0,
                "bearing_rel_deg": 45.0,
            },
        }
    )
    assert "nav hint" in hint and "west gate" in hint and "not routing" in hint
    assert _nav_line({"capture_backend": "beamngpy", "nav": {"mode": "missing"}}) == "nav GPS missing"
    extras = _extras_line(
        {
            "sensors": {
                "imu": "lua",
                "gps": "lua_pose",
                "lidar": "missing",
                "radar": "missing",
                "foxglove": "off",
                "drive_uses": "vision",
            }
        }
    )
    assert "imu=lua" in extras and "gps=lua_pose" in extras and "drive=vision" in extras
    assert _extras_line({}) == ""

    import numpy as np

    st = {
        "engaged": True,
        "loop_hz": 12.0,
        "path_conf": 0.9,
        "path_width": 2.0,
        "path_debug_preview": False,
        "gvd_show_path": True,
        "show_agent_ghosts": True,
        "viz_smoke": True,
        "ego": {"speed_mps": 14.0, "brake": 0.0},
        "planner": {"cipv_id": 1, "aeb": "off", "target_v": 11.0, "ttc_lead": 2.4},
        "path_ego": [{"x": 0.0, "y": float(i)} for i in range(0, 36)],
        "tracks": [
            {"id": 1, "class": "vehicle", "x": 0.1, "y": 16, "speed_mps": 12, "yaw": 1.57},
            {"id": 2, "class": "vehicle", "x": 3.2, "y": 24, "speed_mps": 8, "yaw": 1.57},
        ],
        "lanes_ext": [
            {"points": [{"x": -1.8, "y": float(y)} for y in range(2, 36, 3)],
             "kind": "detected", "side": "left", "style": "unknown", "index": -1},
            {"points": [{"x": 1.8, "y": float(y)} for y in range(2, 36, 3)],
             "kind": "detected", "side": "right", "style": "unknown", "index": 1},
            {"points": [{"x": -5.3, "y": float(y)} for y in range(2, 36, 3)],
             "kind": "predicted", "side": "left", "style": "unknown", "index": -2},
        ],
        "road_edges": [
            {"points": [{"x": -9.2, "y": float(y)} for y in range(2, 36, 3)],
             "kind": "predicted", "side": "left"},
        ],
        "signs": [
            {"cls": "stop_sign", "x": -5.6, "y": 22.0, "conf": 0.5},
            {"cls": "traffic_light", "x": 1.2, "y": 30.0, "conf": 0.5, "state": "unknown"},
        ],
        "missing_state_keys": [],
    }
    ui = VizUI()
    ui.layers = {0}
    ui.show_nerd = False
    frame = render_stage(st, ui=ui)
    assert frame.ndim == 3 and frame.shape[0] == 800 and frame.shape[1] == 1280
    assert frame.dtype == np.uint8
    # void stage is near-black; a fully white or empty frame means the draw failed
    assert int(frame.mean()) < 80
    assert int(frame.max()) > 80, "lexicon should light ice/paper pixels"

    # stub lanes must not be drawn unless viz_smoke
    stub_only = dict(st)
    stub_only["viz_smoke"] = False
    stub_only["lanes_ext"] = [{
        "points": [{"x": -1.8, "y": float(y)} for y in range(2, 36, 3)],
        "kind": "stub", "side": "left", "index": -1,
    }]
    live_stub = render_stage(stub_only, ui=ui)
    smoke_stub = dict(stub_only)
    smoke_stub["viz_smoke"] = True
    smoke_frame = render_stage(smoke_stub, ui=ui)
    # drawing stub paint adds paper-ish pixels; skip vs draw should differ
    assert not np.array_equal(live_stub, smoke_frame)

    heavy = dict(st)
    heavy["loop_hz"] = 6.0
    heavy_frame = render_stage(heavy, ui=ui)
    assert heavy_frame.shape == frame.shape

    print("test_gvd_viz_stage: OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"test_gvd_viz_stage: FAIL — {e}")
        sys.exit(1)
