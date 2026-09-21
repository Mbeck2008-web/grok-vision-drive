#!/usr/bin/env python3
"""Offline checks for the OpenCV GVD VISION lexicon (no display required)."""
from __future__ import annotations

import math
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
    assert "PATH_FADE" not in stage_src
    assert "CORRIDOR_FADE_FRAC" in stage_src

    from python.viz.debug_draw import cabin_drive_word

    assert cabin_drive_word({"engaged": False})[0] == "OFF"
    assert cabin_drive_word({"engaged": True, "cmd_reason": "preview_blocked"})[0] == "HOLD"
    assert cabin_drive_word({"engaged": True, "cmd_reason": "veto:e2e_stub", "veto_reason": "e2e_stub"})[0] == "HOLD"
    assert cabin_drive_word({"engaged": True, "cmd_reason": "cmd_json_pending", "cmd_applied": False})[0] == "ON"
    assert cabin_drive_word({
        "engaged": True, "cmd_reason": "cmd_json_applied", "cmd_applied": True, "lua_applying": True,
    })[0] == "DRIVE"
    assert cabin_drive_word({"bus_link": "MISMATCH", "engaged": True})[0] == "MISMATCH"

    from python.viz.nerd import FS_BODY, TH, VAL_COL_W, _extras_line, _nav_line, _text_size, scene_note
    from python.viz.stage import (
        Cam,
        VizUI,
        _track_dims,
        in_path,
        is_bus_mismatch,
        is_halted,
        is_hazard,
        is_slowing,
        lane_draw_mode,
        pace_scale,
        render_stage,
        resolve_corridors,
        resolve_draw_range,
        track_cls,
    )

    assert _track_dims({}, "vehicle") == (4.2, 1.8, 1.55)
    assert _track_dims({"length": 6.0, "width": 2.2, "height": 2.4}, "vehicle") == (6.0, 2.2, 2.4)

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

    # Loud mismatch: path_ego present must not become a fake corridor.
    mismatch_st = dict(st)
    mismatch_st["bus_link"] = "MISMATCH"
    mismatch_st["link"] = "mismatch"
    mismatch_st["python_bus"] = "C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current/Documents/GVD"
    mismatch_st["lua_bus"] = "C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current/Documents/GVD"
    mismatch_st["product"] = "drive"
    mismatch_st["bus_note"] = "python_bus and lua_bus are not the same folder"
    assert is_bus_mismatch(mismatch_st)
    mm = render_stage(mismatch_st, ui=ui)
    assert mm.shape == (800, 1280, 3)
    path_band = frame[520:720, 500:780]
    mm_band = mm[520:720, 500:780]
    assert int(mm_band.mean()) < 25, "mismatch must not paint a corridor"
    assert int(path_band.mean()) > int(mm_band.mean()), "live corridor lights more than mismatch void"
    assert int(mm.max()) > 80, "MISMATCH overlay must be visible"

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

    # DRIVE / VIZ nerd tabs concatenate a wide panel; occupancy overlay is opt-in.
    nerd = VizUI()
    nerd.show_nerd = True
    nerd.show_drive_tab()
    drive_frame = render_stage(st, ui=nerd)
    assert drive_frame.shape == (800, 1280 + nerd.nerd_width, 3)
    tabs = {h.get("id") for h in nerd.nerd_hits if h.get("kind") == "tab"}
    assert tabs == {"live", "drive", "viz", "model", "cams", "keys"}
    nerd.show_model_tab()
    model_frame = render_stage(st, ui=nerd)
    assert model_frame.shape == drive_frame.shape
    assert _text_size("yolov8n-onnx", FS_BODY, TH)[0] <= VAL_COL_W - 4, "MODEL value column must fit shipped id"
    nerd.show_viz_tab()
    nerd.debug.viz_dense = True
    nerd.debug.apply_dense()
    viz_frame = render_stage(st, ui=nerd)
    assert viz_frame.shape == drive_frame.shape
    nerd.show_nerd = False
    nerd.layers = set()
    dense = render_stage(st, ui=nerd)
    nerd.debug.viz_occ = False
    nerd.debug.viz_ids = False
    nerd.debug.viz_vel = False
    nerd.debug.viz_boxes = False
    nerd.debug.viz_frustums = False
    nerd.debug.viz_cost = False
    sparse = render_stage(st, ui=nerd)
    assert not np.array_equal(dense, sparse)

    from python.sensors.cameras import CAM_IDS
    from python.viz.debug_draw import cam_tile_rects, clamp_front_overexpose, draw_cam_tiles
    from python.viz.stage import CAMS_STAGE_BOX, CAMS_STAGE_GRID

    white = np.full((32, 48, 3), 255, dtype=np.uint8)
    clamped = clamp_front_overexpose(white, "main")
    assert clamped is not None and float(clamped.mean()) < 180, "front overexpose clamp should darken blown-white"
    assert float(clamp_front_overexpose(white, "rear").mean()) == 255, "rear must not be clamped"
    dim = np.full((32, 48, 3), 80, dtype=np.uint8)
    assert np.array_equal(clamp_front_overexpose(dim, "main"), dim)

    colors = {}
    frames = {}
    for i, cid in enumerate(CAM_IDS):
        col = (20 + i * 12, 40 + i * 16, 70 + i * 8)
        colors[cid] = col
        frames[cid] = np.full((36, 48, 3), col, dtype=np.uint8)
    health = {cid: "ok" for cid in CAM_IDS}
    health["rear"] = "missing"
    del frames["rear"]

    cams_ui = VizUI()
    cams_ui.show_cams_tab()
    cams_ui.show_nerd = False
    cams_st = dict(st)
    cams_st["loop_hz"] = 12.0
    cams_st["cam_health"] = health
    cams_st["engaged"] = False
    wall = render_stage(cams_st, ui=cams_ui, cam_frames=frames)
    assert wall.shape == (800, 1280, 3)
    rects = cam_tile_rects(*CAMS_STAGE_BOX, *CAMS_STAGE_GRID)
    assert len(rects) >= 8
    for cid, (x, y, tw, th) in zip(CAM_IDS, rects):
        sx, sy = x + int(tw * 0.78), y + th // 2
        pix = wall[sy, sx]
        if cid == "rear":
            assert int(pix.mean()) < 70, f"{cid} missing tile should stay dark, got {pix}"
        else:
            assert np.allclose(pix, colors[cid], atol=8), f"{cid} tile {pix} != {colors[cid]}"

    # retail / stub: only main (or cam_main) filled; others labelled missing
    retail_frames = {"cam_main": np.full((36, 48, 3), (30, 200, 30), dtype=np.uint8)}
    retail_health = {cid: "missing" for cid in CAM_IDS}
    retail_health["main"] = "ok"
    retail = render_stage(cams_st, ui=cams_ui, cam_frames=retail_frames)
    mx, my, mw, mh = rects[list(CAM_IDS).index("main")]
    nx, ny, nw, nh = rects[list(CAM_IDS).index("narrow")]
    assert int(retail[my + mh // 2, mx + int(mw * 0.78)][1]) > 120
    assert int(retail[ny + nh // 2, nx + int(nw * 0.78)].mean()) < 70

    # under 8 Hz: labelled drop, no crash, no fake fill from the colored frames
    slow = dict(cams_st)
    slow["loop_hz"] = 6.0
    dropped = render_stage(slow, ui=cams_ui, cam_frames=frames)
    assert dropped.shape == wall.shape
    dx, dy, dw, dh = rects[0]
    assert int(dropped[dy + dh // 2, dx + int(dw * 0.78)].mean()) < 70, "dropped blit must not copy live pixels"

    # nerd CAMS tab still concatenates and reports 8 slots
    nerd_cams = VizUI()
    nerd_cams.show_nerd = True
    nerd_cams.show_cams_tab()
    nerd_wall = render_stage(cams_st, ui=nerd_cams, cam_frames=frames)
    assert nerd_wall.shape == (800, 1280 + nerd_cams.nerd_width, 3)
    tabs = {h.get("id") for h in nerd_cams.nerd_hits if h.get("kind") == "tab"}
    assert "cams" in tabs
    grid = next(h for h in nerd_cams.nerd_hits if h.get("kind") == "cams_grid")
    assert int(grid.get("n") or 0) == 8

    # cabin/default stage is unchanged when CAMS is not selected
    cabin_ui = VizUI()
    cabin_ui.layers = {0}
    cabin_ui.show_nerd = False
    cabin = render_stage(st, ui=cabin_ui)
    assert cabin.shape == (800, 1280, 3)
    assert int(cabin.mean()) < 80

    # Forecast fans stay off the box face. A ground line ahead of a car still
    # lands inside the chase silhouette, so the solid fill has to cover it.
    import cv2

    bare = VizUI()
    bare.layers = {0}
    bare.show_nerd = False
    bare.debug.viz_forecast = False
    cabin_bare = render_stage(st, ui=bare)
    face_delta = cv2.absdiff(cabin, cabin_bare)
    cam = Cam()
    covered = np.zeros(cabin.shape[:2], np.uint8)
    for tr in st["tracks"]:
        length, width, height = _track_dims(tr, track_cls(tr))
        yaw = float(tr.get("yaw") or 1.57)
        x, y = float(tr["x"]), float(tr["y"])
        hx, hy = math.cos(yaw), math.sin(yaw)
        sx, sy = hy, -hx
        hl, hw = length * 0.5, width * 0.5
        corners = []
        for z in (0.0, height):
            for s0, s1 in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                corners.append(cam.project(
                    x + hx * hl * s0 + sx * hw * s1,
                    y + hy * hl * s0 + sy * hw * s1,
                    z,
                ))
        mask = np.zeros(cabin.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(np.array(corners, np.int32)), 255)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        covered = cv2.bitwise_or(covered, mask)
        assert int(np.count_nonzero((face_delta.sum(axis=2) > 8) & (mask > 0))) == 0, (
            f"forecast stamped track {tr.get('id')}"
        )
        cx, cy = cam.project(x, y, height * 0.45)
        # Chase sits farther back, so several shaded faces fit in a few dozen
        # pixels. The center still has to match its own face: a disc would not.
        center = cabin[cy, cx].astype(np.int16)
        neigh = cabin[cy - 1:cy + 2, cx - 1:cx + 2].reshape(-1, 3).astype(np.int16)
        same = int(np.all(np.abs(neigh - center) <= 2, axis=1).sum())
        assert same >= 4, f"track {tr.get('id')} face center is not a flat fill ({same})"
        assert int(center.max()) < 230, f"track {tr.get('id')} face center is a bright mark"
    assert int(np.count_nonzero((face_delta.sum(axis=2) > 8) & (covered == 0))) > 0, (
        "moving agents should still draw a thin fan past the box"
    )

    # PIP front clamp: blown-white main preview is not left at 255
    pip_ui = VizUI()
    pip_ui.show_nerd = False
    pip_ui.layers = set()
    pip_ui.debug.viz_pip = True
    pip_st = dict(st)
    pip_st["loop_hz"] = 12.0
    pip_frame = render_stage(pip_st, ui=pip_ui, main_frame=np.full((180, 320, 3), 255, dtype=np.uint8))
    pip = pip_frame[12:192, 12:332]
    assert float(pip.mean()) < 200, "PIP should not stay blown-white"

    scratch = np.full((200, 800, 3), 10, dtype=np.uint8)
    n = draw_cam_tiles(
        scratch, None, {cid: "missing" for cid in CAM_IDS},
        x0=4, y0=4, width=790, height=190, cols=4, rows=2, dropped=False,
    )
    assert n == 8

    # Corridor width stays path_width/2 meters. A 40 m path does not grow ice past itself.
    ribbon = {
        "engaged": False,
        "loop_hz": 12.0,
        "policy": "modular",
        "e2e_backend": "stub",
        "veto_reason": "none",
        "path_conf": 0.95,
        "path_width": 2.0,
        "path_debug_preview": False,
        "viz_smoke": False,
        "ego": {"speed_mps": 8.0, "brake": 0.0},
        "planner": {"aeb": "off", "target_v": 8.0, "ttc_lead": 4.0, "corridor_width": 2.0},
        "path_ego": [{"x": 0.0, "y": float(i), "z": 0.0} for i in range(0, 41)],
        "tracks": [],
        "lanes_ext": [],
        "road_edges": [],
        "signs": [],
    }
    primary, ghost = resolve_corridors(ribbon)
    assert ghost is None
    assert primary[-1]["y"] == 40.0
    assert len(primary) == 41
    rib_ui = VizUI()
    rib_ui.layers = {0}
    rib_ui.show_nerd = False
    rib_ui.debug.viz_forecast = False
    rib_ui.debug.viz_lanes = False
    rib_ui.debug.viz_signs = False
    painted = render_stage(ribbon, ui=rib_ui)
    bare_path = dict(ribbon)
    bare_path["path_ego"] = []
    empty = render_stage(bare_path, ui=rib_ui)
    delta = cv2.absdiff(painted, empty)
    y_near = 12.0
    painted_x = []
    for x in np.linspace(-3.0, 3.0, 121):
        px, py = cam.project(float(x), y_near, 0.03)
        if int(delta[py, px].sum()) > 8:
            painted_x.append(float(x))
    assert painted_x, "corridor should paint near the ego"
    world_half = max(abs(min(painted_x)), abs(max(painted_x)))
    assert abs(world_half - float(ribbon["path_width"]) * 0.5) <= 0.25, world_half
    right = max(painted_x)
    half_px = abs(cam.project(right, y_near, 0.03)[0] - cam.project(0.0, y_near, 0.03)[0])
    expected_half = (float(ribbon["path_width"]) * 0.5) * cam.scale_at(y_near)
    assert abs(half_px - expected_half) <= max(8.0, 0.20 * expected_half), (
        f"half-width {half_px:.1f}px != path_width/2 * scale_at ({expected_half:.1f}px)"
    )
    far_px, far_py = cam.project(2.2, y_near, 0.03)
    assert int(delta[far_py, far_px].sum()) <= 8, "ribbon must not inflate out to the lane fan"

    def _ice_at(y: float) -> int:
        px, py = cam.project(0.0, y, 0.03)
        patch = delta[py - 3:py + 4, px - 3:px + 4]
        return int(np.count_nonzero(patch.sum(axis=2) > 8))

    assert _ice_at(20.0) > 0, "ribbon should cover the path that exists"
    assert _ice_at(46.0) == 0, "path ending at 40 m must not paint ice past ~40 m"
    assert _ice_at(55.0) == 0

    e2e_live = dict(ribbon)
    e2e_live["policy"] = "e2e"
    e2e_live["e2e_backend"] = "onnx"
    e2e_live["veto_reason"] = "none"
    e2e_live["path_e2e"] = [{"x": 1.5, "y": float(i), "z": 0.0} for i in range(0, 21)]
    e2e_primary, e2e_ghost = resolve_corridors(e2e_live)
    assert e2e_ghost is None
    assert e2e_primary[-1]["y"] == 20.0
    assert e2e_primary[5]["x"] == 1.5
    held = dict(e2e_live)
    held["veto_reason"] = "e2e_stub"
    held["e2e_backend"] = "stub"
    held_primary, _held_ghost = resolve_corridors(held)
    assert held_primary[-1]["y"] == 40.0 and held_primary[0]["x"] == 0.0
    steer_only = dict(ribbon)
    steer_only["policy"] = "e2e"
    steer_only["e2e_backend"] = "onnx"
    steer_only["veto_reason"] = "none"
    steer_only["shadow"] = {"steer": 0.35}
    steer_only["path_e2e"] = []
    integrated, _no_ghost = resolve_corridors(steer_only)
    assert len(integrated) == 41 and abs(integrated[-1]["y"] - 40.0) < 0.2
    assert any(abs(p["x"]) > 0.05 for p in integrated), "e2e steer must bend the ribbon"
    shadow = dict(ribbon)
    shadow["policy"] = "shadow"
    shadow["e2e_backend"] = "onnx"
    shadow["shadow"] = {"steer": 0.4}
    shadow["path_e2e"] = [{"x": 4.0, "y": float(i), "z": 0.0} for i in range(0, 41)]
    sh_primary, sh_ghost = resolve_corridors(shadow)
    assert sh_primary[8]["x"] == 0.0 and sh_ghost is not None and sh_ghost[8]["x"] == 4.0
    sh_clean = render_stage(shadow, ui=rib_ui)
    sh_ui = VizUI()
    sh_ui.layers = {1}
    sh_ui.show_nerd = False
    sh_ui.debug.viz_forecast = False
    sh_ui.debug.viz_lanes = False
    sh_ui.debug.viz_signs = False
    sh_nerd = render_stage(shadow, ui=sh_ui)
    assert int(cv2.absdiff(sh_clean, painted).sum()) == 0, "key 0 hides the shadow ghost"
    assert int(cv2.absdiff(sh_nerd, sh_clean).sum()) > 0, "nerd shows the other policy ribbon"

    # Cabin span follows cameras.yaml viz + far_m. The ribbon does not.
    draw = resolve_draw_range({})
    assert draw.ahead_min_m >= 80 and draw.behind_min_m >= 20
    assert draw.ahead_m >= draw.ahead_min_m and draw.behind_m >= draw.behind_min_m
    assert abs(draw.ahead_m - 300.0) < 1e-6, draw
    assert abs(draw.behind_m - 80.0) < 1e-6, draw
    assert abs(resolve_draw_range({"main_far_m": 150}).ahead_m - 150.0) < 1e-6
    assert abs(resolve_draw_range({"main_far_m": 20}).ahead_m - draw.ahead_min_m) < 1e-6
    assert abs(resolve_draw_range({"main_far_m": 900}).ahead_m - 400.0) < 1e-6
    assert abs(resolve_draw_range({"main_far_m": 550}, vram_gb=16.0).ahead_m - 550.0) < 1e-6
    assert abs(resolve_draw_range({"rear_far_m": 40}).behind_m - 40.0) < 1e-6
    assert abs(resolve_draw_range({"rear_far_m": 500}).behind_m - 80.0) < 1e-6
    wish = {
        "viz": {
            "draw_ahead_m": 120,
            "draw_behind_m": 40,
            "ahead_min_m": 80,
            "ahead_max_m": 400,
            "behind_min_m": 20,
            "behind_max_m": 80,
            "fade_frac": 0.28,
        },
        "cameras": [
            {"id": "main", "far_m": 300},
            {"id": "rear", "far_m": 100},
        ],
    }
    assert abs(resolve_draw_range({"main_far_m": 200}, cameras=wish).ahead_m - 120.0) < 1e-6
    span_cam = Cam()
    assert abs(span_cam.ahead_m - draw.ahead_m) < 1e-6
    assert abs(span_cam.behind_m - draw.behind_m) < 1e-6
    # Chase stays just behind the ego. Draw range does not drag the lens back.
    assert abs(span_cam.eye[0] - 1.6) < 1e-6
    assert abs(span_cam.eye[1] - (-16.5)) < 1e-6
    assert abs(span_cam.eye[2] - 5.4) < 1e-6

    import tempfile

    from python.viz.stage import _extend_predicted, _span_ys, smoke

    span_ys = _span_ys(-draw.behind_m, draw.ahead_m, 3.0)
    assert span_ys[0] <= -draw.behind_min_m and span_ys[-1] >= draw.ahead_min_m
    ext_pieces = _extend_predicted(
        [{"x": -1.8, "y": float(y)} for y in range(2, 36, 3)],
        -draw.behind_m,
        draw.ahead_m,
    )
    ext_ys = [p["y"] for piece in ext_pieces for p in piece]
    assert ext_ys and min(ext_ys) <= -draw.behind_min_m and max(ext_ys) >= draw.ahead_min_m

    smoke_path = Path(tempfile.mkdtemp()) / "cabin_span.png"
    smoke(use_perception=False, engaged=False, write_bus=False, out=smoke_path)
    cabin_span = cv2.imread(str(smoke_path))
    assert cabin_span is not None and cabin_span.shape[:2] == (800, 1280)

    def _inside(x: float, y: float, z: float = 0.02) -> tuple[int, int]:
        px, py = span_cam.project(x, y, z)
        assert 0 <= px < 1280 and 22 < py < 798, (x, y, px, py)
        return px, py

    def _ground_at(y: float) -> tuple[int, int, int]:
        px, py = _inside(0.45, y, 0.0)
        pix = cabin_span[py, px]
        return int(pix[0]), int(pix[1]), int(pix[2])

    # 70 is ahead of the car. -6 is behind the rear bumper and still in the chase frame.
    for sample_y in (70.0, -6.0):
        gb, gg, gr = _ground_at(sample_y)
        assert max(gb, gg, gr) >= 13, (sample_y, gb, gg, gr)

    def _lane_lit(y: float) -> bool:
        x = -1.85 + 0.9 * math.sin(y / 17.0)
        px, py = span_cam.project(x, y, 0.02)
        if not (0 <= px < 1280 and 22 < py < 798):
            return False
        patch = cabin_span[py - 6:py + 7, px - 6:px + 7]
        return int(patch.max()) > 40

    assert any(_lane_lit(y) for y in (72.0, 76.0, 80.0, 84.0, 88.0)), "stub lanes must reach ahead_min"
    assert any(_lane_lit(y) for y in (-6.0, -4.0, -2.0)), "stub lanes must show behind the ego"

    # A short live poly is dashed out to the cabin span and is not relabeled in state.
    short = {
        "engaged": False,
        "loop_hz": 12.0,
        "policy": "modular",
        "path_ego": [],
        "path_width": 2.0,
        "path_debug_preview": False,
        "viz_smoke": False,
        "lanes_ext": [{
            "points": [{"x": -1.8, "y": float(y)} for y in range(2, 36, 3)],
            "kind": "detected",
            "index": -1,
        }],
        "road_edges": [],
        "tracks": [],
        "signs": [],
        "missing_state_keys": ["live cameras"],
    }
    lane_ui = VizUI()
    lane_ui.layers = {0}
    lane_ui.show_nerd = False
    lane_ui.debug.viz_forecast = False
    lane_ui.debug.viz_signs = False
    ys_before = [p["y"] for p in short["lanes_ext"][0]["points"]]
    keys_before = list(short["missing_state_keys"])
    lane_on = render_stage(short, ui=lane_ui)
    assert [p["y"] for p in short["lanes_ext"][0]["points"]] == ys_before
    assert short["lanes_ext"][0]["kind"] == "detected"
    assert short["missing_state_keys"] == keys_before
    lane_off = render_stage({**short, "lanes_ext": []}, ui=lane_ui)
    lane_delta = cv2.absdiff(lane_on, lane_off)

    def _ext_hit(y: float) -> int:
        px, py = span_cam.project(-1.8, y, 0.02)
        if not (4 <= px < 1276 and 24 < py < 796):
            return -1
        patch = lane_delta[py - 5:py + 6, px - 5:px + 6]
        return int(np.count_nonzero(patch.sum(axis=2) > 8))

    assert _ext_hit(20.0) > 0, "detected paint still strokes"
    assert any(_ext_hit(y) > 0 for y in (72.0, 80.0, 88.0)), "predicted dash reaches ahead"
    assert any(_ext_hit(y) > 0 for y in (-6.0, -4.0, -2.0)), "predicted dash shows behind the ego"

    # Under 8 Hz the lane stroke and the ribbon stay. Signs drop.
    slow = dict(short)
    slow["loop_hz"] = 6.0
    slow["path_ego"] = [{"x": 0.0, "y": float(i), "z": 0.0} for i in range(0, 41)]
    slow["path_conf"] = 0.95
    slow["signs"] = [{"cls": "stop_sign", "x": -4.0, "y": 22.0}]
    sign_ui = VizUI()
    sign_ui.layers = {0}
    sign_ui.show_nerd = False
    sign_ui.debug.viz_forecast = False
    sign_ui.debug.viz_lanes = True
    sign_ui.debug.viz_signs = True
    slow_full = render_stage(slow, ui=sign_ui)
    slow_nolane = render_stage({**slow, "lanes_ext": []}, ui=sign_ui)
    slow_nopath = render_stage({**slow, "path_ego": []}, ui=sign_ui)
    assert int(cv2.absdiff(slow_full, slow_nolane).sum()) > 0, "under 8 Hz lanes stay"
    assert int(cv2.absdiff(slow_full, slow_nopath).sum()) > 0, "under 8 Hz ribbon stays"
    fast_signs = render_stage({**slow, "loop_hz": 12.0}, ui=sign_ui)
    assert int(cv2.absdiff(slow_full, fast_signs).sum()) > 0, "under 8 Hz signs drop"

    print("test_gvd_viz_stage: OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"test_gvd_viz_stage: FAIL — {e}")
        sys.exit(1)
