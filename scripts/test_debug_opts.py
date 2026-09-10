#!/usr/bin/env python3
"""Offline checks for nerd DRIVE/VIZ knobs (perception, command, chrome)."""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHROME_RE = re.compile(r"tesla|\bfsd\b|full self[- ]driving|autopilot", re.I)

from python.control.actuate import DriveCommand
from python.runtime.debug_opts import (
    CONTROL_ROWS,
    DEBUG_ROWS,
    VIZ_CONTROL_ROWS,
    apply_to_command,
    apply_to_perception,
    DebugOpts,
    row_at,
    viz_row_at,
)
from python.viz.stage import STAGE_H, STAGE_W, VizUI, render_stage


def _pout(**kw):
    base = dict(
        planner={"aeb": "off", "ttc_lead": 0.8, "target_v": 12.0, "cipv_id": 1, "corridor_width": 2.0},
        tracks=[{"id": 1, "class": "vehicle", "x": 0.2, "y": 16.0}],
        tracks_n=1,
        objects_n=1,
        signs=[{"cls": "stop_sign"}],
        dets=[{"cls": "vehicle", "conf": 0.9, "xyxy": [10, 10, 40, 40]}],
        path_debug_preview=False,
        lane_conf=0.8,
        path_conf=0.7,
        path_width=2.0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_apply_perception() -> None:
    opts = DebugOpts()
    out = apply_to_perception(opts, _pout())
    assert out.planner["aeb"] == "brake", out.planner  # ttc 0.8 < default 1.2
    assert out.planner["target_v"] == 0.0

    opts.aeb_on = False
    out = apply_to_perception(opts, _pout())
    assert out.planner["aeb"] == "off"
    assert out.planner["target_v"] == 12.0

    opts = DebugOpts(detector_on=False)
    out = apply_to_perception(opts, _pout())
    assert out.tracks == [] and out.tracks_n == 0 and out.dets == []
    assert out.planner["cipv_id"] is None

    opts = DebugOpts(lanes_on=False)
    out = apply_to_perception(opts, _pout())
    assert out.path_debug_preview is True
    assert out.lane_conf == 0.0

    opts = DebugOpts(cruise_mps=8.0, aeb_on=False, speed_cap=6.0)
    out = apply_to_perception(opts, _pout())
    assert out.planner["target_v"] == 6.0

    opts = DebugOpts(corridor_width=3.4)
    out = apply_to_perception(opts, _pout())
    assert out.path_width == 3.4
    assert out.planner["corridor_width"] == 3.4


def test_apply_command() -> None:
    cmd = DriveCommand(steer=0.5, throttle=0.4, brake=0.0, seq=3, reason="ok")
    opts = DebugOpts()
    out = apply_to_command(opts, cmd)
    assert abs(out.steer - 0.5) < 1e-6 and abs(out.throttle - 0.4) < 1e-6

    opts.invert_steer = True
    opts.steer_gain = 2.0
    opts.max_steer = 0.6
    out = apply_to_command(opts, cmd)
    assert abs(out.steer + 0.6) < 1e-6  # -0.5*2 clipped to -0.6

    opts = DebugOpts(steer_on=False, throttle_on=False)
    out = apply_to_command(opts, cmd)
    assert out.steer == 0.0 and out.throttle == 0.0

    opts = DebugOpts(hold_brake=True)
    out = apply_to_command(opts, cmd)
    assert out.throttle == 0.0 and out.brake == 1.0

    opts = DebugOpts(freeze_cmd=True)
    first = apply_to_command(opts, cmd)
    second = apply_to_command(opts, DriveCommand(steer=-1, throttle=1, brake=1, seq=9, reason="ok"))
    assert abs(second.steer - first.steer) < 1e-6
    assert second.seq == 9


def test_toggle_nudge() -> None:
    opts = DebugOpts()
    row = next(r for r in CONTROL_ROWS if r["id"] == "allow_preview")
    assert opts.allow_preview is False
    opts.toggle(row)
    assert opts.allow_preview is True
    gain = next(r for r in CONTROL_ROWS if r["id"] == "steer_gain")
    opts.nudge(gain, +1)
    assert abs(opts.steer_gain - 1.1) < 1e-6
    pol = next(r for r in CONTROL_ROWS if r["id"] == "policy")
    opts.toggle(pol)
    assert opts.policy == "modular"
    assert opts.effective_policy("shadow") == "modular"
    opts.policy = "session"
    assert opts.effective_policy("e2e") == "e2e"

    dense = next(r for r in VIZ_CONTROL_ROWS if r["id"] == "viz_dense")
    opts.toggle(dense)
    assert opts.viz_dense and opts.viz_occ and opts.viz_ids and opts.viz_frustums
    opts.toggle(dense)
    assert not opts.viz_dense and not opts.viz_occ


def test_ui_keys_clicks() -> None:
    ui = VizUI()
    assert ui.handle_key(ord("d"))
    assert ui.nerd_tab == "drive" and ui.show_nerd
    assert ui.handle_key(ord("g"))
    assert ui.nerd_tab == "viz"
    assert ui.handle_key(ord("]"))
    assert ui.nerd_tab == "keys"
    ui.show_drive_tab()
    n = len(CONTROL_ROWS)
    ui.handle_key(ord("j"))
    assert ui.debug_sel == 1 % n
    allow = row_at(0)
    ui.debug_sel = 0
    ui.handle_key(ord(" "))
    assert ui.debug.allow_preview is True

    st = {
        "engaged": True,
        "loop_hz": 12.0,
        "path_conf": 0.9,
        "path_width": 2.0,
        "path_debug_preview": False,
        "gvd_show_path": True,
        "show_agent_ghosts": True,
        "ego": {"speed_mps": 14.0, "brake": 0.0, "steer_deg": 0.0, "throttle": 0.1},
        "planner": {"cipv_id": 1, "aeb": "off", "target_v": 11.0, "ttc_lead": 2.4, "corridor_width": 2.0},
        "path_ego": [{"x": 0.0, "y": float(i)} for i in range(0, 36)],
        "tracks": [{"id": 1, "class": "vehicle", "x": 0.1, "y": 16, "speed_mps": 12, "yaw": 1.57}],
        "lanes_ext": [],
        "debug": ui.debug.as_dict(),
    }
    ui.nerd_tab = "live"
    ui.show_nerd = True
    ui.layers = set()
    frame = render_stage(st, ui=ui)
    assert frame.shape == (STAGE_H, STAGE_W + ui.nerd_width, 3)
    drive = next(h for h in ui.nerd_hits if h.get("kind") == "tab" and h.get("id") == "drive")
    x0, y0, x1, y1 = drive["rect"]
    assert ui.handle_click(STAGE_W + (x0 + x1) // 2, (y0 + y1) // 2, stage_w=STAGE_W)
    assert ui.nerd_tab == "drive"
    frame = render_stage(st, ui=ui)
    assert any(h.get("kind") == "row" for h in ui.nerd_hits)

    viz = next(h for h in ui.nerd_hits if h.get("kind") == "tab" and h.get("id") == "viz")
    x0, y0, x1, y1 = viz["rect"]
    assert ui.handle_click(STAGE_W + (x0 + x1) // 2, (y0 + y1) // 2, stage_w=STAGE_W)
    assert ui.nerd_tab == "viz"
    render_stage(st, ui=ui)
    row0 = next(h for h in ui.nerd_hits if h.get("kind") == "row" and h.get("i") == 0 and h.get("part") == "value")
    x0, y0, x1, y1 = row0["rect"]
    before = ui.debug.viz_dense
    ui.handle_click(STAGE_W + (x0 + x1) // 2, (y0 + y1) // 2, stage_w=STAGE_W)
    assert ui.debug.viz_dense is (not before)


def test_overlay_pixels() -> None:
    import numpy as np

    st = {
        "engaged": True,
        "loop_hz": 12.0,
        "path_conf": 0.9,
        "path_width": 2.0,
        "path_debug_preview": False,
        "ego": {"speed_mps": 14.0, "brake": 0.0},
        "planner": {"cipv_id": 1, "aeb": "off", "target_v": 11.0, "ttc_lead": 2.4},
        "path_ego": [{"x": 0.0, "y": float(i)} for i in range(0, 36)],
        "tracks": [{"id": 1, "class": "vehicle", "x": 0.1, "y": 16, "speed_mps": 12, "yaw": 1.57}],
        "lanes_bev": [[{"x": -1.8, "y": float(y)} for y in range(2, 30, 4)]],
        "cam_health": {"main": "ok", "narrow": "missing"},
    }
    ui = VizUI()
    ui.show_nerd = False
    ui.debug.viz_hud = False
    a = render_stage(st, ui=ui)
    ui.debug.viz_occ = True
    b = render_stage(st, ui=ui)
    assert not np.array_equal(a, b), "occupancy overlay should change pixels"
    ui.debug.viz_occ = False
    ui.debug.viz_frustums = True
    c = render_stage(st, ui=ui)
    assert not np.array_equal(a, c), "FOV wedges should change pixels"
    ui.debug.viz_frustums = False
    ui.debug.viz_cost = True
    d = render_stage(st, ui=ui)
    assert not np.array_equal(a, d)
    ui.set_layer(1)
    assert ui.debug.viz_occ is True
    ui.set_layer(0)
    clean = render_stage(st, ui=ui)
    assert clean.shape[1] == STAGE_W


def test_no_chrome() -> None:
    files = [
        ROOT / "python" / "runtime" / "debug_opts.py",
        ROOT / "python" / "viz" / "debug_draw.py",
        ROOT / "python" / "viz" / "nerd.py",
        ROOT / "python" / "viz" / "stage.py",
    ]
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        skip = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                body = getattr(node, "body", None) or []
                if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                    skip.add(id(body[0].value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
                assert not CHROME_RE.search(node.value), f"{f.name}: {node.value!r}"


def main() -> None:
    test_apply_perception()
    test_apply_command()
    test_toggle_nudge()
    test_ui_keys_clicks()
    test_overlay_pixels()
    test_no_chrome()
    assert any(r["id"] == "force_engage" for r in DEBUG_ROWS)
    assert viz_row_at(0)["id"] == "viz_dense"
    print("test_debug_opts: OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"test_debug_opts: FAIL — {e}")
        sys.exit(1)
