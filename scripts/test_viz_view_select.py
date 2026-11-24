#!/usr/bin/env python3
"""Offline checks: Reality View is the selected main view; the classic stage
view is a disabled option (config + CLI toggles). No display required."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))



def check_conf_gate_tag_on_default_path() -> None:
    """Default render path (render_main_view), not render_stage / render_reality_placeholder alone."""
    import cv2
    import numpy as np
    import yaml

    from python.runtime.conf_gate import HUD_TAG, hud_tag
    from python.viz.stage import STAGE_H, STAGE_W, VizUI
    from python.viz.view_select import render_main_view, resolve_view

    cams = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))
    choice = resolve_view(cams)
    assert choice.view == "reality"
    ui = VizUI()
    ui.show_nerd = False

    def boom_classic(state, **kw):
        raise AssertionError("default path must not call the classic renderer")

    st_off = {"engaged": False, "no_conf_gate": False}
    st_on = {"engaged": False, "no_conf_gate": True}
    got_off = render_main_view(choice, st_off, boom_classic, width=STAGE_W, height=STAGE_H, ui=ui)
    got_on = render_main_view(choice, st_on, boom_classic, width=STAGE_W, height=STAGE_H, ui=ui)
    assert hud_tag(True) == HUD_TAG and hud_tag(False) == ""
    assert not np.array_equal(got_off, got_on), "flag on vs off must differ on the default path"
    tagged = got_off.copy()
    x = max(260, int(STAGE_W) - 280)
    cv2.putText(tagged, HUD_TAG, (x, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 180, 220), 1, cv2.LINE_AA)
    assert int(np.count_nonzero(np.any(tagged != got_off, axis=2))) > 0, "tag must occupy pixels"
    assert np.array_equal(got_on, tagged), "CONF GATE OFF pixels must be present on the default path"


def check_classic_passthrough_forwards_all() -> None:
    """Classic dispatch must forward dets, main_frame and cam_frames (not just shape)."""
    import numpy as np
    import yaml

    from python.viz.stage import STAGE_H, STAGE_W, VizUI, render_stage
    from python.viz.view_select import render_main_view, resolve_view

    on_cfg = {"viz": {"main_view": "classic", "classic_view_enabled": True}}
    ui = VizUI()
    ui.show_nerd = False
    ui.debug.viz_pip = True
    ui.debug.viz_boxes = True
    ui.debug.viz_cams = True  # cam strip is the only on-stage use of cam_frames
    main_frame = np.full((180, 320, 3), 40, np.uint8)
    main_frame[20:80, 30:120] = (20, 220, 40)
    dets = [{"xyxy": [30, 20, 120, 80], "cls": "vehicle", "conf": 0.9}]
    cam_frames = {"main": np.full((40, 60, 3), (10, 10, 200), np.uint8)}
    st = {"engaged": False}
    calls: list[dict] = []

    def spy(state, **kw):
        calls.append(kw)
        return render_stage(state, **kw)

    got = render_main_view(
        resolve_view(on_cfg),
        st,
        spy,
        width=STAGE_W,
        height=STAGE_H,
        ui=ui,
        main_frame=main_frame,
        cam_frames=cam_frames,
        dets=dets,
    )
    assert len(calls) == 1
    forwarded = calls[0]
    assert forwarded["ui"] is ui
    assert forwarded["main_frame"] is main_frame
    assert forwarded["cam_frames"] is cam_frames
    assert forwarded["dets"] is dets
    assert set(forwarded) == {"ui", "main_frame", "cam_frames", "dets"}
    direct = render_stage(st, ui=ui, main_frame=main_frame, cam_frames=cam_frames, dets=dets)
    assert np.array_equal(got, direct)
    dropped_dets = render_stage(st, ui=ui, main_frame=main_frame, cam_frames=cam_frames, dets=None)
    dropped_main = render_stage(st, ui=ui, main_frame=None, cam_frames=cam_frames, dets=dets)
    dropped_cams = render_stage(st, ui=ui, main_frame=main_frame, cam_frames=None, dets=dets)
    assert not np.array_equal(direct, dropped_dets)
    assert not np.array_equal(direct, dropped_main)
    assert not np.array_equal(direct, dropped_cams)


def check_draw_vision_frame_live_path() -> None:
    """Behavioural pin of the live call helper: width, height, dets, hit-testing."""
    import types

    import numpy as np

    from python.run_vision import draw_vision_frame
    from python.viz.nerd import hit_test
    from python.viz.stage import STAGE_H, STAGE_W, VizUI, render_stage
    from python.viz.view_select import resolve_view

    on_cfg = {"viz": {"main_view": "classic", "classic_view_enabled": True}}
    choice = resolve_view(on_cfg)
    ui = VizUI()
    ui.debug.viz_pip = True
    ui.debug.viz_boxes = True
    main = np.full((180, 320, 3), 50, np.uint8)
    main[10:70, 20:140] = (30, 200, 30)
    dets = [{"xyxy": [20, 10, 140, 70], "cls": "car", "conf": 0.95}]
    cam_frames = {"main": np.full((48, 64, 3), (15, 15, 180), np.uint8)}
    bundle = types.SimpleNamespace(frames=cam_frames)
    pout = types.SimpleNamespace(dets=dets)
    paint = {"engaged": False}

    got = draw_vision_frame(choice, paint, ui, main, bundle, pout)
    ui_c0 = VizUI()
    ui_c0.debug.viz_pip = True
    ui_c0.debug.viz_boxes = True
    classic = render_stage(paint, ui=ui_c0, main_frame=main, cam_frames=cam_frames, dets=dets)
    assert got.shape == (STAGE_H, STAGE_W + ui.nerd_width, 3), got.shape
    assert got.shape == classic.shape
    # The nerd panel can paint viz_ms, which is wall time, so pin the stage pane.
    assert np.array_equal(got[:, :STAGE_W], classic[:, :STAGE_W])

    # Mutants that pass width=STAGE_W//2 or height=STAGE_H//2 change this shape
    # on the default (reality) path, which is the one that consumes width/height.
    reality = resolve_view({"viz": {"main_view": "reality"}})
    ui_r = VizUI()
    got_r = draw_vision_frame(reality, paint, ui_r, main, bundle, pout)
    assert got_r.shape == (STAGE_H, STAGE_W + ui_r.nerd_width, 3), got_r.shape
    # Classic path above already pins STAGE_W x STAGE_H via render_stage defaults.
    assert got.shape[0] == STAGE_H and got.shape[1] == STAGE_W + ui.nerd_width

    ui_live_nod = VizUI()
    ui_live_nod.debug.viz_pip = True
    ui_live_nod.debug.viz_boxes = True
    no_dets = draw_vision_frame(choice, paint, ui_live_nod, main, bundle, types.SimpleNamespace(dets=None))
    ui_nod = VizUI()
    ui_nod.debug.viz_pip = True
    ui_nod.debug.viz_boxes = True
    classic_nod = render_stage(paint, ui=ui_nod, main_frame=main, cam_frames=cam_frames, dets=None)
    assert np.array_equal(no_dets[:, :STAGE_W], classic_nod[:, :STAGE_W])
    assert not np.array_equal(got[:, :STAGE_W], no_dets[:, :STAGE_W]), "live path must forward pout.dets, not None"

    # Hit-testing: a click in the nerd panel (x >= STAGE_W) hits the same control as classic.
    assert ui.nerd_hits, "panel hits must be recorded so mouse hit-testing can match classic"
    tab = next(h for h in ui.nerd_hits if h.get("kind") == "tab")
    rect = tab["rect"]
    cx = STAGE_W + (int(rect[0]) + int(rect[2])) // 2
    cy = (int(rect[1]) + int(rect[3])) // 2
    assert hit_test(ui.nerd_hits, cx - STAGE_W, cy) is not None
    ui_c = VizUI()
    ui_c.debug.viz_pip = True
    ui_c.debug.viz_boxes = True
    classic2 = render_stage(paint, ui=ui_c, main_frame=main, cam_frames=cam_frames, dets=dets)
    assert classic2.shape[1] == STAGE_W + ui_c.nerd_width
    h_live = hit_test(ui.nerd_hits, cx - STAGE_W, cy)
    h_classic = hit_test(ui_c.nerd_hits, cx - STAGE_W, cy)
    assert h_live is not None and h_classic is not None
    assert h_live.get("kind") == h_classic.get("kind") and h_live.get("id") == h_classic.get("id")
    # A click left of STAGE_W is on the stage, not the panel, same as classic.
    assert ui.handle_click(STAGE_W - 10, cy, stage_w=STAGE_W) is False
    assert ui.handle_click(cx, cy, stage_w=STAGE_W) is True


def check_draw_vision_frame_forwards_cam_frames() -> None:
    """Classic cams tab keeps its tiles; cam_frames=None in the helper must fail."""
    import types

    import numpy as np

    from python.run_vision import draw_vision_frame
    from python.viz.stage import STAGE_W, VizUI, render_stage
    from python.viz.view_select import resolve_view

    choice = resolve_view({"viz": {"main_view": "classic", "classic_view_enabled": True}})
    main = np.full((180, 320, 3), 50, np.uint8)
    cam_frames = {"main": np.full((48, 64, 3), (15, 15, 180), np.uint8)}
    bundle = types.SimpleNamespace(frames=cam_frames)
    pout = types.SimpleNamespace(dets=None)
    paint = {"engaged": False}

    def _cams_ui() -> VizUI:
        ui = VizUI()
        ui.nerd_tab = "cams"
        ui.show_nerd = True
        ui.show_help = False
        return ui

    got = draw_vision_frame(choice, paint, _cams_ui(), main, bundle, pout)
    classic = render_stage(paint, ui=_cams_ui(), main_frame=main, cam_frames=cam_frames, dets=None)
    assert np.array_equal(got[:, :STAGE_W], classic[:, :STAGE_W]), (
        "draw_vision_frame must match render_stage when the classic cams tab is on"
    )
    dropped = render_stage(paint, ui=_cams_ui(), main_frame=main, cam_frames=None, dets=None)
    assert not np.array_equal(got[:, :STAGE_W], dropped[:, :STAGE_W]), (
        "classic cams tab must keep its tiles; draw_vision_frame must forward cam_frames"
    )


def check_draw_vision_frame_loop_call_args() -> None:
    """The live loop must call draw_vision_frame(view_choice, paint, ui, main, bundle, pout)."""
    import ast

    src = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    found: list[list[str]] = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != "stage_frame":
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if not isinstance(func, ast.Name) or func.id != "draw_vision_frame":
            continue
        names: list[str] = []
        for arg in call.args:
            assert isinstance(arg, ast.Name), ast.dump(arg)
            names.append(arg.id)
        assert call.keywords == [], "live loop must pass positional args only"
        found.append(names)
    assert found == [["view_choice", "paint", "ui", "main", "bundle", "pout"]], found


def check_docs_view_flags() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "The left part is the stage (the 3D view of what GVD sees)" not in readme
    # CLI table (the supervisor row) must name the new flags.
    table_row = next(line for line in readme.splitlines() if line.startswith("| `py -3 python\\run_vision.py`"))
    for flag in ("--view", "--enable-classic-view", "--enable-viz-cpp"):
        assert flag in table_row, flag
    schema = (ROOT / "docs" / "gvd_state_schema.md").read_text(encoding="utf-8")
    assert "viz_view" in schema
    cams = (ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8")
    assert "cpp_viz_enabled" in cams


def main() -> None:
    import numpy as np
    import yaml

    from python.run_vision import build_parser, resolve_main_view
    from python.viz.stage import STAGE_H, STAGE_W, VizUI, render_stage
    from python.viz.view_select import (
        DEFAULT_VIEW,
        VIEW_CLASSIC,
        VIEW_REALITY,
        render_main_view,
        render_reality_placeholder,
        resolve_view,
    )

    # Default and shipped config: reality selected, classic disabled.
    assert DEFAULT_VIEW == VIEW_REALITY
    cams = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))
    assert cams["viz"]["main_view"] == "reality"
    assert cams["viz"]["classic_view_enabled"] is False
    assert resolve_view(cams).view == VIEW_REALITY
    assert resolve_view(None).view == VIEW_REALITY
    assert resolve_view({}).view == VIEW_REALITY

    # Classic is a disabled option: asking for it without enabling falls back.
    off = resolve_view(cams, cli_view="classic")
    assert off.view == VIEW_REALITY and off.classic_enabled is False
    assert off.note == "classic view is disabled (viz.classic_view_enabled / --enable-classic-view); using reality"
    cfg_only = {"viz": {"main_view": "classic"}}
    assert resolve_view(cfg_only).view == VIEW_REALITY
    # Enabled by CLI or config -> selectable.
    on_cli = resolve_view(cams, cli_view="classic", cli_enable_classic=True)
    assert on_cli.view == VIEW_CLASSIC and on_cli.classic_enabled is True and on_cli.note == ""
    assert resolve_view(cams).classic_enabled is False
    on_cfg = {"viz": {"main_view": "classic", "classic_view_enabled": True}}
    assert resolve_view(on_cfg).view == VIEW_CLASSIC
    # Truthy-but-not-True config does not enable it.
    assert resolve_view({"viz": {"main_view": "classic", "classic_view_enabled": "yes"}}).view == VIEW_REALITY
    # CLI beats config; enabled does not force classic.
    assert resolve_view(on_cfg, cli_view="reality").view == VIEW_REALITY
    assert resolve_view(cams, cli_enable_classic=True).view == VIEW_REALITY
    # Garbage never raises.
    bad = resolve_view({"viz": {"main_view": "fancy"}})
    assert bad.view == VIEW_REALITY and bad.note == "unknown view 'fancy'; using reality"
    assert bad.classic_enabled is False
    assert resolve_view({"viz": {"main_view": "fancy"}}, cli_enable_classic=True).classic_enabled is True
    # Names are trimmed and case-folded; nothing else is rewritten.
    assert resolve_view(on_cfg, cli_view="  Classic ").view == VIEW_CLASSIC
    assert resolve_view(on_cfg, cli_view="class-ic").view == VIEW_REALITY
    assert resolve_view({"viz": "nope"}).view == VIEW_REALITY

    # CLI flags parse and reach the resolver.
    ap = build_parser()
    a = ap.parse_args([])
    assert a.view is None and a.enable_classic_view is False
    assert resolve_main_view(a).view == VIEW_REALITY
    a = ap.parse_args(["--view", "classic"])
    assert resolve_main_view(a).view == VIEW_REALITY
    a = ap.parse_args(["--view", "classic", "--enable-classic-view"])
    assert resolve_main_view(a).view == VIEW_CLASSIC
    import contextlib
    import io

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            ap.parse_args(["--view", "bogus"])
        raise AssertionError("--view bogus should be rejected")
    except SystemExit:
        pass

    # Placeholder: right shape, labelled, keeps the drive word, no GT read.
    img = render_reality_placeholder({"engaged": False}, STAGE_W, STAGE_H)
    assert img.shape == (STAGE_H, STAGE_W, 3) and img.dtype == np.uint8
    assert int(img.max()) > int(img.min())
    eng = render_reality_placeholder({"engaged": True}, STAGE_W, STAGE_H)
    assert not np.array_equal(img, eng), "drive word must change with engage state"
    assert render_reality_placeholder(None, 0, 0).shape == (1, 1, 3)
    # Pixel-exact against an independent reference drawn from the spec'd
    # labels, so text, placement, scale, color, thickness and AA are pinned.
    import cv2

    from python.viz.debug_draw import cabin_drive_word

    def reference(state, w, h):
        ref = np.full((h, w, 3), (24, 20, 18), np.uint8)
        f = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(ref, "GVD VISION - REALITY VIEW", (24, 44), f, 0.9, (235, 220, 200), 2, cv2.LINE_AA)
        cv2.putText(ref, "placeholder: renderer not built yet", (24, 80), f, 0.6, (150, 140, 130), 1, cv2.LINE_AA)
        wd, col = cabin_drive_word(dict(state))
        cv2.putText(ref, wd, (24, h - 28), f, 1.0, tuple(int(c) for c in col), 2, cv2.LINE_AA)
        return ref

    for stt in ({"engaged": False}, {"engaged": True}):
        for w, h in ((STAGE_W, STAGE_H), (640, 360)):
            got = render_reality_placeholder(stt, w, h)
            assert np.array_equal(got, reference(stt, w, h)), (stt, w, h)
    src = (ROOT / "python" / "viz" / "view_select.py").read_text(encoding="utf-8").lower()
    for banned in ("tesla", "fsd", "autopilot", "gt_", "ground_truth", "path_world"):
        assert banned not in src, banned

    # Dispatch: reality never calls the classic renderer; classic does.
    calls: list[dict] = []

    def fake_classic(state, **kw):
        calls.append(kw)
        return np.zeros((2, 2, 3), np.uint8)

    ui = VizUI()
    st: dict = {"engaged": False}
    out = render_main_view(resolve_view(cams), st, fake_classic, width=STAGE_W, height=STAGE_H, ui=ui)
    assert calls == [] and st["viz_view"] == VIEW_REALITY and st["viz_ms"] >= 0.0
    # The placeholder sees the live state (drive word), not an empty dict.
    st_on: dict = {"engaged": True}
    ui_off = VizUI()
    ui_off.show_nerd = False
    got_on = render_main_view(resolve_view(cams), st_on, fake_classic, width=STAGE_W, height=STAGE_H, ui=ui_off)
    assert np.array_equal(got_on, reference({"engaged": True}, STAGE_W, STAGE_H))
    # viz_ms is the placeholder's own wall time, in ms.
    import time as _time

    ticks = iter([10.0, 10.5])
    real_pc = _time.perf_counter
    _time.perf_counter = lambda: next(ticks)
    try:
        st_t: dict = {}
        render_main_view(resolve_view(cams), st_t, fake_classic, width=8, height=8, ui=None)
    finally:
        _time.perf_counter = real_pc
    assert st_t["viz_ms"] == 500.0, st_t["viz_ms"]
    # Panel plumbing: exact args reach nerd.render_panel; layer 0 hides it.
    import python.viz.nerd as nerd_mod

    seen: list[dict] = []
    real_panel = nerd_mod.render_panel

    def fake_panel(state, **kw):
        seen.append({"state": state, **kw})
        return np.zeros((kw["h"], kw["w"], 3), np.uint8)

    class BareUI:  # no show_nerd / layers attrs
        nerd_width = 7
        show_help = "H"

    class NerdUI(BareUI):
        show_nerd = True

    nerd_mod.render_panel = fake_panel
    try:
        assert render_main_view(resolve_view(cams), {}, fake_classic, width=8, height=6, ui=BareUI()).shape == (6, 8, 3)
        assert seen == []
        nu = NerdUI()
        cf = {"front": np.zeros((1, 1, 3), np.uint8)}
        st_p: dict = {"engaged": False}
        out_p = render_main_view(resolve_view(cams), st_p, fake_classic, width=8, height=6, ui=nu, cam_frames=cf)
        assert out_p.shape == (6, 15, 3) and len(seen) == 1
        assert seen[0] == {"state": st_p, "h": 6, "w": 7, "show_help": "H", "ui": nu, "cam_frames": cf}
        nu.layers = {1, 2}
        assert render_main_view(resolve_view(cams), {}, fake_classic, width=8, height=6, ui=nu).shape == (6, 15, 3)
        nu.layers = {0}
        assert render_main_view(resolve_view(cams), {}, fake_classic, width=8, height=6, ui=nu).shape == (6, 8, 3)
    finally:
        nerd_mod.render_panel = real_panel
    # Side panel kept, so window width / mouse hit-test match the classic view.
    assert out.shape == (STAGE_H, STAGE_W + ui.nerd_width, 3), out.shape
    ui.show_nerd = False
    assert render_main_view(resolve_view(cams), {}, fake_classic, width=STAGE_W, height=STAGE_H, ui=ui).shape == (
        STAGE_H, STAGE_W, 3,
    )
    st2: dict = {}
    render_main_view(resolve_view(on_cfg), st2, fake_classic, width=STAGE_W, height=STAGE_H, ui=ui)
    assert len(calls) == 1 and calls[0]["ui"] is ui and st2["viz_view"] == VIEW_CLASSIC

    # Classic path is the unchanged stage renderer.
    ui2 = VizUI()
    real = render_main_view(resolve_view(on_cfg), {}, render_stage, width=STAGE_W, height=STAGE_H, ui=ui2)
    assert real.shape == render_stage({}, ui=VizUI()).shape

    # The live loop draws through the selector, not render_stage directly.
    rv = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    assert "view_choice = resolve_main_view(args)" in rv
    assert "stage_frame = render_stage(" not in rv
    assert "stage_frame = draw_vision_frame(" in rv
    assert "def draw_vision_frame(" in rv

    check_conf_gate_tag_on_default_path()
    check_classic_passthrough_forwards_all()
    check_draw_vision_frame_live_path()
    check_draw_vision_frame_forwards_cam_frames()
    check_draw_vision_frame_loop_call_args()
    check_docs_view_flags()

    print("test_viz_view_select: ok")


if __name__ == "__main__":
    main()
