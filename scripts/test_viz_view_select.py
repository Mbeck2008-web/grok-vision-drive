#!/usr/bin/env python3
"""Offline checks: Reality View is the selected main view; the classic stage
view is a disabled option (config + CLI toggles). No display required."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
    assert "render_main_view(\n                    view_choice," in rv
    assert "stage_frame = render_stage(" not in rv

    print("test_viz_view_select: ok")


if __name__ == "__main__":
    main()
