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
    assert off.view == VIEW_REALITY and not off.classic_enabled and "disabled" in off.note
    cfg_only = {"viz": {"main_view": "classic"}}
    assert resolve_view(cfg_only).view == VIEW_REALITY
    # Enabled by CLI or config -> selectable.
    assert resolve_view(cams, cli_view="classic", cli_enable_classic=True).view == VIEW_CLASSIC
    on_cfg = {"viz": {"main_view": "classic", "classic_view_enabled": True}}
    assert resolve_view(on_cfg).view == VIEW_CLASSIC
    # Truthy-but-not-True config does not enable it.
    assert resolve_view({"viz": {"main_view": "classic", "classic_view_enabled": "yes"}}).view == VIEW_REALITY
    # CLI beats config; enabled does not force classic.
    assert resolve_view(on_cfg, cli_view="reality").view == VIEW_REALITY
    assert resolve_view(cams, cli_enable_classic=True).view == VIEW_REALITY
    # Garbage never raises.
    bad = resolve_view({"viz": {"main_view": "fancy"}})
    assert bad.view == VIEW_REALITY and "unknown" in bad.note
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
