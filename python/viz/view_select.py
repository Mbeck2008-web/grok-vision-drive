"""Main-view selection for the GVD VISION window.

GVD 2.0 (spec /workspace/gvd2/spec.md, Phase 0 decision): Reality View is the
selected main view. The existing stage renderer (``render_stage``, the chase
ribbon/HUD view) stays only as a disabled option called ``classic``.

* ``reality`` (default): Reality View. Until the splat renderer exists this is
  a placeholder frame that says so. It never reads sim GT and costs one
  ``np.full`` plus a few ``putText`` calls, so it cannot delay control.
* ``classic``: the existing stage view. Disabled unless
  ``viz.classic_view_enabled: true`` in ``config/cameras.yaml`` or
  ``--enable-classic-view`` on the CLI. Asking for it while disabled falls
  back to Reality View with a note; it is never an error.

Precedence: CLI ``--view`` > ``viz.main_view`` > ``reality``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

VIEW_REALITY = "reality"
VIEW_CLASSIC = "classic"
VIEWS: tuple[str, ...] = (VIEW_REALITY, VIEW_CLASSIC)
DEFAULT_VIEW = VIEW_REALITY

PLACEHOLDER_BG = (24, 20, 18)
PLACEHOLDER_FG = (235, 220, 200)
PLACEHOLDER_DIM = (150, 140, 130)
# (text, x, y, scale, color, thickness). y < 0 counts up from the bottom edge;
# text None is the drive word in its own color.
PLACEHOLDER_LINES: tuple[tuple[str | None, int, int, float, tuple[int, int, int] | None, int], ...] = (
    ("GVD VISION - REALITY VIEW", 24, 44, 0.9, PLACEHOLDER_FG, 2),
    ("placeholder: renderer not built yet", 24, 80, 0.6, PLACEHOLDER_DIM, 1),
    (None, 24, -28, 1.0, None, 2),
)


@dataclass(frozen=True)
class ViewChoice:
    view: str
    classic_enabled: bool
    note: str = ""


def _viz_cfg(cameras: dict[str, Any] | None) -> dict[str, Any]:
    viz = (cameras or {}).get("viz")
    return viz if isinstance(viz, dict) else {}


def _norm(name: Any) -> str | None:
    if name is None:
        return None
    s = str(name).strip().lower()
    return s or None


def resolve_view(
    cameras: dict[str, Any] | None,
    cli_view: str | None = None,
    cli_enable_classic: bool = False,
) -> ViewChoice:
    """Pick the main view from config + CLI. Never raises."""
    viz = _viz_cfg(cameras)
    enabled = bool(cli_enable_classic) or viz.get("classic_view_enabled") is True
    want = _norm(cli_view) or _norm(viz.get("main_view")) or DEFAULT_VIEW
    if want not in VIEWS:
        return ViewChoice(DEFAULT_VIEW, enabled, f"unknown view {want!r}; using {DEFAULT_VIEW}")
    if want == VIEW_CLASSIC and not enabled:
        return ViewChoice(
            DEFAULT_VIEW,
            enabled,
            "classic view is disabled (viz.classic_view_enabled / --enable-classic-view); using reality",
        )
    return ViewChoice(want, enabled)


def render_reality_placeholder(
    state: dict[str, Any] | None,
    width: int,
    height: int,
) -> np.ndarray:
    """Reality View stand-in until the splat renderer lands.

    Shows only the drive word (OFF / ON / HOLD / ...) so the driver still sees
    engage state, plus a plain label that this is a placeholder.
    """
    import cv2

    from python.viz.debug_draw import cabin_drive_word

    w, h = max(int(width), 1), max(int(height), 1)
    img = np.full((h, w, 3), PLACEHOLDER_BG, dtype=np.uint8)
    word, color = cabin_drive_word(dict(state or {}))
    for text, x, y, scale, fg, thick in PLACEHOLDER_LINES:
        if text is None:
            text, fg = word, tuple(int(c) for c in color)
        org = (x, y if y >= 0 else h + y)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)
    # Cabin top bar: docs/SPEC_no_conf_gate.md requires CONF GATE OFF when the flag is on.
    from python.viz.stage import _draw_conf_gate_tag

    _draw_conf_gate_tag(img, dict(state or {}), x=max(260, w - 280))
    return img


def render_main_view(
    choice: ViewChoice,
    state: dict[str, Any],
    render_classic: Callable[..., np.ndarray],
    *,
    width: int,
    height: int,
    **classic_kwargs: Any,
) -> np.ndarray:
    """Draw the selected view. ``state['viz_ms']`` / ``state['viz_view']`` are set either way."""
    import time

    state["viz_view"] = choice.view
    if choice.view == VIEW_CLASSIC:
        return render_classic(state, **classic_kwargs)
    t0 = time.perf_counter()
    img = render_reality_placeholder(state, width, height)
    state["viz_ms"] = (time.perf_counter() - t0) * 1000.0
    ui = classic_kwargs.get("ui")
    # Keep the side panel (record / AUTO / keys) so mouse hit-testing and the
    # driver's controls match the classic view's window width.
    if ui is not None and getattr(ui, "show_nerd", False) and 0 not in getattr(ui, "layers", set()):
        from python.viz.nerd import render_panel

        panel = render_panel(
            state,
            h=img.shape[0],
            w=ui.nerd_width,
            show_help=ui.show_help,
            ui=ui,
            cam_frames=classic_kwargs.get("cam_frames"),
        )
        return np.concatenate([img, panel], axis=1)
    return img
