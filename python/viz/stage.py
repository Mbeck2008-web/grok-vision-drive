"""Tesla-like cabin BEV stage (OpenCV). Title: GVD. No FSD / Tesla marks."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from python.viz.forecast import cone_widths, predict_modes
from python.viz.nerd import render_panel

VOID = (10, 8, 7)  # #07080a BGR
PAPER = (225, 230, 232)
ICE = (212, 196, 158)  # #9ec4d4
CORRIDOR = (199, 167, 90)
GHOST = (199, 192, 185)
PED = (138, 176, 215)
BIKE = (90, 184, 224)
BLOCKED = (92, 92, 196)

MAX_AGENTS = 32
MAX_FORECAST = 16
PPM = 8.0  # pixels per meter
STAGE_W, STAGE_H = 960, 720


def _world_to_px(x: float, y: float, w: int, h: int, ego_y_shift: float = 80) -> tuple[int, int]:
    # BEV: +Y forward up the image, +X right
    cx, cy = w // 2, h - int(ego_y_shift)
    px = int(cx + x * PPM)
    py = int(cy - y * PPM)
    return px, py


def render_stage(state: dict[str, Any], debug_layers: set[int] | None = None, show_nerd: bool = True) -> np.ndarray:
    t0 = time.perf_counter()
    debug_layers = debug_layers or set()
    clean = 0 in debug_layers
    img = np.full((STAGE_H, STAGE_W, 3), VOID, dtype=np.uint8)

    # dim occupancy slab
    cv2.rectangle(img, (80, 40), (STAGE_W - 80, STAGE_H - 40), (18, 16, 15), -1)

    loop_hz = float(state.get("loop_hz") or 0.0)
    drop_heavy = loop_hz > 0 and loop_hz < 8.0

    # ego path ribbon
    path = state.get("path_ego") or []
    if len(path) >= 2:
        pts = []
        for p in path[:41]:
            pts.append(_world_to_px(float(p.get("x", 0)), float(p.get("y", 0)), STAGE_W, STAGE_H))
        conf = float(state.get("path_conf") or 0.5)
        overlay = img.copy()
        for i in range(len(pts) - 1):
            fade = 1.0 - (i / max(1, len(pts) - 1)) * 0.75
            thick = max(2, int(14 * conf * fade))
            cv2.line(overlay, pts[i], pts[i + 1], CORRIDOR, thick, cv2.LINE_AA)
        alpha = 0.35 * conf
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], ICE, 1, cv2.LINE_AA)

    # ego shell
    ex, ey = _world_to_px(0, 0, STAGE_W, STAGE_H)
    cv2.rectangle(img, (ex - 10, ey - 18), (ex + 10, ey + 10), (60, 60, 60), -1)

    tracks = list(state.get("tracks") or state.get("agents") or [])[:MAX_AGENTS]
    n_forecast = 0
    for tr in tracks:
        cls = str(tr.get("class", "vehicle"))
        col = GHOST if cls == "vehicle" else PED if cls == "pedestrian" else BIKE
        x, y = float(tr.get("x", 0)), float(tr.get("y", 0))
        px, py = _world_to_px(x, y, STAGE_W, STAGE_H)
        cv2.rectangle(img, (px - 8, py - 12), (px + 8, py + 8), col, -1)
        cv2.rectangle(img, (px - 8, py - 12), (px + 8, py + 8), PAPER, 1)

        if clean or drop_heavy:
            continue
        if n_forecast >= MAX_FORECAST:
            continue
        modes = predict_modes(tr)
        n_forecast += 1
        if modes:
            m0 = modes[0]["points"]
            widths = cone_widths(len(m0))
            for i in range(len(m0) - 1):
                a = _world_to_px(m0[i][0], m0[i][1], STAGE_W, STAGE_H)
                b = _world_to_px(m0[i + 1][0], m0[i + 1][1], STAGE_W, STAGE_H)
                cv2.line(img, a, b, ICE, 2, cv2.LINE_AA)
                # thin uncertainty sausage
                rad = max(1, int(widths[i] * PPM * 0.25))
                cv2.circle(img, b, rad, (40, 40, 40), 1, cv2.LINE_AA)
            for m in modes[1:]:
                pts = [_world_to_px(p[0], p[1], STAGE_W, STAGE_H) for p in m["points"]]
                for i in range(len(pts) - 1):
                    cv2.line(img, pts[i], pts[i + 1], (90, 90, 90), 1, cv2.LINE_AA)

    if 1 in debug_layers and state.get("occupancy") is not None:
        cv2.putText(img, "occ debug", (20, STAGE_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLOCKED, 1)

    if not drop_heavy and 2 not in debug_layers:
        # PIP placeholder
        pip = np.full((180, 320, 3), (30, 30, 30), dtype=np.uint8)
        cv2.putText(pip, "cam_main", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PAPER, 1)
        img[12:192, 12:332] = pip
        cv2.rectangle(img, (12, 12), (332, 192), ICE, 1)

    cv2.putText(img, "GVD", (STAGE_W - 90, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ICE, 1, cv2.LINE_AA)
    cv2.putText(img, "VISION", (STAGE_W - 110, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

    viz_ms = (time.perf_counter() - t0) * 1000.0
    state["viz_ms"] = viz_ms

    if show_nerd and not clean:
        panel = render_panel(state, h=STAGE_H, w=420)
        out = np.concatenate([img, panel], axis=1)
    else:
        out = img
    return out


def smoke(window: bool = False, frames: int = 3) -> Path:
    """Offline smoke with fake tracks; writes PNG under docs/."""
    from pathlib import Path

    from python.runtime.state_io import default_state, write_state

    st = default_state(
        engaged=True,
        loop_hz=12.0,
        camera_hz=10.0,
        path_conf=0.85,
        path_debug_preview=False,
        tracks_n=3,
        objects_n=3,
        tracks=[
            {"id": 1, "class": "vehicle", "x": -3.5, "y": 18, "speed_mps": 12, "yaw": 1.55, "yaw_rate": 0.0},
            {"id": 2, "class": "vehicle", "x": 3.0, "y": 25, "speed_mps": 8, "yaw": 1.6, "yaw_rate": 0.05},
            {"id": 3, "class": "pedestrian", "x": 6.0, "y": 12, "speed_mps": 1.2, "yaw": 3.1, "yaw_rate": 0.0},
        ],
        missing_state_keys=["live cameras", "real planner path", "occupancy grid"],
    )
    # richer path
    st["path_ego"] = [{"x": 0.0, "y": float(i), "z": 0.0} for i in range(0, 36)]
    write_state(st)
    frame = render_stage(st)
    out = Path("docs/gvd_viz_smoke.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    if window:
        cv2.imshow("GVD", frame)
        for _ in range(frames):
            if cv2.waitKey(50) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
    return out
