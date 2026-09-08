"""Tesla-like cabin BEV stage (OpenCV). Title chrome only — no FSD / Tesla marks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from python.viz.forecast import cone_widths, predict_modes
from python.viz.nerd import render_panel

VOID = (10, 8, 7)  # #07080a
PAPER = (225, 230, 232)
ICE = (212, 196, 158)  # #9ec4d4
CORRIDOR = (199, 167, 90)  # #5aa7c7-ish BGR
GHOST = (199, 192, 185)
PED = (138, 176, 215)
BIKE = (90, 184, 224)
BLOCKED = (92, 92, 196)

MAX_AGENTS = 32
MAX_FORECAST = 16
PPM = 10.0
STAGE_W, STAGE_H = 960, 720
PATH_WIDTH_M = 2.0


@dataclass
class VizUI:
    show_nerd: bool = True
    show_help: bool = False
    layers: set[int] = field(default_factory=set)
    top_down: bool = True  # v0 BEV; chase perspective later

    def toggle_nerd(self) -> None:
        self.show_nerd = not self.show_nerd

    def toggle_help(self) -> None:
        self.show_help = not self.show_help

    def set_layer(self, k: int) -> None:
        if k == 0:
            self.layers = {0}
            return
        self.layers.discard(0)
        if k in self.layers:
            self.layers.discard(k)
        else:
            self.layers.add(k)


def _to_px(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    cx, cy = w // 2, h - 90
    return int(cx + x * PPM), int(cy - y * PPM)


def _draw_filled_corridor(img: np.ndarray, path: list[dict], conf: float) -> None:
    if len(path) < 2:
        return
    half = (PATH_WIDTH_M * 0.5) * PPM
    left, right, center = [], [], []
    for i, p in enumerate(path[:41]):
        x, y = float(p.get("x", 0)), float(p.get("y", 0))
        # tangent from neighbors
        if i + 1 < len(path):
            nx, ny = float(path[i + 1].get("x", x)), float(path[i + 1].get("y", y))
        else:
            nx, ny = float(path[i - 1].get("x", x)), float(path[i - 1].get("y", y))
            nx, ny = x - (nx - x), y - (ny - y)
        tx, ty = nx - x, ny - y
        norm = (tx * tx + ty * ty) ** 0.5 + 1e-6
        # perpendicular (right-handed in BEV)
        px, py = ty / norm, -tx / norm
        cx, cy = _to_px(x, y, STAGE_W, STAGE_H)
        left.append([int(cx - px * half), int(cy - py * half)])
        right.append([int(cx + px * half), int(cy + py * half)])
        center.append((cx, cy))

    poly = np.array(left + right[::-1], dtype=np.int32)
    overlay = img.copy()
    # brighter first ~15 m (first ~15 samples at 1 m)
    split = min(15, len(left))
    cv2.fillPoly(overlay, [poly], CORRIDOR)
    if split >= 2:
        poly_near = np.array(left[:split] + right[:split][::-1], dtype=np.int32)
        near = img.copy()
        cv2.fillPoly(near, [poly_near], CORRIDOR)
        cv2.addWeighted(near, 0.20, overlay, 0.80, 0, overlay)
    alpha = 0.35 * max(0.2, conf)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    # darker edges + soft centerline
    for i in range(len(left) - 1):
        fade = 1.0 - (i / max(1, len(left) - 1)) * 0.7
        edge = tuple(int(c * (0.55 + 0.45 * fade)) for c in ICE)
        cv2.line(img, tuple(left[i]), tuple(left[i + 1]), edge, 1, cv2.LINE_AA)
        cv2.line(img, tuple(right[i]), tuple(right[i + 1]), edge, 1, cv2.LINE_AA)
        if i < len(center) - 1:
            cv2.line(img, center[i], center[i + 1], ICE, 1, cv2.LINE_AA)


def _draw_underglow(img: np.ndarray, engaged: bool) -> None:
    if not engaged:
        return
    ex, ey = _to_px(0, 0, STAGE_W, STAGE_H)
    overlay = img.copy()
    for r, a in ((38, 0.25), (24, 0.35), (14, 0.45)):
        layer = img.copy()
        cv2.circle(layer, (ex, ey + 4), r, ICE, -1, cv2.LINE_AA)
        cv2.addWeighted(layer, a, overlay, 1 - a, 0, overlay)
    img[:] = overlay


def _draw_ghost(img: np.ndarray, tr: dict[str, Any]) -> None:
    cls = str(tr.get("class", "vehicle"))
    base = GHOST if cls == "vehicle" else PED if cls == "pedestrian" else BIKE
    x, y = float(tr.get("x", 0)), float(tr.get("y", 0))
    yaw = float(tr.get("yaw", tr.get("heading", 1.57)))
    px, py = _to_px(x, y, STAGE_W, STAGE_H)
    # translucent hull (oriented box)
    L, W = 18, 10
    c, s = np.cos(yaw), np.sin(yaw)
    corners = []
    for dx, dy in ((-W, -L), (W, -L), (W, L), (-W, L)):
        # image y is up-forward already via _to_px; rotate in image space roughly
        rx = dx * c - dy * s
        ry = dx * s + dy * c
        corners.append([int(px + rx), int(py - ry)])
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.array(corners, dtype=np.int32)], base)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    cv2.polylines(img, [np.array(corners, dtype=np.int32)], True, PAPER, 1, cv2.LINE_AA)


def _draw_agent_forecasts(img: np.ndarray, tr: dict[str, Any]) -> None:
    modes = predict_modes(tr)
    if not modes:
        return
    m0 = modes[0]["points"]
    widths = cone_widths(len(m0))
    # mode0 solid + widening sausage
    for i in range(len(m0) - 1):
        a = _to_px(m0[i][0], m0[i][1], STAGE_W, STAGE_H)
        b = _to_px(m0[i + 1][0], m0[i + 1][1], STAGE_W, STAGE_H)
        cv2.line(img, a, b, ICE, 2, cv2.LINE_AA)
        rad = max(2, int(widths[i] * PPM * 0.35))
        overlay = img.copy()
        cv2.circle(overlay, b, rad, (50, 45, 40), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
    # modes 1–2 dashed dimmer
    for m in modes[1:]:
        pts = [_to_px(p[0], p[1], STAGE_W, STAGE_H) for p in m["points"]]
        for i in range(0, len(pts) - 1, 2):
            cv2.line(img, pts[i], pts[min(i + 1, len(pts) - 1)], (100, 100, 100), 1, cv2.LINE_AA)


def render_stage(state: dict[str, Any], ui: VizUI | None = None) -> np.ndarray:
    t0 = time.perf_counter()
    ui = ui or VizUI()
    clean = 0 in ui.layers
    img = np.full((STAGE_H, STAGE_W, 3), VOID, dtype=np.uint8)
    cv2.rectangle(img, (60, 30), (STAGE_W - 60, STAGE_H - 30), (18, 16, 15), -1)

    loop_hz = float(state.get("loop_hz") or 0.0)
    drop_heavy = loop_hz > 0 and loop_hz < 8.0
    engaged = bool(state.get("engaged"))

    _draw_underglow(img, engaged)
    path = state.get("path_ego") or []
    _draw_filled_corridor(img, path, float(state.get("path_conf") or 0.5))

    # ego desaturated shell
    ex, ey = _to_px(0, 0, STAGE_W, STAGE_H)
    cv2.rectangle(img, (ex - 10, ey - 18), (ex + 10, ey + 10), (55, 55, 55), -1)

    tracks = list(state.get("tracks") or [])[:MAX_AGENTS]
    n_forecast = 0
    for tr in tracks:
        _draw_ghost(img, tr)
        if clean or drop_heavy:
            continue
        if n_forecast >= MAX_FORECAST:
            continue
        _draw_agent_forecasts(img, tr)
        n_forecast += 1

    if 1 in ui.layers and state.get("occupancy") is not None:
        pass  # occupancy paint later

    # PIP top-left 16:9, thin ice border; no crosshair unless debug 2
    if not drop_heavy and not clean:
        pip = np.full((180, 320, 3), (28, 28, 28), dtype=np.uint8)
        if 2 in ui.layers:
            cv2.putText(pip, "yolo debug", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PAPER, 1)
            cv2.drawMarker(pip, (160, 90), ICE, cv2.MARKER_CROSS, 12, 1)
        img[12:192, 12:332] = pip
        cv2.rectangle(img, (12, 12), (332, 192), ICE, 1)

    # letterbox chrome — titles NOT in world geometry
    cv2.rectangle(img, (0, 0), (STAGE_W, 22), (12, 13, 16), -1)
    cv2.putText(img, "GVD", (12, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ICE, 1, cv2.LINE_AA)
    cv2.putText(img, "VISION", (52, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    if ui.top_down:
        cv2.putText(img, "BEV debug", (STAGE_W - 110, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)

    state["viz_ms"] = (time.perf_counter() - t0) * 1000.0

    if ui.show_nerd and not clean:
        panel = render_panel(state, h=STAGE_H, w=420, show_help=ui.show_help)
        out = np.concatenate([img, panel], axis=1)
    else:
        out = img
    return out


def smoke(ui: VizUI | None = None) -> Path:
    from pathlib import Path

    from python.runtime.state_io import default_state, write_state

    ui = ui or VizUI()
    st = default_state(
        engaged=True,
        loop_hz=12.0,
        camera_hz=10.0,
        path_conf=0.9,
        path_width=2.0,
        path_debug_preview=False,
        tracks_n=3,
        objects_n=3,
        tracks=[
            {"id": 1, "class": "vehicle", "x": -3.5, "y": 18, "speed_mps": 12, "yaw": 1.55, "yaw_rate": 0.0},
            {"id": 2, "class": "vehicle", "x": 3.0, "y": 25, "speed_mps": 8, "yaw": 1.6, "yaw_rate": 0.05},
            {"id": 3, "class": "pedestrian", "x": 6.0, "y": 12, "speed_mps": 1.2, "yaw": 3.1, "yaw_rate": 0.0},
        ],
        planner={"corridor_width": 2.0, "curvature": 0.01, "target_v": 12.0, "ttc_lead": 2.4, "aeb": "off"},
        missing_state_keys=["live cameras", "real planner path", "occupancy grid"],
    )
    st["path_ego"] = [{"x": 0.15 * np.sin(i / 12), "y": float(i), "z": 0.0} for i in range(0, 36)]
    write_state(st)
    frame = render_stage(st, ui=ui)
    out = Path("docs/gvd_viz_smoke.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    return out
