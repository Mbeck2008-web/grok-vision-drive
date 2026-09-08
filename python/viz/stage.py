"""Tesla-like cabin stage (OpenCV). Chase 3/4 bird default; BEV via T. No FSD marks."""

from __future__ import annotations

import math
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
CORRIDOR = (199, 167, 90)  # #5aa7c7
# cooler vehicle ghost — push blue vs warm smoke cast (#b9c0c7 + cooler bias)
GHOST = (210, 200, 170)
PED = (138, 176, 215)
BIKE = (90, 184, 224)
BLOCKED = (92, 92, 196)

MAX_AGENTS = 32
MAX_FORECAST = 16
STAGE_W, STAGE_H = 960, 720
PATH_WIDTH_M = 2.0
PATH_FADE_START_M = 25.0
PATH_FADE_END_M = 40.0


@dataclass
class VizUI:
    show_nerd: bool = True
    show_help: bool = False
    layers: set[int] = field(default_factory=set)
    top_down: bool = False  # default = chase 3/4 bird; T toggles BEV

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


@dataclass
class Cam:
    top_down: bool = False
    ppm: float = 10.0

    def project(self, x: float, y: float, z: float = 0.0) -> tuple[int, int]:
        if self.top_down:
            cx, cy = STAGE_W // 2, STAGE_H - 90
            return int(cx + x * self.ppm), int(cy - y * self.ppm)
        # Chase-up 3/4 bird: slightly behind/right of ego, FOV ~50, look forward-down
        cam = np.array([4.5, -11.0, 7.5], dtype=np.float64)
        target = np.array([0.0, 12.0, 0.0], dtype=np.float64)
        eye = np.array([x, y, z], dtype=np.float64) - cam
        forward = target - cam
        forward = forward / (np.linalg.norm(forward) + 1e-9)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right = right / (np.linalg.norm(right) + 1e-9)
        up = np.cross(right, forward)
        # camera-space
        cx = float(np.dot(eye, right))
        cy = float(np.dot(eye, up))
        cz = float(np.dot(eye, forward))
        if cz < 0.6:
            cz = 0.6
        fov = math.radians(50.0)
        f = (STAGE_H * 0.5) / math.tan(fov * 0.5)
        u = STAGE_W * 0.5 + f * cx / cz
        v = STAGE_H * 0.5 - f * cy / cz
        return int(u), int(v)

    def scale_at(self, y: float) -> float:
        """Approx pixels-per-meter for ribbon width at forward distance y."""
        if self.top_down:
            return self.ppm
        p0 = self.project(0.0, y, 0.0)
        p1 = self.project(1.0, y, 0.0)
        return max(2.0, abs(p1[0] - p0[0]))


def _draw_filled_corridor(img: np.ndarray, path: list[dict], conf: float, cam: Cam) -> None:
    if len(path) < 2:
        return
    # truncate past PATH_FADE_END_M
    trimmed = []
    dist = 0.0
    prev = None
    for p in path:
        x, y = float(p.get("x", 0)), float(p.get("y", 0))
        if prev is not None:
            dist += math.hypot(x - prev[0], y - prev[1])
        if dist > PATH_FADE_END_M:
            break
        trimmed.append((x, y, dist if prev is not None else 0.0))
        prev = (x, y)
    if len(trimmed) < 2:
        return

    left, right, center, alphas = [], [], [], []
    for i, (x, y, d) in enumerate(trimmed):
        if i + 1 < len(trimmed):
            nx, ny = trimmed[i + 1][0], trimmed[i + 1][1]
        else:
            nx, ny = trimmed[i - 1][0], trimmed[i - 1][1]
            nx, ny = x - (nx - x), y - (ny - y)
        tx, ty = nx - x, ny - y
        norm = math.hypot(tx, ty) + 1e-6
        px, py = ty / norm, -tx / norm
        half_m = PATH_WIDTH_M * 0.5
        # fade width + alpha after 25 m
        if d <= 15.0:
            bright = 1.0
        elif d < PATH_FADE_START_M:
            bright = 0.85
        else:
            t = (d - PATH_FADE_START_M) / max(1e-6, PATH_FADE_END_M - PATH_FADE_START_M)
            bright = max(0.0, 1.0 - t)
        half_px = half_m * cam.scale_at(y) * (0.65 + 0.35 * bright)
        cx, cy = cam.project(x, y, 0.05)
        lx, ly = cam.project(x - px * half_m, y - py * half_m, 0.05)
        rx, ry = cam.project(x + px * half_m, y + py * half_m, 0.05)
        # keep edge spacing if projection collapses
        if abs(rx - lx) < 2 and abs(ry - ly) < 2:
            lx, ly = int(cx - half_px), cy
            rx, ry = int(cx + half_px), cy
        left.append([lx, ly])
        right.append([rx, ry])
        center.append((cx, cy))
        alphas.append(bright)

    # draw as segment quads with per-segment alpha (fade out 25–40 m)
    for i in range(len(left) - 1):
        a = 0.35 * max(0.15, conf) * (0.55 * alphas[i] + 0.45 * alphas[i + 1])
        if a < 0.02:
            continue
        quad = np.array([left[i], left[i + 1], right[i + 1], right[i]], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [quad], CORRIDOR)
        # extra punch first 15 m
        if alphas[i] > 0.95:
            a = min(0.45, a + 0.08)
        cv2.addWeighted(overlay, a, img, 1 - a, 0, img)
        edge = tuple(int(c * (0.5 + 0.5 * alphas[i])) for c in ICE)
        cv2.line(img, tuple(left[i]), tuple(left[i + 1]), edge, 1, cv2.LINE_AA)
        cv2.line(img, tuple(right[i]), tuple(right[i + 1]), edge, 1, cv2.LINE_AA)
        cv2.line(img, center[i], center[i + 1], ICE, 1, cv2.LINE_AA)


def _draw_underglow(img: np.ndarray, engaged: bool, cam: Cam) -> None:
    if not engaged:
        return
    ex, ey = cam.project(0.0, 0.5, 0.0)
    overlay = img.copy()
    for r, a in ((42, 0.22), (26, 0.32), (14, 0.42)):
        layer = img.copy()
        cv2.circle(layer, (ex, ey + 2), r, ICE, -1, cv2.LINE_AA)
        cv2.addWeighted(layer, a, overlay, 1 - a, 0, overlay)
    img[:] = overlay


def _draw_ghost(img: np.ndarray, tr: dict[str, Any], cam: Cam) -> None:
    cls = str(tr.get("class", "vehicle"))
    base = GHOST if cls == "vehicle" else PED if cls == "pedestrian" else BIKE
    x, y = float(tr.get("x", 0)), float(tr.get("y", 0))
    yaw = float(tr.get("yaw", tr.get("heading", 1.57)))
    # oriented hull in ego frame, then project corners
    L, W = 2.2, 0.95  # meters
    c, s = math.cos(yaw), math.sin(yaw)
    corners = []
    for dx, dy in ((-W, -L), (W, -L), (W, L), (-W, L)):
        # yaw 0 = +Y forward in ego frame
        wx = x + dx * c - dy * s
        wy = y + dx * s + dy * c
        corners.append(list(cam.project(wx, wy, 0.4)))
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.array(corners, dtype=np.int32)], base)
    cv2.addWeighted(overlay, 0.32, img, 0.68, 0, img)
    cv2.polylines(img, [np.array(corners, dtype=np.int32)], True, PAPER, 1, cv2.LINE_AA)


def _draw_agent_forecasts(img: np.ndarray, tr: dict[str, Any], cam: Cam) -> None:
    modes = predict_modes(tr)
    if not modes:
        return
    m0 = modes[0]["points"]
    widths = cone_widths(len(m0))
    for i in range(len(m0) - 1):
        a = cam.project(m0[i][0], m0[i][1], 0.3)
        b = cam.project(m0[i + 1][0], m0[i + 1][1], 0.3)
        cv2.line(img, a, b, ICE, 2, cv2.LINE_AA)
        rad = max(2, int(widths[i] * cam.scale_at(m0[i][1]) * 0.30))
        overlay = img.copy()
        cv2.circle(overlay, b, rad, (55, 50, 45), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.14, img, 0.86, 0, img)
    for m in modes[1:]:
        pts = [cam.project(p[0], p[1], 0.3) for p in m["points"]]
        for i in range(0, len(pts) - 1, 2):
            cv2.line(img, pts[i], pts[min(i + 1, len(pts) - 1)], (100, 100, 100), 1, cv2.LINE_AA)


def render_stage(state: dict[str, Any], ui: VizUI | None = None) -> np.ndarray:
    t0 = time.perf_counter()
    ui = ui or VizUI()
    clean = 0 in ui.layers
    cam = Cam(top_down=ui.top_down)
    img = np.full((STAGE_H, STAGE_W, 3), VOID, dtype=np.uint8)
    # dim ground slab (void stage — not map texture)
    cv2.rectangle(img, (40, 20), (STAGE_W - 40, STAGE_H - 20), (18, 16, 15), -1)

    loop_hz = float(state.get("loop_hz") or 0.0)
    drop_heavy = loop_hz > 0 and loop_hz < 8.0
    engaged = bool(state.get("engaged"))

    _draw_underglow(img, engaged, cam)
    _draw_filled_corridor(img, state.get("path_ego") or [], float(state.get("path_conf") or 0.5), cam)

    # ego shell
    corners = [
        cam.project(-0.9, -1.2, 0.3),
        cam.project(0.9, -1.2, 0.3),
        cam.project(0.9, 1.8, 0.3),
        cam.project(-0.9, 1.8, 0.3),
    ]
    cv2.fillPoly(img, [np.array(corners, dtype=np.int32)], (55, 55, 55))

    tracks = list(state.get("tracks") or [])[:MAX_AGENTS]
    n_forecast = 0
    for tr in tracks:
        _draw_ghost(img, tr, cam)
        if clean or drop_heavy:
            continue
        if n_forecast >= MAX_FORECAST:
            continue
        _draw_agent_forecasts(img, tr, cam)
        n_forecast += 1

    if not drop_heavy and not clean:
        pip = np.full((180, 320, 3), (28, 28, 28), dtype=np.uint8)
        if 2 in ui.layers:
            cv2.putText(pip, "cam_main", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PAPER, 1)
            cv2.drawMarker(pip, (160, 90), ICE, cv2.MARKER_CROSS, 12, 1)
        img[12:192, 12:332] = pip
        cv2.rectangle(img, (12, 12), (332, 192), ICE, 1)

    cv2.rectangle(img, (0, 0), (STAGE_W, 22), (12, 13, 16), -1)
    cv2.putText(img, "GVD", (12, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ICE, 1, cv2.LINE_AA)
    cv2.putText(img, "VISION", (52, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    cam_lbl = "BEV debug" if ui.top_down else "chase 3/4"
    cv2.putText(img, cam_lbl, (STAGE_W - 120, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)

    state["viz_ms"] = (time.perf_counter() - t0) * 1000.0
    if ui.show_nerd and not clean:
        panel = render_panel(state, h=STAGE_H, w=420, show_help=ui.show_help)
        return np.concatenate([img, panel], axis=1)
    return img


def smoke(ui: VizUI | None = None) -> "Path":
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
    st["path_ego"] = [{"x": 0.12 * math.sin(i / 14), "y": float(i), "z": 0.0} for i in range(0, 45)]
    write_state(st)
    frame = render_stage(st, ui=ui)
    out = Path("docs/gvd_viz_smoke.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    return out
