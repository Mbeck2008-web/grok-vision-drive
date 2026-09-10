"""GVD VISION cabin stage (OpenCV). Chase 3/4 bird default; BEV via T. No FSD/Tesla marks."""

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
STAGE_W, STAGE_H = 1280, 800
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



def _draw_path_world_overlay(img: np.ndarray, path_world: list, cam: Cam) -> None:
    """BEV sanity: project path_world when T (top_down) so game==viz is visible."""
    if not cam.top_down or not path_world or len(path_world) < 2:
        return
    # path_world is absolute; overlay relative to first point as origin for BEV sanity
    x0 = float(path_world[0].get("x", 0))
    y0 = float(path_world[0].get("y", 0))
    pts = []
    for p in path_world[:80]:
        # treat delta as ego-ish for BEV debug (forward≈Δ along path dominant axis)
        dx = float(p.get("x", 0)) - x0
        dy = float(p.get("y", 0)) - y0
        pts.append(cam.project(dx, dy, 0.0))
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], (120, 180, 90), 1, cv2.LINE_AA)  # dim green sanity
    cv2.putText(img, "path_world", (pts[0][0] + 4, pts[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 180, 90), 1, cv2.LINE_AA)


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


def _draw_ghost(img: np.ndarray, tr: dict[str, Any], cam: Cam, *, lead: bool = False) -> None:
    cls = str(tr.get("class", "vehicle"))
    base = GHOST if cls == "vehicle" else PED if cls == "pedestrian" else BIKE
    x, y = float(tr.get("x", 0)), float(tr.get("y", 0))
    yaw = float(tr.get("yaw", tr.get("heading", 1.57)))
    # oriented hull — vehicles ~4.2×1.8 m visual scale (display), peds smaller
    if cls in ("pedestrian", "ped"):
        L, W = 0.6, 0.6
    elif cls in ("bicycle", "bike"):
        L, W = 1.8, 0.6
    else:
        L, W = 2.1, 0.9  # half-extents for Spec 4.2×1.8 m vehicle hull
    c, s = math.cos(yaw), math.sin(yaw)
    corners = []
    for dx, dy in ((-W, -L), (W, -L), (W, L), (-W, L)):
        # yaw 0 = +Y forward in ego frame
        wx = x + dx * c - dy * s
        wy = y + dx * s + dy * c
        corners.append(list(cam.project(wx, wy, 0.4)))
    pts = np.array(corners, dtype=np.int32)
    overlay = img.copy()
    fill_a = 0.45 if lead else 0.32
    cv2.fillPoly(overlay, [pts], base if not lead else ICE)
    cv2.addWeighted(overlay, fill_a, img, 1.0 - fill_a, 0, img)
    stroke = ICE if lead else PAPER
    thick = 3 if lead else 1
    cv2.polylines(img, [pts], True, stroke, thick, cv2.LINE_AA)
    if lead:
        tag = cam.project(x, y + L * 0.55, 1.2)
        cv2.putText(img, "LEAD", (tag[0] - 18, tag[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ICE, 1, cv2.LINE_AA)


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


def render_stage(
    state: dict[str, Any],
    ui: VizUI | None = None,
    main_frame: np.ndarray | None = None,
) -> np.ndarray:
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

    show_path = state.get("gvd_show_path")
    if show_path is None:
        show_path = True
    show_ghosts = state.get("show_agent_ghosts")
    if show_ghosts is None:
        show_ghosts = True

    _draw_underglow(img, engaged, cam)
    if show_path:
        _draw_filled_corridor(img, state.get("path_ego") or [], float(state.get("path_conf") or 0.5), cam)
        _draw_path_world_overlay(img, state.get("path_world") or [], cam)

    # ego shell
    corners = [
        cam.project(-0.9, -1.2, 0.3),
        cam.project(0.9, -1.2, 0.3),
        cam.project(0.9, 1.8, 0.3),
        cam.project(-0.9, 1.8, 0.3),
    ]
    cv2.fillPoly(img, [np.array(corners, dtype=np.int32)], (55, 55, 55))

    # Mode 0 (clean) still shows corridor + all tracks + CIPV — product viz, not empty cabin
    planner = state.get("planner") or {}
    cipv_id = planner.get("cipv_id")
    try:
        cipv_id = int(cipv_id) if cipv_id is not None else None
    except Exception:
        cipv_id = None

    tracks = list(state.get("tracks") or [])[:MAX_AGENTS] if show_ghosts else []
    n_forecast = 0
    for tr in tracks:
        tid = tr.get("id")
        try:
            tid_i = int(tid) if tid is not None else None
        except Exception:
            tid_i = None
        lead = cipv_id is not None and tid_i == cipv_id
        _draw_ghost(img, tr, cam, lead=lead)
        if drop_heavy:
            continue
        # forecasts: skip only when clean cabin AND we need FPS; mode 0 keeps cars, drops fans under 8 Hz already
        if clean:
            continue
        if n_forecast >= MAX_FORECAST:
            continue
        _draw_agent_forecasts(img, tr, cam)
        n_forecast += 1

    # Small cam_main PIP whenever a real frame exists (not only layer 2); drop under 8 Hz
    if main_frame is not None and getattr(main_frame, "size", 0) and not drop_heavy:
        pip = np.full((180, 320, 3), (28, 28, 28), dtype=np.uint8)
        try:
            pip = cv2.resize(main_frame, (320, 180), interpolation=cv2.INTER_AREA)
        except Exception:
            pass
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


def smoke(ui: VizUI | None = None, use_perception: bool = False) -> "Path":
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
            {"id": 1, "class": "vehicle", "x": -0.4, "y": 16, "speed_mps": 12, "yaw": 1.55, "yaw_rate": 0.0},
            {"id": 2, "class": "vehicle", "x": 3.0, "y": 25, "speed_mps": 8, "yaw": 1.6, "yaw_rate": 0.05},
            {"id": 3, "class": "pedestrian", "x": 6.0, "y": 12, "speed_mps": 1.2, "yaw": 3.1, "yaw_rate": 0.0},
        ],
        planner={"corridor_width": 2.0, "curvature": 0.01, "target_v": 12.0, "ttc_lead": 2.4, "aeb": "off", "cipv_id": 1},
        show_agent_ghosts=True,
        missing_state_keys=["live cameras", "real planner path", "occupancy grid"],
    )
    st["path_ego"] = [{"x": 0.12 * math.sin(i / 14), "y": float(i), "z": 0.0} for i in range(0, 45)]
    if use_perception:
        from python.perception.pipeline import ModularPerception
        import numpy as np

        perc = ModularPerception(allow_synthetic=True)
        fake = np.zeros((480, 640, 3), dtype=np.uint8)
        pout = perc.tick(fake, ego_speed_mps=12.0, steer_deg=0.0)
        st["tracks"] = pout.tracks
        st["tracks_n"] = pout.tracks_n
        st["objects_n"] = pout.objects_n
        st["lane_conf"] = pout.lane_conf
        st["lanes_bev"] = pout.lanes_bev
        st["path_ego"] = pout.path_ego
        st["path_width"] = pout.path_width
        st["path_conf"] = pout.path_conf
        st["path_debug_preview"] = pout.path_debug_preview
        st["planner"] = pout.planner
        st["infer_ms"] = pout.infer_ms
        st["detector"] = pout.detector_name
        st["signs"] = pout.signs
        miss = [m for m in (st.get("missing_state_keys") or []) if m not in ("tracks", "lanes_bev", "real path_ego from planner")]
        st["missing_state_keys"] = sorted(set(miss + pout.missing))
    if not st.get("lanes_ext"):
        # --smoke has no camera, so the Hough fit finds nothing. Give the in-game scene
        # something to draw, tagged kind="stub" so it can never read as a live detection.
        span = [float(i) for i in range(2, 40, 3)]
        def _line(off: float) -> list[dict[str, float]]:
            return [{"x": round(off + 0.9 * math.sin(y / 17.0), 2), "y": y} for y in span]
        st["lanes_ext"] = [
            {"points": _line(-1.85), "kind": "stub", "side": "left", "style": "unknown", "index": -1},
            {"points": _line(1.85), "kind": "stub", "side": "right", "style": "unknown", "index": 1},
            {"points": _line(-5.35), "kind": "stub", "side": "left", "style": "unknown", "index": -2},
            {"points": _line(5.35), "kind": "stub", "side": "right", "style": "unknown", "index": 2},
            {"points": _line(-8.85), "kind": "stub", "side": "left", "style": "unknown", "index": -3},
            {"points": _line(8.85), "kind": "stub", "side": "right", "style": "unknown", "index": 3},
        ]
        st["road_edges"] = [
            {"points": _line(-9.25), "kind": "stub", "side": "left"},
            {"points": _line(9.25), "kind": "stub", "side": "right"},
        ]
        st["missing_state_keys"] = sorted(set(
            (st.get("missing_state_keys") or []) + ["live lane paint (smoke uses stub lanes/edges)"]
        ))

    write_state(st)
    # Product smoke: mode 0 cabin with corridor + 3 ghosts + LEAD
    ui.layers = {0}
    ui.show_nerd = False
    frame = render_stage(st, ui=ui)
    out = Path("docs/gvd_viz_smoke.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    return out
