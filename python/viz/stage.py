"""GVD VISION cabin stage (OpenCV). Chase 3/4 bird default; BEV via T.

Lexicon (second-screen product viz): void stage, multi-lane fan, ice-blue ego
corridor + stop bar, agent boxes (CIPV / in-path / BRAKE), warm curbs,
sign/light glyphs. Driven from gvd_state.json. Titles stay GVD / VISION.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from python.runtime.debug_opts import (
    CONTROL_ROWS,
    LAYER_ATTR,
    VIZ_CONTROL_ROWS,
    DebugOpts,
    control_index,
    row_at,
    viz_control_index,
    viz_row_at,
)
from python.viz.debug_draw import (
    draw_cam_strip,
    draw_dense_hud,
    draw_frustums,
    draw_lane_polys,
    draw_occupancy,
    draw_pip_boxes,
    draw_planner_cost,
)
from python.viz.forecast import cone_widths, predict_modes
from python.viz.nerd import NERD_WIDTH, hit_test, render_panel, scene_note

VOID = (10, 8, 7)  # #07080a
PAPER = (225, 230, 232)
ICE = (212, 196, 158)  # #9ec4d4
ICE_HI = (248, 238, 214)  # #dbeef8
IN_PATH = (226, 190, 120)  # #78bee2
CORRIDOR = (199, 167, 90)  # #5aa7c7
GHOST = (188, 178, 168)
HAZARD = (92, 92, 196)
KERB = (124, 131, 168)  # warm grey-red, not lane paint
EGO_BODY = (84, 74, 64)
EGO_EDGE = (150, 136, 122)
PED = (138, 176, 215)
BIKE = (90, 184, 224)
POLE = (136, 128, 120)

MAX_AGENTS = 32
MAX_FORECAST = 16
STAGE_W, STAGE_H = 1280, 800
PATH_FADE_START_M = 25.0
PATH_FADE_END_M = 42.0


@dataclass
class VizUI:
    show_nerd: bool = True
    show_help: bool = False
    layers: set[int] = field(default_factory=set)
    top_down: bool = False  # default = chase 3/4 bird; T toggles BEV
    nerd_tab: str = "live"  # live | drive | viz | keys
    debug: DebugOpts = field(default_factory=DebugOpts)
    debug_sel: int = 0
    viz_sel: int = 0
    nerd_hits: list = field(default_factory=list)
    nerd_width: int = NERD_WIDTH

    def toggle_nerd(self) -> None:
        self.show_nerd = not self.show_nerd

    def toggle_help(self) -> None:
        self.nerd_tab = "live" if self.nerd_tab == "keys" else "keys"
        self.show_help = self.nerd_tab == "keys"
        if self.nerd_tab == "keys":
            self.show_nerd = True

    def show_drive_tab(self) -> None:
        self.nerd_tab = "drive"
        self.show_help = False
        self.show_nerd = True

    def show_viz_tab(self) -> None:
        self.nerd_tab = "viz"
        self.show_help = False
        self.show_nerd = True

    def cycle_tab(self, delta: int = 1) -> None:
        tabs = ("live", "drive", "viz", "keys")
        cur = self.nerd_tab if self.nerd_tab in tabs else "live"
        self.nerd_tab = tabs[(tabs.index(cur) + delta) % len(tabs)]
        self.show_help = self.nerd_tab == "keys"
        self.show_nerd = True

    def set_layer(self, k: int) -> None:
        if k == 0:
            self.layers = {0}
            return
        self.layers.discard(0)
        attr = LAYER_ATTR.get(k)
        if k in self.layers:
            self.layers.discard(k)
            if attr:
                setattr(self.debug, attr, False)
        else:
            self.layers.add(k)
            if attr:
                setattr(self.debug, attr, True)

    def sync_layers_from_debug(self) -> None:
        self.layers.discard(0)
        for k, attr in LAYER_ATTR.items():
            if getattr(self.debug, attr, False):
                self.layers.add(k)
            else:
                self.layers.discard(k)

    def handle_key(self, key: int) -> bool:
        """Consume a GVD VISION key that belongs to the nerd/DRIVE/VIZ tab. True = handled."""
        if key in (ord("d"), ord("D")):
            self.show_drive_tab()
            return True
        if key in (ord("g"), ord("G")):
            self.show_viz_tab()
            return True
        if key == ord("["):
            self.cycle_tab(-1)
            return True
        if key in (ord("]"), 9):  # Tab
            self.cycle_tab(1)
            return True
        if self.nerd_tab not in ("drive", "viz") or not self.show_nerd:
            return False
        if self.nerd_tab == "viz":
            n = max(1, len(VIZ_CONTROL_ROWS))
            sel_attr = "viz_sel"
            row_fn = viz_row_at
        else:
            n = max(1, len(CONTROL_ROWS))
            sel_attr = "debug_sel"
            row_fn = row_at
        if key in (82, 0, ord("k")):  # up
            setattr(self, sel_attr, (int(getattr(self, sel_attr)) - 1) % n)
            return True
        if key in (84, 1, ord("j")):  # down
            setattr(self, sel_attr, (int(getattr(self, sel_attr)) + 1) % n)
            return True
        row = row_fn(int(getattr(self, sel_attr)))
        if key in (81, 2, ord("h")):  # left
            self.debug.nudge(row, -1)
            self.sync_layers_from_debug()
            return True
        if key in (83, 3, ord("l")):  # right
            self.debug.nudge(row, +1)
            self.sync_layers_from_debug()
            return True
        if key in (13, 10, ord(" ")):
            self.debug.toggle(row)
            self.sync_layers_from_debug()
            return True
        return False

    def handle_click(self, x: int, y: int, *, stage_w: int) -> bool:
        if not self.show_nerd or 0 in self.layers:
            return False
        if x < stage_w:
            return False
        hit = hit_test(self.nerd_hits, x - stage_w, y)
        if not hit:
            return False
        if hit.get("kind") == "tab":
            tid = str(hit.get("id") or "live")
            self.nerd_tab = tid
            self.show_help = tid == "keys"
            return True
        if hit.get("kind") == "row":
            i = int(hit.get("i") or 0)
            if self.nerd_tab == "viz":
                self.viz_sel = viz_control_index(i)
                row = viz_row_at(self.viz_sel)
            else:
                self.debug_sel = control_index(i)
                row = row_at(self.debug_sel)
            part = str(hit.get("part") or "row")
            if part == "minus":
                self.debug.nudge(row, -1)
            elif part == "plus":
                self.debug.nudge(row, +1)
            else:
                self.debug.toggle(row)
            self.sync_layers_from_debug()
            return True
        return False


@dataclass
class Cam:
    top_down: bool = False
    ppm: float = 10.0

    @property
    def eye(self) -> tuple[float, float, float]:
        if self.top_down:
            return (0.0, -8.0, 40.0)
        # Chase 3/4, pulled back so ego does not eat the frame and the lane fan reads.
        return (1.6, -16.5, 5.4)

    def project(self, x: float, y: float, z: float = 0.0) -> tuple[int, int]:
        if self.top_down:
            cx, cy = STAGE_W // 2, STAGE_H - 90
            return int(cx + x * self.ppm), int(cy - y * self.ppm)
        cam = np.array(self.eye, dtype=np.float64)
        target = np.array([0.0, 16.0, -0.8], dtype=np.float64)
        eye = np.array([x, y, z], dtype=np.float64) - cam
        forward = target - cam
        forward = forward / (np.linalg.norm(forward) + 1e-9)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right = right / (np.linalg.norm(right) + 1e-9)
        up = np.cross(right, forward)
        cx = float(np.dot(eye, right))
        cy = float(np.dot(eye, up))
        cz = float(np.dot(eye, forward))
        if cz < 0.6:
            cz = 0.6
        fov = math.radians(42.0)
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


def _mix(color: tuple[int, int, int], alpha: float, bg: tuple[int, int, int] = VOID) -> tuple[int, int, int]:
    a = max(0.0, min(1.0, alpha))
    return tuple(int(c * a + b * (1.0 - a)) for c, b in zip(color, bg))  # type: ignore[return-value]


def _shade(color: tuple[int, int, int], k: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(c * k))) for c in color)  # type: ignore[return-value]


def _poly_points(src: Any) -> list[dict[str, float]]:
    """Tolerate lanes_ext.points, Lua pts, or a bare polyline."""
    if isinstance(src, dict):
        src = src.get("points") or src.get("pts") or []
    out: list[dict[str, float]] = []
    for p in src or []:
        if isinstance(p, dict):
            out.append({"x": float(p.get("x", 0)), "y": float(p.get("y", 0))})
        else:
            try:
                out.append({"x": float(p[0]), "y": float(p[1])})
            except (TypeError, IndexError, ValueError):
                continue
    return out


def _proj_poly(cam: Cam, pts: list[dict[str, float]], z: float = 0.0) -> list[tuple[int, int]]:
    return [cam.project(p["x"], p["y"], z) for p in pts]


def _stroke_poly(
    img: np.ndarray,
    pts: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int,
    *,
    dashed: bool = False,
    dash: int = 8,
    gap: int = 7,
) -> None:
    if len(pts) < 2:
        return
    if not dashed:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)
        return
    acc = 0.0
    on = True
    for i in range(len(pts) - 1):
        x0, y0 = float(pts[i][0]), float(pts[i][1])
        x1, y1 = float(pts[i + 1][0]), float(pts[i + 1][1])
        dx, dy = x1 - x0, y1 - y0
        seg = math.hypot(dx, dy)
        if seg < 1e-6:
            continue
        ux, uy = dx / seg, dy / seg
        t = 0.0
        while t < seg:
            period = float(dash if on else gap)
            remain = period - acc
            take = min(remain, seg - t)
            if on:
                a = (int(x0 + ux * t), int(y0 + uy * t))
                b = (int(x0 + ux * (t + take)), int(y0 + uy * (t + take)))
                cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
            t += take
            acc += take
            if acc >= period:
                acc = 0.0
                on = not on


def lane_draw_mode(kind: str | None, *, smoke: bool) -> str | None:
    """How to stroke a lanes_ext boundary: solid / dashed / skip.

    stub is smoke-only — live never ships it, and we must not draw a leftover stub
    as if Hough saw paint.
    """
    k = str(kind or "detected").lower()
    if k == "stub":
        return "dashed" if smoke else None
    if k == "predicted":
        return "dashed"
    return "solid"


def is_slowing(state: dict[str, Any]) -> bool:
    """Chevrons: AEB, a brake command, or target speed under current speed."""
    pl = state.get("planner") or {}
    ego = state.get("ego") or {}
    aeb = str(pl.get("aeb") or "off")
    if aeb in ("brake", "warn"):
        return True
    try:
        if float(ego.get("brake") or 0) > 0.05:
            return True
    except (TypeError, ValueError):
        pass
    try:
        tv = pl.get("target_v")
        sp = ego.get("speed_mps")
        if tv is not None and sp is not None and float(tv) < float(sp) - 0.6:
            return True
    except (TypeError, ValueError):
        pass
    return False


def is_halted(state: dict[str, Any]) -> bool:
    pl = state.get("planner") or {}
    if str(pl.get("aeb") or "off") == "brake":
        return True
    try:
        tv = pl.get("target_v")
        return tv is not None and float(tv) <= 0.2
    except (TypeError, ValueError):
        return False


def pace_scale(state: dict[str, Any]) -> float:
    """Ribbon shade: accelerate bright, coast normal, planned stop faint."""
    if is_halted(state):
        return 0.55
    if is_slowing(state):
        return 0.78
    pl = state.get("planner") or {}
    ego = state.get("ego") or {}
    try:
        tv = pl.get("target_v")
        sp = ego.get("speed_mps")
        if tv is not None and sp is not None and float(tv) > float(sp) + 0.6:
            return 1.15
    except (TypeError, ValueError):
        pass
    return 1.0


def cipv_id_of(state: dict[str, Any]) -> int | None:
    raw = (state.get("planner") or {}).get("cipv_id")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def track_id(tr: dict[str, Any]) -> int | None:
    raw = tr.get("id")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def track_cls(tr: dict[str, Any]) -> str:
    return str(tr.get("class") or tr.get("cls") or "vehicle")


def is_lead(tr: dict[str, Any], cipv_id: int | None) -> bool:
    if tr.get("lead"):
        return True
    tid = track_id(tr)
    return cipv_id is not None and tid is not None and tid == cipv_id


def is_hazard(tr: dict[str, Any], state: dict[str, Any], cipv_id: int | None) -> bool:
    """Red + BRAKE: CIPV and (AEB brake or TTC < 1.5)."""
    if not is_lead(tr, cipv_id):
        return False
    pl = state.get("planner") or {}
    if str(pl.get("aeb") or "off") == "brake":
        return True
    try:
        ttc = pl.get("ttc_lead")
        return ttc is not None and float(ttc) < 1.5
    except (TypeError, ValueError):
        return False


def in_path(tr: dict[str, Any], path: list[dict[str, Any]] | None, path_width: float) -> bool:
    """Inside the planner corridor. No corridor → nobody highlighted."""
    if not path or len(path) < 2:
        return False
    half = max(0.9, float(path_width or 2.0) * 0.5) + 0.45
    tx, ty = float(tr.get("x") or 0), float(tr.get("y") or 0)
    for p in path:
        if abs(float(p.get("y") or 0) - ty) > 2.5:
            continue
        if abs(float(p.get("x") or 0) - tx) <= half:
            return True
    return False


def lead_track(tracks: list[dict[str, Any]], cipv_id: int | None) -> dict[str, Any] | None:
    for tr in tracks:
        if is_lead(tr, cipv_id):
            return tr
    return None


def _drop_heavy(state: dict[str, Any]) -> bool:
    try:
        hz = float(state.get("loop_hz") or 0.0)
    except (TypeError, ValueError):
        return False
    return hz > 0 and hz < 8.0


def _allow_stub(state: dict[str, Any]) -> bool:
    return bool(state.get("viz_smoke"))


def _draw_ground(img: np.ndarray, cam: Cam) -> None:
    far_l, far_r = cam.project(-16, 60, 0), cam.project(16, 60, 0)
    near_l, near_r = cam.project(-16, -16, 0), cam.project(16, -16, 0)
    horizon = max(0, min(STAGE_H - 1, far_l[1]))
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.array([far_l, far_r, near_r, near_l], dtype=np.int32)], (22, 18, 15))
    cv2.addWeighted(overlay, 0.40, img, 0.60, 0, img)
    tick = _mix(ICE, 0.08)
    for d in (10, 20, 30, 40):
        a, b = cam.project(-7, d, 0), cam.project(7, d, 0)
        cv2.line(img, a, b, tick, 1, cv2.LINE_AA)


def _draw_fog(img: np.ndarray, cam: Cam) -> None:
    """Void the sky plus a thin horizon fade — do not wipe the lane fan."""
    horizon = max(0, min(STAGE_H - 1, cam.project(0, 70, 0)[1]))
    sky = img.copy()
    cv2.rectangle(sky, (0, 0), (STAGE_W, max(1, horizon)), VOID, -1)
    cv2.addWeighted(sky, 0.55, img, 0.45, 0, img)
    fade = img.copy()
    cv2.rectangle(fade, (0, max(0, horizon)), (STAGE_W, min(STAGE_H, horizon + 28)), VOID, -1)
    cv2.addWeighted(fade, 0.22, img, 0.78, 0, img)


def _draw_lanes(img: np.ndarray, lanes: list, cam: Cam, *, smoke: bool) -> None:
    for ln in lanes or []:
        kind = ln.get("kind") if isinstance(ln, dict) else None
        mode = lane_draw_mode(kind, smoke=smoke)
        if mode is None:
            continue
        pts = _poly_points(ln)
        if len(pts) < 2:
            continue
        idx = 1
        if isinstance(ln, dict):
            try:
                idx = int(ln.get("index") if ln.get("index") is not None else ln.get("idx") or 1)
            except (TypeError, ValueError):
                idx = 1
        outer = max(0, abs(idx) - 1)
        fade = max(0.35, 1.0 - 0.22 * outer)
        style = str(ln.get("style") or "unknown") if isinstance(ln, dict) else "unknown"
        dashed = mode == "dashed" or style == "dashed"
        if mode == "solid":
            color = tuple(max(90, int(c * max(0.55, fade))) for c in PAPER)
            thick = 2
        else:
            color = tuple(max(70, int(c * max(0.40, 0.70 * fade))) for c in (170, 166, 160))
            thick = 2
        _stroke_poly(
            img, _proj_poly(cam, pts, 0.02), color, thick,
            dashed=dashed, dash=10 if mode == "solid" else 8, gap=5,
        )


def _draw_edges(img: np.ndarray, edges: list, cam: Cam) -> None:
    for e in edges or []:
        pts = _poly_points(e)
        if len(pts) < 2:
            continue
        kind = e.get("kind") if isinstance(e, dict) else "predicted"
        solid = str(kind or "predicted") == "detected"
        base = _proj_poly(cam, pts, 0.0)
        top = _proj_poly(cam, pts, 0.13)
        quad = np.array(base + list(reversed(top)), dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [quad], KERB)
        cv2.addWeighted(overlay, 0.22 if solid else 0.14, img, 1.0 - (0.22 if solid else 0.14), 0, img)
        _stroke_poly(img, top, _mix(KERB, 0.85 if solid else 0.62, (22, 20, 18)), 2, dashed=not solid, dash=6, gap=5)


def _draw_signs(img: np.ndarray, signs: list, cam: Cam) -> None:
    if not signs:
        return
    ordered = sorted(signs, key=lambda s: float(s.get("y") or 0), reverse=True)
    for s in ordered:
        cls = str(s.get("cls") or s.get("class") or "sign")
        x, y = float(s.get("x") or 0), float(s.get("y") or 0)
        light = cls == "traffic_light"
        pole = cls == "pole"
        h = 3.2 if light else (1.1 if pole else 2.1)
        far = max(0.2, min(1.0, 1.0 - (y - 30.0) / 25.0))
        scale = cam.scale_at(y)
        top = cam.project(x, y, h)
        foot = cam.project(x, y, 0.0)
        cv2.line(img, foot, top, _mix(POLE, (0.75 if pole else 0.50) * far), max(1, int((0.14 if pole else 0.06) * scale)), cv2.LINE_AA)
        if pole:
            continue
        if light:
            w = max(6, int(0.34 * scale))
            hh = max(15, int(0.95 * scale))
            routed = s.get("relevant") is True
            x0, y0 = top[0] - w // 2, top[1] - hh
            cv2.rectangle(img, (x0, y0), (x0 + w, top[1]), _mix((36, 31, 26), 0.92 * far), -1)
            cv2.rectangle(img, (x0, y0), (x0 + w, top[1]), _mix(ICE if routed else PAPER, (0.75 if routed else 0.30) * far), 1)
            # No lamp-colour classifier: empty rings unless state names a lamp.
            lamp_r = max(1, int(w * 0.26))
            on = {"red": 0, "amber": 1, "green": 2}.get(str(s.get("state") or ""))
            lamp_col = [(74, 84, 192), (76, 168, 214), (130, 190, 110)]
            for i in range(3):
                cy = y0 + int(hh * (0.22 + 0.28 * i))
                if on == i:
                    cv2.circle(img, (top[0], cy), lamp_r, _mix(lamp_col[i], 0.95 * far), -1, cv2.LINE_AA)
                else:
                    cv2.circle(img, (top[0], cy), lamp_r, _mix(ICE if routed else (102, 94, 86), 0.8 * far), 1, cv2.LINE_AA)
            continue
        # Flat-top octagon + white bar — a red disc on a post reads as a lamp.
        r = max(7, int(0.45 * scale))
        oct_pts = []
        for k in range(8):
            ang = math.pi / 8 + k * math.pi / 4
            oct_pts.append([int(top[0] + r * math.cos(ang)), int(top[1] - r + r * math.sin(ang))])
        cv2.fillPoly(img, [np.array(oct_pts, dtype=np.int32)], _mix((66, 74, 176), 0.9 * far))
        cv2.polylines(img, [np.array(oct_pts, dtype=np.int32)], True, _mix(PAPER, 0.8 * far), max(1, int(r * 0.16)), cv2.LINE_AA)
        cv2.rectangle(
            img,
            (int(top[0] - r * 0.52), int(top[1] - r * 1.1)),
            (int(top[0] + r * 0.52), int(top[1] - r * 1.1 + max(1.5, r * 0.2))),
            _mix(PAPER, 0.92 * far),
            -1,
        )


def _draw_stop_bar(
    img: np.ndarray,
    path: list[dict[str, Any]],
    half_w: float,
    tracks: list[dict[str, Any]],
    cipv_id: int | None,
    cam: Cam,
    halted: bool,
) -> None:
    if not halted:
        return
    lead = lead_track(tracks, cipv_id)
    if not lead:
        return
    y = max(1.5, float(lead.get("y") or 0) - 2.2)
    cx = 0.0
    for p in path:
        cx = float(p.get("x") or 0)
        if float(p.get("y") or 0) >= y:
            break
    a, b = cam.project(cx - half_w, y, 0.05), cam.project(cx + half_w, y, 0.05)
    cv2.line(img, a, b, (252, 248, 244), 3, cv2.LINE_AA)


def _draw_filled_corridor(
    img: np.ndarray,
    path: list[dict],
    conf: float,
    cam: Cam,
    *,
    half_w: float,
    intent: float,
    preview: bool,
) -> None:
    if len(path) < 2:
        return
    trimmed: list[tuple[float, float, float]] = []
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
        px, py = (ty / norm) * half_w, (-tx / norm) * half_w
        fade = 1.0 if d <= PATH_FADE_START_M else max(0.0, 1.0 - (d - PATH_FADE_START_M) / (PATH_FADE_END_M - PATH_FADE_START_M))
        cx, cy = cam.project(x, y, 0.03)
        lx, ly = cam.project(x - px, y - py, 0.03)
        rx, ry = cam.project(x + px, y + py, 0.03)
        if abs(rx - lx) < 3 and abs(ry - ly) < 3:
            half_px = max(3.0, half_w * cam.scale_at(y))
            lx, ly = int(cx - half_px), cy
            rx, ry = int(cx + half_px), cy
        left.append((lx, ly))
        right.append((rx, ry))
        center.append((cx, cy))
        alphas.append(fade)

    nseg = len(left) - 1
    for i in range(nseg):
        near = 1.0 - i / max(1, nseg)
        a = intent * (0.38 + 0.50 * near) * max(0.25, conf) * (0.55 * alphas[i] + 0.45 * alphas[i + 1])
        if a < 0.02:
            continue
        quad = np.array([left[i], left[i + 1], right[i + 1], right[i]], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [quad], CORRIDOR)
        cv2.addWeighted(overlay, min(0.62, a), img, 1 - min(0.62, a), 0, img)

    for i in range(nseg):
        edge = _mix(ICE_HI, 0.8 * alphas[i])
        _stroke_poly(img, [left[i], left[i + 1]], edge, 2, dashed=preview, dash=5, gap=4)
        _stroke_poly(img, [right[i], right[i + 1]], edge, 2, dashed=preview, dash=5, gap=4)
    _stroke_poly(img, center, _mix(ICE, 0.30), 1, dashed=True, dash=4, gap=5)


def _draw_path_world_overlay(img: np.ndarray, path_world: list, cam: Cam) -> None:
    """BEV sanity: project path_world when T (top_down) so game==viz is visible."""
    if not cam.top_down or not path_world or len(path_world) < 2:
        return
    x0 = float(path_world[0].get("x", 0))
    y0 = float(path_world[0].get("y", 0))
    pts = []
    for p in path_world[:80]:
        dx = float(p.get("x", 0)) - x0
        dy = float(p.get("y", 0)) - y0
        pts.append(cam.project(dx, dy, 0.0))
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], (120, 180, 90), 1, cv2.LINE_AA)
    cv2.putText(img, "path_world", (pts[0][0] + 4, pts[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 180, 90), 1, cv2.LINE_AA)


def _draw_underglow(img: np.ndarray, engaged: bool, cam: Cam) -> None:
    if not engaged:
        return
    ex, ey = cam.project(0.0, 0.2, 0.02)
    overlay = img.copy()
    for r, a in ((42, 0.22), (26, 0.32), (14, 0.42)):
        layer = img.copy()
        cv2.circle(layer, (ex, ey + 2), r, ICE, -1, cv2.LINE_AA)
        cv2.addWeighted(layer, a, overlay, 1 - a, 0, overlay)
    img[:] = overlay


def _draw_box(
    img: np.ndarray,
    cam: Cam,
    x: float,
    y: float,
    yaw: float,
    length: float,
    width: float,
    height: float,
    color: tuple[int, int, int],
    alpha: float,
    stroke: tuple[int, int, int],
    stroke_w: int,
) -> None:
    """Low-poly box. yaw is from +X (π/2 = straight ahead), same as the tracker."""
    hx, hy = math.cos(yaw), math.sin(yaw)
    sx, sy = hy, -hx
    hl, hw = length * 0.5, width * 0.5
    signs = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    base, top, shadow = [], [], []
    for s0, s1 in signs:
        px = x + hx * hl * s0 + sx * hw * s1
        py = y + hy * hl * s0 + sy * hw * s1
        base.append(cam.project(px, py, 0.0))
        top.append(cam.project(px, py, height))
        shadow.append(cam.project(x + (px - x) * 1.12, y + (py - y) * 1.12, 0.0))
    cv2.fillPoly(img, [np.array(shadow, dtype=np.int32)], _mix((0, 0, 0), 0.3 * alpha))
    eye = cam.eye
    faces = [
        {"p": [top[0], top[1], top[2], top[3]], "n": (0.0, 0.0, 1.0), "c": (x, y, height), "k": 1.0},
        {"p": [base[1], base[2], top[2], top[1]], "n": (hx, hy, 0.0), "c": (x + hx * hl, y + hy * hl, height * 0.5), "k": 0.62},
        {"p": [base[3], base[0], top[0], top[3]], "n": (-hx, -hy, 0.0), "c": (x - hx * hl, y - hy * hl, height * 0.5), "k": 0.50},
        {"p": [base[2], base[3], top[3], top[2]], "n": (sx, sy, 0.0), "c": (x + sx * hw, y + sy * hw, height * 0.5), "k": 0.40},
        {"p": [base[0], base[1], top[1], top[0]], "n": (-sx, -sy, 0.0), "c": (x - sx * hw, y - sy * hw, height * 0.5), "k": 0.40},
    ]
    vis = []
    for fc in faces:
        vx, vy, vz = fc["c"][0] - eye[0], fc["c"][1] - eye[1], fc["c"][2] - eye[2]
        nx, ny, nz = fc["n"]
        if nx * vx + ny * vy + nz * vz >= 0:
            continue
        fc["d"] = vx * vx + vy * vy + vz * vz
        vis.append(fc)
        vis.sort(key=lambda f: -f["d"])
        for fc in vis:
            # Lead / in-path boxes stay icy; do not crush them with the far-face shade.
            k = fc["k"] if alpha < 0.8 else max(0.72, fc["k"])
            fill = _mix(_shade(color, k), alpha)
            cv2.fillPoly(img, [np.array(fc["p"], dtype=np.int32)], fill)
            cv2.polylines(img, [np.array(fc["p"], dtype=np.int32)], True, _mix(stroke, min(1.0, alpha * 0.8)), stroke_w, cv2.LINE_AA)


def _track_dims(tr: dict[str, Any], cls: str) -> tuple[float, float, float]:
    """One box per agent. Optional length/width/height on the track resize it."""
    if cls in ("pedestrian", "ped"):
        L, W, H = 0.6, 0.6, 1.75
    elif cls in ("bicycle", "bike"):
        L, W, H = 1.8, 0.6, 1.6
    else:
        L, W, H = 4.2, 1.8, 1.55
    for key in ("length", "width", "height"):
        raw = tr.get(key)
        if raw is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if v > 0.2:
            if key == "length":
                L = v
            elif key == "width":
                W = v
            else:
                H = v
    return L, W, H


def _draw_tracks(
    img: np.ndarray,
    tracks: list[dict[str, Any]],
    cam: Cam,
    *,
    path: list[dict[str, Any]],
    path_width: float,
    state: dict[str, Any],
    cipv_id: int | None,
    show_ids: bool = False,
    show_vel: bool = False,
) -> None:
    ordered = sorted(tracks, key=lambda t: float(t.get("y") or 0), reverse=True)
    for t in ordered:
        cls = track_cls(t)
        L, W, H = _track_dims(t, cls)
        far = max(0.18, min(1.0, 1.0 - (float(t.get("y") or 0) - 30.0) / 25.0))
        lead = is_lead(t, cipv_id)
        hazard = is_hazard(t, state, cipv_id)
        hot = lead or in_path(t, path, path_width)
        col = HAZARD if hazard else (IN_PATH if hot else GHOST)
        stroke = (160, 168, 242) if hazard else (ICE_HI if hot else PAPER)
        alpha = (0.85 if hot else 0.74) * far
        try:
            yaw = float(t["yaw"]) if t.get("yaw") is not None else math.pi / 2
        except (TypeError, ValueError):
            yaw = math.pi / 2
        x, y = float(t.get("x") or 0), float(t.get("y") or 0)
        _draw_box(img, cam, x, y, yaw, L, W, H, col, alpha, stroke, 2 if (hot or hazard) else 1)
        if lead:
            tag_h = H + 0.45
            tag = cam.project(x, y, tag_h)
            label = "BRAKE" if hazard else "LEAD"
            cv2.putText(img, label, (tag[0] - 18, tag[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ICE_HI if not hazard else (178, 186, 246), 1, cv2.LINE_AA)
        elif show_ids:
            tag = cam.project(x, y, H + 0.35)
            tid = track_id(t)
            cv2.putText(
                img,
                f"#{tid}" if tid is not None else cls[:4],
                (tag[0] - 10, tag[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                PAPER,
                1,
                cv2.LINE_AA,
            )
        if show_vel:
            try:
                spd = float(t.get("speed_mps") or 0.0)
            except (TypeError, ValueError):
                spd = 0.0
            reach = 0.6 + 0.18 * spd
            hx, hy = math.cos(yaw), math.sin(yaw)
            a = cam.project(x, y, 0.4)
            b = cam.project(x + hx * reach, y + hy * reach, 0.4)
            cv2.arrowedLine(img, a, b, ICE if hot else GHOST, 1, cv2.LINE_AA, tipLength=0.25)
        if show_ids and lead:
            tid = track_id(t)
            if tid is not None:
                extra = cam.project(x, y, H + 0.85)
                cv2.putText(img, f"#{tid}", (extra[0] - 10, extra[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.32, ICE_HI, 1, cv2.LINE_AA)


def _draw_ego(img: np.ndarray, cam: Cam, engaged: bool) -> None:
    if engaged:
        _draw_underglow(img, True, cam)
    yaw = math.pi / 2
    _draw_box(img, cam, 0.0, 0.0, yaw, 4.4, 1.85, 1.5, EGO_BODY, 1.0, EGO_EDGE, 1)


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


def _draw_state_fans(img: np.ndarray, agents: list, cam: Cam) -> None:
    """Mode-0 fans already written into state.agents (same toy math as predict_modes)."""
    for ag in agents or []:
        pts = _poly_points(ag.get("path_ego") if isinstance(ag, dict) else ag)
        if len(pts) < 2:
            continue
        screen = _proj_poly(cam, pts, 0.06)
        _stroke_poly(img, screen, _mix(ICE, 0.30), 1)
        cv2.circle(img, screen[-1], 2, _mix(ICE, 0.35), -1, cv2.LINE_AA)


def render_stage(
    state: dict[str, Any],
    ui: VizUI | None = None,
    main_frame: np.ndarray | None = None,
    cam_frames: dict[str, Any] | None = None,
    dets: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    t0 = time.perf_counter()
    ui = ui or VizUI()
    dbg = ui.debug
    clean = 0 in ui.layers
    cam = Cam(top_down=ui.top_down)
    img = np.full((STAGE_H, STAGE_W, 3), VOID, dtype=np.uint8)

    drop_heavy = _drop_heavy(state)
    engaged = bool(state.get("engaged"))
    smoke = _allow_stub(state)
    path = list(state.get("path_ego") or [])
    try:
        path_width = float(state.get("path_width") or 2.0)
    except (TypeError, ValueError):
        path_width = 2.0
    half_w = max(0.9, path_width * 0.5)
    try:
        path_conf = float(state.get("path_conf") or 0.5)
    except (TypeError, ValueError):
        path_conf = 0.5
    preview = bool(state.get("path_debug_preview"))

    show_path = bool(dbg.viz_path)
    show_ghosts = bool(dbg.viz_ghosts)

    _draw_ground(img, cam)
    if (not clean) and dbg.viz_occ:
        draw_occupancy(img, cam, list(state.get("tracks") or []), path, path_width)
    if dbg.viz_lanes:
        _draw_lanes(img, state.get("lanes_ext") or state.get("lanes") or [], cam, smoke=smoke)
    if dbg.viz_lanes:
        _draw_edges(img, state.get("road_edges") or state.get("edges") or [], cam)
    if (not clean) and dbg.viz_lane_poly:
        draw_lane_polys(img, cam, state.get("lanes_bev") or [])

    cipv_id = cipv_id_of(state)
    all_tracks = list(state.get("tracks") or [])[:MAX_AGENTS]
    tracks = all_tracks if show_ghosts else []

    if show_path:
        _draw_filled_corridor(
            img, path, path_conf, cam,
            half_w=half_w,
            intent=pace_scale(state),
            preview=preview,
        )
        _draw_stop_bar(img, path, half_w, all_tracks, cipv_id, cam, is_halted(state))
        _draw_path_world_overlay(img, state.get("path_world") or [], cam)

    if (not clean) and dbg.viz_cost:
        draw_planner_cost(img, cam, path, path_width, all_tracks)

    if not drop_heavy and dbg.viz_forecast:
        _draw_state_fans(img, state.get("agents") or [], cam)

    if show_ghosts:
        _draw_tracks(
            img, tracks, cam, path=path, path_width=path_width, state=state, cipv_id=cipv_id,
            show_ids=(not clean) and dbg.viz_ids,
            show_vel=(not clean) and dbg.viz_vel,
        )
        n_forecast = 0
        if not drop_heavy and dbg.viz_forecast:
            for tr in tracks:
                if n_forecast >= MAX_FORECAST:
                    break
                _draw_agent_forecasts(img, tr, cam)
                n_forecast += 1

    if not drop_heavy and dbg.viz_signs:
        _draw_signs(img, state.get("signs") or [], cam)

    if (not clean) and dbg.viz_frustums:
        draw_frustums(img, cam, state.get("cam_health") if isinstance(state.get("cam_health"), dict) else None)

    _draw_fog(img, cam)
    _draw_ego(img, cam, engaged)

    if main_frame is not None and getattr(main_frame, "size", 0) and not drop_heavy and dbg.viz_pip:
        pip = np.full((180, 320, 3), (28, 28, 28), dtype=np.uint8)
        try:
            pip = cv2.resize(main_frame, (320, 180), interpolation=cv2.INTER_AREA)
        except Exception:
            pass
        if (not clean) and dbg.viz_boxes and dets:
            try:
                h0, w0 = main_frame.shape[:2]
            except Exception:
                h0, w0 = 180, 320
            draw_pip_boxes(pip, dets, (w0, h0))
        cv2.putText(pip, "cam_main", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PAPER, 1)
        cv2.drawMarker(pip, (160, 90), ICE, cv2.MARKER_CROSS, 12, 1)
        img[12:192, 12:332] = pip
        cv2.rectangle(img, (12, 12), (332, 192), ICE, 1)

    if (not clean) and dbg.viz_cams and not drop_heavy:
        strip_top = draw_cam_strip(
            img, cam_frames, state.get("cam_health") if isinstance(state.get("cam_health"), dict) else None,
            stage_w=STAGE_W, stage_h=STAGE_H,
        )
        hud_pad = STAGE_H - strip_top + 6
    else:
        hud_pad = 0

    cv2.rectangle(img, (0, 0), (STAGE_W, 22), (12, 13, 16), -1)
    cv2.putText(img, "GVD", (12, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ICE, 1, cv2.LINE_AA)
    cv2.putText(img, "VISION", (52, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    cam_lbl = "BEV debug" if ui.top_down else "chase 3/4"
    cv2.putText(img, cam_lbl, (STAGE_W - 120, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
    note = scene_note(state)
    if note and not ((not clean) and dbg.viz_hud):
        cv2.putText(img, note[:72], (12, STAGE_H - 10 - hud_pad), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (134, 124, 114), 1, cv2.LINE_AA)
    if (not clean) and dbg.viz_hud:
        draw_dense_hud(img, state, stage_w=STAGE_W, stage_h=STAGE_H, bottom_pad=hud_pad)

    state["viz_ms"] = (time.perf_counter() - t0) * 1000.0
    state["viz_scene_note"] = note
    if ui.show_nerd and not clean:
        panel = render_panel(state, h=STAGE_H, w=ui.nerd_width, show_help=ui.show_help, ui=ui)
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
        planner={"corridor_width": 2.0, "curvature": 0.01, "target_v": 11.0, "ttc_lead": 2.4, "aeb": "off", "cipv_id": 1},
        show_agent_ghosts=True,
        missing_state_keys=["live cameras", "real planner path", "occupancy grid"],
    )
    st["ego"] = {"speed_mps": 14.0, "steer_deg": 0.0, "throttle": 0.1, "brake": 0.0, "yaw_rate": 0.0}
    authored_path = [{"x": 0.12 * math.sin(i / 14), "y": float(i), "z": 0.0} for i in range(0, 45)]
    st["path_ego"] = authored_path
    st["viz_smoke"] = True
    if use_perception:
        from python.perception.pipeline import ModularPerception

        perc = ModularPerception(allow_synthetic=True)
        fake = np.zeros((480, 640, 3), dtype=np.uint8)
        pout = perc.tick(fake, ego_speed_mps=12.0, steer_deg=0.0)
        st["tracks"] = pout.tracks
        st["tracks_n"] = pout.tracks_n
        st["objects_n"] = pout.objects_n
        st["lane_conf"] = pout.lane_conf
        st["lanes_bev"] = pout.lanes_bev
        st["infer_ms"] = pout.infer_ms
        st["detector"] = pout.detector_name
        st["signs"] = pout.signs
        miss = [m for m in (st.get("missing_state_keys") or []) if m not in ("tracks", "lanes_bev", "real path_ego from planner")]
        st["missing_state_keys"] = sorted(set(miss + pout.missing))
        # Blank-frame Hough cannot see paint. Keep the authored corridor + slowing/CIPV
        # so --smoke actually demonstrates the lexicon, and say so in missing keys.
        if pout.path_debug_preview or len(pout.path_ego or []) < 8:
            st["path_ego"] = authored_path
            st["path_width"] = 2.0
            st["path_conf"] = 0.9
            st["path_debug_preview"] = True
            st["missing_state_keys"] = sorted(set(
                (st.get("missing_state_keys") or []) + ["live corridor (smoke keeps authored path; Hough saw no paint)"]
            ))
        else:
            st["path_ego"] = pout.path_ego
            st["path_width"] = pout.path_width
            st["path_conf"] = pout.path_conf
            st["path_debug_preview"] = pout.path_debug_preview
        pl = dict(pout.planner or {})
        if pl.get("cipv_id") is None:
            for tr in st.get("tracks") or []:
                if track_cls(tr) == "vehicle" and float(tr.get("y") or 0) > 5:
                    pl["cipv_id"] = tr.get("id")
                    break
            st["missing_state_keys"] = sorted(set(
                (st.get("missing_state_keys") or []) + ["live CIPV (smoke tags nearest synthetic vehicle)"]
            ))
        # Authored slowing so the ribbon shade still reads; blank-frame plan_speed has no lead TTC.
        pl["target_v"] = 11.0
        st["planner"] = pl
        st["ego"]["speed_mps"] = 14.0
    if not st.get("lanes_ext"):
        # --smoke has no camera, so the Hough fit finds nothing. Give the stage
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
    if not st.get("signs"):
        st["signs"] = [
            {"cls": "stop_sign", "x": -5.6, "y": 22.0, "conf": 0.5},
            {"cls": "traffic_light", "x": 1.2, "y": 34.0, "conf": 0.5, "state": "unknown"},
        ]
        st["missing_state_keys"] = sorted(set(
            (st.get("missing_state_keys") or []) + ["live signs (smoke uses synthetic furniture)"]
        ))

    write_state(st)
    # Product smoke: clean cabin with the full lexicon (no nerd chrome)
    ui.layers = {0}
    ui.show_nerd = False
    frame = render_stage(st, ui=ui)
    out = Path("docs/gvd_viz_smoke.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    return out
