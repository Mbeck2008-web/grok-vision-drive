"""GVD VISION cabin stage (OpenCV). Chase 3/4 bird default; BEV via T.

Lexicon (second-screen product viz): void stage, multi-lane fan, ice-blue ego
corridor + stop bar, agent boxes (CIPV / in-path / BRAKE), warm curbs,
sign/light glyphs. Driven from gvd_state.json. Titles stay GVD / VISION.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from python.runtime.debug_opts import (
    CAMS_DROP_HZ,
    CONTROL_ROWS,
    LAYER_ATTR,
    MODEL_CONTROL_ROWS,
    NERD_TAB_IDS,
    VIZ_CONTROL_ROWS,
    DebugOpts,
    control_index,
    model_control_index,
    model_row_at,
    row_at,
    viz_control_index,
    viz_row_at,
)
from python.viz.debug_draw import (
    clamp_front_overexpose,
    cabin_drive_word,
    draw_cam_strip,
    draw_cam_tiles,
    draw_dense_hud,
    draw_frustums,
    draw_lane_polys,
    draw_occupancy,
    draw_pip_boxes,
    draw_planner_cost,
)
from python.runtime.state_io import steer_preview_path_ego
from python.viz.forecast import predict_modes
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
# Fade the tail of whatever polyline exists. Do not invent meters past the last point.
# Furniture fade (cameras.yaml viz.fade_frac) is separate and must not pad this ribbon.
CORRIDOR_FADE_FRAC = 0.40
# CAMS tab 4×2 wall on the stage (nerd panel has its own 2×4).
CAMS_STAGE_BOX = (12, 28, STAGE_W - 24, STAGE_H - 40)
CAMS_STAGE_GRID = (4, 2)


@dataclass
class VizUI:
    show_nerd: bool = True
    show_help: bool = False
    layers: set[int] = field(default_factory=set)
    top_down: bool = False  # default = chase 3/4 bird; T toggles BEV
    nerd_tab: str = "live"  # live | drive | viz | model | cams | keys
    debug: DebugOpts = field(default_factory=DebugOpts)
    debug_sel: int = 0
    viz_sel: int = 0
    model_sel: int = 0
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

    def show_model_tab(self) -> None:
        self.nerd_tab = "model"
        self.show_help = False
        self.show_nerd = True

    def show_cams_tab(self) -> None:
        self.nerd_tab = "cams"
        self.show_help = False
        self.show_nerd = True

    def cycle_tab(self, delta: int = 1) -> None:
        tabs = NERD_TAB_IDS
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
        if key in (ord("m"), ord("M")):
            self.show_model_tab()
            return True
        if key in (ord("a"), ord("A")):
            self.show_cams_tab()
            return True
        if key == ord("["):
            self.cycle_tab(-1)
            return True
        if key in (ord("]"), 9):  # Tab
            self.cycle_tab(1)
            return True
        if self.nerd_tab not in ("drive", "viz", "model") or not self.show_nerd:
            return False
        if self.nerd_tab == "viz":
            n = max(1, len(VIZ_CONTROL_ROWS))
            sel_attr = "viz_sel"
            row_fn = viz_row_at
        elif self.nerd_tab == "model":
            n = max(1, len(MODEL_CONTROL_ROWS))
            sel_attr = "model_sel"
            row_fn = model_row_at
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
            elif self.nerd_tab == "model":
                self.model_sel = model_control_index(i)
                row = model_row_at(self.model_sel)
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


@dataclass(frozen=True)
class DrawRange:
    """Cabin world span. The ice ribbon does not use this."""

    ahead_m: float
    behind_m: float
    fade_frac: float
    ahead_min_m: float
    behind_min_m: float


_VIZ_FILE_CACHE: tuple[dict[str, Any], float] | None = None


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_viz_files() -> tuple[dict[str, Any], float]:
    root = Path(__file__).resolve().parents[2]
    cams: dict[str, Any] = {}
    vram = 11.0
    try:
        import yaml  # type: ignore

        cams = yaml.safe_load((root / "config" / "cameras.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        cams = {}
    try:
        import yaml  # type: ignore

        hw = yaml.safe_load((root / "config" / "hardware.yaml").read_text(encoding="utf-8")) or {}
        dgpu = hw.get("dgpu") if isinstance(hw.get("dgpu"), dict) else {}
        got = _num((dgpu or {}).get("vram_gb"))
        if got is not None:
            vram = got
    except Exception:
        vram = 11.0
    return cams, vram


def _far_of(cameras: dict[str, Any], cid: str, fallback: float) -> float:
    for cam in cameras.get("cameras") or []:
        if isinstance(cam, dict) and str(cam.get("id") or "") == cid:
            got = _num(cam.get("far_m"))
            if got is not None and got > 0:
                return got
    return fallback


def _state_far(state: dict[str, Any], key: str) -> float | None:
    got = _num(state.get(key))
    if got is not None and got > 0:
        return got
    return None


def resolve_draw_range(
    state: dict[str, Any] | None = None,
    *,
    cameras: dict[str, Any] | None = None,
    vram_gb: float | None = None,
) -> DrawRange:
    """Clamp cabin ahead/behind once per call.

    auto uses that camera's far_m. A published state far (main_far_m / rear_far_m)
    is the hitch result and replaces the yaml number. vram_gb >= 16 raises the
    ahead cap to 600 so a stronger card can show main far_m 400–600.
    """
    global _VIZ_FILE_CACHE
    state = state or {}
    if cameras is None or vram_gb is None:
        if _VIZ_FILE_CACHE is None:
            _VIZ_FILE_CACHE = _load_viz_files()
        loaded, file_vram = _VIZ_FILE_CACHE
        if cameras is None:
            cameras = loaded
        if vram_gb is None:
            vram_gb = file_vram
    viz = cameras.get("viz") if isinstance(cameras.get("viz"), dict) else {}
    ahead_min = _num(viz.get("ahead_min_m")) or 80.0
    ahead_max = _num(viz.get("ahead_max_m")) or 400.0
    behind_min = _num(viz.get("behind_min_m")) or 20.0
    behind_max = _num(viz.get("behind_max_m")) or 80.0
    fade = _num(viz.get("fade_frac"))
    if fade is None:
        fade = 0.28
    if float(vram_gb) >= 16.0:
        ahead_max = max(ahead_max, 600.0)
    ahead_raw = viz.get("draw_ahead_m", "auto")
    behind_raw = viz.get("draw_behind_m", "auto")
    ahead_auto = isinstance(ahead_raw, str) and ahead_raw.strip().lower() == "auto"
    behind_auto = isinstance(behind_raw, str) and behind_raw.strip().lower() == "auto"
    if ahead_auto:
        ahead_src = _state_far(state, "main_far_m")
        if ahead_src is None:
            ahead_src = _far_of(cameras, "main", 300.0)
    else:
        ahead_src = _num(ahead_raw) or _far_of(cameras, "main", 300.0)
    if behind_auto:
        behind_src = _state_far(state, "rear_far_m")
        if behind_src is None:
            behind_src = _far_of(cameras, "rear", 100.0)
    else:
        behind_src = _num(behind_raw) or _far_of(cameras, "rear", 100.0)
    if ahead_max < ahead_min:
        ahead_max = ahead_min
    if behind_max < behind_min:
        behind_max = behind_min
    return DrawRange(
        ahead_m=min(ahead_max, max(ahead_min, float(ahead_src))),
        behind_m=min(behind_max, max(behind_min, float(behind_src))),
        fade_frac=max(0.0, min(0.95, float(fade))),
        ahead_min_m=float(ahead_min),
        behind_min_m=float(behind_min),
    )


@dataclass
class Cam:
    top_down: bool = False
    ppm: float = 10.0
    ahead_m: float = -1.0
    behind_m: float = -1.0
    fade_frac: float = -1.0

    def __post_init__(self) -> None:
        if self.ahead_m < 0 or self.behind_m < 0 or self.fade_frac < 0:
            draw = resolve_draw_range({})
            if self.ahead_m < 0:
                self.ahead_m = draw.ahead_m
            if self.behind_m < 0:
                self.behind_m = draw.behind_m
            if self.fade_frac < 0:
                self.fade_frac = draw.fade_frac

    @property
    def eye(self) -> tuple[float, float, float]:
        if self.top_down:
            return (0.0, -8.0, 40.0)
        # Chase 3/4, just behind the ego. Draw range is longer; the lens stays here.
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


def is_bus_mismatch(state: dict[str, Any]) -> bool:
    link = str(state.get("bus_link") or state.get("link") or "").strip().lower()
    return link == "mismatch"


def _draw_mismatch(img: np.ndarray, state: dict[str, Any]) -> None:
    """Loud overlay: do not invent a corridor when the bus folder is wrong."""
    note = str(state.get("bus_note") or "python_bus and lua_bus are not the same folder")
    py = str(state.get("python_bus") or "--")
    lua_b = str(state.get("lua_bus") or "--")
    prod = str(state.get("product") or "--")
    cv2.rectangle(img, (0, 0), (STAGE_W, STAGE_H), VOID, -1)
    cv2.putText(img, "GVD", (12, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ICE, 1, cv2.LINE_AA)
    cv2.putText(img, "VISION", (52, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(img, "BUS MISMATCH", (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (80, 80, 210), 3, cv2.LINE_AA)
    cv2.putText(img, "refuse actuation  -  not a fake corridor", (80, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PAPER, 1, cv2.LINE_AA)
    cv2.putText(img, "product=" + prod, (80, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ICE, 1, cv2.LINE_AA)
    cv2.putText(img, "python_bus=" + py[:72], (80, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.48, PAPER, 1, cv2.LINE_AA)
    cv2.putText(img, "lua_bus=" + lua_b[:72], (80, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.48, PAPER, 1, cv2.LINE_AA)
    cv2.putText(img, note[:88], (80, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 140, 210), 1, cv2.LINE_AA)
    cv2.putText(
        img,
        "Drive vs Tech, leftover GVD_DOCS_DIR, or gvd_*.json under current\\",
        (80, 520),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (134, 124, 114),
        1,
        cv2.LINE_AA,
    )


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
    """Under 8 Hz: drop fans, signs, PIP blit, camera strip, CAMS grid blit (tiles stay labelled)."""
    try:
        hz = float(state.get("loop_hz") or 0.0)
    except (TypeError, ValueError):
        return False
    return hz > 0 and hz < CAMS_DROP_HZ


def _allow_stub(state: dict[str, Any]) -> bool:
    return bool(state.get("viz_smoke"))


def _furniture_alpha(cam: Cam, y: float) -> float:
    """Full strength until the last fade_frac of AHEAD. Behind the ego stays full."""
    if y > cam.ahead_m + 1e-3 or y < -cam.behind_m - 1e-3:
        return 0.0
    frac = cam.fade_frac
    if frac <= 0.0:
        return 1.0
    start = cam.ahead_m * (1.0 - frac)
    if y <= start:
        return 1.0
    span = max(1e-3, cam.ahead_m - start)
    return max(0.15, 1.0 - (y - start) / span)


def _clip_poly_y(pts: list[dict[str, float]], y_lo: float, y_hi: float) -> list[dict[str, float]]:
    """Keep the polyline inside [y_lo, y_hi], interpolating the cut."""
    if not pts:
        return []
    ordered = sorted(pts, key=lambda p: (p["y"], p["x"]))
    if len(ordered) == 1:
        p = ordered[0]
        return [p] if y_lo <= p["y"] <= y_hi else []

    def _at(a: dict[str, float], b: dict[str, float], y: float) -> dict[str, float]:
        dy = b["y"] - a["y"]
        t = 0.0 if abs(dy) < 1e-9 else (y - a["y"]) / dy
        t = max(0.0, min(1.0, t))
        return {"x": a["x"] + (b["x"] - a["x"]) * t, "y": y}

    out: list[dict[str, float]] = []
    for i in range(len(ordered) - 1):
        a, b = ordered[i], ordered[i + 1]
        lo, hi = (a, b) if a["y"] <= b["y"] else (b, a)
        if hi["y"] < y_lo or lo["y"] > y_hi:
            continue
        y0 = max(y_lo, lo["y"])
        y1 = min(y_hi, hi["y"])
        p0 = lo if abs(lo["y"] - y0) < 1e-6 else _at(lo, hi, y0)
        p1 = hi if abs(hi["y"] - y1) < 1e-6 else _at(lo, hi, y1)
        if not out or abs(out[-1]["y"] - p0["y"]) > 1e-3 or abs(out[-1]["x"] - p0["x"]) > 1e-3:
            out.append(p0)
        if abs(p1["y"] - p0["y"]) > 1e-3 or abs(p1["x"] - p0["x"]) > 1e-3:
            out.append(p1)
    return out


def _extend_predicted(
    pts: list[dict[str, float]], y_lo: float, y_hi: float,
) -> list[list[dict[str, float]]]:
    """Dashed continuation of a short detected poly. Does not repeat one point as a wall."""
    if len(pts) < 2:
        return []
    pieces: list[list[dict[str, float]]] = []
    y0, y1 = pts[0]["y"], pts[-1]["y"]
    if y1 < y_hi - 0.5:
        a, b = pts[-2], pts[-1]
        dy = b["y"] - a["y"]
        slope = 0.0 if abs(dy) < 1e-6 else (b["x"] - a["x"]) / dy
        ext = [dict(b)]
        y, x = b["y"], b["x"]
        while y < y_hi - 1e-3:
            step = min(8.0, y_hi - y)
            y += step
            x += slope * step
            ext.append({"x": x, "y": y})
        if len(ext) >= 2:
            pieces.append(ext)
    if y0 > y_lo + 0.5:
        a, b = pts[0], pts[1]
        dy = b["y"] - a["y"]
        slope = 0.0 if abs(dy) < 1e-6 else (b["x"] - a["x"]) / dy
        ext = [dict(a)]
        y, x = a["y"], a["x"]
        while y > y_lo + 1e-3:
            step = min(8.0, y - y_lo)
            y -= step
            x -= slope * step
            ext.append({"x": x, "y": y})
        ext.reverse()
        if len(ext) >= 2:
            pieces.append(ext)
    return pieces


def _stroke_world(
    img: np.ndarray,
    cam: Cam,
    pts: list[dict[str, float]],
    color: tuple[int, int, int],
    thickness: int,
    *,
    dashed: bool = False,
    dash: int = 8,
    gap: int = 7,
    z: float = 0.02,
) -> None:
    """Stroke a world polyline, fading only in the last fade_frac of AHEAD."""
    if len(pts) < 2:
        return
    fade_y = cam.ahead_m * (1.0 - cam.fade_frac) if cam.fade_frac > 0 else cam.ahead_m
    runs: list[tuple[float, list[dict[str, float]]]] = []
    run_a: float | None = None
    run: list[dict[str, float]] = []
    for p in pts:
        a = 1.0 if p["y"] <= fade_y else round(_furniture_alpha(cam, p["y"]) * 5.0) / 5.0
        if run_a is None:
            run_a, run = a, [p]
            continue
        if abs(a - run_a) < 0.04:
            run.append(p)
            continue
        runs.append((run_a, run))
        run_a, run = a, [run[-1], p]
    if run_a is not None and len(run) >= 2:
        runs.append((run_a, run))
    for alpha, piece in runs:
        _stroke_poly(
            img, _proj_poly(cam, piece, z), _mix(color, alpha), thickness,
            dashed=dashed, dash=dash, gap=gap,
        )


def _draw_y_lo(cam: Cam) -> float:
    """Near end of the draw span that this camera can actually see.

    Chase sits just behind the ego, so points behind the lens are not stroked.
    Top-down still paints the full behind span.
    """
    span = -float(cam.behind_m)
    if cam.top_down:
        return span
    return max(span, float(cam.eye[1]) + 1.5)


def _draw_ground(img: np.ndarray, cam: Cam) -> None:
    y0, y1 = _draw_y_lo(cam), cam.ahead_m
    far_l, far_r = cam.project(-16, y1, 0), cam.project(16, y1, 0)
    near_l, near_r = cam.project(-16, y0, 0), cam.project(16, y0, 0)
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.array([far_l, far_r, near_r, near_l], dtype=np.int32)], (22, 18, 15))
    cv2.addWeighted(overlay, 0.40, img, 0.60, 0, img)
    tick = _mix(ICE, 0.08)
    y = math.ceil((y0 - 1e-6) / 20.0) * 20.0
    while y <= y1 + 1e-6:
        a, b = cam.project(-7, y, 0), cam.project(7, y, 0)
        cv2.line(img, a, b, tick, 1, cv2.LINE_AA)
        y += 20.0


def _draw_fog(img: np.ndarray, cam: Cam) -> None:
    """Void the sky plus a thin horizon fade — horizon is the draw-ahead range."""
    horizon = max(0, min(STAGE_H - 1, cam.project(0, cam.ahead_m, 0)[1]))
    sky = img.copy()
    cv2.rectangle(sky, (0, 0), (STAGE_W, max(1, horizon)), VOID, -1)
    cv2.addWeighted(sky, 0.55, img, 0.45, 0, img)
    fade = img.copy()
    cv2.rectangle(fade, (0, max(0, horizon)), (STAGE_W, min(STAGE_H, horizon + 28)), VOID, -1)
    cv2.addWeighted(fade, 0.22, img, 0.78, 0, img)


def _lane_color(mode: str, fade: float) -> tuple[tuple[int, int, int], int, bool]:
    if mode == "solid":
        color = tuple(max(90, int(c * max(0.55, fade))) for c in PAPER)
        return color, 2, False
    color = tuple(max(70, int(c * max(0.40, 0.70 * fade))) for c in (170, 166, 160))
    return color, 2, True


def _draw_lanes(img: np.ndarray, lanes: list, cam: Cam, *, smoke: bool) -> None:
    y_lo, y_hi = _draw_y_lo(cam), cam.ahead_m
    for ln in lanes or []:
        kind = ln.get("kind") if isinstance(ln, dict) else None
        mode = lane_draw_mode(kind, smoke=smoke)
        if mode is None:
            continue
        pts = _clip_poly_y(_poly_points(ln), y_lo, y_hi)
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
        color, thick, dash_default = _lane_color(mode, fade)
        _stroke_world(
            img, cam, pts, color, thick,
            dashed=dashed or dash_default, dash=10 if mode == "solid" and not dashed else 8, gap=5,
        )
        # Short live paint is extended as predicted dashes. The source poly is not relabeled.
        ext_color, ext_thick, _ = _lane_color("dashed", fade * 0.85)
        for ext in _extend_predicted(pts, y_lo, y_hi):
            _stroke_world(img, cam, ext, ext_color, ext_thick, dashed=True, dash=8, gap=6)


def _draw_edge_piece(
    img: np.ndarray, cam: Cam, pts: list[dict[str, float]], *, solid: bool, alpha: float,
) -> None:
    if len(pts) < 2 or alpha <= 0.02:
        return
    base = _proj_poly(cam, pts, 0.0)
    top = _proj_poly(cam, pts, 0.13)
    quad = np.array(base + list(reversed(top)), dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [quad], KERB)
    w = (0.22 if solid else 0.14) * alpha
    cv2.addWeighted(overlay, w, img, 1.0 - w, 0, img)
    _stroke_poly(
        img, top, _mix(KERB, (0.85 if solid else 0.62) * alpha, (22, 20, 18)), 2,
        dashed=not solid, dash=6, gap=5,
    )


def _draw_edges(img: np.ndarray, edges: list, cam: Cam) -> None:
    y_lo, y_hi = _draw_y_lo(cam), cam.ahead_m
    fade_y = cam.ahead_m * (1.0 - cam.fade_frac) if cam.fade_frac > 0 else cam.ahead_m
    for e in edges or []:
        raw = _clip_poly_y(_poly_points(e), y_lo, y_hi)
        if len(raw) < 2:
            continue
        kind = e.get("kind") if isinstance(e, dict) else "predicted"
        solid = str(kind or "predicted") == "detected"
        near = [p for p in raw if p["y"] <= fade_y + 1e-6]
        far = [p for p in raw if p["y"] >= fade_y - 1e-6]
        if len(near) >= 2:
            _draw_edge_piece(img, cam, near, solid=solid, alpha=1.0)
        if len(far) >= 2 and far[-1]["y"] > fade_y + 0.5:
            mid = 0.5 * (far[0]["y"] + far[-1]["y"])
            _draw_edge_piece(img, cam, far, solid=solid, alpha=_furniture_alpha(cam, mid))
        for ext in _extend_predicted(raw, y_lo, y_hi):
            ext_near = [p for p in ext if p["y"] <= fade_y + 1e-6]
            ext_far = [p for p in ext if p["y"] >= fade_y - 1e-6]
            if len(ext_near) >= 2:
                _draw_edge_piece(img, cam, ext_near, solid=False, alpha=0.85)
            if len(ext_far) >= 2 and ext_far[-1]["y"] > fade_y + 0.5:
                mid = 0.5 * (ext_far[0]["y"] + ext_far[-1]["y"])
                _draw_edge_piece(
                    img, cam, ext_far, solid=False,
                    alpha=0.85 * _furniture_alpha(cam, mid),
                )


def _draw_signs(img: np.ndarray, signs: list, cam: Cam) -> None:
    if not signs:
        return
    ordered = sorted(signs, key=lambda s: float(s.get("y") or 0), reverse=True)
    for s in ordered:
        cls = str(s.get("cls") or s.get("class") or "sign")
        x, y = float(s.get("x") or 0), float(s.get("y") or 0)
        if y > cam.ahead_m or y < -cam.behind_m:
            continue
        light = cls == "traffic_light"
        pole = cls == "pole"
        h = 3.2 if light else (1.1 if pole else 2.1)
        far = _furniture_alpha(cam, y)
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


def _as_path(raw: Any) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for p in raw or []:
        if isinstance(p, dict):
            out.append({
                "x": float(p.get("x") or 0.0),
                "y": float(p.get("y") or 0.0),
                "z": float(p.get("z") or 0.0),
            })
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            z = float(p[2]) if len(p) > 2 else 0.0
            out.append({"x": float(p[0]), "y": float(p[1]), "z": z})
    return out


def _arc_length(path: list) -> float:
    pts = _as_path(path)
    dist = 0.0
    prev: tuple[float, float] | None = None
    for p in pts:
        if prev is not None:
            dist += math.hypot(p["x"] - prev[0], p["y"] - prev[1])
        prev = (p["x"], p["y"])
    return dist


def corridor_half_width(state: dict[str, Any]) -> float:
    """Half of the planner corridor, in meters. No pixel floor and no lane-fan inflate."""
    raw = state.get("path_width")
    if raw is None:
        planner = state.get("planner") if isinstance(state.get("planner"), dict) else {}
        raw = planner.get("corridor_width")
    try:
        width = float(raw)
    except (TypeError, ValueError):
        width = 2.0
    if width <= 0.0:
        width = 2.0
    return width * 0.5


def _e2e_live(state: dict[str, Any]) -> bool:
    """E2E is the driver only when the net is a real onnx and the modular veto is clear."""
    policy = str(state.get("policy") or "modular").lower()
    backend = str(state.get("e2e_backend") or "stub").lower()
    veto = str(state.get("veto_reason") or "none").lower()
    return policy == "e2e" and backend not in ("", "stub") and veto in ("none", "")


def _e2e_corridor(state: dict[str, Any], modular: list[dict[str, float]]) -> list[dict[str, float]]:
    """Exported E2E polyline, or the same steer integrator as steer_preview_path_ego.

    Length follows the modular path that exists. A missing export does not become a 36 m preview.
    """
    exported = _as_path(state.get("path_e2e"))
    if len(exported) >= 2:
        return exported
    shadow = state.get("shadow") if isinstance(state.get("shadow"), dict) else {}
    if shadow.get("steer") is None:
        return []
    length = _arc_length(modular)
    if length < 1.0:
        return []
    try:
        steer = float(shadow["steer"])
    except (TypeError, ValueError):
        return []
    return steer_preview_path_ego(steer * 30.0, length_m=length, step=1.0)


def resolve_corridors(state: dict[str, Any]) -> tuple[list[dict[str, float]], list[dict[str, float]] | None]:
    """Primary ribbon plus an optional shadow ghost.

    Modular, or e2e/shadow held by a modular veto (including e2e_stub): the planner path.
    policy=e2e with a live onnx and veto none: that net's path, not the modular polyline.
    policy=shadow: primary is the modular command that actuates; ghost is the other path.
    """
    modular = _as_path(state.get("path_ego"))
    policy = str(state.get("policy") or "modular").lower()
    backend = str(state.get("e2e_backend") or "stub").lower()
    if _e2e_live(state):
        return _e2e_corridor(state, modular), None
    ghost: list[dict[str, float]] | None = None
    if policy == "shadow" and backend not in ("", "stub"):
        other = _e2e_corridor(state, modular)
        if len(other) >= 2:
            ghost = other
    return modular, ghost


def _draw_filled_corridor(
    img: np.ndarray,
    path: list[dict],
    conf: float,
    cam: Cam,
    *,
    half_w: float,
    intent: float,
    preview: bool,
    gain: float = 1.0,
) -> None:
    if len(path) < 2 or half_w <= 0.0:
        return
    trimmed: list[tuple[float, float, float]] = []
    dist = 0.0
    prev = None
    for p in path:
        x, y = float(p.get("x", 0)), float(p.get("y", 0))
        if prev is not None:
            dist += math.hypot(x - prev[0], y - prev[1])
        trimmed.append((x, y, dist if prev is not None else 0.0))
        prev = (x, y)
    if len(trimmed) < 2:
        return
    total = trimmed[-1][2]
    fade_from = total * (1.0 - CORRIDOR_FADE_FRAC)

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
        if total <= 1e-3 or d <= fade_from:
            fade = 1.0
        else:
            fade = max(0.0, 1.0 - (d - fade_from) / max(1e-3, total - fade_from))
        cx, cy = cam.project(x, y, 0.03)
        lx, ly = cam.project(x - px, y - py, 0.03)
        rx, ry = cam.project(x + px, y + py, 0.03)
        if abs(rx - lx) < 1 and abs(ry - ly) < 1:
            half_px = half_w * cam.scale_at(y)
            lx, ly = int(round(cx - half_px)), cy
            rx, ry = int(round(cx + half_px)), cy
        left.append((lx, ly))
        right.append((rx, ry))
        center.append((cx, cy))
        alphas.append(fade)

    nseg = len(left) - 1
    for i in range(nseg):
        near = 1.0 - i / max(1, nseg)
        a = gain * intent * (0.38 + 0.50 * near) * max(0.25, conf) * (0.55 * alphas[i] + 0.45 * alphas[i + 1])
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
        y_track = float(t.get("y") or 0)
        if y_track > cam.ahead_m or y_track < -cam.behind_m:
            continue
        far = _furniture_alpha(cam, y_track)
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


def _pt_xy(p: Any) -> tuple[float, float]:
    if isinstance(p, dict):
        return float(p.get("x") or 0.0), float(p.get("y") or 0.0)
    return float(p[0]), float(p[1])


def _fan_ahead(tr: dict[str, Any] | None, points: list) -> list:
    """Drop samples still on the box so a fan never stamps the face.

    The first kept point is at least half a length in front of the track origin.
    """
    if not points:
        return []
    src = tr if isinstance(tr, dict) else {}
    length, _, _ = _track_dims(src, track_cls(src))
    half = max(0.5, float(length) * 0.5)
    if src.get("x") is not None or src.get("y") is not None:
        x0 = float(src.get("x") or 0.0)
        y0 = float(src.get("y") or 0.0)
    else:
        x0, y0 = _pt_xy(points[0])
    yaw = None
    if src.get("yaw") is not None:
        try:
            yaw = float(src["yaw"])
        except (TypeError, ValueError):
            yaw = None
    if yaw is None and len(points) >= 2:
        ax, ay = _pt_xy(points[0])
        bx, by = _pt_xy(points[1])
        if abs(bx - ax) + abs(by - ay) > 1e-4:
            yaw = math.atan2(by - ay, bx - ax)
    if yaw is None:
        yaw = math.pi / 2
    fx, fy = math.cos(yaw), math.sin(yaw)
    ahead = []
    for p in points:
        px, py = _pt_xy(p)
        if (px - x0) * fx + (py - y0) * fy >= half - 1e-6:
            ahead.append(p)
    return ahead


def _draw_agent_forecasts(img: np.ndarray, tr: dict[str, Any], cam: Cam) -> None:
    modes = predict_modes(tr)
    if not modes:
        return
    # Thin ground line only. No disc, halo, or centroid mark on the box face.
    m0 = _fan_ahead(tr, modes[0]["points"])
    for i in range(len(m0) - 1):
        a = cam.project(m0[i][0], m0[i][1], 0.04)
        b = cam.project(m0[i + 1][0], m0[i + 1][1], 0.04)
        cv2.line(img, a, b, ICE, 1, cv2.LINE_AA)
    for m in modes[1:]:
        pts = [cam.project(p[0], p[1], 0.04) for p in _fan_ahead(tr, m["points"])]
        for i in range(0, len(pts) - 1, 2):
            cv2.line(img, pts[i], pts[min(i + 1, len(pts) - 1)], (100, 100, 100), 1, cv2.LINE_AA)


def _draw_state_fans(img: np.ndarray, agents: list, cam: Cam) -> None:
    """Mode-0 fans already written into state.agents (same toy math as predict_modes)."""
    for ag in agents or []:
        raw = ag.get("path_ego") if isinstance(ag, dict) else ag
        pts = _fan_ahead(ag if isinstance(ag, dict) else {}, _poly_points(raw))
        if len(pts) < 2:
            continue
        screen = _proj_poly(cam, pts, 0.04)
        _stroke_poly(img, screen, _mix(ICE, 0.30), 1)


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
    draw = resolve_draw_range(state)
    cam = Cam(
        top_down=ui.top_down,
        ahead_m=draw.ahead_m,
        behind_m=draw.behind_m,
        fade_frac=draw.fade_frac,
    )
    img = np.full((STAGE_H, STAGE_W, 3), VOID, dtype=np.uint8)

    if is_bus_mismatch(state):
        _draw_mismatch(img, state)
        note = str(state.get("bus_note") or "MISMATCH")
        state["viz_ms"] = (time.perf_counter() - t0) * 1000.0
        state["viz_scene_note"] = note
        if ui.show_nerd and not clean:
            panel = render_panel(
                state, h=STAGE_H, w=ui.nerd_width, show_help=ui.show_help, ui=ui, cam_frames=cam_frames,
            )
            return np.concatenate([img, panel], axis=1)
        return img

    drop_heavy = _drop_heavy(state)
    health = state.get("cam_health") if isinstance(state.get("cam_health"), dict) else None
    cams_tab = (not clean) and ui.nerd_tab == "cams"
    if cams_tab:
        draw_cam_tiles(
            img,
            cam_frames,
            health,
            x0=CAMS_STAGE_BOX[0],
            y0=CAMS_STAGE_BOX[1],
            width=CAMS_STAGE_BOX[2],
            height=CAMS_STAGE_BOX[3],
            cols=CAMS_STAGE_GRID[0],
            rows=CAMS_STAGE_GRID[1],
            dropped=drop_heavy,
        )
        cv2.rectangle(img, (0, 0), (STAGE_W, 22), (12, 13, 16), -1)
        cv2.putText(img, "GVD", (12, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ICE, 1, cv2.LINE_AA)
        cv2.putText(img, "VISION", (52, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
        word, word_col = cabin_drive_word(state)
        cv2.putText(img, word, (118, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, word_col, 1, cv2.LINE_AA)
        cam_lbl = "CAMS dropped" if drop_heavy else "CAMS 8-view"
        cv2.putText(img, cam_lbl, (STAGE_W - 150, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
        note = scene_note(state)
        state["viz_ms"] = (time.perf_counter() - t0) * 1000.0
        state["viz_scene_note"] = note
        if ui.show_nerd:
            panel = render_panel(
                state, h=STAGE_H, w=ui.nerd_width, show_help=ui.show_help, ui=ui, cam_frames=cam_frames,
            )
            return np.concatenate([img, panel], axis=1)
        return img

    engaged = bool(state.get("engaged"))
    smoke = _allow_stub(state)
    primary, ghost = resolve_corridors(state)
    path = primary
    half_w = corridor_half_width(state)
    try:
        path_width = float(state.get("path_width") or (half_w * 2.0))
    except (TypeError, ValueError):
        path_width = half_w * 2.0
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
        intent = pace_scale(state)
        # Preview stays dimmer. A shadow ghost is thinner and only off the clean cabin.
        if ghost and not clean:
            _draw_filled_corridor(
                img, ghost, path_conf, cam,
                half_w=half_w * 0.62,
                intent=intent * 0.40,
                preview=False,
                gain=0.55 if preview else 1.0,
            )
        _draw_filled_corridor(
            img, path, path_conf, cam,
            half_w=half_w,
            intent=intent,
            preview=preview,
            gain=0.62 if preview else 1.0,
        )
        _draw_stop_bar(img, path, half_w, all_tracks, cipv_id, cam, is_halted(state))
        _draw_path_world_overlay(img, state.get("path_world") or [], cam)

    if (not clean) and dbg.viz_cost:
        draw_planner_cost(img, cam, path, path_width, all_tracks)

    if not drop_heavy and dbg.viz_forecast:
        # Fans go down before the boxes. A ground line ahead of a car still
        # projects inside the chase silhouette, and a disc there reads as a pupil.
        if show_ghosts:
            n_forecast = 0
            for tr in tracks:
                if n_forecast >= MAX_FORECAST:
                    break
                _draw_agent_forecasts(img, tr, cam)
                n_forecast += 1
        _draw_state_fans(img, state.get("agents") or [], cam)

    if show_ghosts:
        _draw_tracks(
            img, tracks, cam, path=path, path_width=path_width, state=state, cipv_id=cipv_id,
            show_ids=(not clean) and dbg.viz_ids,
            show_vel=(not clean) and dbg.viz_vel,
        )

    if not drop_heavy and dbg.viz_signs:
        _draw_signs(img, state.get("signs") or [], cam)

    if (not clean) and dbg.viz_frustums:
        draw_frustums(img, cam, state.get("cam_health") if isinstance(state.get("cam_health"), dict) else None)

    _draw_fog(img, cam)
    _draw_ego(img, cam, engaged)

    if main_frame is not None and getattr(main_frame, "size", 0) and not drop_heavy and dbg.viz_pip:
        pip = np.full((180, 320, 3), (28, 28, 28), dtype=np.uint8)
        pip_src = clamp_front_overexpose(main_frame, "main")
        try:
            pip = cv2.resize(pip_src if pip_src is not None else main_frame, (320, 180), interpolation=cv2.INTER_AREA)
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

    if (not clean) and dbg.viz_cams:
        # Under 8 Hz keep labelled slots; skip the blit inside draw_cam_tiles.
        strip_top = draw_cam_strip(
            img, cam_frames, health,
            stage_w=STAGE_W, stage_h=STAGE_H, dropped=drop_heavy,
        )
        hud_pad = STAGE_H - strip_top + 6
    else:
        hud_pad = 0

    cv2.rectangle(img, (0, 0), (STAGE_W, 22), (12, 13, 16), -1)
    cv2.putText(img, "GVD", (12, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ICE, 1, cv2.LINE_AA)
    cv2.putText(img, "VISION", (52, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    word, word_col = cabin_drive_word(state)
    cv2.putText(img, word, (118, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, word_col, 1, cv2.LINE_AA)
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
        panel = render_panel(
            state, h=STAGE_H, w=ui.nerd_width, show_help=ui.show_help, ui=ui, cam_frames=cam_frames,
        )
        return np.concatenate([img, panel], axis=1)
    return img


def _span_ys(y0: float, y1: float, step: float = 3.0) -> list[float]:
    """Inclusive samples from y0 to y1. Smoke authors lanes across the cabin span."""
    ys: list[float] = []
    y = float(y0)
    end = float(y1)
    step = float(step) if step > 0 else 3.0
    if end < y:
        return [round(y, 2), round(end, 2)]
    while y < end - 1e-6 and len(ys) < 4000:
        ys.append(round(y, 2))
        y += step
    ys.append(round(end, 2))
    return ys


def smoke(
    ui: VizUI | None = None,
    use_perception: bool = False,
    *,
    engaged: bool = True,
    out: "Path | None" = None,
    write_bus: bool = True,
) -> "Path":
    """Render the product smoke cabin.

    Default path is ``docs/gvd_viz_smoke.png`` (``--smoke``). README media
    calls this with ``engaged=False`` and a ``docs/media/`` dest so the
    parked cabin is synthetic and un-engaged (no ice underglow).
    """
    from pathlib import Path

    from python.runtime.state_io import default_state, write_state

    ui = ui or VizUI()
    st = default_state(
        engaged=bool(engaged),
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
    # Smoke furniture only. Live ticks keep the model path and do not pad it to this span.
    authored_path = [{"x": 0.12 * math.sin(i / 14), "y": float(i), "z": 0.0} for i in range(0, 61)]
    st["path_ego"] = authored_path
    st["viz_smoke"] = True
    if use_perception:
        from python.perception.pipeline import ModularPerception

        perc = ModularPerception(allow_synthetic=True, detector_id="synthetic")
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
        # Author the full cabin span. Live Hough stays short; the drawer may dash-extend it.
        span_draw = resolve_draw_range(st)
        span = _span_ys(-span_draw.behind_m, span_draw.ahead_m, 3.0)

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

    if write_bus:
        write_state(st)
    # Product smoke: clean cabin with the full lexicon (no nerd chrome)
    ui.layers = {0}
    ui.show_nerd = False
    frame = render_stage(st, ui=ui)
    dest = Path(out) if out is not None else Path("docs/gvd_viz_smoke.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), frame)
    return dest
