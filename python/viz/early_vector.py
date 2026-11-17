"""Vector chase stage: black field, dotted lanes, wire boxes.

Drawers stay separate (lanes, edges, path, boxes, signals, ego, hud, overlay)
so a debug board or a showcase frame can call the same pieces. CPU, OpenCV, numpy.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import cv2
import numpy as np

from python.viz.debug_draw import cabin_drive_word
from python.viz.stage import (
    Cam,
    EGO_WIDTH_M,
    _clip_poly_y,
    _lane_span,
    _poly_points,
    _ribbon_display_gain,
    _scene_ckpt_ribbon_path,
    cipv_id_of,
    drawn_lane_records,
    is_lead,
    resolve_corridors,
    resolve_draw_range,
    track_cls,
)

# World spacing. Screen gaps shrink because the chase lens compresses distance.
GROUND_Z = 0.05
LANE_STEP_M = 2.0
PATH_STEP_M = 2.0
EDGE_STEP_M = 1.6
DOT_RADIUS_M = 0.15
DOT_RADIUS_PX_MAX = 6
DRAW_FAR_M = 90.0
CENTER_Z_FRAC = 0.55
LIGHT_Z = 4.8
SIGN_Z = 2.6
EARLY_EYE = (0.0, -8.4, 3.15)
EARLY_LOOK = (0.0, 22.0, 0.15)
EARLY_FOV = 46.0
EGO_GLOW_RADII = (150, 96, 54)

BG = (0, 0, 0)
LANE_WHITE = (245, 245, 245)
EDGE_RED = (40, 40, 255)
LANE_PURPLE = (220, 40, 210)
PATH_BGR = (255, 190, 70)
LEAD_BGR = (70, 255, 80)
CAR_BGR = (255, 100, 40)
TRUCK_BGR = (20, 145, 255)
PED_BGR = (255, 40, 210)
BIKE_BGR = (40, 230, 255)
OTHER_BGR = (40, 220, 255)
OVERLAY_BOX = (30, 150, 255)
OVERLAY_CURB = (40, 90, 255)
HUD = (235, 235, 235)
HUD_DIM = (140, 140, 140)

_PRIORS: dict[str, tuple[float, float, float]] = {
    "pedestrian": (0.6, 0.6, 1.75),
    "ped": (0.6, 0.6, 1.75),
    "person": (0.6, 0.6, 1.75),
    "bicycle": (1.8, 0.6, 1.5),
    "bike": (1.8, 0.6, 1.5),
    "motorcycle": (2.1, 0.7, 1.4),
    "truck": (7.6, 2.5, 3.0),
    "bus": (10.5, 2.6, 3.2),
    "trailer": (12.0, 2.5, 3.4),
    "obstacle": (4.4, 1.9, 1.6),
    "debris": (1.2, 0.8, 0.6),
    "cone": (0.4, 0.4, 0.75),
    "vehicle": (4.5, 1.85, 1.5),
    "car": (4.5, 1.85, 1.5),
}

Project = Callable[[float, float, float], tuple[int, int] | None]


def make_early_cam(
    width: int,
    height: int,
    top_down: bool = False,
    state: dict[str, Any] | None = None,
) -> Cam:
    draw = resolve_draw_range(state or {})
    if top_down:
        return Cam(
            top_down=True,
            ppm=8.0,
            ahead_m=draw.ahead_m,
            behind_m=draw.behind_m,
            fade_frac=draw.fade_frac,
            width=int(width),
            height=int(height),
        )
    return Cam(
        top_down=False,
        ahead_m=draw.ahead_m,
        behind_m=draw.behind_m,
        fade_frac=draw.fade_frac,
        width=int(width),
        height=int(height),
        eye_override=EARLY_EYE,
        look_override=EARLY_LOOK,
        fov_deg=EARLY_FOV,
    )


def walk_dots(pts: Any, step_m: float) -> list[tuple[float, float]]:
    """Points spaced by arc length. The first sample is the first vertex."""
    seq: list[tuple[float, float]] = []
    for p in pts or []:
        if isinstance(p, dict):
            seq.append((float(p.get("x") or 0.0), float(p.get("y") or 0.0)))
        else:
            try:
                seq.append((float(p[0]), float(p[1])))
            except (TypeError, IndexError, ValueError):
                continue
    if not seq:
        return []
    step = float(step_m) if float(step_m) > 1e-3 else 1.0
    out = [seq[0]]
    remain = step
    for (ax, ay), (bx, by) in zip(seq, seq[1:]):
        dx, dy = bx - ax, by - ay
        seg = math.hypot(dx, dy)
        if seg < 1e-8:
            continue
        ux, uy = dx / seg, dy / seg
        travelled = 0.0
        while travelled + remain <= seg + 1e-8:
            travelled += remain
            out.append((ax + ux * travelled, ay + uy * travelled))
            remain = step
        remain -= seg - travelled
        if remain < 1e-6:
            remain = step
    return out


def class_dims(tr: dict[str, Any] | None) -> tuple[float, float, float]:
    src = tr if isinstance(tr, dict) else {}
    cls = track_cls(src)
    length, width, height = _PRIORS.get(cls, _PRIORS["vehicle"])
    for key, cur in (("length", length), ("width", width), ("height", height)):
        raw = src.get(key)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val > 0.2:
            if key == "length":
                length = val
            elif key == "width":
                width = val
            else:
                height = val
    return float(length), float(width), float(height)


def role_bgr(tr: dict[str, Any], cipv_id: int | None) -> tuple[int, int, int]:
    """Lead is green. Other colours follow class, then size."""
    if is_lead(tr, cipv_id):
        return LEAD_BGR
    cls = track_cls(tr)
    length, _w, height = class_dims(tr)
    if cls in ("pedestrian", "ped", "person"):
        return PED_BGR
    if cls in ("bicycle", "bike", "motorcycle", "cyclist"):
        return BIKE_BGR
    if cls in ("truck", "bus", "trailer", "obstacle", "debris", "cone") or length >= 6.2 or height >= 2.4:
        return TRUCK_BGR
    if cls in ("vehicle", "car"):
        return CAR_BGR
    return OTHER_BGR


def lane_bgr(record: dict[str, Any]) -> tuple[int, int, int]:
    """Ego pair stays white. Merge, exit, predicted, and outer lanes are purple."""
    kind = str(record.get("kind") or "")
    role = str(record.get("role") or "through")
    try:
        idx = abs(int(record.get("index") if record.get("index") is not None else 1))
    except (TypeError, ValueError):
        idx = 1
    if kind == "predicted" or role in ("merge", "exit") or idx >= 2:
        return LANE_PURPLE
    return LANE_WHITE


def project_main_xyz(
    x: float, y: float, z: float, width: int, height: int,
) -> tuple[int, int] | None:
    """Ego point to the main camera. z = 0 matches project_main_ground."""
    from python.perception.lanes import camera_pose
    from python.sensors.cameras import yaw_pitch_to_dir_up

    pose = camera_pose("main")
    if pose is None:
        return None
    w = float(width)
    h = float(height)
    fov_v = math.radians(float(pose.fov_v_deg))
    if w < 2.0 or h < 2.0 or fov_v <= 1e-3 or fov_v >= math.pi - 1e-3:
        return None
    fov_h = 2.0 * math.atan(math.tan(fov_v * 0.5) * (w / h))
    forward, up = yaw_pitch_to_dir_up(pose.yaw_deg, pose.pitch_deg)
    fx, fy, fz = forward
    ux, uy, uz = up
    right = (fy * uz - fz * uy, fz * ux - fx * uz, fx * uy - fy * ux)
    d = (float(x) - pose.x, float(y) - pose.y, float(z) - pose.z)
    depth = d[0] * fx + d[1] * fy + d[2] * fz
    if depth <= 1e-3:
        return None
    rx = (d[0] * right[0] + d[1] * right[1] + d[2] * right[2]) / depth
    ry = -(d[0] * ux + d[1] * uy + d[2] * uz) / depth
    u = rx / math.tan(fov_h * 0.5) * (w * 0.5) + (w - 1.0) * 0.5
    v = ry / math.tan(fov_v * 0.5) * (h * 0.5) + (h - 1.0) * 0.5
    return int(round(u)), int(round(v))


def _on_image(img: np.ndarray, u: int, v: int, margin: int = 4) -> bool:
    h, w = img.shape[:2]
    return -margin <= u < w + margin and -margin <= v < h + margin


def _dot(img: np.ndarray, project: Project, x: float, y: float, z: float,
         color: tuple[int, int, int], radius_m: float) -> None:
    p0 = project(x, y, z)
    if p0 is None or not _on_image(img, p0[0], p0[1], 8):
        return
    p1 = project(x + radius_m, y, z)
    if p1 is None:
        rad = 2
    else:
        rad = int(round(math.hypot(p1[0] - p0[0], p1[1] - p0[1])))
    rad = max(2, min(DOT_RADIUS_PX_MAX, rad))
    cv2.circle(img, p0, rad, color, -1, cv2.LINE_8)


def draw_dotted(
    img: np.ndarray,
    pts: Any,
    project: Project,
    color: tuple[int, int, int],
    step_m: float,
    *,
    z: float = GROUND_Z,
    radius_m: float = DOT_RADIUS_M,
) -> None:
    for x, y in walk_dots(pts, step_m):
        if y > DRAW_FAR_M:
            continue
        _dot(img, project, x, y, z, color, radius_m)


def _box_corners(x: float, y: float, yaw: float, length: float, width: float) -> list[tuple[float, float]]:
    hx, hy = math.cos(yaw), math.sin(yaw)
    sx, sy = hy, -hx
    hl, hw = length * 0.5, width * 0.5
    out = []
    for s0, s1 in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        out.append((x + hx * hl * s0 + sx * hw * s1, y + hy * hl * s0 + sy * hw * s1))
    return out


def draw_wire_box(
    img: np.ndarray,
    project: Project,
    x: float,
    y: float,
    yaw: float,
    length: float,
    width: float,
    height: float,
    color: tuple[int, int, int],
    *,
    center: bool = True,
) -> None:
    ground = _box_corners(x, y, yaw, length, width)
    bot, top = [], []
    for px, py in ground:
        b = project(px, py, 0.02)
        t = project(px, py, height)
        if b is None or t is None:
            return
        bot.append(b)
        top.append(t)
    dim = tuple(max(0, c // 3) for c in color)
    edges = []
    for i in range(4):
        j = (i + 1) % 4
        edges.append((bot[i], bot[j]))
        edges.append((top[i], top[j]))
        edges.append((bot[i], top[i]))
    for a, b in edges:
        cv2.line(img, a, b, dim, 2, cv2.LINE_8)
    for a, b in edges:
        cv2.line(img, a, b, color, 1, cv2.LINE_8)
    if center:
        c = project(x, y, height * CENTER_Z_FRAC)
        if c is not None:
            cv2.circle(img, c, 3, color, -1, cv2.LINE_8)


def _yaw_of(tr: dict[str, Any]) -> float:
    try:
        if tr.get("yaw") is None:
            return math.pi / 2
        return float(tr["yaw"])
    except (TypeError, ValueError):
        return math.pi / 2


def draw_tracks(
    img: np.ndarray,
    tracks: list[dict[str, Any]],
    project: Project,
    cipv_id: int | None,
    *,
    force_bgr: tuple[int, int, int] | None = None,
) -> None:
    ordered = sorted(tracks, key=lambda t: float(t.get("y") or 0.0), reverse=True)
    for tr in ordered:
        try:
            x = float(tr.get("x") or 0.0)
            y = float(tr.get("y") or 0.0)
        except (TypeError, ValueError):
            continue
        if y > DRAW_FAR_M or y < -8.0:
            continue
        length, width, height = class_dims(tr)
        color = force_bgr if force_bgr is not None else role_bgr(tr, cipv_id)
        draw_wire_box(img, project, x, y, _yaw_of(tr), length, width, height, color)


def draw_light_icon(img: np.ndarray, anchor: tuple[int, int], state: str, scale: int) -> None:
    w = max(10, int(scale))
    h = int(w * 2.5)
    x, y = anchor
    x0, y0 = x - w // 2, y - h // 2
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (24, 24, 26), -1, cv2.LINE_8)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (190, 190, 190), 1, cv2.LINE_8)
    lamps = {"red": (40, 40, 235), "amber": (30, 180, 255), "green": (50, 230, 60)}
    names = ("red", "amber", "green")
    for i, name in enumerate(names):
        cy = y0 + int(h * (0.22 + 0.28 * i))
        col = lamps[name] if state == name else (55, 55, 55)
        cv2.circle(img, (x, cy), max(2, w // 5), col, -1, cv2.LINE_8)
    # The active lamp is also the anchor pixel so a probe at the icon center hits it.
    if state in lamps:
        cv2.circle(img, (x, y), max(2, w // 5), lamps[state], -1, cv2.LINE_8)


def draw_stop_icon(img: np.ndarray, anchor: tuple[int, int], scale: int) -> None:
    r = max(7, int(scale))
    x, y = anchor
    pts = []
    for k in range(8):
        ang = math.pi / 8 + k * math.pi / 4
        pts.append([int(x + r * math.cos(ang)), int(y + r * math.sin(ang))])
    poly = np.array(pts, dtype=np.int32)
    cv2.fillPoly(img, [poly], (40, 40, 220))
    cv2.polylines(img, [poly], True, (240, 240, 240), 1, cv2.LINE_8)


def draw_crosswalk(img: np.ndarray, project: Project, y: float, half_w: float = 4.2) -> None:
    """Two transverse dotted bars. A stop or a light is enough to place them."""
    for yy in (y, y + 2.4):
        xs = np.arange(-half_w, half_w + 0.01, 0.75)
        for x in xs:
            _dot(img, project, float(x), float(yy), GROUND_Z, LANE_WHITE, 0.12)


def _sedan_faces() -> list[tuple[list[tuple[float, float, float]], tuple[int, int, int]]]:
    w = EGO_WIDTH_M * 0.5
    body = (198, 200, 204)
    roof = (226, 228, 232)
    glass = (42, 40, 38)
    side = (168, 170, 176)
    lamp = (50, 50, 230)
    faces: list[tuple[list[tuple[float, float, float]], tuple[int, int, int]]] = []

    def q(pts: list[tuple[float, float, float]], col: tuple[int, int, int]) -> None:
        faces.append((pts, col))

    q([(-w, -2.15, 0.28), (w, -2.15, 0.28), (w, -2.15, 0.86), (-w, -2.15, 0.86)], body)
    q([(-w * 0.96, -2.05, 0.86), (w * 0.96, -2.05, 0.86), (w * 0.9, -0.55, 0.95), (-w * 0.9, -0.55, 0.95)], body)
    q([(-w * 0.78, -0.50, 0.98), (w * 0.78, -0.50, 0.98), (w * 0.70, -0.15, 1.36), (-w * 0.70, -0.15, 1.36)], glass)
    q([(-w * 0.72, -0.12, 1.38), (w * 0.72, -0.12, 1.38), (w * 0.68, 0.85, 1.34), (-w * 0.68, 0.85, 1.34)], roof)
    q([(-w * 0.72, 0.90, 1.30), (w * 0.72, 0.90, 1.30), (w * 0.86, 1.55, 0.92), (-w * 0.86, 1.55, 0.92)], glass)
    q([(-w * 0.90, 1.50, 0.88), (w * 0.90, 1.50, 0.88), (w * 0.92, 2.20, 0.70), (-w * 0.92, 2.20, 0.70)], body)
    q([(-w, -2.15, 0.28), (-w, 2.20, 0.32), (-w, 2.20, 0.78), (-w, -2.15, 0.86)], side)
    q([(w, -2.15, 0.28), (w, 2.20, 0.32), (w, 2.20, 0.78), (w, -2.15, 0.86)], body)
    q([(-w * 0.72, -2.16, 0.52), (-w * 0.28, -2.16, 0.52), (-w * 0.28, -2.16, 0.72), (-w * 0.72, -2.16, 0.72)], lamp)
    q([(w * 0.28, -2.16, 0.52), (w * 0.72, -2.16, 0.52), (w * 0.72, -2.16, 0.72), (w * 0.28, -2.16, 0.72)], lamp)
    return faces


def draw_ego(img: np.ndarray, cam: Cam, project: Project) -> None:
    u, v = project(0.0, 0.25, 0.02) or cam.project(0.0, 0.25, 0.02)
    scale = float(img.shape[0]) / 800.0
    cols = ((22, 22, 26), (48, 50, 56), (96, 100, 108))
    for rad, col in zip(EGO_GLOW_RADII, cols):
        cv2.circle(img, (u, v), max(4, int(rad * scale)), col, -1, cv2.LINE_8)
    faces = _sedan_faces()
    faces.sort(key=lambda item: -sum(p[1] for p in item[0]) / len(item[0]))
    for pts, col in faces:
        pix = []
        for x, y, z in pts:
            got = project(x, y, z)
            if got is None:
                pix = []
                break
            pix.append(got)
        if len(pix) >= 3:
            cv2.fillPoly(img, [np.array(pix, dtype=np.int32)], col, cv2.LINE_8)
    for wx, wy in ((-0.78, -1.45), (0.78, -1.45), (-0.78, 1.35), (0.78, 1.35)):
        _dot(img, project, wx, wy, 0.32, (24, 24, 26), 0.28)


def draw_ego_sprite(img: np.ndarray, cx: int, cy: int) -> None:
    """Small car glyph for the camera overlay. The lens sits in the cabin."""
    for rad, col in ((36, (30, 30, 34)), (22, (70, 72, 78)), (12, (150, 152, 158))):
        cv2.circle(img, (cx, cy + 8), rad, col, -1, cv2.LINE_8)
    body = np.array(
        [[cx - 16, cy + 18], [cx + 16, cy + 18], [cx + 14, cy - 6], [cx - 14, cy - 6]],
        dtype=np.int32,
    )
    roof = np.array(
        [[cx - 10, cy - 4], [cx + 10, cy - 4], [cx + 7, cy - 18], [cx - 7, cy - 18]],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [body], (210, 212, 216), cv2.LINE_8)
    cv2.fillPoly(img, [roof], (48, 46, 44), cv2.LINE_8)
    cv2.circle(img, (cx - 10, cy + 18), 4, (20, 20, 22), -1, cv2.LINE_8)
    cv2.circle(img, (cx + 10, cy + 18), 4, (20, 20, 22), -1, cv2.LINE_8)


def _speed_mps(state: dict[str, Any]) -> float | None:
    ego = state.get("ego") if isinstance(state.get("ego"), dict) else {}
    raw = ego.get("speed_mps")
    if raw is None:
        raw = state.get("speed_mps")
    try:
        if raw is None:
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def _set_mps(state: dict[str, Any]) -> float | None:
    planner = state.get("planner") if isinstance(state.get("planner"), dict) else {}
    raw = planner.get("target_v")
    try:
        if raw is None:
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def draw_hud(img: np.ndarray, state: dict[str, Any]) -> None:
    spd = _speed_mps(state)
    text = "-- m/s" if spd is None else f"{spd:0.1f} m/s"
    cv2.putText(img, text, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD, 1, cv2.LINE_AA)
    word, col = cabin_drive_word(state)
    cv2.putText(img, word, (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    cv2.putText(img, "GVD", (78, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HUD_DIM, 1, cv2.LINE_AA)


def draw_speed_bar(img: np.ndarray, state: dict[str, Any]) -> None:
    h, w = img.shape[:2]
    bar_h = 44
    y0 = h - bar_h
    cv2.rectangle(img, (0, y0), (w, h), (14, 14, 16), -1, cv2.LINE_8)
    spd = _speed_mps(state) or 0.0
    frac = max(0.0, min(1.0, spd / 30.0))
    x1 = 16 + int(280 * frac)
    cv2.rectangle(img, (16, y0 + 16), (296, h - 12), (60, 60, 64), 1, cv2.LINE_8)
    if x1 > 18:
        cv2.rectangle(img, (16, y0 + 16), (x1, h - 12), PATH_BGR, -1, cv2.LINE_8)
    cv2.putText(img, f"{spd:0.1f} m/s", (312, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, HUD, 1, cv2.LINE_AA)
    set_v = _set_mps(state)
    if set_v is not None:
        cv2.putText(
            img, f"set {set_v:0.1f}", (460, y0 + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_DIM, 1, cv2.LINE_AA,
        )
    word, col = cabin_drive_word(state)
    cv2.putText(img, word, (w - 120, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)


def _chase_project(cam: Cam) -> Project:
    def project(x: float, y: float, z: float) -> tuple[int, int] | None:
        return cam.project(x, y, z)

    return project


def _overlay_project(width: int, height: int) -> Project:
    def project(x: float, y: float, z: float) -> tuple[int, int] | None:
        return project_main_xyz(x, y, z, width, height)

    return project


def _path_points(state: dict[str, Any]) -> tuple[list[dict[str, float]], float]:
    scene = _scene_ckpt_ribbon_path(state)
    if scene is not None:
        poly, conf = scene
        from python.viz.scene_ribbon import scene_ckpt_display_gain

        return poly, scene_ckpt_display_gain(conf)
    primary, _ghost = resolve_corridors(state)
    try:
        conf = float(state.get("path_conf") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    return primary, _ribbon_display_gain(state, conf)


def _scale_bgr(color: tuple[int, int, int], gain: float) -> tuple[int, int, int]:
    g = max(0.0, min(1.0, float(gain)))
    return tuple(max(0, min(255, int(c * g))) for c in color)  # type: ignore[return-value]


def paint_vector_scene(
    img: np.ndarray,
    state: dict[str, Any],
    project: Project,
    *,
    cipv_id: int | None,
    overlay: bool,
    draw_lanes: bool,
    draw_path: bool,
    draw_boxes: bool,
    draw_signs: bool,
    grid: bool,
    cam: Cam | None,
) -> None:
    if grid and cam is not None:
        _paint_grid(img, cam)
    y_lo, y_hi = _lane_span(cam, state) if cam is not None else (0.0, DRAW_FAR_M)
    y_hi = min(float(y_hi), DRAW_FAR_M)
    if draw_lanes:
        records = drawn_lane_records(state, None, cam)
        for ln in records:
            pts = _clip_poly_y(ln.get("points") or [], y_lo, y_hi)
            col = LANE_WHITE if overlay else lane_bgr(ln)
            draw_dotted(img, pts, project, col, LANE_STEP_M)
        curb = OVERLAY_CURB if overlay else EDGE_RED
        for edge in list(state.get("road_edges") or state.get("edges") or []):
            pts = _clip_poly_y(_poly_points(edge), y_lo, y_hi)
            draw_dotted(img, pts, project, curb, EDGE_STEP_M, radius_m=0.12)
    if draw_path:
        poly, gain = _path_points(state)
        if len(poly) >= 2:
            draw_dotted(img, poly, project, _scale_bgr(PATH_BGR, max(gain, 0.45)), PATH_STEP_M)
    if draw_signs:
        signs = list(state.get("signs") or [])
        placed = False
        for s in signs:
            cls = str(s.get("cls") or s.get("class") or "")
            try:
                x, y = float(s.get("x") or 0.0), float(s.get("y") or 0.0)
            except (TypeError, ValueError):
                continue
            if cls == "traffic_light":
                anchor = project(x, y, LIGHT_Z)
                if anchor is not None:
                    draw_light_icon(img, anchor, str(s.get("state") or ""), 12)
            elif cls == "stop_sign":
                anchor = project(x, y, SIGN_Z)
                if anchor is not None:
                    draw_stop_icon(img, anchor, 9)
                if not placed:
                    draw_crosswalk(img, project, y)
                    placed = True
            elif cls == "pole":
                foot = project(x, y, 0.0)
                top = project(x, y, 3.0)
                if foot is not None and top is not None:
                    cv2.line(img, foot, top, (150, 150, 150), 1, cv2.LINE_8)
    if draw_boxes:
        tracks = list(state.get("tracks") or [])
        draw_tracks(
            img, tracks, project, cipv_id,
            force_bgr=OVERLAY_BOX if overlay else None,
        )


def _paint_grid(img: np.ndarray, cam: Cam) -> None:
    color = (40, 90, 40)
    project = _chase_project(cam)
    for x in range(-16, 17, 4):
        a = project(float(x), 0.0, 0.0)
        b = project(float(x), 70.0, 0.0)
        if a and b:
            cv2.line(img, a, b, color, 1, cv2.LINE_8)
    for y in range(0, 72, 4):
        a = project(-16.0, float(y), 0.0)
        b = project(16.0, float(y), 0.0)
        if a and b:
            cv2.line(img, a, b, color, 1, cv2.LINE_8)


def render_early_stage(
    state: dict[str, Any],
    ui: Any = None,
    *,
    width: int = 1280,
    height: int = 720,
    main_frame: np.ndarray | None = None,
    top_down: bool = False,
    overlay: bool = False,
    grid: bool = False,
    hud: bool = True,
) -> np.ndarray:
    """One vector frame. An empty state is the ego on black."""
    state = state if isinstance(state, dict) else {}
    dbg = getattr(ui, "debug", None)
    draw_lanes = True if dbg is None else bool(getattr(dbg, "viz_lanes", True))
    draw_path = True if dbg is None else bool(getattr(dbg, "viz_path", True))
    draw_boxes = True if dbg is None else bool(getattr(dbg, "viz_ghosts", True))
    draw_signs = True if dbg is None else bool(getattr(dbg, "viz_signs", True))
    cam = make_early_cam(width, height, top_down=top_down, state=state)
    want_overlay = bool(overlay) and not top_down
    have_frame = (
        want_overlay
        and isinstance(main_frame, np.ndarray)
        and main_frame.size
        and main_frame.ndim == 3
    )
    if have_frame:
        img = cv2.resize(main_frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = np.ascontiguousarray(img[:, :, :3])
        project: Project = _overlay_project(int(width), int(height))
    else:
        img = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        project = _chase_project(cam)
    paint_vector_scene(
        img, state, project,
        cipv_id=cipv_id_of(state),
        overlay=have_frame,
        draw_lanes=draw_lanes,
        draw_path=draw_path,
        draw_boxes=draw_boxes,
        draw_signs=draw_signs,
        grid=bool(grid) and not have_frame,
        cam=cam,
    )
    if have_frame:
        draw_ego_sprite(img, img.shape[1] // 2, img.shape[0] - 44 - 28)
        draw_speed_bar(img, state)
    else:
        if not top_down:
            draw_ego(img, cam, project)
        if want_overlay:
            draw_speed_bar(img, state)
        elif hud:
            draw_hud(img, state)
    return img
