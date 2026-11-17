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
# Lane paint draws as short dashes (DASH_M long, gap to the next start is
# LANE_STEP_M); far away a dash shrinks to a dot, like the early beta.
LANE_STEP_M = 1.6
DASH_M = 0.55
DASH_WIDTH_M = 0.16
DASH_PX_MAX = 5
DASH_PX_LEN_MAX = 14
PATH_STEP_M = 2.0
EDGE_STEP_M = 0.8
DOT_RADIUS_M = 0.15
DOT_RADIUS_PX_MAX = 6
DRAW_FAR_M = 150.0
THIN_FROM_M = 40.0
CENTER_Z_FRAC = 0.55
LIGHT_Z = 4.8
SIGN_Z = 2.6
# High chase lens looking down about 17 deg: the horizon sits near the top
# fifth, the ego is small in the lower third, like the early-beta refs.
EARLY_EYE = (0.0, -21.0, 6.6)
EARLY_LOOK = (0.0, 14.3, 0.0)
EARLY_FOV = 30.0
# Ground halo around the ego (world meters) and the headlight wash ahead.
EGO_HALO_M = (1.9, 2.9)
EGO_HALO_GAIN = 0.55
EGO_WASH_GAIN = 0.35

BG = (0, 0, 0)
# Lane colour is the real paint: white paint white, yellow paint yellow.
# Red is only road edges and curbs. No purple lane colour.
LANE_WHITE = (245, 245, 245)
LANE_YELLOW = (30, 210, 255)
EDGE_RED = (40, 40, 255)
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
        fov_deg=early_fov_deg(int(width), int(height)),
    )


def early_fov_deg(width: int, height: int) -> float:
    """Vertical FOV. 16:9 and wider use EARLY_FOV. A square or portrait frame
    widens it so the side lanes stay in view (about 0.6 of the 16:9 width)."""
    if width <= 0 or height <= 0:
        return EARLY_FOV
    tan_v = math.tan(math.radians(EARLY_FOV) * 0.5)
    tan_h_min = tan_v * (16.0 / 9.0) * 0.6
    need = tan_h_min * float(height) / float(width)
    return math.degrees(2.0 * math.atan(max(tan_v, need)))


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


_PAINT_KEYS = ("paint", "paint_color", "color", "colour", "material", "line_color")


def lane_bgr(record: dict[str, Any]) -> tuple[int, int, int]:
    """Real paint colour. Yellow paint (a centre line) is yellow, else white.

    The paint comes from the lane record when a label or the fit carries it
    (``paint`` / ``color`` / ``material`` such as ``line_yellow``). Lane index,
    kind and role never change the colour, so nothing is purple. Red stays
    for road edges and curbs (``EDGE_RED``), never a lane line.
    """
    if not isinstance(record, dict):
        return LANE_WHITE
    for key in _PAINT_KEYS:
        raw = record.get(key)
        if isinstance(raw, str) and "yellow" in raw.lower():
            return LANE_YELLOW
    if record.get("yellow") is True:
        return LANE_YELLOW
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
    cv2.circle(img, p0, rad, color, -1, cv2.LINE_AA)


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
    last: tuple[float, float] | None = None
    for x, y in walk_dots(pts, step_m):
        if y > DRAW_FAR_M:
            continue
        # Past THIN_FROM_M the lens packs dots together; thin them with depth.
        if last is not None and y > THIN_FROM_M:
            if math.hypot(x - last[0], y - last[1]) < step_m * (y / THIN_FROM_M) - 1e-6:
                continue
        last = (x, y)
        _dot(img, project, x, y, z, color, radius_m)


def walk_dashes(pts: Any, step_m: float) -> list[tuple[float, float, float, float]]:
    """Dash starts spaced by arc length, each with the unit heading there."""
    seq: list[tuple[float, float]] = []
    for p in pts or []:
        if isinstance(p, dict):
            seq.append((float(p.get("x") or 0.0), float(p.get("y") or 0.0)))
        else:
            try:
                seq.append((float(p[0]), float(p[1])))
            except (TypeError, IndexError, ValueError):
                continue
    if len(seq) < 2:
        return [(seq[0][0], seq[0][1], 0.0, 1.0)] if seq else []
    step = float(step_m) if float(step_m) > 1e-3 else 1.0
    out: list[tuple[float, float, float, float]] = []
    remain = 0.0
    for (ax, ay), (bx, by) in zip(seq, seq[1:]):
        dx, dy = bx - ax, by - ay
        seg = math.hypot(dx, dy)
        if seg < 1e-8:
            continue
        ux, uy = dx / seg, dy / seg
        travelled = remain
        while travelled <= seg + 1e-8:
            out.append((ax + ux * travelled, ay + uy * travelled, ux, uy))
            travelled += step
        remain = travelled - seg
    return out


def draw_dashed(
    img: np.ndarray,
    pts: Any,
    project: Project,
    color: tuple[int, int, int],
    step_m: float = LANE_STEP_M,
    *,
    dash_m: float = DASH_M,
    width_m: float = DASH_WIDTH_M,
    z: float = GROUND_Z,
) -> None:
    """Lane paint as short dashes along the line. Far dashes become dots."""
    h, w = img.shape[:2]
    last: tuple[float, float] | None = None
    for x, y, ux, uy in walk_dashes(pts, step_m):
        if y > DRAW_FAR_M:
            continue
        if last is not None and y > THIN_FROM_M:
            if math.hypot(x - last[0], y - last[1]) < step_m * (y / THIN_FROM_M) - 1e-6:
                continue
        last = (x, y)
        a = project(x, y, z)
        if a is None or not (-8 <= a[0] < w + 8 and -8 <= a[1] < h + 8):
            continue
        b = project(x + ux * dash_m, y + uy * dash_m, z)
        side = project(x + uy * width_m, y - ux * width_m, z)
        if b is None:
            b = a
        th = 2
        if side is not None:
            th = int(round(math.hypot(side[0] - a[0], side[1] - a[1])))
        th = max(2, min(DASH_PX_MAX, th))
        # Near the lens a dash would stretch into a bar; keep it a short tick.
        du, dv = b[0] - a[0], b[1] - a[1]
        dl = math.hypot(du, dv)
        cap = DASH_PX_LEN_MAX
        if dl > cap:
            b = (a[0] + int(du * cap / dl), a[1] + int(dv * cap / dl))
        cv2.line(img, a, b, color, th, cv2.LINE_AA)


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
    # Neon tube: a wide dim halo, a mid band, then a bright core. Line width
    # follows distance so near boxes read thick like the refs.
    span = max(abs(top[0][1] - bot[0][1]), abs(top[2][1] - bot[2][1]))
    core = 2 if span < 36 else (3 if span < 110 else 4)
    halo = tuple(max(0, c // 5) for c in color)
    mid = tuple(max(0, (c * 5) // 9) for c in color)
    hot = tuple(min(255, c + (255 - c) // 4) for c in color)
    edges = []
    for i in range(4):
        j = (i + 1) % 4
        edges.append((bot[i], bot[j]))
        edges.append((top[i], top[j]))
        edges.append((bot[i], top[i]))
    for a, b in edges:
        cv2.line(img, a, b, halo, core + 5, cv2.LINE_8)
    for a, b in edges:
        cv2.line(img, a, b, mid, core + 2, cv2.LINE_AA)
    for a, b in edges:
        cv2.line(img, a, b, color, core, cv2.LINE_AA)
    if core >= 3:
        for a, b in edges:
            cv2.line(img, a, b, hot, 1, cv2.LINE_AA)
    if center:
        c = project(x, y, height * CENTER_Z_FRAC)
        if c is not None:
            rad = max(3, min(14, int(span * 0.11)))
            cv2.circle(img, c, rad + 2, mid, -1, cv2.LINE_AA)
            cv2.circle(img, c, rad, color, -1, cv2.LINE_AA)


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


def _meters_px(project: Project, anchor: tuple[int, int], x: float, y: float, z: float,
               meters: float, lo: int, hi: int) -> int:
    """Pixel size of ``meters`` across at that point, clamped."""
    side = project(x + meters, y, z)
    if side is None:
        return lo
    return max(lo, min(hi, int(abs(side[0] - anchor[0]))))


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


ZEBRA_BAR_M = 0.5
ZEBRA_GAP_M = 0.6
ZEBRA_DEPTH_M = 3.0
ZEBRA_OFFSET_M = 1.2
STOP_LINE_M = 0.45
ZEBRA_BGR = (185, 185, 185)
STOP_BGR = (245, 245, 245)


def _ground_quad(project: Project, pts: list[tuple[float, float]]) -> np.ndarray | None:
    pix = []
    for x, y in pts:
        got = project(x, y, GROUND_Z)
        if got is None:
            return None
        pix.append(got)
    return np.array(pix, dtype=np.int32)


def draw_crosswalk(img: np.ndarray, project: Project, y: float, half_w: float = 4.2) -> None:
    """Solid white stop bar at ``y``, then a zebra crosswalk just past it.

    Zebra bars run along the road (ZEBRA_BAR_M wide, ZEBRA_GAP_M apart) and
    are filled grey with a white rim, so they read as paint, not as lanes.
    """
    bar = _ground_quad(project, [(-half_w, y), (half_w, y), (half_w, y + STOP_LINE_M), (-half_w, y + STOP_LINE_M)])
    if bar is not None:
        cv2.fillPoly(img, [bar], STOP_BGR, cv2.LINE_AA)
    y0 = y + STOP_LINE_M + ZEBRA_OFFSET_M
    y1 = y0 + ZEBRA_DEPTH_M
    x = -half_w
    rim = (205, 205, 205)
    while x + ZEBRA_BAR_M <= half_w + 1e-6:
        quad = _ground_quad(project, [(x, y0), (x + ZEBRA_BAR_M, y0), (x + ZEBRA_BAR_M, y1), (x, y1)])
        if quad is not None:
            cv2.fillPoly(img, [quad], ZEBRA_BGR, cv2.LINE_AA)
            cv2.polylines(img, [quad], True, rim, 1, cv2.LINE_AA)
        x += ZEBRA_BAR_M + ZEBRA_GAP_M


# Low-poly ego: cross-sections from the rear bumper to the nose, lofted into
# quads and shaded by one light. Each row: y, body half-width, sill z,
# belt z, roof half-width, roof z, is_cabin (side and top panels are glass).
EGO_SECTIONS: tuple[tuple[float, float, float, float, float, float, bool], ...] = (
    (-2.34, 0.82, 0.36, 0.66, 0.66, 0.74, False),
    (-2.24, 0.90, 0.32, 0.84, 0.74, 0.90, False),
    (-1.80, 0.93, 0.30, 0.88, 0.72, 0.95, False),
    (-1.35, 0.94, 0.30, 0.90, 0.64, 1.00, True),
    (-0.25, 0.94, 0.30, 0.92, 0.58, 1.40, True),
    (0.80, 0.94, 0.30, 0.92, 0.58, 1.42, True),
    (1.35, 0.94, 0.30, 0.88, 0.70, 0.98, True),
    (2.05, 0.90, 0.32, 0.76, 0.72, 0.82, False),
    (2.36, 0.74, 0.36, 0.58, 0.56, 0.64, False),
)
EGO_BODY = (222, 224, 228)
EGO_GLASS = (46, 42, 40)
EGO_TAIL = (40, 30, 225)
EGO_SHADOW = (14, 14, 16)
_LIGHT = (-0.30, 0.40, 0.87)
_GLASS_SEGS = (2, 3, 4)


def _ego_ring(sec: tuple[float, float, float, float, float, float, bool]) -> list[tuple[float, float, float]]:
    """Cross-section: sill, rounded flank, shoulder, roof, then mirrored."""
    y, hw, z0, zb, rw, zr, _cab = sec
    zm = z0 + (zb - z0) * 0.55
    left = [(-hw * 0.93, y, z0), (-hw, y, zm), (-hw * 0.95, y, zb), (-rw, y, zr)]
    right = [(-x, yy, z) for x, yy, z in reversed(left)]
    return left + right


def _ego_faces() -> list[tuple[list[tuple[float, float, float]], tuple[int, int, int], bool]]:
    """(points, base colour, is_glass). Outward winding is fixed at shading."""
    faces: list[tuple[list[tuple[float, float, float]], tuple[int, int, int], bool]] = []
    for a, b in zip(EGO_SECTIONS, EGO_SECTIONS[1:]):
        ra, rb = _ego_ring(a), _ego_ring(b)
        cabin = a[6] and b[6]
        for k in range(len(ra) - 1):
            quad = [ra[k], ra[k + 1], rb[k + 1], rb[k]]
            glass = cabin and k in _GLASS_SEGS
            faces.append((quad, EGO_GLASS if glass else EGO_BODY, glass))
    faces.append((_ego_ring(EGO_SECTIONS[0]), EGO_BODY, False))
    faces.append((_ego_ring(EGO_SECTIONS[-1]), EGO_BODY, False))
    # Tail lamps wrap the rear corners, just under the deck lip.
    y0 = EGO_SECTIONS[0][0] - 0.02
    for sgn in (-1.0, 1.0):
        faces.append(([
            (sgn * 0.40, y0, 0.56), (sgn * 0.80, y0, 0.54),
            (sgn * 0.80, y0, 0.68), (sgn * 0.40, y0, 0.71),
        ], EGO_TAIL, False))
    faces.append(([(-0.70, y0, 0.37), (0.70, y0, 0.37), (0.72, y0, 0.46), (-0.72, y0, 0.46)], (40, 40, 44), False))
    return faces


def _shade(pts: list[tuple[float, float, float]], base: tuple[int, int, int], glass: bool,
           eye: tuple[float, float, float]) -> tuple[tuple[int, int, int], bool]:
    """Lambert + a little specular. Returns colour and whether it faces the eye.

    Tail lamps are emissive: they keep their colour whatever the light.
    """
    p0, p1, p2 = np.array(pts[0]), np.array(pts[1]), np.array(pts[2])
    n = np.cross(p1 - p0, p2 - p0)
    if float(np.linalg.norm(n)) < 1e-9 and len(pts) > 3:
        n = np.cross(p2 - p0, np.array(pts[3]) - p0)
    n = n / (np.linalg.norm(n) + 1e-9)
    cen = np.mean(np.array(pts), axis=0)
    if float(np.dot(n, cen - np.array((0.0, 0.0, 0.6)))) < 0.0:
        n = -n
    to_eye = np.array(eye) - cen
    to_eye = to_eye / (np.linalg.norm(to_eye) + 1e-9)
    facing = float(np.dot(n, to_eye)) > -0.05
    if base == EGO_TAIL:
        return base, facing
    light = np.array(_LIGHT) / np.linalg.norm(_LIGHT)
    lam = max(0.0, float(np.dot(n, light)))
    half = light + to_eye
    half = half / (np.linalg.norm(half) + 1e-9)
    spec = max(0.0, float(np.dot(n, half))) ** (24 if glass else 10)
    amb = 0.62 if not glass else 0.8
    k = amb + (1.0 - amb) * lam
    add = (110.0 if glass else 45.0) * spec
    col = tuple(int(max(0, min(255, c * k + add))) for c in base)
    return col, facing  # type: ignore[return-value]


_EGO_CACHE: dict[tuple, tuple[list[tuple[np.ndarray, tuple[int, int, int]]], np.ndarray | None, tuple[int, int]]] = {}


def _cam_key(cam: Cam, w: int, h: int) -> tuple:
    return (w, h, tuple(cam.eye), tuple(cam.look_override or ()), float(cam.fov_deg))


def _ego_glow_patch(cam: Cam, project: Project, w: int, h: int) -> tuple[np.ndarray | None, tuple[int, int]]:
    """Soft white halo on the ground plus a headlight wash ahead, pre-blurred."""
    mask = np.zeros((h, w), dtype=np.float32)
    ax, ay = EGO_HALO_M
    ring = []
    for k in range(40):
        t = 2.0 * math.pi * k / 40.0
        ring.append(project(ax * math.cos(t), 0.3 + ay * math.sin(t), 0.0))
    if all(p is not None for p in ring):
        cv2.fillPoly(mask, [np.array(ring, dtype=np.int32)], EGO_HALO_GAIN, cv2.LINE_AA)
    wash = [project(-1.0, 2.3, 0.0), project(1.0, 2.3, 0.0), project(2.6, 11.0, 0.0), project(-2.6, 11.0, 0.0)]
    if all(p is not None for p in wash):
        tmp = np.zeros_like(mask)
        cv2.fillPoly(tmp, [np.array(wash, dtype=np.int32)], EGO_WASH_GAIN, cv2.LINE_AA)
        mask = np.maximum(mask, tmp)
    k = max(9, int(h * 0.06) | 1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=k * 0.5)
    ys, xs = np.nonzero(mask > 0.004)
    if ys.size == 0:
        return None, (0, 0)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    patch = np.clip(mask[y0:y1, x0:x1] * 255.0, 0, 255).astype(np.uint8)
    return cv2.merge([patch, patch, patch]), (x0, y0)


def _ego_cached(cam: Cam, project: Project, w: int, h: int):
    key = _cam_key(cam, w, h)
    hit = _EGO_CACHE.get(key)
    if hit is not None:
        return hit
    eye = tuple(float(v) for v in cam.eye)
    polys: list[tuple[float, np.ndarray, tuple[int, int, int]]] = []
    for pts, base, glass in _ego_faces():
        col, facing = _shade(pts, base, glass, eye)
        if not facing:
            continue
        pix = []
        for x, y, z in pts:
            got = project(x, y, z)
            if got is None:
                pix = []
                break
            pix.append(got)
        if len(pix) < 3:
            continue
        cen = np.mean(np.array(pts), axis=0)
        dist = float(np.linalg.norm(cen - np.array(eye)))
        polys.append((dist, np.array(pix, dtype=np.int32), col))
    polys.sort(key=lambda item: -item[0])
    shadow = [project(x, y, 0.0) for x, y in ((-0.86, -1.40), (0.86, -1.40), (0.86, 2.20), (-0.86, 2.20))]
    if all(p is not None for p in shadow):
        polys.insert(0, (1e9, np.array(shadow, dtype=np.int32), EGO_SHADOW))
    glow, origin = _ego_glow_patch(cam, project, w, h)
    out = ([(poly, col) for _d, poly, col in polys], glow, origin)
    if len(_EGO_CACHE) > 16:
        _EGO_CACHE.clear()
    _EGO_CACHE[key] = out
    return out


def draw_ego_glow(img: np.ndarray, cam: Cam, project: Project) -> None:
    h, w = img.shape[:2]
    _polys, glow, (x0, y0) = _ego_cached(cam, project, w, h)
    if glow is None:
        return
    gh, gw = glow.shape[:2]
    x1, y1 = min(w, x0 + gw), min(h, y0 + gh)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    cv2.add(roi, glow[: y1 - y0, : x1 - x0], dst=roi)


def draw_ego(img: np.ndarray, cam: Cam, project: Project) -> None:
    """Shaded white low-poly car. The glow is drawn before the scene."""
    h, w = img.shape[:2]
    polys, _glow, _o = _ego_cached(cam, project, w, h)
    for poly, col in polys:
        cv2.fillPoly(img, [poly], col, cv2.LINE_AA)


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


HUD_FONT = cv2.FONT_HERSHEY_DUPLEX


def draw_hud(img: np.ndarray, state: dict[str, Any]) -> None:
    """Big speed number, unit under it, drive word: the early-beta corner.

    Sizes follow the short side so a 1080p showcase still reads at 720p scale.
    """
    hh, ww = img.shape[:2]
    k = max(1.0, min(hh, ww) / 720.0)
    spd = _speed_mps(state)
    num = "--" if spd is None else f"{spd:0.1f}"
    x0 = int(18 * k)
    y_num = int(50 * k)
    cv2.putText(img, num, (x0, y_num), HUD_FONT, 1.25 * k, HUD, max(2, int(2 * k)), cv2.LINE_AA)
    (tw, _th), _b = cv2.getTextSize(num, HUD_FONT, 1.25 * k, max(2, int(2 * k)))
    cv2.putText(img, "m/s", (x0 + tw + int(6 * k), y_num), HUD_FONT, 0.5 * k, HUD_DIM, 1, cv2.LINE_AA)
    word, col = cabin_drive_word(state)
    y_word = int(76 * k)
    cv2.putText(img, word, (x0 + 2, y_word), HUD_FONT, 0.5 * k, col, 1, cv2.LINE_AA)
    (wl, _wh), _b = cv2.getTextSize(word, HUD_FONT, 0.5 * k, 1)
    cv2.putText(img, "GVD", (x0 + wl + int(14 * k), y_word), HUD_FONT, 0.45 * k, HUD_DIM, 1, cv2.LINE_AA)


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
    cv2.putText(img, f"{spd:0.1f} m/s", (312, y0 + 28), HUD_FONT, 0.6, HUD, 1, cv2.LINE_AA)
    set_v = _set_mps(state)
    if set_v is not None:
        cv2.putText(
            img, f"set {set_v:0.1f}", (460, y0 + 28),
            HUD_FONT, 0.5, HUD_DIM, 1, cv2.LINE_AA,
        )
    word, col = cabin_drive_word(state)
    cv2.putText(img, word, (w - 120, y0 + 28), HUD_FONT, 0.55, col, 1, cv2.LINE_AA)


def _chase_project(cam: Cam) -> Project:
    """Same numbers as ``cam.project`` without a numpy call per point."""
    if cam.top_down or getattr(cam, "_basis", None) is None:
        def project_td(x: float, y: float, z: float) -> tuple[int, int] | None:
            return cam.project(x, y, z)

        return project_td
    eye, fwd, right, up, f = cam._basis  # type: ignore[misc]
    ex, ey, ez = (float(v) for v in eye)
    fx, fy, fz = (float(v) for v in fwd)
    rx, ry, rz = (float(v) for v in right)
    ux, uy, uz = (float(v) for v in up)
    hw, hh = int(cam.width) * 0.5, int(cam.height) * 0.5
    ff = float(f)

    def project(x: float, y: float, z: float) -> tuple[int, int] | None:
        dx, dy, dz = x - ex, y - ey, z - ez
        cz = dx * fx + dy * fy + dz * fz
        if cz < 0.6:
            cz = 0.6
        cx = dx * rx + dy * ry + dz * rz
        cy = dx * ux + dy * uy + dz * uz
        return int(hw + ff * cx / cz), int(hh - ff * cy / cz)

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
            # drawn_lane_records already clipped to the cam span; the far
            # cut (DRAW_FAR_M) happens per dash.
            pts = ln.get("points") or []
            draw_dashed(img, pts, project, lane_bgr(ln), LANE_STEP_M)
        curb = OVERLAY_CURB if overlay else EDGE_RED
        for edge in list(state.get("road_edges") or state.get("edges") or []):
            pts = _clip_poly_y(_poly_points(edge), y_lo, y_hi)
            draw_dotted(img, pts, project, curb, EDGE_STEP_M, radius_m=0.13)
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
                    draw_light_icon(img, anchor, str(s.get("state") or ""), _meters_px(project, anchor, x, y, LIGHT_Z, 0.8, 11, 28))
            elif cls == "stop_sign":
                anchor = project(x, y, SIGN_Z)
                if anchor is not None:
                    draw_stop_icon(img, anchor, _meters_px(project, anchor, x, y, SIGN_Z, 0.45, 8, 22))
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
        if not top_down:
            draw_ego_glow(img, cam, project)
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
