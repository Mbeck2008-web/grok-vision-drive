"""Multi-panel debug board. Drawn only when the board is on.

Panels: perspective grid, camera stack with a raw readout, top-down map, graphs.
Runtime state only. No map pose, no training labels.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from python.viz.early_vector import render_early_stage

_CALLS = 0


def reset_calls() -> None:
    global _CALLS
    _CALLS = 0


def calls() -> int:
    return _CALLS


def panel_rects(width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    """x0, y0, x1, y1 for each panel."""
    w, h = int(width), int(height)
    left = int(w * 0.40)
    mid = int(w * 0.28)
    gap = 4
    rx = left + mid + gap
    map_y = int(h * 0.46)
    speed_y = int(h * 0.64)
    yaw_y = int(h * 0.82)
    return {
        "grid": (0, 0, left, h),
        "cams": (left + gap, 0, rx - gap, h),
        "map": (rx, 0, w, map_y),
        "speed": (rx, map_y + gap, w, speed_y),
        "yaw": (rx, speed_y + gap, w, yaw_y),
        "pedals": (rx, yaw_y + gap, w, h),
    }


def _sample(state: dict[str, Any]) -> dict[str, float]:
    ego = state.get("ego") if isinstance(state.get("ego"), dict) else {}

    def num(*keys, src=None):
        bag = ego if src is None else src
        for key in keys:
            raw = bag.get(key) if isinstance(bag, dict) else None
            if raw is None and src is None:
                raw = state.get(key)
            try:
                if raw is None:
                    continue
                return float(raw)
            except (TypeError, ValueError):
                continue
        return 0.0

    return {
        "speed": num("speed_mps"),
        "yaw": num("yaw_rate"),
        "steer": num("steer", "steer_deg"),
        "throttle": num("throttle"),
        "brake": num("brake"),
    }


def _blit(canvas: np.ndarray, image: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = rect
    tw, th = max(1, x1 - x0), max(1, y1 - y0)
    if image.shape[1] != tw or image.shape[0] != th:
        image = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)
    canvas[y0:y0 + th, x0:x0 + tw] = image


def _label(img: np.ndarray, text: str, x: int, y: int, color=(220, 220, 220)) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1, cv2.LINE_8)


def _flags(state: dict[str, Any]) -> list[tuple[str, float]]:
    raw = state.get("flags")
    if not isinstance(raw, dict):
        raw = state.get("conditions") if isinstance(state.get("conditions"), dict) else {}
    out = []
    for key, val in raw.items():
        try:
            out.append((str(key), float(val)))
        except (TypeError, ValueError):
            continue
    return out[:8]


def _draw_readout(img: np.ndarray, state: dict[str, Any]) -> None:
    """Raw per-frame lines. A value above 0.5 is green."""
    ego = state.get("ego") if isinstance(state.get("ego"), dict) else {}
    speed = ego.get("speed_mps", state.get("speed_mps"))
    lines = []
    try:
        lines.append((f"speed {float(speed):0.2f}", False))
    except (TypeError, ValueError):
        lines.append(("speed --", False))
    raw_t = state.get("t", state.get("time"))
    try:
        lines.append((f"t {float(raw_t):0.2f}", False))
    except (TypeError, ValueError):
        pass
    try:
        hz = float(state.get("loop_hz"))
        lines.append((f"hz {hz:0.1f}", False))
    except (TypeError, ValueError):
        pass
    calib = state.get("calib") if isinstance(state.get("calib"), dict) else {}
    if calib:
        try:
            lines.append((
                f"cal {float(calib.get('pitch') or 0):0.2f} "
                f"{float(calib.get('yaw') or 0):0.2f} "
                f"{float(calib.get('roll') or 0):0.2f}",
                False,
            ))
        except (TypeError, ValueError):
            pass
    for name, val in _flags(state):
        lines.append((f"{val:0.3f} {name}", val > 0.5))
    y = 14
    for text, hot in lines:
        color = (80, 255, 80) if hot else (240, 240, 240)
        cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1, cv2.LINE_8)
        y += 14


def _camera_stack(
    rect: tuple[int, int, int, int],
    state: dict[str, Any],
    cam_frames: dict[str, Any] | None,
    main_frame: np.ndarray | None,
) -> np.ndarray:
    x0, y0, x1, y1 = rect
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    frames: list[tuple[str, np.ndarray | None]] = []
    bag = cam_frames if isinstance(cam_frames, dict) else {}
    main = bag.get("main", bag.get("cam_main", main_frame))
    frames.append(("main", main if isinstance(main, np.ndarray) else None))
    for key, val in bag.items():
        if key in ("main", "cam_main"):
            continue
        if isinstance(val, np.ndarray) and val.size:
            frames.append((str(key), val))
        if len(frames) >= 3:
            break
    while len(frames) < 3:
        frames.append((f"cam{len(frames)}", None))
    slot_h = h // 3
    for i, (name, frame) in enumerate(frames[:3]):
        y = i * slot_h
        hh = slot_h if i < 2 else h - y
        tile = img[y:y + hh]
        if frame is not None and frame.ndim >= 2 and frame.size:
            src = frame
            if src.ndim == 2:
                src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
            tile[:] = cv2.resize(src[:, :, :3], (w, hh), interpolation=cv2.INTER_AREA)
        _label(tile, name, 6, hh - 8)
        if i == 0:
            _draw_readout(tile, state)
            _draw_main_boxes(tile, state)
    return img


def _draw_main_boxes(tile: np.ndarray, state: dict[str, Any]) -> None:
    from python.viz.early_vector import class_dims, draw_wire_box, project_main_xyz, role_bgr
    from python.viz.stage import cipv_id_of

    h, w = tile.shape[:2]
    cipv = cipv_id_of(state)
    for tr in list(state.get("tracks") or [])[:12]:
        if not isinstance(tr, dict):
            continue
        try:
            x, y = float(tr.get("x") or 0.0), float(tr.get("y") or 0.0)
        except (TypeError, ValueError):
            continue
        length, width, height = class_dims(tr)
        try:
            yaw = float(tr.get("yaw") if tr.get("yaw") is not None else 1.5708)
        except (TypeError, ValueError):
            yaw = 1.5708
        draw_wire_box(
            tile,
            lambda a, b, c, ww=w, hh=h: project_main_xyz(a, b, c, ww, hh),
            x, y, yaw, length, width, height,
            role_bgr(tr, cipv),
        )
    for det in list(state.get("dets") or [])[:16]:
        if not isinstance(det, dict):
            continue
        box = det.get("xyxy")
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        try:
            x0, y0, x1, y1 = [float(v) for v in box[:4]]
        except (TypeError, ValueError):
            continue
        # xyxy is in the source camera. Stretch into this tile.
        src_w = float(det.get("src_w") or w)
        src_h = float(det.get("src_h") or h)
        sx, sy = w / max(1.0, src_w), h / max(1.0, src_h)
        p0 = (int(x0 * sx), int(y0 * sy))
        p1 = (int(x1 * sx), int(y1 * sy))
        cv2.rectangle(tile, p0, p1, (0, 255, 255), 1, cv2.LINE_8)


def _draw_ids(img: np.ndarray, state: dict[str, Any]) -> None:
    from python.viz.early_vector import make_early_cam

    cam = make_early_cam(img.shape[1], img.shape[0], top_down=True, state=state)
    for tr in list(state.get("tracks") or [])[:24]:
        if not isinstance(tr, dict):
            continue
        try:
            x, y = float(tr.get("x") or 0.0), float(tr.get("y") or 0.0)
        except (TypeError, ValueError):
            continue
        u, v = cam.project(x, y, 0.4)
        label = str(tr.get("id") if tr.get("id") is not None else "")
        if label:
            cv2.putText(img, label, (u + 4, v - 4), cv2.FONT_HERSHEY_PLAIN, 1.0, (240, 240, 240), 1, cv2.LINE_8)


GRAPH_GRID = (30, 30, 34)
GRAPH_FONT = cv2.FONT_HERSHEY_DUPLEX


def _graph(rect: tuple[int, int, int, int], title: str, series: list[list[float]], colors: list[tuple[int, int, int]]) -> np.ndarray:
    """Scope-style strip: faint grid, soft fill under the first trace, AA lines,
    latest value of each trace on the right in its own colour."""
    x0, y0, x1, y1 = rect
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (10, 10, 12)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (48, 48, 52), 1, cv2.LINE_8)
    cv2.putText(img, title, (6, 15), GRAPH_FONT, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    if h < 8 or w < 8:
        return img
    plot = img[20:h - 4, 4:w - 4]
    ph, pw = plot.shape[:2]
    if ph < 2 or pw < 2:
        return img
    for k in range(1, 4):
        yy = int(ph * k / 4)
        cv2.line(plot, (0, yy), (pw - 1, yy), GRAPH_GRID, 1, cv2.LINE_8)
    for k in range(1, 8):
        xx = int(pw * k / 8)
        cv2.line(plot, (xx, 0), (xx, ph - 1), GRAPH_GRID, 1, cv2.LINE_8)
    readouts: list[tuple[str, tuple[int, int, int]]] = []
    for idx, (values, color) in enumerate(zip(series, colors)):
        if len(values) < 1:
            continue
        lo = min(values)
        hi = max(values)
        span = hi - lo if hi - lo > 1e-6 else 1.0
        pts = []
        n = len(values)
        for i, val in enumerate(values):
            px = int(round(i * (pw - 1) / max(1, n - 1))) if n > 1 else pw // 2
            py = int(round((1.0 - (val - lo) / span) * (ph - 1)))
            pts.append((px, py))
        if len(pts) == 1:
            cv2.circle(plot, pts[0], 2, color, -1, cv2.LINE_8)
        else:
            if idx == 0:
                area = np.array(pts + [(pts[-1][0], ph - 1), (pts[0][0], ph - 1)], dtype=np.int32)
                fill = tuple(max(0, c // 6) for c in color)
                cv2.fillPoly(plot, [area], fill, cv2.LINE_8)
            cv2.polylines(plot, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
            cv2.circle(plot, pts[-1], 3, color, -1, cv2.LINE_AA)
        readouts.append((f"{values[-1]:+.2f}", color))
    xr = w - 8
    for text, color in reversed(readouts):
        (tw, _th), _b = cv2.getTextSize(text, GRAPH_FONT, 0.42, 1)
        xr -= tw
        cv2.putText(img, text, (xr, 15), GRAPH_FONT, 0.42, color, 1, cv2.LINE_AA)
        xr -= 10
    return img


def render_debug_board(
    state: dict[str, Any],
    ui: Any,
    *,
    width: int = 1280,
    height: int = 720,
    main_frame: np.ndarray | None = None,
    cam_frames: dict[str, Any] | None = None,
) -> np.ndarray:
    global _CALLS
    _CALLS += 1
    state = state if isinstance(state, dict) else {}
    rects = panel_rects(width, height)
    canvas = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    gx0, gy0, gx1, gy1 = rects["grid"]
    grid = render_early_stage(
        state, ui, width=gx1 - gx0, height=gy1 - gy0, top_down=False, grid=True, hud=False,
    )
    _blit(canvas, grid, rects["grid"])
    _label(canvas, "grid", 8, 16)
    cams = _camera_stack(rects["cams"], state, cam_frames, main_frame)
    _blit(canvas, cams, rects["cams"])
    mx0, my0, mx1, my1 = rects["map"]
    top = render_early_stage(
        state, ui, width=mx1 - mx0, height=my1 - my0, top_down=True, hud=False,
    )
    _draw_ids(top, state)
    _blit(canvas, top, rects["map"])
    _label(canvas, "map", mx0 + 6, my0 + 14)

    trace = getattr(ui, "kine_trace", None)
    if trace is None:
        trace = []
        try:
            ui.kine_trace = trace
        except Exception:
            pass
    trace.append(_sample(state))
    if len(trace) > 240:
        del trace[:-240]
    speed = [row["speed"] for row in trace]
    yaw = [row["yaw"] for row in trace]
    _blit(canvas, _graph(rects["speed"], "speed", [speed], [(255, 180, 40)]), rects["speed"])
    _blit(canvas, _graph(rects["yaw"], "yaw", [yaw], [(80, 220, 120)]), rects["yaw"])
    _blit(
        canvas,
        _graph(
            rects["pedals"],
            "steer throttle brake",
            [[r["steer"] for r in trace], [r["throttle"] for r in trace], [r["brake"] for r in trace]],
            [(255, 180, 40), (80, 220, 80), (60, 60, 255)],
        ),
        rects["pedals"],
    )
    return canvas
