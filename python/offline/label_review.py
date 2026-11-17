"""Draw training labels on a recorded camera frame.

Offline only. The live supervisor does not import this module. Missing fields
are skipped. Values are drawn as given, with no smoothing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from python.viz.early_vector import (
    LIGHT_Z,
    SIGN_Z,
    _box_corners,
    _yaw_of,
    class_dims,
    project_main_xyz,
)

YELLOW = (0, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)


def _as_float(v: Any, default: float | None = None) -> float | None:
    try:
        if v is None:
            return default
        out = float(v)
    except (TypeError, ValueError):
        return default
    if out != out:
        return default
    return out


def class_letter(cls: str) -> str:
    name = str(cls or "").strip().lower()
    if name in ("pedestrian", "ped", "person"):
        return "G"
    if name in ("cone", "cones"):
        return "F"
    if name in ("truck", "bus", "trailer", "van", "suv"):
        return "V"
    if name in ("traffic_light", "stop_sign", "sign", "pole"):
        return "K"
    if name in ("vehicle", "car", "bike", "bicycle", "motorcycle"):
        return "C"
    if name:
        return name[:1].upper()
    return "?"


def _pts(src: Any) -> list[dict[str, float]]:
    if isinstance(src, dict):
        src = src.get("points") or src.get("pts") or []
    out = []
    for p in src or []:
        if isinstance(p, dict):
            x, y = _as_float(p.get("x")), _as_float(p.get("y"))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            x, y = _as_float(p[0]), _as_float(p[1])
        else:
            continue
        if x is None or y is None:
            continue
        out.append({"x": x, "y": y})
    return out


def _project(x: float, y: float, z: float, w: int, h: int) -> tuple[int, int] | None:
    return project_main_xyz(x, y, z, w, h)


def _polyline(img: np.ndarray, pts: list[dict[str, float]], z: float, color, thickness: int) -> None:
    h, w = img.shape[:2]
    pix = []
    for p in pts:
        uv = _project(p["x"], p["y"], z, w, h)
        if uv is not None:
            pix.append(uv)
    if len(pix) >= 2:
        cv2.polylines(img, [np.array(pix, dtype=np.int32)], False, color, thickness, cv2.LINE_8)


def _wire(img: np.ndarray, corners_xy: list[tuple[float, float]], z0: float, z1: float, color, thickness: int) -> None:
    h, w = img.shape[:2]
    bot, top = [], []
    for x, y in corners_xy:
        b = _project(x, y, z0, w, h)
        t = _project(x, y, z1, w, h)
        if b is None or t is None:
            return
        bot.append(b)
        top.append(t)
    for i in range(len(bot)):
        j = (i + 1) % len(bot)
        cv2.line(img, bot[i], bot[j], color, thickness, cv2.LINE_8)
        cv2.line(img, top[i], top[j], color, thickness, cv2.LINE_8)
        cv2.line(img, bot[i], top[i], color, thickness, cv2.LINE_8)


def _tag(img: np.ndarray, x: int, y: int, dist_m: float, letter: str) -> None:
    y = max(36, int(y) - 8)
    x = max(2, min(img.shape[1] - 48, int(x)))
    cv2.rectangle(img, (x - 2, y - 26), (x + 42, y + 4), (0, 0, 0), -1, cv2.LINE_8)
    cv2.rectangle(img, (x - 2, y - 26), (x + 42, y + 4), WHITE, 1, cv2.LINE_8)
    cv2.putText(img, f"{dist_m:0.0f}m", (x, y - 12), cv2.FONT_HERSHEY_PLAIN, 1.0, RED, 1, cv2.LINE_8)
    cv2.putText(img, letter, (x, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.1, WHITE, 1, cv2.LINE_8)


def _draw_tracks(img: np.ndarray, labels: dict[str, Any]) -> None:
    h, w = img.shape[:2]
    lead = labels.get("cipv_id")
    try:
        lead_id = None if lead is None else int(lead)
    except (TypeError, ValueError):
        lead_id = None
    for tr in list(labels.get("tracks") or labels.get("objects") or []):
        if not isinstance(tr, dict):
            continue
        x, y = _as_float(tr.get("x")), _as_float(tr.get("y"))
        if x is None or y is None:
            continue
        length, width, height = class_dims(tr)
        yaw = _yaw_of(tr)
        corners = _box_corners(x, y, yaw, length, width)
        in_path = bool(tr.get("in_path") or tr.get("lead"))
        try:
            in_path = in_path or (lead_id is not None and int(tr.get("id")) == lead_id)
        except (TypeError, ValueError):
            pass
        color = RED if in_path else YELLOW
        _wire(img, corners, 0.02, height, color, 2)
        top = []
        for cx, cy in corners:
            uv = _project(cx, cy, height, w, h)
            if uv is not None:
                top.append(uv)
        if top:
            anchor = min(top, key=lambda p: p[1])
            dist = _as_float(tr.get("dist_m"))
            if dist is None:
                dist = math.hypot(x, y)
            _tag(img, anchor[0], anchor[1], dist, class_letter(str(tr.get("class") or tr.get("cls") or "")))


def _draw_lanes(img: np.ndarray, labels: dict[str, Any]) -> None:
    lanes = labels.get("lanes") or labels.get("lanes_ext") or []
    for ln in lanes:
        pts = _pts(ln)
        if len(pts) < 2:
            continue
        _polyline(img, pts, 0.02, GREEN, 2)


def _draw_stops(img: np.ndarray, labels: dict[str, Any]) -> bool:
    found = False
    for bar in list(labels.get("stop_lines") or []):
        if not isinstance(bar, dict):
            continue
        y = _as_float(bar.get("y", bar.get("y_m")))
        if y is None:
            continue
        x0 = _as_float(bar.get("x0"), -3.5)
        x1 = _as_float(bar.get("x1"), 3.5)
        _polyline(img, [{"x": x0, "y": y}, {"x": x1, "y": y}], 0.04, RED, 5)
        found = True
    return found


def _draw_edges(img: np.ndarray, labels: dict[str, Any]) -> None:
    h, w = img.shape[:2]
    edges = labels.get("edges") or labels.get("road_edges") or []
    for edge in edges:
        for x, y in _walk(edge, 2.0):
            uv = _project(x, y, 0.02, w, h)
            if uv is None:
                continue
            cv2.rectangle(img, (uv[0] - 2, uv[1] - 2), (uv[0] + 2, uv[1] + 2), BLUE, -1, cv2.LINE_8)


def _walk(src: Any, step: float) -> list[tuple[float, float]]:
    from python.viz.early_vector import walk_dots

    return walk_dots(_pts(src), step)


def _draw_signs(img: np.ndarray, labels: dict[str, Any]) -> bool:
    stop = False
    h, w = img.shape[:2]
    for s in list(labels.get("signs") or []):
        if not isinstance(s, dict):
            continue
        x, y = _as_float(s.get("x")), _as_float(s.get("y"))
        if x is None or y is None:
            continue
        cls = str(s.get("cls") or s.get("class") or "")
        if cls == "stop_sign":
            stop = True
            color = RED
            z = SIGN_Z
        elif cls == "traffic_light":
            color = GREEN if str(s.get("state") or "") == "green" else YELLOW
            z = LIGHT_Z
        else:
            color = YELLOW
            z = SIGN_Z
        corners = _box_corners(x, y, math.pi / 2, 0.4, 0.35)
        _wire(img, corners, z - 0.4, z + 0.4, color, 1)
    return stop


def _draw_cones(img: np.ndarray, labels: dict[str, Any]) -> None:
    h, w = img.shape[:2]
    for cone in list(labels.get("cones") or []):
        if not isinstance(cone, dict):
            continue
        x, y = _as_float(cone.get("x")), _as_float(cone.get("y"))
        if x is None or y is None:
            continue
        base = [(x - 0.2, y - 0.2), (x + 0.2, y - 0.2), (x + 0.2, y + 0.2), (x - 0.2, y + 0.2)]
        pix = []
        for bx, by in base:
            uv = _project(bx, by, 0.02, w, h)
            if uv is not None:
                pix.append(uv)
        apex = _project(x, y, 0.75, w, h)
        if apex is None or len(pix) < 3:
            continue
        for i in range(len(pix)):
            cv2.line(img, pix[i], pix[(i + 1) % len(pix)], ORANGE, 1, cv2.LINE_8)
            cv2.line(img, pix[i], apex, ORANGE, 1, cv2.LINE_8)


def _bev(img: np.ndarray, labels: dict[str, Any]) -> None:
    h, w = img.shape[:2]
    strip = max(90, int(w * 0.16))

    def map_pt(x: float, y: float) -> tuple[int, int]:
        u = w - strip + 8 + int((x + 8.0) / 16.0 * (strip - 28))
        v = int(h - 24 - max(0.0, min(40.0, y)) / 40.0 * (h - 48))
        return u, v

    for ln in list(labels.get("lanes") or labels.get("lanes_ext") or []):
        pts = _walk(ln, 2.0)
        for i, (x, y) in enumerate(pts):
            if i % 2:
                continue
            u, v = map_pt(x, y)
            cv2.circle(img, (u, v), 2, WHITE, -1, cv2.LINE_8)
    path = _pts(labels.get("path") or labels.get("path_ego") or [])
    if len(path) >= 2:
        pix = [map_pt(p["x"], p["y"]) for p in path if 0.0 <= p["y"] <= 45.0]
        if len(pix) >= 2:
            cv2.polylines(img, [np.array(pix, dtype=np.int32)], False, WHITE, 2, cv2.LINE_8)
    # Distance ruler on the right edge.
    for meters in (10, 20, 30, 40):
        _u, v = map_pt(0.0, float(meters))
        cv2.line(img, (w - 18, v), (w - 4, v), WHITE, 1, cv2.LINE_8)
        cv2.putText(img, f"{meters}m", (w - 52, v - 2), cv2.FONT_HERSHEY_PLAIN, 0.8, WHITE, 1, cv2.LINE_8)


def _hud(img: np.ndarray, labels: dict[str, Any]) -> None:
    ego = labels.get("ego") if isinstance(labels.get("ego"), dict) else {}
    lines: list[tuple[str, bool]] = []
    speed = _as_float(ego.get("speed_mps"), _as_float(labels.get("speed_mps")))
    lines.append((f"speed {speed:0.2f}" if speed is not None else "speed --", False))
    raw_t = _as_float(labels.get("t", labels.get("time")))
    if raw_t is not None:
        lines.append((f"time {raw_t:0.2f}", False))
    calib = labels.get("calib") if isinstance(labels.get("calib"), dict) else {}
    if calib or "calib" in labels:
        p = _as_float(calib.get("pitch"), 0.0) or 0.0
        yw = _as_float(calib.get("yaw"), 0.0) or 0.0
        r = _as_float(calib.get("roll"), 0.0) or 0.0
        lines.append((f"cal {p:0.2f} {yw:0.2f} {r:0.2f}", False))
    fps = labels.get("fps") if isinstance(labels.get("fps"), dict) else {}
    if fps:
        try:
            lines.append((
                f"fps {float(fps.get('vision') or 0):0.1f} {float(fps.get('draw') or 0):0.1f}",
                False,
            ))
        except (TypeError, ValueError):
            pass
    elif _as_float(labels.get("loop_hz")) is not None:
        lines.append((f"hz {_as_float(labels.get('loop_hz')):0.1f}", False))
    flags = labels.get("flags") if isinstance(labels.get("flags"), dict) else {}
    for key, val in list(flags.items())[:8]:
        num = _as_float(val)
        if num is None:
            continue
        lines.append((f"{num:0.4f} {key}", num > 0.5))
    y = 16
    for text, hot in lines:
        color = GREEN if hot else WHITE
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1, cv2.LINE_8)
        y += 14
    right = []
    for key in ("path_conf", "lane_conf"):
        num = _as_float(labels.get(key))
        if num is not None:
            right.append(f"{key} {num:0.2f}")
    word = str(labels.get("state_word") or labels.get("engage") or "")
    if word:
        right.append(word)
    y = 16
    h, w = img.shape[:2]
    for text in right:
        cv2.putText(img, text, (w - 180, y), cv2.FONT_HERSHEY_PLAIN, 1.0, WHITE, 1, cv2.LINE_8)
        y += 14
    cv2.putText(img, "MAIN", (8, h - 10), cv2.FONT_HERSHEY_PLAIN, 1.2, WHITE, 1, cv2.LINE_8)


def _stop_icon(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    cx, cy, r = w // 2, 36, 16
    pts = []
    for k in range(8):
        ang = math.pi / 8 + k * math.pi / 4
        pts.append([int(cx + r * math.cos(ang)), int(cy + r * math.sin(ang))])
    poly = np.array(pts, dtype=np.int32)
    cv2.fillPoly(img, [poly], RED, cv2.LINE_8)
    cv2.polylines(img, [poly], True, WHITE, 1, cv2.LINE_8)


def render_label_frame(frame: np.ndarray | None, labels: dict[str, Any] | None) -> np.ndarray:
    """Grayscale camera plus whatever label fields are present."""
    labels = labels if isinstance(labels, dict) else {}
    if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim < 2:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        src = frame
        if src.ndim == 2:
            src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(src[:, :, :3], cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    _draw_lanes(img, labels)
    had_bar = _draw_stops(img, labels)
    _draw_edges(img, labels)
    _draw_tracks(img, labels)
    had_sign = _draw_signs(img, labels)
    _draw_cones(img, labels)
    _bev(img, labels)
    _hud(img, labels)
    if had_bar or had_sign or any(
        str((s or {}).get("cls") or (s or {}).get("class") or "") == "stop_sign"
        for s in list(labels.get("signs") or [])
        if isinstance(s, dict)
    ):
        _stop_icon(img)
    return img


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _frame_files(folder: Path) -> list[Path]:
    patterns = (
        "frames/*",
        "raw/main/*",
        "main/*",
        "*",
    )
    for pattern in patterns:
        files = [
            p for p in sorted(folder.glob(pattern))
            if p.suffix.lower() in (".png", ".jpg", ".jpeg") and p.is_file()
        ]
        if files:
            return files
    return []


def load_recording(folder: Path | str) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """Pair images with label rows. A short label file leaves later frames empty."""
    root = Path(folder)
    rows: list[dict[str, Any]] = []
    for name in ("labels.jsonl", "gt.jsonl", "state.jsonl"):
        got = _read_jsonl(root / name)
        if got:
            rows = got
            if name == "labels.jsonl":
                break
    files = _frame_files(root)
    pairs = []
    if not files and rows:
        for row in rows:
            pairs.append((np.zeros((480, 640, 3), dtype=np.uint8), row))
        return pairs
    for i, path in enumerate(files):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        labels = rows[i] if i < len(rows) else {}
        pairs.append((img, labels))
    return pairs


def write_frames(pairs: list[tuple[np.ndarray, dict[str, Any]]], dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for i, (frame, labels) in enumerate(pairs):
        painted = render_label_frame(frame, labels)
        path = dest / f"frame_{i:04d}.png"
        cv2.imwrite(str(path), painted)
        out.append(path)
    return out


def encode_mp4(frames: list[np.ndarray], dest: Path, fps: float = 18.0) -> bool:
    """ffmpeg libx264. Returns False when ffmpeg is missing or the encode fails."""
    import shutil
    import subprocess
    import tempfile

    if not frames:
        return False
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    with tempfile.TemporaryDirectory(prefix="gvd_label_") as tmp:
        folder = Path(tmp)
        for i, frame in enumerate(frames):
            img = frame
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(folder / f"{i:04d}.png"), img)
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", str(folder / "%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(dest),
        ]
        try:
            subprocess.run(cmd, check=True)
        except (OSError, subprocess.CalledProcessError):
            return False
    return dest.is_file() and dest.stat().st_size > 0
