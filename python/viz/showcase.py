"""Share-ready frames. Vector stage, optional camera beside it, a small mark.

Sizes: stage 1280x800, square 1080x1080, story 1080x1920, wide 1920x1080.
Smoothing lives in the offline script. This module draws one frame.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from python.viz.early_vector import render_early_stage

PRESETS: dict[str, tuple[int, int]] = {
    "stage": (1280, 800),
    "square": (1080, 1080),
    "story": (1080, 1920),
    "wide": (1920, 1080),
}


def preset_size(name: str | None) -> tuple[int, int]:
    return PRESETS.get(str(name or "stage"), PRESETS["stage"])


def _camera(main_frame: np.ndarray | None, cam_frames: dict[str, Any] | None) -> np.ndarray | None:
    if isinstance(cam_frames, dict):
        for key in ("main", "cam_main"):
            frame = cam_frames.get(key)
            if isinstance(frame, np.ndarray) and frame.size and frame.ndim >= 2:
                return frame
    if isinstance(main_frame, np.ndarray) and main_frame.size and main_frame.ndim >= 2:
        return main_frame
    return None


def _fit(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    src = frame
    if src.ndim == 2:
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    return cv2.resize(src[:, :, :3], (int(width), int(height)), interpolation=cv2.INTER_AREA)


def paint_mark(img: np.ndarray, *, card: bool = False) -> None:
    h, w = img.shape[:2]
    cv2.putText(img, "GVD", (w - 72, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    if card:
        # Title card top-right so the speed HUD (top-left) stays readable.
        text = "GVD VISION"
        (tw, _th), _b = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)
        cv2.putText(img, text, (w - tw - 18, 34), cv2.FONT_HERSHEY_DUPLEX, 0.8, (235, 235, 235), 1, cv2.LINE_AA)


def render_showcase(
    state: dict[str, Any],
    ui: Any = None,
    *,
    width: int = 1280,
    height: int = 800,
    main_frame: np.ndarray | None = None,
    cam_frames: dict[str, Any] | None = None,
    card: bool = False,
) -> np.ndarray:
    state = state if isinstance(state, dict) else {}
    layout = str(getattr(ui, "showcase_layout", "full") or "full")
    cam = _camera(main_frame, cam_frames)
    if layout == "split" and cam is not None:
        left_w = int(width) // 2
        right_w = int(width) - left_w
        vector = render_early_stage(state, ui, width=right_w, height=int(height), hud=True)
        img = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        img[:, :left_w] = _fit(cam, left_w, int(height))
        img[:, left_w:] = vector
    elif layout == "pip":
        img = render_early_stage(state, ui, width=int(width), height=int(height), hud=True)
        if cam is not None:
            pw, ph = max(80, int(width) // 4), max(60, int(height) // 5)
            img[12:12 + ph, 12:12 + pw] = _fit(cam, pw, ph)
            cv2.rectangle(img, (12, 12), (12 + pw, 12 + ph), (230, 230, 230), 1, cv2.LINE_8)
    else:
        img = render_early_stage(state, ui, width=int(width), height=int(height), hud=True)
    paint_mark(img, card=card)
    return img


def lerp_state(a: dict[str, Any], b: dict[str, Any], t: float) -> dict[str, Any]:
    """Linear blend of ego speed, path points, and tracks that share an id.

    Used by the offline showcase export. The live frame is not blended.
    """
    t = max(0.0, min(1.0, float(t)))
    out = dict(a)

    def mix(x, y):
        try:
            return float(x) * (1.0 - t) + float(y) * t
        except (TypeError, ValueError):
            return x

    ego_a = a.get("ego") if isinstance(a.get("ego"), dict) else {}
    ego_b = b.get("ego") if isinstance(b.get("ego"), dict) else {}
    ego = dict(ego_a)
    if "speed_mps" in ego_a or "speed_mps" in ego_b:
        ego["speed_mps"] = mix(ego_a.get("speed_mps") or 0.0, ego_b.get("speed_mps") or 0.0)
    out["ego"] = ego
    path_a = list(a.get("path_ego") or [])
    path_b = list(b.get("path_ego") or [])
    n = min(len(path_a), len(path_b))
    path = []
    for i in range(n):
        pa, pb = path_a[i], path_b[i]
        if isinstance(pa, dict) and isinstance(pb, dict):
            path.append({"x": mix(pa.get("x"), pb.get("x")), "y": mix(pa.get("y"), pb.get("y"))})
    if path:
        out["path_ego"] = path
    by_id = {}
    for tr in list(b.get("tracks") or []):
        if isinstance(tr, dict) and tr.get("id") is not None:
            by_id[tr.get("id")] = tr
    tracks = []
    for tr in list(a.get("tracks") or []):
        if not isinstance(tr, dict):
            continue
        other = by_id.get(tr.get("id"))
        if not isinstance(other, dict):
            tracks.append(dict(tr))
            continue
        merged = dict(tr)
        for key in ("x", "y"):
            if key in tr or key in other:
                merged[key] = mix(tr.get(key) or 0.0, other.get(key) or 0.0)
        tracks.append(merged)
    out["tracks"] = tracks
    return out
