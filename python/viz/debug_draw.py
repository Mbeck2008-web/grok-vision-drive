"""Dense debug overlays for GVD VISION (occupancy, FOV, planner samples, HUD).

Occupancy is derived from tracks + the corridor — not a learned grid. Camera FOV
wedges come from config/cameras.yaml. Planner samples are a toy lateral cost,
not an optimizer.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from python.sensors.cameras import CAM_IDS

OCC = (88, 86, 196)
FREE = (120, 150, 92)
FOV = (168, 156, 120)
COST_COLD = (212, 196, 158)
COST_HOT = (92, 92, 196)
HUD = (212, 204, 200)
HUD_DIM = (120, 120, 120)
ICE_HI = (248, 238, 214)
PAPER = (225, 230, 232)

_FRUSTUMS: list[dict[str, Any]] | None = None


def load_frustums() -> list[dict[str, Any]]:
    global _FRUSTUMS
    if _FRUSTUMS is not None:
        return _FRUSTUMS
    out: list[dict[str, Any]] = []
    try:
        from python.sensors.cameras import load_camera_config

        cfg = load_camera_config()
        for cam in cfg.get("cameras") or []:
            pos = cam.get("pos_m") or [0.0, 0.0, 1.0]
            fov_v = float(cam.get("fov_v") or 40.0)
            out.append(
                {
                    "id": str(cam.get("id") or ""),
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(pos[2]) if len(pos) > 2 else 1.2,
                    "yaw_deg": float(cam.get("yaw_deg") or 0.0),
                    "fov_h": min(140.0, fov_v * 1.45),
                    "range_m": 28.0 if fov_v < 30 else (16.0 if fov_v < 70 else 10.0),
                }
            )
    except Exception:
        out = []
    _FRUSTUMS = out
    return out


def draw_occupancy(
    img: np.ndarray,
    cam: Any,
    tracks: list[dict[str, Any]],
    path: list[dict[str, Any]],
    path_width: float,
) -> None:
    """Ground cells: occupied near a track, free inside the corridor. Unknown stays void."""
    half = max(0.9, float(path_width or 2.0) * 0.5)
    cell = 1.6
    occupied = [
        (float(tr.get("x") or 0.0), float(tr.get("y") or 0.0), 2.1)
        for tr in tracks or []
    ]
    path_xy = [(float(p.get("x") or 0.0), float(p.get("y") or 0.0)) for p in path or []]
    overlay = img.copy()
    painted = False
    y = -2.0
    while y <= 48.0:
        x = -10.0
        while x <= 10.0:
            occ = any((x - ox) ** 2 + (y - oy) ** 2 < r * r for ox, oy, r in occupied)
            on_path = False
            if not occ and path_xy:
                for px, py in path_xy:
                    if abs(py - y) <= cell and abs(px - x) <= half + 0.35:
                        on_path = True
                        break
            if occ or on_path:
                p0 = cam.project(x - cell * 0.42, y - cell * 0.42, 0.02)
                p1 = cam.project(x + cell * 0.42, y - cell * 0.42, 0.02)
                p2 = cam.project(x + cell * 0.42, y + cell * 0.42, 0.02)
                p3 = cam.project(x - cell * 0.42, y + cell * 0.42, 0.02)
                cv2.fillPoly(
                    overlay,
                    [np.array([p0, p1, p2, p3], dtype=np.int32)],
                    OCC if occ else FREE,
                )
                painted = True
            x += cell
        y += cell
    if painted:
        cv2.addWeighted(overlay, 0.32, img, 0.68, 0, img)


def draw_frustums(img: np.ndarray, cam: Any, health: dict[str, Any] | None = None) -> None:
    health = health or {}
    overlay = img.copy()
    for fr in load_frustums():
        cid = str(fr.get("id") or "")
        ok = str(health.get(cid) or health.get("main" if cid == "main" else cid) or "") == "ok"
        yaw = math.radians(float(fr["yaw_deg"]))
        # GVD frame: +Y forward, +X right. yaw 0 = forward.
        heading = math.pi / 2 + yaw
        fov = math.radians(float(fr["fov_h"]))
        reach = float(fr["range_m"])
        x0, y0 = float(fr["x"]), float(fr["y"])
        left = heading - fov * 0.5
        right = heading + fov * 0.5
        p_origin = cam.project(x0, y0, 0.4)
        p_l = cam.project(x0 + math.cos(left) * reach, y0 + math.sin(left) * reach, 0.05)
        p_r = cam.project(x0 + math.cos(right) * reach, y0 + math.sin(right) * reach, 0.05)
        col = FOV if ok else (70, 66, 62)
        cv2.polylines(
            overlay,
            [np.array([p_origin, p_l, p_r], dtype=np.int32)],
            True,
            col,
            1,
            cv2.LINE_AA,
        )
        alpha = 0.16 if ok else 0.06
        layer = overlay.copy()
        cv2.fillPoly(layer, [np.array([p_origin, p_l, p_r], dtype=np.int32)], col)
        cv2.addWeighted(layer, alpha, overlay, 1.0 - alpha, 0, overlay)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)


def draw_planner_cost(
    img: np.ndarray,
    cam: Any,
    path: list[dict[str, Any]],
    path_width: float,
    tracks: list[dict[str, Any]],
) -> None:
    if not path:
        return
    half = max(0.9, float(path_width or 2.0) * 0.5)
    offsets = (-1.15, -0.55, 0.0, 0.55, 1.15)
    ys = (8.0, 16.0, 24.0, 32.0)
    for y in ys:
        cx = 0.0
        for p in path:
            cx = float(p.get("x") or 0.0)
            if float(p.get("y") or 0.0) >= y:
                break
        for off in offsets:
            x = cx + off * half
            cost = abs(off)
            for tr in tracks or []:
                dx = x - float(tr.get("x") or 0.0)
                dy = y - float(tr.get("y") or 0.0)
                if dx * dx + dy * dy < 6.0:
                    cost += 1.4
            t = max(0.0, min(1.0, cost / 2.2))
            col = tuple(int(a * (1.0 - t) + b * t) for a, b in zip(COST_COLD, COST_HOT))
            pt = cam.project(x, y, 0.12)
            rad = 4 if off == 0.0 else 3
            cv2.circle(img, pt, rad, col, -1, cv2.LINE_AA)


def draw_lane_polys(img: np.ndarray, cam: Any, lanes_bev: list) -> None:
    for i, poly in enumerate(lanes_bev or []):
        pts = []
        for p in poly or []:
            if isinstance(p, dict):
                pts.append(cam.project(float(p.get("x") or 0.0), float(p.get("y") or 0.0), 0.04))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append(cam.project(float(p[0]), float(p[1]), 0.04))
        if len(pts) < 2:
            continue
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, ICE_HI, 1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"L{i}",
            (pts[0][0] + 4, pts[0][1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            ICE_HI,
            1,
            cv2.LINE_AA,
        )


def draw_pip_boxes(pip: np.ndarray, dets: list[dict[str, Any]], src_wh: tuple[int, int]) -> None:
    if not dets:
        return
    ph, pw = pip.shape[:2]
    sw, sh = src_wh
    if sw <= 0 or sh <= 0:
        return
    sx, sy = pw / float(sw), ph / float(sh)
    for d in dets:
        xyxy = d.get("xyxy") if isinstance(d, dict) else None
        if not xyxy or len(xyxy) < 4:
            continue
        x1, y1, x2, y2 = [int(float(v)) for v in xyxy[:4]]
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx), int(y2 * sy))
        cv2.rectangle(pip, p1, p2, ICE_HI, 1)
        cls = str(d.get("cls") or d.get("class") or "")[:8]
        if cls:
            cv2.putText(pip, cls, (p1[0], max(12, p1[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, PAPER, 1, cv2.LINE_AA)


def draw_cam_strip(
    img: np.ndarray,
    frames: dict[str, Any] | None,
    health: dict[str, Any] | None,
    *,
    stage_w: int,
    stage_h: int,
    y0: int | None = None,
) -> int:
    """Row of tiny camera tiles along the bottom — missing feeds stay labelled missing.

    Returns the top y of the strip so the HUD can sit above it.
    """
    frames = frames or {}
    health = health or {}
    n = len(CAM_IDS)
    slot_w = min(150, max(72, (stage_w - 24) // n))
    slot_h = 64
    if y0 is None:
        y0 = stage_h - slot_h - 8
    x = 12
    for cid in CAM_IDS:
        frame = frames.get(cid) if cid != "main" else (frames.get("main") or frames.get("cam_main"))
        tile = np.full((slot_h, slot_w, 3), (22, 20, 18), dtype=np.uint8)
        ok = False
        if frame is not None and getattr(frame, "size", 0):
            try:
                tile = cv2.resize(frame, (slot_w, slot_h), interpolation=cv2.INTER_AREA)
                ok = True
            except Exception:
                ok = False
        status = str(health.get(cid) or ("ok" if ok else "missing"))
        if not ok:
            cv2.putText(tile, "missing", (6, slot_h // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, HUD_DIM, 1, cv2.LINE_AA)
        cv2.putText(tile, cid[:8], (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.28, PAPER if ok else HUD_DIM, 1, cv2.LINE_AA)
        img[y0 : y0 + slot_h, x : x + slot_w] = tile
        border = ICE_HI if status == "ok" else (50, 46, 44)
        cv2.rectangle(img, (x, y0), (x + slot_w - 1, y0 + slot_h - 1), border, 1)
        x += slot_w + 2
    return y0


def draw_dense_hud(
    img: np.ndarray,
    state: dict[str, Any],
    *,
    stage_w: int,
    stage_h: int,
    bottom_pad: int = 0,
) -> None:
    """Corner HUD: speed, engage/policy, TTC/AEB, last command."""
    ego = state.get("ego") or {}
    pl = state.get("planner") or {}
    try:
        v = float(ego.get("speed_mps") or 0.0)
    except (TypeError, ValueError):
        v = 0.0
    mph = v * 2.23694
    engaged = bool(state.get("engaged"))
    policy = str(state.get("policy") or "?")
    aeb = str(pl.get("aeb") or "off")
    ttc = pl.get("ttc_lead")
    ttc_s = f"{float(ttc):.1f}s" if ttc is not None else "—"
    cmd = f"s{float(ego.get('steer_deg') or 0):+.0f} t{float(ego.get('throttle') or 0):.1f} b{float(ego.get('brake') or 0):.1f}"
    base = stage_h - bottom_pad
    # Speed stack, lower-left.
    cv2.putText(img, f"{mph:.0f}", (18, base - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ICE_HI, 2, cv2.LINE_AA)
    cv2.putText(img, "mph", (18, base - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, HUD_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, f"{v:.1f} m/s", (18, base - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, HUD, 1, cv2.LINE_AA)
    # Right cluster.
    drive = "DRIVE" if engaged else "idle"
    cv2.putText(
        img,
        f"{policy}  {drive}",
        (stage_w - 220, base - 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        ICE_HI if engaged else HUD_DIM,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"TTC {ttc_s}  AEB {aeb}",
        (stage_w - 220, base - 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COST_HOT if aeb == "brake" else HUD,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        cmd,
        (stage_w - 220, base - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        HUD_DIM,
        1,
        cv2.LINE_AA,
    )
