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

from python.sensors.cameras import CAM_IDS, frame_is_unrendered

# Windshield cluster — Tech attach can blow these white; side/rear stay as-is.
FRONT_CAM_IDS = frozenset({"narrow", "main", "wide", "cam_main"})
CAM_TILE_GAP = 4
_OVEREXPOSE_MEAN = 165.0
_OVEREXPOSE_TARGET = 140.0

OCC = (88, 86, 196)
FREE = (120, 150, 92)
FOV = (168, 156, 120)
COST_COLD = (212, 196, 158)
COST_HOT = (92, 92, 196)
HUD = (212, 204, 200)
HUD_DIM = (120, 120, 120)
ICE_HI = (248, 238, 214)
AMBER = (74, 165, 216)  # #d8a54a
MISMATCH_RED = (112, 112, 208)
PAPER = (225, 230, 232)


def cabin_drive_word(state: dict[str, Any]) -> tuple[str, tuple[int, int, int]]:
    """Same glance word as the in-game app: OFF / ENGAGED is ON here, HOLD, DRIVE, MISMATCH.

    Preview, veto, and AEB are HOLD. DRIVE only when engaged and the command is a live apply.
    """
    link = str(state.get("link") or "")
    bus = str(state.get("bus_link") or "")
    if link == "mismatch" or bus == "MISMATCH":
        return "MISMATCH", MISMATCH_RED
    if not state.get("engaged"):
        return "OFF", HUD_DIM
    reason = str(state.get("cmd_reason") or "")
    planner = state.get("planner") if isinstance(state.get("planner"), dict) else {}
    aeb = str((planner or {}).get("aeb") or "off")
    veto = str(state.get("veto_reason") or "none")
    if (
        reason in ("preview_blocked", "heartbeat_stale", "bus_mismatch")
        or reason.startswith("veto:")
        or aeb in ("brake", "warn")
        or veto == "e2e_stub"
    ):
        return "HOLD", AMBER
    applied = bool(state.get("cmd_applied"))
    lua_applying = bool(state.get("lua_applying"))
    actuator = str(state.get("actuator") or "")
    # Pending retail writes are armed, not driving. DRIVE needs a live apply.
    if lua_applying or (actuator == "beamngpy" and applied) or (reason == "cmd_json_applied" and applied):
        return "DRIVE", ICE_HI
    if reason in ("ok", "plan") and applied:
        return "DRIVE", ICE_HI
    return "ON", (212, 196, 158)

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


def clamp_front_overexpose(frame: np.ndarray | None, cid: str = "main") -> np.ndarray | None:
    """Scale down blown-white Tech front previews. Side/rear and missing feeds are untouched."""
    if frame is None or not getattr(frame, "size", 0):
        return frame
    key = "main" if cid == "cam_main" else str(cid or "main")
    if key not in FRONT_CAM_IDS:
        return frame
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3:
        return frame
    mean = float(frame.mean())
    if mean <= _OVEREXPOSE_MEAN:
        return frame
    scale = _OVEREXPOSE_TARGET / mean
    return np.clip(frame.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def cam_slot_live(frames: dict[str, Any] | None, health: dict[str, Any] | None, cid: str) -> bool:
    """True when this slot has a real picture and health says ok.

    Health ``ok`` plus an all-zero buffer is not 8/8. The tile stays labelled.
    """
    status = str((health or {}).get(cid) or "")
    if status != "ok":
        return False
    return not frame_is_unrendered(cam_frame_for(frames, cid))


def cam_frame_for(frames: dict[str, Any] | None, cid: str) -> Any:
    """Look up one slot. `main` also accepts `cam_main`. Never synthesizes a missing feed."""
    if not frames:
        return None
    frame = frames.get(cid)
    if cid == "main" and (frame is None or not getattr(frame, "size", 0)):
        frame = frames.get("cam_main")
    return frame


def cam_tile_rects(
    x0: int,
    y0: int,
    width: int,
    height: int,
    cols: int,
    rows: int,
    *,
    gap: int = CAM_TILE_GAP,
) -> list[tuple[int, int, int, int]]:
    """(x, y, w, h) for a cols×rows grid. Caller paints at most 8 slots."""
    cols = max(1, int(cols))
    rows = max(1, int(rows))
    gap = max(0, int(gap))
    slot_w = max(1, (int(width) - gap * (cols - 1)) // cols)
    slot_h = max(1, (int(height) - gap * (rows - 1)) // rows)
    out: list[tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            x = int(x0) + c * (slot_w + gap)
            y = int(y0) + r * (slot_h + gap)
            out.append((x, y, slot_w, slot_h))
    return out


def _paint_cam_tile(
    slot_w: int,
    slot_h: int,
    cid: str,
    frame: Any,
    status: str,
    *,
    dropped: bool,
) -> tuple[np.ndarray, str, bool]:
    """Build one tile. Missing/dropped stay labelled; never invents pixels from another cam."""
    tile = np.full((slot_h, slot_w, 3), (22, 20, 18), dtype=np.uint8)
    # A zero buffer is not a camera picture. Leave the slot labelled.
    if frame_is_unrendered(frame):
        frame = None
    has = frame is not None and getattr(frame, "size", 0)
    ok = False
    if dropped:
        label = "dropped" if has else "missing"
    elif has:
        try:
            src = clamp_front_overexpose(frame, cid)
            if src is None or not getattr(src, "size", 0):
                label = str(status or "missing")
            else:
                tile = cv2.resize(src, (slot_w, slot_h), interpolation=cv2.INTER_AREA)
                ok = True
                label = str(status or "ok")
        except Exception:
            label = str(status or "error")
    else:
        label = str(status or "missing")
        if label == "ok":
            label = "missing"
    return tile, label, ok


def draw_cam_tiles(
    img: np.ndarray,
    frames: dict[str, Any] | None,
    health: dict[str, Any] | None,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
    cols: int = 4,
    rows: int = 2,
    ids: tuple[str, ...] | None = None,
    dropped: bool = False,
) -> int:
    """Paint ≤8 honest camera tiles. Empty slots stay labelled; a slow loop may skip the blit.

    Returns how many slots were drawn. Never raises on a bad/missing frame.
    """
    health = health or {}
    ids = tuple(ids or CAM_IDS)[:8]
    rects = cam_tile_rects(x0, y0, width, height, cols, rows)
    ih, iw = img.shape[:2]
    n = 0
    for cid, rect in zip(ids, rects):
        x, y, slot_w, slot_h = rect
        x2 = min(iw, x + slot_w)
        y2 = min(ih, y + slot_h)
        if x >= iw or y >= ih or x2 <= max(0, x) or y2 <= max(0, y):
            continue
        x = max(0, x)
        y = max(0, y)
        tw, th = x2 - x, y2 - y
        status = str(health.get(cid) or "")
        raw = cam_frame_for(frames, cid)
        tile, label, ok = _paint_cam_tile(tw, th, cid, raw, status, dropped=dropped)
        fs_id = 0.28 if th < 80 else 0.45
        fs_st = 0.32 if th < 80 else 0.50
        live = ok and label == "ok"
        if not ok:
            cv2.putText(
                tile,
                label[:10],
                (6, th // 2 + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs_st,
                HUD_DIM,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            tile,
            cid[:8],
            (4, 12 if th < 80 else 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs_id,
            PAPER if live else HUD_DIM,
            1,
            cv2.LINE_AA,
        )
        if ok and label not in ("ok",) and th >= 80:
            cv2.putText(
                tile,
                label[:10],
                (4, th - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                HUD_DIM,
                1,
                cv2.LINE_AA,
            )
        img[y:y2, x:x2] = tile
        border = ICE_HI if live else (50, 46, 44)
        cv2.rectangle(img, (x, y), (x2 - 1, y2 - 1), border, 1)
        n += 1
    return n


def draw_cam_strip(
    img: np.ndarray,
    frames: dict[str, Any] | None,
    health: dict[str, Any] | None,
    *,
    stage_w: int,
    stage_h: int,
    y0: int | None = None,
    dropped: bool = False,
) -> int:
    """Row of tiny camera tiles along the bottom — missing feeds stay labelled missing.

    Returns the top y of the strip so the HUD can sit above it. Under 8 Hz the caller
    should pass dropped=True (or skip the strip); labelled slots still render.
    """
    n = len(CAM_IDS)
    slot_w = min(150, max(72, (stage_w - 24) // n))
    slot_h = 64
    if y0 is None:
        y0 = stage_h - slot_h - 8
    draw_cam_tiles(
        img,
        frames,
        health,
        x0=12,
        y0=y0,
        width=n * (slot_w + 2) - 2,
        height=slot_h,
        cols=n,
        rows=1,
        ids=CAM_IDS,
        dropped=dropped,
    )
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
    drive, drive_col = cabin_drive_word(state)
    cv2.putText(
        img,
        f"{policy}  {drive}",
        (stage_w - 220, base - 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        drive_col,
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
