"""Monospace nerd overlay panel (toggle V)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

BG = (16, 13, 12)  # BGR ~ #0c0d10
FG = (212, 204, 200)  # ~ #c8ccd4
ICE = (212, 196, 158)


def render_panel(state: dict[str, Any], h: int = 720, w: int = 420) -> np.ndarray:
    img = np.full((h, w, 3), BG, dtype=np.uint8)
    lines = _lines(state)
    y = 28
    cv2.putText(img, "GVD NERD", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ICE, 1, cv2.LINE_AA)
    y += 28
    for line in lines:
        cv2.putText(img, line[:56], (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, FG, 1, cv2.LINE_AA)
        y += 18
        if y > h - 20:
            break
    return img


def _lines(s: dict[str, Any]) -> list[str]:
    ego = s.get("ego") or {}
    pl = s.get("planner") or {}
    cams = s.get("cam_health") or {}
    cam_s = " ".join(f"{k[0]}:{str(v)[:1]}" for k, v in cams.items())
    miss = s.get("missing_state_keys") or []
    return [
        f"policy: {s.get('policy', '?')}  engage: {s.get('engaged')}",
        f"disengage: {s.get('disengage_reason', 'none')}",
        f"loop {s.get('loop_hz', 0):.1f}Hz cam {s.get('camera_hz', 0):.1f}Hz",
        f"infer {s.get('infer_ms', 0):.1f}ms viz {s.get('viz_ms', 0):.1f}ms hb {s.get('heartbeat_ms', 0):.0f}ms",
        f"VRAM {s.get('gpu_vram_used_gb', 0):.1f}/{s.get('gpu_vram_total_gb', 11):.0f} GB  {s.get('gpu_name', '')}",
        f"cams {cam_s}",
        f"lane {s.get('lane_conf', 0):.2f} bev {s.get('bev_coverage', 0):.0%} obj {s.get('objects_n', 0)} trk {s.get('tracks_n', 0)}",
        f"ego v={ego.get('speed_mps', 0):.1f}m/s steer={ego.get('steer_deg', 0):.1f} thr={ego.get('throttle', 0):.2f} brk={ego.get('brake', 0):.2f}",
        f"yawr={ego.get('yaw_rate', 0):.2f} accel={ego.get('accel', 0):.2f}",
        f"planner w={pl.get('corridor_width', 0):.2f} curv={pl.get('curvature', 0):.4f} vt={pl.get('target_v', 0):.1f}",
        f"TTC {pl.get('ttc_lead')}  AEB {pl.get('aeb')}",
        f"path_conf {s.get('path_conf', 0):.2f} width {s.get('path_width', 0):.2f} preview={s.get('path_debug_preview')}",
        f"clip {s.get('last_clip_trigger', 'none')}",
        f"keys 1 occ 2 yolo 3 lanes 4 frust 5 cost 0 clean  V nerd",
        "missing: " + (", ".join(miss) if miss else "none"),
    ]
