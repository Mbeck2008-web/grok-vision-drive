"""Monospace nerd overlay (toggle V). Help via ?."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

BG = (16, 13, 12)
FG = (212, 204, 200)
ICE = (212, 196, 158)
DIM = (120, 120, 120)


def render_panel(state: dict[str, Any], h: int = 720, w: int = 420, show_help: bool = False) -> np.ndarray:
    img = np.full((h, w, 3), BG, dtype=np.uint8)
    lines = _help_lines() if show_help else _lines(state)
    y = 28
    cv2.putText(img, "GVD  ·  VISION", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ICE, 1, cv2.LINE_AA)
    y += 26
    for line in lines:
        col = DIM if line.startswith(" ") or line.startswith("keys") else FG
        cv2.putText(img, line[:58], (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
        y += 17
        if y > h - 24:
            break
    cv2.putText(img, "V nerd  ? help  0 clean  q quit", (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, DIM, 1, cv2.LINE_AA)
    return img


def _help_lines() -> list[str]:
    return [
        "keys:",
        "  V  toggle this nerd panel",
        "  0  clean cabin (stage only)",
        "  1  occupancy grid",
        "  2  YOLO boxes in PIP",
        "  3  lane polynomials",
        "  4  camera frustums",
        "  5  planner cost samples",
        "  T  toggle BEV debug / chase 3/4 bird",
        "  q  quit",
    ]


def _lines(s: dict[str, Any]) -> list[str]:
    ego = s.get("ego") or {}
    pl = s.get("planner") or {}
    cams = s.get("cam_health") or {}
    cam_s = " ".join(f"{k}:{str(v)[:1]}" for k, v in cams.items())
    cap = s.get("capture_backend") or "?"
    cap_note = s.get("capture_note") or ""
    v = float(ego.get("speed_mps") or 0)
    mph = v * 2.23694
    miss = s.get("missing_state_keys") or []
    lead = None
    for tr in s.get("tracks") or []:
        if str(tr.get("class", "vehicle")) == "vehicle" and float(tr.get("y", 0)) > 5:
            lead = tr
            break
    sel = []
    if lead:
        # miss distance stub at 1/2/3s along ego x=0 corridor
        for t in (1, 2, 3):
            pred_y = float(lead.get("y", 0)) + float(lead.get("speed_mps", 0)) * t
            miss_m = abs(float(lead.get("x", 0)))
            sel.append(f"miss@{t}s={miss_m:.1f}m (lead y~{pred_y:.0f})")
    return [
        f"policy: {s.get('policy', '?')}   engage: {s.get('engaged')}",
        f"disengage: {s.get('disengage_reason', 'none')}",
        f"loop {s.get('loop_hz', 0):.1f}Hz  cam {s.get('camera_hz', 0):.1f}Hz",
        f"infer {s.get('infer_ms', 0):.1f}ms  viz {s.get('viz_ms', 0):.1f}ms  hb {s.get('heartbeat_ms', 0):.0f}ms",
        f"VRAM {s.get('gpu_vram_used_gb', 0):.1f}/{s.get('gpu_vram_total_gb', 11):.0f} GB  {s.get('gpu_name', '')}",
        f"cams {cam_s}",
        f"capture {cap} {cap_note}".rstrip(),
        f"rss {float(s.get('rss_mb') or 0):.0f} MB",
        f"lane {s.get('lane_conf', 0):.2f}  bev {s.get('bev_coverage', 0):.0%}  obj {s.get('objects_n', 0)}  trk {s.get('tracks_n', 0)}",
        f"ego {v:.1f} m/s ({mph:.0f} mph)  steer {ego.get('steer_deg', 0):.1f}",
        f"thr {ego.get('throttle', 0):.2f}  brk {ego.get('brake', 0):.2f}  yawr {ego.get('yaw_rate', 0):.2f}",
        f"planner w={pl.get('corridor_width', 0):.2f}  curv={pl.get('curvature', 0):.4f}  vt={pl.get('target_v', 0):.1f}",
        f"TTC {pl.get('ttc_lead')}   AEB {pl.get('aeb')}",
        f"path_conf {s.get('path_conf', 0):.2f}  width {s.get('path_width', 0):.2f}  preview={s.get('path_debug_preview')}",
        f"clip {s.get('last_clip_trigger', 'none')}",
        *sel,
        "missing: " + (", ".join(miss) if miss else "none"),
    ]
