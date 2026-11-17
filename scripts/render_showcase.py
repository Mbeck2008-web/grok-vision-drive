#!/usr/bin/env python3
"""Offline showcase stills and a short clip from a synthetic drive.

    python scripts/render_showcase.py --out docs/images/viz_showcase
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _scene(t: float) -> dict:
    y_lead = 12.0 + 6.0 * t
    path = []
    for i, y in enumerate(np.linspace(2.0, 48.0, 24)):
        path.append({"x": 1.4 * math.sin((y / 12.0) + t), "y": float(y)})
    lanes = []
    for x, idx in ((-1.8, -1), (1.8, 1), (-5.4, -2), (5.4, 2)):
        lanes.append({
            "points": [
                {"x": x + 0.6 * math.sin(y / 18.0 + t), "y": float(y)}
                for y in range(2, 64, 2)
            ],
            "kind": "detected" if abs(idx) == 1 else "predicted",
            "index": idx,
        })
    return {
        "engaged": True,
        "path_conf": 0.9,
        "ego": {"speed_mps": 8.0 + 4.0 * t},
        "planner": {"cipv_id": 1, "aeb": "off", "target_v": 12.0},
        "cmd_reason": "cmd_json_applied",
        "cmd_applied": True,
        "lua_applying": True,
        "path_ego": path,
        "lanes_ext": lanes,
        "road_edges": [
            {"points": [{"x": -8.5, "y": float(y)} for y in range(2, 64, 2)], "kind": "detected"},
            {"points": [{"x": 8.5, "y": float(y)} for y in range(2, 64, 2)], "kind": "detected"},
        ],
        "tracks": [
            {"id": 1, "class": "vehicle", "x": 0.2 * math.sin(t * 3), "y": y_lead, "yaw": 1.57},
            {"id": 2, "class": "vehicle", "x": 3.8, "y": 18.0 + t, "yaw": 1.57},
            {"id": 3, "class": "truck", "x": -4.5, "y": 22.0, "yaw": 1.57},
        ],
        "signs": [
            {"cls": "traffic_light", "x": 1.5, "y": 40.0, "state": "green"},
        ],
    }


def _dusk(h: int, w: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        u = y / max(1, h - 1)
        img[y, :] = (int(28 + 20 * (1 - u)), int(36 + 24 * (1 - u)), int(48 + 30 * (1 - u)))
    img[int(h * 0.45):, :] = (26, 30, 34)
    return img


def main() -> int:
    from python.offline.label_review import encode_mp4
    from python.viz.showcase import PRESETS, lerp_state, render_showcase
    from python.viz.stage import VizUI

    dest = ROOT / "docs" / "images" / "viz_showcase"
    dest.mkdir(parents=True, exist_ok=True)
    ui = VizUI()
    ui.show_nerd = False
    ui.style = "early"
    ui.showcase = True
    cam = _dusk(720, 1280)
    for name, (w, h) in PRESETS.items():
        if name == "stage":
            continue
        ui.showcase_layout = "full"
        frame = render_showcase(_scene(0.4), ui, width=w, height=h, main_frame=cam, card=(name == "wide"))
        cv2.imwrite(str(dest / f"{name}.png"), frame)
        print(dest / f"{name}.png", frame.shape)
    ui.showcase_layout = "split"
    split = render_showcase(_scene(0.4), ui, width=1280, height=720, main_frame=cam)
    cv2.imwrite(str(dest / "split.png"), split)
    ui.showcase_layout = "pip"
    pip = render_showcase(_scene(0.4), ui, width=1280, height=720, main_frame=cam)
    cv2.imwrite(str(dest / "pip.png"), pip)
    ui.showcase_layout = "full"
    a, b = _scene(0.0), _scene(1.0)
    frames = []
    steps = 12
    for i in range(steps):
        st = lerp_state(a, b, i / (steps - 1))
        frames.append(render_showcase(st, ui, width=1280, height=720, main_frame=_dusk(720, 1280), card=(i == 0)))
    ok = encode_mp4(frames, dest / "demo.mp4", fps=12.0)
    print(dest / "demo.mp4", "ok" if ok else "encode failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
