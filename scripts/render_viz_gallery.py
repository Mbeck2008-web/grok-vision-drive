#!/usr/bin/env python3
"""Write offline sample frames for the vector stage. Not a live capture."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _ui():
    from python.viz.stage import VizUI

    ui = VizUI()
    ui.show_nerd = False
    ui.style = "early"
    return ui


def _pts(fn, y0, y1, step=2.0):
    out = []
    y = float(y0)
    while y <= y1 + 1e-6:
        out.append({"x": float(fn(y)), "y": round(y, 2)})
        y += step
    return out


def intersection():
    def bend(x0):
        return lambda y: x0 + 4.2 * math.sin(y / 18.0)

    lanes = []
    for x, idx in ((-1.7, -1), (1.7, 1), (-5.2, -2), (5.2, 2)):
        lanes.append({
            "points": _pts(bend(x), 2, 70),
            "kind": "detected" if abs(idx) == 1 else "predicted",
            "index": idx,
            "side": "left" if x < 0 else "right",
        })
    # Cross street.
    lanes.append({
        "points": [{"x": x, "y": 28.0 + 0.15 * abs(x)} for x in np.linspace(-14, 14, 16)],
        "kind": "detected",
        "index": 3,
        "side": "right",
    })
    edges = [
        {"points": _pts(bend(-8.6), 2, 70), "kind": "detected", "side": "left"},
        {"points": _pts(bend(8.6), 2, 70), "kind": "detected", "side": "right"},
    ]
    tracks = [
        {"id": 1, "class": "vehicle", "x": 0.2, "y": 16, "yaw": 1.57},
        {"id": 2, "class": "vehicle", "x": 4.6, "y": 18, "yaw": 1.7},
        {"id": 3, "class": "vehicle", "x": -4.4, "y": 14, "yaw": 1.45},
        {"id": 4, "class": "truck", "x": -7.2, "y": 22, "yaw": 1.2},
        {"id": 5, "class": "pedestrian", "x": 6.4, "y": 12, "yaw": 3.0},
        {"id": 6, "class": "bike", "x": 7.5, "y": 20, "yaw": 1.57},
        {"id": 7, "class": "vehicle", "x": 3.8, "y": 30, "yaw": 0.4},
    ]
    signs = [
        {"cls": "traffic_light", "x": -2.2, "y": 32, "state": "red"},
        {"cls": "traffic_light", "x": 2.4, "y": 33, "state": "green"},
        {"cls": "traffic_light", "x": 6.0, "y": 31, "state": "red"},
        {"cls": "stop_sign", "x": -6.5, "y": 26},
    ]
    path = _pts(lambda y: 0.15 * y * math.sin(y / 14.0), 2, 48, 1.5)
    return {
        "engaged": True,
        "path_conf": 0.9,
        "ego": {"speed_mps": 8.5},
        "planner": {"cipv_id": 1, "aeb": "off", "target_v": 9.0},
        "cmd_reason": "cmd_json_applied",
        "cmd_applied": True,
        "lua_applying": True,
        "path_ego": path,
        "lanes_ext": lanes,
        "road_edges": edges,
        "tracks": tracks,
        "signs": signs,
    }


def multilane():
    lanes = []
    for x, idx in ((-5.4, -2), (-1.8, -1), (1.8, 1), (5.4, 2)):
        lanes.append({
            "points": _pts(lambda y, x=x: x + 0.4 * math.sin(y / 22.0), 2, 80),
            "kind": "detected",
            "index": idx,
            "side": "left" if x < 0 else "right",
        })
    edges = [
        {"points": _pts(lambda y: -9.0, 2, 80), "kind": "detected"},
        {"points": _pts(lambda y: 9.0, 2, 80), "kind": "detected"},
    ]
    tracks = [
        {"id": 1, "class": "vehicle", "x": 0.1, "y": 14, "yaw": 1.57},
        {"id": 2, "class": "vehicle", "x": 3.6, "y": 12, "yaw": 1.57},
        {"id": 3, "class": "vehicle", "x": -3.6, "y": 18, "yaw": 1.57},
        {"id": 4, "class": "vehicle", "x": 3.7, "y": 26, "yaw": 1.57},
        {"id": 5, "class": "truck", "x": -7.2, "y": 20, "yaw": 1.57},
        {"id": 6, "class": "vehicle", "x": 7.2, "y": 24, "yaw": 1.57},
        {"id": 7, "class": "pedestrian", "x": 8.4, "y": 10, "yaw": 0.2},
    ]
    signs = [
        {"cls": "traffic_light", "x": -3.0, "y": 48, "state": "green"},
        {"cls": "traffic_light", "x": 0.2, "y": 50, "state": "green"},
        {"cls": "traffic_light", "x": 3.2, "y": 48, "state": "red"},
    ]
    return {
        "engaged": True,
        "path_conf": 0.88,
        "ego": {"speed_mps": 16.0},
        "planner": {"cipv_id": 1, "target_v": 15.0, "aeb": "off"},
        "cmd_reason": "cmd_json_applied",
        "cmd_applied": True,
        "lua_applying": True,
        "path_ego": _pts(lambda y: 0.35 * math.sin(y / 16.0), 2, 55, 1.5),
        "lanes_ext": lanes,
        "road_edges": edges,
        "tracks": tracks,
        "signs": signs,
    }


def parking():
    tracks = []
    tid = 1
    for row, y in enumerate((8, 13, 18, 23)):
        for col, x in enumerate((-7.5, -4.2, -1.0, 2.4, 5.6)):
            if row == 1 and col == 2:
                continue
            cls = "obstacle" if (row + col) % 3 else "truck"
            if row == 0 and col == 4:
                cls = "pedestrian"
            tracks.append({
                "id": tid,
                "class": cls,
                "x": x,
                "y": y,
                "yaw": 0.15 if col % 2 == 0 else math.pi - 0.1,
            })
            tid += 1
    tracks.append({"id": 99, "class": "vehicle", "x": 0.2, "y": 16, "yaw": 1.57})
    edges = []
    for x in (-9.5, 8.5):
        edges.append({"points": _pts(lambda y, x=x: x, 2, 40), "kind": "detected"})
    edges.append({"points": [{"x": x, "y": 6.0} for x in np.linspace(-9, 8, 14)], "kind": "detected"})
    return {
        "engaged": False,
        "path_conf": 0.7,
        "ego": {"speed_mps": 3.2},
        "planner": {"cipv_id": 99, "target_v": 4.0, "aeb": "off"},
        "path_ego": _pts(lambda y: 1.8 * math.sin((y - 4) / 6.0), 3, 28, 1.2),
        "lanes_ext": [
            {"points": _pts(lambda y: -1.6, 4, 30), "kind": "detected", "index": -1},
            {"points": _pts(lambda y: 1.6, 4, 30), "kind": "detected", "index": 1},
        ],
        "road_edges": edges,
        "tracks": tracks,
        "signs": [],
    }


def dusk_frame(h=720, w=1280):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / max(1, h - 1)
        sky = np.array([40, 50, 70]) * (1.0 - t) + np.array([18, 22, 30]) * t
        img[y, :] = sky.astype(np.uint8)
    # Road plane-ish dark band.
    img[int(h * 0.42):, :] = (28, 32, 36)
    return img


def main() -> None:
    from python.viz.stage import render_stage

    dest = ROOT / "docs" / "images" / "viz_early_fsd"
    dest.mkdir(parents=True, exist_ok=True)
    ui = _ui()
    scenes = {
        "intersection": intersection(),
        "multilane_lights": multilane(),
        "parking": parking(),
    }
    for name, st in scenes.items():
        frame = render_stage(st, ui=ui, width=1280, height=800)
        path = dest / f"{name}.png"
        cv2.imwrite(str(path), frame)
        print(path, frame.shape)
    ov = _ui()
    ov.overlay = True
    frame = render_stage(multilane(), ui=ov, width=1280, height=720, main_frame=dusk_frame())
    path = dest / "overlay_dusk.png"
    cv2.imwrite(str(path), frame)
    print(path, frame.shape)


if __name__ == "__main__":
    main()
