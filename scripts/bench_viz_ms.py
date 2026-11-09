#!/usr/bin/env python3
"""Report CPU viz milliseconds: legacy per-quad copies vs current render_stage."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def legacy_per_quad_ms(width: int = 640, height: int = 400, nseg: int = 36, loops: int = 6) -> float:
    import cv2

    img = np.zeros((height, width, 3), dtype=np.uint8)
    times = []
    for _ in range(loops):
        work = img.copy()
        t0 = time.perf_counter()
        for i in range(nseg):
            overlay = work.copy()
            cv2.rectangle(overlay, (10 + i, 8), (18 + i, height - 8), (234, 126, 61), -1)
            cv2.addWeighted(overlay, 0.45, work, 0.55, 0, work)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def once_blend_ms(width: int = 1280, height: int = 800, nseg: int = 36, loops: int = 6) -> float:
    import cv2

    img = np.zeros((height, width, 3), dtype=np.uint8)
    times = []
    for _ in range(loops):
        t0 = time.perf_counter()
        overlay = np.zeros_like(img)
        for i in range(nseg):
            cv2.rectangle(overlay, (10 + i, 8), (18 + i, height - 8), (234, 126, 61), -1)
        work = img.copy()
        cv2.addWeighted(overlay, 0.45, work, 0.55, 0, work)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def render_ms(width: int = 640, height: int = 400, loops: int = 6) -> tuple[float, np.ndarray]:
    from python.viz.stage import VizUI, render_stage

    st = {
        "engaged": False,
        "loop_hz": 12.0,
        "policy": "modular",
        "path_conf": 0.95,
        "path_width": 2.0,
        "ego": {"speed_mps": 8.0, "brake": 0.0},
        "planner": {"aeb": "off", "target_v": 8.0},
        "path_ego": [{"x": 0.1 * (i / 10.0), "y": float(i)} for i in range(0, 37)],
        "tracks": [],
        "lanes_ext": [],
        "road_edges": [],
        "signs": [],
        "scene_ckpt": {
            "confidence": 0.9,
            "centers": [
                {"x": 0.0, "y": 5.0},
                {"x": 0.4, "y": 10.0},
                {"x": 1.2, "y": 20.0},
                {"x": 2.0, "y": 30.0},
            ],
        },
    }
    ui = VizUI()
    ui.show_nerd = False
    ui.layers = {0}
    render_stage(dict(st), ui=ui, width=width, height=height)
    times = []
    img = None
    for _ in range(loops):
        t0 = time.perf_counter()
        img = render_stage(dict(st), ui=ui, width=width, height=height)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)), img


def main() -> int:
    before = legacy_per_quad_ms(1280, 800)
    once = once_blend_ms(1280, 800)
    after, img = render_ms()
    after1280, _ = render_ms(1280, 800)
    print(f"device: CPU")
    print(f"before (legacy per-quad copy+blend, 1280x800, 36 quads): {before:.2f} ms")
    print(f"after  (single overlay blend, 1280x800, 36 quads):       {once:.2f} ms")
    print(f"after  (render_stage clip view 640x400):                 {after:.2f} ms")
    print(f"after  (render_stage clip view 1280x800):                {after1280:.2f} ms")
    print(f"frame shape 640: {img.shape if img is not None else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
