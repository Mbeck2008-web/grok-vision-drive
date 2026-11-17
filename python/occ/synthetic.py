"""Synthetic scene for tests: flat road + one car-sized box, rendered analytically.

Gives the RGB a camera would see plus the depth / semantic planes the
recorder would store, so labels, the overfit test and the polyline
tolerance test share one known geometry. Test support only.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from python.occ.camera import CameraModel

SEM_CLASSES = {"kind": "gvd_sem_classes", "train_only": True, "classes": {"0": "unknown", "1": "ASPHALT", "2": "Grass", "3": "PROP"}}


def ground_depth(cam: CameraModel, ground_z: float = 0.0, box: dict | None = None) -> np.ndarray:
    """Analytic z-depth of a flat ground plane (and an optional upright box face); NaN where nothing is hit."""
    vv, uu = np.meshgrid(np.arange(cam.height) + 0.5, np.arange(cam.width) + 0.5, indexing="ij")
    rays = np.stack([(uu - cam.width / 2) / cam.f, (vv - cam.height / 2) / cam.f, np.ones_like(uu)], axis=-1)
    ego_dir = rays @ cam.basis()
    pz = cam.pos[2] - ground_z
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(ego_dir[..., 2] < -1e-6, pz / -ego_dir[..., 2], np.inf)
    if box is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            sb = (box["y0"] - cam.pos[1]) / ego_dir[..., 1]
        hx = cam.pos[0] + sb * ego_dir[..., 0]
        hz = cam.pos[2] + sb * ego_dir[..., 2]
        hit = (sb > 0) & (np.abs(hx - box["x"]) < box["w"] / 2) & (hz >= ground_z) & (hz < ground_z + box["h"])
        s = np.where(hit & (sb < s), sb, s)
    return np.where(np.isfinite(s) & (s < 300), s, np.nan)


# World-frame paint: a solid yellow centre line and a dashed white lane line (3 m dash, 9 m period),
# so a forward-moving camera sees longitudinal texture (VO needs it).
PAINT = [{"x0": -1.75, "y0": -20.0, "x1": -1.75, "y1": 120.0, "yellow": True}] + [
    {"x0": 1.75, "y0": float(y), "x1": 1.75, "y1": float(y) + 3.0, "yellow": False} for y in range(-18, 120, 9)]
STOP_BARS = [{"y_m": 30.0, "x0": -1.75, "x1": 1.75, "valid": True, "src": "stop_line"}]
PAINT_HALF_W = 0.2


def scene(cam: CameraModel, *, road_half_w: float = 3.5, box: dict | None = None, paint: bool = True, ego_y: float = 0.0,
          texture: bool = True, leg: dict | None = None, with_box: bool = True) -> dict[str, Any]:
    """RGB uint8, depth float32 metres, sem uint8, the box, paint strokes and stop bars for one camera.

    ``ego_y``: how far the ego has driven forward (+y) through this static world;
    the box and the stop bar move back by that much in the ego frame.
    ``texture``: a faint world-fixed asphalt pattern (real asphalt is never
    uniform); RGB only, labels are unaffected.
    """
    box = dict(box) if box is not None else {"x": 0.0, "y0": 15.0, "w": 1.8, "h": 1.4, "l": 4.0}
    box["y0"] = box["y0"] - ego_y
    bars = [{**b, "y_m": b["y_m"] - ego_y} for b in STOP_BARS]
    strokes = [{**p, "y0": p["y0"] - ego_y, "y1": p["y1"] - ego_y} for p in PAINT]
    d = ground_depth(cam, box=box if with_box else None)
    g = ground_depth(cam)
    on_box = np.isfinite(d) & ~(np.isfinite(g) & np.isclose(d, g, rtol=1e-6))
    ego, valid = cam.backproject(np.nan_to_num(d, nan=0.0))
    x = ego[:, 0].reshape(d.shape)
    y = ego[:, 1].reshape(d.shape)
    on_road = np.isfinite(d) & ~on_box & (np.abs(x) <= road_half_w)
    if leg is not None:
        yw0 = y + ego_y
        on_road |= np.isfinite(d) & ~on_box & (x >= leg["x0"]) & (x <= leg["x1"]) & (yw0 >= leg["y0"]) & (yw0 <= leg["y1"])
    sem = np.zeros(d.shape, np.uint8)
    sem[np.isfinite(d)] = 2
    sem[on_road] = 1
    sem[on_box] = 3
    rgb = np.zeros(d.shape + (3,), np.uint8)
    rgb[...] = (110, 160, 230)          # sky
    rgb[sem == 2] = (60, 130, 50)       # grass
    rgb[sem == 1] = (70, 70, 75)        # asphalt
    if texture:  # world-anchored low-contrast asphalt texture (static in the world, moves with ego_y in the image)
        yw = y + ego_y
        tex = (np.sin(2.3 * x + 1.7 * yw) + np.sin(-0.9 * x + 2.9 * yw + 1.0) + np.sin(0.5 * x + 4.1 * yw + 2.0)
               + np.sin(5.3 * x - 1.1 * yw + 0.5))
        t8 = np.clip(70.0 + 4.0 * tex, 0, 255).astype(np.uint8)
        rgb[sem == 1] = np.stack([t8, t8, np.clip(t8.astype(np.int16) + 5, 0, 255).astype(np.uint8)], -1)[sem == 1]
    if paint:
        for seg in strokes:
            m = on_road & (np.abs(x - seg["x0"]) <= PAINT_HALF_W) & (y >= seg["y0"]) & (y <= seg["y1"])
            rgb[m] = (230, 190, 30) if seg["yellow"] else (235, 235, 235)
        for bar in bars:
            m = on_road & (np.abs(y - bar["y_m"]) <= 0.3) & (x >= bar["x0"]) & (x <= bar["x1"])
            rgb[m] = (240, 240, 240)
    rgb[sem == 3] = (200, 40, 40)       # car
    car = {"x": box["x"], "y": box["y0"] + box.get("l", 4.0) / 2, "yaw": np.pi / 2, "vx": 0.0, "vy": 0.0} if with_box else None
    return {"rgb": rgb, "depth": d.astype(np.float32), "sem": sem, "box": car,
            "paint": strokes if paint else [], "stop_bars": bars if paint else []}
