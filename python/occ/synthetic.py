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


def scene(cam: CameraModel, *, road_half_w: float = 3.5, box: dict | None = None) -> dict[str, Any]:
    """RGB uint8, depth float32 metres, sem uint8, and the box (ego frame) for one frame."""
    box = box if box is not None else {"x": 0.0, "y0": 15.0, "w": 1.8, "h": 1.4, "l": 4.0}
    d = ground_depth(cam, box=box)
    g = ground_depth(cam)
    on_box = np.isfinite(d) & ~(np.isfinite(g) & np.isclose(d, g, rtol=1e-6))
    ego, valid = cam.backproject(np.nan_to_num(d, nan=0.0))
    x = ego[:, 0].reshape(d.shape)
    on_road = np.isfinite(d) & ~on_box & (np.abs(x) <= road_half_w)
    sem = np.zeros(d.shape, np.uint8)
    sem[np.isfinite(d)] = 2
    sem[on_road] = 1
    sem[on_box] = 3
    rgb = np.zeros(d.shape + (3,), np.uint8)
    rgb[...] = (110, 160, 230)          # sky
    rgb[sem == 2] = (60, 130, 50)       # grass
    rgb[sem == 1] = (70, 70, 75)        # asphalt
    rgb[sem == 3] = (200, 40, 40)       # car
    car = {"x": box["x"], "y": box["y0"] + box.get("l", 4.0) / 2, "yaw": np.pi / 2}
    return {"rgb": rgb, "depth": d.astype(np.float32), "sem": sem, "box": car}
