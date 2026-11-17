"""Tiny analytic scene for the occupancy tests: a ground plane and one box.

Renders planar depth (uint16 cm), class ids, and RGB for any ``PinholeCam``
by ray casting. No BeamNG, no GPU.
"""
from __future__ import annotations

import numpy as np

from python.perception.occ_geometry import PinholeCam, pixel_rays

GROUND_Z = 0.2          # mid-cell of the default 0.4 m grid (z cell 5)
BOX_LO = (-1.0, 10.0, 0.2)
BOX_HI = (1.0, 14.4, 1.8)
SEM_CLASSES = {"kind": "gvd_sem_classes", "train_only": True,
               "classes": {"0": "unknown", "1": "ASPHALT", "2": "CAR", "3": "SKY"}}
SEM_GROUND, SEM_BOX, SEM_SKY = 1, 2, 3


def _box_hit(o: np.ndarray, d: np.ndarray, lo, hi) -> np.ndarray:
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t0 = (lo - o) * inv
        t1 = (hi - o) * inv
    tmin = np.nanmax(np.minimum(t0, t1), axis=-1)
    tmax = np.nanmin(np.maximum(t0, t1), axis=-1)
    hit = (tmax >= tmin) & (tmax > 0)
    return np.where(hit, np.maximum(tmin, 0.0), np.inf)


def render(cam: PinholeCam, *, box: bool = True, box_lo=BOX_LO, box_hi=BOX_HI):
    """(depth_cm uint16, sem uint8, bgr uint8) for ``cam``. Sky is the far plane."""
    vs, us = np.mgrid[0 : cam.height, 0 : cam.width]
    d = pixel_rays(cam, us.reshape(-1), vs.reshape(-1))
    o = np.asarray(cam.pos, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(d[:, 2] < -1e-9, (GROUND_Z - o[2]) / d[:, 2], np.inf)
    tb = _box_hit(o[None, :], d, box_lo, box_hi) if box else np.full(tg.shape, np.inf)
    t = np.minimum(tg, tb)
    sem = np.where(tb <= tg, SEM_BOX, SEM_GROUND)
    sky = ~np.isfinite(t) | (t >= cam.far_m)
    sem = np.where(sky, SEM_SKY, sem)
    depth = np.where(sky, cam.far_m, t)
    cm = np.clip(np.rint(depth * 100.0), 0, 65535).astype(np.uint16)
    p = o + d * np.where(sky, 0.0, t)[:, None]
    bgr = np.zeros((d.shape[0], 3), dtype=np.uint8)
    checker = ((np.floor(p[:, 0]) + np.floor(p[:, 1])) % 2).astype(np.uint8)
    bgr[sem == SEM_GROUND] = np.stack([60 + 80 * checker, 60 + 80 * checker, 60 + 80 * checker], -1)[sem == SEM_GROUND]
    bgr[sem == SEM_BOX] = (30, 30, 220)
    bgr[sem == SEM_SKY] = (230, 180, 120)
    shape = (cam.height, cam.width)
    return cm.reshape(shape), sem.astype(np.uint8).reshape(shape), bgr.reshape(shape + (3,))
