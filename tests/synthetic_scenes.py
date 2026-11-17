"""Synthetic intersection / roundabout clips with occluders (offline, no BeamNG).

World frame = ego frame at t=0; the ego drives straight along +y (heading
fixed), so world -> ego is a translation and boxes stay axis-aligned. Ground at
``GROUND_Z``; road = within width/2 of a centerline (ASPHALT), else GRASS;
buildings / roundabout island are boxes (BUILDING -> occupancy "other").
Used by ``test_occ_unseen`` and ``scripts/eval_occ_unseen.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from python.perception.occ_geometry import PinholeCam, pixel_rays
from python.train.occ_map_labels import MapRoad, _seg_dist, _segments
from synthetic_occ import GROUND_Z, _box_hit

SEM_CLASSES = {"kind": "gvd_sem_classes", "train_only": True,
               "classes": {"0": "unknown", "1": "ASPHALT", "2": "CAR", "3": "SKY", "4": "GRASS", "5": "BUILDING"}}
SEM_ROAD, SEM_CAR, SEM_SKY, SEM_GRASS, SEM_BUILDING = 1, 2, 3, 4, 5


@dataclass
class Scene:
    kind: str
    roads: list[MapRoad]
    boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = field(default_factory=list)
    speed: float = 6.0

    def shifted(self, dy: float) -> "Scene":
        """The scene in the ego frame after driving ``dy`` metres along +y."""
        roads = [MapRoad(r.xy - np.array([0.0, dy]), r.width_m, None, r.closed) for r in self.roads]
        boxes = [((lo[0], lo[1] - dy, lo[2]), (hi[0], hi[1] - dy, hi[2])) for lo, hi in self.boxes]
        return Scene(self.kind, roads, boxes, self.speed)


def intersection(seed: int) -> Scene:
    rng = np.random.default_rng(seed)
    yc, w = float(rng.uniform(24, 34)), float(rng.uniform(6.0, 8.0))
    main = MapRoad(np.array([[0.0, -20.0], [0.0, 80.0]]), w)
    cross = MapRoad(np.array([[-40.0, yc], [40.0, yc]]), float(rng.uniform(6.0, 8.0)))
    m = w / 2.0 + float(rng.uniform(1.0, 2.5))
    boxes = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            x0, x1 = sorted((sx * m, sx * (m + rng.uniform(8, 14))))
            y0, y1 = sorted((yc + sy * m, yc + sy * (m + rng.uniform(8, 14))))
            boxes.append(((x0, y0, 0.0), (x1, y1, float(rng.uniform(4, 8)))))
    return Scene("intersection", [main, cross], boxes, float(rng.uniform(5, 8)))


def roundabout(seed: int) -> Scene:
    rng = np.random.default_rng(seed)
    yc, R, w = float(rng.uniform(28, 36)), float(rng.uniform(9, 12)), float(rng.uniform(6.0, 7.5))
    a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    ring = MapRoad(np.stack([R * np.cos(a), yc + R * np.sin(a)], 1), w, None, closed=True)
    approach = MapRoad(np.array([[0.0, -20.0], [0.0, yc - R]]), w)
    exits = [MapRoad(np.array([[0.0, yc + R], [0.0, 90.0]]), w)]
    if rng.uniform() < 0.7:
        exits.append(MapRoad(np.array([[R, yc], [45.0, yc]]), w))
    if rng.uniform() < 0.7:
        exits.append(MapRoad(np.array([[-R, yc], [-45.0, yc]]), w))
    isl = R - w / 2.0 - 0.5
    boxes = [((-isl * 0.8, yc - isl * 0.8, 0.0), (isl * 0.8, yc + isl * 0.8, float(rng.uniform(2.0, 3.5))))]
    return Scene("roundabout", [ring, approach] + exits, boxes, float(rng.uniform(4, 7)))


def on_road(scene: Scene, xy: np.ndarray) -> np.ndarray:
    out = np.zeros(xy.shape[0], bool)
    for r in scene.roads:
        a, b = _segments(r)
        out |= _seg_dist(xy, a, b) <= r.width_m / 2.0
    return out


def render(cam: PinholeCam, scene: Scene):
    """(depth_cm uint16, sem uint8, bgr uint8) for ``cam`` in the scene's ego frame."""
    vs, us = np.mgrid[0:cam.height, 0:cam.width]
    d = pixel_rays(cam, us.reshape(-1), vs.reshape(-1))
    o = np.asarray(cam.pos, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d[:, 2] < -1e-9, (GROUND_Z - o[2]) / d[:, 2], np.inf)
    sem = np.full(t.shape, SEM_ROAD, np.uint8)
    shade = np.zeros(t.shape)
    for i, (lo, hi) in enumerate(scene.boxes):
        tb = _box_hit(o[None, :], d, lo, hi)
        closer = tb < t
        t = np.where(closer, tb, t)
        sem = np.where(closer, SEM_BUILDING, sem)
        shade = np.where(closer, (i * 37) % 80, shade)
    sky = ~np.isfinite(t) | (t >= cam.far_m)
    depth = np.where(sky, cam.far_m, t)
    p = o + d * np.where(sky, 0.0, t)[:, None]
    ground = (sem == SEM_ROAD) & ~sky
    road = on_road(scene, p[ground, :2])
    gi = np.flatnonzero(ground)
    sem[gi[~road]] = SEM_GRASS
    sem = np.where(sky, SEM_SKY, sem)
    bgr = np.zeros((d.shape[0], 3), np.uint8)
    chk = ((np.floor(p[:, 0] / 2) + np.floor(p[:, 1] / 2)) % 2).astype(np.int32)
    bgr[sem == SEM_ROAD] = np.stack([50 + 30 * chk] * 3, -1)[sem == SEM_ROAD]
    bgr[sem == SEM_GRASS] = np.stack([30 + 10 * chk, 120 + 30 * chk, 30 + 10 * chk], -1)[sem == SEM_GRASS]
    b = sem == SEM_BUILDING
    bgr[b] = np.stack([60 + shade, 90 + shade, 150 + shade * 0.5], -1)[b].astype(np.uint8)
    bgr[sem == SEM_SKY] = (230, 180, 120)
    cm = np.clip(np.rint(depth * 100.0), 0, 65535).astype(np.uint16)
    shape = (cam.height, cam.width)
    return cm.reshape(shape), sem.reshape(shape), bgr.reshape(shape + (3,))


def clip(scene: Scene, frames: int = 6, dt: float = 0.25):
    """[(t, ego-frame scene)] driving straight at the scene's speed."""
    return [(i * dt, scene.shifted(scene.speed * i * dt)) for i in range(frames)]
