"""Sparse occupancy grid and pinhole camera geometry (numpy only).

Shared by the runtime occupancy net (projection) and the training-only label
builder (back-projection). Nothing here reads depth, semantics, or vehicle
boxes. Everything is the GVD vehicle frame: +X right, +Y forward, +Z up, the
frame of ``config/cameras.yaml``.

The grid is an index space only. No dense array of it is ever allocated:
occupied cells are a sorted vector of linear keys (see ``occ_labels``) and the
net is queried at point sets (see ``occ_sparse``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class OccGrid:
    """Axis-aligned ego-frame box cut into cubic cells of ``voxel_m``."""

    x_min: float = -25.6
    x_max: float = 25.6
    y_min: float = -12.8
    y_max: float = 51.2
    z_min: float = -2.0
    z_max: float = 4.4
    voxel_m: float = 0.4

    @property
    def shape(self) -> tuple[int, int, int]:
        v = float(self.voxel_m)
        return (
            int(round((self.x_max - self.x_min) / v)),
            int(round((self.y_max - self.y_min) / v)),
            int(round((self.z_max - self.z_min) / v)),
        )

    @property
    def dense_cells(self) -> int:
        nx, ny, nz = self.shape
        return int(nx * ny * nz)

    @property
    def lo(self) -> np.ndarray:
        return np.array([self.x_min, self.y_min, self.z_min], dtype=np.float64)

    @property
    def hi(self) -> np.ndarray:
        return np.array([self.x_max, self.y_max, self.z_max], dtype=np.float64)

    def coarse(self, factor: int) -> "OccGrid":
        """Same box, cells ``factor`` times larger. The box must divide evenly."""
        f = int(factor)
        nx, ny, nz = self.shape
        if f < 1 or nx % f or ny % f or nz % f:
            raise ValueError(f"grid {self.shape} does not divide by {f}")
        return OccGrid(self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max, self.voxel_m * f)

    def inside(self, pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64)
        return np.all((p >= self.lo) & (p < self.hi), axis=-1)

    def cell_of(self, pts: np.ndarray) -> np.ndarray:
        """Integer (ix, iy, iz) of each point. Callers mask with ``inside`` first."""
        p = np.asarray(pts, dtype=np.float64)
        return np.floor((p - self.lo) / float(self.voxel_m)).astype(np.int64)

    def key_of_cell(self, cell: np.ndarray) -> np.ndarray:
        nx, ny, nz = self.shape
        c = np.asarray(cell, dtype=np.int64)
        return (c[..., 0] * ny + c[..., 1]) * nz + c[..., 2]

    def cell_of_key(self, key: np.ndarray) -> np.ndarray:
        nx, ny, nz = self.shape
        k = np.asarray(key, dtype=np.int64)
        iz = k % nz
        iy = (k // nz) % ny
        ix = k // (nz * ny)
        return np.stack([ix, iy, iz], axis=-1)

    def keys_of(self, pts: np.ndarray) -> np.ndarray:
        """Linear key of each point inside the box; -1 outside."""
        p = np.asarray(pts, dtype=np.float64)
        keys = np.full(p.shape[:-1], -1, dtype=np.int64)
        ok = self.inside(p)
        if np.any(ok):
            keys[ok] = self.key_of_cell(self.cell_of(p[ok]))
        return keys

    def centers_of_keys(self, key: np.ndarray) -> np.ndarray:
        return self.lo + (self.cell_of_key(key).astype(np.float64) + 0.5) * float(self.voxel_m)


DEFAULT_GRID = OccGrid()


@dataclass(frozen=True)
class PinholeCam:
    """One rig camera: ego-frame pose and a square-pixel pinhole."""

    cam_id: str
    pos: tuple[float, float, float]
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    fwd: tuple[float, float, float]
    width: int
    height: int
    fov_v_deg: float
    near_m: float
    far_m: float

    @property
    def focal_px(self) -> float:
        return (float(self.height) / 2.0) / math.tan(math.radians(float(self.fov_v_deg)) / 2.0)

    def rotation(self) -> np.ndarray:
        """Rows are the camera right / up / forward axes in the ego frame."""
        return np.array([self.right, self.up, self.fwd], dtype=np.float64)

    def scaled(self, long_side: int) -> "PinholeCam":
        """Same camera at a new resolution (long side), same vertical FOV."""
        s = float(long_side) / float(max(self.width, self.height))
        return PinholeCam(
            self.cam_id, self.pos, self.right, self.up, self.fwd,
            max(1, int(round(self.width * s))), max(1, int(round(self.height * s))),
            self.fov_v_deg, self.near_m, self.far_m,
        )


def _unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n <= 0.0:
        raise ValueError("zero-length axis")
    return a / n


def cam_from_sidecar(rec: Mapping[str, Any]) -> PinholeCam:
    """``raw_frames.camera_sidecar`` record -> ``PinholeCam``. right = fwd x up."""
    fwd = _unit(rec["direction"])
    up = _unit(rec["up"])
    right = _unit(np.cross(fwd, up))
    up = _unit(np.cross(right, fwd))
    w, h = rec["resolution"]
    return PinholeCam(
        str(rec["id"]),
        tuple(float(v) for v in rec["pos_m"]),
        tuple(float(v) for v in right),
        tuple(float(v) for v in up),
        tuple(float(v) for v in fwd),
        int(w), int(h), float(rec["fov_v"]), float(rec["near_m"]), float(rec["far_m"]),
    )


def rig_cams(cfg: Mapping[str, Any] | None) -> list[PinholeCam]:
    """All rig cameras from a cameras.yaml dict, in yaml order."""
    from python.data.raw_frames import camera_sidecar

    return [cam_from_sidecar(rec) for rec in camera_sidecar(cfg)]


def project(cam: PinholeCam, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ego points (N,3) -> pixel (u, v), camera z, and an in-frustum mask."""
    p = np.asarray(pts, dtype=np.float64) - np.asarray(cam.pos, dtype=np.float64)
    c = p @ cam.rotation().T
    z = c[:, 2]
    f = cam.focal_px
    safe = np.where(z > 1e-6, z, 1.0)
    u = cam.width / 2.0 + f * c[:, 0] / safe
    v = cam.height / 2.0 - f * c[:, 1] / safe
    ok = (z > cam.near_m) & (z < cam.far_m) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    return np.stack([u, v], axis=-1), z, ok


def pixel_rays(cam: PinholeCam, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """Ego-frame ray directions with camera-z component 1 (planar depth scale)."""
    f = cam.focal_px
    x = (np.asarray(us, dtype=np.float64) + 0.5 - cam.width / 2.0) / f
    y = -(np.asarray(vs, dtype=np.float64) + 0.5 - cam.height / 2.0) / f
    cam_dirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    return cam_dirs @ cam.rotation()


def backproject(
    cam: PinholeCam, depth_m: np.ndarray, us: np.ndarray, vs: np.ndarray, *, radial: bool = False
) -> np.ndarray:
    """Pixels with metric depth -> ego points.

    ``radial=False`` treats depth as camera-z (planar). BeamNG's depth buffer
    convention is unverified in this repo (``raw_frames.train_gt_from_raw``);
    ``radial=True`` treats it as distance along the ray.
    """
    rays = pixel_rays(cam, us, vs)
    d = np.asarray(depth_m, dtype=np.float64)
    if radial:
        d = d / np.linalg.norm(rays, axis=-1)
    return np.asarray(cam.pos, dtype=np.float64) + rays * d[:, None]
