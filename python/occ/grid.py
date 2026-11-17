"""Voxel grid in the ego frame: x right, y forward, z up, metres.

The grid is a coordinate system only. Occupancy is stored as a sparse set
of unique ``(ix, iy, iz)`` int32 indices plus per-cell values, never as a
dense array.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VoxelGrid:
    x_min: float = -20.0
    x_max: float = 20.0
    y_min: float = -10.0
    y_max: float = 70.0
    z_min: float = -1.0
    z_max: float = 3.0
    voxel: float = 0.5
    coarse: int = 4  # coarse BEV pillar = coarse x coarse fine voxels

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            int(round((self.x_max - self.x_min) / self.voxel)),
            int(round((self.y_max - self.y_min) / self.voxel)),
            int(round((self.z_max - self.z_min) / self.voxel)),
        )

    @property
    def n_dense(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    @property
    def coarse_shape(self) -> tuple[int, int]:
        nx, ny, _ = self.shape
        return (nx + self.coarse - 1) // self.coarse, (ny + self.coarse - 1) // self.coarse

    def lo(self) -> np.ndarray:
        return np.array([self.x_min, self.y_min, self.z_min], dtype=np.float64)

    def to_index(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Indices (N,3) int32 and an in-bounds mask (N,). Non-finite points are out."""
        pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        finite = np.isfinite(pts).all(axis=1)
        safe = np.where(finite[:, None], pts, 0.0)
        idx = np.floor((safe - self.lo()) / self.voxel).astype(np.int64)
        shape = np.array(self.shape, dtype=np.int64)
        ok = finite & (idx >= 0).all(axis=1) & (idx < shape).all(axis=1)
        return np.where(ok[:, None], idx, 0).astype(np.int32), ok

    def to_center(self, idx: np.ndarray) -> np.ndarray:
        ii = np.asarray(idx, dtype=np.float64).reshape(-1, 3)
        return (self.lo() + (ii + 0.5) * self.voxel).astype(np.float32)

    def flat(self, idx: np.ndarray) -> np.ndarray:
        nx, ny, nz = self.shape
        ii = np.asarray(idx, dtype=np.int64).reshape(-1, 3)
        return (ii[:, 0] * ny + ii[:, 1]) * nz + ii[:, 2]

    def unflat(self, flat: np.ndarray) -> np.ndarray:
        nx, ny, nz = self.shape
        f = np.asarray(flat, dtype=np.int64).reshape(-1)
        iz = f % nz
        iy = (f // nz) % ny
        ix = f // (ny * nz)
        return np.stack([ix, iy, iz], axis=1).astype(np.int32)

    def coarse_of(self, idx: np.ndarray) -> np.ndarray:
        ii = np.asarray(idx, dtype=np.int64).reshape(-1, 3)
        return (ii[:, :2] // self.coarse).astype(np.int32)

    def coarse_centers(self, height: float = 0.0) -> np.ndarray:
        """Centre (x, y, height) of every coarse pillar, (Cx*Cy, 3)."""
        cx, cy = self.coarse_shape
        step = self.voxel * self.coarse
        gx, gy = np.meshgrid(np.arange(cx), np.arange(cy), indexing="ij")
        x = self.x_min + (gx.reshape(-1) + 0.5) * step
        y = self.y_min + (gy.reshape(-1) + 0.5) * step
        return np.stack([x, y, np.full_like(x, height)], axis=1).astype(np.float32)

    def fine_in_coarse(self, cidx: np.ndarray) -> np.ndarray:
        """All in-bounds fine voxel indices inside the given coarse pillars (unique)."""
        c = np.asarray(cidx, dtype=np.int64).reshape(-1, 2)
        if c.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.int32)
        nx, ny, nz = self.shape
        k = self.coarse
        ox, oy, oz = np.meshgrid(np.arange(k), np.arange(k), np.arange(nz), indexing="ij")
        off = np.stack([ox.reshape(-1), oy.reshape(-1), oz.reshape(-1)], axis=1)
        base = np.concatenate([c * k, np.zeros((c.shape[0], 1), dtype=np.int64)], axis=1)
        allv = (base[:, None, :] + off[None, :, :]).reshape(-1, 3)
        ok = (allv[:, 0] < nx) & (allv[:, 1] < ny) & (allv >= 0).all(axis=1)
        allv = allv[ok]
        return self.unflat(np.unique(self.flat(allv)))


DEFAULT_GRID = VoxelGrid()
