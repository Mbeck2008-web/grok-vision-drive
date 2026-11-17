"""Pinhole camera model from ``config/cameras.yaml`` (mount pose + fov_v + res).

Runtime-safe: geometry only, no ground truth. Ego frame x right, y forward,
z up. ``yaw_deg`` < 0 turns left, ``pitch_deg`` < 0 looks down. Pixels are
square, the principal point is the image centre. BeamNG depth is treated as
z-depth along the optical axis (unverified, same caveat as
``raw_frames.train_gt_from_raw``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    cam_id: str
    pos: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float
    fov_v_deg: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if not (self.width > 0 and self.height > 0):
            raise ValueError(f"camera {self.cam_id}: resolution must be positive, got {self.width}x{self.height}")
        if not (0.0 < float(self.fov_v_deg) < 179.0):
            raise ValueError(f"camera {self.cam_id}: fov_v must be in (0, 179), got {self.fov_v_deg}")
        vals = (*self.pos, self.yaw_deg, self.pitch_deg)
        if not all(math.isfinite(float(v)) for v in vals):
            raise ValueError(f"camera {self.cam_id}: pose must be finite")

    def scaled(self, width: int, height: int) -> "CameraModel":
        """Same camera at another resolution (fov_v kept)."""
        return CameraModel(self.cam_id, self.pos, self.yaw_deg, self.pitch_deg, self.fov_v_deg, int(width), int(height))

    @property
    def f(self) -> float:
        return (self.height / 2.0) / math.tan(math.radians(self.fov_v_deg) / 2.0)

    def K(self) -> np.ndarray:
        f = self.f
        return np.array([[f, 0.0, self.width / 2.0], [0.0, f, self.height / 2.0], [0.0, 0.0, 1.0]])

    def basis(self) -> np.ndarray:
        """Rows: right, down, forward (camera axes) in the ego frame."""
        yaw, pitch = math.radians(self.yaw_deg), math.radians(self.pitch_deg)
        fwd = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
        right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
        down = np.cross(fwd, right)
        return np.stack([right, down, fwd])

    def project(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ego points (N,3) -> pixel (N,2), z-depth (N,), in-image mask (N,)."""
        pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        cam = (pts - np.asarray(self.pos)) @ self.basis().T
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.f * cam[:, 0] / z + self.width / 2.0
            v = self.f * cam[:, 1] / z + self.height / 2.0
        ok = np.isfinite(u) & np.isfinite(v) & (z > 0.1) & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return np.stack([u, v], axis=1), z, ok

    def backproject(self, depth_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Depth image (H,W) metres -> ego points (H*W,3) and a valid mask.

        Invalid: non-finite, <= 0, or not the camera's resolution (raises).
        """
        d = np.asarray(depth_m, dtype=np.float64)
        if d.shape != (self.height, self.width):
            raise ValueError(f"depth shape {d.shape} != camera {self.cam_id} {(self.height, self.width)}")
        vv, uu = np.meshgrid(np.arange(self.height) + 0.5, np.arange(self.width) + 0.5, indexing="ij")
        z = d.reshape(-1)
        valid = np.isfinite(z) & (z > 0.0)
        zs = np.where(valid, z, 0.0)
        xc = (uu.reshape(-1) - self.width / 2.0) / self.f * zs
        yc = (vv.reshape(-1) - self.height / 2.0) / self.f * zs
        cam = np.stack([xc, yc, zs], axis=1)
        ego = cam @ self.basis() + np.asarray(self.pos)
        return ego, valid


def cameras_from_config(path: Path | str | None = None, ids: tuple[str, ...] = ("main",)) -> dict[str, CameraModel]:
    import yaml

    root = Path(__file__).resolve().parents[2]
    cfg_path = Path(path) if path else root / "config" / "cameras.yaml"
    cfg: Mapping[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out: dict[str, CameraModel] = {}
    for spec in cfg.get("cameras") or []:
        cid = str(spec.get("id"))
        if cid not in ids:
            continue
        w, h = (int(v) for v in spec.get("live_res"))
        pos = tuple(float(v) for v in spec.get("pos_m"))
        out[cid] = CameraModel(cid, pos, float(spec.get("yaw_deg", 0)), float(spec.get("pitch_deg", 0)), float(spec["fov_v"]), w, h)
    missing = [c for c in ids if c not in out]
    if missing:
        raise KeyError(f"cameras not in {cfg_path.name}: {missing}")
    return out
