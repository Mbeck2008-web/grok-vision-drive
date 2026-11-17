"""Ego motion for the occupancy tracker, deploy-valid sources only.

Default (``camera_vo``): camera-only odometry from the net's own sparse
occupancy (sparse: sorted BEV cell keys, no raster or dense grid). The static (non-vehicle) BEV cells of the previous frame are
registered onto the current frame by a coarse-to-fine search over forward
travel and yaw; the best overlap gives ``(speed_mps, yaw_rate)``. Nothing but
the predicted cells is read.

``--ego-from-vehicle`` (default OFF, GVD ruling): the car's own wheel speed and
IMU yaw rate, read through ``VEHICLE_SIGNAL_KEYS`` only. Which BeamNG signals
those map to (``electrics.wheelspeed``, an IMU sensor's gyro z) is for GVD to
confirm; they are the vehicle's sensors, not the sim pose.

BeamNG ground-truth pose, scenetree, ``state.jsonl`` ego rows or any other sim
truth are training labels only: this module imports none of them and never
reads a key outside ``VEHICLE_SIGNAL_KEYS`` (tested with an import trace and a
spy mapping).

Limitation: a featureless straight road (only uniform drivable cells) has no
forward constraint; static non-drivable cells (edges, posts, buildings) are
what VO registers. With fewer than ``VO_MIN_CELLS`` of them, or a weak match,
the last estimate is held and ``ok`` is False.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from python.perception.occ_geometry import DEFAULT_GRID, OccGrid

VEHICLE = 2
DRIVABLE = 1
VEHICLE_SIGNAL_KEYS = ("wheelspeed_mps", "imu_yaw_rate")
VO_MIN_CELLS = 12
VO_MIN_SCORE = 0.4
# Residual VO: unmatched-at-zero cells are used only when they are a real
# share of the previous set, or when they themselves register with high
# confidence (ego-fixed clutter is a small share but a sharp residual peak).
# An absolute 12-cell floor on the residual (the first version) treated
# parked-ego flicker as motion.
VO_RESIDUAL_SHARE = 0.15
VO_RESIDUAL_CONF = 0.6
FWD_RANGE_M = (-1.0, 4.0)     # per frame: -10..40 m/s at 10 Hz
YAW_RANGE = (-0.15, 0.15)     # rad per frame


def add_ego_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--ego-from-vehicle", action="store_true", default=False,
                    help="ego motion from the car's wheel speed + IMU yaw rate (default: camera-only VO)")
    return ap


def _static_bev(keys: np.ndarray, cls: np.ndarray, grid: OccGrid) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64)
    cls = np.asarray(cls)
    m = (cls != VEHICLE) & (cls != DRIVABLE)
    if not m.any():
        return np.zeros((0, 2), dtype=np.float64)
    c = grid.cell_of_key(keys[m])
    road = cls == DRIVABLE
    if road.any():
        # flat ground beside the road (grass, pavement) is the same at every
        # along-track shift and swamps the score: keep cells >= 1 voxel above
        # the road surface (median z cell of this frame's drivable cells)
        ground = int(np.median(grid.cell_of_key(keys[road])[:, 2]))
        c = c[c[:, 2] >= ground + 1]
        if not len(c):
            return np.zeros((0, 2), dtype=np.float64)
    xy = np.unique(c[:, :2], axis=0)
    return (xy + 0.5) * float(grid.voxel_m) + np.array([grid.x_min, grid.y_min])


def _warp(xy: np.ndarray, fwd: float, dyaw: float) -> np.ndarray:
    """Previous-frame ego points -> current ego frame (same convention as ``OccTracker``)."""
    c, s = math.cos(dyaw), math.sin(dyaw)
    x, y = xy[:, 0], xy[:, 1] - fwd
    return np.stack([c * x + s * y, -s * x + c * y], axis=1)


def _bev_keys(xy: np.ndarray, grid: OccGrid) -> np.ndarray:
    v = float(grid.voxel_m)
    nx, ny, _ = grid.shape
    ix = np.floor((xy[:, 0] - grid.x_min) / v).astype(np.int64)
    iy = np.floor((xy[:, 1] - grid.y_min) / v).astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    return np.where(ok, ix * ny + iy, -1)


def _score(prev: np.ndarray, cur_keys: np.ndarray, grid: OccGrid, fwd: float, dyaw: float) -> float:
    """Share of warped previous cells that hit a current cell. Sparse: sorted key lookup, no raster."""
    if len(prev) == 0 or len(cur_keys) == 0:
        return 0.0
    k = _bev_keys(_warp(prev, fwd, dyaw), grid)
    pos = np.clip(np.searchsorted(cur_keys, k), 0, len(cur_keys) - 1)
    return float(((cur_keys[pos] == k) & (k >= 0)).sum()) / float(len(prev))


def _register(prev: np.ndarray, cur_keys: np.ndarray, grid: OccGrid) -> tuple[float, float, float]:
    """Coarse-to-fine search: (forward m, yaw rad, overlap score)."""
    if len(prev) < VO_MIN_CELLS or len(cur_keys) == 0:
        return 0.0, 0.0, 0.0
    best = (-1.0, 0.0, 0.0)
    for step_f, step_y, span_f, span_y in ((0.2, 0.01, None, None), (0.05, 0.0025, 0.3, 0.015)):
        if span_f is None:
            fs = np.arange(FWD_RANGE_M[0], FWD_RANGE_M[1] + 1e-9, step_f)
            ys = np.arange(YAW_RANGE[0], YAW_RANGE[1] + 1e-9, step_y)
        else:
            fs = best[1] + np.arange(-span_f, span_f + 1e-9, step_f)
            ys = best[2] + np.arange(-span_y, span_y + 1e-9, step_y)
        for f in fs:
            for yw in ys:
                sc = _score(prev, cur_keys, grid, float(f), float(yw))
                if sc > best[0] + 1e-9:
                    best = (sc, float(f), float(yw))
    return best[1], best[2], best[0]


def occ_odometry(prev_keys, prev_cls, cur_keys, cur_cls, *, grid: OccGrid = DEFAULT_GRID) -> tuple[float, float, float]:
    """(forward m, yaw rad, overlap score) moving the previous frame onto the current one."""
    prev, cur = _static_bev(prev_keys, prev_cls, grid), _static_bev(cur_keys, cur_cls, grid)
    if len(prev) < VO_MIN_CELLS or len(cur) < VO_MIN_CELLS:
        return 0.0, 0.0, 0.0
    cur_keys = np.unique(_bev_keys(cur, grid))
    cur_keys = cur_keys[cur_keys >= 0]
    # Score the full set and, if enough cells miss at zero, the residual set.
    # Ego-fixed decode_unseen clutter scores 0.8+ at (0, 0) on the full set and
    # would lock speed=0; the residual of the moving posts still registers.
    # A parked ego with a few flickering cells must keep the full (0, 0) hit:
    # residual is used only when the unmatched share is large, or the residual
    # itself scores with high confidence (not a ~0.4 flicker peak).
    full = _register(prev, cur_keys, grid)
    k0 = _bev_keys(_warp(prev, 0.0, 0.0), grid)
    pos0 = np.clip(np.searchsorted(cur_keys, k0), 0, len(cur_keys) - 1)
    hit0 = (cur_keys[pos0] == k0) & (k0 >= 0)
    residual = prev[~hit0]
    if len(residual) >= VO_MIN_CELLS:
        res = _register(residual, cur_keys, grid)
        share = len(residual) / float(len(prev))
        if res[2] >= VO_MIN_SCORE and (share >= VO_RESIDUAL_SHARE or res[2] >= VO_RESIDUAL_CONF):
            return res
    return full


@dataclass
class EgoEstimate:
    speed_mps: float
    yaw_rate: float
    source: str
    ok: bool


class EgoMotion:
    """Per-frame ego speed / yaw rate for ``OccTracker``. Default: camera VO."""

    def __init__(self, *, from_vehicle: bool = False, grid: OccGrid = DEFAULT_GRID) -> None:
        self.from_vehicle = bool(from_vehicle)
        self.source = "vehicle_signals" if self.from_vehicle else "camera_vo"
        self.grid = grid
        self._prev: tuple[np.ndarray, np.ndarray, float] | None = None
        self._last = EgoEstimate(0.0, 0.0, self.source, False)

    @classmethod
    def from_args(cls, a: argparse.Namespace, **kw: Any) -> "EgoMotion":
        return cls(from_vehicle=bool(getattr(a, "ego_from_vehicle", False)), **kw)

    def update(self, keys: np.ndarray, cls: np.ndarray, t: float, vehicle: Mapping[str, Any] | None = None) -> EgoEstimate:
        if self.from_vehicle:
            sig = {k: vehicle[k] for k in VEHICLE_SIGNAL_KEYS if vehicle is not None and k in vehicle}
            if len(sig) == len(VEHICLE_SIGNAL_KEYS):
                self._last = EgoEstimate(float(sig["wheelspeed_mps"]), float(sig["imu_yaw_rate"]), self.source, True)
            else:
                self._last = EgoEstimate(self._last.speed_mps, self._last.yaw_rate, self.source, False)
            return self._last
        prev, self._prev = self._prev, (np.asarray(keys), np.asarray(cls), float(t))
        if prev is None or float(t) <= prev[2]:
            self._last = EgoEstimate(self._last.speed_mps, self._last.yaw_rate, self.source, False)
            return self._last
        dt = float(t) - prev[2]
        fwd, dyaw, score = occ_odometry(prev[0], prev[1], keys, cls, grid=self.grid)
        if score < VO_MIN_SCORE:
            self._last = EgoEstimate(self._last.speed_mps, self._last.yaw_rate, self.source, False)
        else:
            self._last = EgoEstimate(fwd / dt, dyaw / dt, self.source, True)
        return self._last


def runtime_frame(pred: Any, tracker: Any, ego: EgoMotion, t: float, vehicle: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One live frame downstream of the net: ego motion -> tracker -> outputs.

    Reads ``pred.keys`` / ``pred.cls`` and (only with ``--ego-from-vehicle``)
    ``VEHICLE_SIGNAL_KEYS`` of ``vehicle``. No image, no pose.
    """
    from python.perception.occ_outputs import occ_outputs

    est = ego.update(pred.keys, pred.cls, t, vehicle)
    out = occ_outputs(pred, tracker, t, ego_speed_mps=est.speed_mps, ego_yaw_rate=est.yaw_rate, grid=ego.grid)
    out["ego"].update({"source": est.source, "motion_ok": bool(est.ok), "yaw_rate": float(est.yaw_rate)})
    return out
