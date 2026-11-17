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
from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np

from python.perception.occ_geometry import DEFAULT_GRID, OccGrid, project

VEHICLE = 2
DRIVABLE = 1
VEHICLE_SIGNAL_KEYS = ("wheelspeed_mps", "imu_yaw_rate")
VO_MIN_CELLS = 12
VO_MIN_SCORE = 0.4
FWD_RANGE_M = (-1.0, 4.0)     # per frame: -10..40 m/s at 10 Hz
YAW_RANGE = (-0.15, 0.15)     # rad per frame


def add_ego_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--ego-from-vehicle", action="store_true", default=False,
                    help="ego motion from the car's wheel speed + IMU yaw rate (default: camera-only VO)")
    return ap


def _static_bev(keys: np.ndarray, cls: np.ndarray, grid: OccGrid,
                obs: np.ndarray | None = None) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64)
    cls = np.asarray(cls)
    m = (cls != VEHICLE) & (cls != DRIVABLE)
    if obs is not None:
        m = m & np.asarray(obs, dtype=bool)
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


def _visible_xy(xy: np.ndarray, cams: Sequence[Any] | None, z: float = 0.6,
                occl: np.ndarray | None = None, grid: OccGrid | None = None) -> np.ndarray:
    """Which BEV points the current live cameras can see (at height ``z``).

    Frustum only when no occluders are given; otherwise the same observed
    test the runtime uses (frustum and no occluder on the ray).
    """
    if not cams or len(xy) == 0:
        return np.ones(len(xy), dtype=bool)
    pts = np.column_stack([xy, np.full(len(xy), z)])
    if occl is None or grid is None:
        vis = np.zeros(len(xy), dtype=bool)
        for cam in cams:
            _, _, ok = project(cam, pts)
            vis |= ok
        return vis
    from python.perception.occ_out import observed_mask
    return observed_mask(pts, cams, occl, grid)


def _score(prev: np.ndarray, cur_keys: np.ndarray, grid: OccGrid, fwd: float, dyaw: float,
           cams: Sequence[Any] | None = None, occl: np.ndarray | None = None) -> float:
    """Share of warped previous cells that hit a current cell. Sparse: sorted key lookup, no raster.

    When ``cams`` is set, the denominator is only the previous cells that
    fall in the current observed region after the warp (live cameras, and
    this tick's occluders when ``occl`` is given), so a side/rear camera
    going stale does not collapse the score.
    """
    if len(prev) == 0 or len(cur_keys) == 0:
        return 0.0
    warped = _warp(prev, fwd, dyaw)
    k = _bev_keys(warped, grid)
    pos = np.clip(np.searchsorted(cur_keys, k), 0, len(cur_keys) - 1)
    hits = (cur_keys[pos] == k) & (k >= 0)
    if cams:
        vis = _visible_xy(warped, cams, occl=occl, grid=grid)
        n = int(vis.sum())
        if n < VO_MIN_CELLS:
            return 0.0
        return float((hits & vis).sum()) / float(n)
    return float(hits.sum()) / float(len(prev))


def _register(prev: np.ndarray, cur_keys: np.ndarray, grid: OccGrid,
              cams: Sequence[Any] | None = None, occl: np.ndarray | None = None) -> tuple[float, float, float]:
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
                sc = _score(prev, cur_keys, grid, float(f), float(yw), cams=cams, occl=occl)
                if sc > best[0] + 1e-9:
                    best = (sc, float(f), float(yw))
    return best[1], best[2], best[0]


def occ_odometry(prev_keys, prev_cls, cur_keys, cur_cls, *, grid: OccGrid = DEFAULT_GRID,
                 prev_obs: np.ndarray | None = None, cur_obs: np.ndarray | None = None,
                 cams: Sequence[Any] | None = None, occl: np.ndarray | None = None,
                 ) -> tuple[float, float, float]:
    """(forward m, yaw rad, overlap score) moving the previous frame onto the current one.

    Only *observed* static cells vote (``prev_obs`` / ``cur_obs``; default all
    observed). Unobserved decode_unseen clutter is ignored, so there is no
    residual path: a parked ego with a moving or jittering subset keeps the
    (0, 0) hypothesis the same way full-set registration always did. Forward
    and yaw come from the same search. The score's denominator is the
    previous observed cells that, after the warp, fall in the current
    observed region (all previous observed cells if ``cams`` is omitted).
    """
    prev = _static_bev(prev_keys, prev_cls, grid, obs=prev_obs)
    cur = _static_bev(cur_keys, cur_cls, grid, obs=cur_obs)
    if len(prev) < VO_MIN_CELLS or len(cur) < VO_MIN_CELLS:
        return 0.0, 0.0, 0.0
    cur_k = np.unique(_bev_keys(cur, grid))
    cur_k = cur_k[cur_k >= 0]
    return _register(prev, cur_k, grid, cams=cams, occl=occl)


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
        self._prev: tuple[np.ndarray, np.ndarray, float, np.ndarray | None] | None = None
        self._last = EgoEstimate(0.0, 0.0, self.source, False)

    @classmethod
    def from_args(cls, a: argparse.Namespace, **kw: Any) -> "EgoMotion":
        return cls(from_vehicle=bool(getattr(a, "ego_from_vehicle", False)), **kw)

    def update(self, keys: np.ndarray, cls: np.ndarray, t: float, vehicle: Mapping[str, Any] | None = None,
               obs: np.ndarray | None = None, cams: Sequence[Any] | None = None,
               occl: np.ndarray | None = None) -> EgoEstimate:
        if self.from_vehicle:
            sig = {k: vehicle[k] for k in VEHICLE_SIGNAL_KEYS if vehicle is not None and k in vehicle}
            if len(sig) == len(VEHICLE_SIGNAL_KEYS):
                self._last = EgoEstimate(float(sig["wheelspeed_mps"]), float(sig["imu_yaw_rate"]), self.source, True)
            else:
                self._last = EgoEstimate(self._last.speed_mps, self._last.yaw_rate, self.source, False)
            return self._last
        obs_a = None if obs is None else np.asarray(obs, dtype=bool)
        prev, self._prev = self._prev, (np.asarray(keys), np.asarray(cls), float(t), obs_a)
        if prev is None or float(t) <= prev[2]:
            self._last = EgoEstimate(self._last.speed_mps, self._last.yaw_rate, self.source, False)
            return self._last
        dt = float(t) - prev[2]
        fwd, dyaw, score = occ_odometry(prev[0], prev[1], keys, cls, grid=self.grid,
                                        prev_obs=prev[3], cur_obs=obs_a, cams=cams, occl=occl)
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
