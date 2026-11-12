"""Training-only obstacle mask and clearance.

Runtime and the visualizer do not import this module. Inside the drivable
corridor a cell is blocked when its height above the cross-sloped road
surface is past ``clear_eps_m`` (0.15 m). The slope is
``(edge_z - z_center) / signed half-width`` from ``RoadPoly`` ``left_z`` /
``right_z``. When that side has no edge height, a RANSAC fit of paved
in-corridor points supplies it. If neither exists the point stays unknown.
``depth_range_pad_m`` is the only range pad; it stays 0 until spec #217
defines ``pad(r)``. A point within ``edge_band_m`` (0.5 m) outside the
corridor is blocked at ``curb_eps_m`` (0.08 m). Those three numbers are
written on every ``obstacles.jsonl`` row. The difference is in the world
frame. Ego z, ego pitch, and a missing height are not a road surface.

The corridor half-width is half the painted lane width about the painted
center. When that lane is missing it is ``road_width_m / 2`` about
``-road_offset_m + y * sin(heading)``, the same center ``pack_road_edge``
writes. Semantic blocked cells are used only for names a human copied out of
this recording's ``sem_classes.json``. ``PROP`` and ``OBJECT`` never count.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from python.debug.lane_targets import LOOKAHEADS_M, lane_center_is_valid
from python.train.sensor_aux import DRIVABLE_CLASS_ALLOWLIST

# In-corridor world Z above the interpolated road surface. Not an ego-frame height.
# Strictly greater than this is blocked. Written on every obstacles.jsonl row.
CLEAR_EPS_M = 0.15
OBSTACLE_Z_EPS_M = CLEAR_EPS_M
# Outside the corridor, within EDGE_BAND_M of the edge: a curb, not a lip.
CURB_EPS_M = 0.08
EDGE_BAND_M = 0.5
# Fallback slope fit. A curb face (0.08 m) cannot be an inlier.
FIT_RESIDUAL_M = 0.05
FIT_MIN_INLIERS = 20
GRID_H = 64
GRID_W = 64
X_MIN = -16.0
X_MAX = 16.0
Y_MIN = 0.0
Y_MAX = 40.0
UNKNOWN = 0
DRIVABLE = 1
BLOCKED = 2
CLEAR_FOOT_M = 0.5
# Coarse annotation buckets are not obstacle classes, even if a caller lists them.
COARSE_CLASS_NAMES = frozenset({"PROP", "OBJECT"})
# A parked prop has not been measured against clear_m. The loss weight stays 0.
PROP_CHECK_PASSED = False
CLEARANCE_LOSS_WEIGHT = 0.0

_SIDECAR = "obstacles.jsonl"


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _vec3(value: Any, *, missing_z: float | None = 0.0) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x, y = _finite(value[0]), _finite(value[1])
    if x is None or y is None:
        return None
    if len(value) >= 3:
        z = _finite(value[2])
        if z is None:
            return None
        return x, y, z
    if missing_z is None:
        return None
    return x, y, float(missing_z)


def _norm(vec: tuple[float, float, float]) -> tuple[float, float, float] | None:
    length = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    if length < 1e-8:
        return None
    return vec[0] / length, vec[1] / length, vec[2] / length


def _cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vehicle_axes(
    ego_dir: Any, ego_up: Any
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None:
    """``(right, forward, up)`` in the world frame.

    A 2-D direction is a level vehicle (world up). A 3-D direction keeps its
    pitch. ``ego_up`` is Gram-Schmidt'd against forward so the two axes stay
    orthonormal. Returns None when the direction is missing or parallel to up.
    """
    forward = _vec3(ego_dir, missing_z=0.0)
    if forward is None:
        return None
    forward_u = _norm(forward)
    if forward_u is None:
        return None
    if ego_up is None:
        up_raw = (0.0, 0.0, 1.0)
    else:
        parsed = _vec3(ego_up, missing_z=None)
        if parsed is None:
            return None
        up_raw = parsed
    up = (
        up_raw[0] - _dot(up_raw, forward_u) * forward_u[0],
        up_raw[1] - _dot(up_raw, forward_u) * forward_u[1],
        up_raw[2] - _dot(up_raw, forward_u) * forward_u[2],
    )
    up_u = _norm(up)
    if up_u is None:
        return None
    right = _norm(_cross(forward_u, up_u))
    if right is None:
        return None
    return right, forward_u, up_u


def _camera_basis(pose: Any) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], float, float] | None:
    from python.sensors.cameras import yaw_pitch_to_dir_up

    width = float(getattr(pose, "width", 0) or 0)
    height = float(getattr(pose, "height", 0) or 0)
    fov_v = math.radians(float(getattr(pose, "fov_v_deg", 0) or 0))
    if width < 2.0 or height < 2.0 or fov_v <= 1e-3 or fov_v >= math.pi - 1e-3:
        return None
    fov_h = 2.0 * math.atan(math.tan(fov_v * 0.5) * (width / height))
    forward, up = yaw_pitch_to_dir_up(float(pose.yaw_deg), float(pose.pitch_deg))
    right = _cross(forward, up)
    return forward, up, right, fov_v, fov_h


def view_depth_m(
    pose: Any, x: float, y: float, z: float, width: int, height: int
) -> tuple[float, float, float] | None:
    """Ego point to ``(u, v, depth_m)`` along the camera forward axis.

    Same basis as ``pose_pixel_of_ground``. ``depth_m`` is the view-Z stored
    in the uint16 centimetre PNG (``depth_01 * far``), not a ground intersection.
    """
    basis = _camera_basis(pose)
    if basis is None:
        return None
    forward, up, right, fov_v, fov_h = basis
    w = float(width)
    h = float(height)
    if w < 2.0 or h < 2.0:
        return None
    d = (float(x) - float(pose.x), float(y) - float(pose.y), float(z) - float(pose.z))
    depth = _dot(d, forward)
    if depth <= 1e-3:
        return None
    rx = _dot(d, right) / depth
    ry = -_dot(d, up) / depth
    u = rx / math.tan(fov_h * 0.5) * (w * 0.5) + (w - 1.0) * 0.5
    v = ry / math.tan(fov_v * 0.5) * (h * 0.5) + (h - 1.0) * 0.5
    return float(u), float(v), float(depth)


def _unproject(pose: Any, depth_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """uint16 centimetres to ego xyz. Invalid where the depth is 0."""
    basis = _camera_basis(pose)
    height, width = int(depth_cm.shape[0]), int(depth_cm.shape[1])
    if basis is None:
        empty = np.zeros((height, width), np.float64)
        return empty, empty, empty, np.zeros((height, width), np.bool_)
    forward, up, right, fov_v, fov_h = basis
    vs = np.arange(height, dtype=np.float64)
    us = np.arange(width, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    w = float(width)
    h = float(height)
    nx = (uu - (w - 1.0) * 0.5) / (w * 0.5)
    ny = (vv - (h - 1.0) * 0.5) / (h * 0.5)
    rx = nx * math.tan(fov_h * 0.5)
    ry = ny * math.tan(fov_v * 0.5)
    ray_x = forward[0] + rx * right[0] - ry * up[0]
    ray_y = forward[1] + rx * right[1] - ry * up[1]
    ray_z = forward[2] + rx * right[2] - ry * up[2]
    depth_m = np.asarray(depth_cm, dtype=np.float64) / 100.0
    valid = np.asarray(depth_cm) > 0
    return (
        float(pose.x) + depth_m * ray_x,
        float(pose.y) + depth_m * ray_y,
        float(pose.z) + depth_m * ray_z,
        valid,
    )


def _world_of(
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    ego_z: np.ndarray,
    pos: tuple[float, float, float],
    axes: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    right, forward, up = axes
    wx = pos[0] + ego_x * right[0] + ego_y * forward[0] + ego_z * up[0]
    wy = pos[1] + ego_x * right[1] + ego_y * forward[1] + ego_z * up[1]
    wz = pos[2] + ego_x * right[2] + ego_y * forward[2] + ego_z * up[2]
    return wx, wy, wz


def _side_height(values: Any, index: int) -> float | None:
    if not isinstance(values, (list, tuple)) or index >= len(values):
        return None
    return _finite(values[index])


def road_profile_z(
    road: Any, world_x: np.ndarray, world_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Center, left-edge, and right-edge Z at each query, along the centerline.

    A query farther than half the road width plus 2 m is NaN. None when the
    road has no centerline heights. A side is NaN where that edge has no height.
    """
    if road is None or getattr(road, "z", None) is None:
        return None
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    lzs: list[float | None] = []
    rzs: list[float | None] = []
    for index, ((x, y), z) in enumerate(zip(road.xy, road.z)):
        height = _finite(z)
        if height is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
        zs.append(height)
        lzs.append(_side_height(getattr(road, "left_z", None), index))
        rzs.append(_side_height(getattr(road, "right_z", None), index))
    if len(xs) < 2:
        return None
    wx = np.asarray(world_x, dtype=np.float64)
    wy = np.asarray(world_y, dtype=np.float64)
    best_d = np.full(wx.shape, np.inf, dtype=np.float64)
    best_z = np.full(wx.shape, np.nan, dtype=np.float64)
    best_l = np.full(wx.shape, np.nan, dtype=np.float64)
    best_r = np.full(wx.shape, np.nan, dtype=np.float64)
    for index in range(len(xs) - 1):
        ax, ay, az = xs[index], ys[index], zs[index]
        bx, by, bz = xs[index + 1], ys[index + 1], zs[index + 1]
        vx, vy = bx - ax, by - ay
        length2 = vx * vx + vy * vy
        if length2 < 1e-8:
            continue
        t = np.clip(((wx - ax) * vx + (wy - ay) * vy) / length2, 0.0, 1.0)
        dist = np.hypot(wx - (ax + t * vx), wy - (ay + t * vy))
        better = dist < best_d
        best_d = np.where(better, dist, best_d)
        best_z = np.where(better, az + t * (bz - az), best_z)
        lz0, lz1 = lzs[index], lzs[index + 1]
        rz0, rz1 = rzs[index], rzs[index + 1]
        if lz0 is not None and lz1 is not None:
            best_l = np.where(better, lz0 + t * (lz1 - lz0), best_l)
        else:
            best_l = np.where(better, np.nan, best_l)
        if rz0 is not None and rz1 is not None:
            best_r = np.where(better, rz0 + t * (rz1 - rz0), best_r)
        else:
            best_r = np.where(better, np.nan, best_r)
    limit = max(float(getattr(road, "width_m", 0.0) or 0.0) * 0.5 + 2.0, 4.0)
    near = best_d <= limit
    return (
        np.where(near, best_z, np.nan),
        np.where(near, best_l, np.nan),
        np.where(near, best_r, np.nan),
    )


def interpolated_road_z(road: Any, world_x: np.ndarray, world_y: np.ndarray) -> np.ndarray | None:
    """World Z of ``road`` at each query, interpolated along the centerline.

    ``RoadPoly.z`` is already world Z. A query farther than half the road
    width plus 2 m from that centerline is NaN. None when the road has no heights.
    """
    profile = road_profile_z(road, world_x, world_y)
    if profile is None:
        return None
    return profile[0]


def _painted_samples(lane_gt: Any, lane_valid: Any) -> list[tuple[float, float, float]]:
    """``(y, center_x, half_width)`` where center, left, and right are valid."""
    if lane_gt is None or lane_valid is None:
        return []
    values = np.asarray(lane_gt, dtype=np.float64).reshape(-1)
    mask = np.asarray(lane_valid, dtype=np.float64).reshape(-1)
    samples: list[tuple[float, float, float]] = []
    for index, y_m in enumerate(LOOKAHEADS_M):
        slot = 3 * index
        if slot + 2 >= values.size or slot + 2 >= mask.size:
            break
        if float(mask[slot]) <= 0.5 or float(mask[slot + 1]) <= 0.5 or float(mask[slot + 2]) <= 0.5:
            continue
        center, left, right = float(values[slot]), float(values[slot + 1]), float(values[slot + 2])
        if not (math.isfinite(center) and math.isfinite(left) and math.isfinite(right)):
            continue
        samples.append((float(y_m), center, 0.5 * abs(right - left)))
    return samples


def corridor_center_half(
    ego_y: np.ndarray,
    lane_gt: Any,
    lane_valid: Any,
    road_width_m: Any,
    road_offset_m: Any,
    road_heading_err_rad: Any,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Ego-frame ``(center_x, half_width)`` at each ``ego_y``, or None.

    Painted samples win. Otherwise the center and half-width are the ones
    ``pack_road_edge`` stores: ``half = road_width_m / 2`` and
    ``center = -road_offset_m + y * sin(heading)``.
    """
    y = np.asarray(ego_y, dtype=np.float64)
    samples = _painted_samples(lane_gt, lane_valid)
    if samples:
        ys = np.array([item[0] for item in samples], dtype=np.float64)
        centers = np.array([item[1] for item in samples], dtype=np.float64)
        halves = np.array([item[2] for item in samples], dtype=np.float64)
        if len(samples) == 1:
            return np.full(y.shape, centers[0]), np.full(y.shape, halves[0])
        return np.interp(y, ys, centers), np.interp(y, ys, halves)
    width = _finite(road_width_m)
    offset = _finite(road_offset_m)
    if width is None or offset is None or width <= 0.0:
        return None
    heading = _finite(road_heading_err_rad)
    angle = 0.0 if heading is None else heading
    half = float(width) * 0.5
    center = -float(offset) + y * math.sin(angle)
    return center, np.full(y.shape, half)


def _paved_mask(sem: np.ndarray | None, sem_classes: Mapping[str, Any] | None, shape: tuple[int, ...]) -> np.ndarray:
    """True where the semantic id is a paved allowlist name."""
    mask = np.zeros(shape, dtype=bool)
    if sem is None:
        return mask
    ids = np.asarray(sem)
    if ids.shape != shape:
        return mask
    for cid, name in _class_names(sem_classes).items():
        if name in DRIVABLE_CLASS_ALLOWLIST:
            mask |= ids == cid
    return mask


def _ransac_slope(lateral: np.ndarray, height: np.ndarray) -> float | None:
    """Slope through the centerline, or None when fewer than 20 inliers agree.

    One hypothesis per sample (``height / lateral``) plus a flat road.
    Inliers sit within ``FIT_RESIDUAL_M`` (0.05 m), under a curb face.
    """
    side = np.asarray(lateral, dtype=np.float64).reshape(-1)
    values = np.asarray(height, dtype=np.float64).reshape(-1)
    if int(side.size) < FIT_MIN_INLIERS:
        return None
    hypotheses = [0.0]
    usable = np.abs(side) >= 0.25
    if bool(usable.any()):
        hypotheses.extend((values[usable] / side[usable]).tolist())
    best_count = 0
    best: np.ndarray | None = None
    for slope in hypotheses:
        if not math.isfinite(float(slope)):
            continue
        inl = np.abs(values - float(slope) * side) <= FIT_RESIDUAL_M
        count = int(inl.sum())
        if count > best_count:
            best_count = count
            best = inl
    if best is None or best_count < FIT_MIN_INLIERS:
        return None
    samples = side[best]
    denom = float(np.dot(samples, samples))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(samples, values[best]) / denom)


def _fit_side_slopes(
    lateral: np.ndarray,
    rough: np.ndarray,
    inside: np.ndarray,
    in_band: np.ndarray,
    valid: np.ndarray,
    sem: np.ndarray | None,
    sem_classes: Mapping[str, Any] | None,
) -> tuple[float | None, float | None]:
    """Per-side fallback slope from paved in-corridor points.

    The 0.5 m edge band and obstacle candidates (``|height| > 0.15`` m above
    the centerline) are not seeds. A side with fewer than 20 inliers has no slope.
    """
    finite = np.asarray(valid, dtype=bool) & np.isfinite(rough) & np.isfinite(lateral)
    paved = _paved_mask(sem, sem_classes, np.shape(rough))
    # In the corridor, not in the edge band, and not an obstacle candidate.
    seed = finite & paved & np.asarray(inside, dtype=bool) & ~np.asarray(in_band, dtype=bool)
    seed = seed & (np.abs(rough) <= 0.15)
    left = right = None
    for sign, slot in ((1.0, "right"), (-1.0, "left")):
        sel = seed & (lateral * sign >= 0.25)
        slope = _ransac_slope(lateral[sel], rough[sel]) if int(sel.sum()) else None
        if slot == "right":
            right = slope
        else:
            left = slope
    return left, right


def _edge_slopes(
    z_center: np.ndarray, z_left: np.ndarray, z_right: np.ndarray, half_width: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(edge_z - z_center) / signed half-width``. Right is the positive side."""
    left = np.full(np.shape(z_center), np.nan, dtype=np.float64)
    right = np.full(np.shape(z_center), np.nan, dtype=np.float64)
    if half_width <= 1e-6:
        return left, right
    ok_l = np.isfinite(z_left) & np.isfinite(z_center)
    ok_r = np.isfinite(z_right) & np.isfinite(z_center)
    left = np.where(ok_l, (z_left - z_center) / (-half_width), left)
    right = np.where(ok_r, (z_right - z_center) / half_width, right)
    return left, right


def depth_range_pad_m(depth_m: np.ndarray) -> np.ndarray:
    """Metres added to a height threshold for depth quantization.

    Spec #217 has not defined ``pad(r)``. This stays 0 so a follow-up is this
    return. A blocked curb needs ``pad <= 0.02`` m; do not invent a range curve here.
    """
    return np.zeros(np.shape(depth_m), dtype=np.float64)


def _class_names(sem_classes: Mapping[str, Any] | None) -> dict[int, str]:
    if not isinstance(sem_classes, Mapping):
        return {}
    raw = sem_classes.get("classes", sem_classes)
    if not isinstance(raw, Mapping):
        return {}
    names: dict[int, str] = {}
    for key, name in raw.items():
        try:
            names[int(key)] = str(name)
        except (TypeError, ValueError):
            continue
    return names


def proven_blocked_names(
    sem_classes: Mapping[str, Any] | None, blocked_names: Iterable[str] | None
) -> frozenset[str]:
    """Names a human copied from this file. Coarse buckets and strangers are dropped.

    Nothing is added because it sounds like a wall, a tree, or a pole.
    """
    if not blocked_names:
        return frozenset()
    present = set(_class_names(sem_classes).values())
    proven: list[str] = []
    for name in blocked_names:
        text = str(name).strip()
        if not text or text.upper() in COARSE_CLASS_NAMES:
            continue
        if text not in present or text in DRIVABLE_CLASS_ALLOWLIST or text.lower() == "unknown":
            continue
        if text not in proven:
            proven.append(text)
    return frozenset(proven)


def _bin(ego_x: np.ndarray, ego_y: np.ndarray, classes: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Blocked wins over drivable. Class 0 does not vote."""
    grid = np.zeros((GRID_H, GRID_W), np.uint8)
    col = np.floor((ego_x - X_MIN) / (X_MAX - X_MIN) * GRID_W).astype(np.int64)
    row = np.floor((ego_y - Y_MIN) / (Y_MAX - Y_MIN) * GRID_H).astype(np.int64)
    inside = valid & (col >= 0) & (col < GRID_W) & (row >= 0) & (row < GRID_H)
    drivable = np.zeros((GRID_H, GRID_W), np.int32)
    blocked = np.zeros((GRID_H, GRID_W), np.int32)
    drive = inside & (classes == DRIVABLE)
    block = inside & (classes == BLOCKED)
    if bool(drive.any()):
        np.add.at(drivable, (row[drive], col[drive]), 1)
    if bool(block.any()):
        np.add.at(blocked, (row[block], col[block]), 1)
    grid[drivable > 0] = DRIVABLE
    grid[blocked > 0] = BLOCKED
    return grid


def _clearance(
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    blocked: np.ndarray,
    lane_gt: Any,
    lane_valid: Any,
) -> tuple[float | None, bool]:
    """Minimum path station of a blocked foot within 0.5 m of the painted center.

    Masked unless the 5 m and 10 m center slots are valid. The station is the
    lookahead distance along that polyline, not a straight invented corridor.
    """
    if not lane_center_is_valid(lane_valid):
        return None, False
    if lane_gt is None or lane_valid is None:
        return None, False
    values = np.asarray(lane_gt, dtype=np.float64).reshape(-1)
    mask = np.asarray(lane_valid, dtype=np.float64).reshape(-1)
    samples: list[tuple[float, float]] = []
    for index, y_m in enumerate(LOOKAHEADS_M):
        slot = 3 * index
        if slot >= values.size or slot >= mask.size:
            break
        if float(mask[slot]) <= 0.5 or not math.isfinite(float(values[slot])):
            continue
        samples.append((float(y_m), float(values[slot])))
    if len(samples) < 2 or not bool(np.asarray(blocked).any()):
        return None, False
    xs = ego_x[blocked]
    ys = ego_y[blocked]
    best = None
    for (s0, x0), (s1, x1) in zip(samples, samples[1:]):
        vx, vy = x1 - x0, s1 - s0
        length2 = vx * vx + vy * vy
        if length2 < 1e-8:
            continue
        t = np.clip(((xs - x0) * vx + (ys - s0) * vy) / length2, 0.0, 1.0)
        dist = np.hypot(xs - (x0 + t * vx), ys - (s0 + t * vy))
        near = dist <= CLEAR_FOOT_M
        if not bool(near.any()):
            continue
        station = float(np.min(s0 + t[near] * (s1 - s0)))
        best = station if best is None else min(best, station)
    if best is None:
        return None, False
    return float(best), True


def _empty(cam: str, t: Any, mask_src: str) -> dict[str, Any]:
    return {
        "t": t,
        "cam": cam,
        "grid_h": GRID_H,
        "grid_w": GRID_W,
        "x_min": X_MIN,
        "x_max": X_MAX,
        "y_min": Y_MIN,
        "y_max": Y_MAX,
        "mask": [UNKNOWN] * (GRID_H * GRID_W),
        "mask_src": mask_src,
        "clear_eps_m": CLEAR_EPS_M,
        "curb_eps_m": CURB_EPS_M,
        "edge_band_m": EDGE_BAND_M,
        "clear_m": None,
        "clear_valid": False,
    }


def _semantic_classes(
    sem: np.ndarray, depth_valid: np.ndarray, sem_classes: Mapping[str, Any] | None, proven: frozenset[str]
) -> np.ndarray:
    names = _class_names(sem_classes)
    present = set(names.values())
    classes = np.zeros(sem.shape, np.uint8)
    ids = np.asarray(sem)
    for cid, name in names.items():
        if cid == 0 or name not in present:
            continue
        here = depth_valid & (ids == cid)
        if name in DRIVABLE_CLASS_ALLOWLIST:
            classes[here] = DRIVABLE
        elif name in proven:
            classes[here] = BLOCKED
    return classes


def label_obstacle_frame(
    depth_cm: np.ndarray | None,
    *,
    pose: Any,
    lane_gt: Any = None,
    lane_gt_valid: Any = None,
    road_width_m: Any = None,
    road_offset_m: Any = None,
    road_heading_err_rad: Any = None,
    road: Any = None,
    ego_pos: Any = None,
    ego_dir: Any = None,
    ego_up: Any = None,
    sem: np.ndarray | None = None,
    sem_classes: Mapping[str, Any] | None = None,
    blocked_names: Iterable[str] | None = None,
    cam: str = "main",
    t: Any = None,
) -> dict[str, Any]:
    """One tick's BEV mask. Missing depth, height, or corridor stays unknown."""
    proven = proven_blocked_names(sem_classes, blocked_names)
    use_sem = bool(proven) and sem is not None
    record = _empty(cam, t, "sem" if use_sem else "geom")
    if depth_cm is None:
        return record
    depth = np.asarray(depth_cm)
    if depth.ndim != 2 or depth.size == 0:
        return record
    if use_sem and np.asarray(sem).shape != depth.shape:
        use_sem = False
        record["mask_src"] = "geom"
    ego_x, ego_y, ego_z, valid = _unproject(pose, depth)
    if not bool(valid.any()):
        return record
    if use_sem:
        classes = _semantic_classes(np.asarray(sem), valid, sem_classes, proven)
        blocked = valid & (classes == BLOCKED)
    else:
        pos = _vec3(ego_pos, missing_z=None)
        axes = vehicle_axes(ego_dir, ego_up)
        if pos is None or axes is None:
            return record
        world_x, world_y, world_z = _world_of(ego_x, ego_y, ego_z, pos, axes)
        profile = road_profile_z(road, world_x, world_y)
        corridor = corridor_center_half(
            ego_y, lane_gt, lane_gt_valid, road_width_m, road_offset_m, road_heading_err_rad
        )
        if profile is None or corridor is None:
            return record
        road_z, z_left, z_right = profile
        center, half = corridor
        lateral = ego_x - center
        inside = np.abs(lateral) <= (half + 1e-6)
        in_band = (np.abs(lateral) > (half + 1e-6)) & (np.abs(lateral) <= (half + EDGE_BAND_M + 1e-6))
        rough = world_z - road_z
        half_width = float(getattr(road, "width_m", 0.0) or 0.0) * 0.5
        edge_left, edge_right = _edge_slopes(road_z, z_left, z_right, half_width)
        fit_left, fit_right = _fit_side_slopes(
            lateral, rough, inside, in_band, valid & np.isfinite(rough), sem, sem_classes
        )
        slope = np.full(lateral.shape, np.nan, dtype=np.float64)
        if fit_right is not None:
            slope = np.where(lateral >= 0.0, fit_right, slope)
        if fit_left is not None:
            slope = np.where(lateral < 0.0, fit_left, slope)
        slope = np.where((lateral >= 0.0) & np.isfinite(edge_right), edge_right, slope)
        slope = np.where((lateral < 0.0) & np.isfinite(edge_left), edge_left, slope)
        have = valid & np.isfinite(rough) & np.isfinite(slope)
        height = np.where(have, rough - slope * lateral, np.nan)
        pad = depth_range_pad_m(np.asarray(depth, dtype=np.float64) / 100.0)
        clear_tol = CLEAR_EPS_M + pad
        curb_tol = CURB_EPS_M + pad
        classes = np.zeros(depth.shape, np.uint8)
        corridor_pts = have & inside
        classes[corridor_pts & (height > clear_tol)] = BLOCKED
        classes[corridor_pts & (height <= clear_tol) & (height >= -clear_tol)] = DRIVABLE
        band = have & in_band
        classes[band & (height >= curb_tol)] = BLOCKED
        blocked = classes == BLOCKED
    grid = _bin(ego_x, ego_y, classes, valid)
    clear_m, clear_ok = _clearance(ego_x, ego_y, blocked, lane_gt, lane_gt_valid)
    record["mask"] = [int(value) for value in grid.reshape(-1)]
    record["clear_m"] = None if not clear_ok else float(clear_m)
    record["clear_valid"] = bool(clear_ok)
    return record


def _sidecar_index(root: Path) -> dict[str, dict[Any, dict[str, Any]]] | None:
    path = Path(root) / _SIDECAR
    if not path.is_file():
        return None
    by_i: dict[int, dict[str, Any]] = {}
    by_t: dict[float, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            continue
        if row.get("i") is not None:
            try:
                by_i[int(row["i"])] = row
            except (TypeError, ValueError):
                pass
        if row.get("t") is not None:
            try:
                by_t[round(float(row["t"]), 6)] = row
            except (TypeError, ValueError):
                pass
    return {"i": by_i, "t": by_t}


def obstacle_targets(root: Path | str, index: int | None = None, t: Any = None) -> dict[str, np.ndarray]:
    """Trainer tensors. A missing sidecar or row is present=0 and an all-unknown mask."""
    mask = np.zeros((GRID_H, GRID_W), np.int64)
    clear = np.float32(0.0)
    clear_valid = np.float32(0.0)
    present = np.float32(0.0)
    found = None
    index_map = _sidecar_index(Path(root))
    if index_map is not None:
        if index is not None and int(index) in index_map["i"]:
            found = index_map["i"][int(index)]
        elif t is not None:
            try:
                found = index_map["t"].get(round(float(t), 6))
            except (TypeError, ValueError):
                found = None
    if found is not None:
        flat = np.asarray(found.get("mask", []), dtype=np.int64).reshape(-1)
        view = mask.reshape(-1)
        count = min(int(flat.size), int(view.size))
        view[:count] = flat[:count]
        if found.get("clear_valid") and found.get("clear_m") is not None:
            number = _finite(found.get("clear_m"))
            if number is not None:
                clear = np.float32(number)
                clear_valid = np.float32(1.0)
        present = np.float32(1.0)
    return {
        "obstacle_mask": mask,
        "obstacle_clear": clear,
        "obstacle_clear_valid": clear_valid,
        "obstacle_present": present,
    }
