"""Reality View step 2 (C prototype, CPU, offline).

Spec: ``docs/specs/reality_view_3dgs.md`` (Feasibility plan, step 2).
Unprojects valid metric planar depth + RGB from one or more pinhole views into
3D points / screen-space billboards rendered from a chase pose, producing an RGB
image, a finite rendered depth buffer, and an observed mask.

Isolation & safety rules:
* Offline / test-process only. Never imported by the supervisor, live vision
  loop, or control/safety paths (AEB, engage, heartbeat).
* Flag defaults OFF (``REALITY_VIEW_ENABLED_DEFAULT = False``).
* Unknown / invalid depth (NaN, +/-inf, <= 0, < near_m, > far_m, masked out)
  never produces rendered points or observed/free pixels.
* No sim GT reads or imports. No new runtime dependencies (NumPy + stdlib only).
"""
from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

REALITY_VIEW_ENABLED_DEFAULT: bool = False
UNKNOWN_RGB: tuple[int, int, int] = (24, 20, 18)

MAX_RENDER_W: int = 1920
MAX_RENDER_H: int = 1080
MAX_INPUT_PIXELS: int = 4_000_000
MAX_POINTS: int = 300_000
MAX_BILLBOARD_RADIUS_PX: int = 16

DEFAULT_CHASE_POS_M: tuple[float, float, float] = (0.0, -8.0, 4.0)
DEFAULT_CHASE_HEIGHT_M: float = 4.0
DEFAULT_CHASE_YAW_DEG: float = 0.0
DEFAULT_CHASE_PITCH_DEG: float = -20.0
DEFAULT_CHASE_FOV_V_DEG: float = 60.0
DEFAULT_CHASE_NEAR_M: float = 0.05
DEFAULT_CHASE_FAR_M: float = 100.0


@dataclass(frozen=True)
class CameraPose:
    """Pinhole camera intrinsics + ego-frame pose (+X right, +Y forward, +Z up)."""

    cam_id: str
    pos: tuple[float, float, float]
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    fwd: tuple[float, float, float]
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    near_m: float
    far_m: float

    def rotation_matrix(self) -> np.ndarray:
        """3x3 matrix whose rows are [right, up, fwd] in ego coordinates."""
        return np.array([self.right, self.up, self.fwd], dtype=np.float64)


@dataclass(frozen=True)
class DepthColorView:
    """One camera RGB + planar metric depth frame in ego coordinates."""

    cam: CameraPose
    rgb: np.ndarray
    depth_m: np.ndarray
    valid_mask: np.ndarray | None = None


@dataclass(frozen=True)
class RenderResult:
    """Chase-camera render output: RGB, finite depth buffer, and observed mask."""

    rgb: np.ndarray
    depth_m: np.ndarray
    observed: np.ndarray
    point_count: int


def is_reality_view_enabled(cfg: Mapping[str, Any] | None = None, *, cli_enabled: bool = False) -> bool:
    """Return whether Reality View is enabled (strictly defaults to False)."""
    if cli_enabled:
        return True
    if not isinstance(cfg, Mapping):
        return REALITY_VIEW_ENABLED_DEFAULT
    viz = cfg.get("viz")
    if not isinstance(viz, Mapping):
        return REALITY_VIEW_ENABLED_DEFAULT
    return viz.get("reality_view_enabled") is True



_FORBIDDEN_GT_CAM_IDS: frozenset[str] = frozenset(
    {
        "depth_gt",
        "gt_depth",
        "sim_depth",
        "aux_depth",
        "sem_gt",
        "gt_sem",
        "path_world",
        "debug_gt",
        "labels",
    }
)


def _check_cam_id(cam_id: Any) -> str:
    if not isinstance(cam_id, str) or not cam_id.strip():
        raise ValueError("cam_id must be a non-empty string")
    low = cam_id.strip().lower()
    if low in _FORBIDDEN_GT_CAM_IDS or low.startswith("gt_") or low.endswith("_gt") or "sim_depth" in low:
        raise ValueError(f"forbidden GT/aux cam_id on prototype path: {cam_id!r}")
    return cam_id


def _check_int_dim(name: str, val: Any, lo: int, hi: int) -> int:
    if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {type(val).__name__}")
    iv = int(val)
    if iv < lo or iv > hi:
        raise ValueError(f"{name}={iv} out of range [{lo}, {hi}]")
    return iv


def _check_finite_float(name: str, val: Any) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be numeric, got {type(val).__name__}")
    fv = float(val)
    if not math.isfinite(fv):
        raise ValueError(f"{name} must be finite, got {val!r}")
    return fv


def _unit_vec(v: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(v) != 3:
        raise ValueError(f"{name} must have length 3")
    arr = np.array([_check_finite_float(name, x) for x in v], dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    arr = arr / norm
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def camera_from_angles(
    cam_id: str,
    pos_m: Sequence[float],
    yaw_deg: float,
    pitch_deg: float,
    fov_v_deg: float,
    width: int,
    height: int,
    *,
    near_m: float = 0.05,
    far_m: float = 100.0,
) -> CameraPose:
    """Build a square-pixel pinhole ``CameraPose`` in the GVD ego frame."""
    cid = _check_cam_id(cam_id)
    w = _check_int_dim("width", width, 1, MAX_RENDER_W)
    h = _check_int_dim("height", height, 1, MAX_RENDER_H)
    fov_v = _check_finite_float("fov_v_deg", fov_v_deg)
    if fov_v <= 0.0 or fov_v >= 180.0:
        raise ValueError(f"fov_v_deg={fov_v} must be in (0, 180)")
    near = _check_finite_float("near_m", near_m)
    far = _check_finite_float("far_m", far_m)
    if near <= 0.0 or far <= near:
        raise ValueError(f"invalid clip planes near_m={near}, far_m={far}")
    if len(pos_m) != 3:
        raise ValueError("pos_m must have length 3")
    pos = tuple(_check_finite_float("pos_m", v) for v in pos_m)

    yaw = math.radians(_check_finite_float("yaw_deg", yaw_deg))
    pitch = math.radians(_check_finite_float("pitch_deg", pitch_deg))
    fwd_raw = (
        math.sin(yaw) * math.cos(pitch),
        math.cos(yaw) * math.cos(pitch),
        math.sin(pitch),
    )
    up_raw = (
        -math.sin(yaw) * math.sin(pitch),
        -math.cos(yaw) * math.sin(pitch),
        math.cos(pitch),
    )
    fwd_np = np.array(fwd_raw, dtype=np.float64)
    up_np = np.array(up_raw, dtype=np.float64)
    right_np = np.cross(fwd_np, up_np)
    right = _unit_vec(right_np.tolist(), "right")
    fwd = _unit_vec(fwd_np.tolist(), "fwd")
    up_ortho = np.cross(np.array(right, dtype=np.float64), np.array(fwd, dtype=np.float64))
    up = _unit_vec(up_ortho.tolist(), "up")

    fy = (float(h) / 2.0) / math.tan(math.radians(fov_v) / 2.0)
    return CameraPose(
        cam_id=cid,
        pos=(pos[0], pos[1], pos[2]),
        right=right,
        up=up,
        fwd=fwd,
        width=w,
        height=h,
        fx=fy,
        fy=fy,
        cx=float(w) / 2.0,
        cy=float(h) / 2.0,
        near_m=near,
        far_m=far,
    )


def chase_pose_default(width: int = 320, height: int = 180) -> CameraPose:
    """Default chase pose matching ``config/cameras.yaml`` ``chase_camera``."""
    return camera_from_angles(
        cam_id="chase",
        pos_m=DEFAULT_CHASE_POS_M,
        yaw_deg=DEFAULT_CHASE_YAW_DEG,
        pitch_deg=DEFAULT_CHASE_PITCH_DEG,
        fov_v_deg=DEFAULT_CHASE_FOV_V_DEG,
        width=width,
        height=height,
        near_m=DEFAULT_CHASE_NEAR_M,
        far_m=DEFAULT_CHASE_FAR_M,
    )


def _validate_pose(cam: CameraPose) -> None:
    _check_cam_id(cam.cam_id)
    _check_int_dim("width", cam.width, 1, MAX_RENDER_W)
    _check_int_dim("height", cam.height, 1, MAX_RENDER_H)
    fx = _check_finite_float("fx", cam.fx)
    fy = _check_finite_float("fy", cam.fy)
    _check_finite_float("cx", cam.cx)
    _check_finite_float("cy", cam.cy)
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"focal lengths must be positive, got fx={fx}, fy={fy}")
    near = _check_finite_float("near_m", cam.near_m)
    far = _check_finite_float("far_m", cam.far_m)
    if near <= 0.0 or far <= near:
        raise ValueError(f"invalid clip planes near_m={near}, far_m={far}")
    for v in cam.pos:
        _check_finite_float("pos", v)
    _unit_vec(cam.right, "right")
    _unit_vec(cam.up, "up")
    _unit_vec(cam.fwd, "fwd")


def _unproject_view(view: DepthColorView) -> tuple[np.ndarray, np.ndarray]:
    """Unproject valid pixels of one view into (N, 3) ego points and (N, 3) uint8 RGB."""
    _validate_pose(view.cam)
    rgb = np.asarray(view.rgb)
    depth = np.asarray(view.depth_m)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {rgb.shape}")
    if depth.ndim != 2:
        raise ValueError(f"depth_m must have shape (H, W), got {depth.shape}")
    h, w = depth.shape
    if h < 1 or w < 1:
        raise ValueError(f"input view dimensions must be >= 1, got {depth.shape}")
    if h * w > MAX_INPUT_PIXELS:
        raise ValueError(f"input view pixel count {h * w} exceeds MAX_INPUT_PIXELS={MAX_INPUT_PIXELS}")
    if rgb.shape[:2] != (h, w):
        raise ValueError(f"rgb shape {rgb.shape[:2]} does not match depth_m shape {(h, w)}")
    if rgb.dtype != np.uint8:
        rgb = np.clip(np.nan_to_num(rgb.astype(np.float64), nan=0.0, posinf=255.0, neginf=0.0), 0.0, 255.0).astype(
            np.uint8
        )

    d64 = depth.astype(np.float64, copy=False)
    valid = np.isfinite(d64) & (d64 > 0.0) & (d64 >= view.cam.near_m) & (d64 <= view.cam.far_m)
    if view.valid_mask is not None:
        mask = np.asarray(view.valid_mask)
        if mask.shape != (h, w):
            raise ValueError(f"valid_mask shape {mask.shape} does not match depth_m shape {(h, w)}")
        valid = valid & mask.astype(bool, copy=False)

    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)

    ds = d64[ys, xs]
    xc = ((xs.astype(np.float64) + 0.5 - view.cam.cx) / view.cam.fx) * ds
    yc = -((ys.astype(np.float64) + 0.5 - view.cam.cy) / view.cam.fy) * ds
    zc = ds
    rot = view.cam.rotation_matrix()
    cam_pts = np.stack([xc, yc, zc], axis=-1)
    ego_pts = np.asarray(view.cam.pos, dtype=np.float64)[None, :] + cam_pts @ rot
    cols = rgb[ys, xs].astype(np.uint8, copy=False)
    return ego_pts, cols


def render_chase_depth_billboards(
    views: Sequence[DepthColorView],
    chase_cam: CameraPose,
    *,
    billboard_radius_px: int = 1,
    max_points: int = MAX_POINTS,
) -> RenderResult:
    """Unproject valid depth across ``views`` and render chase-pose billboards on CPU."""
    _validate_pose(chase_cam)
    rad = _check_int_dim("billboard_radius_px", billboard_radius_px, 0, MAX_BILLBOARD_RADIUS_PX)
    cap = _check_int_dim("max_points", max_points, 1, MAX_POINTS)

    out_h = int(chase_cam.height)
    out_w = int(chase_cam.width)
    out_rgb = np.empty((out_h, out_w, 3), dtype=np.uint8)
    out_rgb[:, :] = np.asarray(UNKNOWN_RGB, dtype=np.uint8)
    out_depth = np.zeros((out_h, out_w), dtype=np.float32)
    out_obs = np.zeros((out_h, out_w), dtype=bool)

    if not views:
        return RenderResult(rgb=out_rgb, depth_m=out_depth, observed=out_obs, point_count=0)

    pts_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    for view in views:
        pts, cols = _unproject_view(view)
        if pts.shape[0] > 0:
            pts_list.append(pts)
            cols_list.append(cols)

    if not pts_list:
        return RenderResult(rgb=out_rgb, depth_m=out_depth, observed=out_obs, point_count=0)

    all_pts = np.concatenate(pts_list, axis=0)
    all_cols = np.concatenate(cols_list, axis=0)
    if all_pts.shape[0] > cap:
        keep_idx = np.linspace(0, all_pts.shape[0] - 1, cap, dtype=np.int64)
        all_pts = all_pts[keep_idx]
        all_cols = all_cols[keep_idx]

    # Transform ego points into chase camera coordinates and cull behind-camera / out-of-range points.
    rel = all_pts - np.asarray(chase_cam.pos, dtype=np.float64)[None, :]
    cam_pts = rel @ chase_cam.rotation_matrix().T
    zc = cam_pts[:, 2]
    in_z = np.isfinite(zc) & (zc > chase_cam.near_m) & (zc <= chase_cam.far_m)
    if not np.any(in_z):
        return RenderResult(rgb=out_rgb, depth_m=out_depth, observed=out_obs, point_count=0)

    cam_pts = cam_pts[in_z]
    cols_z = all_cols[in_z]
    zc = cam_pts[:, 2]

    uc = chase_cam.cx + chase_cam.fx * (cam_pts[:, 0] / zc)
    vc = chase_cam.cy - chase_cam.fy * (cam_pts[:, 1] / zc)
    px = np.floor(uc).astype(np.int64)
    py = np.floor(vc).astype(np.int64)

    in_frustum = (px >= 0) & (px < out_w) & (py >= 0) & (py < out_h)
    if not np.any(in_frustum):
        return RenderResult(rgb=out_rgb, depth_m=out_depth, observed=out_obs, point_count=0)

    px = px[in_frustum]
    py = py[in_frustum]
    pz = zc[in_frustum]
    pcols = cols_z[in_frustum]
    n_pts = int(px.size)

    if rad == 0:
        hy, hx, hz, hcols, htie = py, px, pz, pcols, np.arange(n_pts, dtype=np.int64)
    else:
        offs = np.arange(-rad, rad + 1, dtype=np.int64)
        dy_g, dx_g = np.meshgrid(offs, offs, indexing="ij")
        dy_f = dy_g.reshape(-1)
        dx_f = dx_g.reshape(-1)
        hy = (py[:, None] + dy_f[None, :]).reshape(-1)
        hx = (px[:, None] + dx_f[None, :]).reshape(-1)
        hz = np.repeat(pz, dy_f.size)
        hcols = np.repeat(pcols, dy_f.size, axis=0)
        htie = np.repeat(np.arange(n_pts, dtype=np.int64), dy_f.size)
        inside = (hx >= 0) & (hx < out_w) & (hy >= 0) & (hy < out_h)
        hy = hy[inside]
        hx = hx[inside]
        hz = hz[inside]
        hcols = hcols[inside]
        htie = htie[inside]

    # Deterministic painter's sort: farthest z painted first so nearest z wins on overwrite;
    # ties broken deterministically by point index.
    order = np.lexsort((htie, -hz))
    sy = hy[order]
    sx = hx[order]
    out_rgb[sy, sx] = hcols[order]
    out_depth[sy, sx] = hz[order].astype(np.float32)
    out_obs[sy, sx] = True

    return RenderResult(rgb=out_rgb, depth_m=out_depth, observed=out_obs, point_count=n_pts)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag)
    crc = zlib.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def encode_png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an ``(H, W, 3) uint8`` RGB array as deterministic PNG bytes (stdlib only)."""
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError(f"encode_png_bytes expects (H, W, 3) uint8, got {arr.shape} {arr.dtype}")
    h, w, _ = arr.shape
    if h < 1 or w < 1 or w > MAX_RENDER_W or h > MAX_RENDER_H:
        raise ValueError(f"invalid PNG image shape {(h, w)}")
    arr = np.ascontiguousarray(arr)
    # Prepend PNG filter byte 0 to each RGB scanline
    rows = np.concatenate([np.zeros((h, 1), dtype=np.uint8), arr.reshape(h, w * 3)], axis=1)
    idat = zlib.compress(rows.tobytes(), level=6)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")
