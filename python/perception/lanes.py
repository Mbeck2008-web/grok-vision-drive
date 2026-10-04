"""OpenCV lanes in the ego frame. Forward cameras keep the windshield map.

Side and rear cameras use each camera's pose so a pixel lands on the ground
beside or behind the car.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

_LOG = logging.getLogger("gvd.perception.lanes")
# One warning per failure text. A fit that fails every tick must not flood the console.
_lane_fit_logged: set[str] = set()
# A later piece joins a chain only when its near end still lies on that chain.
# The live kink steps about 2.6 m sideways between y=10 m and y=12 m, so it
# stays its own polyline. Pieces of one curve stay under this miss.
_JOIN_LATERAL_M = 0.9


@dataclass
class LaneResult:
    conf: float
    lanes_bev: list[list[dict[str, float]]]  # polylines in ego frame
    curvature: float  # 1/m approx


def hough_segments(lines: Any) -> list[tuple[int, int, int, int]]:
    """Normalize HoughLinesP rows to ``(x1, y1, x2, y2)``.

    OpenCV 4 returns ``(N, 1, 4)``. OpenCV 5 returns ``(N, 4)``. Indexing
    ``lines[:, 0]`` on the OpenCV 5 layout yields scalars; ``row[1]`` then
    raises ``IndexError``. ``estimate_lanes`` used to swallow that and report
    ``lane_conf`` 0 on a frame that already had lane paint.
    """
    if lines is None:
        return []
    arr = np.asarray(lines)
    if arr.size == 0:
        return []
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2 or arr.shape[1] < 4:
        return []
    out: list[tuple[int, int, int, int]] = []
    for row in arr[:, :4]:
        out.append((int(row[0]), int(row[1]), int(row[2]), int(row[3])))
    return out


# cam_main maps the top of the frame to about 35 m. The blue path runs to
# 36 m, so a stitch fit uses a longer scale: the same horizon row lands
# past that path. The join rule is unchanged.
_MAIN_FAR_M = 35.0
STITCH_LANE_FAR_M = 70.0


def estimate_lanes(
    bgr: np.ndarray | None,
    *,
    far_m: float = _MAIN_FAR_M,
    roi_top: float = 0.40,
    x_spans: tuple[tuple[int, int], ...] | None = None,
) -> LaneResult:
    """Fit ego-lane paint on one image.

    White and yellow stripes are masked in HSV, so a hazy yellow line that
    grayscale Canny misses can still clear the engage gate. An empty or
    undecodable frame is ``lane_conf`` 0. A Hough row-layout ``IndexError``
    (OpenCV 5 ``(N, 4)`` indexed as OpenCV 4 ``(N, 1, 4)``) is logged and
    re-raised so it cannot look like an empty road. Other fit errors
    (``cv2.error``, ``ValueError``, ``TypeError``) log once and return 0.
    ``far_m`` is the ego distance of the top row. The default is the
    cam_main scale.
    """
    if bgr is None or bgr.size == 0:
        return LaneResult(conf=0.0, lanes_bev=[], curvature=0.0)
    try:
        return _estimate_lanes_impl(
            bgr,
            far_m=float(far_m),
            roi_top=float(roi_top),
            x_spans=x_spans,
        )
    except IndexError as exc:
        _LOG.error(
            "estimate_lanes Hough/layout IndexError (%s); refusing lane_conf=0",
            exc,
        )
        raise
    except (cv2.error, ValueError, TypeError) as exc:
        _log_lane_fit_failure(exc)
        return LaneResult(conf=0.0, lanes_bev=[], curvature=0.0)


def _log_lane_fit_failure(exc: BaseException) -> None:
    key = f"{type(exc).__name__}:{exc}"
    if key in _lane_fit_logged:
        return
    _lane_fit_logged.add(key)
    _LOG.warning(
        "estimate_lanes failed (%s: %s); reporting lane_conf=0",
        type(exc).__name__,
        exc,
    )


def _as_bgr(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    if bgr.shape[2] > 3:
        return np.ascontiguousarray(bgr[:, :, :3])
    return bgr


def lane_paint_mask(bgr: np.ndarray) -> np.ndarray:
    """White and yellow road paint. Grayscale Canny misses a hazy yellow stripe.

    OpenCV HSV: H 0–180. Yellow sits near 15–35. White is low saturation and
    high value, so a bright gray line still counts. Saturation stops at 18:
    hazy sky (BGR 175,185,195 → HSV 15,26,195) is not a white stripe. A flat
    gray frame (V below 185) does not count either.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 185), (180, 18, 255))
    yellow = cv2.inRange(hsv, (8, 40, 80), (42, 255, 255))
    return cv2.bitwise_or(white, yellow)


def lane_roi_mask(height: int, width: int, top_frac: float = 0.40) -> np.ndarray:
    """Forward-cam trapezoid. Top sits above mid-frame so hood-cam lines that
    converge near the horizon are still inside the mask.
    """
    h, w = int(height), int(width)
    top = int(max(0.0, min(0.95, float(top_frac))) * h)
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.array(
        [[
            (int(0.05 * w), h - 1),
            (int(0.36 * w), top),
            (int(0.64 * w), top),
            (int(0.95 * w), h - 1),
        ]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, poly, 255)
    return mask


def _span_of(
    x: float,
    spans: tuple[tuple[int, int], ...] | None,
) -> tuple[int, int] | None:
    """Sector that contains ``x``. A gap pixel uses the nearest sector."""
    if not spans:
        return None
    best: tuple[int, int] | None = None
    best_d = 1e9
    for x0, x1 in spans:
        if x0 <= x < x1:
            return (x0, x1)
        d = min(abs(x - x0), abs(x - (x1 - 1)))
        if d < best_d:
            best, best_d = (x0, x1), d
    return best


def _ego_x(
    x: float,
    width: float,
    x_spans: tuple[tuple[int, int], ...] | None = None,
) -> float:
    """Lateral meters. One camera is ±3 m across its own width.

    A windshield band is several cameras side by side. Scaling the whole
    band as one camera squeezes a real lane under a meter. Each painted
    sector keeps the single-camera scale, so 0.9 m is still a road distance.
    """
    span = _span_of(float(x), x_spans)
    if span is None:
        return ((float(x) / float(width)) - 0.5) * 6.0
    x0, x1 = span
    return ((float(x) - x0) / float(max(1, x1 - x0)) - 0.5) * 6.0


def _ego_point(
    x: float,
    y: float,
    width: float,
    height: float,
    far_m: float = _MAIN_FAR_M,
    x_spans: tuple[tuple[int, int], ...] | None = None,
) -> dict[str, float]:
    """Crude pixel → ego meters (x right, y forward)."""
    ex = _ego_x(x, width, x_spans)
    ey = max(2.0, float(far_m) * (1.0 - float(y) / float(height)))
    return {"x": float(ex), "y": float(ey), "z": 0.0}


def _ordered_ego_segment(
    seg: tuple,
    width: float,
    height: float,
    far_m: float = _MAIN_FAR_M,
    x_spans: tuple[tuple[int, int], ...] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    x1, y1, x2, y2 = seg[0], seg[1], seg[2], seg[3]
    a = _ego_point(x1, y1, width, height, far_m, x_spans)
    b = _ego_point(x2, y2, width, height, far_m, x_spans)
    if a["y"] <= b["y"]:
        return a, b
    return b, a


def _lateral_miss(
    a: dict[str, float],
    b: dict[str, float],
    pt: dict[str, float],
) -> float:
    """Meters of sideways error if `pt` is read off the line through `a` and `b`."""
    dy = b["y"] - a["y"]
    dx = b["x"] - a["x"]
    if abs(dy) < 1e-4:
        return abs(pt["x"] - b["x"])
    pred = b["x"] + (dx / dy) * (pt["y"] - b["y"])
    return abs(pt["x"] - pred)


def chain_lane_segments(
    segs: list,
    width: int,
    height: int,
    far_m: float = _MAIN_FAR_M,
    x_spans: tuple[tuple[int, int], ...] | None = None,
    project: Callable[[float, float], dict[str, float] | None] | None = None,
) -> list[list[dict[str, float]]]:
    """Polylines for one slope group. Pieces that do not meet stay apart.

    Up to eight Hough segments are mapped with the same ego scale as before,
    unless ``project`` is set. A pose projector is how a side or rear pixel
    becomes a ground point. It is not the forward-camera map. A piece is
    appended only when its nearer end lies within ``_JOIN_LATERAL_M`` of a
    segment already on that chain. A sideways jump starts a new polyline
    instead of kinking the one in hand. Each returned line is ordered by
    increasing forward distance, which is negative behind the car.
    """
    pieces: list[tuple[dict[str, float], dict[str, float]]] = []
    for seg in segs[:8]:
        if project is None:
            pieces.append(_ordered_ego_segment(seg, width, height, far_m, x_spans))
            continue
        a = project(float(seg[0]), float(seg[1]))
        b = project(float(seg[2]), float(seg[3]))
        if a is None or b is None:
            continue
        pieces.append((a, b) if a["y"] <= b["y"] else (b, a))
    pieces.sort(key=lambda ab: (ab[0]["y"], ab[0]["x"], ab[1]["y"]))
    chains: list[list[tuple[dict[str, float], dict[str, float]]]] = []
    for near, far in pieces:
        best_i: int | None = None
        best_miss: float | None = None
        for i, chain in enumerate(chains):
            miss = min(_lateral_miss(a, b, near) for a, b in chain)
            if miss <= _JOIN_LATERAL_M and (best_miss is None or miss < best_miss):
                best_i = i
                best_miss = miss
        if best_i is None:
            chains.append([(near, far)])
        else:
            chains[best_i].append((near, far))
    out: list[list[dict[str, float]]] = []
    for chain in chains:
        pts: list[dict[str, float]] = []
        for near, far in chain:
            pts.extend((near, far))
        pts.sort(key=lambda p: (p["y"], p["x"]))
        deduped: list[dict[str, float]] = []
        for p in pts:
            if (
                deduped
                and abs(deduped[-1]["y"] - p["y"]) < 1e-3
                and abs(deduped[-1]["x"] - p["x"]) < 1e-3
            ):
                continue
            deduped.append(p)
        if len(deduped) >= 2:
            out.append(deduped)
    return out


@dataclass(frozen=True)
class CameraPose:
    cam_id: str
    x: float
    y: float
    z: float
    yaw_deg: float
    pitch_deg: float
    fov_v_deg: float
    width: int
    height: int


# Pillar, repeater, and rear. Forward cameras stay on ``_ego_point``.
SIDE_REAR_CAM_IDS = ("pillarL", "pillarR", "repeatL", "repeatR", "rear")

_POSES: dict[str, CameraPose] | None = None
_POSE_MISSING: tuple[str, ...] = ()


def _pose_from_spec(cid: str, spec: dict[str, Any]) -> CameraPose | None:
    pos = spec.get("pos_m")
    res = spec.get("live_res")
    yaw = spec.get("yaw_deg")
    pitch = spec.get("pitch_deg")
    fov = spec.get("fov_v")
    if not isinstance(pos, (list, tuple)) or len(pos) < 3:
        return None
    if yaw is None or pitch is None or fov is None:
        return None
    if not isinstance(res, (list, tuple)) or len(res) < 2:
        return None
    try:
        return CameraPose(
            cam_id=cid,
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            yaw_deg=float(yaw),
            pitch_deg=float(pitch),
            fov_v_deg=float(fov),
            width=int(res[0]),
            height=int(res[1]),
        )
    except (TypeError, ValueError):
        return None


def camera_poses() -> dict[str, CameraPose]:
    """Poses from ``config/cameras.yaml``. A camera with no pose is named and skipped."""
    global _POSES, _POSE_MISSING
    if _POSES is not None:
        return _POSES
    from python.sensors.cameras import CAM_IDS, load_camera_config

    cfg = load_camera_config()
    by_id: dict[str, dict[str, Any]] = {}
    for spec in cfg.get("cameras") or []:
        if isinstance(spec, dict) and spec.get("id"):
            by_id[str(spec["id"])] = spec
    found: dict[str, CameraPose] = {}
    missing: list[str] = []
    for cid in CAM_IDS:
        pose = _pose_from_spec(cid, by_id.get(cid) or {})
        if pose is None:
            missing.append(cid)
            continue
        found[cid] = pose
    _POSES = found
    _POSE_MISSING = tuple(missing)
    return found


def missing_pose_ids() -> tuple[str, ...]:
    camera_poses()
    return _POSE_MISSING


def camera_pose(cid: str) -> CameraPose | None:
    return camera_poses().get(cid)


def project_pose_pixel(
    pose: CameraPose,
    u: float,
    v: float,
    *,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, float] | None:
    """Pixel on this camera to ego ground (x right, y forward, z up).

    The ray uses the camera position, yaw, pitch, vertical fov, and the
    image size. Image-right is camera-right. Image-down is camera-down.
    A ray that does not meet the ground in front of the lens is omitted.
    This is not the forward-camera map: y is not forced to at least 2 m
    and x is not forced into ±3 m.
    """
    from python.sensors.cameras import yaw_pitch_to_dir_up

    w = float(width or pose.width)
    h = float(height or pose.height)
    if w < 2.0 or h < 2.0 or pose.z <= 0.05:
        return None
    fov_v = math.radians(float(pose.fov_v_deg))
    if fov_v <= 1e-3 or fov_v >= math.pi - 1e-3:
        return None
    fov_h = 2.0 * math.atan(math.tan(fov_v * 0.5) * (w / h))
    nx = (float(u) - (w - 1.0) * 0.5) / (w * 0.5)
    ny = (float(v) - (h - 1.0) * 0.5) / (h * 0.5)
    rx = nx * math.tan(fov_h * 0.5)
    ry = ny * math.tan(fov_v * 0.5)
    rz = 1.0
    forward, up = yaw_pitch_to_dir_up(pose.yaw_deg, pose.pitch_deg)
    fx, fy, fz = forward
    ux, uy, uz = up
    right = (
        fy * uz - fz * uy,
        fz * ux - fx * uz,
        fx * uy - fy * ux,
    )
    vx = rz * fx + rx * right[0] - ry * ux
    vy = rz * fy + rx * right[1] - ry * uy
    vz = rz * fz + rx * right[2] - ry * uz
    if vz >= -1e-5:
        return None
    travel = -float(pose.z) / vz
    if travel <= 0.0:
        return None
    return {
        "x": float(pose.x + travel * vx),
        "y": float(pose.y + travel * vy),
        "z": 0.0,
    }


def _full_stitch(bgr: np.ndarray) -> bool:
    from python.perception.stitch360 import GAP_PX, SECTOR_H, SECTOR_W, STITCH_ORDER

    if not isinstance(bgr, np.ndarray) or bgr.ndim < 2:
        return False
    height, width = int(bgr.shape[0]), int(bgr.shape[1])
    expect_w = len(STITCH_ORDER) * SECTOR_W + (len(STITCH_ORDER) - 1) * GAP_PX
    return height == SECTOR_H and width == expect_w


def _stitch_sector(bgr: np.ndarray, cam_id: str) -> np.ndarray | None:
    from python.perception.stitch360 import GAP_PX, SECTOR_W, STITCH_ORDER

    if cam_id not in STITCH_ORDER:
        return None
    index = STITCH_ORDER.index(cam_id)
    x0 = index * (SECTOR_W + GAP_PX)
    return np.ascontiguousarray(bgr[:, x0 : x0 + SECTOR_W])


def posed_lanes_from_image(bgr: np.ndarray, cam_id: str) -> list[list[dict[str, float]]]:
    """Lane pieces on one side or rear camera, in ego meters.

    Forward cameras are not accepted here. A camera with no pose is skipped.
    """
    from python.sensors.cameras import frame_is_unrendered

    if cam_id not in SIDE_REAR_CAM_IDS:
        return []
    pose = camera_pose(cam_id)
    if pose is None or bgr is None or frame_is_unrendered(bgr):
        return []
    image = _as_bgr(bgr)
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    paint = lane_paint_mask(image)
    grad = cv2.morphologyEx(paint, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    near = cv2.dilate(paint, np.ones((3, 3), np.uint8))
    roi = np.zeros((height, width), dtype=np.uint8)
    roi[int(0.12 * height) :, :] = 255
    crop = cv2.bitwise_and(cv2.bitwise_or(cv2.bitwise_and(edges, near), grad), roi)
    min_len = max(12, int(0.22 * max(height, width)))
    lines = cv2.HoughLinesP(
        crop, 1, np.pi / 180, threshold=20, minLineLength=min_len, maxLineGap=40
    )
    segs: list[tuple[int, int, int, int, float]] = []
    for x1, y1, x2, y2 in hough_segments(lines):
        if x1 == x2 and y1 == y2:
            continue
        if x1 == x2:
            slope = 1.0e6
        else:
            slope = (y2 - y1) / float(x2 - x1)
        # A pillar or repeater looks across the car, so a lane beside it
        # runs across the frame. A rear lane does the same. Keep that line.
        segs.append((x1, y1, x2, y2, slope))
    if not segs:
        return []

    def project(x: float, y: float) -> dict[str, float] | None:
        # Sector pixels back into the camera's own resolution, then the pose.
        u = (float(x) + 0.5) * float(pose.width) / float(width) - 0.5
        v = (float(y) + 0.5) * float(pose.height) / float(height) - 0.5
        return project_pose_pixel(pose, u, v)

    return chain_lane_segments(segs, width, height, project=project)


def lanes_from_view(
    main_bgr: np.ndarray | None,
    stitch_bgr: np.ndarray | None = None,
) -> LaneResult:
    """Windshield band on the forward map, plus side and rear pose points.

    An image that is not the eight-camera strip is still the windshield fit.
    An empty windshield band falls back to ``cam_main``. Pillar, repeater,
    and rear sectors on a full strip become their own polylines. They are
    not run through ``_ego_point``. A camera with no pose is skipped.
    """
    from python.perception.stitch360 import forward_lane_view, windshield_x_spans
    from python.sensors.cameras import frame_is_unrendered

    forward: LaneResult | None = None
    posed: list[list[dict[str, float]]] = []
    if stitch_bgr is not None and not frame_is_unrendered(stitch_bgr):
        if _full_stitch(stitch_bgr):
            band = forward_lane_view(stitch_bgr)
            if not frame_is_unrendered(band):
                forward = estimate_lanes(
                    band,
                    far_m=STITCH_LANE_FAR_M,
                    x_spans=windshield_x_spans(band),
                )
            for cid in SIDE_REAR_CAM_IDS:
                if camera_pose(cid) is None:
                    continue
                sector = _stitch_sector(stitch_bgr, cid)
                if sector is None or frame_is_unrendered(sector):
                    continue
                posed.extend(posed_lanes_from_image(sector, cid))
        else:
            forward = estimate_lanes(
                stitch_bgr,
                far_m=STITCH_LANE_FAR_M,
                x_spans=windshield_x_spans(stitch_bgr),
            )
    if forward is None:
        forward = estimate_lanes(main_bgr)
    if not posed:
        return forward
    conf = float(forward.conf)
    if forward.lanes_bev:
        conf = min(1.0, conf + 0.15)
    else:
        conf = max(conf, 0.4)
    return LaneResult(
        conf=conf,
        lanes_bev=[*forward.lanes_bev, *posed],
        curvature=forward.curvature,
    )


def _estimate_lanes_impl(
    bgr: np.ndarray,
    far_m: float = _MAIN_FAR_M,
    roi_top: float = 0.40,
    x_spans: tuple[tuple[int, int], ...] | None = None,
) -> LaneResult:
    bgr = _as_bgr(bgr)
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    paint = lane_paint_mask(bgr)
    # Boundary of a solid stripe. A low-gradient yellow line has almost no Canny edge.
    grad = cv2.morphologyEx(paint, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    # Canny only on the paint. A sky/road wedge is not a stripe, so that
    # silhouette cannot score a left lane and a right lane.
    near = cv2.dilate(paint, np.ones((3, 3), np.uint8))
    crop = cv2.bitwise_and(
        cv2.bitwise_or(cv2.bitwise_and(edges, near), grad),
        lane_roi_mask(h, w, roi_top),
    )
    lines = cv2.HoughLinesP(crop, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=80)
    left, right = [], []
    for x1, y1, x2, y2 in hough_segments(lines):
        if x2 == x1:
            continue
        slope = (y2 - y1) / float(x2 - x1)
        if abs(slope) < 0.3:
            continue
        (left if slope < 0 else right).append((x1, y1, x2, y2, slope))
    conf = 0.0
    lanes_bev: list[list[dict[str, float]]] = []
    curv = 0.0
    if left:
        lanes_bev.extend(chain_lane_segments(left, w, h, far_m, x_spans))
        conf += 0.4
    if right:
        lanes_bev.extend(chain_lane_segments(right, w, h, far_m, x_spans))
        conf += 0.4
    if left and right:
        # crude curvature from mean slope asymmetry
        ls = np.mean([s[-1] for s in left])
        rs = np.mean([s[-1] for s in right])
        curv = float(np.clip((ls + rs) * 0.02, -0.05, 0.05))
        conf = min(1.0, conf + 0.15)
    return LaneResult(conf=float(conf), lanes_bev=lanes_bev, curvature=curv)
