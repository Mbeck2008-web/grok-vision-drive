"""OpenCV lanes on cam_main only (Udacity/Aly-style toy)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

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
        return _estimate_lanes_impl(bgr, far_m=float(far_m), roi_top=float(roi_top))
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


def _ego_point(
    x: float,
    y: float,
    width: float,
    height: float,
    far_m: float = _MAIN_FAR_M,
) -> dict[str, float]:
    """Crude pixel → ego meters (x right, y forward)."""
    ex = ((float(x) / float(width)) - 0.5) * 6.0
    ey = max(2.0, float(far_m) * (1.0 - float(y) / float(height)))
    return {"x": float(ex), "y": float(ey), "z": 0.0}


def _ordered_ego_segment(
    seg: tuple,
    width: float,
    height: float,
    far_m: float = _MAIN_FAR_M,
) -> tuple[dict[str, float], dict[str, float]]:
    x1, y1, x2, y2 = seg[0], seg[1], seg[2], seg[3]
    a = _ego_point(x1, y1, width, height, far_m)
    b = _ego_point(x2, y2, width, height, far_m)
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
) -> list[list[dict[str, float]]]:
    """Polylines for one slope group. Pieces that do not meet stay apart.

    Up to eight Hough segments are mapped with the same ego scale as before.
    A piece is appended only when its nearer end lies within
    ``_JOIN_LATERAL_M`` of a segment already on that chain. A sideways jump
    starts a new polyline instead of kinking the one in hand. Each returned
    line is ordered by increasing forward distance.
    """
    pieces = [_ordered_ego_segment(s, width, height, far_m) for s in segs[:8]]
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


def lanes_from_view(
    main_bgr: np.ndarray | None,
    stitch_bgr: np.ndarray | None = None,
) -> LaneResult:
    """Fit lanes on the 360 strip when that frame has pixels, else on main.

    The strip's windshield band is the image. Repeater and rear sectors stay
    out of this fit. An empty stitch falls back to ``cam_main``.
    """
    from python.perception.stitch360 import forward_lane_view
    from python.sensors.cameras import frame_is_unrendered

    if stitch_bgr is not None and not frame_is_unrendered(stitch_bgr):
        return estimate_lanes(forward_lane_view(stitch_bgr), far_m=STITCH_LANE_FAR_M)
    return estimate_lanes(main_bgr)


def _estimate_lanes_impl(
    bgr: np.ndarray,
    far_m: float = _MAIN_FAR_M,
    roi_top: float = 0.40,
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
        lanes_bev.extend(chain_lane_segments(left, w, h, far_m))
        conf += 0.4
    if right:
        lanes_bev.extend(chain_lane_segments(right, w, h, far_m))
        conf += 0.4
    if left and right:
        # crude curvature from mean slope asymmetry
        ls = np.mean([s[-1] for s in left])
        rs = np.mean([s[-1] for s in right])
        curv = float(np.clip((ls + rs) * 0.02, -0.05, 0.05))
        conf = min(1.0, conf + 0.15)
    return LaneResult(conf=float(conf), lanes_bev=lanes_bev, curvature=curv)
