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


def estimate_lanes(bgr: np.ndarray | None) -> LaneResult:
    """Fit ego-lane paint on ``cam_main``.

    White and yellow stripes are masked in HSV, so a hazy yellow line that
    grayscale Canny misses can still clear the engage gate. An empty or
    undecodable frame is ``lane_conf`` 0. A Hough row-layout ``IndexError``
    (OpenCV 5 ``(N, 4)`` indexed as OpenCV 4 ``(N, 1, 4)``) is logged and
    re-raised so it cannot look like an empty road. Other fit errors
    (``cv2.error``, ``ValueError``, ``TypeError``) log once and return 0.
    """
    if bgr is None or bgr.size == 0:
        return LaneResult(conf=0.0, lanes_bev=[], curvature=0.0)
    try:
        return _estimate_lanes_impl(bgr)
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
    high value, so a bright gray line still counts. A flat gray frame does not.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 185), (180, 55, 255))
    yellow = cv2.inRange(hsv, (8, 40, 80), (42, 255, 255))
    return cv2.bitwise_or(white, yellow)


def lane_roi_mask(height: int, width: int) -> np.ndarray:
    """Forward-cam trapezoid. Top sits above mid-frame so hood-cam lines that
    converge near the horizon are still inside the mask.
    """
    h, w = int(height), int(width)
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.array(
        [[
            (int(0.05 * w), h - 1),
            (int(0.36 * w), int(0.40 * h)),
            (int(0.64 * w), int(0.40 * h)),
            (int(0.95 * w), h - 1),
        ]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, poly, 255)
    return mask


def _estimate_lanes_impl(bgr: np.ndarray) -> LaneResult:
    bgr = _as_bgr(bgr)
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    paint = lane_paint_mask(bgr)
    # Boundary of a solid stripe. A low-gradient yellow line has almost no Canny edge.
    grad = cv2.morphologyEx(paint, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    crop = cv2.bitwise_and(cv2.bitwise_or(edges, grad), lane_roi_mask(h, w))
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
    def _to_ego_poly(segs: list) -> list[dict[str, float]]:
        pts = []
        for x1, y1, x2, y2, _ in segs[:8]:
            for x, y in ((x1, y1), (x2, y2)):
                # image → crude ego
                ex = ((x / w) - 0.5) * 6.0
                ey = max(2.0, 35.0 * (1.0 - y / h))
                pts.append({"x": float(ex), "y": float(ey), "z": 0.0})
        pts.sort(key=lambda p: p["y"])
        return pts

    if left:
        lanes_bev.append(_to_ego_poly(left))
        conf += 0.4
    if right:
        lanes_bev.append(_to_ego_poly(right))
        conf += 0.4
    if left and right:
        # crude curvature from mean slope asymmetry
        ls = np.mean([s[-1] for s in left])
        rs = np.mean([s[-1] for s in right])
        curv = float(np.clip((ls + rs) * 0.02, -0.05, 0.05))
        conf = min(1.0, conf + 0.15)
    return LaneResult(conf=float(conf), lanes_bev=lanes_bev, curvature=curv)
