"""OpenCV lanes on cam_main only (Udacity/Aly-style toy)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LaneResult:
    conf: float
    lanes_bev: list[list[dict[str, float]]]  # polylines in ego frame
    curvature: float  # 1/m approx


def estimate_lanes(bgr: np.ndarray | None) -> LaneResult:
    if bgr is None or bgr.size == 0:
        return LaneResult(conf=0.0, lanes_bev=[], curvature=0.0)
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    # ROI: lower half trapezoid
    mask = np.zeros_like(edges)
    poly = np.array([[(int(0.1 * w), h), (int(0.45 * w), int(0.55 * h)), (int(0.55 * w), int(0.55 * h)), (int(0.9 * w), h)]], dtype=np.int32)
    cv2.fillPoly(mask, poly, 255)
    crop = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(crop, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=80)
    left, right = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
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
