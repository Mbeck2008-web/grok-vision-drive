#!/usr/bin/env python3
"""Regression: Hough endpoints must cast to Python int (live gate TypeError)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.perception.lanes import estimate_lanes, hough_segments
from python.runtime.shadow import ShadowConfig


def main() -> None:
    # OpenCV 4 is (N, 1, 4); OpenCV 5 is (N, 4). Both must yield the same endpoints.
    cv5 = np.array([[80, 470, 300, 280], [560, 470, 340, 280]], dtype=np.int32)
    cv4 = cv5.reshape(-1, 1, 4)
    assert hough_segments(cv5) == [(80, 470, 300, 280), (560, 470, 340, 280)]
    assert hough_segments(cv4) == hough_segments(cv5)
    assert hough_segments(None) == []
    assert hough_segments(np.zeros((0, 4), dtype=np.int32)) == []

    # synthetic frame with edges — visible paint must clear the engage lane gate
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2 = __import__("cv2")
    cv2.line(img, (80, 470), (300, 280), (200, 200, 200), 3)
    cv2.line(img, (560, 470), (340, 280), (200, 200, 200), 3)
    r = estimate_lanes(img)
    assert r is not None
    gate = ShadowConfig().lane_conf_min
    assert r.conf >= gate, r.conf
    assert len(r.lanes_bev) >= 2, r.lanes_bev
    # Empty perception stays empty. Do not invent a lane_conf.
    r2 = estimate_lanes(None)
    assert r2.conf == 0.0 and r2.lanes_bev == []
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    r3 = estimate_lanes(blank)
    assert r3.conf == 0.0 and r3.lanes_bev == [], r3
    flat = np.full((360, 640, 3), 140, dtype=np.uint8)
    r4 = estimate_lanes(flat)
    assert r4.conf < gate and r4.lanes_bev == [], r4
    print("test_lanes_hough_int: OK")


def test_lanes_hough_int() -> None:
    main()


if __name__ == "__main__":
    main()
