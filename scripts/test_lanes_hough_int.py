#!/usr/bin/env python3
"""Regression: Hough endpoints must cast to Python int (live gate TypeError)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.perception.lanes import estimate_lanes


def main() -> None:
    # synthetic frame with edges
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2 = __import__("cv2")
    cv2.line(img, (80, 470), (300, 280), (200, 200, 200), 3)
    cv2.line(img, (560, 470), (340, 280), (200, 200, 200), 3)
    r = estimate_lanes(img)
    assert r is not None
    # must not raise; conf may be 0 on blankish frames but cast path exercised when lines found
    r2 = estimate_lanes(None)
    assert r2.conf == 0.0
    print("test_lanes_hough_int: OK")


if __name__ == "__main__":
    main()
