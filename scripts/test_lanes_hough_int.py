#!/usr/bin/env python3
"""Regression: OpenCV 5 HoughLinesP rows are (N, 4).

OpenCV 4 returns (N, 1, 4). Indexing that layout's ``lines[:, 0]`` on an
OpenCV 5 (N, 4) array yields scalars, and ``row[1]`` raises IndexError.
That used to be swallowed as lane_conf 0. It is not a numpy-int TypeError.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.perception import lanes as lanes_mod
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

    # A Hough layout IndexError must not collapse to a silent lane_conf 0.
    # Other frame failures stay a zero fit, but the handler has to say why.
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    orig = lanes_mod._estimate_lanes_impl
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    cap = _Capture(level=logging.WARNING)
    log = logging.getLogger("gvd.perception.lanes")
    prev_level = log.level
    log.addHandler(cap)
    log.setLevel(logging.WARNING)

    def _layout_break(_bgr: np.ndarray):
        raise IndexError("(N, 4) Hough row")

    lanes_mod._estimate_lanes_impl = _layout_break
    try:
        try:
            estimate_lanes(frame)
        except IndexError as exc:
            assert "(N, 4)" in str(exc)
        else:
            raise AssertionError("layout IndexError was swallowed as lane_conf 0")
        assert any("refusing lane_conf=0" in rec.getMessage() for rec in records)

        records.clear()

        def _bad_frame(_bgr: np.ndarray):
            raise ValueError("undecodable frame")

        lanes_mod._estimate_lanes_impl = _bad_frame
        failed = estimate_lanes(frame)
    finally:
        lanes_mod._estimate_lanes_impl = orig
        log.removeHandler(cap)
        log.setLevel(prev_level)
    assert failed.conf == 0.0 and failed.lanes_bev == []
    assert records, "estimate_lanes swallowed ValueError with no log"
    assert any("undecodable frame" in rec.getMessage() for rec in records)
    print("test_lanes_hough_int: OK")


def test_lanes_hough_int() -> None:
    main()


if __name__ == "__main__":
    main()
