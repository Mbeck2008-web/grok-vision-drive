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

from python.control.e2e import E2EIntent
from python.perception import lanes as lanes_mod
from python.perception.lanes import estimate_lanes, hough_segments
from python.runtime.shadow import ShadowConfig


class _BrakeStub:
    """Untrained e2e stand-in that would jab the brake if modular copied it."""

    backend = "stub"

    def forward(self, *args, **kwargs) -> E2EIntent:
        return E2EIntent(
            steer=0.4, accel=-1.0, throttle=0.0, brake=1.0, ok=True, backend="stub", reason="stub",
        )


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

    # Hazy solid yellow + white. Grayscale Canny scores this 0; the live path
    # is estimate_lanes (ModularPerception.tick on cam_main).
    haze = _yellow_white_road(
        asphalt=(168, 170, 172),
        yellow=(140, 190, 205),
        white=(198, 198, 200),
        width=2,
    )
    r5 = estimate_lanes(haze)
    assert r5.conf >= gate, r5.conf
    assert len(r5.lanes_bev) >= 2, r5.lanes_bev
    # Same function ModularPerception.tick calls on cam_main, engaged or not.
    from python.perception.detect import EmptyDetector
    from python.perception.pipeline import ModularPerception
    from python.runtime.shadow import shadow_tick

    perc = ModularPerception(allow_synthetic=False, detector_id="empty")
    perc.set_detector(EmptyDetector(), ["yolo_weights"])
    pout = perc.tick(haze)
    assert pout.lane_conf >= gate, pout.lane_conf
    assert "lanes" not in pout.missing
    assert pout.path_debug_preview is False
    held = shadow_tick(
        policy="modular",
        engaged=True,
        heartbeat_ok=True,
        path_debug_preview=bool(pout.path_debug_preview),
        allow_preview_drive=False,
        path_ego=pout.path_ego,
        planner=pout.planner,
        ego_speed_mps=5.0,
        lane_conf=float(pout.lane_conf),
        path_conf=float(pout.path_conf),
        seq=1,
        e2e_policy=_BrakeStub(),
        main_bgr=haze,
    )
    assert held.veto_reason != "low_lane_conf"
    assert held.should_disengage is False
    assert held.applied.reason == "ok"
    assert held.applied.brake != 1.0
    clear = _yellow_white_road(
        asphalt=(110, 108, 105),
        yellow=(40, 210, 235),
        white=(245, 245, 245),
        width=4,
    )
    r6 = estimate_lanes(clear)
    assert r6.conf >= gate, r6.conf
    assert len(r6.lanes_bev) >= 2, r6.lanes_bev

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
    src = (ROOT / "python" / "perception" / "lanes.py").read_text(encoding="utf-8")
    assert "except (cv2.error, ValueError, TypeError)" in src
    assert "except Exception" not in src
    print("test_lanes_hough_int: OK")


def _yellow_white_road(
    *,
    asphalt: tuple[int, int, int],
    yellow: tuple[int, int, int],
    white: tuple[int, int, int],
    width: int,
    vp_y: float = 0.46,
) -> np.ndarray:
    """640×480 hood view: solid yellow left, solid white right, low contrast ok."""
    cv2 = __import__("cv2")
    w, h = 640, 480
    img = np.full((h, w, 3), (175, 185, 195), dtype=np.uint8)
    vp = (int(w * 0.5), int(h * vp_y))
    cv2.fillPoly(
        img,
        [np.array([[int(w * 0.02), h - 1], vp, [int(w * 0.98), h - 1]], np.int32)],
        asphalt,
    )

    def _draw(xb: float, color: tuple[int, int, int]) -> None:
        a = (int(xb), h - 8)
        b = (int(w * 0.5 + (xb / w - 0.5) * 18), int(h * vp_y) + 4)
        cv2.line(img, a, b, color, width)

    _draw(w * 0.28, yellow)
    _draw(w * 0.72, white)
    cv2.rectangle(img, (0, int(h * 0.93)), (w, h), (35, 35, 38), -1)
    return img


def test_lanes_hough_int() -> None:
    main()


if __name__ == "__main__":
    main()
