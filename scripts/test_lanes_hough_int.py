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
from python.perception.lanes import chain_lane_segments, estimate_lanes, hough_segments, lane_paint_mask
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
    # Dark sky, yellow left and white right, still clears the engage gate.
    night = _yellow_white_road(
        sky=(24, 26, 32),
        asphalt=(48, 50, 54),
        yellow=(40, 210, 235),
        white=(236, 236, 238),
        width=3,
    )
    r_night = estimate_lanes(night)
    assert r_night.conf >= gate, r_night.conf
    assert len(r_night.lanes_bev) >= 2, r_night.lanes_bev
    # Unmarked hazy road: stripes removed. Sky (HSV 15,26,195) is not white
    # paint, and the road/sky wedge must stay under lane_conf_min.
    bare = _yellow_white_road(
        asphalt=(168, 170, 172),
        yellow=(140, 190, 205),
        white=(198, 198, 200),
        width=2,
        stripes=False,
    )
    assert int(lane_paint_mask(bare)[8, 8]) == 0
    r_bare = estimate_lanes(bare)
    assert r_bare.conf < 0.25, r_bare.conf
    assert r_bare.conf < gate
    assert r_bare.lanes_bev == []
    # Same wedge at higher contrast. Edges that are not paint stay under the gate.
    sharp = _yellow_white_road(
        asphalt=(40, 42, 48),
        yellow=(40, 210, 235),
        white=(245, 245, 245),
        width=2,
        stripes=False,
    )
    r_sharp = estimate_lanes(sharp)
    assert r_sharp.conf < 0.25, r_sharp.conf
    assert r_sharp.lanes_bev == []

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

    def _layout_break(_bgr: np.ndarray, *args, **kwargs):
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

        def _bad_frame(_bgr: np.ndarray, *args, **kwargs):
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
    _check_gap_is_not_one_polyline()
    _check_curve_stays_one_line()
    _check_stitch_view_and_past_path()
    print("test_lanes_hough_int: OK")


def _check_stitch_view_and_past_path() -> None:
    """A stitch frame is the lane image. Drawn lines pass the blue path."""
    cv2 = __import__("cv2")
    from python.perception.lanes import STITCH_LANE_FAR_M, lanes_from_view
    from python.perception.stitch360 import stitch_frames
    from python.viz.stage import Cam, VizUI, drawn_lane_records, render_stage

    road = _yellow_white_road(
        asphalt=(40, 40, 42),
        yellow=(0, 220, 220),
        white=(230, 230, 230),
        width=6,
    )
    blank = np.zeros_like(road)
    flat = np.full_like(road, 140)
    gate = ShadowConfig().lane_conf_min
    from_stitch = lanes_from_view(blank, road)
    assert from_stitch.conf >= gate and from_stitch.lanes_bev, from_stitch
    assert lanes_from_view(road, flat).conf < gate
    assert lanes_from_view(road, None).conf >= gate
    # An all-zero stitch is not a frame, so the main camera still fits.
    assert lanes_from_view(road, blank).conf >= gate
    via_canvas = lanes_from_view(blank, stitch_frames({"main": road}).bgr)
    assert via_canvas.lanes_bev, via_canvas
    horizon_m = STITCH_LANE_FAR_M * (1.0 - 0.40)
    assert horizon_m > 36.0, horizon_m

    short = [{"x": -1.7, "y": float(y)} for y in (4.0, 10.0, 18.0)]
    path = [{"x": 0.0, "y": float(y)} for y in range(0, 31, 2)]
    state = {
        "engaged": False,
        "loop_hz": 12.0,
        "policy": "modular",
        "path_ego": path,
        "path_width": 3.5,
        "path_debug_preview": False,
        "viz_smoke": False,
        "lanes_ext": [{"points": list(short), "kind": "detected", "index": -1, "side": "left"}],
        "road_edges": [],
        "tracks": [],
        "signs": [],
        "missing_state_keys": ["live cameras"],
    }
    drawn = drawn_lane_records(state)
    far = max(float(p["y"]) for p in drawn[0]["points"])
    assert far > 30.0, far
    assert far < 50.0, far
    held = drawn_lane_records({**state, "path_ego": []})
    assert max(float(p["y"]) for p in held[0]["points"]) == 18.0

    ui = VizUI()
    ui.layers = {0}
    ui.show_nerd = False
    ui.debug.viz_forecast = False
    ui.debug.viz_signs = False
    on = render_stage(state, ui=ui)
    off = render_stage({**state, "lanes_ext": []}, ui=ui)
    delta = cv2.absdiff(on, off)
    cam = Cam()

    def _hit(y: float) -> int:
        px, py = cam.project(-1.7, y, 0.02)
        if not (4 <= px < on.shape[1] - 4 and 24 < py < on.shape[0] - 4):
            return -1
        patch = delta[py - 4:py + 5, px - 4:px + 5]
        return int(np.count_nonzero(patch.sum(axis=2) > 8))

    assert _hit(32.0) > 0, "lane line runs past the blue path"
    assert _hit(80.0) == 0, "lane line does not run out to the cabin span"


def _px(ex: float, ey: float, w: int = 640, h: int = 480) -> tuple[float, float]:
    return ((ex / 6.0 + 0.5) * w, h * (1.0 - ey / 35.0))


def _seg(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float, float, float, float]:
    x1, y1 = _px(*a)
    x2, y2 = _px(*b)
    return (x1, y1, x2, y2, -1.0)


def _spans_jump(poly: list[dict]) -> bool:
    xs = [float(p["x"]) for p in poly]
    return min(xs) < -1.0 and max(xs) > 0.5


def _check_gap_is_not_one_polyline() -> None:
    """Pieces at about x=-1.5,y=10 and x=+1.1,y=12 are not one polyline."""
    segs = [
        _seg((-1.55, 6.0), (-1.45, 10.0)),
        _seg((1.05, 12.0), (1.15, 14.0)),
    ]
    polys = chain_lane_segments(segs, 640, 480)
    assert len(polys) == 2, polys
    assert not any(_spans_jump(p) for p in polys), polys
    cv2 = __import__("cv2")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for seg in segs:
        cv2.line(
            img,
            (int(round(seg[0])), int(round(seg[1]))),
            (int(round(seg[2])), int(round(seg[3]))),
            (220, 220, 220),
            3,
        )
    # The second piece is short in ego y. Lengthen it so Hough can see it.
    far = _seg((1.05, 12.0), (1.15, 18.0))
    cv2.line(
        img,
        (int(round(far[0])), int(round(far[1]))),
        (int(round(far[2])), int(round(far[3]))),
        (220, 220, 220),
        3,
    )
    fit = estimate_lanes(img)
    assert fit.lanes_bev, fit
    assert not any(_spans_jump(p) for p in fit.lanes_bev), fit.lanes_bev


def _check_curve_stays_one_line() -> None:
    """A curve whose pieces meet stays one polyline, in forward order."""
    knots = [(-1.8, 4.0), (-1.5, 10.0), (-0.9, 18.0), (-0.2, 28.0)]
    segs = [_seg(a, b) for a, b in zip(knots, knots[1:])]
    polys = chain_lane_segments(segs, 640, 480)
    assert len(polys) == 1, polys
    poly = polys[0]
    ys = [float(p["y"]) for p in poly]
    assert ys == sorted(ys), poly
    assert poly[0]["y"] < 5.0 and poly[-1]["y"] > 25.0, poly
    assert poly[-1]["x"] > poly[0]["x"], poly
    cv2 = __import__("cv2")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    pts = np.array([(int(round(x)), int(round(y))) for x, y in (_px(*k) for k in knots)], np.int32)
    cv2.polylines(img, [pts], False, (220, 220, 220), 3)
    fit = estimate_lanes(img)
    assert fit.lanes_bev, fit
    spanned = [
        p for p in fit.lanes_bev
        if min(float(q["y"]) for q in p) < 6.0 and max(float(q["y"]) for q in p) > 18.0
    ]
    assert spanned, fit.lanes_bev
    for poly in spanned:
        got = [float(p["y"]) for p in poly]
        assert got == sorted(got), poly
        assert not _spans_jump(poly), poly


def _yellow_white_road(
    *,
    asphalt: tuple[int, int, int],
    yellow: tuple[int, int, int],
    white: tuple[int, int, int],
    width: int,
    vp_y: float = 0.46,
    sky: tuple[int, int, int] = (175, 185, 195),
    stripes: bool = True,
) -> np.ndarray:
    """640×480 hood view. Default sky is the hazy BGR that must not read as white paint."""
    cv2 = __import__("cv2")
    w, h = 640, 480
    img = np.full((h, w, 3), sky, dtype=np.uint8)
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

    if stripes:
        _draw(w * 0.28, yellow)
        _draw(w * 0.72, white)
    cv2.rectangle(img, (0, int(h * 0.93)), (w, h), (35, 35, 38), -1)
    return img


def test_lanes_hough_int() -> None:
    main()


if __name__ == "__main__":
    main()
