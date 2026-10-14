#!/usr/bin/env python3
"""The ecf5d35 frame must see a boundary on the left of the car.

Opening ticks, before this frame was saved: ground truth at 8 m was left
x=-1.21, right x=+2.54, center +0.66, while det_left_x sat at +2.8..+3.0
and det_right_x at +8..+16. The yellow stripe is on the right of the main
camera and the white dashes are on the left. The old fit clamped the yellow
stripe to the right edge of the sector and called that the left line.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.debug.gt_logger import detected_lane_xs  # noqa: E402
from python.perception.lanes import (  # noqa: E402
    STITCH_LANE_FAR_M,
    estimate_lanes,
    lane_paint_mask,
    lanes_from_view,
)
from python.perception.stitch360 import forward_lane_view, windshield_x_spans  # noqa: E402
from python.planning.path_predictor import LANE_BOUNDARY_WIDTHS, LANE_W_DEFAULT  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "gt_run3_frame.jpg"
# Opening-tick ground truth, before the car had yawed off the lane.
_GT_LEFT = -1.21


def main() -> None:
    frame = cv2.imread(str(FIXTURE))
    assert frame is not None, FIXTURE
    band = forward_lane_view(frame)
    spans = windshield_x_spans(band)
    assert spans is not None, band.shape
    # Hazy west-coast yellow (S about 33) stays paint. Sky (S about 25) does not.
    haze = np_bgr(214, 236, 246)
    sky = np_bgr(190, 200, 210)
    assert int(lane_paint_mask(haze)[0, 0]) == 255
    assert int(lane_paint_mask(sky)[0, 0]) == 0

    fit = estimate_lanes(band, far_m=STITCH_LANE_FAR_M, x_spans=spans)
    assert fit.lanes_bev, fit
    for poly in fit.lanes_bev:
        for pt in poly:
            # The old clamp published exactly ±3. Interior pixels stay under 2.9.
            assert abs(float(pt["x"])) < 2.9, pt
    left, right, _center = detected_lane_xs(fit.lanes_bev, 8.0)
    assert left is not None and left < 0.0, (left, right, _summarize(fit.lanes_bev))
    # Closer to the opening-tick left line than to the old +3 m clamp.
    assert abs(left - _GT_LEFT) < abs(3.0 - left), (left, right)
    gate = LANE_BOUNDARY_WIDTHS * LANE_W_DEFAULT
    if right is not None:
        assert 0.0 < right <= gate, (left, right, gate)
    # Side cameras still emit |x| of tens of metres. The logged lane must not.
    full = lanes_from_view(None, frame)
    full_left, full_right, _full_center = detected_lane_xs(full.lanes_bev, 8.0)
    assert full_left is not None and full_left < 0.0, (full_left, full_right)
    assert full_right is None or abs(full_right) <= gate, (full_left, full_right)
    print(
        f"test_gt_run3_left_line: OK left={left:.2f} right={right} "
        f"full={full_left:.2f},{full_right}"
    )


def np_bgr(b: int, g: int, r: int):
    import numpy as np

    return np.array([[[b, g, r]]], dtype=np.uint8)


def _summarize(lanes) -> list[tuple[float, float]]:
    out = []
    for poly in lanes:
        xs = [float(p["x"]) for p in poly]
        out.append((min(xs), max(xs)))
    return out


if __name__ == "__main__":
    main()
