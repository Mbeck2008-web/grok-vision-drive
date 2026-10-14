#!/usr/bin/env python3
"""Stale lanes from the efc25e3 run.

Companions were attached at 1 s, so the same lane lines came back for
several ticks while the car yawed. A 10.7 m pair was still called a lane.
A right turn has to move lane_x left, and speed stays near 6 m/s while the
picture is old.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import steer_sample_point  # noqa: E402
from python.control.launch_guard import (  # noqa: E402
    STALE_SPEED_MPS,
    apply_stale_speed_cap,
)
from python.perception.pipeline import ModularPerception  # noqa: E402
from python.planning.path_predictor import (  # noqa: E402
    LANE_PAIR_W_MAX,
    LANE_PAIR_W_MIN,
    predict_path,
    reckon_ego_x,
)
from python.sensors.cameras import (  # noqa: E402
    COMPANION_OFFSCREEN_UPDATE_S,
    LANE_CAM_UPDATE_S,
    sensor_requested_update_s,
)


class _Cmd:
    def __init__(self) -> None:
        self.throttle = 0.8
        self.brake = 0.0
        self.steer = 0.2


def _line(x: float) -> list[dict[str, float]]:
    return [{"x": float(x), "y": float(y)} for y in range(2, 18, 2)]


def _road():
    import cv2
    import numpy as np

    img = np.full((96, 160, 3), (40, 42, 44), dtype=np.uint8)
    cv2.line(img, (30, 90), (70, 30), (40, 210, 235), 3)
    cv2.line(img, (130, 90), (90, 30), (245, 245, 245), 3)
    return img


def main() -> None:
    assert LANE_CAM_UPDATE_S <= 0.1
    assert sensor_requested_update_s(-1.0, "pillarL") == LANE_CAM_UPDATE_S
    assert sensor_requested_update_s(-1.0, "pillarR") == LANE_CAM_UPDATE_S
    assert sensor_requested_update_s(-1.0, "wide") == LANE_CAM_UPDATE_S
    assert sensor_requested_update_s(-1.0, "narrow") == LANE_CAM_UPDATE_S
    assert sensor_requested_update_s(-1.0, "rear") == COMPANION_OFFSCREEN_UPDATE_S
    assert sensor_requested_update_s(-1.0, "repeatL") == COMPANION_OFFSCREEN_UPDATE_S
    assert sensor_requested_update_s(0.067, "main") == 0.067
    assert 1.0 / LANE_CAM_UPDATE_S >= 10.0

    wide = predict_path(
        lanes_bev=[_line(-4.11), _line(6.64)],
        lane_conf=0.8,
        ego_speed_mps=8.0,
    )
    assert "pair" not in wide.lane_source, wide.lane_source
    assert wide.drivable is False, wide.lane_source
    good = predict_path(
        lanes_bev=[_line(-1.2), _line(2.3)],
        lane_conf=0.8,
        ego_speed_mps=8.0,
    )
    width = 2.3 - (-1.2)
    assert LANE_PAIR_W_MIN <= width <= LANE_PAIR_W_MAX
    assert good.lane_source.startswith("pair"), good.lane_source

    # 0.5 s of a right turn (yaw_rate < 0) at the 8 m point.
    moved = reckon_ego_x(0.64, 8.0, yaw_rate=-0.2, dt=0.5)
    assert moved < 0.0, moved

    img = _road()
    perc = ModularPerception(allow_synthetic=False, detector_id="empty")
    first = perc.tick(img, ego_speed_mps=5.0, yaw_rate=0.0, dt_s=0.16)
    assert first.frame_age_s == 0.0
    time.sleep(0.3)
    second = perc.tick(img, ego_speed_mps=8.0, yaw_rate=-0.25, dt_s=0.3)
    assert second.frame_age_s > 0.25, second.frame_age_s
    assert "stale" in second.lane_source, second.lane_source
    a = steer_sample_point(first.path_ego)
    b = steer_sample_point(second.path_ego)
    if a is not None and b is not None:
        assert b["x"] < a["x"], (a, b, second.lane_source)

    cmd = _Cmd()
    assert apply_stale_speed_cap(
        cmd, speed_mps=10.8, lane_age_s=0.5, lane_src="pair/stale",
    )
    assert cmd.throttle == 0.0 and cmd.brake >= 0.3
    held = _Cmd()
    assert apply_stale_speed_cap(
        held, speed_mps=4.0, lane_age_s=0.0, lane_src="pair/hold",
    )
    assert held.throttle <= 0.25
    fresh = _Cmd()
    assert apply_stale_speed_cap(
        fresh, speed_mps=10.0, lane_age_s=0.0, lane_src="pair",
    ) is False
    assert fresh.throttle == 0.8
    assert STALE_SPEED_MPS == 6.0
    print("test_lane_rate: OK")


if __name__ == "__main__":
    main()
