#!/usr/bin/env python3
"""A4 (#151): the tracker filters position and reports signed speed.

Offline repro from the issue: a 640x480 frame, box bottom at row 300,
box height 60 +/- 4 px, fed to ``IoUTracker`` at 15 Hz. The checks are
direction only (spec Q4: no unmeasured thresholds).

    py -3 scripts/test_track_smoothing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.perception.detect import Detection, project_box_to_ego  # noqa: E402
from python.perception.track import IoUTracker  # noqa: E402

HZ = 15.0
W, H = 640, 480



def _det(bh: float) -> Detection:
    box = (300.0, 300.0 - bh, 360.0, 300.0)
    x, y, yaw = project_box_to_ego(box, W, H)
    return Detection(cls="vehicle", conf=0.9, xyxy=box, x=x, y=y, yaw=yaw)


def _run(heights: list[float], tracker: IoUTracker) -> tuple[list[float], list[float], list[float]]:
    raw, ys, speeds = [], [], []
    for k, bh in enumerate(heights):
        d = _det(bh)
        tracks = tracker.update([d], k / HZ)
        assert len(tracks) == 1, tracks
        raw.append(d.y)
        ys.append(tracks[0].y)
        speeds.append(tracks[0].speed_mps)
    return raw, ys, speeds


def _noisy() -> list[float]:
    rng = np.random.default_rng(0)
    return [60.0 + float(v) for v in rng.uniform(-4.0, 4.0, int(4 * HZ))]


def _reference_speed_2s() -> float:
    """The pre-#151 update rule on the same input: EMA of abs(dy/dt), position copied."""
    speed, y = 0.0, None
    out = []
    for bh in _noisy():
        d = _det(bh)
        if y is not None:
            pred = y + speed * (1 / HZ) * 0.15
            speed = max(0.0, min(40.0, 0.8 * speed + 0.2 * abs((d.y - pred) * HZ)))
        y = d.y
        out.append(speed)
    return out[int(2 * HZ)]


def _closing_track(speed: float) -> dict:
    return {"id": 1, "class": "vehicle", "x": 0.0, "y": 20.0, "yaw": 1.5708, "speed_mps": speed,
            "yaw_rate": 0.0, "hits": 5, "misses": 0, "age": 5, "conf": 0.9}


def check_consumers_keep_magnitude() -> None:
    """Signed speed must not change any consumer: -v behaves exactly like the old +v."""
    from python.perception.cipv import select_cipv
    from python.planning.path_predictor import _predict_occupants
    from python.run_vision import _forecast_agents
    from python.train.scene_live import live_strip_fields
    from python.viz.forecast import predict_modes
    from python.viz.nerd import _lines
    from python.viz.stage import Cam, _draw_tracks

    neg, pos = _closing_track(-5.0), _closing_track(5.0)
    fan_neg, fan_pos = _forecast_agents([neg]), _forecast_agents([pos])
    assert fan_neg, "closing track got no forecast fan (parked-car check read signed speed)"
    assert fan_neg == fan_pos, (fan_neg, fan_pos)
    print("run_vision forecast: closing track keeps its fan OK")
    assert predict_modes(neg) == predict_modes(pos)
    print("forecast.predict_modes: magnitude OK")
    a = select_cipv([neg], ego_speed_mps=10.0)
    b = select_cipv([pos], ego_speed_mps=10.0)
    # The old assertion required -v and +v to produce the same TTC/AEB. That
    # locked in abs(speed_mps), so a lead pulling away looked like one we were
    # catching, and a matched-speed car looked stopped (TTC = gap / ego).
    assert a.ttc_lead != b.ttc_lead, (a, b)
    assert b.ttc_lead is None and b.aeb == "off", b
    print("cipv: opening is not closing OK")
    # Occupants keep signed relative speed: closing coasts the gap shut,
    # opening coasts it open. abs() here was the planner's phantom-AEB bug.
    occ_n = _predict_occupants([dict(neg, unseen_s=1.0)])
    occ_p = _predict_occupants([dict(pos, unseen_s=1.0)])
    assert occ_n and occ_p
    assert occ_n[0]["speed_mps"] == -5.0 and occ_p[0]["speed_mps"] == 5.0, (occ_n, occ_p)
    assert occ_n[0]["y"] < neg["y"] < occ_p[0]["y"], (occ_n[0]["y"], neg["y"], occ_p[0]["y"])
    print("path_predictor occupants: signed coast OK")
    sn = [line for line in _lines({"tracks": [neg]}) if "miss@" in line]
    sp = [line for line in _lines({"tracks": [pos]}) if "miss@" in line]
    assert sn == sp, (sn, sp)
    print("nerd lead line: magnitude OK")
    imgs = []
    for tr in (neg, pos):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        _draw_tracks(img, [tr], Cam(), path=[], path_width=2.0, state={}, cipv_id=None, show_vel=True)
        imgs.append(img)
    assert np.array_equal(imgs[0], imgs[1])
    print("stage velocity arrow: magnitude OK")
    fn, fp = live_strip_fields(tracks=[neg]), live_strip_fields(tracks=[pos])
    assert fn == fp, (fn, fp)
    assert neg["speed_mps"] == -5.0  # the caller's dict is not changed
    print("scene_live strip labels: magnitude OK")


def main() -> None:
    raw, ys, speeds = _run(_noisy(), IoUTracker())
    skip = int(HZ)  # let the filter settle for 1 s
    raw_std, trk_std = float(np.std(raw[skip:])), float(np.std(ys[skip:]))
    print(f"stationary: raw y std={raw_std:.4f} track y std={trk_std:.4f}")
    assert trk_std < raw_std, (trk_std, raw_std)

    ref = _reference_speed_2s()
    now = abs(speeds[int(2 * HZ)])
    print(f"stationary: |speed| at 2 s={now:.4f} (old rule {ref:.4f})")
    assert now < ref, (now, ref)

    # Closing: range drops at a steady rate; box grows as it approaches.
    tracker = IoUTracker()
    heights = [40.0 + 1.0 * k for k in range(int(3 * HZ))]
    _raw, _ys, speeds = _run(heights, tracker)
    print(f"closing: speed at end={speeds[-1]:.4f}")
    assert speeds[-1] < 0.0, speeds[-1]

    check_consumers_keep_magnitude()
    keys = set(tracker.as_dicts()[0])
    assert keys == {"id", "class", "x", "y", "yaw", "speed_mps", "yaw_rate", "hits", "misses", "age", "conf"}, keys
    print("test_track_smoothing: OK")


if __name__ == "__main__":
    main()
