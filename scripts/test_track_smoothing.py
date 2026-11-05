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

    keys = set(tracker.as_dicts()[0])
    assert keys == {"id", "class", "x", "y", "yaw", "speed_mps", "yaw_rate", "hits", "misses", "age", "conf"}, keys
    print("test_track_smoothing: OK")


if __name__ == "__main__":
    main()
