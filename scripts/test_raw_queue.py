#!/usr/bin/env python3
"""Capture tick time stays flat when the disk writer is slow."""
from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from python.data.raw_frames import RawFrameWriter  # noqa: E402
from python.data.raw_queue import RawWriteQueue  # noqa: E402
from python.debug.lane_targets import tech_from_hit  # noqa: E402


class _Slow:
    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.n = 0
        self.entered: list[float] = []
        self.left: list[float] = []

    def write(self, frames: dict, **kwargs: object) -> None:
        self.entered.append(time.perf_counter())
        time.sleep(self.delay_s)
        self.n += 1
        self.left.append(time.perf_counter())


class _Hold:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.n = 0

    def write(self, frames: dict, **kwargs: object) -> None:
        self.started.set()
        self.release.wait(2.0)
        self.n += 1


def _payload(i: int) -> dict:
    return {
        "t": float(i),
        "driver": "gvd",
        "ego": {"speed_mps": 4.0, "yaw_rate": 0.01, "pos": [1.0, 2.0, 3.0], "dir": [0.0, 1.0, 0.0]},
        "wheel": {"steer": -0.2},
        "pedals": {"throttle": 0.15, "brake": 0.0},
        "tech": tech_from_hit(0.1, 0.0, 3.5, False),
    }


def check_tick_does_not_wait_for_disk() -> None:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :, 1] = 40
    slow = _Slow(0.04)
    queue = RawWriteQueue(slow, max_depth=8)
    recorded = []
    plain = []
    try:
        for i in range(6):
            t0 = time.perf_counter()
            assert queue.submit({"main": image, "wide": image}, **_payload(i))
            recorded.append(time.perf_counter() - t0)
            t1 = time.perf_counter()
            _ = np.ascontiguousarray(image)
            plain.append(time.perf_counter() - t1)
        stats = queue.stats()
        assert max(recorded) < 0.015, recorded
        assert max(recorded) < 0.04
        # The disk sleep is 40 ms. The capture tick stays with the no-write copy.
        assert max(recorded) < max(plain) + 0.012
        deadline = time.time() + 2.0
        while slow.n < 6 and time.time() < deadline:
            time.sleep(0.01)
        assert slow.n == 6
        assert stats["queue_peak"] >= 1 or queue.stats()["written"] == 6
        assert slow.left and slow.entered
        assert (slow.left[0] - slow.entered[0]) >= 0.03
    finally:
        queue.close()


def check_full_queue_drops_and_stays_fast() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    hold = _Hold()
    queue = RawWriteQueue(hold, max_depth=1)
    try:
        assert queue.submit({"main": image}, **_payload(0))
        assert hold.started.wait(1.0)
        dropped_at = None
        durs = []
        for i in range(1, 8):
            t0 = time.perf_counter()
            ok = queue.submit({"main": image}, **_payload(i))
            durs.append(time.perf_counter() - t0)
            if not ok and dropped_at is None:
                dropped_at = i
        assert dropped_at is not None
        assert max(durs) < 0.015, durs
        stats = queue.stats()
        assert stats["dropped_frames"] >= 1
        assert stats["queue_depth"] <= 1
    finally:
        hold.release.set()
        queue.close()


def check_queue_writes_controls_off_thread() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    with tempfile.TemporaryDirectory() as tmp:
        queue = RawWriteQueue(RawFrameWriter(Path(tmp), []), max_depth=2)
        try:
            t0 = time.perf_counter()
            assert queue.submit({"main": image}, **_payload(1))
            assert time.perf_counter() - t0 < 0.05
            deadline = time.time() + 2.0
            while queue.stats()["written"] < 1 and time.time() < deadline:
                time.sleep(0.01)
        finally:
            queue.close()
        line = (Path(tmp) / "state.jsonl").read_text(encoding="utf-8")
        assert '"driver":"gvd"' in line or '"driver": "gvd"' in line
        assert "steer" in line and "yaw_rate" in line
        gt = (Path(tmp) / "gt.jsonl").read_text(encoding="utf-8")
        assert "lane_gt" in gt


def main() -> None:
    check_tick_does_not_wait_for_disk()
    check_full_queue_drops_and_stays_fast()
    check_queue_writes_controls_off_thread()
    print("raw queue ok")


if __name__ == "__main__":
    main()
