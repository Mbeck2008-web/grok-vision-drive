#!/usr/bin/env python3
"""Capture tick time stays flat when the disk writer is slow."""
from __future__ import annotations

import json
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



def check_failed_write_is_not_counted_written() -> None:
    """A write error must not increment written or kill later ticks."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    class _Boom:
        def __init__(self) -> None:
            self.calls = 0
        def write_encoded(self, payload, **kwargs: object) -> None:
            self.calls += 1
            if self.calls == 1:
                raise OSError("disk full")
    boom = _Boom()
    queue = RawWriteQueue(boom, max_depth=8)
    try:
        assert queue.submit({"main": image}, **_payload(0))
        assert queue.submit({"main": image}, **_payload(1))
        deadline = time.time() + 2.0
        while time.time() < deadline:
            stats = queue.stats()
            if stats["written"] + int(stats.get("failed") or 0) >= 2 and stats.get("error"):
                break
            time.sleep(0.01)
        stats = queue.stats()
        assert stats["written"] == 1, stats
        assert int(stats.get("failed") or 0) == 1, stats
        assert stats.get("error") and "disk full" in str(stats["error"])
        assert boom.calls == 2
    finally:
        queue.close()
    print("failed write is not counted as written")


def check_empty_submit_does_not_create_a_row() -> None:
    """Empty camera dicts must not become gt rows with no PNGs."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        queue = RawWriteQueue(RawFrameWriter(root, []), max_depth=4)
        try:
            assert queue.submit({}, **_payload(0)) is True
            assert queue.submit({"main": np.zeros((4, 4, 3), dtype=np.uint8)}, **_payload(1))
            assert queue.wait_empty(2.0)
        finally:
            queue.close()
        state = list((root / "state.jsonl").read_text(encoding="utf-8").splitlines())
        assert len(state) == 1
        assert json.loads(state[0])["i"] == 0
        assert list(root.joinpath("raw").rglob("*.png"))
    print("empty submit creates no hollow row")


def check_empty_submit_does_not_enqueue() -> None:
    """Empty ticks must not occupy a queue slot or count as enqueued.

    ``write_encoded([])`` and a failed empty encode already refuse a
    hollow row, so a test that only checks state.jsonl cannot kill a
    mutant that drops the early return in ``submit()`` and still
    enqueues the empty tick (later marked failed, no PNG).
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        queue = RawWriteQueue(RawFrameWriter(root, []), max_depth=4)
        try:
            assert queue.submit({}, **_payload(0)) is True
            assert queue.submit(None, **_payload(1)) is True
            assert queue.submit({"main": None}, **_payload(2)) is True
            assert queue.submit({"main": np.zeros((0, 0, 3), dtype=np.uint8)}, **_payload(3)) is True
            stats = queue.stats()
            assert int(stats["enqueued"]) == 0, stats
            assert int(stats["queue_depth"]) == 0, stats
            assert int(stats["written"]) == 0, stats
            assert int(stats["failed"]) == 0, stats
            assert int(stats["dropped_frames"]) == 0, stats
            assert queue.wait_empty(1.0)
        finally:
            queue.close()
        state = root / "state.jsonl"
        assert (not state.exists()) or state.read_text(encoding="utf-8") == ""
        assert not list(root.joinpath("raw").rglob("*.png"))
    print("empty submit does not enqueue")



def check_queue_close_closes_writer() -> None:
    """RawWriteQueue.close must close the writer, which closes the gt log."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[0, 0] = (1, 2, 3)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = RawFrameWriter(root, [{"id": "main"}])
        queue = RawWriteQueue(writer, max_depth=4)
        try:
            assert queue.submit({"main": image}, **_payload(0)) is True
            assert queue.wait_empty(2.0)
        finally:
            queue.close()
        assert writer._state_log._fd is None
        assert writer._gt_log._fd is None
    print("queue close closes the writer")


def main() -> None:
    check_tick_does_not_wait_for_disk()
    check_full_queue_drops_and_stays_fast()
    check_queue_writes_controls_off_thread()
    check_failed_write_is_not_counted_written()
    check_empty_submit_does_not_create_a_row()
    check_empty_submit_does_not_enqueue()
    check_queue_close_closes_writer()
    print("raw queue ok")


if __name__ == "__main__":
    main()
