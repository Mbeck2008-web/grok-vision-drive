"""Disk writes off the capture thread.

The queue is bounded. A full queue drops the tick and counts it. The
capture call returns after a copy and a ``put_nowait``, so a slow disk
does not stretch the tick.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

import numpy as np

_STOP = object()


class RawWriteQueue:
    def __init__(self, writer: Any, max_depth: int = 8) -> None:
        self.writer = writer
        self.max_depth = max(1, int(max_depth))
        self._q: queue.Queue = queue.Queue(self.max_depth)
        self.dropped = 0
        self.enqueued = 0
        self.written = 0
        self._peak = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="gvd-raw-writer", daemon=True)
        self._thread.start()

    def submit(self, frames: Any, **kwargs: Any) -> bool:
        copied = {}
        for cam_id, image in dict(frames or {}).items():
            if isinstance(image, np.ndarray) and image.size and image.ndim >= 2:
                copied[str(cam_id)] = np.ascontiguousarray(image)
        try:
            self._q.put_nowait((copied, kwargs))
        except queue.Full:
            with self._lock:
                self.dropped += 1
            return False
        with self._lock:
            self.enqueued += 1
            self._peak = max(self._peak, self._q.qsize())
        return True

    def stats(self) -> dict[str, int]:
        with self._lock:
            dropped = int(self.dropped)
            enqueued = int(self.enqueued)
            written = int(self.written)
            peak = int(self._peak)
        return {
            "queue_depth": int(self._q.qsize()),
            "queue_max": int(self.max_depth),
            "queue_peak": peak,
            "dropped_frames": dropped,
            "enqueued": enqueued,
            "written": written,
        }

    def close(self, timeout_s: float = 30.0) -> None:
        try:
            self._q.put(_STOP, timeout=max(0.1, float(timeout_s)))
        except queue.Full:
            return
        self._thread.join(timeout=max(0.1, float(timeout_s)))

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is _STOP:
                    return
                frames, kwargs = item
                self.writer.write(frames, **kwargs)
                with self._lock:
                    self.written += 1
            finally:
                self._q.task_done()
