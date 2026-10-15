"""Disk writes off the capture thread.

Several threads encode lossless PNG. One commit keeps state lines in
capture order. The queue is bounded by item count and by the bytes of
the arrays still waiting, so a slow disk does not grow without a limit.
A full queue drops the tick, counts it, and the caller logs it.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

import numpy as np

from python.data.raw_frames import encode_cameras

_STOP = object()
DEFAULT_QUEUE_BYTES = 512 * 1024 * 1024
DEFAULT_WORKERS = 4


class RawWriteQueue:
    def __init__(
        self,
        writer: Any,
        max_depth: int = 64,
        *,
        max_bytes: int = DEFAULT_QUEUE_BYTES,
        workers: int = DEFAULT_WORKERS,
    ) -> None:
        self.writer = writer
        self.max_depth = max(1, int(max_depth))
        self.max_bytes = max(1, int(max_bytes))
        self._q: queue.Queue = queue.Queue()
        self.dropped = 0
        self.enqueued = 0
        self.written = 0
        self._peak = 0
        self._queued = 0
        self._encoding = 0
        self._ready_n = 0
        self._committing = 0
        self._bytes = 0
        self._encoded = callable(getattr(writer, "write_encoded", None))
        self._seq = 0
        self._next = 1
        self._ready: dict[int, tuple[Any, dict[str, Any], int]] = {}
        self._stops = 0
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        nworkers = max(1, int(workers))
        self._nworkers = nworkers
        self._threads = [
            threading.Thread(target=self._encode, name=f"gvd-raw-encode-{i}", daemon=True)
            for i in range(nworkers)
        ]
        self._commit = threading.Thread(target=self._commit_loop, name="gvd-raw-commit", daemon=True)
        for thread in self._threads:
            thread.start()
        self._commit.start()

    def _inflight(self) -> int:
        return int(self._queued + self._encoding + self._ready_n + self._committing)

    def submit(self, frames: Any, **kwargs: Any) -> bool:
        copied = {}
        nbytes = 0
        for cam_id, image in dict(frames or {}).items():
            if isinstance(image, np.ndarray) and image.size and image.ndim >= 2:
                arr = np.ascontiguousarray(image)
                copied[str(cam_id)] = arr
                nbytes += int(arr.nbytes)
        with self._lock:
            if self._inflight() >= self.max_depth or self._bytes + nbytes > self.max_bytes:
                self.dropped += 1
                return False
            self._seq += 1
            seq = self._seq
            self._queued += 1
            self._bytes += nbytes
            self.enqueued += 1
            self._peak = max(self._peak, self._inflight())
        self._q.put((seq, copied, kwargs, nbytes))
        return True

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "queue_depth": self._inflight(),
                "queue_max": int(self.max_depth),
                "queue_bytes": int(self._bytes),
                "queue_max_bytes": int(self.max_bytes),
                "queue_peak": int(self._peak),
                "dropped_frames": int(self.dropped),
                "enqueued": int(self.enqueued),
                "written": int(self.written),
                "bytes_written": int(getattr(self.writer, "stored_bytes", 0) or 0),
            }

    def close(self, timeout_s: float = 120.0) -> None:
        for _ in self._threads:
            self._q.put(_STOP)
        deadline = timeout_s
        for thread in self._threads:
            thread.join(timeout=max(0.1, float(deadline)))
        self._commit.join(timeout=max(0.1, float(deadline)))

    def _encode(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is _STOP:
                    with self._cv:
                        self._stops += 1
                        self._cv.notify_all()
                    return
                seq, frames, kwargs, nbytes = item
                with self._lock:
                    self._queued -= 1
                    self._encoding += 1
                try:
                    payload: Any = encode_cameras(frames) if self._encoded else frames
                except Exception as exc:
                    print(f"collect encode failed: {type(exc).__name__}: {exc}", flush=True)
                    payload = [] if self._encoded else {}
                with self._cv:
                    self._encoding -= 1
                    self._ready[int(seq)] = (payload, kwargs, int(nbytes))
                    self._ready_n += 1
                    self._cv.notify_all()
            finally:
                self._q.task_done()

    def _commit_loop(self) -> None:
        while True:
            with self._cv:
                while self._next not in self._ready and self._stops < self._nworkers:
                    self._cv.wait()
                if self._next not in self._ready:
                    return
                payload, kwargs, nbytes = self._ready.pop(self._next)
                self._ready_n -= 1
                self._committing += 1
                self._next += 1
            try:
                if self._encoded:
                    self.writer.write_encoded(payload, **kwargs)
                else:
                    self.writer.write(payload, **kwargs)
            finally:
                with self._lock:
                    self._committing -= 1
                    self._bytes = max(0, self._bytes - nbytes)
                    self.written += 1
