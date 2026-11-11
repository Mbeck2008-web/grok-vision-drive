"""Latest-frame viz pump.

Paint (stage draw, imshow, waitKey) runs off the control thread. A newer
job replaces one the painter has not started. The control thread only
submits and drains keys, so a slow draw does not delay actuation.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable


class VizPump:
    """One painter thread. ``submit`` keeps the latest job and returns."""

    def __init__(
        self,
        paint: Callable[[Any], int | None],
        *,
        setup: Callable[[], None] | None = None,
        teardown: Callable[[], None] | None = None,
    ) -> None:
        self._paint = paint
        self._setup = setup
        self._teardown = teardown
        self._cond = threading.Condition()
        self._job: tuple[int, Any] | None = None
        self._seq = 0
        self._taken = 0
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._events: queue.SimpleQueue[tuple] = queue.SimpleQueue()
        self._setup_error: BaseException | None = None
        self._last_error = ""
        self.viz_ms = 0.0
        self.painted = 0
        self.dropped = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, name="gvd-viz", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=2.0)
        if self._setup_error is not None:
            raise self._setup_error

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)

    def submit(self, job: Any) -> None:
        with self._cond:
            if self._job is not None and self._job[0] != self._taken:
                self.dropped += 1
            self._seq += 1
            self._job = (self._seq, job)
            self._cond.notify()

    def poll_event(self) -> tuple | None:
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def push_event(self, event: tuple) -> None:
        self._events.put(event)

    def _run(self) -> None:
        try:
            if self._setup is not None:
                self._setup()
        except BaseException as exc:
            self._setup_error = exc
            self._ready.set()
            return
        self._ready.set()
        last = 0
        while not self._stop.is_set():
            with self._cond:
                item = self._job
                if item is None or item[0] == last:
                    self._cond.wait(timeout=0.03)
                    item = self._job
                if item is None or item[0] == last:
                    continue
                seq, job = item
                self._taken = seq
            last = seq
            t0 = time.perf_counter()
            key: int | None = None
            try:
                key = self._paint(job)
            except Exception as exc:
                text = str(exc)
                if text != self._last_error:
                    self._last_error = text
                    print(f"[GVD] viz paint: {text}", flush=True)
            self.viz_ms = (time.perf_counter() - t0) * 1000.0
            self.painted += 1
            if isinstance(key, int) and key >= 0:
                self._events.put(("key", key))
        if self._teardown is not None:
            try:
                self._teardown()
            except Exception as exc:
                print(f"[GVD] viz teardown: {exc}", flush=True)
