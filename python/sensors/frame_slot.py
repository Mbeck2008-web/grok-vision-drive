"""Latest-frame capture slot for the vision supervisor.

The ep11 shadow run logged ``frame_age`` p90 of 11.5 s at about 3.1 Hz.
That number is not how long ``grab()`` blocked. ``PerceptionOut.frame_age_s``
is the lane-decision age (time since the lane picture last changed, or since
the lane source was a fresh take). A scene-ckpt run whose pixel fitter stays
on ``none`` / ``hold`` climbs for the whole run, and
``apply_stale_speed_cap`` then treats anything over 0.25 s as a stale lane.

The camera path makes the pixels themselves old:

* ``BeamNGPyBackend.grab`` reads all eight cameras on the control thread.
  Scene-ckpt only consumes ``main``, but the other seven stay attached and
  scheduled, so the Tech sensor manager falls behind on a shared 1080 Ti.
* ``stream_raw`` (BeamNGpy, shared memory) returns the latest buffer and does
  not send ``PollCamera``. It also does not say when that buffer was rendered.
  The grab stamped every camera with ``time.time()`` at the start of the
  pass, so a repeated buffer looked new in ``frame_cmd_ms`` while the bytes
  did not change.
* A late read was kept in ``_cache_frames`` and not published. Nothing dropped
  a stale frame in favour of a newer one; the control thread just waited.

``LatestFrameSlot`` keeps one bundle. A put that replaces a frame the
consumer has not taken increments ``dropped``. BeamNGpy itself is unchanged:
``stream_raw`` is already the latest shared-memory read. The pin stays 1.36.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np


def percentile(values: Iterable[float], p: float) -> float:
    """Linear percentile in ``[0, 1]``. Empty input is 0."""
    data = sorted(float(v) for v in values)
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    frac = min(1.0, max(0.0, float(p)))
    k = (len(data) - 1) * frac
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return data[lo]
    weight = k - lo
    return data[lo] * (1.0 - weight) + data[hi] * weight


class AgeWindow:
    """Rolling frame ages, in milliseconds, for the live p50 / p90 line."""

    def __init__(self, cap: int = 128) -> None:
        self._values: deque[float] = deque(maxlen=max(1, int(cap)))

    def add(self, age_ms: float) -> None:
        self._values.append(float(age_ms))

    def p50(self) -> float:
        return percentile(self._values, 0.50)

    def p90(self) -> float:
        return percentile(self._values, 0.90)

    def __len__(self) -> int:
        return len(self._values)


def bundle_frame_age_s(bundle: Any, now: float | None = None) -> float:
    """Seconds from the content stamp to the decision clock.

    ``captured_at`` is ``time.monotonic()`` when the pixels were first seen.
    A missing, non-finite, or non-positive stamp is infinitely old: a waiting
    bundle and a grab with no main picture must not count as fresh. A stamp
    ahead of ``now`` returns a negative age, which the gate treats as stale.
    """
    raw = getattr(bundle, "captured_at", None)
    try:
        captured = float(raw)
    except (TypeError, ValueError):
        return math.inf
    if not math.isfinite(captured) or captured <= 0.0:
        return math.inf
    if now is None:
        clock = time.monotonic()
    else:
        try:
            clock = float(now)
        except (TypeError, ValueError):
            return math.inf
    if not math.isfinite(clock):
        return math.inf
    return clock - captured


def format_latency_line(
    *,
    capture_ms: float,
    infer_ms: float,
    viz_ms: float,
    total_ms: float,
    frame_age_ms: float,
    frame_age_p50_ms: float,
    frame_age_p90_ms: float,
    rss_mb: float,
    dropped: int,
) -> str:
    """One supervisor tick. ``frame_age_*`` is capture age, not lane-decision age."""
    return (
        f"[GVD] lat capture_ms={float(capture_ms):.2f} infer_ms={float(infer_ms):.2f} "
        f"viz_ms={float(viz_ms):.2f} total_ms={float(total_ms):.2f} "
        f"frame_age_ms={float(frame_age_ms):.1f} "
        f"frame_age_p50_ms={float(frame_age_p50_ms):.1f} "
        f"frame_age_p90_ms={float(frame_age_p90_ms):.1f} "
        f"rss_mb={float(rss_mb):.1f} dropped={int(dropped)}"
    )


def rss_mb_from_statm(text: str, page_size: int) -> float:
    """Resident set from a ``/proc/self/statm`` line. The second field is pages."""
    pages = int(str(text).split()[1])
    return pages * int(page_size) / (1024.0 * 1024.0)


def _windows_working_set_bytes() -> int | None:
    """Current working set via ``GetProcessMemoryInfo``. None off Windows."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_mem = ctypes.windll.psapi.GetProcessMemoryInfo
    ok = get_mem(get_process(), ctypes.byref(counters), counters.cb)
    if not ok:
        return None
    return int(counters.WorkingSetSize)


def process_rss_mb() -> float:
    """Current resident set in MiB. Never the peak ``ru_maxrss`` when a current source exists.

    Linux reads ``/proc/self/statm`` (Windows has no ``resource`` module, which
    is why the ep11 log showed ``rss_mb`` 0 when psutil was also absent).
    Windows uses ``GetProcessMemoryInfo``. psutil, then ``resource``, are fallbacks.
    """
    try:
        text = open("/proc/self/statm", encoding="ascii").read()
        page = int(os.sysconf("SC_PAGE_SIZE"))
        mb = rss_mb_from_statm(text, page)
        if mb > 0.0:
            return mb
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    try:
        win = _windows_working_set_bytes()
        if win is not None and win > 0:
            return win / (1024.0 * 1024.0)
    except (AttributeError, OSError, ValueError):
        pass
    try:
        import psutil  # type: ignore

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except Exception:
        return 0.0


@dataclass
class HeldFrame:
    seq: int
    payload: Any


class LatestFrameSlot:
    """One slot. A newer put drops an untaken frame instead of queueing it."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: HeldFrame | None = None
        self._taken_seq = 0
        self.dropped = 0
        self.published = 0

    def put(self, payload: Any) -> int:
        with self._cond:
            if self._item is not None and self._item.seq != self._taken_seq:
                self.dropped += 1
            self.published += 1
            self._item = HeldFrame(seq=self.published, payload=payload)
            self._cond.notify_all()
            return self.published

    def take(self) -> HeldFrame | None:
        """The newest frame. Taking the same seq twice means nothing newer arrived."""
        with self._cond:
            if self._item is None:
                return None
            self._taken_seq = self._item.seq
            return self._item

    def wait(self, timeout_s: float) -> bool:
        with self._cond:
            if self._item is not None:
                return True
            self._cond.wait(timeout=max(0.0, float(timeout_s)))
            return self._item is not None


class CapturePump:
    """Calls ``produce`` off the control thread and publishes into ``slot``."""

    def __init__(
        self,
        slot: LatestFrameSlot,
        produce: Callable[[], Any],
        *,
        idle_s: float = 0.002,
    ) -> None:
        self.slot = slot
        self._produce = produce
        self._idle_s = max(0.0, float(idle_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gvd-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            bundle = None
            try:
                bundle = self._produce()
            except Exception:
                bundle = None
            if bundle is not None:
                self.slot.put(bundle)
            spent = time.perf_counter() - t0
            pause = self._idle_s if spent < self._idle_s else 0.0
            if self._stop.wait(pause):
                return


class LatencyStubBackend:
    """Fake camera. Each ``grab`` sleeps ``latency_s`` and stamps the request time.

    The bytes change every grab, so a latest-slot consumer sees a new frame.
    A FIFO in front of this backend is what makes the age grow; the backend
    itself does not queue.
    """

    name = "stub"

    def __init__(self, latency_s: float = 0.02, *, cams: Sequence[str] = ("main",)) -> None:
        self.latency_s = max(0.0, float(latency_s))
        self.cams = tuple(cams)
        self.grabs = 0
        self._closed = False

    def open(self) -> None:
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def grab(self) -> Any:
        from python.sensors.cameras import CamHealth, CameraFrameBundle

        started = time.perf_counter()
        stamp = time.monotonic()
        if self.latency_s > 0.0:
            time.sleep(self.latency_s)
        self.grabs += 1
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        img[0, 0, 0] = self.grabs % 251 or 1
        frames = {cid: img for cid in self.cams}
        if "main" in frames:
            frames["cam_main"] = img
        health = {cid: CamHealth.OK for cid in self.cams}
        return CameraFrameBundle(
            frames=frames,
            timestamps={cid: stamp for cid in self.cams},
            health=health,
            backend=self.name,
            note="latency-stub",
            grab_ms=(time.perf_counter() - started) * 1000.0,
            captured_at=stamp,
            unique_gpu_n=1,
            unique_gpu_ids=tuple(self.cams),
        )
