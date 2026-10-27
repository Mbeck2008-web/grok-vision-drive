#!/usr/bin/env python3
"""Test-only: the one clock a converted supervisor check runs on (not a test itself).

``SimClock`` is owned by the test. It moves only when the test calls ``advance``. The fake
mod, the fake car, ``test_unknown_ego._session``'s ``t`` and the supervisor (through
scripts/gvd_clock_child.py and ``run_vision.main(clock=...)``) all read this one value, so a
slow or loaded host changes how long a check takes, never what the supervisor sees.

``t`` stands in for time.monotonic() and time.perf_counter(); ``wall()`` stands in for
time.time(), as ``W0 + (t - C0)`` with W0 the real wall time read once at spawn.
"""
from __future__ import annotations

import time

C0 = 1000.0


class SimClock:
    def __init__(self, t0: float = C0) -> None:
        self.t = float(t0)
        self.w0 = time.time()

    def advance(self, dt: float) -> float:
        self.t += float(dt)
        return self.t

    def mono(self) -> float:
        return self.t

    def wall(self) -> float:
        return self.w0 + (self.t - C0)
