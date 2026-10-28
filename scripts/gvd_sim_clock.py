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


def install_bus_identity_wrap(clock) -> None:
    """Judge the supervisor's by-name ``bus_identity`` imports on ``clock.wall()``; clock=None: no-op.

    ``actuate.py`` (CmdJsonActuator.apply's bus check) and ``state_io.py`` (attach_bus_identity
    in write_state) import ``bus_identity`` by name and call it without ``now=``, so on a
    loaded host real time runs ahead of the sim wall the lua_bus files are stamped to and the
    actuator refuses with bus_mismatch. Patching ``paths.bus_identity`` would not reach them.
    Every argument passes through; an explicit ``now=`` from a caller wins.
    """
    if clock is None:
        return
    from python.control import actuate
    from python.runtime import paths, state_io

    real_bi = paths.bus_identity
    wrap = lambda *a, **k: real_bi(*a, **{"now": clock.wall(), **k})  # noqa: E731
    actuate.bus_identity = wrap
    state_io.bus_identity = wrap


class ActuateTime:
    """``actuate.time`` on the test clock: exactly time/monotonic/perf_counter, nothing else.

    time -> clock.wall, monotonic -> clock.mono, perf_counter -> clock.perf. Any other
    attribute (sleep, strftime, ...) raises AttributeError, so a new time use in actuate.py
    fails loudly in the clock child instead of silently reading the real clock.
    """

    __slots__ = ("_clock",)

    def __init__(self, clock) -> None:
        self._clock = clock

    def time(self) -> float:
        return self._clock.wall()

    def monotonic(self) -> float:
        return self._clock.mono()

    def perf_counter(self) -> float:
        return self._clock.perf()


def install_actuate_time_shim(clock) -> None:
    """Point ``actuate.time`` at ``ActuateTime(clock)``; clock=None: no-op (stdlib ``time`` stays).

    Stamps actuate writes (gvd_cmd.json heartbeat_mtime, gvd_engage.json mtime, ...) on the
    sim wall, so they compare against sim-wall stamps on a loaded host. state_io is not patched.
    """
    if clock is None:
        return
    from python.control import actuate

    actuate.time = ActuateTime(clock)
