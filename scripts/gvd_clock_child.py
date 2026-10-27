#!/usr/bin/env python3
"""Test-only launcher: ``python/run_vision.py`` stepped on the test's clock (scripts/gvd_sim_clock.py).

Not a test itself (the suite only runs ``test_*``). ``test_unknown_ego._session(clock=...)``
starts the supervisor through this file. It installs, unedited:

* the painted-road stub camera (``gvd_road_stub_child._grab``), and
* the echo-gate clock wrap (``gvd_echo_clock_child._observe``; the test writes the sim time
  to ``GVD_TEST_ECHO_CLOCK_FILE`` before each tick),

then calls ``run_vision.main(clock=PipeClock())``. ``PipeClock.sleep`` is the step barrier:
it writes ``done N`` on the protocol pipe (the original stdout) and blocks on reading
``go <t>`` from stdin. run_vision's own prints go to stderr. Nothing here sleeps on wall time.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# The protocol pipe is the stdout the test reads; run_vision's prints go to stderr instead.
_PROTO = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
os.dup2(2, 1)

import gvd_echo_clock_child  # noqa: E402 - needs GVD_TEST_ECHO_CLOCK_FILE, set by the test
import gvd_road_stub_child  # noqa: E402
from gvd_sim_clock import C0  # noqa: E402
from python.control import echo_link  # noqa: E402
from python.sensors import cameras  # noqa: E402


class PipeClock:
    """The supervisor's view of the test's SimClock: the last ``t`` the test sent."""

    def __init__(self, t: float, w0: float) -> None:
        self.t = float(t)
        self.w0 = float(w0)
        self.n = 0

    def mono(self) -> float:
        return self.t

    def perf(self) -> float:
        return self.t

    def wall(self) -> float:
        return self.w0 + (self.t - C0)

    def sleep(self, _s: float) -> None:
        """One tick is done: tell the test, then wait for it to release the next one."""
        self.n += 1
        _PROTO.write(f"done {self.n}\n")
        _PROTO.flush()
        line = sys.stdin.readline()
        if not line:
            raise SystemExit(0)  # the test closed the pipe
        word, t = line.split()
        if word != "go":
            raise SystemExit(f"gvd_clock_child: bad step line {line!r}")
        self.t = float(t)


def main() -> None:
    t0, w0 = (float(v) for v in os.environ["GVD_TEST_SIM_CLOCK"].split())
    cameras.StubBackend.grab = gvd_road_stub_child._grab
    echo_link.EchoGate.observe = gvd_echo_clock_child._observe
    script = ROOT / "python" / "run_vision.py"
    sys.argv = [str(script), *sys.argv[1:]]
    g = runpy.run_path(str(script), run_name="gvd_clock_child")
    g["main"](clock=PipeClock(t0, w0))


if __name__ == "__main__":
    main()
