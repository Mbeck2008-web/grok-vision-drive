#!/usr/bin/env python3
"""Test-only launcher: ``python/run_vision.py`` with the echo gate on the fake mod's clock.

Not a test itself (the suite only runs ``test_*``). ``scripts/test_m6_retail.py`` starts the
supervisor through this file when a check plays the mod's echo on the mod's own clock: each
echo the fake mod writes is ``FAKE_ECHO_PERIOD_S`` of mod time, and the test writes that time
to ``GVD_TEST_ECHO_CLOCK_FILE`` (``{"t": seconds}``). Here ``EchoGate.observe`` is wrapped so
its ``now`` is that clock instead of the supervisor's ``time.monotonic()``; the gate's logic,
``ECHO_STALE_MS`` and everything else in run_vision are unchanged. A slow test host then cannot
starve the fake mod into an ``echo_stale`` it would never show at 10 Hz of mod time.
"""
from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.control import echo_link  # noqa: E402

CLOCK = Path(os.environ["GVD_TEST_ECHO_CLOCK_FILE"])
_REAL_OBSERVE = echo_link.EchoGate.observe
_last = [0.0]


def echo_clock() -> float:
    """The fake mod's time; the last good value when the file is mid-replace."""
    try:
        _last[0] = float(json.loads(CLOCK.read_text(encoding="utf-8"))["t"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return _last[0]


def _observe(self, *, engaged, ego_fb, now):  # noqa: ARG001 - the supervisor's monotonic now is replaced
    return _REAL_OBSERVE(self, engaged=engaged, ego_fb=ego_fb, now=echo_clock())


def main() -> None:
    echo_link.EchoGate.observe = _observe
    script = ROOT / "python" / "run_vision.py"
    sys.argv = [str(script), *sys.argv[1:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
