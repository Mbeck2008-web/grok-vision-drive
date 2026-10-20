"""Per-tick timing for the collector, one line per loop tick in ``ticks.jsonl``.

A recording shows 0.25-0.5 s gaps between state rows and the rows do
not say where the time went. Each line here splits one tick into the parts the loop can
time itself:

``poll_s``    BeamNG camera poll (shared-memory reads)
``veh_s``     BeamNG vehicle sensor poll
``submit_s``  hand-off to the writer queue (array copies)
``sleep_s``   the planned sleep to the 0.1 s period
``top_s``     loop top before the poll (respawns, level loads, status)
``other_s``   the rest: labels, excursion control, impact checks

``sim_t``     BeamNG's sim clock from the State sensor (``None`` when not sent)
``sim_dt``    sim time since the previous tick's ``sim_t``. A wall ``dt``
              much longer than ``sim_dt`` is our loop or the transport; a
              long ``sim_dt`` too means the simulation itself took the time.

plus the writer queue depth and bytes at the hand-off, and the writer
threads' last encode and commit times. ``dt`` is the whole tick, start
to next start. A tick of two periods or more names its largest part in
``cause``. The encode and commit threads share the GIL with the loop, so
a slow tick with a deep queue and a long ``other_s`` points at the writer
even though no single timed part is large.

This file explains slow ticks. It does not change them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

TICKS_FILE = "ticks.jsonl"
_PARTS = ("top_s", "poll_s", "veh_s", "submit_s", "sleep_s")


def slow_cause(dt: float, parts: dict[str, float], period_s: float) -> str | None:
    """Largest timed part of a tick that took two periods or more. ``None`` otherwise."""
    if float(dt) < 2.0 * float(period_s):
        return None
    named = {k: float(v) for k, v in parts.items() if k != "sleep_s"}
    if not named:
        return "other_s"
    key = max(named, key=lambda k: named[k])
    return key if named[key] > 0.0 else "other_s"


class TickLog:
    """Buffered ``ticks.jsonl`` writer. ``start`` closes the previous tick."""

    def __init__(self, path: Path | str | None, period_s: float = 0.1, flush_every: int = 50) -> None:
        self.path = None if path is None else Path(path)
        self.period_s = float(period_s)
        self.flush_every = max(1, int(flush_every))
        self.n = 0
        self.slow = 0
        self._handle = None
        self._pending = 0
        self._t0: float | None = None
        self._start: float | None = None
        self._end: float | None = None
        self._parts: dict[str, float] = {}
        self._notes: dict[str, Any] = {}
        self._last_sim_t: float | None = None

    def _open(self) -> Any:
        if self._handle is None and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(self.path, "a", encoding="utf-8", newline="\n")
        return self._handle

    def start(self, mono: float) -> dict[str, Any] | None:
        """A new tick starts at ``mono``. Writes and returns the previous tick's line."""
        mono = float(mono)
        line = None
        if self._start is not None:
            if self._end is not None:
                self._parts["top_s"] = self._parts.get("top_s", 0.0) + max(0.0, mono - self._end)
            line = self._emit(mono - self._start)
        if self._t0 is None:
            self._t0 = mono
        self._start = mono
        self._end = None
        self._parts = {}
        self._notes = {}
        return line

    def add(self, part: str, seconds: float) -> None:
        if self._start is None:
            return
        self._parts[part] = self._parts.get(part, 0.0) + max(0.0, float(seconds))

    def note(self, **values: Any) -> None:
        if self._start is not None:
            self._notes.update(values)

    def end(self, mono: float) -> None:
        """The tick's own work and sleep are done; later time is loop top."""
        if self._start is not None:
            self._end = float(mono)

    def _emit(self, dt: float) -> dict[str, Any]:
        timed = sum(self._parts.get(k, 0.0) for k in _PARTS)
        parts = {k: self._parts.get(k, 0.0) for k in _PARTS}
        parts["other_s"] = max(0.0, float(dt) - timed)
        line: dict[str, Any] = {
            "k": self.n,
            "t": round(float(self._start or 0.0) - float(self._t0 or 0.0), 4),
            "dt": round(float(dt), 4),
        }
        for key, value in parts.items():
            line[key] = round(float(value), 4)
        line.update(self._notes)
        sim_t = self._notes.get("sim_t")
        if isinstance(sim_t, (int, float)):
            if self._last_sim_t is not None:
                line["sim_dt"] = round(float(sim_t) - self._last_sim_t, 4)
            self._last_sim_t = float(sim_t)
        cause = slow_cause(dt, parts, self.period_s)
        if cause is not None:
            line["cause"] = cause
            self.slow += 1
        self.n += 1
        handle = self._open()
        if handle is not None:
            handle.write(json.dumps(line, separators=(",", ":")) + "\n")
            self._pending += 1
            if self._pending >= self.flush_every or cause is not None:
                handle.flush()
                self._pending = 0
        return line

    def close(self) -> None:
        if self._start is not None and self._end is not None:
            self._emit(self._end - self._start)
            self._start = None
        if self._handle is not None:
            try:
                self._handle.flush()
                os.fsync(self._handle.fileno())
            except OSError:
                pass
            self._handle.close()
            self._handle = None
