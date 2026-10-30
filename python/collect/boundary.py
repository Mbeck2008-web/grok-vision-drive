"""Time-based recording boundaries wait for steady driving.

A ``--segment-min`` chunk switch (level reload or next segment) and a
``--respawn-min`` teleport are due on the clock. Cutting a weave, lane change
or recover in half, or a settle that just started, leaves a stub excursion
in the recording, so a due boundary is held until the excursion clock is back
in ``cruise`` and the settle window is over, for at most
``BOUNDARY_DEFER_CAP_S`` of driving time; then it switches anyway.

A held segment switch carries its wait over: the collector moves the
schedule back by ``waited_s``, so the next chunk starts where the switch
first fell due and keeps its full length (a 30 s hold no longer leaves a
20 s chunk only its settle window).

Forced boundaries (aborts, stalls, impacts, a cut schedule) are not held.
Logs: ``collect boundary_defer <kind>:<phase> 0.0`` when the hold starts,
``collect boundary_defer <kind>:cruise <s>`` or ``<kind>:cap <s>`` when it is
released, ``<s>`` being the seconds of driving clock it waited.
"""

from __future__ import annotations

BOUNDARY_DEFER_CAP_S = 30.0


def boundary_wait_reason(phase: str, settling: bool) -> str:
    """Why a due boundary should wait: the settle window, a non-cruise phase, or '' (switch now)."""
    if settling:
        return "settle"
    if phase != "cruise":
        return str(phase or "unknown")
    return ""


class BoundaryDefer:
    """One kind of due boundary (``segment`` or ``respawn``) on the driving clock."""

    def __init__(self, kind: str, cap_s: float = BOUNDARY_DEFER_CAP_S) -> None:
        self.kind = str(kind)
        self.cap_s = float(cap_s)
        self.since: float | None = None
        self.waited_s = 0.0  # driving seconds the last released hold waited

    def pending(self) -> bool:
        return self.since is not None

    def clear(self) -> None:
        """The boundary is no longer due (something else switched); drop the hold silently."""
        self.since = None

    def hold(self, now_s: float, why: str) -> bool:
        """Call while the boundary is due. True: keep driving the current chunk or spawn this tick."""
        now = float(now_s)
        if self.since is None:
            if not why:
                return False
            self.since = now
            print(f"collect boundary_defer {self.kind}:{why} 0.0", flush=True)
            return True
        waited = now - self.since
        if why and waited < self.cap_s - 1e-6:
            return True
        print(f"collect boundary_defer {self.kind}:{'cap' if why else 'cruise'} {waited:.1f}", flush=True)
        self.since = None
        self.waited_s = max(0.0, waited)
        return False
