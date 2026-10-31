"""Time-based recording boundaries wait for steady driving.

A ``--segment-min`` chunk switch (level reload or next segment) and a
``--respawn-min`` teleport are due on the clock. Cutting a weave, lane change
or recover in half, or a settle that just started, leaves a stub excursion
in the recording, so a due boundary is held until the excursion clock is back
in ``cruise`` and the settle window is over, for at most
``BOUNDARY_DEFER_CAP_S`` of driving time; then it switches anyway.

A held segment switch carries its wait over: the collector moves the
schedule back by ``waited_s``, so the next chunk starts where the switch
first fell due (a 30 s hold no longer leaves a 20 s chunk only its settle
window). ``CarryOver`` then owes that time back evenly over the rest of the
run (until the run total), so the carried seconds do not pile up and all come
out of the last chunk: every later chunk runs ``rest / (rest + owed)`` of its
slot instead. That keeps chunks near their slot only for EARLY holds, with a
long rest to spread over. A late hold has little rest left, so the repay rate
can reach or pass 1 and the next chunk(s) pay most of it: a 30 s hold at the
last boundary of a 200 s run leaves the tail 30 of 60 s, and a 115 s run held
at 60 s leaves italy#1 5.3 of 20 s. The floor is the cap: a chunk loses at
most ``BOUNDARY_DEFER_CAP_S`` (30 s) per hold, so the tail keeps at least its
slot - 30 s per hold it pays for.

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


class CarryOver:
    """Seconds a held segment switch moved the schedule back, owed back evenly by the run's end."""

    def __init__(self) -> None:
        self.total_s = 0.0  # every carried second this run
        self.owed_s = 0.0  # carried and not yet repaid
        self._rate = 0.0  # owed seconds repaid per driving second
        self._at = 0.0

    def add(self, waited_s: float, now_s: float, end_s: float) -> float:
        """A hold released at ``now_s`` carried ``waited_s``; repay all that is owed by ``end_s``.

        Before the new rate is set, whatever fell due on the old rate since the last ``repay()``
        is settled and RETURNED: the caller must give it back to the schedule like a ``repay()``
        result. It is 0.0 when the caller already called ``repay(now_s)`` this tick (the
        collector does)."""
        paid = self.repay(now_s)
        w = max(0.0, float(waited_s))
        self.total_s += w
        self.owed_s += w
        left = float(end_s) - float(now_s)
        self._rate = self.owed_s / left if left > 0.0 and left != float("inf") else 0.0
        return paid

    def repay(self, now_s: float) -> float:
        """Seconds to give back to the schedule since the last call (never more than owed)."""
        now = float(now_s)
        pay = min(self.owed_s, self._rate * max(0.0, now - self._at))
        self._at = max(self._at, now)
        self.owed_s -= pay
        if self.owed_s <= 1e-9:
            self.owed_s, self._rate = 0.0, 0.0
        return pay

    def forgive(self) -> None:
        """The schedule restarted (wrapped to 0): nothing is owed any more."""
        self.owed_s, self._rate = 0.0, 0.0

    def summary(self) -> dict[str, float]:
        return {"carried_s": round(self.total_s, 1), "owed_s": round(self.owed_s, 1)}
