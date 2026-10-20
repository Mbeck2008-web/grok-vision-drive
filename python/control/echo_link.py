"""Which reads of gvd_ego.json are new samples of the wheel.

The mod writes the electrics echo at 10 Hz (`EGO_POLL_S`). The supervisor reads it every loop
tick, at 15-20 Hz. Most ticks therefore re-read the echo the previous tick already judged, and
under CPU load a tick can also miss one. The override filter used to take every read as a new
sample. A repeated read, or a skipped one, put two same-sign force-feedback kicks next to each
other. Neither was a spike, so the EMA followed them. The kicks after that were spikes, so the
filter stayed frozen high while the dwell kept counting, and chatter read as `player_steer`.

The mod now stamps each echo with `echo_seq`, its own counter, one per echo written.
`EchoGate.observe` turns each read into one of:

* ``fresh``    — `echo_seq` moved on. The override filter and its dwell clock consume it.
  ``gap`` is set when it moved on by more than one, so echoes were missed in between.
* ``repeat``   — the echo already consumed. Nothing new about the wheel; the filter waits.
* ``waiting``  — no echo with an `echo_seq` seen yet this engagement. Nothing is applied
  until the mod acks a command either (the cmd_json actuator stays `cmd_json_pending`).
* ``stale``    — the link was live this engagement, and `echo_seq` has not moved on for
  `stale_ms` (about 300 ms, three echoes), or the last `GAP_RUN_MAX` fresh echoes all arrived
  with gaps. Frozen, gone, or never contiguous, the override has nothing it can judge. That
  is a link fault, never "no override": the supervisor disengages with `echo_stale` on the
  same path a player override takes. One gap after a supervisor stall is not a fault; the
  override re-seeds on it and the next contiguous echo decides.
* ``outdated`` — a live echo with no `echo_seq`. The mod predates this contract and cannot say
  whether a read is new, so engage is refused with `lua_echo_outdated`. There is no fallback to
  judging every read.
* ``off``      — not engaged. The gate forgets everything, so a re-engage starts clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ECHO_STALE_MS = 300.0
# Fresh echoes in a row that each skipped some: the filter re-seeds on every one, so it is blind.
GAP_RUN_MAX = 3

REASON_ECHO_STALE = "echo_stale"
REASON_ECHO_OUTDATED = "lua_echo_outdated"

STATUS_FRESH = "fresh"
STATUS_REPEAT = "repeat"
STATUS_WAITING = "waiting"
STATUS_STALE = "stale"
STATUS_OUTDATED = "outdated"
STATUS_OFF = "off"

FAULT_REASONS = {STATUS_STALE: REASON_ECHO_STALE, STATUS_OUTDATED: REASON_ECHO_OUTDATED}


@dataclass(frozen=True)
class EchoRead:
    status: str = STATUS_OFF
    seq: int | None = None
    skipped: int = 0          # echoes missed between this fresh one and the last
    since_ms: float = 0.0     # since echo_seq last moved on (0 before the link is up)

    @property
    def fresh(self) -> bool:
        return self.status == STATUS_FRESH

    @property
    def gap(self) -> bool:
        return self.status == STATUS_FRESH and self.skipped != 0

    @property
    def fault(self) -> str | None:
        """The disengage reason this read calls for, or None."""
        return FAULT_REASONS.get(self.status)

    def telemetry(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "seq": self.seq,
            "skipped": int(self.skipped),
            "since_ms": round(float(self.since_ms), 1),
        }


class EchoGate:
    """Per-loop freshness of the mod's electrics echo, keyed on its `echo_seq`."""

    def __init__(self, stale_ms: float = ECHO_STALE_MS) -> None:
        self.stale_s = max(0.05, float(stale_ms) / 1000.0)
        self.read = EchoRead()
        self.reset()

    def reset(self) -> None:
        self._last_seq: int | None = None
        self._moved_at: float | None = None
        self._gap_run = 0
        self.read = EchoRead()

    def observe(self, *, engaged: bool, ego_fb: Any, now: float) -> EchoRead:
        """Classify this tick's read. `ego_fb` is an EgoFeedback or None (missing / torn)."""
        if not engaged:
            self.reset()
            return self.read
        t = float(now)
        live = ego_fb is not None and bool(getattr(ego_fb, "fresh", False))
        seq = getattr(ego_fb, "echo_seq", None) if live else None
        if live and seq is None:
            self.read = EchoRead(status=STATUS_OUTDATED)
            return self.read
        if seq is not None and seq != self._last_seq:
            skipped = 0
            if self._last_seq is not None:
                # A mod reload restarts the counter: a step back is a gap too, not a repeat.
                skipped = seq - self._last_seq - 1 if seq > self._last_seq else -1
            self._last_seq = int(seq)
            self._moved_at = t
            self._gap_run = self._gap_run + 1 if skipped != 0 else 0
            if self._gap_run >= GAP_RUN_MAX:
                self.read = EchoRead(status=STATUS_STALE, seq=int(seq), skipped=skipped)
                return self.read
            self.read = EchoRead(status=STATUS_FRESH, seq=int(seq), skipped=skipped)
            return self.read
        if self._moved_at is None:
            self.read = EchoRead(status=STATUS_WAITING)
            return self.read
        since = max(0.0, t - self._moved_at)
        status = STATUS_STALE if since >= self.stale_s else STATUS_REPEAT
        self.read = EchoRead(status=status, seq=self._last_seq, since_ms=since * 1000.0)
        return self.read
