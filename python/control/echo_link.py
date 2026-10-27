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
* ``outdated`` — a live echo from a mod older than `MIN_MOD_VERSION`: no `echo_seq`, or a
  `mod_version` missing or below the minimum. That mod cannot say whether a read is new (or
  says it with an older meaning), so engage is refused with `lua_echo_outdated`, and the message
  names the minimum mod version (`outdated_message`). There is no fallback to judging every
  read.

`echo_seq` is the mod's own counter (+1 per echo, `onEgoFeedback`), never the last command seq
it received (`applied_seq`). A command seq stops moving whenever Python stalls or a cmd write is
lost, while the wheel keeps being read and echoed; keyed on that, the gate would call a live
wheel stale.
* ``off``      — not engaged. The gate forgets everything, so a re-engage starts clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ECHO_STALE_MS = 300.0
# Fresh echoes in a row that each skipped some: the filter re-seeds on every one, so it is blind.
GAP_RUN_MAX = 3

# First mod that stamps echo_seq as its own per-echo counter and writes mod_version. Bump only
# when the mod/bus contract the supervisor relies on changes, not on every release.
MIN_MOD_VERSION = "1.7.13"

REASON_ECHO_STALE = "echo_stale"
REASON_ECHO_OUTDATED = "lua_echo_outdated"

STATUS_FRESH = "fresh"
STATUS_REPEAT = "repeat"
STATUS_WAITING = "waiting"
STATUS_STALE = "stale"
STATUS_OUTDATED = "outdated"
STATUS_OFF = "off"

FAULT_REASONS = {STATUS_STALE: REASON_ECHO_STALE, STATUS_OUTDATED: REASON_ECHO_OUTDATED}


def parse_version(v: Any) -> tuple[int, ...] | None:
    """'1.7.13' (or '1.7.13-alpha-<sha>') -> (1, 7, 13). None when it is not a version."""
    if not isinstance(v, str):
        return None
    head = v.strip().split("-", 1)[0]
    parts = head.split(".")
    if not parts or not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def mod_outdated(ego_fb: Any, minimum: str = MIN_MOD_VERSION) -> bool:
    """True for a live echo from a mod older than `minimum` (no echo_seq, or old/no mod_version)."""
    if getattr(ego_fb, "echo_seq", None) is None:
        return True
    have = parse_version(getattr(ego_fb, "mod_version", None))
    need = parse_version(minimum) or ()
    return have is None or have < need


def outdated_message(ego_fb: Any, minimum: str = MIN_MOD_VERSION) -> str:
    """The refusal line: what the mod reported, and the minimum mod version required."""
    have = getattr(ego_fb, "mod_version", None) if ego_fb is not None else None
    found = f"mod {have}" if isinstance(have, str) and have else "a mod without mod_version"
    if ego_fb is not None and getattr(ego_fb, "echo_seq", None) is None:
        found += " and no echo_seq"
    return (
        f"{REASON_ECHO_OUTDATED}: gvd_ego.json is from {found}; "
        f"Grok Vision Drive mod {minimum} or newer is required"
    )


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

    def __init__(self, stale_ms: float = ECHO_STALE_MS, min_mod_version: str = MIN_MOD_VERSION) -> None:
        self.stale_s = max(0.05, float(stale_ms) / 1000.0)
        self.min_mod_version = min_mod_version
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
        if live and mod_outdated(ego_fb, self.min_mod_version):
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


class SessionEcho:
    """The echo read for the ego speed: every tick, engaged or not, keyed on this session.

    The first read is only a baseline. An echo an old supervisor session left behind can still
    be under 1 s old, so its seq says nothing about now. Until ``echo_seq`` moves on past that
    baseline the read is ``waiting`` (speed unknown). After that an ``EchoGate`` judges every
    read, so the 300 ms stale window, the gap run and the outdated check are the gate's own.
    ``speed_mps`` is the speed in the last advanced echo, or None.
    """

    def __init__(self, stale_ms: float = ECHO_STALE_MS, min_mod_version: str = MIN_MOD_VERSION) -> None:
        self.min_mod_version = min_mod_version
        self._gate = EchoGate(stale_ms=stale_ms, min_mod_version=min_mod_version)
        self._baselined = False
        self._baseline_seq: int | None = None
        self._advanced = False
        self.speed_mps: float | None = None
        self.read = EchoRead(status=STATUS_WAITING)

    @property
    def status(self) -> str:
        return self.read.status

    def baseline(self, ego_fb: Any) -> None:
        """Take the session baseline (once). Any seq on disk now belongs to before this session."""
        if self._baselined:
            return
        self._baselined = True
        seq = getattr(ego_fb, "echo_seq", None) if ego_fb is not None else None
        self._baseline_seq = int(seq) if seq is not None else None

    def observe(self, *, ego_fb: Any, now: float) -> EchoRead:
        """This tick's read. ``ego_fb`` is an EgoFeedback or None (missing / torn)."""
        if not self._baselined:
            self.baseline(ego_fb)
            self.speed_mps = None
            self.read = EchoRead(status=STATUS_WAITING)
            return self.read
        if not self._advanced:
            live = ego_fb is not None and bool(getattr(ego_fb, "fresh", False))
            if live and mod_outdated(ego_fb, self.min_mod_version):
                self.speed_mps = None
                self.read = EchoRead(status=STATUS_OUTDATED)
                return self.read
            seq = getattr(ego_fb, "echo_seq", None) if live else None
            if seq is not None and self._baseline_seq is None:
                # No echo at baseline: the first seq seen is the baseline, not an advance.
                self._baseline_seq = int(seq)
            if seq is None or seq == self._baseline_seq:
                self.speed_mps = None
                self.read = EchoRead(status=STATUS_WAITING)
                return self.read
            self._advanced = True
            self._gate.reset()
        read = self._gate.observe(engaged=True, ego_fb=ego_fb, now=now)
        if read.status == STATUS_FRESH:
            speed = getattr(ego_fb, "speed_mps", None)
            self.speed_mps = float(speed) if speed is not None else None
        elif read.status != STATUS_REPEAT:
            self.speed_mps = None
        self.read = read
        return self.read

    def telemetry(self) -> dict[str, Any]:
        out = self.read.telemetry()
        out["baseline_seq"] = self._baseline_seq
        out["advanced"] = bool(self._advanced)
        return out
