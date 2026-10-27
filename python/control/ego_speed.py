"""Ego speed is known only from a fresh measurement taken in THIS supervisor session.

Otherwise it is ``None``. Code that needs a speed and gets ``None`` refuses: throttle 0,
brake 0, a named disengage reason. Nothing stands in for an unknown speed: not 0.0, not
the last value, not "slow".

When the speed counts as known:

* Retail (cmd_json bus): the session echo read (``echo_link.SessionEcho``) is FRESH or
  REPEAT. That needs ``echo_seq`` to have moved on at least once since this supervisor
  started (the first read is only a baseline: an echo an old session left behind can still
  be under 1 s old), and the last advance to be under ``ECHO_STALE_MS`` ago on the
  supervisor's monotonic clock. The value is the speed in the last advanced echo.
* Tech (beamngpy): this tick's poll or electrics speed, with ``StaleEgo`` not blocking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from python.control.echo_link import (
    REASON_ECHO_OUTDATED,
    REASON_ECHO_STALE,
    STATUS_FRESH,
    STATUS_OUTDATED,
    STATUS_REPEAT,
    STATUS_STALE,
)

REASON_EGO_UNKNOWN = "ego_unknown"
REASON_LATCHED_BEFORE_START = "latched_before_start"
REASON_HEARTBEAT_STALE = "heartbeat_stale"

# Every reason an unknown ego speed refuses or disengages with. Each one releases the car
# (throttle 0, brake 0); none of them is a held stop.
UNKNOWN_EGO_REASONS = frozenset({
    REASON_EGO_UNKNOWN,
    REASON_LATCHED_BEFORE_START,
    REASON_ECHO_STALE,
    REASON_ECHO_OUTDATED,
    REASON_HEARTBEAT_STALE,
})


@dataclass(frozen=True)
class EgoSpeed:
    """This tick's ego speed. ``mps`` is None when it is not known; ``why`` says which way."""

    mps: float | None
    source: str
    why: str

    @property
    def known(self) -> bool:
        return self.mps is not None

    @property
    def reason(self) -> str:
        """The disengage reason for an unknown speed (``why`` when it is one, else ego_unknown)."""
        if self.mps is not None:
            return "none"
        return self.why if self.why in UNKNOWN_EGO_REASONS else REASON_EGO_UNKNOWN


def resolve_ego(
    *,
    echo_status: str | None,
    echo_speed_mps: float | None,
    tech_speed_mps: float | None,
    tech_stale: bool | None,
) -> EgoSpeed:
    """One tick's speed, from what was measured this tick in this session.

    ``echo_status`` is the session echo read (``SessionEcho.observe``), ``echo_speed_mps``
    the speed in the last advanced echo. ``tech_speed_mps`` is this tick's beamngpy poll or
    electrics speed and ``tech_stale`` whether ``StaleEgo`` blocks it. Anything missing is
    None, never a stand-in.
    """
    if tech_speed_mps is not None:
        if tech_stale:
            return EgoSpeed(None, "beamngpy", REASON_HEARTBEAT_STALE)
        return EgoSpeed(max(0.0, float(tech_speed_mps)), "beamngpy", "poll")
    if echo_status in (STATUS_FRESH, STATUS_REPEAT) and echo_speed_mps is not None:
        return EgoSpeed(max(0.0, float(echo_speed_mps)), "lua", str(echo_status))
    if echo_status == STATUS_STALE:
        return EgoSpeed(None, "lua", REASON_ECHO_STALE)
    if echo_status == STATUS_OUTDATED:
        return EgoSpeed(None, "lua", REASON_ECHO_OUTDATED)
    if echo_status is not None:
        return EgoSpeed(None, "lua", f"no advancing echo this session ({echo_status})")
    return EgoSpeed(None, "none", "nothing measured this session")


def engage_gate(latch_on: bool, seen_off_this_session: bool, ego: EgoSpeed | Any) -> tuple[bool, str]:
    """May an engage request be honoured this tick? ``(allow, reason)``.

    A latch that was already on when this session started is refused with
    ``latched_before_start``: ON counts only on an off->on edge seen in this session.
    A request while the speed is unknown is refused with the unknown reason.
    """
    if not latch_on:
        return False, "not_engaged"
    if not seen_off_this_session:
        return False, REASON_LATCHED_BEFORE_START
    if getattr(ego, "mps", None) is None:
        reason = getattr(ego, "reason", REASON_EGO_UNKNOWN)
        return False, str(reason or REASON_EGO_UNKNOWN)
    return True, "ok"
