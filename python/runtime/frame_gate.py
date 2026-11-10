"""Refuse a drive command when the camera frame is older than the cap.

Heartbeat, AEB, and engage are left alone. A stale frame replaces only a
command that would still drive (``reason == ok``, or a preview hold that is
still steering). An existing safety stop keeps its reason.
"""

from __future__ import annotations

from typing import Any

from python.control.actuate import stop_command

DEFAULT_MAX_FRAME_AGE_MS = 300.0
STALE_CAPTION = "LOW/STALE"


def frame_is_stale(age_s: float, max_age_ms: float) -> bool:
    return float(age_s) * 1000.0 > float(max_age_ms)


def gate_drive_command(
    cmd: Any,
    *,
    engaged: bool,
    stale: bool,
    holding: bool = False,
) -> tuple[Any, bool]:
    """Return ``(command, replaced)``.

    ``replaced`` is true only when a drive command was turned into
    ``frame_stale``. Disengaged ticks and safety stops are returned unchanged.
    """
    if cmd is None or not stale or not engaged:
        return cmd, False
    reason = str(getattr(cmd, "reason", "") or "")
    steering_hold = bool(holding) and reason == "preview_blocked"
    if reason != "ok" and not steering_hold:
        return cmd, False
    seq = int(getattr(cmd, "seq", 0) or 0)
    return stop_command(seq=seq, reason="frame_stale"), True
