"""Refuse a drive command when the camera frame is older than the cap.

Heartbeat, AEB, and engage are left alone. A stale frame replaces only a
command that would still drive (``reason == ok``, or a preview hold that is
still steering). An existing safety stop keeps its reason.
"""

from __future__ import annotations

import math
from typing import Any

from python.control.actuate import stop_command

DEFAULT_MAX_FRAME_AGE_MS = 300.0
STALE_CAPTION = "STALE"


def require_max_frame_age_ms(value: float) -> float:
    """Accept a finite cap above 0. ``nan`` and ``inf`` would disable the gate."""
    try:
        age = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(2) from exc
    if not math.isfinite(age) or age <= 0.0:
        print(
            f"[GVD] --max-frame-age-ms must be a finite number greater than 0, got {value}",
            flush=True,
        )
        raise SystemExit(2)
    return age


def frame_is_stale(age_s: float, max_age_ms: float) -> bool:
    """True when the frame must not drive.

    A missing stamp is ``inf``. A negative age means the stamp is ahead of the
    decision clock. Both are stale. A non-finite cap is stale too, so a bad
    limit cannot leave the gate open.
    """
    try:
        age = float(age_s)
        cap = float(max_age_ms)
    except (TypeError, ValueError):
        return True
    if not math.isfinite(age) or age < 0.0:
        return True
    if not math.isfinite(cap) or cap <= 0.0:
        return True
    return age * 1000.0 > cap


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
