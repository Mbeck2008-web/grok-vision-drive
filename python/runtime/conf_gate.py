"""Single switch for perception/lane/path confidence vetoes.

``--no-conf-gate`` is opt-in and off by default. Every confidence veto must
call ``conf_veto_blocks`` so the flag cannot be applied at only one site.
AEB, heartbeat, frame-staleness, missing frames, user disengage, and Soft Esc
are not confidence vetoes and must not consult this helper.
"""

from __future__ import annotations

HUD_TAG = "CONF GATE OFF"
STARTUP_WARNING = (
    "[GVD] WARNING: --no-conf-gate: confidence vetoes are OFF; "
    "low lane/path/scene confidence will not block Engage or force a hold/stop. "
    "AEB, heartbeat, frame-staleness, missing-frame stops, user disengage, and Soft Esc stay on."
)


def conf_veto_blocks(*, no_conf_gate: bool, triggered: bool) -> bool:
    """Return whether a confidence comparison should block drive or Engage.

    ``triggered`` is the call site's existing comparison (today's default).
    When ``no_conf_gate`` is set, this always returns False.
    """
    if no_conf_gate:
        return False
    return bool(triggered)


def hud_tag(no_conf_gate: bool) -> str:
    """Cabin top-bar label, or empty when the gate is on (the default)."""
    return HUD_TAG if no_conf_gate else ""
