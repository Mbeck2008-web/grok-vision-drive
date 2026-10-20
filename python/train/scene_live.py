"""Torch-free scene-net pieces the live supervisor uses every tick.

``python/run_vision.py`` imports these at startup. ``scene_net`` imports torch
at module load (about 0.8 s on a CPU box), and the stub / retail supervisor
never runs the net, so these live here and ``scene_net`` re-exports them.
Nothing in this module may import torch.
"""

from __future__ import annotations

from typing import Any

_BAD_REASONS = frozenset({"player_steer", "player_brake", "player_throttle"})

# ``stop_command`` reason while the scene ONNX input shape and the live canvas
# disagree (``scene_lane_runtime.SceneLaneGuard``).
SCENE_SHAPE_HOLD = "scene_hold:input_shape"


def _has_labels(value: Any) -> bool:
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return True


def live_strip_fields(
    *,
    lanes: Any = None,
    tracks: Any = None,
    signs: Any = None,
    curbs: Any = None,
    steer: float = 0.0,
    throttle: float = 0.0,
    brake: float = 0.0,
    override_reason: Any = None,
) -> dict[str, Any]:
    """Scene labels and the wheel/pedal gate for one live strip.

    ``tech`` is present only when this tick has lanes, tracks, signs, or curbs,
    which is what ``loss_mask`` requires before a step is supervised. A player
    override reason is stored on the wheel or the pedals so that moment is gated.
    """
    tech: dict[str, Any] = {}
    if _has_labels(lanes):
        tech["lanes"] = lanes
    if _has_labels(tracks):
        tech["tracks"] = tracks
    if _has_labels(signs):
        tech["signs"] = signs
    if _has_labels(curbs):
        tech["curbs"] = curbs
    wheel: dict[str, Any] = {"steer": float(steer or 0.0)}
    pedals: dict[str, Any] = {
        "throttle": float(throttle or 0.0),
        "brake": float(brake or 0.0),
    }
    reason = str(override_reason or "").strip().lower()
    if reason in _BAD_REASONS:
        if reason == "player_steer":
            wheel["reason"] = reason
        else:
            pedals["reason"] = reason
    return {"wheel": wheel, "pedals": pedals, "tech": tech or None}
