"""Trainer 2.0 wide-camera lane points through the live near-range gate.

docs/specs/trainer_2.md, "Wide near-range gating" (option a): the trainer adds
no wide range or speed rule of its own. Wide-derived lane points come from the
same ``lanes_from_view`` -> ``estimate_lanes`` -> ``_estimate_span_lanes``
path the live planner uses, whose default ``gate_wide=True`` drops wide points
past ``WIDE_NEAR_MAX_Y_M`` ahead or ``WIDE_NEAR_MAX_X_M`` to the side. The two
constants are imported here, never redefined
(``scripts/test_trainer2_wide_gate.py``).

Training-only module. Nothing on a live path imports it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from python.perception.lanes import WIDE_NEAR_MAX_X_M, WIDE_NEAR_MAX_Y_M, lanes_from_view

__all__ = ["WIDE_NEAR_MAX_X_M", "WIDE_NEAR_MAX_Y_M", "wide_gate", "wide_lane_points"]


def wide_gate() -> tuple[float, float]:
    """(max ahead m, max side m) of the live wide gate, for the run manifest."""
    return (WIDE_NEAR_MAX_Y_M, WIDE_NEAR_MAX_X_M)


def wide_lane_points(stitch_bgr: np.ndarray) -> list[list[dict[str, Any]]]:
    """Wide-camera lane polylines of a stitched strip, gated exactly as live.

    Calls ``lanes_from_view(None, stitch_bgr)`` (the live call) and keeps the
    points tagged ``src == "wide"``, in fit order.
    """
    out: list[list[dict[str, Any]]] = []
    for poly in lanes_from_view(None, stitch_bgr).lanes_bev:
        wide = [dict(pt) for pt in poly if pt.get("src") == "wide"]
        if wide:
            out.append(wide)
    return out
