"""Turn tracks into multimodal polylines + uncertainty cones (toy CV / CYR baseline)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MAX_MODES = 3
HORIZON_S = 4.0
DT = 0.2


def predict_modes(track: dict[str, Any], max_modes: int = MAX_MODES) -> list[dict[str, Any]]:
    """Constant-velocity / constant-yaw-rate multimodal stub.

    Mode 0: CYR / CV most-likely. Modes 1–2: mild yaw / speed variants.
    """
    x = float(track.get("x", 0.0))
    y = float(track.get("y", 0.0))
    speed = float(track.get("speed_mps", track.get("speed", 0.0)))
    yaw = float(track.get("yaw", track.get("heading", 0.0)))
    yaw_rate = float(track.get("yaw_rate", 0.0))
    steps = int(HORIZON_S / DT)
    variants = [
        (1.0, yaw_rate, speed),
        (0.55, yaw_rate + 0.12, speed * 0.9),
        (0.35, yaw_rate - 0.12, speed * 1.05),
    ][:max_modes]
    modes: list[dict[str, Any]] = []
    for mi, (prob, yr, sp) in enumerate(variants):
        pts = []
        px, py, pyaw = x, y, yaw
        for _ in range(steps + 1):
            pts.append((px, py))
            pyaw += yr * DT
            px += math.cos(pyaw) * sp * DT
            py += math.sin(pyaw) * sp * DT
        modes.append({"mode": mi, "prob": prob, "points": pts})
    return modes


def cone_widths(n: int, base: float = 0.4, growth: float = 0.35) -> np.ndarray:
    t = np.linspace(0.0, HORIZON_S, n)
    return base + growth * t
