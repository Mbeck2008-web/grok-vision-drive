#!/usr/bin/env python3
"""Test-only launcher: ``python/run_vision.py`` with the stub cameras showing a painted road.

Not a test itself (the suite only runs ``test_*``). ``scripts/test_launch_window.py`` starts
the supervisor through this file so the stub backend's main camera shows two painted lane
lines (the ``_road()`` picture from test_unknown_ego) instead of nothing, and the modular
planner has a lane to drive. ``StubBackend.grab`` is the only thing replaced; run_vision and
everything it calls are unchanged.
"""
from __future__ import annotations

import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from python.sensors import cameras  # noqa: E402


def road() -> np.ndarray:
    """Two painted lines: a lane pair with a ~36 m path starting at y=0."""
    img = np.full((96, 160, 3), (40, 42, 44), dtype=np.uint8)
    cv2.line(img, (30, 90), (70, 30), (40, 210, 235), 3)
    cv2.line(img, (130, 90), (90, 30), (245, 245, 245), 3)
    return img


def _grab(self) -> cameras.CameraFrameBundle:  # noqa: ARG001 - replaces StubBackend.grab
    now = time.time()
    health = {cid: cameras.CamHealth.MISSING for cid in cameras.CAM_IDS}
    health["main"] = cameras.CamHealth.OK
    return cameras.CameraFrameBundle(
        frames={"main": road()},
        timestamps={"main": now},
        health=health,
        backend=self.name,
        note="stub: painted road (test launcher)",
        unique_gpu_n=1,
    )


def main() -> None:
    cameras.StubBackend.grab = _grab
    script = ROOT / "python" / "run_vision.py"
    sys.argv = [str(script), *sys.argv[1:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
