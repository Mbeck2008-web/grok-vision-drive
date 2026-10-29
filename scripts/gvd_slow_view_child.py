#!/usr/bin/env python3
"""Test-only launcher: run_vision on the stub road with a slow fake scene-net viewer runtime.

Not a test itself (the suite only runs ``test_*``). ``scripts/test_scene_view.py`` (budget check)
starts run_vision through this file. Like ``gvd_road_stub_child.py`` the stub main camera shows a
painted road. Two more things are replaced, both test-only:

* ``GVD_SLOW_VIEW_MODE``: ``sleep`` or ``spin`` swaps the viewer's runtime factory for a fake
  whose forward takes 400 ms, sleeping (releases the GIL) or spinning in Python (holds it).
  ``off`` (default) leaves everything alone.
* ``GVD_SLOW_VIEW_LOG``: a file that gets one ``time.monotonic()`` line per gvd_cmd.json write.
* The detector is the synthetic one in every mode (off included), so the tick measures the loop
  and the viewer, not CPU YOLO on a shared runner (~70 ms there, already over the 66.7 ms budget).

run_vision and everything it calls are otherwise unchanged.
"""
from __future__ import annotations

import os
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

import gvd_road_stub_child as road_stub  # noqa: E402
from python.control import actuate  # noqa: E402
from python.perception import detect  # noqa: E402
from python.sensors import cameras  # noqa: E402
from python.train.cam_strip import canvas_width  # noqa: E402
from python.viz import scene_view  # noqa: E402

FORWARD_S = 0.4


class SlowRuntime:
    """SceneLaneRuntime stand-in: a main-only lane_gt model whose forward takes FORWARD_S."""

    cams = ("main",)
    canvas_w = canvas_width(("main",))
    cam_age = False
    ready = True
    reason = "ok"
    untrained = ()

    def __init__(self, path=None, *, feeds_cam_age=False, intra_op_threads=None, mode="sleep") -> None:
        self.path = path
        self.mode = mode
        self.last_timers: dict = {}

    def observe(self, strip, dt_s, ages=None):
        if self.mode == "spin":
            end = time.perf_counter() + FORWARD_S
            n = 0
            while time.perf_counter() < end:
                n += 1  # pure Python: holds the GIL between switch intervals
        else:
            time.sleep(FORWARD_S)
        self.last_timers = {"scene_ms": FORWARD_S * 1000.0}
        return 0.0, 0.9

    def output(self, name):
        vec = np.zeros(14, dtype=np.float32)
        vec[13] = 0.9
        return vec

    def reset_state(self) -> None:
        pass


def main() -> None:
    cameras.StubBackend.grab = road_stub._grab
    detect.make_detector = lambda *, allow_synthetic=True: (detect.SyntheticDetector(), ["yolo_weights"])
    mode = os.environ.get("GVD_SLOW_VIEW_MODE", "off")
    if mode in ("sleep", "spin"):
        scene_view.DEFAULT_RUNTIME_FACTORY = lambda path, **kw: SlowRuntime(path, mode=mode, **kw)
    log = os.environ.get("GVD_SLOW_VIEW_LOG")
    if log:
        real = actuate.atomic_write_json
        target = actuate.cmd_path()

        def logged(path, payload, *args, **kwargs):
            ok = real(path, payload, *args, **kwargs)
            if Path(path) == target:
                with open(log, "a", encoding="utf-8") as fh:
                    fh.write(f"{time.monotonic():.6f}\n")
            return ok

        actuate.atomic_write_json = logged
    script = ROOT / "python" / "run_vision.py"
    sys.argv = [str(script), *sys.argv[1:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
