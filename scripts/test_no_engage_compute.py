#!/usr/bin/env python3
"""With --no-engage the supervisor still runs everything: detector, planner / AEB, scene net.

A stub supervisor run with ``--force-engage --no-engage --cam-age-baseline``
and a small non-age scene model logs one ``[GVD] cam_age`` line per tick.
Every tick is engaged=0, and every tick has the planner (``plan_ms``,
``aeb``) and scene net (``scene_ms``) timers > 0 and the detector timer
(``detect_ms``) logged on every tick (> 0 whenever there is a main frame;
the stub backend has none): the check scripts/live_age_capture.py's
``compute`` bar makes.
Engage gates actuation only.
"""
# gvd-ci: requires torch
from __future__ import annotations

import io
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from live_age_capture import compute_gaps, parse_log  # noqa: E402
from test_live_default_path import run_supervisor  # noqa: E402


def main() -> None:
    import torch

    from python.train.scene_net import SceneNet, export_scene_onnx

    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "scene_main.onnx"
        torch.manual_seed(0)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            export_scene_onnx(model, SceneNet(cams=("main",)), cams=("main",), heads=("lane_gt",))
        text, states = run_supervisor(
            ["--lane-source", "scene", "--scene-onnx", str(model), "--cam-age-baseline", "--force-engage", "--no-engage"],
            seconds=3.0,
        )
    assert "engage HARD-disabled" in text and "cam-age: baseline" in text, text[-2000:]
    log = parse_log(text.splitlines())
    ticks = log["ticks"]
    assert len(ticks) >= 10, (len(ticks), text[-2000:])
    assert log["model"] is not None and log["model"].get("hold") == "off"
    assert "detector" in log["providers"] and "scene" in log["providers"], log["providers"]
    assert all(t["engaged"] == "0" for t in ticks)
    bad = compute_gaps(ticks)
    assert not bad, bad[:3]
    # The stub backend has no camera frames: the detector is called every tick (detect_ms
    # logged) and returns at once on no picture. Detection on a real frame is covered by
    # test_live_hold.check_perception_timers and by the capture's compute bar.
    assert all(t["detect_ms"] is not None for t in ticks)
    with_main = [t for t in ticks if t["main_luma"] is not None]
    assert states and not any(st.get("engaged") for st in states)
    print(
        f"no-engage: {len(ticks)} ticks engaged=0 with --force-engage; planner+AEB / scene timers > 0 and the detector "
        f"called on every tick ({len(with_main)} with a stub main frame; detector {log['providers']['detector']}) OK"
    )
    print("test_no_engage_compute: all OK")


if __name__ == "__main__":
    main()
