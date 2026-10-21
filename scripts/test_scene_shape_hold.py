#!/usr/bin/env python3
"""Supervisor handling of SceneInputShapeError (PR 87 review condition).

A wrong-size live strip mid-drive is a HOLD: the error is caught, the command
is the same ``stop_command`` (steer 0, throttle 0, brake 1) the other gate
holds use, the reason is ``scene_hold:input_shape`` on the glance word and in
the ``scene_lane`` block of gvd_state.json, and nothing escapes the loop. A
wrong-size graph at load refuses Engage. No BeamNG, no bus files; the real-
graph check writes only to a temp dir.

    py -3 scripts/test_scene_shape_hold.py
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gvd_test_env import skip_check  # noqa: E402

from python.control.actuate import DriveCommand, stop_command  # noqa: E402
from python.train.scene_lane_runtime import (  # noqa: E402
    SCENE_SHAPE_HOLD,
    SceneInputShapeError,
    SceneLaneGuard,
)
from python.viz.debug_draw import cabin_drive_word  # noqa: E402

NEUTRAL = (0.0, 0.0, 1.0)  # stop_command: steer, throttle, brake


class FakeRuntime:
    """Drives a clean lane for ``good`` ticks, then the strip is the wrong size."""

    def __init__(self, good: int) -> None:
        self.good = good
        self.calls = 0

    def observe(self, strip, dt_s):
        self.calls += 1
        if self.calls > self.good:
            raise SceneInputShapeError("live strip (96, 128, 3) does not give the model canvas 192x1024x3")
        return 0.1, 0.9


class RecordingActuator:
    """Same contract as the real actuators: stop() sends stop_command."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, DriveCommand]] = []

    def apply(self, cmd: DriveCommand) -> DriveCommand:
        self.sent.append(("apply", cmd))
        return cmd

    def stop(self, seq: int = 0, reason: str = "stop") -> DriveCommand:
        cmd = stop_command(seq=seq, reason=reason)
        self.sent.append(("stop", cmd))
        return cmd


def _tick(guard: SceneLaneGuard, runtime, actuator: RecordingActuator, seq: int, engaged: bool) -> dict:
    """The scene branch and dispatch of run_vision.main, one tick."""
    lane_hat, conf = guard.observe(runtime, object(), 0.05, engaged=engaged)
    scene_hold = guard.hold is not None
    if scene_hold:
        cmd = stop_command(seq=seq, reason=SCENE_SHAPE_HOLD)
    elif lane_hat is None:
        cmd = stop_command(seq=seq, reason="scene_lane_low")
    else:
        cmd = DriveCommand(steer=0.2, throttle=0.3, brake=0.0, seq=seq, reason="ok")
    holding = False  # the lane-hold branch needs cmd.reason ok / preview_blocked
    if engaged and (cmd.reason == "ok" or holding):
        applied = actuator.apply(cmd)
    else:
        applied = actuator.stop(seq=seq, reason=cmd.reason)
    disengage_reason = "scene_input_shape" if (engaged and scene_hold) else "none"
    return {
        "engaged": engaged,
        "disengage_reason": disengage_reason,
        "cmd_reason": applied.reason,
        "cmd_applied": False,
        "scene_lane": guard.status(),
        "_applied": applied,
    }


def check_mid_drive_hold() -> None:
    guard = SceneLaneGuard()
    runtime = FakeRuntime(good=3)
    act = RecordingActuator()
    states = []
    for seq in range(1, 9):
        states.append(_tick(guard, runtime, act, seq, engaged=True))  # must not raise
    # Ticks 1-3 drive, the error on tick 4 is a HOLD, and the HOLD stays.
    assert [k for k, _ in act.sent[:3]] == ["apply"] * 3, act.sent
    assert [k for k, _ in act.sent[3:]] == ["stop"] * 5, act.sent
    for _kind, cmd in act.sent[3:]:
        assert (cmd.steer, cmd.throttle, cmd.brake) == NEUTRAL, cmd
        assert cmd.reason == SCENE_SHAPE_HOLD, cmd
    ref = stop_command(seq=0, reason="bus_mismatch")  # the existing gate-hold command
    assert (ref.steer, ref.throttle, ref.brake) == NEUTRAL
    assert runtime.calls == 4, runtime.calls  # latched: the bad runtime is not asked again
    st = states[-1]
    assert st["engaged"] is True and st["disengage_reason"] == "scene_input_shape", st
    assert st["scene_lane"]["state"] == "hold" and st["scene_lane"]["reason"] == SCENE_SHAPE_HOLD, st
    assert "does not give the model canvas" in st["scene_lane"]["detail"], st
    assert cabin_drive_word(st)[0] == "HOLD", cabin_drive_word(st)
    # Only a disengage clears the latch. Once the strip is right again, a fresh
    # engage asks the runtime again and drives.
    runtime.good = 100
    _tick(guard, runtime, act, 9, engaged=True)
    assert act.sent[-1][0] == "stop" and runtime.calls == 4, act.sent[-1]
    _tick(guard, runtime, act, 10, engaged=False)
    assert guard.hold is None and runtime.calls == 5 and act.sent[-1][0] == "stop"
    st = _tick(guard, runtime, act, 11, engaged=True)
    assert act.sent[-1][0] == "apply" and st["scene_lane"]["state"] == "ok", st
    print(f"mid-drive: tick 4 raised, ticks 4-8 stop {NEUTRAL} reason={SCENE_SHAPE_HOLD}, glance HOLD, no exception OK")


def check_load_refuses_engage() -> None:
    def factory(path):
        raise SceneInputShapeError(f"{path}: strip input is [1, 3, 96, 512], the live canvas is [1, 3, 192, 1024]")

    guard = SceneLaneGuard()
    runtime = guard.load(factory, "bad.onnx")
    assert runtime is None and guard.refused and "the live canvas is" in guard.refused, guard.refused
    st = guard.status()
    assert st["state"] == "refused" and st["reason"] == SCENE_SHAPE_HOLD and "bad.onnx" in st["detail"], st
    # Anything else is not swallowed.
    guard2 = SceneLaneGuard()
    try:
        guard2.load(lambda p: (_ for _ in ()).throw(RuntimeError("boom")), None)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a non-shape error was swallowed")
    print("load: SceneInputShapeError -> no runtime, refused with the graph and canvas sizes OK")


def check_real_graph_refused() -> None:
    """The real loader on a half-canvas export, through the guard."""
    from python.train.pretrained_net import build_scene_net
    from python.train.scene_lane_runtime import SceneLaneRuntime
    from python.train.scene_net import export_scene_onnx

    with tempfile.TemporaryDirectory() as tmp:
        net = build_scene_net(("main",), 0.5, "mobilenetv3", pretrained=False).eval()
        path = Path(tmp) / "half.onnx"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            export_scene_onnx(path, net, cams=("main",), heads=("lane_gt",))
        guard = SceneLaneGuard()
        assert guard.load(SceneLaneRuntime, path) is None
        assert guard.refused and "96" in guard.refused and "fixed dims" in guard.refused, guard.refused
    print("load: real 96-row graph refused through SceneLaneGuard OK")


def check_supervisor_wiring() -> None:
    """run_vision.main uses the guard the way _tick does."""
    rv = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    # The runtime goes through the guard; feeds_cam_age says whether this backend
    # can feed the 8 hold ages (scheduled BeamNG companions), else a cam-age model is refused.
    assert re.search(
        r"scene_lane = scene_guard\.load\(\s*functools\.partial\(SceneLaneRuntime, feeds_cam_age=feeds_cam_age\), "
        r"args\.scene_onnx or None\s*\)",
        rv,
    )
    assert 'feeds_cam_age = bool(getattr(backend, "can_feed_cam_age", False))' in rv
    assert "scene_lane.observe(" not in rv, "a bare observe call can raise out of the loop"
    assert "scene_guard.observe(scene_lane, strip, tick_dt, engaged=engaged, ages=tick_ages)" in rv
    assert "cmd = stop_command(seq=cmd_seq, reason=SCENE_SHAPE_HOLD)" in rv
    assert 'st["scene_lane"] = scene_guard.status()' in rv
    assert "if engaged and (cmd.reason == \"ok\" or holding):\n                applied = actuator.apply(cmd)" in rv.replace("\r\n", "\n")
    # The HOLD is checked before the shadow veto, so a veto cannot turn it into a release.
    i_hold = rv.index("elif scene_hold:")
    i_veto = rv.index("elif tick.should_disengage:")
    assert i_hold < i_veto
    # Load refusal forces engaged False every tick and names the reason.
    assert re.search(r"if scene_refuse:\s+#[^\n]*\n\s+#[^\n]*\n\s+engaged = False", rv.replace("\r\n", "\n"))
    assert 'write_engage_flag(False, disengage_reason="scene_input_shape")' in rv
    lua = (ROOT / "beamng_mod" / "lua" / "ge" / "extensions" / "gvd" / "main.lua").read_text(encoding="utf-8")
    js = (ROOT / "beamng_mod" / "ui" / "modules" / "apps" / "GVD" / "app.js").read_text(encoding="utf-8")
    assert "reason:sub(1, 11) == 'scene_hold:'" in lua
    assert "reason.indexOf('scene_hold:') === 0" in js
    assert len("scene_hold:") == 11
    print("wiring: run_vision load/observe/stop/status, HOLD before veto, in-game HOLD word OK")


def main() -> None:
    import importlib.util

    check_mid_drive_hold()
    check_load_refuses_engage()
    check_supervisor_wiring()
    if importlib.util.find_spec("torch") is None:
        skip_check("real_graph_refused", "torch missing (it builds and exports a net)")
    else:
        check_real_graph_refused()
    print("test_scene_shape_hold: OK")


if __name__ == "__main__":
    main()