#!/usr/bin/env python3
"""Opt-in --no-conf-gate: bypass confidence vetoes only.

Default off keeps every confidence veto. The flag lets low lane/path/scene
confidence keep steering. AEB, heartbeat, and missing lane_x still stop.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _path() -> list[dict[str, float]]:
    return [{"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 0.4, "y": 8.0, "z": 0.0}, {"x": 0.8, "y": 16.0, "z": 0.0}]


def _pred(lane_x: float = -0.8, confidence: float = 0.9) -> object:
    from python.train.scene_ckpt_runtime import SceneCkptPred

    return SceneCkptPred(
        lane_gt=np.zeros(14, dtype=np.float32),
        lane_x=lane_x,
        confidence=confidence,
        canvas_shape=(192, 256, 3),
    )


def _pout(**extra):
    base = dict(
        path_ego=[{"x": 3.0, "y": 0.0, "z": 0.0}, {"x": 3.0, "y": 8.0, "z": 0.0}],
        lane_conf=0.99,
        path_conf=0.99,
        lane_source="hough",
        path_debug_preview=True,
        planner={"aeb": "off", "target_v": 25.0, "ttc_lead": None},
    )
    base.update(extra)
    return SimpleNamespace(**base)


def _e2e():
    from python.control.e2e import E2EPolicy

    return E2EPolicy()


def _cfg(*, no_conf_gate: bool = False):
    from python.runtime.shadow import ShadowConfig

    return ShadowConfig(lane_conf_min=0.25, steer_disagree_max=0.55, path_conf_min=0.15, no_conf_gate=no_conf_gate)


def _tick(*, policy: str = "e2e", lane_conf: float = 0.05, path_conf: float = 0.8, heartbeat_ok: bool = True, planner=None, no_conf_gate: bool = False, engaged: bool = True):
    from python.runtime.shadow import shadow_tick

    return shadow_tick(
        policy=policy,
        engaged=engaged,
        heartbeat_ok=heartbeat_ok,
        path_debug_preview=False,
        allow_preview_drive=False,
        path_ego=_path(),
        planner=planner if planner is not None else {"target_v": 10, "aeb": "off"},
        ego_speed_mps=5.0,
        lane_conf=lane_conf,
        path_conf=path_conf,
        seq=7,
        e2e_policy=_e2e(),
        main_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
        wide_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
        cfg=_cfg(no_conf_gate=no_conf_gate),
    )


def check_helper_default_and_flag() -> None:
    from python.runtime.conf_gate import HUD_TAG, STARTUP_WARNING, conf_veto_blocks, hud_tag

    assert conf_veto_blocks(no_conf_gate=False, triggered=True) is True
    assert conf_veto_blocks(no_conf_gate=False, triggered=False) is False
    assert conf_veto_blocks(no_conf_gate=True, triggered=True) is False
    assert conf_veto_blocks(no_conf_gate=True, triggered=False) is False
    assert HUD_TAG == "CONF GATE OFF"
    assert hud_tag(False) == ""
    assert hud_tag(True) == HUD_TAG
    assert "WARNING" in STARTUP_WARNING
    assert "confidence" in STARTUP_WARNING.lower()
    print("helper keyed on the flag; HUD tag and startup warning present OK")


def check_default_keeps_vetoes() -> None:
    from python.runtime.shadow import evaluate_veto
    from python.control.e2e import E2EIntent
    from python.control.actuate import DriveCommand
    from python.train.scene_lane_runtime import scene_lane_decision
    from python.train.scene_ckpt_runtime import apply_scene_ckpt_lane

    cmd = DriveCommand(steer=0.1, throttle=0.2, brake=0.0, seq=1, reason="ok")
    intent = E2EIntent(ok=True, steer=0.0, throttle=0.0, brake=0.0)
    kwargs = dict(
        modular=cmd,
        e2e=intent,
        heartbeat_ok=True,
        path_debug_preview=False,
        allow_preview_drive=False,
        cfg=_cfg(),
        policy="shadow",
        planner={"aeb": "off"},
    )
    assert evaluate_veto(lane_conf=0.05, path_conf=0.8, **kwargs) == "low_lane_conf"
    assert evaluate_veto(lane_conf=0.9, path_conf=0.01, **kwargs) == "low_path_conf"

    tick = _tick(policy="e2e", lane_conf=0.05, no_conf_gate=False)
    assert tick.veto_reason == "low_lane_conf"
    assert tick.applied.reason.startswith("veto:")
    assert tick.should_disengage is True
    assert tick.applied.brake >= 0.99

    tick_m = _tick(policy="modular", lane_conf=0.05, no_conf_gate=False)
    assert tick_m.veto_reason == "low_lane_conf"
    assert tick_m.applied.reason == "veto:low_lane_conf"

    tick_p = _tick(policy="modular", lane_conf=0.9, path_conf=0.01, no_conf_gate=False)
    assert tick_p.veto_reason == "low_path_conf"

    assert scene_lane_decision(0.2, 0.1) == "stop"
    assert scene_lane_decision(0.2, 0.1, no_conf_gate=False) == "stop"

    pout = _pout()
    decision = apply_scene_ckpt_lane(pout, _pred(lane_x=-0.4, confidence=0.10), drive=True)
    assert decision == "stop", decision
    assert pout.path_ego == []
    print("default off: lane/path/scene/ckpt confidence vetoes still fire OK")


def check_flag_bypasses_each_conf_veto() -> None:
    from python.runtime.shadow import evaluate_veto
    from python.control.e2e import E2EIntent
    from python.control.actuate import DriveCommand
    from python.train.scene_lane_runtime import scene_lane_decision
    from python.train.scene_ckpt_runtime import apply_scene_ckpt_lane

    cmd = DriveCommand(steer=0.1, throttle=0.2, brake=0.0, seq=1, reason="ok")
    intent = E2EIntent(ok=True, steer=0.0, throttle=0.0, brake=0.0)
    kwargs = dict(
        modular=cmd,
        e2e=intent,
        heartbeat_ok=True,
        path_debug_preview=False,
        allow_preview_drive=False,
        cfg=_cfg(no_conf_gate=True),
        policy="shadow",
        planner={"aeb": "off"},
    )
    assert evaluate_veto(lane_conf=0.05, path_conf=0.8, **kwargs) == "none"
    assert evaluate_veto(lane_conf=0.9, path_conf=0.01, **kwargs) == "none"

    tick = _tick(policy="shadow", lane_conf=0.05, path_conf=0.8, no_conf_gate=True)
    assert tick.veto_reason == "none", tick.veto_reason
    assert tick.should_disengage is False
    assert tick.applied.reason == "ok", tick.applied
    assert tick.applied.brake < 0.5
    tick_e2e = _tick(policy="e2e", lane_conf=0.05, path_conf=0.8, no_conf_gate=True)
    assert tick_e2e.veto_reason != "low_lane_conf", tick_e2e.veto_reason

    tick_m = _tick(policy="modular", lane_conf=0.05, no_conf_gate=True)
    assert tick_m.veto_reason == "none", tick_m.veto_reason
    assert tick_m.applied.reason == "ok", tick_m.applied

    tick_p = _tick(policy="modular", lane_conf=0.9, path_conf=0.01, no_conf_gate=True)
    assert tick_p.veto_reason == "none", tick_p.veto_reason
    assert tick_p.applied.reason == "ok", tick_p.applied

    assert scene_lane_decision(-0.4, 0.10, no_conf_gate=True) == "lane"
    pout = _pout()
    decision = apply_scene_ckpt_lane(
        pout, _pred(lane_x=-0.4, confidence=0.10), drive=True, no_conf_gate=True
    )
    assert decision == "lane", decision
    assert pout.lane_source == "scene_ckpt"
    assert abs(float(pout.path_ego[0]["x"]) - (-0.4)) < 1e-6
    print("flag on: each confidence veto is bypassed and the model keeps a path OK")


def check_aeb_and_heartbeat_still_stop() -> None:
    from python.runtime.shadow import evaluate_veto
    from python.control.e2e import E2EIntent
    from python.control.actuate import DriveCommand, plan_command
    from python.train.scene_ckpt_runtime import apply_scene_ckpt_lane

    cmd = DriveCommand(steer=0.1, throttle=0.2, brake=0.0, seq=1, reason="ok")
    intent = E2EIntent(ok=True, steer=0.0, throttle=0.0, brake=0.0)
    veto_aeb = evaluate_veto(
        modular=cmd,
        e2e=intent,
        lane_conf=0.05,
        path_conf=0.05,
        heartbeat_ok=True,
        path_debug_preview=False,
        allow_preview_drive=False,
        cfg=_cfg(no_conf_gate=True),
        policy="shadow",
        planner={"aeb": "brake"},
    )
    assert veto_aeb == "aeb_brake", veto_aeb

    tick = _tick(
        policy="e2e",
        lane_conf=0.05,
        heartbeat_ok=False,
        no_conf_gate=True,
    )
    assert tick.veto_reason == "heartbeat_stale"
    assert tick.applied.reason == "heartbeat_stale"
    assert tick.should_disengage is True
    assert float(tick.applied.throttle) == 0.0

    pout = _pout()
    apply_scene_ckpt_lane(pout, _pred(lane_x=-0.9, confidence=0.10), drive=True, no_conf_gate=True)
    pout.planner = {"aeb": "brake", "target_v": 15.0, "ttc_lead": 0.8}
    planned = plan_command(path_ego=pout.path_ego, planner=pout.planner, ego_speed_mps=14.0, seq=7)
    assert planned.brake >= 0.99 and planned.throttle == 0.0, planned
    print("flag on: AEB and heartbeat still stop / release OK")


def check_missing_lane_x_still_stops() -> None:
    from python.train.scene_lane_runtime import scene_lane_decision
    from python.train.scene_ckpt_runtime import apply_scene_ckpt_lane

    assert scene_lane_decision(None, 0.9, no_conf_gate=True) == "stop"
    assert scene_lane_decision(float("nan"), 0.9, no_conf_gate=True) == "stop"
    pout = _pout()
    decision = apply_scene_ckpt_lane(pout, _pred(lane_x=float("nan"), confidence=0.99), drive=True, no_conf_gate=True)
    assert decision == "stop"
    print("flag on: missing / non-finite lane_x still stops OK")


def check_flag_parse() -> None:
    from python.run_vision import build_parser

    parser = build_parser()
    default = parser.parse_args([])
    assert default.no_conf_gate is False
    opted = parser.parse_args(["--no-conf-gate"])
    assert opted.no_conf_gate is True
    print("parser: default off; --no-conf-gate accepted without error OK")


def check_paths_call_helper() -> None:
    """The named veto functions must call conf_veto_blocks (mutation lock)."""

    def _fn_source(path: Path, name: str) -> str:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(path.read_text(encoding="utf-8"), node) or ast.dump(node)
        raise AssertionError(f"{name} not found in {path}")

    shadow = ROOT / "python" / "runtime" / "shadow.py"
    scene = ROOT / "python" / "train" / "scene_lane_runtime.py"
    run = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    ev = _fn_source(shadow, "evaluate_veto")
    tick = _fn_source(shadow, "shadow_tick")
    dec = _fn_source(scene, "scene_lane_decision")
    assert "conf_veto_blocks" in ev, "evaluate_veto must route conf vetoes through conf_veto_blocks"
    assert "conf_veto_blocks" in tick, "shadow_tick must route modular conf vetoes through conf_veto_blocks"
    assert "conf_veto_blocks" in dec, "scene_lane_decision must route the conf floor through conf_veto_blocks"
    assert "--no-conf-gate" in run
    assert "STARTUP_WARNING" in run
    assert "no_conf_gate" in run
    stage = (ROOT / "python" / "viz" / "stage.py").read_text(encoding="utf-8")
    assert "hud_tag" in stage or "CONF GATE OFF" in stage or "HUD_TAG" in stage
    print("AST: veto paths and HUD/startup wiring call the helper OK")


def main() -> None:
    check_helper_default_and_flag()
    check_default_keeps_vetoes()
    check_flag_bypasses_each_conf_veto()
    check_aeb_and_heartbeat_still_stop()
    check_missing_lane_x_still_stops()
    check_flag_parse()
    check_paths_call_helper()
    print("test_no_conf_gate: all OK")


if __name__ == "__main__":
    main()
