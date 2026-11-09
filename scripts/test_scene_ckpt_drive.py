#!/usr/bin/env python3
"""Opt-in --scene-ckpt-drive: ckpt lane_x steers through the modular controller.

Shadow (--scene-ckpt alone) stays the default and still cannot engage.
Drive keeps heartbeat, AEB/CIPV, and the confidence veto in the loop.
"""
# gvd-ci: requires torch
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


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


def check_flag_off_by_default() -> None:
    from python.train.scene_ckpt_runtime import (
        apply_scene_ckpt_lane,
        apply_scene_ckpt_to_command,
        shadow_engage_allowed,
    )
    from python.control.actuate import DriveCommand

    assert shadow_engage_allowed() is False
    assert shadow_engage_allowed(drive=False) is False
    cmd = DriveCommand(steer=0.31, throttle=0.4, brake=0.0, seq=3, reason="lane")
    after = apply_scene_ckpt_to_command(cmd, _pred())
    assert after.steer == cmd.steer and after.throttle == cmd.throttle and after.brake == cmd.brake

    pout = _pout()
    before = list(pout.path_ego)
    decision = apply_scene_ckpt_lane(pout, _pred(lane_x=-1.25, confidence=0.92), drive=False)
    assert decision == "shadow"
    assert pout.path_ego == before
    assert pout.lane_source == "hough"
    print("flag off by default: engage blocked, command and path unchanged OK")


def check_drive_feeds_modular_path() -> None:
    from python.train.scene_ckpt_runtime import apply_scene_ckpt_lane, cap_drive_speed, scene_ckpt_as_state

    pout = _pout()
    pred = _pred(lane_x=-1.25, confidence=0.88)
    decision = apply_scene_ckpt_lane(pout, pred, drive=True)
    assert decision == "lane", decision
    assert pout.lane_source == "scene_ckpt"
    assert abs(float(pout.path_ego[8]["x"]) - (-1.25)) < 1e-6
    assert abs(float(pout.lane_conf) - 0.88) < 1e-6
    assert pout.path_debug_preview is False
    cap_drive_speed(pout, 15.0)
    assert float(pout.planner["target_v"]) == 15.0
    st = scene_ckpt_as_state(pred, drive=True, engaged=True, steer=0.22)
    assert st["shadow"] is False and st["drive"] is True
    assert st["lane_x"] == -1.25 and abs(st["confidence"] - 0.88) < 1e-6
    assert st["engaged"] is True and abs(float(st["steer"]) - 0.22) < 1e-6
    print("drive feeds modular path from lane_x and caps target_v OK")


def check_low_conf_and_aeb_override() -> None:
    from python.control.actuate import plan_command
    from python.runtime.shadow import ShadowConfig, evaluate_veto
    from python.train.scene_ckpt_runtime import apply_scene_ckpt_lane

    pout = _pout()
    decision = apply_scene_ckpt_lane(pout, _pred(lane_x=-0.4, confidence=0.10), drive=True)
    assert decision == "stop", decision
    assert pout.lane_source == "scene_ckpt/stop"
    assert pout.path_ego == []

    # A confident path still loses to AEB: plan_command must brake, not chase target_v.
    pout = _pout()
    apply_scene_ckpt_lane(pout, _pred(lane_x=-0.9, confidence=0.91), drive=True)
    pout.planner = {"aeb": "brake", "target_v": 15.0, "ttc_lead": 0.8}
    cmd = plan_command(path_ego=pout.path_ego, planner=pout.planner, ego_speed_mps=14.0, seq=7)
    assert cmd.brake >= 0.99 and cmd.throttle == 0.0, cmd

    from python.control.e2e import E2EIntent

    veto = evaluate_veto(
        modular=cmd,
        e2e=E2EIntent(ok=True, steer=0.0, throttle=0.0, brake=0.0),
        lane_conf=0.10,
        path_conf=0.10,
        heartbeat_ok=True,
        path_debug_preview=False,
        allow_preview_drive=False,
        cfg=ShadowConfig(),
        policy="shadow",
        planner={"aeb": "off"},
    )
    assert veto == "low_lane_conf", veto
    veto_aeb = evaluate_veto(
        modular=cmd,
        e2e=E2EIntent(ok=True, steer=0.0, throttle=0.0, brake=0.0),
        lane_conf=0.91,
        path_conf=0.91,
        heartbeat_ok=True,
        path_debug_preview=False,
        allow_preview_drive=False,
        cfg=ShadowConfig(),
        policy="shadow",
        planner={"aeb": "brake"},
    )
    assert veto_aeb == "aeb_brake", veto_aeb
    print("low confidence stops; AEB and veto still override drive path OK")


def _write_ckpt(tmp: Path, net):
    from python.train.epoch_ckpt import epoch_ckpt_path, write_epoch_ckpt

    cams = list(getattr(net, "cams", ()) or ("main",))
    meta = {
        "epoch": 0,
        "step": 1,
        "cams": cams,
        "scale": float(getattr(net, "scale", 1.0) or 1.0),
        "backbone": "scratch",
        "heads": ["lane_gt"],
        "cam_age": False,
        "ctl": False,
        "arch": getattr(net, "pause_arch", None),
    }
    weights = {key: value.detach().cpu().numpy() for key, value in net.state_dict().items()}
    path = epoch_ckpt_path(tmp, tmp / "scene_lanes", "20261107T011144Z", 0, 1)
    write_epoch_ckpt(path, weights, meta, min_free_bytes=0)
    return path


def check_supervisor_shadow_default_and_drive_opt_in() -> None:
    import tempfile

    import torch

    from python.train.scene_net import SceneNet
    from test_live_default_path import run_supervisor

    torch.manual_seed(5)
    net = SceneNet(cams=("main",)).cpu().eval()
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_ckpt(Path(tmp), net)
        text, states = run_supervisor(
            ["--scene-ckpt", str(path), "--force-engage", "--run-seconds", "2"],
            seconds=2.0,
        )
        assert states and all(not st.get("engaged") for st in states), [st.get("engaged") for st in states[:5]]
        for st in states:
            ckpt = st.get("scene_ckpt") or {}
            if ckpt:
                assert ckpt.get("shadow") is True
                assert ckpt.get("drive") in (None, False)

        text_d, states_d = run_supervisor(
            [
                "--scene-ckpt",
                str(path),
                "--scene-ckpt-drive",
                "--force-engage",
                "--run-seconds",
                "2",
            ],
            seconds=2.0,
            echo=True,
        )
    assert "scene-ckpt" in text_d.lower() or "DRIVE" in text_d, text_d[-2000:]
    assert states_d, text_d[-2000:]
    assert any(st.get("engaged") for st in states_d), (
        "drive + --force-engage with a session echo should be allowed to engage",
        [st.get("engaged") for st in states_d[:8]],
        text_d[-2000:],
    )
    ckpt_rows = [st.get("scene_ckpt") or {} for st in states_d if st.get("scene_ckpt")]
    assert ckpt_rows, "drive run wrote no scene_ckpt fields"
    hit = ckpt_rows[-1]
    assert hit.get("drive") is True
    assert hit.get("shadow") is False
    assert "lane_x" in hit and "confidence" in hit
    assert "engaged" in hit and "steer" in hit
    print(
        f"supervisor: shadow default engaged=0; drive opt-in engaged "
        f"on {sum(1 for st in states_d if st.get('engaged'))}/{len(states_d)} ticks OK"
    )


def check_drive_without_ckpt_refused() -> None:
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-u", str(ROOT / "python" / "run_vision.py"), "--backend", "stub", "--scene-ckpt-drive"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0, proc.stdout[-1500:] + proc.stderr[-1500:]
    blob = (proc.stdout + proc.stderr).lower()
    assert "scene-ckpt" in blob and "drive" in blob, blob[-1500:]
    print("--scene-ckpt-drive without --scene-ckpt refused at startup OK")


def check_multicam_ckpt_refuses_drive() -> None:
    """A cams!=('main',) ckpt must raise at load; drive must not steer on it."""
    import tempfile

    import torch

    from python.train.epoch_ckpt import epoch_ckpt_path, write_epoch_ckpt
    from python.train.scene_ckpt_runtime import (
        SceneCkptInputError,
        SceneCkptRuntime,
        apply_scene_ckpt_lane,
    )
    from python.train.scene_net import SceneNet

    torch.manual_seed(11)
    net = SceneNet(cams=("main", "wide")).cpu().eval()
    meta = {
        "epoch": 0,
        "step": 1,
        "cams": ["main", "wide"],
        "scale": 1.0,
        "backbone": "scratch",
        "heads": ["lane_gt"],
        "cam_age": False,
        "ctl": False,
        "arch": getattr(net, "pause_arch", None),
    }
    weights = {key: value.detach().cpu().numpy() for key, value in net.state_dict().items()}
    with tempfile.TemporaryDirectory() as tmp:
        path = epoch_ckpt_path(Path(tmp), Path(tmp) / "scene_lanes", "20261109T000000Z", 0, 1)
        write_epoch_ckpt(path, weights, meta, min_free_bytes=0)
        try:
            SceneCkptRuntime(path)
        except SceneCkptInputError as exc:
            msg = str(exc).lower()
            assert "main" in msg and ("black" in msg or "camera" in msg or "cams" in msg), exc
        else:
            raise AssertionError("multi-cam ckpt loaded; drive would steer on black sectors")
    # Even a forged multi-cam pred must not be treated as a usable drive lane
    # if decision somehow ran; the load refuse above is the gate of record.
    pout = _pout()
    # No runtime means no observe → None pred → stop under drive.
    decision = apply_scene_ckpt_lane(pout, None, drive=True)
    assert decision == "stop"
    assert pout.path_ego == []
    print("multi-cam ckpt refuses load; drive does not steer OK")


def check_nonfinite_lane_x_stops() -> None:
    """±inf lane_x (alone or from a finite/inf vector) must stop, not full-lock."""
    import math

    from python.debug.lane_targets import LANE_GT_DIM, lane_x_from_vector
    from python.train.scene_ckpt_runtime import apply_scene_ckpt_lane
    from python.train.scene_lane_runtime import scene_lane_decision

    assert scene_lane_decision(float("inf"), 0.99) == "stop"
    assert scene_lane_decision(float("-inf"), 0.99) == "stop"
    assert scene_lane_decision(float("nan"), 0.99) == "stop"

    for raw in (float("inf"), float("-inf")):
        pout = _pout()
        decision = apply_scene_ckpt_lane(pout, _pred(lane_x=raw, confidence=0.99), drive=True)
        assert decision == "stop", (raw, decision)
        assert pout.path_ego == []
        assert pout.lane_source == "scene_ckpt/stop"

    vec = __import__("numpy").zeros(LANE_GT_DIM, dtype="float32")
    vec[0] = 0.5  # 5 m center finite
    vec[3] = float("inf")  # 10 m center inf → 8 m interp is inf
    vec[-1] = 0.95
    mixed = lane_x_from_vector(vec)
    assert mixed is not None and not math.isfinite(float(mixed)), mixed
    pout = _pout()
    decision = apply_scene_ckpt_lane(pout, _pred(lane_x=float(mixed), confidence=0.95), drive=True)
    assert decision == "stop", decision
    assert pout.path_ego == []
    print("non-finite lane_x (inf alone and 5 m finite / 10 m inf) stops OK")


def check_steer_sign_locked_to_lane_x() -> None:
    """Flipping path_steer_from_ego's sign must fail this test."""
    from python.control.actuate import path_steer_from_ego, plan_command
    from python.train.scene_ckpt_runtime import apply_scene_ckpt_lane

    pout = _pout()
    apply_scene_ckpt_lane(pout, _pred(lane_x=-1.25, confidence=0.9), drive=True)
    steer = path_steer_from_ego(pout.path_ego)
    assert steer < 0.0, f"lane_x=-1.25 (left) must steer left, got {steer}"
    assert abs(float(pout.path_ego[8]["x"]) - (-1.25)) < 1e-9
    cmd = plan_command(path_ego=pout.path_ego, planner={"aeb": "off", "target_v": 10.0}, ego_speed_mps=8.0, seq=1)
    assert cmd.steer < 0.0, cmd

    pout_r = _pout()
    apply_scene_ckpt_lane(pout_r, _pred(lane_x=1.25, confidence=0.9), drive=True)
    steer_r = path_steer_from_ego(pout_r.path_ego)
    assert steer_r > 0.0, f"lane_x=+1.25 (right) must steer right, got {steer_r}"
    assert abs(steer + steer_r) < 1e-6, (steer, steer_r)
    print(f"steer sign locked: lane_x=-1.25 -> {steer:.4f}; +1.25 -> {steer_r:.4f} OK")


def check_modular_low_conf_and_stale_no_ckpt_steer() -> None:
    """Drive under live modular policy: low conf / stale must not ckpt-steer.

    Also asserts shadow_tick still calls evaluate_veto (mutation: removing that
    call from the tick path must fail this test).
    """
    from unittest import mock

    from python.control.e2e import E2EPolicy
    from python.runtime.shadow import ShadowConfig, evaluate_veto, shadow_tick
    from python.train.scene_ckpt_runtime import apply_scene_ckpt_lane

    # Low confidence: apply_scene_ckpt_lane stops before shadow_tick; no path to chase.
    pout = _pout()
    decision = apply_scene_ckpt_lane(pout, _pred(lane_x=-0.8, confidence=0.10), drive=True)
    assert decision == "stop"
    assert pout.path_ego == []
    assert pout.lane_source == "scene_ckpt/stop"

    calls: list[dict] = []
    real_veto = evaluate_veto

    def _spy_veto(**kwargs):
        calls.append(dict(kwargs))
        return real_veto(**kwargs)

    e2e = E2EPolicy()
    with mock.patch("python.runtime.shadow.evaluate_veto", side_effect=_spy_veto):
        tick_low = shadow_tick(
            policy="modular",
            engaged=True,
            heartbeat_ok=True,
            path_debug_preview=False,
            allow_preview_drive=False,
            path_ego=pout.path_ego,
            planner={"aeb": "off", "target_v": 15.0},
            ego_speed_mps=8.0,
            lane_conf=float(pout.lane_conf),
            path_conf=float(pout.path_conf),
            seq=11,
            e2e_policy=e2e,
            cfg=ShadowConfig(),
        )
    assert calls, "shadow_tick must call evaluate_veto under modular policy"
    assert float(calls[0]["lane_conf"]) < 0.40
    # Empty path → modular steer 0; ckpt_stop in run_vision would replace with stop.
    assert abs(float(tick_low.applied.steer)) < 1e-9, tick_low.applied
    assert abs(float(tick_low.modular.steer)) < 1e-9, tick_low.modular

    # Stale heartbeat: release, no throttle; evaluate_veto still invoked.
    calls.clear()
    pout_ok = _pout()
    apply_scene_ckpt_lane(pout_ok, _pred(lane_x=-0.8, confidence=0.91), drive=True)
    with mock.patch("python.runtime.shadow.evaluate_veto", side_effect=_spy_veto):
        tick_stale = shadow_tick(
            policy="modular",
            engaged=True,
            heartbeat_ok=False,
            path_debug_preview=False,
            allow_preview_drive=False,
            path_ego=pout_ok.path_ego,
            planner={"aeb": "off", "target_v": 15.0},
            ego_speed_mps=8.0,
            lane_conf=float(pout_ok.lane_conf),
            path_conf=float(pout_ok.path_conf),
            seq=12,
            e2e_policy=e2e,
            cfg=ShadowConfig(),
        )
    assert calls, "shadow_tick must call evaluate_veto on the stale path too"
    assert tick_stale.veto_reason == "heartbeat_stale"
    assert float(tick_stale.applied.throttle) == 0.0
    assert tick_stale.should_disengage is True
    # Applied must not keep chasing the ckpt lane after a stale heartbeat.
    assert tick_stale.applied.reason == "heartbeat_stale", tick_stale.applied
    print("modular drive: low-conf and stale yield no ckpt steer; evaluate_veto called OK")


def check_same_tick_observe_documented() -> None:
    """Pred is computed on this tick's main frame (no cached / aged pred).

    Code pointer: python/run_vision.py observe(main, tick_dt) then
    apply_scene_ckpt_lane in the same loop body. No pred-age/frame-id gate
    is required while observe always uses the current frame.
    """
    src = (ROOT / "python" / "run_vision.py").read_text()
    assert "ckpt_pred = scene_ckpt.observe(main, tick_dt)" in src
    assert "apply_scene_ckpt_lane(pout, ckpt_pred, drive=True)" in src
    # No last-pred reuse cache in the runtime.
    rt = (ROOT / "python" / "train" / "scene_ckpt_runtime.py").read_text()
    assert "self._last_pred" not in rt
    assert "pred_age" not in rt
    print("same-tick observe documented (no pred-age cache) OK")


def main() -> None:
    check_flag_off_by_default()
    check_drive_feeds_modular_path()
    check_low_conf_and_aeb_override()
    check_drive_without_ckpt_refused()
    check_multicam_ckpt_refuses_drive()
    check_nonfinite_lane_x_stops()
    check_steer_sign_locked_to_lane_x()
    check_modular_low_conf_and_stale_no_ckpt_steer()
    check_same_tick_observe_documented()
    check_supervisor_shadow_default_and_drive_opt_in()
    print("test_scene_ckpt_drive: all OK")


if __name__ == "__main__":
    main()
