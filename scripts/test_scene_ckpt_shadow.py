#!/usr/bin/env python3
"""Shadow-only live load of a train_scene epoch checkpoint.

* npz → SceneNet round-trip matches train_scene forward on a fixture frame.
* main-camera shape and channel checks (live_input_long_side 640).
* --scene-ckpt cannot actuate, even with Engage / --force-engage pressed.
* the overlay renders on a stub camera frame.
"""
# gvd-ci: requires torch
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _fixture_main(seed: int = 0, h: int = 480, w: int = 640) -> np.ndarray:
    """A live-shaped main frame: long side 640, 4:3, BGR uint8."""
    rng = np.random.RandomState(seed)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def _write_ckpt(tmp: Path, net, meta_extra: dict | None = None, epoch: int = 0, step: int = 1) -> Path:
    from python.train.epoch_ckpt import epoch_ckpt_path, write_epoch_ckpt

    cams = list(getattr(net, "cams", ()) or ("main",))
    meta = {
        "epoch": epoch,
        "step": step,
        "cams": cams,
        "scale": float(getattr(net, "scale", 1.0) or 1.0),
        "backbone": "scratch",
        "heads": ["lane_gt"],
        "cam_age": bool(getattr(net, "cam_age", False)),
        "ctl": bool(getattr(net, "ctl", False)),
        "arch": getattr(net, "pause_arch", None),
    }
    if meta_extra:
        meta.update(meta_extra)
    weights = {key: value.detach().cpu().numpy() for key, value in net.state_dict().items()}
    path = epoch_ckpt_path(tmp, tmp / "scene_lanes", "20261107T011144Z", epoch, step)
    write_epoch_ckpt(path, weights, meta, min_free_bytes=0)
    return path


def _teacher_forward(net, frame: np.ndarray, dt: float = 0.1):
    import torch

    from python.data.strip_writer import compose_strip
    from python.train.cam_strip import extract_cams

    strip = compose_strip({"main": frame})
    canvas = extract_cams(strip, tuple(net.cams) or ("main",))
    net.eval()
    with torch.no_grad():
        pred, state = net.forward(canvas, dt, net.initial_state(device="cpu"))
    return pred, state, canvas


def check_loader_round_trip() -> None:
    import torch

    from python.train.scene_ckpt_runtime import SceneCkptRuntime
    from python.train.scene_net import SceneNet

    torch.manual_seed(7)
    net = SceneNet(cams=("main",)).cpu().eval()
    frame = _fixture_main()
    pred, _state, canvas = _teacher_forward(net, frame, 0.1)
    assert canvas.shape == (192, 256, 3), canvas.shape
    teacher = pred.lane_gt.detach().cpu().numpy().reshape(-1)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_ckpt(Path(tmp), net)
        runtime = SceneCkptRuntime(path)
        out = runtime.observe(frame, 0.1)
    got = np.asarray(out.lane_gt, dtype=np.float32).reshape(-1)
    assert got.shape == teacher.shape, (got.shape, teacher.shape)
    if not np.allclose(got, teacher, atol=1e-5, rtol=1e-5):
        raise AssertionError(
            f"loaded forward diverged from SceneNet.forward: max abs {float(np.max(np.abs(got - teacher)))}"
        )
    assert out.lane_x is not None
    print(
        f"loader round-trip: lane_gt dim={got.size} max_abs_err="
        f"{float(np.max(np.abs(got - teacher))):.2e} lane_x={out.lane_x:.4f} OK"
    )


def check_aux_ckpt_without_meta_flag() -> None:
    import tempfile

    import torch

    from python.train.scene_ckpt_runtime import SceneCkptRuntime
    from python.train.scene_net import SceneNet

    torch.manual_seed(11)
    net = SceneNet(cams=("main",), aux_sensors=True).cpu().eval()
    frame = _fixture_main(seed=3)
    pred, _state, _canvas = _teacher_forward(net, frame, 0.05)
    teacher = pred.lane_gt.detach().cpu().numpy().reshape(-1)
    with tempfile.TemporaryDirectory() as tmp:
        # Production aux-run files list aux heads in the weights / arch string, not always in meta.
        path = _write_ckpt(Path(tmp), net, meta_extra={"aux_sensors": None})
        runtime = SceneCkptRuntime(path)
        assert runtime.aux_sensors is True
        out = runtime.observe(frame, 0.05)
    got = np.asarray(out.lane_gt, dtype=np.float32).reshape(-1)
    assert np.allclose(got, teacher, atol=1e-5, rtol=1e-5), float(np.max(np.abs(got - teacher)))
    print("aux ckpt without meta.aux_sensors: inferred and matched teacher OK")


def check_main_cam_shape() -> None:
    import tempfile

    import torch

    from python.train.scene_ckpt_runtime import SceneCkptInputError, SceneCkptRuntime
    from python.train.scene_net import SceneNet

    torch.manual_seed(2)
    net = SceneNet(cams=("main",)).cpu().eval()
    with tempfile.TemporaryDirectory() as tmp:
        runtime = SceneCkptRuntime(_write_ckpt(Path(tmp), net))
        empty = runtime.observe(None, 0.1)
        assert empty.lane_gt is None and empty.lane_x is None
        ok = runtime.observe(_fixture_main(), 0.1)
        assert ok.canvas_shape == (192, 256, 3), ok.canvas_shape
        assert np.asarray(ok.lane_gt).reshape(-1).size >= 14
        try:
            runtime.observe(np.zeros((480, 640), dtype=np.uint8), 0.1)
        except SceneCkptInputError as exc:
            assert "channel" in str(exc).lower() or "shape" in str(exc).lower(), exc
        else:
            raise AssertionError("grayscale main frame was accepted")
        try:
            runtime.observe(np.zeros((480, 640, 4), dtype=np.uint8), 0.1)
        except SceneCkptInputError as exc:
            assert "channel" in str(exc).lower() or "shape" in str(exc).lower(), exc
        else:
            raise AssertionError("4-channel main frame was accepted")
    print("main cam shape: 480x640x3 packed to 192x256x3; gray and 4ch refused; missing frame empty OK")


def check_shadow_cannot_actuate() -> None:
    from python.control.actuate import DriveCommand
    from python.train.scene_ckpt_runtime import SceneCkptPred, apply_scene_ckpt_to_command, shadow_engage_allowed

    assert shadow_engage_allowed() is False
    cmd = DriveCommand(steer=0.31, throttle=0.4, brake=0.0, seq=3, reason="lane")
    pred = SceneCkptPred(
        lane_gt=np.zeros(14, dtype=np.float32),
        lane_x=-1.25,
        confidence=0.9,
        canvas_shape=(192, 256, 3),
    )
    after = apply_scene_ckpt_to_command(cmd, pred)
    assert after.steer == cmd.steer and after.throttle == cmd.throttle and after.brake == cmd.brake
    assert after.reason == cmd.reason
    print("shadow isolation helper: command unchanged by a confident prediction OK")

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
    assert "scene-ckpt" in text.lower() or "SHADOW" in text or "shadow" in text.lower(), text[-2000:]
    assert states, text[-2000:]
    assert all(not st.get("engaged") for st in states), [st.get("engaged") for st in states[:5]]
    # Prediction must not become the lane the wheel follows.
    for st in states:
        hold = st.get("lane_hold") or {}
        src = str(hold.get("source") or st.get("lane_src") or "")
        assert src != "scene", (src, hold)
        ckpt = st.get("scene_ckpt") or {}
        if ckpt.get("lane_x") is not None:
            assert hold.get("lateral_m") != ckpt.get("lane_x") or hold.get("source") != "scene"
    print(
        f"shadow cannot actuate: {len(states)} stub states engaged=0 with --force-engage "
        f"and --scene-ckpt (no --no-engage on CLI) OK"
    )


def check_overlay_on_stub() -> None:
    from python.train.scene_ckpt_runtime import SceneCkptPred, overlay_bgr, draw_scene_ckpt_overlay
    from python.debug.lane_targets import LANE_GT_DIM, LOOKAHEADS_M

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    values = np.zeros((LANE_GT_DIM,), dtype=np.float32)
    for i, _y in enumerate(LOOKAHEADS_M):
        values[i * 3] = 0.8 + 0.15 * i
        values[i * 3 + 1] = values[i * 3] - 1.8
        values[i * 3 + 2] = values[i * 3] + 1.8
    values[-1] = 0.77
    pred = SceneCkptPred(lane_gt=values, lane_x=0.95, confidence=0.77, canvas_shape=(192, 256, 3))
    painted = draw_scene_ckpt_overlay(frame, pred)
    assert painted is not frame
    assert painted.shape == frame.shape
    assert not np.array_equal(painted, frame), "overlay left the stub frame unchanged"
    # Distinct from the ice-blue corridor used by the existing viz (ICE = 212,196,158).
    ice = np.array(overlay_bgr("ice"), dtype=np.int16)
    mark = np.array(overlay_bgr("shadow"), dtype=np.int16)
    assert int(np.sum(np.abs(ice - mark))) > 40, (ice, mark)
    shadow = overlay_bgr("shadow")
    hit = np.any(np.all(painted == np.array(shadow, dtype=np.uint8), axis=-1))
    assert hit, "shadow overlay color was not drawn"
    # Caption so a clip can be read without guessing the polyline.
    low = painted.mean(axis=2)
    assert float((low > 0).mean()) > 0
    print("overlay: stub 480x640 frame marked SHADOW in a non-ice color OK")


def main() -> None:
    check_loader_round_trip()
    check_aux_ckpt_without_meta_flag()
    check_main_cam_shape()
    check_shadow_cannot_actuate()
    check_overlay_on_stub()
    print("test_scene_ckpt_shadow: all OK")


if __name__ == "__main__":
    main()
