#!/usr/bin/env python3
"""--scene-ckpt takes the live 8-camera frame dict (Trainer 2.0 PR2b; spec docs/specs/trainer_2.md l.31-36).

Before: ``scene_ckpt_runtime._main_canvas`` composed ``compose_strip({"main": frame})``, so a
checkpoint that lists more than ``main`` saw black side sectors and was refused at load.
After: ``observe(frames)`` composes the whole live dict (wide included) with ``compose_strip``,
the same call the live lane path makes at run_vision.py (``compose_strip(frames)``).

Checks
* check_train_live_input_identical: on one REAL recorded 8-camera tick
  (tests/fixtures/cams8_clip015_000001, see its README) three builders give byte-identical
  float32 NCHW tensors, pinned by sha256:
    train   : train_scene._load_bgr -> cam_strip.extract_cams (8 cams) -> scene_net._strip_nchw  (augment off)
    live    : strip_writer.compose_strip(frames) -> scene_lane_runtime._strip_nchw
    ckpt    : SceneCkptRuntime.observe(frames) -> the tensor SceneNet.forward actually receives
* the 8-cam checkpoint loads; main-only is gone (an ndarray frame is refused; a dict without main is empty).
* engage stays hard-blocked in --scene-ckpt shadow mode with an 8-cam checkpoint.
* run_vision passes the live frame dict (not ``main``) to observe.

    py -3 scripts/test_scene_ckpt_8cam.py
"""
# gvd-ci: requires torch
from __future__ import annotations

import ast
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

FIXTURE = ROOT / "tests" / "fixtures" / "cams8_clip015_000001"
FRAME_INDEX = 1
# One pin per fixture (see the fixture README). Any drift in cam order (STRIP_ORDER, 8 names incl. wide),
# sector crop/x0/x1, resolution, dtype, layout or /255 changes them.
PIN_CANVAS_SHA256 = "1a678eac4a83bd912f18830a7a85a88f8dbba1183b7bd200d8eac872209a21bd"
PIN_TENSOR_SHA256 = "4e39b640d592fd3d2094bf5112cb3cd25939e890a906f9397f4f863455dfe329"
FIXTURE_FILE_SHA256 = {
    "repeatL": "f694e64e39dad361e7a211d2b343de44d554f106f9424cf4b27fc106acef7bb5",
    "pillarL": "4905d68091ac01f9fbabbd3b1b2a2a2d841c482eff368ee1618a0478eca056ad",
    "wide": "6821ac4cca0a02a27dbef6c158d0ace7e9861a661317f9c0f577d457e3985e5c",
    "main": "4e7fed544022ac151743712097ad7b82f1b625b4433e05237f0253308ed0adba",
    "narrow": "aa827d65a11c89e2ff1b8f4d3ed7ea54df75d5238eaffb74ea72c3d2d763e20b",
    "pillarR": "881c862e4c62d4f2c9e585ba6ea03a291891257e3001a98de1982f78a3f473da",
    "repeatR": "d4b89560cb4f97a68e822803b757946a7a99850435f87211016250832a339bb6",
    "rear": "ceced09f794aca2280e25d7d44a914b58de79ba59701378aef404a7dd8b4602b",
}


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _bytes(array) -> bytes:
    """Logical C-order bytes (a torch permute view is not contiguous in memory)."""
    return np.ascontiguousarray(np.asarray(array)).tobytes()


def _fixture_frames() -> dict:
    from python.data.raw_frames import load_raw_cameras

    frames = load_raw_cameras(FIXTURE, FRAME_INDEX)
    return frames


def _runtime(tmp: Path, cams: tuple[str, ...] | None, seed: int = 3):
    import torch

    from python.train.scene_ckpt_runtime import SceneCkptRuntime
    from python.train.scene_net import SceneNet
    from test_scene_ckpt_shadow import _write_ckpt

    torch.manual_seed(seed)
    net = (SceneNet(cams=cams) if cams else SceneNet()).cpu().eval()
    return SceneCkptRuntime(_write_ckpt(tmp, net)), net


def _observe_capture(runtime, frames):
    """observe(frames) and the (uint8 canvas, float tensor) that SceneNet.forward really got."""
    from python.train import scene_net

    real = scene_net._strip_nchw
    seen: list = []

    def spy(strip, **kwargs):
        out = real(strip, **kwargs)
        seen.append((np.array(strip, copy=True), out.detach().cpu().numpy()))
        return out

    scene_net._strip_nchw = spy
    try:
        pred = runtime.observe(frames, 0.1)
    finally:
        scene_net._strip_nchw = real
    assert len(seen) == 1, f"forward built {len(seen)} input tensors, expected 1"
    return pred, seen[0][0], seen[0][1]


# ------------------------------------------------------------------ fixture itself


def check_fixture_is_real_8cam() -> None:
    from python.data.strip_writer import SECTORS, STRIP_ORDER

    for cam, want in FIXTURE_FILE_SHA256.items():
        path = FIXTURE / "raw" / cam / f"{FRAME_INDEX:06d}.png"
        assert path.is_file(), f"fixture file missing: {path}"
        assert _sha(path.read_bytes()) == want, f"{path.name} for {cam} changed"
    assert tuple(STRIP_ORDER) == ("repeatL", "pillarL", "wide", "main", "narrow", "pillarR", "repeatR", "rear"), STRIP_ORDER
    frames = _fixture_frames()
    assert set(frames) == set(STRIP_ORDER), sorted(frames)
    assert "wide" in frames
    for cam, img in frames.items():
        assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3, (cam, img.shape, img.dtype)
    from python.data.strip_writer import compose_strip

    strip = compose_strip(frames)
    for sec in SECTORS:
        assert int(strip[:, sec.x0 : sec.x1].max()) > 0, f"{sec.cam_id} sector is black in the fixture strip"
    print(f"fixture: 8 cams {list(STRIP_ORDER)} all present, native sizes, every sector non-black OK")


# ------------------------------------------------------------------ the required check


def check_train_live_input_identical() -> None:
    from python.data.strip_writer import STRIP_ORDER, compose_strip
    from python.train.cam_strip import extract_cams
    from python.train.scene_lane_runtime import _strip_nchw as live_nchw
    from python.train.scene_net import _strip_nchw as train_nchw
    from python.train.train_scene import _load_bgr

    frames = _fixture_frames()
    # 1. train builder, augment off: loader -> (8-cam extract = full strip) -> trainer _strip_nchw
    train_canvas = extract_cams(_load_bgr(FIXTURE, FRAME_INDEX), tuple(STRIP_ORDER))
    train_t = train_nchw(train_canvas).detach().cpu().numpy()
    # 2. live lane builder: compose_strip of the live dict -> live _strip_nchw
    live_canvas = compose_strip(frames)
    live_t = live_nchw(live_canvas)
    # 3. --scene-ckpt builder: observe(frames) -> what forward is fed
    with tempfile.TemporaryDirectory() as tmp:
        runtime, _net = _runtime(Path(tmp), None)
        assert tuple(runtime.cams) == tuple(STRIP_ORDER), runtime.cams
        pred, ckpt_canvas, ckpt_t = _observe_capture(runtime, frames)
    assert pred.canvas_shape == (192, 2272, 3), pred.canvas_shape

    for name, arr in (("train", train_t), ("live", live_t), ("scene-ckpt", ckpt_t)):
        assert arr.dtype == np.float32, (name, arr.dtype)
        assert tuple(arr.shape) == (1, 3, 192, 2272), (name, arr.shape)  # NCHW, not NHWC
    assert _bytes(train_canvas) == _bytes(live_canvas) == _bytes(ckpt_canvas), "uint8 canvases differ"
    assert _sha(_bytes(live_canvas)) == PIN_CANVAS_SHA256, _sha(_bytes(live_canvas))
    b_train, b_live, b_ckpt = _bytes(train_t), _bytes(live_t), _bytes(ckpt_t)
    assert b_train == b_live, "train builder != live lane builder"
    assert b_live == b_ckpt, "live lane builder != --scene-ckpt builder"
    assert _sha(b_live) == PIN_TENSOR_SHA256, _sha(b_live)
    print(
        "train == live lane == --scene-ckpt: float32 NCHW (1,3,192,2272) byte-identical; "
        f"canvas sha256 {PIN_CANVAS_SHA256[:16]}.., tensor sha256 {PIN_TENSOR_SHA256[:16]}.."
    )


# ------------------------------------------------------------------ runtime behaviour


def check_main_canvas_is_the_full_strip() -> None:
    from python.data.strip_writer import SECTORS, STRIP_ORDER, compose_strip
    from python.train.scene_ckpt_runtime import _main_canvas

    frames = _fixture_frames()
    canvas = _main_canvas(frames, tuple(STRIP_ORDER), 192, 2272, 1.0)
    assert _bytes(canvas) == _bytes(compose_strip(frames)), "canvas is not compose_strip(live 8-cam dict)"
    wide = next(s for s in SECTORS if s.cam_id == "wide")
    assert int(canvas[:, wide.x0 : wide.x1].max()) > 0, "wide sector is black"
    # the old main-only strip is a different image
    main_only = compose_strip({"main": frames["main"]})
    assert _bytes(canvas) != _bytes(main_only)
    # a changed wide picture changes the canvas, in the wide sector only
    moved = dict(frames)
    moved["wide"] = np.full_like(frames["wide"], 200)
    other = _main_canvas(moved, tuple(STRIP_ORDER), 192, 2272, 1.0)
    diff_cols = np.where(np.any(other != canvas, axis=(0, 2)))[0]
    assert diff_cols.size and int(diff_cols.min()) >= wide.x0 and int(diff_cols.max()) < wide.x1, "wide change leaked or was dropped"
    print("_main_canvas == compose_strip(8-cam dict); wide sector present and wide-only edits stay in the wide sector OK")


def check_multicam_loads_and_observes() -> None:
    from python.data.strip_writer import SECTORS

    frames = _fixture_frames()
    with tempfile.TemporaryDirectory() as tmp:
        rt2, _ = _runtime(Path(tmp), ("main", "wide"), seed=4)
        assert tuple(rt2.cams) == ("main", "wide"), rt2.cams
        pred, canvas, tensor = _observe_capture(rt2, frames)
        assert tensor.shape[-2] == 192 and tensor.shape[-1] == canvas.shape[1], (tensor.shape, canvas.shape)
        # sectors repacked in strip order: wide then main, both from the live dict
        wide_w = next(s for s in SECTORS if s.cam_id == "wide").x1 - next(s for s in SECTORS if s.cam_id == "wide").x0
        assert int(canvas[:, :wide_w].max()) > 0 and int(canvas[:, -256:].max()) > 0, "a sector of the 2-cam canvas is black"
        assert pred.lane_gt is not None and pred.canvas_shape == tuple(canvas.shape)
    print(f"2-cam (main,wide) ckpt loads and sees both cams from the live dict (canvas {tuple(canvas.shape)}) OK")


def check_main_only_input_is_gone() -> None:
    from python.train.scene_ckpt_runtime import SceneCkptInputError

    frames = _fixture_frames()
    with tempfile.TemporaryDirectory() as tmp:
        rt, _ = _runtime(Path(tmp), None)
        # a bare main ndarray is the old API: refused, never composed as a main-only strip
        try:
            rt.observe(frames["main"], 0.1)
        except SceneCkptInputError as exc:
            assert "frame" in str(exc).lower() and "dict" in str(exc).lower(), exc
        else:
            raise AssertionError("a bare main frame was accepted (main-only path still alive)")
        assert rt.observe(None, 0.1).lane_gt is None
        assert rt.observe({}, 0.1).lane_gt is None
        no_main = {k: v for k, v in frames.items() if k != "main"}
        assert rt.observe(no_main, 0.1).lane_gt is None, "no main picture must give an empty prediction"
        bad = dict(frames)
        bad["wide"] = np.zeros((360, 640), dtype=np.uint8)
        try:
            rt.observe(bad, 0.1)
        except SceneCkptInputError as exc:
            assert "wide" in str(exc) and ("channel" in str(exc).lower() or "shape" in str(exc).lower()), exc
        else:
            raise AssertionError("a grayscale wide frame was accepted")
        ok = rt.observe(frames, 0.1)
        assert ok.lane_gt is not None and ok.canvas_shape == (192, 2272, 3), ok.canvas_shape
    print("main-only input gone: bare ndarray refused, None/{}/no-main empty, bad wide refused, full dict predicts OK")


# ------------------------------------------------------------------ engage stays blocked


def check_engage_still_blocked_with_8cam_ckpt() -> None:
    from python.control.actuate import DriveCommand
    from python.train.scene_ckpt_runtime import SceneCkptPred, apply_scene_ckpt_to_command, shadow_engage_allowed
    from test_live_default_path import run_supervisor

    assert shadow_engage_allowed() is False
    cmd = DriveCommand(steer=0.2, throttle=0.3, brake=0.0, seq=1, reason="lane")
    pred = SceneCkptPred(lane_gt=np.zeros(14, dtype=np.float32), lane_x=-1.0, confidence=0.9, canvas_shape=(192, 2272, 3))
    after = apply_scene_ckpt_to_command(cmd, pred)
    assert (after.steer, after.throttle, after.brake, after.reason) == (cmd.steer, cmd.throttle, cmd.brake, cmd.reason)

    with tempfile.TemporaryDirectory() as tmp:
        runtime, _net = _runtime(Path(tmp), None, seed=5)
        assert len(runtime.cams) == 8
        text, states = run_supervisor(["--scene-ckpt", str(runtime.path), "--force-engage", "--run-seconds", "2"], seconds=2.0)
    assert "scene-ckpt shadow" in text and "engage blocked" in text, text[-2000:]
    assert "scene-ckpt input" not in text, text[-2000:]
    assert states, text[-2000:]
    assert all(not st.get("engaged") for st in states), [st.get("engaged") for st in states[:5]]
    for st in states:
        assert str((st.get("lane_hold") or {}).get("source") or st.get("lane_src") or "") != "scene"
    print(f"engage blocked with an 8-cam --scene-ckpt: {len(states)} stub states engaged=0 under --force-engage OK")


# ------------------------------------------------------------------ run_vision wiring


def check_run_vision_passes_the_frame_dict() -> None:
    src = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "observe"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "scene_ckpt"
    ]
    assert len(calls) == 1, f"expected one scene_ckpt.observe call, found {len(calls)}"
    arg0 = calls[0].args[0]
    assert isinstance(arg0, ast.Name) and arg0.id == "ckpt_frames", ast.dump(arg0)
    # ckpt_frames is the live camera frame dict of this tick, the same source the live lane strip uses
    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "ckpt_frames" for t in node.targets)
    ]
    assert assigns, "ckpt_frames is never assigned"
    for node in assigns:
        text = ast.unparse(node.value)
        assert text == "getattr(bundle, 'frames', None)", text
    assert "scene_ckpt.observe(main," not in src
    print("run_vision: scene_ckpt.observe(ckpt_frames, ...) with ckpt_frames = getattr(bundle, 'frames', None) OK")


CHECKS = (
    check_fixture_is_real_8cam,
    check_train_live_input_identical,
    check_main_canvas_is_the_full_strip,
    check_multicam_loads_and_observes,
    check_main_only_input_is_gone,
    check_engage_still_blocked_with_8cam_ckpt,
    check_run_vision_passes_the_frame_dict,
)


def main() -> None:
    import traceback

    failed = []
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except BaseException as exc:  # report every check
            failed.append(fn.__name__)
            print(f"FAILED {fn.__name__}: {type(exc).__name__}: {str(exc)[:300]}")
            traceback.print_exc(limit=3)
    if failed:
        print(f"test_scene_ckpt_8cam: {len(failed)} FAILED: {', '.join(failed)}")
        sys.exit(1)
    print("test_scene_ckpt_8cam: all OK")


if __name__ == "__main__":
    main()
