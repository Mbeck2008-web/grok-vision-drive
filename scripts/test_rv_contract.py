#!/usr/bin/env python3
"""Offline contract tests for Reality View Step 1 (contract + flag pin).

Spec: docs/specs/reality_view_3dgs.md (Step 1).
Tests:
- check_flag_off_is_placeholder
- check_live_path_is_separate_process
- check_rv_input_schema
- check_rv_output_schema
- check_shm_slot_roundtrip_and_copy_only
"""
from __future__ import annotations

import ast
import hashlib
import os
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(arr) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def check_flag_off_is_placeholder() -> None:
    """Flag defaults OFF; when OFF, no RV process starts and placeholder is bit-identical."""
    import cv2
    import numpy as np
    import yaml

    from python.run_vision import build_parser, start_reality_view
    from python.viz.debug_draw import cabin_drive_word
    from python.viz.rv_contract import reality_view_enabled
    from python.viz.stage import STAGE_H, STAGE_W, VizUI
    from python.viz.view_select import render_main_view, render_reality_placeholder, resolve_view

    cams = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))
    assert "reality_view_enabled" in cams["viz"], "config/cameras.yaml viz.reality_view_enabled missing"
    assert cams["viz"]["reality_view_enabled"] is False, "viz.reality_view_enabled must default to False"
    assert cams["viz"]["main_view"] == "reality"

    ap = build_parser()
    bare = ap.parse_args([])
    assert bare.reality_view is False, "--reality-view CLI flag must default to False"

    # Flag helper: literal True only, no truthy coercions, no env-var bypass.
    assert reality_view_enabled(cams) is False
    assert reality_view_enabled(None) is False
    assert reality_view_enabled({}) is False
    for truthy in ("yes", "true", "1", 1, ["true"]):
        assert reality_view_enabled({"viz": {"reality_view_enabled": truthy}}) is False, truthy
    assert reality_view_enabled({"viz": {"reality_view_enabled": True}}) is True
    assert reality_view_enabled(cams, cli_reality_view=True) is True

    env_keys = ("GVD_REALITY_VIEW", "REALITY_VIEW", "GVD_RV_ENABLED")
    old_env = {k: os.environ.get(k) for k in env_keys}
    try:
        for k in env_keys:
            os.environ[k] = "1"
        assert reality_view_enabled(cams) is False
        assert start_reality_view(bare, cameras=cams) is None
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # When flag is OFF, start_reality_view must be a no-op returning None.
    assert start_reality_view(bare, cameras=cams) is None

    # Bit-identical PR #242 placeholder verification against independent reference.
    def _ref_placeholder(state: dict, w: int, h: int) -> np.ndarray:
        ref = np.full((h, w, 3), (24, 20, 18), dtype=np.uint8)
        f = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(ref, "GVD VISION - REALITY VIEW", (24, 44), f, 0.9, (235, 220, 200), 2, cv2.LINE_AA)
        cv2.putText(ref, "placeholder: renderer not built yet", (24, 80), f, 0.6, (150, 140, 130), 1, cv2.LINE_AA)
        wd, col = cabin_drive_word(dict(state))
        cv2.putText(ref, wd, (24, h - 28), f, 1.0, tuple(int(c) for c in col), 2, cv2.LINE_AA)
        return ref

    choice = resolve_view(cams)
    ui = VizUI()
    ui.show_nerd = False

    def _boom_classic(*_a, **_kw):
        raise AssertionError("flag OFF reality view must never call classic renderer")

    for st in ({"engaged": False}, {"engaged": True}):
        direct = render_reality_placeholder(dict(st), STAGE_W, STAGE_H)
        main_out = render_main_view(choice, dict(st), _boom_classic, width=STAGE_W, height=STAGE_H, ui=ui)
        ref = _ref_placeholder(st, STAGE_W, STAGE_H)
        assert np.array_equal(direct, ref), f"placeholder altered for {st}"
        assert np.array_equal(main_out, ref), f"main view placeholder altered for {st}"
        assert _sha256(main_out) == _sha256(ref)


def _ast_imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.add(node.module)
                for alias in node.names:
                    mods.add(f"{node.module}.{alias.name}")
    return mods


def check_live_path_is_separate_process() -> None:
    """RV renderer runs only in a separate child process; no safety-path imports."""
    from python.run_vision import build_parser, start_reality_view
    from python.viz.rv_contract import RV_RENDERER_MODULE, RV_SCHEMA_VERSION, RvClient

    assert RV_RENDERER_MODULE == "python.viz.rv_renderer_stub"

    # 1. Static check: run_vision.py and rv_contract.py never import any RV renderer module in-process.
    rv_path = ROOT / "python" / "run_vision.py"
    contract_path = ROOT / "python" / "viz" / "rv_contract.py"
    stub_path = ROOT / "python" / "viz" / "rv_renderer_stub.py"
    assert contract_path.is_file(), "missing python/viz/rv_contract.py"
    assert stub_path.is_file(), "missing python/viz/rv_renderer_stub.py"

    for mod_path in (rv_path, contract_path):
        imported = _ast_imported_modules(mod_path)
        bad_renderer = [m for m in imported if "rv_renderer" in m or "gsplat" in m or "gaussian_splatting" in m]
        assert not bad_renderer, f"{mod_path.name} must not import renderer modules in-process: {bad_renderer}"

    # 2. Static check: no RV module imports safety-path modules (heartbeat / AEB / engage / frame_gate / recorder).
    forbidden_prefixes = (
        "python.control",
        "python.runtime.frame_gate",
        "python.runtime.state_io",
        "python.data.record",
    )
    forbidden_substrings = ("actuate", "heartbeat", "aeb", "engage", "frame_gate", "override", "launch_guard")
    for rv_mod in (contract_path, stub_path):
        imported = _ast_imported_modules(rv_mod)
        for m in imported:
            assert not m.startswith(forbidden_prefixes), f"{rv_mod.name} imports safety module {m}"
            low = m.lower()
            for sub in forbidden_substrings:
                assert sub not in low, f"{rv_mod.name} imports forbidden safety symbol {m}"

    # 3. Runtime check: clean Python subprocess importing rv_contract never loads rv_renderer_stub or python.control.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, python.viz.rv_contract; "
                "bad=[m for m in sys.modules if 'rv_renderer' in m or m.startswith('python.control') or 'frame_gate' in m]; "
                "assert not bad, bad; print('OK')"
            ),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert probe.returncode == 0, f"clean import check failed: {probe.stdout} {probe.stderr}"

    # 4. Separate-process entrypoint runs in a different PID and reports schema version.
    ping = subprocess.run(
        [sys.executable, "-m", RV_RENDERER_MODULE, "--ping"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert ping.returncode == 0, f"stub ping failed: {ping.stdout} {ping.stderr}"
    import json

    info = json.loads(ping.stdout.strip().splitlines()[-1])
    assert info["schema_version"] == RV_SCHEMA_VERSION
    assert isinstance(info["pid"], int) and info["pid"] != os.getpid()

    # 5. Flag ON starts a separate child process (via start_reality_view) and closes cleanly.
    ap = build_parser()
    on_args = ap.parse_args(["--reality-view"])
    assert on_args.reality_view is True
    client = start_reality_view(on_args, cameras={"viz": {"reality_view_enabled": False}})
    assert isinstance(client, RvClient)
    try:
        assert client.child_pid is not None and client.child_pid != os.getpid()
        assert client.is_alive() is True
    finally:
        client.close()
    assert client.is_alive() is False


def check_rv_input_schema() -> None:
    """Versioned RV input schema validates shapes/cameras and rejects sim GT / wrong version."""
    import numpy as np

    from python.viz.rv_contract import (
        RV_SCHEMA_VERSION,
        RvCameraInput,
        RvInputSpec,
        RvSchemaError,
        validate_rv_input,
    )

    k = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    t_cam = np.eye(4, dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    good = RvInputSpec(
        schema_version=RV_SCHEMA_VERSION,
        seq=1,
        timestamp_s=1.25,
        ego_pose_4x4=pose,
        camera_frames={"main": RvCameraInput(cam_id="main", rgb=frame, intrinsics_3x3=k, extrinsics_4x4=t_cam)},
        predicted_depth=None,
    )
    validate_rv_input(good)

    def _expect_fail(spec: RvInputSpec, msg: str) -> None:
        try:
            validate_rv_input(spec)
        except RvSchemaError:
            return
        raise AssertionError(f"expected RvSchemaError for {msg}")

    # Wrong version fails loudly.
    for bad_v in (0, 2, -1, True, "1"):
        _expect_fail(
            RvInputSpec(
                schema_version=bad_v,  # type: ignore[arg-type]
                seq=1,
                timestamp_s=1.0,
                ego_pose_4x4=pose,
                camera_frames=good.camera_frames,
            ),
            f"schema_version={bad_v!r}",
        )

    # Long side > 640 fails loudly.
    _expect_fail(
        RvInputSpec(
            schema_version=RV_SCHEMA_VERSION,
            seq=1,
            timestamp_s=1.0,
            ego_pose_4x4=pose,
            camera_frames={
                "main": RvCameraInput(
                    cam_id="main",
                    rgb=np.zeros((480, 641, 3), dtype=np.uint8),
                    intrinsics_3x3=k,
                    extrinsics_4x4=t_cam,
                )
            },
        ),
        "long side > 640",
    )

    # Non-finite pose fails loudly.
    bad_pose = pose.copy()
    bad_pose[0, 3] = float("nan")
    _expect_fail(
        RvInputSpec(
            schema_version=RV_SCHEMA_VERSION,
            seq=1,
            timestamp_s=1.0,
            ego_pose_4x4=bad_pose,
            camera_frames=good.camera_frames,
        ),
        "non-finite pose",
    )

    # Sim GT / aux_depth keys rejected on live input contract.
    for gt_key in ("depth_gt", "gt_depth", "sim_depth", "aux_depth", "path_world", "debug_gt"):
        _expect_fail(
            RvInputSpec(
                schema_version=RV_SCHEMA_VERSION,
                seq=1,
                timestamp_s=1.0,
                ego_pose_4x4=pose,
                camera_frames=good.camera_frames,
                predicted_depth={gt_key: np.ones((360, 640), dtype=np.float32)},
            ),
            f"forbidden depth key {gt_key}",
        )


def check_rv_output_schema() -> None:
    """Output contract requires RGB + rendered depth buffer at identical pose and resolution."""
    import numpy as np

    from python.viz.rv_contract import RV_SCHEMA_VERSION, RvOutputSpec, RvSchemaError, validate_rv_output

    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    depth = np.full((120, 160), 12.5, dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)

    validate_rv_output(
        RvOutputSpec(
            schema_version=RV_SCHEMA_VERSION,
            seq=5,
            timestamp_s=2.0,
            pose_4x4=pose,
            rgb=rgb,
            depth=depth,
            stale=False,
        )
    )

    def _expect_fail(out: RvOutputSpec, msg: str) -> None:
        try:
            validate_rv_output(out)
        except RvSchemaError:
            return
        raise AssertionError(f"expected RvSchemaError for {msg}")

    # Depth buffer is required.
    _expect_fail(
        RvOutputSpec(schema_version=RV_SCHEMA_VERSION, seq=5, timestamp_s=2.0, pose_4x4=pose, rgb=rgb, depth=None),  # type: ignore[arg-type]
        "missing depth",
    )
    # Resolution mismatch fails loudly.
    _expect_fail(
        RvOutputSpec(
            schema_version=RV_SCHEMA_VERSION,
            seq=5,
            timestamp_s=2.0,
            pose_4x4=pose,
            rgb=rgb,
            depth=np.zeros((120, 159), dtype=np.float32),
        ),
        "depth shape mismatch",
    )
    # Wrong depth dtype fails loudly.
    _expect_fail(
        RvOutputSpec(
            schema_version=RV_SCHEMA_VERSION,
            seq=5,
            timestamp_s=2.0,
            pose_4x4=pose,
            rgb=rgb,
            depth=np.zeros((120, 160), dtype=np.float64),  # type: ignore[arg-type]
        ),
        "depth wrong dtype",
    )
    # Wrong version fails loudly.
    _expect_fail(
        RvOutputSpec(schema_version=99, seq=5, timestamp_s=2.0, pose_4x4=pose, rgb=rgb, depth=depth),
        "wrong version",
    )


def check_shm_slot_roundtrip_and_copy_only() -> None:
    """SHM slot schema round-trips RGB+depth, enforces version/magic/depth, and copies non-blocking."""
    import numpy as np

    from python.viz.rv_contract import (
        RV_FLAG_HAS_DEPTH,
        RV_MAX_SPLATS,
        RV_SCHEMA_VERSION,
        RV_SHM_MAGIC,
        SLOT_HEADER_SIZE,
        DoubleBufferSlotLayout,
        RvOutputSpec,
        RvSchemaError,
        RvShmHeader,
        copy_slot_bytes,
        pack_output_slot,
        unpack_output_slot,
    )

    assert SLOT_HEADER_SIZE == 128

    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
    depth = rng.uniform(0.5, 60.0, size=(24, 32)).astype(np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = -8.0
    pose[2, 3] = 4.0

    spec = RvOutputSpec(
        schema_version=RV_SCHEMA_VERSION,
        seq=42,
        timestamp_s=10.5,
        pose_4x4=pose,
        rgb=rgb,
        depth=depth,
        stale=True,
    )
    raw = pack_output_slot(spec, age_ms=250.0, generation=1)
    hdr, unpacked = unpack_output_slot(raw)
    assert hdr.magic == RV_SHM_MAGIC
    assert hdr.schema_version == RV_SCHEMA_VERSION
    assert hdr.seq == 42
    assert hdr.age_ms == 250.0
    assert (hdr.flags & RV_FLAG_HAS_DEPTH) != 0
    assert unpacked.seq == 42
    assert unpacked.stale is True
    assert np.array_equal(unpacked.rgb, rgb)
    assert np.array_equal(unpacked.depth, depth)
    assert np.allclose(unpacked.pose_4x4, pose)

    # Wrong version in header bytes must fail loudly.
    bad_ver = bytearray(raw)
    struct.pack_into("<I", bad_ver, 8, RV_SCHEMA_VERSION + 1)
    try:
        unpack_output_slot(bytes(bad_ver))
        raise AssertionError("wrong SHM schema_version must fail loudly")
    except RvSchemaError:
        pass

    # Wrong magic must fail loudly.
    bad_magic = bytearray(raw)
    bad_magic[0:8] = b"BADMAGIC"
    try:
        unpack_output_slot(bytes(bad_magic))
        raise AssertionError("wrong SHM magic must fail loudly")
    except RvSchemaError:
        pass

    # Missing depth flag or truncated depth payload must fail loudly.
    no_depth_flag = bytearray(raw)
    orig_flags = struct.unpack_from("<I", no_depth_flag, 12)[0]
    struct.pack_into("<I", no_depth_flag, 12, orig_flags & ~RV_FLAG_HAS_DEPTH)
    try:
        unpack_output_slot(bytes(no_depth_flag))
        raise AssertionError("missing RV_FLAG_HAS_DEPTH must fail loudly")
    except RvSchemaError:
        pass

    try:
        unpack_output_slot(raw[:-4])
        raise AssertionError("truncated depth payload must fail loudly")
    except RvSchemaError:
        pass

    # Splat count exceeding RV_MAX_SPLATS fails loudly.
    try:
        RvShmHeader(
            magic=RV_SHM_MAGIC,
            schema_version=RV_SCHEMA_VERSION,
            flags=RV_FLAG_HAS_DEPTH,
            seq=1,
            timestamp_s=1.0,
            age_ms=0.0,
            width=32,
            height=24,
            splat_count=RV_MAX_SPLATS + 1,
            payload_bytes=0,
            generation=1,
            pose_4x4=tuple(float(x) for x in pose.reshape(-1)),
        ).pack()
        raise AssertionError("splat_count > RV_MAX_SPLATS must fail loudly")
    except RvSchemaError:
        pass

    # Double-buffered copy-only slot handoff + backpressure drop.
    slot_cap = len(raw) + 64
    ring = DoubleBufferSlotLayout(slot_bytes=slot_cap)
    assert ring.num_slots == 2

    dst_buf = bytearray(slot_cap)
    copied = copy_slot_bytes(raw, dst_buf)
    assert copied == len(raw)
    assert bytes(dst_buf[: len(raw)]) == raw

    assert ring.write_slot(raw, seq=1) is True
    assert ring.write_slot(raw, seq=2) is True
    latest = ring.read_latest()
    assert latest is not None
    _, unpacked2 = unpack_output_slot(latest)
    assert np.array_equal(unpacked2.rgb, rgb)

    # Simulate consumer lock / backpressure on both slots -> non-blocking drop, never raises.
    ring.set_consumer_busy(True)
    assert ring.write_slot(raw, seq=3) is False
    assert ring.dropped == 1
    ring.set_consumer_busy(False)


def main() -> None:
    check_flag_off_is_placeholder()
    check_live_path_is_separate_process()
    check_rv_input_schema()
    check_rv_output_schema()
    check_shm_slot_roundtrip_and_copy_only()
    print("test_rv_contract: ok")


if __name__ == "__main__":
    main()
