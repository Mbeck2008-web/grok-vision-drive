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

    # Flag OFF path in run_vision must not import python.viz.rv_contract into sys.modules.
    lazy_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from python.run_vision import build_parser, start_reality_view; "
                "assert 'python.viz.rv_contract' not in sys.modules, 'imported on run_vision import'; "
                "args = build_parser().parse_args([]); "
                "assert start_reality_view(args) is None; "
                "assert 'python.viz.rv_contract' not in sys.modules, 'rv_contract imported when flag OFF (default)'; "
                "assert start_reality_view(args, cameras={'viz': {'reality_view_enabled': 'true'}}) is None; "
                "assert 'python.viz.rv_contract' not in sys.modules, 'rv_contract imported on truthy string'; "
                "print('LAZY_OK')"
            ),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert lazy_probe.returncode == 0, f"lazy import check failed: {lazy_probe.stdout} {lazy_probe.stderr}"
    assert "LAZY_OK" in lazy_probe.stdout


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

    # 6. Intermediate parent SIGKILL must not orphan the --serve stub (PDEATHSIG on Linux, ppid poll fallback).
    import signal
    import time

    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _wait_pid_exit(pid: int, timeout_s: float = 4.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not _pid_alive(pid):
                return True
            time.sleep(0.05)
        return not _pid_alive(pid)

    # Case A: intermediate parent launches stub via RvClient, then gets SIGKILLed.
    parent_a = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import time; "
                "from python.viz.rv_contract import RvClient; "
                "c = RvClient(); "
                "pid = c.launch(); "
                "print(pid, flush=True); "
                "time.sleep(60)"
            ),
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stub_pid_a: int | None = None
    try:
        assert parent_a.stdout is not None
        line_a = parent_a.stdout.readline().strip()
        assert line_a.isdigit(), f"unexpected parent_a output: {line_a!r}"
        stub_pid_a = int(line_a)
        assert _pid_alive(stub_pid_a), f"stub_pid_a {stub_pid_a} not alive"
        parent_a.kill()
        parent_a.wait(timeout=5)
        assert _wait_pid_exit(stub_pid_a, timeout_s=4.0), f"stub_pid_a {stub_pid_a} orphaned after parent SIGKILL"
    finally:
        if parent_a.poll() is None:
            parent_a.kill()
            parent_a.wait(timeout=5)
        if stub_pid_a is not None and _pid_alive(stub_pid_a):
            try:
                os.kill(stub_pid_a, signal.SIGKILL)
            except OSError:
                pass

    # Case B (POSIX): intermediate parent spawns --serve with stdin kept open by grandparent pipe (so EOF cannot trigger).
    if sys.platform != "win32":
        r_fd, w_fd = os.pipe()
        os.set_inheritable(r_fd, True)
        parent_b = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import subprocess, sys, time; "
                    "from python.viz.rv_contract import RV_RENDERER_MODULE, pdeathsig_preexec; "
                    f"p = subprocess.Popen([sys.executable, '-m', RV_RENDERER_MODULE, '--serve'], stdin={r_fd}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=pdeathsig_preexec()); "
                    "print(p.pid, flush=True); "
                    "time.sleep(60)"
                ),
            ],
            cwd=str(ROOT),
            pass_fds=(r_fd,),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        os.close(r_fd)
        stub_pid_b: int | None = None
        try:
            assert parent_b.stdout is not None
            line_b = parent_b.stdout.readline().strip()
            assert line_b.isdigit(), f"unexpected parent_b output: {line_b!r} stderr={parent_b.stderr.read() if parent_b.stderr else ''}"
            stub_pid_b = int(line_b)
            assert _pid_alive(stub_pid_b), f"stub_pid_b {stub_pid_b} not alive"
            parent_b.kill()
            parent_b.wait(timeout=5)
            assert _wait_pid_exit(stub_pid_b, timeout_s=4.0), (
                f"stub_pid_b {stub_pid_b} orphaned after parent SIGKILL with open stdin"
            )
        finally:
            try:
                os.close(w_fd)
            except OSError:
                pass
            if parent_b.poll() is None:
                parent_b.kill()
                parent_b.wait(timeout=5)
            if stub_pid_b is not None and _pid_alive(stub_pid_b):
                try:
                    os.kill(stub_pid_b, signal.SIGKILL)
                except OSError:
                    pass

    # Case C (Linux): verify prctl PR_SET_PDEATHSIG(SIGTERM) is armed by both pdeathsig_preexec and stub._install_pdeathsig.
    if sys.platform.startswith("linux"):
        pdeath_probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import ctypes, signal, subprocess, sys; "
                    "from python.viz.rv_contract import pdeathsig_preexec; "
                    "fn = pdeathsig_preexec(); "
                    "assert fn is not None; "
                    "p = subprocess.run([sys.executable, '-c', "
                    "'import ctypes; libc=ctypes.CDLL(None); pr=libc.prctl; "
                    "pr.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong]; "
                    "s=ctypes.c_int(0); assert pr(2, ctypes.addressof(s), 0,0,0)==0; print(s.value)'], "
                    "preexec_fn=fn, capture_output=True, text=True, check=True); "
                    "assert int(p.stdout.strip()) == signal.SIGTERM, p.stdout; "
                    "from python.viz.rv_renderer_stub import _install_pdeathsig; "
                    "assert _install_pdeathsig() == signal.SIGTERM; "
                    "libc = ctypes.CDLL(None); pr = libc.prctl; "
                    "pr.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]; "
                    "s = ctypes.c_int(0); assert pr(2, ctypes.addressof(s), 0, 0, 0) == 0; "
                    "assert s.value == signal.SIGTERM, s.value; "
                    "print('PDEATHSIG_OK')"
                ),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert pdeath_probe.returncode == 0, f"pdeathsig probe failed: {pdeath_probe.stdout} {pdeath_probe.stderr}"
        assert "PDEATHSIG_OK" in pdeath_probe.stdout


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

    # Non-finite ego_pose_4x4, K (intrinsics_3x3), and T_cam_ego (extrinsics_4x4) fail loudly.
    for bad_val in (float("nan"), float("inf"), float("-inf")):
        bad_pose = pose.copy()
        bad_pose[0, 3] = bad_val
        _expect_fail(
            RvInputSpec(
                schema_version=RV_SCHEMA_VERSION,
                seq=1,
                timestamp_s=1.0,
                ego_pose_4x4=bad_pose,
                camera_frames=good.camera_frames,
            ),
            f"non-finite ego_pose_4x4 ({bad_val})",
        )

        bad_k = k.copy()
        bad_k[0, 2] = bad_val
        _expect_fail(
            RvInputSpec(
                schema_version=RV_SCHEMA_VERSION,
                seq=1,
                timestamp_s=1.0,
                ego_pose_4x4=pose,
                camera_frames={
                    "main": RvCameraInput(cam_id="main", rgb=frame, intrinsics_3x3=bad_k, extrinsics_4x4=t_cam)
                },
            ),
            f"non-finite intrinsics_3x3 K ({bad_val})",
        )

        bad_t = t_cam.copy()
        bad_t[1, 3] = bad_val
        _expect_fail(
            RvInputSpec(
                schema_version=RV_SCHEMA_VERSION,
                seq=1,
                timestamp_s=1.0,
                ego_pose_4x4=pose,
                camera_frames={
                    "main": RvCameraInput(cam_id="main", rgb=frame, intrinsics_3x3=k, extrinsics_4x4=bad_t)
                },
            ),
            f"non-finite extrinsics_4x4 T_cam_ego ({bad_val})",
        )

    # Sim GT / aux_depth keys rejected on live input contract by the forbidden-key check
    # (pinned with a matching camera_frames entry so 'no matching camera frame' cannot mask it).
    forbidden_keys = (
        "depth_gt",
        "gt_depth",
        "sim_depth",
        "aux_depth",
        "sem_gt",
        "gt_sem",
        "path_world",
        "debug_gt",
        "labels",
        "custom_gt",
        "gt_custom",
        "extra_sim_depth",
    )
    for gt_key in forbidden_keys:
        matching_cam = RvCameraInput(cam_id="main", rgb=frame, intrinsics_3x3=k, extrinsics_4x4=t_cam)
        depth_val = np.ones((360, 640), dtype=np.float32)

        # (a) Valid matching camera_frames entry + matching predicted_depth -> only forbidden-key check can reject.
        for spec_case, desc in (
            (
                RvInputSpec(
                    schema_version=RV_SCHEMA_VERSION,
                    seq=1,
                    timestamp_s=1.0,
                    ego_pose_4x4=pose,
                    camera_frames={"main": good.camera_frames["main"], gt_key: matching_cam},
                    predicted_depth={gt_key: depth_val},
                ),
                f"matching frame + predicted_depth forbidden key {gt_key}",
            ),
            (
                RvInputSpec(
                    schema_version=RV_SCHEMA_VERSION,
                    seq=1,
                    timestamp_s=1.0,
                    ego_pose_4x4=pose,
                    camera_frames=good.camera_frames,
                    predicted_depth={gt_key: depth_val},
                ),
                f"predicted_depth forbidden key {gt_key}",
            ),
            (
                RvInputSpec(
                    schema_version=RV_SCHEMA_VERSION,
                    seq=1,
                    timestamp_s=1.0,
                    ego_pose_4x4=pose,
                    camera_frames={gt_key: matching_cam},
                    predicted_depth=None,
                ),
                f"camera_frames forbidden key {gt_key}",
            ),
        ):
            try:
                validate_rv_input(spec_case)
            except RvSchemaError as exc:
                assert "forbidden" in str(exc).lower(), f"{desc}: expected forbidden error, got {exc!r}"
            else:
                raise AssertionError(f"expected RvSchemaError for {desc}")


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
    # Non-finite output pose fails loudly.
    for bad_val in (float("nan"), float("inf"), float("-inf")):
        bad_out_pose = pose.copy()
        bad_out_pose[2, 3] = bad_val
        _expect_fail(
            RvOutputSpec(schema_version=RV_SCHEMA_VERSION, seq=5, timestamp_s=2.0, pose_4x4=bad_out_pose, rgb=rgb, depth=depth),
            f"non-finite output pose ({bad_val})",
        )
    # Max legal output (640x640) validates, and oversize output (>640) is rejected.
    validate_rv_output(
        RvOutputSpec(
            schema_version=RV_SCHEMA_VERSION,
            seq=5,
            timestamp_s=2.0,
            pose_4x4=pose,
            rgb=np.zeros((640, 640, 3), dtype=np.uint8),
            depth=np.zeros((640, 640), dtype=np.float32),
        )
    )
    for bad_hw in ((641, 640), (640, 641)):
        _expect_fail(
            RvOutputSpec(
                schema_version=RV_SCHEMA_VERSION,
                seq=5,
                timestamp_s=2.0,
                pose_4x4=pose,
                rgb=np.zeros((bad_hw[0], bad_hw[1], 3), dtype=np.uint8),
                depth=np.zeros(bad_hw, dtype=np.float32),
            ),
            f"oversize output {bad_hw}",
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

    # Wrong version in header bytes must fail loudly (both RvShmHeader.unpack directly and unpack_output_slot).
    bad_ver = bytearray(raw)
    struct.pack_into("<I", bad_ver, 8, RV_SCHEMA_VERSION + 1)
    try:
        RvShmHeader.unpack(bytes(bad_ver))
        raise AssertionError("RvShmHeader.unpack must fail loudly on wrong schema_version")
    except RvSchemaError:
        pass
    try:
        unpack_output_slot(bytes(bad_ver))
        raise AssertionError("wrong SHM schema_version must fail loudly")
    except RvSchemaError:
        pass

    # Wrong magic and short header must fail loudly in RvShmHeader.unpack directly.
    bad_magic = bytearray(raw)
    bad_magic[0:8] = b"BADMAGIC"
    try:
        RvShmHeader.unpack(bytes(bad_magic))
        raise AssertionError("RvShmHeader.unpack must fail loudly on wrong magic")
    except RvSchemaError:
        pass
    try:
        RvShmHeader.unpack(raw[: SLOT_HEADER_SIZE - 1])
        raise AssertionError("RvShmHeader.unpack must fail loudly on short buffer")
    except RvSchemaError:
        pass
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

    # Real contended-lock backpressure: another thread holds ring._lock -> write_slot drops without blocking.
    import threading

    acquired = threading.Event()
    release = threading.Event()

    def _hold_lock(lock: threading.Lock) -> None:
        lock.acquire()
        try:
            acquired.set()
            release.wait(timeout=5.0)
        finally:
            lock.release()

    holder = threading.Thread(target=_hold_lock, args=(ring._lock,), daemon=True)
    holder.start()
    assert acquired.wait(timeout=2.0), "lock holder did not acquire lock"
    write_result: dict[str, object] = {}

    def _try_write() -> None:
        write_result["ok"] = ring.write_slot(raw, seq=4)

    writer = threading.Thread(target=_try_write, daemon=True)
    before_drops = ring.dropped
    writer.start()
    writer.join(timeout=1.0)
    timed_out = writer.is_alive()
    release.set()
    holder.join(timeout=2.0)
    writer.join(timeout=2.0)
    assert not timed_out, "DoubleBufferSlotLayout.write_slot blocked on contended lock instead of dropping"
    assert write_result.get("ok") is False
    assert ring.dropped == before_drops + 1

    # Default RvClient slot is sized from the schema (max legal 640x640 RGB+depth+header = 2,867,328 B).
    from python.viz.rv_contract import RV_MAX_LONG_SIDE, RvClient

    expected_max_slot = SLOT_HEADER_SIZE + RV_MAX_LONG_SIDE * RV_MAX_LONG_SIDE * (3 + 4)
    default_client = RvClient()
    assert default_client._ring.slot_bytes >= expected_max_slot, (
        f"default RvClient slot_bytes={default_client._ring.slot_bytes} < max legal frame {expected_max_slot}"
    )

    # 360x640 legal frame (1,612,928 B) and max legal 640x640 frame (2,867,328 B) pack and fit.
    for h_out, w_out, expected_bytes, seq_no in (
        (360, 640, 1_612_928, 10),
        (RV_MAX_LONG_SIDE, RV_MAX_LONG_SIDE, expected_max_slot, 11),
    ):
        rgb_leg = np.zeros((h_out, w_out, 3), dtype=np.uint8)
        rgb_leg[0, 0, :] = [11, 22, 33]
        depth_leg = np.full((h_out, w_out), 7.5, dtype=np.float32)
        packed = pack_output_slot(
            RvOutputSpec(
                schema_version=RV_SCHEMA_VERSION,
                seq=seq_no,
                timestamp_s=3.0,
                pose_4x4=pose,
                rgb=rgb_leg,
                depth=depth_leg,
            )
        )
        assert len(packed) == expected_bytes, f"{(h_out, w_out)} packed {len(packed)} != {expected_bytes}"
        assert default_client._ring.write_slot(packed, seq=seq_no) is True
        latest_leg = default_client._ring.read_latest()
        assert latest_leg is not None
        _, roundtrip = unpack_output_slot(latest_leg)
        assert roundtrip.rgb.shape == (h_out, w_out, 3) and roundtrip.depth.shape == (h_out, w_out)
        assert int(roundtrip.rgb[0, 0, 0]) == 11

    # Oversize frame (>640) is rejected by pack_output_slot, and oversize raw slot fails loudly in write_slot.
    try:
        pack_output_slot(
            RvOutputSpec(
                schema_version=RV_SCHEMA_VERSION,
                seq=12,
                timestamp_s=3.0,
                pose_4x4=pose,
                rgb=np.zeros((641, 640, 3), dtype=np.uint8),
                depth=np.zeros((641, 640), dtype=np.float32),
            )
        )
        raise AssertionError("oversize 641x640 output must fail loudly")
    except RvSchemaError:
        pass
    try:
        default_client._ring.write_slot(b"\x00" * (default_client._ring.slot_bytes + 1), seq=13)
        raise AssertionError("oversize raw slot payload must fail loudly")
    except RvSchemaError:
        pass


def main() -> None:
    check_flag_off_is_placeholder()
    check_live_path_is_separate_process()
    check_rv_input_schema()
    check_rv_output_schema()
    check_shm_slot_roundtrip_and_copy_only()
    print("test_rv_contract: ok")


if __name__ == "__main__":
    main()
