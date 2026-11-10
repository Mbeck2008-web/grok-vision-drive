#!/usr/bin/env python3
"""Camera frame age: a FIFO goes stale, a latest-slot does not.

The stub backend sleeps on each grab (injected latency). One consumer runs
at a fixed Hz. The FIFO path keeps every frame and the consumer reads the
oldest, so frame_age grows. The supervisor path keeps one slot and drops the
rest, so steady-state frame_age p90 stays under 250 ms.
"""
from __future__ import annotations

import ast
import math
import subprocess
import sys
import threading
import time
import types
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import DriveCommand, release_command, stop_command  # noqa: E402
from python.runtime.frame_gate import (  # noqa: E402
    DEFAULT_MAX_FRAME_AGE_MS,
    STALE_CAPTION,
    frame_is_stale,
    gate_drive_command,
    require_max_frame_age_ms,
)
from python.sensors.cameras import (  # noqa: E402
    CAM_IDS,
    BeamNGPyBackend,
    CamHealth,
    CameraFrameBundle,
    load_camera_config,
    note_content_time,
)
from python.sensors.frame_slot import (  # noqa: E402
    CapturePump,
    LatestFrameSlot,
    LatencyStubBackend,
    bundle_frame_age_s,
    format_latency_line,
    percentile,
    process_rss_mb,
    rss_mb_from_statm,
)
from python.train.scene_ckpt_runtime import (  # noqa: E402
    SceneCkptPred,
    draw_scene_ckpt_overlay,
    scene_ckpt_overlay_caption,
)

TARGET_HZ = 10.0
MAX_AGE_MS = 250.0
LATENCY_S = 0.02


def _consume_ages(consume, *, seconds: float, hz: float, warmup_s: float) -> list[float]:
    period = 1.0 / hz
    ages: list[float] = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        time.sleep(period)
        age_s = consume()
        if age_s is None:
            continue
        if time.perf_counter() - t0 >= warmup_s:
            ages.append(float(age_s) * 1000.0)
    return ages


def _fifo_ages() -> tuple[list[float], int]:
    """The bug: every grab is queued, the consumer pops the oldest."""
    backend = LatencyStubBackend(LATENCY_S)
    queued: deque = deque()
    lock = threading.Lock()
    stop = threading.Event()

    def produce() -> None:
        while not stop.is_set():
            bundle = backend.grab()
            with lock:
                queued.append(bundle)

    thread = threading.Thread(target=produce, name="gvd-fifo", daemon=True)
    thread.start()

    def consume() -> float | None:
        with lock:
            bundle = queued.popleft() if queued else None
        if bundle is None:
            return None
        return bundle_frame_age_s(bundle)

    try:
        ages = _consume_ages(consume, seconds=1.8, hz=TARGET_HZ, warmup_s=0.4)
    finally:
        stop.set()
        thread.join(timeout=2.0)
    with lock:
        leftover = len(queued)
    return ages, leftover


def _latest_ages() -> tuple[list[float], int, float, float]:
    """The fix: one slot. Untaken frames are dropped."""
    backend = LatencyStubBackend(LATENCY_S)
    slot = LatestFrameSlot()
    pump = CapturePump(slot, backend.grab, idle_s=0.0)
    pump.start()

    def consume() -> float | None:
        held = slot.take()
        if held is None:
            return None
        return bundle_frame_age_s(held.payload)

    try:
        ages = _consume_ages(consume, seconds=1.6, hz=TARGET_HZ, warmup_s=0.3)
    finally:
        pump.stop()
    return ages, slot.dropped, percentile(ages, 0.50), percentile(ages, 0.90)


def check_queue_versus_latest() -> tuple[float, float, float, int]:
    fifo, leftover = _fifo_ages()
    assert len(fifo) >= 8, fifo
    fifo_p90 = percentile(fifo, 0.90)
    assert leftover > 1, leftover
    assert fifo_p90 > 400.0, (fifo_p90, leftover)
    latest, dropped, p50, p90 = _latest_ages()
    assert len(latest) >= 8, latest
    assert dropped > 0, dropped
    assert p90 < MAX_AGE_MS, (p50, p90, dropped, latest)
    assert p90 < fifo_p90
    print(
        f"fifo frame_age_p90_ms={fifo_p90:.1f} queued_left={leftover} "
        f"(stale frames kept)"
    )
    print(
        f"latest target_hz={TARGET_HZ:.0f} latency_ms={LATENCY_S * 1000:.0f} "
        f"frame_age_p50_ms={p50:.1f} frame_age_p90_ms={p90:.1f} "
        f"dropped={dropped} n={len(latest)}"
    )
    return p50, p90, fifo_p90, dropped


def check_gate() -> None:
    drive = DriveCommand(steer=0.25, throttle=0.4, brake=0.0, seq=4, reason="ok")
    fresh, replaced = gate_drive_command(drive, engaged=True, stale=False, holding=False)
    assert replaced is False and fresh.steer == 0.25 and fresh.reason == "ok"

    stale, replaced = gate_drive_command(drive, engaged=True, stale=True, holding=False)
    assert replaced is True
    assert stale.reason == "frame_stale"
    assert stale.steer == 0.0 and stale.throttle == 0.0 and stale.brake == 1.0
    assert stale.seq == 4

    idle = release_command(seq=2, reason="not_engaged")
    out, replaced = gate_drive_command(idle, engaged=False, stale=True)
    assert replaced is False and out.reason == "not_engaged" and out.brake == 0.0

    beat = release_command(seq=3, reason="heartbeat_stale")
    out, replaced = gate_drive_command(beat, engaged=True, stale=True, holding=True)
    assert replaced is False and out.reason == "heartbeat_stale" and out.brake == 0.0

    veto = stop_command(seq=5, reason="veto:low_lane_conf")
    out, replaced = gate_drive_command(veto, engaged=True, stale=True)
    assert replaced is False and out.reason == "veto:low_lane_conf" and out.brake == 1.0

    held = stop_command(seq=6, reason="preview_blocked")
    held.steer = 0.2
    out, replaced = gate_drive_command(held, engaged=True, stale=True, holding=True)
    assert replaced is True and out.reason == "frame_stale" and out.steer == 0.0

    waiting = CameraFrameBundle(note="capture: waiting")
    age = bundle_frame_age_s(waiting, now=100.0)
    assert math.isinf(age), age
    drive = DriveCommand(steer=0.2, throttle=0.4, brake=0.0, seq=9, reason="ok")
    stopped, replaced = gate_drive_command(
        drive, engaged=True, stale=frame_is_stale(age, DEFAULT_MAX_FRAME_AGE_MS),
    )
    assert replaced is True and stopped.reason == "frame_stale"
    assert stopped.steer == 0.0 and stopped.throttle == 0.0 and stopped.brake == 1.0
    print("gate: stale drive -> frame_stale; heartbeat/AEB veto/disengage unchanged")


def check_caption() -> None:
    pred = SceneCkptPred(lane_x=-0.4, confidence=0.91)
    assert scene_ckpt_overlay_caption(pred, drive=True).startswith("DRIVE")
    assert scene_ckpt_overlay_caption(pred, drive=False).startswith("SHADOW")
    assert scene_ckpt_overlay_caption(pred, drive=True, frame_stale=True) == STALE_CAPTION
    assert scene_ckpt_overlay_caption(None, frame_stale=True) == STALE_CAPTION
    frame = np.zeros((48, 180, 3), dtype=np.uint8)
    stale = draw_scene_ckpt_overlay(frame, pred, drive=True, frame_stale=True)
    fresh = draw_scene_ckpt_overlay(frame, pred, drive=True, frame_stale=False)
    assert stale is not None and fresh is not None
    assert not np.array_equal(stale, frame)
    assert not np.array_equal(stale, fresh)
    print(f"caption: stale frame draws {STALE_CAPTION}")


def check_content_stamp() -> None:
    stamps: dict[str, float] = {}
    sigs: dict[str, tuple] = {}
    first = note_content_time("main", ("a",), 10.0, stamps, sigs)
    sigs["main"] = ("a",)
    second = note_content_time("main", ("a",), 18.0, stamps, sigs)
    third = note_content_time("main", ("b",), 18.5, stamps, sigs)
    assert first == 10.0 and second == 10.0 and third == 18.5, (first, second, third)
    age = bundle_frame_age_s(types.SimpleNamespace(captured_at=10.0), now=10.2)
    assert abs(age - 0.2) < 1e-9, age
    missing = bundle_frame_age_s(types.SimpleNamespace(captured_at=0.0), now=50.0)
    assert math.isinf(missing), missing
    assert math.isinf(bundle_frame_age_s(types.SimpleNamespace(), now=50.0))
    assert frame_is_stale(missing, DEFAULT_MAX_FRAME_AGE_MS)
    print("content stamp: repeated buffer keeps its first monotonic time")


def check_needed_main_only() -> None:
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[0, 0, 0] = 40
    names: list[str] = []

    class FakeCamera:
        def __init__(self, name, _bng, _vehicle, **kwargs):
            self.name = name
            self.kwargs = kwargs
            self.is_streaming = True
            names.append(name)

        def stream_raw(self):
            return {"colour": pixels.copy()}

        def remove(self):
            return None

        def get_update_priority(self):
            return float(self.kwargs.get("update_priority", 0.0))

    sensors = types.ModuleType("beamngpy.sensors")
    sensors.Camera = FakeCamera
    beamngpy = types.ModuleType("beamngpy")
    beamngpy.sensors = sensors
    old = {k: sys.modules.get(k) for k in ("beamngpy", "beamngpy.sensors")}
    sys.modules["beamngpy"] = beamngpy
    sys.modules["beamngpy.sensors"] = sensors
    try:
        import python.control.actuate as act

        prev = act.soft_esc_sensors_every_tick()
        engage = act.engage_path()
        prev_bytes = engage.read_bytes() if engage.is_file() else None
        act.note_soft_esc_engaged(False)
        act.write_engage_flag(False)
        be = BeamNGPyBackend(
            config=load_camera_config(),
            tech_config={
                "wait_vehicle_s": 0,
                "cameras": {"attach": True, "rgb_only": True, "shared_memory": True, "streaming": True},
            },
        )
        be.set_needed_cameras(("main",))

        def _connect(explicit=True):
            be.session.vehicle = object()
            be.session.bng = object()
            return True

        be.session.connect = _connect  # type: ignore[method-assign]
        be.session.attach_vehicle_sensors = lambda: {}  # type: ignore[method-assign]
        be.open()
        assert set(be._sensors) == {"main"}, set(be._sensors)
        assert names == ["gvd_main"], names
        first = be.grab()
        assert first.health["main"] == CamHealth.OK
        assert "needed=main" in first.note
        assert "same_tick=8" not in first.note
        for cid in CAM_IDS:
            if cid == "main":
                continue
            assert cid not in first.frames
        time.sleep(0.05)
        second = be.grab()
        assert abs(second.captured_at - first.captured_at) < 1e-6, (first.captured_at, second.captured_at)
        assert abs(float(second.timestamps["main"]) - second.captured_at) < 1e-9
        assert time.monotonic() - second.captured_at >= 0.04
        pixels[0, 0, 0] = 90
        third = be.grab()
        assert third.captured_at > second.captured_at
        be.close()
        act.note_soft_esc_engaged(prev)
        if prev_bytes is None:
            engage.unlink(missing_ok=True)
        else:
            engage.write_bytes(prev_bytes)
    finally:
        for key, value in old.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    print("beamng grab: main-only attach; repeated shmem keeps captured_at")


def _call_name(node: ast.AST) -> tuple[str, str]:
    if not isinstance(node, ast.Call):
        return "", ""
    func = node.func
    if isinstance(func, ast.Name):
        return "", func.id
    if isinstance(func, ast.Attribute):
        base = func.value.id if isinstance(func.value, ast.Name) else ""
        return base, func.attr
    return "", ""


def check_gate_is_wired() -> None:
    """main() must gate before it applies or stops, and the cap default stays 300."""
    import python.run_vision as run_vision

    assert DEFAULT_MAX_FRAME_AGE_MS == 300.0
    src = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    main_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = [node for node in ast.walk(main_fn) if isinstance(node, ast.Call)]
    calls.sort(key=lambda node: (node.lineno, node.col_offset))
    gate_at: list[int] = []
    apply_at: list[int] = []
    stop_at: list[int] = []
    for call in calls:
        base, name = _call_name(call)
        if name == "gate_drive_command":
            gate_at.append(call.lineno)
        elif base == "actuator" and name == "apply":
            apply_at.append(call.lineno)
        elif base == "actuator" and name == "stop":
            stop_at.append(call.lineno)
    assert gate_at, "main stopped calling gate_drive_command"
    assert apply_at and stop_at
    assert min(gate_at) < min(apply_at), (gate_at, apply_at)
    assert min(gate_at) < min(stop_at), (gate_at, stop_at)
    found_default = False
    for call in calls:
        base, name = _call_name(call)
        if name != "add_argument":
            continue
        if not call.args or not isinstance(call.args[0], ast.Constant):
            continue
        if call.args[0].value != "--max-frame-age-ms":
            continue
        for keyword in call.keywords:
            if keyword.arg == "default":
                assert isinstance(keyword.value, ast.Name)
                assert keyword.value.id == "DEFAULT_MAX_FRAME_AGE_MS"
                found_default = True
    assert found_default
    assert any(_call_name(call)[1] == "require_max_frame_age_ms" for call in calls)
    assert run_vision.DEFAULT_MAX_FRAME_AGE_MS == 300.0
    print("run_vision: gate_drive_command precedes actuator.apply/stop; cap default is 300")


def check_bad_max_age() -> None:
    assert require_max_frame_age_ms(300) == 300.0
    for bad in (0, -1, float("nan"), float("inf"), float("-inf")):
        try:
            require_max_frame_age_ms(bad)
        except SystemExit as exc:
            assert exc.code == 2, (bad, exc.code)
        else:
            raise AssertionError(bad)
    for bad in ("nan", "inf", "-inf", "0", "-5"):
        proc = subprocess.run(
            [sys.executable, "-m", "python.run_vision", "--backend", "stub", "--max-frame-age-ms", bad],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode != 0, (bad, proc.returncode, proc.stdout[-400:], proc.stderr[-400:])
    print("max-frame-age-ms: non-finite and non-positive values are refused")


def check_mocked_clock() -> None:
    import python.sensors.frame_slot as slot

    def frozen() -> float:
        return 20.0

    real = slot.time.monotonic
    slot.time.monotonic = frozen
    try:
        fresh = slot.bundle_frame_age_s(types.SimpleNamespace(captured_at=19.5))
        assert abs(fresh - 0.5) < 1e-9, fresh
        ahead = slot.bundle_frame_age_s(types.SimpleNamespace(captured_at=25.0))
        assert ahead < 0.0, ahead
        assert frame_is_stale(ahead, DEFAULT_MAX_FRAME_AGE_MS)
        assert frame_is_stale(slot.bundle_frame_age_s(types.SimpleNamespace(captured_at=0.0)), 300)
    finally:
        slot.time.monotonic = real
    print("monotonic clock: negative age is stale; a mocked clock supplies now")


def check_latency_line(p50: float, p90: float, dropped: int) -> None:
    from python.runtime.state_io import default_state
    from python.viz.stage import render_stage

    state = default_state(engaged=False, loop_hz=TARGET_HZ)
    render_stage(state)
    viz_ms = float(state.get("viz_ms") or 0.0)
    rss = process_rss_mb()
    assert rss_mb_from_statm("0 2048 0 0", 4096) == 8.0
    assert viz_ms > 0.0, viz_ms
    assert rss > 1.0, rss
    line = format_latency_line(
        capture_ms=LATENCY_S * 1000.0,
        infer_ms=1.0,
        viz_ms=viz_ms,
        total_ms=viz_ms + LATENCY_S * 1000.0,
        frame_age_ms=p50,
        frame_age_p50_ms=p50,
        frame_age_p90_ms=p90,
        rss_mb=rss,
        dropped=dropped,
    )
    assert "capture_ms=" in line and "infer_ms=" in line and "viz_ms=" in line
    assert "total_ms=" in line and "frame_age_p50_ms=" in line and "frame_age_p90_ms=" in line
    assert "rss_mb=" in line and f"dropped={dropped}" in line
    print(line)
    print(f"rss_mb={rss:.1f} viz_ms={viz_ms:.2f}")


def main() -> None:
    check_content_stamp()
    check_gate()
    check_mocked_clock()
    check_bad_max_age()
    check_gate_is_wired()
    check_caption()
    check_needed_main_only()
    p50, p90, _fifo_p90, dropped = check_queue_versus_latest()
    check_latency_line(p50, p90, dropped)
    print("test_frame_staleness: OK")


if __name__ == "__main__":
    main()
