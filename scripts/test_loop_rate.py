#!/usr/bin/env python3
"""Control-loop rate: phased camera grab, stage log, viz handoff.

The live scene-ckpt loop on IQU45 sat near 2.3-3.8 Hz with unique=1 per tick.
A capture pump that grabbed again after 2 ms could hold the camera socket.
Sleeping a full control period after each grab made the taken frame about one
period old, so AEB saw stale pixels. The pump now starts the grab so it
finishes just before the next take. Age at take stays under one third of the
control period. The rest of the 300 ms frame_stale gate is left for the tick.

Viz submit does not wait for a paint that releases the GIL (sleep, waitKey).
A paint that holds the GIL still stalls the control thread. The pump does not
move paint into another process.

Stale-frame refuse goes through run_vision.main. Heartbeat stops are unchanged.

Measure on IQU45 (one GPU job, BeamNG.tech already up):

    py -3 -m python.run_vision --backend beamngpy --viz --scene-ckpt <ep11.npz> --scene-ckpt-drive --scene-ckpt-drive-speed 15 --live-long-side 640 --no-conf-gate --precision fp32

Read ``[GVD] stage`` for grab_ms, infer_ms, viz_ms, actuate_ms and loop_hz.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from python.runtime.frame_gate import DEFAULT_MAX_FRAME_AGE_MS  # noqa: E402
from python.sensors.cameras import CAM_IDS, CamHealth, CameraFrameBundle  # noqa: E402
from python.sensors.frame_slot import (  # noqa: E402
    CapturePump,
    LatestFrameSlot,
    LatencyStubBackend,
    bundle_frame_age_s,
    format_stage_line,
)
from python.viz.present import VizPump, paint_job_snapshot  # noqa: E402


TARGET_HZ = 10.0
IQU45_CMD = (
    "py -3 -m python.run_vision --backend beamngpy --viz "
    "--scene-ckpt <ep11.npz> --scene-ckpt-drive --scene-ckpt-drive-speed 15 "
    "--live-long-side 640 --no-conf-gate --precision fp32"
)


def _hold_gil(seconds: float) -> int:
    end = time.perf_counter() + float(seconds)
    x = 0
    while time.perf_counter() < end:
        x = (x + 1) % 1000003
    return x


def check_stage_line() -> None:
    line = format_stage_line(
        grab_ms=12.5,
        grab_wait_ms=0.4,
        infer_ms=80.0,
        actuate_ms=3.0,
        viz_ms=48.0,
        viz_wait_ms=0.2,
        tick_ms=90.0,
        loop_hz=9.5,
        ckpt_ms=70.0,
    )
    for key in (
        "grab_ms=12.50",
        "grab_wait_ms=0.40",
        "infer_ms=80.00",
        "actuate_ms=3.00",
        "viz_ms=48.00",
        "viz_wait_ms=0.20",
        "tick_ms=90.00",
        "loop_hz=9.50",
        "ckpt_ms=70.00",
    ):
        assert key in line, (key, line)
    assert line.startswith("[GVD] stage ")
    print(line)


def check_phased_grab_age() -> None:
    """Taken frames are younger than one third of the control period.

    The leftover of the 300 ms frame_stale gate must still cover a full
    control period of AEB and actuation after the take.
    """
    period = 1.0 / TARGET_HZ
    backend = LatencyStubBackend(0.004)
    slot = LatestFrameSlot()
    pump = CapturePump(slot, backend.grab, idle_s=0.002, phase_s=period)
    pump.start()
    ages: list[float] = []
    try:
        assert slot.wait(1.0)
        slot.take()
        nxt = time.perf_counter() + period
        for _ in range(8):
            pump.note_next_take(nxt)
            delay = nxt - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)
            held = slot.take()
            assert held is not None
            ages.append(bundle_frame_age_s(held.payload))
            nxt = time.perf_counter() + period
    finally:
        pump.stop()
    steady = ages[2:]
    assert steady, ages
    oldest = max(steady)
    cap = period / 3.0
    gate_s = DEFAULT_MAX_FRAME_AGE_MS / 1000.0
    remainder = gate_s - oldest
    print(
        f"phased take age_s p50={sorted(steady)[len(steady)//2]:.4f} "
        f"max={oldest:.4f} cap={cap:.4f} stale_remainder_s={remainder:.4f} "
        f"grabs={backend.grabs}"
    )
    assert oldest < cap, (steady, cap)
    assert remainder > period, (remainder, period, oldest)
    assert backend.grabs < 40, backend.grabs


def check_phased_pump_keeps_the_control_rate() -> None:
    """Control work that shares a lock with grab still reaches 10 Hz.

    Grab counts are not compared with a ratio. That comparison moved with
    the scheduler on a busy host.
    """
    period = 1.0 / TARGET_HZ
    backend = LatencyStubBackend(0.008)
    lock = threading.Lock()

    def produce():
        with lock:
            return backend.grab()

    slot = LatestFrameSlot()
    pump = CapturePump(slot, produce, idle_s=0.002, phase_s=period)
    pump.start()
    ticks = 0
    t0 = time.perf_counter()
    try:
        assert slot.wait(1.0)
        while time.perf_counter() - t0 < 0.8:
            with lock:
                time.sleep(0.02)
            slot.take()
            now = time.perf_counter()
            remain = max(0.0, period - (now - t0) % period)
            # Schedule the next take on the period grid from t0.
            pump.note_next_take(now + remain)
            time.sleep(remain)
            ticks += 1
    finally:
        pump.stop()
    elapsed = max(1e-6, time.perf_counter() - t0)
    hz = ticks / elapsed
    print(f"phased contended loop_hz={hz:.2f} grabs={backend.grabs} ticks={ticks}")
    assert hz >= TARGET_HZ * 0.8, (hz, ticks, backend.grabs)


def _control_hz_with_sleeping_paint(*, off_thread: bool, seconds: float = 0.7) -> tuple[float, int]:
    """30 ms of control work plus an 80 ms paint that releases the GIL."""
    painted = {"n": 0}

    def paint(job: dict) -> int:
        del job
        painted["n"] += 1
        time.sleep(0.08)
        return -1

    pump = VizPump(paint) if off_thread else None
    if pump is not None:
        pump.start()
    ticks = 0
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < seconds:
            time.sleep(0.03)
            if pump is not None:
                pump.submit({"i": ticks})
            else:
                paint({"i": ticks})
            ticks += 1
    finally:
        if pump is not None:
            pump.stop()
    elapsed = max(1e-6, time.perf_counter() - t0)
    return ticks / elapsed, painted["n"]


def check_sleeping_paint_overlaps_control() -> None:
    inline_hz, inline_n = _control_hz_with_sleeping_paint(off_thread=False)
    pumped_hz, pumped_n = _control_hz_with_sleeping_paint(off_thread=True)
    print(
        f"stub viz loop_hz sleep-paint before={inline_hz:.2f} (paints={inline_n}) "
        f"after={pumped_hz:.2f} (paints={pumped_n})"
    )
    assert inline_hz < TARGET_HZ, inline_hz
    assert pumped_hz >= 15.0, pumped_hz
    assert pumped_n < pumped_hz * 0.7, (pumped_n, pumped_hz)


def _wall_until_control_resumes(paint) -> float:
    """Time from submit until a short control slice finishes.

    ``time.sleep`` cannot return while the paint holds the GIL, so a
    GIL-holding paint shows up in this wall time. A paint that sleeps does not.
    """
    pump = VizPump(paint)
    pump.start()
    try:
        t0 = time.perf_counter()
        pump.submit({"i": 0})
        time.sleep(0.01)
        target = time.perf_counter() + 0.02
        while time.perf_counter() < target:
            pass
        return time.perf_counter() - t0
    finally:
        pump.stop()


def check_gil_holding_paint_stalls_control() -> None:
    """A paint that holds the GIL is not overlapped with AEB."""
    text = (ROOT / "python" / "viz" / "present.py").read_text(encoding="utf-8")
    assert "holds the GIL" in text

    def sleeping(job: dict) -> int:
        del job
        time.sleep(0.20)
        return -1

    def holding(job: dict) -> int:
        del job
        # The default switch interval lets a busy loop drop the GIL every few
        # milliseconds. Stretch it so this paint actually keeps the GIL.
        old = sys.getswitchinterval()
        sys.setswitchinterval(1.0)
        try:
            _hold_gil(0.20)
        finally:
            sys.setswitchinterval(old)
        return -1

    sleep_wall = _wall_until_control_resumes(sleeping)
    hold_wall = _wall_until_control_resumes(holding)
    print(f"paint overlap sleep_wall_s={sleep_wall:.3f} gil_wall_s={hold_wall:.3f}")
    assert sleep_wall < 0.08, sleep_wall
    assert hold_wall > 0.15, hold_wall


def check_viz_submit_returns() -> None:
    paints = {"n": 0}

    def paint(job: dict) -> int:
        paints["n"] += 1
        time.sleep(0.04)
        return -1 if job.get("quit") else int(job.get("key", -1))

    pump = VizPump(paint)
    pump.start()
    try:
        t0 = time.perf_counter()
        for i in range(12):
            pump.submit({"i": i, "key": ord("v") if i == 11 else -1})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 80.0, elapsed_ms
        deadline = time.perf_counter() + 1.0
        saw_key = False
        while time.perf_counter() < deadline:
            ev = pump.poll_event()
            if ev == ("key", ord("v")):
                saw_key = True
                break
            time.sleep(0.01)
        assert saw_key, paints
        assert paints["n"] < 12, paints
        assert paints["n"] >= 1
    finally:
        pump.stop()
    print(f"viz submit returns: submit_12_ms={elapsed_ms:.1f} painted={paints['n']}")


def check_paint_job_is_one_snapshot() -> None:
    dets = [{"cls": "car", "conf": 0.91}]
    other = [{"cls": "sign", "conf": 0.2}]
    state = {"cmd_seq": 4, "objects_n": 1}
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    job = paint_job_snapshot(
        state=state,
        main=image,
        frames={"main": image},
        dets=dets,
        matched=True,
        screen="primary",
        fullscreen=False,
    )
    assert job is not None
    dets.clear()
    dets.extend(other)
    state["cmd_seq"] = 9
    image[0, 0, 0] = 255
    assert job["dets"] == [{"cls": "car", "conf": 0.91}]
    assert job["dets"] != other
    assert job["state"]["cmd_seq"] == 4
    assert int(job["main"][0, 0, 0]) == 0
    assert job["state"] is not state
    assert job["dets"] is not dets

    class _NoCopy:
        def __deepcopy__(self, _memo):
            raise RuntimeError("live state")

    refused = paint_job_snapshot(
        state={"bad": _NoCopy()},
        main=None,
        frames=None,
        dets=[{"cls": "car"}],
        matched=True,
        screen="",
        fullscreen=False,
    )
    assert refused is None
    print("paint job: swapping the live dets leaves the snapshot on the previous tick")


def check_frozen_frame_through_main() -> None:
    """A captured_at 350 ms ago reaches frame_stale through run_vision.main."""
    import io

    import test_m6_retail as retail
    from gvd_test_env import isolated_home

    import python.run_vision as run_vision
    import python.sensors.cameras as cameras

    class _OneTick:
        def __init__(self) -> None:
            self.t = 1000.0
            self.snap: dict | None = None

        def mono(self) -> float:
            return self.t

        def perf(self) -> float:
            return self.t

        def wall(self) -> float:
            return time.time()

        def sleep(self, _seconds: float) -> None:
            from python.control.actuate import cmd_path

            self.snap = json.loads(cmd_path().read_text(encoding="utf-8"))
            raise SystemExit(0)

    frozen_at = time.monotonic() - 0.35

    def _grab(self) -> CameraFrameBundle:
        del self
        retail._write_ego(1, applying=True, speed=5.5)
        health = {cid: CamHealth.MISSING for cid in CAM_IDS}
        health["main"] = CamHealth.OK
        image = np.full((96, 160, 3), (40, 42, 44), dtype=np.uint8)
        try:
            import cv2

            cv2.line(image, (30, 90), (70, 30), (40, 210, 235), 3)
            cv2.line(image, (130, 90), (90, 30), (245, 245, 245), 3)
        except Exception:
            pass
        return CameraFrameBundle(
            frames={"main": image, "cam_main": image},
            timestamps={"main": frozen_at},
            health=health,
            backend="stub",
            note="frozen captured_at",
            unique_gpu_n=1,
            captured_at=frozen_at,
        )

    previous_grab = cameras.StubBackend.grab
    previous_argv = sys.argv
    stdout = io.StringIO()
    real_stdout = sys.stdout
    tick = _OneTick()
    try:
        cameras.StubBackend.grab = _grab
        sys.argv = [
            str(ROOT / "python" / "run_vision.py"),
            "--backend",
            "stub",
            "--hz",
            "20",
            "--encode",
            "cpu",
            "--force-engage",
        ]
        sys.stdout = stdout
        with isolated_home(prefix="gvd-frozen-frame-"):
            retail._touch_lua_bus()
            retail._write_ego(1, applying=True, speed=5.5)
            retail.write_engage_flag(True)
            try:
                run_vision.main(clock=tick)
            except SystemExit:
                pass
    finally:
        sys.stdout = real_stdout
        sys.argv = previous_argv
        cameras.StubBackend.grab = previous_grab

    text = stdout.getvalue()
    snap = tick.snap
    assert snap is not None, f"supervisor did not finish a tick: {text[-1500:]}"
    assert snap.get("reason") == "frame_stale", (snap.get("reason"), snap, text[-800:])
    assert float(snap.get("brake") or 0.0) == 1.0, snap
    assert float(snap.get("steer") or 0.0) == 0.0, snap
    assert float(snap.get("throttle") or 0.0) == 0.0, snap
    print(f"main frozen captured_at: reason={snap.get('reason')} brake={snap.get('brake')}")
    print(f"IQU45: {IQU45_CMD}")


def main() -> None:
    check_stage_line()
    check_phased_grab_age()
    check_phased_pump_keeps_the_control_rate()
    check_sleeping_paint_overlaps_control()
    check_gil_holding_paint_stalls_control()
    check_viz_submit_returns()
    check_paint_job_is_one_snapshot()
    check_frozen_frame_through_main()
    print("test_loop_rate: OK")


if __name__ == "__main__":
    main()
