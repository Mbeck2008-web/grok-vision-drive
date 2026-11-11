#!/usr/bin/env python3
"""Control-loop rate: stage log, paced camera grab, viz off the control thread.

The live scene-ckpt loop on IQU45 sat near 2.3-3.8 Hz with unique=1 per tick.
A capture pump that grabs again after 2 ms re-takes the shared camera socket
(or the GIL) before the control thread can run inference, so the loop rate
collapses even though each tick still sees one new frame. Pacing the pump to
the control period leaves that resource free between grabs. Drawing the
stage on the control thread was the slow stub stage (~48 ms); the viz pump
keeps only the latest paint and does not block actuation.

Stale-frame refuse and heartbeat_stale are unchanged.

Measure on IQU45 (one GPU job, BeamNG.tech already up):

    py -3 -m python.run_vision --backend beamngpy --viz --scene-ckpt <ep11.npz> --scene-ckpt-drive --scene-ckpt-drive-speed 15 --live-long-side 640 --no-conf-gate --precision fp32

Read ``[GVD] stage`` for grab_ms, infer_ms, viz_ms, actuate_ms and loop_hz.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import DriveCommand, release_command  # noqa: E402
from python.runtime.frame_gate import frame_is_stale, gate_drive_command  # noqa: E402
from python.sensors.frame_slot import (  # noqa: E402
    CapturePump,
    LatestFrameSlot,
    LatencyStubBackend,
    bundle_frame_age_s,
    format_stage_line,
)
from python.viz.present import VizPump  # noqa: E402


TARGET_HZ = 10.0
IQU45_CMD = (
    "py -3 -m python.run_vision --backend beamngpy --viz "
    "--scene-ckpt <ep11.npz> --scene-ckpt-drive --scene-ckpt-drive-speed 15 "
    "--live-long-side 640 --no-conf-gate --precision fp32"
)


def _contended_hz(period_s: float, idle_s: float, *, seconds: float = 1.0) -> tuple[float, float]:
    """Grab holds a lock the control tick also needs. Returns (loop_hz, grab_hz)."""
    backend = LatencyStubBackend(0.015)
    lock = threading.Lock()

    def produce():
        with lock:
            return backend.grab()

    slot = LatestFrameSlot()
    pump = CapturePump(slot, produce, idle_s=idle_s, period_s=period_s)
    pump.start()
    time.sleep(0.05)
    ticks = 0
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < seconds:
            with lock:
                time.sleep(0.05)
            slot.take()
            ticks += 1
    finally:
        pump.stop()
    elapsed = max(1e-6, time.perf_counter() - t0)
    return ticks / elapsed, backend.grabs / elapsed


def _grab_hz(period_s: float, idle_s: float, *, seconds: float = 0.45) -> float:
    backend = LatencyStubBackend(0.0)
    slot = LatestFrameSlot()
    pump = CapturePump(slot, backend.grab, idle_s=idle_s, period_s=period_s)
    pump.start()
    time.sleep(seconds)
    pump.stop()
    return backend.grabs / seconds


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


def check_paced_grab_does_not_spin() -> None:
    spun = _grab_hz(0.0, 0.002)
    paced = _grab_hz(1.0 / 15.0, 0.0)
    print(f"stub grab_hz spin_idle_2ms={spun:.1f} paced_15hz={paced:.1f}")
    assert spun > 80.0, spun
    assert 8.0 <= paced <= 25.0, paced


def check_paced_pump_keeps_the_control_rate() -> None:
    """Shared lock: the spinning pump and the paced pump.

    Which thread wins the lock depends on the scheduler, so the spinning
    loop rate is not a fixed number. The paced pump must still clear 10 Hz,
    and it must grab far less often than the 2 ms idle pump.
    """
    spin_hz, spin_grabs = _contended_hz(0.0, 0.002)
    paced_hz, paced_grabs = _contended_hz(1.0 / 15.0, 0.0)
    print(
        f"stub contended loop_hz before={spin_hz:.2f} (grab_hz={spin_grabs:.1f}) "
        f"after={paced_hz:.2f} (grab_hz={paced_grabs:.1f})"
    )
    assert paced_hz >= TARGET_HZ, (paced_hz, paced_grabs)
    assert spin_grabs > paced_grabs * 2.0, (spin_grabs, paced_grabs)


def _control_hz_with_viz(*, off_thread: bool, seconds: float = 0.7) -> tuple[float, int]:
    """30 ms of control work plus an 80 ms paint. Inline paint misses 10 Hz."""
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


def check_viz_off_thread_raises_loop_hz() -> None:
    inline_hz, inline_n = _control_hz_with_viz(off_thread=False)
    pumped_hz, pumped_n = _control_hz_with_viz(off_thread=True)
    print(
        f"stub viz loop_hz before={inline_hz:.2f} (paints={inline_n}) "
        f"after={pumped_hz:.2f} (paints={pumped_n})"
    )
    assert inline_hz < TARGET_HZ, inline_hz
    assert pumped_hz >= 15.0, pumped_hz
    assert pumped_n < pumped_hz * 0.7, (pumped_n, pumped_hz)


def check_viz_does_not_block_the_control_thread() -> None:
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
    print(f"viz off control thread: submit_12_ms={elapsed_ms:.1f} painted={paints['n']}")


def check_stale_and_heartbeat_unchanged() -> None:
    backend = LatencyStubBackend(0.0)
    slot = LatestFrameSlot()
    pump = CapturePump(slot, backend.grab, idle_s=0.0, period_s=0.05)
    pump.start()
    time.sleep(0.12)
    try:
        held = slot.take()
    finally:
        pump.stop()
    assert held is not None
    held.payload.captured_at = time.monotonic() - 0.35
    age = bundle_frame_age_s(held.payload)
    assert frame_is_stale(age, 300.0), age
    drive = DriveCommand(steer=0.2, throttle=0.3, brake=0.0, seq=4, reason="ok")
    stopped, replaced = gate_drive_command(drive, engaged=True, stale=True, holding=False)
    assert replaced is True and stopped.reason == "frame_stale"
    assert stopped.steer == 0.0 and stopped.throttle == 0.0 and stopped.brake == 1.0
    beat = release_command(seq=5, reason="heartbeat_stale")
    out, replaced = gate_drive_command(beat, engaged=True, stale=True, holding=True)
    assert replaced is False and out.reason == "heartbeat_stale" and out.brake == 0.0
    print(f"paced grab: stale age_s={age:.3f} refuses drive; heartbeat_stale kept")
    print(f"IQU45: {IQU45_CMD}")


def main() -> None:
    check_stage_line()
    check_paced_grab_does_not_spin()
    check_paced_pump_keeps_the_control_rate()
    check_viz_off_thread_raises_loop_hz()
    check_viz_does_not_block_the_control_thread()
    check_stale_and_heartbeat_unchanged()
    print("test_loop_rate: OK")


if __name__ == "__main__":
    main()
