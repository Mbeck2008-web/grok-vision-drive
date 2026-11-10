#!/usr/bin/env python3
"""Camera stall: duplicate buffers age out, and a 20 Hz camera stays visible.

The capture pump reads shared memory much faster than the sensor's
requested_update_time. Each re-show is a new bundle with unique_gpu_n 0.
LatestFrameSlot keeps only the newest bundle, so the supervisor's 15 Hz
take usually observes unique=0 and camera_hz collapses, while the content
stamp on that duplicate is still the fresh unique frame. An identical
buffer must not move that stamp, so a camera that truly stops ages out
and the frame-age gate refuses to drive.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import DriveCommand  # noqa: E402
from python.runtime.frame_gate import (  # noqa: E402
    DEFAULT_MAX_FRAME_AGE_MS,
    frame_is_stale,
    gate_drive_command,
)
from python.runtime.hw_probe import ema_hz, unique_frame_hz_inst  # noqa: E402
from python.sensors.cameras import BeamNGPyBackend, CameraFrameBundle  # noqa: E402
from python.sensors.frame_slot import (  # noqa: E402
    CapturePump,
    LatestFrameSlot,
    bundle_frame_age_s,
)


def _bundle(unique: int, stamp: float, *, grab: int = 0) -> CameraFrameBundle:
    return CameraFrameBundle(
        note=f"beamngpy: 1 colour frame(s) grab={grab} unique={unique}",
        unique_gpu_n=unique,
        unique_gpu_ids=("main",) if unique else (),
        captured_at=stamp,
    )


def check_untaken_unique_survives_duplicate() -> None:
    """A duplicate re-show must not erase a unique frame the loop has not taken."""
    slot = LatestFrameSlot()
    slot.put(_bundle(1, 10.0, grab=0))
    slot.put(_bundle(0, 10.0, grab=1))
    held = slot.take()
    assert held is not None
    assert held.payload.unique_gpu_n >= 1, held.payload.unique_gpu_n
    assert held.payload.captured_at == 10.0
    assert "unique=0" not in held.payload.note
    # Consumed. Later duplicates are not a new frame.
    slot.put(_bundle(0, 10.0, grab=2))
    again = slot.take()
    assert again is not None
    assert again.payload.unique_gpu_n == 0, again.payload.unique_gpu_n
    assert again.payload.captured_at == 10.0
    print("slot: untaken unique survives a duplicate re-show; a taken one does not stick")


def _ready_backend(cam: object) -> BeamNGPyBackend:
    be = BeamNGPyBackend(
        long_side=64,
        config={"cameras": [], "hitch": {}, "origin_offset": [0.0, 0.0, 0.0]},
        tech_config={"wait_vehicle_s": 0, "launch": False, "host": "127.0.0.1", "port": 1},
    )
    be._ok = True
    be._sensors = {"main": cam}
    be._needed = frozenset({"main"})
    be._resolution = {"main": (16, 16)}
    be._live_narrow_hitch = False
    return be


class _Identical:
    def stream_raw(self) -> dict:
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        img[0, 0, 0] = 40
        return {"colour": img.copy()}


class _HzCamera:
    """New pixels every 1/hz seconds. Repeats the current frame in between."""

    def __init__(self, hz: float) -> None:
        self.hz = float(hz)
        self.t0 = time.perf_counter()
        self.reads = 0

    def stream_raw(self) -> dict:
        self.reads += 1
        n = int((time.perf_counter() - self.t0) * self.hz)
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        img[0, 0, 0] = (n % 200) + 1
        img[0, 0, 1] = (n // 200) % 255
        return {"colour": img}


def check_identical_buffer_ages_out() -> None:
    be = _ready_backend(_Identical())
    first = be.grab()
    assert first.unique_gpu_n == 1, first.unique_gpu_n
    assert first.captured_at > 0.0
    time.sleep(0.35)
    second = be.grab()
    assert second.unique_gpu_n == 0, second.unique_gpu_n
    assert abs(second.captured_at - first.captured_at) < 1e-6, (first.captured_at, second.captured_at)
    age = bundle_frame_age_s(second)
    assert age >= 0.30, age
    stale = frame_is_stale(age, DEFAULT_MAX_FRAME_AGE_MS)
    drive = DriveCommand(steer=0.2, throttle=0.4, brake=0.0, seq=3, reason="ok")
    stopped, replaced = gate_drive_command(drive, engaged=True, stale=stale)
    assert replaced is True and stopped.reason == "frame_stale"
    assert stopped.steer == 0.0 and stopped.throttle == 0.0 and stopped.brake == 1.0
    print(f"identical buffer keeps its stamp; age_s={age:.3f} refuses drive")


def check_supervisor_sees_20hz() -> None:
    """Pump the way run_vision does, take at the default 15 Hz loop."""
    cam = _HzCamera(20.0)
    be = _ready_backend(cam)
    slot = LatestFrameSlot()
    pump = CapturePump(slot, be.grab, idle_s=0.002)
    pump.start()
    cam_hz = 0.0
    last = time.perf_counter()
    unique_takes = 0
    takes = 0
    time.sleep(0.05)
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < 1.0:
            held = slot.take()
            if held is None and slot.wait(0.05):
                held = slot.take()
            if held is None:
                continue
            bundle = held.payload
            now = time.perf_counter()
            if takes > 0:
                dt = max(1e-6, now - last)
                n = int(getattr(bundle, "unique_gpu_n", 0) or 0)
                cam_hz = ema_hz(cam_hz, unique_frame_hz_inst(n, dt))
                if n > 0:
                    unique_takes += 1
            last = now
            takes += 1
            time.sleep(1.0 / 15.0)
    finally:
        pump.stop()
    assert takes >= 8, takes
    assert unique_takes > 0, unique_takes
    assert cam_hz >= 10.0, (cam_hz, unique_takes, takes, cam.reads, slot.dropped)
    print(
        f"20 Hz camera via supervisor poll: camera_hz={cam_hz:.2f} "
        f"unique_takes={unique_takes}/{takes} reads={cam.reads} dropped={slot.dropped}"
    )


def main() -> None:
    check_untaken_unique_survives_duplicate()
    check_identical_buffer_ages_out()
    check_supervisor_sees_20hz()
    print("test_camera_stall: OK")


if __name__ == "__main__":
    main()
