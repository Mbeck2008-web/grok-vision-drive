#!/usr/bin/env python3
"""The drive gate must re-read frame age at the decision, not reuse the take.

A mocked ``time.monotonic`` reports the bundle ~50 ms old at the grab sample
and past 300 ms at the second ``bundle_frame_age_s`` call, the one immediately
before ``gate_drive_command``. Reusing the take-time value would leave an
engaged ``ok`` command driving and would log ~50 ms.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from python.sensors.cameras import CAM_IDS, CamHealth, CameraFrameBundle  # noqa: E402

CAPTURED_AT = 1000.0
TAKE_AGE_S = 0.050
DECISION_AGE_S = 0.450


def _road() -> np.ndarray:
    import cv2

    img = np.full((96, 160, 3), (40, 42, 44), dtype=np.uint8)
    cv2.line(img, (30, 90), (70, 30), (40, 210, 235), 3)
    cv2.line(img, (130, 90), (90, 30), (245, 245, 245), 3)
    return img


class _AgeClock:
    """Only ``bundle_frame_age_s`` reads this. The first read is the take."""

    def __init__(self) -> None:
        self.calls = 0

    def monotonic(self) -> float:
        self.calls += 1
        if self.calls == 1:
            return CAPTURED_AT + TAKE_AGE_S
        return CAPTURED_AT + DECISION_AGE_S


class _OneTick:
    def __init__(self) -> None:
        self.t = 1000.0
        self.snap: dict | None = None

    def mono(self) -> float:
        return self.t

    def perf(self) -> float:
        return self.t

    def wall(self) -> float:
        import time

        return time.time()

    def sleep(self, _seconds: float) -> None:
        from python.control.actuate import cmd_path

        self.snap = json.loads(cmd_path().read_text(encoding="utf-8"))
        raise SystemExit(0)


def _grab(self) -> CameraFrameBundle:
    del self
    # Baseline echo is taken before the loop. A new echo during the grab is
    # this session's speed, so --force-engage can arm an ok command.
    import test_m6_retail as retail

    retail._write_ego(1, applying=True, speed=5.5)
    health = {cid: CamHealth.MISSING for cid in CAM_IDS}
    health["main"] = CamHealth.OK
    image = _road()
    return CameraFrameBundle(
        frames={"main": image, "cam_main": image},
        timestamps={"main": CAPTURED_AT},
        health=health,
        backend="stub",
        note="decision-age road",
        unique_gpu_n=1,
        captured_at=CAPTURED_AT,
    )


def _logged_age_ms(text: str) -> float | None:
    """Decision age from the stale line, else from the lat line."""
    for line in text.splitlines():
        if line.startswith("[GVD] frame stale "):
            for part in line.split():
                if part.startswith("age_ms="):
                    return float(part.split("=", 1)[1])
        if line.startswith("[GVD] lat "):
            for part in line.split():
                if part.startswith("frame_age_ms="):
                    return float(part.split("=", 1)[1])
    return None


def check_decision_age_not_take_age() -> None:
    import test_m6_retail as retail
    from gvd_test_env import isolated_home

    import python.run_vision as run_vision
    import python.sensors.cameras as cameras
    import python.sensors.frame_slot as frame_slot

    clock = _AgeClock()
    previous_time = frame_slot.time
    previous_grab = cameras.StubBackend.grab
    previous_argv = sys.argv
    stdout = io.StringIO()
    real_stdout = sys.stdout
    tick = _OneTick()
    try:
        frame_slot.time = clock
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
        with isolated_home(prefix="gvd-decision-age-"):
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
        frame_slot.time = previous_time

    text = stdout.getvalue()
    snap = tick.snap
    logged = _logged_age_ms(text)
    assert snap is not None, "supervisor did not finish a tick"
    assert snap.get("engaged") is True, snap
    assert snap.get("reason") == "frame_stale", (snap.get("reason"), logged, snap)
    assert float(snap.get("brake") or 0.0) == 1.0, snap
    assert float(snap.get("steer") or 0.0) == 0.0 and float(snap.get("throttle") or 0.0) == 0.0, snap
    assert logged is not None and logged > 300.0, (logged, text[-800:])
    assert abs(logged - (TAKE_AGE_S * 1000.0)) > 1.0, logged
    assert clock.calls >= 2, clock.calls
    print(
        f"decision age: take {TAKE_AGE_S * 1000:.0f} ms, "
        f"logged {logged:.0f} ms, reason {snap.get('reason')} brake {snap.get('brake')}"
    )


def main() -> None:
    check_decision_age_not_take_age()
    print("test_frame_age_decision: OK")


if __name__ == "__main__":
    main()
