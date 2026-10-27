#!/usr/bin/env python3
"""The 6 s launch window on retail, run headless with a painted road (#125, option 2).

The retail LaunchGuard now gets the session-echo speed. These checks run run_vision.py on the
retail cmd_json bus through scripts/gvd_road_stub_child.py (stub cameras that return a painted
road, so the planner has a path) with a fake mod that integrates the commanded pedals into its
echoed speed.

1. Known speed: a fresh echo at 0 m/s pulls away inside the window. Throttle goes above 0 and
   the echoed speed rises.
2. Known speed: above LAUNCH_SPEED_MPS inside the window the command is throttle 0 with the
   0.2 launch brake, and the car stays near that speed.
3. Unknown speed inside the window: the echo freezes, and the supervisor disengages with
   echo_stale on every surface (thr 0 / brk 0). There is no silent throttle-0 hold.
"""
from __future__ import annotations

import contextlib
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import test_unknown_ego as T  # noqa: E402
from gvd_test_env import isolated_home  # noqa: E402
from python.control.actuate import cmd_path, write_engage_flag  # noqa: E402
from python.control.launch_guard import LAUNCH_S, LAUNCH_SPEED_MPS, LAUNCH_THROTTLE  # noqa: E402

LAUNCHER = ROOT / "scripts" / "gvd_road_stub_child.py"
ACCEL_PER_THROTTLE = 10.0  # m/s^2 at throttle 1.0 (fake car)
DECEL_PER_BRAKE = 8.0  # m/s^2 at brake 1.0
ENGAGE_AT_S = 0.5
DROP_BAR_S = 0.6  # same bar as test_unknown_ego test 9


class _Car(T._Mod):
    """The fake mod, plus a car: echoed speed follows the commanded pedals while engaged."""

    def __init__(self) -> None:
        super().__init__(speed=0.0)
        self.echoing = True
        self._last: float | None = None

    def echo(self, t_spawn: float) -> None:
        now = time.monotonic()
        if now >= self.next_echo and self.echoing and self.frozen is None:
            cmd = T._read(cmd_path())
            if self._last is not None and cmd.get("engaged") is True:
                acc = (ACCEL_PER_THROTTLE * float(cmd.get("throttle") or 0.0)
                       - DECEL_PER_BRAKE * float(cmd.get("brake") or 0.0))
                self.speed = max(0.0, self.speed + acc * (now - self._last))
            self._last = now
        super().echo(t_spawn)


@contextlib.contextmanager
def _road_launcher():
    """T._session spawns run_vision.py; run it through the painted-road launcher instead."""
    real = subprocess.Popen

    def popen(args, *a, **k):
        if len(args) > 2 and str(args[2]).endswith("run_vision.py"):
            args = list(args)
            args[2] = str(LAUNCHER)
        return real(args, *a, **k)

    subprocess.Popen = popen
    try:
        yield
    finally:
        subprocess.Popen = real


def _speed(snap: dict) -> float | None:
    v = (snap["state"].get("ego") or {}).get("speed_mps")
    return None if v is None else float(v)


def _drive(seconds: float, *, freeze_after_s: float | None = None) -> tuple[list[dict], dict]:
    mark: dict = {}
    with isolated_home(prefix="gvd-launch-"), _road_launcher():
        write_engage_flag(False)
        car = _Car()

        def step(t: float) -> None:
            if "engage" not in mark and t >= ENGAGE_AT_S:
                mark["engage"] = t
                write_engage_flag(True)
            if freeze_after_s is not None and "freeze" not in mark and "engage" in mark:
                cmd = T._read(cmd_path())
                if (cmd.get("engaged") is True and float(cmd.get("throttle") or 0.0) > 0.0
                        and t >= mark["engage"] + freeze_after_s):
                    mark["freeze"] = t
                    car.freeze()

        seen = T._session(seconds, car, step)
    return seen, mark


def check_known_speed_pulls_away_and_launch_brake() -> None:
    seen, mark = _drive(5.5)
    on = [s for s in seen if s["cmd"].get("engaged") is True]
    assert on, ("never engaged", sorted({str(s["cmd"].get("reason")) for s in seen}))
    t_on = on[0]["t"]
    assert all(s["t"] - t_on < LAUNCH_S for s in on), "run must stay inside the launch window"
    off = [s["state"].get("disengage_reason") for s in seen if s["t"] > t_on and s["cmd"].get("engaged") is not True]
    assert not off, ("disengaged during the launch", off[:5])
    # 1. Pull away: throttle above 0 at a measured 0 m/s, then the echoed speed rises.
    slow = [s for s in on if (_speed(s) is not None and _speed(s) < 0.5)]
    assert any(float(s["cmd"].get("throttle") or 0.0) > 0.0 for s in slow), \
        ("no throttle at a known 0 m/s", [(s["cmd"].get("throttle"), _speed(s)) for s in on[:5]])
    speeds = [_speed(s) for s in on if _speed(s) is not None]
    assert speeds and max(speeds) >= 2.0, ("speed did not rise", speeds[:3], max(speeds or [0.0]))
    thr = [float(s["cmd"].get("throttle") or 0.0) for s in on]
    assert max(thr) <= LAUNCH_THROTTLE + 1e-9, ("launch throttle cap", max(thr))
    # 2. Launch brake: above LAUNCH_SPEED_MPS the command is thr 0 with the 0.2 brake.
    braked = [s for s in on if float(s["cmd"].get("brake") or 0.0) > 0.0]
    assert braked, ("launch brake never applied", max(speeds))
    assert all(float(s["cmd"].get("throttle") or 0.0) == 0.0 for s in braked), "brake with throttle"
    assert all(abs(float(s["cmd"].get("brake")) - 0.2) < 1e-9 for s in braked), \
        sorted({s["cmd"].get("brake") for s in braked})
    early = [s for s in on if _speed(s) is not None and _speed(s) < LAUNCH_SPEED_MPS - 0.5]
    assert not any(float(s["cmd"].get("brake") or 0.0) > 0.0 for s in early), "launch brake below launch speed"
    assert max(speeds) < LAUNCH_SPEED_MPS + 1.0, ("launch window did not hold the speed", max(speeds))


def check_unknown_speed_in_window_disengages() -> None:
    seen, mark = _drive(5.0, freeze_after_s=1.5)
    assert "freeze" in mark, ("never drove with throttle > 0, so the echo was not frozen", mark)
    on = [s for s in seen if s["cmd"].get("engaged") is True]
    assert on, "never engaged"
    t_on, t_freeze = on[0]["t"], mark["freeze"]
    assert t_freeze - t_on < LAUNCH_S - 1.0, ("freeze must be inside the launch window", t_on, t_freeze)
    T._refusal(seen, "echo_stale", "echo froze inside the launch window")
    drops = [s["t"] for s in seen if s["t"] >= t_freeze and s["state"].get("disengage_reason") == "echo_stale"]
    assert drops and drops[0] - t_freeze <= DROP_BAR_S, ("echo_stale too late", drops[:1], t_freeze)
    held = [(round(s["t"], 2), s["cmd"].get("reason"), s["cmd"].get("throttle"), s["cmd"].get("brake"))
            for s in seen if s["t"] > t_freeze + DROP_BAR_S and s["cmd"].get("engaged") is True]
    assert not held, ("still engaged after the echo froze: a silent hold", held[:5])


CHECKS = (
    check_known_speed_pulls_away_and_launch_brake,
    check_unknown_speed_in_window_disengages,
)


def main() -> None:
    failed: list[str] = []
    for fn in CHECKS:
        try:
            fn()
            print(f"{fn.__name__}: ok", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            kind = "FAIL" if isinstance(exc, AssertionError) else f"ERROR {type(exc).__name__}"
            print(f"{fn.__name__}: {kind} {exc!r}"[:2000], flush=True)
            if not isinstance(exc, AssertionError):
                traceback.print_exc()
    if failed:
        raise SystemExit("test_launch_window failed: " + ", ".join(failed))
    print("test_launch_window: OK")


if __name__ == "__main__":
    main()
