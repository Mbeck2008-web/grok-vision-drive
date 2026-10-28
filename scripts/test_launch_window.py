#!/usr/bin/env python3
"""The 6 s launch window on retail, run headless with a painted road (#125, option 2).

The retail LaunchGuard now gets the session-echo speed. These checks run run_vision.py on the
retail cmd_json bus through scripts/gvd_clock_child.py (stub cameras that return a painted
road, so the planner has a path) with a fake mod that integrates the commanded pedals into its
echoed speed. The supervisor, the fake mod and the car all run on one test-owned clock
(scripts/gvd_sim_clock.py), one supervisor tick per step, so a loaded host cannot change what
the checks see.

1. Known speed: a fresh echo at 0 m/s pulls away inside the window. Throttle goes above 0 and
   the echoed speed rises.
2. Known speed: above LAUNCH_SPEED_MPS inside the window the command is throttle 0 with the
   0.2 launch brake, and the car stays near that speed.
3. Unknown speed inside the window: the echo freezes, and the supervisor disengages with
   echo_stale on every surface (thr 0 / brk 0). There is no silent throttle-0 hold.
4. The stale-lane cap after the window (apply_stale_speed_cap), with a known speed: throttle
   capped at 0.25 at or below STALE_SPEED_MPS (6.0 m/s, inclusive), throttle 0 and brake 0.3
   above it. Unit level at 5.9 / 6.0 / 6.1 m/s, and on the supervisor: after the window the
   car is held near 6 m/s with the 0.3 brake.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import test_unknown_ego as T  # noqa: E402
from gvd_sim_clock import SimClock  # noqa: E402
from gvd_test_env import isolated_home  # noqa: E402
from python.control.actuate import cmd_path, write_engage_flag  # noqa: E402
from python.control.actuate import DriveCommand  # noqa: E402
from python.control.launch_guard import (  # noqa: E402
    LAUNCH_S,
    LAUNCH_SPEED_MPS,
    LAUNCH_THROTTLE,
    STALE_LANE_S,
    STALE_SPEED_MPS,
    apply_stale_speed_cap,
)

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
        now = self._mono()
        if now >= self.next_echo and self.echoing and self.frozen is None:
            cmd = T._read(cmd_path())
            if self._last is not None and cmd.get("engaged") is True:
                acc = (ACCEL_PER_THROTTLE * float(cmd.get("throttle") or 0.0)
                       - DECEL_PER_BRAKE * float(cmd.get("brake") or 0.0))
                self.speed = max(0.0, self.speed + acc * (now - self._last))
            self._last = now
        super().echo(t_spawn)


def _speed(snap: dict) -> float | None:
    v = (snap["state"].get("ego") or {}).get("speed_mps")
    return None if v is None else float(v)


def _drive(seconds: float, *, freeze_after_s: float | None = None) -> tuple[list[dict], dict]:
    mark: dict = {}
    with isolated_home(prefix="gvd-launch-"):
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

        seen = T._session(seconds, car, step, clock=SimClock())
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


STALE_THROTTLE = 0.25
STALE_BRAKE = 0.3


def _capped(speed: float, *, throttle: float = 0.5, brake: float = 0.0,
            lane_age_s: float = 0.5, lane_src: str = "pair/stale") -> tuple[bool, float, float]:
    cmd = DriveCommand(steer=0.0, throttle=throttle, brake=brake, seq=1, reason="ok")
    changed = apply_stale_speed_cap(cmd, speed_mps=speed, lane_age_s=lane_age_s, lane_src=lane_src)
    return changed, cmd.throttle, cmd.brake


def check_stale_cap_numbers_unit() -> None:
    assert STALE_SPEED_MPS == 6.0 and STALE_LANE_S == 0.25, (STALE_SPEED_MPS, STALE_LANE_S)
    # At or below 6.0 m/s (the boundary is inclusive: the brake needs speed > 6.0): throttle 0.25.
    assert _capped(5.9) == (True, STALE_THROTTLE, 0.0), _capped(5.9)
    assert _capped(6.0) == (True, STALE_THROTTLE, 0.0), _capped(6.0)
    # Above 6.0 m/s: throttle 0, brake 0.3.
    assert _capped(6.1) == (True, 0.0, STALE_BRAKE), _capped(6.1)
    # The cap only lowers throttle and only raises brake.
    assert _capped(5.9, throttle=0.1) == (True, 0.1, 0.0), _capped(5.9, throttle=0.1)
    assert _capped(6.1, brake=0.5) == (True, 0.0, 0.5), _capped(6.1, brake=0.5)
    # A held center caps the same way; a fresh lane is left alone.
    assert _capped(6.1, lane_age_s=0.0, lane_src="pair/hold") == (True, 0.0, STALE_BRAKE)
    assert _capped(5.9, lane_age_s=0.0, lane_src="pair/hold") == (True, STALE_THROTTLE, 0.0)
    assert _capped(6.1, lane_age_s=STALE_LANE_S, lane_src="pair") == (False, 0.5, 0.0)


def check_stale_cap_after_window_holds_6mps() -> None:
    seen, _mark = _drive(10.5)
    on = [s for s in seen if s["cmd"].get("engaged") is True]
    assert on, "never engaged"
    t_on = on[0]["t"]
    off = [s["state"].get("disengage_reason") for s in seen if s["t"] > t_on and s["cmd"].get("engaged") is not True]
    assert not off, ("disengaged", off[:5])
    after = [s for s in on if s["t"] - t_on > LAUNCH_S + 0.3]
    assert len(after) >= 20, ("too few ticks after the launch window", len(after))
    pedals = sorted({(float(s["cmd"].get("throttle") or 0.0), float(s["cmd"].get("brake") or 0.0)) for s in after})
    assert set(pedals) <= {(STALE_THROTTLE, 0.0), (0.0, STALE_BRAKE)}, ("stale cap pedals", pedals)
    braked = [s for s in after if float(s["cmd"].get("brake") or 0.0) > 0.0]
    assert braked, ("the 0.3 stale-cap brake never applied", max(_speed(s) or 0.0 for s in after))
    assert any(float(s["cmd"].get("throttle") or 0.0) == STALE_THROTTLE for s in after), pedals
    first_brake = braked[0]["t"]
    held = [_speed(s) for s in after if s["t"] >= first_brake and _speed(s) is not None]
    assert held and min(held) >= STALE_SPEED_MPS - 1.0 and max(held) <= STALE_SPEED_MPS + 1.0, \
        ("not held near 6 m/s after the window", min(held or [0.0]), max(held or [0.0]))
    slow = [s for s in braked if _speed(s) is not None and _speed(s) < STALE_SPEED_MPS - 0.75]
    assert not slow, ("stale-cap brake well below 6 m/s", [(_speed(s), s["cmd"].get("brake")) for s in slow[:5]])


CHECKS = (
    check_known_speed_pulls_away_and_launch_brake,
    check_unknown_speed_in_window_disengages,
    check_stale_cap_numbers_unit,
    check_stale_cap_after_window_holds_6mps,
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
