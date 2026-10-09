#!/usr/bin/env python3
"""Tech Engage: controller steer sticks, a 0.25 wheel grab drops Engage, quit restores the wheel.

BeamNGpy ``vehicle.control`` is a pad-filter event on source ``local``. The physical
wheel writes that same source every frame after it. Without a whitelist the wheel
wins, and a centered wheel makes the car go straight. These fakes are that input
rule. Live BeamNG is not on this machine.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from python.control.actuate import (  # noqa: E402
    TECH_STEER_RELEASE_LUA,
    BeamNGPyActuator,
    DriveCommand,
    tech_steer_hold_lua,
)
from python.control.override import (  # noqa: E402
    REASON_STEER,
    OverrideDetector,
    load_override_config,
    tech_override_steer,
)

CONTROL_YAML = ROOT / "config" / "control.yaml"
MAIN_LUA = ROOT / "beamng_mod" / "lua" / "ge" / "extensions" / "gvd" / "main.lua"
RUN_VISION = ROOT / "python" / "run_vision.py"
TICK = 0.05

_ALLOWED_RE = re.compile(
    r"setAllowedInputSource\('([a-z]+)',(nil|'[a-z]+')(?:,(true|false))?\)"
)
_EVENT_RE = re.compile(
    r"input\.event\('([a-z]+)',(-?\d+(?:\.\d+)?),\d+,\d+,\d+,nil,'([a-z]+)'\)"
)


class PlayerWheel:
    """BeamNG ``input.event`` rule: last allowed source wins.

    ``allowed[itype] is None`` means every source applies. Once the table
    exists, only sources set true apply. ``lastInputs`` still records a
    blocked source. That is the 0.34+ ``input.lua`` behavior the retail
    lock already depends on.
    """

    def __init__(self, wheel: float = 0.0) -> None:
        self.wheel = float(wheel)
        self.allowed: dict[str, dict[str, bool] | None] = {}
        self.state: dict[str, float] = {"steering": 0.0}
        self.last: dict[str, dict[str, float]] = {}

    def set_allowed(self, itype: str, source: str | None, enabled: bool) -> None:
        if source is None:
            self.allowed[itype] = None
            return
        table = self.allowed.get(itype)
        if not isinstance(table, dict):
            table = {}
            self.allowed[itype] = table
        table[source] = bool(enabled)

    def event(self, itype: str, value: float, source: str = "local") -> None:
        self.last.setdefault(source, {})[itype] = float(value)
        allowed = self.allowed.get(itype)
        if allowed is None or (isinstance(allowed, dict) and allowed.get(source)):
            self.state[itype] = float(value)

    def player_frame(self) -> None:
        """The physical wheel writes source local after every software event."""
        self.event("steering", self.wheel, "local")


def apply_steer_chunk(wheel: PlayerWheel, chunk: str) -> None:
    """Run the steer-hold / release snippets in source order. Not a Lua VM."""
    marks: list[tuple[int, str, re.Match[str]]] = []
    for match in _ALLOWED_RE.finditer(chunk):
        marks.append((match.start(), "allow", match))
    for match in _EVENT_RE.finditer(chunk):
        marks.append((match.start(), "event", match))
    for _, kind, match in sorted(marks, key=lambda item: item[0]):
        if kind == "allow":
            itype = match.group(1)
            raw = match.group(2)
            if raw == "nil":
                wheel.set_allowed(itype, None, False)
            else:
                wheel.set_allowed(itype, raw.strip("'"), match.group(3) == "true")
        else:
            wheel.event(match.group(1), float(match.group(2)), match.group(3))


class TechVeh:
    """``vehicle.control`` then the wheel frame, then any queued vehicle Lua."""

    def __init__(self, wheel: PlayerWheel) -> None:
        self.wheel_sim = wheel
        self.calls: list[dict] = []
        self.shifts: list[str] = []
        self.ai_modes: list[str] = []
        self.lua: list[str] = []
        self.fail_queue = False
        self.sensors = {"electrics": {"gear": "D", "wheelspeed": 5.0, "steering_input": 0.0}}

    def set_shift_mode(self, mode: str) -> None:
        self.shifts.append(mode)

    def ai_set_mode(self, mode: str) -> None:
        self.ai_modes.append(mode)

    def control(self, **kw) -> None:
        self.calls.append(dict(kw))
        if "steering" in kw:
            # BeamNGpy Control: filter 1, source local. The wheel writes next.
            self.wheel_sim.event("steering", float(kw["steering"]), "local")
            self.wheel_sim.player_frame()

    def queue_lua_command(self, chunk: str, response: bool = False) -> None:
        del response
        if self.fail_queue:
            raise RuntimeError("queue failed")
        self.lua.append(chunk)
        apply_steer_chunk(self.wheel_sim, chunk)


class _Ego:
    def __init__(self, wheel: float | None, *, fresh: bool = True, device: bool = True) -> None:
        self.player_steering = wheel
        self.player_device = device
        self._fresh = fresh

    @property
    def fresh(self) -> bool:
        return self._fresh


def _cfg() -> OverrideDetector:
    raw = yaml.safe_load(CONTROL_YAML.read_text(encoding="utf-8"))
    return OverrideDetector(load_override_config(raw))


def check_unfiltered_wheel_zeros_command() -> None:
    """The bug: control steer 0.3, then a centered wheel, applied steer is 0."""
    wheel = PlayerWheel(0.0)
    wheel.event("steering", 0.3, "local")
    assert wheel.state["steering"] == 0.3
    wheel.player_frame()
    assert wheel.state["steering"] == 0.0


def check_engaged_command_sticks() -> None:
    """Engaged, command 0.3, wheel reports 0 → applied steer is 0.3."""
    wheel = PlayerWheel(0.0)
    veh = TechVeh(wheel)
    act = BeamNGPyActuator(veh)
    act.note_engaged(True)
    out = act.apply(DriveCommand(steer=0.3, throttle=0.2, brake=0.0, seq=1, reason="ok"))
    assert out.applied is True, out
    assert abs(veh.calls[-1]["steering"] - 0.3) < 1e-9
    assert act.steer_locked is True
    hold = veh.lua[-1]
    assert hold == tech_steer_hold_lua(0.3)
    assert "setAllowedInputSource('steering','local',false)" in hold
    assert "input.event('steering',0.3000,2,900,0,nil,'gvd')" in hold
    assert abs(wheel.state["steering"] - 0.3) < 1e-9, wheel.state
    # Another wheel frame, and another local control, leave the gvd steer in place.
    wheel.player_frame()
    wheel.event("steering", 0.0, "local")
    assert abs(wheel.state["steering"] - 0.3) < 1e-9
    act.apply(DriveCommand(steer=0.3, throttle=0.2, brake=0.0, seq=2, reason="ok"))
    assert abs(wheel.state["steering"] - 0.3) < 1e-9
    # The wheel angle is still recorded for the grab test.
    assert abs(wheel.last["local"]["steering"] - 0.0) < 1e-9


def _hold_until_steer(det: OverrideDetector, *, cmd: float, wheel: float, t0: float) -> bool:
    t = t0
    for i in range(40):
        t += TICK
        verdict = det.update(
            engaged=True,
            steering_input=wheel,
            throttle_input=0.2,
            brake_input=0.0,
            now=t,
            own_axes=True,
        )
        det.note_command(seq=100 + i, steer=cmd, throttle=0.2, brake=0.0, now=t)
        if verdict.active:
            assert verdict.reason == REASON_STEER, verdict
            return True
    return False


def check_wheel_grab_releases() -> None:
    """A wheel held 0.25 past the command drops Engage and drives again."""
    wheel = PlayerWheel(0.0)
    veh = TechVeh(wheel)
    act = BeamNGPyActuator(veh)
    act.note_engaged(True)
    act.apply(DriveCommand(steer=0.3, throttle=0.2, brake=0.0, seq=1, reason="ok"))
    assert abs(wheel.state["steering"] - 0.3) < 1e-9

    det = _cfg()
    t = 100.0
    armed = False
    for i in range(30):
        t += TICK
        det.update(
            engaged=True,
            steering_input=0.0,
            throttle_input=0.2,
            brake_input=0.0,
            now=t,
            own_axes=True,
        )
        det.note_command(seq=i + 1, steer=0.3, throttle=0.2, brake=0.0, now=t)
        if det.armed(t):
            armed = True
            break
    assert armed
    # Centered wheel against command 0.3 stays inside the 1.7.4 span.
    for i in range(10):
        t += TICK
        held = det.update(
            engaged=True,
            steering_input=0.0,
            throttle_input=0.2,
            brake_input=0.0,
            now=t,
            own_axes=True,
        )
        det.note_command(seq=50 + i, steer=0.3, throttle=0.2, brake=0.0, now=t)
        assert not held.active, held

    assert _hold_until_steer(det, cmd=0.3, wheel=0.55, t0=t), "0.25 past 0.3 must be player_steer"
    act.note_engaged(False)
    released = act.stop(seq=9, reason=REASON_STEER)
    assert released.brake == 0.0 and released.throttle == 0.0
    assert act.steer_locked is False
    assert wheel.allowed.get("steering") is None
    wheel.wheel = 0.55
    wheel.player_frame()
    assert abs(wheel.state["steering"] - 0.55) < 1e-9, wheel.state

    # Straight command, wheel held at 0.25, same rule.
    wheel2 = PlayerWheel(0.0)
    veh2 = TechVeh(wheel2)
    act2 = BeamNGPyActuator(veh2)
    act2.note_engaged(True)
    act2.apply(DriveCommand(steer=0.0, throttle=0.2, brake=0.0, seq=1, reason="ok"))
    det2 = _cfg()
    t = 100.0
    for i in range(30):
        t += TICK
        det2.update(
            engaged=True,
            steering_input=0.0,
            throttle_input=0.2,
            brake_input=0.0,
            now=t,
            own_axes=True,
        )
        det2.note_command(seq=i + 1, steer=0.0, throttle=0.2, brake=0.0, now=t)
    assert det2.armed(t)
    assert _hold_until_steer(det2, cmd=0.0, wheel=0.25, t0=t), "held 0.25 on a straight command"
    act2.note_engaged(False)
    act2.stop(seq=3, reason=REASON_STEER)
    assert act2.steer_locked is False
    wheel2.wheel = 0.25
    wheel2.player_frame()
    assert abs(wheel2.state["steering"] - 0.25) < 1e-9


def check_shutdown_restores_wheel() -> None:
    """Supervisor exit while engaged restores player steering."""
    wheel = PlayerWheel(0.0)
    veh = TechVeh(wheel)
    act = BeamNGPyActuator(veh)
    act.note_engaged(True)
    act.apply(DriveCommand(steer=0.3, throttle=0.2, brake=0.0, seq=4, reason="ok"))
    assert act.steer_locked is True
    # Same order as run_vision's finally block.
    act.note_engaged(False)
    stopped = act.stop(seq=5, reason="shutdown")
    assert stopped.reason == "shutdown"
    assert act.steer_locked is False
    assert TECH_STEER_RELEASE_LUA in veh.lua
    assert wheel.allowed.get("steering") is None
    wheel.wheel = -0.4
    wheel.player_frame()
    assert abs(wheel.state["steering"] - (-0.4)) < 1e-9
    # A second stop does not require the lock again.
    n = len(veh.lua)
    act.stop(seq=6, reason="not_engaged")
    assert veh.lua[n - 1 :] == [] or TECH_STEER_RELEASE_LUA not in veh.lua[n:]
    assert act.steer_locked is False


def check_release_retries_and_control_failure() -> None:
    """A failed release stays latched. A later control error still frees the wheel."""
    wheel = PlayerWheel(0.0)
    veh = TechVeh(wheel)
    act = BeamNGPyActuator(veh)
    act.note_engaged(True)
    act.apply(DriveCommand(steer=0.3, throttle=0.2, brake=0.0, seq=1, reason="ok"))
    veh.fail_queue = True
    act.note_engaged(False)
    act.stop(seq=2, reason="shutdown")
    assert act.steer_locked is True
    assert isinstance(wheel.allowed.get("steering"), dict)
    veh.fail_queue = False
    act.stop(seq=3, reason="shutdown")
    assert act.steer_locked is False
    assert wheel.allowed.get("steering") is None

    wheel_b = PlayerWheel(0.0)
    veh_b = TechVeh(wheel_b)
    act_b = BeamNGPyActuator(veh_b)
    act_b.note_engaged(True)
    act_b.apply(DriveCommand(steer=0.2, throttle=0.0, brake=0.0, seq=1, reason="ok"))
    assert act_b.steer_locked is True

    def _boom(**_kw):
        raise RuntimeError("control down")

    veh_b.control = _boom  # type: ignore[method-assign]
    act_b.note_engaged(False)
    act_b.stop(seq=2, reason="not_engaged")
    assert act_b.steer_locked is False
    assert wheel_b.allowed.get("steering") is None
    wheel_b.wheel = 0.15
    wheel_b.player_frame()
    assert abs(wheel_b.state["steering"] - 0.15) < 1e-9


def check_override_signal_is_the_wheel() -> None:
    """Locked electrics are the command. The grab signal is the player wheel."""
    ego = _Ego(0.0)
    assert tech_override_steer(0.3, ego, steer_locked=True) == 0.0
    assert tech_override_steer(0.3, None, steer_locked=True) is None
    assert tech_override_steer(0.0, None, steer_locked=False) == 0.0
    assert tech_override_steer(0.3, _Ego(0.55), steer_locked=True) == 0.55
    quiet = _Ego(0.0, device=False)
    assert tech_override_steer(0.3, quiet, steer_locked=True) is None
    src = RUN_VISION.read_text(encoding="utf-8")
    assert "tech_override_steer(" in src
    assert 'st["tech_steer_hold"]' in src


def check_release_text_matches_mod() -> None:
    lua = MAIN_LUA.read_text(encoding="utf-8")
    assert "M.techSteerRelease" in lua
    assert "input.event('steering',0,2,900,0,nil,'gvd')" in TECH_STEER_RELEASE_LUA
    assert "setAllowedInputSource('steering',nil)" in TECH_STEER_RELEASE_LUA
    # The mod string is concatenated; the pieces are the Python constant.
    assert "input.event('steering',0,2,900,0,nil,'gvd');" in lua
    assert "input.setAllowedInputSource('steering',nil);" in lua
    assert "M.techSteerOwed" in lua
    assert "tech_steer_hold" in lua
    body = lua[lua.index("M.techSteerRelease"):lua.index("local VE_RELEASE")]
    compact = body.replace(" ", "").replace("\n", "").replace("..", "").replace('"', "")
    want = TECH_STEER_RELEASE_LUA.replace(" ", "")
    assert want in compact, compact


def main() -> None:
    check_unfiltered_wheel_zeros_command()
    check_engaged_command_sticks()
    check_wheel_grab_releases()
    check_shutdown_restores_wheel()
    check_release_retries_and_control_failure()
    check_override_signal_is_the_wheel()
    check_release_text_matches_mod()
    print("test_tech_steer_hold: OK")


if __name__ == "__main__":
    main()
