#!/usr/bin/env python3
"""Tech Engage: controller steer sticks, a 0.25 wheel grab drops Engage, quit restores the wheel.

BeamNGpy ``vehicle.control`` with a steering field is a pad-filter event on source
``local``, the same slot the physical wheel writes. The wheel writes that slot
once per change (onChange), not every frame. A later Control message replaces
``lastInputs.local.steering`` even when the whitelist keeps that source off the
hydros, so the grab echo becomes the command. These fakes are that input rule.
Live BeamNG is not on this machine.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
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
    locked_steer_residual,
    tech_override_steer,
)

CONTROL_YAML = ROOT / "config" / "control.yaml"
MAIN_LUA = ROOT / "beamng_mod" / "lua" / "ge" / "extensions" / "gvd" / "main.lua"
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
    blocked source. The physical wheel writes source ``local`` once when the
    angle changes. A Control message that includes steering writes that same
    slot again. That is the 0.34+ ``input.lua`` behavior the retail lock
    already depends on.
    """

    def __init__(self, wheel: float = 0.0) -> None:
        self.wheel = float(wheel)
        self._sent: float | None = None
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

    def set_wheel(self, angle: float) -> bool:
        """One local event when the held angle changes. Repeats do not write."""
        angle = float(angle)
        self.wheel = angle
        if self._sent is not None and abs(self._sent - angle) < 1e-9:
            return False
        self._sent = angle
        self.event("steering", angle, "local")
        return True


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
    # Release replays the stored local slot. The value is not a numeric literal.
    if "input.lastInputs['local']" in chunk and "nil,'local')" in chunk:
        stored = float((wheel.last.get("local") or {}).get("steering", 0.0))
        wheel.event("steering", stored, "local")


class TechVeh:
    """``vehicle.control`` shares the local slot. The wheel does not write again.

    A steering field on control overwrites ``lastInputs.local.steering``.
    The physical wheel does not. Queued vehicle Lua runs after that write.
    """

    def __init__(self, wheel: PlayerWheel) -> None:
        self.wheel_sim = wheel
        self.calls: list[dict] = []
        self.shifts: list[str] = []
        self.ai_modes: list[str] = []
        self.lua: list[str] = []
        self.fail_queue = False
        self.die_in_queue = False
        self.accept_then_die = False
        self.expect_flag_before_queue = False
        self.flag_seen = False
        self.sensors = {"electrics": {"gear": "D", "wheelspeed": 5.0, "steering_input": 0.0}}

    def set_shift_mode(self, mode: str) -> None:
        self.shifts.append(mode)

    def ai_set_mode(self, mode: str) -> None:
        self.ai_modes.append(mode)

    def control(self, **kw) -> None:
        self.calls.append(dict(kw))
        if "steering" in kw:
            # Control and the wheel share source local. This is not an onChange.
            self.wheel_sim.event("steering", float(kw["steering"]), "local")

    def queue_lua_command(self, chunk: str, response: bool = False) -> None:
        del response
        if self.expect_flag_before_queue and "setAllowedInputSource('steering','gvd',true)" in chunk:
            from python.runtime.state_io import read_state

            st = read_state()
            assert st is not None and st.get("tech_steer_hold") is True, st
            self.flag_seen = True
        if self.accept_then_die and "setAllowedInputSource('steering','gvd',true)" in chunk:
            self.lua.append(chunk)
            apply_steer_chunk(self.wheel_sim, chunk)
            raise SystemExit("killed after accept")
        if self.die_in_queue and "setAllowedInputSource('steering','gvd',true)" in chunk:
            raise SystemExit("killed during queue")
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
    """Without a whitelist, a later wheel onChange replaces the command on the hydros."""
    wheel = PlayerWheel(0.0)
    wheel.event("steering", 0.3, "local")
    assert wheel.state["steering"] == 0.3
    assert wheel.set_wheel(0.0) is True
    assert wheel.state["steering"] == 0.0
    assert wheel.set_wheel(0.0) is False


def _pedals_kept(call: dict) -> None:
    assert "steering" not in call, call
    assert "throttle" in call and "brake" in call and "parkingbrake" in call


def check_engaged_command_sticks() -> None:
    """Engaged, command 0.3, wheel reports 0 → applied steer is 0.3."""
    wheel = PlayerWheel(0.0)
    veh = TechVeh(wheel)
    act = BeamNGPyActuator(veh)
    act.note_engaged(True)
    out = act.apply(DriveCommand(steer=0.3, throttle=0.2, brake=0.0, seq=1, reason="ok"))
    assert out.applied is True, out
    _pedals_kept(veh.calls[-1])
    assert veh.calls[-1]["throttle"] == 0.2
    assert "clutch" in veh.calls[-1]
    assert act.steer_locked is True
    hold = veh.lua[-1]
    assert hold == tech_steer_hold_lua(0.3)
    assert "setAllowedInputSource('steering','local',false)" in hold
    assert "input.event('steering',0.3000,2,900,0,nil,'gvd')" in hold
    assert abs(wheel.state["steering"] - 0.3) < 1e-9, wheel.state
    # One centered wheel event. Later Control messages do not write the local slot.
    assert wheel.set_wheel(0.0) is True
    assert abs(wheel.state["steering"] - 0.3) < 1e-9
    act.apply(DriveCommand(steer=0.3, throttle=0.2, brake=0.0, seq=2, reason="ok"))
    _pedals_kept(veh.calls[-1])
    assert abs(wheel.state["steering"] - 0.3) < 1e-9
    assert abs(wheel.last["local"]["steering"] - 0.0) < 1e-9
    assert wheel.set_wheel(0.0) is False


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
            steer_locked=True,
        )
        det.note_command(seq=100 + i, steer=cmd, throttle=0.2, brake=0.0, now=t)
        if verdict.active:
            assert verdict.reason == REASON_STEER, verdict
            return True
    return False


def check_grab_survives_control_messages() -> None:
    """One wheel event at 0.55, then several Control messages, still player_steer.

    The detector is fed from ``lastInputs.local.steering`` after those messages.
    """
    wheel = PlayerWheel(0.0)
    veh = TechVeh(wheel)
    act = BeamNGPyActuator(veh)
    act.note_engaged(True)
    act.apply(DriveCommand(steer=0.3, throttle=0.2, brake=0.0, seq=1, reason="ok"))
    assert abs(wheel.state["steering"] - 0.3) < 1e-9
    assert wheel.set_wheel(0.0) is True

    det = _cfg()
    t = 100.0
    armed = False
    for i in range(30):
        t += TICK
        det.update(
            engaged=True,
            steering_input=wheel.last["local"]["steering"],
            throttle_input=0.2,
            brake_input=0.0,
            now=t,
            own_axes=True,
            steer_locked=True,
        )
        det.note_command(seq=i + 1, steer=0.3, throttle=0.2, brake=0.0, now=t)
        if det.armed(t):
            armed = True
            break
    assert armed
    for i in range(10):
        t += TICK
        held = det.update(
            engaged=True,
            steering_input=wheel.last["local"]["steering"],
            throttle_input=0.2,
            brake_input=0.0,
            now=t,
            own_axes=True,
            steer_locked=True,
        )
        det.note_command(seq=50 + i, steer=0.3, throttle=0.2, brake=0.0, now=t)
        assert not held.active, held

    assert wheel.set_wheel(0.55) is True
    assert abs(wheel.state["steering"] - 0.3) < 1e-9
    for n in range(4):
        act.apply(DriveCommand(steer=0.3, throttle=0.2, brake=0.0, seq=10 + n, reason="ok"))
        _pedals_kept(veh.calls[-1])
        assert abs(wheel.last["local"]["steering"] - 0.55) < 1e-9
    assert wheel.set_wheel(0.55) is False
    slot = wheel.last["local"]["steering"]
    echo = tech_override_steer(0.3, _Ego(slot), steer_locked=True)
    assert echo == 0.55
    assert _hold_until_steer(det, cmd=0.3, wheel=float(echo), t0=t), "0.55 slot must be player_steer"
    act.note_engaged(False)
    released = act.stop(seq=20, reason=REASON_STEER)
    assert released.brake == 0.0 and released.throttle == 0.0
    assert act.steer_locked is False
    assert wheel.allowed.get("steering") is None
    _pedals_kept(veh.calls[-1])
    # Release replays the stored wheel. No second onChange.
    assert abs(wheel.state["steering"] - 0.55) < 1e-9, wheel.state
    assert abs(wheel.last["local"]["steering"] - 0.55) < 1e-9


def _feed_missing_echo(det: OverrideDetector, t: float) -> float:
    """None through warmup. Returns the first time the detector is armed."""
    det.note_command(seq=1, steer=0.0, throttle=0.2, brake=0.0, now=t)
    while not det.armed(t + TICK):
        t += TICK
        det.update(
            engaged=True,
            steering_input=None,
            throttle_input=0.2,
            brake_input=0.0,
            now=t,
            own_axes=True,
            steer_locked=True,
        )
    t += TICK
    assert det.armed(t)
    return t


def check_wheel_grab_releases() -> None:
    """None through warmup and arm seeds rest at 0. A later 0.25 is player_steer.

    No wheel sample at 0 is fed first. A grab already present on the first
    armed sample is not stored as rest. An unlocked detector still uses that
    first sample as the resting wheel.
    """
    det2 = _cfg()
    t = _feed_missing_echo(det2, 100.0)
    armed = det2.update(
        engaged=True,
        steering_input=None,
        throttle_input=0.2,
        brake_input=0.0,
        now=t,
        own_axes=True,
        steer_locked=True,
    )
    assert not armed.active
    assert det2._base_steer == 0.0
    assert _hold_until_steer(det2, cmd=0.0, wheel=0.25, t0=t), "held 0.25 with no prior rest sample"

    grabbed = _cfg()
    t = _feed_missing_echo(grabbed, 300.0)
    grabbed.update(
        engaged=True,
        steering_input=0.25,
        throttle_input=0.2,
        brake_input=0.0,
        now=t,
        own_axes=True,
        steer_locked=True,
    )
    assert grabbed._base_steer == 0.0
    assert _hold_until_steer(grabbed, cmd=0.0, wheel=0.25, t0=t), "grab already held at arm time"

    plain = _cfg()
    t = 500.0
    plain.note_command(seq=1, steer=0.0, throttle=0.0, brake=0.0, now=t)
    while not plain.armed(t + TICK):
        t += TICK
        plain.update(
            engaged=True,
            steering_input=None,
            now=t,
            own_axes=True,
            steer_locked=False,
        )
    t += TICK
    plain.update(
        engaged=True,
        steering_input=0.25,
        now=t,
        own_axes=True,
        steer_locked=False,
    )
    assert plain._base_steer == 0.25

    wheel2 = PlayerWheel(0.0)
    veh2 = TechVeh(wheel2)
    act2 = BeamNGPyActuator(veh2)
    act2.note_engaged(True)
    act2.apply(DriveCommand(steer=0.0, throttle=0.2, brake=0.0, seq=1, reason="ok"))
    assert wheel2.set_wheel(0.25) is True
    act2.note_engaged(False)
    act2.stop(seq=3, reason=REASON_STEER)
    assert act2.steer_locked is False
    assert abs(wheel2.state["steering"] - 0.25) < 1e-9


def check_shutdown_restores_wheel() -> None:
    """Supervisor exit while engaged restores player steering."""
    wheel = PlayerWheel(0.0)
    veh = TechVeh(wheel)
    act = BeamNGPyActuator(veh)
    act.note_engaged(True)
    act.apply(DriveCommand(steer=0.3, throttle=0.2, brake=0.0, seq=4, reason="ok"))
    assert act.steer_locked is True
    assert wheel.set_wheel(-0.4) is True
    assert abs(wheel.state["steering"] - 0.3) < 1e-9
    # Same order as run_vision's finally block.
    act.note_engaged(False)
    stopped = act.stop(seq=5, reason="shutdown")
    assert stopped.reason == "shutdown"
    assert act.steer_locked is False
    assert TECH_STEER_RELEASE_LUA in veh.lua
    assert wheel.allowed.get("steering") is None
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
    assert wheel_b.set_wheel(0.15) is True

    def _boom(**_kw):
        raise RuntimeError("control down")

    veh_b.control = _boom  # type: ignore[method-assign]
    act_b.note_engaged(False)
    act_b.stop(seq=2, reason="not_engaged")
    assert act_b.steer_locked is False
    assert wheel_b.allowed.get("steering") is None
    assert abs(wheel_b.state["steering"] - 0.15) < 1e-9


def check_override_signal_is_the_wheel() -> None:
    """Locked electrics are the command. The grab signal is the player wheel."""
    ego = _Ego(0.0)
    assert tech_override_steer(0.3, ego, steer_locked=True) == 0.0
    assert tech_override_steer(0.3, None, steer_locked=True) is None
    assert tech_override_steer(0.0, None, steer_locked=False) == 0.0
    assert tech_override_steer(0.3, _Ego(0.55), steer_locked=True) == 0.55
    quiet = _Ego(0.0, device=False)
    assert tech_override_steer(0.3, quiet, steer_locked=True) == 0.0
    # A fresh numeric echo counts even when player_device is false.
    assert tech_override_steer(0.3, _Ego(0.22, device=False), steer_locked=True) == 0.22

    from python.control.actuate import ego_path
    from python.run_vision import sample_tech_wheel

    class Retail:
        name = "cmd_json"

    assert sample_tech_wheel(Retail(), 0.2) == 0.2
    ego_path().write_text(
        json.dumps({"player_device": True, "player_steering": 0.55, "steering_input": 0.3}),
        encoding="utf-8",
    )
    locked = BeamNGPyActuator(TechVeh(PlayerWheel()))
    locked._steer_locked = True
    assert sample_tech_wheel(locked, 0.3) == 0.55
    ego_path().write_text(json.dumps({"player_device": False, "steering_input": 0.3}), encoding="utf-8")
    import io
    from contextlib import redirect_stdout

    import python.run_vision as rv

    rv._STEER_GRAB_BLIND_LOGGED = False
    rv._STEER_GRAB_BLIND_SINCE = None
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert sample_tech_wheel(locked, 0.3, now=10.0) is None
        assert sample_tech_wheel(locked, 0.3, now=10.1) is None
    assert buf.getvalue().count("tech steer grab blind") == 1, buf.getvalue()
    assert rv.tech_wheel_blind_reason(now=12.0) is None
    assert rv.tech_wheel_blind_reason(now=12.1) == "tech_wheel_blind"
    locked._steer_locked = False
    with redirect_stdout(io.StringIO()):
        assert sample_tech_wheel(locked, 0.3, now=13.0) == 0.3
    assert rv._STEER_GRAB_BLIND_LOGGED is False
    assert rv._STEER_GRAB_BLIND_SINCE is None
    locked._steer_locked = True
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        assert sample_tech_wheel(locked, 0.3, now=14.0) is None
    assert buf2.getvalue().count("tech steer grab blind") == 1, buf2.getvalue()
    # Fresh player_steering with player_device false is the wheel. The 2 s
    # timer starts when that field is absent or gvd_ego.json is stale.
    import time

    rv._STEER_GRAB_BLIND_LOGGED = False
    rv._STEER_GRAB_BLIND_SINCE = None
    locked._steer_locked = True
    ego_path().write_text(
        json.dumps(
            {"player_device": False, "player_steering": 0.22, "steering_input": 0.3}
        ),
        encoding="utf-8",
    )
    assert sample_tech_wheel(locked, 0.3, now=20.0) == 0.22
    assert rv._STEER_GRAB_BLIND_SINCE is None
    stale = time.time() - 5.0
    os.utime(ego_path(), (stale, stale))
    buf3 = io.StringIO()
    with redirect_stdout(buf3):
        assert sample_tech_wheel(locked, 0.3, now=21.0) is None
    assert rv._STEER_GRAB_BLIND_SINCE == 21.0
    assert buf3.getvalue().count("tech steer grab blind") == 1, buf3.getvalue()
    open_wheel = BeamNGPyActuator(TechVeh(PlayerWheel()))
    ego_path().write_text("{}", encoding="utf-8")
    assert sample_tech_wheel(open_wheel, 0.0) == 0.0


def check_hold_flag_before_queue() -> None:
    """The rising edge publishes tech_steer_hold before queue_lua_command.

    A kill inside that call leaves the flag set. A second engaged tick does
    not publish again.
    """
    import python.control.actuate as act_mod

    n = {"n": 0}
    orig = act_mod.flush_tech_steer_hold

    def spy(held: bool, **_k: object) -> bool:
        n["n"] += 1
        assert held is True
        return bool(orig(held, **_k))

    act_mod.flush_tech_steer_hold = spy
    try:
        wheel = PlayerWheel(0.0)
        veh = TechVeh(wheel)
        veh.expect_flag_before_queue = True
        act = BeamNGPyActuator(veh)
        act.note_engaged(True)
        act.apply(DriveCommand(steer=0.3, throttle=0.2, brake=0.0, seq=1, reason="ok"))
        act.apply(DriveCommand(steer=0.3, throttle=0.2, brake=0.0, seq=2, reason="ok"))
        assert veh.flag_seen is True
        assert act.steer_locked is True
        assert n["n"] == 1, n

        killed = PlayerWheel(0.0)
        veh_k = TechVeh(killed)
        veh_k.die_in_queue = True
        veh_k.expect_flag_before_queue = True
        act_k = BeamNGPyActuator(veh_k)
        act_k.note_engaged(True)
        raised = False
        try:
            act_k.apply(DriveCommand(steer=0.4, throttle=0.2, brake=0.0, seq=1, reason="ok"))
        except SystemExit:
            raised = True
        assert raised
        assert veh_k.flag_seen is True
        assert act_k.steer_locked is False
        from python.runtime.state_io import read_state

        st = read_state()
        assert st is not None and st.get("tech_steer_hold") is True, st
        assert n["n"] == 2, n
    finally:
        act_mod.flush_tech_steer_hold = orig


def check_failed_hold_not_applied() -> None:
    """A raising queue leaves applied false and sends no vehicle.control.

    Each pre-lock try flushes the hold flag. A missing queue and a failed
    flag write also leave the command unapplied, with no throttle or brake.
    """
    import python.control.actuate as act_mod

    n = {"n": 0}
    orig = act_mod.flush_tech_steer_hold

    def spy(held: bool, **_k: object) -> bool:
        n["n"] += 1
        assert held is True
        return bool(orig(held, **_k))

    act_mod.flush_tech_steer_hold = spy
    try:
        wheel = PlayerWheel(0.0)
        veh = TechVeh(wheel)
        veh.fail_queue = True
        act = BeamNGPyActuator(veh)
        act.note_engaged(True)
        sent = act.apply(DriveCommand(steer=0.4, throttle=0.2, brake=0.0, seq=1, reason="ok"))
        assert sent.applied is False, sent
        assert sent.reason == "tech_steer_hold_queue", sent.reason
        assert act.steer_locked is False
        assert veh.calls == []
        sent2 = act.apply(DriveCommand(steer=0.4, throttle=0.2, brake=0.0, seq=2, reason="ok"))
        assert sent2.applied is False and act.steer_locked is False
        assert veh.calls == []
        assert n["n"] == 2, n
    finally:
        act_mod.flush_tech_steer_hold = orig

    class Bare:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self.shifts: list[str] = []

        def set_shift_mode(self, mode: str) -> None:
            self.shifts.append(mode)

        def control(self, **kw) -> None:
            self.calls.append(kw)

    bare = Bare()
    act_b = BeamNGPyActuator(bare)
    act_b.note_engaged(True)
    missing = act_b.apply(DriveCommand(steer=0.4, throttle=0.2, brake=0.0, seq=1, reason="ok"))
    assert missing.applied is False and missing.reason == "tech_steer_hold_no_queue"
    assert act_b.steer_locked is False
    assert bare.calls == [] and bare.shifts == []
    again = act_b.apply(DriveCommand(steer=0.4, throttle=0.2, brake=0.0, seq=2, reason="ok"))
    assert again.applied is False and bare.calls == []

    import python.runtime.state_io as sio

    orig_write = sio.write_state

    def fail_write(*_a, **_k):
        return None

    sio.write_state = fail_write
    try:
        blocked = TechVeh(PlayerWheel(0.0))
        act_w = BeamNGPyActuator(blocked)
        act_w.note_engaged(True)
        wrote = act_w.apply(DriveCommand(steer=0.4, throttle=0.2, brake=0.0, seq=1, reason="ok"))
        assert wrote.applied is False and wrote.reason == "tech_steer_hold_flag", wrote
        assert act_w.steer_locked is False
        assert blocked.lua == []
        assert blocked.calls == []
    finally:
        sio.write_state = orig_write


def _steady(act: BeamNGPyActuator) -> None:
    """Write the supervisor snapshot the way run_vision does after a tick."""
    from python.runtime.state_io import read_state, write_state

    prev = read_state()
    st = dict(prev) if isinstance(prev, dict) else {}
    st["tech_steer_hold"] = act.tech_steer_hold_for_state()
    if st["tech_steer_hold"] and act.steer_hold_vid is not None:
        st["tech_steer_hold_vid"] = act.steer_hold_vid
    elif "tech_steer_hold_vid" in st and not st["tech_steer_hold"]:
        st.pop("tech_steer_hold_vid", None)
    write_state(st)


def check_retry_kill_keeps_disk_flag() -> None:
    """Failed queue, steady snapshot, re-engage, accepted queue, then kill.

    The disk flag stays true. The snapshot after the failure keeps the
    watcher armed, and the queue that finally lands flushes again before
    it runs.
    """
    from python.runtime.state_io import read_state

    wheel = PlayerWheel(0.0)
    veh = TechVeh(wheel)
    veh.vid = 303
    veh.fail_queue = True
    act = BeamNGPyActuator(veh)
    act.note_engaged(True)
    failed = act.apply(DriveCommand(steer=0.4, throttle=0.2, brake=0.0, seq=1, reason="ok"))
    assert failed.applied is False and failed.reason == "tech_steer_hold_queue"
    assert act.tech_steer_hold_for_state() is True
    assert veh.calls == []
    _steady(act)
    st = read_state()
    assert st is not None and st.get("tech_steer_hold") is True, st
    assert st.get("tech_steer_hold_vid") == 303, st

    act.note_engaged(False)
    assert act.tech_steer_hold_for_state() is False
    _steady(act)
    cleared = read_state()
    assert cleared is not None and cleared.get("tech_steer_hold") is False, cleared

    veh.fail_queue = False
    veh.accept_then_die = True
    act.note_engaged(True)
    raised = False
    try:
        act.apply(DriveCommand(steer=0.4, throttle=0.2, brake=0.0, seq=2, reason="ok"))
    except SystemExit:
        raised = True
    assert raised
    assert tech_steer_hold_lua(0.4) in veh.lua
    disk = read_state()
    assert disk is not None and disk.get("tech_steer_hold") is True, disk
    assert disk.get("tech_steer_hold_vid") == 303, disk


def _stay_engaged(det: OverrideDetector, *, cmd: float, wheel: float, t0: float) -> None:
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
            steer_locked=True,
        )
        det.note_command(seq=200 + i, steer=cmd, throttle=0.2, brake=0.0, now=t)
        assert not verdict.active, (cmd, wheel, verdict)


def check_command_side_grab() -> None:
    """Locked residual: between 0 and the command is 0; past it is |w-c|.

    Opposite side of 0, and a straight command, use max(0, |w| - 0.15).
    A held 0.25 in any direction drops Engage. A wheel at 0.10, at 0, or
    at -0.08 against command 0.35 stays engaged for the full dwell.
    """
    d = 0.15
    assert locked_steer_residual(0.0, 0.35, d) == 0.0
    assert abs(locked_steer_residual(-0.25, 0.35, d) - 0.10) < 1e-9
    assert abs(locked_steer_residual(-1.0, 0.35, d) - 0.85) < 1e-9
    assert locked_steer_residual(0.10, 0.0, d) == 0.0
    assert abs(locked_steer_residual(0.25, 0.0, d) - 0.10) < 1e-9
    assert abs(locked_steer_residual(0.60, 0.35, d) - 0.25) < 1e-9
    assert locked_steer_residual(-0.08, 0.35, d) == 0.0
    assert abs(locked_steer_residual(0.25, -0.35, d) - 0.10) < 1e-9

    cases = (
        (0.35, -0.25, True),
        (-0.35, 0.25, True),
        (0.35, -1.0, True),
        (0.35, 0.0, False),
        (0.0, 0.10, False),
        (0.0, 0.25, True),
        (0.35, 0.60, True),
        (0.35, -0.08, False),
    )
    for n, (cmd, wheel, trips) in enumerate(cases):
        det = _cfg()
        t = _feed_missing_echo(det, 700.0 + n * 100.0)
        det.note_command(seq=700 + n, steer=cmd, throttle=0.2, brake=0.0, now=t)
        if trips:
            assert _hold_until_steer(det, cmd=cmd, wheel=wheel, t0=t), (cmd, wheel)
        else:
            _stay_engaged(det, cmd=cmd, wheel=wheel, t0=t)


def check_locked_refresh_sends_pedals() -> None:
    """After the lock is up, a failed hold refresh still sends pedals.

    Steering stays off the control message. The pre-lock failure path is
    covered by check_failed_hold_not_applied.
    """
    wheel = PlayerWheel(0.0)
    veh = TechVeh(wheel)
    act = BeamNGPyActuator(veh)
    act.note_engaged(True)
    first = act.apply(DriveCommand(steer=0.2, throttle=0.2, brake=0.0, seq=1, reason="ok"))
    assert first.applied is True and act.steer_locked is True
    veh.fail_queue = True
    veh.calls.clear()
    sent = act.apply(DriveCommand(steer=0.2, throttle=0.1, brake=0.8, seq=2, reason="ok"))
    assert sent.applied is True, sent
    assert sent.reason == "ok", sent.reason
    assert act.steer_locked is True
    assert veh.calls, "locked refresh sent no vehicle.control"
    call = veh.calls[-1]
    assert call.get("throttle") == 0.1, call
    assert call.get("brake") == 0.8, call
    assert "steering" not in call, call


def check_release_text_matches_mod() -> None:
    lua = MAIN_LUA.read_text(encoding="utf-8")
    assert "M.techSteerRelease" in lua
    assert "setAllowedInputSource('steering',nil)" in TECH_STEER_RELEASE_LUA
    assert "input.lastInputs['local']" in TECH_STEER_RELEASE_LUA
    assert "input.event('steering',s,flt,ang,lk,nil,'local')" in TECH_STEER_RELEASE_LUA
    assert "steeringAngle" in TECH_STEER_RELEASE_LUA
    assert "steeringFilter" in TECH_STEER_RELEASE_LUA
    assert "lockType" in TECH_STEER_RELEASE_LUA
    assert "900" in TECH_STEER_RELEASE_LUA
    assert "input.setAllowedInputSource('steering',nil);" in lua
    assert "input.lastInputs['local']" in lua
    assert "M.techSteerOwed" in lua
    assert "tech_steer_hold" in lua
    assert "queueVehicle(M.techSteerTarget(), M.techSteerRelease)" in lua
    assert "function M.techSteerTarget()" in lua
    assert "scenetree.findObject" in lua
    assert "getObjectByID" in lua
    assert "return getPlayerVeh()" in lua
    assert "function M.finishTechSteerLoadReplay()" in lua
    assert "tech steer release (load replay)" in lua
    assert "M.noteTechSteerHold" in lua
    body = lua[lua.index("M.techSteerRelease"):lua.index("local VE_RELEASE")]
    compact = body.replace(" ", "").replace("\n", "").replace("..", "").replace('"', "")
    want = TECH_STEER_RELEASE_LUA.replace(" ", "")
    assert want in compact, compact


def main() -> None:
    prev = os.environ.get("GVD_DOCS_DIR")
    tmp = tempfile.TemporaryDirectory()
    os.environ["GVD_DOCS_DIR"] = tmp.name
    try:
        check_unfiltered_wheel_zeros_command()
        check_engaged_command_sticks()
        check_grab_survives_control_messages()
        check_wheel_grab_releases()
        check_shutdown_restores_wheel()
        check_release_retries_and_control_failure()
        check_override_signal_is_the_wheel()
        check_hold_flag_before_queue()
        check_failed_hold_not_applied()
        check_retry_kill_keeps_disk_flag()
        check_command_side_grab()
        check_locked_refresh_sends_pedals()
        check_release_text_matches_mod()
    finally:
        if prev is None:
            os.environ.pop("GVD_DOCS_DIR", None)
        else:
            os.environ["GVD_DOCS_DIR"] = prev
        tmp.cleanup()
    print("test_tech_steer_hold: OK")


if __name__ == "__main__":
    main()
