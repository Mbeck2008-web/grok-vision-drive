#!/usr/bin/env python3
"""Offline M3 actuation gates (Spec verify 2–4)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.control.actuate import (
    TECH_DRIVE_GEAR,
    TECH_HOLD_GEAR,
    TECH_PLAYER_SHIFT_MODE,
    TECH_SHIFT_MODE,
    BeamNGPyActuator,
    CmdJsonActuator,
    DriveCommand,
    heartbeat_fresh,
    is_arcade_reverse_hold,
    is_reverse_control,
    may_drive,
    plan_command,
    last_electrics_ms,
    publish_vehicle_sensor_snap,
    read_electrics,
    read_electrics_speed,
    reset_soft_esc_state_writes,
    safe_command,
    soft_esc_state_write_due,
    soft_esc_state_write_mark,
    soft_esc_state_write_skips,
    TECH_DRIVE_ARM_S,
    TECH_DRIVE_SHIFT_LUA,
    gear_is_forward,
    tech_control_kwargs,
)


class _Pollable(dict):
    def poll(self):
        return self


def check_electrics_segment_timer() -> None:
    """read_electrics wall-ms. Snapshot reuse does not open a second sensors.poll."""
    import time

    class Slow:
        def __init__(self) -> None:
            self.n = 0
            self.sensors = self

        def poll(self):
            self.n += 1
            time.sleep(0.02)
            return {"electrics": {"wheelspeed": 1.5, "steering_input": 0.0}}

    slow = Slow()
    el = read_electrics(slow)
    assert el is not None and el["wheelspeed"] == 1.5
    assert slow.n == 1
    assert last_electrics_ms() >= 10.0, last_electrics_ms()

    class Held:
        def __init__(self) -> None:
            self.n = 0
            self.sensors = self

        def poll(self):
            self.n += 1
            return {"electrics": {"wheelspeed": 9.0}}

    held = Held()
    publish_vehicle_sensor_snap(held, {"electrics": {"wheelspeed": 4.0, "steering_input": 0.25}})
    reused = read_electrics(held)
    assert reused is not None and reused["wheelspeed"] == 4.0 and reused["steering_input"] == 0.25
    assert held.n == 0
    assert 0.0 <= last_electrics_ms() < 10.0, last_electrics_ms()
    assert read_electrics(None) is None
    assert last_electrics_ms() >= 0.0


def check_drive_gear_arm() -> None:
    """Throttle>0 leaves Neutral: gear>=1, clutch and parking brake cleared.

    set_shift_mode failures are logged and retried. The vehicle-Lua arm shifts
    up from N or jumps to the D letter. It never shiftDown and never requests -1.
    """
    import io
    import time
    from contextlib import redirect_stdout

    assert gear_is_forward(None) is False
    assert gear_is_forward("N") is False and gear_is_forward("n") is False
    assert gear_is_forward("P") is False and gear_is_forward("R") is False
    assert gear_is_forward(0) is False and gear_is_forward("0") is False
    assert gear_is_forward(-1) is False
    assert gear_is_forward("D") and gear_is_forward("S") and gear_is_forward("M2")
    assert gear_is_forward(1) and gear_is_forward("3") and gear_is_forward(TECH_DRIVE_GEAR)
    assert "shiftDown" not in TECH_DRIVE_SHIFT_LUA
    assert "-1" not in TECH_DRIVE_SHIFT_LUA
    assert "shiftUp" in TECH_DRIVE_SHIFT_LUA
    assert "automaticModes" in TECH_DRIVE_SHIFT_LUA
    assert "parkingbrake" in TECH_DRIVE_SHIFT_LUA and "clutch" in TECH_DRIVE_SHIFT_LUA

    class ArmVeh:
        def __init__(self, gear: object, *, fail_shift: bool = False) -> None:
            self.calls: list[dict] = []
            self.shifts: list[str] = []
            self.lua: list[str] = []
            self.fail_shift = fail_shift
            self.sensors = {"electrics": {"gear": gear}}

        def set_shift_mode(self, mode: str) -> None:
            self.shifts.append(mode)
            if self.fail_shift and mode == TECH_SHIFT_MODE:
                raise RuntimeError("ShiftModeSet ack timeout")

        def queue_lua_command(self, chunk: str, response: bool = False) -> None:
            self.lua.append(chunk)

        def control(self, **kw):
            self.calls.append(kw)

    neutral = ArmVeh("N")
    tech = BeamNGPyActuator(neutral)
    tech.note_engaged(True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = tech.apply(DriveCommand(steer=0.1, throttle=0.55, brake=0.0, seq=1, reason="ok"))
    log = buf.getvalue()
    assert out.applied is True
    kw = neutral.calls[-1]
    assert kw["throttle"] == 0.55 and kw["parkingbrake"] == 0.0 and kw["clutch"] == 0.0
    assert int(kw["gear"]) >= TECH_DRIVE_GEAR and int(kw["gear"]) != -1
    assert not is_reverse_control(kw)
    assert neutral.shifts == [TECH_SHIFT_MODE]
    assert "ok" in log and TECH_SHIFT_MODE in log
    assert neutral.lua == [TECH_DRIVE_SHIFT_LUA]
    assert tech.drive_arm_n == 1
    with redirect_stdout(io.StringIO()):
        tech.apply(DriveCommand(steer=0.1, throttle=0.55, brake=0.0, seq=2, reason="ok"))
    assert len(neutral.lua) == 1, "second arm inside the delay must not shiftUp again"
    tech._drive_arm_mono = time.monotonic() - (TECH_DRIVE_ARM_S + 0.05)
    with redirect_stdout(io.StringIO()):
        tech.apply(DriveCommand(steer=0.1, throttle=0.55, brake=0.0, seq=3, reason="ok"))
    assert len(neutral.lua) == 2
    assert all("shiftDown" not in chunk and "-1" not in chunk for chunk in neutral.lua)

    forward = ArmVeh("D")
    tech_d = BeamNGPyActuator(forward)
    tech_d.note_engaged(True)
    with redirect_stdout(io.StringIO()):
        tech_d.apply(DriveCommand(steer=0.0, throttle=0.4, brake=0.0, seq=4, reason="ok"))
    assert forward.lua == []
    assert forward.calls[-1]["gear"] >= TECH_DRIVE_GEAR
    assert forward.calls[-1]["parkingbrake"] == 0.0

    failing = ArmVeh("N", fail_shift=True)
    tech_f = BeamNGPyActuator(failing)
    tech_f.note_engaged(True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        failed = tech_f.apply(DriveCommand(steer=0.0, throttle=0.4, brake=0.0, seq=5, reason="ok"))
        tech_f.apply(DriveCommand(steer=0.0, throttle=0.4, brake=0.0, seq=6, reason="ok"))
    flog = buf.getvalue()
    assert failed.applied is True
    assert tech_f._shift_set is False
    assert failing.shifts == [TECH_SHIFT_MODE, TECH_SHIFT_MODE]
    assert "failed" in flog and "ShiftModeSet" in flog
    assert failing.calls[-1]["gear"] >= TECH_DRIVE_GEAR
    assert failing.lua  # still armed the lever; the ack failure is not silent

    class NoClutch:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def control(self, steering, throttle, brake, parkingbrake=0.0, gear=None):
            self.calls.append(
                {
                    "steering": steering,
                    "throttle": throttle,
                    "brake": brake,
                    "parkingbrake": parkingbrake,
                    "gear": gear,
                }
            )

    bare = NoClutch()
    tech_b = BeamNGPyActuator(bare)
    tech_b.note_engaged(True)
    with redirect_stdout(io.StringIO()):
        tech_b.apply(DriveCommand(steer=0.2, throttle=0.3, brake=0.0, seq=7, reason="ok"))
    assert "clutch" not in bare.calls[-1]
    assert bare.calls[-1]["gear"] >= TECH_DRIVE_GEAR
    assert bare.calls[-1]["parkingbrake"] == 0.0


def check_grab_loop_poll_before_electrics() -> None:
    """Soft Esc grab calls poll_vehicle() before read_electrics_inputs(vehicle).

    Electrics reuse the snapshot that poll just published. A swap is a hard fail.
    """
    rv = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    start = rv.find("bundle = backend.grab()")
    assert start >= 0, "run_vision grab missing"
    grab = rv[start:]
    end = grab.find("extras_bundle = extras.poll(")
    assert end > 0, "run_vision grab loop missing extras.poll"
    grab = grab[:end]
    poll_at = grab.find("backend.poll_vehicle()")
    el_at = grab.find("read_electrics_inputs(vehicle)")
    assert poll_at >= 0, "grab loop must call poll_vehicle()"
    assert el_at >= 0, "grab loop must call read_electrics_inputs(vehicle)"
    assert poll_at < el_at, (
        "grab loop must call poll_vehicle() before read_electrics_inputs(vehicle)"
    )


def check_soft_esc_heartbeat_coalesce() -> None:
    """Soft Esc gvd_state gate and Tech release-cmd: at most one rewrite per 100 ms.

    Engage state writes stay every tick. The rising edge flushes inside the
    window. A latched Disengage and shutdown still rewrite gvd_cmd.json.
    """
    import json

    from python.control.actuate import SOFT_ESC_FILE_PERIOD_S, cmd_path

    period = float(SOFT_ESC_FILE_PERIOD_S)

    def _due(engaged: bool, now: float, *, rising: bool = False) -> bool:
        ok = soft_esc_state_write_due(engaged, rising=rising, now=now)
        if ok:
            soft_esc_state_write_mark(now)
        return ok

    reset_soft_esc_state_writes()
    try:
        assert _due(False, 10.0) is True
        assert _due(False, 10.05) is False
        assert soft_esc_state_write_skips() == 1
        assert _due(False, 10.06, rising=True) is True
        assert soft_esc_state_write_skips() == 1
        assert _due(True, 10.07) is True
        assert _due(True, 10.08) is True
        assert soft_esc_state_write_skips() == 1
        assert _due(False, 10.09) is False
        assert soft_esc_state_write_skips() == 2
        stamped = 10.08 + period + 0.001
        assert _due(False, stamped) is True
        assert _due(False, stamped + period - 0.001) is False
        assert soft_esc_state_write_skips() == 3
        assert _due(False, stamped + period + 0.001) is True
        assert soft_esc_state_write_skips() == 3
    finally:
        reset_soft_esc_state_writes()

    rv = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    loop = rv.split("while True:", 1)[1]
    assert "rising_engage = bool(engaged) and not prev_engaged" in loop
    assert "soft_esc_state_write_due(bool(engaged), rising=rising_engage)" in loop
    assert "state_write_skips=" in loop
    assert "release_cmd_skips=" in loop
    assert "if state_due:\n                write_state(st)\n                soft_esc_state_write_mark()" in loop
    assert rv.count("write_state(") == 2

    class Veh:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self.shifts: list[str] = []
            self.ai_modes: list[str] = []

        def set_shift_mode(self, mode: str) -> None:
            self.shifts.append(mode)

        def ai_set_mode(self, mode: str) -> None:
            self.ai_modes.append(mode)

        def control(self, **kw):
            self.calls.append(kw)

    veh = Veh()
    tech = BeamNGPyActuator(veh)
    tech.note_engaged(False)
    # Injected clock, same pattern as the Soft Esc state-write gate. A stall
    # between stops must not cross SOFT_ESC_FILE_PERIOD_S on the wall clock.
    t0 = 10.0
    first = tech.stop(seq=1, reason="not_engaged", now=t0)
    assert first.brake == 0.0 and first.throttle == 0.0 and veh.calls == []
    payload = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert payload["seq"] == 1 and payload["engaged"] is False and payload["brake"] == 0.0
    held = tech.stop(seq=2, reason="not_engaged", now=t0 + (period - 0.001))
    assert held.brake == 0.0 and held.throttle == 0.0
    assert tech.release_cmd_skips == 1
    quiet = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert quiet["seq"] == 1 and quiet["brake"] == 0.0 and quiet["engaged"] is False
    tech.stop(seq=3, reason="not_engaged", now=t0 + period + 0.01)
    renewed = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert renewed["seq"] == 3 and renewed["brake"] == 0.0 and renewed["engaged"] is False
    assert tech.release_cmd_skips == 1
    tech.note_engaged(True)
    tech._latched = True
    tech.note_engaged(False)
    tech.stop(seq=4, reason="not_engaged", now=t0 + period + 0.02)
    edge = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert edge["seq"] == 4 and edge["brake"] == 0.0 and edge["engaged"] is False
    assert tech.release_cmd_skips == 1
    assert veh.calls[-1]["brake"] == 0.0 and veh.calls[-1]["throttle"] == 0.0
    tech.stop(seq=5, reason="shutdown", now=t0 + period + 0.03)
    shut = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert shut["seq"] == 5 and shut["reason"] == "shutdown"
    assert shut["brake"] == 0.0 and shut["engaged"] is False
    assert tech.release_cmd_skips == 1


def check_soft_esc_engage_rising_edge() -> None:
    """Grab order is poll, then note_engaged. The first engaged poll is real.

    The latch is still false at that poll. A live gvd_engage.json must refuse
    Soft Esc-hold. Also checks empty-map, thrown poll, close / no-vehicle
    cache clears, map copy, and latch restore.
    """
    import time

    import python.control.actuate as act
    from python.sensors.tech import TechSession

    class EgoSensors(dict):
        def __init__(self) -> None:
            super().__init__()
            self.n = 0
            self.speed = 3.0
            self.empty = False
            self.boom = False

        def poll(self) -> None:
            self.n += 1
            if self.boom:
                raise RuntimeError("sensors.poll failed")
            self.clear()
            if self.empty:
                return None
            self["electrics"] = {
                "wheelspeed": float(self.speed),
                "steering_input": 0.1,
                "throttle_input": 0.0,
                "brake_input": 0.0,
            }
            return None

    class EgoVeh:
        vid = "etk_player"
        options = {"model": "etk800"}

        def __init__(self) -> None:
            self.sensors = EgoSensors()
            self.state = {
                "pos": (1.0, 2.0, 0.0),
                "dir": (0.0, 1.0, 0.0),
                "up": (0.0, 0.0, 1.0),
                "vel": (0.0, 3.0, 0.0),
            }

    prev_latch = act.soft_esc_sensors_every_tick()
    engage = act.engage_path()
    prev_bytes = engage.read_bytes() if engage.is_file() else None
    session = TechSession({"wait_vehicle_s": 0, "sensors": {"electrics": True}})
    veh = EgoVeh()
    session.vehicle = veh
    session.attached = {"electrics": True}
    try:
        act.note_soft_esc_engaged(False)
        act.write_engage_flag(False)
        assert act.read_engage_flag(default=False) is False
        assert act.soft_esc_sensors_every_tick() is False

        first = session.poll()
        assert veh.sensors.n == 1
        assert first.speed_mps == 3.0
        assert first.pos == (1.0, 2.0, 0.0)
        cached = session._last_sensor_map
        assert cached is not None and cached is not veh.sensors
        assert cached["electrics"] is not veh.sensors["electrics"]
        veh.sensors["electrics"]["wheelspeed"] = 0.0
        assert cached["electrics"]["wheelspeed"] == 3.0

        session._sensors_poll_mono = time.monotonic()
        held = session.poll()
        assert veh.sensors.n == 1
        assert held.sensors_poll_ms == 0.0
        assert held.speed_mps == 3.0
        assert held.pos == (1.0, 2.0, 0.0)
        assert "coalesced" in held.note
        el_held = act.read_electrics(veh)
        assert el_held is not None and el_held["wheelspeed"] == 3.0
        assert veh.sensors.n == 1

        veh.sensors.speed = 9.0
        veh.state["pos"] = (9.0, 1.0, 0.0)
        veh.state["vel"] = (9.0, 0.0, 0.0)
        act.write_engage_flag(True)
        assert act.read_engage_flag(default=False) is True
        assert act.soft_esc_sensors_every_tick() is False
        session._sensors_poll_mono = time.monotonic()
        engaged_grab = session.poll()
        assert act.soft_esc_sensors_every_tick() is False
        assert veh.sensors.n == 2
        assert engaged_grab.speed_mps == 9.0
        assert engaged_grab.pos == (9.0, 1.0, 0.0)
        assert "coalesced" not in (engaged_grab.note or "")
        el_on = act.read_electrics(veh)
        assert el_on is not None and el_on["wheelspeed"] == 9.0
        assert veh.sensors.n == 2
        act.BeamNGPyActuator(veh).note_engaged(True)
        assert act.soft_esc_sensors_every_tick() is True
        veh.sensors.speed = 6.0
        veh.state["pos"] = (6.0, 1.0, 0.0)
        session._sensors_poll_mono = time.monotonic()
        latched = session.poll()
        assert veh.sensors.n == 3
        assert latched.speed_mps == 6.0
        assert latched.pos == (6.0, 1.0, 0.0)
        assert "coalesced" not in (latched.note or "")

        act.note_soft_esc_engaged(False)
        act.write_engage_flag(False)
        prior_map = session._last_sensor_map
        prior_data = session._last_vehicle_data
        assert prior_map is not None and prior_data is not None
        assert prior_data.speed_mps == 6.0
        session._sensors_poll_mono = time.monotonic() - 1.0
        frozen_mono = session._sensors_poll_mono
        veh.sensors.empty = True
        n_empty = veh.sensors.n
        session.poll()
        assert veh.sensors.n == n_empty + 1
        assert session._last_sensor_map is prior_map
        assert session._last_vehicle_data is prior_data
        assert session._sensors_poll_mono == frozen_mono
        assert prior_map["electrics"]["wheelspeed"] == 6.0
        veh.sensors.empty = False

        veh.sensors.speed = 4.0
        rearmed = session.poll()
        assert rearmed.speed_mps == 4.0
        assert session._sensors_poll_mono is not None
        assert session._last_sensor_map is not None
        assert session._last_sensor_map["electrics"]["wheelspeed"] == 4.0
        session._sensors_poll_mono = time.monotonic()
        veh.sensors.boom = True
        act.write_engage_flag(True)
        n_throw = veh.sensors.n
        session.poll()
        assert veh.sensors.n == n_throw + 1
        assert session._sensors_poll_mono is None
        assert session._sensors_poll_vehicle_id is None
        veh.sensors.boom = False
        veh.sensors.speed = 8.0
        act.write_engage_flag(False)
        assert act.soft_esc_sensors_every_tick() is False
        n_retry = veh.sensors.n
        retry = session.poll()
        assert veh.sensors.n == n_retry + 1
        assert retry.speed_mps == 8.0
        assert "coalesced" not in (retry.note or "")

        session._sensors_poll_mono = time.monotonic()
        n_none = veh.sensors.n
        session.vehicle = None
        missing = session.poll()
        assert missing.connected is False
        assert session._last_sensor_map is None
        assert session._last_vehicle_data is None
        assert session._sensors_poll_mono is None
        assert session._sensors_poll_vehicle_id is None
        session.vehicle = veh
        back = session.poll()
        assert veh.sensors.n == n_none + 1
        assert back.speed_mps == 8.0
        assert "coalesced" not in (back.note or "")

        session._sensors_poll_mono = time.monotonic()
        n_close = veh.sensors.n
        session.close()
        assert session.vehicle is None
        assert session._last_sensor_map is None
        assert session._last_vehicle_data is None
        assert session._sensors_poll_mono is None
        assert session._sensors_poll_vehicle_id is None
        session.vehicle = veh
        after_close = session.poll()
        assert veh.sensors.n == n_close + 1
        assert after_close.speed_mps == 8.0
        assert "coalesced" not in (after_close.note or "")
    finally:
        act.note_soft_esc_engaged(prev_latch)
        if prev_bytes is None:
            engage.unlink(missing_ok=True)
        else:
            engage.write_bytes(prev_bytes)


def main() -> None:
    # 2) stale heartbeat → zero throttle + brake
    cmd = safe_command(
        engaged=True,
        heartbeat_ok=False,
        path_debug_preview=False,
        allow_preview_drive=False,
        path_ego=[{"x": 0, "y": 5, "z": 0}, {"x": 0.2, "y": 10, "z": 0}, {"x": 0.3, "y": 15, "z": 0}],
        planner={"target_v": 12, "aeb": "off", "ttc_lead": None},
        ego_speed_mps=8.0,
        seq=1,
    )
    assert cmd.throttle == 0.0 and cmd.brake == 1.0 and cmd.reason == "heartbeat_stale", cmd
    assert heartbeat_fresh(__import__("time").time() - 1.0) is False
    assert heartbeat_fresh(__import__("time").time()) is True

    # 3) preview path → no actuate unless flag
    cmd2 = safe_command(
        engaged=True,
        heartbeat_ok=True,
        path_debug_preview=True,
        allow_preview_drive=False,
        path_ego=[{"x": 0, "y": 5, "z": 0}] * 5,
        planner={"target_v": 10, "aeb": "off"},
        ego_speed_mps=5.0,
        seq=2,
    )
    assert cmd2.reason == "preview_blocked" and cmd2.throttle == 0.0 and cmd2.brake == 1.0, cmd2
    ok, _ = may_drive(engaged=True, heartbeat_ok=True, path_debug_preview=True, allow_preview_drive=True)
    assert ok
    cmd3 = safe_command(
        engaged=True,
        heartbeat_ok=True,
        path_debug_preview=True,
        allow_preview_drive=True,
        path_ego=[{"x": 0.5, "y": float(i), "z": 0} for i in range(12)],
        planner={"target_v": 10, "aeb": "off"},
        ego_speed_mps=5.0,
        seq=3,
    )
    assert cmd3.reason == "ok" and cmd3.throttle >= 0.0, cmd3

    # AEB
    cmd_aeb = safe_command(
        engaged=True,
        heartbeat_ok=True,
        path_debug_preview=False,
        allow_preview_drive=False,
        path_ego=[{"x": 0, "y": float(i), "z": 0} for i in range(12)],
        planner={"target_v": 10, "aeb": "brake", "ttc_lead": 0.5},
        ego_speed_mps=12.0,
        seq=4,
    )
    assert cmd_aeb.throttle == 0.0 and cmd_aeb.brake == 1.0, cmd_aeb

    # 4) Electrics mock → speed
    class V:
        def __init__(self):
            self.sensors = _Pollable(
                {"electrics": {"wheelspeed": 7.5, "airspeed": 7.4, "steering_input": 0.1}}
            )

    spd, steeri = read_electrics_speed(V())
    assert spd == 7.5 and steeri == 0.1, (spd, steeri)

    # cmd json actuator writes the file but never claims applied without a Lua ack (M6 bus)
    act = CmdJsonActuator()
    out = act.stop(seq=9, reason="unit_stop")
    assert out.applied is False and out.reason == "cmd_json_idle" and out.brake == 1.0
    from python.control.actuate import cmd_path

    assert cmd_path().is_file()
    import json

    payload = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert payload["engaged"] is False and payload["brake"] == 1.0 and payload["seq"] == 9
    # Retail bus unchanged: no parkingbrake/gear in gvd_cmd.json (Lua zeros PB itself).
    assert "parkingbrake" not in payload and "gear" not in payload, payload

    rest = tech_control_kwargs(0.0, 0.0, 1.0)
    assert rest["gear"] == TECH_HOLD_GEAR == 0
    assert rest["brake"] == 1.0 and rest["parkingbrake"] == 1.0
    assert rest.get("gear") != -1
    assert not is_reverse_control(rest)
    assert not is_arcade_reverse_hold(rest)
    assert is_reverse_control({"gear": -1})
    assert is_arcade_reverse_hold({"steering": 0.0, "throttle": 0.0, "brake": 1.0})
    rolling = tech_control_kwargs(0.1, 0.0, 1.0, speed_mps=12.0)
    assert rolling["brake"] == 1.0 and rolling["parkingbrake"] == 0.0 and rolling["gear"] == 0
    assert rolling.get("gear") != -1
    drive_kw = tech_control_kwargs(0.2, 0.4, 0.0)
    assert drive_kw["throttle"] == 0.4 and drive_kw["parkingbrake"] == 0.0
    assert drive_kw["clutch"] == 0.0
    assert drive_kw["gear"] >= TECH_DRIVE_GEAR and drive_kw["gear"] != -1
    slow = tech_control_kwargs(0.0, 0.0, 0.6)
    assert slow["brake"] == 0.6 and slow.get("parkingbrake", 0.0) == 0.0
    assert slow["gear"] == TECH_HOLD_GEAR == 0  # forward gear only with throttle>0
    coast = tech_control_kwargs(0.0, 0.0, 0.0)
    assert coast["gear"] == 0 and coast["throttle"] == 0.0
    rel = tech_control_kwargs(0.9, 0.9, 0.9, release=True)
    assert rel == {"steering": 0.0, "throttle": 0.0, "brake": 0.0, "parkingbrake": 0.0}
    assert rel.get("gear", 0) != -1

    # Tech: disengaged must not call vehicle.control (no brake takeover). One zero
    # release on the falling edge, then silence. Engaged stop/hold: realistic_automatic,
    # gear=0 + brake ±parkingbrake, never gear=-1. Drive pins gear>=1.
    class FakeVeh:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self.shifts: list[str] = []
            self.ai_modes: list[str] = []

        def set_shift_mode(self, mode: str) -> None:
            self.shifts.append(mode)

        def ai_set_mode(self, mode: str) -> None:
            self.ai_modes.append(mode)

        def control(self, **kw):
            self.calls.append(kw)

    def _assert_no_reverse(kw: dict) -> None:
        assert not is_reverse_control(kw), kw
        gear = kw.get("gear")
        assert gear != -1, kw
        assert not isinstance(gear, str), kw  # never letter "D"
        if gear is not None:
            assert isinstance(gear, int) and gear >= 0, kw
        if float(kw.get("throttle") or 0.0) > 1e-6:
            assert int(kw["gear"]) >= TECH_DRIVE_GEAR, kw
        if float(kw.get("throttle") or 0.0) <= 1e-6 and float(kw.get("brake") or 0.0) >= 0.99:
            assert float(kw.get("brake") or 0.0) > 0.0 or float(kw.get("parkingbrake") or 0.0) > 0.0, kw
            if gear is not None:
                assert gear == 0 or gear >= 1, kw
        assert not is_arcade_reverse_hold(kw), kw

    veh = FakeVeh()
    tech = BeamNGPyActuator(veh)
    tech.note_engaged(False)
    idle = tech.stop(seq=1, reason="not_engaged")
    assert idle.applied is False and idle.throttle == 0.0 and idle.brake == 0.0 and veh.calls == [], veh.calls
    # A disengaged tick clears a stale brake:1 cmd bus even before this process latched the car.
    idle_payload = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert idle_payload["engaged"] is False and idle_payload["brake"] == 0.0 and idle_payload["throttle"] == 0.0
    tech.note_engaged(True)
    drive = tech.apply(DriveCommand(steer=0.2, throttle=0.4, brake=0.0, seq=2, reason="ok"))
    assert drive.applied is True and veh.calls[-1]["throttle"] == 0.4
    assert veh.calls[-1].get("parkingbrake", 0.0) == 0.0
    assert veh.calls[-1]["gear"] >= TECH_DRIVE_GEAR
    assert veh.shifts == [TECH_SHIFT_MODE] and TECH_SHIFT_MODE == "realistic_automatic"
    _assert_no_reverse(veh.calls[-1])

    hold = tech.stop(seq=3, reason="preview_blocked")
    assert hold.applied is True and hold.brake == 1.0 and hold.throttle == 0.0
    kw = veh.calls[-1]
    _assert_no_reverse(kw)
    assert kw["gear"] == TECH_HOLD_GEAR == 0
    assert kw["brake"] == 1.0 and kw["parkingbrake"] == 1.0

    aeb_cmd = plan_command(
        path_ego=[{"x": 0, "y": float(i), "z": 0} for i in range(12)],
        planner={"target_v": 10, "aeb": "brake", "ttc_lead": 0.5},
        ego_speed_mps=12.0,
        seq=30,
    )
    aeb_cmd.reason = "ok"
    aeb_out = tech.apply(aeb_cmd)
    assert aeb_out.applied is True and aeb_out.brake == 1.0
    _assert_no_reverse(veh.calls[-1])
    assert veh.calls[-1]["gear"] == 0 and veh.calls[-1]["brake"] == 1.0

    class MovingVeh(FakeVeh):
        def __init__(self) -> None:
            super().__init__()
            self.sensors = _Pollable(
                {"electrics": {"wheelspeed": 12.0, "steering_input": 0.0, "throttle_input": 0.0, "brake_input": 0.0}}
            )

    moving = MovingVeh()
    tech_m = BeamNGPyActuator(moving)
    tech_m.note_engaged(True)
    tech_m.apply(DriveCommand(steer=0.0, throttle=0.0, brake=1.0, seq=31, reason="ok"))
    mkw = moving.calls[-1]
    assert mkw["brake"] == 1.0 and mkw["parkingbrake"] == 0.0 and mkw["gear"] == 0
    _assert_no_reverse(mkw)

    class NoGearVeh:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def control(self, steering, throttle, brake, parkingbrake=0.0):
            self.calls.append(
                {"steering": steering, "throttle": throttle, "brake": brake, "parkingbrake": parkingbrake}
            )

    ng = NoGearVeh()
    tech_ng = BeamNGPyActuator(ng)
    tech_ng.note_engaged(True)
    tech_ng.stop(seq=32, reason="preview_blocked")
    assert ng.calls[-1]["brake"] == 1.0 and ng.calls[-1]["parkingbrake"] == 1.0
    assert "gear" not in ng.calls[-1]
    _assert_no_reverse(ng.calls[-1])

    resume = tech.apply(DriveCommand(steer=0.0, throttle=0.35, brake=0.0, seq=33, reason="ok"))
    assert resume.applied is True
    assert veh.calls[-1]["gear"] >= 1 and veh.calls[-1]["parkingbrake"] == 0.0
    _assert_no_reverse(veh.calls[-1])

    tech.note_engaged(False)
    n = len(veh.calls)
    # Injected clock so a stall between these stops cannot expire the window.
    edge = tech.stop(seq=4, reason="not_engaged", now=40.0)
    assert edge.applied is False and edge.throttle == 0.0 and edge.brake == 0.0
    assert veh.calls[-1] == {"steering": 0.0, "throttle": 0.0, "brake": 0.0, "parkingbrake": 0.0}
    assert veh.calls[-1].get("gear", 0) != -1
    assert veh.ai_modes[-1] == "disabled"
    assert veh.shifts[-1] == TECH_PLAYER_SHIFT_MODE == "arcade"
    edge_payload = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert edge_payload["engaged"] is False and edge_payload["brake"] == 0.0 and edge_payload["throttle"] == 0.0
    tech.stop(seq=5, reason="not_engaged", now=40.05)
    assert len(veh.calls) == n + 1, veh.calls  # no further takeover while OFF
    assert len(veh.ai_modes) == 1
    quiet = json.loads(cmd_path().read_text(encoding="utf-8"))
    # Soft Esc release-cmd inside 100 ms keeps the falling-edge file (seq 4).
    assert quiet["brake"] == 0.0 and quiet["engaged"] is False and quiet["seq"] == 4
    assert tech.release_cmd_skips == 1

    # Hold left brake=1 on the car; Disengage must zero it and release AI, not leave brake=1.
    tech.note_engaged(True)
    held = tech.stop(seq=6, reason="preview_blocked")
    assert held.applied is True and held.brake == 1.0 and veh.calls[-1]["brake"] == 1.0
    assert veh.shifts[-1] == TECH_SHIFT_MODE
    tech.note_engaged(False)
    n_hold = len(veh.calls)
    n_ai = len(veh.ai_modes)
    hand = tech.stop(seq=7, reason="not_engaged")
    assert hand.applied is False and hand.throttle == 0.0 and hand.brake == 0.0
    assert veh.calls[-1]["throttle"] == 0.0 and veh.calls[-1]["brake"] == 0.0
    assert veh.calls[-1]["parkingbrake"] == 0.0
    assert veh.ai_modes[-1] == "disabled" and len(veh.ai_modes) == n_ai + 1
    assert veh.shifts[-1] == "arcade"
    hand_payload = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert hand_payload["engaged"] is False and hand_payload["brake"] == 0.0 and hand_payload["throttle"] == 0.0
    tech.stop(seq=8, reason="not_engaged")
    assert len(veh.calls) == n_hold + 1
    assert len(veh.ai_modes) == n_ai + 1
    again = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert again["brake"] == 0.0 and again["throttle"] == 0.0 and again["engaged"] is False

    # Falling-edge release with the shifter never armed must not call realistic_automatic.
    bare = FakeVeh()
    tech_bare = BeamNGPyActuator(bare)
    tech_bare._latched = True
    tech_bare._shift_set = False
    tech_bare.note_engaged(False)
    bare_edge = tech_bare.stop(seq=40, reason="not_engaged")
    assert bare_edge.throttle == 0.0 and bare_edge.brake == 0.0
    assert bare.calls[-1] == {"steering": 0.0, "throttle": 0.0, "brake": 0.0, "parkingbrake": 0.0}
    assert TECH_SHIFT_MODE not in bare.shifts, bare.shifts
    assert bare.shifts == [TECH_PLAYER_SHIFT_MODE]
    assert bare.ai_modes == ["disabled"]
    assert tech_bare._latched is False and tech_bare._shift_set is False
    n_bare = len(bare.calls)
    tech_bare.stop(seq=41, reason="not_engaged")
    assert len(bare.calls) == n_bare

    # Arcade restore throw keeps the latch so the next disengaged tick retries.
    class ArcadeFailVeh(FakeVeh):
        def __init__(self) -> None:
            super().__init__()
            self.arcade_ok = False

        def set_shift_mode(self, mode: str) -> None:
            self.shifts.append(mode)
            if mode == TECH_PLAYER_SHIFT_MODE and not self.arcade_ok:
                raise RuntimeError("arcade handoff failed")

    fail = ArcadeFailVeh()
    tech_fail = BeamNGPyActuator(fail)
    tech_fail.note_engaged(True)
    tech_fail.apply(DriveCommand(steer=0.0, throttle=0.3, brake=0.0, seq=42, reason="ok"))
    assert fail.shifts == [TECH_SHIFT_MODE] and tech_fail._latched is True
    tech_fail.note_engaged(False)
    missed = tech_fail.stop(seq=43, reason="not_engaged")
    assert missed.throttle == 0.0 and missed.brake == 0.0
    assert fail.calls[-1]["brake"] == 0.0 and fail.calls[-1]["throttle"] == 0.0
    assert fail.shifts == [TECH_SHIFT_MODE, TECH_PLAYER_SHIFT_MODE]
    assert tech_fail._latched is True and tech_fail._shift_set is True
    n_fail = len(fail.calls)
    n_realistic = fail.shifts.count(TECH_SHIFT_MODE)
    tech_fail.stop(seq=44, reason="not_engaged")
    assert tech_fail._latched is True
    assert len(fail.calls) == n_fail + 1
    assert fail.calls[-1]["brake"] == 0.0 and fail.calls[-1]["parkingbrake"] == 0.0
    assert fail.shifts.count(TECH_SHIFT_MODE) == n_realistic
    assert fail.shifts[-1] == TECH_PLAYER_SHIFT_MODE
    fail.arcade_ok = True
    n_retry = len(fail.calls)
    landed = tech_fail.stop(seq=45, reason="not_engaged")
    assert landed.brake == 0.0 and landed.throttle == 0.0
    assert tech_fail._latched is False and tech_fail._shift_set is False
    assert fail.shifts[-1] == TECH_PLAYER_SHIFT_MODE
    assert len(fail.calls) == n_retry + 1
    tech_fail.stop(seq=46, reason="not_engaged")
    assert len(fail.calls) == n_retry + 1
    fail_payload = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert fail_payload["engaged"] is False and fail_payload["brake"] == 0.0 and fail_payload["throttle"] == 0.0

    for kw in veh.calls + moving.calls + ng.calls + bare.calls + fail.calls:
        _assert_no_reverse(kw)

    check_drive_gear_arm()
    check_grab_loop_poll_before_electrics()
    check_electrics_segment_timer()
    check_soft_esc_engage_rising_edge()
    check_soft_esc_heartbeat_coalesce()

    print("test_m3_actuate: OK")


def test_m3_actuate() -> None:
    import python.control.actuate as _act

    prev = _act.soft_esc_sensors_every_tick()
    try:
        main()
    finally:
        _act.note_soft_esc_engaged(prev)


if __name__ == "__main__":
    test_m3_actuate()
