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
    read_electrics_speed,
    safe_command,
    tech_control_kwargs,
)


class _Pollable(dict):
    def poll(self):
        return self


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
    edge = tech.stop(seq=4, reason="not_engaged")
    assert edge.applied is False and edge.throttle == 0.0 and edge.brake == 0.0
    assert veh.calls[-1] == {"steering": 0.0, "throttle": 0.0, "brake": 0.0, "parkingbrake": 0.0}
    assert veh.calls[-1].get("gear", 0) != -1
    assert veh.ai_modes[-1] == "disabled"
    assert veh.shifts[-1] == TECH_PLAYER_SHIFT_MODE == "arcade"
    edge_payload = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert edge_payload["engaged"] is False and edge_payload["brake"] == 0.0 and edge_payload["throttle"] == 0.0
    tech.stop(seq=5, reason="not_engaged")
    assert len(veh.calls) == n + 1, veh.calls  # no further takeover while OFF
    assert len(veh.ai_modes) == 1
    quiet = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert quiet["brake"] == 0.0 and quiet["engaged"] is False and quiet["seq"] == 5

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

    check_grab_loop_poll_before_electrics()

    print("test_m3_actuate: OK")


if __name__ == "__main__":
    main()
