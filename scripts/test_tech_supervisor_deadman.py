#!/usr/bin/env python3
"""Tech: a frozen control loop trips the mod's dead-man too (no BeamNG).

On Tech ``BeamNGPyActuator`` drives with ``vehicle.control``. Right after
that call, on the same thread, it writes the retail ``gvd_cmd.json``
seq/heartbeat with ``heartbeat_only``. ``gvd_main.applyCmdJson`` keeps its
hands off while that stream is fresh, and runs its retail stale (brake hold)
and dead (release + disengage ``command_stream_dead``) logic when it stops.

A child process runs the tick loop (``write_state`` + ``BeamNGPyActuator``
on a fake vehicle) and after ``FREEZE_AT`` ticks the whole loop stops: no
grab, no control, no write. The real ``main.lua`` runs on the sim-clock
lockstep harness of ``test_frozen_supervisor_deadman.py`` at its frame rates,
with the same declared limits. A second check disengages cleanly through
``stop()`` and then stops writing: no brake and no ``command_stream_dead``.

Not covered: whether the mod's brake beats BeamNGpy's last vehicle.control
in the real game.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from test_frozen_supervisor_deadman import (  # noqa: E402
    FPS,
    FREEZE_AT,
    Lockstep,
    _wait_bus,
    check_bounds,
    check_model,
    hold_and_disengage,
)

STEER, THROTTLE = 0.1, 0.3
QUIET_S = 2.0  # sim time after a clean disengage with no writes: no brake, no command_stream_dead


class _FakeVehicle:
    """What BeamNGPyActuator calls on a BeamNGpy vehicle. Records control calls."""

    sensors = None

    def __init__(self, log=None) -> None:
        self.calls: list[dict] = []
        self.log = log

    def control(self, **kw) -> None:
        self.calls.append(dict(kw))
        if self.log is not None:
            self.log("control", kw)

    def set_shift_mode(self, mode) -> None:
        pass

    def set_lights(self, **kw) -> None:
        if self.log is not None:
            self.log("set_lights", kw)

    def queue_lua_command(self, code, *a) -> None:
        if self.log is not None:
            self.log("queue_lua_command", code)


def _supervisor() -> int:
    """Child: one grab -> state -> vehicle.control tick per stdin line.

    ``freeze`` stops the whole tick loop on its own thread (no grab, no control, no write).
    ``disengage`` is one disengaged tick through the real ``stop()`` path.
    """
    from python.control.actuate import BeamNGPyActuator, DriveCommand
    from python.runtime.state_io import write_state

    if not _wait_bus():
        return 3
    veh = _FakeVehicle()
    act = BeamNGPyActuator(veh)
    act.note_engaged(True)
    seq = 0
    for line in sys.stdin:
        c = line.strip()
        if c == "freeze":
            print(f"FREEZE {seq} controls={len(veh.calls)} threads={threading.active_count()}", flush=True)
            threading.Event().wait()  # the whole tick loop stops here, on its own thread
        seq += 1
        if c == "disengage":
            write_state({"engaged": False, "policy": "modular", "cmd_seq": seq, "disengage_reason": "none"})
            act.note_engaged(False)
            act.stop(seq=seq, reason="not_engaged")
        else:
            write_state({"engaged": True, "policy": "modular", "cmd_seq": seq, "disengage_reason": "none"})
            out = act.apply(DriveCommand(steer=STEER, throttle=THROTTLE, brake=0.0, seq=seq, reason="ok"))
            assert out.applied, out.reason
        print(f"TICK {seq}", flush=True)
    return 0


def check_stamp_follows_control_on_same_thread() -> None:
    """The seq/heartbeat write is the next thing after vehicle.control, same thread."""
    import python.control.actuate as A

    order: list[tuple] = []
    veh = _FakeVehicle(log=lambda what, arg: order.append((what, threading.get_ident(), arg)))
    real_write, real_path = A.atomic_write_json, A.cmd_path
    try:
        A.atomic_write_json = lambda p, payload, **kw: order.append(("write", threading.get_ident(), payload)) or True
        A.cmd_path = lambda: Path("gvd_cmd.json")
        act = A.BeamNGPyActuator(veh)
        act.note_engaged(True)
        out = act.apply(A.DriveCommand(steer=STEER, throttle=THROTTLE, brake=0.0, seq=7, reason="ok"))
        assert out.applied, out.reason
        names = [o[0] for o in order]
        i = names.index("control")
        assert names[i + 1] == "write", names
        assert order[i][1] == order[i + 1][1] == threading.get_ident(), order
        beat = order[i + 1][2]
        assert beat["seq"] == 7 and beat["engaged"] is True and beat["heartbeat_only"] is True, beat
        assert set(beat) == {"steer", "throttle", "brake", "seq", "engaged", "heartbeat_mtime", "reason", "heartbeat_only"}, beat
        # A release (disengaged) is not a heartbeat: it still writes engaged:false only.
        order.clear()
        act.note_engaged(False)
        act.stop(seq=8, reason="not_engaged")
        writes = [o[2] for o in order if o[0] == "write"]
        assert writes and all(w["engaged"] is False and "heartbeat_only" not in w for w in writes), writes
    finally:
        A.atomic_write_json, A.cmd_path = real_write, real_path


def check_frozen_tech_loop_brakes_then_disengages(fps) -> None:
    ls = Lockstep(Path(__file__).resolve(), fps)
    try:
        res = ls.run(["tick"] * FREEZE_AT + ["freeze"], tail_s=2.0)
        live = [ev for t, ev, _ in res["frames"] if t <= res["t_freeze"]]
        assert len(live) >= FREEZE_AT, len(live)
        fight = [e for ev in live for e in ev]
        assert fight == [], f"mod actuated on top of vehicle.control while fresh: {fight[:6]}"
        assert all(eng for t, _, eng in res["frames"] if t <= res["t_freeze"]), "mod disengaged while the Tech loop was live"
        parts = dict(kv.split("=") for kv in res["freeze_line"].split()[2:])
        assert int(parts["controls"]) == FREEZE_AT, res["freeze_line"]  # one vehicle.control per tick
        assert int(parts["threads"]) == 1, res["freeze_line"]  # no helper thread keeps a heartbeat alive
        t_hold, t_dis, before_hold, throttle_while_held = hold_and_disengage(res)
        assert t_hold is not None, "no brake hold after the Tech loop froze"
        assert t_dis is not None, "no disengage after the Tech loop froze"
        assert before_hold == [], f"mod input before the hold: {before_hold[:3]}"
        assert throttle_while_held == [], throttle_while_held
        cmd = json.loads((ls.bus / "gvd_cmd.json").read_text(encoding="utf-8"))
        assert cmd["seq"] == FREEZE_AT and cmd["engaged"] is True and cmd["heartbeat_only"] is True, cmd
        engage = json.loads((ls.bus / "gvd_engage.json").read_text(encoding="utf-8"))
        assert engage["engaged"] is False and engage["disengage_reason"] == "command_stream_dead", engage
        check_bounds("test_tech_supervisor_deadman", fps, t_hold - res["t_last"], t_dis - res["t_last"], res["dt_max"])
        check_model("test_tech_supervisor_deadman", fps, res, t_hold - res["t_last"], t_dis - res["t_last"])
    finally:
        ls.close()


def check_clean_disengage_then_silence(fps) -> None:
    """Disengage through stop(), then no more writes: the last file is engaged:false, so nothing fires.

    Only the actuator disengages here (no gvd_engage.json write), so the mod is still engaged and
    the cmd file alone decides: a heartbeat_only engaged:true left behind would brake at CMD_STALE_S.
    """
    ls = Lockstep(Path(__file__).resolve(), fps)
    try:
        res = ls.run(["tick"] * FREEZE_AT + ["disengage", "freeze"], tail_s=QUIET_S + 0.5)
        t_off = res["marks"]["disengage"]
        after = [(t, ev, eng) for t, ev, eng in res["frames"] if t > t_off]
        assert after[-1][0] - t_off >= QUIET_S, (after[-1][0], t_off)
        brake = [(round(t - t_off, 4), ev) for t, ev, _ in after
                 if dict(ev).get("parkingbrake", 0.0) > 0.0 or dict(ev).get("brake", 0.0) > 0.0]
        assert brake == [], f"brake input after a clean disengage: {brake[:3]}"
        moved = [(round(t - t_off, 4), ev) for t, ev, _ in after if ev]
        assert moved == [], f"mod input after a clean disengage: {moved[:3]}"
        assert all(eng for _, _, eng in after), "mod disengaged itself after a clean supervisor disengage"
        cmd = json.loads((ls.bus / "gvd_cmd.json").read_text(encoding="utf-8"))
        assert cmd["engaged"] is False and "heartbeat_only" not in cmd, cmd
        engage = json.loads((ls.bus / "gvd_engage.json").read_text(encoding="utf-8"))
        assert engage.get("disengage_reason") != "command_stream_dead", engage
        print(
            f"test_tech_supervisor_deadman {fps} fps: clean disengage, {after[-1][0] - t_off:.2f} s sim of silence: "
            "no brake, no command_stream_dead",
            flush=True,
        )
    finally:
        ls.close()


def main() -> None:
    if "--supervisor" in sys.argv:
        sys.exit(_supervisor())
    check_stamp_follows_control_on_same_thread()
    for fps in FPS:
        check_clean_disengage_then_silence(fps)
        check_frozen_tech_loop_brakes_then_disengages(fps)
    print("test_tech_supervisor_deadman: OK")


def test_tech_supervisor_deadman() -> None:
    main()


if __name__ == "__main__":
    main()
