#!/usr/bin/env python3
"""Tech: a frozen control loop trips the mod's dead-man too (no BeamNG).

On Tech ``BeamNGPyActuator`` drives with ``vehicle.control``. Right after
that call, on the same thread, it writes the retail ``gvd_cmd.json``
seq/heartbeat with ``heartbeat_only``. ``gvd_main.applyCmdJson`` keeps its
hands off while that stream is fresh, and runs its retail stale (brake hold)
and dead (release + disengage ``command_stream_dead``) logic when it stops.

A child process runs the tick loop (``write_state`` + ``BeamNGPyActuator``
on a fake vehicle) and after ``FREEZE_AT`` ticks the whole loop stops: no
grab, no control, no write. ``scripts/gvd_lua_sim.lua`` runs the real
``main.lua`` on the same folder in wall time. Bounds are the mod's own
constants plus two of its 20 Hz polls.

Not covered: whether the mod's brake beats BeamNGpy's last vehicle.control
in the real game.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gvd_test_env import isolated_env  # noqa: E402
from test_frozen_supervisor_deadman import _Sim, _lua_const  # noqa: E402

FREEZE_AT = 40  # ticks of live driving, then the whole loop stops
TICK_S = 0.05
STEER, THROTTLE = 0.1, 0.3


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
    """Child: grab -> state -> vehicle.control tick loop; after FREEZE_AT ticks it stops."""
    from python.control.actuate import BeamNGPyActuator, DriveCommand
    from python.runtime.paths import bus_identity
    from python.runtime.state_io import write_state

    deadline = time.monotonic() + 10.0
    while not bus_identity().matched:
        if time.monotonic() > deadline:
            print("NO_BUS", flush=True)
            return 3
        time.sleep(0.02)
    veh = _FakeVehicle()
    act = BeamNGPyActuator(veh)
    act.note_engaged(True)
    seq = 0
    while True:
        t0 = time.monotonic()
        if seq == FREEZE_AT:
            print(f"FREEZE {seq} controls={len(veh.calls)} threads={threading.active_count()}", flush=True)
            threading.Event().wait()  # the whole tick loop stops here, on its own thread
        seq += 1
        write_state({"engaged": True, "policy": "modular", "cmd_seq": seq, "disengage_reason": "none"})
        out = act.apply(DriveCommand(steer=STEER, throttle=THROTTLE, brake=0.0, seq=seq, reason="ok"))
        assert out.applied, out.reason
        time.sleep(max(0.0, TICK_S - (time.monotonic() - t0)))


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


def check_frozen_tech_loop_brakes_then_disengages() -> None:
    stale_s = _lua_const("CMD_STALE_S")
    dead_s = _lua_const("CMD_DEAD_S")
    poll_s = _lua_const("CMD_POLL_S")
    lua = shutil.which("lua5.1") or shutil.which("lua")
    assert lua, "lua5.1 is not on PATH (the mod harness needs it; not a skip)"
    root = Path(tempfile.mkdtemp(prefix="gvd-tech-deadman-"))
    sup = sim = None
    try:
        env = isolated_env(os.environ, root)
        env["GVD_PRODUCT"] = "drive"
        user = Path(env["LOCALAPPDATA"]) / "BeamNG" / "BeamNG.drive" / "current"
        bus = user / "Documents" / "GVD"
        bus.mkdir(parents=True, exist_ok=True)
        sim = _Sim(lua, {**env, "GVD_SIM_USER": str(user)})
        _, engaged = sim.send("engage")
        assert engaged, "Alt+G in the sim did not engage"
        sim.send("step 0.02")  # gvd_link.json carries lua_bus before the supervisor starts
        sup = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--supervisor"],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            text=True,
        )
        out: list[str] = []
        threading.Thread(target=lambda: out.extend(sup.stdout), daemon=True).start()

        live_steps = 0
        mod_events_while_fresh: list[tuple[str, float]] = []
        freeze_line: str | None = None
        t_hold = t_dis = None
        throttle_while_held = []
        last = time.time()
        end = last + 20.0
        while time.time() < end:
            time.sleep(1.0 / 60.0)
            now = time.time()
            events, engaged = sim.send(f"step {now - last:.5f}")
            last = now
            stamp = time.time()
            if freeze_line is None:
                for line in list(out):
                    if line.startswith("FREEZE"):
                        freeze_line = line.strip()
                    assert not line.startswith("NO_BUS"), "supervisor never saw a live lua_bus"
                    assert not line.startswith("Traceback"), out
            if freeze_line is None:
                live_steps += 1
                mod_events_while_fresh.extend(events)  # vehicle.control drives; the mod must not
                assert engaged, "mod disengaged while the Tech loop was live"
                continue
            got = dict(events)
            if t_hold is None and got.get("throttle") == 0.0 and got.get("steering") == 0.0 and got.get("parkingbrake") == 1.0:
                t_hold = stamp
            elif t_hold is None and got:
                mod_events_while_fresh.extend(events)  # anything before the hold is a fight
            elif t_hold is not None and t_dis is None and got.get("throttle", 0.0) > 0.0:
                throttle_while_held.append(got["throttle"])
            if not engaged and t_dis is None:
                t_dis = stamp
                break
        assert freeze_line is not None, out
        assert live_steps >= 20, f"too few live steps before the freeze ({live_steps})"
        assert mod_events_while_fresh == [], f"mod actuated on top of vehicle.control while fresh: {mod_events_while_fresh[:6]}"
        assert t_hold is not None, "no brake hold after the Tech loop froze"
        assert t_dis is not None, "no disengage after the Tech loop froze"
        parts = dict(kv.split("=") for kv in freeze_line.split()[2:])
        assert int(parts["controls"]) == FREEZE_AT, freeze_line  # one vehicle.control per tick
        assert int(parts["threads"]) == 1, freeze_line  # no helper thread keeps a heartbeat alive
        assert sup.poll() is None, "supervisor exited; its loop should be stuck"
        cmd = json.loads((bus / "gvd_cmd.json").read_text(encoding="utf-8"))
        assert cmd["seq"] == FREEZE_AT and cmd["engaged"] is True and cmd["heartbeat_only"] is True, cmd
        t_last = float(cmd["heartbeat_mtime"])  # stamped right after the last vehicle.control
        hold_s = t_hold - t_last
        dis_s = t_dis - t_last
        print(
            f"test_tech_supervisor_deadman: brake hold {hold_s:.3f} s (CMD_STALE_S {stale_s:g}), "
            f"disengage {dis_s:.3f} s (CMD_DEAD_S {dead_s:g}) after the last vehicle.control"
        )
        assert stale_s - poll_s <= hold_s <= stale_s + 2 * poll_s, hold_s
        assert dead_s - poll_s <= dis_s <= dead_s + 2 * poll_s, dis_s
        assert throttle_while_held == [], throttle_while_held
        engage = json.loads((bus / "gvd_engage.json").read_text(encoding="utf-8"))
        assert engage["engaged"] is False and engage["disengage_reason"] == "command_stream_dead", engage
    finally:
        if sup is not None:
            sup.kill()
            sup.wait()
        if sim is not None:
            sim.close()
        shutil.rmtree(root, ignore_errors=True)


def main() -> None:
    if "--supervisor" in sys.argv:
        sys.exit(_supervisor())
    check_stamp_follows_control_on_same_thread()
    check_frozen_tech_loop_brakes_then_disengages()
    print("test_tech_supervisor_deadman: OK")


def test_tech_supervisor_deadman() -> None:
    main()


if __name__ == "__main__":
    main()
