#!/usr/bin/env python3
"""A supervisor frozen inside a camera read: the mod's dead-man stops the car (no BeamNG).

Retail drive bus. The supervisor writes ``gvd_cmd.json`` (``CmdJsonActuator``)
from the same loop, on the same thread, as ``backend.grab()``; a read that
never returns stops the ``seq``. ``gvd_main.applyCmdJson`` holds the brake
after ``CMD_STALE_S`` without a new seq and releases + disengages after
``CMD_DEAD_S``.

A child process runs that loop with the real writers (``write_state``,
``CmdJsonActuator.apply``) and then blocks forever in ``stream_raw`` on its
main thread. ``scripts/gvd_lua_sim.lua`` runs the real ``main.lua`` on the
same folder in wall time. The mod's own constants are the bounds, plus two
of its 20 Hz polls.

Tech (``BeamNGPyActuator``, ``vehicle.control``) does not go through this
file and is not covered here.
"""
from __future__ import annotations

import json
import os
import re
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

MAIN_LUA = ROOT / "beamng_mod" / "lua" / "ge" / "extensions" / "gvd" / "main.lua"
SIM_LUA = ROOT / "scripts" / "gvd_lua_sim.lua"
FREEZE_AT = 40  # ticks of live driving before the read that never returns
TICK_S = 0.05
STEER, THROTTLE = 0.1, 0.3


def _lua_const(name: str) -> float:
    m = re.search(rf"^local {name} = ([0-9.]+)", MAIN_LUA.read_text(encoding="utf-8"), re.M)
    assert m, name
    return float(m.group(1))


def _supervisor() -> int:
    """Child: the grab -> state -> command loop, then a read that never returns."""
    from python.control.actuate import CmdJsonActuator, DriveCommand
    from python.runtime.paths import bus_identity
    from python.runtime.state_io import write_state

    class FrozenCam:
        n = 0

        def stream_raw(self):
            FrozenCam.n += 1
            if FrozenCam.n > FREEZE_AT:
                threading.Event().wait()  # never returns; main thread
            return {"colour": None}

    deadline = time.monotonic() + 10.0
    while not bus_identity().matched:
        if time.monotonic() > deadline:
            print("NO_BUS", flush=True)
            return 3
        time.sleep(0.02)
    cam = FrozenCam()
    act = CmdJsonActuator()
    act.note_engaged(True)
    seq = 0
    while True:
        t0 = time.monotonic()
        if FrozenCam.n == FREEZE_AT:
            print(f"FREEZE {seq}", flush=True)
        cam.stream_raw()
        seq += 1
        write_state({"engaged": True, "policy": "modular", "cmd_seq": seq, "disengage_reason": "none"})
        act.apply(DriveCommand(steer=STEER, throttle=THROTTLE, brake=0.0, seq=seq, reason="ok"))
        time.sleep(max(0.0, TICK_S - (time.monotonic() - t0)))


class _Sim:
    def __init__(self, lua: str, env: dict[str, str]) -> None:
        self.p = subprocess.Popen(
            [lua, str(SIM_LUA)],
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.read()

    def read(self) -> tuple[list[tuple[str, float]], bool]:
        events: list[tuple[str, float]] = []
        engaged = False
        while True:
            line = self.p.stdout.readline()
            assert line, "gvd_lua_sim.lua exited"
            parts = line.split()
            if parts[0] == "ok":
                return events, engaged
            if parts[0] == "ev":
                events.append((parts[1], float(parts[2])))
            elif parts[0] == "st":
                engaged = parts[1] == "1"

    def send(self, cmd: str) -> tuple[list[tuple[str, float]], bool]:
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()
        return self.read()

    def close(self) -> None:
        try:
            self.p.stdin.write("quit\n")
            self.p.stdin.flush()
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def check_frozen_read_brakes_then_disengages() -> None:
    stale_s = _lua_const("CMD_STALE_S")
    dead_s = _lua_const("CMD_DEAD_S")
    poll_s = _lua_const("CMD_POLL_S")
    lua = shutil.which("lua5.1") or shutil.which("lua")
    assert lua, "lua5.1 is not on PATH (the mod harness needs it; not a skip)"
    root = Path(tempfile.mkdtemp(prefix="gvd-deadman-"))
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

        drove = False
        frozen_seq: int | None = None
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
            if frozen_seq is None:
                for line in list(out):
                    if line.startswith("FREEZE"):
                        frozen_seq = int(line.split()[1])
                    assert not line.startswith("NO_BUS"), "supervisor never saw a live lua_bus"
            got = dict(events)
            if frozen_seq is None:
                if abs(got.get("throttle", -1.0) - THROTTLE) < 1e-3 and abs(got.get("steering", -9.0) - STEER) < 1e-3:
                    drove = True
                continue
            if t_hold is None and got.get("throttle") == 0.0 and got.get("steering") == 0.0 and got.get("parkingbrake") == 1.0:
                t_hold = stamp
            elif t_hold is not None and t_dis is None and got.get("throttle", 0.0) > 0.0:
                throttle_while_held.append(got["throttle"])
            if not engaged and t_dis is None:
                t_dis = stamp
                break
        assert drove, "the mod never applied the live command before the freeze"
        assert frozen_seq is not None, out
        assert sup.poll() is None, "supervisor exited; it should be stuck in the read"
        cmd = json.loads((bus / "gvd_cmd.json").read_text(encoding="utf-8"))
        assert cmd["seq"] == frozen_seq and cmd["engaged"] is True, cmd
        t_last = float(cmd["heartbeat_mtime"])  # the last command before the read froze
        assert t_hold is not None, "no brake hold after the supervisor froze"
        assert t_dis is not None, "no disengage after the supervisor froze"
        hold_s = t_hold - t_last
        dis_s = t_dis - t_last
        print(
            f"test_frozen_supervisor_deadman: brake hold {hold_s:.3f} s (CMD_STALE_S {stale_s:g}), "
            f"disengage {dis_s:.3f} s (CMD_DEAD_S {dead_s:g}) after the last command"
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


def check_supervisor_writes_cmd_on_grab_thread() -> None:
    """The command writer is on the grab's loop, so a stuck read stops the seq."""
    src = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    loop_at = src.find("        while True:")
    grab_at = src.find("bundle = backend.grab()")
    apply_at = src.find("applied = actuator.apply(cmd)")
    assert 0 <= loop_at < grab_at < apply_at, (loop_at, grab_at, apply_at)
    assert "threading" not in src
    act = (ROOT / "python" / "control" / "actuate.py").read_text(encoding="utf-8")
    assert "threading" not in act


def main() -> None:
    if "--supervisor" in sys.argv:
        sys.exit(_supervisor())
    check_supervisor_writes_cmd_on_grab_thread()
    check_frozen_read_brakes_then_disengages()
    print("test_frozen_supervisor_deadman: OK")


def test_frozen_supervisor_deadman() -> None:
    main()


if __name__ == "__main__":
    main()
