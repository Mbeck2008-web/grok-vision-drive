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
same folder. The harness owns a sim clock: the mod gets ``onUpdate`` at a
fixed frame dt (60, 20, 300, 2400 fps) or a random dt in 2..60 ms (seeds 101..120), the child ticks every ``TICK_S`` of sim
time in lockstep, and every time below is sim time. A child write lands on a
frame boundary right after that frame's ``onUpdate``, so the mod sees it one
frame later (the worst phase).

Limits are declared up front from main.lua's own constants:
fixed rate: [thr, thr + CMD_POLL_S + dt); random dt: [thr, thr + 2 * (CMD_POLL_S + dt_max))
(see ``limits``), thr = CMD_STALE_S for the hold and CMD_DEAD_S for the disengage. Random-dt
runs are also checked against an independent model of the poll design (``model_hold_dis``).

Tech (``BeamNGPyActuator``) is in ``test_tech_supervisor_deadman.py`` on the
same harness.
"""
from __future__ import annotations

import json
import math
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gvd_test_env import isolated_env  # noqa: E402

MAIN_LUA = ROOT / "beamng_mod" / "lua" / "ge" / "extensions" / "gvd" / "main.lua"
SIM_LUA = ROOT / "scripts" / "gvd_lua_sim.lua"
FREEZE_AT = 40  # supervisor ticks of live driving before the freeze
TICK_S = 0.05  # supervisor tick, sim time
# Frame rates: a fixed fps, or ("rand", seed) = dt uniform in RAND_DT_S from that seed.
RAND_DT_S = (0.002, 0.060)
RAND_SEEDS = tuple(range(101, 121))  # declared before the first run; every seed must pass
FPS = (60, 20, 300, 2400) + tuple(("rand", seed) for seed in RAND_SEEDS)
STEER, THROTTLE = 0.1, 0.3
# Test-only knob: a real sleep after every sim frame, to play a slow runner (wall time runs
# ahead of sim time). GVD_TEST_STEP_SLEEP_S sets it for every rate; SLOW_RUNS always run with it.
STEP_SLEEP_S = float(os.environ.get("GVD_TEST_STEP_SLEEP_S") or 0.0)
# 2400 fps with 1.5 ms per frame: about 3 s of wall per 1 s of sim. Before the mod's wall clock
# followed the sim, this disengaged before the 0.35 s hold (CI saw 0.55 s and 0.90 s).
SLOW_RUNS = ((2400, 0.0015),)


def _lua_const(name: str) -> float:
    m = re.search(rf"^local {name} = ([0-9.]+)", MAIN_LUA.read_text(encoding="utf-8"), re.M)
    assert m, name
    return float(m.group(1))


def limits(rate, dt_max: float) -> dict[str, float]:
    """Declared before any run: [lower, upper) from main.lua's constants and the largest frame dt sent.

    Fixed rate (dt_max = that rate's dt): every poll spans exactly CMD_POLL_S, so the only slack
    is up to one poll before the mod sees that writes stopped, plus one frame:
    thr + CMD_POLL_S + dt.
    Random dt: a poll fires on the first frame that brings the accumulated time to CMD_POLL_S,
    so one poll spans up to CMD_POLL_S + dt_max. The slack is up to one poll span before the
    mod sees the writes stopped (its counter starts at that poll), plus up to one poll span of
    overshoot past the threshold (the compare runs only on polls): thr + 2 * (CMD_POLL_S + dt_max).
    """
    stale, dead, poll = _lua_const("CMD_STALE_S"), _lua_const("CMD_DEAD_S"), _lua_const("CMD_POLL_S")
    slack = 2.0 * (poll + dt_max) if isinstance(rate, tuple) else poll + dt_max
    return {"hold_lo": stale, "hold_hi": stale + slack, "dis_lo": dead, "dis_hi": dead + slack}


def frame_dts(rate):
    """Endless frame dts for a rate: 1/fps, or ("rand", seed) uniform in RAND_DT_S."""
    if isinstance(rate, tuple):
        rng = random.Random(rate[1])
        while True:
            yield rng.uniform(*RAND_DT_S)
    dt = 1.0 / rate
    while True:
        yield dt


def _wait_bus() -> bool:
    from python.runtime.paths import bus_identity

    deadline = time.monotonic() + 20.0
    while not bus_identity().matched:
        if time.monotonic() > deadline:
            print("NO_BUS", flush=True)
            return False
        time.sleep(0.01)
    print("READY", flush=True)
    return True


def _supervisor() -> int:
    """Child: one grab -> state -> command tick per stdin line; ``freeze`` = a read that never returns."""
    from python.control.actuate import CmdJsonActuator, DriveCommand
    from python.runtime.state_io import write_state

    class FrozenCam:
        frozen = False

        def stream_raw(self):
            if FrozenCam.frozen:
                threading.Event().wait()  # never returns; main thread
            return {"colour": None}

    if not _wait_bus():
        return 3
    cam = FrozenCam()
    act = CmdJsonActuator()
    act.note_engaged(True)
    seq = 0
    for line in sys.stdin:
        if line.strip() == "freeze":
            print(f"FREEZE {seq} threads={threading.active_count()}", flush=True)
            FrozenCam.frozen = True
        cam.stream_raw()
        seq += 1
        write_state({"engaged": True, "policy": "modular", "cmd_seq": seq, "disengage_reason": "none"})
        act.apply(DriveCommand(steer=STEER, throttle=THROTTLE, brake=0.0, seq=seq, reason="ok"))
        print(f"TICK {seq}", flush=True)
    return 0


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


class Lockstep:
    """Real main.lua at a fixed frame dt + a supervisor child ticking in the same sim time."""

    def __init__(self, script: Path, rate, step_sleep_s: float | None = None) -> None:
        self.dts = frame_dts(rate)
        self.step_sleep_s = STEP_SLEEP_S if step_sleep_s is None else float(step_sleep_s)
        self.warm_dt = 0.05  # one mod poll per warm-up frame: every run starts at the same poll phase
        lua = shutil.which("lua5.1") or shutil.which("lua")
        assert lua, "lua5.1 is not on PATH (the mod harness needs it; not a skip)"
        self.root = Path(tempfile.mkdtemp(prefix="gvd-deadman-"))
        self.sim = self.sup = None
        env = isolated_env(os.environ, self.root)
        env["GVD_PRODUCT"] = "drive"
        user = Path(env["LOCALAPPDATA"]) / "BeamNG" / "BeamNG.drive" / "current"
        self.bus = user / "Documents" / "GVD"
        self.bus.mkdir(parents=True, exist_ok=True)
        self.sim = _Sim(lua, {**env, "GVD_SIM_USER": str(user)})
        _, engaged = self.sim.send("engage")
        assert engaged, "Alt+G in the sim did not engage"
        self.sup = subprocess.Popen(
            [sys.executable, str(script), "--supervisor"],
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.lines: queue.Queue[str] = queue.Queue()
        self.out: list[str] = []
        threading.Thread(target=self._pump, daemon=True).start()
        # Warm-up (not timed): frames until the child sees the mod's lua_bus.
        deadline = time.monotonic() + 20.0
        while True:
            self.sim.send(f"step {self.warm_dt!r}")
            if any(x.startswith(("READY", "NO_BUS", "Traceback")) for x in self.out):
                self._reply("READY")
                break
            assert time.monotonic() < deadline, f"supervisor never got READY: {self.out}"
            time.sleep(0.005)

    def _pump(self) -> None:
        for line in self.sup.stdout:
            self.out.append(line.rstrip())
            self.lines.put(line.rstrip())

    def _reply(self, want: str) -> str:
        while True:  # the writers' own log lines ride on the same stdout
            line = self.lines.get(timeout=20.0)
            if line.split(" ", 1)[0] in ("READY", "TICK", "FREEZE", "NO_BUS") or line.startswith("Traceback"):
                break
        assert line.startswith(want), (want, line, self.out[-20:])
        return line

    def run(self, plan: list[str], tail_s: float) -> dict:
        """Send ``plan`` one command per supervisor tick; after ``freeze``, run ``tail_s`` more sim time.

        Returns per-frame (sim_t, events, engaged) and the sim time of the last child write.
        A write at sim time t (between frames) is seen by the onUpdate that ends at t + dt.

        The mod's wall clock runs on sim time too: each frame carries ``wall = hb + (t_end - t_w)``,
        hb = the ``heartbeat_mtime`` of the last child write and t_w its sim time. The mod reads it
        in whole seconds (os.time), so a slow host (real time ahead of sim time) changes nothing.
        """
        frames: list[tuple[float, list[tuple[str, float]], bool]] = []
        res: dict = {"frames": frames, "t_last": None, "t_freeze": None, "marks": {}, "freeze_line": None, "dt_max": 0.0,
                     "dts": [], "hb_last": None}
        cmds = iter(plan)
        t = Fraction(0)  # exact sum of the float dts handed to the mod
        next_tick = Fraction(0)
        t_end = None
        t_w = None
        while t_end is None or t < t_end:
            if t >= next_tick - Fraction(1, 10**9):
                next_tick += Fraction(TICK_S)
                c = next(cmds, None)
                if c is not None:
                    self.sup.stdin.write(c + "\n")
                    self.sup.stdin.flush()
                    if c == "freeze":
                        res["freeze_line"] = self._reply("FREEZE")
                        res["t_freeze"] = float(t)
                        t_end = t + Fraction(tail_s)
                    else:
                        self._reply("TICK")
                        res["t_last"] = float(t)
                        t_w = t
                        res["hb_last"] = float(json.loads((self.bus / "gvd_cmd.json").read_text(encoding="utf-8"))["heartbeat_mtime"])
                        res["k_last"] = len(res["dts"])
                        res["marks"].setdefault(c, float(t))
            dt = next(self.dts)
            res["dt_max"] = max(res["dt_max"], dt)
            res["dts"].append(dt)
            if t_w is None:
                events, engaged = self.sim.send(f"step {dt!r}")
            else:
                wall = sim_wall(res["hb_last"], t + Fraction(dt) - t_w)
                events, engaged = self.sim.send(f"step {dt!r} {wall!r}")
            if self.step_sleep_s > 0.0:
                time.sleep(self.step_sleep_s)
            t += Fraction(dt)
            frames.append((float(t), events, engaged))
        assert self.sup.poll() is None, "supervisor exited; it should be stuck"
        return res

    def close(self) -> None:
        if self.sup is not None:
            self.sup.kill()
            self.sup.wait()
        if self.sim is not None:
            self.sim.close()
        shutil.rmtree(self.root, ignore_errors=True)


def sim_wall(hb: float, since_write: Fraction) -> float:
    """Sim wall time (Unix s) ``since_write`` of sim time after the write stamped ``hb``."""
    return hb + float(since_write)


def hold_and_disengage(res: dict) -> tuple[float | None, float | None, list, list]:
    """Sim time of the first brake hold and the disengage after the freeze, plus what came between."""
    t_hold = t_dis = None
    before_hold: list = []
    throttle_while_held: list = []
    for t, events, engaged in res["frames"]:
        if t <= res["t_freeze"]:
            continue
        got = dict(events)
        if t_hold is None and got.get("throttle") == 0.0 and got.get("steering") == 0.0 and got.get("parkingbrake") == 1.0:
            t_hold = t
        elif t_hold is None and got:
            before_hold.append((t, events))
        elif t_hold is not None and t_dis is None and got.get("throttle", 0.0) > 0.0:
            throttle_while_held.append(got["throttle"])
        if not engaged and t_dis is None:
            t_dis = t
            break
    return t_hold, t_dis, before_hold, throttle_while_held


def model_hold_dis(dts: list[float], k_last: int, hb: float | None = None) -> tuple[float, float]:
    """Independent model of the DESIGN (not a copy of main.lua): regression detector, not proof.

    The mod polls whenever its accumulated frame time reaches CMD_POLL_S (the run starts on a
    poll boundary: warm-up frames are each one poll). A write lands on the boundary before frame
    ``k_last``, so the first poll at or after that frame's end sees it and starts the clock. Each
    later poll compares the sim time elapsed since that poll, in whole microseconds, against
    CMD_STALE_S then CMD_DEAD_S. Times are exact rationals of the dts sent. Returns (hold, dead)
    in seconds after the write.
    With ``hb`` (the write's heartbeat_mtime), each poll also applies the leftover-file rule:
    dead once whole sim-wall seconds minus hb exceed CMD_DEAD_S (``sim_wall``, floored like os.time).
    """
    stale, dead, poll = (Fraction(_lua_const(n)) for n in ("CMD_STALE_S", "CMD_DEAD_S", "CMD_POLL_S"))
    t = Fraction(0)
    since_poll = Fraction(0)
    t_write = sum((Fraction(d) for d in dts[:k_last]), Fraction(0))
    seen = hold = None
    for k, d in enumerate(dts):
        t += Fraction(d)
        since_poll += Fraction(d)
        if since_poll < poll:
            continue
        since_poll = Fraction(0)
        if k < k_last:
            continue
        if seen is None:
            seen = t
            continue
        elapsed_us = math.floor((t - seen) * 10**6 + Fraction(1, 2))
        ancient = hb is not None and (math.floor(sim_wall(hb, t - t_write)) - hb) > float(dead)
        if elapsed_us >= dead * 10**6 or ancient:
            assert hold is not None, "model: dead before the hold"
            return float(hold - t_write), float(t - t_write)
        if hold is None and elapsed_us >= stale * 10**6:
            hold = t
    raise AssertionError("model: no disengage within the dts sent")


def check_model(name: str, rate, res: dict, hold_s: float, dis_s: float) -> None:
    if not isinstance(rate, tuple):
        return  # exact-boundary fixed rates depend on float rounding of the poll gate; not modelled
    m_hold, m_dis = model_hold_dis(res["dts"], res["k_last"], res.get("hb_last"))
    print(f"{name} {rate}: model hold {m_hold:.6f} dis {m_dis:.6f}; measured {hold_s:.6f} {dis_s:.6f}", flush=True)
    assert abs(m_hold - hold_s) <= 1e-6 and abs(m_dis - dis_s) <= 1e-6, (rate, m_hold, hold_s, m_dis, dis_s)


def check_bounds(name: str, fps, hold_s: float, dis_s: float, dt_max: float) -> None:
    lim = limits(fps, dt_max)
    print(
        f"{name} {fps} fps (dt_max {dt_max * 1000:.3f} ms): brake hold {hold_s:.4f} s in [{lim['hold_lo']:.4f}, {lim['hold_hi']:.4f}), "
        f"disengage {dis_s:.4f} s in [{lim['dis_lo']:.4f}, {lim['dis_hi']:.4f}) (sim time after the last write)",
        flush=True,
    )
    assert lim["hold_lo"] <= hold_s < lim["hold_hi"], (fps, "hold", hold_s, lim)
    assert lim["dis_lo"] <= dis_s < lim["dis_hi"], (fps, "disengage", dis_s, lim)


def check_frozen_read_brakes_then_disengages(fps, step_sleep_s: float | None = None) -> None:
    ls = Lockstep(Path(__file__).resolve(), fps, step_sleep_s)
    try:
        res = ls.run(["tick"] * FREEZE_AT + ["freeze"], tail_s=2.0)
        drove = any(
            abs(dict(ev).get("throttle", -1.0) - THROTTLE) < 1e-3 and abs(dict(ev).get("steering", -9.0) - STEER) < 1e-3
            for t, ev, _ in res["frames"] if t <= res["t_freeze"]
        )
        assert drove, "the mod never applied the live command before the freeze"
        cmd = json.loads((ls.bus / "gvd_cmd.json").read_text(encoding="utf-8"))
        assert cmd["seq"] == FREEZE_AT and cmd["engaged"] is True, cmd
        t_hold, t_dis, _, throttle_while_held = hold_and_disengage(res)
        assert t_hold is not None, "no brake hold after the supervisor froze"
        assert t_dis is not None, "no disengage after the supervisor froze"
        assert throttle_while_held == [], throttle_while_held
        engage = json.loads((ls.bus / "gvd_engage.json").read_text(encoding="utf-8"))
        assert engage["engaged"] is False and engage["disengage_reason"] == "command_stream_dead", engage
        check_bounds("test_frozen_supervisor_deadman", fps, t_hold - res["t_last"], t_dis - res["t_last"], res["dt_max"])
        check_model("test_frozen_supervisor_deadman", fps, res, t_hold - res["t_last"], t_dis - res["t_last"])
    finally:
        ls.close()


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
    for fps in FPS:
        check_frozen_read_brakes_then_disengages(fps)
    for fps, sleep_s in SLOW_RUNS:
        print(f"test_frozen_supervisor_deadman slow runner: {fps} fps, {sleep_s * 1000:.1f} ms real sleep per frame", flush=True)
        check_frozen_read_brakes_then_disengages(fps, sleep_s)
    print("test_frozen_supervisor_deadman: OK")


def test_frozen_supervisor_deadman() -> None:
    main()


if __name__ == "__main__":
    main()
