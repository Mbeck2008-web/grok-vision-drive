#!/usr/bin/env python3
"""The echo gate's gap-run rule on the supervisor, with the skips injected by the test.

python/control/echo_link.py: GAP_RUN_MAX advancing reads in a row that each skipped at least
one echo_seq make the read stale, so the supervisor disengages with echo_stale while the echo
is still moving (since_ms 0). This runs run_vision.py headless (stub backend, retail cmd_json
bus) and drives that rule from the fake mod, the same way on a fast or a slow runner:

* The fake mod runs in lockstep (test_unknown_ego._Mod.lockstep_hold): each echo waits until
  gvd_state.ego_echo.seq shows the supervisor read the one before. So the supervisor sees
  every echo the test writes, and only the skips the test writes.
* The supervisor runs through scripts/gvd_echo_clock_child.py, so the echo gate's clock is
  the fake mod's (one echo period per echo, two for an echo that skips one). The 300 ms timer
  cannot fire on a slow runner, so an echo_stale here can only be the gap run.

1. Control: GAP_RUN_MAX - 1 skips, a clean echo, GAP_RUN_MAX - 1 skips, clean echoes. The
   supervisor reads each skip (skipped 1) and stays engaged: the run must be consecutive.
2. GAP_RUN_MAX skips in a row: the first GAP_RUN_MAX - 1 read fresh and engaged; the read of
   the last one is the disengage, echo_stale on gvd_engage / gvd_state / gvd_cmd (thr 0 /
   brk 0). On that tick an echo gate (gvd_state.ego_echo, the session echo that is read first,
   or gvd_state.echo) reads status stale at that seq with skipped >= 1 and since_ms 0.
"""
from __future__ import annotations

import contextlib
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import test_unknown_ego as T  # noqa: E402
from gvd_test_env import isolated_home  # noqa: E402
from python.control.actuate import cmd_path, write_engage_flag  # noqa: E402
from python.control.echo_link import GAP_RUN_MAX  # noqa: E402
from python.runtime.state_io import atomic_write_json, state_path  # noqa: E402

LAUNCHER = ROOT / "scripts" / "gvd_echo_clock_child.py"
ENGAGE_AT_S = 0.5
SETTLE_S = 1.0  # engaged this long on clean echoes before the control run starts
CONTROL = [1] * (GAP_RUN_MAX - 1) + [0] + [1] * (GAP_RUN_MAX - 1) + [0] * 4
HANG_GUARD_S = 30.0


class _SkipMod(T._Mod):
    """The fake mod in lockstep, on its own echo clock; `plan` lists the skips for the next echoes."""

    def __init__(self, clock: Path) -> None:
        super().__init__(lockstep=True)
        self.echoing = True
        self.clock = clock
        self.mod_t = 0.0
        atomic_write_json(self.clock, {"t": self.mod_t}, indent=None)
        self.plan: list[int] = []  # echo_seq values skipped before each next echo (0 = clean)
        self.written: list[tuple[int, int]] = []  # (echo_seq, skipped) of every echo written

    def echo(self, t_spawn: float) -> None:
        m6 = T._m6()
        now = time.monotonic()
        if now < self.next_echo or self.lockstep_hold():
            return
        self.next_echo = now + m6.FAKE_ECHO_PERIOD_S
        gap = self.plan.pop(0) if self.plan else 0
        # Clock before echo: the gate never reads this echo against an older mod time.
        self.mod_t += m6.FAKE_ECHO_PERIOD_S * (1 + gap)
        atomic_write_json(self.clock, {"t": self.mod_t}, indent=None)
        cmd = T._read(cmd_path())
        seq = int(cmd.get("seq", 0) or 0) if float(cmd.get("heartbeat_mtime") or 0.0) >= t_spawn - 0.5 else 0
        self.k += 1 + gap
        m6._write_ego(
            seq, applying=bool(seq), speed=self.speed, steer_in=float(cmd.get("steer", 0.0) or 0.0),
            throttle_in=float(cmd.get("throttle", 0.0) or 0.0), brake_in=float(cmd.get("brake", 0.0) or 0.0),
            echo_seq=self.k,
        )
        self.written.append((self.k, gap))


@contextlib.contextmanager
def _clock_launcher(clock: Path):
    """T._session spawns run_vision.py; run it through the echo-clock launcher instead."""
    real = subprocess.Popen

    def popen(args, *a, **k):
        if len(args) > 2 and str(args[2]).endswith("run_vision.py"):
            args = list(args)
            args[2] = str(LAUNCHER)
            k["env"] = dict(k.get("env") or {}, GVD_TEST_ECHO_CLOCK_FILE=str(clock))
        return real(args, *a, **k)

    subprocess.Popen = popen
    try:
        yield
    finally:
        subprocess.Popen = real


def _echo(snap: dict) -> dict:
    e = snap["state"].get("echo")
    return e if isinstance(e, dict) else {}


def _run() -> tuple[list[dict], dict, _SkipMod]:
    mark: dict = {}
    with tempfile.TemporaryDirectory(prefix="gvd-gaprun-clock-") as tmp, \
            isolated_home(prefix="gvd-gaprun-"), _clock_launcher(Path(tmp) / "echo_clock.json"):
        write_engage_flag(False)
        mod = _SkipMod(Path(tmp) / "echo_clock.json")

        def caught_up(st: dict) -> bool:
            tele = st.get("ego_echo")
            return isinstance(tele, dict) and tele.get("seq") == mod.k

        def step(t: float) -> None:
            if "on" not in mark:
                if t >= ENGAGE_AT_S:
                    mark["on"] = t
                    write_engage_flag(True)
                return
            st = T._read(state_path())
            if st.get("engaged") is not True:
                return
            mark.setdefault("live", t)
            if "control" not in mark:
                if t >= mark["live"] + SETTLE_S and caught_up(st):
                    mark["control"] = t
                    mark["control_k"] = mod.k
                    mod.plan = list(CONTROL)
            elif "inject" not in mark and not mod.plan and caught_up(st):
                mark["inject"] = t
                mark["inject_k"] = mod.k
                mod.plan = [1] * GAP_RUN_MAX

        def until(snap: dict) -> bool:
            if "inject" not in mark:
                return False
            if snap["engage"].get("engaged") is False:
                mark.setdefault("off", snap["t"])
                return snap["t"] >= mark["off"] + 0.5  # a few more ticks: the refusal on every surface
            if not mod.plan and caught_up(snap["state"]):
                mark.setdefault("read_all", snap["t"])
            # Every injected echo was read and the supervisor is still engaged: give it 1 s more.
            return "read_all" in mark and snap["t"] >= mark["read_all"] + 1.0

        seen = T._session(HANG_GUARD_S, mod, step, until=until)
    return seen, mark, mod


def check_gap_run_disengages_supervisor() -> None:
    assert GAP_RUN_MAX == 3, GAP_RUN_MAX
    seen, mark, mod = _run()
    assert "control" in mark, ("never engaged on clean echoes", sorted({str(s["state"].get("disengage_reason"))
                                                                        for s in seen}))
    k0, k1 = mark["control_k"], mark["inject_k"]
    by_seq: dict[int, list[dict]] = {}
    for s in seen:
        seq = _echo(s).get("seq")
        if isinstance(seq, int):
            by_seq.setdefault(seq, []).append(s)

    # 1. Control: every skip was read as one (skipped 1, fresh) and the supervisor stayed engaged.
    control = [(k, gap) for k, gap in mod.written if k0 < k <= k1]
    assert [gap for _k, gap in control] == CONTROL, ("control echoes written", control)
    for k, gap in control:
        reads = [s for s in by_seq.get(k, []) if _echo(s).get("status") == "fresh"]
        assert reads, ("the supervisor never read control echo", k, gap)
        assert all(_echo(s).get("skipped") == gap for s in reads), ("control skipped", k, [_echo(s) for s in reads])
    window = [s for s in seen if mark["control"] <= s["t"] <= mark["inject"]]
    dropped = [(round(s["t"], 2), s["state"].get("disengage_reason"), _echo(s)) for s in window
               if s["state"].get("engaged") is not True]
    assert not dropped, ("disengaged during the control run (not GAP_RUN_MAX in a row)", dropped[:3])

    # 2. GAP_RUN_MAX in a row: the last one is the disengage, nothing earlier.
    run = [(k, gap) for k, gap in mod.written if k > k1][:GAP_RUN_MAX]
    assert [gap for _k, gap in run] == [1] * GAP_RUN_MAX, ("skip echoes written", run)
    for k, _gap in run[:-1]:
        reads = by_seq.get(k, [])
        ok = (any(_echo(s).get("status") == "fresh" and _echo(s).get("skipped") == 1 for s in reads)
              and all(s["state"].get("engaged") is True and _echo(s).get("status") in ("fresh", "repeat")
                      for s in reads))
        assert ok, ("before the last skip", k, [(_echo(s), s["state"].get("engaged")) for s in reads])
    last = run[-1][0]
    off = [s for s in seen if s["t"] >= mark["inject"] and s["state"].get("disengage_reason") == "echo_stale"]
    assert off, ("no echo_stale after GAP_RUN_MAX skips", sorted({str(s["state"].get("disengage_reason"))
                                                                  for s in seen if s["t"] >= mark["inject"]}))
    # Both gates are EchoGates: the session echo (ego speed, read first in the tick) and the
    # engaged link gate. Whichever reads the run first names echo_stale; the other may already
    # be reset to off. The read that did it carries the gap-run signature.
    gates = {name: off[0]["state"].get(name) for name in ("ego_echo", "echo")}
    print(f"  disengage tick (skip echoes {[k for k, _g in run]}): {gates}")
    hit = [name for name, tele in gates.items() if isinstance(tele, dict) and tele.get("status") == "stale"
           and tele.get("seq") == last and int(tele.get("skipped") or 0) >= 1 and tele.get("since_ms") == 0.0]
    assert hit, ("no gate read stale at the last skip with since_ms 0 and skipped >= 1", last, gates)
    T._refusal([s for s in seen if s["t"] >= mark["inject"]], "echo_stale", "gap run")


CHECKS = (check_gap_run_disengages_supervisor,)


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
        raise SystemExit("test_gap_run_disengage failed: " + ", ".join(failed))
    print("test_gap_run_disengage: OK")


if __name__ == "__main__":
    main()
