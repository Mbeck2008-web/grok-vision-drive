#!/usr/bin/env python3
"""Unknown ego speed is None, and refuses (SPEC_unknown_ego rev 4, tests 1-14).

Ego speed is known only from a fresh measurement taken in THIS supervisor session.
Otherwise it is None, and code that needs speed refuses: throttle 0, brake 0, a
named disengage reason (in gvd_state.disengage_reason), never a 0.0 / last-value
stand-in. A measured low speed (0.02 m/s, a stopped car) is still known.

Tests 1-6 and 11-13 call the speed functions directly (the stub view from
test_held_plan_pedals: two painted lines, a stopped vehicle track or a stop
sign in the path). Tests 7-10 and 14 run run_vision.py headless (stub cameras) on the
retail cmd_json bus with a fake mod, like test_m6_retail.

Red first: on the code before the fix these checks fail on the post-fix value
they assert. The new module python/control/ego_speed.py is imported inside the
checks that need it, so a missing module fails only those checks.
"""
from __future__ import annotations

import contextlib
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from gvd_test_env import isolated_home  # noqa: E402
from python.control.actuate import (  # noqa: E402
    cmd_path,
    ego_path,
    engage_path,
    write_engage_flag,
)
from python.perception import pipeline  # noqa: E402
from python.perception.pipeline import ModularPerception  # noqa: E402
from python.runtime.shadow import E2EPolicy, shadow_tick  # noqa: E402
from python.runtime.state_io import state_path  # noqa: E402

DT = 0.05
UNKNOWN = "ego_unknown"
SPEED_FIELDS = ("target_v", "ttc_lead", "aeb", "stop_reason", "pred_brake")


# ---- scene helpers (unit level) -----------------------------------------------


def _road() -> np.ndarray:
    """Two painted lines: a lane pair with a ~36 m path starting at y=0."""
    import cv2

    img = np.full((96, 160, 3), (40, 42, 44), dtype=np.uint8)
    cv2.line(img, (30, 90), (70, 30), (40, 210, 235), 3)
    cv2.line(img, (130, 90), (90, 30), (245, 245, 245), 3)
    return img


class _Tracker:
    """A stopped vehicle track in the path at y (or no track)."""

    def __init__(self, y: float | None) -> None:
        self.y = y

    def update(self, dets: Any, t: Any) -> list:
        return []

    def as_dicts(self) -> list[dict[str, Any]]:
        if self.y is None:
            return []
        return [{"id": 7, "class": "vehicle", "x": 0.0, "y": float(self.y), "yaw": 1.5707963,
                 "speed_mps": 0.0, "yaw_rate": 0.0, "hits": 5, "misses": 0, "age": 5, "conf": 0.9}]


@contextlib.contextmanager
def _stop_sign(y: float | None):
    """A held stop sign at (1.0, y) on every tick (None: the real sign memory)."""
    real = pipeline._hold_signs
    if y is not None:
        pipeline._hold_signs = lambda prior, fresh: [
            {"cls": "stop_sign", "x": 1.0, "y": float(y), "conf": 0.9, "state": None, "partial": False, "misses": 0}
        ]
    try:
        yield
    finally:
        pipeline._hold_signs = real


def _perc(car_y: float | None) -> ModularPerception:
    perc = ModularPerception(allow_synthetic=False, detector_id="empty")
    perc.tracker = _Tracker(car_y)
    return perc


def _shadow(pout: Any, ego: float | None, *, heartbeat_ok: bool = True) -> Any:
    return shadow_tick(
        policy="modular", engaged=True, heartbeat_ok=heartbeat_ok,
        path_debug_preview=bool(pout.path_debug_preview), allow_preview_drive=False,
        path_ego=pout.path_ego, planner=pout.planner, ego_speed_mps=ego,
        lane_conf=float(pout.lane_conf), path_conf=float(pout.path_conf), seq=1,
        e2e_policy=E2EPolicy(), main_bgr=None, wide_bgr=None, stitch_bgr=None, steer_deg=0.0,
        cfg=None, loop_hz=20.0, camera_hz=20.0, engage_age_s=0.0,
    )


def _call(what: str, fn: Callable[[], Any]) -> Any:
    """Run fn; a raise is a failed refusal, reported by name instead of crashing the check."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - the raise itself is the finding
        raise AssertionError(f"{what} raised {type(exc).__name__}: {exc} (expected an explicit refusal)") from exc


def _refused(tick: Any, what: str) -> None:
    """The unknown-ego response: throttle 0, brake 0, ego_unknown, disengage."""
    cmd = tick.applied
    got = {"throttle": cmd.throttle, "brake": cmd.brake, "reason": cmd.reason,
           "should_disengage": tick.should_disengage}
    want = {"throttle": 0.0, "brake": 0.0, "reason": UNKNOWN, "should_disengage": True}
    assert got == want, (what, got, want)


def _unknown_then_echo(
    *, car_y: float | None = None, sign_y: float | None = None, echo: float,
    want_brake: float | None = None, want_throttle: float | None = None, tol: float = 1e-6,
    want_target: float | None = None,
) -> None:
    """No echo yet (ego None): refuse with no fit or command. Then the echo arrives: fresh values."""
    frame = _road()
    with _stop_sign(sign_y):
        perc = _perc(car_y)
        pout = _call("ModularPerception.tick(ego_speed_mps=None)",
                     lambda: perc.tick(frame, stitch_bgr=frame, ego_speed_mps=None, dt_s=DT))
        unset = {k: pout.planner.get(k) for k in SPEED_FIELDS if pout.planner.get(k) is not None}
        assert not unset, ("no echo: speed fields must be None, not a 0 m/s fit", unset)
        _refused(_call("shadow_tick(ego_speed_mps=None)", lambda: _shadow(pout, None)), "no echo")
        # The echo arrives on the next tick, same camera bytes: a fresh fit at the echo speed,
        # not a held fit from the unknown tick.
        pout = perc.tick(frame, stitch_bgr=frame, ego_speed_mps=echo, dt_s=DT)
        cmd = _shadow(pout, echo).applied
    got = {"throttle": cmd.throttle, "brake": cmd.brake, "reason": cmd.reason,
           "lane_source": pout.planner.get("lane_source"), "target_v": pout.planner.get("target_v")}
    assert cmd.reason == "ok", ("echo", got)
    if want_brake is not None:
        assert abs(cmd.brake - want_brake) <= tol and cmd.throttle == 0.0, ("echo brake", got, want_brake)
    if want_throttle is not None:
        assert abs(cmd.throttle - want_throttle) <= tol and cmd.brake == 0.0, ("echo throttle", got, want_throttle)
    if want_target is not None:
        assert abs(float(pout.planner["target_v"]) - want_target) <= 0.01, ("echo target_v", got, want_target)


# ---- tests 1-6: the held 0 m/s engage fit cannot happen -------------------------


def check_01_car_20m_ego_15() -> None:
    """Car 20 m, ego 15: no echo refuses (today a 0 m/s fit, held thr 0.24); echo 15 brakes 1.0."""
    _unknown_then_echo(car_y=20.0, echo=15.0, want_brake=1.0)


def check_02_car_30m_ego_15() -> None:
    """Car 30 m, ego 15: no echo refuses; echo 15 gives brk 0.203 +- 0.01 (target 10.94)."""
    _unknown_then_echo(car_y=30.0, echo=15.0, want_brake=0.203, tol=0.01, want_target=10.94)


def check_03_car_15m_ego_15() -> None:
    """Car 15 m, ego 15: no echo refuses with brake 0 (today brk 1.0 at a 0 m/s fit); echo 15 brakes 1.0."""
    _unknown_then_echo(car_y=15.0, echo=15.0, want_brake=1.0)


def check_04_sign_12m_ego_12() -> None:
    """Stop sign 12 m, ego 12: no echo refuses (today held brk 0.261); echo 12 brakes 1.0."""
    _unknown_then_echo(sign_y=12.0, echo=12.0, want_brake=1.0)


def check_05_sign_20m_ego_12() -> None:
    """Stop sign 20 m, ego 12: no echo refuses; echo 12 gives brk 0.136."""
    _unknown_then_echo(sign_y=20.0, echo=12.0, want_brake=0.136, tol=0.001)


def check_06_clear_road_ego_15() -> None:
    """Clear road, ego 15: no echo refuses; echo 15 gives thr 0.24."""
    _unknown_then_echo(echo=15.0, want_throttle=0.24)


# ---- tests 7-10: the supervisor (run_vision, stub cameras, fake mod) ------------


def _m6():
    import test_m6_retail as m6  # noqa: PLC0415 - shared fake-mod helpers

    return m6


def _read(path: Path) -> dict:
    return _m6()._read_json(path)


class _Mod:
    """The fake gvd_main: 10 Hz echo with its own echo_seq, Alt+G engage heartbeat."""

    def __init__(self, *, speed: float = 5.5, echo_seq0: int = 0, mod_version: str | None = None) -> None:
        self.speed = speed
        self.mod_version = mod_version
        self.k = echo_seq0
        self.next_echo = 0.0
        self.next_hb = 0.0
        self.echoing = False
        self.frozen: dict | None = None  # rewrite this echo (fresh mtime, same echo_seq)

    def echo(self, t_spawn: float) -> None:
        m6 = _m6()
        now = time.monotonic()
        if now < self.next_echo:
            return
        self.next_echo = now + m6.FAKE_ECHO_PERIOD_S
        if self.frozen is not None:
            m6._write_ego(**self.frozen)
            return
        if not self.echoing:
            return
        cmd = _read(cmd_path())
        seq = int(cmd.get("seq", 0) or 0) if float(cmd.get("heartbeat_mtime") or 0.0) >= t_spawn - 0.5 else 0
        self.k += 1
        m6._write_ego(
            seq, applying=bool(seq), speed=self.speed, steer_in=float(cmd.get("steer", 0.0) or 0.0),
            throttle_in=float(cmd.get("throttle", 0.0) or 0.0), brake_in=float(cmd.get("brake", 0.0) or 0.0),
            echo_seq=self.k, mod_version=self.mod_version,
        )

    def freeze(self) -> None:
        m6 = _m6()
        cmd = _read(cmd_path())
        seq = int(cmd.get("seq", 0) or 0)
        self.frozen = dict(seq=seq, applying=bool(seq), speed=self.speed, steer_in=float(cmd.get("steer", 0.0) or 0.0),
                           throttle_in=float(cmd.get("throttle", 0.0) or 0.0),
                           brake_in=float(cmd.get("brake", 0.0) or 0.0), echo_seq=self.k)
        m6._write_ego(**self.frozen)

    def heartbeat(self) -> None:
        """Lua refreshes gvd_engage.json while engaged; it never turns it on by itself."""
        now = time.monotonic()
        if now < self.next_hb:
            return
        self.next_hb = now + 1.0
        eng = _read(engage_path())
        if eng.get("engaged") is True:
            eng["mtime"] = time.time()
            engage_path().write_text(json.dumps(eng), encoding="utf-8")


def _session(
    seconds: float,
    mod: _Mod,
    step: Callable[[float], None] | None = None,
    *,
    extra: tuple[str, ...] = ("--allow-preview-drive",),
    until: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """Run run_vision headless; `step(t)` runs every pass with t = seconds since its first tick.

    Returns every distinct (engage, cmd, state) read, tagged with t. Files from an earlier
    run are removed before spawn; gvd_engage.json is left as the caller set it.

    With `until`, the run ends on the first read where until(snap) is true (tick-driven), and
    `seconds` is only a hang guard: reaching it is a failure.
    """
    m6 = _m6()
    m6._touch_lua_bus()
    for stale in (cmd_path(), state_path()):
        with contextlib.suppress(FileNotFoundError):
            stale.unlink()
    args = [sys.executable, "-u", str(ROOT / "python" / "run_vision.py"), "--backend", "stub",
            "--hz", "20", "--encode", "cpu", *extra]
    t_spawn = time.time()
    m_spawn = time.monotonic()
    proc = subprocess.Popen(args, cwd=str(ROOT), env=dict(os.environ, PYTHONPATH=str(ROOT)),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    seen: list[dict] = []
    last_key = None
    try:
        while not (cmd_path().exists() and state_path().exists()):
            if proc.poll() is not None:
                raise AssertionError(f"run_vision exited during startup ({proc.returncode})")
            if time.monotonic() > m_spawn + m6.SUPERVISOR_STARTUP_S:
                raise AssertionError("run_vision wrote no gvd_cmd.json / gvd_state.json in startup")
            m6._touch_lua_bus()
            mod.heartbeat()
            mod.echo(t_spawn)
            time.sleep(0.01)
        m0 = time.monotonic()
        while True:
            t = time.monotonic() - m0
            if t >= seconds:
                if until is not None:
                    raise AssertionError(f"hang guard: the `until` condition did not hold within {seconds:.0f}s")
                break
            if proc.poll() is not None:
                raise AssertionError(f"run_vision exited early ({proc.returncode})")
            m6._touch_lua_bus()
            if step is not None:
                step(t)
            mod.heartbeat()
            mod.echo(t_spawn)
            snap = {"t": t, "engage": _read(engage_path()), "cmd": _read(cmd_path()), "state": _read(state_path())}
            key = json.dumps([snap["engage"], snap["cmd"], snap["state"].get("engaged"),
                              snap["state"].get("disengage_reason"), snap["state"].get("last_refusal_reason"),
                              snap["state"].get("cmd_seq")], sort_keys=True, default=str)
            if key != last_key:
                seen.append(snap)
                last_key = key
            if until is not None and until(snap):
                break
            time.sleep(0.01)
    finally:
        proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)
    return seen


def _engaged_snaps(seen: list[dict]) -> list[dict]:
    return [s for s in seen if s["cmd"].get("engaged") is True or s["state"].get("engaged") is True]


def _refusal(seen: list[dict], reason: str, what: str) -> None:
    """The refusal is named on all three surfaces and its command is thr 0 / brk 0."""
    eng = [s["engage"] for s in seen if s["engage"].get("engaged") is False
           and s["engage"].get("disengage_reason") == reason]
    st = [s for s in seen if s["state"].get("disengage_reason") == reason]
    cmds = [s["cmd"] for s in seen if s["cmd"].get("reason") == reason]
    reasons = sorted({str(s["state"].get("disengage_reason")) for s in seen})
    assert eng, (what, f"gvd_engage.json never named {reason}", reasons)
    assert st, (what, f"gvd_state.disengage_reason never named {reason}", reasons)
    assert cmds, (what, f"no gvd_cmd.json with reason {reason}", sorted({str(s['cmd'].get('reason')) for s in seen}))
    bad = [(c.get("throttle"), c.get("brake"), c.get("engaged")) for c in cmds
           if not (c.get("throttle") == 0.0 and c.get("brake") == 0.0 and c.get("engaged") is False)]
    assert not bad, (what, f"{reason} commands must be thr 0 / brk 0 / engaged false", bad[:5])


def _engage_now() -> None:
    write_engage_flag(True)


def check_07_engage_before_first_echo() -> None:
    """Alt+G with no echo yet: today the engage tick drives at the 0 m/s placeholder.

    Unit: the clear road at the placeholder gave thr 0.55; at None it refuses.
    Supervisor: never engaged, ego_unknown on every surface, thr 0 / brk 0.
    """
    problems: list[str] = []
    frame = _road()
    perc = _perc(None)
    try:
        pout = _call("ModularPerception.tick(ego_speed_mps=None)",
                     lambda: perc.tick(frame, stitch_bgr=frame, ego_speed_mps=None, dt_s=DT))
        _refused(_call("shadow_tick(ego_speed_mps=None)", lambda: _shadow(pout, None)), "engage tick, clear road")
    except AssertionError as exc:
        problems.append(f"unit: {exc}")

    write_engage_flag(False)
    with contextlib.suppress(FileNotFoundError):
        ego_path().unlink()
    mod = _Mod()  # never echoes
    fired = {"on": False}

    def step(t: float) -> None:
        if not fired["on"] and t >= 0.5:
            fired["on"] = True
            _engage_now()

    seen = _session(3.0, mod, step)
    try:
        live = _engaged_snaps(seen)
        assert not live, ("engaged with no echo this session",
                          [(round(s["t"], 2), s["cmd"].get("reason"), s["cmd"].get("throttle"), s["cmd"].get("brake"))
                           for s in live[:5]])
        _refusal(seen, UNKNOWN, "engage before the first echo")
    except AssertionError as exc:
        problems.append(f"supervisor: {exc}")
    assert not problems, " | ".join(problems)


def check_08_restart_with_latch_on() -> None:
    """Supervisor restart with the latch already ON: refused as latched_before_start.

    Main case: the echo on disk is stale (an old session's). Variant: the old session's
    echo is under 1 s old (the mod keeps rewriting it, same echo_seq). Then a new off->on
    edge after a seq advance engages.
    """
    m6 = _m6()
    problems: list[str] = []
    for variant in ("echo_stale", "old_echo_under_1s"):
        write_engage_flag(True)  # the latch an earlier session left ON
        mod = _Mod(echo_seq0=500)
        if variant == "echo_stale":
            m6._write_ego(0, applying=False, speed=15.0, echo_seq=500, age_s=5.0)
            os.utime(ego_path(), (time.time() - 5.0, time.time() - 5.0))
        else:
            mod.frozen = dict(seq=0, applying=False, speed=15.0, echo_seq=500)
            m6._write_ego(**mod.frozen)
        phase = {"edge": None}

        def step(t: float) -> None:
            if t >= 2.0 and phase["edge"] is None:
                # The mod comes back (seq advances), the player presses Alt+G again.
                mod.frozen = None
                mod.echoing = True
                write_engage_flag(False)
                phase["edge"] = t
            elif phase["edge"] is not None and t >= phase["edge"] + 0.6 and _read(engage_path()).get("engaged") is not True:
                if t < phase["edge"] + 0.7:
                    _engage_now()

        seen = _session(4.5, mod, step)
        before = [s for s in seen if s["t"] < 2.0]
        try:
            live = _engaged_snaps(before)
            assert not live, (variant, "a latch ON at start drove",
                              [(round(s["t"], 2), s["cmd"].get("reason"), s["cmd"].get("throttle"),
                                s["cmd"].get("brake")) for s in live[:5]])
            _refusal(before, "latched_before_start", variant)
        except AssertionError as exc:
            problems.append(str(exc))
        after = [s for s in seen if phase["edge"] is not None and s["t"] >= phase["edge"] + 0.6]
        if not any(s["cmd"].get("engaged") is True for s in after):
            problems.append(f"{variant}: a new off->on edge after a seq advance did not engage "
                            f"{sorted({str(s['state'].get('disengage_reason')) for s in after})}")
    assert not problems, " | ".join(problems)


def check_09_stale_echo_mid_run() -> None:
    """Echo freezes while engaged: unknown at 300 ms (monotonic), thr 0 / brk 0, echo_stale named."""
    write_engage_flag(False)
    mod = _Mod()
    mod.echoing = True
    mark: dict = {}

    def step(t: float) -> None:
        if "on" not in mark and t >= 0.5:
            mark["on"] = t
            _engage_now()
        if "frozen" not in mark and t >= 1.7:
            mod.freeze()
            mark["frozen"] = time.time()

    seen = _session(3.5, mod, step)
    assert _engaged_snaps(seen), "never engaged before the freeze"
    _refusal(seen, "echo_stale", "stale echo mid-run")
    drops = [float(s["engage"].get("mtime") or 0.0) for s in seen
             if s["engage"].get("disengage_reason") == "echo_stale"]
    after = min(drops) - mark["frozen"]
    print(f"  echo_stale {after * 1000.0:.0f} ms after the last advancing echo")
    assert 0.0 < after < 0.6, f"echo_stale {after:.2f}s after the echo froze (300 ms window + a few ticks)"


def check_10_waiting_while_latched() -> None:
    """Latched while the echo gate is WAITING (an echo on disk, never a fresh one): refuse, ego_unknown."""
    m6 = _m6()
    write_engage_flag(False)
    m6._write_ego(0, applying=False, speed=15.0, echo_seq=9, age_s=5.0)
    os.utime(ego_path(), (time.time() - 5.0, time.time() - 5.0))
    mod = _Mod()  # silent: the gate never sees a fresh echo
    fired = {"on": False}

    def step(t: float) -> None:
        if not fired["on"] and t >= 0.5:
            fired["on"] = True
            _engage_now()

    seen = _session(3.0, mod, step)
    live = _engaged_snaps(seen)
    assert not live, ("drove while the echo gate was WAITING", [(round(s["t"], 2), s["cmd"].get("reason"),
                                                                s["cmd"].get("throttle"), s["cmd"].get("brake"))
                                                               for s in live[:5]])
    _refusal(seen, UNKNOWN, "WAITING while latched")


# ---- test 11: Tech StaleEgo -----------------------------------------------------


def check_11_tech_stale_ego() -> None:
    """StaleEgo blocks drive -> heartbeat_stale: brake 0 (released), should disengage.

    Offline this checks the command only. The `heartbeat_stale` in gvd_state part needs
    BeamNG.tech: UNPROVEN offline, it is live check (d) in the spec.
    """
    from types import SimpleNamespace

    from python.sensors.live_hold import StaleEgo

    stale = StaleEgo()
    for _ in range(3):
        stale.observe(SimpleNamespace(note="sensors.poll coalesced"))
    assert stale.blocks_drive(True), "two coalesced polls in a row must block drive"
    frame = _road()
    pout = _perc(None).tick(frame, stitch_bgr=frame, ego_speed_mps=15.0, dt_s=DT)
    tick = _shadow(pout, 15.0, heartbeat_ok=not stale.blocks_drive(True))
    got = {"throttle": tick.applied.throttle, "brake": tick.applied.brake, "reason": tick.applied.reason,
           "should_disengage": tick.should_disengage}
    want = {"throttle": 0.0, "brake": 0.0, "reason": "heartbeat_stale", "should_disengage": True}
    assert got == want, ("Tech StaleEgo", got, want)


# ---- test 12: None into every speed function ------------------------------------


def _cmd_refused(cmd: Any) -> str | None:
    got = (cmd.throttle, cmd.brake, cmd.reason)
    return None if got == (0.0, 0.0, UNKNOWN) else f"cmd {got}, want (0.0, 0.0, {UNKNOWN!r})"


def _fields_none(obj: Any, names: tuple[str, ...]) -> str | None:
    get = obj.get if isinstance(obj, dict) else (lambda k: getattr(obj, k, None))
    left = {k: get(k) for k in names if get(k) is not None}
    return None if not left else f"fields not None: {left}"


def check_12_none_into_every_speed_function() -> None:
    """Every function in the spec list takes None: no raise, no default, None fields or a refusal."""
    from python.control import actuate, launch_guard
    from python.control.e2e import E2EPolicy as Policy
    from python.perception.cipv import select_cipv
    from python.planning import corridor, path_predictor as pp
    from python.planning.speed import plan_speed
    from python.runtime import shadow
    from python.sensors.live_hold import RespawnWatch

    frame = _road()
    car = _Tracker(20.0).as_dicts()
    seen = _perc(20.0)
    seen.tick(frame, stitch_bgr=frame, ego_speed_mps=15.0, dt_s=DT)
    lanes = list(seen._last_lanes.lanes_bev)
    plan0 = pp.predict_path(lanes_bev=lanes, lane_conf=0.9, tracks=car, ego_speed_mps=15.0)
    path = [dict(p) for p in plan0.path_ego]
    sign = [{"cls": "stop_sign", "x": 1.0, "y": 12.0, "conf": 0.9, "state": None, "partial": False, "misses": 0}]
    results: list[str] = []

    def case(name: str, fn: Callable[[], str | None]) -> None:
        try:
            why = fn()
        except Exception as exc:  # noqa: BLE001
            why = f"raised {type(exc).__name__}: {exc}"
        results.append(f"{name}: {'ok' if why is None else 'FAIL ' + why}")

    def _tick():
        out = _perc(20.0).tick(frame, stitch_bgr=frame, ego_speed_mps=None, dt_s=DT)
        return _fields_none(out.planner, SPEED_FIELDS)

    def _drive_plan():
        plan, _ = pipeline.drive_plan_from_lanes(lanes, lanes, lane_conf=0.9, curvature=0.0, tracks=car,
                                                 signs=sign, ego_speed_mps=None)
        return _fields_none(plan, SPEED_FIELDS)

    def _cipv():
        res = select_cipv(car, ego_speed_mps=None)
        return None if (res.ttc_lead, res.aeb) == (None, None) else f"ttc/aeb {(res.ttc_lead, res.aeb)}, want (None, None)"

    def _predict():
        return _fields_none(pp.predict_path(lanes_bev=lanes, lane_conf=0.9, tracks=car, ego_speed_mps=None), SPEED_FIELDS)

    def _follow():
        return _cmd_refused(pp.follow_path(plan0, ego_speed_mps=None))

    def _expose():
        from dataclasses import replace

        out = pp._expose_pedals(replace(plan0), None)
        return None if out.pred_brake is None else f"pred_brake {out.pred_brake}, want None"

    def _reckon():
        out = pp.reckon_path(path, speed_mps=None, yaw_rate=0.0, dt=DT)
        return None if out in (None, []) else f"moved {len(out)} points at an unknown speed (a 0 m/s reckon)"

    def _scene():
        out = pp._apply_scene(path, [], sign, 3.5, ego_speed_mps=None)
        return None if out[3] is None else f"ttc {out[3]} from an unknown speed"

    def _sign_ttc():
        out = pp._sign_ttc(12.0, None)
        return None if out is None else f"ttc {out}, want None"

    def _flags():
        aeb, target, ttc = pp._speed_flags(curv=0.0, stop_reason="vehicle", ttc=1.0, stop_dist=12.0, closing_mps=None,
                                           ego_speed_mps=None, signs=[], path=path)
        return None if (aeb, target) == (None, None) else f"aeb/target {(aeb, target)}, want (None, None)"

    def _corridor():
        sig = inspect.signature(corridor.build_path_ego)
        p = sig.parameters.get("ego_speed_mps")
        if p is None or p.default is not inspect.Parameter.empty:
            return "build_path_ego must take ego_speed_mps with no default (it calls predict_path)"
        out = corridor.build_path_ego(lanes_bev=lanes, lane_conf=0.9, curvature=0.0, cipv=car[0], ego_speed_mps=None)
        return _fields_none(out, SPEED_FIELDS)

    def _plan_speed():
        out = plan_speed(ego_speed_mps=None, curvature=0.0, ttc_lead=3.0, aeb="off")
        return None if out.target_v is None else f"target_v {out.target_v}, want None"

    def _plan_command():
        return _cmd_refused(actuate.plan_command(path_ego=path, planner=plan0.planner_dict(), ego_speed_mps=None, seq=1))

    def _safe_command():
        return _cmd_refused(actuate.safe_command(engaged=True, heartbeat_ok=True, path_debug_preview=False,
                                                 allow_preview_drive=False, path_ego=path,
                                                 planner=plan0.planner_dict(), ego_speed_mps=None, seq=1))

    def _lookahead():
        out = actuate.steer_lookahead_m(None)
        return None if out is None else f"{out} m (the 0 m/s lookahead), want None"

    def _curve():
        out = actuate.curve_steer(0.5, 0.0, 0.01, None)
        return None if out is None else f"steer {out} (a default), want None"

    def _preprocess():
        feats = Policy().preprocess(frame, None, stitch_bgr=frame, speed_mps=None)
        return None if feats.get("kin") is None else f"kin {feats.get('kin')} from an unknown speed, want None"

    def _forward():
        intent = Policy().forward(frame, None, stitch_bgr=frame, speed_mps=None)
        return None if (intent.ok, intent.reason) == (False, UNKNOWN) else f"ok/reason {(intent.ok, intent.reason)}"

    def _modular():
        return _cmd_refused(shadow.modular_intent(path_ego=path, planner=plan0.planner_dict(), ego_speed_mps=None, seq=1))

    def _shadow_tick():
        pout = _perc(20.0).tick(frame, stitch_bgr=frame, ego_speed_mps=15.0, dt_s=DT)
        tick = _shadow(pout, None)
        why = _cmd_refused(tick.applied)
        return why or (None if tick.should_disengage else "should_disengage False")

    def _stale_cap():
        cmd = actuate.DriveCommand(steer=0.0, throttle=0.5, brake=0.0, seq=1, reason="ok")
        launch_guard.apply_stale_speed_cap(cmd, speed_mps=None, lane_age_s=5.0, lane_src="pair")
        return None if (cmd.throttle, cmd.brake) == (0.0, 0.0) else f"thr/brk {(cmd.throttle, cmd.brake)} (None read as slow)"

    def _launch():
        cmd = actuate.DriveCommand(steer=0.0, throttle=0.5, brake=0.0, seq=1, reason="ok")
        out = launch_guard.LaunchGuard().update(cmd, engaged=True, dt_s=DT, heading_rad=0.0, speed_mps=None,
                                                damage=0.0, pos_xy=(0.0, 0.0))
        if out not in (None, UNKNOWN):
            return f"returned {out!r}"
        return None if (cmd.throttle, cmd.brake) == (0.0, 0.0) else f"thr/brk {(cmd.throttle, cmd.brake)} (None read as slow)"

    def _respawn():
        from types import SimpleNamespace

        watch = RespawnWatch()
        a = SimpleNamespace(connected=True, vid=1, pos=(0.0, 0.0, 0.0), vel=(30.0, 0.0, 0.0), speed_mps=None, note="")
        b = SimpleNamespace(connected=True, vid=1, pos=(25.0, 0.0, 0.0), vel=(30.0, 0.0, 0.0), speed_mps=None, note="")
        watch.observe(a, None, mono=0.0)
        out = watch.observe(b, None, mono=1.0)
        return None if out is None else f"{out!r}: 25 m in 1 s at 30 m/s body speed read as a respawn (speed_mps None -> 0)"

    def _resolve():
        from python.control import ego_speed  # noqa: PLC0415 - new module

        sig = inspect.signature(ego_speed.resolve_ego)
        ego = ego_speed.resolve_ego(**{n: None for n, p in sig.parameters.items()
                                       if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)})
        return None if ego.mps is None and ego.why else f"resolve_ego with nothing measured gave {ego}"

    def _run_vision_block():
        src = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
        left = [s for s in ("last_ego_v", "float(ego_v or 0.0)", "spd_now if spd_now is not None else ego_v")
                if s in src]
        return None if not left else f"run_vision still substitutes: {left}"

    for name, fn in (
        ("pipeline.ModularPerception.tick", _tick),
        ("pipeline.drive_plan_from_lanes", _drive_plan),
        ("cipv.select_cipv", _cipv),
        ("path_predictor.predict_path", _predict),
        ("path_predictor.follow_path", _follow),
        ("path_predictor._expose_pedals", _expose),
        ("path_predictor.reckon_path", _reckon),
        ("path_predictor._apply_scene", _scene),
        ("path_predictor._sign_ttc", _sign_ttc),
        ("path_predictor._speed_flags", _flags),
        ("corridor.build_path_ego", _corridor),
        ("speed.plan_speed", _plan_speed),
        ("actuate.plan_command", _plan_command),
        ("actuate.safe_command", _safe_command),
        ("actuate.steer_lookahead_m", _lookahead),
        ("actuate.curve_steer", _curve),
        ("e2e.E2EPolicy.preprocess", _preprocess),
        ("e2e.E2EPolicy.forward", _forward),
        ("shadow.modular_intent", _modular),
        ("shadow.shadow_tick", _shadow_tick),
        ("launch_guard.apply_stale_speed_cap", _stale_cap),
        ("launch_guard.LaunchGuard.update", _launch),
        ("live_hold.RespawnWatch.observe", _respawn),
        ("run_vision ego resolution (ego_speed.resolve_ego)", _resolve),
        ("run_vision curve cap/steer block", _run_vision_block),
    ):
        case(name, fn)
    for line in results:
        print(f"  {line}")
    bad = [r for r in results if ": FAIL " in r]
    assert not bad, f"{len(bad)}/{len(results)} speed functions do not refuse None: " + "; ".join(bad)


# ---- test 13: a measured 0.02 m/s is known --------------------------------------


def check_13_measured_002_is_known() -> None:
    """0.02 m/s (a real stopped car) is a measurement: known, the kept guards decide, no ZeroDivisionError."""
    from python.control.actuate import plan_command
    from python.perception.cipv import select_cipv
    from python.planning import path_predictor as pp

    v = 0.02
    results: list[str] = []

    def case(name: str, fn: Callable[[], str | None]) -> None:
        try:
            why = fn()
        except Exception as exc:  # noqa: BLE001
            why = f"raised {type(exc).__name__}: {exc}"
        results.append(f"{name}: {'ok' if why is None else 'FAIL ' + why}")

    frame = _road()
    car = _Tracker(20.0).as_dicts()
    seen = _perc(None)
    seen.tick(frame, stitch_bgr=frame, ego_speed_mps=v, dt_s=DT)
    lanes = list(seen._last_lanes.lanes_bev)
    path = [dict(p) for p in pp.predict_path(lanes_bev=lanes, lane_conf=0.9, ego_speed_mps=v).path_ego]

    def _cipv():
        res = select_cipv(car, ego_speed_mps=v)
        return None if (res.ttc_lead, res.aeb) == (None, "off") else f"ttc/aeb {(res.ttc_lead, res.aeb)}"

    def _sign():
        out = pp._sign_ttc(12.0, v)
        return None if out is None else f"_sign_ttc {out}"

    def _flags():
        aeb, _target, _ttc = pp._speed_flags(curv=0.0, stop_reason="vehicle", ttc=None, stop_dist=1.0, closing_mps=0.0,
                                             ego_speed_mps=v, signs=[], path=path)
        return None if aeb == "off" else f"aeb {aeb!r} (the >0.5 m/s guard must keep a bumper gap off hard)"

    def _plan():
        plan = pp.predict_path(lanes_bev=lanes, lane_conf=0.9, tracks=[], ego_speed_mps=v)
        cmd = plan_command(path_ego=plan.path_ego, planner=plan.planner_dict(), ego_speed_mps=v, seq=1)
        return None if cmd.reason == "plan" and cmd.throttle > 0.0 else f"cmd {(cmd.throttle, cmd.brake, cmd.reason)}"

    def _tick():
        pout = _perc(None).tick(frame, stitch_bgr=frame, ego_speed_mps=v, dt_s=DT)
        cmd = _shadow(pout, v).applied
        return None if cmd.reason == "ok" else f"shadow reason {cmd.reason!r}"

    def _known_and_engage():
        from python.control import ego_speed  # noqa: PLC0415 - new module

        ego = ego_speed.EgoSpeed(mps=v, source="echo", why="fresh")
        if ego.mps != v:
            return f"EgoSpeed.mps {ego.mps}"
        allow, reason = ego_speed.engage_gate(True, True, ego)
        if not allow or reason == UNKNOWN:
            return f"engage_gate refused a measured {v} m/s: {reason!r}"
        unknown = ego_speed.EgoSpeed(mps=None, source="none", why="no echo this session")
        allow, reason = ego_speed.engage_gate(True, True, unknown)
        return None if (allow, reason) == (False, UNKNOWN) else f"engage_gate(None) gave {(allow, reason)}"

    for name, fn in (
        ("cipv.select_cipv 0.02 -> ttc None / aeb off", _cipv),
        ("path_predictor._sign_ttc 0.02 -> None", _sign),
        ("path_predictor._speed_flags 0.02 -> L1309 not hard", _flags),
        ("actuate.plan_command 0.02 -> a normal cmd", _plan),
        ("pipeline + shadow_tick 0.02 -> drives (not ego_unknown)", _tick),
        ("ego_speed: EgoSpeed.mps=0.02 known, engage allowed", _known_and_engage),
    ):
        case(name, fn)
    for line in results:
        print(f"  {line}")
    bad = [r for r in results if ": FAIL " in r]
    assert not bad, f"{len(bad)}/{len(results)}: " + "; ".join(bad)


# ---- test 14: the refusal reason is sticky ---------------------------------------

STICKY_TICKS = 20


def check_14_refusal_reason_sticky() -> None:
    """After a refusal, gvd_state.last_refusal_reason still names it STICKY_TICKS ticks later.

    Tick-driven: ticks are counted by gvd_state.cmd_seq (one per supervisor tick). The time
    limit is only a hang guard. The refusal is lua_echo_outdated (a 1.7.12 echo), which
    refuses today, so the only thing under test is that `not_engaged` does not overwrite it.
    """
    write_engage_flag(False)
    with contextlib.suppress(FileNotFoundError):
        ego_path().unlink()
    mod = _Mod(mod_version="1.7.12")
    mod.echoing = True
    mark: dict = {}

    def _seq(snap: dict) -> int | None:
        v = snap["state"].get("cmd_seq")
        return int(v) if isinstance(v, int) else None

    def step(_t: float) -> None:
        if "on" in mark:
            return
        seq = _seq({"state": _read(state_path())})
        if seq is not None and seq >= 10:  # ten ticks in, the player presses Alt+G
            mark["on"] = time.time()
            _engage_now()

    def until(snap: dict) -> bool:
        if "on" not in mark:
            return False
        eng = snap["engage"]
        if "s0" not in mark:
            if (eng.get("engaged") is False and eng.get("disengage_reason") == "lua_echo_outdated"
                    and float(eng.get("mtime") or 0.0) >= mark["on"] and _seq(snap) is not None):
                mark["s0"] = _seq(snap)
            return False
        seq = _seq(snap)
        return seq is not None and seq >= mark["s0"] + STICKY_TICKS

    seen = _session(20.0, mod, step, until=until)
    assert "s0" in mark, "lua_echo_outdated never refused the engage"
    st = seen[-1]["state"]
    got = {k: st.get(k) for k in ("last_refusal_reason", "disengage_reason", "engaged", "cmd_seq")}
    print(f"  {STICKY_TICKS} ticks after the refusal (cmd_seq {mark['s0']} -> {got['cmd_seq']}): {got}")
    assert st.get("engaged") is not True, ("engaged on a 1.7.12 echo", got)
    assert st.get("last_refusal_reason") == "lua_echo_outdated", (
        f"gvd_state.last_refusal_reason after {STICKY_TICKS} idle ticks", got)
    write_engage_flag(False)


CHECKS = (
    check_01_car_20m_ego_15,
    check_02_car_30m_ego_15,
    check_03_car_15m_ego_15,
    check_04_sign_12m_ego_12,
    check_05_sign_20m_ego_12,
    check_06_clear_road_ego_15,
    check_07_engage_before_first_echo,
    check_08_restart_with_latch_on,
    check_09_stale_echo_mid_run,
    check_10_waiting_while_latched,
    check_11_tech_stale_ego,
    check_12_none_into_every_speed_function,
    check_13_measured_002_is_known,
    check_14_refusal_reason_sticky,
)


def main() -> None:
    only = set(sys.argv[1:])
    failed: list[str] = []
    for fn in CHECKS:
        if only and fn.__name__ not in only:
            continue
        try:
            with isolated_home(prefix=f"gvd-unknown-ego-{fn.__name__}-"):
                fn()
            print(f"{fn.__name__}: ok", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            kind = "FAIL" if isinstance(exc, AssertionError) else f"ERROR {type(exc).__name__}"
            print(f"{fn.__name__}: {kind} {exc!r}"[:2000], flush=True)
            if not isinstance(exc, AssertionError):
                traceback.print_exc()
    if failed:
        raise SystemExit("test_unknown_ego failed: " + ", ".join(failed))
    print("test_unknown_ego: OK")


if __name__ == "__main__":
    main()
