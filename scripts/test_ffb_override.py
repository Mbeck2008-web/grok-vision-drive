#!/usr/bin/env python3
"""Offline driver-override checks: force-feedback steer chatter must not disengage GVD.

Everything here is deterministic — `now` is passed explicitly, so the dwell and hysteresis are
exercised at exact times instead of at whatever rate the host happens to run.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from python.control.override import (  # noqa: E402
    OVERRIDE_REASON,
    OverrideConfig,
    OverrideDetector,
    config_mirror,
    deadband,
    load_override_config,
    residual,
)

CONTROL_YAML = ROOT / "config" / "control.yaml"
MAIN_LUA = ROOT / "beamng_mod" / "lua" / "ge" / "extensions" / "gvd" / "main.lua"
TICK = 0.05  # 20 Hz, the retail apply rate


def _yaml_override() -> dict:
    return yaml.safe_load(CONTROL_YAML.read_text(encoding="utf-8"))["override"]


class Sim:
    """Drives a detector at a fixed tick so dwell assertions are exact."""

    def __init__(self, cfg: OverrideConfig, tick: float = TICK) -> None:
        self.det = OverrideDetector(cfg)
        self.tick = tick
        self.t = 100.0

    def step(self, *, cmd=(0.0, 0.0, 0.0), echo=(0.0, 0.0, 0.0), engaged: bool = True):
        self.t += self.tick
        self.det.note_command(steer=cmd[0], throttle=cmd[1], brake=cmd[2], now=self.t)
        return self.det.update(
            engaged=engaged,
            steering_input=echo[0],
            throttle_input=echo[1],
            brake_input=echo[2],
            now=self.t,
        )

    def run(self, n: int, **kw) -> list:
        return [self.step(**kw) for _ in range(n)]

    def warm(self, **kw):
        """Command past the warm-up window so the next step is actually judged."""
        out = []
        while not self.det.armed(self.t) and len(out) < 200:
            out.append(self.step(**kw))
        assert self.det.armed(self.t), "warm-up did not arm the detector"
        return out


def check_primitives() -> None:
    assert residual(0.9, 0.0, 0.0) == 0.9
    assert residual(-0.9, 0.0, 0.0) == -0.9
    assert residual(0.4, 0.0, 1.0) == 0.0, "inside the command envelope is not a driver"
    assert round(residual(0.4, -0.2, 0.1), 4) == 0.3
    # Subtractive: inside the band there is no driver, outside it the band comes off.
    assert deadband(0.09, 0.10) == 0.0 and round(deadband(0.60, 0.10), 4) == 0.50
    assert deadband(-0.09, 0.10) == 0.0 and round(deadband(-0.60, 0.10), 4) == -0.50
    assert deadband(0.5, 0.0) == 0.5, "a zero-width band must pass everything through"


def check_config() -> None:
    """config/control.yaml owns the thresholds; nonsense values fall back to the defaults."""
    cfg = load_override_config(yaml.safe_load(CONTROL_YAML.read_text(encoding="utf-8")))
    ov = _yaml_override()
    assert cfg.steer_deadband == ov["steer_deadband"] > 0.0
    assert cfg.steer_enter == ov["steer_enter"]
    assert cfg.steer_clear == ov["steer_clear"] < cfg.steer_enter
    assert cfg.steer_hold_s == ov["steer_hold_s"] > 0.0
    assert cfg.brake_enter == ov["brake_enter"] and cfg.throttle_enter == ov["throttle_enter"]
    assert cfg.pedal_hold_s == 0.0, "pedals stay hard: no dwell"
    assert cfg.ffb_assume_wheel is True

    assert load_override_config(None) == OverrideConfig()
    assert load_override_config({"override": None}) == OverrideConfig()
    junk = load_override_config({"override": {"steer_enter": "nope", "brake_enter": None, "steer_hold_s": float("nan")}})
    assert junk.steer_enter == OverrideConfig().steer_enter
    assert junk.brake_enter == OverrideConfig().brake_enter
    assert junk.steer_hold_s == OverrideConfig().steer_hold_s
    # A clear band at/above the trip point would freeze the dwell forever, and a deadband near
    # full lock would put the trip point out of a driver's reach.
    silly = load_override_config({"override": {"steer_enter": 0.4, "steer_deadband": 0.9, "steer_clear": 0.8}})
    assert silly.steer_clear < silly.steer_enter and silly.steer_deadband <= 0.5

    # The mirror Lua reads must round-trip back to the same thresholds.
    mirror = config_mirror(cfg)
    assert load_override_config({"override": mirror}) == cfg, mirror
    assert set(mirror) == {f.name for f in OverrideConfig.__dataclass_fields__.values()}


def check_lua_defaults_match_yaml() -> None:
    """The mod runs standalone until the first gvd_state arrives, so its defaults must agree."""
    body = MAIN_LUA.read_text(encoding="utf-8")
    block = re.search(r"local OVR = \{(.*?)\n\}", body, re.S)
    assert block, "local OVR defaults table not found in main.lua"
    lua: dict[str, object] = {}
    for key, val in re.findall(r"(\w+)\s*=\s*([^,\n]+),", block.group(1)):
        val = val.strip()
        lua[key] = True if val == "true" else False if val == "false" else float(val)
    ov = _yaml_override()
    assert set(lua) == set(ov), (sorted(lua), sorted(ov))
    for key, want in ov.items():
        got = lua[key]
        if isinstance(want, bool):
            assert got is want, f"OVR.{key} = {got}, control.yaml = {want}"
        else:
            assert abs(float(got) - float(want)) < 1e-9, f"OVR.{key} = {got}, control.yaml = {want}"


def check_ffb_chatter_holds() -> None:
    """The bug: a force-feedback wheel chatters and GVD used to hand the car back."""
    cfg = load_override_config(yaml.safe_load(CONTROL_YAML.read_text(encoding="utf-8")))

    # Alternating kicks well past the trip point — the classic FFB signature over bumps.
    sim = Sim(cfg)
    for i in range(60):
        v = sim.step(echo=(0.9 if i % 2 else -0.9, 0.0, 0.0))
        assert not v.active, f"chatter tick {i} disengaged: {v}"
    assert sim.det.verdict.steer_held_s < cfg.steer_hold_s

    # Steady jitter inside the deadband: never even reaches the dwell.
    sim = Sim(cfg)
    for i in range(60):
        v = sim.step(echo=(cfg.steer_deadband * (0.9 if i % 3 else -0.9), 0.0, 0.0))
        assert not v.active and v.steer_residual == 0.0, v
    assert sim.det.verdict.deadband_swallowed

    # The deadband earns its keep here: a hold that sits between the old bare 0.55 threshold and
    # the compensated trip point is exactly the FFB case M6 disengaged on.
    borderline = cfg.steer_enter + cfg.steer_deadband * 0.5
    sim = Sim(cfg)
    sim.warm()
    for _ in range(20):
        v = sim.step(echo=(borderline, 0.0, 0.0))
        assert not v.active, f"{borderline:.2f} of lock is inside the deadband allowance: {v}"
    no_band = Sim(OverrideConfig(**{**cfg.__dict__, "steer_deadband": 0.0}))
    no_band.warm()
    assert any(no_band.step(echo=(borderline, 0.0, 0.0)).active for _ in range(20)), (
        "without the deadband the same hold disengages — that is the compensation"
    )

    # A single wheel kick past the trip point (what the old bare threshold disengaged on).
    sim = Sim(cfg)
    sim.warm()
    assert not sim.step(echo=(0.95, 0.0, 0.0)).active, "one-tick spike must not disengage"
    assert not sim.step(echo=(0.0, 0.0, 0.0)).active

    # Chatter riding on top of GVD's own steer: the echo of our command is not a driver.
    sim = Sim(cfg)
    for i in range(40):
        v = sim.step(cmd=(0.15, 0.2, 0.0), echo=(0.15 + (0.08 if i % 2 else -0.08), 0.2, 0.0))
        assert not v.active, v

    # Command released a tick ago, echo still showing the old value → inside the envelope.
    sim = Sim(cfg)
    sim.warm(cmd=(0.0, 0.0, 1.0), echo=(0.0, 0.0, 1.0))
    for _ in range(4):
        v = sim.step(cmd=(0.0, 0.3, 0.0), echo=(0.0, 0.0, 1.0))
        assert not v.active, f"stale echo of our own brake read as a driver: {v}"


def check_real_driver_wins() -> None:
    """Deadband compensation must not cost us a real takeover."""
    cfg = load_override_config(yaml.safe_load(CONTROL_YAML.read_text(encoding="utf-8")))

    # Sustained pull, one direction: trips once the dwell fills, and stays tripped.
    sim = Sim(cfg)
    sim.warm()
    seen = [sim.step(echo=(0.8, 0.0, 0.0)) for _ in range(6)]
    assert not seen[0].active, "no override on the first tick past the trip point (that is the dwell)"
    first = next(i for i, v in enumerate(seen) if v.active)
    assert first * TICK <= cfg.steer_hold_s + 2 * TICK, f"override took {first} ticks"
    assert seen[first].channel == "steer" and seen[first].reason == OVERRIDE_REASON
    assert all(v.active for v in seen[first:])
    # Left is the same story with the opposite sign.
    sim = Sim(cfg)
    sim.warm()
    assert any(sim.step(echo=(-0.8, 0.0, 0.0)).active for _ in range(6))

    # Pedals are hard: a real press is an override on the tick it is seen.
    sim = Sim(cfg)
    sim.warm(cmd=(0.0, 0.3, 0.0), echo=(0.0, 0.3, 0.0))
    v = sim.step(cmd=(0.0, 0.3, 0.0), echo=(0.0, 0.3, 0.3))
    assert v.active and v.channel == "brake" and v.pedal_residual >= cfg.brake_enter, v

    sim = Sim(cfg)
    sim.warm()
    v = sim.step(echo=(0.0, 0.4, 0.0))
    assert v.active and v.channel == "throttle", v

    # Just under the pedal thresholds is still noise.
    sim = Sim(cfg)
    sim.warm()
    assert not sim.step(echo=(0.0, cfg.throttle_enter * 0.9, cfg.brake_enter * 0.9)).active

    # Brake wins the tie so the reason players see matches what the car did.
    sim = Sim(cfg)
    sim.warm()
    assert sim.step(echo=(0.0, 0.9, 0.9)).channel == "brake"


def check_hysteresis_and_stickiness() -> None:
    cfg = load_override_config(yaml.safe_load(CONTROL_YAML.read_text(encoding="utf-8")))
    mid = (cfg.steer_clear + cfg.steer_enter) / 2.0

    # Inside the hysteresis band the dwell is frozen: it neither trips nor forgets.
    sim = Sim(cfg)
    sim.warm()
    sim.step(echo=(0.8, 0.0, 0.0))
    held = sim.det.verdict.steer_held_s
    assert held > 0.0
    for _ in range(10):
        v = sim.step(echo=(mid, 0.0, 0.0))
        assert not v.active and v.steer_held_s == held, v

    # Falling under the clear threshold discharges it, so the next trip starts from scratch.
    sim.step(echo=(cfg.steer_clear * 0.5, 0.0, 0.0))
    assert sim.det.verdict.steer_held_s == 0.0
    assert not sim.step(echo=(0.8, 0.0, 0.0)).active

    # Disengaged: never an override, and the dwell + envelope reset for the next Alt+A.
    sim = Sim(cfg)
    sim.warm()
    sim.step(echo=(0.8, 0.0, 0.0))
    assert sim.det.verdict.steer_held_s > 0.0
    v = sim.step(echo=(0.9, 0.0, 1.0), engaged=False)
    assert not v.active and v.channel == "none" and v.steer_held_s == 0.0

    # A driver fighting GVD mid-corner must still win — M6's 0.20 gate could not see this.
    sim = Sim(cfg)
    corner = 0.45
    assert corner > 0.20, "the point is a steer M6's gate would have ignored"
    sim.warm(cmd=(corner, 0.0, 0.0), echo=(corner, 0.0, 0.0))
    assert any(sim.step(cmd=(corner, 0.0, 0.0), echo=(-0.6, 0.0, 0.0)).active for _ in range(8))

    # Near full lock the residual saturates against the +/-1 clamp, so the steer channel bows out.
    sim = Sim(cfg)
    hard = min(1.0, cfg.steer_cmd_max + 0.1)
    sim.warm(cmd=(hard, 0.0, 0.0), echo=(hard, 0.0, 0.0))
    for _ in range(8):
        v = sim.step(cmd=(hard, 0.0, 0.0), echo=(1.0, 0.0, 0.0))
        assert not v.active and not v.steer_judged, v
    # ...but the pedals still take the car back out of a hard corner.
    assert sim.step(cmd=(hard, 0.0, 0.0), echo=(1.0, 0.0, 0.5)).channel == "brake"


def check_both_guards_are_load_bearing() -> None:
    """Neither guard is decoration: chatter and a transient kick fail differently.

    Alternating kicks stay past the trip point on every tick, so the dwell alone keeps charging —
    only the sign-flip reset discharges it. A one-tick spike keeps its sign, so only the dwell
    catches that one. Dropping either default puts the accidental disengage back.
    """

    def chatter_trips(cfg: OverrideConfig, n: int = 60) -> bool:
        sim = Sim(cfg)
        return any(sim.step(echo=(0.9 if i % 2 else -0.9, 0.0, 0.0)).active for i in range(n))

    def kick_trips(cfg: OverrideConfig) -> bool:
        sim = Sim(cfg)
        sim.warm()
        return sim.step(echo=(0.95, 0.0, 0.0)).active

    shipped = load_override_config(yaml.safe_load(CONTROL_YAML.read_text(encoding="utf-8")))
    assert not chatter_trips(shipped) and not kick_trips(shipped)
    # M6's bare threshold: both failure modes disengaged.
    m6 = OverrideConfig(steer_deadband=0.0, steer_hold_s=0.0, steer_sign_flip_resets=False)
    assert chatter_trips(m6) and kick_trips(m6)
    assert chatter_trips(OverrideConfig(steer_sign_flip_resets=False)), "sign-flip reset carries the chatter case"
    assert kick_trips(OverrideConfig(steer_hold_s=0.0)), "dwell carries the transient-kick case"


def check_warmup() -> None:
    """Straight after engage there is no command history to attribute the echo to.

    Without this a player already resting on the brake pedal as they press Alt+A would override
    themselves on the first tick and GVD would look impossible to engage.
    """
    cfg = load_override_config(yaml.safe_load(CONTROL_YAML.read_text(encoding="utf-8")))
    sim = Sim(cfg)
    early, judged = [], None
    for _ in range(40):
        v = sim.step(echo=(0.95, 0.9, 0.9))
        if v.armed:
            judged = v
            break
        early.append(v)
    assert len(early) >= math.floor(cfg.cmd_window_s / TICK) - 1, len(early)
    assert not any(v.active for v in early), "override fired during the warm-up"
    # The warm-up must end, not silence the check: the same input is an override once armed.
    assert judged is not None and judged.active, judged

    # Disengaging re-arms the warm-up, so Alt+A is always usable.
    sim.step(echo=(0.0, 0.0, 0.0), engaged=False)
    assert not sim.det.armed(sim.t)
    assert not sim.step(echo=(0.0, 0.0, 0.9)).active


def check_missing_echo() -> None:
    """No electrics echo at all (mod off, Tech path without Electrics) → never an override."""
    cfg = load_override_config(yaml.safe_load(CONTROL_YAML.read_text(encoding="utf-8")))
    det = OverrideDetector(cfg)
    t = 0.0
    for _ in range(20):
        t += TICK
        det.note_command(steer=0.0, throttle=0.2, brake=0.0, now=t)
        v = det.update(engaged=True, now=t)
        assert not v.active and v.channel == "none", v


def check_deadband_assumption() -> None:
    """No dependable device type, so the deadband is on unless we positively know otherwise."""
    on = OverrideDetector(OverrideConfig(ffb_assume_wheel=True))
    assert on.steer_deadband_on(None) and on.steer_deadband_on(False) and on.steer_deadband_on(True)
    off = OverrideDetector(OverrideConfig(ffb_assume_wheel=False))
    assert off.steer_deadband_on(None), "unknown must still mean deadband on"
    assert off.steer_deadband_on(True)
    assert not off.steer_deadband_on(False)

    # With the band switched off, the same one-tick kick trips again — that is the regression
    # the deadband exists to stop.
    sim = Sim(OverrideConfig(ffb_assume_wheel=False, steer_hold_s=0.0))
    sim.warm()
    sim.t += TICK
    assert sim.det.update(engaged=True, steering_input=0.9, ffb_wheel=False, now=sim.t).active


def main() -> None:
    check_primitives()
    check_config()
    check_lua_defaults_match_yaml()
    check_ffb_chatter_holds()
    check_real_driver_wins()
    check_hysteresis_and_stickiness()
    check_both_guards_are_load_bearing()
    check_warmup()
    check_missing_echo()
    check_deadband_assumption()
    print("test_ffb_override: OK")


if __name__ == "__main__":
    main()
