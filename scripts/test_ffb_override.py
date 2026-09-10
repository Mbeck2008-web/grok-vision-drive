#!/usr/bin/env python3
"""Offline player-override checks: force-feedback steer noise must not disengage GVD.

The signal is |steering_input − aligned cmd.steer|, never an absolute angle. Spike reject +
EMA + hysteresis + dwell are exercised at an exact tick so assertions do not depend on host
timing. Live FFB behaviour is UNPROVEN — these are the research-pin magnitudes, not a wheel.
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
    LPF_TAU_MS,
    OPPOSITION_GAIN,
    REASON_BRAKE,
    REASON_STEER,
    REASON_THROTTLE,
    STEER_ENTER,
    STEER_EXIT,
    STEER_HOLD_MS,
    STEER_SPIKE,
    OverrideConfig,
    OverrideDetector,
    config_mirror,
    ema_alpha,
    load_override_config,
    opposition,
)

CONTROL_YAML = ROOT / "config" / "control.yaml"
MAIN_LUA = ROOT / "beamng_mod" / "lua" / "ge" / "extensions" / "gvd" / "main.lua"
TICK = 0.05  # 20 Hz, the retail apply rate
PIN_KEYS = (
    "steer_enter",
    "steer_exit",
    "steer_hold_ms",
    "steer_spike",
    "brake_enter",
    "throttle_enter",
    "lpf_tau_ms",
)


class Sim:
    """Drives a detector in production order: update, then note the applied command.

    `applied_seq` on this tick is the seq we noted last tick — the same one-tick lag the mod
    acks in gvd_ego.json, so a steer ramp does not look like a driver.
    """

    def __init__(self, cfg: OverrideConfig, tick: float = TICK) -> None:
        self.det = OverrideDetector(cfg)
        self.tick = tick
        self.t = 100.0
        self.seq = 0
        self.acked = -1

    def step(self, *, cmd=(0.0, 0.0, 0.0), echo=(0.0, 0.0, 0.0), engaged: bool = True):
        self.t += self.tick
        self.seq += 1
        v = self.det.update(
            engaged=engaged,
            steering_input=echo[0],
            throttle_input=echo[1],
            brake_input=echo[2],
            applied_seq=self.acked,
            now=self.t,
        )
        self.det.note_command(seq=self.seq, steer=cmd[0], throttle=cmd[1], brake=cmd[2], now=self.t)
        self.acked = self.seq
        return v

    def warm(self, **kw):
        """Command past the warm-up so the next step is actually judged."""
        out = []
        while not self.det.armed(self.t) and len(out) < 200:
            out.append(self.step(**kw))
        assert self.det.armed(self.t), "warm-up did not arm the detector"
        return out


def check_config() -> None:
    cfg = load_override_config(yaml.safe_load(CONTROL_YAML.read_text(encoding="utf-8")))
    ov = yaml.safe_load(CONTROL_YAML.read_text(encoding="utf-8"))["override"]
    assert set(ov) == set(PIN_KEYS), sorted(ov)
    assert cfg.steer_enter == STEER_ENTER == ov["steer_enter"] == 0.08
    assert cfg.steer_exit == STEER_EXIT == ov["steer_exit"] == 0.04
    assert cfg.steer_hold_ms == STEER_HOLD_MS == ov["steer_hold_ms"] == 200
    assert cfg.steer_spike == STEER_SPIKE == ov["steer_spike"] == 0.20
    assert cfg.brake_enter == ov["brake_enter"] == 0.06
    assert cfg.throttle_enter == ov["throttle_enter"] == 0.10
    assert cfg.lpf_tau_ms == LPF_TAU_MS == ov["lpf_tau_ms"] == 80
    assert cfg.steer_exit < cfg.steer_enter
    assert cfg.brake_enter < cfg.throttle_enter, "pedals are asymmetric: brake is tighter"

    assert load_override_config(None) == OverrideConfig()
    assert load_override_config({"override": None}) == OverrideConfig()
    junk = load_override_config({"override": {"steer_enter": "nope", "lpf_tau_ms": float("nan")}})
    assert junk.steer_enter == STEER_ENTER and junk.lpf_tau_ms == LPF_TAU_MS
    silly = load_override_config({"override": {"steer_enter": 0.08, "steer_exit": 0.9}})
    assert silly.steer_exit < silly.steer_enter

    mirror = config_mirror(cfg)
    assert set(mirror) == set(PIN_KEYS)
    assert load_override_config({"override": mirror}) == cfg, mirror


def check_lua_defaults_match_yaml() -> None:
    body = MAIN_LUA.read_text(encoding="utf-8")
    block = re.search(r"local OVR = \{(.*?)\n\}", body, re.S)
    assert block, "local OVR defaults table not found in main.lua"
    lua: dict[str, object] = {}
    for key, val in re.findall(r"(\w+)\s*=\s*([^,\n]+),", block.group(1)):
        lua[key] = float(val.strip())
    ov = yaml.safe_load(CONTROL_YAML.read_text(encoding="utf-8"))["override"]
    assert set(lua) == set(ov) == set(PIN_KEYS), (sorted(lua), sorted(ov))
    for key, want in ov.items():
        assert abs(float(lua[key]) - float(want)) < 1e-9, f"OVR.{key} = {lua[key]}, yaml = {want}"
    assert re.search(r"local CMD_DEAD_S = 1\.0", body), "CMD_DEAD_S must stay 1.0 — this is not a dead-man change"
    assert "player_steer" in body and "player_brake" in body and "player_throttle" in body


def check_residual_not_absolute() -> None:
    """A wheel at 0.5 while GVD also commands 0.5 is following, not a driver."""
    cfg = OverrideConfig(steer_hold_ms=0, lpf_tau_ms=0)
    sim = Sim(cfg)
    sim.warm(cmd=(0.5, 0.0, 0.0), echo=(0.5, 0.0, 0.0))
    for _ in range(20):
        v = sim.step(cmd=(0.5, 0.0, 0.0), echo=(0.5, 0.0, 0.0))
        assert not v.active and abs(v.steer_raw) < 1e-6, v

    # The same 0.5 of lock against a straight command is a driver.
    sim = Sim(cfg)
    sim.warm()
    assert any(sim.step(echo=(0.5, 0.0, 0.0)).active for _ in range(8))


def check_aligned_cmd_not_latest() -> None:
    """Echo lags the command by a tick: residual is against the aligned cmd, not the new one."""
    cfg = OverrideConfig(steer_hold_ms=0, lpf_tau_ms=0, steer_spike=1.0)
    sim = Sim(cfg)
    sim.warm(cmd=(0.0, 0.0, 0.0), echo=(0.0, 0.0, 0.0))
    # GVD ramps; the echo still shows last tick's 0. Against the new command that would be -0.4.
    v = sim.step(cmd=(0.4, 0.0, 0.0), echo=(0.0, 0.0, 0.0))
    assert not v.active and abs(v.steer_raw) < 1e-6, f"ramp lag read as a driver: {v}"
    assert v.ref_seq == sim.seq - 1
    # Next tick the echo has caught up.
    v = sim.step(cmd=(0.4, 0.0, 0.0), echo=(0.4, 0.0, 0.0))
    assert not v.active and abs(v.steer_raw) < 1e-6, v


def check_ema_on_residual_only() -> None:
    """The filter is a first-order EMA of the residual, time-correct so 10 Hz and 20 Hz agree."""
    assert abs(ema_alpha(0.05, 0.08) - (1 - math.exp(-0.05 / 0.08))) < 1e-12
    assert ema_alpha(0.05, 0.0) == 1.0

    cfg = OverrideConfig(lpf_tau_ms=80, steer_hold_ms=10_000, steer_spike=1.0)
    sim = Sim(cfg)
    sim.warm()
    v = sim.step(echo=(0.5, 0.0, 0.0))
    a = ema_alpha(TICK, cfg.lpf_tau_s)
    assert abs(v.steer_filt - a * 0.5) < 1e-3, v
    v = sim.step(echo=(0.5, 0.0, 0.0))
    expect = a * 0.5 + a * (0.5 - a * 0.5)
    assert abs(v.steer_filt - expect) < 1e-3, v
    # tau=0 is passthrough: the filtered residual is the raw one.
    raw = Sim(OverrideConfig(lpf_tau_ms=0, steer_hold_ms=10_000, steer_spike=1.0))
    raw.warm()
    v = raw.step(echo=(0.3, 0.0, 0.0))
    assert abs(v.steer_filt - 0.3) < 1e-6 and abs(v.steer_raw - 0.3) < 1e-6, v


def check_spike_reject() -> None:
    """A sample-to-sample jump past steer_spike is mechanical: the EMA holds."""
    cfg = OverrideConfig(steer_hold_ms=10_000, lpf_tau_ms=0, steer_spike=0.20)
    sim = Sim(cfg)
    sim.warm()
    # Rest → 0.7 is a spike; filter stays at 0 so a one-tick kick cannot trip.
    v = sim.step(echo=(0.7, 0.0, 0.0))
    assert v.spike and abs(v.steer_filt) < 1e-6 and not v.active, v
    # Holding 0.7 after that is a zero jump, so the filter is allowed to follow.
    v = sim.step(echo=(0.7, 0.0, 0.0))
    assert not v.spike and abs(v.steer_filt - 0.7) < 1e-6, v


def check_ffb_chatter_holds() -> None:
    """The bug: a force-feedback wheel chatters and GVD used to hand the car back."""
    cfg = load_override_config(yaml.safe_load(CONTROL_YAML.read_text(encoding="utf-8")))

    sim = Sim(cfg)
    sim.warm()
    for i in range(80):
        v = sim.step(echo=(0.9 if i % 2 else -0.9, 0.0, 0.0))
        assert not v.active, f"chatter tick {i} disengaged: {v}"
        assert v.spike or abs(v.steer_filt) <= cfg.steer_enter, v
    assert sim.det.verdict.steer_held_ms < cfg.steer_hold_ms

    # Steady residual inside the exit band: never even reaches the dwell.
    sim = Sim(cfg)
    sim.warm()
    for _ in range(40):
        v = sim.step(echo=(cfg.steer_exit * 0.5, 0.0, 0.0))
        assert not v.active and v.steer_held_ms == 0.0, v

    # Chatter riding on top of GVD's own steer: residual is the difference, still a spike.
    sim = Sim(cfg)
    sim.warm(cmd=(0.3, 0.2, 0.0), echo=(0.3, 0.2, 0.0))
    for i in range(40):
        echo_s = 0.3 + (0.7 if i % 2 else -0.7)
        v = sim.step(cmd=(0.3, 0.2, 0.0), echo=(echo_s, 0.2, 0.0))
        assert not v.active, v


def check_real_driver_wins() -> None:
    """Compensation must not cost us a real takeover."""
    cfg = load_override_config(yaml.safe_load(CONTROL_YAML.read_text(encoding="utf-8")))

    sim = Sim(cfg)
    sim.warm()
    seen = [sim.step(echo=(0.25, 0.0, 0.0)) for _ in range(int(cfg.steer_hold_s / TICK) + 12)]
    # The first tick off rest is a spike (0.25 > 0.20); the hold starts on the next.
    assert seen[0].spike and not seen[0].active, seen[0]
    first = next(i for i, v in enumerate(seen) if v.active)
    assert seen[first].channel == "steer" and seen[first].reason == REASON_STEER
    assert all(v.active and v.reason == REASON_STEER for v in seen[first:])
    # Left is the same story.
    sim = Sim(cfg)
    sim.warm()
    assert any(sim.step(echo=(-0.25, 0.0, 0.0)).active for _ in range(20))

    # Pedals: asymmetric, tight, no filter, no dwell. Brake is the lower threshold.
    sim = Sim(cfg)
    sim.warm(cmd=(0.0, 0.3, 0.0), echo=(0.0, 0.3, 0.0))
    v = sim.step(cmd=(0.0, 0.3, 0.0), echo=(0.0, 0.3, 0.10))
    assert v.active and v.channel == "brake" and v.reason == REASON_BRAKE, v
    assert v.pedal_residual >= cfg.brake_enter

    sim = Sim(cfg)
    sim.warm()
    v = sim.step(echo=(0.0, 0.20, 0.0))
    assert v.active and v.channel == "throttle" and v.reason == REASON_THROTTLE, v

    sim = Sim(cfg)
    sim.warm()
    under = (0.0, cfg.throttle_enter * 0.9, cfg.brake_enter * 0.9)
    assert not sim.step(echo=under).active

    # Brake wins the tie so the reason players see matches what the car did.
    sim = Sim(cfg)
    sim.warm()
    assert sim.step(echo=(0.0, 0.9, 0.9)).reason == REASON_BRAKE

    # A driver fighting GVD mid-corner still wins — residual, not absolute angle.
    sim = Sim(cfg)
    sim.warm(cmd=(0.4, 0.0, 0.0), echo=(0.4, 0.0, 0.0))
    assert any(sim.step(cmd=(0.4, 0.0, 0.0), echo=(-0.1, 0.0, 0.0)).active for _ in range(20))


def check_hysteresis_and_opposition() -> None:
    cfg = load_override_config(yaml.safe_load(CONTROL_YAML.read_text(encoding="utf-8")))
    mid = (cfg.steer_exit + cfg.steer_enter) / 2.0

    sim = Sim(cfg)
    sim.warm()
    # Get the filter onto a real residual (second tick is not a spike), then freeze in the band.
    sim.step(echo=(0.25, 0.0, 0.0))
    sim.step(echo=(0.25, 0.0, 0.0))
    held = sim.det.verdict.steer_held_ms
    assert held > 0.0
    for _ in range(8):
        v = sim.step(echo=(mid, 0.0, 0.0))
        assert not v.active
        # Filter is still settling toward `mid`; the dwell must not discharge in the band.
        assert v.steer_held_ms >= held - 1e-6, v

    sim.step(echo=(cfg.steer_exit * 0.25, 0.0, 0.0))
    # Give the EMA a tick to fall under exit.
    for _ in range(8):
        v = sim.step(echo=(0.0, 0.0, 0.0))
        if v.steer_held_ms == 0.0:
            break
    assert sim.det.verdict.steer_held_ms == 0.0

    # Soft opposition: fighting GVD's steer inflates the residual, going with it does not.
    assert opposition(0.4, -0.1) == 1.0
    assert opposition(0.4, 0.1) == 0.0
    assert opposition(0.0, 0.2) == 0.0
    raw = 0.07  # just under enter
    assert raw < STEER_ENTER < raw * (1.0 + OPPOSITION_GAIN)


def check_warmup() -> None:
    cfg = load_override_config(yaml.safe_load(CONTROL_YAML.read_text(encoding="utf-8")))
    sim = Sim(cfg)
    early, judged = [], None
    for _ in range(40):
        v = sim.step(echo=(0.4, 0.9, 0.9))
        if v.armed:
            judged = v
            break
        early.append(v)
    assert len(early) >= math.floor(cfg.warmup_s / TICK) - 1, len(early)
    assert not any(v.active for v in early), "override fired during the warm-up"
    assert judged is not None and judged.active, judged
    assert judged.reason in (REASON_STEER, REASON_BRAKE, REASON_THROTTLE)

    sim.step(echo=(0.0, 0.0, 0.0), engaged=False)
    assert not sim.det.armed(sim.t)
    assert not sim.step(echo=(0.0, 0.0, 0.9)).active


def check_spike_is_load_bearing() -> None:
    """Without the spike reject, a one-tick FFB kick is an override on the tick it lands."""

    def kick_trips(cfg: OverrideConfig) -> bool:
        sim = Sim(cfg)
        sim.warm()
        return sim.step(echo=(0.7, 0.0, 0.0)).active

    shipped = load_override_config(yaml.safe_load(CONTROL_YAML.read_text(encoding="utf-8")))
    instant = OverrideConfig(
        steer_enter=shipped.steer_enter,
        steer_exit=shipped.steer_exit,
        steer_hold_ms=0,
        steer_spike=shipped.steer_spike,
        brake_enter=shipped.brake_enter,
        throttle_enter=shipped.throttle_enter,
        lpf_tau_ms=0,  # passthrough, so a kick would trip this tick if it reached the filter
    )
    assert not kick_trips(instant), "a one-tick kick must not disengage"
    naive = OverrideConfig(
        steer_enter=shipped.steer_enter,
        steer_exit=shipped.steer_exit,
        steer_hold_ms=0,
        steer_spike=2.0,  # larger than any residual jump: every sample is believable
        brake_enter=shipped.brake_enter,
        throttle_enter=shipped.throttle_enter,
        lpf_tau_ms=0,
    )
    assert kick_trips(naive), "steer_spike is what keeps a one-tick FFB kick out of the filter"

    # Alternating chatter still must not trip the shipped defaults (spike + EMA + dwell).
    sim = Sim(shipped)
    sim.warm()
    assert not any(sim.step(echo=(0.9 if i % 2 else -0.9, 0.0, 0.0)).active for i in range(80))


def check_missing_echo() -> None:
    cfg = load_override_config(yaml.safe_load(CONTROL_YAML.read_text(encoding="utf-8")))
    det = OverrideDetector(cfg)
    t = 0.0
    for i in range(30):
        t += TICK
        det.note_command(seq=i, steer=0.0, throttle=0.2, brake=0.0, now=t)
        v = det.update(engaged=True, applied_seq=i, now=t)
        assert not v.active and v.channel == "none", v


def check_reference_fallback() -> None:
    """No applied_seq (BeamNGpy electrics) → previous command, the same one-tick lag."""
    det = OverrideDetector(OverrideConfig(lpf_tau_ms=0, steer_hold_ms=0, steer_spike=1.0))
    t = 0.0
    det.note_command(seq=1, steer=0.0, throttle=0.0, brake=0.0, now=t)
    t += 0.3
    det.note_command(seq=2, steer=0.4, throttle=0.0, brake=0.0, now=t)
    ref = det.reference(None)
    assert ref.seq == 1 and ref.steer == 0.0
    ref = det.reference(2)
    assert ref.seq == 2 and ref.steer == 0.4
    assert det.reference(99).seq == 2  # unknown seq → latest


def check_aeb_echo_not_brake() -> None:
    cfg = load_override_config(yaml.safe_load(CONTROL_YAML.read_text(encoding="utf-8")))
    sim = Sim(cfg)
    sim.warm(cmd=(0.0, 0.0, 1.0), echo=(0.0, 0.0, 1.0))
    for _ in range(20):
        v = sim.step(cmd=(0.0, 0.0, 1.0), echo=(0.0, 0.0, 1.0))
        assert not v.active, f"our own brake=1 echoed back: {v}"


def main() -> None:
    check_config()
    check_lua_defaults_match_yaml()
    check_residual_not_absolute()
    check_aligned_cmd_not_latest()
    check_ema_on_residual_only()
    check_spike_reject()
    check_ffb_chatter_holds()
    check_real_driver_wins()
    check_hysteresis_and_opposition()
    check_warmup()
    check_spike_is_load_bearing()
    check_missing_echo()
    check_reference_fallback()
    check_aeb_echo_not_brake()
    print("test_ffb_override: OK")


if __name__ == "__main__":
    main()
