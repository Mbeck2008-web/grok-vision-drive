"""Driver-override detection with a force-feedback tolerant steer deadband.

M6 called it an override when `steering_input` passed 0.55 while GVD steered near-straight.
Force-feedback / racing wheels break that rule: self-aligning torque, spring centering and
kicks over bumps move `steering_input` around whatever GVD commands, so FFB chatter read as
a driver grabbing the wheel and accidentally disengaged GVD.

The steer channel now goes through

    residual (echo minus what GVD asked for) -> deadband -> hysteresis -> dwell

before it counts as a driver. The deadband is subtractive, so `steer_enter` is a threshold on
the *compensated* residual and the effective trip point sits at `steer_enter + steer_deadband`
of lock; on top of that the residual must hold one direction for `steer_hold_s`, which is what
separates a driver leaning on the wheel from a wheel shaking itself. Pedals stay hard: a brake
or throttle press past its threshold is an override on the tick it is seen, because there is no
benign reason for a pedal to move.

Both sides of the retail bus share these thresholds — the supervisor here and the GELua that
actually holds the vehicle (`beamng_mod/lua/ge/extensions/gvd/main.lua`). `config/control.yaml`
owns them; `run_vision.py` mirrors them into `gvd_state.json` as `override_cfg` so Lua tracks
the yaml without parsing it.

Known limitation: while GVD itself steers past `steer_cmd_max` the steer channel is not judged,
because near full lock the residual saturates against the +/-1 input clamp. M6 had the same gate
at a far lower 0.20, which meant a driver's steer could not be seen in any real corner; residual
against the command envelope is what makes the wider band safe. Pedals and Alt+A are unaffected.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

STEER_DEADBAND = 0.10
STEER_ENTER = 0.55
STEER_CLEAR = 0.30
STEER_HOLD_S = 0.12
STEER_CMD_MAX = 0.85
BRAKE_ENTER = 0.08
THROTTLE_ENTER = 0.15
PEDAL_HOLD_S = 0.0
CMD_WINDOW_S = 0.30

OVERRIDE_REASON = "driver_override"
_CMD_RING = 256


@dataclass(frozen=True)
class OverrideConfig:
    steer_deadband: float = STEER_DEADBAND
    steer_enter: float = STEER_ENTER
    steer_clear: float = STEER_CLEAR
    steer_hold_s: float = STEER_HOLD_S
    steer_cmd_max: float = STEER_CMD_MAX
    steer_sign_flip_resets: bool = True
    brake_enter: float = BRAKE_ENTER
    throttle_enter: float = THROTTLE_ENTER
    pedal_hold_s: float = PEDAL_HOLD_S
    cmd_window_s: float = CMD_WINDOW_S
    ffb_assume_wheel: bool = True


@dataclass
class OverrideVerdict:
    """One tick of judgement. `active` is what disengages; the rest is telemetry."""

    active: bool = False
    channel: str = "none"          # none | steer | brake | throttle
    steer_residual: float = 0.0    # after the deadband (what the dwell actually sees)
    steer_raw: float = 0.0         # before the deadband
    pedal_residual: float = 0.0
    steer_held_s: float = 0.0
    deadband_swallowed: bool = False  # residual existed but the deadband read it as zero
    steer_judged: bool = True         # false while GVD's own steer exceeds steer_cmd_max
    armed: bool = True                # false during the warm-up after engage

    @property
    def reason(self) -> str:
        return OVERRIDE_REASON if self.active else "none"


def load_override_config(cfg: dict[str, Any] | None = None) -> OverrideConfig:
    """Read the `override:` block of config/control.yaml. Bad values fall back to defaults."""
    ov = (cfg or {}).get("override") or {}
    if not isinstance(ov, dict):
        ov = {}

    def num(key: str, dflt: float, lo: float, hi: float) -> float:
        try:
            v = float(ov[key])
        except (KeyError, TypeError, ValueError):
            return dflt
        if v != v:  # NaN
            return dflt
        return min(hi, max(lo, v))

    steer_enter = num("steer_enter", STEER_ENTER, 0.02, 1.0)
    # The deadband shifts the trip point out by its own width, so cap it well short of full
    # lock or a driver could never reach the threshold. A clear band at or above `enter` would
    # leave the dwell unable to discharge.
    steer_deadband = num("steer_deadband", STEER_DEADBAND, 0.0, 0.5)
    steer_clear = min(num("steer_clear", STEER_CLEAR, 0.0, 1.0), steer_enter * 0.95)
    return OverrideConfig(
        steer_deadband=steer_deadband,
        steer_enter=steer_enter,
        steer_clear=steer_clear,
        steer_hold_s=num("steer_hold_s", STEER_HOLD_S, 0.0, 2.0),
        steer_cmd_max=num("steer_cmd_max", STEER_CMD_MAX, 0.0, 1.0),
        steer_sign_flip_resets=bool(ov.get("steer_sign_flip_resets", True)),
        brake_enter=num("brake_enter", BRAKE_ENTER, 0.01, 1.0),
        throttle_enter=num("throttle_enter", THROTTLE_ENTER, 0.01, 1.0),
        pedal_hold_s=num("pedal_hold_s", PEDAL_HOLD_S, 0.0, 2.0),
        cmd_window_s=num("cmd_window_s", CMD_WINDOW_S, 0.0, 2.0),
        ffb_assume_wheel=bool(ov.get("ffb_assume_wheel", True)),
    )


def config_mirror(cfg: OverrideConfig) -> dict[str, Any]:
    """Flat `override_cfg` for gvd_state.json so GELua runs the same thresholds as the yaml."""
    return {
        "steer_deadband": round(cfg.steer_deadband, 4),
        "steer_enter": round(cfg.steer_enter, 4),
        "steer_clear": round(cfg.steer_clear, 4),
        "steer_hold_s": round(cfg.steer_hold_s, 4),
        "steer_cmd_max": round(cfg.steer_cmd_max, 4),
        "steer_sign_flip_resets": bool(cfg.steer_sign_flip_resets),
        "brake_enter": round(cfg.brake_enter, 4),
        "throttle_enter": round(cfg.throttle_enter, 4),
        "pedal_hold_s": round(cfg.pedal_hold_s, 4),
        "cmd_window_s": round(cfg.cmd_window_s, 4),
        "ffb_assume_wheel": bool(cfg.ffb_assume_wheel),
    }


def _clamp(v: Any, lo: float, hi: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if f != f:
        return 0.0
    return min(hi, max(lo, f))


def residual(value: float, lo: float, hi: float) -> float:
    """Signed distance of `value` outside the [lo, hi] command envelope (0 while inside).

    The envelope, not a single command, is the baseline: the electrics echo lags the command
    by a tick or two on both buses, so a stale echo of what GVD asked for a moment ago must
    not read as a driver.
    """
    if value > hi:
        return value - hi
    if value < lo:
        return value - lo
    return 0.0


def deadband(value: float, width: float) -> float:
    """Subtractive deadband, the way a wheel axis dead zone works.

    Inside the band there is no driver at all; outside it the band is *subtracted* rather than
    stepped over, so the thresholds keep their meaning on the compensated signal and the whole
    trip point shifts out by `width`. A hard cutoff would do nothing here: anything it zeroed
    already sat under `steer_clear`.
    """
    mag = abs(value) - width
    if mag <= 0.0:
        return 0.0
    return mag if value > 0 else -mag


class OverrideDetector:
    """Stateful per-loop judgement. Feed it every command and every electrics echo.

    `note_command` records what was actually applied to the car — not the pre-gate intent, or a
    gate hold riding along as `brake=1` would echo back and read as the driver standing on the
    pedal. `update` then compares this tick's echo against the envelope of those commands, so
    call it before the actuator and note the applied command after.

    Judgement waits `cmd_window_s` after the first command: until GVD has asked for something
    across the whole lookback window there is nothing to attribute the echo to, and a player
    holding the brake as they press Alt+A would otherwise override themselves instantly.

    Disengage is sticky at the caller (gvd_engage.json stays false until Alt+A) — here
    `update(engaged=False)` just clears the state so re-engaging starts clean.
    """

    def __init__(self, cfg: OverrideConfig | None = None) -> None:
        self.cfg = cfg or OverrideConfig()
        self.verdict = OverrideVerdict()
        self._cmds: deque[tuple[float, float, float, float]] = deque(maxlen=_CMD_RING)
        self._steer_held = 0.0
        self._steer_sign = 0
        self._pedal_held = 0.0
        self._last_t: float | None = None
        self._armed_at: float | None = None

    def note_command(self, *, steer: float, throttle: float, brake: float, now: float | None = None) -> None:
        t = time.monotonic() if now is None else float(now)
        if self._armed_at is None:
            self._armed_at = t
        self._cmds.append((t, _clamp(steer, -1.0, 1.0), _clamp(throttle, 0.0, 1.0), _clamp(brake, 0.0, 1.0)))
        self._trim(t)

    def reset(self) -> None:
        """Forget the dwell and the command envelope (re-engage starts from a clean slate)."""
        self._cmds.clear()
        self._steer_held = 0.0
        self._steer_sign = 0
        self._pedal_held = 0.0
        self._armed_at = None
        self.verdict = OverrideVerdict()

    def armed(self, now: float) -> bool:
        return self._armed_at is not None and (now - self._armed_at) >= self.cfg.cmd_window_s

    def _trim(self, t: float) -> None:
        # Always keep one sample: an empty envelope would make every echo look like a driver.
        while len(self._cmds) > 1 and (t - self._cmds[0][0]) > self.cfg.cmd_window_s:
            self._cmds.popleft()

    def _envelope(self, idx: int) -> tuple[float, float]:
        if not self._cmds:
            return 0.0, 0.0
        vals = [c[idx] for c in self._cmds]
        return min(vals), max(vals)

    def steer_deadband_on(self, ffb_wheel: bool | None = None) -> bool:
        """Deadband applies whenever engaged unless we *positively* know there is no FFB device.

        The supervisor never sees input devices (`ffb_wheel` stays None on the retail bus), so
        in practice this is always true — the documented assumption.
        """
        return bool(self.cfg.ffb_assume_wheel) or ffb_wheel is not False

    def update(
        self,
        *,
        engaged: bool,
        steering_input: float | None = None,
        throttle_input: float | None = None,
        brake_input: float | None = None,
        ffb_wheel: bool | None = None,
        now: float | None = None,
    ) -> OverrideVerdict:
        cfg = self.cfg
        t = time.monotonic() if now is None else float(now)
        dt = 0.0 if self._last_t is None else min(1.0, max(0.0, t - self._last_t))
        self._last_t = t
        if not engaged:
            self.reset()
            return self.verdict
        self._trim(t)
        if not self.armed(t):
            self._steer_held = 0.0
            self._steer_sign = 0
            self._pedal_held = 0.0
            self.verdict = OverrideVerdict(armed=False)
            return self.verdict

        # Pedals: hard and sensitive. Only a press *beyond* what GVD asked for counts, so an
        # AEB brake hold echoing back at 1.0 is not mistaken for the driver standing on it.
        thr_lo, thr_hi = self._envelope(2)
        brk_lo, brk_hi = self._envelope(3)
        thr_r = max(0.0, residual(_clamp(throttle_input, 0.0, 1.0), thr_lo, thr_hi)) if throttle_input is not None else 0.0
        brk_r = max(0.0, residual(_clamp(brake_input, 0.0, 1.0), brk_lo, brk_hi)) if brake_input is not None else 0.0
        pedal_channel = "none"
        pedal_r = 0.0
        if brk_r >= cfg.brake_enter and brk_r >= thr_r:
            pedal_channel, pedal_r = "brake", brk_r
        elif thr_r >= cfg.throttle_enter:
            pedal_channel, pedal_r = "throttle", thr_r
        if pedal_channel == "none":
            self._pedal_held = 0.0
        else:
            self._pedal_held += dt
        pedal_active = pedal_channel != "none" and self._pedal_held >= cfg.pedal_hold_s

        # Steer: residual against the command envelope, then deadband, hysteresis and dwell.
        steer_lo, steer_hi = self._envelope(1)
        steer_judged = (
            steering_input is not None
            and max(abs(steer_lo), abs(steer_hi)) <= cfg.steer_cmd_max
        )
        raw = residual(_clamp(steering_input, -1.0, 1.0), steer_lo, steer_hi) if steer_judged else 0.0
        band = cfg.steer_deadband if self.steer_deadband_on(ffb_wheel) else 0.0
        steer_r = deadband(raw, band)
        mag = abs(steer_r)
        sign = 1 if steer_r > 0 else (-1 if steer_r < 0 else 0)
        if mag >= cfg.steer_enter:
            if cfg.steer_sign_flip_resets and self._steer_sign != 0 and sign != self._steer_sign:
                # Chatter alternates side to side; a driver pulling the wheel does not.
                self._steer_held = 0.0
            self._steer_sign = sign
            self._steer_held += dt
        elif mag <= cfg.steer_clear:
            self._steer_held = 0.0
            self._steer_sign = 0
        # Between clear and enter the dwell is frozen — that band is the hysteresis.
        steer_active = mag >= cfg.steer_enter and self._steer_held >= cfg.steer_hold_s

        if pedal_active:
            channel = pedal_channel
        elif steer_active:
            channel = "steer"
        else:
            channel = "none"
        self.verdict = OverrideVerdict(
            active=channel != "none",
            channel=channel,
            steer_residual=round(steer_r, 4),
            steer_raw=round(raw, 4),
            pedal_residual=round(pedal_r, 4),
            steer_held_s=round(self._steer_held, 4),
            deadband_swallowed=steer_r == 0.0 and raw != 0.0,
            steer_judged=steer_judged,
            armed=True,
        )
        return self.verdict
