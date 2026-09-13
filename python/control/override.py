"""Driver-override detection on the steer residual, so force-feedback noise cannot disengage.

M6 judged an override on the absolute wheel angle: `|steering_input| > 0.55` while GVD steered
near-straight. Force-feedback wheels break that — self-aligning torque, spring centering and
kicks over bumps move `steering_input` around whatever GVD commands, so chatter read as a driver
grabbing the wheel and accidentally disengaged GVD.

The signal is now the **residual** `steering_input - cmd.steer`, never an absolute angle, and it
is conditioned before anyone believes it:

    residual -> spike reject -> EMA (lpf_tau_ms) -> soft opposition bias -> hysteresis -> dwell

* **Aligned reference.** `cmd.steer` is the command that was in force when the echo was sampled,
  found by the `applied_seq` the mod acks in `gvd_ego.json`. Comparing a fresh echo against a
  fresh command would read GVD's own steer ramp as a driver, because the echo lags by a tick.
* **Spike reject.** A sample that jumps more than `steer_spike` from the previous one is
  mechanical, not muscular: the filter holds instead of following it. This is what kills the
  large alternating chatter an EMA alone would only attenuate.
* **EMA on the residual only** — not on the inputs and not on the command. For a linear filter
  those are the same thing (`lpf(a) - lpf(b) == lpf(a - b)`), and one state is cheaper to reason
  about than three.
* **Soft opposition bias.** A residual pushing *against* GVD's steer is likelier to be a driver
  than one going along with it, so it counts up to `OPPOSITION_GAIN` more — a nudge on the
  threshold, not a separate rule, and it fades to nothing when GVD is steering straight.
* **Hysteresis + dwell.** Charge above `steer_enter`, discharge below `steer_exit`, and only
  call it an override once the dwell reaches `steer_hold_ms`.

Pedals are deliberately asymmetric and tight: no filter, no dwell, one-sided (only a press
*beyond* what GVD asked for counts) and brake trips lower than throttle. There is no benign
reason for a pedal to move.

Both sides of the retail bus share this: the supervisor here and the GELua that actually holds
the vehicle (`beamng_mod/lua/ge/extensions/gvd/main.lua`). `config/control.yaml` owns the
thresholds; `run_vision.py` mirrors them into `gvd_state.json` as `override_cfg` so the mod
tracks the yaml without parsing it.

Force-feedback behaviour on a real wheel is **UNPROVEN** — no live BeamNG here. The magnitudes
are Michael's research pin, every one is tunable from the yaml, and none of this touches the
`CMD_DEAD_S` dead-man.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

STEER_ENTER = 0.08
STEER_EXIT = 0.04
STEER_HOLD_MS = 200.0
STEER_SPIKE = 0.20
BRAKE_ENTER = 0.06
THROTTLE_ENTER = 0.10
LPF_TAU_MS = 80.0

# Soft opposition bias. Not yaml knobs: the pin fixes the thresholds, and this only leans on
# them. A residual fighting GVD's steer counts up to +25 % more, ramped in by how hard GVD is
# actually steering — there is nothing to oppose when the command is straight.
OPPOSITION_GAIN = 0.25
OPPOSITION_FULL_STEER = 0.25

REASON_STEER = "player_steer"
REASON_BRAKE = "player_brake"
REASON_THROTTLE = "player_throttle"
OVERRIDE_REASONS = (REASON_STEER, REASON_BRAKE, REASON_THROTTLE)
_CHANNEL_REASON = {"steer": REASON_STEER, "brake": REASON_BRAKE, "throttle": REASON_THROTTLE}

# Judgement waits this many filter time constants after taking the car: the EMA has to settle,
# and until GVD has been commanding for a while the echo says nothing about our commands. It
# also keeps a player already resting on the brake as they press Alt+A from overriding
# themselves on the spot. Derived from lpf_tau_ms, not a separate knob.
_WARMUP_TAUS = 3.0
_CMD_RING = 256


@dataclass(frozen=True)
class OverrideConfig:
    steer_enter: float = STEER_ENTER
    steer_exit: float = STEER_EXIT
    steer_hold_ms: float = STEER_HOLD_MS
    steer_spike: float = STEER_SPIKE
    brake_enter: float = BRAKE_ENTER
    throttle_enter: float = THROTTLE_ENTER
    lpf_tau_ms: float = LPF_TAU_MS

    @property
    def steer_hold_s(self) -> float:
        return self.steer_hold_ms / 1000.0

    @property
    def lpf_tau_s(self) -> float:
        return self.lpf_tau_ms / 1000.0

    @property
    def warmup_s(self) -> float:
        return _WARMUP_TAUS * self.lpf_tau_s


@dataclass
class Command:
    """One command GVD actually applied to the car."""

    seq: int = -1
    t: float = 0.0
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0


@dataclass
class OverrideVerdict:
    """One tick of judgement. `active` is what disengages; the rest is telemetry."""

    active: bool = False
    channel: str = "none"        # none | steer | brake | throttle
    steer_raw: float = 0.0       # steering_input - aligned cmd.steer
    steer_filt: float = 0.0      # after the EMA
    steer_eff: float = 0.0       # after the opposition bias — what the thresholds see
    pedal_residual: float = 0.0
    steer_held_ms: float = 0.0
    spike: bool = False          # this sample jumped too far to be a driver
    opposition: float = 0.0      # 0..1, how much the residual fights GVD's steer
    ref_seq: int = -1            # command seq the residual was measured against
    armed: bool = True           # false during the warm-up after engage

    @property
    def reason(self) -> str:
        return _CHANNEL_REASON.get(self.channel, "none")


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

    steer_enter = num("steer_enter", STEER_ENTER, 0.005, 1.0)
    return OverrideConfig(
        steer_enter=steer_enter,
        # Exit at or above enter would leave the dwell unable to discharge.
        steer_exit=min(num("steer_exit", STEER_EXIT, 0.0, 1.0), steer_enter * 0.95),
        steer_hold_ms=num("steer_hold_ms", STEER_HOLD_MS, 0.0, 2000.0),
        steer_spike=num("steer_spike", STEER_SPIKE, 0.01, 2.0),
        brake_enter=num("brake_enter", BRAKE_ENTER, 0.005, 1.0),
        throttle_enter=num("throttle_enter", THROTTLE_ENTER, 0.005, 1.0),
        lpf_tau_ms=num("lpf_tau_ms", LPF_TAU_MS, 0.0, 1000.0),
    )


def config_mirror(cfg: OverrideConfig) -> dict[str, Any]:
    """Flat `override_cfg` for gvd_state.json so GELua runs the same thresholds as the yaml."""
    return {
        "steer_enter": round(cfg.steer_enter, 4),
        "steer_exit": round(cfg.steer_exit, 4),
        "steer_hold_ms": round(cfg.steer_hold_ms, 3),
        "steer_spike": round(cfg.steer_spike, 4),
        "brake_enter": round(cfg.brake_enter, 4),
        "throttle_enter": round(cfg.throttle_enter, 4),
        "lpf_tau_ms": round(cfg.lpf_tau_ms, 3),
    }


def _clamp(v: Any, lo: float, hi: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if f != f:
        return 0.0
    return min(hi, max(lo, f))


def ema_alpha(dt: float, tau_s: float) -> float:
    """Time-correct first-order weight, so the filter does not depend on the loop rate."""
    if tau_s <= 0.0 or dt <= 0.0:
        return 1.0
    return 1.0 - math.exp(-dt / tau_s)


def opposition(ref_steer: float, residual: float) -> float:
    """0..1: how much `residual` fights a steer command of `ref_steer`.

    Zero when the two agree, and zero when GVD is steering straight — a residual cannot oppose
    a command that is not there.
    """
    if ref_steer * residual >= 0.0:
        return 0.0
    return min(1.0, abs(ref_steer) / OPPOSITION_FULL_STEER)


class OverrideDetector:
    """Stateful per-loop judgement. Feed it every applied command and every electrics echo.

    `note_command` records what was actually applied — not the pre-gate intent, or a gate hold
    riding along as `brake=1` would echo back and read as the driver standing on the pedal.
    `update` then measures the echo against the command the mod says it applied (`applied_seq`),
    so call it before the actuator and note the applied command after.

    Disengage is sticky at the caller (gvd_engage.json stays false until Alt+A) — here
    `update(engaged=False)` just clears the state so re-engaging starts clean.
    """

    def __init__(self, cfg: OverrideConfig | None = None) -> None:
        self.cfg = cfg or OverrideConfig()
        self.verdict = OverrideVerdict()
        self._cmds: deque[Command] = deque(maxlen=_CMD_RING)
        self._filt = 0.0
        # Compared against the next raw residual so the first sample can be a spike too
        # (a kick off rest). None would let that kick into the filter and freeze it there.
        self._last_raw: float = 0.0
        self._steer_held = 0.0
        self._steer_sign = 0
        self._last_t: float | None = None
        self._armed_at: float | None = None

    def note_command(self, *, seq: int, steer: float, throttle: float, brake: float, now: float | None = None) -> None:
        t = time.monotonic() if now is None else float(now)
        if self._armed_at is None:
            self._armed_at = t
        self._cmds.append(
            Command(
                seq=int(seq),
                t=t,
                steer=_clamp(steer, -1.0, 1.0),
                throttle=_clamp(throttle, 0.0, 1.0),
                brake=_clamp(brake, 0.0, 1.0),
            )
        )

    def reset(self) -> None:
        """Forget the filter, the dwell and the command history (re-engage starts clean)."""
        self._cmds.clear()
        self._filt = 0.0
        self._last_raw = 0.0
        self._steer_held = 0.0
        self._steer_sign = 0
        self._armed_at = None
        self.verdict = OverrideVerdict()

    def armed(self, now: float) -> bool:
        return self._armed_at is not None and (now - self._armed_at) >= self.cfg.warmup_s

    def reference(self, applied_seq: int | None = None) -> Command:
        """The command in force when the echo was sampled.

        `applied_seq` is the mod's own ack, so on the retail bus this is exact. Without one
        (BeamNGpy polls electrics directly) fall back to the previous command, which is the same
        one-tick lag by a less certain route.
        """
        if not self._cmds:
            return Command()
        if applied_seq is not None:
            want = int(applied_seq)
            for cmd in reversed(self._cmds):
                if cmd.seq == want:
                    return cmd
            return self._cmds[-1]
        return self._cmds[-2] if len(self._cmds) >= 2 else self._cmds[-1]

    def update(
        self,
        *,
        engaged: bool,
        steering_input: float | None = None,
        throttle_input: float | None = None,
        brake_input: float | None = None,
        applied_seq: int | None = None,
        now: float | None = None,
        player_device: bool = False,
    ) -> OverrideVerdict:
        cfg = self.cfg
        t = time.monotonic() if now is None else float(now)
        dt = 0.0 if self._last_t is None else min(1.0, max(0.0, t - self._last_t))
        self._last_t = t
        if not engaged:
            self.reset()
            return self.verdict
        if not self.armed(t):
            self._steer_held = 0.0
            self._steer_sign = 0
            self.verdict = OverrideVerdict(armed=False)
            return self.verdict

        ref = self.reference(applied_seq)
        # Retail Direct Drive lock: electrics.steering_input is GVD's own command, so residual
        # vs cmd is 0. The physical wheel/pedals live in lastInputs (player_device=True) as
        # an absolute axis — centered wheel is 0, a real pull is not. Opposition still uses
        # GVD's steer so a pull against the command counts a little more.
        ref_steer = 0.0 if player_device else ref.steer
        ref_thr = 0.0 if player_device else ref.throttle
        ref_brk = 0.0 if player_device else ref.brake

        # Pedals: asymmetric and tight. Only a press beyond what GVD asked for counts, so an AEB
        # brake hold echoing back at 1.0 is not the driver standing on it. Brake trips lower than
        # throttle and wins a tie, because that is the reason a player most needs to be told.
        thr_r = max(0.0, _clamp(throttle_input, 0.0, 1.0) - ref_thr) if throttle_input is not None else 0.0
        brk_r = max(0.0, _clamp(brake_input, 0.0, 1.0) - ref_brk) if brake_input is not None else 0.0
        pedal_channel = "none"
        pedal_r = 0.0
        if brk_r >= cfg.brake_enter:
            pedal_channel, pedal_r = "brake", brk_r
        elif thr_r >= cfg.throttle_enter:
            pedal_channel, pedal_r = "throttle", thr_r

        # Steer: residual against the aligned command, spike-rejected, then filtered.
        raw = _clamp(steering_input, -1.0, 1.0) - ref_steer if steering_input is not None else 0.0
        # Sample-to-sample jump past steer_spike is mechanical: skip the EMA so a kick does
        # not drag the filter with it. last_raw still advances, so a hold after the kick is
        # a zero jump next tick and the filter is allowed to follow.
        spike = abs(raw - self._last_raw) > cfg.steer_spike
        self._last_raw = raw
        if not spike:
            self._filt += ema_alpha(dt, cfg.lpf_tau_s) * (raw - self._filt)
        opp = opposition(ref.steer, self._filt)
        eff = self._filt * (1.0 + OPPOSITION_GAIN * opp)

        mag = abs(eff)
        sign = 1 if eff > 0 else (-1 if eff < 0 else 0)
        if sign != 0 and self._steer_sign != 0 and sign != self._steer_sign:
            # "How long has the player been pushing one way" — a side change restarts the clock.
            self._steer_held = 0.0
        if mag >= cfg.steer_enter:
            self._steer_sign = sign
            self._steer_held += dt
        elif mag <= cfg.steer_exit:
            self._steer_held = 0.0
            self._steer_sign = 0
        # Between exit and enter the dwell is frozen — that band is the hysteresis.
        steer_active = mag >= cfg.steer_enter and self._steer_held >= cfg.steer_hold_s

        if pedal_channel != "none":
            channel = pedal_channel
        elif steer_active:
            channel = "steer"
        else:
            channel = "none"
        self.verdict = OverrideVerdict(
            active=channel != "none",
            channel=channel,
            steer_raw=round(raw, 4),
            steer_filt=round(self._filt, 4),
            steer_eff=round(eff, 4),
            pedal_residual=round(pedal_r, 4),
            steer_held_ms=round(self._steer_held * 1000.0, 1),
            spike=bool(spike),
            opposition=round(opp, 4),
            ref_seq=int(ref.seq),
            armed=True,
        )
        return self.verdict
