"""Sim-only actuation: BeamNGpy vehicle.control (Tech) or gvd_cmd.json → GELua (retail).

Safety gates (must all pass to drive):
  engaged + heartbeat fresh + (not path_debug_preview OR allow_preview_drive)

Retail bus (M6): Python writes Documents/GVD/gvd_cmd.json every tick; the mod's
gvd_main.applyCmdJson feeds steer/throttle/brake to the player vehicle as a secondary
Direct Drive wheel + pedals (`input.event` FILTER_DIRECT + source `gvd` + `setAllowedInputSource`
on steering/throttle/brake/parkingbrake/clutch) so a connected keyboard/pad/wheel/pedal cluster
cannot overwrite the software. Echoes wheelspeed /
inputs / applied seq back through gvd_ego.json. `cmd_applied` is only claimed once that
ack is fresh. No DLL / hooks / process inject. Live BeamNG still UNPROVEN on Linux.

Tech `vehicle.control` is a `Control` message. BeamNG applies a steering field as
pad filter 1 on source `local`, the same slot as the physical wheel. That write is
blocked from the hydros once the gvd hold is up, but it still overwrites
``lastInputs.local.steering``, so a held wheel grab is replaced by the command and
the override echo stays at the command. While the hold is the steer path, Tech
omits ``steering`` from ``vehicle.control`` and queues a steering-only Direct Drive
hold (source `gvd`, local blocked). Throttle, brake, parking brake, and clutch stay
on ``vehicle.control``. The rising edge publishes ``tech_steer_hold`` before that
queue, and only after the write lands. A failed write or a failed queue leaves
the command unapplied. Disengage, shutdown, and the mod's watcher clear the
whitelist and replay ``lastInputs.local.steering`` so a wheel already held drives
again. The replay uses a binding angle and lock type when they sit beside that
value. Otherwise it is Direct Drive filter 2, angle 900, lock type 0.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from python.runtime.paths import bus_identity
from python.runtime.state_io import atomic_write_json, gvd_docs_dir

# Link / ribbon stale. A single grab hitch is well under a second and must
# not flip the in-game link to stale (that flash is the Engage HOLD). The
# command dead-man stays at 0.35 s in the mod; this is only the link.
HEARTBEAT_STALE_S = 1.5
AEB_BRAKE_TTC = 1.2
EGO_FRESH_S = 1.0          # gvd_ego.json older than this → treat Lua feedback as gone
CMD_ACK_SLACK = 5          # Lua acks lag a few seqs (20 Hz apply, 10 Hz echo, 15 Hz loop)
# Lua heartbeats gvd_engage.json while its in-memory latch is on (os.time is 1 s).
# A leftover engaged:true from a crashed session must not start Tech vehicle.control.
ENGAGE_FRESH_S = 2.5

# Drive uses arcade. Set once at engage. A gear field on vehicle.control is
# shiftToGearIndex, and arcade's shiftToGearIndex enters realistic, so control
# omits gear. Echo letters D/S/M/L and indexes >= 1 are already forward.
TECH_SHIFT_MODE = "arcade"
# Player arrows/pedals expect arcade. Restored only on the Disengage handoff.
TECH_PLAYER_SHIFT_MODE = "arcade"
TECH_HOLD_BRAKE = 0.99
# Electrics neutral / first forward index. Read from the echo. Not sent.
TECH_HOLD_GEAR = 0
TECH_DRIVE_GEAR = 1
# Let a resting handbrake finish releasing before asking again.
TECH_DRIVE_ARM_S = 0.55
# Vehicle VM. Parking brake and clutch are zeroed on source gvd.
# This chunk must not call setGearboxMode or shiftToGearIndex. Arcade's
# shiftToGearIndex is switchToRealisticBehavior, and setGearboxMode always
# runs gearboxBehaviorChanged. Repeating either one flips arcade ↔ realistic
# and the car brakes, then the throttle catches.
TECH_DRIVE_SHIFT_LUA = (
    "pcall(function() "
    "if input and input.event then pcall(function() "
    "input.event('parkingbrake',0,2,0,0,nil,'gvd'); "
    "input.event('clutch',0,2,0,0,nil,'gvd') end) end "
    "end)"
)
# Steering only. Pedals stay on vehicle.control (source local). Once any source is
# listed, BeamNG allows only sources set true, so local and any other device are out
# and the gvd event is the steer that sticks. Release clears that whitelist, then
# replays the stored local wheel angle: a held wheel is onChange and will not
# write again until it moves. Must match the mod's M.techSteerRelease.
# lastInputs normally stores only the axis value, so a non-direct player binding
# is not reproduced. When the stored steering is a table, value / angle / lockType
# (or lock) are used. When it is a number, sibling steeringAngle or angle and
# steeringLockType or lockType are used. Otherwise the replay is Direct Drive
# filter 2, angle 900, lockType 0.
TECH_STEER_RELEASE_LUA = (
    "if input and input.setAllowedInputSource then "
    "input.setAllowedInputSource('steering',nil);"
    "end;"
    "local s,ang,lk=0,900,0;"
    "if input and input.lastInputs and input.lastInputs['local'] then "
    "local slot=input.lastInputs['local'];"
    "local raw=slot.steering;"
    "if type(raw)=='table' then "
    "s=tonumber(raw.value or raw[1]) or 0;"
    "ang=tonumber(raw.angle) or ang;"
    "lk=tonumber(raw.lockType or raw.lock) or lk;"
    "else "
    "s=tonumber(raw) or 0;"
    "ang=tonumber(slot.steeringAngle or slot.angle) or ang;"
    "lk=tonumber(slot.steeringLockType or slot.lockType) or lk;"
    "end;"
    "end;"
    "if s~=s or s==math.huge or s==-math.huge then s=0 end;"
    "if ang~=ang or ang==math.huge or ang==-math.huge then ang=900 end;"
    "if lk~=lk or lk==math.huge or lk==-math.huge then lk=0 end;"
    "input.event('steering',s,2,ang,lk,nil,'local')"
)


def tech_steer_hold_lua(steer: float) -> str:
    """Vehicle Lua: controller steer wins, the physical wheel does not move the car.

    FILTER_DIRECT (2), angle 900, lockType 0, source ``gvd``. Same Direct Drive
    steering event retail uses. ``local`` is blocked so the wheel's later
    ``input.event`` is recorded in ``lastInputs`` and does not change hydros.
    """
    try:
        s = float(steer)
    except (TypeError, ValueError):
        s = 0.0
    if s != s:
        s = 0.0
    s = max(-1.0, min(1.0, s))
    return (
        "if input and input.setAllowedInputSource then "
        "input.setAllowedInputSource('steering','gvd',true);"
        "input.setAllowedInputSource('steering','local',false);"
        "end;"
        f"input.event('steering',{s:.4f},2,900,0,nil,'gvd')"
    )


def flush_tech_steer_hold(held: bool) -> bool:
    """Publish ``tech_steer_hold`` before ``queue_lua_command`` on the rising edge.

    True only when the file write landed. A failed write must not be followed
    by the hold queue. The steady tick writes the full state after the queue
    returns. A kill inside that call never reaches the steady write, so this
    patch is what arms the mod's watcher.
    """
    from python.runtime.state_io import read_state, write_state

    try:
        prev = read_state()
    except Exception:
        prev = None
    if not isinstance(prev, dict):
        prev = {}
    prev["tech_steer_hold"] = bool(held)
    prev["actuator"] = "beamngpy"
    try:
        written = write_state(prev)
    except Exception as e:
        print(
            f"[GVD] tech steer hold flag publish failed: {type(e).__name__}",
            flush=True,
        )
        return False
    if written is None:
        print("[GVD] tech steer hold flag publish failed: write", flush=True)
        return False
    return True


@dataclass
class DriveCommand:
    steer: float = 0.0      # [-1, 1]
    throttle: float = 0.0  # [0, 1]
    brake: float = 0.0     # [0, 1]
    seq: int = 0
    applied: bool = False
    reason: str = "ok"
    blinker: str = "off"  # off | left | right, from the predicted path


def cmd_path() -> Path:
    return gvd_docs_dir() / "gvd_cmd.json"


def engage_path() -> Path:
    return gvd_docs_dir() / "gvd_engage.json"


def bot_engage_path() -> Path:
    """Ship-bot latch. Same folder as ``gvd_engage.json``. No window focus."""
    return gvd_docs_dir() / "gvd_bot_engage.json"


def ego_path() -> Path:
    return gvd_docs_dir() / "gvd_ego.json"


@dataclass
class EgoFeedback:
    """Vehicle echo written by gvd_main (retail): electrics + which cmd seq Lua applied."""

    speed_mps: float | None = None
    steering_input: float | None = None
    throttle_input: float | None = None
    brake_input: float | None = None
    applied_seq: int = -1
    applying: bool = False
    age_s: float = 0.0
    gx: float | None = None
    gy: float | None = None
    gz: float | None = None
    yaw_rate: float | None = None
    pos: tuple[float, float, float] | None = None
    dir: tuple[float, float, float] | None = None
    player_device: bool = False
    player_steering: float | None = None
    player_throttle: float | None = None
    player_brake: float | None = None

    @property
    def fresh(self) -> bool:
        return self.age_s <= EGO_FRESH_S


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def read_ego_feedback(now: float | None = None) -> EgoFeedback | None:
    """Parse gvd_ego.json (Lua writes it non-atomically; a torn read just returns None)."""
    p = ego_path()
    try:
        if not p.is_file():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        mtime = p.stat().st_mtime
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    t = now if now is not None else time.time()
    try:
        seq = int(data.get("applied_seq", -1))
    except (TypeError, ValueError):
        seq = -1
    pos = None
    raw_pos = data.get("pos")
    if isinstance(raw_pos, dict):
        try:
            pos = (float(raw_pos["x"]), float(raw_pos["y"]), float(raw_pos["z"]))
        except (KeyError, TypeError, ValueError):
            pos = None
    direction = None
    raw_dir = data.get("dir")
    if isinstance(raw_dir, dict):
        try:
            direction = (float(raw_dir["x"]), float(raw_dir["y"]), float(raw_dir["z"]))
        except (KeyError, TypeError, ValueError):
            direction = None
    return EgoFeedback(
        speed_mps=_num(data.get("speed_mps")),
        steering_input=_num(data.get("steering_input")),
        throttle_input=_num(data.get("throttle_input")),
        brake_input=_num(data.get("brake_input")),
        applied_seq=seq,
        applying=bool(data.get("applying", False)),
        age_s=max(0.0, t - mtime),
        gx=_num(data.get("gx")),
        gy=_num(data.get("gy")),
        gz=_num(data.get("gz")),
        yaw_rate=_num(data.get("yaw_rate")),
        pos=pos,
        dir=direction,
        player_device=bool(data.get("player_device", False)),
        player_steering=_num(data.get("player_steering")),
        player_throttle=_num(data.get("player_throttle")),
        player_brake=_num(data.get("player_brake")),
    )


def read_engage_flag(default: bool = False) -> bool:
    """Mirror Lua Alt+G. ``engaged:true`` counts only while ``mtime`` is fresh.

    Lua rewrites the stamp while the in-game latch is on. A file left true after a
    crash is not a new human engage — Tech must not call ``vehicle.control`` from it.
    ``engaged:false`` is always off. A missing file returns *default*.
    """
    p = engage_path()
    if not p.is_file():
        return default
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default
    if not isinstance(data, dict) or not bool(data.get("engaged", False)):
        return False
    try:
        age = time.time() - float(data["mtime"])
    except (KeyError, TypeError, ValueError):
        return False
    # os.time() is whole seconds, so a heartbeat can look ~1 s old. A stamp far
    # in the future is not a live latch.
    if age < -1.0:
        return False
    return age <= ENGAGE_FRESH_S


def write_engage_flag(engaged: bool, disengage_reason: str | None = None) -> None:
    """Write gvd_engage.json. On engage clear reason to none; on disengage pass reason when known."""
    payload: dict = {"engaged": bool(engaged), "mtime": time.time()}
    if engaged or disengage_reason is None:
        payload["disengage_reason"] = "none"
    else:
        payload["disengage_reason"] = str(disengage_reason)
    atomic_write_json(engage_path(), payload, indent=None)


def _read_bot_engage() -> dict | None:
    p = bot_engage_path()
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def bot_engage_fresh(now: float | None = None) -> bool:
    """True when ``gvd_bot_engage.json`` is engaged and ``mtime`` is fresh.

    Same window as ``read_engage_flag`` (``ENGAGE_FRESH_S``). The grab calls
    ``poll_vehicle`` before ``note_engaged``, so the in-memory latch is still
    false on the first bot-engage tick. A fresh file is Engage for that poll.
    A stale leftover is not.
    """
    data = _read_bot_engage()
    if not data or not bool(data.get("engaged", False)):
        return False
    try:
        age = (time.time() if now is None else float(now)) - float(data["mtime"])
    except (KeyError, TypeError, ValueError):
        return False
    if age < -1.0:
        return False
    return age <= ENGAGE_FRESH_S


def write_bot_engage(engaged: bool, *, mtime: float | None = None) -> float:
    """One write engages or clears. ``mtime`` must be fresh for a new engage."""
    stamp = time.time() if mtime is None else float(mtime)
    atomic_write_json(bot_engage_path(), {"engaged": bool(engaged), "mtime": stamp}, indent=None)
    return float(json.loads(json.dumps(stamp)))


class BotEngage:
    """Latch for ``gvd_bot_engage.json``.

    A new ``engaged:true`` counts only while ``mtime`` is inside the same
    window as Alt+G (``ENGAGE_FRESH_S``). After that the latch stays on
    until ``engaged:false``, ``clear()`` (driver override, a veto that
    drops Engage, shutdown), or a newer command. A leftover true from a
    crashed session does not start the car.
    """

    def __init__(self) -> None:
        self.latched = False
        self._seen: float | None = None

    def poll(self, now: float | None = None) -> bool:
        now_s = time.time() if now is None else float(now)
        data = _read_bot_engage()
        if not data or "mtime" not in data:
            return self.latched
        try:
            mtime = float(data["mtime"])
        except (TypeError, ValueError):
            return self.latched
        if self._seen is not None and mtime == self._seen:
            return self.latched
        self._seen = mtime
        if not bool(data.get("engaged", False)):
            self.latched = False
            return False
        age = now_s - mtime
        if -1.0 <= age <= ENGAGE_FRESH_S:
            self.latched = True
        return self.latched

    def clear(self) -> None:
        """Sticky off. The old true is consumed so it cannot re-engage."""
        self.latched = False
        write_bot_engage(False)
        data = _read_bot_engage()
        if data and data.get("engaged") is False:
            try:
                self._seen = float(data["mtime"])
                return
            except (TypeError, ValueError):
                pass
        self._seen = None


def heartbeat_fresh(heartbeat_mtime: float | None, now: float | None = None, stale_s: float = HEARTBEAT_STALE_S) -> bool:
    if heartbeat_mtime is None:
        return False
    t = now if now is not None else time.time()
    return (t - float(heartbeat_mtime)) <= stale_s


def may_drive(
    *,
    engaged: bool,
    heartbeat_ok: bool,
    path_debug_preview: bool,
    allow_preview_drive: bool,
    policy: str = "modular",
) -> tuple[bool, str]:
    if not engaged:
        return False, "not_engaged"
    if not heartbeat_ok:
        return False, "heartbeat_stale"
    if policy in ("stub", "map-ai") and not allow_preview_drive:
        # stub policy never drives unless explicitly allowed
        if policy == "stub":
            return False, "stub_policy"
    if path_debug_preview and not allow_preview_drive:
        return False, "preview_blocked"
    return True, "ok"


def stop_command(seq: int = 0, reason: str = "stop") -> DriveCommand:
    return DriveCommand(steer=0.0, throttle=0.0, brake=1.0, seq=seq, applied=False, reason=reason)


def release_command(seq: int = 0, reason: str = "not_engaged") -> DriveCommand:
    """Disengage handoff: pedals at rest. Engaged holds still use stop_command (brake=1)."""
    return DriveCommand(steer=0.0, throttle=0.0, brake=0.0, seq=seq, applied=False, reason=reason)


def _clip01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _clip_steer(v: float) -> float:
    return float(max(-1.0, min(1.0, v)))


def gear_is_forward(gear: Any) -> bool:
    """True when the electrics gear string/index is already a forward range.

    Automatics report the shifter letter (``D`` / ``S`` / ``M`` / ``L``). Manuals
    report a gear index (``1`` and up). ``N``, ``P``, ``R``, ``0``, and a missing
    echo are not forward — those still need a drive arm.
    """
    if gear is None or isinstance(gear, bool):
        return False
    if isinstance(gear, (int, float)):
        try:
            g = int(gear)
        except (TypeError, ValueError):
            return False
        if g == TECH_HOLD_GEAR:
            return False
        return g >= TECH_DRIVE_GEAR
    text = str(gear).strip().upper()
    if text in {"D", "S", "M", "L"}:
        return True
    if len(text) >= 2 and text[0] == "M" and text[1:].isdigit():
        return int(text[1:]) >= TECH_DRIVE_GEAR
    if text.isdigit():
        return int(text) >= TECH_DRIVE_GEAR
    return False


def _gear_value(el: Any) -> Any:
    """Gear field from a dict or a BeamNGpy electrics object (``.data`` / ``.gear``)."""
    if el is None:
        return None
    if isinstance(el, dict):
        return el.get("gear")
    data = getattr(el, "data", None)
    if isinstance(data, dict):
        return data.get("gear")
    gear = getattr(el, "gear", None)
    if gear is not None and not callable(gear):
        return gear
    return None


def _unexpected_kw(err: BaseException) -> str | None:
    """Pull the name out of ``got an unexpected keyword argument 'clutch'``."""
    msg = str(err)
    marker = "unexpected keyword argument "
    i = msg.find(marker)
    if i < 0:
        return None
    rest = msg[i + len(marker) :].strip()
    if len(rest) >= 2 and rest[0] in "'\"":
        end = rest.find(rest[0], 1)
        if end > 1:
            return rest[1:end]
    return None


def tech_control_kwargs(
    steer: float,
    throttle: float,
    brake: float,
    *,
    release: bool = False,
    speed_mps: float | None = None,
) -> dict[str, Any]:
    """Arcade forward drive. A full hold does not send the reverse pedal.

    Shift mode stays the arcade set at engage. The kwargs omit gear. BeamNG
    routes a gear field through shiftToGearIndex, and in arcade that function
    is switchToRealisticBehavior, so every control tick was leaving arcade.
    Throttle>0 releases the parking brake and the clutch. A hold, stop, or
    AEB (throttle ~0 and brake >= 0.99) is parkingbrake=1 and service brake 0.
    Arcade treats that held service brake, with no throttle, as reverse. The
    hold ignores speed and always sets the parking brake.
    """
    del speed_mps  # moving or stopped, the reverse pedal stays off
    if release:
        return {
            "steering": 0.0,
            "throttle": 0.0,
            "brake": 0.0,
            "parkingbrake": 0.0,
        }
    steer_v = _clip_steer(steer)
    throttle_v = _clip01(throttle)
    brake_v = _clip01(brake)
    if throttle_v > 1e-6:
        return {
            "steering": steer_v,
            "throttle": throttle_v,
            "brake": brake_v,
            "parkingbrake": 0.0,
            "clutch": 0.0,
        }
    if brake_v >= TECH_HOLD_BRAKE:
        return {
            "steering": steer_v,
            "throttle": 0.0,
            "brake": 0.0,
            "parkingbrake": 1.0,
        }
    return {
        "steering": steer_v,
        "throttle": 0.0,
        "brake": brake_v,
        "parkingbrake": 0.0,
    }


def is_reverse_control(kwargs: dict[str, Any]) -> bool:
    """True when control kwargs select reverse (gear=-1)."""
    gear = kwargs.get("gear")
    try:
        return gear is not None and int(gear) < 0
    except (TypeError, ValueError):
        return False


def is_arcade_reverse_hold(kwargs: dict[str, Any]) -> bool:
    """True when arcade would select reverse from these kwargs.

    A held service brake with no throttle is reverse throttle. Gear 0 does
    not make that safe, and neither does the parking brake. Explicit gear < 0
    is reverse. Parking brake with the service brake released is not.
    """
    if is_reverse_control(kwargs):
        return True
    throttle = float(kwargs.get("throttle") or 0.0)
    brake = float(kwargs.get("brake") or 0.0)
    return throttle <= 1e-6 and brake >= TECH_HOLD_BRAKE


def plan_command(
    *,
    path_ego: list[dict[str, float]] | None,
    planner: dict[str, Any] | None,
    ego_speed_mps: float,
    seq: int,
) -> DriveCommand:
    """Map corridor + speed plan → DriveCommand pedals.

    AEB / full stop still return brake=1, throttle=0 (retail JSON + override
    semantics). Tech remaps that pair in tech_control_kwargs: arcade stays the
    shift mode, and the hold is parkingbrake=1 with the service brake released
    so arcade does not select reverse. The Tech control message has no gear field.
    """
    planner = planner or {}
    aeb = str(planner.get("aeb") or "off")
    target_v = float(planner.get("target_v") or 0.0)
    ttc = planner.get("ttc_lead")

    steer = 0.0
    if path_ego and len(path_ego) >= 3:
        steer = path_steer_from_ego(path_ego)

    blinker = str(planner.get("blinker") or "off")
    if blinker not in ("left", "right"):
        blinker = "off"

    throttle = 0.0
    brake = 0.0
    if aeb == "brake" or (ttc is not None and float(ttc) < AEB_BRAKE_TTC):
        # Keep the command semantic (brake=1). Arcade would treat a held service
        # brake as reverse; tech_control_kwargs sends parking brake instead.
        throttle = 0.0
        brake = 1.0
    elif aeb == "warn":
        throttle = 0.0
        brake = 0.35
    else:
        err = target_v - float(ego_speed_mps)
        if err > 0.5:
            throttle = max(0.0, min(0.55, 0.08 * err))
        elif err < -1.0:
            brake = max(0.0, min(0.6, -0.05 * err))
        else:
            throttle = 0.12 if target_v > 1.0 else 0.0

    return DriveCommand(
        steer=steer, throttle=throttle, brake=brake, seq=seq, reason="plan", blinker=blinker,
    )


def safe_command(
    *,
    engaged: bool,
    heartbeat_ok: bool,
    path_debug_preview: bool,
    allow_preview_drive: bool,
    path_ego: list[dict[str, float]] | None = None,
    planner: dict[str, Any] | None = None,
    ego_speed_mps: float = 0.0,
    seq: int = 0,
    policy: str = "modular",
) -> DriveCommand:
    ok, reason = may_drive(
        engaged=engaged,
        heartbeat_ok=heartbeat_ok,
        path_debug_preview=path_debug_preview,
        allow_preview_drive=allow_preview_drive,
        policy=policy,
    )
    if not ok:
        return stop_command(seq=seq, reason=reason)
    cmd = plan_command(path_ego=path_ego, planner=planner, ego_speed_mps=ego_speed_mps, seq=seq)
    cmd.reason = "ok"
    return cmd


@dataclass
class DriverInputs:
    """Electrics echo of what the car is actually doing (Tech path; mirrors EgoFeedback)."""

    speed_mps: float | None = None
    steering_input: float | None = None
    throttle_input: float | None = None
    brake_input: float | None = None


# Soft Esc grab calls TechSession.poll, then read_electrics, on one tick.
# vehicle.sensors.poll is one GE roundtrip. Reuse that map once. The clock
# restarts when poll() returns so PollGPSGE inside the same poll cannot
# expire the snapshot before electrics are read. A miss still polls; Engage
# hold reads speed on that path.
# Soft Esc (latch false, gvd_engage.json not live, and gvd_bot_engage.json
# not fresh) may skip the GE poll for 200 ms and republish the last-good map
# here so this read does not open a second sensors.poll. TechSession.poll
# reads both files before that hold, because note_engaged runs after poll_vehicle.
SENSOR_POLL_REUSE_S = 0.05

# Engage latch updated by note_engaged after poll_vehicle. Default false:
# Soft Esc coalesces without a run_vision edit. True: one sensors.poll per
# grab (Tip #1). The rising edge does not wait for this latch; poll() also
# refuses the hold when read_engage_flag() is true.
_soft_esc_engaged = False


def note_soft_esc_engaged(engaged: bool) -> None:
    """Latch Engage so a grab with this bit set polls every tick.

    Callers that flip this must restore it. ``TechSession.poll`` does not
    wait for the latch on the rising edge; it also reads ``read_engage_flag``
    and a fresh ``gvd_bot_engage.json``.
    """
    global _soft_esc_engaged
    _soft_esc_engaged = bool(engaged)


def soft_esc_sensors_every_tick() -> bool:
    """True when the engage latch forbids the Soft Esc sensors.poll coalesce."""
    return bool(_soft_esc_engaged)


# Max rewrite rate for Soft Esc gvd_state.json and the Tech release cmd.
# Matches Lua gvd_main.pollEvery (0.10): at most one rewrite per window.
# Heartbeat age can still pass HEARTBEAT_STALE_S. When the Soft Esc loop
# is already slower than this period, write age tracks the loop. Engage
# does not use this cap.
SOFT_ESC_FILE_PERIOD_S = 0.10

_soft_esc_state_mono: float | None = None
_soft_esc_state_skips = 0


def reset_soft_esc_state_writes() -> None:
    """Clear the Soft Esc gvd_state.json window. Callers that flip it restore it."""
    global _soft_esc_state_mono, _soft_esc_state_skips
    _soft_esc_state_mono = None
    _soft_esc_state_skips = 0


def soft_esc_state_write_skips() -> int:
    """Soft Esc ticks that did not rewrite gvd_state.json."""
    return int(_soft_esc_state_skips)


def soft_esc_state_write_due(
    engaged: bool, *, rising: bool = False, now: float | None = None
) -> bool:
    """True when the supervisor should call write_state on this tick.

    Soft Esc (engaged false, not a rising edge): at most one True per
    SOFT_ESC_FILE_PERIOD_S, measured from the last ``soft_esc_state_write_mark``.
    A False increments soft_esc_state_write_skips. Engage is True every tick.
    A rising edge flushes on that same tick when the Soft Esc window has not
    elapsed. ``now`` is monotonic seconds for tests; the supervisor omits it.
    """
    global _soft_esc_state_skips
    t = time.monotonic() if now is None else float(now)
    if engaged or rising:
        return True
    last = _soft_esc_state_mono
    if last is None or (t - last) >= SOFT_ESC_FILE_PERIOD_S:
        return True
    _soft_esc_state_skips += 1
    return False


def soft_esc_state_write_mark(now: float | None = None) -> None:
    """Stamp the Soft Esc window after write_state returns.

    The cap is on the rewrite, not the earlier due-check, so recorder time
    between the check and the write cannot bunch two files inside 100 ms.
    """
    global _soft_esc_state_mono
    _soft_esc_state_mono = time.monotonic() if now is None else float(now)


class _SensorSnap:
    """One vehicle.sensors.poll payload, consumed by the next same-tick reader."""

    __slots__ = ("mono", "data", "used")

    def __init__(self, mono: float, data: dict[str, Any]) -> None:
        self.mono = mono
        self.data = data
        self.used = False


_sensor_snaps: dict[int, _SensorSnap] = {}


def publish_vehicle_sensor_snap(vehicle: Any, data: dict[str, Any]) -> None:
    """Remember this tick's vehicle.sensors.poll map for one follow-up read."""
    if vehicle is None:
        return
    payload = data if isinstance(data, dict) else {}
    _sensor_snaps[id(vehicle)] = _SensorSnap(time.monotonic(), payload)


def touch_vehicle_sensor_snap(vehicle: Any) -> None:
    """Restart the reuse window when TechSession.poll returns."""
    if vehicle is None:
        return
    snap = _sensor_snaps.get(id(vehicle))
    if snap is None or snap.used:
        return
    snap.mono = time.monotonic()


def take_vehicle_sensor_snap(vehicle: Any) -> dict[str, Any] | None:
    """Return the unconsumed snapshot, or None when it is missing or old."""
    if vehicle is None:
        return None
    snap = _sensor_snaps.get(id(vehicle))
    if snap is None or snap.used:
        return None
    if time.monotonic() - snap.mono > SENSOR_POLL_REUSE_S:
        return None
    snap.used = True
    return snap.data


_last_electrics_ms = 0.0


def last_electrics_ms() -> float:
    """Wall-ms of the most recent ``read_electrics`` call."""
    return _last_electrics_ms


def read_electrics(vehicle: Any) -> dict[str, Any] | None:
    """Electrics dict. None when the sensor is absent.

    Soft Esc calls this immediately after ``TechSession.poll``. That poll
    already issued this tick's ``vehicle.sensors.poll``, or republished the
    last-good map when the 200 ms Soft Esc window skipped the GE poll.
    Reuse that snapshot once so the grab does not open a second roundtrip.
    A miss (no snapshot, already consumed, or older than the reuse window)
    still polls. Engage hold reads speed on that miss path.
    """
    global _last_electrics_ms
    t0 = time.perf_counter()
    try:
        if vehicle is None:
            return None
        snap = take_vehicle_sensor_snap(vehicle)
        if isinstance(snap, dict):
            el = snap.get("electrics") if "electrics" in snap else snap
            return el if isinstance(el, dict) else None
        from python.sensors.cameras import camera_ge_socket_busy

        if camera_ge_socket_busy():
            return None
        sensors = getattr(vehicle, "sensors", None)
        if sensors is None:
            return None
        data = None
        if hasattr(sensors, "poll"):
            data = sensors.poll()
        elif callable(sensors):
            data = sensors()
        if not isinstance(data, dict):
            # try electrics attribute
            el = None
            if hasattr(vehicle, "electrics"):
                el = vehicle.electrics
                if hasattr(el, "poll"):
                    el = el.poll()
            if isinstance(el, dict):
                data = {"electrics": el}
            else:
                return None
        el = data.get("electrics") if "electrics" in data else data
        return el if isinstance(el, dict) else None
    except Exception:
        return None
    finally:
        _last_electrics_ms = (time.perf_counter() - t0) * 1000.0


def read_electrics_inputs(vehicle: Any) -> DriverInputs:
    """Speed + steering/throttle/brake inputs in one poll (override detection needs pedals)."""
    el = read_electrics(vehicle)
    if not el:
        return DriverInputs()
    try:
        # wheelspeed / airspeed often m/s already in BeamNG electrics
        spd = el.get("wheelspeed")
        if spd is None:
            spd = el.get("airspeed")
        return DriverInputs(
            speed_mps=_num(spd),
            steering_input=_num(el.get("steering_input")),
            throttle_input=_num(el.get("throttle_input")),
            brake_input=_num(el.get("brake_input")),
        )
    except Exception:
        return DriverInputs()


def read_electrics_speed(vehicle: Any) -> tuple[float | None, float | None]:
    """Return (speed_mps, steering_input) from BeamNGpy Electrics when available."""
    di = read_electrics_inputs(vehicle)
    return di.speed_mps, di.steering_input


def attach_electrics(vehicle: Any, bng: Any = None) -> bool:
    """Best-effort attach Electrics (classic BeamNGpy Sensor API). Prefer TechSession.attach_vehicle_sensors."""
    if vehicle is None:
        return False
    try:
        sensors = getattr(vehicle, "sensors", None)
        if sensors is not None and "electrics" in sensors:
            return True
    except Exception:
        pass
    try:
        from beamngpy.sensors import Electrics  # type: ignore

        el = Electrics()
        if hasattr(vehicle, "attach_sensor"):
            vehicle.attach_sensor("electrics", el)
            vehicle._gvd_electrics = el
            return True
    except Exception:
        pass
    try:
        return hasattr(vehicle, "sensors") and vehicle.sensors is not None
    except Exception:
        return False


class Actuator(Protocol):
    name: str

    def apply(self, cmd: DriveCommand) -> DriveCommand: ...

    def stop(self, seq: int = 0, reason: str = "stop") -> DriveCommand: ...


class NullActuator:
    name = "null"

    def apply(self, cmd: DriveCommand) -> DriveCommand:
        cmd.applied = False
        if cmd.reason == "ok":
            cmd.reason = "null"
        return cmd

    def stop(self, seq: int = 0, reason: str = "stop") -> DriveCommand:
        return stop_command(seq=seq, reason=reason)


class BeamNGPyActuator:
    """Tech drive: vehicle.control only while engaged.

    Disengaged ticks must not slam brake=1 (that is takeover). On the falling
    edge we zero throttle/brake/parkingbrake, set AI mode to disabled, restore
    arcade shift, and rewrite gvd_cmd.json to brake=0 / engaged=false so a
    stale brake:1 file cannot keep the pedals. Release does not arm a new
    shift mode when the shifter was never set. A failed arcade restore leaves
    the latch set so the next disengaged tick retries. After arcade succeeds,
    later Soft Esc ticks refresh the cmd file at most once per
    SOFT_ESC_FILE_PERIOD_S. Engaged gate holds (preview_blocked, AEB, veto)
    still apply the stop command. Arcade stays the shift mode set once at
    engage. Forward drive is throttle with the parking brake and clutch
    released. A hold is parkingbrake=1 and service brake 0, so the brake
    pedal is not reverse throttle. Control omits gear: that field is
    shiftToGearIndex, and arcade uses it to enter realistic. A failed
    set_shift_mode is logged
    and retried.     Throttle>0 while the echoed gear is not forward also queues
    a vehicle-Lua arm that releases the parking brake and the clutch.
    While engaged, steer is a Direct Drive hold on source ``gvd``, not a field
    on ``vehicle.control``. That field would overwrite ``lastInputs.local.steering``
    and erase a held wheel grab. ``steer_locked`` stays set until the release
    chunk is queued, including shutdown. A failed release keeps the latch
    so the next disengaged tick retries. The rising edge writes
    ``tech_steer_hold`` before the first queue so a kill inside that call
    still leaves the mod a watcher. A failed flag write or a failed queue
    leaves ``applied`` false and does not latch the lock. The mod clears the
    whitelist and replays the stored wheel if this process dies.
    """

    name = "beamngpy"

    def __init__(self, vehicle: Any) -> None:
        self.vehicle = vehicle
        self._shift_set = False
        self.engaged = False
        self._latched = False
        self._release_cmd_mono: float | None = None
        self.release_cmd_skips = 0
        self._shift_fail_logged = False
        self._shift_ok_logged = False
        self._drive_arm_mono: float | None = None
        self.drive_arm_n = 0
        self._blinker: str | None = None
        self._steer_locked = False
        self._steer_hold_logged = False
        self._steer_hold_move_logged = False
        self._steer_hold_logged_v = 0.0
        self._steer_hold_fail_logged = False
        self._steer_hold_flag_sent = False

    @property
    def steer_locked(self) -> bool:
        """True after the player-wheel whitelist is queued, until release lands."""
        return bool(self._steer_locked)

    def note_engaged(self, engaged: bool) -> None:
        self.engaged = bool(engaged)
        # Grab loop: poll_vehicle, then note_engaged. The latch covers later
        # grabs (and force_engage, which does not write gvd_engage.json).
        # The rising-edge poll reads read_engage_flag itself.
        note_soft_esc_engaged(self.engaged)

    def _write_release_cmd(
        self, seq: int, reason: str, *, force: bool = False, now: float | None = None
    ) -> bool:
        """gvd_cmd must not keep brake=1 after Disengage. Lua applies only engaged:true.

        Steady Soft Esc refreshes at most once per SOFT_ESC_FILE_PERIOD_S.
        The falling-edge latch, the first rewrite, and shutdown pass
        ``force`` and write this tick. A skipped tick increments
        ``release_cmd_skips`` and leaves the last release file in place.
        ``now`` is monotonic seconds for tests; the supervisor omits it.
        """
        t = time.monotonic() if now is None else float(now)
        last = self._release_cmd_mono
        if not force and last is not None and (t - last) < SOFT_ESC_FILE_PERIOD_S:
            self.release_cmd_skips += 1
            return False
        payload = {
            "steer": 0.0,
            "throttle": 0.0,
            "brake": 0.0,
            "seq": int(seq),
            "engaged": False,
            "heartbeat_mtime": time.time(),
            "reason": str(reason),
        }
        ok = bool(atomic_write_json(cmd_path(), payload, indent=None))
        if ok:
            self._release_cmd_mono = t
        return ok

    def _release_ai(self) -> None:
        """Drop BeamNG AI so keyboard arrows and pedals own the car again."""
        veh = self.vehicle
        if veh is None:
            return
        fn = getattr(veh, "ai_set_mode", None)
        if not callable(fn):
            ai = getattr(veh, "ai", None)
            fn = getattr(ai, "set_mode", None) if ai is not None else None
        if callable(fn):
            try:
                fn("disabled")
                return
            except Exception:
                pass
        q = getattr(veh, "queue_lua_command", None)
        if callable(q):
            try:
                q("if ai and ai.setMode then ai.setMode('disabled') end")
            except Exception:
                pass

    def _restore_player_shift(self) -> bool:
        """Arcade is what player arrows drive. True only after that restore lands.

        A throw leaves ``_shift_set`` unchanged and returns False so ``stop``
        keeps ``_latched`` and the next disengaged tick retries.
        """
        veh = self.vehicle
        if veh is None or not hasattr(veh, "set_shift_mode"):
            self._shift_set = False
            return True
        try:
            veh.set_shift_mode(TECH_PLAYER_SHIFT_MODE)
        except Exception:
            return False
        self._shift_set = False
        self._drive_arm_mono = None
        self.drive_arm_n = 0
        self._shift_ok_logged = False
        self._shift_fail_logged = False
        return True

    def _echo_gear(self) -> Any:
        """Cached electrics gear, if the last poll left it on the vehicle. No extra poll.

        BeamNGpy's sensor container is not a dict. ``.items()`` or ``.data`` hold
        electrics, and electrics itself may be an object whose ``.data`` is the map.
        A dict-only read never sees ``D`` and re-queues the drive arm on every throttle.
        """
        sensors = getattr(self.vehicle, "sensors", None)
        if sensors is None:
            return None
        el = None
        if isinstance(sensors, dict):
            el = sensors.get("electrics")
        else:
            try:
                el = {str(k): v for k, v in sensors.items()}.get("electrics")
            except Exception:
                el = None
            if el is None:
                data = getattr(sensors, "data", None)
                if isinstance(data, dict):
                    el = data.get("electrics")
            if el is None:
                try:
                    el = sensors["electrics"]
                except Exception:
                    el = None
        return _gear_value(el)

    def _arm_shift_mode(self) -> None:
        """Engage arms arcade. A thrown ack is logged and retried, not swallowed."""
        veh = self.vehicle
        if veh is None or not hasattr(veh, "set_shift_mode"):
            return
        try:
            veh.set_shift_mode(TECH_SHIFT_MODE)
        except Exception as e:
            self._shift_set = False
            if not self._shift_fail_logged:
                self._shift_fail_logged = True
                print(
                    f"[GVD] set_shift_mode({TECH_SHIFT_MODE}) failed: {type(e).__name__}: {e}",
                    flush=True,
                )
            return
        self._shift_set = True
        if not self._shift_ok_logged:
            self._shift_ok_logged = True
            print(f"[GVD] set_shift_mode({TECH_SHIFT_MODE}) ok", flush=True)

    def _queue_drive_shift(self, gear_echo: Any, now: float) -> None:
        """Release a resting parking brake and clutch. Rate-limited.

        Queued only while the echoed gear is not forward. The chunk does not
        change gearbox mode and does not call shiftToGearIndex.
        """
        last = self._drive_arm_mono
        if last is not None and (now - last) < TECH_DRIVE_ARM_S:
            return
        q = getattr(self.vehicle, "queue_lua_command", None)
        if not callable(q):
            return
        self._drive_arm_mono = now
        self.drive_arm_n += 1
        if self.drive_arm_n == 1 or self.drive_arm_n % 8 == 0:
            print(
                f"[GVD] drive arm gear={gear_echo!r} attempt={self.drive_arm_n} "
                "parkingbrake=0 clutch=0",
                flush=True,
            )
        try:
            q(TECH_DRIVE_SHIFT_LUA, False)
        except TypeError:
            try:
                q(TECH_DRIVE_SHIFT_LUA)
            except Exception as e:
                print(
                    f"[GVD] drive arm queue_lua_command failed: {type(e).__name__}: {e}",
                    flush=True,
                )
        except Exception as e:
            print(
                f"[GVD] drive arm queue_lua_command failed: {type(e).__name__}: {e}",
                flush=True,
            )

    def _queue_vehicle_lua(self, chunk: str) -> bool:
        """Queue a vehicle-VM snippet. False when the handle cannot take it."""
        veh = self.vehicle
        q = getattr(veh, "queue_lua_command", None) if veh is not None else None
        if not callable(q):
            return False
        try:
            try:
                q(chunk, False)
            except TypeError:
                q(chunk)
        except Exception:
            return False
        return True

    def _log_steer_hold(self, steer: float) -> None:
        """One line when the lock lands, and one when the command leaves that value."""
        if not self._steer_hold_logged:
            self._steer_hold_logged = True
            self._steer_hold_logged_v = steer
            print(
                f"[GVD] tech steer hold {steer:.3f} "
                "(player wheel locked out; source=gvd)",
                flush=True,
            )
            return
        if self._steer_hold_move_logged:
            return
        if abs(steer - self._steer_hold_logged_v) < 0.05:
            return
        self._steer_hold_move_logged = True
        print(
            f"[GVD] tech steer hold {steer:.3f} "
            "(player wheel locked out; source=gvd)",
            flush=True,
        )

    def _queue_steer_hold(self, steer: float) -> str | None:
        """Block local steering and write the controller steer on source gvd.

        Queued after ``vehicle.control``. That message no longer carries
        steering, so it cannot replace ``lastInputs.local.steering``. An error
        string means the hold did not land: ``apply`` leaves ``applied`` false,
        and the next tick still omits steering. The hold flag is written once
        per pre-lock episode, and only a successful write is followed by the
        queue. A later failed queue does not publish the flag again. A kill
        inside the queue still leaves the flag that was written first.
        """
        chunk = tech_steer_hold_lua(steer)
        q = getattr(self.vehicle, "queue_lua_command", None) if self.vehicle is not None else None
        if not callable(q):
            if not self._steer_hold_fail_logged:
                self._steer_hold_fail_logged = True
                print(
                    "[GVD] tech steer hold skipped: no queue_lua_command",
                    flush=True,
                )
            return "tech_steer_hold_no_queue"
        if not self._steer_locked and not self._steer_hold_flag_sent:
            if not flush_tech_steer_hold(True):
                return "tech_steer_hold_flag"
            self._steer_hold_flag_sent = True
        if not self._queue_vehicle_lua(chunk):
            if not self._steer_hold_fail_logged:
                self._steer_hold_fail_logged = True
                print("[GVD] tech steer hold queue failed", flush=True)
            return "tech_steer_hold_queue"
        self._steer_hold_fail_logged = False
        self._steer_locked = True
        self._log_steer_hold(float(steer))
        return None

    def _queue_steer_release(self) -> bool:
        """Give steering back to the player. True when no lock remains.

        Retries while ``steer_locked`` is set. A vehicle with no queue
        cannot have taken the lock, so that case is already clear.
        """
        if not self._steer_locked:
            return True
        if not self._queue_vehicle_lua(TECH_STEER_RELEASE_LUA):
            print("[GVD] tech steer release failed; will retry", flush=True)
            return False
        self._steer_locked = False
        self._steer_hold_flag_sent = False
        self._steer_hold_logged = False
        self._steer_hold_move_logged = False
        self._steer_hold_logged_v = 0.0
        print("[GVD] tech steer release (player wheel restored)", flush=True)
        return True

    def _invoke_control(self, kwargs: dict[str, Any]) -> None:
        """Send control kwargs. Never fall back to arcade brake-hold (no gear, brake=1).

        An unexpected keyword (older ``control`` without ``clutch``) is dropped and
        retried. ``gear`` is dropped only when the error names it, and that drop is logged.
        """
        kw = dict(kwargs)
        last_err: Exception | None = None
        for _ in range(6):
            if is_reverse_control(kw) or is_arcade_reverse_hold(kw):
                break
            try:
                self.vehicle.control(**kw)
                return
            except TypeError as e:
                last_err = e
                name = _unexpected_kw(e)
                if name in ("steering", "throttle", "brake") or name not in kw:
                    raise
                if name == "gear":
                    print(
                        f"[GVD] vehicle.control rejected gear: {e}",
                        flush=True,
                    )
                kw = {k: v for k, v in kw.items() if k != name}
                continue
        if last_err is not None:
            raise last_err
        raise TypeError("vehicle.control rejected non-reverse Tech kwargs")

    def _apply_blinker(self, blinker: str) -> None:
        """Turn signals after the drive command. Skipped while the camera holds the socket.

        ``_control`` returns before this when ``camera_ge_socket_busy`` is set, so a
        blinker never opens a second GE call on a busy tick. The signal changes only
        when the predicted path asks for a different side.
        """
        mode = blinker if blinker in ("left", "right") else "off"
        if mode == self._blinker:
            return
        fn = getattr(self.vehicle, "set_lights", None)
        if not callable(fn):
            self._blinker = mode
            return
        try:
            fn(left_signal=(mode == "left"), right_signal=(mode == "right"), hazard_signal=False)
        except TypeError:
            try:
                fn(left_signal=(mode == "left"), right_signal=(mode == "right"))
            except Exception:
                return
        except Exception:
            return
        self._blinker = mode

    def _control(
        self,
        steer: float,
        throttle: float,
        brake: float,
        *,
        release: bool = False,
        blinker: str = "off",
    ) -> str | None:
        if self.vehicle is None:
            return "no_vehicle"
        from python.sensors.cameras import camera_ge_socket_busy

        if camera_ge_socket_busy():
            return "camera_io_busy"
        try:
            # Engage arms arcade. A release with the shifter still unset must
            # only clear pedals, not arm that mode on the way out.
            # A failed ack is logged and retried next tick (_shift_set stays false).
            if not release and not self._shift_set:
                self._arm_shift_mode()
            speed_mps = None
            if (
                not release
                and float(throttle) <= 1e-6
                and float(brake) >= TECH_HOLD_BRAKE
            ):
                speed_mps = read_electrics_inputs(self.vehicle).speed_mps
            kwargs = tech_control_kwargs(
                steer, throttle, brake, release=release, speed_mps=speed_mps
            )
            # Steer is the gvd hold. A steering field on this message is
            # input.event on source local: blocked from the hydros, but it
            # overwrites lastInputs.local.steering, so the next echo is the
            # command and a held grab never reaches the detector.
            kwargs.pop("steering", None)
            self._invoke_control(kwargs)
            self._apply_blinker("off" if release else blinker)
            if not release and float(throttle) > 1e-6:
                gear_echo = self._echo_gear()
                if not gear_is_forward(gear_echo):
                    self._queue_drive_shift(gear_echo, time.monotonic())
            if not release:
                hold_err = self._queue_steer_hold(steer)
                if hold_err:
                    return hold_err
            return None
        except Exception as e:
            return f"beamngpy_err:{type(e).__name__}"

    def apply(self, cmd: DriveCommand) -> DriveCommand:
        if not self.engaged:
            cmd.applied = False
            if cmd.reason in ("ok", "plan"):
                cmd.reason = "not_engaged"
            return cmd
        err = self._control(
            cmd.steer, cmd.throttle, cmd.brake, blinker=getattr(cmd, "blinker", "off"),
        )
        if err:
            cmd.applied = False
            cmd.reason = err
            return cmd
        cmd.applied = True
        self._latched = True
        return cmd

    def stop(
        self, seq: int = 0, reason: str = "stop", *, now: float | None = None
    ) -> DriveCommand:
        if not self.engaged:
            # Handoff, not a brake hold. ego.brake must not stay at 1, and the
            # cmd bus must not keep a stale brake:1 while engaged is false.
            cmd = release_command(seq=seq, reason=reason)
            # Latched handoff, the first rewrite, and shutdown must hit the
            # file this tick. Later Soft Esc ticks coalesce to pollEvery.
            force = (
                bool(self._latched)
                or self._release_cmd_mono is None
                or str(reason) == "shutdown"
            )
            self._write_release_cmd(seq, reason, force=force, now=now)
            # Wheel first, even when the camera holds the socket or control throws.
            # A failed release keeps the latch so the next disengaged tick retries.
            steer_ok = self._queue_steer_release()
            if self._latched:
                err = self._control(0.0, 0.0, 0.0, release=True)
                if steer_ok and (err is None or err == "no_vehicle"):
                    self._release_ai()
                    if self._restore_player_shift():
                        self._latched = False
            cmd.applied = False
            return cmd
        return self.apply(stop_command(seq=seq, reason=reason))


class CmdJsonActuator:
    """Retail drive bus: atomic Documents/GVD/gvd_cmd.json polled by gvd_main.applyCmdJson.

    Payload carries `engaged` so Lua only touches the player vehicle while the supervisor
    is engaged (gate holds like preview_blocked ride along as brake=1). Lua echoes the seq it
    applied via gvd_ego.json; `applied` is claimed only when that ack is fresh — never on
    the strength of having written a file.
    """

    name = "cmd_json"
    _PASSTHROUGH_TAGS = ("ok", "plan", "stop", "shutdown", "unit_stop")

    def __init__(self) -> None:
        self.engaged = False
        self.ack: EgoFeedback | None = None
        self.last_seq = 0
        self.write_ok = True

    def note_engaged(self, engaged: bool) -> None:
        self.engaged = bool(engaged)
        # Same latch as BeamNGPyActuator. poll_vehicle already ran this grab.
        note_soft_esc_engaged(self.engaged)

    def note_ack(self, fb: EgoFeedback | None) -> None:
        self.ack = fb

    def acked(self, seq: int) -> bool:
        fb = self.ack
        if fb is None or not fb.fresh or not fb.applying:
            return False
        # Ack must be this seq or a few behind. A high applied_seq from a previous
        # supervisor (seq restarts at 1) is not an ack of the command we just wrote.
        try:
            lag = int(seq) - int(fb.applied_seq)
        except (TypeError, ValueError):
            return False
        return 0 <= lag <= CMD_ACK_SLACK

    def apply(self, cmd: DriveCommand) -> DriveCommand:
        driving = bool(self.engaged)
        if not bus_identity().matched:
            # Fresh lua_bus is required. Do not drive the predicted LOCALAPPDATA guess.
            if driving:
                cmd.reason = "bus_mismatch"
                cmd.steer = 0.0
                cmd.throttle = 0.0
                cmd.brake = 1.0
            driving = False
        payload = {
            "steer": float(max(-1.0, min(1.0, cmd.steer))),
            "throttle": float(max(0.0, min(1.0, cmd.throttle))),
            "brake": float(max(0.0, min(1.0, cmd.brake))),
            "seq": int(cmd.seq),
            "engaged": driving,
            "heartbeat_mtime": time.time(),
            "reason": cmd.reason,
        }
        self.write_ok = atomic_write_json(cmd_path(), payload)
        self.last_seq = int(cmd.seq)
        cmd.applied = bool(driving and self.write_ok and self.acked(cmd.seq))
        if cmd.reason in self._PASSTHROUGH_TAGS or cmd.reason.startswith("beamngpy"):
            if cmd.applied:
                cmd.reason = "cmd_json_applied"
            elif driving:
                cmd.reason = "cmd_json_pending"  # written; no fresh Lua ack (mod off / no vehicle)
            else:
                cmd.reason = "cmd_json_idle"  # not engaged: Lua keeps its hands off the car
        return cmd

    def stop(self, seq: int = 0, reason: str = "stop") -> DriveCommand:
        cmd = stop_command(seq=seq, reason=reason)
        return self.apply(cmd)


def make_actuator(vehicle: Any | None = None, prefer_beamngpy: bool = True) -> Actuator:
    if prefer_beamngpy and vehicle is not None and hasattr(vehicle, "control"):
        return BeamNGPyActuator(vehicle)
    # Retail: cmd json applied by the mod's GELua on the player vehicle. Still no DLL.
    return CmdJsonActuator()


def steer_sample_point(path_ego: list[dict[str, float]]) -> dict[str, float] | None:
    """Point the wheel reads.

    A path that starts at the bumper with 1 m steps has its old index-8
    sample at about 8 m. Prefer the first point at least that far ahead.
    Points beside and behind the car stay on the route, but they are not
    the sample. If the visible piece ends before 8 m, use the farthest
    point still ahead of the bumper. Nothing ahead leaves the wheel centered.
    """
    ahead = [p for p in path_ego if float(p.get("y", 0.0)) >= 8.0]
    if ahead:
        return ahead[0]
    forward = [p for p in path_ego if float(p.get("y", 0.0)) > 0.0]
    if not forward:
        return None
    return max(forward, key=lambda p: float(p.get("y", 0.0)))


def path_steer_from_ego(path_ego: list[dict[str, float]] | None) -> float:
    if not path_ego or len(path_ego) < 2:
        return 0.0
    mid = steer_sample_point(path_ego)
    if mid is None:
        return 0.0
    return max(-1.0, min(1.0, float(mid.get("x", 0.0)) / 2.5))
