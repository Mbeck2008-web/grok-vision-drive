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
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from python.runtime.paths import bus_identity
from python.runtime.state_io import atomic_write_json, gvd_docs_dir

HEARTBEAT_STALE_S = 0.35
AEB_BRAKE_TTC = 1.2
EGO_FRESH_S = 1.0          # gvd_ego.json older than this → treat Lua feedback as gone
CMD_ACK_SLACK = 5          # Lua acks lag a few seqs (20 Hz apply, 10 Hz echo, 15 Hz loop)
# Lua heartbeats gvd_engage.json while its in-memory latch is on (os.time is 1 s).
# A leftover engaged:true from a crashed session must not start Tech vehicle.control.
ENGAGE_FRESH_S = 2.5

# Tech no-R (research pin): arcade + brake-hold + no throttle auto-selects R.
# Use realistic_automatic so brake-hold is not reverse throttle. Gear is int only
# (-1 R, 0 N, 1+ forward) — never gear=-1, never letter "D".
TECH_SHIFT_MODE = "realistic_automatic"
# Player arrows/pedals expect arcade. Restored only on the Disengage handoff.
TECH_PLAYER_SHIFT_MODE = "arcade"
TECH_HOLD_BRAKE = 0.99
# Below this, a full brake is a rest hold (parkingbrake on). Rolling AEB keeps PB off.
TECH_HOLD_SPEED_MPS = 0.5
TECH_HOLD_GEAR = 0
TECH_DRIVE_GEAR = 1
# Let neutralSelectionDelay (~0.5 s) finish before asking the lever to move again.
TECH_DRIVE_ARM_S = 0.55
# Vehicle VM. Idempotent: shiftUp only from N/0, otherwise jump to the 'D' letter in
# automaticModes (1-based string.find). D/S/M/L, numeric >=1, and M1/M2 already count
# as forward (same as gear_is_forward), so a stale Python echo cannot move that lever.
# Never shiftDown, never gear -1. Parking brake and clutch are zeroed on source gvd.
TECH_DRIVE_SHIFT_LUA = (
    "pcall(function() "
    "local c=controller and controller.mainController; "
    "if c and c.setGearboxMode then pcall(function() c.setGearboxMode('realistic') end) end; "
    "if input and input.event then pcall(function() "
    "input.event('parkingbrake',0,2,0,0,nil,'gvd'); "
    "input.event('clutch',0,2,0,0,nil,'gvd') end) end; "
    "local logic=c and c.shiftLogic; "
    "local pos=''; "
    "if logic and logic.getGearPosition then local ok,p=pcall(logic.getGearPosition); "
    "if ok and p~=nil then pos=tostring(p) end end; "
    "if pos=='' and electrics and electrics.values and electrics.values.gear~=nil then "
    "pos=tostring(electrics.values.gear) end; "
    "pos=string.upper(pos); "
    "local n=tonumber(pos); "
    "local manual=pos:match('^M(%d+)$'); "
    "if pos=='D' or pos=='S' or pos=='M' or pos=='L' or (n and n>=1) "
    "or (manual and tonumber(manual)>=1) then return end; "
    "if (pos=='N' or pos=='0' or n==0) and c and c.shiftUp then "
    "local up=pcall(function() c.shiftUp() end); "
    "if not up then up=pcall(function() c:shiftUp() end) end; "
    "if up then return end end; "
    "if logic and c and c.shiftToGearIndex then "
    "local modes=logic.automaticModes or logic.modes; "
    "if type(modes)=='string' then local d=string.find(modes,'D',1,true); "
    "if d then local jumped=pcall(function() c.shiftToGearIndex(d) end); "
    "if not jumped then pcall(function() c:shiftToGearIndex(d) end) end end end end "
    "end)"
)


@dataclass
class DriveCommand:
    steer: float = 0.0      # [-1, 1]
    throttle: float = 0.0  # [0, 1]
    brake: float = 0.0     # [0, 1]
    seq: int = 0
    applied: bool = False
    reason: str = "ok"


def cmd_path() -> Path:
    return gvd_docs_dir() / "gvd_cmd.json"


def engage_path() -> Path:
    return gvd_docs_dir() / "gvd_engage.json"


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
            return int(gear) >= TECH_DRIVE_GEAR
        except (TypeError, ValueError):
            return False
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


def _tech_gear(value: int) -> int:
    """Clamp to a non-reverse BeamNGpy gear int (0 N, 1+ forward). Never -1 or 'D'."""
    try:
        g = int(value)
    except (TypeError, ValueError):
        return TECH_HOLD_GEAR
    if g < 0:
        return TECH_HOLD_GEAR
    return g


def tech_control_kwargs(
    steer: float,
    throttle: float,
    brake: float,
    *,
    release: bool = False,
    speed_mps: float | None = None,
) -> dict[str, Any]:
    """BeamNGpy vehicle.control kwargs that never select reverse.

    Shift mode is realistic_automatic (not arcade). Holds: throttle=0, brake=1,
    gear=0, ±parkingbrake. Forward motion only with gear>=1 and throttle>0.
    gear is int only; never -1, never letter D. Drive also sends clutch=0 so a
    resting clutch pedal cannot hold the gearbox out of gear.
    """
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
            "gear": _tech_gear(TECH_DRIVE_GEAR),
        }
    parking = 0.0
    if brake_v >= TECH_HOLD_BRAKE:
        moving = speed_mps is not None and float(speed_mps) > TECH_HOLD_SPEED_MPS
        parking = 0.0 if moving else 1.0
    return {
        "steering": steer_v,
        "throttle": 0.0,
        "brake": brake_v,
        "parkingbrake": parking,
        "gear": _tech_gear(TECH_HOLD_GEAR),
    }


def is_reverse_control(kwargs: dict[str, Any]) -> bool:
    """True when control kwargs select reverse (gear=-1)."""
    gear = kwargs.get("gear")
    try:
        return gear is not None and int(gear) < 0
    except (TypeError, ValueError):
        return False


def is_arcade_reverse_hold(kwargs: dict[str, Any]) -> bool:
    """Backward-compat alias: arcade brake-hold → R, or an explicit reverse gear."""
    if is_reverse_control(kwargs):
        return True
    throttle = float(kwargs.get("throttle") or 0.0)
    brake = float(kwargs.get("brake") or 0.0)
    parkingbrake = float(kwargs.get("parkingbrake") or 0.0)
    gear = kwargs.get("gear")
    try:
        if gear is not None and int(gear) == TECH_HOLD_GEAR:
            return False
        if gear is not None and int(gear) >= TECH_DRIVE_GEAR:
            return False
    except (TypeError, ValueError):
        pass
    return throttle <= 1e-6 and brake >= TECH_HOLD_BRAKE and parkingbrake < 0.5


def plan_command(
    *,
    path_ego: list[dict[str, float]] | None,
    planner: dict[str, Any] | None,
    ego_speed_mps: float,
    seq: int,
) -> DriveCommand:
    """Map corridor + speed plan → DriveCommand pedals.

    AEB / full stop still return brake=1, throttle=0 (retail JSON + override
    semantics). Tech remaps that pair in BeamNGPyActuator via tech_control_kwargs
    (realistic_automatic, hold gear=0 + brake ±parkingbrake; never gear=-1).
    """
    planner = planner or {}
    aeb = str(planner.get("aeb") or "off")
    target_v = float(planner.get("target_v") or 0.0)
    ttc = planner.get("ttc_lead")

    steer = 0.0
    if path_ego and len(path_ego) >= 3:
        # lateral of a near point ≈ steering demand
        mid = path_ego[min(8, len(path_ego) - 1)]
        x = float(mid.get("x", 0.0))
        steer = max(-1.0, min(1.0, x / 2.5))

    throttle = 0.0
    brake = 0.0
    if aeb == "brake" or (ttc is not None and float(ttc) < AEB_BRAKE_TTC):
        # Keep the command semantic (brake=1). Arcade would treat this as reverse
        # throttle; Tech remaps at control() (realistic_automatic, gear=0 hold).
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

    return DriveCommand(steer=steer, throttle=throttle, brake=brake, seq=seq, reason="plan")


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
# Soft Esc (latch false and gvd_engage.json not live) may skip the GE poll
# for 200 ms and republish the last-good map here so this read does not open
# a second sensors.poll. TechSession.poll reads the engage flag before that
# hold, because note_engaged runs after poll_vehicle.
SENSOR_POLL_REUSE_S = 0.05

# Engage latch updated by note_engaged after poll_vehicle. Default false:
# Soft Esc coalesces without a run_vision edit. True: one sensors.poll per
# grab (Tip #1). The rising edge does not wait for this latch; poll() also
# refuses the hold when read_engage_flag() is true.
_soft_esc_engaged = False


def note_soft_esc_engaged(engaged: bool) -> None:
    """Latch Engage so a grab with this bit set polls every tick.

    Callers that flip this must restore it. ``TechSession.poll`` does not
    wait for the latch on the rising edge; it also reads ``read_engage_flag``.
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
    stale brake:1 file cannot keep the pedals. Release does not arm
    realistic_automatic when the shifter was never set. A failed arcade
    restore leaves the latch set so the next disengaged tick retries. After
    arcade succeeds, later Soft Esc ticks refresh the cmd file at most once
    per SOFT_ESC_FILE_PERIOD_S. Engaged gate holds
    (preview_blocked, AEB, veto) still apply the stop command, remapped off
    reverse: realistic_automatic, hold gear=0 + brake ±parkingbrake, drive
    gear>=1 plus clutch=0, never gear=-1. A failed set_shift_mode is logged
    and retried. Throttle>0 while the echoed gear is not forward also queues
    a vehicle-Lua arm that leaves N/P without selecting reverse.
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
        """Engage arms realistic_automatic. A thrown ack is logged and retried, not swallowed."""
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
        """Ask the vehicle VM to leave N/P without selecting reverse. Rate-limited.

        The chunk itself no-ops once the lever is already in a forward range, so a
        stale Python echo of ``N`` cannot walk D → 2.
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

    def _control(
        self, steer: float, throttle: float, brake: float, *, release: bool = False
    ) -> str | None:
        if self.vehicle is None:
            return "no_vehicle"
        try:
            # Engage arms realistic_automatic. A release with the shifter
            # still unset must only clear pedals, not arm that mode on the way out.
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
            self._invoke_control(kwargs)
            if not release and float(throttle) > 1e-6:
                gear_echo = self._echo_gear()
                if not gear_is_forward(gear_echo):
                    self._queue_drive_shift(gear_echo, time.monotonic())
            return None
        except Exception as e:
            return f"beamngpy_err:{type(e).__name__}"

    def apply(self, cmd: DriveCommand) -> DriveCommand:
        if not self.engaged:
            cmd.applied = False
            if cmd.reason in ("ok", "plan"):
                cmd.reason = "not_engaged"
            return cmd
        err = self._control(cmd.steer, cmd.throttle, cmd.brake)
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
            if self._latched:
                err = self._control(0.0, 0.0, 0.0, release=True)
                if err is None or err == "no_vehicle":
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


def path_steer_from_ego(path_ego: list[dict[str, float]] | None) -> float:
    if not path_ego or len(path_ego) < 2:
        return 0.0
    mid = path_ego[min(8, len(path_ego) - 1)]
    return max(-1.0, min(1.0, float(mid.get("x", 0.0)) / 2.5))
