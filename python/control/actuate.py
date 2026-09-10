"""Sim-only actuation: BeamNGpy vehicle.control (Tech) or gvd_cmd.json → GELua (retail).

Safety gates (must all pass to drive):
  engaged + heartbeat fresh + (not path_debug_preview OR allow_preview_drive)

Retail bus (M6): Python writes Documents/GVD/gvd_cmd.json every tick; the mod's
gvd_main.applyCmdJson feeds steer/throttle/brake to the player vehicle with the same
vehicle-Lua `input.event` calls BeamNG's own AI and BeamNGpy use, and echoes
wheelspeed / inputs / applied seq back through gvd_ego.json. `cmd_applied` is only
claimed once that ack is fresh. No DLL / hooks / process inject. Live BeamNG still
UNPROVEN on Linux.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from python.runtime.state_io import atomic_write_json, gvd_docs_dir

HEARTBEAT_STALE_S = 0.35
AEB_BRAKE_TTC = 1.2
EGO_FRESH_S = 1.0          # gvd_ego.json older than this → treat Lua feedback as gone
CMD_ACK_SLACK = 5          # Lua acks lag a few seqs (20 Hz apply, 10 Hz echo, 15 Hz loop)


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
    return EgoFeedback(
        speed_mps=_num(data.get("speed_mps")),
        steering_input=_num(data.get("steering_input")),
        throttle_input=_num(data.get("throttle_input")),
        brake_input=_num(data.get("brake_input")),
        applied_seq=seq,
        applying=bool(data.get("applying", False)),
        age_s=max(0.0, t - mtime),
    )


def read_engage_flag(default: bool = False) -> bool:
    """Lua Alt+A writes gvd_engage.json; Python mirrors that (do not invent engage)."""
    p = engage_path()
    if not p.is_file():
        return default
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return bool(data.get("engaged", default))
    except Exception:
        return default


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


def plan_command(
    *,
    path_ego: list[dict[str, float]] | None,
    planner: dict[str, Any] | None,
    ego_speed_mps: float,
    seq: int,
) -> DriveCommand:
    """Map corridor + speed plan → arcade controls. AEB brake forces throttle=0."""
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
        # Arcade + hold brake can auto-shift reverse — keep throttle=0 (README caveat).
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


def read_electrics(vehicle: Any) -> dict[str, Any] | None:
    """One best-effort poll of the BeamNGpy Electrics dict. None when the sensor is absent."""
    if vehicle is None:
        return None
    try:
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
    """Best-effort attach Electrics sensor (BeamNGpy). Returns True if attached/available."""
    if vehicle is None:
        return False
    try:
        from beamngpy.sensors import Electrics  # type: ignore

        # Avoid double-attach
        if getattr(vehicle, "_gvd_electrics", None) is not None:
            return True
        el = Electrics("gvd_electrics", bng, vehicle) if bng is not None else Electrics()
        if hasattr(vehicle, "attach_sensor"):
            vehicle.attach_sensor("electrics", el)
        vehicle._gvd_electrics = el
        return True
    except Exception:
        # Older APIs: vehicle.sensors may already include electrics after poll setup
        try:
            if hasattr(vehicle, "sensors") and vehicle.sensors is not None:
                return True
        except Exception:
            pass
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
    name = "beamngpy"

    def __init__(self, vehicle: Any) -> None:
        self.vehicle = vehicle
        self._shift_set = False

    def apply(self, cmd: DriveCommand) -> DriveCommand:
        if self.vehicle is None:
            cmd.applied = False
            cmd.reason = "no_vehicle"
            return cmd
        try:
            if not self._shift_set and hasattr(self.vehicle, "set_shift_mode"):
                try:
                    self.vehicle.set_shift_mode("arcade")
                    self._shift_set = True
                except Exception:
                    pass
            # steering [-1,1], throttle/brake [0,1]
            self.vehicle.control(
                steering=float(max(-1.0, min(1.0, cmd.steer))),
                throttle=float(max(0.0, min(1.0, cmd.throttle))),
                brake=float(max(0.0, min(1.0, cmd.brake))),
            )
            cmd.applied = True
            return cmd
        except Exception as e:
            cmd.applied = False
            cmd.reason = f"beamngpy_err:{type(e).__name__}"
            return cmd

    def stop(self, seq: int = 0, reason: str = "stop") -> DriveCommand:
        cmd = stop_command(seq=seq, reason=reason)
        return self.apply(cmd)


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

    def note_ack(self, fb: EgoFeedback | None) -> None:
        self.ack = fb

    def acked(self, seq: int) -> bool:
        fb = self.ack
        if fb is None or not fb.fresh or not fb.applying:
            return False
        return fb.applied_seq >= int(seq) - CMD_ACK_SLACK

    def apply(self, cmd: DriveCommand) -> DriveCommand:
        payload = {
            "steer": float(max(-1.0, min(1.0, cmd.steer))),
            "throttle": float(max(0.0, min(1.0, cmd.throttle))),
            "brake": float(max(0.0, min(1.0, cmd.brake))),
            "seq": int(cmd.seq),
            "engaged": bool(self.engaged),
            "heartbeat_mtime": time.time(),
            "reason": cmd.reason,
        }
        self.write_ok = atomic_write_json(cmd_path(), payload)
        self.last_seq = int(cmd.seq)
        cmd.applied = bool(self.engaged and self.write_ok and self.acked(cmd.seq))
        if cmd.reason in self._PASSTHROUGH_TAGS or cmd.reason.startswith("beamngpy"):
            if cmd.applied:
                cmd.reason = "cmd_json_applied"
            elif self.engaged:
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
