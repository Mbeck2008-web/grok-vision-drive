"""Sim-only actuation: BeamNGpy vehicle.control preferred; gvd_cmd.json fallback.

Safety gates (must all pass to drive):
  engaged + heartbeat fresh + (not path_debug_preview OR allow_preview_drive)

No DLL / hooks / process inject. Live BeamNG still UNPROVEN on Linux.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from python.runtime.state_io import gvd_docs_dir

HEARTBEAT_STALE_S = 0.35
AEB_BRAKE_TTC = 1.2


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


def write_engage_flag(engaged: bool) -> None:
    p = engage_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"engaged": bool(engaged), "mtime": time.time()}), encoding="utf-8")
    tmp.replace(p)


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


def read_electrics_speed(vehicle: Any) -> tuple[float | None, float | None]:
    """Return (speed_mps, steering_input) from BeamNGpy Electrics when available."""
    if vehicle is None:
        return None, None
    try:
        sensors = getattr(vehicle, "sensors", None)
        if sensors is None:
            return None, None
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
                return None, None
        el = data.get("electrics") if "electrics" in data else data
        if not isinstance(el, dict):
            return None, None
        # wheelspeed / airspeed often m/s already in BeamNG electrics
        spd = el.get("wheelspeed")
        if spd is None:
            spd = el.get("airspeed")
        steer_in = el.get("steering_input")
        speed = float(spd) if spd is not None else None
        steeri = float(steer_in) if steer_in is not None else None
        return speed, steeri
    except Exception:
        return None, None


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
    """Atomic Documents/GVD/gvd_cmd.json for GELua poll — fallback only."""

    name = "cmd_json"

    def apply(self, cmd: DriveCommand) -> DriveCommand:
        p = cmd_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "steer": float(cmd.steer),
            "throttle": float(cmd.throttle),
            "brake": float(cmd.brake),
            "seq": int(cmd.seq),
            "heartbeat_mtime": time.time(),
            "reason": cmd.reason,
        }
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(p)
        cmd.applied = True
        return cmd

    def stop(self, seq: int = 0, reason: str = "stop") -> DriveCommand:
        cmd = stop_command(seq=seq, reason=reason)
        return self.apply(cmd)


def make_actuator(vehicle: Any | None = None, prefer_beamngpy: bool = True) -> Actuator:
    if prefer_beamngpy and vehicle is not None and hasattr(vehicle, "control"):
        return BeamNGPyActuator(vehicle)
    # Fallback: cmd json (Lua optional poll). Still no DLL.
    return CmdJsonActuator()


def path_steer_from_ego(path_ego: list[dict[str, float]] | None) -> float:
    if not path_ego or len(path_ego) < 2:
        return 0.0
    mid = path_ego[min(8, len(path_ego) - 1)]
    return max(-1.0, min(1.0, float(mid.get("x", 0.0)) / 2.5))
