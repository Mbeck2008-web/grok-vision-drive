"""BeamNG.tech session: connect, player vehicle, electrics/damage/pose. Vision-only.

Never attaches LiDAR / radar / ultrasonic / GPS. Camera RGB is handled by cameras.py.
Coordinates: GVD vehicle frame is +X right, +Y forward, +Z up. BeamNGpy Camera vehicle
space is +X left, +Y backward, +Z up (default dir=(0,-1,0) is forward). Convert at attach.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_SENSORS = ("lidar", "radar", "ultrasonic", "gps", "idealradar", "ideal_radar")


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _vec3(v: Any) -> tuple[float, float, float] | None:
    if v is None:
        return None
    try:
        if isinstance(v, dict):
            return (float(v["x"]), float(v["y"]), float(v["z"]))
        seq = list(v)
        if len(seq) < 3:
            return None
        return (float(seq[0]), float(seq[1]), float(seq[2]))
    except Exception:
        return None


def _norm(x: float, y: float, z: float) -> tuple[float, float, float]:
    L = math.sqrt(x * x + y * y + z * z) + 1e-9
    return x / L, y / L, z / L


def gvd_to_bng_vehicle(x: float, y: float, z: float) -> tuple[float, float, float]:
    """GVD (+X right, +Y forward, +Z up) → BeamNGpy Camera vehicle space (+X left, +Y back, +Z up)."""
    return (-float(x), -float(y), float(z))


def bng_to_gvd_vehicle(x: float, y: float, z: float) -> tuple[float, float, float]:
    return gvd_to_bng_vehicle(x, y, z)


def path_ego_to_world(
    path_ego: list[dict[str, float]] | None,
    pos: tuple[float, float, float] | None,
    fwd: tuple[float, float, float] | None,
    up: tuple[float, float, float] | None = None,
) -> list[dict[str, float]] | None:
    """Kinematics-only: GVD path_ego × vehicle pose → world. Not a map."""
    if not path_ego or pos is None or fwd is None:
        return None
    if up is None:
        up = (0.0, 0.0, 1.0)
    try:
        px, py, pz = pos
        fx, fy, fz = _norm(*fwd)
        ux, uy, uz = _norm(*up)
        rx, ry, rz = _norm(fy * uz - fz * uy, fz * ux - fx * uz, fx * uy - fy * ux)
        out: list[dict[str, float]] = []
        for p in path_ego:
            ex, ey, ez = float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0))
            out.append(
                {
                    "x": px + rx * ex + fx * ey + ux * ez,
                    "y": py + ry * ex + fy * ey + uy * ez,
                    "z": pz + rz * ex + fz * ey + uz * ez,
                }
            )
        return out
    except Exception:
        return None


def load_tech_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (ROOT / "config" / "tech.yaml")
    data: dict[str, Any] = {}
    try:
        import yaml  # type: ignore

        if cfg_path.is_file():
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return data


def apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Env wins over yaml. Does not invent a live session."""
    out = dict(cfg)
    host = os.environ.get("GVD_BEAMNG_HOST")
    if host:
        out["host"] = host
    port = os.environ.get("GVD_BEAMNG_PORT")
    if port:
        try:
            out["port"] = int(port)
        except ValueError:
            pass
    home = os.environ.get("BNG_HOME") or os.environ.get("BEAMNG_HOME")
    if home:
        out["home"] = home
    user = os.environ.get("GVD_BEAMNG_USER")
    if user:
        out["user"] = user
    launch = os.environ.get("GVD_TECH_LAUNCH", "").strip().lower()
    if launch in ("1", "true", "yes"):
        out["launch"] = True
    elif launch in ("0", "false", "no"):
        out["launch"] = False
    vid = os.environ.get("GVD_TECH_VEHICLE")
    if vid:
        out["vehicle"] = vid
    return out


def wants_tech_attach() -> bool:
    return os.environ.get("GVD_BEAMNG", "").strip().lower() in ("1", "true", "yes")


@dataclass
class VehicleData:
    """One poll of Tech vehicle kinematics. None fields = honest miss, not invented."""

    vid: str | None = None
    model: str | None = None
    connected: bool = False
    speed_mps: float | None = None
    steering_input: float | None = None
    throttle_input: float | None = None
    brake_input: float | None = None
    gear: Any = None
    rpm: float | None = None
    parkingbrake: float | None = None
    damage: float | None = None
    pos: tuple[float, float, float] | None = None
    dir: tuple[float, float, float] | None = None
    up: tuple[float, float, float] | None = None
    vel: tuple[float, float, float] | None = None
    gx: float | None = None
    gy: float | None = None
    gz: float | None = None
    yaw_rate: float | None = None
    accel: float | None = None
    note: str = ""
    sensors: dict[str, str] = field(default_factory=dict)

    @property
    def pose_ok(self) -> bool:
        return self.pos is not None and self.dir is not None


class TechSession:
    """Connect to a running (or launched) Tech instance and attach vehicle data sensors."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = apply_env_overrides(config if config is not None else load_tech_config())
        self.bng: Any = None
        self.vehicle: Any = None
        self.attached: dict[str, bool] = {}
        self._logged = False
        self._last_dir: tuple[float, float, float] | None = None
        self._last_t: float | None = None
        self.note = ""

    def connect(self, *, explicit: bool = True) -> bool:
        """Open BeamNGpy. explicit=True when the user asked for --backend beamngpy."""
        if not explicit and not wants_tech_attach():
            self.note = "set GVD_BEAMNG=1 or --backend beamngpy to attach"
            return False
        try:
            from beamngpy import BeamNGpy  # type: ignore
        except Exception as e:
            self.note = f"beamngpy not available ({e})"
            self._log(f"[GVD] {self.note}; vehicle data missing.")
            return False

        host = str(self.config.get("host") or "localhost")
        port = int(self.config.get("port") or 25252)
        home = str(self.config.get("home") or "").strip() or None
        user = str(self.config.get("user") or "").strip() or None
        launch = bool(self.config.get("launch"))
        kwargs: dict[str, Any] = {}
        if home:
            kwargs["home"] = home
        if user:
            kwargs["user"] = user
        try:
            bng = BeamNGpy(host, port, **kwargs) if kwargs else BeamNGpy(host, port)
            bng.open(launch=launch)
            self.bng = bng
        except Exception as e:
            self.note = f"beamngpy connect failed ({e})"
            self._log(
                f"[GVD] {self.note}. Is BeamNG.tech listening on {host}:{port} "
                f"with tech.key in the install dir?"
            )
            return False

        wait_s = float(self.config.get("wait_vehicle_s") or 0.0)
        vehicle = self._wait_vehicle(wait_s)
        if vehicle is None:
            self.note = "no vehicle to attach (spawn one in Tech, then retry)"
            self._log(f"[GVD] beamngpy: {self.note}.")
            return False
        self.vehicle = vehicle
        self.note = f"connected vid={self._vid(vehicle)}"
        return True

    def _wait_vehicle(self, wait_s: float) -> Any:
        deadline = time.time() + max(0.0, wait_s)
        first = True
        while True:
            veh = self._resolve_vehicle()
            if veh is not None:
                return veh
            if first:
                self._log(
                    f"[GVD] beamngpy: waiting up to {max(0.0, wait_s):.0f}s for a spawned vehicle…"
                )
                first = False
            if time.time() >= deadline:
                return None
            time.sleep(0.4)

    def _resolve_vehicle(self) -> Any:
        bng = self.bng
        if bng is None:
            return None
        want = str(self.config.get("vehicle") or "current").strip()
        vehicles = self._current_vehicles(bng)
        if not vehicles:
            return None
        pick = None
        if want and want not in ("current", "*"):
            pick = vehicles.get(want)
        if pick is None:
            pid = self._player_vid(bng)
            if pid and pid in vehicles:
                pick = vehicles[pid]
        if pick is None:
            pick = next(iter(vehicles.values()), None)
        if pick is None:
            return None
        return self._connect_vehicle(bng, pick)

    def _current_vehicles(self, bng: Any) -> dict[str, Any]:
        try:
            api = getattr(bng, "vehicles", None)
            if api is not None and hasattr(api, "get_current"):
                cur = api.get_current()
                if isinstance(cur, dict):
                    return {str(k): v for k, v in cur.items() if v is not None}
        except Exception:
            pass
        return {}

    def _player_vid(self, bng: Any) -> str | None:
        try:
            api = getattr(bng, "vehicles", None)
            if api is None or not hasattr(api, "get_player_vehicle_id"):
                return None
            info = api.get_player_vehicle_id()
            if isinstance(info, dict):
                vid = info.get("vid") or info.get("name")
                return str(vid) if vid else None
            return str(info) if info else None
        except Exception:
            return None

    def _connect_vehicle(self, bng: Any, veh: Any) -> Any:
        try:
            if hasattr(veh, "is_connected") and not veh.is_connected():
                if hasattr(veh, "connect"):
                    veh.connect(bng)
                elif hasattr(bng, "connect_vehicle"):
                    bng.connect_vehicle(veh)
        except Exception as e:
            self._log(f"[GVD] beamngpy vehicle connect: {e}")
        return veh

    def attach_vehicle_sensors(self) -> dict[str, bool]:
        """Electrics + Damage + GForces only. Refuses perception extras."""
        flags = self.config.get("sensors") if isinstance(self.config.get("sensors"), dict) else {}
        for name in FORBIDDEN_SENSORS:
            if flags.get(name):
                raise ValueError(f"GVD refuses Tech sensor {name!r} (vision-only)")
        vehicle = self.vehicle
        bng = self.bng
        if vehicle is None:
            return dict(self.attached)
        if flags.get("electrics", True):
            self.attached["electrics"] = self._attach_classic(vehicle, bng, "electrics", "Electrics")
        if flags.get("damage", True):
            self.attached["damage"] = self._attach_classic(vehicle, bng, "damage", "Damage")
        if flags.get("gforces", True):
            self.attached["gforces"] = self._attach_classic(vehicle, bng, "gforces", "GForces")
        ok = [k for k, v in self.attached.items() if v]
        self._log(
            f"[GVD] tech vehicle sensors: {', '.join(ok) if ok else 'none'} "
            f"(RGB cameras separate; no LiDAR/radar/GPS)."
        )
        return dict(self.attached)

    def _attach_classic(self, vehicle: Any, bng: Any, key: str, cls_name: str) -> bool:
        """Old BeamNGpy Sensor API: Electrics() / Damage() / GForces() + attach_sensor."""
        try:
            sensors = getattr(vehicle, "sensors", None)
            if sensors is not None and key in sensors:
                return True
        except Exception:
            pass
        try:
            import beamngpy.sensors as sn  # type: ignore

            cls = getattr(sn, cls_name, None)
            if cls is None:
                return False
            inst = cls()
            if hasattr(vehicle, "attach_sensor"):
                vehicle.attach_sensor(key, inst)
                return True
        except Exception as e:
            self._log(f"[GVD] attach {key} failed: {e}")
        return False

    def poll(self) -> VehicleData:
        data = VehicleData(
            vid=self._vid(self.vehicle),
            model=self._model(self.vehicle),
            connected=self.vehicle is not None,
            sensors={k: ("ok" if v else "missing") for k, v in self.attached.items()},
            note=self.note,
        )
        vehicle = self.vehicle
        if vehicle is None:
            data.note = self.note or "no vehicle"
            return data
        sensors = self._poll_sensors(vehicle)
        el = self._extract(sensors, "electrics")
        dmg = self._extract(sensors, "damage")
        gf = self._extract(sensors, "gforces")
        if el:
            spd = el.get("wheelspeed")
            if spd is None:
                spd = el.get("airspeed")
            data.speed_mps = _num(spd)
            data.steering_input = _num(el.get("steering_input"))
            data.throttle_input = _num(el.get("throttle_input"))
            data.brake_input = _num(el.get("brake_input"))
            data.gear = el.get("gear")
            data.rpm = _num(el.get("rpm"))
            data.parkingbrake = _num(el.get("parkingbrake"))
        if dmg:
            data.damage = _num(dmg.get("damage"))
        if gf:
            data.gx = _num(gf.get("gx") if gf.get("gx") is not None else gf.get("gx2"))
            data.gy = _num(gf.get("gy") if gf.get("gy") is not None else gf.get("gy2"))
            data.gz = _num(gf.get("gz") if gf.get("gz") is not None else gf.get("gz2"))
            if data.gx is not None and data.gy is not None:
                data.accel = math.sqrt(data.gx * data.gx + data.gy * data.gy)
        st = self._state(vehicle, sensors)
        if st:
            data.pos = _vec3(st.get("pos"))
            data.dir = _vec3(st.get("dir") or st.get("forward"))
            data.up = _vec3(st.get("up"))
            data.vel = _vec3(st.get("vel") or st.get("velocity"))
        now = time.time()
        if data.dir is not None and self._last_dir is not None and self._last_t is not None:
            dt = now - self._last_t
            if 0.001 < dt < 1.0:
                ax, ay, az = self._last_dir
                bx, by, bz = _norm(*data.dir)
                dot = max(-1.0, min(1.0, ax * bx + ay * by + az * bz))
                # yaw around world Z from consecutive forward vectors
                cross_z = ax * by - ay * bx
                data.yaw_rate = math.atan2(cross_z, dot) / dt
        if data.dir is not None:
            self._last_dir = _norm(*data.dir)
            self._last_t = now
        if data.speed_mps is None and data.vel is not None:
            vx, vy, vz = data.vel
            data.speed_mps = math.sqrt(vx * vx + vy * vy + vz * vz)
        return data

    def _poll_sensors(self, vehicle: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        try:
            sensors = getattr(vehicle, "sensors", None)
            if sensors is None:
                return out
            if hasattr(sensors, "poll"):
                sensors.poll()
            if isinstance(sensors, dict):
                return {str(k): sensors[k] for k in sensors}
            # SensorContainer: iterate keys
            try:
                for k in list(sensors):
                    out[str(k)] = sensors[k]
            except Exception:
                pass
        except Exception:
            return out
        return out

    def _extract(self, sensors: dict[str, Any], key: str) -> dict[str, Any] | None:
        raw = sensors.get(key)
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw
        data = getattr(raw, "data", None)
        if isinstance(data, dict):
            return data
        return None

    def _state(self, vehicle: Any, sensors: dict[str, Any]) -> dict[str, Any] | None:
        st = getattr(vehicle, "state", None)
        if isinstance(st, dict) and st.get("pos") is not None:
            return st
        raw = self._extract(sensors, "state")
        if isinstance(raw, dict) and raw.get("pos") is not None:
            return raw
        # Fallback: vehicles.get_states([vid])
        try:
            vid = self._vid(vehicle)
            api = getattr(self.bng, "vehicles", None)
            if vid and api is not None and hasattr(api, "get_states"):
                mapping = api.get_states([vid])
                if isinstance(mapping, dict):
                    got = mapping.get(vid) or mapping.get(str(vid))
                    if isinstance(got, dict):
                        return got
        except Exception:
            pass
        return None

    def origin_offset_gvd(self) -> tuple[float, float, float]:
        base = list(self.config.get("origin_offset") or [0.0, 0.0, 0.0])
        while len(base) < 3:
            base.append(0.0)
        model = (self._model(self.vehicle) or "").lower()
        origins = self.config.get("vehicle_origins") if isinstance(self.config.get("vehicle_origins"), dict) else {}
        extra = origins.get(model) if model else None
        if extra:
            extra = list(extra) + [0.0, 0.0, 0.0]
            return (float(base[0]) + float(extra[0]), float(base[1]) + float(extra[1]), float(base[2]) + float(extra[2]))
        return (float(base[0]), float(base[1]), float(base[2]))

    def camera_mount(self, pos_gvd: list[float], dir_gvd: tuple[float, float, float], up_gvd: tuple[float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        """Apply origin_offset then convert GVD frame → BeamNG Camera vehicle space when enabled."""
        ox, oy, oz = self.origin_offset_gvd()
        px = float(pos_gvd[0]) + ox
        py = float(pos_gvd[1]) + oy
        pz = float(pos_gvd[2]) + oz
        cams = self.config.get("cameras") if isinstance(self.config.get("cameras"), dict) else {}
        if cams.get("convert_gvd_frame", True):
            return gvd_to_bng_vehicle(px, py, pz), gvd_to_bng_vehicle(*dir_gvd), gvd_to_bng_vehicle(*up_gvd)
        return (px, py, pz), dir_gvd, up_gvd

    def close(self) -> None:
        self.vehicle = None
        if self.bng is not None:
            try:
                self.bng.disconnect()
            except Exception:
                try:
                    self.bng.close()
                except Exception:
                    pass
        self.bng = None

    def _vid(self, vehicle: Any) -> str | None:
        if vehicle is None:
            return None
        for attr in ("vid", "vid_name", "name"):
            v = getattr(vehicle, attr, None)
            if v:
                return str(v)
        return None

    def _model(self, vehicle: Any) -> str | None:
        if vehicle is None:
            return None
        m = getattr(vehicle, "model", None)
        if m:
            return str(m)
        opts = getattr(vehicle, "options", None)
        if isinstance(opts, dict) and opts.get("model"):
            return str(opts["model"])
        return None

    def _log(self, msg: str) -> None:
        print(msg, flush=True)


def run_probe(config: dict[str, Any] | None = None) -> int:
    """Connect, list the player vehicle, poll electrics/pose/damage. No fake cameras."""
    cfg = apply_env_overrides(config if config is not None else load_tech_config())
    # Probe should not wait a full minute on a box with no Tech.
    cfg.setdefault("wait_vehicle_s", cfg.get("wait_vehicle_s", 15))
    session = TechSession(cfg)
    print("[GVD] tech probe: vision-only (RGB + ego kinematics). No LiDAR/radar/GPS.")
    if not session.connect(explicit=True):
        print(f"[GVD] tech probe FAIL: {session.note}")
        session.close()
        return 1
    session.attach_vehicle_sensors()
    data = session.poll()
    print(
        f"[GVD] tech probe OK vid={data.vid} model={data.model} "
        f"speed={data.speed_mps} steer_in={data.steering_input} damage={data.damage} "
        f"pose={'ok' if data.pose_ok else 'missing'} sensors={data.sensors}"
    )
    if data.pos:
        print(f"[GVD] pose pos={data.pos} dir={data.dir}")
    session.close()
    return 0
