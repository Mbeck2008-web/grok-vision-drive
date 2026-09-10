"""BeamNG.tech session: connect, player vehicle, electrics/damage/pose + GPS nav hint.

Camera RGB is handled by cameras.py. GPS is a coarse nav hint, not localization.
LiDAR / radar / AdvancedIMU attach when config/sensors.yaml enables them — Foxglove /
future fusion only; the corridor planner stays vision-only. Ultrasonic stays refused.
Coordinates: GVD vehicle frame is +X right, +Y forward, +Z up. BeamNGpy Camera/GPS
vehicle space is +X left, +Y backward, +Z up. Convert at attach.
"""

from __future__ import annotations

import inspect
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_SENSORS = ("ultrasonic", "idealradar", "ideal_radar")
EARTH_R_M = 6371000.0
# BeamNG maps have no real-world lat/lon. These put world (0, 0) on the Italy demo sphere.
DEFAULT_REF_LON = 8.8017
DEFAULT_REF_LAT = 53.0793


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


def wrap180(deg: float) -> float:
    x = (float(deg) + 180.0) % 360.0 - 180.0
    if x <= -180.0:
        return 180.0
    return x


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2.0) ** 2
    a = min(1.0, max(0.0, a))
    return 2.0 * EARTH_R_M * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing, degrees clockwise from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    x = math.sin(dlmb) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def heading_from_world_dir(direction: tuple[float, float, float] | None) -> float | None:
    """Heading clockwise from north, assuming world +X east / +Y north (GPS map origin)."""
    if direction is None:
        return None
    dx, dy = float(direction[0]), float(direction[1])
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def _flatten_gps_samples(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if raw is None:
        return out
    if isinstance(raw, dict):
        if any(k in raw for k in ("lon", "lat", "x", "y")):
            out.append(raw)
            return out
        data = raw.get("data")
        if data is not None and data is not raw:
            return _flatten_gps_samples(data)
        for v in raw.values():
            out.extend(_flatten_gps_samples(v))
        return out
    if isinstance(raw, (list, tuple)):
        for item in raw:
            out.extend(_flatten_gps_samples(item))
        return out
    data = getattr(raw, "data", None)
    if data is not None:
        return _flatten_gps_samples(data)
    return out


def latest_gps_reading(raw: Any) -> dict[str, Any] | None:
    """Pick the newest lon/lat sample from a BeamNGpy GPS.poll() payload (list, dict, or bulk)."""
    samples = _flatten_gps_samples(raw)
    if not samples:
        return None
    timed: list[tuple[float, dict[str, Any]]] = []
    for s in samples:
        t = _num(s.get("time"))
        if t is not None:
            timed.append((t, s))
    if timed:
        timed.sort(key=lambda p: p[0])
        return timed[-1][1]
    return samples[-1]


def empty_nav(*, note: str = "retail / no Tech GPS") -> dict[str, Any]:
    return {
        "mode": "missing",
        "drive_to_pin": False,
        "gps": None,
        "pin": None,
        "range_m": None,
        "bearing_deg": None,
        "bearing_rel_deg": None,
        "note": note,
    }


def nav_snapshot(data: VehicleData | None) -> dict[str, Any]:
    """gvd_state.json `nav` block. mode=hint never means the planner is routing to the pin."""
    if data is None or not data.connected:
        return empty_nav()
    gps_ok = data.lat is not None and data.lon is not None
    pin = None
    if data.pin_lat is not None and data.pin_lon is not None:
        pin = {"lat": data.pin_lat, "lon": data.pin_lon, "name": data.pin_name or ""}
    return {
        "mode": "hint" if gps_ok else "missing",
        "drive_to_pin": False,
        "gps": {
            "lat": data.lat,
            "lon": data.lon,
            "x": data.gps_x,
            "y": data.gps_y,
            "ok": gps_ok,
        },
        "pin": pin,
        "range_m": data.range_m,
        "bearing_deg": data.bearing_deg,
        "bearing_rel_deg": data.bearing_rel_deg,
        "note": "nav hint only; not localization; pin is not a route",
    }


def apply_nav_missing(missing: list[str] | None, nav: dict[str, Any] | None) -> list[str]:
    """Keep `nav` honest: Tech GPS is a hint; drive-to-pin stays missing until a planner exists."""
    out = [m for m in (missing or []) if m not in ("nav", "nav drive-to-pin")]
    mode = str((nav or {}).get("mode") or "missing")
    if mode == "hint":
        out.append("nav drive-to-pin")
    else:
        out.append("nav")
    return sorted(set(out))


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
    gps_env = os.environ.get("GVD_GPS", "").strip().lower()
    if gps_env:
        sensors = dict(out["sensors"]) if isinstance(out.get("sensors"), dict) else {}
        if gps_env in ("1", "true", "yes"):
            sensors["gps"] = True
        elif gps_env in ("0", "false", "no"):
            sensors["gps"] = False
        out["sensors"] = sensors
    pin_lat = os.environ.get("GVD_NAV_PIN_LAT")
    pin_lon = os.environ.get("GVD_NAV_PIN_LON")
    pin_name = os.environ.get("GVD_NAV_PIN_NAME")
    if pin_lat or pin_lon or pin_name:
        nav = dict(out["nav"]) if isinstance(out.get("nav"), dict) else {}
        if pin_lat:
            n = _num(pin_lat)
            if n is not None:
                nav["pin_lat"] = n
        if pin_lon:
            n = _num(pin_lon)
            if n is not None:
                nav["pin_lon"] = n
        if pin_name:
            nav["pin_name"] = pin_name
        out["nav"] = nav
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
    lat: float | None = None
    lon: float | None = None
    gps_x: float | None = None
    gps_y: float | None = None
    gps_time: float | None = None
    gps_heading_deg: float | None = None
    pin_lat: float | None = None
    pin_lon: float | None = None
    pin_name: str | None = None
    range_m: float | None = None
    bearing_deg: float | None = None
    bearing_rel_deg: float | None = None
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
        self._gps: Any = None
        self._lidar: Any = None
        self._radar: Any = None
        self._imu: Any = None
        self._extra_sensors_yaml: dict[str, Any] | None = None
        self._last_gps: tuple[float, float] | None = None
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

    def _extra_yaml(self) -> dict[str, Any]:
        cached = getattr(self, "_extra_sensors_yaml", None)
        if isinstance(cached, dict):
            return cached
        extra: dict[str, Any] = {}
        try:
            import yaml  # type: ignore

            p = ROOT / "config" / "sensors.yaml"
            if p.is_file():
                extra = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            extra = {}
        if not isinstance(extra, dict):
            extra = {}
        for key in ("lidar", "radar", "gps", "imu", "lidar_lua"):
            env = os.environ.get(f"GVD_{key.upper()}", "").strip().lower()
            if env in ("1", "true", "yes"):
                extra[key] = True
            elif env in ("0", "false", "no"):
                extra[key] = False
        adv = os.environ.get("GVD_ADVANCED_IMU", "").strip().lower()
        if adv in ("1", "true", "yes"):
            extra["advanced_imu"] = True
        elif adv in ("0", "false", "no"):
            extra["advanced_imu"] = False
        self._extra_sensors_yaml = extra
        return extra

    def _merged_sensor_flags(self) -> dict[str, Any]:
        flags = dict(self.config.get("sensors") or {}) if isinstance(self.config.get("sensors"), dict) else {}
        extra = self._extra_yaml()
        for key in ("lidar", "radar"):
            if extra.get(key):
                flags[key] = True
        if extra.get("advanced_imu"):
            flags["advanced_imu"] = True
        return flags

    def attach_vehicle_sensors(self) -> dict[str, bool]:
        """Electrics + Damage + GForces + GPS. Optional LiDAR/radar/AdvancedIMU (not for driving)."""
        flags = self._merged_sensor_flags()
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
        if flags.get("gps", True):
            self.attached["gps"] = self._attach_gps(vehicle, bng)
        if flags.get("lidar"):
            self.attached["lidar"] = self._attach_lidar(vehicle, bng)
        if flags.get("radar"):
            self.attached["radar"] = self._attach_radar(vehicle, bng)
        if flags.get("advanced_imu"):
            self.attached["advanced_imu"] = self._attach_advanced_imu(vehicle, bng)
        ok = [k for k, v in self.attached.items() if v]
        extra = [k for k in ("lidar", "radar", "advanced_imu") if self.attached.get(k)]
        extra_note = (
            f" extras={','.join(extra)} (Foxglove only; planner ignores them)"
            if extra
            else " (LiDAR/radar off; enable in config/sensors.yaml)"
        )
        self._log(
            f"[GVD] tech vehicle sensors: {', '.join(ok) if ok else 'none'} "
            f"(RGB cameras separate; GPS is a nav hint, not localization){extra_note}."
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

    def _attach_gps(self, vehicle: Any, bng: Any) -> bool:
        """CommBase GPS like Camera: GPS(name, bng, vehicle, pos=..., ref_lon=...)."""
        if self._gps is not None:
            return True
        try:
            from beamngpy.sensors import GPS  # type: ignore
        except Exception as e:
            self._log(f"[GVD] GPS class missing ({e}); nav lat/lon missing.")
            return False
        gps_cfg = self.config.get("gps") if isinstance(self.config.get("gps"), dict) else {}
        pos = list(gps_cfg.get("pos_m") or [0.0, 0.0, 1.7])
        while len(pos) < 3:
            pos.append(0.0)
        cams = self.config.get("cameras") if isinstance(self.config.get("cameras"), dict) else {}
        if cams.get("convert_gvd_frame", True):
            pos_bng = gvd_to_bng_vehicle(float(pos[0]), float(pos[1]), float(pos[2]))
        else:
            pos_bng = (float(pos[0]), float(pos[1]), float(pos[2]))
        ref_lon = _num(gps_cfg.get("ref_lon") if gps_cfg.get("ref_lon") is not None else gps_cfg.get("refLon"))
        ref_lat = _num(gps_cfg.get("ref_lat") if gps_cfg.get("ref_lat") is not None else gps_cfg.get("refLat"))
        if ref_lon is None:
            ref_lon = DEFAULT_REF_LON
        if ref_lat is None:
            ref_lat = DEFAULT_REF_LAT
        vis = bool(gps_cfg.get("visualised", False))
        update_s = float(gps_cfg.get("update_s") or 0.05)
        try:
            params = set()
            try:
                params = set(inspect.signature(GPS.__init__).parameters)
            except (TypeError, ValueError):
                params = set()
            kwargs: dict[str, Any] = {}
            if not params or "pos" in params:
                kwargs["pos"] = pos_bng
            if not params or "is_visualised" in params:
                kwargs["is_visualised"] = vis
            if "is_snapping_desired" in params:
                kwargs["is_snapping_desired"] = False
            if not params or "ref_lon" in params:
                kwargs["ref_lon"] = ref_lon
                kwargs["ref_lat"] = ref_lat
            elif "refLon" in params:
                kwargs["refLon"] = ref_lon
                kwargs["refLat"] = ref_lat
            if "gfx_update_time" in params:
                kwargs["gfx_update_time"] = update_s
            elif "requested_update_time" in params:
                kwargs["requested_update_time"] = update_s
            self._gps = GPS("gvd_gps", bng, vehicle, **kwargs)
            return True
        except Exception as e:
            self._log(f"[GVD] attach GPS failed: {e}")
            self._gps = None
            return False

    def _mount_pos_dir(
        self, pos_m: Any, dir_gvd: tuple[float, float, float] = (0.0, 1.0, 0.0)
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        pos = list(pos_m or [0.0, 0.0, 0.0])
        while len(pos) < 3:
            pos.append(0.0)
        cams = self.config.get("cameras") if isinstance(self.config.get("cameras"), dict) else {}
        if cams.get("convert_gvd_frame", True):
            return (
                gvd_to_bng_vehicle(float(pos[0]), float(pos[1]), float(pos[2])),
                gvd_to_bng_vehicle(*dir_gvd),
            )
        return (float(pos[0]), float(pos[1]), float(pos[2])), dir_gvd

    def _comm_kwargs(
        self,
        cls: Any,
        *,
        pos: tuple[float, float, float],
        direction: tuple[float, float, float] | None = None,
        update_s: float = 0.1,
        shared: bool | None = None,
        is_360: bool | None = None,
        vis: bool = False,
    ) -> dict[str, Any]:
        try:
            params = set(inspect.signature(cls.__init__).parameters)
        except (TypeError, ValueError):
            params = set()
        kwargs: dict[str, Any] = {}

        def put(value: Any, *names: str) -> None:
            for name in names:
                if not params or name in params:
                    kwargs[name] = value
                    return

        put(pos, "pos")
        if direction is not None:
            put(direction, "dir", "direction")
        put(update_s, "requested_update_time", "gfx_update_time")
        if shared is not None:
            put(shared, "is_using_shared_memory")
        if is_360 is not None:
            put(is_360, "is_360_mode", "is_360")
        put(vis, "is_visualised", "is_visualized")
        if "is_snapping_desired" in params:
            kwargs["is_snapping_desired"] = False
        return kwargs

    def _attach_lidar(self, vehicle: Any, bng: Any) -> bool:
        """Optional BeamNGpy Lidar. Foxglove / future fusion only — planner ignores it."""
        if self._lidar is not None:
            return True
        try:
            from beamngpy.sensors import Lidar  # type: ignore
        except Exception as e:
            self._log(f"[GVD] Lidar class missing ({e}); lidar stays missing.")
            return False
        mount = self._extra_yaml().get("lidar_mount") if isinstance(self._extra_yaml().get("lidar_mount"), dict) else {}
        pos, direction = self._mount_pos_dir(mount.get("pos_m") or [0.0, 0.0, 1.7])
        update_s = float(mount.get("update_s") or 0.1)
        shared = bool(mount.get("shared_memory", True))
        is_360 = bool(mount.get("is_360", True))
        try:
            kwargs = self._comm_kwargs(
                Lidar, pos=pos, direction=direction, update_s=update_s, shared=shared, is_360=is_360, vis=False
            )
            self._lidar = Lidar("gvd_lidar", bng, vehicle, **kwargs)
            return True
        except Exception as e:
            self._log(f"[GVD] attach Lidar failed: {e}")
            self._lidar = None
            return False

    def _attach_radar(self, vehicle: Any, bng: Any) -> bool:
        """Optional BeamNGpy Radar. Foxglove / future fusion only — planner ignores it."""
        if self._radar is not None:
            return True
        try:
            from beamngpy.sensors import Radar  # type: ignore
        except Exception as e:
            self._log(f"[GVD] Radar class missing ({e}); radar stays missing.")
            return False
        mount = self._extra_yaml().get("radar_mount") if isinstance(self._extra_yaml().get("radar_mount"), dict) else {}
        pos, direction = self._mount_pos_dir(mount.get("pos_m") or [0.0, 2.0, 0.5])
        update_s = float(mount.get("update_s") or 0.1)
        try:
            kwargs = self._comm_kwargs(Radar, pos=pos, direction=direction, update_s=update_s, vis=False)
            self._radar = Radar("gvd_radar", bng, vehicle, **kwargs)
            return True
        except Exception as e:
            self._log(f"[GVD] attach Radar failed: {e}")
            self._radar = None
            return False

    def _attach_advanced_imu(self, vehicle: Any, bng: Any) -> bool:
        """Optional BeamNGpy AdvancedIMU. Default IMU is GForces / Lua; this is opt-in."""
        if self._imu is not None:
            return True
        try:
            from beamngpy.sensors import AdvancedIMU  # type: ignore
        except Exception as e:
            self._log(f"[GVD] AdvancedIMU class missing ({e}).")
            return False
        mount = self._extra_yaml().get("imu_mount") if isinstance(self._extra_yaml().get("imu_mount"), dict) else {}
        pos, direction = self._mount_pos_dir(mount.get("pos_m") or [0.0, 0.0, 0.5])
        vis = bool(mount.get("visualised", False))
        try:
            kwargs = self._comm_kwargs(AdvancedIMU, pos=pos, direction=direction, update_s=0.05, vis=vis)
            self._imu = AdvancedIMU("gvd_imu", bng, vehicle, **kwargs)
            return True
        except Exception as e:
            self._log(f"[GVD] attach AdvancedIMU failed: {e}")
            self._imu = None
            return False

    def poll_lidar(self) -> Any:
        inst = self._lidar
        if inst is None:
            return None
        try:
            if hasattr(inst, "poll"):
                return inst.poll()
            return getattr(inst, "data", None)
        except Exception as e:
            self._log(f"[GVD] LiDAR poll failed: {e}")
            return None

    def poll_radar(self) -> Any:
        inst = self._radar
        if inst is None:
            return None
        try:
            if hasattr(inst, "poll"):
                return inst.poll()
            return getattr(inst, "data", None)
        except Exception as e:
            self._log(f"[GVD] Radar poll failed: {e}")
            return None

    def poll_advanced_imu(self) -> Any:
        inst = self._imu
        if inst is None:
            return None
        try:
            if hasattr(inst, "poll"):
                return inst.poll()
            return getattr(inst, "data", None)
        except Exception as e:
            self._log(f"[GVD] AdvancedIMU poll failed: {e}")
            return None

    def _nav_pin(self) -> tuple[float | None, float | None, str]:
        nav = self.config.get("nav") if isinstance(self.config.get("nav"), dict) else {}
        gps = self.config.get("gps") if isinstance(self.config.get("gps"), dict) else {}
        lat = _num(nav.get("pin_lat"))
        if lat is None:
            lat = _num(gps.get("pin_lat"))
        lon = _num(nav.get("pin_lon"))
        if lon is None:
            lon = _num(gps.get("pin_lon"))
        name = str(nav.get("pin_name") or gps.get("pin_name") or "").strip()
        return lat, lon, name

    def _poll_gps(self) -> dict[str, Any] | None:
        gps = self._gps
        if gps is None:
            return None
        try:
            raw = gps.poll() if hasattr(gps, "poll") else None
        except Exception as e:
            self._log(f"[GVD] GPS poll failed: {e}")
            return None
        return latest_gps_reading(raw)

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
        self._fill_nav(data)
        return data

    def _fill_nav(self, data: VehicleData) -> None:
        reading = self._poll_gps()
        if reading:
            data.lat = _num(reading.get("lat"))
            data.lon = _num(reading.get("lon"))
            data.gps_x = _num(reading.get("x"))
            data.gps_y = _num(reading.get("y"))
            data.gps_time = _num(reading.get("time"))
            data.sensors["gps"] = "ok" if data.lat is not None and data.lon is not None else "missing"
        elif self.attached.get("gps"):
            data.sensors["gps"] = "missing"
        pin_lat, pin_lon, pin_name = self._nav_pin()
        data.pin_lat, data.pin_lon = pin_lat, pin_lon
        data.pin_name = pin_name or None
        heading: float | None = None
        if data.lat is not None and data.lon is not None:
            if self._last_gps is not None:
                plat, plon = self._last_gps
                if haversine_m(plat, plon, data.lat, data.lon) > 0.5:
                    heading = bearing_deg(plat, plon, data.lat, data.lon)
            self._last_gps = (data.lat, data.lon)
        if heading is None:
            heading = heading_from_world_dir(data.dir)
        data.gps_heading_deg = heading
        if data.lat is not None and data.lon is not None and pin_lat is not None and pin_lon is not None:
            data.range_m = haversine_m(data.lat, data.lon, pin_lat, pin_lon)
            data.bearing_deg = bearing_deg(data.lat, data.lon, pin_lat, pin_lon)
            if heading is not None:
                data.bearing_rel_deg = wrap180(data.bearing_deg - heading)

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

    def _drop_handle(self, attr: str) -> None:
        inst = getattr(self, attr, None)
        if inst is None:
            return
        try:
            inst.remove()
        except Exception:
            try:
                inst.detach()
            except Exception:
                pass
        setattr(self, attr, None)

    def close(self) -> None:
        for attr in ("_gps", "_lidar", "_radar", "_imu"):
            self._drop_handle(attr)
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
    """Connect, list the player vehicle, poll electrics/pose/damage/GPS. No fake cameras."""
    cfg = apply_env_overrides(config if config is not None else load_tech_config())
    # Probe should not wait a full minute on a box with no Tech.
    cfg.setdefault("wait_vehicle_s", cfg.get("wait_vehicle_s", 15))
    session = TechSession(cfg)
    print(
        "[GVD] tech probe: RGB + ego kinematics + GPS nav hint. "
        "LiDAR/radar/AdvancedIMU attach only if config/sensors.yaml enables them "
        "(Foxglove / future fusion; planner stays vision-only). Pin is not a route."
    )
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
    if data.lat is not None and data.lon is not None:
        pin = f" pin={data.pin_name or '-'} range={data.range_m} bearing={data.bearing_deg}"
        print(f"[GVD] gps lat={data.lat} lon={data.lon}{pin} (hint only; not routing)")
    else:
        print("[GVD] gps missing (Tech GPS did not return lat/lon)")
    session.close()
    return 0
