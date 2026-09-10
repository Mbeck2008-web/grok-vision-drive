"""Optional LiDAR / radar / IMU / GPS bus. Not used for driving.

Retail (no BeamNGpy): IMU + pose from gvd_ego.json, GPS from world xy × ref lat/lon,
optional coarse Lua ray sweep in gvd_scan.json. Tech: BeamNGpy Lidar / Radar /
AdvancedIMU when enabled in config/sensors.yaml. Planner stays vision-only.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from python.runtime.state_io import atomic_write_json, gvd_docs_dir
from python.sensors.tech import (
    DEFAULT_REF_LAT,
    DEFAULT_REF_LON,
    EARTH_R_M,
    ROOT,
    VehicleData,
    apply_env_overrides,
    bearing_deg,
    empty_nav,
    haversine_m,
    load_tech_config,
    wrap180,
)

MAX_LIDAR_POINTS = 4000
MAX_RADAR_RETURNS = 64


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


def world_xy_to_ll(
    x: float,
    y: float,
    *,
    ref_lat: float = DEFAULT_REF_LAT,
    ref_lon: float = DEFAULT_REF_LON,
) -> tuple[float, float]:
    """BeamNG world metres → lat/lon using the same sphere origin as Tech GPS."""
    lat = ref_lat + (float(y) / EARTH_R_M) * (180.0 / math.pi)
    lon = ref_lon + (float(x) / (EARTH_R_M * math.cos(math.radians(ref_lat)) + 1e-12)) * (
        180.0 / math.pi
    )
    return lat, lon


def load_sensors_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (ROOT / "config" / "sensors.yaml")
    data: dict[str, Any] = {}
    try:
        import yaml  # type: ignore

        if cfg_path.is_file():
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return apply_sensor_env(data)


def apply_sensor_env(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    fox = dict(out["foxglove"]) if isinstance(out.get("foxglove"), dict) else {}
    env_fox = os.environ.get("GVD_FOXGLOVE", "").strip().lower()
    if env_fox in ("1", "true", "yes"):
        fox["enabled"] = True
    elif env_fox in ("0", "false", "no"):
        fox["enabled"] = False
    port = os.environ.get("GVD_FOXGLOVE_PORT")
    if port:
        try:
            fox["port"] = int(port)
        except ValueError:
            pass
    if fox:
        out["foxglove"] = fox
    for key, env in (("lidar", "GVD_LIDAR"), ("radar", "GVD_RADAR"), ("lidar_lua", "GVD_LIDAR_LUA")):
        raw = os.environ.get(env, "").strip().lower()
        if raw in ("1", "true", "yes"):
            out[key] = True
        elif raw in ("0", "false", "no"):
            out[key] = False
    return out


def scan_path() -> Path:
    return gvd_docs_dir() / "gvd_scan.json"


def sensors_snapshot_path() -> Path:
    return gvd_docs_dir() / "gvd_sensors.json"


@dataclass
class ImuSample:
    gx: float | None = None
    gy: float | None = None
    gz: float | None = None
    yaw_rate: float | None = None
    source: str = "missing"


@dataclass
class GpsFix:
    lat: float | None = None
    lon: float | None = None
    x: float | None = None
    y: float | None = None
    source: str = "missing"
    ok: bool = False


@dataclass
class LidarScan:
    points: list[tuple[float, float, float]] = field(default_factory=list)
    source: str = "missing"
    note: str = ""


@dataclass
class RadarScan:
    returns: list[dict[str, Any]] = field(default_factory=list)
    source: str = "missing"
    note: str = ""


@dataclass
class SensorBundle:
    """One tick of extras. Empty/missing is honest; never invented for the planner."""

    imu: ImuSample = field(default_factory=ImuSample)
    gps: GpsFix = field(default_factory=GpsFix)
    lidar: LidarScan = field(default_factory=LidarScan)
    radar: RadarScan = field(default_factory=RadarScan)
    foxglove: str = "off"
    drive_uses: str = "vision"
    lidar_lua: bool = False

    def health(self) -> dict[str, Any]:
        return {
            "imu": self.imu.source if self.imu.source != "missing" else "missing",
            "gps": self.gps.source if self.gps.ok else "missing",
            "lidar": self.lidar.source if self.lidar.points else "missing",
            "radar": self.radar.source if self.radar.returns else "missing",
            "foxglove": self.foxglove,
            "drive_uses": self.drive_uses,
            "lidar_lua": bool(self.lidar_lua),
        }

    def as_dict(self) -> dict[str, Any]:
        pts = self.lidar.points[:64]
        return {
            "drive_uses": self.drive_uses,
            "imu": {
                "gx": self.imu.gx,
                "gy": self.imu.gy,
                "gz": self.imu.gz,
                "yaw_rate": self.imu.yaw_rate,
                "source": self.imu.source,
            },
            "gps": {
                "lat": self.gps.lat,
                "lon": self.gps.lon,
                "x": self.gps.x,
                "y": self.gps.y,
                "ok": self.gps.ok,
                "source": self.gps.source,
            },
            "lidar": {
                "n": len(self.lidar.points),
                "source": self.lidar.source,
                "note": self.lidar.note,
                "preview": [{"x": p[0], "y": p[1], "z": p[2]} for p in pts],
            },
            "radar": {
                "n": len(self.radar.returns),
                "source": self.radar.source,
                "note": self.radar.note,
            },
            "foxglove": self.foxglove,
        }


def parse_lidar_points(raw: Any, *, limit: int = MAX_LIDAR_POINTS) -> list[tuple[float, float, float]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        for key in ("pointCloud", "points", "pos", "xyz"):
            if raw.get(key) is not None:
                return parse_lidar_points(raw.get(key), limit=limit)
        return []
    try:
        import numpy as np

        arr = np.asarray(raw, dtype=float)
        if arr.size == 0:
            return []
        if arr.ndim == 1:
            if arr.size % 3 != 0:
                arr = arr[: arr.size - (arr.size % 3)]
            arr = arr.reshape(-1, 3)
        elif arr.ndim >= 2:
            arr = arr.reshape(-1, arr.shape[-1])[:, :3]
        out: list[tuple[float, float, float]] = []
        for row in arr[:limit]:
            out.append((float(row[0]), float(row[1]), float(row[2])))
        return out
    except Exception:
        pass
    if isinstance(raw, (list, tuple)):
        out = []
        for item in raw:
            if len(out) >= limit:
                break
            if isinstance(item, dict):
                v = _vec3(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                try:
                    v = (float(item[0]), float(item[1]), float(item[2]))
                except (TypeError, ValueError):
                    v = None
            else:
                v = None
            if v is not None:
                out.append(v)
        return out
    return []


def parse_radar_returns(raw: Any, *, limit: int = MAX_RADAR_RETURNS) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        for key in ("returns", "detections", "tracks", "data"):
            if isinstance(raw.get(key), (list, tuple)):
                return parse_radar_returns(raw.get(key), limit=limit)
        dist = _num(raw.get("distance") or raw.get("range") or raw.get("dist"))
        if dist is not None:
            return [{"distance": dist, "az": _num(raw.get("az") or raw.get("azimuth")), "doppler": _num(raw.get("doppler") or raw.get("velocity"))}]
        return []
    if isinstance(raw, (list, tuple)):
        out: list[dict[str, Any]] = []
        for item in raw:
            if len(out) >= limit:
                break
            if isinstance(item, dict):
                out.append(
                    {
                        "distance": _num(item.get("distance") or item.get("range") or item.get("dist")),
                        "az": _num(item.get("az") or item.get("azimuth")),
                        "doppler": _num(item.get("doppler") or item.get("velocity")),
                    }
                )
        return out
    return []


def parse_imu_advanced(raw: Any) -> ImuSample | None:
    if raw is None:
        return None
    sample = raw
    if isinstance(raw, dict) and "accSmooth" not in raw and "acc" not in raw:
        timed = []
        for v in raw.values():
            if isinstance(v, dict) and ("accSmooth" in v or "acc" in v or "time" in v):
                t = _num(v.get("time"))
                timed.append((t if t is not None else -1.0, v))
        if timed:
            timed.sort(key=lambda p: p[0])
            sample = timed[-1][1]
        elif isinstance(raw.get("data"), dict):
            return parse_imu_advanced(raw.get("data"))
    if not isinstance(sample, dict):
        return None
    acc = sample.get("accSmooth") or sample.get("acc") or sample.get("acceleration")
    ang = sample.get("angVelSmooth") or sample.get("angVel") or sample.get("angular_velocity")
    gx = gy = gz = yaw = None
    if isinstance(acc, (list, tuple)) and len(acc) >= 3:
        gx, gy, gz = _num(acc[0]), _num(acc[1]), _num(acc[2])
    if isinstance(ang, (list, tuple)) and len(ang) >= 3:
        yaw = _num(ang[2])
    if gx is None and gy is None and gz is None:
        return None
    return ImuSample(gx=gx, gy=gy, gz=gz, yaw_rate=yaw, source="tech")


def gps_from_world(
    pos: tuple[float, float, float] | None,
    *,
    ref_lat: float,
    ref_lon: float,
    source: str,
) -> GpsFix:
    if pos is None:
        return GpsFix(source="missing")
    x, y = float(pos[0]), float(pos[1])
    lat, lon = world_xy_to_ll(x, y, ref_lat=ref_lat, ref_lon=ref_lon)
    return GpsFix(lat=lat, lon=lon, x=x, y=y, source=source, ok=True)


def ref_ll(cfg: dict[str, Any] | None = None) -> tuple[float, float]:
    tech = cfg if cfg is not None else load_tech_config()
    gps = tech.get("gps") if isinstance(tech.get("gps"), dict) else {}
    lat = _num(gps.get("ref_lat"))
    lon = _num(gps.get("ref_lon"))
    return (lat if lat is not None else DEFAULT_REF_LAT, lon if lon is not None else DEFAULT_REF_LON)


class ExtraSensors:
    """Poll Lua + Tech extras. Never feeds the planner."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config if config is not None else load_sensors_config()
        self.drive_uses = str(self.config.get("drive_uses") or "vision")

    def poll(
        self,
        *,
        vdata: VehicleData | None = None,
        ego_fb: Any = None,
        session: Any = None,
    ) -> SensorBundle:
        bundle = SensorBundle(drive_uses=self.drive_uses)
        bundle.lidar_lua = bool(self.config.get("lidar_lua"))
        ref_lat, ref_lon = ref_ll()
        want_imu = bool(self.config.get("imu", True))
        want_gps = bool(self.config.get("gps", True))

        if want_imu:
            bundle.imu = self._imu(vdata, ego_fb, session)
        if want_gps:
            bundle.gps = self._gps(vdata, ego_fb, ref_lat, ref_lon)
        bundle.lidar = self._lidar(session)
        bundle.radar = self._radar(session)
        return bundle

    def _imu(self, vdata: VehicleData | None, ego_fb: Any, session: Any) -> ImuSample:
        if session is not None:
            adv = parse_imu_advanced(getattr(session, "poll_advanced_imu", lambda: None)())
            if adv is not None:
                return adv
        if vdata is not None and (vdata.gx is not None or vdata.gy is not None):
            return ImuSample(
                gx=vdata.gx,
                gy=vdata.gy,
                gz=vdata.gz,
                yaw_rate=vdata.yaw_rate,
                source="tech",
            )
        if ego_fb is not None:
            gx = getattr(ego_fb, "gx", None)
            gy = getattr(ego_fb, "gy", None)
            gz = getattr(ego_fb, "gz", None)
            yr = getattr(ego_fb, "yaw_rate", None)
            if gx is not None or gy is not None or gz is not None or yr is not None:
                return ImuSample(gx=gx, gy=gy, gz=gz, yaw_rate=yr, source="lua")
        return ImuSample()

    def _gps(self, vdata: VehicleData | None, ego_fb: Any, ref_lat: float, ref_lon: float) -> GpsFix:
        if vdata is not None and vdata.lat is not None and vdata.lon is not None:
            return GpsFix(
                lat=vdata.lat,
                lon=vdata.lon,
                x=vdata.gps_x,
                y=vdata.gps_y,
                source="tech",
                ok=True,
            )
        pos = None
        if vdata is not None and vdata.pos is not None:
            pos = vdata.pos
        elif ego_fb is not None:
            pos = getattr(ego_fb, "pos", None)
        if pos is not None:
            return gps_from_world(pos, ref_lat=ref_lat, ref_lon=ref_lon, source="lua_pose")
        return GpsFix()

    def _lidar(self, session: Any) -> LidarScan:
        limit = int((self.config.get("lidar_mount") or {}).get("max_points") or MAX_LIDAR_POINTS)
        if self.config.get("lidar") and session is not None and getattr(session, "poll_lidar", None):
            pts = parse_lidar_points(session.poll_lidar(), limit=limit)
            if pts:
                return LidarScan(points=pts, source="tech")
        if self.config.get("lidar_lua"):
            lua_scan = _read_lua_scan()
            if lua_scan:
                return lua_scan
            return LidarScan(source="missing", note="Lua ray sweep enabled but gvd_scan.json empty")
        if self.config.get("lidar"):
            return LidarScan(source="missing", note="Tech Lidar enabled but no points")
        return LidarScan(note="off")

    def _radar(self, session: Any) -> RadarScan:
        if self.config.get("radar") and session is not None and getattr(session, "poll_radar", None):
            hits = parse_radar_returns(session.poll_radar())
            if hits:
                return RadarScan(returns=hits, source="tech")
        if self.config.get("radar"):
            return RadarScan(source="missing", note="Tech Radar enabled but no returns")
        return RadarScan(note="off")


def _read_lua_scan() -> LidarScan | None:
    p = scan_path()
    try:
        if not p.is_file():
            return None
        if time.time() - p.stat().st_mtime > 1.5:
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    pts = parse_lidar_points(data.get("points") or data.get("pointCloud"))
    if not pts:
        return None
    return LidarScan(points=pts, source="lua_raycast", note=str(data.get("note") or "coarse retail sweep"))


def nav_hint_from_bundle(
    bundle: SensorBundle,
    *,
    heading_deg: float | None = None,
    tech_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill gvd_state `nav` from extras GPS (Tech GPS or retail lua_pose). Pin is not a route."""
    if not bundle.gps.ok or bundle.gps.lat is None or bundle.gps.lon is None:
        return empty_nav(note="no GPS fix")
    cfg = apply_env_overrides(tech_cfg if tech_cfg is not None else load_tech_config())
    nav_cfg = cfg.get("nav") if isinstance(cfg.get("nav"), dict) else {}
    gps_cfg = cfg.get("gps") if isinstance(cfg.get("gps"), dict) else {}
    pin_lat = _num(nav_cfg.get("pin_lat"))
    if pin_lat is None:
        pin_lat = _num(gps_cfg.get("pin_lat"))
    pin_lon = _num(nav_cfg.get("pin_lon"))
    if pin_lon is None:
        pin_lon = _num(gps_cfg.get("pin_lon"))
    pin_name = str(nav_cfg.get("pin_name") or gps_cfg.get("pin_name") or "").strip()
    pin = None
    if pin_lat is not None and pin_lon is not None:
        pin = {"lat": pin_lat, "lon": pin_lon, "name": pin_name}
    range_m = bearing = rel = None
    if pin_lat is not None and pin_lon is not None:
        range_m = haversine_m(bundle.gps.lat, bundle.gps.lon, pin_lat, pin_lon)
        bearing = bearing_deg(bundle.gps.lat, bundle.gps.lon, pin_lat, pin_lon)
        if heading_deg is not None:
            rel = wrap180(bearing - heading_deg)
    note = "nav hint only; not localization; pin is not a route"
    if bundle.gps.source == "lua_pose":
        note = "pose-derived nav hint; not localization; pin is not a route"
    return {
        "mode": "hint",
        "drive_to_pin": False,
        "gps": {
            "lat": bundle.gps.lat,
            "lon": bundle.gps.lon,
            "x": bundle.gps.x,
            "y": bundle.gps.y,
            "ok": True,
        },
        "pin": pin,
        "range_m": range_m,
        "bearing_deg": bearing,
        "bearing_rel_deg": rel,
        "note": note,
    }


def write_sensors_snapshot(bundle: SensorBundle) -> None:
    try:
        atomic_write_json(sensors_snapshot_path(), bundle.as_dict(), indent=None)
    except Exception:
        pass
