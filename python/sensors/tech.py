"""BeamNG.tech session: connect, player vehicle, electrics/damage/pose + GPS nav hint.

Soft Esc (supervisor engaged=false): at most one ``vehicle.sensors.poll`` per
``SOFT_ESC_SENSOR_POLL_S``. Inside that window the grab reuses the last-good
ego map and does not send ``PollGPSGE``. A reused GPS sample is
``sensors["gps"]="stale"`` (same contract as the GPS window below): lat/lon
stay, and it is not a new fix. Skipped polls record 0 ms. Before that hold,
``poll`` reads ``read_engage_flag`` and the engage latch. A live
``gvd_engage.json`` refuses last-good on the rising edge, where the grab loop
has not called ``note_engaged`` yet. Engage keeps Tip #1: one
``vehicle.sensors.poll`` every grab. ``PollGPSGE`` stays on ``GPS_POLL_PERIOD_S``.
Camera RGB is handled by cameras.py. GPS is a coarse nav hint, not localization.
LiDAR / radar / AdvancedIMU attach when config/sensors.yaml enables them — Foxglove /
future fusion only; the corridor planner stays vision-only. Ultrasonic stays refused.
Coordinates: GVD vehicle frame is +X right, +Y forward, +Z up. BeamNGpy Camera/GPS
vehicle space is +X left, +Y backward, +Z up. Convert at attach.
"""

from __future__ import annotations

import ctypes
import inspect
import math
import os
import socket
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_SENSORS = ("ultrasonic", "idealradar", "ideal_radar")
EARTH_R_M = 6371000.0
# BeamNG maps have no real-world lat/lon. These put world (0, 0) on the Italy demo sphere.
DEFAULT_REF_LON = 8.8017
DEFAULT_REF_LAT = 53.0793
# GPS.poll sends PollGPSGE, a separate GE roundtrip. Nav is a hint, so grabs
# inside this window coalesce onto the last sample. That sample is
# sensors["gps"]="stale": lat/lon stay, and it is not a new fix.
GPS_POLL_PERIOD_S = 0.5
# Soft Esc only (engaged=false). One vehicle.sensors.poll per this window.
# A grab inside it reuses the last-good map and does not send PollGPSGE.
# A live gvd_engage.json or the engage latch ignores the window: one
# sensors.poll per grab (Tip #1), including the first grab of the rising edge.
SOFT_ESC_SENSOR_POLL_S = 0.2


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


def nav_heading_from_sources(
    vdata: Any = None,
    ego_fb: Any = None,
) -> float | None:
    """Heading for extras-GPS nav fill. Safe when ego_fb is missing (Tech attach).

    Wait for vehicle pose / GPS heading first; Lua ego_fb.dir is retail-only and
    stays None after beamngpy Camera attach. Never require engage.
    """
    if vdata is not None:
        heading = getattr(vdata, "gps_heading_deg", None)
        if heading is not None:
            try:
                return float(heading)
            except (TypeError, ValueError):
                pass
        heading = heading_from_world_dir(getattr(vdata, "dir", None))
        if heading is not None:
            return heading
    if ego_fb is not None:
        return heading_from_world_dir(getattr(ego_fb, "dir", None))
    return None


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


def _sensor_map(sensors: Any) -> dict[str, Any]:
    """Read the vehicle sensor container after poll().

    BeamNGpy ``Sensors.poll`` returns None and the container is not a dict.
    ``.items()`` / ``.data`` hold the updated Electrics, Damage, and GForces.
    A dict (tests, older containers) is copied by key.
    """
    if sensors is None:
        return {}
    if isinstance(sensors, dict):
        return {str(k): sensors[k] for k in sensors}
    try:
        items = sensors.items()
        return {str(k): v for k, v in items}
    except Exception:
        pass
    data = getattr(sensors, "data", None)
    if isinstance(data, dict):
        try:
            return {str(k): data[k] for k in data}
        except Exception:
            pass
    try:
        return {str(k): sensors[k] for k in list(sensors)}
    except Exception:
        return {}


def _copy_sensor_map(sensors: dict[str, Any]) -> dict[str, Any]:
    """Copy the poll map so last-good does not alias the live sensor dicts."""
    out: dict[str, Any] = {}
    for key, val in sensors.items():
        out[str(key)] = dict(val) if isinstance(val, dict) else val
    return out


def _soft_esc_map_has_ego(sensors: dict[str, Any]) -> bool:
    """True when the map still carries electrics or pose worth caching.

    An empty non-throwing ``sensors.poll`` must not arm the Soft Esc window
    or replace the previous sample.
    """
    if not isinstance(sensors, dict) or not sensors:
        return False
    if _value_has_electrics(sensors.get("electrics")):
        return True
    for key in ("state", "pose"):
        if _value_has_pose(sensors.get(key)):
            return True
    return False


def _value_has_electrics(el: Any) -> bool:
    if el is None:
        return False
    if isinstance(el, dict):
        return bool(el)
    data = getattr(el, "data", None)
    if isinstance(data, dict):
        return bool(data)
    return True


def _value_has_pose(st: Any) -> bool:
    if st is None:
        return False
    src = st if isinstance(st, dict) else getattr(st, "data", None)
    if not isinstance(src, dict):
        return False
    return src.get("pos") is not None or src.get("dir") is not None or src.get("forward") is not None


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
    gps_stale = (data.sensors or {}).get("gps") == "stale"
    pin = None
    if data.pin_lat is not None and data.pin_lon is not None:
        pin = {"lat": data.pin_lat, "lon": data.pin_lon, "name": data.pin_name or ""}
    note = "nav hint only; not localization; pin is not a route"
    if gps_ok and gps_stale:
        # ok means a fix is present. stale means this grab did not send PollGPSGE.
        note += "; gps is the last PollGPSGE sample (off grab cadence, not a new fix)"
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
        "note": note,
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
    py_pin = os.environ.get("GVD_BEAMNGPY_PIN", "").strip()
    if py_pin:
        out["beamngpy_pin"] = py_pin
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


# Research socket. Official binary is the install-root exe, not Bin64.
TECH_RESEARCH_PORT = 25252
TECH_ROOT_EXE = "BeamNG.tech.exe"
TECH_GFX = "dx11"
# BeamNGpy Hello recv. None blocks past the prove window; this cap REFUSEs.
TECH_SOCKET_TIMEOUT_S = 10.0


class TechHelloTimeout(TimeoutError):
    """Hello did not return before socket_timeout. Not a missing vehicle."""
# Human one-starter and the BeamNGpy gfx switch. BeamNGpy also adds -nosteam/-tport.
TECH_HUMAN_TAIL = ("-tcom", "-console", "-gfx", TECH_GFX)
# BeamNGpy minor line for the Tech build in use. 0.38 → 1.35, 0.39 → 1.36.
BEAMNGPY_FOR_TECH = {"1.35": "0.38", "1.36": "0.39"}


def wants_tech_attach() -> bool:
    return os.environ.get("GVD_BEAMNG", "").strip().lower() in ("1", "true", "yes")


def research_port_listening(host: str, port: int, *, timeout_s: float = 0.4) -> bool:
    """True when a TCP listener accepts on the Tech research port."""
    import socket

    try:
        with socket.create_connection((str(host), int(port)), timeout=timeout_s):
            return True
    except OSError:
        return False


def resolve_tech_launch(cfg: dict[str, Any], *, port_listening: bool) -> tuple[bool, str]:
    """One starter. Port already up → attach, even if launch was requested.

    ``True`` only when launch was asked and nothing is listening yet (BeamNGpy
    is the only starter). Never launch a second Tech onto a live port.
    """
    requested = bool(cfg.get("launch"))
    if port_listening:
        if requested:
            return (
                False,
                "research port already LISTENING; attach only "
                "(GVD_TECH_LAUNCH=1 would double-start BeamNG.tech)",
            )
        return False, "attach"
    if requested:
        return (
            True,
            "single launch path: port is down and GVD_TECH_LAUNCH=1. "
            "The bat must not also start BeamNG.tech.",
        )
    return False, "attach; research port is not LISTENING"


def hold_tech_process(bng: Any) -> None:
    """Esc/q must not quit Tech. BeamNGpy.close() sends quit_beamng when this is true."""
    if bng is None:
        return
    try:
        bng.quit_on_close = False
    except Exception:
        pass


def release_tech_beamngpy(bng: Any) -> None:
    """Disconnect the research socket. Do not close, quit, or kill BeamNG.tech / CrashSender."""
    if bng is None:
        return
    hold_tech_process(bng)
    disc = getattr(bng, "disconnect", None)
    if not callable(disc):
        return
    try:
        disc()
    except Exception:
        pass


class HelloAbandoned(Exception):
    """Stops a BeamNGpy ``open()`` reconnect loop after the Hello deadline.

    This is not an ``OSError``. ``_recv_exactly`` handles ``socket.error`` by
    calling ``reconnect()``; a raise from that call leaves ``open()``.
    """


_held_bng: Any = None
_held_lock = threading.Lock()
_HELLO_JOIN_GRACE_S = 0.05


def park_tech_beamngpy(bng: Any) -> None:
    """Keep one live BeamNGpy for the supervisor to reuse after the hold."""
    global _held_bng
    if bng is None:
        return
    with _held_lock:
        prev = _held_bng
        _held_bng = bng
    if prev is not None and prev is not bng:
        release_tech_beamngpy(prev)


def take_tech_beamngpy(port: int | None = None) -> Any:
    """Remove and return the parked session when its research port matches."""
    global _held_bng
    with _held_lock:
        bng = _held_bng
        _held_bng = None
    if bng is None:
        return None
    if port is not None:
        bport = getattr(bng, "port", None)
        try:
            mismatch = bport is not None and int(bport) != int(port)
        except (TypeError, ValueError):
            mismatch = True
        if mismatch:
            release_tech_beamngpy(bng)
            return None
    return bng


def drop_tech_beamngpy() -> None:
    """Disconnect a parked session. Idempotent."""
    release_tech_beamngpy(take_tech_beamngpy())


def _close_raw_socket(sock: Any) -> None:
    if not isinstance(sock, socket.socket):
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        sock.close()
    except Exception:
        pass


def _patch_reconnect(obj: Any) -> None:
    if obj is None or isinstance(obj, socket.socket):
        return
    if not callable(getattr(obj, "reconnect", None)):
        return

    def _stop(*_a: Any, **_k: Any) -> None:
        raise HelloAbandoned("Hello abandoned after socket_timeout")

    try:
        obj.reconnect = _stop
    except Exception:
        pass


def _hello_objects(bng: Any) -> list[Any]:
    objs: list[Any] = []
    seen: set[int] = set()

    def add(obj: Any) -> None:
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        objs.append(obj)

    add(bng)
    conn = getattr(bng, "connection", None)
    add(conn)
    add(getattr(bng, "skt", None))
    add(getattr(bng, "_hello_sock", None))
    if conn is not None:
        add(getattr(conn, "skt", None))
    return objs


def _hello_sockets(bng: Any) -> list[socket.socket]:
    found: list[socket.socket] = []
    seen: set[int] = set()

    def add(obj: Any) -> None:
        if isinstance(obj, socket.socket) and id(obj) not in seen:
            seen.add(id(obj))
            found.append(obj)

    for obj in _hello_objects(bng):
        add(obj)
        for attr in ("skt", "socket", "sock", "_socket", "_sock"):
            try:
                add(getattr(obj, attr, None))
            except Exception:
                pass
        try:
            for val in vars(obj).values():
                add(val)
        except Exception:
            pass
    return found


def _async_raise(worker: threading.Thread, exc_type: type[BaseException]) -> None:
    ident = worker.ident
    if not ident:
        return
    for tid_type in (ctypes.c_ulong, ctypes.c_long):
        try:
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid_type(ident), ctypes.py_object(exc_type))
        except Exception:
            return
        if res == 0:
            continue
        if res > 1:
            try:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(tid_type(ident), None)
            except Exception:
                pass
        return


def _abandon_hello(bng: Any, worker: threading.Thread) -> None:
    """Stop a daemon still inside ``bng.open()`` so it cannot reconnect forever.

    BeamNGpy 1.35.1/1.36 treat a closed socket as ``socket.error`` and call
    ``reconnect()``. 1.35.1 then sets the new socket to blocking. Patch
    ``reconnect`` before closing so that follow-up raises out of ``open()``.
    """
    try:
        setattr(bng, "_gvd_abandon", True)
    except Exception:
        pass
    for obj in _hello_objects(bng):
        _patch_reconnect(obj)
    for sock in _hello_sockets(bng):
        _close_raw_socket(sock)
    worker.join(_HELLO_JOIN_GRACE_S)
    if worker.is_alive():
        _async_raise(worker, HelloAbandoned)
        worker.join(_HELLO_JOIN_GRACE_S)


def human_one_starter(home: str | None) -> str:
    """Install-root ``BeamNG.tech.exe -tcom -console -gfx dx11``."""
    exe = TECH_ROOT_EXE
    if home and str(home).strip():
        exe = str(Path(home) / TECH_ROOT_EXE)
    return " ".join((exe, *TECH_HUMAN_TAIL))


def beamngpy_launch_argv(home: str | None, port: int, user: str | None = None) -> list[str]:
    """Argv BeamNGpy 1.36 builds for the root exe with ``gfx=dx11``.

    Matches ``_prepare_call`` plus ``open()``'s default ``-tcom-listen-ip``.
    Used when a launch fails before ``last_command_line`` is set. Not a Bin64 path.
    """
    exe = TECH_ROOT_EXE
    if home and str(home).strip():
        exe = str(Path(home) / TECH_ROOT_EXE)
    call = [
        exe,
        "-nosteam",
        "-tcom",
        "-tport",
        str(int(port)),
        "-console",
        "-tcom-listen-ip",
        "127.0.0.1",
        "-gfx",
        TECH_GFX,
    ]
    if user and str(user).strip():
        call.extend(("-userpath", str(user)))
    return call


def real_launch_argv(bng: Any, home: str | None, port: int, user: str | None) -> str:
    """Command line BeamNGpy actually used, else the root-exe reconstruction.

    ``get_launch_arguments()`` ignores ``self.binary`` when ``last_command_line``
    is empty, so it is not the failure argv.
    """
    line = getattr(bng, "last_command_line", None) if bng is not None else None
    if isinstance(line, (list, tuple)) and line:
        return " ".join(str(part) for part in line)
    if isinstance(line, str) and line.strip():
        return line.strip()
    return " ".join(beamngpy_launch_argv(home, port, user))


def resolve_socket_timeout(cfg: dict[str, Any] | None = None) -> float:
    """Attach/Hello cap in seconds. Env, then yaml, then ``TECH_SOCKET_TIMEOUT_S``."""
    raw: Any = os.environ.get("GVD_TECH_SOCKET_TIMEOUT", "").strip()
    if not raw and cfg is not None:
        raw = cfg.get("socket_timeout")
    if raw is None or raw == "":
        raw = TECH_SOCKET_TIMEOUT_S
    try:
        val = float(raw)
    except (TypeError, ValueError):
        val = TECH_SOCKET_TIMEOUT_S
    if val <= 0:
        return TECH_SOCKET_TIMEOUT_S
    return val


def _force_root_dx11(bng: Any, socket_timeout: float) -> None:
    """Root exe, dx11, and a finite Hello socket timeout. No Bin64 default."""
    try:
        bng.binary = TECH_ROOT_EXE
    except Exception:
        pass
    try:
        bng.gfx = TECH_GFX
    except Exception:
        pass
    try:
        bng.socket_timeout = socket_timeout
    except Exception:
        pass


def _hello_refuse_line(timeout: float) -> None:
    print(
        f"[GVD] phase=Hello REFUSE: socket_timeout {timeout:g}s. Not a missing vehicle.",
        flush=True,
    )


def _open_with_deadline(bng: Any, launch: bool, timeout: float, host: str, port: int) -> None:
    """Return when ``bng.open`` finishes, or REFUSE at ``timeout``.

    BeamNGpy 1.35.1 has no ``socket_timeout`` argument, so a silent port blocks
    inside ``open()``. 1.36 accepts the argument and may retry once. The join
    is the cap for both pins. On expiry the helper logs one REFUSE, stops the
    ``open()`` thread, and disconnects. Any other ``open()`` error disconnects
    too, so a Hello version mismatch cannot leave the socket up.
    """
    box: dict[str, BaseException] = {}

    def _run() -> None:
        try:
            bng.open(launch=bool(launch))
        except BaseException as exc:
            box["exc"] = exc

    worker = threading.Thread(target=_run, name="gvd-hello", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        _hello_refuse_line(timeout)
        _abandon_hello(bng, worker)
        release_tech_beamngpy(bng)
        raise TechHelloTimeout(f"Hello timeout after {timeout:g}s on {host}:{int(port)}")
    exc = box.get("exc")
    if exc is None:
        return
    release_tech_beamngpy(bng)
    if isinstance(exc, TechHelloTimeout):
        _hello_refuse_line(timeout)
        raise exc
    if isinstance(exc, TimeoutError):
        _hello_refuse_line(timeout)
        raise TechHelloTimeout(f"Hello timeout after {timeout:g}s on {host}:{int(port)}") from exc
    raise exc


def open_tech_beamngpy(
    host: str,
    port: int,
    *,
    home: str | None,
    user: str | None,
    launch: bool,
    socket_timeout: float | None = None,
) -> Any:
    """Connect with quit_on_close=False. Launch uses the root exe and ``-gfx dx11``.

    A listening research port forces ``launch=False``. ``bng.open()`` is joined
    for ``socket_timeout`` seconds on BeamNGpy 1.35 and 1.36. Expiry is
    ``TechHelloTimeout`` (one ``phase=Hello REFUSE``), not a missing vehicle.
    A launch failure prints the real argv. Attach failures do not print a launch line.
    """
    from beamngpy import BeamNGpy  # type: ignore

    if socket_timeout is None:
        timeout = resolve_socket_timeout(load_tech_config())
    else:
        timeout = float(socket_timeout)
    if timeout <= 0:
        timeout = TECH_SOCKET_TIMEOUT_S
    listening = research_port_listening(host, int(port))
    if listening:
        launch = False
    print(
        f"[GVD] phase=attach {host}:{int(port)} "
        f"{'LISTENING' if listening else 'down'} "
        f"launch={launch} socket_timeout={timeout:g}s",
        flush=True,
    )
    kwargs: dict[str, Any] = {
        "quit_on_close": False,
        "binary": TECH_ROOT_EXE,
        "gfx": TECH_GFX,
        "socket_timeout": timeout,
    }
    if home:
        kwargs["home"] = home
    if user:
        kwargs["user"] = user
    bng = None
    try:
        try:
            bng = BeamNGpy(host, int(port), **kwargs)
        except TypeError:
            slim: dict[str, Any] = {}
            if home:
                slim["home"] = home
            if user:
                slim["user"] = user
            try:
                bng = BeamNGpy(host, int(port), quit_on_close=False, **slim)
            except TypeError:
                bng = BeamNGpy(host, int(port), **slim) if slim else BeamNGpy(host, int(port))
        _force_root_dx11(bng, timeout)
        hold_tech_process(bng)
        print("[GVD] phase=Hello", flush=True)
        _open_with_deadline(bng, bool(launch), timeout, host, int(port))
        print("[GVD] phase=Hello ok", flush=True)
        hold_tech_process(bng)
        _force_root_dx11(bng, timeout)
        return bng
    except TechHelloTimeout:
        raise
    except Exception:
        if launch:
            argv = real_launch_argv(bng, home, port, user)
            print(f"[GVD] beamngpy launch failed. argv: {argv}", flush=True)
        raise


def beamngpy_minor(version: str | None) -> str | None:
    if version is None:
        return None
    parts = str(version).strip().split(".")
    if len(parts) < 2:
        return None
    try:
        return f"{int(parts[0])}.{int(parts[1])}"
    except ValueError:
        return None


def beamngpy_pin_ok(installed: str | None, pin: str | None) -> bool:
    """True when the installed BeamNGpy minor matches the Tech-build pin."""
    want = beamngpy_minor(pin)
    got = beamngpy_minor(installed)
    if not want or not got:
        return False
    return want == got


def installed_beamngpy_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("beamngpy")
    except Exception:
        pass
    try:
        import beamngpy  # type: ignore

        raw = getattr(beamngpy, "__version__", None)
        return str(raw) if raw else None
    except Exception:
        return None


def tech_key_status(home: str | None) -> bool | None:
    """Diagnose only. True when install-root ``tech.key`` is non-empty.

    A Bin64 ``tech.key`` does not count. Empty or whitespace is False.
    None when home is unset. This does not gate the wait hold.
    """
    if home is None or not str(home).strip():
        return None
    key = Path(home) / "tech.key"
    try:
        if not key.is_file():
            return False
        text = key.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return bool(text.strip())


def tech_mod_roots() -> list[Path]:
    """Unpacked GVD mod: legacy ``BeamNG.tech\\current`` first, then nested."""
    from python.runtime.paths import local_appdata_dir

    la = local_appdata_dir()
    if la is None:
        return []
    return [
        la / "BeamNG.tech" / "current" / "mods" / "unpacked" / "gvd",
        la / "BeamNG" / "BeamNG.tech" / "current" / "mods" / "unpacked" / "gvd",
    ]


def tech_mod_present() -> bool:
    marker = Path("lua") / "ge" / "extensions" / "gvd" / "main.lua"
    for root in tech_mod_roots():
        try:
            if (root / marker).is_file():
                return True
        except OSError:
            continue
    return False


def probe_spawned_vehicle(
    host: str,
    port: int,
    *,
    socket_timeout: float | None = None,
    retain: bool = False,
) -> bool | None:
    """Attach-only vehicle poll. None when beamngpy is missing. Never launches or kills.

    ``TechHelloTimeout`` propagates. It is not an empty vehicle list.
    ``retain=True`` parks the live socket when a vehicle is present so the
    supervisor can reuse that one Hello.
    """
    try:
        from beamngpy import BeamNGpy  # type: ignore  # noqa: F401
    except Exception:
        return None
    bng = None
    parked = False
    try:
        bng = open_tech_beamngpy(
            host, port, home=None, user=None, launch=False, socket_timeout=socket_timeout
        )
        vehicles = None
        getter = getattr(bng, "get_current_vehicles", None)
        if callable(getter):
            vehicles = getter()
        else:
            api = getattr(bng, "vehicles", None)
            info = getattr(api, "get_current", None) if api is not None else None
            if callable(info):
                vehicles = info()
        if isinstance(vehicles, dict):
            found = len(vehicles) > 0
        else:
            found = bool(vehicles)
        if retain and found:
            park_tech_beamngpy(bng)
            parked = True
        return found
    except TechHelloTimeout:
        raise
    except Exception:
        return False
    finally:
        if not parked:
            release_tech_beamngpy(bng)


@dataclass(frozen=True)
class TechHoldGate:
    """Wait gate before vision / unique-frame Hz. Launch policy is separate."""

    port_listening: bool
    mod_present: bool
    vehicle_spawned: bool
    lua_fresh: bool
    buses_same: bool
    link: str
    pin_ok: bool
    note: str
    python_bus: str
    lua_bus: str
    lua_age_s: float | None
    beamngpy_pin: str
    beamngpy_version: str
    tech_key: bool | None
    will_launch: bool
    host: str
    port: int
    hello: str = "skipped"
    lua_folder: str = ""

    @property
    def ok(self) -> bool:
        """Port LISTENING + mod + vehicle + fresh lua_bus + buses_same. Not the pin."""
        return bool(
            self.port_listening
            and self.mod_present
            and self.vehicle_spawned
            and self.lua_fresh
            and self.buses_same
            and self.link == "ok"
        )

    @property
    def proceed(self) -> bool:
        """Wait gate plus the BeamNGpy pin for the Tech build in use."""
        return self.ok and self.pin_ok

    @property
    def line(self) -> str:
        age = "--" if self.lua_age_s is None else f"{self.lua_age_s:.3f}s"
        if self.hello == "timeout":
            vehicle = "Hello-timeout"
        else:
            vehicle = "yes" if self.vehicle_spawned else "no"
        folder = (self.lua_folder or "").strip() or "--"
        return (
            f"[GVD] phase=wait-gate port={'LISTENING' if self.port_listening else 'down'} "
            f"mod={'yes' if self.mod_present else 'no'} "
            f"vehicle={vehicle} "
            f"lua_bus={'fresh' if self.lua_fresh else 'stale'} folder={folder} age={age} "
            f"buses_same={'yes' if self.buses_same else 'no'} link={self.link} "
            f"pin={self.beamngpy_pin or '--'} beamngpy={self.beamngpy_version or 'missing'} "
            f"launch={'yes' if self.will_launch else 'attach'}"
        )


def tech_hold_gate(
    config: dict[str, Any] | None = None,
    *,
    port_up: bool | None = None,
    vehicle_spawned: bool | None = None,
    mod_present: bool | None = None,
    beamngpy_version: str | None = None,
    retain: bool = False,
) -> TechHoldGate:
    """Preflight before cameras / unique-frame Hz.

    Wait gate (unchanged): research port LISTENING, unpacked GVD mod, a spawned
    vehicle, fresh lua_bus (age < 1s), and buses_same(python_bus, lua_bus).
    ``tech.key`` and the Tech user path are status only and are not part of this gate.
    ``retain=True`` parks the Hello socket when the gate proceeds so
    ``TechSession.connect`` reuses that one session.
    """
    from python.runtime.paths import bus_identity, buses_same_folder, read_live_lua_bus, youngest_lua_handshake

    drop_tech_beamngpy()
    cfg = apply_env_overrides(config if config is not None else load_tech_config())
    host = str(cfg.get("host") or "localhost")
    port = int(cfg.get("port") or TECH_RESEARCH_PORT)
    listening = research_port_listening(host, port) if port_up is None else bool(port_up)
    will_launch, _launch_note = resolve_tech_launch(cfg, port_listening=listening)

    if mod_present is None:
        mod_ok = tech_mod_present()
    else:
        mod_ok = bool(mod_present)

    vehicle_note = ""
    hello = "skipped"
    hello_note = ""
    if vehicle_spawned is None:
        if not listening:
            veh_ok = False
            print(
                f"[GVD] phase=attach {host}:{port} down launch={will_launch} "
                f"socket_timeout={resolve_socket_timeout(cfg):g}s",
                flush=True,
            )
            print("[GVD] phase=Hello skipped", flush=True)
        else:
            try:
                probed = probe_spawned_vehicle(
                    host,
                    port,
                    socket_timeout=resolve_socket_timeout(cfg),
                    retain=retain,
                )
            except TechHelloTimeout as exc:
                veh_ok = False
                hello = "timeout"
                hello_note = str(exc)
            else:
                hello = "ok"
                if probed is None:
                    veh_ok = False
                    vehicle_note = "beamngpy not available"
                else:
                    veh_ok = bool(probed)
    else:
        veh_ok = bool(vehicle_spawned)

    ident = bus_identity()
    live, live_note = read_live_lua_bus()
    _raw, age = youngest_lua_handshake()
    fresh = live is not None and live_note == "ok"
    lua_folder = ident.lua_bus_s if fresh else (str(_raw).strip() if _raw else "")
    same = bool(
        buses_same_folder(ident.python_bus, ident.lua_bus)
        and live is not None
        and buses_same_folder(ident.python_bus, live)
    )
    link = "ok" if fresh and same else "MISMATCH"
    pin = str(cfg.get("beamngpy_pin") or "").strip()
    installed = beamngpy_version if beamngpy_version is not None else installed_beamngpy_version()
    pin_ok = beamngpy_pin_ok(installed, pin)
    home = str(cfg.get("home") or "").strip() or None
    key = tech_key_status(home)

    reasons: list[str] = []
    if not listening:
        reasons.append(f"research port {host}:{port} not LISTENING")
    if not mod_ok:
        reasons.append(r"GVD mod missing under Tech current\mods\unpacked\gvd")
    if hello == "timeout":
        reasons.append(hello_note or "Hello timeout")
    elif not veh_ok:
        reasons.append(vehicle_note or "no vehicle spawned")
    if not fresh:
        reasons.append(live_note or "lua_bus missing or stale")
    elif not same:
        reasons.append("python_bus and lua_bus differ")
    if not pin_ok:
        got = installed or "missing"
        tech = BEAMNGPY_FOR_TECH.get(beamngpy_minor(pin) or "", "")
        extra = f" (Tech {tech})" if tech else ""
        reasons.append(f"beamngpy {got} != pin {pin or '--'}{extra}")
    note = "ok" if not reasons else "; ".join(reasons)
    gate = TechHoldGate(
        port_listening=listening,
        mod_present=mod_ok,
        vehicle_spawned=veh_ok,
        lua_fresh=fresh,
        buses_same=same,
        link=link,
        pin_ok=pin_ok,
        note=note,
        python_bus=ident.python_bus_s,
        lua_bus=ident.lua_bus_s if fresh else "",
        lua_age_s=age,
        beamngpy_pin=pin,
        beamngpy_version=str(installed or ""),
        tech_key=key,
        will_launch=will_launch,
        host=host,
        port=port,
        hello=hello,
        lua_folder=lua_folder,
    )
    if not (retain and gate.proceed):
        drop_tech_beamngpy()
    print(gate.line, flush=True)
    return gate


def run_tech_hold(config: dict[str, Any] | None = None) -> int:
    """Print the wait gate and return 0 only when vision/Hz is allowed to start."""
    gate = tech_hold_gate(config)
    cfg = apply_env_overrides(config if config is not None else load_tech_config())
    home = str(cfg.get("home") or "").strip() or None
    print(f"[GVD] one starter: {human_one_starter(home)}", flush=True)
    if gate.tech_key is True:
        print("[GVD] tech.key: non-empty install-root tech.key (status only, not a hold gate).", flush=True)
    elif gate.tech_key is False:
        print("[GVD] tech.key: missing or empty at the install root (status only, not a hold gate).", flush=True)
    else:
        print("[GVD] tech.key: not confirmed (set BNG_HOME). Status only, not a hold gate.", flush=True)
    if gate.proceed:
        print("[GVD] tech-hold OK. Unique-frame Hz is not measured here.", flush=True)
        return 0
    print(f"[GVD] REFUSE: {gate.note}", flush=True)
    print(
        "[GVD] Supervisor stays down. Do not kill BeamNG.tech or CrashSender. "
        f"One starter: {human_one_starter(home)} already running, mod loaded, vehicle spawned, then attach.",
        flush=True,
    )
    return 1


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
    # Soft Esc segment timers. 0 / False when that call did not run this poll.
    sensors_poll_ms: float = 0.0
    poll_gps_ms: float = 0.0
    poll_gps_sent: bool = False

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
        self._gps_reading: dict[str, Any] | None = None
        self._gps_mono: float | None = None
        self._sensors_poll_ms = 0.0
        self._poll_gps_ms = 0.0
        self._poll_gps_sent = False
        self._sensors_poll_ok = False
        self._sensors_poll_mono: float | None = None
        self._sensors_poll_vehicle_id: int | None = None
        self._last_vehicle_data: VehicleData | None = None
        self._last_sensor_map: dict[str, Any] | None = None
        self.note = ""

    def connect(self, *, explicit: bool = True) -> bool:
        """Open BeamNGpy. explicit=True when the user asked for --backend beamngpy.

        Prefer attach. If the research port is already LISTENING, launch is forced
        off so a second Tech is not started. A session parked by the wait gate is
        reused (one Hello). The socket is opened with quit_on_close=False; Esc/q
        disconnects and leaves the Tech process running.
        """
        if not explicit and not wants_tech_attach():
            self.note = "set GVD_BEAMNG=1 or --backend beamngpy to attach"
            return False

        host = str(self.config.get("host") or "localhost")
        port = int(self.config.get("port") or TECH_RESEARCH_PORT)
        home = str(self.config.get("home") or "").strip() or None
        user = str(self.config.get("user") or "").strip() or None
        listening = research_port_listening(host, port)
        launch, launch_note = resolve_tech_launch(self.config, port_listening=listening)
        if launch_note not in ("attach",):
            self._log(f"[GVD] {launch_note}")
        if not listening:
            drop_tech_beamngpy()
        if not listening and not launch:
            self.note = f"research port {host}:{port} not LISTENING"
            self._log(
                f"[GVD] {self.note}. Start BeamNG.tech once: {human_one_starter(home)} "
                "then attach. This path does not start a second Tech."
            )
            return False
        parked = take_tech_beamngpy(port) if listening else None
        if parked is not None:
            hold_tech_process(parked)
            self.bng = parked
            self._log(f"[GVD] phase=Hello reuse {host}:{int(port)}")
        else:
            try:
                bng = open_tech_beamngpy(
                    host,
                    port,
                    home=home,
                    user=user,
                    launch=launch,
                    socket_timeout=resolve_socket_timeout(self.config),
                )
                self.bng = bng
            except TechHelloTimeout as e:
                # The open helper already logged this Hello failure once.
                self.note = str(e)
                return False
            except Exception as e:
                self.note = f"beamngpy connect failed ({e})"
                self._log(f"[GVD] {self.note}. Research port {host}:{port}.")
                return False

        wait_s = float(self.config.get("wait_vehicle_s") or 0.0)
        vehicle = self._wait_vehicle(wait_s)
        if vehicle is None:
            self.note = "no vehicle to attach (spawn one in Tech, then retry)"
            self._log(f"[GVD] beamngpy: {self.note}.")
            release_tech_beamngpy(self.bng)
            self.bng = None
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
        wanted = [k for k in ("lidar", "radar", "advanced_imu") if flags.get(k)]
        if extra:
            extra_note = f" extras={','.join(extra)} (Foxglove only; planner ignores them)"
        elif wanted:
            extra_note = f" extras requested={','.join(wanted)} but missing (planner stays vision-only)"
        else:
            extra_note = " (LiDAR/radar off; enable in config/sensors.yaml)"
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

    def _poll_gps(self) -> tuple[dict[str, Any] | None, str]:
        """Return ``(reading, status)`` with status ``ok`` / ``stale`` / ``missing``.

        ``ok``: this call sent PollGPSGE and stored a lat/lon sample.
        ``stale``: this grab skipped PollGPSGE (inside ``GPS_POLL_PERIOD_S``)
        or the new poll failed; lat/lon are the previous sample, not a new fix.
        ``missing``: no sample. A failed poll still arms the period so a down
        GPS does not retry on every grab.
        """
        gps = self._gps
        self._poll_gps_ms = 0.0
        self._poll_gps_sent = False
        if gps is None:
            return None, "missing"
        now = time.monotonic()
        due = self._gps_mono is None or (now - self._gps_mono) >= GPS_POLL_PERIOD_S
        if not due:
            if self._gps_reading is not None:
                return self._gps_reading, "stale"
            return None, "missing"
        self._gps_mono = now
        reading: dict[str, Any] | None = None
        sent = hasattr(gps, "poll")
        t0 = time.perf_counter()
        try:
            raw = gps.poll() if sent else None
            reading = latest_gps_reading(raw)
        except Exception as e:
            self._log(f"[GVD] GPS poll failed: {e}")
            reading = None
        finally:
            if sent:
                self._poll_gps_ms = (time.perf_counter() - t0) * 1000.0
                self._poll_gps_sent = True
        if (
            reading is not None
            and _num(reading.get("lat")) is not None
            and _num(reading.get("lon")) is not None
        ):
            self._gps_reading = reading
            return reading, "ok"
        if self._gps_reading is not None:
            return self._gps_reading, "stale"
        return None, "missing"

    def poll(self) -> VehicleData:
        self._sensors_poll_ms = 0.0
        self._poll_gps_ms = 0.0
        self._poll_gps_sent = False
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
            self._clear_soft_esc_cache()
            return data
        if self._soft_esc_hold(vehicle):
            return self._coalesced_vehicle_data(vehicle)
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
        data.sensors_poll_ms = float(self._sensors_poll_ms)
        data.poll_gps_ms = float(self._poll_gps_ms)
        data.poll_gps_sent = bool(self._poll_gps_sent)
        from python.control.actuate import touch_vehicle_sensor_snap

        touch_vehicle_sensor_snap(vehicle)
        self._remember_sensor_poll(vehicle, data, sensors)
        return data

    def _soft_esc_engage_blocks_hold(self) -> bool:
        """True when Engage requires a real ``sensors.poll`` on this grab.

        The grab loop calls ``note_engaged`` after ``poll_vehicle``, so the
        latch is still false on the rising edge. Lua has already written
        ``gvd_engage.json``. Either signal refuses Soft Esc-hold.
        """
        from python.control.actuate import read_engage_flag, soft_esc_sensors_every_tick

        if soft_esc_sensors_every_tick():
            return True
        return bool(read_engage_flag(default=False))

    def _soft_esc_hold(self, vehicle: Any) -> bool:
        """True when this Soft Esc grab must not send ``vehicle.sensors.poll``.

        Engage (live flag or latch) never holds. With no last-good map yet, the
        grab polls. A different vehicle object does not reuse the previous car.
        """
        if self._soft_esc_engage_blocks_hold():
            return False
        if (
            self._last_vehicle_data is None
            or self._last_sensor_map is None
            or self._sensors_poll_mono is None
            or self._sensors_poll_vehicle_id != id(vehicle)
        ):
            return False
        return (time.monotonic() - self._sensors_poll_mono) < SOFT_ESC_SENSOR_POLL_S

    def _remember_sensor_poll(self, vehicle: Any, data: VehicleData, sensors: dict[str, Any]) -> None:
        """Arm the Soft Esc window only after a poll that still has ego data.

        An empty map keeps the previous sample and does not refresh the window.
        The stored map is a copy, not the live sensor container.
        """
        if not self._sensors_poll_ok:
            return
        if not _soft_esc_map_has_ego(sensors):
            return
        self._sensors_poll_mono = time.monotonic()
        self._sensors_poll_vehicle_id = id(vehicle)
        self._last_vehicle_data = replace(data, sensors=dict(data.sensors))
        self._last_sensor_map = _copy_sensor_map(sensors)

    def _expire_soft_esc_window(self) -> None:
        """Drop Soft Esc eligibility after a thrown ``sensors.poll``.

        The previous sample stays, but the next grab must poll again.
        """
        self._sensors_poll_ok = False
        self._sensors_poll_mono = None
        self._sensors_poll_vehicle_id = None

    def _clear_soft_esc_cache(self) -> None:
        """Drop last-good, the window clock, and the vehicle id together."""
        self._expire_soft_esc_window()
        self._last_vehicle_data = None
        self._last_sensor_map = None

    def _coalesced_vehicle_data(self, vehicle: Any) -> VehicleData:
        """Last-good ego/GPS. No ``sensors.poll``, no ``PollGPSGE``.

        GPS that was ``ok`` on the real poll becomes ``stale``: lat/lon stay,
        and this grab did not take a new fix. Timers stay 0. The last-good
        map is published again so ``read_electrics`` does not poll.
        """
        prev = self._last_vehicle_data
        if prev is None or self._last_sensor_map is None:
            raise RuntimeError("soft esc coalesce without a last-good sample")
        sensors_status = dict(prev.sensors)
        if sensors_status.get("gps") == "ok":
            sensors_status["gps"] = "stale"
        note = prev.note or ""
        tag = "soft esc: sensors.poll coalesced; last-good (not a new GPS fix)"
        if tag not in note:
            note = f"{note}; {tag}" if note else tag
        data = replace(
            prev,
            sensors=sensors_status,
            note=note,
            sensors_poll_ms=0.0,
            poll_gps_ms=0.0,
            poll_gps_sent=False,
        )
        self._sensors_poll_ms = 0.0
        self._poll_gps_ms = 0.0
        self._poll_gps_sent = False
        from python.control.actuate import publish_vehicle_sensor_snap, touch_vehicle_sensor_snap

        publish_vehicle_sensor_snap(vehicle, _copy_sensor_map(self._last_sensor_map or {}))
        touch_vehicle_sensor_snap(vehicle)
        return data

    def _fill_nav(self, data: VehicleData) -> None:
        reading, status = self._poll_gps()
        if reading and status in ("ok", "stale"):
            data.lat = _num(reading.get("lat"))
            data.lon = _num(reading.get("lon"))
            data.gps_x = _num(reading.get("x"))
            data.gps_y = _num(reading.get("y"))
            # Sample time from the sensor payload, never wall-clock of a reuse.
            data.gps_time = _num(reading.get("time"))
            have = data.lat is not None and data.lon is not None
            if not have:
                data.sensors["gps"] = "missing"
            elif status == "ok":
                data.sensors["gps"] = "ok"
            else:
                data.sensors["gps"] = "stale"
        elif self.attached.get("gps"):
            data.sensors["gps"] = "missing"
        pin_lat, pin_lon, pin_name = self._nav_pin()
        data.pin_lat, data.pin_lon = pin_lat, pin_lon
        data.pin_name = pin_name or None
        heading: float | None = None
        # Only a PollGPSGE this call may advance the GPS track. A reused
        # sample must not look like the car sat still for a fresh fix.
        if status == "ok" and data.lat is not None and data.lon is not None:
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
        """One vehicle.sensors.poll. The map is the tick's ego snapshot.

        A throw expires the Soft Esc window so the next grab polls again
        instead of replaying the pre-throw sample. An empty map stays
        ``_sensors_poll_ok`` true and is filtered in ``_remember_sensor_poll``.
        """
        out: dict[str, Any] = {}
        self._sensors_poll_ms = 0.0
        self._sensors_poll_ok = False
        try:
            sensors = getattr(vehicle, "sensors", None)
            if sensors is None:
                return out
            if hasattr(sensors, "poll"):
                t0 = time.perf_counter()
                try:
                    sensors.poll()
                finally:
                    self._sensors_poll_ms = (time.perf_counter() - t0) * 1000.0
            out = _sensor_map(sensors)
        except Exception:
            self._expire_soft_esc_window()
            return out
        from python.control.actuate import publish_vehicle_sensor_snap

        publish_vehicle_sensor_snap(vehicle, out)
        self._sensors_poll_ok = True
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
        """Esc/q / supervisor exit. Disconnect only — never BeamNGpy.close().

        BeamNGpy.close() sends quit_beamng even when this instance did not launch
        Tech (quit_on_close defaults true, process is None) and may kill the
        process tree, which is how CrashSender shows up after a Soft Esc.
        The Soft Esc cache is cleared here so a later vehicle object that
        reuses this ``id()`` cannot replay the previous car inside 200 ms.
        """
        self._clear_soft_esc_cache()
        for attr in ("_gps", "_lidar", "_radar", "_imu"):
            self._drop_handle(attr)
        self.vehicle = None
        bng = self.bng
        self.bng = None
        release_tech_beamngpy(bng)

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


def exit_if_tech_connect_failed(backend: Any, backend_name: str) -> None:
    """Exit 1 when a beamngpy backend fails connect after the wait gate.

    ``connect_failed`` is the connect/import miss. ``_ok`` later means cameras
    attached, so a zero-camera open stays in the vision loop.
    """
    if backend_name != "beamngpy" or not getattr(backend, "connect_failed", False):
        return
    drop_tech_beamngpy()
    session = getattr(backend, "session", None)
    note = getattr(session, "note", "") if session is not None else ""
    note = str(note or "beamngpy connect failed")
    print(f"[GVD] REFUSE: {note}", flush=True)
    print(
        "[GVD] Post-hold connect failed. Supervisor exits. Unique-frame Hz is not measured. "
        "Do not kill BeamNG.tech or CrashSender.",
        flush=True,
    )
    raise SystemExit(1)


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
