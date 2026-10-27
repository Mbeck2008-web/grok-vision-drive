"""Load a Tech level as a started scenario with one vehicle.

The 45-minute capture read ``scenario.get_road_network`` on a scenario that
was already started. Freeroam is not that: Tech answers ``Need to be in a
started scenario to get its DecalRoad data.`` Each named level is therefore
``Scenario`` + ``etk800``, ``make``, ``load_scenario``, ``start_scenario``,
and only then the road network. Session errors (including ``BNGError``) stay
inside the call and come back as text.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from python.debug.gt_geom import project_road

# Poses taken from BeamNGpy examples, not from a zeroed info.json spawn.
# west_coast_usa: examples/west_coast_IMU.py and the README.
# italy: examples/feature_overview.ipynb and examples/GPS_trajectory.py.
# east_coast_usa: examples/east_coast_random.py.
# hirochi_raceway: examples/ai_waypoints.py.
# gridmap_v2: examples/ultrasonic_demo.py.
# small_island: BeamNGRL's small_island start pose.
KNOWN_SPAWNS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {
    "west_coast_usa": ((-717.121, 101.0, 118.675), (0.0, 0.0, 0.3826834, 0.9238795)),
    "italy": ((245.11, -906.94, 247.46), (0.0010, 0.1242, 0.9884, -0.0872)),
    "east_coast_usa": ((-426.68, -43.59, 31.11), (0.0, 0.0, 1.0, 0.0)),
    "hirochi_raceway": ((-408.5, 260.2, 25.22), (-0.006664, -0.002505, -0.2799, 0.96)),
    "gridmap_v2": ((531.91, 263.99, 100.3), (0.0053479, 0.0082936, 0.7021236, 0.7119867)),
    "small_island": ((-67.0, 336.0, 34.5), (0.0, 0.0, 0.3826834, 0.9238795)),
}
ROAD_WAIT_S = 8.0
PLAYER_WAIT_S = 8.0
POLL_S = 0.5
SPAWN_Z_M = 250.0
SCENARIO_TIMEOUT_S = 120.0


@dataclass
class OpenedLevel:
    vehicle: Any = None
    roads: list[Any] = field(default_factory=list)
    error: str = ""
    spawn: tuple[float, float, float] | None = None
    spawn_source: str = ""
    cameras: dict[str, Any] = field(default_factory=dict)
_SESSION_NAMES = {"BNGError", "BNGValueError", "BNGDisconnectedError"}


def session_text(exc: BaseException) -> str:
    detail = str(exc).strip() or type(exc).__name__
    return f"{type(exc).__name__}: {detail}"


def is_session_error(exc: BaseException) -> bool:
    """True for BeamNGpy failures and dead sockets. A fake ``BNGError`` counts."""
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    return any(cls.__name__ in _SESSION_NAMES for cls in type(exc).__mro__)


def level_token(name: str) -> str:
    text = str(name or "").strip().replace("\\", "/")
    if not text:
        return ""
    parts = [part for part in text.split("/") if part and part not in ("levels", "info.json")]
    token = parts[-1] if parts else text
    return "".join(ch for ch in token if ch.isalnum() or ch in "_-")


def level_names(raw: Any) -> set[str]:
    """Level ids from ``get_levels``, a dict of infos, or a list."""
    names: set[str] = set()
    if isinstance(raw, dict):
        nested = raw.get("levels") if "levelName" not in raw and "title" not in raw else None
        if nested is not None and nested is not raw:
            return level_names(nested)
        for key, value in raw.items():
            if str(key) in ("levels", "status"):
                continue
            if isinstance(value, dict):
                picked = value.get("levelName") or value.get("name") or value.get("level") or key
                token = level_token(str(picked))
            else:
                token = level_token(str(key))
            if token:
                names.add(token)
        return names
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, str):
                token = level_token(item)
            elif isinstance(item, dict):
                token = level_token(str(item.get("levelName") or item.get("name") or item.get("level") or ""))
            else:
                token = ""
            if token:
                names.add(token)
    return names


def installed_levels(bng: Any) -> tuple[set[str] | None, str]:
    """Level ids the running install reports. ``None`` when the list cannot be read."""
    owners = [bng, getattr(bng, "scenario", None), getattr(bng, "control", None)]
    notes: list[str] = []
    for owner in owners:
        if owner is None:
            continue
        for name in ("get_levels", "get_level_list"):
            fn = getattr(owner, name, None)
            if not callable(fn):
                continue
            try:
                raw = fn()
            except TypeError:
                try:
                    raw = fn(bng)
                except Exception as exc:
                    if not is_session_error(exc):
                        raise
                    notes.append(session_text(exc))
                    continue
            except Exception as exc:
                if not is_session_error(exc):
                    raise
                notes.append(session_text(exc))
                continue
            found = level_names(raw)
            if found:
                return found, ""
            notes.append(f"{name} returned no level names")
    return None, "; ".join(notes) if notes else "no get_levels"


def select_maps(requested: tuple[str, ...] | list[str], installed: set[str] | None) -> tuple[list[str], list[str]]:
    """Keep requested levels that are installed. ``installed is None`` keeps every name."""
    if installed is None:
        return [level_token(name) for name in requested if level_token(name)], []
    kept: list[str] = []
    skipped: list[str] = []
    for name in requested:
        token = level_token(name)
        if not token:
            continue
        if token in installed:
            kept.append(token)
        else:
            skipped.append(token)
    return kept, skipped


def player_vehicle(bng: Any) -> tuple[Any | None, str]:
    """The current player vehicle, or ``(None, reason)``. ``BNGError`` is a reason, not a crash."""
    try:
        api = getattr(bng, "vehicles", None)
        vehicles: dict[str, Any] = {}
        if api is not None and hasattr(api, "get_current"):
            cur = api.get_current()
            if isinstance(cur, dict):
                vehicles = {str(key): value for key, value in cur.items() if value is not None}
        vid = None
        if api is not None and hasattr(api, "get_player_vehicle_id"):
            info = api.get_player_vehicle_id()
            if isinstance(info, dict):
                vid = info.get("vid") or info.get("name")
            elif info:
                vid = info
        veh = vehicles.get(str(vid)) if vid else None
        if veh is None and vehicles:
            veh = next(iter(vehicles.values()))
        if veh is None:
            return None, "There is no vehicle."
        if hasattr(veh, "connect"):
            try:
                if not getattr(veh, "is_connected", lambda: True)():
                    veh.connect(bng)
            except Exception as exc:
                if not is_session_error(exc):
                    raise
                return None, session_text(exc)
        return veh, ""
    except Exception as exc:
        if is_session_error(exc):
            return None, session_text(exc)
        raise


def _polyline_length(pts: list[tuple[float, float]]) -> float:
    total = 0.0
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        total += math.hypot(bx - ax, by - ay)
    return total


def _yaw_quat(dx: float, dy: float) -> tuple[float, float, float, float]:
    half = math.atan2(float(dy), float(dx)) * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def road_spawn(roads: list[Any]) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
    """A point about 40% along the longest centerline, dropped onto the mesh from above."""
    usable = [road for road in roads if len(getattr(road, "xy", ()) or []) >= 2]
    if not usable:
        return None
    road = max(usable, key=lambda item: _polyline_length(list(item.xy)))
    pts = [(float(x), float(y)) for x, y in road.xy]
    target = max(5.0, _polyline_length(pts) * 0.4)
    walked = 0.0
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        seg = math.hypot(bx - ax, by - ay)
        if seg < 1e-6:
            continue
        if walked + seg >= target or (bx, by) == pts[-1]:
            mix = 0.0 if seg < 1e-6 else min(1.0, (target - walked) / seg)
            x = ax + mix * (bx - ax)
            y = ay + mix * (by - ay)
            return (x, y, SPAWN_Z_M), _yaw_quat(bx - ax, by - ay)
        walked += seg
    return None


def choose_spawn(level: str, roads: list[Any]) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
    """Known highway pose when it still lies on a road, otherwise a centerline point."""
    known = KNOWN_SPAWNS.get(level_token(level))
    if known and roads:
        pos, rot = known
        hit = project_road(list(roads), float(pos[0]), float(pos[1]), None)
        if hit is not None and not hit.off_road and float(hit.dist_m) < 20.0:
            return pos, rot
    return road_spawn(roads)


def spawn_is_origin(pos: tuple[float, float, float] | None) -> bool:
    """True for a missing point or the (0, 0, 0) info.json parse."""
    if pos is None or len(pos) < 3:
        return True
    return abs(float(pos[0])) + abs(float(pos[1])) + abs(float(pos[2])) < 1.0


class ScriptScenario:
    """Scenario stand-in used when ``beamngpy`` cannot be imported."""

    def __init__(self, level: str, name: str, pos: tuple[float, float, float], rot: tuple[float, float, float, float]) -> None:
        self.level = level
        self.name = name
        self.pos = pos
        self.rot = rot
        self.path: str | None = None
        self.made = False
        self.vehicles = [("gvd_ego", pos, rot)]
        self.safe_spawn = True

    def make(self, bng: Any) -> None:
        del bng
        self.made = True
        self.path = f"script/{self.level}/{self.name}"


def _add_vehicle(scenario: Any, vehicle: Any, pos: tuple[float, float, float], rot: tuple[float, float, float, float]) -> None:
    """``safe_spawn`` is the current flag. ``cling`` only remains for older BeamNGpy."""
    try:
        scenario.add_vehicle(vehicle, pos=pos, rot_quat=rot, safe_spawn=True)
        return
    except TypeError:
        scenario.add_vehicle(vehicle, pos=pos, rot_quat=rot)


def make_collect_scenario(
    level: str,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    *,
    name: str,
) -> Any:
    """A one-vehicle scenario. The real BeamNGpy classes are used when they import."""
    try:
        from beamngpy import Scenario, Vehicle  # type: ignore
    except Exception:
        return ScriptScenario(level, name, pos, rot)
    scenario = Scenario(level, name, description="GVD scene collect")
    vehicle = Vehicle("gvd_ego", model="etk800")
    _add_vehicle(scenario, vehicle, pos, rot)
    return scenario


def _send(bng: Any, command: str) -> str:
    send = getattr(getattr(bng, "control", None), "queue_lua_command", None) or getattr(bng, "queue_lua_command", None)
    if not callable(send):
        return "no queue_lua_command"
    try:
        send(command)
        return ""
    except TypeError:
        try:
            send(command, False)
            return ""
        except Exception as exc:
            if is_session_error(exc):
                return session_text(exc)
            raise
    except Exception as exc:
        if is_session_error(exc):
            return session_text(exc)
        raise


def queue_level(bng: Any, level: str) -> str:
    """Ask Tech to open ``level``. Empty string means the command was queued."""
    clean = level_token(level)
    if not clean:
        return "blank level name"
    return _send(bng, f'freeroam_freeroam.startFreeroam("{clean}")')


def _scenario_ops(bng: Any) -> tuple[Callable[..., Any] | None, Callable[..., Any] | None]:
    scenario = getattr(bng, "scenario", None)
    load = getattr(scenario, "load", None) if scenario is not None else None
    start = getattr(scenario, "start", None) if scenario is not None else None
    if not callable(load):
        load = getattr(bng, "load_scenario", None)
    if not callable(start):
        start = getattr(bng, "start_scenario", None)
    return (load if callable(load) else None), (start if callable(start) else None)


def widen_scenario_timeout(bng: Any, seconds: float = SCENARIO_TIMEOUT_S) -> None:
    """One scenario load can sit longer than the 10 s Hello cap. Big maps need about 120 s."""
    from python.sensors.tech import widen_socket_timeout

    widen_socket_timeout(bng, float(seconds))
    conn = getattr(bng, "connection", None)
    if conn is not None:
        for name in ("timeout", "socket_timeout"):
            try:
                setattr(conn, name, float(seconds))
            except Exception:
                pass


def spawn_scenario(bng: Any, level: str, pos: tuple[float, float, float], rot: tuple[float, float, float, float]) -> str:
    """``make``, then ``load_scenario``, then ``start_scenario``. One retry after a timeout.

    ``make`` has to set ``scenario.path``. Loading without that path is the
    ``Scenario not found, empty path`` error.
    """
    clean = level_token(level)
    if not clean:
        return "spawn needs a level name"
    if spawn_is_origin(pos):
        return f"refusing origin spawn for {clean}"
    last = "scenario did not start"
    for attempt, name in enumerate((f"gvd_collect_{clean}", f"gvd_collect_{clean}_retry"), start=1):
        widen_scenario_timeout(bng, SCENARIO_TIMEOUT_S)
        try:
            scenario = make_collect_scenario(clean, pos, rot, name=name)
        except Exception as exc:
            if is_session_error(exc) or isinstance(exc, (TypeError, ValueError, AttributeError)):
                last = session_text(exc)
                continue
            raise
        make = getattr(scenario, "make", None)
        if not callable(make):
            return "scenario has no make()"
        try:
            make(bng)
        except Exception as exc:
            if is_session_error(exc) or isinstance(exc, (TypeError, ValueError, AttributeError, OSError)):
                last = session_text(exc)
                continue
            raise
        if not getattr(scenario, "path", None):
            last = "scenario.make() did not set a path"
            continue
        load, start = _scenario_ops(bng)
        if load is None:
            return "no scenario load"
        try:
            load(scenario)
            if start is not None:
                start()
            return ""
        except Exception as exc:
            if is_session_error(exc) or isinstance(exc, (TypeError, ValueError, AttributeError)):
                last = session_text(exc)
                if attempt == 2:
                    return last
                continue
            raise
    return last


def spawn_direct(bng: Any, pos: tuple[float, float, float], rot: tuple[float, float, float, float]) -> str:
    """Spawn without reloading the level, when the scenario path did not leave a vehicle."""
    x, y, z = pos
    qx, qy, qz, qw = rot
    try:
        from beamngpy import Vehicle  # type: ignore
    except Exception:
        Vehicle = None  # type: ignore
    if Vehicle is not None:
        vehicle = Vehicle("gvd_ego", model="etk800")
        fn = getattr(bng, "spawn_vehicle", None)
        if not callable(fn):
            fn = getattr(getattr(bng, "vehicles", None), "spawn", None)
        if callable(fn):
            try:
                fn(vehicle, pos, rot_quat=rot, safe_spawn=True)
                return ""
            except TypeError:
                try:
                    fn(vehicle, pos, rot)
                    return ""
                except Exception as exc:
                    if is_session_error(exc):
                        return session_text(exc)
                    raise
            except Exception as exc:
                if is_session_error(exc):
                    return session_text(exc)
                raise
    return _send(
        bng,
        "core_vehicles.spawnNewVehicle(\"etk800\", "
        f"{{pos=vec3({x:.2f},{y:.2f},{z:.2f}), rot=quat({qx:.5f},{qy:.5f},{qz:.5f},{qw:.5f})}})",
    )


def _wait(ready: Callable[[], Any], timeout_s: float) -> Any:
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        found = ready()
        if found:
            return found
        if time.monotonic() >= deadline:
            return None
        time.sleep(POLL_S)


def _roads(bng: Any) -> tuple[list[Any], str]:
    from python.debug.gt_logger import load_road_polys

    try:
        roads, note = load_road_polys(bng)
    except Exception as exc:
        if is_session_error(exc):
            return [], session_text(exc)
        raise
    return list(roads or []), str(note or "")


def lua_call(bng: Any, command: str) -> Any:
    """Run one Lua chunk and return its reply. ``None`` when this build cannot reply."""
    for owner in (getattr(bng, "control", None), bng):
        if owner is None:
            continue
        fn = getattr(owner, "queue_lua_command", None)
        if not callable(fn):
            continue
        try:
            return fn(command, True)
        except TypeError:
            try:
                return fn(command, response=True)
            except TypeError:
                return None
        except Exception as exc:
            if is_session_error(exc):
                return None
            raise
    return None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_spawn_point(raw: Any) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
    """One spawn from a level info record, a reply string, or a pos/rot pair."""
    if isinstance(raw, dict):
        for key in ("result", "resp", "data", "return", "ret"):
            if key in raw:
                parsed = parse_spawn_point(raw[key])
                if parsed is not None:
                    return parsed
        pos = raw.get("pos") or raw.get("position") or raw.get("spawn")
        rot = raw.get("rot") or raw.get("rotQuat") or raw.get("rot_quat")
        if pos is None and all(key in raw for key in ("x", "y")):
            pos = raw
        if pos is not None:
            return parse_spawn_point((pos, rot if rot is not None else (0.0, 0.0, 0.0, 1.0)))
        return None
    if isinstance(raw, str):
        parts = raw.replace(",", " ").split()
        try:
            nums = [float(part) for part in parts]
        except ValueError:
            return None
        if len(nums) >= 7:
            return (nums[0], nums[1], nums[2]), (nums[3], nums[4], nums[5], nums[6])
        if len(nums) >= 3:
            return (nums[0], nums[1], nums[2]), (0.0, 0.0, 0.0, 1.0)
        return None
    if isinstance(raw, (list, tuple)) and len(raw) >= 2 and isinstance(raw[0], (list, tuple, dict)):
        pos = parse_spawn_point(raw[0])
        quat = raw[1]
        if pos is None:
            return None
        if isinstance(quat, (list, tuple)) and len(quat) >= 4:
            return pos[0], (_as_float(quat[0]), _as_float(quat[1]), _as_float(quat[2]), _as_float(quat[3], 1.0))
        return pos
    if isinstance(raw, (list, tuple)) and len(raw) >= 3 and not isinstance(raw[0], (list, tuple, dict)):
        return (_as_float(raw[0]), _as_float(raw[1]), _as_float(raw[2])), (0.0, 0.0, 0.0, 1.0)
    return None


def _levels_payload(bng: Any) -> Any:
    for owner in (bng, getattr(bng, "scenario", None)):
        if owner is None:
            continue
        fn = getattr(owner, "get_levels", None)
        if not callable(fn):
            continue
        try:
            return fn()
        except Exception as exc:
            if is_session_error(exc):
                return None
            raise
    return None


def _spawn_from_info(info: Any) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
    if not isinstance(info, dict):
        return None
    points = info.get("spawnPoints") or info.get("spawnpoints") or info.get("spawn_points")
    if isinstance(points, dict):
        points = list(points.values())
    if isinstance(points, (list, tuple)) and points:
        parsed = parse_spawn_point(points[0])
        if parsed is not None:
            return parsed
    return parse_spawn_point(info.get("spawn") or info.get("defaultSpawn"))


def official_spawn(bng: Any, level: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
    """Spawn published by the level list or ``levels/<name>/info.json``."""
    clean = level_token(level)
    if not clean:
        return None
    payload = _levels_payload(bng)
    if isinstance(payload, dict):
        nested = payload.get("levels") if "levelName" not in payload else None
        table = nested if isinstance(nested, dict) else payload
        if isinstance(table, dict):
            parsed = _spawn_from_info(table.get(clean))
            if parsed is not None:
                return parsed
    reply = lua_call(
        bng,
        "local info = jsonReadFile('levels/"
        + clean
        + "/info.json') or {} "
        "local pts = info.spawnPoints or info.spawnpoints or {} "
        "local sp = pts[1] "
        "if type(sp) ~= 'table' then return '' end "
        "local p = sp.pos or sp.position or {} "
        "local x = tonumber(p.x or p[1]) or 0 "
        "local y = tonumber(p.y or p[2]) or 0 "
        "local z = tonumber(p.z or p[3]) or 0 "
        "local r = sp.rot or sp.rotQuat or {0,0,0,1} "
        "local qx = tonumber(r.x or r[1]) or 0 "
        "local qy = tonumber(r.y or r[2]) or 0 "
        "local qz = tonumber(r.z or r[3]) or 0 "
        "local qw = tonumber(r.w or r[4]) or 1 "
        "return string.format('%.3f %.3f %.3f %.5f %.5f %.5f %.5f', x, y, z, qx, qy, qz, qw)",
    )
    return parse_spawn_point(reply)


def resolve_spawn(bng: Any, level: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float], str]:
    """Table pose first. An info.json point is used only when it is not the origin."""
    clean = level_token(level)
    known = KNOWN_SPAWNS.get(clean)
    if known is not None and not spawn_is_origin(known[0]):
        return known[0], known[1], "known"
    official = official_spawn(bng, clean)
    if official is not None and not spawn_is_origin(official[0]):
        return official[0], official[1], "info"
    return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), "missing"


def _xyz(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
        z = float(value[2]) if len(value) > 2 else 0.0
    except (TypeError, ValueError):
        return None
    return x, y, z


def polylines_xyz(spec: Any) -> list[list[tuple[float, float, float]]]:
    """Centerline nodes with z, from a ``get_road_network`` payload."""
    if isinstance(spec, dict) and isinstance(spec.get("roads"), dict):
        spec = spec["roads"]
    if not isinstance(spec, dict):
        return []
    lines: list[list[tuple[float, float, float]]] = []
    for road in spec.values():
        edges = None
        if isinstance(road, dict):
            try:
                if float(road.get("drivability", road.get("drivable", 1)) or 0) <= 0.0:
                    continue
            except (TypeError, ValueError):
                pass
            edges = road.get("edges") or road.get("nodes")
        elif isinstance(road, list):
            edges = road
        if not isinstance(edges, list):
            continue
        line: list[tuple[float, float, float]] = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            point = _xyz(edge.get("middle") or edge.get("mid") or edge.get("center"))
            if point is None:
                left = _xyz(edge.get("left"))
                right = _xyz(edge.get("right"))
                if left is not None and right is not None:
                    point = ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5, (left[2] + right[2]) * 0.5)
            if point is not None:
                line.append(point)
        if len(line) >= 2:
            lines.append(line)
    return lines


def _network_spec(bng: Any) -> Any:
    """The current level's road network from the shared loader (read once per level)."""
    from python.collect.road_loader import shared

    return shared().network(bng)


def recover_points(
    pos: tuple[float, float, float] | None,
    spawn: tuple[float, float, float] | None,
) -> list[tuple[float, float, float]]:
    """Where to look for a centerline after a stall. The car's own point first.

    The italy lane change stopped near (165, -849). Snapping the original
    spawn instead threw the car back to the start, and the next pose tipped.
    """
    points: list[tuple[float, float, float]] = []
    for raw in (pos, spawn):
        if raw is None or len(raw) < 3:
            continue
        try:
            point = (float(raw[0]), float(raw[1]), float(raw[2]))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in point):
            continue
        if points and all(abs(point[i] - points[0][i]) < 1.0 for i in range(3)):
            continue
        points.append(point)
    return points


def snap_to_road(
    bng: Any,
    near: tuple[float, float, float] | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
    """Nearest centerline pose: node z plus half a metre, heading along the road.

    Points farther than 40 m are ignored so a bad ray does not throw the car
    across the map at z=250.
    """
    lines = polylines_xyz(_network_spec(bng))
    if not lines or near is None:
        return None
    px, py = float(near[0]), float(near[1])
    best: tuple[float, float, float, float, float, float] | None = None
    for line in lines:
        for (ax, ay, az), (bx, by, bz) in zip(line, line[1:]):
            dx, dy = bx - ax, by - ay
            length2 = dx * dx + dy * dy
            if length2 < 1e-6:
                continue
            t = ((px - ax) * dx + (py - ay) * dy) / length2
            t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
            cx = ax + t * dx
            cy = ay + t * dy
            cz = az + t * (bz - az)
            dist = math.hypot(px - cx, py - cy)
            if best is None or dist < best[0]:
                best = (dist, cx, cy, cz, dx, dy)
    if best is None or best[0] > 40.0:
        return None
    _dist, cx, cy, cz, dx, dy = best
    return (cx, cy, cz + 0.5), _yaw_quat(dx, dy)


def place_vehicle(
    bng: Any,
    vehicle: Any,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
) -> str:
    """Upright move onto the road. Empty string when the call was accepted."""
    fn = getattr(vehicle, "teleport", None)
    if callable(fn):
        try:
            fn(pos, rot_quat=rot, reset=True)
            return ""
        except TypeError:
            try:
                fn(pos, rot_quat=rot)
                return ""
            except Exception as exc:
                if not is_session_error(exc) and not isinstance(exc, TypeError):
                    raise
        except Exception as exc:
            if not is_session_error(exc):
                raise
    x, y, z = pos
    qx, qy, qz, qw = rot
    return _send(
        bng,
        "local v = be:getPlayerVehicle(0) "
        "if v then "
        f"v:setPositionRotation({x:.2f},{y:.2f},{z:.2f},{qx:.5f},{qy:.5f},{qz:.5f},{qw:.5f}) "
        "end",
    )


def begin_level(bng: Any, level: str) -> OpenedLevel:
    """Open ``level`` as a started scenario. Empty keeps the session already running.

    Roads are read only after ``start_scenario``, through the shared road
    loader: once per level, reused for the same level and for recovery snaps.
    An empty level is the 45-minute path: the vehicle and road network that
    are already there.
    """
    from python.collect.road_loader import current_level, shared

    clean = level_token(level)
    try:
        if not clean:
            vehicle, player_note = player_vehicle(bng)
            shared().enter(bng, current_level(bng))
            roads, note = _roads(bng)
            if vehicle is not None and roads:
                return OpenedLevel(vehicle, list(roads), "", None, "current")
            detail = player_note or note or "no vehicle after load"
            return OpenedLevel(vehicle, list(roads or []), detail, None, "current")
        pos, rot, source = resolve_spawn(bng, clean)
        if source == "missing" or spawn_is_origin(pos):
            return OpenedLevel(None, [], f"no known spawn for {clean}", pos, source)
        print(
            f"collect spawn {clean} source={source} ({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})",
            flush=True,
        )
        started = spawn_scenario(bng, clean, pos, rot)
        if started:
            return OpenedLevel(None, [], started, pos, source)
        vehicle, player_note = player_vehicle(bng)
        if vehicle is None:
            vehicle = _wait(lambda: player_vehicle(bng)[0], PLAYER_WAIT_S)
            if vehicle is None:
                direct = spawn_direct(bng, pos, rot)
                if direct:
                    player_note = direct
                vehicle = _wait(lambda: player_vehicle(bng)[0], PLAYER_WAIT_S)
                if vehicle is None:
                    vehicle, player_note = player_vehicle(bng)
        if vehicle is None:
            return OpenedLevel(None, [], f"no vehicle after load ({player_note or 'There is no vehicle.'})", pos, source)
        # One network read per level: the same level keeps the cached roads,
        # a new level drops them and the first _roads call reads once.
        shared().enter(bng, clean)
        roads, note = _roads(bng)
        if not roads:
            found = _wait(lambda: _roads(bng)[0] or None, ROAD_WAIT_S)
            if found:
                roads = list(found)
                note = ""
            else:
                roads, note = _roads(bng)
                roads = list(roads or [])
        if not roads:
            detail = note or "no road network"
            return OpenedLevel(vehicle, [], f"no road network after loading {clean} ({detail})", pos, source)
        placed = snap_to_road(bng, pos)
        if placed is not None:
            moved, mrot = placed
            print(
                f"collect place {clean} ({moved[0]:.1f},{moved[1]:.1f},{moved[2]:.1f})",
                flush=True,
            )
            shift = place_vehicle(bng, vehicle, moved, mrot)
            if shift:
                return OpenedLevel(vehicle, list(roads), shift, pos, source)
            pos = moved
            from python.collect.liveness import SETTLE_S

            time.sleep(SETTLE_S)
        return OpenedLevel(vehicle, list(roads), "", pos, source)
    except Exception as exc:
        if is_session_error(exc) or isinstance(exc, (TypeError, ValueError, AttributeError)):
            return OpenedLevel(None, [], session_text(exc))
        raise
