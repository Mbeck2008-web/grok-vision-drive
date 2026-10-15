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

# BeamNGpy's west_coast_usa highway example. Used when that point still lies on a road.
KNOWN_SPAWNS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {
    "west_coast_usa": ((-717.121, 101.0, 118.675), (0.0, 0.0, 0.3826834, 0.9238795)),
}
ROAD_WAIT_S = 8.0
PLAYER_WAIT_S = 8.0
POLL_S = 0.5
SPAWN_Z_M = 250.0
# Used only so a scenario can start. The car is moved onto a centerline after.
DEFAULT_SPAWN = ((0.0, 0.0, 50.0), (0.0, 0.0, 0.0, 1.0))


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


class ScriptScenario:
    """Scenario stand-in used when ``beamngpy`` cannot be imported."""

    def __init__(self, level: str, name: str, pos: tuple[float, float, float], rot: tuple[float, float, float, float]) -> None:
        self.level = level
        self.name = name
        self.pos = pos
        self.rot = rot
        self.vehicles = [("gvd_ego", pos, rot)]

    def make(self, bng: Any) -> None:
        del bng


def make_collect_scenario(level: str, pos: tuple[float, float, float], rot: tuple[float, float, float, float]) -> Any:
    """A one-vehicle scenario. The real BeamNGpy classes are used when they import."""
    try:
        from beamngpy import Scenario, Vehicle  # type: ignore
    except Exception:
        return ScriptScenario(level, "gvd_collect", pos, rot)
    scenario = Scenario(level, "gvd_collect", description="GVD scene collect")
    vehicle = Vehicle("gvd_ego", model="etk800")
    try:
        scenario.add_vehicle(vehicle, pos=pos, rot_quat=rot, cling=True)
    except TypeError:
        scenario.add_vehicle(vehicle, pos=pos, rot_quat=rot)
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


def spawn_scenario(bng: Any, level: str, pos: tuple[float, float, float], rot: tuple[float, float, float, float]) -> str:
    """``Scenario.add_vehicle`` / ``make`` / ``load`` / ``start``. Empty string on a queued start."""
    clean = level_token(level)
    if not clean:
        return "spawn needs a level name"
    try:
        scenario = make_collect_scenario(clean, pos, rot)
    except Exception as exc:
        if is_session_error(exc) or isinstance(exc, (TypeError, ValueError, AttributeError)):
            return session_text(exc)
        raise
    make = getattr(scenario, "make", None)
    if callable(make):
        try:
            make(bng)
        except Exception as exc:
            if not is_session_error(exc) and not isinstance(exc, (TypeError, ValueError, AttributeError, OSError)):
                raise
    load, start = _scenario_ops(bng)
    if load is None:
        return "no scenario load"
    try:
        load(scenario)
        if start is not None:
            start()
    except Exception as exc:
        if is_session_error(exc) or isinstance(exc, (TypeError, ValueError, AttributeError)):
            return session_text(exc)
        raise
    return ""


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
                fn(vehicle, pos, rot_quat=rot, cling=True)
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
    """Where to place ``etk800`` so the scenario can start."""
    clean = level_token(level)
    official = official_spawn(bng, clean)
    if official is not None:
        return official[0], official[1], "info"
    known = KNOWN_SPAWNS.get(clean)
    if known is not None:
        return known[0], known[1], "known"
    return DEFAULT_SPAWN[0], DEFAULT_SPAWN[1], "default"


def teleport_vehicle(
    bng: Any,
    vehicle: Any,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
) -> str:
    """Move the player onto a centerline. Empty string when the call was queued."""
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
        f"local x, y = {x:.2f}, {y:.2f} "
        f"local z = {z:.2f} "
        "local ok, h = pcall(be.getSurfaceHeightBelow, be, vec3(x, y, 2000)) "
        "if ok and type(h) == 'number' then z = h + 0.6 end "
        f"v:setPositionRotation(x, y, z, {qx:.5f}, {qy:.5f}, {qz:.5f}, {qw:.5f}) "
        "end",
    )


def _on_road(roads: list[Any], pos: tuple[float, float, float] | None) -> bool:
    if pos is None or not roads:
        return False
    hit = project_road(list(roads), float(pos[0]), float(pos[1]), None)
    return hit is not None and not bool(hit.off_road)


def begin_level(bng: Any, level: str) -> OpenedLevel:
    """Open ``level`` as a started scenario. Empty keeps the session already running.

    Roads are read only after ``start_scenario``. An empty level is the
    45-minute path: the vehicle and ``get_road_network`` that are already there.
    """
    clean = level_token(level)
    try:
        if not clean:
            vehicle, player_note = player_vehicle(bng)
            roads, note = _roads(bng)
            if vehicle is not None and roads:
                return OpenedLevel(vehicle, list(roads), "", None, "current")
            detail = player_note or note or "no vehicle after load"
            return OpenedLevel(vehicle, list(roads or []), detail, None, "current")
        pos, rot, source = resolve_spawn(bng, clean)
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
        if source == "default" or not _on_road(roads, pos):
            spot = road_spawn(list(roads))
            if spot is not None:
                moved, mrot = spot
                print(
                    f"collect teleport {clean} ({moved[0]:.1f},{moved[1]:.1f},{moved[2]:.1f})",
                    flush=True,
                )
                shift = teleport_vehicle(bng, vehicle, moved, mrot)
                if shift:
                    return OpenedLevel(vehicle, list(roads), shift, pos, source)
                pos = moved
                source = "centerline"
        return OpenedLevel(vehicle, list(roads), "", pos, source)
    except Exception as exc:
        if is_session_error(exc) or isinstance(exc, (TypeError, ValueError, AttributeError)):
            return OpenedLevel(None, [], session_text(exc))
        raise
