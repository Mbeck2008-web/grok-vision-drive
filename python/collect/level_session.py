"""Load a Tech level and make sure a player vehicle exists.

``freeroam_freeroam.startFreeroam`` opens the map and does not spawn a car.
This waits for the road network, then builds a scenario with one vehicle on
that network. Session errors (including BeamNGpy ``BNGError``) stay inside
the call and come back as text.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

from python.debug.gt_geom import project_road

# BeamNGpy's west_coast_usa highway example. Used when that point still lies on a road.
KNOWN_SPAWNS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {
    "west_coast_usa": ((-717.121, 101.0, 118.675), (0.0, 0.0, 0.3826834, 0.9238795)),
}
ROAD_WAIT_S = 45.0
PLAYER_WAIT_S = 15.0
POLL_S = 0.5
SPAWN_Z_M = 250.0
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


def current_level_name(bng: Any) -> str:
    """Name of the loaded level, or empty when this BeamNGpy build does not say."""
    owners = (bng, getattr(bng, "scenario", None), getattr(bng, "control", None))
    for owner in owners:
        if owner is None:
            continue
        for name in ("get_current_level", "get_level_name", "get_current_level_name"):
            fn = getattr(owner, name, None)
            if not callable(fn):
                continue
            try:
                raw = fn()
            except Exception as exc:
                if is_session_error(exc):
                    return ""
                raise
            if isinstance(raw, dict):
                raw = raw.get("level") or raw.get("levelName") or raw.get("name") or ""
            token = level_token(str(raw or ""))
            if token:
                return token
    return ""


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


def begin_level(bng: Any, level: str) -> tuple[Any | None, list[Any], str]:
    """Open ``level`` (empty keeps the current one) and return ``(vehicle, roads, error)``.

    The error is empty when a player vehicle and a road network are both in hand.
    ``BNGError: There is no vehicle`` is retried by spawning, and if that still
    fails the same text is returned instead of escaping.
    """
    clean = level_token(level)
    notes: list[str] = []
    try:
        before, _before_note = _roads(bng) if clean else ([], "")
        before_ids = {str(getattr(road, "road_id", "")) for road in before}
        names_levels = bool(current_level_name(bng))
        if clean:
            queued = queue_level(bng, clean)
            if queued and queued != "no queue_lua_command":
                notes.append(queued)

        def _fresh_roads() -> list[Any] | None:
            if clean and names_levels:
                current = current_level_name(bng)
                if current and current != clean:
                    return None
            now, _note = _roads(bng)
            if not now:
                return None
            ids = {str(getattr(road, "road_id", "")) for road in now}
            # Freeroam can keep serving the previous level's roads for a moment.
            if before_ids and ids == before_ids and not (names_levels and current_level_name(bng) == clean):
                return None
            return list(now)

        found = _wait(_fresh_roads, ROAD_WAIT_S) if clean else _wait(lambda: _roads(bng)[0] or None, ROAD_WAIT_S)
        if found:
            roads = list(found)
            note = ""
        else:
            roads, note = _roads(bng)
            roads = list(roads or [])
            if clean and before_ids and {str(getattr(road, "road_id", "")) for road in roads} == before_ids:
                notes.append("roads still match the previous level")
        if not roads:
            detail = note or (notes[-1] if notes else "no road network")
            label = clean or "current"
            return None, [], f"no road network after loading {label} ({detail})"
        vehicle, player_note = player_vehicle(bng)
        if vehicle is None:
            if player_note:
                notes.append(player_note)
            spawn = choose_spawn(clean, roads)
            if spawn is None:
                return None, list(roads), f"no vehicle after load ({player_note or 'There is no vehicle.'})"
            pos, rot = spawn
            print(
                f"collect spawn {clean or 'current'} ({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})",
                flush=True,
            )
            spawned = spawn_scenario(bng, clean, pos, rot) if clean else "spawn needs a level name"
            if spawned:
                notes.append(spawned)
            vehicle, player_note = player_vehicle(bng)
            if vehicle is None:
                # Scenario start can return before the car is registered.
                vehicle = _wait(lambda: player_vehicle(bng)[0], 2.0)
            if vehicle is None:
                direct = spawn_direct(bng, pos, rot)
                if direct:
                    notes.append(direct)
                vehicle = _wait(lambda: player_vehicle(bng)[0], PLAYER_WAIT_S)
                if vehicle is None:
                    vehicle, player_note = player_vehicle(bng)
            fresh, note = _roads(bng)
            if fresh:
                roads = list(fresh)
            elif note:
                notes.append(note)
        if vehicle is None:
            detail = player_note or (notes[-1] if notes else "There is no vehicle.")
            return None, list(roads or []), f"no vehicle after load ({detail})"
        if not roads:
            detail = note or "no road network"
            return None, [], f"no road network after spawn ({detail})"
        return vehicle, list(roads), ""
    except Exception as exc:
        if is_session_error(exc) or isinstance(exc, (TypeError, ValueError, AttributeError)):
            return None, [], session_text(exc)
        raise
