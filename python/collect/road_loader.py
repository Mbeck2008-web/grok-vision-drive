"""The one place that reads the level's road network from BeamNG.

Every caller that needs road geometry at runtime goes through here: the GT
logger (R1), ``level_session`` road waits and recovery snaps (R2, R3) and
the offline road dumps (R4). One read per level, cached until the level
changes, so a recovery or a tick never costs another socket round trip.

``get_road_network`` is asked for left/middle/right edges of drivable roads
(BeamNGpy 1.36). Builds that only have the deprecated ``get_roads`` /
``get_road_edges`` pair still work through the legacy fallback. An empty
result is never cached, so a caller that polls while the level is still
loading keeps polling.

The cache is keyed by the session object and the level name. ``enter``
switches the key; a different level drops the cache and the next caller
reads once. Nothing here runs per tick.
"""
from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass, field
from typing import Any

from python.debug.gt_geom import RoadPoly, parse_road_edges, road_width_floor

LEVEL_LUA = "return getCurrentLevelIdentifier() or ''"


@dataclass
class RoadSnapshot:
    """One read of the network: the raw spec, parsed centerlines, and why it is empty.

    ``error`` is the first exception ``get_road_network`` raised on a read
    that found no roads (its text is also in ``note``). ``network()``
    re-raises it unless it is a session error.
    """

    spec: Any = None
    polys: list[RoadPoly] = field(default_factory=list)
    note: str = ""
    error: BaseException | None = field(default=None, compare=False, repr=False)


def _exc_text(exc: BaseException) -> str:
    detail = str(exc).strip() or type(exc).__name__
    return f"{type(exc).__name__}: {detail}"


def _call_exc(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, str, BaseException | None]:
    try:
        return fn(*args, **kwargs), "", None
    except TypeError:
        try:
            return fn(*args), "", None
        except Exception as exc:
            return None, _exc_text(exc), exc
    except Exception as exc:
        return None, _exc_text(exc), exc


def _call(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, str]:
    got, note, _exc = _call_exc(fn, *args, **kwargs)
    return got, note


def polys_from_network(spec: Any) -> list[RoadPoly]:
    """Centerlines from a road network spec that carries left/middle/right edges."""
    if isinstance(spec, dict) and "roads" in spec and isinstance(spec["roads"], dict):
        spec = spec["roads"]
    elif isinstance(spec, dict) and "network" in spec and isinstance(spec["network"], dict):
        spec = spec["network"]
    if not isinstance(spec, dict):
        return []
    polys: list[RoadPoly] = []
    for road_id, road in spec.items():
        edges = None
        if isinstance(road, dict):
            driv = road.get("drivability", road.get("drivable", 1))
            try:
                if float(driv) <= 0.0:
                    continue
            except (TypeError, ValueError):
                pass
            edges = road.get("edges") or road.get("nodes")
        elif isinstance(road, list):
            edges = road
        poly = parse_road_edges(str(road_id), edges)
        if poly is not None:
            if isinstance(road, dict):
                try:
                    count = int(float(road.get("lanesLeft") or 0)) + int(float(road.get("lanesRight") or 0))
                except (TypeError, ValueError):
                    count = 0
                if 1 <= count <= 8:
                    poly.lanes = count
            polys.append(road_width_floor(poly, road))
    return polys


def read_network(bng: Any) -> RoadSnapshot:
    """One uncached read. ``spec`` is the first non-empty network reply.

    BeamNGpy 1.36 deprecates ``get_roads``; Tech 0.39 raises ``BNGError``
    from it. The scenario's network call with edges is the one that returns
    left/middle/right edges. The note keeps the exception text.
    """
    if bng is None:
        return RoadSnapshot(None, [], "no beamng")
    notes: list[str] = []
    first_spec = None
    first_error: BaseException | None = None
    owners = []
    scenario = getattr(bng, "scenario", None)
    if scenario is not None:
        owners.append(scenario)
    owners.append(bng)
    for owner in owners:
        fn = getattr(owner, "get_road_network", None)
        if not callable(fn):
            continue
        spec, note, exc = _call_exc(fn, include_edges=True, drivable_only=True)
        if note:
            notes.append(f"get_road_network {note}")
            first_error = first_error or exc
            spec, note2, exc2 = _call_exc(fn)
            if note2:
                notes.append(f"get_road_network() {note2}")
                first_error = first_error or exc2
        if spec is not None and first_spec is None:
            first_spec = spec
        polys = polys_from_network(spec)
        if polys:
            return RoadSnapshot(spec, polys, "")
        if spec is not None and not note:
            notes.append("get_road_network returned no edges")
    get_roads = getattr(bng, "get_roads", None)
    get_edges = getattr(bng, "get_road_edges", None)
    if callable(get_roads):
        roads, note = _call(get_roads)
        if note:
            notes.append(f"get_roads failed: {note}")
        elif isinstance(roads, dict) and roads and callable(get_edges):
            polys = []
            for road_id in roads:
                edges, edge_note = _call(get_edges, road_id)
                if edge_note or edges is None:
                    continue
                poly = parse_road_edges(str(road_id), edges)
                if poly is not None:
                    polys.append(poly)
            if polys:
                return RoadSnapshot(first_spec, polys, "")
            notes.append("get_road_edges returned no centerlines")
        elif not note:
            notes.append("get_roads returned no roads")
    elif not notes:
        notes.append("beamngpy has no get_road_network or get_roads")
    return RoadSnapshot(first_spec, [], "; ".join(notes) if notes else "no road network", first_error)


def read_legacy_roads(bng: Any) -> tuple[Any, str]:
    """``get_roads`` as a dict, for the offline road dumps only. ``(None, note)`` when absent."""
    get_roads = getattr(bng, "get_roads", None)
    if not callable(get_roads):
        return None, ""
    got, note = _call(get_roads)
    if note:
        return None, f"get_roads failed: {note}"
    return (got, "") if isinstance(got, dict) else (None, "")


def current_level(bng: Any) -> str:
    """The loaded level's name, or "" when this session cannot tell.

    Asks the game first (one Lua reply), then the scenario BeamNGpy last
    saw. Only called when roads are (re)loaded, never per tick.
    """
    if bng is None:
        return ""
    from python.collect.level_session import level_token, lua_call

    try:
        got = lua_call(bng, LEVEL_LUA)
    except Exception:
        got = None
    if isinstance(got, dict):
        got = got.get("result", got.get("value"))
    token = level_token(got) if isinstance(got, str) else ""
    if token:
        return token
    scenario = getattr(bng, "scenario", None)
    fn = getattr(scenario, "get_current", None)
    if callable(fn):
        try:
            current = fn(connect=False)
        except Exception:
            current = None
        level = getattr(current, "level", None)
        name = getattr(level, "name", level)
        if isinstance(name, str) and level_token(name):
            return level_token(name)
    return ""


def _same(ref: Any, bng: Any) -> bool:
    if ref is None:
        return False
    if isinstance(ref, weakref.ref):
        return ref() is bng
    return ref == id(bng)


def _ref(bng: Any) -> Any:
    try:
        return weakref.ref(bng)
    except TypeError:
        return id(bng)


class RoadLoader:
    """Road network cached per (session, level). ``reads`` counts session reads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: Any = None
        self._level: str | None = None
        self._snap: RoadSnapshot | None = None
        self.reads = 0

    @property
    def level(self) -> str | None:
        return self._level

    def enter(self, bng: Any, level: str | None) -> bool:
        """Point the cache at ``level`` on ``bng``. True when that dropped a cached network.

        The same level on the same session keeps the cache (a scenario restart
        or a recovery does not move roads). An unknown level ("") only matches
        an unknown level.
        """
        from python.collect.level_session import level_token

        key = level_token(level or "")
        with self._lock:
            if _same(self._owner, bng) and self._level == key:
                return False
            dropped = self._snap is not None
            self._owner = _ref(bng)
            self._level = key
            self._snap = None
            return dropped

    def invalidate(self) -> None:
        with self._lock:
            self._snap = None

    def snapshot(self, bng: Any) -> RoadSnapshot:
        """Cached roads for ``bng``, read once when nothing usable is cached."""
        if bng is None:
            return RoadSnapshot(None, [], "no beamng")
        with self._lock:
            if self._snap is not None and _same(self._owner, bng):
                return self._snap
            if not _same(self._owner, bng):
                self._owner = _ref(bng)
                self._level = None
                self._snap = None
        snap = read_network(bng)
        with self._lock:
            self.reads += 1
            if snap.polys and _same(self._owner, bng):
                self._snap = snap
        return snap

    def polys(self, bng: Any) -> tuple[list[RoadPoly], str]:
        snap = self.snapshot(bng)
        return list(snap.polys), snap.note

    def network(self, bng: Any) -> Any:
        """The raw network spec (for recovery snaps).

        A session error from ``get_road_network`` gives None (no roads to
        snap to). Any other exception is re-raised: a bad reply must not look
        like "no road here". The snapshot's note keeps the text either way;
        ``polys()`` and ``snapshot()`` stay note-only.
        """
        snap = self.snapshot(bng)
        if snap.error is not None:
            from python.collect.level_session import is_session_error

            if not is_session_error(snap.error):
                raise snap.error
        return snap.spec


_SHARED = RoadLoader()


def shared() -> RoadLoader:
    """The process-wide loader every runtime caller uses."""
    return _SHARED


def reset_shared() -> RoadLoader:
    """Fresh process-wide loader. Tests only."""
    global _SHARED
    _SHARED = RoadLoader()
    return _SHARED
