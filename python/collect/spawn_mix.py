"""Stratified west-coast spawns from a dumped navgraph.

Highway is a wide or one-way multi-lane road. A ramp is a short one-way
road that touches both a highway and a local road. Everything else that
is drivable is a city street. The mix is about 40% city, 35% highway,
25% ramps, and a road is used once per run.
"""

from __future__ import annotations

import gzip
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CITY = "city"
HIGHWAY = "highway"
RAMP = "ramp"
KINDS = (CITY, HIGHWAY, RAMP)
TARGET = {CITY: 0.40, HIGHWAY: 0.35, RAMP: 0.25}
SPEED = {CITY: (8.0, 14.0), HIGHWAY: (20.0, 30.0), RAMP: (10.0, 18.0)}
# A link counts when the ends sit about this close. The cell is only a search.
LINK_M = 30.0
CELL_M = 20.0


@dataclass(frozen=True)
class SpawnPick:
    kind: str
    road_id: str
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float]
    speed: tuple[float, float]


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _yaw_quat(dx: float, dy: float) -> tuple[float, float, float, float]:
    half = math.atan2(float(dy), float(dx)) * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _middles(road: dict[str, Any]) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for edge in road.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        middle = edge.get("middle")
        if isinstance(middle, (list, tuple)) and len(middle) >= 3:
            points.append((float(middle[0]), float(middle[1]), float(middle[2])))
    return points


def _pose_at(points: list[tuple[float, float, float]], fraction: float = 0.4) -> tuple[
    tuple[float, float, float], tuple[float, float, float, float]
] | None:
    if len(points) < 2:
        return None
    length = 0.0
    segs: list[tuple[tuple[float, float, float], tuple[float, float, float], float]] = []
    for left, right in zip(points, points[1:]):
        step = math.hypot(right[0] - left[0], right[1] - left[1])
        if step < 1e-3:
            continue
        segs.append((left, right, step))
        length += step
    if not segs or length < 1.0:
        return None
    target = min(length * float(fraction), max(0.0, length - 1.0))
    walked = 0.0
    left, right, step = segs[-1]
    for left, right, step in segs:
        if walked + step >= target:
            break
        walked += step
    mix = 0.0 if step < 1e-6 else min(1.0, max(0.0, (target - walked) / step))
    x = left[0] + (right[0] - left[0]) * mix
    y = left[1] + (right[1] - left[1]) * mix
    z = left[2] + (right[2] - left[2]) * mix + 0.5
    return (x, y, z), _yaw_quat(right[0] - left[0], right[1] - left[1])


def load_nav_roads(path: Path | str) -> list[dict[str, Any]]:
    """Drivable navgraph roads with a centerline of at least two points."""
    file = Path(path)
    opener = gzip.open if file.suffix == ".gz" else open
    with opener(file, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    network = payload.get("road_network") if isinstance(payload, dict) else None
    if not isinstance(network, dict):
        return []
    roads: list[dict[str, Any]] = []
    for road_id, road in network.items():
        if not isinstance(road, dict):
            continue
        points = _middles(road)
        if len(points) < 2:
            continue
        lanes = int(_num(road.get("lanesLeft"))) + int(_num(road.get("lanesRight")))
        length = _num(road.get("debugRoadLength"))
        drivability = _num(road.get("drivability"), -1.0)
        if drivability < 0.3 or length < 12.0 or lanes < 1:
            continue
        oneway = str(road.get("oneWay") or "0").lower() in {"1", "true"}
        roads.append(
            {
                "id": str(road_id),
                "pts": points,
                "lanes": lanes,
                "length": length,
                "oneway": oneway,
            }
        )
    return roads


def _is_highway(road: dict[str, Any]) -> bool:
    if int(road["lanes"]) >= 4:
        return True
    return bool(road["oneway"]) and int(road["lanes"]) >= 2 and float(road["length"]) >= 150.0


def _link_ids(roads: list[dict[str, Any]]) -> dict[str, set[str]]:
    cells: dict[tuple[int, int], list[tuple[str, float, float]]] = defaultdict(list)
    for road in roads:
        for point in (road["pts"][0], road["pts"][-1]):
            key = (int(point[0] // CELL_M), int(point[1] // CELL_M))
            cells[key].append((road["id"], point[0], point[1]))
    links: dict[str, set[str]] = {road["id"]: set() for road in roads}
    for road in roads:
        for point in (road["pts"][0], road["pts"][-1]):
            cx, cy = int(point[0] // CELL_M), int(point[1] // CELL_M)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for other, ox, oy in cells.get((cx + dx, cy + dy), ()):
                        if other == road["id"]:
                            continue
                        if math.hypot(ox - point[0], oy - point[1]) <= LINK_M:
                            links[road["id"]].add(other)
    return links


def classify_roads(roads: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Split ``roads`` into city, highway, and ramp pools."""
    highway = [road for road in roads if _is_highway(road)]
    highway_ids = {road["id"] for road in highway}
    links = _link_ids(roads)
    pools: dict[str, list[dict[str, Any]]] = {CITY: [], HIGHWAY: highway, RAMP: []}
    for road in roads:
        if road["id"] in highway_ids:
            continue
        neighbors = links.get(road["id"], set())
        touches_highway = any(item in highway_ids for item in neighbors)
        touches_local = any(item not in highway_ids for item in neighbors)
        ramp = (
            bool(road["oneway"])
            and int(road["lanes"]) <= 2
            and float(road["length"]) <= 250.0
            and touches_highway
            and touches_local
        )
        pools[RAMP if ramp else CITY].append(road)
    return pools


class SpawnDeck:
    """Hands out unused spawns in the 40/35/25 mix."""

    def __init__(self, pools: dict[str, list[dict[str, Any]]], rng: random.Random | None = None) -> None:
        self.pools = {kind: list(pools.get(kind) or []) for kind in KINDS}
        self.rng = rng or random.Random(0)
        for pool in self.pools.values():
            self.rng.shuffle(pool)
        self.used: set[str] = set()
        self.counts = {kind: 0 for kind in KINDS}

    @classmethod
    def load(cls, path: Path | str, rng: random.Random | None = None) -> SpawnDeck:
        return cls(classify_roads(load_nav_roads(path)), rng)

    def take(self) -> SpawnPick | None:
        kind = self._next_kind()
        if kind is None:
            return None
        pool = [road for road in self.pools[kind] if road["id"] not in self.used]
        if not pool:
            return None
        road = pool[0]
        pose = _pose_at(road["pts"])
        if pose is None:
            self.used.add(road["id"])
            return self.take()
        self.used.add(road["id"])
        self.counts[kind] += 1
        pos, rot = pose
        return SpawnPick(kind, road["id"], pos, rot, SPEED[kind])

    def _next_kind(self) -> str | None:
        total = sum(self.counts.values()) + 1
        best: str | None = None
        score = -1e9
        for kind in KINDS:
            if not any(road["id"] not in self.used for road in self.pools[kind]):
                continue
            deficit = TARGET[kind] * total - self.counts[kind]
            if deficit > score:
                best = kind
                score = deficit
        return best


def traffic_count(requested: int | None, map_name: str) -> int:
    """Companion cars. West coast defaults to 2. An explicit 0 stays 0."""
    if requested is None:
        return 2 if map_name == "west_coast_usa" else 0
    return max(0, int(requested))


def traffic_lua(pick: SpawnPick, count: int) -> list[str]:
    """GE Lua that replaces the companion cars ahead of and behind the ego.

    The player vehicle is not deleted and is not switched into traffic mode.
    """
    if count <= 0:
        return []
    dx = 0.0
    dy = 1.0
    # The yaw quaternion is (0, 0, sin(yaw/2), cos(yaw/2)). Recover the heading.
    _qx, _qy, qz, qw = pick.rot
    yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
    # BeamNG yaw-from-dir used atan2(dy, dx), so the road tangent is that heading.
    dx = math.cos(yaw)
    dy = math.sin(yaw)
    offsets = (28.0, -22.0)
    lines = [
        "local player = be:getPlayerVehicle(0)",
        "for i = be:getObjectCount() - 1, 0, -1 do",
        "  local v = be:getObject(i)",
        "  if v and v ~= player then",
        "    local name = ''",
        "    pcall(function() name = tostring(v:getField('name', 0)) end)",
        "    if string.find(name, 'gvd_traffic', 1, true) then v:delete() end",
        "  end",
        "end",
    ]
    speed = (float(pick.speed[0]) + float(pick.speed[1])) * 0.5
    x0, y0, z0 = pick.pos
    for slot in range(min(int(count), len(offsets))):
        dist = offsets[slot]
        x = x0 + dx * dist
        y = y0 + dy * dist
        z = z0
        qx, qy, qz, qw = pick.rot
        name = f"gvd_traffic_{slot}"
        lines.append(
            "local veh = core_vehicles.spawnNewVehicle(\"etk800\", "
            f"{{pos=vec3({x:.2f},{y:.2f},{z:.2f}), "
            f"rot=quat({qx:.5f},{qy:.5f},{qz:.5f},{qw:.5f}), "
            "autoEnterVehicle=false})"
        )
        lines.append("if veh then")
        lines.append(f"  pcall(function() veh:setField('name', 0, '{name}') end)")
        lines.append("  veh:queueLuaCommand('ai.setMode(\"span\")')")
        lines.append("  veh:queueLuaCommand('ai.driveInLane(\"on\")')")
        lines.append("  veh:queueLuaCommand('ai.setSpeedMode(\"limit\")')")
        lines.append(f"  veh:queueLuaCommand('ai.setSpeed({speed:.1f})')")
        lines.append("end")
    return [" ".join(lines)]
