"""Stratified west-coast spawns from a dumped navgraph.

Highway is a wide or one-way multi-lane road. A ramp is a short one-way
road that touches both a highway and a local road. Everything else that
is drivable is a city street. The mix is about 40% city, 35% highway,
25% ramps, and a road is used once per run.

With a painted-lane map, ``SpawnDeck`` can also skip a road whose next
300 m carries little line paint (``min_paint_cov``). The labeler reads the
paint, so a spawn on unpainted road records minutes of masked rows.
"""

from __future__ import annotations

import gzip
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

CITY = "city"
HIGHWAY = "highway"
RAMP = "ramp"
KINDS = (CITY, HIGHWAY, RAMP)
TARGET = {CITY: 0.40, HIGHWAY: 0.35, RAMP: 0.25}
SPEED = {CITY: (8.0, 14.0), HIGHWAY: (20.0, 30.0), RAMP: (10.0, 18.0)}
# A link counts when the ends sit about this close. The cell is only a search.
LINK_M = 30.0
CELL_M = 20.0
# Paint coverage of the road ahead of a spawn: one probe every 10 m over the
# next 300 m of the spawn road and the roads it continues into. A probe counts
# when the painted labeler finds the ego lane 10 m ahead of it, on the road
# centerline or half a lane to either side (a two-way road only looks right).
PAINT_STEP_M = 10.0
PAINT_AHEAD_M = 300.0
PAINT_PROBE_Y_M = 10.0
PAINT_LANE_M = 1.75
# Road ends this close join the walk ahead. The turn there must be under 60 deg.
PAINT_JOIN_M = 8.0
PAINT_JOIN_COS = 0.5
# Low-coverage candidates looked at per kind before falling back.
PAINT_MAX_TRIES = 40


@dataclass(frozen=True)
class SpawnPick:
    kind: str
    road_id: str
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float]
    speed: tuple[float, float]
    # Share of probes ahead with a painted ego lane. None: not measured.
    paint_cov: float | None = None


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


def _tail_from(
    points: list[tuple[float, float, float]], fraction: float = 0.4
) -> list[tuple[float, float, float]]:
    """The centerline from the ``_pose_at`` point to the road end, at road height."""
    if len(points) < 2:
        return list(points)
    total = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
    target = min(total * float(fraction), max(0.0, total - 1.0))
    walked = 0.0
    for i, (a, b) in enumerate(zip(points, points[1:])):
        step = math.hypot(b[0] - a[0], b[1] - a[1])
        if step >= 1e-3 and walked + step >= target:
            mix = min(1.0, max(0.0, (target - walked) / step))
            start = tuple(a[k] + (b[k] - a[k]) * mix for k in range(3))
            return [start] + [tuple(p) for p in points[i + 1 :]]
        walked += step
    return [tuple(p) for p in points[-2:]]


def _polyline_m(points: list[tuple[float, float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def road_ahead(
    road: dict[str, Any],
    roads_by_id: dict[str, dict[str, Any]],
    ends: dict[tuple[int, int], list[tuple[str, bool]]],
    ahead_m: float = PAINT_AHEAD_M,
    fraction: float = 0.4,
) -> list[tuple[float, float, float]]:
    """Centerline from the spawn point forward, continued into joining roads.

    At each road end the walk takes the joining road that turns least (under
    60 deg). A one-way road is only entered at its start. It stops at a dead
    end, a sharp turn, or ``ahead_m``.
    """
    chain = _tail_from(road["pts"], fraction)
    seen = {str(road["id"])}
    length = _polyline_m(chain)
    while length < float(ahead_m) and len(chain) >= 2 and len(seen) < 64:
        tail, prev = chain[-1], chain[-2]
        hx, hy = tail[0] - prev[0], tail[1] - prev[1]
        norm = math.hypot(hx, hy)
        if norm < 1e-6:
            break
        hx, hy = hx / norm, hy / norm
        cx, cy = int(tail[0] // PAINT_JOIN_M), int(tail[1] // PAINT_JOIN_M)
        best: tuple[float, str, list[tuple[float, float, float]]] | None = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for rid, at_start in ends.get((cx + dx, cy + dy), ()):
                    if rid in seen:
                        continue
                    other = roads_by_id.get(rid)
                    if other is None or (not at_start and other["oneway"]):
                        continue
                    seq = list(other["pts"]) if at_start else list(other["pts"])[::-1]
                    if math.hypot(seq[0][0] - tail[0], seq[0][1] - tail[1]) > PAINT_JOIN_M:
                        continue
                    k = 1
                    while k < len(seq) - 1 and math.hypot(seq[k][0] - seq[0][0], seq[k][1] - seq[0][1]) < 1.0:
                        k += 1
                    ux, uy = seq[k][0] - seq[0][0], seq[k][1] - seq[0][1]
                    un = math.hypot(ux, uy)
                    if un < 1e-6:
                        continue
                    dot = (ux * hx + uy * hy) / un
                    if dot >= PAINT_JOIN_COS and (best is None or dot > best[0]):
                        best = (dot, rid, seq)
        if best is None:
            break
        seen.add(best[1])
        added = [tuple(p) for p in best[2][1:]]
        length += _polyline_m([chain[-1]] + added)
        chain.extend(added)
    return chain


def sample_along(
    chain: list[tuple[float, float, float]],
    step_m: float = PAINT_STEP_M,
    ahead_m: float = PAINT_AHEAD_M,
) -> list[tuple[float, float, float, float, float]]:
    """``(x, y, z, fx, fy)`` every ``step_m`` from the chain start, below ``ahead_m``."""
    out: list[tuple[float, float, float, float, float]] = []
    step = max(1.0, float(step_m))
    next_at = 0.0
    walked = 0.0
    for a, b in zip(chain, chain[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg < 1e-6:
            continue
        fx, fy = (b[0] - a[0]) / seg, (b[1] - a[1]) / seg
        while next_at <= walked + seg + 1e-9 and next_at < float(ahead_m) - 1e-9:
            mix = (next_at - walked) / seg
            out.append(
                (
                    a[0] + (b[0] - a[0]) * mix,
                    a[1] + (b[1] - a[1]) * mix,
                    a[2] + (b[2] - a[2]) * mix,
                    fx,
                    fy,
                )
            )
            next_at += step
        walked += seg
    return out


def paint_coverage(
    painted: Any,
    samples: list[tuple[float, float, float, float, float]],
    *,
    oneway: bool = False,
    probe_y_m: float = PAINT_PROBE_Y_M,
) -> float:
    """Share of ``samples`` where ``painted.lane_at`` finds the ego lane at ``probe_y_m``.

    ``painted`` only needs ``lane_at(pos, direction, ys)``. No samples is 0.
    """
    if not samples:
        return 0.0
    offsets = (0.0, PAINT_LANE_M, -PAINT_LANE_M) if oneway else (0.0, PAINT_LANE_M)
    good = 0
    for x, y, z, fx, fy in samples:
        rx, ry = fy, -fx
        for off in offsets:
            got = painted.lane_at((x + rx * off, y + ry * off, z), (fx, fy, 0.0), (float(probe_y_m),))
            if got and got[0] is not None:
                good += 1
                break
    return good / float(len(samples))


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
    """Hands out unused spawns in the 40/35/25 mix.

    With ``painted`` and ``min_paint_cov`` > 0, a road whose paint coverage
    ahead is below the threshold is skipped (and used up). After
    ``max_tries`` skips in a kind the next kind is tried. When every kind is
    out of tries, the best-covered skipped road is returned anyway, so a
    respawn never fails because of paint. ``log`` gets one line per pick and
    per skip.
    """

    def __init__(
        self,
        pools: dict[str, list[dict[str, Any]]],
        rng: random.Random | None = None,
        *,
        painted: Any = None,
        min_paint_cov: float = 0.0,
        max_tries: int = PAINT_MAX_TRIES,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.pools = {kind: list(pools.get(kind) or []) for kind in KINDS}
        self.rng = rng or random.Random(0)
        for pool in self.pools.values():
            self.rng.shuffle(pool)
        self.used: set[str] = set()
        self.counts = {kind: 0 for kind in KINDS}
        self.painted = painted
        self.min_paint_cov = max(0.0, float(min_paint_cov or 0.0))
        self.max_tries = max(1, int(max_tries))
        self.log = log
        self.skipped = 0
        # Picks of a kind other than the one the mix asked for, and picks of
        # the best-covered skipped road because nothing passed. One pick can
        # be both.
        self.cross_kind = 0
        self.best_seen = 0
        self.last_ms = 0.0
        self.select_ms: list[float] = []
        self._by_id: dict[str, dict[str, Any]] = {}
        self._ends: dict[tuple[int, int], list[tuple[str, bool]]] = defaultdict(list)
        for pool in self.pools.values():
            for road in pool:
                self._by_id[str(road["id"])] = road
                for point, at_start in ((road["pts"][0], True), (road["pts"][-1], False)):
                    key = (int(point[0] // PAINT_JOIN_M), int(point[1] // PAINT_JOIN_M))
                    self._ends[key].append((str(road["id"]), at_start))

    @classmethod
    def load(
        cls,
        path: Path | str,
        rng: random.Random | None = None,
        **kwargs: Any,
    ) -> SpawnDeck:
        return cls(classify_roads(load_nav_roads(path)), rng, **kwargs)

    def filtering(self) -> bool:
        return self.painted is not None and bool(self.painted) and self.min_paint_cov > 0.0

    def coverage(self, road: dict[str, Any]) -> float:
        """Paint coverage over the next ``PAINT_AHEAD_M`` from this road's spawn point."""
        chain = road_ahead(road, self._by_id, self._ends, PAINT_AHEAD_M)
        return paint_coverage(self.painted, sample_along(chain), oneway=bool(road["oneway"]))

    def _say(self, text: str) -> None:
        if self.log is not None:
            self.log(text)

    def _pick(self, kind: str, road: dict[str, Any], pose: Any, cov: float | None, wanted: str) -> SpawnPick:
        self.counts[kind] += 1
        if kind != wanted:
            self.cross_kind += 1
        pos, rot = pose
        return SpawnPick(kind, road["id"], pos, rot, SPEED[kind], cov)

    def take(self) -> SpawnPick | None:
        """Next spawn, or None. ``last_ms`` is this call's wall time."""
        start = time.perf_counter()
        try:
            return self._take()
        finally:
            self.last_ms = (time.perf_counter() - start) * 1000.0
            self.select_ms.append(self.last_ms)

    def summary(self) -> dict[str, Any]:
        """Spawn mix so far, for the periodic log line and collect.json."""
        picks = sum(self.counts.values())
        times = self.select_ms
        return {
            "picks": picks,
            "counts": dict(self.counts),
            "share": {kind: (self.counts[kind] / picks if picks else 0.0) for kind in KINDS},
            "target": dict(TARGET),
            "skipped": self.skipped,
            "cross_kind_fallback": self.cross_kind,
            "best_seen_fallback": self.best_seen,
            "min_paint_cov": self.min_paint_cov if self.filtering() else 0.0,
            "select_ms_mean": (sum(times) / len(times)) if times else None,
            "select_ms_max": max(times) if times else None,
        }

    def summary_line(self) -> str:
        s = self.summary()
        mix = " ".join(f"{kind}={s['counts'][kind]}" for kind in KINDS)
        mean = "na" if s["select_ms_mean"] is None else format(s["select_ms_mean"], ".0f")
        peak = "na" if s["select_ms_max"] is None else format(s["select_ms_max"], ".0f")
        return (
            f"spawn mix picks={s['picks']} {mix} skipped={s['skipped']} "
            f"cross_kind={s['cross_kind_fallback']} best_seen={s['best_seen_fallback']} "
            f"select_ms mean={mean} max={peak}"
        )

    def _take(self) -> SpawnPick | None:
        first = self._next_kind()
        if first is None:
            return None
        order = [first] + [kind for kind in self._kinds_by_deficit() if kind != first]
        filtering = self.filtering()
        tried: list[tuple[float, str, dict[str, Any], Any]] = []
        for kind in order:
            budget = self.max_tries
            for road in [item for item in self.pools[kind] if item["id"] not in self.used]:
                pose = _pose_at(road["pts"])
                self.used.add(road["id"])
                if pose is None:
                    continue
                if not filtering:
                    return self._pick(kind, road, pose, None, first)
                cov = self.coverage(road)
                if cov >= self.min_paint_cov:
                    self._say(
                        f"spawn pick {kind} road={road['id']} paint_cov={cov:.2f} "
                        f"(>={self.min_paint_cov:.2f}, skipped {len(tried)})"
                    )
                    return self._pick(kind, road, pose, cov, first)
                self.skipped += 1
                tried.append((cov, kind, road, pose))
                self._say(
                    f"spawn skip {kind} road={road['id']} paint_cov={cov:.2f} (<{self.min_paint_cov:.2f})"
                )
                budget -= 1
                if budget <= 0:
                    break
        if not tried:
            return None
        cov, kind, road, pose = max(tried, key=lambda item: item[0])
        self._say(
            f"spawn fallback {kind} road={road['id']} paint_cov={cov:.2f} "
            f"(no road >= {self.min_paint_cov:.2f} in {len(tried)} tries)"
        )
        self.best_seen += 1
        return self._pick(kind, road, pose, cov, first)

    def _kinds_by_deficit(self) -> list[str]:
        total = sum(self.counts.values()) + 1
        return sorted(KINDS, key=lambda kind: -(TARGET[kind] * total - self.counts[kind]))

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
