"""A real route for a long regular-driving clip: highway end to end, then a loop back.

Michael: "have the Game AI start on the end of the highway go to the other end
of it and start driving around and do a big loop around back to the highway
using different routes each time in different directions".

The planner works on the road dump ``spawn_mix`` already reads. Its graph has
one node per centerline point; consecutive points are linked in the legal
direction (``flipDirection`` and one-way honoured), and a road end that sits
within ``JOIN_M`` of another road joins it. A route is

* a **highway leg**: from one end of a highway corridor to the far end, over
  highway roads only;
* a **return leg**: from the far end back to the start end over non-highway
  roads only (city, ramps, and unpaved roads where the dump's road material
  says dirt / mud / gravel), through a seeded via point with seeded edge-cost
  jitter, so each clip loops back a different way.

The line is offset into the legal right-hand lane (``_offset_candidates``, the
rule the spawn deck uses), so it can be sent to the AI as ``vehicle.ai.set_line``.

``RouteDriver`` turns a position into AI commands: it sends the remaining
route once as a ``set_line`` command (the helper ``excursion.py`` already
makes) and hands the car back to ``span`` at the end or when it is off the route.
Mid-drive speed changes go out as ``ai.setSpeed`` only: re-issuing ``set_line``
makes Tech ``driveUsingPath({script=...})`` teleport to the first script point.
``measure_route`` scores a recording against the route from the ego positions.

No BeamNG here, so everything runs offline. Nothing in this module was run
against a live BeamNG; the collector wiring is described in
``scripts/collect_scene_lanes.py`` (``--route-plan``).
"""

from __future__ import annotations

import heapq
from functools import lru_cache
import json
import math
import random
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from python.collect.excursion import set_line_command
from python.collect.spawn_mix import (
    CITY,
    HIGHWAY,
    RAMP,
    SPEED,
    SpawnPick,
    _effective_oneway,
    _offset_candidates,
    _station_width,
    _yaw_quat,
    classify_roads,
)

# A road end within this of another road's centerline point joins it.
JOIN_M = 15.0
JOIN_DZ_M = 5.0
# A junction hop costs this much extra, so the planner does not zig-zag across roads.
LINK_COST_M = 8.0
# Smallest highway leg worth calling "end to end".
MIN_HIGHWAY_M = 800.0
# Fractions of the distance budget (clip seconds x AVG_SPEED_MPS) for each leg.
AVG_SPEED_MPS = 11.0
HIGHWAY_BUDGET_FRAC = 0.45
MIN_RETURN_M = 1500.0
# A route is at most this multiple of the clip's driving budget (duration x AVG_SPEED_MPS); it prefers at least the lower one unless that costs the requested direction.
BUDGET_MAX_FRAC = 1.25
BUDGET_MIN_FRAC = 0.6
# Unpaved roads are cheaper on the return leg, so they are chosen when present.
DIRT_COST = 0.5
JITTER_LO, JITTER_HI = 0.85, 1.7
DIRT_WORDS = ("dirt", "mud", "gravel", "tiretracks")

# Driver and measure tolerances.
# The first set_line of a segment is the remaining route (no mid-drive window
# slice). Re-issuing set_line mid-drive teleports: Tech driveUsingPath({script})
# always places the car at script[1].
WINDOW_M = float("inf")
# Historical mid-drive re-issue distance (80 m). Kept so tests can pin that we
# no longer re-send set_line at this gap. Not used to trigger a new line.
RESEND_M = 80.0
OFF_ROUTE_M = 45.0
OFF_TICKS = 8
# Mid-drive wrong-deck: this many consecutive over-limit ticks, so a one-sample z glitch does not end the segment.
DECK_TICKS = 3
# Prefer a still-on-route projection within this forward arc over a stacked later leg
# inside the 400 m look-ahead (smoke-c: s~2604 under s~2941, 337 m later).
MAX_ARC_JUMP_M = 200.0
END_TOL_M = 25.0
FAR_END_TOL_M = 60.0
SEND_SPEED_DELTA = 0.5
# A respawn on the route (see ``RouteDriver.resume``): past where the car was, so a spot that stalled it is not
# the spawn again; at most MAX_RESUMES per clip; and not within RESUME_MIN_LEFT_M of the end.
RESUME_NUDGE_M = 40.0
RESUME_MIN_LEFT_M = 150.0
MAX_RESUMES = 3

# Highest AI speed (m/s) on each kind of road; the highway takes the clip's own speed.
KIND_SPEED_CAP = {"highway": None, "ramp": 16.0, "city": 14.0, "dirt": 11.0}


def route_place_z_error(
    requested: Sequence[float] | None,
    settled: Sequence[float] | None,
    *,
    limit_m: float = JOIN_DZ_M,
) -> str | None:
    """Return an error if the settled z is more than ``limit_m`` from the reference z.

    The first pose is the route plan z (or any reference z the caller chooses).
    After a route teleport, and mid-drive, the car must rest on the planned deck,
    not a stacked surface above or below it. ``JOIN_DZ_M`` (5 m) is the same
    vertical join used when linking roads. A missing or short pose is also an error.
    """
    if requested is None or settled is None:
        return "route place: settled pose missing z"
    try:
        if len(requested) < 3 or len(settled) < 3:
            return "route place: settled pose missing z"
        req_z = float(requested[2])
        got_z = float(settled[2])
    except (TypeError, ValueError):
        return "route place: settled pose missing z"
    if not (math.isfinite(req_z) and math.isfinite(got_z)):
        return "route place: settled pose missing z"
    dz = got_z - req_z
    if abs(dz) > float(limit_m):
        return f"route place landed {dz:+.1f} m off the planned z (limit {float(limit_m):.0f} m)"
    return None


def _interp_plan_z(route: "Route", i: int, s: float) -> float:
    """Plan z at arc length ``s`` on segment ``i``."""
    pts, cum = route.points, route.cum
    if i >= len(pts) - 1:
        return float(pts[-1][2])
    seg = max(1e-6, cum[i + 1] - cum[i])
    t = min(1.0, max(0.0, (s - cum[i]) / seg))
    return float(pts[i][2] + (pts[i + 1][2] - pts[i][2]) * t)


def route_plan_z(route: "Route", x: float, y: float, start_index: int = 0) -> float:
    """The planned route height at the projection of ``(x, y)``."""
    i, s, _d = project_forward(route, x, y, start_index, ahead_m=1e9)
    return _interp_plan_z(route, i, s)


def is_dirt_material(material: object) -> bool:
    """True for a decal-road material that is unpaved (dirt, mud, gravel, tire tracks)."""
    text = str(material or "").lower()
    return any(word in text for word in DIRT_WORDS)


def _hash01(seed: int, tag: str, a: int, b: int) -> float:
    """Deterministic 0..1 from (seed, tag, edge): cheap enough for every relaxed edge."""
    h = (int(seed) * 1000003 ^ hash_tag(tag) * 7919 ^ int(a) * 2654435761 ^ int(b) * 40503) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 2246822519) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 3266489917) & 0xFFFFFFFF
    h ^= h >> 16
    return h / 4294967296.0


@lru_cache(maxsize=256)
def hash_tag(tag: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(tag)) & 0xFFFF


@dataclass(frozen=True)
class Route:
    """One planned long clip route. ``points`` are right-lane x, y, z; ``kinds`` is per point."""

    map: str
    seed: int
    direction: int
    requested_direction: int
    points: tuple[tuple[float, float, float], ...]
    kinds: tuple[str, ...]
    far_end_index: int
    highway_m: float
    return_m: float
    dirt_m: float
    city_m: float
    ramp_m: float
    legs: tuple[str, ...]
    start_road: str
    far_road: str
    via: tuple[float, float] | None = None
    cum: tuple[float, ...] = field(default=(), repr=False, compare=False)

    @property
    def total_m(self) -> float:
        return self.cum[-1] if self.cum else 0.0

    @property
    def far_end_m(self) -> float:
        return self.cum[self.far_end_index] if self.cum else 0.0

    def start_pose(self, z_lift: float = 0.5) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Spawn pose at the start end of the highway, facing along the route."""
        p0 = self.points[0]
        look = next((p for p in self.points[1:] if math.hypot(p[0] - p0[0], p[1] - p0[1]) >= 10.0), self.points[-1])
        return (p0[0], p0[1], p0[2] + z_lift), _yaw_quat(look[0] - p0[0], look[1] - p0[1])

    def bearing_deg(self) -> float:
        """Compass-style bearing of start -> far end (0 = +y, clockwise)."""
        a, b = self.points[0], self.points[self.far_end_index]
        return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360.0

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "gvd_long_route",
            "map": self.map, "seed": self.seed, "direction": self.direction,
            "requested_direction": self.requested_direction,
            "points": [[round(x, 2), round(y, 2), round(z, 2)] for x, y, z in self.points],
            "kinds": list(self.kinds),
            "far_end_index": self.far_end_index,
            "highway_m": round(self.highway_m, 1), "return_m": round(self.return_m, 1),
            "dirt_m": round(self.dirt_m, 1), "city_m": round(self.city_m, 1), "ramp_m": round(self.ramp_m, 1),
            "legs": list(self.legs), "start_road": self.start_road, "far_road": self.far_road,
            "via": None if self.via is None else [round(self.via[0], 1), round(self.via[1], 1)],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Route":
        pts = tuple((float(a), float(b), float(c)) for a, b, c in data["points"])
        via = data.get("via")
        return cls(
            map=str(data["map"]), seed=int(data["seed"]), direction=int(data["direction"]),
            requested_direction=int(data.get("requested_direction", data["direction"])),
            points=pts, kinds=tuple(str(k) for k in data["kinds"]),
            far_end_index=int(data["far_end_index"]),
            highway_m=float(data["highway_m"]), return_m=float(data["return_m"]),
            dirt_m=float(data.get("dirt_m", 0.0)), city_m=float(data.get("city_m", 0.0)),
            ramp_m=float(data.get("ramp_m", 0.0)),
            legs=tuple(str(x) for x in data["legs"]), start_road=str(data.get("start_road", "")),
            far_road=str(data.get("far_road", "")), via=None if via is None else (float(via[0]), float(via[1])),
            cum=_cumulative(pts),
        )


def _cumulative(points: Sequence[Sequence[float]]) -> tuple[float, ...]:
    out = [0.0]
    for a, b in zip(points, points[1:]):
        out.append(out[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    return tuple(out)


class RoadGraph:
    """Directed point graph of the dump's roads."""

    def __init__(self, roads: Sequence[dict[str, Any]], highway_ids: Iterable[str], *, materials: dict[str, str] | None = None) -> None:
        self.roads = list(roads)
        hw = {str(i) for i in highway_ids}
        mats = materials or {}
        self.is_highway = [str(r["id"]) in hw for r in self.roads]
        self.is_dirt = [is_dirt_material(mats.get(str(r["id"]), r.get("material"))) for r in self.roads]
        self.is_ramp = [False] * len(self.roads)
        self.node_of: dict[tuple[int, int], int] = {}
        self.node_road: list[int] = []
        self.xyz: list[tuple[float, float, float]] = []
        for ri, road in enumerate(self.roads):
            for pi, p in enumerate(road["pts"]):
                self.node_of[(ri, pi)] = len(self.xyz)
                self.node_road.append(ri)
                self.xyz.append((float(p[0]), float(p[1]), float(p[2])))
        self.node_pi = [pi for ri, road in enumerate(self.roads) for pi in range(len(road["pts"]))]
        # adj[n] = [(to, length_m, road_index or -1 for a junction link)]
        self.adj: list[list[tuple[int, float, int]]] = [[] for _ in self.xyz]
        for ri, road in enumerate(self.roads):
            one = _effective_oneway(road)
            flip = bool(road.get("flip"))
            fwd_ok = (not one) or (not flip)
            back_ok = (not one) or flip
            for pi in range(len(road["pts"]) - 1):
                a, b = self.node_of[(ri, pi)], self.node_of[(ri, pi + 1)]
                d = math.hypot(self.xyz[a][0] - self.xyz[b][0], self.xyz[a][1] - self.xyz[b][1])
                if d < 0.05:
                    continue
                if fwd_ok:
                    self.adj[a].append((b, d, ri))
                if back_ok:
                    self.adj[b].append((a, d, ri))
        self._quality: list[Any] | None = None
        self._grid: dict[Any, Any] = {}
        self._link()
        self.ends = [
            n for ri, road in enumerate(self.roads) if self.is_highway[ri]
            for n in (self.node_of[(ri, 0)], self.node_of[(ri, len(road["pts"]) - 1)])
        ]

    def _link(self) -> None:
        cell: dict[tuple[int, int], list[int]] = defaultdict(list)
        for n, (x, y, _z) in enumerate(self.xyz):
            cell[(int(x // JOIN_M), int(y // JOIN_M))].append(n)
        seen: set[tuple[int, int]] = set()
        for ri, road in enumerate(self.roads):
            for pi in (0, len(road["pts"]) - 1):
                a = self.node_of[(ri, pi)]
                ax, ay, az = self.xyz[a]
                best: dict[int, tuple[float, int]] = {}
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for b in cell.get((int(ax // JOIN_M) + dx, int(ay // JOIN_M) + dy), ()):
                            rb = self.node_road[b]
                            if rb == ri:
                                continue
                            bx, by, bz = self.xyz[b]
                            d = math.hypot(bx - ax, by - ay)
                            if d <= JOIN_M and abs(bz - az) <= JOIN_DZ_M and (rb not in best or d < best[rb][0]):
                                best[rb] = (d, b)
                for d, b in best.values():
                    key = (min(a, b), max(a, b))
                    if key in seen:
                        continue
                    seen.add(key)
                    self.adj[a].append((b, d + LINK_COST_M, -1))
                    self.adj[b].append((a, d + LINK_COST_M, -1))

    def pose_legal(self, pos: Sequence[float], rot: Sequence[float]) -> bool:
        """The spawn deck's own check: right side of the road, with the traffic."""
        from python.collect.spawn_mix import _quality_from_road
        from python.train.frame_quality import build_road_grid, classify_wrong_way, spawn_pose_rejected

        if self._quality is None:
            self._quality = [q for q in (_quality_from_road(r) for r in self.roads) if q is not None]
            self._grid = build_road_grid(self._quality) if self._quality else {}
        if not self._quality:
            return False
        _qx, _qy, qz, qw = rot
        yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
        got = classify_wrong_way(pos[0], pos[1], yaw, self._quality, self._grid, z=pos[2])
        return str(got.get("verdict") or "") == "ok" and not spawn_pose_rejected(got)

    def road_kind(self, ri: int) -> str:
        return HIGHWAY if self.is_highway[ri] else (RAMP if self.is_ramp[ri] else CITY)

    def dijkstra(
        self,
        src: int,
        cost_fn: Callable[[int, int, float, int], float | None],
        *,
        goal: int | None = None,
        limit: float = math.inf,
    ) -> tuple[dict[int, float], dict[int, int]]:
        dist = {src: 0.0}
        prev: dict[int, int] = {}
        heap = [(0.0, src)]
        while heap:
            d, n = heapq.heappop(heap)
            if d > dist.get(n, math.inf) + 1e-9:
                continue
            if goal is not None and n == goal:
                break
            for to, length, ri in self.adj[n]:
                c = cost_fn(n, to, length, ri)
                if c is None:
                    continue
                nd = d + c
                if nd < dist.get(to, math.inf) and nd <= limit:
                    dist[to] = nd
                    prev[to] = n
                    heapq.heappush(heap, (nd, to))
        return dist, prev

    @staticmethod
    def path(prev: dict[int, int], src: int, dst: int) -> list[int]:
        if dst != src and dst not in prev:
            return []
        out = [dst]
        while out[-1] != src:
            out.append(prev[out[-1]])
        out.reverse()
        return out

    def _is_hw_node(self, n: int) -> bool:
        return self.is_highway[self.node_road[n]]

    def highway_cost(self, frm: int, to: int, length: float, ri: int) -> float | None:
        """Highway leg: highway road edges, and links between two highway nodes."""
        if ri >= 0:
            return length if self.is_highway[ri] else None
        return length if (self._is_hw_node(frm) and self._is_hw_node(to)) else None

    def return_cost(self, seed: int, tag: str, avoid: set[int] | None = None) -> Callable[[int, int, float, int], float | None]:
        """Return leg: no highway edges, unpaved cheaper, seeded jitter per edge."""
        avoid = avoid or set()

        def cost(frm: int, to: int, length: float, ri: int) -> float | None:
            if ri >= 0:
                if self.is_highway[ri]:
                    return None
                c = length * (DIRT_COST if self.is_dirt[ri] else 1.0)
            else:
                if self._is_hw_node(frm) and self._is_hw_node(to):
                    return None
                c = length
            c *= JITTER_LO + (JITTER_HI - JITTER_LO) * _hash01(seed, tag, frm, to)
            if to in avoid:
                c *= 4.0
            return c

        return cost

    def lane_offset(self, n: int) -> float:
        """Metres right of the travel tangent for the node's road (rightmost lane, spawn-deck rule)."""
        ri, pi = self.node_road[n], self.node_pi[n]
        road = self.roads[ri]
        widths = road.get("widths") or []
        local = widths[pi] if pi < len(widths) else None
        width = _station_width(local, road)
        if width is None or not road.get("width"):
            return 0.0
        try:
            cands = _offset_candidates(road, width)
        except (KeyError, TypeError, ValueError):
            return 0.0
        return float(cands[0]) if cands else 0.0


def build_graph(roads: Sequence[dict[str, Any]], pools: dict[str, list[dict[str, Any]]] | None = None) -> RoadGraph:
    pools = pools if pools is not None else classify_roads(list(roads))
    graph = RoadGraph(roads, [r["id"] for r in pools.get(HIGHWAY, [])])
    ramps = {str(r["id"]) for r in pools.get(RAMP, [])}
    graph.is_ramp = [str(r["id"]) in ramps for r in graph.roads]
    return graph


def _polyline(graph: RoadGraph, nodes: Sequence[int]) -> tuple[list[tuple[float, float, float]], list[str]]:
    """Right-lane points and a kind per point for a node path."""
    pts: list[tuple[float, float, float]] = []
    kinds: list[str] = []
    for i, n in enumerate(nodes):
        x, y, z = graph.xyz[n]
        a = graph.xyz[nodes[max(0, i - 1)]]
        b = graph.xyz[nodes[min(len(nodes) - 1, i + 1)]]
        tx, ty = b[0] - a[0], b[1] - a[1]
        norm = math.hypot(tx, ty)
        off = graph.lane_offset(n) if norm > 1e-6 else 0.0
        if norm > 1e-6:
            tx, ty = tx / norm, ty / norm
            x, y = x + ty * off, y - tx * off
        if pts and math.hypot(x - pts[-1][0], y - pts[-1][1]) < 1.0:
            continue
        pts.append((x, y, z))
        ri = graph.node_road[n]
        kind = HIGHWAY if graph.is_highway[ri] else ("dirt" if graph.is_dirt[ri] else (RAMP if graph.is_ramp[ri] else CITY))
        kinds.append(kind)
    return pts, kinds


def _leg_lengths(points: Sequence[Sequence[float]], kinds: Sequence[str], lo: int, hi: int) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for i in range(lo, hi):
        out[kinds[i + 1]] += math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
    return out


def _entry_nodes(graph: RoadGraph) -> list[int]:
    """Highway nodes a car can drive onto from a non-highway road (an on-ramp or a joining street)."""
    out = []
    for n in range(len(graph.xyz)):
        if not graph._is_hw_node(n):
            continue
        has_in_link = any(
            ri < 0 and not graph._is_hw_node(to) for to, _l, ri in graph.adj[n]
        )
        has_out_hw = any(ri >= 0 and graph.is_highway[ri] for _to, _l, ri in graph.adj[n])
        if has_in_link and has_out_hw:
            out.append(n)
    return out


def _exit_nodes(graph: RoadGraph) -> set[int]:
    """Highway nodes with a junction link to a non-highway road (where a car can leave)."""
    return {
        n for n in range(len(graph.xyz))
        if graph._is_hw_node(n) and any(ri < 0 and not graph._is_hw_node(to) for to, _l, ri in graph.adj[n])
    }


def _bearing(a: Sequence[float], b: Sequence[float]) -> float:
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360.0


def plan_highway_route(
    graph: RoadGraph,
    *,
    map_name: str,
    seed: int,
    direction: int,
    duration_s: float = 750.0,
    tries: int = 24,
) -> Route | None:
    """Highway end to end, then a different loop back to the highway. None when the map has none.

    ``direction`` (0 or 1) asks for a start -> far-end bearing in the first or second half
    of the compass; the route's own ``direction`` says which it got (a map whose highways
    only run one way cannot honour both).
    """
    if not graph.ends:
        return None
    rng = random.Random(f"gvd-long-route:{map_name}:{int(seed)}")
    budget = float(duration_s) * AVG_SPEED_MPS
    hw_cap = max(MIN_HIGHWAY_M * 1.5, HIGHWAY_BUDGET_FRAC * budget)
    want = int(direction) % 2
    exits = _exit_nodes(graph)
    entries = _entry_nodes(graph)
    has_in = {to for n in range(len(graph.xyz)) for to, _l, ri in graph.adj[n] if ri >= 0 and graph.is_highway[ri]}
    starts = [n for n in dict.fromkeys(graph.ends) if n not in has_in] or list(dict.fromkeys(graph.ends))
    rng.shuffle(starts)
    best: Route | None = None
    # First a real loop that fills the clip, then any real loop, then, only if the map has none, any loop.
    # Never longer than BUDGET_MAX_FRAC x the budget: a route the clip cannot finish is not planned.
    max_total = BUDGET_MAX_FRAC * budget
    for min_return, min_total in ((MIN_RETURN_M, BUDGET_MIN_FRAC * budget), (MIN_RETURN_M, 0.0), (0.0, 0.0)):
        for s0 in starts[: int(tries)]:
            dist, prev = graph.dijkstra(s0, graph.highway_cost, limit=hw_cap)
            far = sorted((n for n in exits if n != s0 and dist.get(n, 0.0) >= MIN_HIGHWAY_M), key=lambda n: -dist[n])
            for f0 in far[:3]:
                hw_nodes = graph.path(prev, s0, f0)
                if len(hw_nodes) < 2:
                    continue
                ret = _plan_return(graph, rng, seed, s0, f0, entries, budget - dist[f0], min_return)
                if ret is None:
                    continue
                route = _assemble(graph, map_name, seed, want, hw_nodes, ret)
                if route is None or route.return_m < min_return or not (min_total <= route.total_m <= max_total):
                    continue
                if not graph.pose_legal(*route.start_pose()):
                    continue
                if route.direction == want:
                    return route
                best = best or route
                break
    return best


def _assemble(graph: RoadGraph, map_name: str, seed: int, want: int, hw_nodes: list[int], ret: tuple[list[int], int | None]) -> Route | None:
    ret_nodes, via = ret
    hw_pts, hw_kinds = _polyline(graph, hw_nodes)
    rt_pts, rt_kinds = _polyline(graph, ret_nodes[1:])
    pts = hw_pts + rt_pts
    kinds = hw_kinds + rt_kinds
    if len(pts) < 4 or len(hw_pts) < 2:
        return None
    cum = _cumulative(pts)
    far_i = len(hw_pts) - 1
    leg_rt = _leg_lengths(pts, kinds, far_i, len(pts) - 1)
    legs = ["highway"]
    if leg_rt.get(CITY, 0) + leg_rt.get(RAMP, 0) > 0:
        legs.append("city")
    if leg_rt.get("dirt", 0) > 0:
        legs.append("dirt")
    legs.append("highway")
    bearing = _bearing(pts[0], pts[far_i])
    return Route(
        map=str(map_name), seed=int(seed), direction=(0 if bearing < 180.0 else 1), requested_direction=int(want),
        points=tuple(pts), kinds=tuple(kinds), far_end_index=far_i,
        highway_m=cum[far_i], return_m=cum[-1] - cum[far_i],
        dirt_m=leg_rt.get("dirt", 0.0), city_m=leg_rt.get(CITY, 0.0), ramp_m=leg_rt.get(RAMP, 0.0),
        legs=tuple(legs), start_road=str(graph.roads[graph.node_road[hw_nodes[0]]]["id"]),
        far_road=str(graph.roads[graph.node_road[hw_nodes[-1]]]["id"]),
        via=None if via is None else (graph.xyz[via][0], graph.xyz[via][1]), cum=cum,
    )


def _plan_return(
    graph: RoadGraph, rng: random.Random, seed: int, start: int, far_end: int, entries: Sequence[int], remaining_m: float, min_return_m: float = 0.0,
) -> tuple[list[int], int | None] | None:
    """Far end -> back onto the highway near the start, over non-highway roads, via a seeded point."""
    target = max(MIN_RETURN_M, remaining_m)
    cost1 = graph.return_cost(seed, "out")
    dist, prev = graph.dijkstra(far_end, cost1, limit=target * 1.6)
    sx, sy = graph.xyz[start][0], graph.xyz[start][1]
    reach = [e for e in entries if e in dist and e != far_end and dist[e] >= float(min_return_m)]
    if not reach:
        return None
    reach.sort(key=lambda e: math.hypot(graph.xyz[e][0] - sx, graph.xyz[e][1] - sy))
    home = reach[min(len(reach) - 1, rng.randrange(0, min(3, len(reach))))]
    candidates = [n for n, d in dist.items() if 0.3 * target <= d <= 0.6 * target and not graph._is_hw_node(n)]
    dirt = [n for n in candidates if graph.is_dirt[graph.node_road[n]]]
    rng.shuffle(candidates)
    rng.shuffle(dirt)
    pool = (dirt[:6] + candidates[:10]) if dirt and rng.random() < 0.7 else candidates[:12]
    for via in pool:
        first = graph.path(prev, far_end, via)
        if len(first) < 2:
            continue
        cost2 = graph.return_cost(seed, "back", avoid=set(first))
        d2, p2 = graph.dijkstra(via, cost2, goal=home)
        second = graph.path(p2, via, home)
        if len(second) >= 2:
            return first + second[1:], via
    direct = graph.path(prev, far_end, home)
    if len(direct) >= 2:
        return direct, None
    return None


@lru_cache(maxsize=8)
def graph_for_map(map_name: str) -> RoadGraph | None:
    """The map's road graph from its dump (cached). None when there is no dump."""
    from python.collect.spawn_mix import load_nav_roads, spawn_roads_path

    path = spawn_roads_path(str(map_name))
    if path is None or not path.is_file():
        return None
    return build_graph(load_nav_roads(path))


def map_unpaved_m(map_name: str) -> float:
    """Total length of the map's unpaved roads (dirt, mud, gravel, tire tracks) in its dump; 0 with no dump."""
    graph = graph_for_map(map_name)
    if graph is None:
        return 0.0
    return float(sum(
        _cumulative(r["pts"])[-1] for ri, r in enumerate(graph.roads) if graph.is_dirt[ri]
    ))


def route_for_map(map_name: str, *, seed: int, direction: int, duration_s: float = 750.0) -> Route | None:
    """Plan from the map's road dump. None when there is no dump or no highway."""
    graph = graph_for_map(map_name)
    if graph is None:
        return None
    return plan_highway_route(graph, map_name=map_name, seed=seed, direction=direction, duration_s=duration_s)


# --------------------------------------------------------------------------- driver


def span_commands(speed_mps: float) -> list[str]:
    """Hand the car back to the AI's own roaming."""
    return ['ai.setMode("span")', 'ai.driveInLane("on")', 'ai.setSpeedMode("limit")', f"ai.setSpeed({float(speed_mps)})"]


class RouteDriver:
    """Position in, AI commands out. ``commands`` is called once per tick by the collector."""

    def __init__(
        self, route: Route, *, window_m: float = WINDOW_M, resend_m: float = RESEND_M,
        off_route_m: float = OFF_ROUTE_M, off_ticks: int = OFF_TICKS, end_tol_m: float = END_TOL_M,
    ) -> None:
        self.route = route
        self.window_m = float(window_m)
        self.resend_m = float(resend_m)
        self.off_route_m = float(off_route_m)
        self.off_ticks = int(off_ticks)
        self.end_tol_m = float(end_tol_m)
        self.state = "idle"  # idle -> following -> done | off_route
        self.index = 0
        self.progress_m = 0.0
        self.lines_sent = 0
        self.max_off_route_m = 0.0
        self.far_end_reached = False
        self.spawns = 0  # route spawns taken: the start (1) and every resume
        self.resumes = 0
        self.skipped_m = 0.0  # route metres jumped over by resumes (the nudges), not driven
        self._off = 0
        self._deck = 0
        self._sent_at: float | None = None
        self._sent_speed: float | None = None
        self.deck_error: str | None = None

    @property
    def active(self) -> bool:
        return self.state in ("idle", "following")

    def _project(self, x: float, y: float) -> tuple[int, float, float]:
        # Prefer a still-on-route local arc so a stacked later leg cannot steal the
        # projection (smoke-c: s~2604 under s~2941, 337 m later).
        return project_forward(
            self.route, x, y, self.index, prefer_s=self.progress_m, max_jump_m=MAX_ARC_JUMP_M,
        )

    def commands(self, x: float, y: float, speed_mps: float, z: float | None = None) -> list[object]:
        if not self.active:
            return []
        i, s, d = self._project(x, y)
        self.max_off_route_m = max(self.max_off_route_m, d)
        if d > self.off_route_m:
            self._off += 1
            if self._off >= self.off_ticks:
                self.state = "off_route"
                return span_commands(speed_mps)
            return []
        self._off = 0
        self.index = i
        self.progress_m = max(self.progress_m, s)
        if z is not None:
            err = route_place_z_error((x, y, _interp_plan_z(self.route, i, s)), (x, y, z))
            if err:
                self._deck += 1
                if self._deck >= DECK_TICKS:
                    # Stay active so resume() still works; the collector ends the segment.
                    self.deck_error = err
                    return []
            else:
                self._deck = 0
        if self.progress_m >= self.route.far_end_m - FAR_END_TOL_M:
            self.far_end_reached = True
        if self.progress_m >= self.route.total_m - self.end_tol_m:
            self.state = "done"
            return span_commands(speed_mps)
        v = float(speed_mps)
        cap = KIND_SPEED_CAP.get(self.route.kinds[min(len(self.route.kinds) - 1, i + 1)])
        if cap is not None:
            v = min(v, float(cap))
        first = self._sent_at is None
        faster = self._sent_speed is None or abs(v - self._sent_speed) >= SEND_SPEED_DELTA
        if not first and not faster:
            return []
        if not first:
            # Mid-drive: speed only. A new set_line would teleport to script[1].
            self._sent_speed = v
            return [f"ai.setSpeed({v})"]
        line = self._window(i, s)
        self._sent_at = self.progress_m
        self._sent_speed = v
        self.lines_sent += 1
        self.state = "following"
        # Per-node caps: setScriptedPath clears setSpeed, so wpSpeeds govern.
        # Mid-drive setSpeed may still go out, but the line must carry the caps.
        return [
            'ai.setMode("manual")',
            'ai.driveInLane("off")',
            'ai.setSpeedMode("limit")',
            f"ai.setSpeed({v})",
            set_line_command(line, self._speeds_for_line(line, v), cling=True),
        ]

    def _speeds_for_line(self, line: list[tuple[float, float, float]], v: float) -> list[float]:
        """min(v, KIND_SPEED_CAP[kind]) at each line point (kinds is per point; segment i uses kinds[i+1])."""
        out: list[float] = []
        idx = 0
        nkind = len(self.route.kinds)
        for x, y, _z in line:
            j, _s, _d = project_forward(self.route, x, y, idx, ahead_m=1e9)
            idx = j
            cap = KIND_SPEED_CAP.get(self.route.kinds[min(nkind - 1, j + 1)])
            out.append(min(float(v), float(cap)) if cap is not None else float(v))
        return out

    def _window(self, i: int, s: float) -> list[tuple[float, float, float]]:
        """The car's place on the route, then the remaining route (capped by window_m if finite)."""
        pts = self.route.points
        cum = self.route.cum
        seg = max(1e-6, cum[i + 1] - cum[i])
        t = min(1.0, max(0.0, (s - cum[i]) / seg))
        a, b = pts[i], pts[i + 1]
        out = [(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)]
        j = i + 1
        ahead = self.window_m if math.isfinite(self.window_m) else float("inf")
        while j < len(pts) and cum[j] - s <= ahead:
            if math.hypot(pts[j][0] - out[-1][0], pts[j][1] - out[-1][1]) >= 0.5:
                out.append(pts[j])
            j += 1
        if j < len(pts) and math.hypot(pts[j][0] - out[-1][0], pts[j][1] - out[-1][1]) >= 0.5:
            out.append(pts[j])
        return out

    def note_start(self) -> None:
        """The collector took the route's own start pose."""
        self.spawns += 1
        self.deck_error = None
        self._deck = 0

    def resume(self) -> tuple[SpawnPick | None, str]:
        """A respawn on the route itself, ``RESUME_NUDGE_M`` past the farthest point the car reached.

        Called when a segment ends (a stationary abort that a recover did not fix) while the route is still
        being driven. The deck would pick a road anywhere on the map, kilometres off the route. Returns
        ``(pick, "")`` and moves the driver to the new place (the next tick re-sends the mode and window), or
        ``(None, why)`` when the route cannot be resumed: the driver is done or off the route, MAX_RESUMES is
        used, or too little of the route is left. The caller ends the clip then, rather than record off the plan.
        """
        if not self.active:
            return None, f"route driver is {self.state}"
        if self.resumes >= MAX_RESUMES:
            return None, f"the car was already put back on the route {self.resumes}x and stopped again"
        base = max(self.progress_m, self.route.cum[self.index])
        s = base + RESUME_NUDGE_M
        left = self.route.total_m - s
        if left < RESUME_MIN_LEFT_M:
            return None, f"only {max(0.0, left):.0f} m of the route are left past {base:.0f} m (need {RESUME_MIN_LEFT_M:.0f} m)"
        i, pick = route_resume_pick(self.route, s)
        self.index = i
        self.progress_m = s
        self.skipped_m += s - base
        self.resumes += 1
        self.spawns += 1
        self._off = 0
        self._deck = 0
        self.deck_error = None
        self.reset_send()
        return pick, ""

    def reset_send(self) -> None:
        """Forget what was sent (the AI was reset by a spawn or recover): the next tick sends the window again, mode included."""
        self._sent_at = None
        self._sent_speed = None

    def summary(self) -> dict[str, Any]:
        return {
            "state": self.state, "progress_m": round(self.progress_m, 1), "total_m": round(self.route.total_m, 1),
            "far_end_reached": bool(self.far_end_reached), "lines_sent": int(self.lines_sent),
            "max_off_route_m": round(self.max_off_route_m, 1),
            "resumes": int(self.resumes), "skipped_m": round(self.skipped_m, 1),
        }


def project_forward(
    route: Route, x: float, y: float, start_index: int, *, back: int = 3, ahead_m: float = 400.0,
    prefer_s: float | None = None, max_jump_m: float | None = None,
) -> tuple[int, float, float]:
    """Nearest route segment at or just behind ``start_index``, looking ``ahead_m`` forward.

    Returns ``(segment index, arc length m, distance m)``.

    When ``max_jump_m`` is set, a near-arc candidate (arc no more than that
    ahead of ``prefer_s`` or ``cum[start_index]``, and within ``OFF_ROUTE_M``
    in xy) wins over a stacked later leg that is closer in xy. If there is no
    such local candidate, the uncapped nearest is used, so a real jump ahead
    still projects. Backward search is still the ``back`` segments already in
    the window.
    """
    pts, cum = route.points, route.cum
    lo = max(0, int(start_index) - back)
    base = cum[min(len(pts) - 1, max(0, int(start_index)))]
    origin = base if prefer_s is None else float(prefer_s)

    def _search(limit: float | None) -> tuple[int, float, float]:
        best = (lo, cum[lo], math.inf)
        j = lo
        while j < len(pts) - 1 and cum[j] <= base + ahead_m:
            ax, ay = pts[j][0], pts[j][1]
            bx, by = pts[j + 1][0], pts[j + 1][1]
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            t = 0.0 if seg2 < 1e-9 else min(1.0, max(0.0, ((x - ax) * dx + (y - ay) * dy) / seg2))
            px, py = ax + t * dx, ay + t * dy
            d = math.hypot(x - px, y - py)
            s = cum[j] + t * math.sqrt(seg2)
            if limit is not None and s > origin + limit:
                j += 1
                continue
            if d < best[2]:
                best = (j, s, d)
            j += 1
        return best

    if max_jump_m is not None:
        local = _search(float(max_jump_m))
        # Prefer a still-on-route local projection over a later stacked leg
        # (smoke-c: lower s~2604 under upper s~2941, 0.3-1 m off the line).
        if local[2] <= OFF_ROUTE_M:
            return local
    return _search(None)


# --------------------------------------------------------------------------- collector hooks


DRIVER_FILE = "route_driver.json"
ROUTE_AI_KEY = ("route-plan",)  # the collector's last_ai while the driver owns the AI


def load_route_plan(path: str | Path, map_names: Sequence[str]) -> Route:
    """Read the route the clip runner wrote. It must be for the one map this collector run loads."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        route = Route.from_json(data)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"route plan {path} unreadable: {type(exc).__name__}: {exc}") from exc
    maps = [str(m) for m in map_names if str(m)]
    if maps != [route.map]:
        raise ValueError(f"route plan is for map {route.map}; this run loads {maps or 'no map'} (--maps must be exactly that map)")
    if len(route.points) < 2 or not (0 < route.far_end_index < len(route.points)):
        raise ValueError(f"route plan {path} has no usable route")
    return route


def route_spawn_pick(route: Route, speed: tuple[float, float] | None = None) -> SpawnPick:
    """The collector's first spawn: the highway end the route starts at, facing along it."""
    pos, rot = route.start_pose()
    return SpawnPick(kind=HIGHWAY, road_id=str(route.start_road), pos=pos, rot=rot,
                     speed=tuple(speed) if speed is not None else SPEED[HIGHWAY])


def route_resume_pick(route: Route, s: float, speed: tuple[float, float] | None = None) -> tuple[int, SpawnPick]:
    """A spawn pose ``s`` metres along the route, facing along it. Returns ``(segment index, pick)``.

    The points are the legal right-lane line the AI is sent, so the pose is on the road and on the route.
    Its kind comes from the route's own kinds (``dirt`` takes the city speed range).
    """
    cum = route.cum
    s = min(max(0.0, float(s)), cum[-1])
    i = max(0, min(len(route.points) - 2, next((k for k in range(len(cum) - 1) if cum[k + 1] >= s), len(cum) - 2)))
    a, b = route.points[i], route.points[i + 1]
    t = min(1.0, max(0.0, (s - cum[i]) / max(1e-6, cum[i + 1] - cum[i])))
    x, y, z = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t
    look = next((p for p in route.points[i + 1:] if math.hypot(p[0] - x, p[1] - y) >= 10.0), route.points[-1])
    kind = route.kinds[min(len(route.kinds) - 1, i + 1)]
    kind = kind if kind in SPEED else CITY
    pick = SpawnPick(kind=kind, road_id=f"route@{s:.0f}", pos=(x, y, z + 0.5), rot=_yaw_quat(look[0] - x, look[1] - y),
                     speed=tuple(speed) if speed is not None else SPEED[kind])
    return i, pick


def drive_tick(driver: RouteDriver, pos: Sequence[float] | None, speed_mps: float, queue: Callable[[str], None]) -> bool:
    """One collector tick of route driving. True when the route driver owns the AI this tick.

    ``queue`` is the collector's ``_queue(vehicle, command)``. The tick that finishes or abandons
    the route sends the span commands itself and returns True; after that the collector's own AI
    logic runs as before.
    """
    if not driver.active:
        return False
    if pos is None or len(pos) < 2:
        return True
    z: float | None = None
    if len(pos) >= 3:
        try:
            z_val = float(pos[2])
        except (TypeError, ValueError):
            z_val = None
        else:
            if math.isfinite(z_val):
                z = z_val
    for cmd in driver.commands(float(pos[0]), float(pos[1]), float(speed_mps), z=z):
        queue(cmd)
    return True


def write_driver_summary(out: str | Path, driver: RouteDriver) -> Path:
    """``route_driver.json`` in the clip folder (replaced each save; the run clip_tags.json embeds the last one)."""
    path = Path(out) / DRIVER_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(driver.summary()), encoding="utf-8")
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------- measure


def measure_route(route: Route, samples: Iterable[tuple[float, float]]) -> dict[str, Any]:
    """How much of the planned route the recorded ego positions actually covered.

    ``samples`` are x, y from the recording (GT pose), in time order. The car is
    tracked forward along the route; a sample farther than ``OFF_ROUTE_M`` from it
    is off the route and earns no progress.
    """
    idx = 0
    progress = 0.0
    n = 0
    off = 0
    max_d = 0.0
    sum_d = 0.0
    dirt_reach = 0.0
    last_good = None
    for x, y in samples:
        n += 1
        i, s, d = project_forward(route, float(x), float(y), idx)
        max_d = max(max_d, d) if d < math.inf else max_d
        if d > OFF_ROUTE_M:
            off += 1
            continue
        sum_d += d
        idx = i
        if s > progress:
            progress = s
        last_good = (i, s)
    hw = max(route.highway_m, 1e-9)
    ret = max(route.return_m, 1e-9)
    far_reached = progress >= route.far_end_m - FAR_END_TOL_M
    done = progress >= route.total_m - END_TOL_M
    # Unpaved length actually passed: the unpaved segments up to the progress reached.
    reached = 0.0
    for k in range(len(route.points) - 1):
        if route.cum[k + 1] <= progress + 1e-6 and route.kinds[k + 1] == "dirt":
            reached += route.cum[k + 1] - route.cum[k]
    dirt_reach = reached
    good = n - off
    return {
        "planned_m": round(route.total_m, 1),
        "highway_m": round(route.highway_m, 1),
        "return_m": round(route.return_m, 1),
        "driven_m": round(progress, 1),
        "highway_fraction": round(min(1.0, progress / hw), 4),
        "return_fraction": round(min(1.0, max(0.0, progress - route.far_end_m) / ret), 4),
        "far_end_reached": bool(far_reached),
        "loop_closed": bool(done),
        "dirt_driven_m": round(dirt_reach, 1),
        "samples": int(n),
        "off_route_samples": int(off),
        "max_off_route_m": round(max_d, 1),
        "mean_off_route_m": round(sum_d / good, 2) if good else None,
    }
