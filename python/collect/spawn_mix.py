"""Stratified west-coast spawns from a dumped navgraph.

Highway is a wide or one-way multi-lane road. A ramp is a short one-way
road that touches both a highway and a local road. Everything else that
is drivable is a city street. The mix is about 40% city, 35% highway,
25% ramps, and a road is used once per run.

With a painted-lane map, ``SpawnDeck`` can also skip a road whose next
300 m carries little line paint (``min_paint_cov``). The labeler reads the
paint, so a spawn on unpainted road records minutes of masked rows.

The AI does not stay on those 300 m for a whole segment, so
``coverage_from`` scores the same probe from wherever the car is now (the
collector's ``--recheck-cov-min``). The deck also counts picks and recorded
time per spawn road (``road_repeats``): a resumed run reshuffles with the
same seed, so ``skip_recorded`` marks the roads already in collect.json as
used and the resumed deck goes on to new ones.

``road_type`` refines city/highway/ramp with ``bridge`` (within 25 m of where
the road passes 4 m or more over another drivable navgraph road) and
``tunnel`` (4 m or more under one: an underpass or a tunnel under another
road; a tunnel under bare terrain is not in the navgraph and stays its pool
kind). ``type_summary`` counts picks by that type, deck spawns and recheck
respawns apart. The type is only logged; no pick is filtered on it.

``take`` returning None sets ``exhausted``: every road is used. The
collector then keeps the current spawn. ``traffic_slots`` places the
companion cars along the road (``TRAFFIC_AHEAD_M`` ahead, ``TRAFFIC_BEHIND_M``
behind), not on the straight spawn tangent.
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
# The walk ahead stops at a dead end or where no road end joins (a ramp that
# merges into the middle of a highway). A walk shorter than this counts its
# missing probes as unpainted, so a 9 m stub with one lucky probe is not 1.00.
PAINT_MIN_PROBE_M = 100.0
# Recheck from the car's pose: the nearest navgraph road within this distance,
# preferring one that runs along the heading (under 60 deg) at the car's height.
RECHECK_NEAR_M = 12.0
RECHECK_CELL_M = 25.0
# A spawn road picked this many times is flagged in road_repeats.
REPEAT_FLAG_PICKS = 3
# Bridge / tunnel: a drivable road crossing this far above or below, within
# this distance of the pose.
BRIDGE = "bridge"
TUNNEL = "tunnel"
ROAD_TYPES = (CITY, HIGHWAY, RAMP, BRIDGE, TUNNEL)
DECK_DZ_M = 4.0
DECK_NEAR_M = 25.0
DECK_CELL_M = 50.0
# Pick sources counted apart in type_summary.
SOURCES = ("spawn", "recheck")
# Companion cars, measured along the road from the ego spawn. A slot the road
# walk cannot reach falls back to the farthest point it reached; under
# TRAFFIC_MIN_M there is no valid spot and the car is not spawned.
TRAFFIC_AHEAD_M = 60.0
TRAFFIC_BEHIND_M = 40.0
TRAFFIC_MIN_M = 8.0


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
    return _walk_ahead(chain, {str(road["id"])}, roads_by_id, ends, ahead_m)


def _walk_ahead(
    chain: list[tuple[float, float, float]],
    seen: set[str],
    roads_by_id: dict[str, dict[str, Any]],
    ends: dict[tuple[int, int], list[tuple[str, bool]]],
    ahead_m: float = PAINT_AHEAD_M,
    reverse: bool = False,
) -> list[tuple[float, float, float]]:
    """Continue ``chain`` into joining roads (the ``road_ahead`` rules) up to ``ahead_m``.

    ``reverse`` walks against the direction of travel (where a car behind
    comes from): a one-way road is then only entered at its end.
    """
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
                    if other is None or (other["oneway"] and (at_start if reverse else not at_start)):
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


def chain_from_pose(
    road: dict[str, Any],
    pos: tuple[float, float, float] | list[float],
    heading: tuple[float, float],
    roads_by_id: dict[str, dict[str, Any]],
    ends: dict[tuple[int, int], list[tuple[str, bool]]],
    ahead_m: float = PAINT_AHEAD_M,
    reverse: bool = False,
) -> list[tuple[float, float, float]]:
    """Centerline from the point of ``road`` nearest ``pos``, in the direction of travel.

    ``reverse`` is the walk behind a car: pass the reversed heading.

    The road is walked the way ``heading`` points along it (a two-way road
    either way), then continued like ``road_ahead``.
    """
    pts = [tuple(float(v) for v in p[:3]) for p in road["pts"]]
    if len(pts) < 2:
        return list(pts)
    qx, qy = float(pos[0]), float(pos[1])
    best: tuple[float, int, float] | None = None
    for k, (a, b) in enumerate(zip(pts, pts[1:])):
        dx, dy = b[0] - a[0], b[1] - a[1]
        l2 = dx * dx + dy * dy
        t = 0.0 if l2 < 1e-12 else min(1.0, max(0.0, ((qx - a[0]) * dx + (qy - a[1]) * dy) / l2))
        d = math.hypot(a[0] + dx * t - qx, a[1] + dy * t - qy)
        if best is None or d < best[0]:
            best = (d, k, t)
    assert best is not None
    _d, k, t = best
    a, b = pts[k], pts[k + 1]
    here = tuple(a[i] + (b[i] - a[i]) * t for i in range(3))
    forward = (b[0] - a[0]) * float(heading[0]) + (b[1] - a[1]) * float(heading[1]) >= 0.0
    rest = pts[k + 1 :] if forward else pts[k::-1]
    # Drop a vertex the car is standing on, so the first step has a heading.
    chain = [here] + [p for p in rest if math.hypot(p[0] - here[0], p[1] - here[1]) > 1e-3]
    if len(chain) < 2:
        return chain
    return _walk_ahead(chain, {str(road["id"])}, roads_by_id, ends, ahead_m, reverse)


def point_along(
    chain: list[tuple[float, float, float]], dist_m: float
) -> tuple[tuple[float, float, float], tuple[float, float], float] | None:
    """``(point, unit tangent, reached_m)`` at ``dist_m`` along ``chain``, or its far end if shorter."""
    last: tuple[tuple[float, float, float], tuple[float, float], float] | None = None
    walked = 0.0
    for a, b in zip(chain, chain[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg < 1e-6:
            continue
        tangent = ((b[0] - a[0]) / seg, (b[1] - a[1]) / seg)
        if walked + seg >= float(dist_m):
            mix = (float(dist_m) - walked) / seg
            return tuple(a[k] + (b[k] - a[k]) * mix for k in range(3)), tangent, float(dist_m)
        walked += seg
        last = (tuple(b), tangent, walked)
    return last


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
    min_samples: int = 0,
) -> float:
    """Share of ``samples`` where ``painted.lane_at`` finds the ego lane at ``probe_y_m``.

    ``painted`` only needs ``lane_at(pos, direction, ys)``. No samples is 0.
    Fewer than ``min_samples`` samples divide by ``min_samples``: the missing
    ones count as unpainted.
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
    return good / float(max(len(samples), int(min_samples)))


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
        min_probe_m: float = PAINT_MIN_PROBE_M,
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
        self.min_samples = max(0, int(round(float(min_probe_m or 0.0) / PAINT_STEP_M)))
        self.skipped = 0
        # Picks of a kind other than the one the mix asked for, and picks of
        # the best-covered skipped road because nothing passed. One pick can
        # be both.
        self.cross_kind = 0
        self.best_seen = 0
        self.last_ms = 0.0
        self.select_ms: list[float] = []
        # Spawn picks and recorded seconds per road (road_repeats).
        self.road_picks: dict[str, int] = {}
        self.road_time_s: dict[str, float] = {}
        self.road_kind: dict[str, str] = {}
        self._seg_grid: dict[tuple[int, int], list[tuple[str, int]]] | None = None
        self._kind_of: dict[str, str] = {}
        self._decks: dict[str, list[tuple[float, float, bool]]] | None = None
        self.type_counts: dict[str, dict[str, int]] = {src: {t: 0 for t in ROAD_TYPES} for src in SOURCES}
        self.last_type: str | None = None
        self.resume_skipped = 0
        self.exhausted = False
        self._by_id: dict[str, dict[str, Any]] = {}
        self._ends: dict[tuple[int, int], list[tuple[str, bool]]] = defaultdict(list)
        for pool_kind, pool in self.pools.items():
            for road in pool:
                self._by_id[str(road["id"])] = road
                self._kind_of[str(road["id"])] = pool_kind
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
        """Paint coverage over the next ``PAINT_AHEAD_M`` from this road's spawn point.

        A walk under ``min_probe_m`` scores its missing probes as unpainted.
        """
        chain = road_ahead(road, self._by_id, self._ends, PAINT_AHEAD_M)
        return paint_coverage(
            self.painted, sample_along(chain), oneway=bool(road["oneway"]), min_samples=self.min_samples
        )

    def _say(self, text: str) -> None:
        if self.log is not None:
            self.log(text)

    def _pick(self, kind: str, road: dict[str, Any], pose: Any, cov: float | None, wanted: str) -> SpawnPick:
        self.counts[kind] += 1
        rid = str(road["id"])
        self.road_picks[rid] = self.road_picks.get(rid, 0) + 1
        self.road_kind[rid] = kind
        if kind != wanted:
            self.cross_kind += 1
        pos, rot = pose
        return SpawnPick(kind, road["id"], pos, rot, SPEED[kind], cov)

    def take(self, source: str = "spawn") -> SpawnPick | None:
        """Next spawn, or None. ``last_ms`` is this call's wall time.

        ``source`` ("spawn" or "recheck") is the type_summary column the pick
        counts in; ``last_type`` is its road type at the spawn pose.
        """
        start = time.perf_counter()
        try:
            pick = self._take()
            self.last_type = None
            if pick is None:
                self.exhausted = True
            else:
                self.last_type = self.road_type(pick.road_id, pick.pos)
                column = self.type_counts.setdefault(str(source), {t: 0 for t in ROAD_TYPES})
                column[self.last_type] = column.get(self.last_type, 0) + 1
            return pick
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
            "exhausted": self.exhausted,
            "used": len(self.used),
            "roads": sum(len(pool) for pool in self.pools.values()),
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

    # ------------------------------------------------------- road repeats
    def note_time(self, road_id: str | None, dt_s: float) -> None:
        """Credit ``dt_s`` recorded seconds to the spawn road the car came from."""
        if not road_id:
            return
        rid = str(road_id)
        self.road_time_s[rid] = self.road_time_s.get(rid, 0.0) + max(0.0, float(dt_s or 0.0))

    def seed_repeats(self, prior: Any) -> int:
        """Add picks and seconds from an earlier ``road_repeats`` (a resumed run). Returns roads read."""
        rows = (prior or {}).get("by_road") if isinstance(prior, dict) else None
        if not isinstance(rows, dict):
            return 0
        n = 0
        for rid, item in rows.items():
            if not isinstance(item, dict):
                continue
            rid = str(rid)
            self.road_picks[rid] = self.road_picks.get(rid, 0) + int(_num(item.get("picks")))
            self.road_time_s[rid] = self.road_time_s.get(rid, 0.0) + _num(item.get("time_s"))
            if item.get("kind"):
                self.road_kind.setdefault(rid, str(item["kind"]))
            n += 1
        return n

    def road_repeats(self, flag_picks: int = REPEAT_FLAG_PICKS) -> dict[str, Any]:
        """Picks and share of recorded time per spawn road; roads picked ``flag_picks``+ times are flagged."""
        total_t = sum(self.road_time_s.values())
        ids = set(self.road_picks) | set(self.road_time_s)
        by_road: dict[str, Any] = {}
        for rid in sorted(ids, key=lambda r: (-self.road_picks.get(r, 0), -self.road_time_s.get(r, 0.0), r)):
            t = self.road_time_s.get(rid, 0.0)
            by_road[rid] = {
                "kind": self.road_kind.get(rid),
                "picks": int(self.road_picks.get(rid, 0)),
                "time_s": round(t, 1),
                "time_share": round(t / total_t, 4) if total_t > 0 else 0.0,
            }
        flagged = [rid for rid, item in by_road.items() if item["picks"] >= int(flag_picks)]
        flagged_t = sum(self.road_time_s.get(rid, 0.0) for rid in flagged)
        return {
            "flag_picks": int(flag_picks),
            "roads": len(by_road),
            "picks": int(sum(self.road_picks.values())),
            "max_picks": max((item["picks"] for item in by_road.values()), default=0),
            "time_s": round(total_t, 1),
            "flagged": flagged,
            "flagged_time_share": round(flagged_t / total_t, 4) if total_t > 0 else 0.0,
            "by_road": by_road,
        }

    def repeat_lines(self) -> list[str]:
        """End-of-run log: one summary line, then one line per flagged road."""
        r = self.road_repeats()
        out = [
            f"spawn repeats roads={r['roads']} picks={r['picks']} max_picks={r['max_picks']} "
            f"flagged={len(r['flagged'])} (>= {r['flag_picks']} picks) "
            f"flagged_time={r['flagged_time_share']:.0%} of {r['time_s']:.0f}s"
        ]
        for rid in r["flagged"]:
            item = r["by_road"][rid]
            out.append(
                f"spawn repeat flagged road={rid} kind={item['kind'] or '-'} picks={item['picks']} "
                f"time={item['time_s']:.0f}s share={item['time_share']:.0%}"
            )
        return out

    def skip_recorded(self, prior: Any) -> int:
        """Mark the roads an earlier ``road_repeats`` picked as used. Returns how many were new to ``used``.

        A resumed run builds a fresh deck with the same seed, which would
        start over on the same roads.
        """
        rows = (prior or {}).get("by_road") if isinstance(prior, dict) else None
        if not isinstance(rows, dict):
            return 0
        n = 0
        for rid, item in rows.items():
            rid = str(rid)
            if isinstance(item, dict) and int(_num(item.get("picks"))) > 0 and rid in self._by_id and rid not in self.used:
                self.used.add(rid)
                n += 1
        self.resume_skipped += n
        return n

    # --------------------------------------------------------- road types
    def _deck_crossings(self) -> dict[str, list[tuple[float, float, bool]]]:
        """Per road, where it passes over (True) or under (False) another drivable road by DECK_DZ_M+."""
        if self._decks is not None:
            return self._decks
        segs: list[tuple[str, Any, Any]] = []
        grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        for rid, road in self._by_id.items():
            pts = road["pts"]
            for a, b in zip(pts, pts[1:]):
                j = len(segs)
                segs.append((rid, a, b))
                for cx in range(int(min(a[0], b[0]) // DECK_CELL_M), int(max(a[0], b[0]) // DECK_CELL_M) + 1):
                    for cy in range(int(min(a[1], b[1]) // DECK_CELL_M), int(max(a[1], b[1]) // DECK_CELL_M) + 1):
                        grid[(cx, cy)].append(j)
        found: dict[str, list[tuple[float, float, bool]]] = defaultdict(list)
        seen: set[tuple[int, int]] = set()
        for js in grid.values():
            for x in range(len(js)):
                for y in range(x + 1, len(js)):
                    pair = (js[x], js[y])
                    if pair in seen:
                        continue
                    seen.add(pair)
                    r1, a1, b1 = segs[pair[0]]
                    r2, a2, b2 = segs[pair[1]]
                    if r1 == r2:
                        continue
                    dx1, dy1 = b1[0] - a1[0], b1[1] - a1[1]
                    dx2, dy2 = b2[0] - a2[0], b2[1] - a2[1]
                    den = dx1 * dy2 - dy1 * dx2
                    if abs(den) < 1e-9:
                        continue
                    t = ((a2[0] - a1[0]) * dy2 - (a2[1] - a1[1]) * dx2) / den
                    u = ((a2[0] - a1[0]) * dy1 - (a2[1] - a1[1]) * dx1) / den
                    if not (0.0 <= t <= 1.0 and 0.0 <= u <= 1.0):
                        continue
                    z1 = a1[2] + (b1[2] - a1[2]) * t
                    z2 = a2[2] + (b2[2] - a2[2]) * u
                    if abs(z1 - z2) < DECK_DZ_M:
                        continue
                    px, py = a1[0] + dx1 * t, a1[1] + dy1 * t
                    found[r1].append((px, py, z1 > z2))
                    found[r2].append((px, py, z2 > z1))
        self._decks = dict(found)
        return self._decks

    def road_type(self, road_id: str | None, pos: Any = None) -> str:
        """city/highway/ramp, or bridge/tunnel within DECK_NEAR_M of a crossing on this road."""
        rid = str(road_id) if road_id is not None else ""
        kind = self._kind_of.get(rid, CITY)
        if pos is None or not rid:
            return kind
        best: tuple[float, bool] | None = None
        for px, py, over in self._deck_crossings().get(rid, ()):
            d = math.hypot(px - float(pos[0]), py - float(pos[1]))
            if d <= DECK_NEAR_M and (best is None or d < best[0]):
                best = (d, over)
        if best is None:
            return kind
        return BRIDGE if best[1] else TUNNEL

    def type_summary(self) -> dict[str, Any]:
        """Picks by road type, deck spawns and recheck respawns apart, for collect.json."""
        out: dict[str, Any] = {src: dict(col) for src, col in self.type_counts.items()}
        out["total"] = {t: sum(col.get(t, 0) for col in self.type_counts.values()) for t in ROAD_TYPES}
        out["resume_skipped_roads"] = self.resume_skipped
        return out

    def type_lines(self) -> list[str]:
        """End-of-run spawn mix by road type: one line per source."""
        lines = []
        for src in list(SOURCES) + [s for s in self.type_counts if s not in SOURCES]:
            col = self.type_counts.get(src, {})
            n = sum(col.values())
            mix = " ".join(f"{t}={col.get(t, 0)}" for t in ROAD_TYPES)
            lines.append(f"spawn types {src} n={n} {mix}")
        return lines

    # ------------------------------------------------------------ traffic
    def traffic_slots(
        self,
        pick: SpawnPick,
        count: int,
        ahead_m: float = TRAFFIC_AHEAD_M,
        behind_m: float = TRAFFIC_BEHIND_M,
    ) -> tuple[list[tuple[str, tuple[float, float, float], tuple[float, float, float, float], float]], list[str]]:
        """Companion-car poses along the road: slot 0 ``ahead_m`` ahead, slot 1 ``behind_m`` behind.

        Returns ``([(where, pos, rot, along_m)], notes)``. A slot the walk
        cannot reach uses the farthest point reached (a note says so); under
        TRAFFIC_MIN_M, or with no road, the slot is dropped (also noted).
        Both cars face the direction of travel.
        """
        road = self._by_id.get(str(pick.road_id))
        _qx, _qy, qz, qw = pick.rot
        yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
        hx, hy = math.cos(yaw), math.sin(yaw)
        wants = [("ahead", float(ahead_m), False), ("behind", float(behind_m), True)][: max(0, int(count))]
        slots: list[tuple[str, tuple[float, float, float], tuple[float, float, float, float], float]] = []
        notes: list[str] = []
        for where, want, back in wants:
            if road is None:
                notes.append(f"traffic {where} skipped: spawn road {pick.road_id} not in the deck")
                continue
            heading = (-hx, -hy) if back else (hx, hy)
            chain = chain_from_pose(road, pick.pos, heading, self._by_id, self._ends, want + 1.0, reverse=back)
            got = point_along(chain, want)
            if got is None or got[2] < TRAFFIC_MIN_M:
                reached = 0.0 if got is None else got[2]
                notes.append(f"traffic {where} skipped: road walk reaches {reached:.1f} m < {TRAFFIC_MIN_M:.0f} m (wanted {want:.0f} m)")
                continue
            (x, y, z), (tx, ty), along = got
            if along < want - 1e-6:
                notes.append(f"traffic {where} fallback: farthest valid spot {along:.1f} m along the road (wanted {want:.0f} m)")
            fx, fy = (-tx, -ty) if back else (tx, ty)
            slots.append((where, (x, y, z + 0.5), _yaw_quat(fx, fy), along))
        return slots, notes

    # ------------------------------------------------------------ recheck
    def nearest_road(
        self, pos: Any, heading: tuple[float, float] | None = None, near_m: float = RECHECK_NEAR_M
    ) -> tuple[dict[str, Any], float] | None:
        """Navgraph road nearest ``pos`` (x, y, z), or None past ``near_m``.

        A road more than 4 m above or below, or crossing the heading by more
        than 60 deg (the cross street at a junction), is ranked as farther.
        """
        if self._seg_grid is None:
            grid: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
            for rid, road in self._by_id.items():
                pts = road["pts"]
                for k, (a, b) in enumerate(zip(pts, pts[1:])):
                    x0, x1 = sorted((a[0], b[0]))
                    y0, y1 = sorted((a[1], b[1]))
                    for cx in range(int(x0 // RECHECK_CELL_M), int(x1 // RECHECK_CELL_M) + 1):
                        for cy in range(int(y0 // RECHECK_CELL_M), int(y1 // RECHECK_CELL_M) + 1):
                            grid[(cx, cy)].append((rid, k))
            self._seg_grid = dict(grid)
        qx, qy = float(pos[0]), float(pos[1])
        qz = float(pos[2]) if len(pos) > 2 and pos[2] is not None else None
        hx = hy = 0.0
        if heading is not None:
            n = math.hypot(float(heading[0]), float(heading[1]))
            if n > 1e-6:
                hx, hy = float(heading[0]) / n, float(heading[1]) / n
        cx, cy = int(qx // RECHECK_CELL_M), int(qy // RECHECK_CELL_M)
        best: tuple[float, float, str] | None = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for rid, k in self._seg_grid.get((cx + dx, cy + dy), ()):
                    a, b = self._by_id[rid]["pts"][k], self._by_id[rid]["pts"][k + 1]
                    ux, uy = b[0] - a[0], b[1] - a[1]
                    l2 = ux * ux + uy * uy
                    t = 0.0 if l2 < 1e-12 else min(1.0, max(0.0, ((qx - a[0]) * ux + (qy - a[1]) * uy) / l2))
                    d = math.hypot(a[0] + ux * t - qx, a[1] + uy * t - qy)
                    score = d
                    if qz is not None and abs(a[2] + (b[2] - a[2]) * t - qz) > 4.0:
                        score += 6.0
                    if (hx or hy) and l2 > 1e-12 and abs(ux * hx + uy * hy) / math.sqrt(l2) < PAINT_JOIN_COS:
                        score += 6.0
                    if best is None or score < best[0]:
                        best = (score, d, rid)
        if best is None or best[1] > float(near_m):
            return None
        return self._by_id[best[2]], best[1]

    def coverage_from(self, pos: Any, direction: Any) -> tuple[float | None, str | None]:
        """The spawn coverage probe, from the car's pose: ``(cov, road_id)``.

        ``(None, None)`` when filtering is off, the pose is missing, or no
        road is within ``RECHECK_NEAR_M`` (a lot or a field: not scored).
        """
        if not self.filtering() or pos is None or direction is None:
            return None, None
        try:
            fx, fy = float(direction[0]), float(direction[1])
        except (TypeError, ValueError, IndexError):
            return None, None
        if math.hypot(fx, fy) < 1e-6:
            return None, None
        found = self.nearest_road(pos, (fx, fy))
        if found is None:
            return None, None
        road, _dist = found
        chain = chain_from_pose(road, pos, (fx, fy), self._by_id, self._ends, PAINT_AHEAD_M)
        cov = paint_coverage(
            self.painted, sample_along(chain), oneway=bool(road["oneway"]), min_samples=self.min_samples
        )
        return cov, str(road["id"])

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


def traffic_lua(pick: SpawnPick, count: int, slots: list[Any] | None = None) -> list[str]:
    """GE Lua that replaces the companion cars ahead of and behind the ego.

    ``slots`` (from ``SpawnDeck.traffic_slots``) gives each car's pose along
    the road; without it the cars go 28 m ahead and 22 m behind on the
    spawn tangent. The player vehicle is not deleted and is not switched into
    traffic mode.
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
    poses = []
    if slots is not None:
        poses = [(tuple(item[1]), tuple(item[2])) for item in slots][: int(count)]
    else:
        for slot in range(min(int(count), len(offsets))):
            dist = offsets[slot]
            poses.append(((x0 + dx * dist, y0 + dy * dist, z0), tuple(pick.rot)))
    for slot, ((x, y, z), (qx, qy, qz, qw)) in enumerate(poses):
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

