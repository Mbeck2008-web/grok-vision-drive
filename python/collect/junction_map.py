"""Junction map per level, and per-row junction labels (``jct.jsonl``).

Offline: ``build_junction_map`` turns a road dump (``roads/<map>.json.gz``) into
``junctions/<map>.json.gz``. Live and offline: ``JunctionLabeler.step(row)``
labels one state row, and ``JctSidecar`` writes those labels create-only. The
recorder and ``scripts/label_junctions.py`` share this one code path, so the
live and the relabelled ``jct.jsonl`` are the same bytes.

Labels are for grading and training data only. Nothing that steers, brakes or
plans imports this module.

Graph rules (moved here from ``scripts/junction_grade.py``, which imports them):
road ends within ``MERGE_M`` share a node, and a road end that lands on the
middle of another road splits that road there. With ``dz_max`` set, both only
happen between points whose heights are within ``dz_max`` (decks).

Junction rules:

- filter: keep roads with ``drivability > 0``; drop ``hiddenInNavi``; keep
  ``gatedRoad`` (driveways, parking connectors) and tag those arms minor;
- a node with 3+ arms counts; such nodes within ``CLUSTER_M`` in plan and
  ``JOIN_DZ_M`` in height are one junction; centre = mean of the nodes;
  ``r_m`` = largest node offset from the centre + largest arm half-width;
- an arm is minor when gated or ``drivability < MINOR_DRIVABILITY``; a junction
  is minor when fewer than 3 arms are not minor;
- type: ``T`` (3 arms, an opposite pair 180 +/- ``THROUGH_DEG``), ``Y`` (3, none),
  ``4way``, ``multi`` (5+); ``merge`` overrides them when two one-way arms leave
  less than ``MERGE_DEG`` apart.

``jct_id`` is stable only for one ``src_sha256`` + ``params``.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from python.debug.gt_geom import JOIN_DZ_M, parse_road_edges

VERSION = 1
LEAD_M = 25.0  # how far along a road its direction at a node is measured
MERGE_M = 4.0  # road ends this close together are one node
CLUSTER_M = 20.0  # junction nodes this close (and within JOIN_DZ_M) are one junction
THROUGH_DEG = 25.0  # an opposite pair is 180 +/- this apart
MERGE_DEG = 30.0  # two one-way arms closer than this make a merge
MINOR_DRIVABILITY = 0.5
FILTER = "keep drivability>0; drop hiddenInNavi=1; keep gatedRoad=1 as minor"
# Labeler
GRID_M = 50.0
HEADING_MAX_DEG = 60.0
HYST_M = 1.0
JUMP_M = 30.0
CAP_M = 150.0
FLUSH_ROWS = 10
# Navgraph check (dev PC, metric 1(a)); frozen with the golden file.
ARM_POINT_M = 10.0
ARM_MAP_M = 3.0
MATCH_M = 25.0
UNMAPPED_MAX = 0.05

PARAMS = {
    "CLUSTER_M": CLUSTER_M,
    "JOIN_DZ_M": JOIN_DZ_M,
    "MERGE_M": MERGE_M,
    "LEAD_M": LEAD_M,
    "THROUGH_DEG": THROUGH_DEG,
    "MERGE_DEG": MERGE_DEG,
    "MINOR_DRIVABILITY": MINOR_DRIVABILITY,
    "filter": FILTER,
}

LABEL_KEYS = ("t", "in_jct", "jct_id", "next_jct_m", "jct_type", "minor", "src")


def wrap(rad: float) -> float:
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Edge:
    a: str
    b: str
    xy: list[tuple[float, float]]
    oneway: bool = False
    name: str = ""
    road_id: str = ""

    @property
    def length(self) -> float:
        return sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(self.xy, self.xy[1:]))


@dataclass
class Graph:
    nodes: dict[str, tuple[float, float]] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    node_z: dict[str, float | None] = field(default_factory=dict)
    road_width: dict[str, float] = field(default_factory=dict)

    def incident(self, node: str) -> list[Edge]:
        return [e for e in self.edges if e.a == node or e.b == node]

    def junctions(self) -> list[str]:
        return [n for n in self.nodes if len(self.incident(n)) >= 3]


def _point_along(xy: list[tuple[float, float]], dist: float) -> tuple[float, float]:
    left = dist
    for p, q in zip(xy, xy[1:]):
        seg = math.hypot(q[0] - p[0], q[1] - p[1])
        if seg >= left and seg > 0.0:
            f = left / seg
            return (p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f)
        left -= seg
    return xy[-1]


def leave_angle(edge: Edge, node: str) -> float:
    """Direction a car leaving ``node`` along ``edge`` drives in."""
    xy = edge.xy if edge.a == node else list(reversed(edge.xy))
    far = _point_along(xy, min(LEAD_M, edge.length))
    return math.atan2(far[1] - xy[0][1], far[0] - xy[0][0])


# ---------------------------------------------------------------- graph


def graph_from_fixture(data: dict[str, Any], level: str | None = None) -> Graph:
    """The hand-built ``junctions_synthetic.json`` format."""
    levels = data.get("levels") or {}
    if level is None:
        level = next(iter(levels))
    lvl = levels[level]
    graph = Graph({k: (float(v[0]), float(v[1])) for k, v in lvl["nodes"].items()})
    for e in lvl["edges"]:
        graph.edges.append(
            Edge(e["a"], e["b"], [graph.nodes[e["a"]], graph.nodes[e["b"]]], bool(e.get("oneway")), f"{e['a']}-{e['b']}")
        )
    return graph


def _drivable(road: Any) -> bool:
    if not isinstance(road, dict):
        return True
    try:
        return float(road.get("drivability", road.get("drivable", 1))) > 0.0
    except (TypeError, ValueError):
        return True


def _cell(p: tuple[float, float], size: float) -> tuple[int, int]:
    return (math.floor(p[0] / size), math.floor(p[1] / size))


def _dz_ok(za: float | None, zb: float | None, dz_max: float | None) -> bool:
    return dz_max is None or za is None or zb is None or abs(za - zb) <= dz_max


def graph_from_network(spec: Any, dz_max: float | None = None) -> Graph:
    """A node/edge graph from a raw road network (an offline road dump).

    Road ends within ``MERGE_M`` share a node (the first node made within
    ``MERGE_M``). A road end that lands on the middle of another road splits
    that road there, so a side street that ends on a through road makes a T.
    With ``dz_max``, a merge or split also needs the two heights within
    ``dz_max`` (skipped where a height is unknown). Grids only speed up the
    search; the result is the same as checking every pair.
    """
    if isinstance(spec, dict) and isinstance(spec.get("roads"), dict):
        spec = spec["roads"]
    lines: list[tuple[str, list[tuple[float, float]], bool, list[float | None]]] = []
    graph = Graph()
    for road_id, road in (spec or {}).items() if isinstance(spec, dict) else []:
        if not _drivable(road):
            continue
        edges = road.get("edges") or road.get("nodes") if isinstance(road, dict) else road
        poly = parse_road_edges(str(road_id), edges)
        if poly is None or len(poly.xy) < 2:
            continue
        oneway = _flag(road, "oneWay") or _flag(road, "oneway")  # "0" is two-way
        zs = list(poly.z) if poly.z is not None and dz_max is not None else [None] * len(poly.xy)
        lines.append((str(road_id), [(float(x), float(y)) for x, y in poly.xy], oneway, zs))
        graph.road_width[str(road_id)] = float(poly.width_m)

    node_grid: dict[tuple[int, int], list[int]] = {}
    node_list: list[str] = []

    def node_for(p: tuple[float, float], z: float | None) -> str:
        cx, cy = _cell(p, MERGE_M)
        best = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for idx in node_grid.get((cx + dx, cy + dy), ()):
                    if best is not None and idx >= best:
                        continue
                    name = node_list[idx]
                    q = graph.nodes[name]
                    if math.hypot(p[0] - q[0], p[1] - q[1]) <= MERGE_M and _dz_ok(z, graph.node_z.get(name), dz_max):
                        best = idx
        if best is not None:
            return node_list[best]
        name = f"n{len(graph.nodes)}"
        graph.nodes[name] = p
        graph.node_z[name] = z
        node_grid.setdefault((cx, cy), []).append(len(node_list))
        node_list.append(name)
        return name

    ends: list[tuple[tuple[float, float], float | None]] = [
        (p, z) for _rid, xy, _ow, zs in lines for p, z in ((xy[0], zs[0]), (xy[-1], zs[-1]))
    ]
    for p, z in ends:
        node_for(p, z)
    end_grid: dict[tuple[int, int], list[int]] = {}
    for i, (p, _z) in enumerate(ends):
        end_grid.setdefault(_cell(p, MERGE_M), []).append(i)
    for rid, xy, oneway, zs in lines:
        # Split points: other roads' ends that sit on this road's interior.
        cuts: list[tuple[float, tuple[float, float]]] = []
        run = 0.0
        for k, (p, q) in enumerate(zip(xy, xy[1:])):
            dx, dy = q[0] - p[0], q[1] - p[1]
            seg2 = dx * dx + dy * dy
            if seg2 <= 1e-9:
                continue
            lo = _cell((min(p[0], q[0]) - MERGE_M, min(p[1], q[1]) - MERGE_M), MERGE_M)
            hi = _cell((max(p[0], q[0]) + MERGE_M, max(p[1], q[1]) + MERGE_M), MERGE_M)
            near: list[int] = []
            for gx in range(lo[0], hi[0] + 1):
                for gy in range(lo[1], hi[1] + 1):
                    near.extend(end_grid.get((gx, gy), ()))
            for i in sorted(near):
                e, ez = ends[i]
                t = max(0.0, min(1.0, ((e[0] - p[0]) * dx + (e[1] - p[1]) * dy) / seg2))
                c = (p[0] + dx * t, p[1] + dy * t)
                if math.hypot(e[0] - c[0], e[1] - c[1]) <= MERGE_M:
                    za, zb = zs[k], zs[k + 1]
                    cz = None if za is None or zb is None else za + (zb - za) * t
                    if _dz_ok(ez, cz, dz_max):
                        cuts.append((run + math.sqrt(seg2) * t, c))
            run += math.sqrt(seg2)
        total = run
        cuts = sorted(c for c in cuts if MERGE_M < c[0] < total - MERGE_M)
        pieces: list[list[tuple[float, float]]] = []
        piece_z: list[tuple[float | None, float | None]] = []
        cur = [xy[0]]
        cur_z0 = zs[0]
        run = 0.0
        ci = 0
        for k, (p, q) in enumerate(zip(xy, xy[1:])):
            seg = math.hypot(q[0] - p[0], q[1] - p[1])
            while ci < len(cuts) and cuts[ci][0] <= run + seg:
                za, zb = zs[k], zs[k + 1]
                f = 0.0 if seg <= 0.0 else min(1.0, max(0.0, (cuts[ci][0] - run) / seg))
                cz = None if za is None or zb is None else za + (zb - za) * f
                cur.append(cuts[ci][1])
                pieces.append(cur)
                piece_z.append((cur_z0, cz))
                cur = [cuts[ci][1]]
                cur_z0 = cz
                ci += 1
            cur.append(q)
            run += seg
        pieces.append(cur)
        piece_z.append((cur_z0, zs[-1]))
        for i, piece in enumerate(pieces):
            if len(piece) < 2:
                continue
            a, b = node_for(piece[0], piece_z[i][0]), node_for(piece[-1], piece_z[i][1])
            if a == b:
                continue
            graph.edges.append(Edge(a, b, piece, oneway, rid if len(pieces) == 1 else f"{rid}#{i}", rid))
    return graph


# ---------------------------------------------------------------- junctions


def _flag(road: Any, key: str) -> bool:
    value = road.get(key) if isinstance(road, dict) else None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in ("1", "true", "yes")


def _drivability(road: Any) -> float:
    try:
        return float(road.get("drivability", road.get("drivable", 1)))
    except (AttributeError, TypeError, ValueError):
        return 1.0


def road_minor(road: Any) -> bool:
    return _flag(road, "gatedRoad") or _drivability(road) < MINOR_DRIVABILITY


def filter_network(spec: Any) -> dict[str, Any]:
    """The junction-map filter: drivability > 0, no hiddenInNavi (gated roads stay)."""
    if isinstance(spec, dict) and isinstance(spec.get("roads"), dict):
        spec = spec["roads"]
    out: dict[str, Any] = {}
    for rid, road in (spec or {}).items() if isinstance(spec, dict) else []:
        if not isinstance(road, dict) or not _drivable(road) or _flag(road, "hiddenInNavi"):
            continue
        out[str(rid)] = road
    return out


def _lanes(road: dict[str, Any]) -> int | None:
    try:
        return int(float(road.get("lanesLeft"))) + int(float(road.get("lanesRight")))
    except (TypeError, ValueError):
        return None


def junction_type(arms: list[dict[str, Any]]) -> str:
    heads = [float(a["heading_rad"]) for a in arms]
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            if arms[i]["oneway"] and arms[j]["oneway"]:
                if abs(math.degrees(wrap(heads[i] - heads[j]))) < MERGE_DEG:
                    return "merge"
    if len(arms) >= 5:
        return "multi"
    if len(arms) == 4:
        return "4way"
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            if abs(math.degrees(wrap(heads[i] - heads[j]))) >= 180.0 - THROUGH_DEG:
                return "T"
    return "Y"


def _r2(v: float) -> float:
    out = round(float(v), 2)
    return 0.0 if out == 0.0 else out


def build_junction_map(network: Any, map_name: str, src_sha256: str) -> dict[str, Any]:
    """The ``junctions/<map>.json.gz`` document for one road network."""
    roads = filter_network(network)
    graph = graph_from_network(roads, dz_max=JOIN_DZ_M)
    order = {name: i for i, name in enumerate(graph.nodes)}
    incident: dict[str, list[Edge]] = {n: [] for n in graph.nodes}
    for e in graph.edges:
        incident[e.a].append(e)
        incident[e.b].append(e)
    jnodes = [n for n in graph.nodes if len(incident[n]) >= 3]
    parent = {n: n for n in jnodes}

    def find(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    grid: dict[tuple[int, int], list[str]] = {}
    for n in jnodes:
        grid.setdefault(_cell(graph.nodes[n], CLUSTER_M), []).append(n)
    for n in jnodes:
        p, zn = graph.nodes[n], graph.node_z.get(n)
        cx, cy = _cell(p, CLUSTER_M)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for m in grid.get((cx + dx, cy + dy), ()):
                    if order[m] <= order[n]:
                        continue
                    q = graph.nodes[m]
                    if math.hypot(p[0] - q[0], p[1] - q[1]) <= CLUSTER_M and _dz_ok(zn, graph.node_z.get(m), JOIN_DZ_M):
                        ra, rb = find(n), find(m)
                        if ra != rb:
                            if order[ra] < order[rb]:
                                parent[rb] = ra
                            else:
                                parent[ra] = rb
    clusters: dict[str, list[str]] = {}
    for n in jnodes:
        clusters.setdefault(find(n), []).append(n)
    junctions: list[dict[str, Any]] = []
    for members in clusters.values():
        mset = set(members)
        arms = []
        half = 0.0
        for n in members:
            for e in incident[n]:
                other = e.b if e.a == n else e.a
                if other in mset:
                    continue
                road = roads.get(e.road_id, {})
                half = max(half, 0.5 * graph.road_width.get(e.road_id, 0.0))
                arms.append({
                    "road_id": e.road_id,
                    "heading_rad": round(leave_angle(e, n), 4),
                    "oneway": _flag(road, "oneWay") or _flag(road, "oneway"),
                    "lanes": _lanes(road),
                    "minor": road_minor(road),
                })
        if len(arms) < 3:
            continue
        cx = sum(graph.nodes[n][0] for n in members) / len(members)
        cy = sum(graph.nodes[n][1] for n in members) / len(members)
        zs = [graph.node_z.get(n) for n in members]
        cz = None if any(z is None for z in zs) else sum(zs) / len(zs)  # type: ignore[arg-type]
        offset = max(math.hypot(graph.nodes[n][0] - cx, graph.nodes[n][1] - cy) for n in members)
        arms.sort(key=lambda a: (a["heading_rad"], a["road_id"]))
        junctions.append({
            "c": [_r2(cx), _r2(cy), None if cz is None else _r2(cz)],
            "r_m": _r2(offset + half),
            "type": junction_type(arms),
            "minor": sum(1 for a in arms if not a["minor"]) < 3,
            "arms": arms,
        })
    junctions.sort(key=lambda j: (round(j["c"][0], 1), round(j["c"][1], 1), j["c"][2] if j["c"][2] is not None else 0.0))
    for i, j in enumerate(junctions):
        j["id"] = f"j{i:04d}"
    node_ix = {}
    gnodes = []
    gedges = []
    for e in graph.edges:
        for n in (e.a, e.b):
            if n not in node_ix:
                node_ix[n] = len(gnodes)
                gnodes.append([_r2(graph.nodes[n][0]), _r2(graph.nodes[n][1])])
        gedges.append({"a": node_ix[e.a], "b": node_ix[e.b], "road_id": e.road_id, "oneway": e.oneway,
                       "xy": [[_r2(x), _r2(y)] for x, y in e.xy]})
    return {
        "map": str(map_name),
        "version": VERSION,
        "src_sha256": str(src_sha256),
        "params": dict(PARAMS),
        "junctions": junctions,
        # Additive: the edge graph the labeler walks (the recorder loads only this file).
        "graph": {"nodes": gnodes, "edges": gedges},
    }


def canonical_bytes(doc: dict[str, Any]) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def map_sha256(doc: dict[str, Any]) -> str:
    """sha256 of the canonical (uncompressed) map JSON: the golden sha and every label's ``src``."""
    return hashlib.sha256(canonical_bytes(doc)).hexdigest()


def summary(doc: dict[str, Any]) -> dict[str, Any]:
    js = doc.get("junctions") or []
    types: dict[str, int] = {}
    types_major: dict[str, int] = {}
    for j in js:
        types[j["type"]] = types.get(j["type"], 0) + 1
        if not j["minor"]:
            types_major[j["type"]] = types_major.get(j["type"], 0) + 1
    major = sum(1 for j in js if not j["minor"])
    return {"junctions": len(js), "major": major, "minor": len(js) - major,
            "types": dict(sorted(types.items())), "types_major": dict(sorted(types_major.items()))}


def map_path(maps_dir: Path | str, map_name: str) -> Path:
    return Path(maps_dir) / f"{map_name}.json.gz"


def write_map(doc: dict[str, Any], maps_dir: Path | str) -> Path:
    """Write ``<maps_dir>/<map>.json.gz`` create-only (gzip mtime 0, canonical JSON)."""
    path = map_path(maps_dir, doc["map"])
    blob = gzip.compress(canonical_bytes(doc), compresslevel=9, mtime=0)
    with open(path, "xb") as fh:
        fh.write(blob)
    return path


def load_map(path: Path | str) -> dict[str, Any]:
    return load_map_sha(path)[0]


def load_map_sha(path: Path | str) -> tuple[dict[str, Any], str]:
    """The map document and the sha256 of its uncompressed bytes (the labels' ``src``)."""
    with open(path, "rb") as fh:
        raw = gzip.decompress(fh.read())
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------- labeler


def _seg_dist(x: float, y: float, p, q) -> tuple[float, float]:
    vx, vy = q[0] - p[0], q[1] - p[1]
    seg2 = vx * vx + vy * vy
    u = 0.0 if seg2 <= 1e-12 else max(0.0, min(1.0, ((x - p[0]) * vx + (y - p[1]) * vy) / seg2))
    return math.hypot(p[0] + u * vx - x, p[1] + u * vy - y), u


def _circle_entry(p, q, c, r: float) -> float | None:
    """Fraction along p->q where it first enters circle (c, r), or None."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    fx, fy = p[0] - c[0], p[1] - c[1]
    a = dx * dx + dy * dy
    cc = fx * fx + fy * fy - r * r
    if cc <= 0.0:
        return 0.0
    if a <= 1e-12:
        return None
    b = 2.0 * (fx * dx + fy * dy)
    disc = b * b - 4.0 * a * cc
    if disc < 0.0:
        return None
    s = (-b - math.sqrt(disc)) / (2.0 * a)
    return s if 0.0 <= s <= 1.0 else None


class JunctionLabeler:
    """Causal per-row junction labels for one map.

    ``step(row)`` reads only ``t``, ``map``, ``ego.pos`` and ``ego.dir``. Its
    state is the matched edge and the last pose; ``undo()`` rolls back the
    last ``step`` (the recorder calls it when a row is not queued).
    """

    def __init__(self, doc: dict[str, Any], src: str | None = None) -> None:
        self.map = doc.get("map")
        self.src = src if src is not None else map_sha256(doc)
        graph = doc.get("graph") or {}
        self.nodes = [tuple(n) for n in graph.get("nodes") or []]
        self.edges = [(int(e["a"]), int(e["b"]), [tuple(p) for p in e["xy"]]) for e in graph.get("edges") or []]
        self.adj: dict[int, list[int]] = {}
        for i, (a, b, _xy) in enumerate(self.edges):
            self.adj.setdefault(a, []).append(i)
            self.adj.setdefault(b, []).append(i)
        self.jcts = [(j["id"], (float(j["c"][0]), float(j["c"][1])), float(j["r_m"]), j["type"], bool(j["minor"]))
                     for j in doc.get("junctions") or []]
        self.seg_grid: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for ei, (_a, _b, xy) in enumerate(self.edges):
            for si, (p, q) in enumerate(zip(xy, xy[1:])):
                for cell in self._cells(min(p[0], q[0]), min(p[1], q[1]), max(p[0], q[0]), max(p[1], q[1])):
                    self.seg_grid.setdefault(cell, []).append((ei, si))
        self.jct_grid: dict[tuple[int, int], list[int]] = {}
        for k, (_id, c, r, _t, _m) in enumerate(self.jcts):
            for cell in self._cells(c[0] - r, c[1] - r, c[0] + r, c[1] + r):
                self.jct_grid.setdefault(cell, []).append(k)
        # Junction circles each edge passes through.
        self.edge_jcts: list[list[int]] = []
        for _a, _b, xy in self.edges:
            hit: set[int] = set()
            for p, q in zip(xy, xy[1:]):
                cands: set[int] = set()
                for cell in self._cells(min(p[0], q[0]), min(p[1], q[1]), max(p[0], q[0]), max(p[1], q[1])):
                    cands.update(self.jct_grid.get(cell, ()))
                for k in cands:
                    if _seg_dist(self.jcts[k][1][0], self.jcts[k][1][1], p, q)[0] <= self.jcts[k][2]:
                        hit.add(k)
            self.edge_jcts.append(sorted(hit))
        self.resets = 0
        self._edge: int | None = None
        self._last: tuple[float, float] | None = None
        self._saved: tuple | None = None

    @staticmethod
    def _cells(x0: float, y0: float, x1: float, y1: float) -> Iterable[tuple[int, int]]:
        for gx in range(math.floor(x0 / GRID_M), math.floor(x1 / GRID_M) + 1):
            for gy in range(math.floor(y0 / GRID_M), math.floor(y1 / GRID_M) + 1):
                yield (gx, gy)

    def reset(self) -> None:
        self._edge = None
        self._last = None

    def undo(self) -> None:
        if self._saved is not None:
            self._edge, self._last, self.resets = self._saved
            self._saved = None

    def _blank(self, t: Any) -> dict[str, Any]:
        return {"t": t, "in_jct": None, "jct_id": None, "next_jct_m": None, "jct_type": None, "minor": None,
                "src": self.src}

    def _out(self, t: Any, inside: bool, k: int | None, dist: float | None) -> dict[str, Any]:
        if k is None:
            return {"t": t, "in_jct": inside, "jct_id": None, "next_jct_m": None, "jct_type": None,
                    "minor": None, "src": self.src}
        jid, _c, _r, kind, minor = self.jcts[k]
        return {"t": t, "in_jct": inside, "jct_id": jid, "next_jct_m": _r2(dist or 0.0), "jct_type": kind,
                "minor": minor, "src": self.src}

    def step(self, row: dict[str, Any]) -> dict[str, Any]:
        self._saved = (self._edge, self._last, self.resets)
        t = row.get("t")
        ego = row.get("ego") if isinstance(row.get("ego"), dict) else {}
        pos, direction = ego.get("pos"), ego.get("dir")
        try:
            x, y = float(pos[0]), float(pos[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError
        except (TypeError, ValueError, IndexError):
            return self._blank(t)
        heading = None
        try:
            dx, dy = float(direction[0]), float(direction[1])
            if math.hypot(dx, dy) > 1e-6:
                heading = math.atan2(dy, dx)
        except (TypeError, ValueError, IndexError):
            heading = None
        if self._last is not None and math.hypot(x - self._last[0], y - self._last[1]) > JUMP_M:
            self._edge = None
            self.resets += 1
        self._last = (x, y)
        match = self._match(x, y, heading)
        self._edge = None if match is None else match[0]
        inside = self._inside(x, y)
        if inside is not None:
            return self._out(t, True, inside, 0.0)
        if match is None:
            return self._out(t, False, None, None)
        found = self._ahead(*match)
        if found is None:
            return self._out(t, False, None, None)
        return self._out(t, False, found[0], found[1])

    def _inside(self, x: float, y: float) -> int | None:
        best = None
        for k in self.jct_grid.get((math.floor(x / GRID_M), math.floor(y / GRID_M)), ()):
            _id, c, r, _t, _m = self.jcts[k]
            d = math.hypot(x - c[0], y - c[1])
            if d <= r and (best is None or (d, k) < best):
                best = (d, k)
        return None if best is None else best[1]

    def _match(self, x: float, y: float, heading: float | None):
        """(edge, segment, u, sense) of the matched edge, or None."""
        gx, gy = math.floor(x / GRID_M), math.floor(y / GRID_M)
        per_edge: dict[int, tuple[float, int, float, int]] = {}
        seen: set[tuple[int, int]] = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for key in self.seg_grid.get((gx + dx, gy + dy), ()):
                    if key in seen:
                        continue
                    seen.add(key)
                    ei, si = key
                    xy = self.edges[ei][2]
                    p, q = xy[si], xy[si + 1]
                    if heading is None:
                        continue
                    seg_h = math.atan2(q[1] - p[1], q[0] - p[0])
                    err_f = abs(math.degrees(wrap(heading - seg_h)))
                    if err_f < HEADING_MAX_DEG:
                        sense = 1
                    elif 180.0 - err_f < HEADING_MAX_DEG:
                        sense = -1
                    else:
                        continue
                    d, u = _seg_dist(x, y, p, q)
                    got = per_edge.get(ei)
                    if got is None or (d, si) < (got[0], got[1]):
                        per_edge[ei] = (d, si, u, sense)
        if not per_edge:
            return None
        best = min(per_edge, key=lambda ei: (per_edge[ei][0], ei))
        if self._edge is not None and self._edge in per_edge and per_edge[self._edge][0] <= per_edge[best][0] + HYST_M:
            best = self._edge
        d, si, u, sense = per_edge[best]
        return best, si, u, sense

    def _ahead(self, ei: int, si: int, u: float, sense: int) -> tuple[int, float] | None:
        """(junction, distance) of the first circle ahead along the edge chain, capped at CAP_M."""
        xy = self.edges[ei][2]
        p, q = xy[si], xy[si + 1]
        start = (p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u)
        head = (q[0] - p[0], q[1] - p[1]) if sense > 0 else (p[0] - q[0], p[1] - q[1])
        skip: set[int] = set()
        for k in self.jct_grid.get((math.floor(start[0] / GRID_M), math.floor(start[1] / GRID_M)), ()):
            _id, c, r, _t, _m = self.jcts[k]
            if math.hypot(start[0] - c[0], start[1] - c[1]) <= r:
                if (start[0] - c[0]) * head[0] + (start[1] - c[1]) * head[1] >= 0.0:
                    skip.add(k)  # leaving it
                else:
                    return k, 0.0
        # Path: from start to the end of this edge in the travel direction.
        if sense > 0:
            pts = [start] + xy[si + 1:]
            end_node = self.edges[ei][1]
        else:
            pts = [start] + list(reversed(xy[: si + 1]))
            end_node = self.edges[ei][0]
        total = 0.0
        visited = {ei}
        while True:
            best: tuple[float, int] | None = None
            cands = [k for k in self.edge_jcts[ei] if k not in skip]
            run = total
            for a, b in zip(pts, pts[1:]):
                seg = math.hypot(b[0] - a[0], b[1] - a[1])
                if cands:
                    for k in cands:
                        s = _circle_entry(a, b, self.jcts[k][1], self.jcts[k][2])
                        if s is not None and (best is None or (run + s * seg, k) < best):
                            best = (run + s * seg, k)
                    if best is not None:
                        break
                run += seg
                if run > CAP_M:
                    break
            if best is not None:
                return (best[1], best[0]) if best[0] <= CAP_M else None
            total = run
            if total > CAP_M:
                return None
            nxt = [e for e in self.adj.get(end_node, ()) if e != ei]
            if len(self.adj.get(end_node, ())) != 2 or len(nxt) != 1 or nxt[0] in visited:
                return None
            ei = nxt[0]
            visited.add(ei)
            a, b, exy = self.edges[ei]
            if a == end_node:
                pts_n, end_node = list(exy), b
            else:
                pts_n, end_node = list(reversed(exy)), a
            total += math.hypot(pts_n[0][0] - pts[-1][0], pts_n[0][1] - pts[-1][1])
            pts = pts_n
            if total > CAP_M:
                return None


def label_line(out: dict[str, Any]) -> str:
    return json.dumps(out, sort_keys=True, separators=(",", ":"), allow_nan=False)


class JctSidecar:
    """``jct.jsonl`` writer shared by the recorder and the relabel.

    One line per state row. The file opens create-only (mode ``"x"``) on the
    first row whose map has a junction map; earlier rows' lines are written
    then. An existing file is never written. A row on a map without a junction
    map gets a null line (``src`` null). Errors become ``{"err", "t"}`` lines
    and are never raised.
    """

    def __init__(self, path: Path | str, maps_dir: Path | str, log: Callable[[str], None] = print,
                 prefix: str = "collect jct", flush_every: int = FLUSH_ROWS) -> None:
        self.path = Path(path)
        self.maps_dir = Path(maps_dir)
        self.log = log
        self.prefix = prefix
        self.flush_every = int(flush_every)
        self.labelers: dict[str, JunctionLabeler | None] = {}
        self.fh = None
        self.pending: list[str] = []
        self.rows = 0
        self.errors = 0
        self.disabled = False
        self._last_map: Any = None
        self._saved: tuple | None = None
        self._cur: JunctionLabeler | None = None
        if self.path.exists():
            self._exists()

    def _exists(self) -> None:
        self.disabled = True
        self.pending = []
        self.log(f"{self.prefix} {self.path.name} exists in {self.path.parent.name}; not written")

    def prepare(self, map_name: Any) -> None:
        """Load the junction map for ``map_name`` (once). Call outside the tick loop."""
        key = str(map_name or "current")
        if key in self.labelers:
            return
        path = map_path(self.maps_dir, key) if map_name else None
        try:
            if path is None or not path.is_file():
                raise FileNotFoundError
            self.labelers[key] = JunctionLabeler(*load_map_sha(path))
            self.log(f"{self.prefix} on map={key} junctions={len(self.labelers[key].jcts)}")
        except FileNotFoundError:
            self.labelers[key] = None
            self.log(f"{self.prefix} off map={key}")
        except Exception as exc:  # noqa: BLE001 - labels are never fatal
            self.labelers[key] = None
            self.log(f"{self.prefix} off map={key} load failed: {type(exc).__name__}: {exc}")

    def step(self, row: dict[str, Any]) -> str | None:
        """The label line for ``row`` (not yet written; see ``commit``)."""
        if self.disabled:
            return None
        t = row.get("t")
        try:
            key = row.get("map")
            self._saved = (self._last_map, None)
            if key is not None and str(key) not in self.labelers:
                self.prepare(key)
            lab = self.labelers.get(str(key)) if key is not None else None
            self._cur = lab
            if lab is not None and key != self._last_map:
                lab.reset()
            self._last_map = key
            if lab is None:
                return label_line({"t": t, "in_jct": None, "jct_id": None, "next_jct_m": None,
                                   "jct_type": None, "minor": None, "src": None})
            return label_line(lab.step(row))
        except Exception as exc:  # noqa: BLE001 - labels are never fatal
            self.errors += 1
            self._cur = None
            return json.dumps({"err": f"{type(exc).__name__}: {exc}", "t": t}, sort_keys=True, separators=(",", ":"))

    def undo(self) -> None:
        """Forget the last ``step`` (its row was not written)."""
        if self._saved is not None:
            self._last_map = self._saved[0]
            if self._cur is not None:
                self._cur.undo()
        self._saved = None
        self._cur = None

    def commit(self, line: str | None) -> None:
        if self.disabled or line is None:
            return
        self._saved = None
        self._cur = None
        try:
            if self.fh is None:
                if '"src":null' in line or '"err"' in line and not any(self.labelers.values()):
                    self.pending.append(line)
                    return
                try:
                    self.fh = open(self.path, "x", encoding="utf-8", newline="\n")
                except FileExistsError:
                    self._exists()
                    return
                for old in self.pending:
                    self.fh.write(old + "\n")
                self.rows += len(self.pending)
                self.pending = []
            self.fh.write(line + "\n")
            self.rows += 1
            if self.rows % self.flush_every == 0:
                self.fh.flush()
        except OSError as exc:
            self.disabled = True
            self.log(f"{self.prefix} write failed, labels off: {type(exc).__name__}: {exc}")

    def flush(self) -> None:
        try:
            if self.fh is not None:
                self.fh.flush()
        except OSError as exc:
            self.disabled = True
            self.log(f"{self.prefix} flush failed, labels off: {type(exc).__name__}: {exc}")

    def close(self) -> None:
        try:
            if self.fh is not None:
                self.fh.close()
        except OSError as exc:
            self.log(f"{self.prefix} close failed: {type(exc).__name__}: {exc}")
        self.fh = None


# ---------------------------------------------------------------- exits (offline only)


def junction_exits(rows: list[dict[str, Any]], labels: list[dict[str, Any]],
                   classify: Callable[[float], str] | None = None) -> list[dict[str, Any]]:
    """``jct_exit.jsonl`` records: the turn at each junction circle driven through.

    Needs the row after the circle, so it is offline only. The heading change
    from the row before the circle to the row after it goes through
    ``junction_grade.classify``.
    """
    if classify is None:
        import importlib

        classify = importlib.import_module("scripts.junction_grade").classify

    def pose(row):
        ego = row.get("ego") if isinstance(row.get("ego"), dict) else {}
        try:
            p, d = ego.get("pos"), ego.get("dir")
            return float(p[0]), float(p[1]), math.atan2(float(d[1]), float(d[0]))
        except (TypeError, ValueError, IndexError):
            return None

    poses = [pose(r) for r in rows]
    out: list[dict[str, Any]] = []
    i, n = 0, len(rows)
    while i < n:
        lab = labels[i] if i < len(labels) else {}
        if not lab.get("in_jct"):
            i += 1
            continue
        jid = lab.get("jct_id")
        j = i
        while j + 1 < n and labels[j + 1].get("in_jct") and labels[j + 1].get("jct_id") == jid:
            j += 1
        before, after = i - 1, j + 1
        ok = before >= 0 and after < n and poses[before] is not None and poses[after] is not None
        if ok:
            for k in range(before, after):
                a, b = poses[k], poses[k + 1]
                if a is None or b is None or rows[k].get("map") != rows[k + 1].get("map") or \
                        math.hypot(b[0] - a[0], b[1] - a[1]) > JUMP_M:
                    ok = False
                    break
        if ok and not labels[after].get("in_jct"):
            out.append({"jct_id": jid, "t_in": rows[i].get("t"), "t_out": rows[after].get("t"),
                        "exit": classify(poses[after][2] - poses[before][2])})
        i = j + 1
    return out


# ---------------------------------------------------------------- navgraph check (dev PC)


def _road_index(network: Any) -> tuple[list[tuple[str, Any, Any, dict]], dict[tuple[int, int], list[int]]]:
    roads = filter_network(network)
    segs: list[tuple[str, Any, Any, dict]] = []
    grid: dict[tuple[int, int], list[int]] = {}
    for rid, road in roads.items():
        poly = parse_road_edges(rid, road.get("edges") or road.get("nodes"))
        if poly is None:
            continue
        for p, q in zip(poly.xy, poly.xy[1:]):
            k = len(segs)
            segs.append((rid, p, q, road))
            for gx in range(math.floor((min(p[0], q[0]) - ARM_MAP_M) / GRID_M), math.floor((max(p[0], q[0]) + ARM_MAP_M) / GRID_M) + 1):
                for gy in range(math.floor((min(p[1], q[1]) - ARM_MAP_M) / GRID_M), math.floor((max(p[1], q[1]) + ARM_MAP_M) / GRID_M) + 1):
                    grid.setdefault((gx, gy), []).append(k)
    return segs, grid


def map_arm_point(index, xy: tuple[float, float]) -> dict | None:
    """The road nearest ``xy`` within ``ARM_MAP_M`` (its road dict), or None."""
    segs, grid = index
    best = None
    for k in grid.get((math.floor(xy[0] / GRID_M), math.floor(xy[1] / GRID_M)), ()):
        rid, p, q, road = segs[k]
        d, _u = _seg_dist(xy[0], xy[1], p, q)
        if d <= ARM_MAP_M and (best is None or (d, rid) < best[:2]):
            best = (d, rid, road)
    return None if best is None else best[2]


def classify_navgraph_nodes(navgraph: dict[str, Any], network: Any) -> dict[str, Any]:
    """Major / minor / unmapped navgraph junction nodes (the frozen metric-1(a) rule).

    A node with any arm that maps to no road is unmapped (checked first);
    otherwise it is minor when any arm maps to a minor road, else major.

    ``navgraph`` is the ``NavigraphData`` shape: ``graph`` (node -> linked
    nodes) and ``coords`` (node -> xyz). A candidate has more than 2 links.
    Each link's arm point is ``ARM_POINT_M`` along it (or the far node) and maps
    to the nearest filtered road centreline within ``ARM_MAP_M``.
    """
    graph = navgraph.get("graph") or {}
    coords = {str(k): v for k, v in (navgraph.get("coords") or {}).items()}
    links: dict[str, set[str]] = {}
    for a, nbrs in graph.items():
        names = nbrs.keys() if isinstance(nbrs, dict) else nbrs
        for b in names:
            if str(a) == str(b):
                continue
            links.setdefault(str(a), set()).add(str(b))
            links.setdefault(str(b), set()).add(str(a))
    index = _road_index(network)
    out: dict[str, Any] = {"major": [], "minor": [], "unmapped": [], "candidates": 0}
    for node in sorted(links):
        if len(links[node]) <= 2 or node not in coords:
            continue
        out["candidates"] += 1
        px, py = float(coords[node][0]), float(coords[node][1])
        mapped = []
        for other in sorted(links[node]):
            if other not in coords:
                mapped.append(None)
                continue
            ox, oy = float(coords[other][0]), float(coords[other][1])
            length = math.hypot(ox - px, oy - py)
            f = 1.0 if length <= ARM_POINT_M else ARM_POINT_M / length
            mapped.append(map_arm_point(index, (px + (ox - px) * f, py + (oy - py) * f)))
        entry = {"id": node, "pos": [round(px, 2), round(py, 2)]}
        # Unmapped first: a node with any unmapped arm is unmapped, even when another arm is minor.
        if any(road is None for road in mapped):
            out["unmapped"].append(entry)
        elif any(road_minor(road) for road in mapped):
            out["minor"].append(entry)
        else:
            out["major"].append(entry)
    return out


def navgraph_metric(doc: dict[str, Any], classified: dict[str, Any], match_m: float = MATCH_M) -> dict[str, Any]:
    """Metric 1(a): recall, precision, splits and merges of our major junctions vs navgraph major nodes."""

    def score(ours: list[tuple[float, float]], theirs: list[tuple[float, float]]) -> dict[str, Any]:
        hits = [[k for k, c in enumerate(ours) if math.hypot(p[0] - c[0], p[1] - c[1]) <= match_m] for p in theirs]
        recall_n = sum(1 for h in hits if len(h) == 1)
        matched: dict[int, list[int]] = {}
        for i, h in enumerate(hits):
            for k in h:
                matched.setdefault(k, []).append(i)
        merges = sum(1 for k, idx in matched.items()
                     if any(math.dist(theirs[a], theirs[b]) > match_m for a in idx for b in idx if a < b))
        return {"recall": None if not theirs else recall_n / len(theirs),
                "precision": None if not ours else len(matched) / len(ours),
                "splits": sum(1 for h in hits if len(h) >= 2), "merges": merges,
                "ours": len(ours), "theirs": len(theirs)}

    js = doc.get("junctions") or []
    major = score([tuple(j["c"][:2]) for j in js if not j["minor"]], [tuple(n["pos"]) for n in classified["major"]])
    minor = score([tuple(j["c"][:2]) for j in js if j["minor"]], [tuple(n["pos"]) for n in classified["minor"]])
    cand = int(classified.get("candidates") or 0)
    share = None if not cand else len(classified["unmapped"]) / cand
    return {"major": major, "minor": minor, "unmapped": len(classified["unmapped"]), "unmapped_share": share,
            "void": share is not None and share > UNMAPPED_MAX}


def navgraph_verdict(metric: dict[str, Any], golden: dict[str, Any]) -> str:
    """``PASS``, ``FAIL`` or ``VOID`` for metric 1(a), from the frozen golden values only.

    VOID first: an unmapped share above the golden ``UNMAPPED_MAX`` means the
    navgraph comparison does not count. Otherwise PASS needs major recall and
    major precision at or above the golden bars.
    """
    share = metric.get("unmapped_share")
    if share is None or share > float(golden["params"]["UNMAPPED_MAX"]):
        return "VOID"
    major = metric.get("major") or {}
    recall, precision = major.get("recall"), major.get("precision")
    bars = golden["bars"]
    if recall is None or precision is None:
        return "FAIL"
    if recall >= float(bars["major_recall_min"]) and precision >= float(bars["major_precision_min"]):
        return "PASS"
    return "FAIL"


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

