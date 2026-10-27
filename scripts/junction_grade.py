#!/usr/bin/env python3
"""Offline junction grader: which way did the car go at each junction in a GT log?

For every junction the logged path crosses, two answers are compared:

- **log**: left / through / right from the logged heading change between
  entering and leaving the junction (``heading_rad``, counter-clockwise
  positive, so a positive change is a left turn);
- **graph**: the branch actually taken in the offline road graph (the road
  the car came in on and the road it left on), classified the same way.

A crossing that drives a one-way edge against its direction is flagged
``wrong_way`` and does not count as a match. ``route`` is a plain shortest
path that honours one-way edges and answers ``no_route`` when the target
cannot be reached.

Offline only. It reads a GT log (``gt_*.jsonl``, optionally gzipped) and
either the synthetic graph fixture or an offline road dump
(``roads/<map>.json.gz`` from ``collect_scene_lanes.py --dump-roads``). It
never talks to BeamNG and nothing at runtime imports it.

    py -3 scripts/junction_grade.py --gt logs/gt_X.jsonl --roads roads/map.json.gz
    py -3 scripts/junction_grade.py --gt ... --roads ... --bar logs/bar.txt --bar-sha256 <hex>

With ``--bar`` the grader prints the bar file's sha256 and refuses (exit 2)
a file whose hash is not ``--bar-sha256``. The bar text reads like
"turn match >=95% over N >=10 junctions"; exit 0 means the bar is met.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.debug.gt_geom import parse_road_edges  # noqa: E402

THROUGH_DEG = 20.0  # |heading change| below this is "through"
NEAR_M = 20.0  # a pose this close to a junction node is inside the junction
LEAD_M = 25.0  # how far along a road its direction at a node is measured
MERGE_M = 4.0  # road ends this close together are one node
JUMP_M = 30.0  # a pose step longer than this is a respawn, not driving
CLASSES = ("left", "through", "right")


def wrap(rad: float) -> float:
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def classify(delta_rad: float) -> str:
    """left / through / right from a heading change (CCW positive)."""
    deg = math.degrees(wrap(delta_rad))
    if abs(deg) < THROUGH_DEG:
        return "through"
    return "left" if deg > 0.0 else "right"


@dataclass
class Edge:
    a: str
    b: str
    xy: list[tuple[float, float]]
    oneway: bool = False
    name: str = ""

    @property
    def length(self) -> float:
        return sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(self.xy, self.xy[1:]))


@dataclass
class Graph:
    nodes: dict[str, tuple[float, float]] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

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


# ---------------------------------------------------------------- loading


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


def graph_from_network(spec: Any) -> Graph:
    """A node/edge graph from a raw road network (an offline road dump).

    Road ends within ``MERGE_M`` share a node. A road end that lands on the
    middle of another road splits that road there, so a side street that
    ends on a through road makes a T.
    """
    if isinstance(spec, dict) and isinstance(spec.get("roads"), dict):
        spec = spec["roads"]
    lines: list[tuple[str, list[tuple[float, float]], bool]] = []
    for road_id, road in (spec or {}).items() if isinstance(spec, dict) else []:
        if not _drivable(road):
            continue
        edges = road.get("edges") or road.get("nodes") if isinstance(road, dict) else road
        poly = parse_road_edges(str(road_id), edges)
        if poly is None or len(poly.xy) < 2:
            continue
        oneway = bool(isinstance(road, dict) and (road.get("oneWay") or road.get("oneway")))
        lines.append((str(road_id), [(float(x), float(y)) for x, y in poly.xy], oneway))
    graph = Graph()

    def node_for(p: tuple[float, float]) -> str:
        for name, q in graph.nodes.items():
            if math.hypot(p[0] - q[0], p[1] - q[1]) <= MERGE_M:
                return name
        name = f"n{len(graph.nodes)}"
        graph.nodes[name] = p
        return name

    ends = [p for _rid, xy, _ow in lines for p in (xy[0], xy[-1])]
    for p in ends:
        node_for(p)
    for rid, xy, oneway in lines:
        # Split points: other roads' ends that sit on this road's interior.
        cuts: list[tuple[float, tuple[float, float]]] = []
        run = 0.0
        for p, q in zip(xy, xy[1:]):
            dx, dy = q[0] - p[0], q[1] - p[1]
            seg2 = dx * dx + dy * dy
            if seg2 <= 1e-9:
                continue
            for e in ends:
                t = max(0.0, min(1.0, ((e[0] - p[0]) * dx + (e[1] - p[1]) * dy) / seg2))
                c = (p[0] + dx * t, p[1] + dy * t)
                if math.hypot(e[0] - c[0], e[1] - c[1]) <= MERGE_M:
                    cuts.append((run + math.sqrt(seg2) * t, c))
            run += math.sqrt(seg2)
        total = run
        cuts = sorted(c for c in cuts if MERGE_M < c[0] < total - MERGE_M)
        pieces: list[list[tuple[float, float]]] = []
        cur = [xy[0]]
        run = 0.0
        ci = 0
        for p, q in zip(xy, xy[1:]):
            seg = math.hypot(q[0] - p[0], q[1] - p[1])
            while ci < len(cuts) and cuts[ci][0] <= run + seg:
                cur.append(cuts[ci][1])
                pieces.append(cur)
                cur = [cuts[ci][1]]
                ci += 1
            cur.append(q)
            run += seg
        pieces.append(cur)
        for i, piece in enumerate(pieces):
            if len(piece) < 2:
                continue
            a, b = node_for(piece[0]), node_for(piece[-1])
            if a == b:
                continue
            graph.edges.append(Edge(a, b, piece, oneway, rid if len(pieces) == 1 else f"{rid}#{i}"))
    return graph


def load_graph(path: Path, level: str | None = None) -> Graph:
    path = Path(path)
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if isinstance(data, dict) and "levels" in data:
        return graph_from_fixture(data, level)
    if isinstance(data, dict) and "road_network" in data:
        return graph_from_network(data["road_network"])
    return graph_from_network(data)


def load_rows(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    rows = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------- routing


def route(graph: Graph, src: str, dst: str) -> dict[str, Any]:
    """Shortest path that honours one-way edges. ``{"status": "no_route"}`` when there is none."""
    if src not in graph.nodes or dst not in graph.nodes:
        return {"status": "no_route", "nodes": [], "length_m": None}
    adj: dict[str, list[tuple[str, float]]] = {n: [] for n in graph.nodes}
    for e in graph.edges:
        adj[e.a].append((e.b, e.length))
        if not e.oneway:
            adj[e.b].append((e.a, e.length))
    dist = {src: 0.0}
    prev: dict[str, str] = {}
    heap = [(0.0, src)]
    while heap:
        d, n = heapq.heappop(heap)
        if n == dst:
            break
        if d > dist.get(n, math.inf):
            continue
        for m, w in adj[n]:
            nd = d + w
            if nd < dist.get(m, math.inf):
                dist[m] = nd
                prev[m] = n
                heapq.heappush(heap, (nd, m))
    if dst not in dist:
        return {"status": "no_route", "nodes": [], "length_m": None}
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return {"status": "ok", "nodes": list(reversed(path)), "length_m": dist[dst]}


# ---------------------------------------------------------------- grading


def _pose(row: dict[str, Any]) -> tuple[float, float, float | None] | None:
    pos = row.get("pos")
    if not isinstance(pos, (list, tuple)) or len(pos) < 2 or pos[0] is None or pos[1] is None:
        return None
    heading = row.get("heading_rad")
    return float(pos[0]), float(pos[1]), None if heading is None else float(heading)


def _best_edge(graph: Graph, node: str, angle: float) -> Edge | None:
    best = None
    for e in graph.incident(node):
        err = abs(wrap(leave_angle(e, node) - angle))
        if best is None or err < best[0]:
            best = (err, e)
    return None if best is None else best[1]


def crossings(rows: list[dict[str, Any]], graph: Graph) -> list[dict[str, Any]]:
    """One record per junction the log drives through, in log order."""
    poses = [_pose(r) for r in rows]
    # Split the log at respawns and gaps: a crossing must be driven, so the
    # poses before, inside and after it share one continuous stretch.
    stretch: list[int] = []
    cur, last = 0, None
    for p in poses:
        if p is None or (last is not None and math.hypot(p[0] - last[0], p[1] - last[1]) > JUMP_M):
            cur += 1
        stretch.append(cur)
        last = p
    found: list[dict[str, Any]] = []
    for node in graph.junctions():
        nx, ny = graph.nodes[node]
        inside = [p is not None and math.hypot(p[0] - nx, p[1] - ny) <= NEAR_M for p in poses]
        i = 0
        while i < len(inside):
            if not inside[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(inside) and inside[j + 1] and stretch[j + 1] == stretch[i]:
                j += 1
            before, after = i - 1, j + 1
            if (
                before >= 0
                and after < len(poses)
                and poses[before] is not None
                and poses[after] is not None
                and stretch[before] == stretch[i] == stretch[after]
            ):
                pb, pa = poses[before], poses[after]
                h_in = pb[2] if pb[2] is not None else math.atan2(ny - pb[1], nx - pb[0])
                h_out = pa[2] if pa[2] is not None else math.atan2(pa[1] - ny, pa[0] - nx)
                log_class = classify(h_out - h_in)
                e_in = _best_edge(graph, node, math.atan2(pb[1] - ny, pb[0] - nx))
                e_out = _best_edge(graph, node, math.atan2(pa[1] - ny, pa[0] - nx))
                graph_class = None
                wrong_way = False
                if e_in is not None and e_out is not None and e_in is not e_out:
                    travel_in = wrap(leave_angle(e_in, node) + math.pi)
                    graph_class = classify(leave_angle(e_out, node) - travel_in)
                    # Arriving on a one-way edge must be a -> b (ending here);
                    # leaving on one must start here.
                    wrong_way = (e_in.oneway and e_in.b != node) or (e_out.oneway and e_out.a != node)
                found.append(
                    {
                        "row": i,
                        "node": node,
                        "log": log_class,
                        "graph": graph_class,
                        "in": None if e_in is None else e_in.name,
                        "out": None if e_out is None else e_out.name,
                        "wrong_way": wrong_way,
                        "match": graph_class is not None and log_class == graph_class and not wrong_way,
                        "turn_deg": round(math.degrees(wrap(h_out - h_in)), 1),
                    }
                )
            i = j + 1
    found.sort(key=lambda c: c["row"])
    return found


def summarize(found: Iterable[dict[str, Any]]) -> dict[str, Any]:
    found = list(found)
    n = len(found)
    k = sum(1 for c in found if c["match"])
    return {"junctions": n, "matched": k, "rate": (k / n) if n else None}


# ---------------------------------------------------------------- bar


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_bar(text: str) -> tuple[float, int]:
    """``(min rate, min N)`` from text like "turn match >=95% over N >=10 junctions"."""
    pct = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    count = re.search(r"N\s*(?:≥|>=|=>)\s*(\d+)", text)
    if pct is None or count is None:
        raise ValueError("bar file needs a percentage and 'N >= <count>'")
    return float(pct.group(1)) / 100.0, int(count.group(1))


def check_bar(path: Path, want_sha: str) -> tuple[float, int, str]:
    """The bar, or ValueError when its hash is not the recorded one."""
    got = sha256_file(path)
    print(f"bar file sha256 {got}")
    if not want_sha or got.lower() != want_sha.strip().lower():
        raise ValueError(f"bar file hash {got} != recorded {want_sha or '(none)'}; refusing to grade")
    rate, n_min = parse_bar(Path(path).read_text(encoding="utf-8"))
    return rate, n_min, got


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gt", required=True, type=Path, help="GT log (jsonl or jsonl.gz)")
    ap.add_argument("--roads", required=True, type=Path, help="offline road dump or synthetic graph json")
    ap.add_argument("--level", default=None, help="level inside a synthetic graph file")
    ap.add_argument("--bar", type=Path, default=None, help="bar file written before the run")
    ap.add_argument("--bar-sha256", default="", help="recorded sha256 of the bar file")
    ap.add_argument("--json", type=Path, default=None, help="write the crossings here")
    args = ap.parse_args(argv)

    bar = None
    if args.bar is not None:
        try:
            bar = check_bar(args.bar, args.bar_sha256)
        except (OSError, ValueError) as exc:
            print(f"junction_grade: {exc}", file=sys.stderr)
            return 2
    graph = load_graph(args.roads, args.level)
    found = crossings(load_rows(args.gt), graph)
    for c in found:
        flag = "ok " if c["match"] else "BAD"
        extra = " wrong_way" if c["wrong_way"] else ""
        print(f"{flag} row {c['row']:6d} node {c['node']}: log {c['log']:7s} graph {str(c['graph']):7s} "
              f"turn {c['turn_deg']:+6.1f} deg  {c['in']} -> {c['out']}{extra}")
    summary = summarize(found)
    rate = "n/a" if summary["rate"] is None else f"{100.0 * summary['rate']:.1f}%"
    print(f"turn match {summary['matched']}/{summary['junctions']} ({rate})")
    if args.json is not None:
        args.json.write_text(json.dumps({"summary": summary, "crossings": found}, indent=1), encoding="utf-8")
    if bar is None:
        return 0
    need_rate, need_n, _sha = bar
    ok = summary["junctions"] >= need_n and (summary["rate"] or 0.0) >= need_rate
    print(f"bar: rate >= {100.0 * need_rate:.1f}% over N >= {need_n}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
