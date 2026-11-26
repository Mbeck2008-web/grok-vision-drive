"""Build a route_intent.Graph from a collected road dump.

Offline / labeler only. Do not import this module from run_vision, the
planner, or junction_grade. The walk is adapted from
``scripts/junction_grade.py`` ``graph_from_network`` (merge nearby ends,
split a through road where a side street meets it) but emits
``python.sensors.route_intent.Graph``.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

from python.debug.gt_geom import parse_road_edges
from python.sensors.route_intent import Graph

MERGE_M = 4.0  # road ends this close together are one node


def _drivable(road: Any) -> bool:
    if not isinstance(road, dict):
        return True
    try:
        return float(road.get("drivability", road.get("drivable", 1))) > 0.0
    except (TypeError, ValueError):
        return True


def _oneway(road: Any) -> bool:
    """BeamNG dumps store oneWay as the string '0' / '1', not a bool."""
    if not isinstance(road, dict):
        return False
    value = road.get("oneWay")
    if value is None:
        value = road.get("oneway")
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return value != 0
    return bool(value)


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
        lines.append((str(road_id), [(float(x), float(y)) for x, y in poly.xy], _oneway(road)))

    nodes: dict[str, tuple[float, float]] = {}
    edge_rows: list[tuple[str, str, bool]] = []

    def node_for(p: tuple[float, float]) -> str:
        for name, q in nodes.items():
            if math.hypot(p[0] - q[0], p[1] - q[1]) <= MERGE_M:
                return name
        name = f"n{len(nodes)}"
        nodes[name] = p
        return name

    ends = [p for _rid, xy, _ow in lines for p in (xy[0], xy[-1])]
    for p in ends:
        node_for(p)
    for _rid, xy, oneway in lines:
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
        for piece in pieces:
            if len(piece) < 2:
                continue
            a, b = node_for(piece[0]), node_for(piece[-1])
            if a == b:
                continue
            edge_rows.append((a, b, oneway))
    return Graph(nodes=nodes, edges=tuple(edge_rows))


def _read_json(path: Path) -> Any:
    raw = path.read_bytes()
    if path.suffix == ".gz" or raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def load_label_graph(path: Path | str, level: str | None = None) -> Graph:
    """Fixture levels, {nodes, edges} JSON, or a gzip road dump."""
    data = _read_json(Path(path))
    if not isinstance(data, dict):
        raise ValueError(f"road graph must be a JSON object: {path}")
    if "levels" in data:
        name = level or next(iter(data["levels"]))
        return Graph.from_fixture(data, name)
    if "nodes" in data and "edges" in data:
        nodes = {str(k): (float(v[0]), float(v[1])) for k, v in data["nodes"].items()}
        edges = tuple((str(e["a"]), str(e["b"]), bool(e.get("oneway", False))) for e in data["edges"])
        return Graph(nodes=nodes, edges=edges)
    if "road_network" in data:
        return graph_from_network(data["road_network"])
    return graph_from_network(data)
