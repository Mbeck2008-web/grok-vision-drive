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
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.collect.junction_map import (  # noqa: E402,F401 - the graph code lives there
    LEAD_M,
    MERGE_M,
    Edge,
    Graph,
    _drivable,
    _point_along,
    graph_from_fixture,
    graph_from_network,
    leave_angle,
    wrap,
)

THROUGH_DEG = 20.0  # |heading change| below this is "through"
NEAR_M = 20.0  # a pose this close to a junction node is inside the junction
JUMP_M = 30.0  # a pose step longer than this is a respawn, not driving
CLASSES = ("left", "through", "right")


def classify(delta_rad: float) -> str:
    """left / through / right from a heading change (CCW positive)."""
    deg = math.degrees(wrap(delta_rad))
    if abs(deg) < THROUGH_DEG:
        return "through"
    return "left" if deg > 0.0 else "right"


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
