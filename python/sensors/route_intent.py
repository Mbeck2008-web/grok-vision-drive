"""GPS route intent for the LIVE tab (SPEC M1). Display only.

A route over an offline graph yields ``{turn, next_jct_m}``. ``route_src`` is
derived in-process when that pair is merged into ``nav``. Nothing here is a
path, a stop, or a lane. The planner, actuation, and visualizer must not
import this module. There is no per-tick Navigraph or ``get_road_network``
call: a caller that already has a graph (tests, or a later worker) asks
``intent_from_pose``.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

TURNS = ("left", "right", "straight", "none")
RESULT_KEYS = frozenset({"turn", "next_jct_m"})
THROUGH_DEG = 20.0


def result_message(turn: str, next_jct_m: float | None) -> dict[str, Any]:
    """The only payload that may cross a result queue."""
    label = str(turn) if str(turn) in TURNS else "none"
    dist: float | None
    if next_jct_m is None:
        dist = None
    else:
        dist = float(next_jct_m)
        if not math.isfinite(dist):
            dist = None
    return {"turn": label, "next_jct_m": dist}


def publish(queue: Any, turn: str, next_jct_m: float | None) -> None:
    """Push one schema-checked result. The AST scan requires ``result_message`` here."""
    queue.put(result_message(turn, next_jct_m))


def _wrap(rad: float) -> float:
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def classify_turn(delta_rad: float) -> str:
    """left / straight / right from a heading change (CCW positive)."""
    deg = math.degrees(_wrap(delta_rad))
    if abs(deg) < THROUGH_DEG:
        return "straight"
    return "left" if deg > 0.0 else "right"


@dataclass(frozen=True)
class Graph:
    nodes: dict[str, tuple[float, float]]
    edges: tuple[tuple[str, str, bool], ...]

    @classmethod
    def from_fixture(cls, data: dict[str, Any], level: str) -> "Graph":
        lvl = data["levels"][level]
        nodes = {str(name): (float(xy[0]), float(xy[1])) for name, xy in lvl["nodes"].items()}
        edges = tuple((str(e["a"]), str(e["b"]), bool(e.get("oneway", False))) for e in lvl["edges"])
        return cls(nodes=nodes, edges=edges)

    def degree(self, node: str) -> int:
        return sum(1 for a, b, _oneway in self.edges if a == node or b == node)

    def junctions(self) -> frozenset[str]:
        return frozenset(n for n in self.nodes if self.degree(n) >= 3)


def _shortest_path(graph: Graph, src: str, dst: str) -> list[str] | None:
    if src not in graph.nodes or dst not in graph.nodes:
        return None
    adj: dict[str, list[tuple[str, float]]] = {n: [] for n in graph.nodes}
    for a, b, oneway in graph.edges:
        ax, ay = graph.nodes[a]
        bx, by = graph.nodes[b]
        length = math.hypot(bx - ax, by - ay)
        adj[a].append((b, length))
        if not oneway:
            adj[b].append((a, length))
    dist = {src: 0.0}
    prev: dict[str, str] = {}
    heap = [(0.0, src)]
    while heap:
        d, n = heapq.heappop(heap)
        if n == dst:
            break
        if d > dist.get(n, math.inf):
            continue
        for nxt, w in adj[n]:
            nd = d + w
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                prev[nxt] = n
                heapq.heappush(heap, (nd, nxt))
    if dst not in dist:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def _path_length(graph: Graph, path: list[str]) -> float:
    total = 0.0
    for a, b in zip(path, path[1:]):
        ax, ay = graph.nodes[a]
        bx, by = graph.nodes[b]
        total += math.hypot(bx - ax, by - ay)
    return total


def _project(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> tuple[tuple[float, float], float, float]:
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        proj = (ax, ay)
        return proj, 0.0, math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    qx, qy = ax + t * dx, ay + t * dy
    return (qx, qy), t, math.hypot(px - qx, py - qy)


def _finite_xy(ego_xy: Any) -> tuple[float, float] | None:
    """Two finite metre coords, or None (never raise on NaN / inf / junk)."""
    try:
        px = float(ego_xy[0])
        py = float(ego_xy[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(px) and math.isfinite(py)):
        return None
    return px, py


def _heading_err(a: float, b: float) -> float:
    return abs(_wrap(a - b))


def _snap_edge(
    graph: Graph,
    px: float,
    py: float,
    *,
    heading: float | None = None,
    sticky: tuple[str, str] | None = None,
) -> tuple[str, str, bool, float, float, float, float] | None:
    """The edge ego is actually on.

    Interior projections (``0 < t < 1``) win over endpoint clamps, so a side
    street that only shares the upcoming junction vertex cannot steal the snap
    in the last metre. When several edges still tie (exactly at a node), the
    last snapped edge stays selected (hysteresis) and a travel heading, if
    given, picks the aligned edge.
    """
    scored: list[tuple[float, bool, str, str, bool, float, float, float]] = []
    for a, b, oneway in graph.edges:
        ax, ay = graph.nodes[a]
        bx, by = graph.nodes[b]
        (qx, qy), t, off = _project(px, py, ax, ay, bx, by)
        interior = 0.0 < t < 1.0
        scored.append((off, interior, a, b, oneway, t, qx, qy))
    if not scored:
        return None
    interiors = [item for item in scored if item[1]]
    pool = interiors if interiors else scored
    nearest = min(item[0] for item in pool)
    # Float slack only — never the old 1.0 m window that admitted side streets.
    tight = [item for item in pool if item[0] <= nearest + 1e-3]
    if sticky is not None:
        want = {sticky[0], sticky[1]}
        held = [item for item in tight if {item[2], item[3]} == want]
        if held:
            tight = held
    if heading is not None:
        try:
            travel = float(heading)
        except (TypeError, ValueError):
            travel = None
        else:
            if not math.isfinite(travel):
                travel = None
        if travel is not None:
            def _align(item: tuple[float, bool, str, str, bool, float, float, float]) -> tuple[float, float]:
                ax, ay = graph.nodes[item[2]]
                bx, by = graph.nodes[item[3]]
                edge_h = math.atan2(by - ay, bx - ax)
                return (min(_heading_err(travel, edge_h), _heading_err(travel, edge_h + math.pi)), item[0])

            picked = min(tight, key=_align)
            return picked[2], picked[3], picked[4], picked[5], picked[6], picked[7], picked[0]
    picked = min(tight, key=lambda item: item[0])
    return picked[2], picked[3], picked[4], picked[5], picked[6], picked[7], picked[0]


def intent_from_pose(
    graph: Graph,
    ego_xy: tuple[float, float],
    *,
    goal: str,
    heading: float | None = None,
    sticky: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Snap to the edge ego is on, then the next junction on the path to ``goal``."""
    if goal not in graph.nodes:
        return result_message("none", None)
    xy = _finite_xy(ego_xy)
    if xy is None:
        return result_message("none", None)
    px, py = xy
    snapped = _snap_edge(graph, px, py, heading=heading, sticky=sticky)
    if snapped is None:
        return result_message("none", None)
    a, b, oneway, _t, qx, qy, _off = snapped
    ax, ay = graph.nodes[a]
    bx, by = graph.nodes[b]
    scored: list[tuple[float, str, float, float, list[str]]] = []
    candidates: list[tuple[str, float, float]] = [(b, bx, by)]
    if not oneway:
        candidates.append((a, ax, ay))
    for via, vx, vy in candidates:
        path = _shortest_path(graph, via, goal)
        if path is None:
            continue
        remain = math.hypot(vx - qx, vy - qy)
        incoming = math.atan2(vy - (ay if via == b else by), vx - (ax if via == b else bx))
        scored.append((remain + _path_length(graph, path), via, remain, incoming, path))
    if not scored:
        return result_message("none", None)
    _rest, via, remain, incoming, path = min(scored, key=lambda item: item[0])
    junctions = graph.junctions()
    dist = float(remain)
    jct: str | None = None
    in_heading = float(incoming)
    prev = via
    if via in junctions:
        jct = via
    else:
        for nxt in path[1:]:
            ax, ay = graph.nodes[prev]
            bx, by = graph.nodes[nxt]
            dist += math.hypot(bx - ax, by - ay)
            if nxt in junctions:
                jct = nxt
                in_heading = math.atan2(by - ay, bx - ax)
                break
            prev = nxt
    if jct is None:
        return result_message("none", None)
    try:
        idx = path.index(jct)
    except ValueError:
        return result_message("none", None)
    if idx + 1 >= len(path):
        return result_message("none", dist)
    ox, oy = graph.nodes[path[idx + 1]]
    jx, jy = graph.nodes[jct]
    outgoing = math.atan2(oy - jy, ox - jx)
    return result_message(classify_turn(outgoing - in_heading), dist)


def intent_caption(nav: dict[str, Any] | None) -> str:
    """LIVE-tab words for an active turn. Empty when there is no intent."""
    turn = str((nav or {}).get("turn") or "none")
    if turn not in ("left", "right", "straight"):
        return ""
    bits = [f"intent {turn}"]
    nxt = (nav or {}).get("next_jct_m")
    if nxt is not None:
        try:
            bits.append(f"{float(nxt):.0f}m")
        except (TypeError, ValueError):
            pass
    return " ".join(bits)


class RouteInbox:
    """Last-result reuse when a non-blocking get is empty."""

    def __init__(self) -> None:
        self._last = result_message("none", None)
        self.route_age_ticks = 0

    def observe(self, message: dict[str, Any] | None) -> dict[str, Any]:
        if message is None:
            self.route_age_ticks += 1
            return dict(self._last)
        keys = set(message)
        if keys != RESULT_KEYS:
            extra = sorted(keys - RESULT_KEYS)
            missing = sorted(RESULT_KEYS - keys)
            raise ValueError(f"route intent schema extra={extra} missing={missing}")
        turn = message["turn"]
        if turn not in TURNS:
            raise ValueError("route intent schema: bad turn")
        self._last = result_message(str(turn), message.get("next_jct_m"))
        self.route_age_ticks = 0
        return dict(self._last)


def merge_nav_intent(
    nav: dict[str, Any] | None,
    result: dict[str, Any] | None,
    *,
    route_age_ticks: int = 0,
) -> dict[str, Any]:
    """Copy ``nav`` and attach display-only intent. Never sets ``drive_to_pin``."""
    out = dict(nav or {})
    msg = result_message("none", None) if result is None else result_message(
        str(result.get("turn") or "none"), result.get("next_jct_m")
    )
    if result is not None and set(result) != RESULT_KEYS:
        extra = sorted(set(result) - RESULT_KEYS)
        missing = sorted(RESULT_KEYS - set(result))
        raise ValueError(f"route intent schema extra={extra} missing={missing}")
    out["turn"] = msg["turn"]
    out["next_jct_m"] = msg["next_jct_m"]
    out["route_src"] = "gps" if msg["turn"] != "none" or msg["next_jct_m"] is not None else "none"
    out["route_age_ticks"] = int(route_age_ticks)
    out["drive_to_pin"] = False
    return out
