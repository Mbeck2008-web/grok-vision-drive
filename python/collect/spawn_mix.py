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

A road with a measured width is spawned in the legal direction: node order,
reversed when ``flipDirection`` is set, and shifted into the right-hand lane
(rightmost lane on a one-way road). ``classify_wrong_way`` has to call the
pose ``ok`` on that same road. A ``wrong_way`` or ``twoway_centerline`` result
is not returned; the deck tries a few lanes and stations, then skips the road.
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
from typing import Any, Callable, Mapping

CITY = "city"
HIGHWAY = "highway"
RAMP = "ramp"
KINDS = (CITY, HIGHWAY, RAMP)
TARGET = {CITY: 0.40, HIGHWAY: 0.35, RAMP: 0.25}
SPEED = {CITY: (8.0, 14.0), HIGHWAY: (20.0, 30.0), RAMP: (10.0, 18.0)}


def clamp_speed_range(
    kind_range: tuple[float, float],
    flag_min: float | None,
    flag_max: float | None,
) -> tuple[tuple[float, float], str, bool]:
    """``((lo, hi), src, empty)``: the spawn-type range clamped to ``--speed-min/--speed-max``.

    No flag: the spawn-type range stands (``src`` spawn_type). With a flag the
    range is the intersection (``src`` flag when it narrowed the range). An
    empty intersection uses the flag range itself (``empty`` True); a missing
    bound there is the given one, so ``--speed-max 15`` on a 20-30 highway is 15-15.
    """
    lo, hi = float(min(kind_range)), float(max(kind_range))
    if flag_min is None and flag_max is None:
        return (lo, hi), "spawn_type", False
    f_lo = None if flag_min is None else float(flag_min)
    f_hi = None if flag_max is None else float(flag_max)
    if f_lo is not None and f_hi is not None and f_lo > f_hi:
        f_lo, f_hi = f_hi, f_lo
    a = lo if f_lo is None else max(lo, f_lo)
    b = hi if f_hi is None else min(hi, f_hi)
    if a <= b:
        return (a, b), ("spawn_type" if (a, b) == (lo, hi) else "flag"), False
    a = f_lo if f_lo is not None else float(f_hi)
    b = f_hi if f_hi is not None else float(f_lo)
    return (a, b), "flag", True
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
# Legal-lane search. The first fraction is the historical 40% station.
LEGAL_FRACTIONS = (0.40, 0.30, 0.50, 0.22, 0.62)
LEGAL_ATTEMPTS = 8
LANE_W_M = 3.5
# Half the etk800 body. A pose whose center is closer than this to the decal
# edge hangs the car off the road.
VEHICLE_HALF_M = 0.9
# Stay this far inside the painted edge so the pose is still on the road.
EDGE_MARGIN_M = 0.15
# Past the 0.75 m centerline deadzone, so a later projection does not fall
# back inside it.
LANE_CLEAR_M = 0.08


@dataclass(frozen=True)
class SpawnPick:
    kind: str
    road_id: str
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float]
    speed: tuple[float, float]
    # Share of probes ahead with a painted ego lane. None: not measured.
    paint_cov: float | None = None
    # Set when the pose was checked against the wrong-way classifier.
    flip: bool | None = None
    align: float | None = None
    offset_m: float | None = None
    verdict: str | None = None


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _yaw_quat(dx: float, dy: float) -> tuple[float, float, float, float]:
    half = math.atan2(float(dy), float(dx)) * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "0").strip().lower() in {"1", "true", "yes", "on"}


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
            # LATERAL FRAME: only whether a lane exists is used; frame-insensitive.
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


def _mix_width(left: float | None, right: float | None, mix: float) -> float | None:
    if left is None and right is None:
        return None
    if left is None:
        return float(right)  # type: ignore[arg-type]
    if right is None:
        return float(left)
    return float(left) + (float(right) - float(left)) * mix


def _sample_center(
    points: list[tuple[float, float, float]],
    fraction: float,
    widths: list[float | None] | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float], float | None] | None:
    """``(center xyz, node-order tangent, local width)`` at ``fraction`` along ``points``."""
    paired: list[float | None]
    if widths is not None and len(widths) == len(points):
        paired = list(widths)
    else:
        paired = [None] * len(points)
    segs: list[tuple[tuple[float, float, float], tuple[float, float, float], float, float | None, float | None]] = []
    length = 0.0
    for (left, left_w), (right, right_w) in zip(zip(points, paired), zip(points[1:], paired[1:])):
        step = math.hypot(right[0] - left[0], right[1] - left[1])
        if step < 1e-3:
            continue
        segs.append((left, right, step, left_w, right_w))
        length += step
    if not segs or length < 1.0:
        return None
    target = min(length * float(fraction), max(0.0, length - 1.0))
    walked = 0.0
    left, right, step, left_w, right_w = segs[-1]
    for left, right, step, left_w, right_w in segs:
        if walked + step >= target:
            break
        walked += step
    mix = 0.0 if step < 1e-6 else min(1.0, max(0.0, (target - walked) / step))
    center = tuple(left[k] + (right[k] - left[k]) * mix for k in range(3))
    norm = step or 1.0
    return center, ((right[0] - left[0]) / norm, (right[1] - left[1]) / norm), _mix_width(left_w, right_w, mix)


def _centers_and_widths(road: Mapping[str, Any]) -> tuple[list[tuple[float, float, float]], list[float | None]]:
    """Centerline points and the left/right span at each point. Missing edges are None."""
    points: list[tuple[float, float, float]] = []
    widths: list[float | None] = []
    for edge in road.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        middle = edge.get("middle")
        if not (isinstance(middle, (list, tuple)) and len(middle) >= 3):
            continue
        points.append((float(middle[0]), float(middle[1]), float(middle[2])))
        left, right = edge.get("left"), edge.get("right")
        span: float | None = None
        if (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) >= 2
            and len(right) >= 2
        ):
            got = math.hypot(float(right[0]) - float(left[0]), float(right[1]) - float(left[1]))
            if got >= 0.5:
                span = got
        widths.append(span)
    return points, widths


def _median_width(widths: list[float | None]) -> float | None:
    spans = [float(item) for item in widths if item is not None and float(item) >= 0.5]
    if not spans:
        return None
    spans.sort()
    return spans[len(spans) // 2]


def _width_of_edges(edges: Any) -> float | None:
    spans: list[float] = []
    if not isinstance(edges, list):
        return None
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        left, right = edge.get("left"), edge.get("right")
        if not (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) >= 2
            and len(right) >= 2
        ):
            continue
        span = math.hypot(float(right[0]) - float(left[0]), float(right[1]) - float(left[1]))
        if span >= 0.5:
            spans.append(span)
    if not spans:
        return None
    spans.sort()
    return spans[len(spans) // 2]


def _explicit_lanes(road: Mapping[str, Any] | dict[str, Any]) -> tuple[int, int] | None:
    if isinstance(road, dict):
        keys = road.keys()
    else:
        return None
    if not any(key in keys for key in ("lanes_left", "lanes_right", "lanesLeft", "lanesRight")):
        return None
    left = road["lanes_left"] if "lanes_left" in road else road.get("lanesLeft")
    right = road["lanes_right"] if "lanes_right" in road else road.get("lanesRight")
    return int(_num(left)), int(_num(right))


def _effective_oneway(road: dict[str, Any]) -> bool:
    """Same rule as ``classify_wrong_way``: oneWay, or no left lanes and some right lanes."""
    if bool(road.get("oneway")):
        return True
    counts = _explicit_lanes(road)
    if counts is None:
        return False
    left, right = counts
    return left == 0 and right >= 1


def _derived_lanes(width: float) -> int:
    return max(1, min(8, int(round(float(width) / LANE_W_M))))


def _auto_lanes(road: dict[str, Any]) -> bool:
    """BeamNG ignores ``lanesLeft`` / ``lanesRight`` when ``autoLanes`` is set."""
    if "auto_lanes" in road:
        return bool(road["auto_lanes"])
    return _flag(road.get("autoLanes"))


def _tagged_lanes(road: dict[str, Any]) -> int:
    counts = _explicit_lanes(road)
    if counts is not None and sum(counts) >= 1:
        return sum(counts)
    return int(_num(road.get("lanes")))


def _lane_count(road: dict[str, Any], width: float) -> int:
    """Lane count used for the right-hand offset.

    ``autoLanes`` drops the tags. A tag that underfills the decal (fewer lanes
    than the width holds at ``LANE_W_M``) is replaced by that width count too.
    Highway/ramp classification keeps the stored tag on ``road["lanes"]``.
    """
    derived = _derived_lanes(width)
    tagged = _tagged_lanes(road)
    if _auto_lanes(road) or tagged < 1 or tagged < derived:
        return derived
    return tagged


def body_inside(offset_m: float, local_width: float | None) -> bool:
    """False when the car body (``VEHICLE_HALF_M``) crosses the local decal edge."""
    if local_width is None:
        return True
    try:
        half = float(local_width) * 0.5
    except (TypeError, ValueError):
        return True
    if not math.isfinite(half) or half <= 0.0:
        return True
    return abs(float(offset_m)) + VEHICLE_HALF_M <= half + 1e-6


def _station_width(local: float | None, road: dict[str, Any]) -> float | None:
    if local is not None:
        try:
            value = float(local)
        except (TypeError, ValueError):
            value = None
        else:
            if math.isfinite(value) and value > 0.0:
                return value
    fallback = road.get("width")
    try:
        value = float(fallback)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def _offset_candidates(road: dict[str, Any], width: float | None = None) -> list[float]:
    """Metres to the right of the legal tangent, rightmost lane first.

    ``width`` is the decal width at the station. Lane tags that underfill that
    width, and ``autoLanes``, are counted from it.
    """
    from python.train.frame_quality import CENTER_DEADZONE_M

    width = float(road["width"] if width is None else width)
    count = _lane_count(road, width)
    lane_w = width / count
    centers = [-width / 2.0 + (index + 0.5) * lane_w for index in range(count)]
    cap = max(0.0, width / 2.0 - EDGE_MARGIN_M)
    if _effective_oneway(road):
        raw = [min(max(0.0, center), cap) for center in centers if center >= -1e-6]
        if not raw:
            raw = [0.0]
    else:
        clear = CENTER_DEADZONE_M + LANE_CLEAR_M
        raw = [min(center, cap) for center in centers if clear - 1e-9 <= center <= cap + 1e-6]
        if not raw and cap >= clear:
            raw = [min(clear, cap)]
    ordered: list[float] = []
    for center in sorted(raw, reverse=True):
        if not ordered or abs(ordered[-1] - center) > 0.05:
            ordered.append(center)
    return ordered


def _pose_shifted(
    center: tuple[float, float, float],
    node_tangent: tuple[float, float],
    flip: bool,
    offset_m: float,
    z_lift: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float], tuple[float, float]]:
    nx, ny = node_tangent
    if flip:
        nx, ny = -nx, -ny
    rx, ry = ny, -nx
    pos = (center[0] + rx * offset_m, center[1] + ry * offset_m, center[2] + z_lift)
    return pos, _yaw_quat(nx, ny), (nx, ny)


def _fmt_num(value: Any, spec: str) -> str:
    if value is None:
        return "na"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "na"
    if not math.isfinite(number):
        return "na"
    return format(number, spec)


def format_spawn_line(
    prefix: str,
    kind: str,
    road_id: str,
    flip: bool,
    align: Any,
    offset_m: Any,
    verdict: str,
    reason: str = "",
    road_mismatch: str = "",
) -> str:
    """One spawn line. ``verdict`` and ``reason`` stay separate fields.

    ``road_mismatch`` is the classifier's road id when it is not ``road_id``.
    """
    line = (
        f"{prefix} road={road_id} kind={kind} flipDirection={1 if flip else 0} "
        f"align={_fmt_num(align, '.3f')} offset_m={_fmt_num(offset_m, '.2f')} "
        f"verdict={verdict or 'na'} reason={reason or 'na'}"
    )
    if road_mismatch:
        line += f" road_mismatch={road_mismatch}"
    return line


def reject_spawn_line(kind: str, road_id: str, flip: bool, last: Mapping[str, Any] | None) -> str:
    """Skip line: ``verdict=`` and ``reason=`` are separate, plus a road mismatch."""
    logged = str(road_id)
    verdict = "rejected"
    reason = "no_candidate"
    mismatch = ""
    align = offset = None
    if last:
        if last.get("attempted_id"):
            logged = str(last["attempted_id"])
        verdict = str(last.get("verdict") or "rejected")
        reason = str(last.get("reason") or "na")
        align = last.get("align")
        offset = last.get("offset_m")
        got = str(last.get("road_id") or "")
        if got and got != logged:
            mismatch = got
    return (
        format_spawn_line("spawn reject", kind, logged, bool(flip), align, offset, verdict, reason, mismatch)
        + " (skipped)"
    )


def _quality_from_road(road: dict[str, Any]) -> Any:
    from python.train.frame_quality import QualityRoad

    points = road.get("pts") or []
    if len(points) < 2:
        return None
    try:
        width = float(road.get("width"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(width) or width < 0.5:
        return None
    counts = _explicit_lanes(road)
    left = counts[0] if counts is not None else 0
    right = counts[1] if counts is not None else 0
    lanes = left + right if left + right >= 1 else int(_num(road.get("lanes")))
    return QualityRoad(
        road_id=str(road.get("id")),
        xy=[(float(point[0]), float(point[1])) for point in points],
        width_m=width,
        oneway=bool(road.get("oneway")),
        lanes_left=left,
        lanes_right=right,
        lanes=max(0, lanes),
        drivability=1.0,
        flip=bool(road.get("flip")),
        z=[float(point[2]) for point in points],
    )


def _pose_accepted(got: Mapping[str, Any], road_id: str) -> bool:
    from python.train.frame_quality import spawn_pose_rejected

    if str(got.get("verdict") or "") != "ok":
        return False
    if str(got.get("road_id")) != str(road_id):
        return False
    return not spawn_pose_rejected(got)


def seek_legal_pose(
    road: dict[str, Any],
    quality: list[Any],
    grid: dict[Any, Any] | None,
    *,
    fractions: tuple[float, ...] = LEGAL_FRACTIONS,
    attempts: int | None = None,
    z_lift: float = 0.5,
    classify_z: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """A legal right-lane pose, or ``(None, last classification)``.

    Roads with no measured width keep the old centerline pose and are not
    classified (callers that only stored a centerline have nothing to offset).
    """
    from python.train.frame_quality import classify_wrong_way

    if attempts is None:
        attempts = LEGAL_ATTEMPTS
    points = road.get("pts") or []
    if not road.get("width"):
        pose = _pose_at(points)
        if pose is None:
            return None, None
        pos, rot = pose
        return {
            "pos": pos,
            "rot": rot,
            "flip": False,
            "offset_m": None,
            "align": None,
            "verdict": None,
            "reason": "",
            "legacy": True,
        }, None
    flip = bool(road.get("flip"))
    road_id = str(road.get("id"))
    if not quality:
        return None, {"verdict": "rejected", "reason": "no_roads", "road_id": road_id}
    if road.get("width") is None:
        return None, {"verdict": "rejected", "reason": "no_candidate", "road_id": road_id}
    last: dict[str, Any] | None = None
    tried = 0
    past_edge = False
    for fraction in fractions:
        sampled = _sample_center(points, fraction, road.get("widths"))
        if sampled is None:
            continue
        center, tangent, local = sampled
        station = _station_width(local, road)
        offsets = _offset_candidates(road, station)
        for offset in offsets:
            if not body_inside(offset, station):
                past_edge = True
                continue
            if tried >= int(attempts):
                return None, last
            tried += 1
            pos, rot, heading = _pose_shifted(center, tangent, flip, offset, z_lift)
            yaw = math.atan2(heading[1], heading[0])
            got = classify_wrong_way(
                pos[0], pos[1], yaw, quality, grid, z=pos[2] if classify_z else None
            )
            last = dict(got)
            last["attempted_id"] = road_id
            last["flip"] = flip
            if _pose_accepted(got, road_id):
                return {
                    "pos": pos,
                    "rot": rot,
                    "flip": flip,
                    "offset_m": float(got["offset_m"]),
                    "align": float(got["align"]),
                    "verdict": str(got.get("verdict") or "ok"),
                    "reason": str(got.get("reason") or ""),
                    "legacy": False,
                }, got
    if last is None:
        last = {
            "verdict": "rejected",
            "reason": "past_edge" if past_edge else "no_candidate",
            "road_id": road_id,
        }
    return None, last


def _project_station(
    points: list[tuple[float, float, float]],
    x: float,
    y: float,
    widths: list[float | None] | None = None,
) -> tuple[float, tuple[float, float, float], tuple[float, float], float | None] | None:
    """``(distance, center, node tangent, local width)`` of the closest centerline point."""
    paired: list[float | None]
    if widths is not None and len(widths) == len(points):
        paired = list(widths)
    else:
        paired = [None] * len(points)
    best: tuple[float, tuple[float, float, float], tuple[float, float], float | None] | None = None
    for (left, left_w), (right, right_w) in zip(zip(points, paired), zip(points[1:], paired[1:])):
        dx, dy = right[0] - left[0], right[1] - left[1]
        length2 = dx * dx + dy * dy
        if length2 < 1e-6:
            continue
        tee = ((x - left[0]) * dx + (y - left[1]) * dy) / length2
        tee = 0.0 if tee < 0.0 else 1.0 if tee > 1.0 else tee
        center = tuple(left[k] + (right[k] - left[k]) * tee for k in range(3))
        dist = math.hypot(x - center[0], y - center[1])
        norm = math.sqrt(length2)
        tangent = (dx / norm, dy / norm)
        local = _mix_width(left_w, right_w, tee)
        if best is None or dist < best[0]:
            best = (dist, center, tangent, local)
    return best


def legal_pose_near(
    roads: list[dict[str, Any]],
    x: float,
    y: float,
    z: float | None = None,
    *,
    max_dist: float = 40.0,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Legal right-lane pose on the centerline nearest ``(x, y)``, within ``max_dist``.

    ``z`` is ignored for the search. The returned height is the node height
    plus half a metre, which is what a recovery snap uses.
    """
    del z
    from python.train.frame_quality import build_road_grid, classify_wrong_way

    ranked: list[tuple[float, dict[str, Any], tuple[float, float, float], tuple[float, float], float | None]] = []
    for road in roads:
        hit = _project_station(road.get("pts") or [], x, y, road.get("widths"))
        if hit is None or hit[0] > float(max_dist):
            continue
        ranked.append((hit[0], road, hit[1], hit[2], hit[3]))
    ranked.sort(key=lambda item: item[0])
    if not ranked:
        return None, None
    nearest = ranked[0]
    if not nearest[1].get("width"):
        center, tangent = nearest[2], nearest[3]
        pos = (center[0], center[1], center[2] + 0.5)
        return {
            "pos": pos,
            "rot": _yaw_quat(tangent[0], tangent[1]),
            "flip": False,
            "offset_m": None,
            "align": None,
            "verdict": None,
            "reason": "",
            "legacy": True,
            "road_id": str(nearest[1].get("id")),
        }, None
    quality = [item for item in (_quality_from_road(road) for _d, road, _c, _t, _w in ranked) if item is not None]
    grid = build_road_grid(quality) if quality else {}
    last: dict[str, Any] | None = None
    tried = 0
    past_edge = False
    for _dist, road, center, tangent, local in ranked[:4]:
        if not road.get("width"):
            continue
        flip = bool(road.get("flip"))
        station = _station_width(local, road)
        for offset in _offset_candidates(road, station):
            if not body_inside(offset, station):
                past_edge = True
                continue
            if tried >= LEGAL_ATTEMPTS:
                return None, last
            tried += 1
            pos, rot, heading = _pose_shifted(center, tangent, flip, offset, 0.5)
            got = classify_wrong_way(
                pos[0], pos[1], math.atan2(heading[1], heading[0]), quality, grid, z=pos[2]
            )
            last = dict(got)
            last["attempted_id"] = str(road.get("id"))
            last["flip"] = flip
            if _pose_accepted(got, str(road.get("id"))):
                return {
                    "pos": pos,
                    "rot": rot,
                    "flip": flip,
                    "offset_m": float(got["offset_m"]),
                    "align": float(got["align"]),
                    "verdict": str(got.get("verdict") or "ok"),
                    "reason": str(got.get("reason") or ""),
                    "legacy": False,
                    "road_id": str(road.get("id")),
                }, got
    if last is None:
        last = {
            "verdict": "rejected",
            "reason": "past_edge" if past_edge else "no_candidate",
            "road_id": str(nearest[1].get("id")),
        }
    return None, last


def view_from_network_road(road_id: str, road: Any) -> dict[str, Any] | None:
    """One spawn-road dict from a ``get_road_network`` value (dict or edge list)."""
    if isinstance(road, list):
        road = {"edges": road}
    if not isinstance(road, dict):
        return None
    try:
        if float(road.get("drivability", road.get("drivable", 1)) or 0) <= 0.0:
            return None
    except (TypeError, ValueError):
        pass
    points, widths = _centers_and_widths(road)
    if len(points) < 2:
        return None
    counts = _explicit_lanes(road)
    lanes = (sum(counts) if counts is not None else 0) or int(_num(road.get("lanes")))
    width_f = _median_width(widths)
    if width_f is None:
        raw_width = road.get("width")
        try:
            width_f = float(raw_width) if raw_width is not None else None
        except (TypeError, ValueError):
            width_f = None
    return {
        "id": str(road_id),
        "pts": points,
        "widths": widths,
        "lanes": lanes,
        "lanes_left": counts[0] if counts is not None else 0,
        "lanes_right": counts[1] if counts is not None else 0,
        "length": _num(road.get("debugRoadLength")) or _polyline_m(points),
        "oneway": _flag(road.get("oneWay", road.get("oneway"))),
        "flip": _flag(road.get("flipDirection", road.get("flip"))),
        "auto_lanes": _flag(road.get("autoLanes", road.get("auto_lanes"))),
        "width": width_f,
    }


def views_from_network(spec: Any) -> list[dict[str, Any]]:
    """Spawn roads from a road-network payload, keeping flip and lane counts."""
    if isinstance(spec, dict) and isinstance(spec.get("roads"), dict):
        spec = spec["roads"]
    elif isinstance(spec, dict) and isinstance(spec.get("road_network"), dict):
        spec = spec["road_network"]
    if not isinstance(spec, dict):
        return []
    views: list[dict[str, Any]] = []
    for road_id, road in spec.items():
        view = view_from_network_road(str(road_id), road)
        if view is not None:
            views.append(view)
    return views


def index_spawn_roads(views: list[dict[str, Any]]) -> tuple[list[Any], dict[Any, Any]]:
    """Quality roads for ``classify_wrong_way``, in the same order as ``views``."""
    from python.train.frame_quality import build_road_grid

    built = [item for item in (_quality_from_road(view) for view in views) if item is not None]
    return built, (build_road_grid(built) if built else {})


def view_from_poly(road: Any) -> dict[str, Any] | None:
    """A ``RoadPoly`` or any object with ``xy`` (and optional lane fields)."""
    xy = getattr(road, "xy", None)
    if not xy or len(xy) < 2:
        return None
    heights = getattr(road, "z", None) or []
    points: list[tuple[float, float, float]] = []
    for index, pair in enumerate(xy):
        height = 0.0
        if heights and index < len(heights) and heights[index] is not None:
            try:
                height = float(heights[index])
            except (TypeError, ValueError):
                height = 0.0
        points.append((float(pair[0]), float(pair[1]), height))
    width = getattr(road, "width_m", None)
    try:
        width_f = float(width) if width is not None else None
    except (TypeError, ValueError):
        width_f = None
    has_sides = getattr(road, "lanes_left", None) is not None or getattr(road, "lanes_right", None) is not None
    left = int(getattr(road, "lanes_left", 0) or 0) if has_sides else 0
    right = int(getattr(road, "lanes_right", 0) or 0) if has_sides else 0
    lanes = int(getattr(road, "lanes", 0) or 0)
    view = {
        "id": str(getattr(road, "road_id", None) or getattr(road, "id", None) or ""),
        "pts": points,
        "lanes": lanes or (left + right),
        "length": _polyline_m(points),
        "oneway": bool(getattr(road, "oneway", False)),
        "flip": bool(getattr(road, "flip", False)),
        "width": width_f if width_f and width_f >= 0.5 else None,
    }
    if has_sides:
        view["lanes_left"] = left
        view["lanes_right"] = right
    return view


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
        points, widths = _centers_and_widths(road)
        if len(points) < 2:
            continue
        lanes_left = int(_num(road.get("lanesLeft")))
        lanes_right = int(_num(road.get("lanesRight")))
        lanes = lanes_left + lanes_right
        length = _num(road.get("debugRoadLength"))
        drivability = _num(road.get("drivability"), -1.0)
        if drivability < 0.3 or length < 12.0 or lanes < 1:
            continue
        oneway = str(road.get("oneWay") or "0").lower() in {"1", "true"}
        roads.append(
            {
                "id": str(road_id),
                "pts": points,
                "widths": widths,
                "lanes": lanes,
                "lanes_left": lanes_left,
                "lanes_right": lanes_right,
                "length": length,
                "oneway": oneway,
                "flip": _flag(road.get("flipDirection")),
                "auto_lanes": _flag(road.get("autoLanes")),
                "width": _median_width(widths),
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
        self._quality: list[Any] | None = None
        self._quality_grid: dict[Any, Any] | None = None
        self._quality_ready = False
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
        deck = cls(classify_roads(load_nav_roads(path)), rng, **kwargs)
        deck._bind_dump(path)
        return deck

    def _bind_dump(self, path: Path | str) -> None:
        """Classify poses with the same road dump the offline filter uses."""
        from python.train.frame_quality import build_road_grid, load_quality_roads

        try:
            roads = load_quality_roads(path)
        except Exception:
            return
        if not roads:
            return
        self._quality = roads
        self._quality_grid = build_road_grid(roads)
        self._quality_ready = True

    def _ensure_quality(self) -> tuple[list[Any], dict[Any, Any]]:
        if not self._quality_ready:
            built = [item for item in (_quality_from_road(road) for road in self._by_id.values()) if item is not None]
            from python.train.frame_quality import build_road_grid

            self._quality = built
            self._quality_grid = build_road_grid(built) if built else {}
            self._quality_ready = True
        return self._quality or [], self._quality_grid or {}

    def _legal_for(self, kind: str, road: dict[str, Any]) -> dict[str, Any] | None:
        from python.train.frame_quality import classify_wrong_way

        quality, grid = self._ensure_quality()
        found, last = seek_legal_pose(road, quality, grid)
        if found is not None and found.get("legacy") and quality:
            pos, rot = found["pos"], found["rot"]
            _qx, _qy, qz, qw = rot
            yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
            last = classify_wrong_way(pos[0], pos[1], yaw, quality, grid, z=pos[2])
            if _pose_accepted(last, str(road.get("id"))):
                found = {
                    "pos": pos,
                    "rot": rot,
                    "flip": bool(road.get("flip")),
                    "offset_m": float(last["offset_m"]),
                    "align": float(last["align"]),
                    "verdict": str(last.get("verdict") or "ok"),
                    "reason": str(last.get("reason") or ""),
                    "legacy": False,
                }
            else:
                found = None
        if found is not None:
            return found
        if not road.get("width") and not quality:
            return None
        self._say(reject_spawn_line(kind, str(road.get("id")), bool(road.get("flip")), last))
        return None

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

    def _pick(
        self, kind: str, road: dict[str, Any], info: dict[str, Any], cov: float | None, wanted: str
    ) -> SpawnPick:
        self.counts[kind] += 1
        rid = str(road["id"])
        self.road_picks[rid] = self.road_picks.get(rid, 0) + 1
        self.road_kind[rid] = kind
        if kind != wanted:
            self.cross_kind += 1
        if not info.get("legacy"):
            self._say(
                format_spawn_line(
                    "spawn legal",
                    kind,
                    rid,
                    bool(info.get("flip")),
                    info.get("align"),
                    info.get("offset_m"),
                    str(info.get("verdict") or "ok"),
                    str(info.get("reason") or ""),
                )
            )
        return SpawnPick(
            kind,
            road["id"],
            info["pos"],
            info["rot"],
            SPEED[kind],
            cov,
            None if info.get("legacy") else bool(info.get("flip")),
            None if info.get("legacy") else info.get("align"),
            None if info.get("legacy") else info.get("offset_m"),
            None if info.get("legacy") else info.get("verdict"),
        )

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
            shift = 0.0 if pick.offset_m is None else float(pick.offset_m)
            rx, ry = fy, -fx
            x += rx * shift
            y += ry * shift
            if shift and not self._slot_is_legal(x, y, z + 0.5, fx, fy):
                notes.append(f"traffic {where} skipped: not in a legal lane")
                continue
            slots.append((where, (x, y, z + 0.5), _yaw_quat(fx, fy), along))
        return slots, notes

    def _slot_is_legal(self, x: float, y: float, z: float, fx: float, fy: float) -> bool:
        """Companion cars use the ego's lane shift, and are dropped when that is rejected."""
        from python.train.frame_quality import classify_wrong_way, spawn_pose_rejected

        quality, grid = self._ensure_quality()
        if not quality:
            return True
        got = classify_wrong_way(x, y, math.atan2(fy, fx), quality, grid, z=z)
        if spawn_pose_rejected(got):
            return False
        return str(got.get("verdict") or "") == "ok"

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
        tried: list[tuple[float, str, dict[str, Any], dict[str, Any]]] = []
        for kind in order:
            budget = self.max_tries
            for road in [item for item in self.pools[kind] if item["id"] not in self.used]:
                self.used.add(road["id"])
                info = self._legal_for(kind, road)
                if info is None:
                    continue
                if not filtering:
                    return self._pick(kind, road, info, None, first)
                cov = self.coverage(road)
                if cov >= self.min_paint_cov:
                    self._say(
                        f"spawn pick {kind} road={road['id']} paint_cov={cov:.2f} "
                        f"(>={self.min_paint_cov:.2f}, skipped {len(tried)})"
                    )
                    return self._pick(kind, road, info, cov, first)
                self.skipped += 1
                tried.append((cov, kind, road, info))
                self._say(
                    f"spawn skip {kind} road={road['id']} paint_cov={cov:.2f} (<{self.min_paint_cov:.2f})"
                )
                budget -= 1
                if budget <= 0:
                    break
        if not tried:
            return None
        cov, kind, road, info = max(tried, key=lambda item: item[0])
        self._say(
            f"spawn fallback {kind} road={road['id']} paint_cov={cov:.2f} "
            f"(no road >= {self.min_paint_cov:.2f} in {len(tried)} tries)"
        )
        self.best_seen += 1
        return self._pick(kind, road, info, cov, first)

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

