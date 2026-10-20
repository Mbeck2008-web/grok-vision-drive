#!/usr/bin/env python3
"""A weave line walked across road-piece joins: no repeated tip, no step at the join.

``_chain`` walks a road and its colinear continuations. Each continuation
piece starts at the previous tip again (s = 0 on that piece). Kept, that is a
second sample at the same s with the new piece's heading, a few degrees off.
Offset by the lane center it becomes a step of offset * dheading over 0 m of
road, which ``path_lat_acc`` read as 13-38 m/s^2 (rec5b refusals).

Tolerances:
* s strictly increases: every s_j - s_{j-1} > 1e-6 m.
* no duplicated sample: no two centerline samples within 1e-6 m of each other.
* the sent line is continuous at every join: consecutive sent points move at
  most 1.5 m per metre of s, plus 0.01 m (|dp| <= 1.5 * ds + 0.01). A
  repeated tip has ds = 0, so a 1 cm step fails; at a 1.75 m offset and a 3 deg
  join the old step was about 9 cm. The signed lateral offset of each sent
  point from the planned lane center stays within 0.02 m of the sine.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect import excursion as exc  # noqa: E402
from python.collect.excursion import Excursion, excursion_path, path_lat_acc  # noqa: E402
from python.debug.gt_geom import RoadPoly, project_road  # noqa: E402

HW = 0.9
S_EPS = 1e-6
DUP_M = 1e-6
STEP_K = 1.5
STEP_TOL_M = 0.01
LAT_TOL_M = 0.02


def _piece(road_id: str, x0: float, y0: float, heading: float, length: float, step: float = 10.0) -> RoadPoly:
    n = int(round(length / step))
    pts = [(x0 + math.cos(heading) * step * k, y0 + math.sin(heading) * step * k) for k in range(n + 1)]
    return RoadPoly(road_id, pts, 7.0, 2)


def _roads(kinks_deg: tuple[float, ...], lengths: tuple[float, ...], side_gap_m: tuple[float, ...]) -> list[RoadPoly]:
    """Pieces end to start. Piece k+1 turns kinks_deg[k] at the join and starts side_gap_m[k] to the right."""
    heading = math.pi / 2.0
    x, y = 0.0, 0.0
    roads = []
    for k, length in enumerate(lengths):
        roads.append(_piece(f"p{k}", x, y, heading, length))
        x, y = roads[-1].xy[-1]
        if k < len(kinks_deg):
            # Right of a heading h is (sin h, -cos h).
            x += math.sin(heading) * side_gap_m[k]
            y -= math.cos(heading) * side_gap_m[k]
            heading += math.radians(kinks_deg[k])
    return roads


def _check(name: str, roads: list[RoadPoly], speed: float) -> str:
    x, y, heading = 1.75, 6.0, math.pi / 2.0
    hit = project_road(roads, x, y, heading)
    assert hit is not None and hit.road_id == "p0", hit
    road = roads[0]
    need = exc.weave_length_m(speed)
    _length, _fork, _span, samples = exc._chain(roads, road, x, y, heading, need, exc.PATH_STEP_M)
    # The car is 1.75 m right of p0 (x = 0), so its centerline point is (0, y).
    tip_s = math.hypot(road.xy[-1][0], road.xy[-1][1] - y)
    assert samples[-1][4] > tip_s + exc.PATH_STEP_M, (name, "chain must cross a join", tip_s, samples[-1][4])

    s = [p[4] for p in samples]
    bad_s = [(j, s[j - 1], s[j]) for j in range(1, len(s)) if not s[j] - s[j - 1] > S_EPS]
    assert not bad_s, (name, "s not strictly increasing", bad_s)
    dups = [(i, j) for i in range(len(samples)) for j in range(i + 1, len(samples))
            if math.hypot(samples[i][0] - samples[j][0], samples[i][1] - samples[j][1]) <= DUP_M]
    assert not dups, (name, "duplicated centerline sample", dups)

    info: dict = {}
    pts, _d = excursion_path(roads, x, y, heading, Excursion("weave", 0.0, 0.0, 1, ""), hit, speed,
                             paint_lane=(-1.75, 1.75), half_width_m=HW, info=info)
    assert pts is not None, (name, "weave refused", info)
    assert len(pts) == len(samples), (name, len(pts), len(samples))
    amp, wave = info["amp_m"], info["wave_m"]
    worst_step = -math.inf
    for j in range(1, len(pts)):
        ds = s[j] - s[j - 1]
        dp = math.hypot(pts[j][0] - pts[j - 1][0], pts[j][1] - pts[j - 1][1])
        worst_step = max(worst_step, dp - STEP_K * ds)
        assert dp <= STEP_K * ds + STEP_TOL_M, (name, "sent line steps", j, s[j], ds, dp)
    for (cx, cy, fx, fy, s_m), (px, py) in zip(samples, pts):
        lat = (px - cx) * fy - (py - cy) * fx - 1.75
        want = amp * math.sin(2.0 * math.pi * s_m / wave)
        assert abs(lat - want) <= LAT_TOL_M, (name, "lateral offset off the sine", s_m, lat, want)
    centerline = [(p[0], p[1]) for p in samples]
    acc, _total = path_lat_acc(pts, centerline, speed)
    assert acc <= exc.WEAVE_LAT_ACC_MPS2 + 1e-9, (name, acc)
    return (f"{name}: {len(samples)} samples over {s[-1]:.1f} m, joins crossed, amp={amp:.2f} m "
            f"lat_acc={acc:.2f} m/s2, max(dp - 1.5 ds)={worst_step:+.2f} m")


def main() -> None:
    out = [
        # Two pieces, 3 deg left at the join, ends meet exactly.
        _check("two pieces, 3 deg left", _roads((3.0,), (30.0, 200.0), (0.0,)), 8.0),
        # Three pieces, 2.5 deg right then 3 deg left, the second join 5 cm to the side.
        _check("three pieces, -2.5 / +3 deg, 5 cm side gap", _roads((-2.5, 3.0), (30.0, 20.0, 200.0), (0.0, 0.05)), 8.0),
        # 11 m/s: 55 m line, two joins inside it.
        _check("three pieces at 11 m/s", _roads((3.0, -3.0), (20.0, 20.0, 200.0), (0.0, 0.0)), 11.0),
    ]
    for line in out:
        print(line)
    print("collect chain join ok")


if __name__ == "__main__":
    main()
