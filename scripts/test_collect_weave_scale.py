#!/usr/bin/env python3
"""Speed-compensated weaves: wavelength, amplitude that fits the painted lane,
sent-path lateral accel, the painted body-edge abort, vehicle width, run-up."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect import excursion as exc  # noqa: E402
from python.collect.excursion import (  # noqa: E402
    EDGE_M,
    WEAVE_SLACK_M,
    Excursion,
    ExcursionClock,
    body_past_paint_edge,
    excursion_path,
    gate_reason,
    path_lat_acc,
    targets_for,
    unsafe_reason,
    weave_amp_cap_m,
    weave_lat_acc,
    weave_length_m,
    weave_plan,
    weave_wave_m,
)
from python.collect.liveness import PRE_IMPACT_S, ImpactWatch  # noqa: E402
from python.debug.gt_geom import RoadPoly, project_road  # noqa: E402
from python.debug.painted_lanes import PaintedLanes  # noqa: E402
from scripts.collect_scene_lanes import bbox_width_m, vehicle_half_width  # noqa: E402

HW = 0.9


def _paint(xs, y0=-100.0, y1=600.0) -> PaintedLanes:
    ys = [y0 + 10.0 * k for k in range(int((y1 - y0) / 10.0) + 1)]
    return PaintedLanes([{"material": "line_white", "nodes": [[x, y, 0.4, 0.15] for y in ys]} for x in xs])


def _straight(width=7.0, lanes=0):
    return [RoadPoly("r", [(0.0, 0.0), (0.0, 1000.0)], width, lanes)]


def check_scaling() -> None:
    assert exc.WEAVE_PERIOD_S == 5.0
    assert weave_wave_m(None) == 40.0 and weave_wave_m(8.0) == 40.0 and weave_wave_m(10.0) == 50.0
    assert weave_wave_m(25.0) == 125.0 and weave_wave_m(-33.0) == 165.0
    assert weave_amp_cap_m(10.0) == 0.8 and abs(weave_amp_cap_m(16.0) - 0.575) < 1e-9 and weave_amp_cap_m(22.0) == 0.35
    caps = [weave_amp_cap_m(v / 2.0) for v in range(0, 80)]
    assert all(a >= b - 1e-12 for a, b in zip(caps, caps[1:]))
    # 3.5 m painted lane, 0.9 m half width: room = 1.75 - 0.9 - 0.3 = 0.55.
    for v, amp in ((8.0, 0.55), (10.0, 0.55), (16.0, 0.55), (19.0, weave_amp_cap_m(19.0)), (22.0, 0.35), (33.0, 0.35)):
        spec = targets_for("weave", 1, 3.5, road_width_m=7.0, speed_mps=v, weave_lane_w=3.5, half_width_m=HW)
        assert spec is not None and abs(spec[0] - amp) < 1e-9, (v, spec, amp)
    print("weave scaling: wave=max(40,5v), cap 0.8@<=10 -> 0.35@>=22 m/s, highways still weave OK")


def check_amp_fits_lane() -> None:
    assert WEAVE_SLACK_M == 0.3
    # Cap = lane_w/2 - half_width - 0.3; refused under 0.3 m.
    for lane_w, hw, want in ((3.0, 0.9, 0.3), (2.99, 0.9, None), (3.2, 1.0, 0.3), (3.19, 1.0, None),
                             (3.4, 0.9, 0.5), (4.0, 0.9, 0.6), (5.0, 0.9, 0.6)):
        spec = targets_for("weave", 1, 9.9, road_width_m=14.0, lanes=2, speed_mps=8.0, weave_lane_w=lane_w, half_width_m=hw)
        if want is None:
            assert spec is None, (lane_w, hw, spec)
        else:
            assert spec is not None and abs(spec[0] - want) < 1e-9, (lane_w, hw, spec)
    # The painted width is used, not the navgraph's 7 m "lane" of a 14 m 1/1 road.
    roads = _straight(14.0, 2)
    hit = project_road(roads, 1.75, 20.0, math.pi / 2.0)
    assert abs(hit.lane_width_m - 7.0) < 1e-9
    assert gate_reason(roads, hit, 1.75, 20.0, math.pi / 2.0, "weave", 8.0, weave_lane_w=2.9, half_width_m=HW) == "narrow"
    assert gate_reason(roads, hit, 1.75, 20.0, math.pi / 2.0, "weave", 8.0, weave_lane_w=3.5, half_width_m=HW) == "open"
    print("amp fits lane: lane_w/2 - half_w - 0.3, refused under 0.3 m (3.0 m lane at 0.9 m half width) OK")


def check_length_and_sent_accel() -> list[str]:
    # Whole half-waves nearest 60 m, at least one: ends on the lane center.
    for v, want in ((5.0, 60.0), (8.0, 60.0), (10.0, 50.0), (16.0, 80.0), (17.0, 42.5), (24.0, 60.0), (25.0, 62.5), (33.0, 82.5)):
        assert abs(weave_length_m(v) - want) < 1e-9, (v, weave_length_m(v))
    out = []
    paint = _paint([-3.5, 0.0, 3.5])
    roads = _straight(7.0)
    for v in (8.0, 16.0, 22.0, 33.0):
        lane = paint.lane_at((1.75, 20.0, 0.4), (0.0, 1.0, 0.0), ys=(0.0,))[0]
        hit = project_road(roads, 1.75, 20.0, math.pi / 2.0)
        info: dict = {}
        pts, _d = excursion_path(roads, 1.75, 20.0, math.pi / 2.0, Excursion("weave", 0.0, 0.0, 1, ""), hit, v,
                                 paint_lane=lane, half_width_m=HW, info=info)
        assert pts is not None, (v, info)
        assert abs(info["length_m"] - weave_length_m(v)) < 1e-6 and abs(pts[-1][0] - 1.75) < 0.02, (v, info, pts[-1])
        dev = max(abs(x - 1.75) for x, _y in pts)
        assert dev <= info["amp_m"] + 1e-9 and info["lat_acc_mps2"] <= exc.WEAVE_LAT_ACC_MPS2 + 1e-9, (v, info)
        out.append(f"v={v:.0f} line={info['length_m']:.1f}m amp={info['amp_m']:.2f}m wave={info['wave_m']:.0f}m "
                   f"sent peak={info['lat_acc_mps2']:.3f} (analytic {weave_lat_acc(info['amp_m'], info['wave_m'], v):.3f}) m/s2")
    # 33 m/s on a 400 m radius bend: the road alone is 2.7 m/s^2; the weave adds <= 1.0.
    r = 400.0
    arc = [(r - r * math.cos(t / r), r * math.sin(t / r)) for t in [4.0 * k for k in range(80)]]
    bend = [RoadPoly("b", arc, 7.0)]
    hit = project_road(bend, arc[2][0] + 1.75, arc[2][1], math.atan2(arc[3][1] - arc[1][1], arc[3][0] - arc[1][0]))
    info = {}
    pts, _d = excursion_path(bend, arc[2][0] + 1.75, arc[2][1], math.atan2(arc[3][1] - arc[1][1], arc[3][0] - arc[1][0]),
                             Excursion("weave", 0.0, 0.0, 1, ""), hit, 33.0, paint_lane=(-1.75, 1.75), half_width_m=HW, info=info)
    assert pts is not None and info["lat_acc_mps2"] <= 1.0 and info["total_lat_acc_mps2"] > 2.5, info
    out.append(f"v=33 on R=400 m: sent weave peak={info['lat_acc_mps2']:.3f}, line total={info['total_lat_acc_mps2']:.2f} m/s2")
    # The check is on the sent points: with the planner's guard bypassed, a 9 m/s 0.55 m wave on 40 m
    # (1.08 m/s^2) is shrunk to the limit; at 20 m/s it would need < 0.3 m and is refused.
    keep = (exc.weave_plan, exc.weave_wave_m)
    try:
        exc.weave_plan = lambda v, room: (min(0.6, room), 40.0)  # type: ignore[assignment]
        exc.weave_wave_m = lambda v: 40.0  # type: ignore[assignment]
        hit = project_road(roads, 1.75, 20.0, math.pi / 2.0)
        info = {}
        pts, _d = excursion_path(roads, 1.75, 20.0, math.pi / 2.0, Excursion("weave", 0.0, 0.0, 1, ""), hit, 9.0,
                                 paint_lane=(-1.75, 1.75), half_width_m=HW, info=info)
        assert pts is not None and info["shrunk"] >= 1 and info["lat_acc_mps2"] <= 1.0 and info["amp_m"] < 0.55, info
        acc_pts, _t = path_lat_acc(pts, [(0.0, p[1]) for p in pts], 9.0)
        assert abs(acc_pts - info["lat_acc_mps2"]) < 1e-9
        out.append(f"sent-path guard: 9 m/s 40 m wave shrunk to amp={info['amp_m']:.2f}m peak={info['lat_acc_mps2']:.3f} m/s2")
        info = {}
        pts, _d = excursion_path(roads, 1.75, 20.0, math.pi / 2.0, Excursion("weave", 0.0, 0.0, 1, ""), hit, 20.0,
                                 paint_lane=(-1.75, 1.75), half_width_m=HW, info=info)
        assert pts is None and info.get("refused") == "lat_acc", info
    finally:
        exc.weave_plan, exc.weave_wave_m = keep  # type: ignore[assignment]
    for line in out:
        print("  " + line)
    print("weave line: whole half-waves near 60 m; sent-path peak lateral accel <= 1.0 m/s^2 (shrink / refuse) OK")
    return out


def check_painted_lane_edge() -> None:
    """2 lanes per direction: paint at x = -7, -3.5, 0, 3.5, 7 on a 14 m road heading +y. The navgraph
    (lanesLeft/Right 1/1) calls it 2 lanes of 7 m; the abort follows the 3.5 m painted lane."""
    paint = _paint([-7.0, -3.5, 0.0, 3.5, 7.0])
    roads = _straight(14.0, 2)
    assert EDGE_M == 0.2

    def tick(x: float, phase: str = "weave") -> str:
        lane = paint.lane_at((x, 20.0, 0.4), (0.0, 1.0, 0.0), ys=(0.0,))[0]
        hit = project_road(roads, x, 20.0, math.pi / 2.0)
        clock = ExcursionClock()
        clock.phase = phase
        move = clock.update(0.1, offset_m=hit.lane_offset_m, heading_rad=0.0, lane_width_m=hit.lane_width_m,
                            road_offset_m=hit.offset_m, road_width_m=14.0, speed_mps=10.0,
                            half_width_m=HW, paint_lane=lane)
        return move.reason if move.phase == "abort" else ""

    # Painted lane (0, 3.5): right body edge x + 0.9 past 3.5 + 0.2 at x > 2.8.
    assert tick(2.75) == "" and tick(2.85) == "lane_edge"
    assert tick(0.75) == "" and tick(0.65) == "lane_edge"  # left edge x - 0.9 < -0.2
    # The navgraph lane here is (0, 7), center 3.5: x=2.85 is 0.65 m from its center, well inside.
    hit = project_road(roads, 2.85, 20.0, math.pi / 2.0)
    assert abs(hit.lane_width_m - 7.0) < 1e-9 and abs(hit.lane_offset_m) < 0.7
    assert abs(hit.offset_m) + HW < 7.0, "nowhere near the road edge"
    # Second lane of the same direction (3.5, 7): same rule against its own lines.
    # Left body edge x - 0.9 under 3.5 - 0.2 at x < 4.2; right x + 0.9 over 7.0 + 0.2 at x > 6.3.
    assert tick(4.25) == "" and tick(4.15) == "lane_edge" and tick(6.25) == "" and tick(6.35) == "lane_edge"
    # Unit helper and the no-paint path.
    assert not body_past_paint_edge((-1.75, 1.75), 0.9) and body_past_paint_edge((-1.75, 0.69), 0.9)
    assert unsafe_reason(offset_m=0.0, heading_rad=0.0, lane_width_m=3.5, off_road=False, near_junction=False,
                         traffic=False, phase="weave", half_width_m=HW, paint_lane=None, paint_lost=True) == "no_paint"
    print("painted lane edge: 4-lane mock, abort at painted lane edge + 0.2 m (x>2.8 / x<0.7), not the 7 m nav lane or road edge OK")


def check_straddle_phases() -> None:
    """x=0.4 on a 7 m road with paint at -3.5/0/3.5: a weave aborts lane_edge; lane_change and
    recover keep the old paved-edge rule and continue."""
    paint = _paint([-3.5, 0.0, 3.5])
    roads = _straight(7.0)
    lane = paint.lane_at((0.4, 20.0, 0.4), (0.0, 1.0, 0.0), ys=(0.0,))[0]
    assert lane == (-0.4, 3.1), lane
    hit = project_road(roads, 0.4, 20.0, math.pi / 2.0)
    got = {}
    for phase in ("weave", "lane_change", "recover"):
        clock = ExcursionClock()
        clock.phase = phase
        move = clock.update(0.1, offset_m=hit.lane_offset_m, heading_rad=0.0, lane_width_m=hit.lane_width_m,
                            road_offset_m=hit.offset_m, road_width_m=7.0, speed_mps=10.0, half_width_m=HW, paint_lane=lane)
        got[phase] = move.reason if move.phase == "abort" else move.phase
    assert got == {"weave": "lane_edge", "lane_change": "lane_change", "recover": "recover"}, got
    print(f"straddle x=0.4: {got} OK")


def check_vehicle_width() -> None:
    class _V:
        def __init__(self, box=None, model=None, err=None):
            self.box, self.model, self.err = box, model, err

        def get_bbox(self):
            if self.err:
                raise self.err
            return self.box

    def box(w):
        return {"front_bottom_left": (0.0, 2.0, 0.0), "front_bottom_right": (w, 2.0, 0.0),
                "rear_bottom_left": (0.0, -2.0, 0.0), "rear_bottom_right": (w * 0.98, -2.0, 0.0)}

    assert abs(bbox_width_m(box(2.04)) - 2.04) < 1e-9 and bbox_width_m(None) is None
    half, note = vehicle_half_width(_V(box(2.04), "etk800"))
    assert abs(half - 1.02) < 1e-9 and note == "source=bbox width=2.04m half=1.02m", note
    half, note = vehicle_half_width(_V(model="etk800", err=RuntimeError("not handled")))
    assert half == 0.9 and note.startswith("source=etk800 width=1.80m half=0.90m (bbox failed: RuntimeError"), note
    half, note = vehicle_half_width(_V(box(9.0), "etk800"))
    assert half == 0.9 and "bbox width 9.00m outside 1.2-3 m" in note, note
    half, note = vehicle_half_width(_V(model="pickup", err=RuntimeError("x")))
    assert half is None and "model=pickup is not etk800" in note, note

    class _NoBox:
        model = None

    half, note = vehicle_half_width(_NoBox())
    assert half is None and note.startswith("no get_bbox; model=unknown"), note
    print("vehicle width: bbox -> half, etk800 fallback logged, other model -> None (run stops) OK")


def check_impact_runup() -> None:
    assert PRE_IMPACT_S == 3.0
    watch = ImpactWatch()
    for k in range(60):
        watch.note_row(100.0 + 0.1 * k, k)
        watch.update(0.1, 15.0)
    assert watch.update(0.1, 2.0) is not None
    t0, i0 = watch.pre_window()
    # The impact tick is clock 6.1 s (row t 106.1): the first row inside 3 s is 103.1.
    assert i0 == 31 and abs(t0 - (106.1 - PRE_IMPACT_S)) < 1e-9, (t0, i0)
    print("impact run-up: pre_t0 is 3 s before the impact OK")


def main() -> None:
    check_scaling()
    check_amp_fits_lane()
    check_length_and_sent_accel()
    check_painted_lane_edge()
    check_straddle_phases()
    check_vehicle_width()
    check_impact_runup()
    print("collect weave scale ok")


if __name__ == "__main__":
    main()
