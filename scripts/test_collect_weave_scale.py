#!/usr/bin/env python3
"""Speed-compensated weaves: wavelength, amplitude cap, lateral-accel guard,
the body-edge lane abort, and the impact run-up / abort_t0 the mask uses."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect import excursion as exc  # noqa: E402
from python.collect.excursion import (  # noqa: E402
    EDGE_M,
    EGO_HALF_WIDTH_M,
    Excursion,
    ExcursionClock,
    body_past_lane_edge,
    excursion_path,
    gate_reason,
    targets_for,
    unsafe_reason,
    weave_amp_cap_m,
    weave_lat_acc,
    weave_plan,
    weave_wave_m,
)
from python.collect.liveness import PRE_IMPACT_S, ImpactWatch  # noqa: E402
from python.debug.gt_geom import RoadPoly, lane_center_offsets, project_road  # noqa: E402


def check_scaling() -> None:
    assert exc.WEAVE_PERIOD_S == 5.0
    assert weave_wave_m(None) == 40.0 and weave_wave_m(0.0) == 40.0 and weave_wave_m(8.0) == 40.0
    assert weave_wave_m(10.0) == 50.0 and weave_wave_m(25.0) == 125.0 and weave_wave_m(-30.0) == 150.0
    assert weave_amp_cap_m(None) == 0.8 and weave_amp_cap_m(5.0) == 0.8 and weave_amp_cap_m(10.0) == 0.8
    assert abs(weave_amp_cap_m(16.0) - 0.575) < 1e-9
    assert weave_amp_cap_m(22.0) == 0.35 and weave_amp_cap_m(35.0) == 0.35
    caps = [weave_amp_cap_m(v / 2.0) for v in range(0, 80)]
    assert all(a >= b - 1e-12 for a, b in zip(caps, caps[1:])), "cap must not rise with speed"
    # 3.5 m lane on a 7 m road: room is 1.75 - 0.9 = 0.85, so the 0.6 m nominal binds slow.
    for v, amp in ((8.0, 0.6), (10.0, 0.6), (16.0, 0.575), (22.0, 0.35), (30.0, 0.35)):
        spec = targets_for("weave", 1, 3.5, road_width_m=7.0, speed_mps=v)
        assert spec is not None and abs(spec[0] - amp) < 1e-9, (v, spec)
        assert abs(spec[1] + math.atan(amp * 2.0 * math.pi / weave_wave_m(v))) < 1e-12
    # Highways keep weaving (scaled, not off): 33 m/s on a 3.5 m lane.
    assert targets_for("weave", 1, 3.5, road_width_m=7.0, speed_mps=33.0) is not None
    assert gate_reason([RoadPoly("r", [(0.0, 0.0), (0.0, 500.0)], 7.0)],
                       project_road([RoadPoly("r", [(0.0, 0.0), (0.0, 500.0)], 7.0)], -1.75, 20.0, math.pi / 2.0),
                       -1.75, 20.0, math.pi / 2.0, "weave", 33.0) == "open"
    # The body stays in the lane at the planned peak: 2.3 m lane -> room 0.25 < 0.3 floor.
    assert targets_for("weave", 1, 2.3, road_width_m=4.6, lanes=2, speed_mps=10.0) is None
    assert targets_for("weave", 1, 2.5, road_width_m=5.0, lanes=2, speed_mps=10.0) is not None
    print("weave scaling: wave=max(40,5v), cap 0.8@<=10 -> 0.35@>=22 m/s, body-room limit OK")


def check_lat_acc_guard() -> None:
    assert abs(weave_lat_acc(0.6, 40.0, 8.0) - 0.6 * (2 * math.pi / 40) ** 2 * 64) < 1e-12
    # With the shipped constants no planned weave is over 1.0 m/s^2 at any speed.
    for v10 in range(0, 450):
        v = v10 / 10.0
        plan = weave_plan(v, 0.85)
        assert plan is not None, v
        assert weave_lat_acc(plan[0], plan[1], v) <= exc.WEAVE_LAT_ACC_MPS2 + 1e-9, (v, plan)
    old = exc.WEAVE_LAT_ACC_MPS2
    try:
        exc.WEAVE_LAT_ACC_MPS2 = 0.5  # the guard binds: 20 m/s, wave 100 m, cap 0.425
        amp, wave = weave_plan(20.0, 0.85)
        assert wave == 100.0 and abs(weave_lat_acc(amp, wave, 20.0) - 0.5) < 1e-9 and amp < weave_amp_cap_m(20.0)
        exc.WEAVE_LAT_ACC_MPS2 = 0.4  # shrunk under the 0.3 m floor -> no weave
        assert weave_plan(20.0, 0.85) is None
        assert targets_for("weave", 1, 3.5, road_width_m=7.0, speed_mps=20.0) is None
    finally:
        exc.WEAVE_LAT_ACC_MPS2 = old
    print("lateral-accel guard: shrinks to 1.0 m/s^2, refuses under the 0.3 m floor OK")


def check_path_follows_speed() -> None:
    roads = [RoadPoly("r", [(0.0, 0.0), (0.0, 500.0)], 7.0)]
    hit = project_road(roads, -1.75, 20.0, math.pi / 2.0)
    center = lane_center_offsets(7.0)[int(hit.lane_id)]
    weave = Excursion("weave", 0.0, 0.0, 1, "")
    slow, _ = excursion_path(roads, -1.75, 20.0, math.pi / 2.0, weave, hit, 8.0)
    fast, _ = excursion_path(roads, -1.75, 20.0, math.pi / 2.0, weave, hit, 25.0)
    assert slow and fast
    dev_slow = max(abs(x - center) for x, _y in slow)
    dev_fast = max(abs(x - center) for x, _y in fast)
    # 4 m path steps miss the exact crest; the peak never exceeds the planned amplitude.
    assert 0.55 < dev_slow <= 0.6 + 1e-9 and 0.33 < dev_fast <= 0.35 + 1e-9, (dev_slow, dev_fast)
    # Fast: half of the 125 m wave is sent (out and back), not 60 m of it.
    assert fast[-1][1] - fast[0][1] >= 62.5 - 1e-6 and abs(fast[-1][0] - center) < 0.02
    # Slow: 60 m of a 40 m wave.
    assert abs((slow[-1][1] - slow[0][1]) - 60.0) < 1e-6
    print(f"weave path: 8 m/s peak {dev_slow:.2f} m over 60 m, 25 m/s peak {dev_fast:.2f} m over {fast[-1][1] - fast[0][1]:.0f} m OK")


def check_body_edge_abort() -> None:
    assert EDGE_M == 0.2 and EGO_HALF_WIDTH_M == 0.9
    # 3.5 m lane: abort once |center offset| + 0.9 > 1.75 + 0.2, i.e. offset > 1.05 m.
    assert not body_past_lane_edge(1.049, 3.5)
    assert body_past_lane_edge(1.051, 3.5) and body_past_lane_edge(-1.06, 3.5)
    assert not body_past_lane_edge(None, 3.5) and not body_past_lane_edge(1.5, None)
    kw = dict(heading_rad=0.0, lane_width_m=3.5, off_road=False, near_junction=False, traffic=False,
              road_offset_m=-1.75 + 1.1, road_width_m=7.0)
    # The old weave rule (1 m past the paved edge) would not fire here; the body-edge rule does.
    assert unsafe_reason(offset_m=1.1, phase="weave", **kw) == ""  # no half width given: old rule only
    assert unsafe_reason(offset_m=1.1, phase="weave", half_width_m=EGO_HALF_WIDTH_M, **kw) == "lane_edge"
    assert unsafe_reason(offset_m=1.0, phase="weave", half_width_m=EGO_HALF_WIDTH_M, **kw) == ""
    # Lane changes and recoveries leave the lane by design: not this rule.
    assert unsafe_reason(offset_m=1.1, phase="lane_change", half_width_m=EGO_HALF_WIDTH_M, **kw) == ""
    assert unsafe_reason(offset_m=1.1, phase="recover", half_width_m=EGO_HALF_WIDTH_M, **kw) == ""
    clock = ExcursionClock()
    clock.phase = "weave"
    move = clock.update(0.1, offset_m=0.8, heading_rad=0.0, lane_width_m=3.5, speed_mps=20.0,
                        half_width_m=EGO_HALF_WIDTH_M, road_offset_m=-0.95, road_width_m=7.0)
    assert move.phase == "weave" and abs(move.target_offset_m - weave_amp_cap_m(20.0)) < 1e-3, move
    move = clock.update(0.1, offset_m=-1.2, heading_rad=0.0, lane_width_m=3.5, speed_mps=20.0,
                        half_width_m=EGO_HALF_WIDTH_M, road_offset_m=-2.95, road_width_m=7.0)
    assert move.phase == "abort" and move.reason == "lane_edge", move
    print("weave abort: body edge (lane offset +/- 0.9 m) past world lane edge + 0.2 m -> lane_edge OK")


def check_impact_runup() -> None:
    assert PRE_IMPACT_S == 3.0
    watch = ImpactWatch()
    hit = None
    for k in range(60):
        watch.note_row(100.0 + 0.1 * k, k)
        hit = watch.update(0.1, 15.0)
    hit = watch.update(0.1, 2.0)
    assert hit is not None
    t0, i0 = watch.pre_window()
    # The impact tick is clock 6.1 s (row t 106.1): the first row inside 3 s is 103.1.
    assert i0 == 31 and abs(t0 - (106.1 - PRE_IMPACT_S)) < 1e-9, (t0, i0)
    print("impact run-up: pre_t0 is 3 s before the impact OK")


def main() -> None:
    check_scaling()
    check_lat_acc_guard()
    check_path_follows_speed()
    check_body_edge_abort()
    check_impact_runup()
    print("collect weave scale ok")


if __name__ == "__main__":
    main()
