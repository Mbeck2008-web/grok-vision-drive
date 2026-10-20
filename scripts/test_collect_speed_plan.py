#!/usr/bin/env python3
"""Planning speed = max(measured, target) for weaves, lane changes and recovers;
the mid-move overspeed abort; --speed-min/--speed-max clamping; weave_stats."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect import excursion as exc  # noqa: E402
from python.collect.excursion import (  # noqa: E402
    Excursion,
    WeaveStats,
    excursion_path,
    overspeed,
    path_lat_acc,
    plan_speed,
    speed_band,
)
from python.collect.spawn_mix import CITY, HIGHWAY, RAMP, SPEED, clamp_speed_range  # noqa: E402
from python.debug.gt_geom import RoadPoly, project_road  # noqa: E402

HW = 0.9
LIMIT = exc.WEAVE_LAT_ACC_MPS2


def _plan(phase: str, v: float, roads, x: float, direction: int = 1, paint=None):
    hit = project_road(roads, x, 20.0, math.pi / 2.0)
    info: dict = {}
    pts, used = excursion_path(roads, x, 20.0, math.pi / 2.0, Excursion(phase, 0.0, 0.0, direction, ""), hit, v,
                               paint_lane=paint, half_width_m=HW, info=info)
    return pts, used, info


def _sent_peak(pts, v: float) -> float:
    """Peak lateral accel of a sent line on the straight +y road, driven at ``v``."""
    return path_lat_acc(pts, [(0.0, p[1]) for p in pts], v)[0]


def check_plan_speed() -> list[str]:
    # Measured above the target: the measured speed plans the move.
    assert plan_speed(30.0, 17.0) == (30.0, 30.0, "measured")
    # Measured below the target: the target.
    assert plan_speed(7.8, 11.0) == (11.0, 7.8, "target")
    # No measured speed: the target, flagged so the collector logs it.
    assert plan_speed(None, 17.0) == (17.0, None, "target_fallback")
    assert plan_speed(float("nan"), 17.0)[2] == "target_fallback"
    out = []
    straight = [RoadPoly("r", [(0.0, 0.0), (0.0, 1000.0)], 7.0)]
    four = [RoadPoly("r", [(0.0, 0.0), (0.0, 1000.0)], 14.0, 4)]
    cases = (
        ("weave", straight, 1.75, 1, (-1.75, 1.75)),
        ("lane_change", four, 1.75, 1, None),
        ("recover", straight, 1.75, 1, None),
    )
    for phase, roads, x, direction, paint in cases:
        v_plan = plan_speed(30.0, 17.0)[0]
        pts, _used, info = _plan(phase, v_plan, roads, x, direction, paint)
        assert pts is not None, (phase, info)
        peak = _sent_peak(pts, 30.0)
        assert peak <= LIMIT + 1e-9 and abs(peak - info["lat_acc_mps2"]) < 1e-9, (phase, peak, info)
        # The old planner sized the line for the 17 m/s target; the car drove it at 30.
        old, _u, old_info = _plan(phase, 17.0, roads, x, direction, paint)
        assert old is not None, (phase, old_info)
        old_peak = _sent_peak(old, 30.0)
        assert old_peak > LIMIT, (phase, old_peak)
        # Measured below target: the plan is the target's.
        low, _u, low_info = _plan(phase, plan_speed(12.0, 17.0)[0], roads, x, direction, paint)
        assert low == old and low_info == old_info, phase
        out.append(f"{phase}: meas 30 / target 17 -> v_plan 30, sent peak at 30 m/s = {peak:.3f} m/s2 "
                   f"(planned at 17, driven at 30: {old_peak:.3f}); "
                   f"line={info['length_m']:.0f}m" + (f" ramp={info['ramp_m']:.0f}m" if info.get("ramp_m") else
                                                       f" amp={info['amp_m']:.2f}m wave={info['wave_m']:.0f}m"))
    for line in out:
        print("  " + line)
    print("planning speed: max(measured, target); 30 vs 17 keeps every sent line <= 1.0 m/s2 OK")
    return out


def check_refusals_carry_reason() -> None:
    # A road chain under 85% of the line: refused no_path (was a silent None).
    short = [RoadPoly("s", [(0.0, 0.0), (0.0, 40.0)], 7.0)]
    pts, _u, info = _plan("weave", 10.0, short, 1.75, 1, (-1.75, 1.75))
    assert pts is None and info.get("refused") == "no_path", info
    # A fast lane change needs more road: 120 m is enough at 10 m/s, not at 30.
    mid = [RoadPoly("m", [(0.0, 0.0), (0.0, 140.0)], 14.0, 4)]
    assert _plan("lane_change", 10.0, mid, 1.75)[0] is not None
    pts, _u, info = _plan("lane_change", 30.0, mid, 1.75)
    assert pts is None and info.get("refused") == "no_path", info
    hit = project_road(mid, 1.75, 20.0, math.pi / 2.0)
    assert exc.gate_reason(mid, hit, 1.75, 20.0, math.pi / 2.0, "lane_change", 10.0) == "open"
    assert exc.gate_reason(mid, hit, 1.75, 20.0, math.pi / 2.0, "lane_change", 30.0) == "short"
    # No room on a one-lane road: narrow.
    one = [RoadPoly("n", [(0.0, 0.0), (0.0, 500.0)], 3.5)]
    pts, _u, info = _plan("lane_change", 10.0, one, 0.0)
    assert pts is None and info.get("refused") == "narrow", info
    print("refusals: no_path (short chain, also a fast lane change), narrow; gate short at 30 m/s OK")


def check_overspeed_rule() -> None:
    assert exc.OVERSPEED_FRAC == 0.15
    assert not overspeed(11.5, 10.0) and overspeed(11.51, 10.0) and not overspeed(None, 10.0)
    assert not overspeed(12.0, None)
    assert [speed_band(v) for v in (0.0, 11.99, 12.0, 17.99, 18.0, 23.99, 24.0, 33.0)] == [
        "<12", "<12", "12-18", "12-18", "18-24", "18-24", ">=24", ">=24"]
    print("overspeed: > 15% over v_plan; bands <12, 12-18, 18-24, >=24 OK")


def check_flag_clamp() -> None:
    # None: the spawn-type ranges stand (highway 20-30 is intended).
    for kind in (CITY, HIGHWAY, RAMP):
        assert clamp_speed_range(SPEED[kind], None, None) == (SPEED[kind], "spawn_type", False)
    # Intersect.
    assert clamp_speed_range(SPEED[CITY], 8.0, 17.0) == ((8.0, 14.0), "spawn_type", False)
    assert clamp_speed_range(SPEED[RAMP], 8.0, 17.0) == ((10.0, 17.0), "flag", False)
    assert clamp_speed_range(SPEED[HIGHWAY], 15.0, 25.0) == ((20.0, 25.0), "flag", False)
    assert clamp_speed_range(SPEED[HIGHWAY], None, 26.0) == ((20.0, 26.0), "flag", False)
    # Empty intersection: the flag range itself.
    assert clamp_speed_range(SPEED[HIGHWAY], 8.0, 17.0) == ((8.0, 17.0), "flag", True)
    assert clamp_speed_range(SPEED[HIGHWAY], None, 15.0) == ((15.0, 15.0), "flag", True)
    assert clamp_speed_range(SPEED[CITY], 16.0, None) == ((16.0, 16.0), "flag", True)
    print("speed flags: None keeps the spawn type, a flag intersects, no overlap uses the flag range OK")


def check_weave_stats() -> None:
    st = WeaveStats()
    st.start("weave", 30.0, direction=1, one_sided=True)
    st.start("weave", 26.0, direction=-1, one_sided=True)
    st.start("weave", 25.0, direction=1, one_sided=True)
    st.start("weave", 10.0, direction=1, one_sided=False)  # full waves: no side
    st.start("lane_change", 20.0, direction=-1, one_sided=True)  # not a weave: no side
    for reason, v in (("no_paint", 9.0), ("narrow", 13.0), ("short", 19.0), ("curve", 19.5), ("paint_ahead", 8.0),
                      ("junction_ahead", 8.0), ("lat_acc", 30.0), ("no_path", 24.0), ("overspeed_abort", 11.0),
                      ("no_path", 12.0)):
        st.refuse(reason, v)
    got = st.summary()
    assert got["started"] == {"weave": 4, "lane_change": 1} and got["started_total"] == 5, got
    assert got["refused"] == {"no_paint": 1, "narrow": 1, "short": 1, "curve": 1, "paint_ahead": 1,
                              "junction_ahead": 1, "lat_acc": 1, "no_path": 2, "overspeed_abort": 1}, got
    assert got["refused_total"] == 10
    assert got["by_speed_band"] == {"<12": {"started": 1, "refused": 4}, "12-18": {"started": 0, "refused": 2},
                                    "18-24": {"started": 1, "refused": 2}, ">=24": {"started": 3, "refused": 2}}, got
    assert got["half_wave_sides"] == {"left": 1, "right": 2}, got
    json.dumps(got)
    print(f"weave_stats: {got['started_total']} started, {got['refused_total']} refused, sides {got['half_wave_sides']} OK")


def check_collector_run() -> None:
    """The mocked collector (test_collect_spawn_fix harness), painted everywhere, in lane at x=1.75:
    target 8 m/s, car at 10 -> the weave is planned at 10; the car then speeds to 12 (> 11.5) -> overspeed."""
    import scripts.collect_scene_lanes as collect
    import scripts.test_collect_spawn_fix as harness

    previous = collect.painted_for
    collect.painted_for = lambda level: harness._Paint(lambda x, y: True)  # type: ignore[assignment]
    try:
        code, text, out = harness._run_collector(
            {"speed_at": lambda n: 10.0 if n <= 215 else 12.0, "stop_at": 300, "x": 1.75},
            ["--speed-min", "8", "--speed-max", "17"],
        )
    finally:
        collect.painted_for = previous  # type: ignore[assignment]
    assert code == 0, text[-2000:]
    assert "collect speed_range seg=current#0 kind=level range=8-17 src=flag" in text, text[-3000:]
    assert "collect weave current#0 v_plan=10.0 v_meas=10.0 v_target=8.0 " in text, text[-3000:]
    assert "collect gate current#0 weave open v_plan=10.0" in text
    assert "collect excursion current#0 weave abort overspeed v_meas=12.0 v_plan=10.0" in text, text[-3000:]
    rows = harness._rows(out)
    reasons = {(r.get("excursion") or {}).get("reason") for r in rows if (r.get("excursion") or {}).get("phase") == "abort"}
    assert reasons == {"overspeed"}, reasons
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    ws = status["weave_stats"]
    assert ws["started"] == {"weave": 1} and ws["refused"] == {"overspeed_abort": 1}, ws
    assert ws["by_speed_band"]["<12"] == {"started": 1, "refused": 1}, ws
    assert status["speed_ranges"]["current#0"][0]["range"] == [8.0, 17.0], status["speed_ranges"]
    # Under 15% over: no abort.
    collect.painted_for = lambda level: harness._Paint(lambda x, y: True)  # type: ignore[assignment]
    try:
        code, text, out = harness._run_collector(
            {"speed_at": lambda n: 10.0 if n <= 215 else 11.4, "stop_at": 300, "x": 1.75}, [])
    finally:
        collect.painted_for = previous  # type: ignore[assignment]
    assert code == 0 and "overspeed" not in text, text[-2000:]
    assert "collect speed_range seg=current#0 kind=level range=8-17 src=default" in text, text[-3000:]
    print("collector: weave planned at the measured 10 m/s, abort overspeed at 12, none at 11.4, stats in collect.json OK")


def check_west_coast_spawn_ranges() -> None:
    """Every deck spawn logs its effective range; the flags clamp it, no flag keeps the spawn type."""
    import scripts.test_collect_spawn_fix as harness

    paint = harness._Paint(lambda x, y: True)

    def pools() -> dict:
        lanes = {kind: [harness._road(f"{kind}{j}", [(5000 + 400 * k + 50 * j, 0), (5000 + 400 * k + 50 * j, 500)])
                        for j in range(4)] for k, kind in enumerate((CITY, HIGHWAY, RAMP))}
        return lanes

    flags = ["--speed-min", "8", "--speed-max", "17", "--respawn-min", "0.1"]
    code, text, out = harness._run_west_coast(paint, pools(), flags, 700)
    assert code == 0, text[-2000:]
    spawns = [x for x in text.splitlines() if x.startswith("collect spawn west_coast_usa#") and " failed " not in x]
    ranges = [x for x in text.splitlines() if x.startswith("collect speed_range seg=west_coast_usa#")]
    deck = [x for x in ranges if "kind=level" not in x]
    assert spawns and len(deck) == len(spawns), (spawns, ranges)
    want = {CITY: "range=8-14 src=spawn_type", RAMP: "range=10-17 src=flag", HIGHWAY: "range=8-17 src=flag"}
    kinds = set()
    for line in deck:
        kind = line.split("kind=", 1)[1].split()[0]
        kinds.add(kind)
        assert want[kind] in line, line
    assert kinds == {CITY, HIGHWAY, RAMP}, kinds
    assert "collect speed_clamp kind=highway range=20-30 flags=8.0-17.0 -> 8-17 (no overlap: flag range)" in text
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    seg = status["speed_ranges"]["west_coast_usa#0"]
    assert {(e["kind"], tuple(e["range"]), e["src"]) for e in seg} >= {
        ("level", (8.0, 17.0), "flag"), (CITY, (8.0, 14.0), "spawn_type"), (HIGHWAY, (8.0, 17.0), "flag"),
        (RAMP, (10.0, 17.0), "flag")}, seg
    code, text, out = harness._run_west_coast(paint, pools(), ["--respawn-min", "0.1"], 700)
    deck = [x for x in text.splitlines() if x.startswith("collect speed_range seg=") and "kind=level" not in x]
    assert deck and all("src=spawn_type" in x for x in deck), deck
    assert any("kind=highway range=20-30" in x for x in deck), deck
    assert "speed_clamp" not in text
    print(f"west coast spawns: every spawn logs its range ({len(spawns)} with flags: city 8-14, ramp 10-17, "
          f"highway clamped 8-17); no flags: highway 20-30 OK")


def main() -> None:
    check_plan_speed()
    check_refusals_carry_reason()
    check_overspeed_rule()
    check_flag_clamp()
    check_weave_stats()
    check_collector_run()
    check_west_coast_spawn_ranges()
    print("collect speed plan ok")


if __name__ == "__main__":
    main()
