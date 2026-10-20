#!/usr/bin/env python3
"""Planning speed = max(measured, target) for weaves, lane changes and recovers;
the mid-move overspeed abort; --speed-min/--speed-max clamping (and Automatic
Mode passing them only when set); weave_stats; lane_change_incomplete."""
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
    WEAVE_STATS_NOTES,
    WeaveStats,
    excursion_path,
    overspeed,
    path_lat_acc,
    plan_speed,
    ramp_heading_reached,
    ramp_length_m,
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
    # Each refusal below is its own stretch: an excursion start or a respawn ends the one before.
    for reason, v in (("no_paint", 9.0), ("narrow", 13.0), ("short", 19.0), ("curve", 19.5), ("paint_ahead", 8.0),
                      ("junction_ahead", 8.0), ("lat_acc", 30.0), ("no_path", 24.0), ("no_path", 12.0)):
        st.refuse(reason, v)
        st.end_stretch()
    # Started moves cut short are aborted, not refused.
    st.abort("weave", "overspeed", 11.0)
    st.abort("lane_change", "overspeed", 20.0)
    st.lane_change_end(False)
    st.lane_change_end(True)
    got = st.summary()
    assert got["started"] == {"weave": 4, "lane_change": 1} and got["started_total"] == 5, got
    assert got["refused"] == {"no_paint": 1, "narrow": 1, "short": 1, "curve": 1, "paint_ahead": 1,
                              "junction_ahead": 1, "lat_acc": 1, "no_path": 2}, got
    assert got["refused_total"] == 9 and got["refused_attempts"] == got["refused"], got
    assert got["aborted"] == {"weave": {"overspeed": 1}, "lane_change": {"overspeed": 1}} and got["aborted_total"] == 2
    assert "overspeed_abort" not in got["refused"] and "overspeed_abort" not in got["refused_attempts"]
    assert got["lane_change_complete"] == 1 and got["lane_change_incomplete"] == 1, got
    assert got["by_speed_band"] == {
        "<12": {"started": 1, "refused": 3, "refused_attempts": 3, "aborted": 1},
        "12-18": {"started": 0, "refused": 2, "refused_attempts": 2, "aborted": 0},
        "18-24": {"started": 1, "refused": 2, "refused_attempts": 2, "aborted": 1},
        ">=24": {"started": 3, "refused": 2, "refused_attempts": 2, "aborted": 0}}, got
    assert got["half_wave_sides"] == {"left": 1, "right": 2}, got
    json.dumps(got)
    # A blocked stretch: the gate retries every 2 s; one refusal, every attempt kept.
    st = WeaveStats()
    for reason in ("curve", "curve", "short", "curve"):
        st.refuse(reason, 10.0)
    assert st.summary()["refused"] == {"curve": 1}, st.summary()
    assert st.summary()["refused_attempts"] == {"curve": 3, "short": 1}
    st.start("weave", 10.0)  # an excursion start ends the stretch
    st.refuse("curve", 10.0)
    st.refuse("curve", 10.0)
    st.end_stretch()  # a respawn or the segment end ends it too
    st.refuse("short", 10.0)
    got = st.summary()
    assert got["refused"] == {"curve": 2, "short": 1} and got["refused_total"] == 3, got
    assert got["refused_attempts"] == {"curve": 5, "short": 2} and got["refused_attempts_total"] == 7, got
    assert got["by_speed_band"]["<12"] == {"started": 1, "refused": 3, "refused_attempts": 7, "aborted": 0}, got
    assert set(WEAVE_STATS_NOTES) >= {"started", "refused", "refused_attempts", "aborted"}
    print(f"weave_stats: refused counts stretches, refused_attempts every retry, overspeed under aborted OK")


def check_ramp_heading_reached() -> None:
    ramp = 40.0
    full = math.atan(3.5 * math.pi / (2.0 * ramp))
    assert ramp_heading_reached(3.5, ramp, 0.0) == 0.0
    assert abs(ramp_heading_reached(3.5, ramp, 20.0) - full) < 1e-12  # the slope peaks at the midpoint
    assert ramp_heading_reached(3.5, ramp, 80.0) == ramp_heading_reached(3.5, ramp, 20.0)
    assert 0.0 < ramp_heading_reached(3.5, ramp, 8.0) < ramp_heading_reached(3.5, ramp, 12.0) < full
    assert ramp_heading_reached(-3.5, ramp, 10.0) == ramp_heading_reached(3.5, ramp, 10.0)
    print("ramp heading reached: 0 at the start, the ramp peak from its midpoint on OK")


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
    assert ws["started"] == {"weave": 1} and ws["refused"] == {} and ws["refused_attempts"] == {}, ws
    assert ws["aborted"] == {"weave": {"overspeed": 1}} and ws["aborted_total"] == 1, ws
    assert ws["by_speed_band"]["<12"] == {"started": 1, "refused": 0, "refused_attempts": 0, "aborted": 1}, ws
    assert status["notes"]["weave_stats"] == WEAVE_STATS_NOTES, status.get("notes")
    assert status["lane_change_tag"] == "peak_so_far", status.get("lane_change_tag")
    assert status["notes"]["lane_change_tag"].startswith("peak_so_far: "), status.get("notes")
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


def check_refused_stretches_run() -> None:
    """No paint: the weave gate is blocked no_paint and retried every 2 s. One stretch per spawn;
    the impact respawn starts a second one. Every retry stays in refused_attempts."""
    import scripts.test_collect_spawn_fix as harness

    code, text, out = harness._run_collector(
        {"speed_at": lambda n: 0.2 if 300 < n <= 340 else 10.0, "stop_at": 700}, ["--no-impact-respawn"])
    assert code == 0, text[-2000:]
    assert "collect recover current#0" in text, text[-3000:]
    before, after = text.split("collect recover current#0", 1)
    assert "weave blocked no_paint" in before and "weave blocked no_paint" in after, text[-3000:]
    ws = json.loads((out / "collect.json").read_text(encoding="utf-8"))["weave_stats"]
    assert ws["refused"] == {"no_paint": 2} and ws["refused_total"] == 2, ws
    tries = ws["refused_attempts"]["no_paint"]
    assert set(ws["refused_attempts"]) == {"no_paint"} and tries >= 10, ws
    assert ws["started"] == {} and ws["aborted"] == {}, ws
    print(f"refused stretches: 2 (one per spawn) from {tries} no_paint attempts OK")


def _lane_change_run(v_car: float, flags: list[str]):
    """A 4-lane 14 m road, painted, the car in a lane at a fixed ``v_car``; every start is a lane change."""
    import scripts.collect_scene_lanes as collect
    import scripts.test_collect_spawn_fix as harness

    roads = [RoadPoly("r", [(0.0, 0.0), (0.0, 1000.0)], 14.0, 4)]
    previous = (collect.painted_for, exc.ExcursionClock.peek_kind)
    collect.painted_for = lambda level: harness._Paint(lambda x, y: True)  # type: ignore[assignment]
    exc.ExcursionClock.peek_kind = lambda self, n: "lane_change" if n is not None and int(n) >= 2 else "weave"
    try:
        code, text, out = harness._run_collector(
            {"speed_at": lambda n: v_car, "stop_at": 400, "x": 1.75}, flags, roads=roads)
    finally:
        collect.painted_for, exc.ExcursionClock.peek_kind = previous  # type: ignore[assignment]
    assert code == 0, text[-2000:]
    rows = harness._rows(out)
    heads = [abs(float(r["excursion"]["target_heading_rad"])) for r in rows
             if (r.get("excursion") or {}).get("phase") == "lane_change"]
    ws = json.loads((out / "collect.json").read_text(encoding="utf-8"))["weave_stats"]
    return text, heads, ws


def check_lane_change_incomplete() -> None:
    """Planned at 17 m/s. A car at 2 m/s covers ~16 m of the ~74 m ramp in LANE_CHANGE_S: incomplete,
    and the label heading is the peak the line reached by then. At 17 m/s the ramp is covered."""
    assert exc.LANE_CHANGE_S == 8.0
    flags = ["--speed-min", "17", "--speed-max", "17"]
    text, heads, ws = _lane_change_run(2.0, flags)
    assert "collect lane_change current#0 v_plan=17.0 v_meas=2.0 v_target=17.0 delta=3.50m" in text, text[-3000:]
    line = [x for x in text.splitlines() if "lane_change_incomplete" in x]
    assert len(line) == 1 and line[0].startswith("collect excursion current#0 lane_change_incomplete covered=16m ramp="), line
    assert line[0].endswith(" v_meas=2.0 v_plan=17.0"), line
    ramp = float(line[0].split("ramp=", 1)[1].split("m", 1)[0])
    assert abs(ramp - ramp_length_m(3.5, 17.0)) < 1.0, (ramp, ramp_length_m(3.5, 17.0))
    assert ws["lane_change_incomplete"] == 1 and ws["lane_change_complete"] == 0, ws
    assert ws["started"].get("lane_change") == 1, ws
    full = math.atan(3.5 * math.pi / (2.0 * ramp_length_m(3.5, 17.0)))
    covered = 2.0 * 0.1 * (len(heads) - 1)
    assert len(heads) == 80 and heads == sorted(heads), heads
    assert abs(heads[-1] - ramp_heading_reached(3.5, ramp_length_m(3.5, 17.0), covered)) < 2e-3, (heads[-1], covered)
    assert heads[-1] < 0.7 * full, (heads[-1], full)
    slow_peak = heads[-1]
    text, heads, ws = _lane_change_run(17.0, flags)
    assert "lane_change_incomplete" not in text, text[-3000:]
    assert ws["lane_change_complete"] == 1 and ws["lane_change_incomplete"] == 0, ws
    assert len(heads) == 80 and heads == sorted(heads) and abs(heads[-1] - full) < 2e-3, (heads[-1], full)
    print(f"lane change: 2 m/s on a 17 m/s plan incomplete, label peak {math.degrees(slow_peak):.1f} deg "
          f"(ramp peak {math.degrees(full):.1f}); 17 m/s complete at the ramp peak OK")


def check_auto_mode_speed_flags() -> None:
    """Automatic Mode passes --speed-min/--speed-max only when the user set them."""
    import scripts.test_collect_spawn_fix as harness
    from python.collect.auto_mode import AutoOptions, collect_argv

    def flags_of(opts) -> list[str]:
        return collect_argv(opts, "collect_scene_lanes.py", python="py")[2:]

    opts = AutoOptions(minutes=5, maps="west_coast_usa", excursions=False, smoke_s=0, vary_env=False)
    argv = flags_of(opts)
    assert "--speed-min" not in argv and "--speed-max" not in argv, argv
    paint = harness._Paint(lambda x, y: True)

    def pools() -> dict:
        return {kind: [harness._road(f"{kind}{j}", [(5000 + 400 * k + 50 * j, 0), (5000 + 400 * k + 50 * j, 500)])
                       for j in range(4)] for k, kind in enumerate((CITY, HIGHWAY, RAMP))}

    code, text, _out = harness._run_west_coast(paint, pools(), argv + ["--respawn-min", "0.1"], 700)
    assert code == 0, text[-2000:]
    deck = [x for x in text.splitlines() if x.startswith("collect speed_range seg=") and "kind=level" not in x]
    assert deck and all("src=spawn_type" in x for x in deck), deck
    assert any("kind=highway range=20-30 src=spawn_type" in x for x in deck), deck
    assert "speed_clamp" not in text
    # Speeds the user sets are passed through, and the collector clamps with them.
    opts.cycle("speed+")
    argv = flags_of(opts)
    i = argv.index("--speed-min")
    assert argv[i:i + 4] == ["--speed-min", "8.0", "--speed-max", "18.0"], argv
    for given in (AutoOptions(speed_min=10.0, speed_max=15.0), AutoOptions(speed_set=True)):
        argv = flags_of(given)
        i = argv.index("--speed-min")
        assert argv[i:i + 4] == ["--speed-min", str(given.speed_min), "--speed-max", str(given.speed_max)], argv
    opts = AutoOptions(minutes=5, maps="west_coast_usa", excursions=False, smoke_s=0, vary_env=False,
                       speed_min=8.0, speed_max=17.0, speed_set=True)
    code, text, _out = harness._run_west_coast(paint, pools(), flags_of(opts) + ["--respawn-min", "0.1"], 700)
    assert code == 0, text[-2000:]
    assert "collect speed_clamp kind=highway range=20-30 flags=8.0-17.0 -> 8-17 (no overlap: flag range)" in text
    print(f"auto mode: no speed flags unless set ({len(deck)} spawns at spawn-type ranges, highway 20-30); "
          f"set speeds pass through OK")


def main() -> None:
    check_plan_speed()
    check_refusals_carry_reason()
    check_overspeed_rule()
    check_flag_clamp()
    check_weave_stats()
    check_ramp_heading_reached()
    check_collector_run()
    check_refused_stretches_run()
    check_lane_change_incomplete()
    check_west_coast_spawn_ranges()
    check_auto_mode_speed_flags()
    print("collect speed plan ok")


if __name__ == "__main__":
    main()
