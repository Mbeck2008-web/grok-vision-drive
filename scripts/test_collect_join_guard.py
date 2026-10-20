#!/usr/bin/env python3
"""Excursion walk joins on the west_coast_usa road dump, and the sent-line backstop.

Pure Python: no BeamNG, no torch. Roads come from
``tests/fixtures/roads/west_coast_usa.json.gz``.

Join filter (``road_ahead_split``): a candidate is dropped before the heading
pick when the walk would jump more than JOIN_GAP_M onto it (tip to its
projected point), turn more than JOIN_HEADING_RAD, or change height by more
than JOIN_DZ_M.

* fork: the end of 894247 continues onto 892725 (0.18 m, 3.59 deg). The old
  heading-only rule took 892630 (9.97 m, 3.49 deg); rec5b rows 3281 and 27739
  sent that jump.
* stacked parking decks: the start of 890809 meets the end of 890846 one deck
  up (dz 4.05 m, 1.28 m, 2.3 deg in plan). Refused on dz; the walk takes
  890798 on its own deck.
* a real junction sharper than 30 deg (892485 -> 892493, 0.22 m, 47.4 deg) is
  cut. Deliberate: a cut only shortens the walk, a wrong join sends a line.

Backstop (``sent_line_fault``): line_kink over LINE_KINK_M, line_gap over
LINE_GAP_M, on synthetic lines and on rec5b row 27739 replayed with the old
join rule. Gate: ``gate_reason`` builds that same line and closes on its
refusal.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect import excursion as exc  # noqa: E402
from python.collect.excursion import (  # noqa: E402
    LINE_GAP_M,
    LINE_KINK_M,
    PATH_STEP_M,
    Excursion,
    excursion_path,
    line_kinks_m,
    sent_line_fault,
)
from python.collect.road_dump import load_road_dump  # noqa: E402
from python.debug import gt_geom  # noqa: E402
from python.debug.gt_geom import (  # noqa: E402
    CONNECT_HEADING,
    CONNECT_M,
    JOIN_DZ_M,
    JOIN_GAP_M,
    JOIN_HEADING_RAD,
    RoadHit,
    RoadPoly,
    join_candidates,
    road_ahead_split,
    wrap_rad,
)

ROADS_PATH = ROOT / "tests" / "fixtures" / "roads" / "west_coast_usa.json.gz"
TIP_TOL_M = 0.05
GAP_TOL_M = 0.01
DEG_TOL = 0.01


def _old_road_ahead_split(roads, road, x, y, heading_rad):
    """The rule before the gap-first filter: nearest heading among ends within CONNECT_M."""
    tip = gt_geom._forward_tip(road, x, y, heading_rad)
    if tip is None:
        return None, False
    tip_x, tip_y, tip_h = tip
    cont, cont_delta, others = None, None, 0
    for other in roads:
        if other.road_id == road.road_id or len(other.xy) < 2:
            continue
        for end, nxt in ((other.xy[0], other.xy[1]), (other.xy[-1], other.xy[-2])):
            end_m = math.hypot(end[0] - tip_x, end[1] - tip_y)
            if end_m > CONNECT_M:
                continue
            delta = abs(wrap_rad(math.atan2(nxt[1] - end[1], nxt[0] - end[0]) - tip_h))
            if delta <= CONNECT_HEADING:
                if cont_delta is None or delta < cont_delta:
                    cont_delta, cont = delta, other
            elif delta < 2.6 and end_m <= 4.0:
                others += 1
    return cont, others > 0


def _tip(road: RoadPoly, end: int) -> tuple[float, float, float]:
    """The road's end point and the heading of travel into it."""
    if end == -1:
        (ax, ay), (bx, by) = road.xy[-2], road.xy[-1]
    else:
        (ax, ay), (bx, by) = road.xy[1], road.xy[0]
    return bx, by, math.atan2(by - ay, bx - ax)


def _cand(cands, road_id: str):
    got = [c for c in cands if c.road.road_id == road_id and c.refused != "not_ahead"]
    assert len(got) == 1, (road_id, got)
    return got[0]


def _id(road: RoadPoly | None) -> str | None:
    return None if road is None else road.road_id


def check_constants() -> str:
    assert JOIN_GAP_M == 2.0 and abs(JOIN_HEADING_RAD - math.radians(30.0)) < 1e-12 and JOIN_DZ_M == 1.5
    assert LINE_KINK_M == 0.6 and LINE_GAP_M == 6.0
    return "constants: join gap 2.0 m, heading 30 deg, dz 1.5 m; line kink 0.6 m, gap 6 m"


def check_fork(roads: list[RoadPoly], byid: dict[str, RoadPoly]) -> str:
    road = byid["894247.0"]
    tx, ty, th = _tip(road, -1)
    assert math.hypot(tx + 457.37, ty - 139.67) < TIP_TOL_M, (tx, ty)
    old, _split = _old_road_ahead_split(roads, road, tx, ty, th)
    assert _id(old) == "892630.0", ("the old rule took 892630 (this test fails on main)", _id(old))
    cont, _split = road_ahead_split(roads, road, tx, ty, th)
    assert _id(cont) == "892725.0", _id(cont)
    _t, cands = join_candidates(roads, road, tx, ty, th)
    good, far = _cand(cands, "892725.0"), _cand(cands, "892630.0")
    assert abs(good.gap_m - 0.18) < GAP_TOL_M and abs(math.degrees(good.heading_rad) - 3.59) < DEG_TOL, good
    assert abs(far.gap_m - 9.97) < GAP_TOL_M and abs(math.degrees(far.heading_rad) - 3.49) < DEG_TOL, far
    # The far road is the better heading; only the gap rules it out.
    assert far.heading_rad < good.heading_rad and far.refused == "join_gap" and good.refused is None
    # The walk itself crosses onto 892725: past the tip every sample lies on it.
    start = road.xy[0]
    assert math.hypot(tx - start[0], ty - start[1]) > 15.0
    _len, _fork, _span, samples = exc._chain(roads, road, start[0], start[1], th, 40.0, PATH_STEP_M)
    s_tip = math.hypot(tx - start[0], ty - start[1])
    after = [p for p in samples if p[4] > s_tip + 1.0]
    assert after, "walk stopped at the tip"
    on = byid["892725.0"]
    worst = max(gt_geom._nearest_on(on, p[0], p[1])[0] for p in after)
    assert worst < 0.05, worst
    return (f"fork: 894247 tip -> 892725 (gap {good.gap_m:.2f} m, {math.degrees(good.heading_rad):.2f} deg); "
            f"892630 refused join_gap ({far.gap_m:.2f} m, {math.degrees(far.heading_rad):.2f} deg), old rule took it")


def check_garage(roads: list[RoadPoly], byid: dict[str, RoadPoly]) -> str:
    road = byid["890809.0"]
    tx, ty, th = _tip(road, 0)
    _t, cands = join_candidates(roads, road, tx, ty, th)
    upper = _cand(cands, "890846.0")
    # One of the three parking-deck joins the 2-D filter alone still took:
    # close in plan, straight ahead, one deck (~4 m) up.
    assert upper.gap_m <= JOIN_GAP_M and upper.heading_rad <= JOIN_HEADING_RAD, upper
    assert upper.dz_m is not None and abs(upper.dz_m - 4.05) < 0.01, upper.dz_m
    assert upper.refused == "join_dz", upper.refused
    cont, _split = road_ahead_split(roads, road, tx, ty, th)
    assert _id(cont) == "890798.0", _id(cont)  # same deck, 0.06 m, 4.8 deg
    same = _cand(cands, "890798.0")
    assert same.refused is None and same.dz_m is not None and abs(same.dz_m) < 0.01, same
    # Without heights the same tip takes the deck above: dz is what refuses it.
    flat = [RoadPoly(r.road_id, r.xy, r.width_m, r.lanes) for r in roads]
    flat_road = next(r for r in flat if r.road_id == "890809.0")
    flat_cont, _split = road_ahead_split(flat, flat_road, tx, ty, th)
    assert _id(flat_cont) == "890846.0", _id(flat_cont)
    old, _split = _old_road_ahead_split(roads, road, tx, ty, th)
    assert _id(old) != "890798.0", _id(old)
    return (f"garage: 890809 start -> 890798 (same deck); 890846 refused join_dz "
            f"(dz {upper.dz_m:+.2f} m, gap {upper.gap_m:.2f} m, {math.degrees(upper.heading_rad):.1f} deg)")


def check_sharp(roads: list[RoadPoly], byid: dict[str, RoadPoly]) -> str:
    # A real junction (the two ends overlap in the map.lua re-implementation)
    # that turns 47 deg. It is cut on purpose: we prefer cutting a real join
    # (the walk ends, the move is refused short / no_path) over risking a
    # wrong one (a line sent onto the wrong road).
    road = byid["892485.0"]
    tx, ty, th = _tip(road, -1)
    old, _split = _old_road_ahead_split(roads, road, tx, ty, th)
    assert _id(old) == "892493.0", _id(old)
    cont, _split = road_ahead_split(roads, road, tx, ty, th)
    assert cont is None, _id(cont)
    _t, cands = join_candidates(roads, road, tx, ty, th)
    sharp = _cand(cands, "892493.0")
    assert sharp.gap_m < 0.5 and math.degrees(sharp.heading_rad) > 30.0 and sharp.refused == "join_heading", sharp
    return (f"sharp: 892485 -> 892493 cut on purpose (gap {sharp.gap_m:.2f} m, "
            f"{math.degrees(sharp.heading_rad):.1f} deg > 30)")


def _line(n: int, step: float = 4.0) -> list[tuple[float, float]]:
    return [(0.0, step * k) for k in range(n)]


def check_backstop_synthetic() -> list[str]:
    out = []
    straight = _line(16)
    assert sent_line_fault(straight) is None and max(line_kinks_m(straight)) < 1e-9
    # A legitimate weave: 0.8 m amplitude over a 40 m wave, 4 m steps. The
    # middle sample sits ~half the second difference off the chord:
    # 0.5 * amp * (2 pi / wave)^2 * 16 = 0.16 m. rec5b's legitimate max was 0.367 m.
    weave = [(0.8 * math.sin(2.0 * math.pi * 4.0 * k / 40.0), 4.0 * k) for k in range(16)]
    w = max(line_kinks_m(weave))
    assert sent_line_fault(weave) is None and 0.1 < w < 0.2, w
    out.append(f"backstop: straight 0.000 m, 0.8 m weave {w:.3f} m -> sent")
    # A sideways step of d at 4 m spacing reads about d / 2 (d * 4 / sqrt(64 + d^2)):
    # 1.5 m mid-line is a kink, not a gap.
    step = [(0.0 if k < 8 else 1.5, 4.0 * k) for k in range(16)]
    got = sent_line_fault(step)
    assert got is not None and got[0] == "line_kink" and abs(got[1] - 1.5 * 4.0 / math.hypot(8.0, 1.5)) < 1e-9, got
    # The same step at the very end of the line shows only at the last
    # interior sample. It is still caught at 1.5 m.
    end = [(0.0 if k < 15 else 1.5, 4.0 * k) for k in range(16)]
    got_end = sent_line_fault(end)
    assert got_end is not None and got_end[0] == "line_kink", got_end
    # The limit of the cap: a 1.0 m step reads 0.50 m and is sent. Steps under
    # about 1.2 m pass (rec5b's smallest jump read 0.92 m).
    soft = [(0.0 if k < 8 else 1.0, 4.0 * k) for k in range(16)]
    ks = max(line_kinks_m(soft))
    assert sent_line_fault(soft) is None and 0.45 < ks < LINE_KINK_M, ks
    # Uneven spacing is normalized by 16 / (a * b): a 0.2 m bump between 1 m neighbours.
    tight = [(0.0, 0.0), (0.0, 4.0), (0.0, 8.0), (0.2, 9.0), (0.0, 10.0), (0.0, 14.0)]
    kt = max(line_kinks_m(tight))
    ab = math.hypot(0.2, 1.0)
    assert abs(kt - 0.2 * 16.0 / (ab * ab)) < 1e-9, kt
    assert sent_line_fault(tight)[0] == "line_kink"
    out.append(f"backstop: 1.5 m step mid-line {got[1]:.2f} m, at the end {got_end[1]:.2f} m -> line_kink; "
               f"1.0 m step reads {ks:.2f} m -> sent")
    # A 7 m gap straight ahead: no kink at all, refused line_gap.
    gap = _line(8) + [(0.0, 28.0 + 7.0 + 4.0 * k) for k in range(8)]
    g = sent_line_fault(gap)
    assert g is not None and g[0] == "line_gap" and abs(g[1] - 7.0) < 1e-9, g
    assert max(line_kinks_m(gap)) < 1e-9
    # 5.9 m is under LINE_GAP_M: the gap check alone does not see it. It
    # catches only big jumps, not a second line laid beside the first.
    near = _line(8) + [(0.0, 28.0 + 5.9 + 4.0 * k) for k in range(8)]
    assert sent_line_fault(near) is None, sent_line_fault(near)
    out.append(f"backstop: 7 m gap -> line_gap ({g[1]:.1f} m); 5.9 m straight gap is under the cap")
    return out


# rec5b row 27739: a weave from road 892691 toward 894247 at v_plan 11.0 m/s.
# Car state from the replay over the same road dump (follow_label, painted lane).
REC5B_27739 = dict(
    x=-439.57092064619064, y=186.67360454797745, heading=-1.9966500177595394, v=11.025,
    paint_lane=(-2.2297369312432513, 2.6370763749182826),
    hit=RoadHit(road_id="892691.0", offset_m=2.1776843568442583, heading_err_rad=-0.0034795775240130666,
                lane_id=1, lane_offset_m=0.17768413419996776, lane_width_m=4.000000445288581, off_road=False,
                dist_m=2.1776843568442583, curvature=0.000791682230404304, ahead_m=12.0),
)


def check_rec5b(roads: list[RoadPoly]) -> list[str]:
    c = REC5B_27739

    def run():
        info: dict = {}
        pts, _d = excursion_path(roads, c["x"], c["y"], c["heading"], Excursion("weave", 0.0, 0.0, 1, ""), c["hit"],
                                 c["v"], paint_lane=c["paint_lane"], half_width_m=1.01, info=info)
        return pts, info

    pts, info = run()
    assert pts is not None, info
    assert sent_line_fault(pts) is None
    on = next(r for r in roads if r.road_id == "892725.0")
    assert min(gt_geom._nearest_on(on, p[0], p[1])[0] for p in pts) < 3.0  # the line runs onto 892725
    out = [f"rec5b 27739, new join: sent, kink {max(line_kinks_m(pts)):.3f} m, lat_acc {info['lat_acc_mps2']:.2f}"]
    saved_split, saved_fault = exc.road_ahead_split, exc.sent_line_fault
    try:
        exc.road_ahead_split = _old_road_ahead_split
        # Old join, no backstop: this is the line rec5b sent (lat_acc passed).
        exc.sent_line_fault = lambda _points: None
        old_pts, old_info = run()
        assert old_pts is not None, old_info
        gaps = [math.dist(old_pts[i], old_pts[i + 1]) for i in range(len(old_pts) - 1)]
        assert max(gaps) > 10.0, max(gaps)
        # Old join with the backstop: refused.
        exc.sent_line_fault = saved_fault
        bad_pts, bad_info = run()
    finally:
        exc.road_ahead_split, exc.sent_line_fault = saved_split, saved_fault
    assert bad_pts is None and bad_info.get("refused") == "line_gap", bad_info
    assert max(line_kinks_m(old_pts)) > LINE_KINK_M
    out.append(f"rec5b 27739, old join: line jumps {max(gaps):.2f} m (kink {max(line_kinks_m(old_pts)):.2f} m), "
               f"backstop refuses {bad_info['refused']}")
    return out


def check_gate(roads: list[RoadPoly]) -> str:
    """The gate judges the line it would send (same inputs as the start)."""
    c = REC5B_27739

    def gate() -> str:
        return exc.gate_reason(roads, c["hit"], c["x"], c["y"], c["heading"], "weave", c["v"],
                               weave_lane_w=c["paint_lane"][1] - c["paint_lane"][0], half_width_m=1.01,
                               paint_lane=c["paint_lane"])

    assert gate() == "open", gate()
    saved = exc.road_ahead_split
    try:
        exc.road_ahead_split = _old_road_ahead_split
        # The window alone (length, curvature) is clear on the old join, at
        # both steps: the 8 m _span walk the gate used to read, and the 4 m walk.
        need = max(exc.WINDOW_M, exc.weave_length_m(c["v"]))
        road = next(r for r in roads if r.road_id == c["hit"].road_id)
        for length, _fork, span in (exc._span(roads, road, c["x"], c["y"], c["heading"], need),
                                    exc._chain(roads, road, c["x"], c["y"], c["heading"], need, PATH_STEP_M)[:3]):
            assert length + 1e-3 >= need and span / max(length, 1.0) < exc.CURVATURE_MAX, (length, span)
        old_gate = gate()
    finally:
        exc.road_ahead_split = saved
    # The sent line on that join jumps 10.55 m: the gate stays closed.
    assert old_gate == "line_gap", old_gate
    return "gate: rec5b 27739 open on the new join; on the old join the window is clear but the line is line_gap"


def main() -> None:
    roads, _payload = load_road_dump(ROADS_PATH)
    assert roads and all(r.z is not None for r in roads), "fixture roads should carry z"
    byid = {r.road_id: r for r in roads}
    lines = [check_constants(), check_fork(roads, byid), check_garage(roads, byid), check_sharp(roads, byid)]
    lines += check_backstop_synthetic()
    lines += check_rec5b(roads)
    lines.append(check_gate(roads))
    for line in lines:
        print(line)
    print("collect join guard ok")


if __name__ == "__main__":
    main()
