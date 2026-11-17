#!/usr/bin/env python3
"""Random per-clip length and traffic count, off unless the range flags are set.

Pure Python: no BeamNG, no torch. The _main checks use the existing fake
BeamNG collector harness from test_collect_spawn_fix.py.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import python.collect.clip_draw as clip_draw  # noqa: E402
import scripts.collect_scene_lanes as collect  # noqa: E402
from python.collect.clip_draw import (  # noqa: E402
    TRAFFIC_RANGE_MAX,
    ClipSchedule,
    clip_rng,
    clip_row_tags,
    clip_seed_tag,
    draw_segment_minutes,
    draw_traffic_count,
    maps_allow_traffic_range,
    minutes_to_segment_s,
    parse_int_range,
    parse_minute_range,
    resolve_traffic,
    segment_at,
    segment_boundary,
    validate_clip_flags,
)
from python.collect.spawn_mix import traffic_count, traffic_lua  # noqa: E402

MAPS = ("west_coast_usa", "italy")
SEGMENT_S = 12.0 * 60.0
FIXTURE = ROOT / "tests" / "fixtures" / "roads" / "west_coast_usa.json.gz"
MAPS6 = (
    "west_coast_usa",
    "italy",
    "east_coast_usa",
    "hirochi_raceway",
    "gridmap_v2",
    "small_island",
)


def check_ranges_parse_and_reject() -> None:
    assert parse_minute_range(None) is None
    assert parse_minute_range((5.0, 10.0)) == (5.0, 10.0)
    assert parse_minute_range((7, 7)) == (7.0, 7.0)
    assert parse_int_range(None) is None
    assert parse_int_range((0, 32)) == (0, 32)
    assert parse_int_range((4, 4)) == (4, 4)
    for lo, hi, words in (
        (0.0, 10.0, "below 1"),
        (0.5, 1.0, "below 1"),
        (10.0, 3.0, "above max"),
        (float("nan"), 10.0, "finite"),
    ):
        try:
            parse_minute_range((lo, hi))
        except ValueError as exc:
            assert words in str(exc), exc
        else:
            raise AssertionError((lo, hi))
    for lo, hi, words in (
        (-1, 8, "below 0"),
        (8, 3, "above max"),
        (0, 10000, "above 32"),
        (33, 33, "above 32"),
    ):
        try:
            parse_int_range((lo, hi))
        except ValueError as exc:
            assert words in str(exc), exc
        else:
            raise AssertionError((lo, hi))
    assert TRAFFIC_RANGE_MAX == 32
    assert maps_allow_traffic_range("west_coast_usa,italy")
    assert not maps_allow_traffic_range("italy")
    assert not maps_allow_traffic_range("")
    print("range parse: unset is None; lo>hi, minute LO<1, traffic HI>32 rejected")


def check_seeded_draws_are_reproducible() -> None:
    a = random.Random(0)
    b = random.Random(0)
    c = random.Random(4)
    mins_a = [draw_segment_minutes(a, 5.0, 10.0) for _ in range(20)]
    mins_b = [draw_segment_minutes(b, 5.0, 10.0) for _ in range(20)]
    mins_c = [draw_segment_minutes(c, 5.0, 10.0) for _ in range(20)]
    assert mins_a == mins_b
    assert mins_a != mins_c
    assert all(5.0 <= m <= 10.0 for m in mins_a + mins_c)
    assert draw_segment_minutes(random.Random(0), 8.0, 8.0) == 8.0
    assert minutes_to_segment_s(5.0) == 5.0 * 60.0
    assert minutes_to_segment_s(1.0) == 60.0
    # Inclusive HI: record the randint bounds, and actually draw HI.
    class _Probe:
        def __init__(self) -> None:
            self.called = None

        def randint(self, lo: int, hi: int) -> int:
            self.called = (int(lo), int(hi))
            return int(hi)

    probe = _Probe()
    assert draw_traffic_count(probe, 4, 12) == 12 and probe.called == (4, 12)
    rng = random.Random(1)
    counts = [draw_traffic_count(rng, 0, 32) for _ in range(4000)]
    assert min(counts) == 0 and max(counts) == 32
    assert 0 in counts and 32 in counts
    assert draw_traffic_count(random.Random(0), 6, 6) == 6
    assert clip_rng(7, 3).random() == clip_rng(7, 3).random()
    assert clip_rng(7, 3).random() != clip_rng(7, 4).random()
    assert clip_seed_tag(7, 3) == "7:3"
    print(f"seeded draws: seed 0 repeats; HI 32 drawn in {len(counts)} samples; minutes {min(mins_a):.2f}-{max(mins_a):.2f}")


def check_defaults_off_match_old_schedule() -> None:
    sched = ClipSchedule(MAPS, SEGMENT_S, None, 120.0 * 60.0)
    for t in (0.0, 1.0, SEGMENT_S - 1.0, SEGMENT_S, SEGMENT_S + 1.0, 3 * SEGMENT_S):
        assert sched.at(t) == segment_at(t, MAPS, SEGMENT_S)
        assert sched.boundary(t) == segment_boundary(t, MAPS, SEGMENT_S)
    assert sched.at(0.0, maps=()) == ("", 0)
    assert sched.boundary(0.0, maps=()) is None
    weights = {"west_coast_usa": 0.6}
    total = 120.0 * 60.0
    fixed = ClipSchedule(MAPS6, SEGMENT_S, weights, total)
    for t in (0.0, 60.0, SEGMENT_S, 39.7 * 60.0, total - 1.0):
        assert fixed.at(t) == segment_at(t, MAPS6, SEGMENT_S, weights, total)
        assert fixed.boundary(t) == segment_boundary(t, MAPS6, SEGMENT_S, weights, total)
    assert traffic_count(None, "west_coast_usa") == 2
    assert resolve_traffic(None, "west_coast_usa", None, random.Random(0)) == 2
    assert resolve_traffic(0, "west_coast_usa", None, random.Random(0)) == 0
    assert resolve_traffic(2, "italy", None, random.Random(0)) == 2
    assert clip_row_tags(None, None) == {}
    print("defaults off: schedule and traffic_count unchanged")


def check_ranged_schedule_draws_per_clip() -> None:
    expected = [minutes_to_segment_s(draw_segment_minutes(clip_rng(0, i), 5.0, 10.0)) for i in range(8)]
    sched = ClipSchedule(MAPS, SEGMENT_S, None, None, minute_range=(5.0, 10.0), seed=0)
    cursor = 0.0
    lengths = []
    for i in range(8):
        name, index = sched.at(cursor)
        assert index == i
        assert name == MAPS[i % 2]
        boundary = sched.boundary(cursor)
        assert boundary is not None and boundary > cursor
        lengths.append(boundary - cursor)
        assert sched.at(boundary - 1e-6)[1] == i
        cursor = boundary
    assert all(abs(got - want) < 1e-9 for got, want in zip(lengths, expected)), (lengths, expected)
    assert all(5.0 * 60.0 <= dur <= 10.0 * 60.0 for dur in lengths)
    other = ClipSchedule(MAPS, SEGMENT_S, None, None, minute_range=(5.0, 10.0), seed=4)
    other_len = []
    cursor = 0.0
    for _ in range(8):
        boundary = other.boundary(cursor)
        other_len.append(boundary - cursor)
        cursor = boundary
    assert other_len != lengths
    again = ClipSchedule(MAPS, SEGMENT_S, None, None, minute_range=(5.0, 10.0), seed=0)
    assert abs(again.boundary(0.0) - lengths[0]) < 1e-9
    assert abs(again.duration_min(0) - lengths[0] / 60.0) < 1e-9
    print(f"ranged schedule: 8 clips {min(lengths)/60:.2f}-{max(lengths)/60:.2f} min, seed 4 differs")


def _walk(sched: ClipSchedule, total: float) -> list[tuple[str, int, float]]:
    cursor = 0.0
    pieces = []
    while cursor < total - 1e-3:
        name, index = sched.at(cursor)
        boundary = sched.boundary(cursor)
        assert boundary is not None and boundary > cursor
        end = min(boundary, total)
        pieces.append((name, index, end - cursor))
        cursor = boundary if boundary < total + 1e-9 else total
        if cursor >= total:
            break
    return pieces


def check_ranged_weighted_stays_in_bounds() -> None:
    weights = {"west_coast_usa": 0.6}
    lo, hi = 5.0, 10.0
    for total_min, seed in ((30.0, 0), (30.0, 3), (120.0, 0), (120.0, 4)):
        total = total_min * 60.0
        sched = ClipSchedule(MAPS6, SEGMENT_S, weights, total, minute_range=(lo, hi), seed=seed)
        pieces = _walk(sched, total)
        assert pieces, (total_min, seed)
        assert abs(sum(dur for _n, _i, dur in pieces) - total) < 1.0, (total_min, seed, pieces)
        body, last = pieces[:-1], pieces[-1]
        for name, index, dur in body:
            assert lo * 60.0 - 1e-6 <= dur <= hi * 60.0 + 1e-6, (total_min, seed, name, index, dur / 60.0)
        assert last[2] <= hi * 60.0 + 1e-6, last
        # last may be shorter than LO when the run ends
    print("ranged weighted: 30-min and 120-min clips in [5, 10] except a short last")


def check_resolve_traffic_range_and_tags() -> None:
    rng = clip_rng(0, 0)
    want = draw_traffic_count(clip_rng(0, 0), 4, 12)
    got = resolve_traffic(None, "west_coast_usa", (4, 12), rng)
    assert got == want
    assert resolve_traffic(2, "italy", (0, 8), clip_rng(3, 1)) in range(0, 9)
    tags = clip_row_tags(segment_min=7.25, traffic_n=11)
    assert tags["segment_min"] == 7.25 and tags["traffic_count"] == 11
    assert "traffic_count" not in clip_row_tags(7.25, None)
    rec = {"segment": "west_coast_usa#0", "segment_min": 7.25, "traffic_count": 11, "seed": 0, "clip_seed": "0:0"}
    payload = json.dumps({"clip_draws": [rec]})
    assert "traffic_count" in payload and "clip_seed" in payload
    print(f"traffic range + tags: first seed-0/idx-0 draw is {got}")


def check_parser_defaults_and_flags() -> None:
    bare = collect.build_parser().parse_args([])
    assert bare.segment_min_range is None
    assert bare.traffic_range is None
    assert bare.seed == 0
    assert bare.segment_min == 12.0
    assert bare.traffic is None
    on = collect.build_parser().parse_args(
        [
            "--maps",
            "west_coast_usa",
            "--segment-min-range",
            "5",
            "10",
            "--traffic-range",
            "4",
            "12",
            "--seed",
            "7",
        ]
    )
    assert on.segment_min_range == [5.0, 10.0] or tuple(on.segment_min_range) == (5.0, 10.0)
    assert tuple(int(v) for v in on.traffic_range) == (4, 12)
    assert on.seed == 7
    lo, hi = parse_minute_range(on.segment_min_range)
    assert (lo, hi) == (5.0, 10.0)
    assert validate_clip_flags(on) == ((5.0, 10.0), (4, 12), 7)
    bare_ok = validate_clip_flags(bare)
    assert bare_ok == (None, None, 0)
    try:
        bad = collect.build_parser().parse_args(["--segment-min-range", "8", "2"])
        validate_clip_flags(bad)
    except ValueError as exc:
        assert "above max" in str(exc)
    else:
        raise AssertionError("hi < lo should fail")
    try:
        both = collect.build_parser().parse_args(
            ["--maps", "west_coast_usa", "--traffic", "2", "--traffic-range", "0", "8"]
        )
        validate_clip_flags(both)
    except ValueError as exc:
        assert "traffic-range" in str(exc) and "traffic" in str(exc)
    else:
        raise AssertionError("traffic + traffic-range should fail")
    try:
        validate_clip_flags(
            collect.build_parser().parse_args(["--maps", "italy", "--traffic-range", "0", "8"])
        )
    except ValueError as exc:
        assert "west_coast_usa" in str(exc)
    else:
        raise AssertionError("traffic-range on italy should fail")
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            collect._main(["--segment-min-range", "10", "3"], collect.RunStop())
    except SystemExit as exc:
        assert exc.code == 2, exc.code
    else:
        raise AssertionError("bad range should exit 2")
    assert "segment-min-range" in err.getvalue() or "above max" in err.getvalue()
    err2 = io.StringIO()
    try:
        with contextlib.redirect_stderr(err2):
            collect._main(["--maps", "italy", "--traffic-range", "4", "8"], collect.RunStop())
    except SystemExit as exc:
        assert exc.code == 2, exc.code
    else:
        raise AssertionError("traffic-range without west_coast should exit 2")
    print("parser: defaults off; flags parse; traffic+range, no-deck, and lo>hi exit 2")


def check_no_circular_import() -> None:
    import ast

    tree = ast.parse(inspect.getsource(clip_draw))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "collect_scene_lanes" not in alias.name, alias.name
        if isinstance(node, ast.ImportFrom):
            assert "collect_scene_lanes" not in (node.module or ""), node.module
    print("no circular import: clip_draw does not import collect_scene_lanes")


def check_traffic_slots_keep_two_and_extend() -> None:
    from python.collect.spawn_mix import SpawnDeck

    deck = SpawnDeck.load(FIXTURE)
    picked = deck.take()
    assert picked is not None
    two, notes_two = deck.traffic_slots(picked, 2, 60.0, 40.0)
    assert len(two) <= 2
    if len(two) == 2:
        assert two[0][0] == "ahead" and two[1][0] == "behind"
        assert abs(two[0][3] - 60.0) < 2.0 or any("fallback" in n for n in notes_two)
        assert abs(two[1][3] - 40.0) < 2.0 or any("fallback" in n for n in notes_two)
    eight, notes_eight = deck.traffic_slots(picked, 8, 60.0, 40.0)
    if len(two) == 2 and len(eight) >= 2:
        assert eight[0][1] == two[0][1] and eight[1][1] == two[1][1]
    names = [slot[0] for slot in eight]
    if names:
        assert names[0] == "ahead"
    if len(names) > 1:
        assert names[1] == "behind"
    script = "\n".join(traffic_lua(picked, 8, eight))
    for i in range(len(eight)):
        assert f"gvd_traffic_{i}" in script
    none = "\n".join(traffic_lua(picked, 0, []))
    assert "gvd_traffic" not in none
    fallback = "\n".join(traffic_lua(picked, 4, None))
    assert "gvd_traffic_0" in fallback and "gvd_traffic_3" in fallback
    print(f"traffic slots: count=2 keeps ahead/behind; count=8 placed {len(eight)} cars")


def check_traffic_slots_drop_stacked_poses() -> None:
    """West-coast fixture: placed poses stay distinct; extras are dropped, not stacked."""
    from python.collect.spawn_mix import SpawnDeck, TRAFFIC_MIN_M

    deck = SpawnDeck.load(FIXTURE)
    picks = [p for p in (deck.take() for _ in range(60)) if p is not None]
    assert len(picks) == 60
    placed_over_two = 0
    for count in (3, 4, 8, 12, 32):
        for pick in picks:
            slots, notes = deck.traffic_slots(pick, count, 60.0, 40.0)
            pos = [slot[1] for slot in slots]
            for i in range(len(pos)):
                for j in range(i + 1, len(pos)):
                    dist = math.dist(pos[i][:2], pos[j][:2])
                    assert dist >= TRAFFIC_MIN_M - 1e-6, (count, pick.road_id, dist, notes)
            if count >= 8 and len(slots) >= 3:
                placed_over_two += 1
    assert placed_over_two > 0, "expected some long roads to place more than two cars"
    print(f"traffic slots distinct: 60 picks x N=3,4,8,12,32; {placed_over_two} count>=8 placed>=3")


def check_main_logs_draws_on_fake_beamng() -> None:
    """_main with --traffic-range / --segment-min-range / --seed writes collect.json and state.jsonl."""
    from test_collect_spawn_fix import _Paint, _run_collector, _run_west_coast, _west_pools

    seed, lo_m, hi_m, lo_t, hi_t = 7, 1.0, 2.0, 4, 8
    code, text, out = _run_west_coast(
        _Paint(lambda x, y: True),
        _west_pools(),
        [
            "--segment-min-range",
            str(lo_m),
            str(hi_m),
            "--traffic-range",
            str(lo_t),
            str(hi_t),
            "--seed",
            str(seed),
            "--minutes",
            "3",
        ],
        2000,
    )
    assert code == 0, text[-2500:]
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    draws = status.get("clip_draws") or []
    assert draws, (status.keys(), text[-2000:])
    assert status.get("clip_seed") == seed or draws[0].get("seed") == seed
    for i, rec in enumerate(draws):
        assert rec["seed"] == seed
        assert rec["clip_seed"] == clip_seed_tag(seed, i)
        rng = clip_rng(seed, i)
        want_min = draw_segment_minutes(rng, lo_m, hi_m)
        want_tr = draw_traffic_count(rng, lo_t, hi_t)
        assert abs(float(rec["segment_min"]) - round(want_min, 4)) < 1e-9, rec
        assert int(rec["traffic_requested"]) == want_tr, rec
        assert "traffic_count" in rec
        assert lo_t <= int(rec["traffic_requested"]) <= hi_t
        # Long mocked roads should place the requested cars (kills spawn-ignores-draw).
        assert int(rec["traffic_count"]) == int(rec["traffic_requested"]), rec
        assert int(rec["traffic_requested"]) >= 4
    rows = [
        json.loads(line)
        for line in (out / "state.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tagged = [row for row in rows if row.get("traffic_count") is not None]
    assert tagged, "state.jsonl must carry the placed traffic_count tag"
    by_seg = {}
    for rec in draws:
        by_seg[rec["segment"]] = int(rec["traffic_count"])
    for row in tagged:
        if row.get("segment") in by_seg:
            assert int(row["traffic_count"]) == by_seg[row["segment"]], row
    assert any("requested=" in line and "placed=" in line for line in text.splitlines())
    # Single-level run: ranged ids must advance (current#0, current#1, ...).
    code2, text2, out2 = _run_collector(
        {"speed_at": (lambda n: 10.0), "stop_at": 1300},
        ["--segment-min-range", "1", "1", "--minutes", "3", "--no-excursions"],
    )
    assert code2 == 0, text2[-2000:]
    status2 = json.loads((out2 / "collect.json").read_text(encoding="utf-8"))
    draws2 = status2.get("clip_draws") or []
    segs = [rec["segment"] for rec in draws2]
    assert "current#0" in segs, segs
    assert any(seg != "current#0" for seg in segs), segs
    print(f"main fake BeamNG: {len(draws)} west clips logged; single-level ids {segs}")


def main() -> None:
    check_ranges_parse_and_reject()
    check_seeded_draws_are_reproducible()
    check_defaults_off_match_old_schedule()
    check_ranged_schedule_draws_per_clip()
    check_ranged_weighted_stays_in_bounds()
    check_resolve_traffic_range_and_tags()
    check_parser_defaults_and_flags()
    check_no_circular_import()
    check_traffic_slots_keep_two_and_extend()
    check_traffic_slots_drop_stacked_poses()
    check_main_logs_draws_on_fake_beamng()
    print("clip draw ok")


if __name__ == "__main__":
    main()
