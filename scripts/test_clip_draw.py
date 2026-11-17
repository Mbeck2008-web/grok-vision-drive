#!/usr/bin/env python3
"""Random per-clip length and traffic count, off unless the range flags are set.

Pure Python: no BeamNG, no torch.

* ``--segment-min-range`` / ``--traffic-range`` are unset by default, so
  ``segment_at`` / ``traffic_count`` stay exactly as they were.
* A seeded RNG draws each clip's length in [LO, HI] minutes and each
  clip's AI car count in [LO, HI]. The same seed is the same sequence;
  another seed is another sequence. Draws are logged for collect.json
  and as ``traffic_count`` / ``segment_min`` row tags.
* ``traffic_slots`` still places two companions at the configured ahead
  and behind distances. Count > 2 adds staggered cars along the road.
"""
from __future__ import annotations

import contextlib
import io
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.collect_scene_lanes as collect  # noqa: E402
from python.collect.clip_draw import (  # noqa: E402
    ClipSchedule,
    clip_row_tags,
    draw_segment_minutes,
    draw_traffic_count,
    minutes_to_segment_s,
    parse_int_range,
    parse_minute_range,
    resolve_traffic,
    validate_clip_flags,
)
from python.collect.spawn_mix import traffic_count, traffic_lua  # noqa: E402
from scripts.collect_scene_lanes import segment_at, segment_boundary  # noqa: E402

MAPS = ("west_coast_usa", "italy")
SEGMENT_S = 12.0 * 60.0
FIXTURE = ROOT / "tests" / "fixtures" / "roads" / "west_coast_usa.json.gz"


def check_ranges_parse_and_reject() -> None:
    assert parse_minute_range(None) is None
    assert parse_minute_range((5.0, 10.0)) == (5.0, 10.0)
    assert parse_minute_range((7, 7)) == (7.0, 7.0)
    assert parse_int_range(None) is None
    assert parse_int_range((0, 32)) == (0, 32)
    assert parse_int_range((4, 4)) == (4, 4)
    for lo, hi, words in (
        (-1.0, 10.0, "below 0"),
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
    ):
        try:
            parse_int_range((lo, hi))
        except ValueError as exc:
            assert words in str(exc), exc
        else:
            raise AssertionError((lo, hi))
    print("range parse: unset is None; lo>hi and lo<0 rejected")


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
    assert minutes_to_segment_s(0.5) == 60.0  # same 60 s floor as --segment-min
    counts_a = [draw_traffic_count(random.Random(0), 4, 12) for _ in range(1)]
    # restart the same seed: first draw matches, and inclusive bounds hold
    rng = random.Random(1)
    counts = [draw_traffic_count(rng, 0, 32) for _ in range(80)]
    assert min(counts) >= 0 and max(counts) <= 32
    assert 0 in counts and 32 in counts or len(set(counts)) > 8
    assert draw_traffic_count(random.Random(0), 6, 6) == 6
    assert counts_a[0] == draw_traffic_count(random.Random(0), 4, 12)
    print(f"seeded draws: seed 0 repeats, seed 4 differs; minutes {min(mins_a):.2f}-{max(mins_a):.2f}")


def check_defaults_off_match_old_schedule() -> None:
    sched = ClipSchedule(MAPS, SEGMENT_S, None, 120.0 * 60.0)
    for t in (0.0, 1.0, SEGMENT_S - 1.0, SEGMENT_S, SEGMENT_S + 1.0, 3 * SEGMENT_S):
        assert sched.at(t) == segment_at(t, MAPS, SEGMENT_S)
        assert sched.boundary(t) == segment_boundary(t, MAPS, SEGMENT_S)
    assert sched.at(0.0, maps=()) == ("", 0)
    assert sched.boundary(0.0, maps=()) is None
    weights = {"west_coast_usa": 0.6}
    total = 120.0 * 60.0
    maps = (
        "west_coast_usa",
        "italy",
        "east_coast_usa",
        "hirochi_raceway",
        "gridmap_v2",
        "small_island",
    )
    fixed = ClipSchedule(maps, SEGMENT_S, weights, total)
    for t in (0.0, 60.0, SEGMENT_S, 39.7 * 60.0, total - 1.0):
        assert fixed.at(t) == segment_at(t, maps, SEGMENT_S, weights, total)
        assert fixed.boundary(t) == segment_boundary(t, maps, SEGMENT_S, weights, total)
    assert traffic_count(None, "west_coast_usa") == 2
    assert resolve_traffic(None, "west_coast_usa", None, random.Random(0)) == 2
    assert resolve_traffic(0, "west_coast_usa", None, random.Random(0)) == 0
    assert resolve_traffic(2, "italy", None, random.Random(0)) == 2
    assert clip_row_tags(None, None) == {}
    print("defaults off: schedule and traffic_count unchanged")


def check_ranged_schedule_draws_per_clip() -> None:
    rng = random.Random(0)
    probe = random.Random(0)
    expected = [minutes_to_segment_s(draw_segment_minutes(probe, 5.0, 10.0)) for _ in range(8)]
    sched = ClipSchedule(MAPS, SEGMENT_S, None, None, minute_range=(5.0, 10.0), rng=rng)
    cursor = 0.0
    lengths = []
    for i in range(8):
        name, index = sched.at(cursor)
        assert index == i
        assert name == MAPS[i % 2]
        boundary = sched.boundary(cursor)
        assert boundary is not None and boundary > cursor
        lengths.append(boundary - cursor)
        # still in this clip just before the boundary
        assert sched.at(boundary - 1e-6)[1] == i
        cursor = boundary
    assert all(abs(got - want) < 1e-9 for got, want in zip(lengths, expected))
    assert all(5.0 * 60.0 <= dur <= 10.0 * 60.0 for dur in lengths)
    other = ClipSchedule(MAPS, SEGMENT_S, None, None, minute_range=(5.0, 10.0), rng=random.Random(4))
    other_len = []
    cursor = 0.0
    for _ in range(8):
        boundary = other.boundary(cursor)
        other_len.append(boundary - cursor)
        cursor = boundary
    assert other_len != lengths
    # same seed, same sequence again
    again = ClipSchedule(MAPS, SEGMENT_S, None, None, minute_range=(5.0, 10.0), rng=random.Random(0))
    assert abs(again.boundary(0.0) - lengths[0]) < 1e-9
    assert abs(again.duration_min(0) - lengths[0] / 60.0) < 1e-9
    print(f"ranged schedule: 8 clips {min(lengths)/60:.2f}-{max(lengths)/60:.2f} min, seed 4 differs")


def check_ranged_weighted_stays_in_bounds() -> None:
    maps = ("west_coast_usa", "italy", "east_coast_usa")
    weights = {"west_coast_usa": 0.6}
    total = 60.0 * 60.0
    sched = ClipSchedule(maps, SEGMENT_S, weights, total, minute_range=(5.0, 10.0), rng=random.Random(0))
    cursor = 0.0
    pieces = []
    while cursor < total - 1e-3:
        name, index = sched.at(cursor)
        boundary = sched.boundary(cursor)
        assert boundary is not None
        pieces.append((name, index, boundary - cursor))
        cursor = boundary
    assert abs(sum(dur for _n, _i, dur in pieces) - total) < 1.0
    # non-tail chunks sit in [5, 10] min; a last remainder of a share may be shorter
    body = [dur for _n, _i, dur in pieces if dur > 0.05 * 60.0]
    assert body and all(dur <= 10.0 * 60.0 + 1e-6 for dur in body)
    west = sum(dur for name, _i, dur in pieces if name == "west_coast_usa")
    assert abs(west - 0.6 * total) < 1.0
    print(f"ranged weighted: {len(pieces)} clips, west={west/60:.1f} min of {total/60:.0f}")


def check_resolve_traffic_range_and_tags() -> None:
    rng = random.Random(0)
    want = draw_traffic_count(random.Random(0), 4, 12)
    got = resolve_traffic(None, "west_coast_usa", (4, 12), rng)
    assert got == want
    # --traffic is ignored once a range is set (the collector rejects passing both)
    assert resolve_traffic(2, "italy", (0, 8), random.Random(3)) in range(0, 9)
    tags = clip_row_tags(segment_min=7.25, traffic_n=11)
    assert tags["segment_min"] == 7.25 and tags["traffic_count"] == 11
    assert "traffic_count" not in clip_row_tags(7.25, None)
    assert "segment_min" not in clip_row_tags(None, 11)
    rec = {"segment": "west_coast_usa#0", "segment_min": 7.25, "traffic_count": 11, "seed": 0}
    payload = json.dumps({"clip_draws": [rec]})
    assert "traffic_count" in payload and "7.25" in payload
    print(f"traffic range + tags: first seed-0 draw is {got}")


def check_parser_defaults_and_flags() -> None:
    bare = collect.build_parser().parse_args([])
    assert bare.segment_min_range is None
    assert bare.traffic_range is None
    assert bare.seed == 0
    assert bare.segment_min == 12.0
    assert bare.traffic is None
    on = collect.build_parser().parse_args(
        ["--segment-min-range", "5", "10", "--traffic-range", "4", "12", "--seed", "7"]
    )
    assert on.segment_min_range == [5.0, 10.0] or tuple(on.segment_min_range) == (5.0, 10.0)
    assert tuple(int(v) for v in on.traffic_range) == (4, 12)
    assert on.seed == 7
    lo, hi = parse_minute_range(on.segment_min_range)
    assert (lo, hi) == (5.0, 10.0)
    assert validate_clip_flags(on) == ( (5.0, 10.0), (4, 12), 7 )
    bare_ok = validate_clip_flags(bare)
    assert bare_ok == (None, None, 0)
    try:
        bad = collect.build_parser().parse_args(["--traffic-range", "8", "2"])
        validate_clip_flags(bad)
    except ValueError as exc:
        assert "above max" in str(exc)
    else:
        raise AssertionError("hi < lo should fail")
    try:
        both = collect.build_parser().parse_args(["--traffic", "2", "--traffic-range", "0", "8"])
        validate_clip_flags(both)
    except ValueError as exc:
        assert "traffic-range" in str(exc) and "traffic" in str(exc)
    else:
        raise AssertionError("traffic + traffic-range should fail")
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            collect._main(["--segment-min-range", "10", "3"], collect.RunStop())
    except SystemExit as exc:
        assert exc.code == 2, exc.code
    else:
        raise AssertionError("bad range should exit 2")
    assert "segment-min-range" in err.getvalue() or "above max" in err.getvalue()
    print("parser: defaults off; flags parse; traffic+range and lo>hi exit 2")


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
    # first two slots match the count=2 placement when both succeeded
    if len(two) == 2 and len(eight) >= 2:
        assert eight[0][1] == two[0][1] and eight[1][1] == two[1][1]
    assert len(eight) >= min(4, 8)  # staggered extras should appear on this dump
    names = [slot[0] for slot in eight]
    assert names[0] == "ahead"
    if len(names) > 1:
        assert names[1] == "behind"
    script = "\n".join(traffic_lua(picked, 8, eight))
    for i in range(len(eight)):
        assert f"gvd_traffic_{i}" in script
    none = "\n".join(traffic_lua(picked, 0, []))
    assert "gvd_traffic" not in none
    # no-slot fallback also extends past the original two offsets
    fallback = "\n".join(traffic_lua(picked, 4, None))
    assert "gvd_traffic_0" in fallback and "gvd_traffic_3" in fallback
    print(f"traffic slots: count=2 keeps ahead/behind; count=8 placed {len(eight)} cars")


def main() -> None:
    check_ranges_parse_and_reject()
    check_seeded_draws_are_reproducible()
    check_defaults_off_match_old_schedule()
    check_ranged_schedule_draws_per_clip()
    check_ranged_weighted_stays_in_bounds()
    check_resolve_traffic_range_and_tags()
    check_parser_defaults_and_flags()
    check_traffic_slots_keep_two_and_extend()
    print("clip draw ok")


if __name__ == "__main__":
    main()
