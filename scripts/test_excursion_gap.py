#!/usr/bin/env python3
"""Cruise gap before a weave or lane change.

* Defaults stay on the old 60 s cadence (``CRUISE_S``), and that cadence does
  not consume the run RNG.
* ``--excursion-gap-min-s`` / ``--excursion-gap-max-s`` draw each cruise
  uniformly in that range from ``--excursion-seed`` (default 0). The same
  seed repeats. A new draw is taken only when the clock returns to cruise.
* With either gap flag set, segment load, teleport, respawn, and stall
  recover keep that one clock (elapsed cruise, armed gap, lane-change count).
  A teleport during a move ends it in cruise. With the flags absent, those
  points still build a new clock, matching main.
* min > max, min < 0, and a non-integer seed are rejected.
* The excursion start log names the gap that just elapsed.
* Spawn order does not use the gap generator.
"""
from __future__ import annotations

import contextlib
import io
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from python.collect.auto_mode import AutoOptions, collect_argv  # noqa: E402
from python.collect.excursion import (  # noqa: E402
    CRUISE_S,
    LANE_CHANGE_S,
    RECOVER_S,
    WEAVE_S,
    ExcursionClock,
    excursion_gap_bounds,
)

DT = 1.0
LO = 300.0
HI = 600.0


def _kw(**extra: object) -> dict:
    base: dict = {"offset_m": 0.0, "heading_rad": 0.0, "lane_width_m": 3.5, "n_lanes": 2}
    base.update(extra)
    return base


def _dur(phase: str) -> float:
    if phase == "weave":
        return WEAVE_S
    if phase == "lane_change":
        return LANE_CHANGE_S + RECOVER_S
    raise AssertionError(phase)


@dataclass(frozen=True)
class _Start:
    t: float
    phase: str
    gap: float
    cruise_s: float


def _simulate(clock: ExcursionClock, seconds: float, dt: float = DT) -> list[_Start]:
    """Drive ``clock`` for ``seconds``. Record each weave / lane-change start."""
    t = 0.0
    prev = clock.phase
    cruise_at = 0.0
    starts: list[_Start] = []
    while t < seconds - 1e-9:
        move = clock.update(dt, **_kw())
        t += dt
        if prev != "cruise" and move.phase == "cruise":
            cruise_at = t
        if prev == "cruise" and move.phase in ("weave", "lane_change"):
            gap = float(clock.gap_s)
            starts.append(_Start(t, move.phase, gap, t - cruise_at))
        prev = move.phase
    return starts


def check_defaults_match_the_old_cadence() -> None:
    plain = _simulate(ExcursionClock(), 600.0)
    other = _simulate(ExcursionClock(rng=random.Random(99)), 600.0)
    assert [(s.t, s.phase) for s in plain] == [(s.t, s.phase) for s in other]
    assert plain and plain[0].t == CRUISE_S and plain[0].phase == "weave"
    assert plain[3].phase == "lane_change"
    for i, start in enumerate(plain):
        assert start.gap == CRUISE_S
        assert abs(start.cruise_s - CRUISE_S) < 1e-6
        if i == 0:
            continue
        prev = plain[i - 1]
        assert abs((start.t - prev.t) - (_dur(prev.phase) + CRUISE_S)) < 1e-6, (prev, start)
    # Equal bounds do not draw, so a shared generator stays where it was.
    fresh = random.Random(1)
    used = random.Random(1)
    _simulate(ExcursionClock(rng=used), 200.0)
    assert used.random() == fresh.random()
    print(f"default: {len(plain)} starts on the {CRUISE_S:.0f}s cruise, lane change at {plain[3].t:.0f}s")


def check_range_over_an_hour() -> None:
    def run(seed: int) -> list[_Start]:
        return _simulate(ExcursionClock(LO, HI, random.Random(seed)), 60.0 * 60.0)

    same = run(7)
    assert [(s.t, s.phase, s.gap) for s in same] == [(s.t, s.phase, s.gap) for s in run(7)]
    assert [(s.t, s.gap) for s in same] != [(s.t, s.gap) for s in run(8)]
    assert len(same) >= 5, len(same)
    assert same[0].phase == "weave" and same[3].phase == "lane_change"
    probe = random.Random(7)
    for i, start in enumerate(same):
        expect = probe.uniform(LO, HI)
        assert abs(start.gap - expect) < 1e-9, (i, start.gap, expect)
        assert LO <= start.gap <= HI
        assert 0.0 <= start.cruise_s - start.gap <= DT + 1e-6
        if i == 0:
            continue
        prev = same[i - 1]
        cruise = (start.t - prev.t) - _dur(prev.phase)
        assert abs(cruise - start.gap) <= DT + 1e-6, (prev, start, cruise)
        assert LO - 1e-9 <= cruise <= HI + DT + 1e-6
    gaps = [s.gap for s in same]
    print(
        f"range: {len(same)} starts in 60 min, draws {min(gaps):.1f}-{max(gaps):.1f}s "
        f"from seed 7 (seed 8 differs)"
    )


def check_abort_and_respawn() -> None:
    rng = random.Random(4)
    probe = random.Random(4)
    clock = ExcursionClock(LO, HI, rng)
    t = 0.0
    prev = clock.phase
    while prev == "cruise":
        move = clock.update(DT, **_kw())
        t += DT
        prev = move.phase
        assert t < HI + 5.0
    assert prev == "weave"
    assert abs(clock.gap_s - probe.uniform(LO, HI)) < 1e-9
    aborted = clock.update(DT, **_kw(off_road=True))
    assert aborted.phase == "abort" and aborted.reason == "off_road"
    while clock.phase != "cruise":
        clock.update(DT, **_kw())
        t += DT
        assert t < HI + 40.0
    cruise_at = t
    armed = float(clock.gap_s)
    assert abs(armed - probe.uniform(LO, HI)) < 1e-9
    while clock.phase == "cruise":
        clock.update(DT, **_kw())
        t += DT
        assert t - cruise_at < HI + 5.0
    assert LO <= armed <= HI
    assert 0.0 <= (t - cruise_at) - armed <= DT + 1e-6
    # Teleport during the move: one new gap, lane-change count kept, not a new clock.
    kind = clock._kind
    clock.relocate()
    assert clock.phase == "cruise" and clock.phase_t == 0.0 and clock._kind == kind
    assert abs(clock.gap_s - probe.uniform(LO, HI)) < 1e-9
    for _ in range(30):
        clock.update(DT, **_kw())
    assert clock.phase == "cruise" and clock.phase_t == 30.0
    held_gap = float(clock.gap_s)
    clock.relocate()
    assert clock.phase == "cruise" and clock.phase_t == 30.0 and clock.gap_s == held_gap and clock._kind == kind
    print(f"abort and teleport: next gap {armed:.1f}s, relocate keeps {held_gap:.1f}s of cruise")


def check_rejects_a_bad_range() -> None:
    for lo, hi, words in (
        (-1.0, 10.0, "below 0"),
        (10.0, 3.0, "above max"),
        (-4.0, -1.0, "below 0"),
        (float("nan"), 10.0, "finite"),
    ):
        try:
            excursion_gap_bounds(lo, hi)
        except ValueError as exc:
            assert words in str(exc), exc
        else:
            raise AssertionError((lo, hi))
        try:
            ExcursionClock(lo, hi)
        except ValueError as exc:
            assert words in str(exc), exc
        else:
            raise AssertionError((lo, hi))
    assert excursion_gap_bounds(0, 0) == (0.0, 0.0)
    assert excursion_gap_bounds(CRUISE_S, CRUISE_S) == (CRUISE_S, CRUISE_S)
    import scripts.collect_scene_lanes as collect

    args = collect.build_parser().parse_args([])
    assert args.excursion_gap_min_s == CRUISE_S and args.excursion_gap_max_s == CRUISE_S
    assert args.excursion_gap_set is False and args.excursion_seed == 0
    wide = collect.build_parser().parse_args(
        ["--excursion-gap-min-s", "300", "--excursion-gap-max-s", "600", "--excursion-seed", "4"]
    )
    assert wide.excursion_gap_min_s == 300.0 and wide.excursion_gap_max_s == 600.0
    assert wide.excursion_gap_set is True and wide.excursion_seed == 4
    explicit = collect.build_parser().parse_args(["--excursion-gap-min-s", "60", "--excursion-gap-max-s", "60"])
    assert explicit.excursion_gap_set is True
    for argv in (
        ["--excursion-gap-min-s", "-1"],
        ["--excursion-gap-min-s", "10", "--excursion-gap-max-s", "3"],
        ["--excursion-seed", "nope"],
    ):
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                collect._main(argv, collect.RunStop())
        except SystemExit as exc:
            assert exc.code == 2, exc.code
        else:
            raise AssertionError(argv)
        assert "excursion" in err.getvalue(), err.getvalue()
    bare = collect_argv(AutoOptions(), "collect_scene_lanes.py")
    assert "--excursion-gap-min-s" not in bare and "--excursion-gap-max-s" not in bare
    assert "--excursion-seed" not in bare
    opts = AutoOptions(excursion_gap_min_s=LO, excursion_gap_max_s=HI, excursion_seed=4)
    argv = collect_argv(opts, "collect_scene_lanes.py")
    assert argv[argv.index("--excursion-gap-min-s") + 1] == "300.0"
    assert argv[argv.index("--excursion-gap-max-s") + 1] == "600.0"
    assert argv[argv.index("--excursion-seed") + 1] == "4"
    zero = [s.gap for s in _simulate(ExcursionClock(LO, HI, random.Random(0)), 1200.0)]
    other = [s.gap for s in _simulate(ExcursionClock(LO, HI, random.Random(4)), 1200.0)]
    assert zero != other
    print("reject: min < 0, min > max, and a bad seed exit 2; seed 4 differs from seed 0")


def _simulate_resets(reset_s: float, seconds: float, seed: int) -> list[_Start]:
    """The collector's place function, every ``reset_s``, on one clock."""
    import scripts.collect_scene_lanes as collect

    clock = ExcursionClock(LO, HI, random.Random(seed))

    def rearm():
        raise AssertionError("reset re-armed the clock from zero")

    t = 0.0
    prev = clock.phase
    cruise_at = 0.0
    starts: list[_Start] = []
    next_reset = float(reset_s)
    while t < seconds - 1e-9:
        if t + 1e-9 >= next_reset:
            before = clock.phase
            clock = collect.place_excursion_clock(clock, persist=True, fresh=rearm)
            if before != "cruise":
                cruise_at = t
            prev = clock.phase
            next_reset += float(reset_s)
        move = clock.update(DT, **_kw())
        t += DT
        if prev != "cruise" and move.phase == "cruise":
            cruise_at = t
        if prev == "cruise" and move.phase in ("weave", "lane_change"):
            starts.append(_Start(t, move.phase, float(clock.gap_s), t - cruise_at))
        prev = move.phase
    return starts


def _gap_stats(starts: list[_Start], reset_s: float) -> str:
    lane_changes = sum(1 for s in starts if s.phase == "lane_change")
    waits = [s.cruise_s for s in starts]
    mean = sum(waits) / len(waits)
    in_band = sum(1 for g in waits if LO - 1e-6 <= g <= HI + DT + 1e-6)
    early = [s for s in starts if s.t <= 3600.0 + 1e-6]
    assert any(s.phase == "lane_change" for s in early), (reset_s, [(s.t, s.phase) for s in early])
    assert LO - 1e-6 <= starts[0].t <= HI + DT + 1e-6, starts[0]
    for prev, start in zip(starts, starts[1:]):
        space = start.t - prev.t
        assert LO - 1e-6 <= space <= HI + _dur(prev.phase) + DT + 1e-6, (reset_s, prev, start, space)
    return (
        f"8h reset {reset_s:.0f}s: starts={len(starts)} lane_changes={lane_changes} "
        f"mean_gap={mean:.1f}s in_band={in_band}/{len(waits)}"
    )


def _collector_place_sites(src: str) -> dict[str, tuple[int, str]]:
    """The three collector calls, each named, so one ``persist=False`` fails alone.

    Stall recover and driving start share the same call text. The line number
    is what distinguishes them, so flipping either one is its own failure.
    """
    lines = src.splitlines()
    found: dict[str, tuple[int, str]] = {}
    for i, line in enumerate(lines):
        if "place_excursion_clock(" not in line or line.lstrip().startswith("def "):
            continue
        after = "\n".join(lines[i:i + 6])
        before = "\n".join(lines[max(0, i - 8):i])
        if "segment_aborts = 0" in after:
            name = "segment load"
        elif "weave_stats.end_stretch()" in after:
            name = "driving start"
        elif "stall_s" in before:
            name = "stall recover"
        else:
            raise AssertionError(f"line {i + 1} calls place_excursion_clock outside the three reset sites: {line.strip()}")
        if name in found:
            raise AssertionError(f"two {name} calls")
        found[name] = (i + 1, line)
    missing = {"stall recover", "segment load", "driving start"} - set(found)
    assert not missing, f"reset site missing: {sorted(missing)}"
    return found


def check_resets_keep_one_clock() -> None:
    import scripts.collect_scene_lanes as collect

    src = Path(collect.__file__).read_text(encoding="utf-8")
    assert src.count("clock = new_clock()") == 1, "a later site still builds a new clock"
    for name, (lineno, line) in _collector_place_sites(src).items():
        assert "persist=gap_persist" in line and "persist=False" not in line, (
            f"{name} (line {lineno}) passes persist=False while a gap flag is set: {line.strip()}"
        )
    fresh = ExcursionClock()
    fresh.update(10.0, **_kw())
    replaced = collect.place_excursion_clock(fresh, persist=False, fresh=ExcursionClock)
    assert replaced is not fresh and replaced.phase_t == 0.0 and replaced._kind == 0
    # A teleport during a weave keeps the object and the lane-change count.
    def rearm():
        raise AssertionError("reset re-armed the clock from zero")

    moving = ExcursionClock(LO, HI, random.Random(3))
    while moving.phase == "cruise":
        moving.update(DT, **_kw())
    kind = moving._kind
    moved = collect.place_excursion_clock(moving, persist=True, fresh=rearm)
    assert moved is moving and moved.phase == "cruise" and moved.phase_t == 0.0 and moved._kind == kind
    assert LO <= moved.gap_s <= HI
    lines = []
    for reset_s in (720.0, 180.0):
        starts = _simulate_resets(reset_s, 8.0 * 3600.0, seed=7)
        lines.append(_gap_stats(starts, reset_s))
        print(lines[-1])
    assert lines[0] != lines[1]


def check_gap_flag_keeps_the_clock_across_respawn() -> None:
    """A gap flag set means a respawn keeps the same clock and its cruise time.

    Forcing ``gap_persist = False`` builds a new clock here, so phase_t goes back
    to 0. The call-line check above still requires ``persist=gap_persist``.
    """
    import scripts.collect_scene_lanes as collect
    import python.collect.spawn_mix as spawn_mix
    import test_collect_spawn_fix as harness
    from python.collect.level_session import OpenedLevel

    pools = harness._west_pools()
    real = collect.place_excursion_clock
    seen: list[tuple[int, str, float, bool, int, float]] = []

    def wrapped(clock, *, persist, fresh):
        before_id, before_phase, before_t = id(clock), clock.phase, float(clock.phase_t)
        out = real(clock, persist=persist, fresh=fresh)
        seen.append((before_id, before_phase, before_t, bool(persist), id(out), float(out.phase_t)))
        return out

    def enter(bng, level, cfg, cams):
        del cfg, cams, level
        return OpenedLevel(
            vehicle=bng.vehicles.vehicle,
            roads=[harness.RoadPoly("r", [(0.0, 0.0), (0.0, 80.0)], 7.0)],
            spawn=(0.4, 12.0, 0.5),
            cameras={"main": harness._Cam()},
        )

    previous = (collect.enter_map, collect.painted_for, spawn_mix.SpawnDeck.load, collect.place_excursion_clock)
    collect.enter_map = enter  # type: ignore[assignment]
    collect.painted_for = lambda level: harness._Paint(lambda x, y: True)  # type: ignore[assignment]
    spawn_mix.SpawnDeck.load = classmethod(lambda cls, path, rng=None, **kw: cls(pools, rng, **kw))  # type: ignore[assignment]
    collect.place_excursion_clock = wrapped  # type: ignore[assignment]
    try:
        code, text, _out = harness._run_collector(
            {"speed_at": lambda n: 10.0, "stop_at": 80},
            ["--maps", "west_coast_usa", "--no-vary-env", "--respawn-min", "0.05",
             "--excursion-gap-min-s", "300", "--excursion-gap-max-s", "600"],
        )
    finally:
        collect.enter_map, collect.painted_for, spawn_mix.SpawnDeck.load, collect.place_excursion_clock = previous  # type: ignore[assignment]
    assert code == 0, text[-1500:]
    # The first places are the fresh clock. A later respawn must keep the cruise already driven.
    carried = [row for row in seen if row[2] > 1.0]
    assert carried, [(row[1], row[2], row[3]) for row in seen]
    for before_id, phase, before_t, persist, out_id, after_t in carried:
        assert persist is True and out_id == before_id and phase == "cruise" and after_t == before_t, (
            f"gap flag set but a reset rebuilt the clock "
            f"(persist={persist} phase={phase} phase_t {before_t} -> {after_t})"
        )
    print(f"respawn kept the clock: {len(carried)} places with cruise already {carried[0][2]:.1f}s")


def check_spawn_order_ignores_the_gap_rng() -> None:
    import scripts.collect_scene_lanes as collect
    import python.collect.spawn_mix as spawn_mix
    import test_collect_spawn_fix as harness
    from python.collect.level_session import OpenedLevel

    src = Path(collect.__file__).read_text(encoding="utf-8")
    load_at = src.index("spawn_deck = SpawnDeck.load(")
    load_call = src[load_at:src.index("\n                )", load_at)]
    assert "excursion_rng" not in load_call and "rng=" not in load_call, load_call
    pools = harness._west_pools()

    def enter(bng, level, cfg, cams):
        del cfg, cams, level
        return OpenedLevel(
            vehicle=bng.vehicles.vehicle,
            roads=[harness.RoadPoly("r", [(0.0, 0.0), (0.0, 80.0)], 7.0)],
            spawn=(0.4, 12.0, 0.5),
            cameras={"main": harness._Cam()},
        )

    def run(extra: list[str]) -> list[str]:
        seen: dict[str, object] = {}

        def load(cls, path, rng=None, **kw):
            deck = cls(pools, rng, **kw)
            seen["rng"] = rng
            seen["order"] = [road["id"] for kind in deck.pools.values() for road in kind]
            return deck

        previous = (collect.enter_map, collect.painted_for, spawn_mix.SpawnDeck.load)
        collect.enter_map = enter  # type: ignore[assignment]
        collect.painted_for = lambda level: harness._Paint(lambda x, y: True)  # type: ignore[assignment]
        spawn_mix.SpawnDeck.load = classmethod(load)  # type: ignore[assignment]
        try:
            code, text, _out = harness._run_collector(
                {"speed_at": lambda n: 10.0, "stop_at": 40},
                ["--maps", "west_coast_usa", "--no-vary-env", "--no-excursions"] + extra,
            )
        finally:
            collect.enter_map, collect.painted_for, spawn_mix.SpawnDeck.load = previous  # type: ignore[assignment]
        assert code == 0, text[-1500:]
        assert seen["rng"] is None, seen["rng"]
        return list(seen["order"])  # type: ignore[arg-type]

    plain = run([])
    seeded = run(["--excursion-gap-min-s", "300", "--excursion-gap-max-s", "600", "--excursion-seed", "11"])
    assert plain == seeded and len(plain) > 4, (plain, seeded)
    print(f"spawn order: {len(plain)} roads match with the gap flags on and off")


def check_start_log_names_the_gap() -> None:
    import scripts.collect_scene_lanes as collect
    import test_collect_spawn_fix as harness

    previous = collect.painted_for
    collect.painted_for = lambda level: harness._Paint(lambda x, y: True)  # type: ignore[assignment]
    try:
        code, text, _out = harness._run_collector(
            {"speed_at": lambda n: 10.0, "stop_at": 80, "x": 1.75},
            ["--excursion-gap-min-s", "1", "--excursion-gap-max-s", "1"],
        )
    finally:
        collect.painted_for = previous  # type: ignore[assignment]
    assert code == 0, text[-2000:]
    weave = [line for line in text.splitlines() if line.startswith("collect weave ")]
    assert weave and "gap=1.0s" in weave[0], weave or text[-2000:]
    ai = [line for line in text.splitlines() if "excursion=weave" in line]
    assert any("gap=1.0s" in line for line in ai), ai
    assert ExcursionClock().gap_log() == "gap=60.0s"
    print(f"start log: {weave[0]}")


def main() -> None:
    check_defaults_match_the_old_cadence()
    check_range_over_an_hour()
    check_abort_and_respawn()
    check_rejects_a_bad_range()
    check_resets_keep_one_clock()
    check_gap_flag_keeps_the_clock_across_respawn()
    check_spawn_order_ignores_the_gap_rng()
    check_start_log_names_the_gap()
    print("excursion gap ok")


if __name__ == "__main__":
    main()
