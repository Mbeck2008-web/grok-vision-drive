#!/usr/bin/env python3
"""Excursions, map segments, smoke decode, and the segment holdout."""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect.auto_mode import DEFAULT_MAPS, AutoOptions, collect_argv  # noqa: E402
from python.collect.excursion import (  # noqa: E402
    CRUISE_S,
    HEADING_DEG,
    RECOVER_OFFSET_M,
    WEAVE_AMP_M,
    Excursion,
    ExcursionClock,
    excursion_commands,
    excursion_path,
    near_junction,
    targets_for,
)
from python.data.raw_frames import RawFrameWriter  # noqa: E402
from python.data.raw_queue import RawWriteQueue  # noqa: E402
from python.debug.gt_geom import RoadPoly, lane_center_offsets, project_road  # noqa: E402
from python.train.continual import pass_plan  # noqa: E402
from python.viz.nerd import render_panel  # noqa: E402
from python.viz.stage import VizUI  # noqa: E402
from scripts.collect_scene_lanes import (  # noqa: E402
    achieved_hz,
    decode_issues,
    segment_at,
    segment_env,
    start_freeroam,
    tick_sleep_s,
)


def _hit(ahead: float, *, off: bool = False, broken: bool = False) -> object:
    class Hit:
        pass

    hit = Hit()
    hit.off_road = off
    hit.ahead_m = ahead
    hit.discontinuous = broken
    return hit


def _advance(clock: ExcursionClock, seconds: float, **kwargs: object) -> Excursion:
    step = 0.5
    base: dict[str, object] = {
        "offset_m": 0.0,
        "heading_rad": 0.0,
        "lane_width_m": 3.5,
        "n_lanes": 2,
    }
    base.update(kwargs)
    move = clock.update(0.0, **base)  # type: ignore[arg-type]
    left = float(seconds)
    while left > 1e-9:
        dt = min(step, left)
        move = clock.update(dt, **base)  # type: ignore[arg-type]
        left -= dt
    return move


def check_clock_and_junctions() -> None:
    assert near_junction(None) is True
    assert near_junction(_hit(12.0)) is False
    assert near_junction(_hit(4.0)) is True
    assert near_junction(_hit(12.0, off=True)) is True
    assert near_junction(_hit(12.0, broken=True)) is True
    road = [RoadPoly("r", [(0.0, 0.0), (0.0, 400.0)], 7.0)]
    healthy = project_road(road, -1.75, 30.0, math.pi / 2.0)
    assert healthy is not None and near_junction(healthy) is False
    short = project_road([RoadPoly("s", [(0.0, 0.0), (0.0, 8.0)], 7.0)], -1.75, 2.0, math.pi / 2.0)
    assert short is not None and near_junction(short) is True

    clock = ExcursionClock()
    held = _advance(clock, 5.0, near_junction=True)
    assert held.phase == "cruise"
    assert clock.phase_t == 0.0
    # A junction must not stick after the road opens up again.
    released = _advance(clock, CRUISE_S, near_junction=False)
    assert released.phase == "weave"
    assert abs(released.target_offset_m) <= WEAVE_AMP_M + 1e-6
    assert abs(math.degrees(released.target_heading_rad)) <= HEADING_DEG[1]

    traffic = ExcursionClock()
    assert _advance(traffic, 40.0, traffic=True).phase == "cruise"
    weaving = _advance(ExcursionClock(), CRUISE_S)
    assert weaving.phase == "weave"
    aborted = traffic.update(
        0.1,
        offset_m=0.2,
        heading_rad=0.0,
        lane_width_m=3.5,
        traffic=True,
        n_lanes=2,
    )
    traffic.phase = "weave"
    aborted = traffic.update(
        0.1,
        offset_m=0.2,
        heading_rad=0.0,
        lane_width_m=3.5,
        traffic=True,
        n_lanes=2,
    )
    assert aborted.phase == "abort" and aborted.reason == "traffic"

    edged = ExcursionClock()
    edged.phase = "weave"
    edged = edged.update(0.1, offset_m=5.0, heading_rad=0.0, lane_width_m=3.5, n_lanes=2)
    assert edged.phase == "abort" and edged.reason == "road_edge"
    yawed = ExcursionClock()
    yawed.phase = "lane_change"
    yawed = yawed.update(0.1, offset_m=0.2, heading_rad=math.radians(25.0), lane_width_m=3.5, n_lanes=2)
    assert yawed.phase == "abort" and yawed.reason == "heading"

    single = ExcursionClock()
    # weave, cruise, then the lane-change slot, which a one-lane road turns into a weave.
    _advance(single, CRUISE_S + 12.0 + CRUISE_S, n_lanes=1)
    assert single.phase == "weave"

    recover = ExcursionClock()
    recover.phase = "recover"
    still = recover.update(1.0, offset_m=0.0, heading_rad=0.0, lane_width_m=3.5, n_lanes=2)
    assert still.phase == "recover"
    recover.update(0.5, offset_m=1.0, heading_rad=math.radians(8.0), lane_width_m=3.5, n_lanes=2)
    back = recover.update(0.5, offset_m=0.1, heading_rad=0.0, lane_width_m=3.5, n_lanes=2)
    assert back.phase == "cruise"


def _long_road() -> list[RoadPoly]:
    return [RoadPoly("r", [(0.0, 0.0), (0.0, 500.0)], 7.0)]


def _pose_hit() -> tuple[list[RoadPoly], object]:
    roads = _long_road()
    hit = project_road(roads, -1.75, 20.0, math.pi / 2.0)
    assert hit is not None and int(hit.lane_id) == 0
    return roads, hit


def check_paths_stay_on_the_road() -> None:
    roads, hit = _pose_hit()
    centers = lane_center_offsets(7.0)
    limit = 7.0 * 0.5 - 0.2
    lo, hi = HEADING_DEG
    off_lo, off_hi = RECOVER_OFFSET_M

    weave = Excursion("weave", 0.4, 0.0, 1, "")
    points, used = excursion_path(roads, -1.75, 20.0, math.pi / 2.0, weave, hit, 10.0)
    assert points is not None and used == 1
    xs = [p[0] for p in points]
    assert max(abs(x - centers[0]) for x in xs) <= WEAVE_AMP_M + 1e-3
    assert max(abs(x) for x in xs) <= limit + 1e-6
    assert max(abs(x - centers[0]) for x in xs) > 0.3

    change = Excursion("lane_change", 0.0, 0.0, -1, "")
    points, used = excursion_path(roads, -1.75, 20.0, math.pi / 2.0, change, hit, 10.0)
    assert points is not None and used == 1  # left of the left lane is off the road
    assert abs(points[-1][0] - centers[1]) < 0.15
    assert max(abs(p[0]) for p in points) <= limit + 1e-6
    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    heading = math.degrees(math.atan2(-(dx), dy))
    assert lo - 0.5 <= abs(heading) <= hi + 0.5
    spec = targets_for("lane_change", used, hit.lane_width_m, lane_id=hit.lane_id, road_width_m=7.0)
    assert spec is not None
    assert abs(spec[0]) > off_hi
    assert lo <= abs(math.degrees(spec[1])) <= hi

    recover = Excursion("recover", 0.0, 0.0, 1, "")
    points, used = excursion_path(roads, -1.75, 20.0, math.pi / 2.0, recover, hit, 8.0)
    assert points is not None
    peak = max(abs(p[0] - centers[0]) for p in points)
    assert off_lo - 1e-3 <= peak <= off_hi + 1e-3
    assert abs(points[-1][0] - centers[0]) < 0.3
    assert max(abs(p[0]) for p in points) <= limit + 1e-6
    spec = targets_for("recover", used, hit.lane_width_m, lane_id=0, road_width_m=7.0)
    assert spec is not None
    assert off_lo <= abs(spec[0]) <= off_hi
    assert lo <= abs(math.degrees(spec[1])) <= hi

    narrow = [RoadPoly("n", [(0.0, 0.0), (0.0, 500.0)], 3.5)]
    one = project_road(narrow, 0.0, 20.0, math.pi / 2.0)
    missing, _used = excursion_path(
        narrow, 0.0, 20.0, math.pi / 2.0, Excursion("lane_change", 0.0, 0.0, 1, ""), one, 8.0,
    )
    assert missing is None

    cruise = excursion_commands(10.0, "cruise", 1)
    assert any('setMode("span")' in line for line in cruise)
    assert any('driveInLane("on")' in line for line in cruise)
    assert all("setLine" not in line for line in cruise)
    lined = excursion_commands(10.0, "weave", 1, path=points)
    assert any('setMode("manual")' in line for line in lined)
    assert any('driveInLane("off")' in line for line in lined)
    assert any(line.startswith("ai.setLine(") and line.endswith(", true)") for line in lined)
    abort = excursion_commands(10.0, "abort", 1)
    assert any('driveInLane("on")' in line for line in abort)
    assert all("setLine" not in line for line in abort)


def _segment_rows(names: list[str], count: int = 30) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    stamp = 0.0
    index = 0
    for name in names:
        for _ in range(count):
            rows.append(
                {
                    "i": index,
                    "segment": name,
                    "t": stamp,
                    "dt_s": None if index == 0 else 0.2,
                    "session": "a",
                }
            )
            stamp += 0.2
            index += 1
    return rows


def check_segment_holdout() -> None:
    one = _segment_rows(["west#0"])
    plan = pass_plan(one, 12, 6, fraction=0.2)
    assert plan and all(item["kind"] == "train" for item in plan)
    trained = {int(row["i"]) for item in plan for row in item["rows"]}
    assert trained == {int(row["i"]) for row in one}

    two = _segment_rows(["west#0", "italy#1"])
    plan = pass_plan(two, 12, 6, fraction=0.2)
    hold = [row for item in plan if item["kind"] == "holdout" for row in item["rows"]]
    train = [row for item in plan if item["kind"] == "train" for row in item["rows"]]
    assert hold and {row["segment"] for row in hold} == {"italy#1"}
    assert all(row.get("segment") != "italy#1" for row in train)
    assert all(item["incoming"] == "reset" for item in plan if item["kind"] == "holdout")
    first_ids = {int(row["i"]) for row in two if row["segment"] == "west#0"}
    assert first_ids <= {int(row["i"]) for row in train}

    names = [f"map#{i}" for i in range(6)]
    six = _segment_rows(names, count=20)
    plan = pass_plan(six, 8, 4)
    held = {row["segment"] for item in plan if item["kind"] == "holdout" for row in item["rows"]}
    assert held == {"map#4"}
    trained = {int(row["i"]) for item in plan if item["kind"] == "train" for row in item["rows"]}
    last_seg = {int(row["i"]) for row in six if row["segment"] == "map#5"}
    assert last_seg <= trained

    plain = [
        {"i": index, "t": float(index), "dt_s": None if index == 0 else 1.0, "session": "a"}
        for index in range(10)
    ]
    plain_plan = pass_plan(plain, 8, 4)
    assert any(row["i"] == 9 for item in plain_plan if item["kind"] == "holdout" for row in item["rows"])
    assert all(row["i"] != 9 for item in plain_plan if item["kind"] == "train" for row in item["rows"])


def check_smoke_rate_and_flags() -> None:
    assert abs(tick_sleep_s(0.04, 0.1) - 0.06) < 1e-9
    assert tick_sleep_s(0.25, 0.1) == 0.0
    assert abs(achieved_hz(50, 5.0) - 10.0) < 1e-9
    assert segment_at(0.0, ("west_coast_usa", "italy"), 12 * 60) == ("west_coast_usa", 0)
    assert segment_at(12 * 60, ("west_coast_usa", "italy"), 12 * 60) == ("italy", 1)
    assert segment_at(0.0, (), 60.0) == ("", 0)
    assert segment_env(0) == ("", "")
    assert segment_env(1)[1] == "clear"
    opts = AutoOptions(minutes=120, maps=",".join(DEFAULT_MAPS), excursions=True, smoke_s=60, vary_env=True)
    argv = collect_argv(opts, "collect_scene_lanes.py", python="py")
    text = " ".join(argv)
    assert "--maps" in argv and "jungle_rock_island" in text
    assert "--segment-min" in argv and "--smoke-s" in argv
    assert "--no-excursions" not in argv and "--no-vary-env" not in argv
    opts.excursions = False
    opts.vary_env = False
    argv = collect_argv(opts, "collect_scene_lanes.py", python="py")
    assert "--no-excursions" in argv and "--no-vary-env" in argv

    class _Bng:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def queue_lua_command(self, command: str) -> None:
            self.commands.append(command)

    import scripts.collect_scene_lanes as collect

    collect.time.sleep = lambda _seconds: None
    bng = _Bng()
    assert start_freeroam(bng, 'west"coast_usa') is True
    assert bng.commands == ['freeroam_freeroam.startFreeroam("westcoast_usa")']

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        assert decode_issues(root) == ["smoke wrote no state.jsonl"]
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[0, 0] = (8, 16, 24)
        writer = RawFrameWriter(root, [])
        writer.write(
            {"main": image},
            t=1.0,
            driver="beamng_ai",
            tags={"segment": "italy#0", "map": "italy", "excursion": {"phase": "weave"}},
        )
        import json

        row = json.loads((root / "state.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert row["driver"] == "beamng_ai"
        assert row["segment"] == "italy#0"
        assert row["excursion"]["phase"] == "weave"
        assert decode_issues(root) == []
        queue = RawWriteQueue(RawFrameWriter(root / "queued", []))
        try:
            assert queue.wait_empty(2.0) is True
        finally:
            queue.close()

    ui = VizUI()
    from python.collect.auto_mode import AutoSession

    ui.auto = AutoSession(script=ROOT / "scripts" / "collect_scene_lanes.py")
    ui.auto.options.maps = ",".join(DEFAULT_MAPS)
    ui.nerd_tab = "auto"
    render_panel({}, h=900, w=700, ui=ui)
    ids = [hit.get("id") for hit in ui.nerd_hits if hit.get("kind") == "auto"]
    assert "traffic" in ids and "maps" in ids and "excursion" in ids


def main() -> None:
    check_clock_and_junctions()
    check_paths_stay_on_the_road()
    check_segment_holdout()
    check_smoke_rate_and_flags()
    print("collect variety ok")


if __name__ == "__main__":
    main()
