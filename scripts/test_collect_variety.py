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


def _fast_clock():
    import python.collect.level_session as level_session

    clock = {"t": 0.0}
    previous = (level_session.time.sleep, level_session.time.monotonic)

    def sleep(seconds: float) -> None:
        clock["t"] += float(seconds or 0.0)

    def monotonic() -> float:
        return clock["t"]

    level_session.time.sleep = sleep
    level_session.time.monotonic = monotonic
    return level_session, previous


def _restore_clock(level_session, previous) -> None:
    level_session.time.sleep, level_session.time.monotonic = previous


class _NoVehicle(Exception):
    """Same name BeamNGpy uses, so the collector treats it as a session error."""


_NoVehicle.__name__ = "BNGError"


class _Ego:
    def connect(self, bng: object) -> None:
        del bng

    def is_connected(self) -> bool:
        return True

    def queue_lua_command(self, command: str, *args: object) -> None:
        del command, args

    def poll_sensors(self) -> None:
        return None


class _Vehicles:
    def __init__(self, host: "_Host") -> None:
        self.host = host

    def get_current(self) -> dict[str, _Ego]:
        if not self.host.ready:
            return {}
        return {"gvd_ego": self.host.ego}

    def get_player_vehicle_id(self) -> dict[str, str]:
        self.host.player_calls += 1
        if not self.host.ready:
            raise _NoVehicle("There is no vehicle.")
        return {"vid": "gvd_ego"}


class _ScenarioApi:
    def __init__(self, host: "_Host") -> None:
        self.host = host
        self.loaded: list[object] = []
        self.starts = 0

    def get_road_network(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        self.host.road_reads += 1
        # Freeroam is not a started scenario. Edges exist only after start().
        if self.starts < 1:
            self.host.early_road = True
            raise _NoVehicle("Need to be in a started scenario to get its DecalRoad data.")
        return {
            "roads": {
                "main": [
                    {"middle": [-717.121, 0, 118], "left": [-724, 0, 118], "right": [-710, 0, 118]},
                    {"middle": [-717.121, 500, 118], "left": [-724, 500, 118], "right": [-710, 500, 118]},
                ]
            }
        }

    def load(self, scenario: object) -> None:
        self.loaded.append(scenario)

    def start(self) -> None:
        self.starts += 1
        if self.host.spawn_works:
            self.host.ready = True


class _Host:
    def __init__(self, *, spawn_works: bool, levels: dict[str, object]) -> None:
        self.ready = False
        self.spawn_works = spawn_works
        self.ego = _Ego()
        self.player_calls = 0
        self.road_reads = 0
        self.early_road = False
        self.commands: list[str] = []
        self.levels = levels
        self.vehicles = _Vehicles(self)
        self.scenario = _ScenarioApi(self)

    def get_levels(self) -> dict[str, object]:
        return self.levels

    def queue_lua_command(self, command: str, *args: object) -> None:
        del args
        self.commands.append(str(command))
        if "startFreeroam" in command:
            self.ready = False


def check_level_load_spawns_and_switches() -> None:
    import json
    from python.sensors.cameras import CAM_IDS, load_camera_config
    from python.debug.gt_geom import RoadPoly
    from python.collect.level_session import (
        choose_spawn,
        installed_levels,
        player_vehicle,
        select_maps,
    )
    import scripts.collect_scene_lanes as collect

    cfg = load_camera_config()
    ids = [str(row.get("id")) for row in (cfg.get("cameras") or []) if isinstance(row, dict)]
    assert set(CAM_IDS) <= set(ids)
    assert len(CAM_IDS) == 8
    roads = [RoadPoly("r", [(0.0, 0.0), (0.0, 400.0)], 7.0)]
    spawn = choose_spawn("italy", roads)
    assert spawn is not None
    (x, y, z), rot = spawn
    assert abs(x) < 1e-6 and 100.0 < y < 250.0 and z > 20.0
    assert abs(rot[2] - math.sin(math.pi / 4.0)) < 1e-6

    missing, skipped = select_maps(("west_coast_usa", "italy", "utah"), {"italy", "jungle_rock_island"})
    assert missing == ["italy"] and skipped == ["west_coast_usa", "utah"]
    kept, none_skipped = select_maps(("italy",), None)
    assert kept == ["italy"] and none_skipped == []

    bare = _Host(spawn_works=False, levels={})

    class _DeadList(_Host):
        def get_levels(self) -> dict[str, object]:
            raise _NoVehicle("level list failed")

    found, note = installed_levels(_DeadList(spawn_works=False, levels={}))
    assert found is None and "BNGError" in note
    try:
        collect._player_vehicle(bare)
    except RuntimeError as exc:
        assert "BNGError" in str(exc) and "There is no vehicle" in str(exc)
    else:
        raise AssertionError("BNGError escaped")
    veh, why = player_vehicle(bare)
    assert veh is None and "BNGError" in why

    level_session, previous = _fast_clock()
    removed: list[str] = []
    attached: list[object] = []

    class _Cam:
        def __init__(self, cid: str) -> None:
            self.cid = cid

        def remove(self) -> None:
            removed.append(self.cid)

    def _attach(bng: object, vehicle: object, cfg: object) -> dict[str, _Cam]:
        del bng, cfg
        attached.append(vehicle)
        return {cid: _Cam(cid) for cid in CAM_IDS}

    previous_attach = collect._attach_cameras
    collect._attach_cameras = _attach
    try:
        host = _Host(
            spawn_works=True,
            levels={"west_coast_usa": {"levelName": "west_coast_usa"}, "italy": {}},
        )
        opened = collect.enter_map(host, "west_coast_usa", {}, {})
        assert opened.error == "", opened.error
        assert opened.vehicle is host.ego
        assert opened.roads and len(opened.cameras) == 8
        assert opened.spawn_source == "known"
        assert abs(float(opened.spawn[0]) + 717.121) < 0.05
        assert host.scenario.starts == 1
        assert host.scenario.loaded and getattr(host.scenario.loaded[-1], "level", "") == "west_coast_usa"
        started_at = getattr(host.scenario.loaded[0], "pos", None)
        if started_at is not None:
            assert abs(float(started_at[1]) - 101.0) < 0.05
        assert host.early_road is False
        assert host.player_calls >= 1
        assert not any("startFreeroam" in line for line in host.commands)
        opened2 = collect.enter_map(host, "italy", {}, opened.cameras)
        assert opened2.error == "", opened2.error
        assert opened2.vehicle is host.ego and opened2.roads
        assert opened2.spawn_source == "known"
        assert abs(float(opened2.spawn[1]) + 906.94) < 0.05
        assert len(attached) == 2 and len(opened2.cameras) == 8
        assert len(removed) == 8
        assert host.scenario.starts == 2
        assert host.road_reads >= 2
        assert host.early_road is False
        assert host.scenario.loaded[0].made and host.scenario.loaded[1].made
        assert host.scenario.loaded[0].path
        places = [line for line in host.commands if "setPositionRotation" in line]
        assert len(places) == 1, places
        assert "118.50" in places[0], places[0]
        assert "250.00" not in places[0]
        assert "0.00000,0.00000" in places[0]
        assert not any("startFreeroam" in line for line in host.commands)
        from python.collect.level_session import begin_level, resolve_spawn, spawn_is_origin

        class _Zero:
            def get_levels(self) -> dict[str, object]:
                return {"italy": {"spawnPoints": [{"pos": [0, 0, 0]}]}}

            def queue_lua_command(self, *args: object, **kwargs: object) -> str:
                del args, kwargs
                return "0.000 0.000 0.000 0 0 0 1"

        pos, _rot, source = resolve_spawn(_Zero(), "italy")
        assert source == "known" and abs(pos[1] + 906.94) < 0.05
        pos, _rot, source = resolve_spawn(_Zero(), "east_coast_usa")
        assert source == "known" and abs(pos[0] + 426.68) < 0.05
        pos, _rot, source = resolve_spawn(_Zero(), "jungle_rock_island")
        assert source == "missing" and spawn_is_origin(pos)

        class _Once(_ScenarioApi):
            def __init__(self, owner: _Host) -> None:
                super().__init__(owner)
                self.loads = 0

            def load(self, scenario: object) -> None:
                self.loads += 1
                self.loaded.append(scenario)
                if self.loads == 1:
                    raise TimeoutError("timed out")

        flaky = _Host(spawn_works=True, levels={"italy": {}})
        flaky.scenario = _Once(flaky)
        retried = begin_level(flaky, "italy")
        assert retried.error == "", retried.error
        assert flaky.scenario.loads == 2
        assert flaky.scenario.loaded[-1].made
    finally:
        collect._attach_cameras = previous_attach
        _restore_clock(level_session, previous)


def check_missing_level_and_bng_error_are_status() -> None:
    import io
    import json
    import scripts.collect_scene_lanes as collect
    import python.sensors.tech as tech

    level_session, previous = _fast_clock()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skip_host = _Host(spawn_works=False, levels={"italy": {"levelName": "italy"}})

            def _open(*args: object, **kwargs: object) -> _Host:
                del args, kwargs
                return skip_host

            tech.open_tech_beamngpy = _open  # type: ignore[assignment]
            tech.widen_socket_timeout = lambda *a, **k: None  # type: ignore[assignment]
            tech.release_tech_beamngpy = lambda *a, **k: None  # type: ignore[assignment]
            buf = io.StringIO()
            old_out = sys.stdout
            sys.stdout = buf
            try:
                code = collect.main(
                    ["--minutes", "1", "--maps", "utah,west_coast_usa", "--data-dir", str(root), "--out", str(root / "scene"), "--smoke-s", "0"]
                )
            finally:
                sys.stdout = old_out
            text = buf.getvalue()
            assert code == 2, code
            assert "collect skip level utah" in text
            assert "collect skip level west_coast_usa" in text
            status = json.loads((root / "status" / "collect.json").read_text(encoding="utf-8"))
            assert "none of the requested maps" in str(status.get("error"))
            crash = (root / "status" / "collect_crash.txt").read_text(encoding="utf-8")
            assert "none of the requested maps" in crash
            assert skip_host.player_calls == 0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dead = _Host(
                spawn_works=False,
                levels={"west_coast_usa": {"levelName": "west_coast_usa"}, "italy": {}},
            )

            def _open(*args: object, **kwargs: object) -> _Host:
                del args, kwargs
                return dead

            tech.open_tech_beamngpy = _open  # type: ignore[assignment]
            buf = io.StringIO()
            old_out = sys.stdout
            sys.stdout = buf
            try:
                code = collect.main(
                    [
                        "--minutes", "1", "--maps", "west_coast_usa",
                        "--data-dir", str(root), "--out", str(root / "scene"), "--smoke-s", "0",
                    ]
                )
            finally:
                sys.stdout = old_out
            assert code == 2, (code, buf.getvalue())
            assert dead.player_calls > 1
            assert dead.scenario.starts == 1
            assert dead.early_road is False
            status = json.loads((root / "status" / "collect.json").read_text(encoding="utf-8"))
            crash = (root / "status" / "collect_crash.txt").read_text(encoding="utf-8")
            assert "BNGError" in str(status.get("error"))
            assert "There is no vehicle" in str(status.get("error"))
            assert "BNGError" in crash and "There is no vehicle" in crash
            assert "Traceback" not in buf.getvalue()
    finally:
        _restore_clock(level_session, previous)


def check_map_preflight() -> None:
    import io
    import scripts.collect_scene_lanes as collect
    import python.sensors.tech as tech

    level_session, previous = _fast_clock()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            host = _Host(
                spawn_works=True,
                levels={
                    "west_coast_usa": {"levelName": "west_coast_usa"},
                    "italy": {},
                    "east_coast_usa": {},
                },
            )

            def _open(*args: object, **kwargs: object) -> _Host:
                del args, kwargs
                return host

            tech.open_tech_beamngpy = _open  # type: ignore[assignment]
            tech.widen_socket_timeout = lambda *a, **k: None  # type: ignore[assignment]
            tech.release_tech_beamngpy = lambda *a, **k: None  # type: ignore[assignment]
            previous_attach = collect._attach_cameras

            def _attach(bng: object, vehicle: object, cfg: object) -> dict[str, object]:
                del bng, vehicle, cfg
                from python.sensors.cameras import CAM_IDS

                return {cid: object() for cid in CAM_IDS}

            collect._attach_cameras = _attach
            buf = io.StringIO()
            old_out = sys.stdout
            sys.stdout = buf
            try:
                code = collect.main(
                    [
                        "--check-maps",
                        "--maps",
                        "utah,west_coast_usa,italy",
                        "--data-dir",
                        str(root),
                        "--out",
                        str(root / "scene"),
                    ]
                )
            finally:
                sys.stdout = old_out
                collect._attach_cameras = previous_attach
            text = buf.getvalue()
            assert code == 0, (code, text)
            assert "collect skip level utah" in text
            assert "collect check west_coast_usa" in text and "source=known" in text
            assert "collect check italy" in text and "source=known" in text and "cameras=8" in text
            logs = list((root / "logs").glob("check_maps_*.txt"))
            assert len(logs) == 1, [path.name for path in (root / "logs").glob("*")]
            assert "collect check west_coast_usa" in logs[0].read_text(encoding="utf-8")
            assert logs[0].name != "check_maps.txt"
            assert host.early_road is False
            assert host.scenario.starts == 2
            assert "startFreeroam" not in text
            listed = _Host(
                spawn_works=True,
                levels={
                    "west_coast_usa": {},
                    "italy": {},
                    "gridmap_v2": {},
                    "automation_test_track": {},
                    "hirochi_raceway": {},
                    "small_island": {},
                    "derby": {},
                },
            )
            tech.open_tech_beamngpy = lambda *a, **k: listed  # type: ignore[assignment]
            buf = io.StringIO()
            sys.stdout = buf
            try:
                listed_code = collect.main(
                    ["--check-maps", "--list-levels", "--data-dir", str(root), "--out", str(root / "scene")]
                )
            finally:
                sys.stdout = old_out
            listed_text = buf.getvalue()
            assert listed_code == 0, listed_text
            assert "collect level gridmap_v2 known" in listed_text
            assert "collect level derby no-spawn-table" in listed_text
            assert "collect level automation_test_track no-spawn-table" in listed_text
            assert listed.scenario.starts == 0
    finally:
        _restore_clock(level_session, previous)


def check_windows_and_piece_boundaries() -> None:
    """A decal split is not a junction, and a dead end does not start a weave."""
    from python.collect.excursion import route_is_open
    from python.debug.gt_geom import RoadFollow, follow_label

    first = RoadPoly("a", [(0.0, float(i)) for i in range(0, 41)], 7.0)
    second = RoadPoly("b", [(0.0, 40.0 + float(i)) for i in range(0, 200)], 7.0)
    roads = [first, second]
    hit = project_road(roads, -1.75, 36.0, math.pi / 2.0)
    assert hit is not None
    assert near_junction(hit) is True
    assert near_junction(hit, roads, -1.75, 36.0, math.pi / 2.0) is False
    assert route_is_open(roads, hit, -1.75, 36.0, math.pi / 2.0, "weave", 10.0)
    points, _used = excursion_path(
        roads, -1.75, 36.0, math.pi / 2.0, Excursion("weave", 0.4, 0.0, 1, ""), hit, 10.0,
    )
    assert points is not None and points[-1][1] > 40.0

    dead = [RoadPoly("short", [(0.0, 0.0), (0.0, 30.0)], 7.0)]
    dead_hit = project_road(dead, 0.0, 5.0, math.pi / 2.0)
    assert route_is_open(dead, dead_hit, 0.0, 5.0, math.pi / 2.0, "weave", 10.0) is False
    clock = ExcursionClock()
    held = _advance(clock, CRUISE_S, window_ok=False)
    assert held.phase == "cruise"

    narrow = RoadPoly("n", [(2.0, 0.0), (2.0, 80.0)], 3.0)
    wide = RoadPoly("w", [(0.0, 0.0), (0.0, 80.0)], 12.0)
    follow = RoadFollow()
    chosen, _mem, _jump = follow_label([narrow, wide], 3.2, 20.0, math.pi / 2.0, follow, 8.0, None)
    assert chosen is not None and chosen.road_id == "w", chosen
    nearest = project_road([narrow, wide], 3.2, 20.0, math.pi / 2.0)
    assert nearest is not None and nearest.road_id == "n"
    one = [RoadPoly("only", [(0.0, 0.0), (0.0, 400.0)], 3.5)]
    one_hit = project_road(one, 0.0, 20.0, math.pi / 2.0)
    assert route_is_open(one, one_hit, 0.0, 20.0, math.pi / 2.0, "weave", 8.0)
    assert route_is_open(one, one_hit, 0.0, 20.0, math.pi / 2.0, "lane_change", 8.0) is False
    print("windows skip dead ends and a one-lane weave still fits")


def main() -> None:
    check_clock_and_junctions()
    check_paths_stay_on_the_road()
    check_windows_and_piece_boundaries()
    check_segment_holdout()
    check_smoke_rate_and_flags()
    check_level_load_spawns_and_switches()
    check_missing_level_and_bng_error_are_status()
    check_map_preflight()
    print("collect variety ok")


if __name__ == "__main__":
    main()
