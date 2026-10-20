#!/usr/bin/env python3
"""Collector fixes: paint-coverage spawns, impacts, settle rows, tip check, damage,
spawn-road repeat counts and the mid-spawn coverage recheck, road types
(bridge/tunnel) on picks and rechecks, and no repeated spawns on resume.

Offline. ``lane_at`` and the speed trace are mocked; no BeamNG is needed.
"""
from __future__ import annotations

import gzip
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect.liveness import (  # noqa: E402
    IMPACT_STALL_S,
    PRE_IMPACT_S,
    SETTLE_FLAG_S,
    TIP_DIR_Z_LIVE,
    TIP_Z,
    LIVE_YIELD_MAX_CUTS,
    ImpactWatch,
    Liveness,
    SettleWindow,
    YieldGuard,
    car_is_driving,
    is_tipped,
)
from python.collect.spawn_mix import (  # noqa: E402
    CITY,
    HIGHWAY,
    RAMP,
    REPEAT_FLAG_PICKS,
    BRIDGE,
    TUNNEL,
    SpawnDeck,
    SpawnPick,
    chain_from_pose,
    paint_coverage,
    road_ahead,
    sample_along,
)
from python.debug.gt_geom import RoadPoly  # noqa: E402


# --------------------------------------------------------------------- (a)
class _Paint:
    """``lane_at`` stand-in: a lane exists wherever ``painted(x, y)`` is true."""

    def __init__(self, painted) -> None:
        self.painted = painted
        self.calls = 0

    def __bool__(self) -> bool:
        return True

    def lane_at(self, pos, direction, ys=(5.0, 10.0, 20.0, 30.0), tracker=None):
        self.calls += 1
        out = []
        for y in ys:
            x = float(pos[0]) + float(direction[0]) * float(y)
            yy = float(pos[1]) + float(direction[1]) * float(y)
            out.append((-1.75, 1.75) if self.painted(x, yy) else None)
        return out


def _road(rid: str, pts, *, lanes: int = 2, oneway: bool = False) -> dict:
    pts = [(float(x), float(y), 10.0) for x, y in pts]
    length = sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in zip(pts, pts[1:]))
    return {"id": rid, "pts": pts, "lanes": lanes, "length": length, "oneway": oneway}


def check_coverage_walks_ahead() -> None:
    # A 60 m road joined to a 340 m road. Paint only on x < 100.
    a = _road("a", [(0, 0), (30, 0), (60, 0)])
    b = _road("b", [(60, 0), (200, 0), (400, 0)])
    side = _road("side", [(60, 0), (60, 200)])  # a 90 deg turn is not "ahead"
    deck = SpawnDeck({CITY: [a, b, side], HIGHWAY: [], RAMP: []}, painted=_Paint(lambda x, y: x < 100.0), min_paint_cov=0.5)
    chain = road_ahead(a, deck._by_id, deck._ends)
    assert abs(chain[0][0] - 24.0) < 1e-6, chain[0]  # the spawn point is 40% along
    assert chain[-1][0] == 400.0 and all(abs(p[1]) < 1e-9 for p in chain), chain
    samples = sample_along(chain)
    assert len(samples) == 30 and abs(samples[1][0] - 34.0) < 1e-6
    cov = deck.coverage(a)
    # Probes at x = 24..314; lane 10 m ahead exists for x + 10 < 100 -> 24..84 = 7 of 30.
    assert abs(cov - 7 / 30) < 1e-9, cov
    # A one-way road is not walked against its direction.
    back = _road("back", [(400, 0), (60, 0)], oneway=True)
    deck2 = SpawnDeck({CITY: [a, back], HIGHWAY: [], RAMP: []})
    assert road_ahead(a, deck2._by_id, deck2._ends)[-1][0] == 60.0
    assert paint_coverage(_Paint(lambda x, y: True), []) == 0.0
    print(f"coverage ahead ok cov={cov:.3f}")


def check_short_probe_is_not_full_coverage() -> None:
    """rec3 #1/#3 ramps: the walk ahead stopped after 9-14 m (1-2 probes), so a
    painted stub scored 1.00. Missing probes under 100 m now count as unpainted."""
    stub = _road("stub", [(0, 0), (25, 0)])  # spawn at 10 m, walk 15 m, nothing joins
    everywhere = _Paint(lambda x, y: True)
    deck = SpawnDeck({CITY: [stub], HIGHWAY: [], RAMP: []}, painted=everywhere, min_paint_cov=0.5)
    assert len(sample_along(road_ahead(stub, deck._by_id, deck._ends))) == 2
    assert abs(deck.coverage(stub) - 0.2) < 1e-9, deck.coverage(stub)
    old = SpawnDeck({CITY: [stub], HIGHWAY: [], RAMP: []}, painted=everywhere, min_paint_cov=0.5, min_probe_m=0.0)
    assert old.coverage(stub) == 1.0
    # A walk of 100 m or more is scored as before.
    long_road = _road("long", [(0, 0), (500, 0)])
    deck = SpawnDeck({CITY: [long_road], HIGHWAY: [], RAMP: []}, painted=everywhere, min_paint_cov=0.5)
    assert deck.coverage(long_road) == 1.0
    assert paint_coverage(everywhere, [(0.0, 0.0, 0.0, 1.0, 0.0)], min_samples=10) == 0.1
    print("short probe: a 15 m walk scores 0.20, not 1.00; 100 m+ unchanged")


def _pools(low: int, high: int, kind: str = CITY) -> dict:
    roads = [_road(f"low{j}", [(0, 3000 * j), (400, 3000 * j)]) for j in range(low)]
    roads += [_road(f"high{j}", [(0, -5000 - 3000 * j), (400, -5000 - 3000 * j)]) for j in range(high)]
    pools = {CITY: [], HIGHWAY: [], RAMP: []}
    pools[kind] = roads
    return pools


PAINT_SOUTH = _Paint(lambda x, y: y < -1000.0)  # only the "high" roads are painted


def check_deck_skips_and_logs() -> None:
    log: list[str] = []
    deck = SpawnDeck(_pools(5, 1), painted=PAINT_SOUTH, min_paint_cov=0.5, log=log.append)
    pick = deck.take()
    assert pick is not None and pick.road_id == "high0", pick
    assert pick.paint_cov is not None and pick.paint_cov >= 0.5
    skips = [line for line in log if line.startswith("spawn skip")]
    assert len(skips) == deck.skipped and all("paint_cov=0.00" in line for line in skips), log
    assert log[-1].startswith("spawn pick city road=high0 paint_cov=1.00"), log
    # Skipped roads are spent: the next take never offers them again.
    assert all(f"low{j}" in deck.used for j in range(5) if any(f"road=low{j} " in s for s in skips))
    print(f"deck skip ok skipped={deck.skipped} log={log[-1]!r}")


def check_deck_disabled_is_unchanged() -> None:
    paint = _Paint(lambda x, y: False)
    plain = SpawnDeck(_pools(4, 1))
    off = SpawnDeck(_pools(4, 1), painted=paint, min_paint_cov=0.0)
    for _ in range(5):
        a, b = plain.take(), off.take()
        assert a == b and a.paint_cov is None
    assert paint.calls == 0
    assert plain.take() is None and off.take() is None
    print("deck with min_paint_cov 0 matches the old deck")


def check_deck_fallback_and_next_kind() -> None:
    log: list[str] = []
    deck = SpawnDeck(_pools(6, 0), painted=PAINT_SOUTH, min_paint_cov=0.5, max_tries=2, log=log.append)
    pick = deck.take()
    assert pick is not None and pick.paint_cov == 0.0
    assert deck.skipped == 2 and len(deck.used) == 2, deck.used
    assert log[-1].startswith("spawn fallback"), log
    # City all unpainted, highway painted: the city budget runs out, then highway.
    pools = _pools(4, 0)
    pools[HIGHWAY] = _pools(0, 1, HIGHWAY)[HIGHWAY]
    deck = SpawnDeck(pools, painted=PAINT_SOUTH, min_paint_cov=0.5, max_tries=3)
    pick = deck.take()
    assert pick is not None and pick.kind == HIGHWAY and pick.paint_cov == 1.0, pick
    assert deck.counts[HIGHWAY] == 1 and deck.counts[CITY] == 0
    print("deck fallback ok")


def check_real_west_coast_if_present() -> None:
    nav = Path(r"E:\GVD_data\roads\west_coast_usa.json.gz")
    lines = Path(r"E:\GVD_data\roads\west_coast_usa_lines.json.gz")
    if not (nav.is_file() and lines.is_file()):
        print("real west coast deck skipped (no E:\\GVD_data\\roads dump)")
        return
    from python.debug.painted_lanes import PaintedLanes

    with gzip.open(lines, "rt", encoding="utf-8") as fh:
        painted = PaintedLanes(json.load(fh))
    deck = SpawnDeck.load(nav, painted=painted, min_paint_cov=0.5)
    picks = [deck.take() for _ in range(20)]
    assert all(p is not None and p.paint_cov >= 0.5 for p in picks), [p.paint_cov for p in picks]
    assert "890092.0" not in {p.road_id for p in picks}  # the drift loop at (-1119, 1940)
    print(f"real west coast: 20 picks >= 0.50, skipped {deck.skipped}")


# --------------------------------------------------------------------- (b)
def check_impact_watch() -> None:
    watch = ImpactWatch()
    t = 0.0
    hits = []
    trace = [15.0] * 30 + [14.5] + [14.0] * 10 + [9.0] + [8.0] + [0.2] * 25
    for i, speed in enumerate(trace):
        t += 0.1
        got = watch.update(0.1, speed)
        if got is not None:
            hits.append((i, round(got, 1), watch.pre_window()))
        watch.note_row(t, i)
    # 14 -> 9 m/s in 0.1 s is 50 m/s^2. 15 -> 14.5 (5 m/s^2) is braking, not an impact.
    # 8 -> 0.2 a tick later is inside the 2 s debounce.
    assert [h[0] for h in hits] == [41], hits
    assert hits[0][1] == 50.0
    pre_t, pre_i = hits[0][2]
    assert pre_i == 41 - int(round(PRE_IMPACT_S / 0.1)), hits  # first row inside the 3 s run-up
    assert abs(pre_t - (pre_i + 1) * 0.1) < 1e-6
    assert watch.should_respawn() and watch.stall_s >= IMPACT_STALL_S
    # A long gap (teleport, a stall in the loop) is not a deceleration.
    gap = ImpactWatch()
    gap.update(0.1, 20.0)
    assert gap.update(1.0, 0.0) is None and not gap.open
    # Driving on after an impact closes it.
    on = ImpactWatch()
    on.update(0.1, 12.0)
    assert on.update(0.1, 9.0) is not None
    for _ in range(85):
        on.update(0.1, 9.0)
    assert on.drove_off() and not on.should_respawn()
    on.close()
    assert not on.open
    print(f"impact watch ok hit={hits[0][:2]} pre_i0={pre_i}")


# --------------------------------------------------------------------- (c)
def check_settle_window() -> None:
    win = SettleWindow()
    assert not win.active(0.0)
    win.start(100.0)
    assert win.active(100.0) and win.active(100.0 + SETTLE_FLAG_S - 0.05)
    assert not win.active(100.0 + SETTLE_FLAG_S)
    win.start(500.0)
    assert win.active(501.0)
    print("settle window ok")


# --------------------------------------------------------------------- (d)
def check_tip_rules() -> None:
    # 20 deg grade: dir_z 0.34, up_z 0.94. The old rule called that tipped.
    assert is_tipped(0.34)
    assert not is_tipped(0.34, up_z=0.94)
    assert not is_tipped(0.34, tip_z=TIP_DIR_Z_LIVE)
    assert is_tipped(0.0, up_z=0.2)  # on its side
    assert is_tipped(0.97, tip_z=TIP_DIR_Z_LIVE)  # nose up, no up vector
    assert car_is_driving(8.0, 0.34, 0.94, TIP_DIR_Z_LIVE)
    assert not car_is_driving(8.0, 0.0, -0.9, TIP_DIR_Z_LIVE)
    live = Liveness()
    assert live.update(0.1, 8.0, 0.34, None, ai_should_move=True, up_z=0.94, tip_z=TIP_DIR_Z_LIVE) == ""
    assert live.update(0.1, 8.0, 0.34, None, ai_should_move=True) == "tipped"  # default rule kept
    assert TIP_Z == 0.3
    print("tip rules ok")


def check_weave_guard_helpers() -> None:
    import scripts.collect_scene_lanes as collect
    from python.debug.lane_targets import tech_from_painted

    full = tech_from_painted([(-1.7, 1.7)] * 4)
    half = tech_from_painted([(-1.7, 1.7), (-1.7, 1.7), None, None])
    none = tech_from_painted([None] * 4)
    assert collect.paint_share(full) == 1.0 and collect.paint_share(half) == 0.5 and collect.paint_share(none) == 0.0
    assert collect.weave_guard_reason(1.0, None, 0.5, 40.0) == "open"
    assert collect.weave_guard_reason(0.25, None, 0.5, 40.0) == "paint_ahead"
    assert collect.weave_guard_reason(1.0, 24.0, 0.5, 40.0) == "junction_ahead"
    assert collect.weave_guard_reason(1.0, 48.0, 0.5, 40.0) == "open"
    assert abs(collect.polyline_length([(0, 0), (3, 4), (3, 10)]) - 11.0) < 1e-9
    args = collect.build_parser().parse_args([])
    assert args.min_paint_cov == 0.5 and args.weave_guard is False
    args = collect.build_parser().parse_args(["--min-paint-cov", "0", "--weave-guard"])
    assert args.min_paint_cov == 0.0 and args.weave_guard is True
    assert "impact" in collect.HARD_SEGMENT_REASONS
    print("weave guard helpers ok")


def check_paint_share_slot_order() -> None:
    """paint_share must read the center slot of lane_gt_valid, as lane_targets writes it."""
    import math

    import scripts.collect_scene_lanes as collect
    from python.debug.lane_targets import (
        LOOKAHEADS_M,
        SLOT_CENTER,
        SLOT_LEFT,
        SLOT_RIGHT,
        lookahead_slot,
        tech_from_painted,
    )

    # Distinct left/right/center at every lookahead, so each value has one home.
    lanes = [(-1.0 - 0.3 * k, 2.0 + 0.7 * k) for k in range(len(LOOKAHEADS_M))]
    tech = tech_from_painted(lanes)
    values, valid = list(tech["lane_gt"]), list(tech["lane_gt_valid"])
    triples = 3 * len(LOOKAHEADS_M)
    for k, (left, right) in enumerate(lanes):
        for part, want in ((SLOT_CENTER, 0.5 * (left + right)), (SLOT_LEFT, left), (SLOT_RIGHT, right)):
            # Where the real writer put it, found by value, not by the helper.
            home = [i for i in range(triples) if abs(values[i] - want) < 1e-5]
            assert home == [lookahead_slot(k, part)], (k, part, home)
            assert valid[home[0]] == 1.0  # the mask uses the same slot as the value
    assert collect.paint_share(tech) == 1.0
    # Lookahead 2: NaN center, left and right still valid. Not a usable lane.
    centre = [i for i in range(triples) if abs(values[i] - 0.5 * sum(lanes[2])) < 1e-5][0]
    values[centre] = float("nan")
    valid[centre] = 0.0
    side = [i for i in range(triples) if abs(values[i] - lanes[2][0]) < 1e-5 or abs(values[i] - lanes[2][1]) < 1e-5]
    assert len(side) == 2 and all(valid[i] == 1.0 for i in side)
    broken = dict(tech, lane_gt=values, lane_gt_valid=valid)
    assert math.isnan(broken["lane_gt"][lookahead_slot(2, SLOT_CENTER)])
    assert collect.paint_share(broken) == 0.75, collect.paint_share(broken)
    # The writer's own "no paint at 20 m" vector agrees.
    missing = tech_from_painted([lanes[0], lanes[1], None, lanes[3]])
    assert collect.paint_share(missing) == 0.75
    print("paint_share reads the center slot (NaN center with valid sides counts invalid)")


def check_spawn_mix_summary() -> None:
    import scripts.collect_scene_lanes as collect

    pools = _pools(6, 0)
    pools[HIGHWAY] = _pools(0, 2, HIGHWAY)[HIGHWAY]
    deck = SpawnDeck(pools, painted=PAINT_SOUTH, min_paint_cov=0.5, max_tries=2)
    first = deck.take()  # city wanted, unpainted: 2 skips, then a painted highway
    assert first.kind == HIGHWAY and deck.cross_kind == 1 and deck.best_seen == 0
    second = deck.take()  # city still behind its share: 2 more skips, then highway again
    assert second.kind == HIGHWAY and deck.cross_kind == 2
    third = deck.take()  # no highway left: the best-covered skipped city road
    assert third.kind == CITY and third.paint_cov == 0.0 and deck.best_seen == 1 and deck.cross_kind == 2
    s = deck.summary()
    assert s["picks"] == 3 and s["counts"] == {CITY: 1, HIGHWAY: 2, RAMP: 0}, s
    assert s["cross_kind_fallback"] == deck.cross_kind and s["skipped"] == deck.skipped
    assert len(deck.select_ms) == 3 and deck.last_ms >= 0.0 and s["select_ms_max"] >= s["select_ms_mean"]
    line = deck.summary_line()
    assert line.startswith("spawn mix picks=3 city=1 highway=2 ramp=0 ") and "best_seen=1" in line, line
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "collect.json"
        collect.save_progress(path, 12.5, 60.0, {"spawn_mix": s})
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["spawn_mix"]["counts"] == s["counts"]
        assert collect.load_progress(path) == 12.5
        collect.save_progress(path, 3.0, 60.0)
        assert set(json.loads(path.read_text(encoding="utf-8"))) == {"done_s", "target_s"}
    print(f"spawn mix summary ok: {line}")


# --------------------------------------------------------------------- (e)
class _Sensors(dict):
    pass


class _DamageVehicle:
    def __init__(self) -> None:
        self.state = {"pos": [1.0, 2.0, 3.0], "dir": [0.0, 1.0, 0.1], "up": [0.0, -0.1, 0.99], "vel": [0.0, 5.0, 0.0]}
        self.sensors = _Sensors(damage={"damage": 512.0, "part_damage": {}})

    def poll_sensors(self) -> None:
        pass


def check_damage_and_up_are_read() -> None:
    import scripts.collect_scene_lanes as collect

    sample = collect._read_vehicle(_DamageVehicle())
    assert sample["damage"] == 512.0, sample
    assert sample["up"] == [0.0, -0.1, 0.99]
    print("damage sensor and up vector read")


# ------------------------------------------------------ collector, end to end
class _CrashVehicle:
    """10 m/s, a crash to 0.2 m/s, stopped, then driving again, then the stop file."""

    def __init__(self, stop: Path, speed_at=None, stop_at: int = 120, damage_at=None, x: float = 0.4,
                 bbox_w: float | None = 1.9, model: str | None = "etk800") -> None:
        self.stop = stop
        self.x = float(x)
        self.bbox_w = bbox_w
        self.model = model
        self.bbox_calls = 0
        self.speed_at = speed_at or (lambda n: 10.0 if n <= 40 or n > 75 else 0.2)
        self.damage_at = damage_at
        self.stop_at = int(stop_at)
        self.polls = 0
        self.commands: list[str] = []
        self.state: dict = {}
        self._set(10.0)

    def get_bbox(self) -> dict:
        """Vehicle.get_bbox stand-in: an axis-aligned box ``bbox_w`` wide; None raises like a BNGError."""
        self.bbox_calls += 1
        if self.bbox_w is None:
            raise RuntimeError("The request was not handled by BeamNG")
        x, y, z = self.x, 12.0, 0.5
        h = 0.5 * float(self.bbox_w)
        box = {}
        for end, dy in (("front", 2.3), ("rear", -2.3)):
            for side, dx in (("left", -h), ("right", h)):
                box[f"{end}_bottom_{side}"] = (x + dx, y + dy, z - 0.5)
                box[f"{end}_top_{side}"] = (x + dx, y + dy, z + 1.0)
        return box

    def _set(self, speed: float) -> None:
        self.state = {
            "pos": [self.x, 12.0, 0.5],
            "dir": [0.0, 1.0, 0.0],
            "up": [0.0, 0.0, 1.0],
            "electrics": {"wheelspeed": speed, "steering_input": 0.0, "throttle_input": 0.3, "brake_input": 0.0},
        }

    def poll_sensors(self) -> None:
        self.polls += 1
        self._set(float(self.speed_at(self.polls)))
        if self.damage_at is not None:
            self.state["damage"] = float(self.damage_at(self.polls))
        if self.polls >= self.stop_at:
            self.stop.write_text("stop", encoding="utf-8")

    def queue_lua_command(self, command: str, *args: object) -> None:
        del args
        self.commands.append(str(command))

    def connect(self, bng: object) -> None:
        del bng

    def is_connected(self) -> bool:
        return True


class _Api:
    def __init__(self, vehicle) -> None:
        self.vehicle = vehicle

    def get_current(self):
        return {"ego": self.vehicle}

    def get_player_vehicle_id(self):
        return {"vid": "ego"}


class _Bng:
    def __init__(self, vehicle) -> None:
        self.vehicles = _Api(vehicle)
        self.commands: list[str] = []

    def queue_lua_command(self, command: str, *args: object) -> None:
        del args
        self.commands.append(str(command))

    def get_road_network(self, **kwargs: object):
        del kwargs
        return {"roads": {"main": [{"middle": [0.0, 0.0, 0.4]}, {"middle": [0.0, 80.0, 0.4]}]}}


class _Cam:
    def poll(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[0, 0] = (8, 16, 24)
        return {"colour": image}


def _run_collector(vehicle_args: dict, extra: list[str], root: Path | None = None,
                   cams=None, roads=None) -> tuple[int, str, Path]:
    import python.data.raw_queue as raw_queue
    import python.debug.gt_logger as gt_logger
    import python.sensors.tech as tech
    import scripts.collect_scene_lanes as collect
    from python.collect.auto_mode import stop_file

    base_queue = raw_queue.RawWriteQueue

    class _DeepQueue(base_queue):  # the mocked loop outruns the PNG workers
        def __init__(self, writer, max_depth: int = 64, **kwargs) -> None:
            del max_depth
            super().__init__(writer, 100000, **kwargs)

    root = Path(root) if root is not None else Path(tempfile.mkdtemp())
    out = root / "scene"
    vehicle = _CrashVehicle(stop_file(out), **vehicle_args)
    bng = _Bng(vehicle)
    previous = (
        tech.open_tech_beamngpy,
        tech.widen_socket_timeout,
        tech.release_tech_beamngpy,
        gt_logger.load_road_polys,
        collect._attach_cameras,
        collect.time.sleep,
        collect.time.monotonic,
        collect.tick_sleep_s,
    )
    clock = {"t": 0.0}

    def sleep(seconds: float) -> None:
        clock["t"] += float(seconds or 0.0)

    tech.open_tech_beamngpy = lambda *a, **k: bng  # type: ignore[assignment]
    tech.widen_socket_timeout = lambda *a, **k: None  # type: ignore[assignment]
    tech.release_tech_beamngpy = lambda *a, **k: None  # type: ignore[assignment]
    level_roads = roads if roads is not None else [RoadPoly("r", [(0.0, 0.0), (0.0, 80.0)], 7.0)]
    gt_logger.load_road_polys = lambda session: (list(level_roads), "")  # type: ignore[assignment]
    collect._attach_cameras = lambda session, veh, cfg: (cams() if cams else {"main": _Cam()})  # type: ignore[assignment]
    collect.time.sleep = sleep  # type: ignore[method-assign]
    collect.time.monotonic = lambda: clock["t"]  # type: ignore[method-assign]
    collect.tick_sleep_s = lambda _elapsed, _period=0.1: 0.1  # type: ignore[method-assign]
    raw_queue.RawWriteQueue = _DeepQueue  # type: ignore[misc]
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = collect.main(
            ["--minutes", "5", "--smoke-s", "0", "--data-dir", str(root), "--out", str(out),
             "--host", "127.0.0.1", "--port", "1"] + list(extra)
        )
    finally:
        sys.stdout = old
        (
            tech.open_tech_beamngpy,
            tech.widen_socket_timeout,
            tech.release_tech_beamngpy,
            gt_logger.load_road_polys,
            collect._attach_cameras,
            collect.time.sleep,
            collect.time.monotonic,
            collect.tick_sleep_s,
        ) = previous
        raw_queue.RawWriteQueue = base_queue  # type: ignore[misc]
    return code, buf.getvalue(), out


def check_collector_impact_rows() -> None:
    code, text, out = _run_collector({}, ["--no-excursions", "--no-impact-respawn"])
    assert code == 0, (code, text)
    assert "collect impact current#0 decel=98.0 m/s2 v=10.0->0.2" in text, text
    assert "respawning" in text and "collect abort impact" in text and "collect recover" in text, text
    assert "collect abort stationary" not in text
    rows = [json.loads(line) for line in (out / "state.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    hit = [row for row in rows if row.get("impact")]
    assert len(hit) == 1 and hit[0]["impact_decel_mps2"] == 98.0, hit
    i_hit = int(hit[0]["i"])
    stalled = [row for row in rows if row.get("stall")]
    assert stalled and min(int(r["i"]) for r in stalled) == i_hit, [r["i"] for r in stalled]
    # The impact is about 2 s of stall before the respawn, not 10 s.
    assert len(stalled) <= int(IMPACT_STALL_S / 0.1) + 2, len(stalled)
    rec = [row for row in rows if row.get("recover")]
    assert len(rec) == 1
    i_rec = int(rec[0]["i"])
    settled = [int(row["i"]) for row in rows if row.get("settle")]
    # 3 s of 0.1 s rows; the float clock may let one more row in.
    assert settled[0] == i_rec and len(settled) - int(round(SETTLE_FLAG_S / 0.1)) in (0, 1), settled
    assert settled == list(range(i_rec, i_rec + len(settled)))
    assert not any(row.get("settle") for row in rows if int(row["i"]) < i_rec)
    assert all(row["ego"].get("up") == [0.0, 0.0, 1.0] for row in rows)
    events = [json.loads(line) for line in (out / "impacts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["kind"] for e in events] == ["impact", "impact_end"], events
    ev, end = events
    assert ev["i"] == i_hit and abs(ev["t"] - hit[0]["t"]) < 1e-9, (ev, hit[0])
    pre = [row for row in rows if row["i"] == ev["pre_i0"]]
    assert pre and abs(pre[0]["t"] - ev["pre_t0"]) < 1e-9
    assert i_hit - ev["pre_i0"] == int(round(PRE_IMPACT_S / 0.1)), ev
    assert end["how"] == "respawn" and end["t"] >= ev["t"]
    assert "abort_t0" in ev and ev["abort_t0"] is None, ev  # no abort phase before this impact
    print(f"collector impact rows ok impact i={i_hit} pre_i0={ev['pre_i0']} recover i={i_rec} settle={len(settled)} rows")


def check_weave_guard_runs() -> None:
    """No paint on this mocked level: no painted lane at the car, so no lane edge to weave inside.
    The weave is blocked (no_paint) with or without --weave-guard."""
    code, text, out = _run_collector({"speed_at": lambda n: 10.0, "stop_at": 260}, ["--weave-guard"])
    assert code == 0, (code, text)
    assert "collect gate current#0 weave blocked no_paint" in text, text[-3000:]
    assert "excursion=weave" not in text
    code, text, out = _run_collector({"speed_at": lambda n: 10.0, "stop_at": 260}, [])
    assert code == 0, (code, text)
    assert "collect gate current#0 weave blocked no_paint" in text, text[-3000:]
    assert "excursion=weave" not in text and "line end" not in text
    # Painted everywhere: the guard lets the weave start, then ends it before
    # the line runs out instead of after the 12 s weave. The car does 10 m/s on
    # an 8 m/s target, so the weave is planned at 10: a 50 m line, 4.5 s to the
    # 5 m line-end margin (it was 60 m / 5.5 s when planned at the 8 m/s target).
    import scripts.collect_scene_lanes as collect

    previous = collect.painted_for
    collect.painted_for = lambda level: _Paint(lambda x, y: True)  # type: ignore[assignment]
    try:
        # In the right lane (center x=+1.75 on the 7 m road heading +y): at x=0.4 the body
        # straddles the centerline and the body-edge rule aborts the weave at once.
        code, text, out = _run_collector({"speed_at": lambda n: 10.0, "stop_at": 300, "x": 1.75}, ["--weave-guard"])
    finally:
        collect.painted_for = previous  # type: ignore[assignment]
    assert code == 0, (code, text)
    assert "collect gate current#0 weave open" in text, text[-3000:]
    assert "collect excursion current#0 weave line end, back to lane" in text, text[-3000:]
    assert "collect weave current#0 v_plan=10.0 v_meas=10.0 v_target=8.0" in text, text[-3000:]
    after = text.split("line end, back to lane", 1)[1]
    assert "excursion=abort" in after, after[:400]
    rows = [json.loads(line) for line in (out / "state.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    weave_rows = [row for row in rows if (row.get("excursion") or {}).get("phase") == "weave"]
    assert weave_rows and 41 <= len(weave_rows) <= 49, len(weave_rows)
    print(f"weave guard: unpainted weave blocked, off by default, line end after {len(weave_rows)} rows")


def _rows(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "state.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def check_impact_on_an_abort_tick_names_a_row() -> None:
    """The crash tick also aborts on damage, so it writes no row; the impact
    still names the last written row (rec3 had two ``i: null`` events)."""
    code, text, out = _run_collector({"damage_at": lambda n: 0.0 if n <= 40 else 1000.0}, ["--no-excursions"])
    assert code == 0, (code, text)
    assert "collect impact current#0 decel=98.0" in text and "collect abort damage" in text, text[-2000:]
    rows = _rows(out)
    events = [json.loads(line) for line in (out / "impacts.jsonl").read_text(encoding="utf-8").splitlines()]
    hits = [e for e in events if e["kind"] == "impact"]
    assert len(hits) == 1, events
    ev = hits[0]
    assert isinstance(ev["i"], int) and ev["i_row"] == "previous", ev
    by_i = {int(r["i"]): r for r in rows}
    assert ev["i"] in by_i, ev
    # The mocked loop runs faster than the wall clock, so go by index: the
    # named row is the one right before the recover row that follows the abort.
    i_rec = min(int(r["i"]) for r in rows if r.get("recover"))
    assert ev["i"] == i_rec - 1, (ev, i_rec)
    assert not any(r.get("impact") for r in rows)  # the impact tick itself has no row
    assert ev["pre_i0"] in by_i and ev["pre_i0"] < ev["i"]
    # The normal path still names the impact row itself.
    code, text, out = _run_collector({}, ["--no-excursions"])
    ev2 = [json.loads(line) for line in (out / "impacts.jsonl").read_text(encoding="utf-8").splitlines()][0]
    row = [r for r in _rows(out) if r.get("impact")][0]
    assert ev2["i"] == row["i"] and ev2["i_row"] == "impact", (ev2, row["i"])
    print(f"impact on an abort tick -> i={ev['i']} (previous row), normal impact -> its own row i={ev2['i']}")


def check_yield_guard() -> None:
    off = YieldGuard(0.0)
    assert not off.enabled and not any(off.update(0.1, 10.0, False, False) for _ in range(2000))
    g = YieldGuard(0.25)
    fired = [n for n in range(1, 700) if g.update(0.1, 10.0, False, False)]
    assert fired and fired[0] in (600, 601), fired[:3]  # a full 60 s of moving rows first
    # Settle rows, stopped rows and a missing speed are not counted.
    g = YieldGuard(0.25)
    assert not any(g.update(0.1, 10.0, False, True) for _ in range(1000))
    assert not any(g.update(0.1, 0.5, False, False) for _ in range(1000))
    assert not any(g.update(0.1, None, False, False) for _ in range(1000))
    assert g.moving_s == 0.0
    # A long tick is credited at most 0.5 s.
    g.update(5.0, 10.0, False, False)
    assert abs(g.moving_s - 0.5) < 1e-9
    # 30% valid stays; 20% valid cuts.
    for share, cuts in ((3, False), (2, True)):
        g = YieldGuard(0.25)
        fire = [g.update(0.1, 10.0, (n % 10) < share, False) for n in range(800)]
        assert any(fire) is cuts, (share, g.rate)
    # Cut empties the window; three cuts in a row pause it until a healthy window.
    g = YieldGuard(0.25)
    for k in range(LIVE_YIELD_MAX_CUTS):
        n = 0
        while not g.update(0.1, 10.0, False, False):
            n += 1
            assert n < 700
        assert n + 1 in (600, 601), n
        g.cut()
        assert g.moving_s == 0.0 and g.in_a_row == k + 1
    assert g.paused and g.total_cuts == LIVE_YIELD_MAX_CUTS
    assert not any(g.update(0.1, 10.0, False, False) for _ in range(2000))
    assert not any(g.update(0.1, 10.0, True, False) for _ in range(600))
    assert not g.paused and g.in_a_row == 0
    assert any(g.update(0.1, 10.0, False, False) for _ in range(700))
    # Reset (a spawn) empties the window without counting a cut.
    g = YieldGuard(0.25)
    for _ in range(590):
        g.update(0.1, 10.0, False, False)
    g.reset()
    assert not any(g.update(0.1, 10.0, False, False) for _ in range(590)) and g.total_cuts == 0
    import scripts.collect_scene_lanes as collect

    assert collect.low_yield_plan(True, 500.0, 100.0, 0.0) == ("respawn", 0.0)
    assert collect.low_yield_plan(False, None, 100.0, 0.0) == ("keep", 0.0)
    assert collect.low_yield_plan(False, 720.0, 100.0, 20.0) == ("segment_end", 620.0)
    tech = {"lane_gt_valid": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0] + [1.0] * 8}
    assert collect.label_valid_at(tech, 5.0) and not collect.label_valid_at(tech, 10.0)
    assert not collect.label_valid_at({}, 10.0)
    print("yield guard: off at 0, 60 s window, settle/stopped ignored, pauses after 3 cuts, re-arms")


def check_yield_guard_runs() -> None:
    """Unpainted mock level: the guard logs 3 cuts with position and roads, then
    pauses. Every row is kept. Off by default; painted everywhere, no cut."""
    args = {"speed_at": lambda n: 10.0, "stop_at": 2600}
    code, text, out = _run_collector(args, ["--no-excursions", "--min-live-yield", "0.25"])
    assert code == 0, (code, text[-2000:])
    lines = [line for line in text.splitlines() if line.startswith("collect low_yield current#0")]
    assert len(lines) == LIVE_YIELD_MAX_CUTS, lines
    assert all("valid10=0.00 < 0.25 over 60s moving pos=(0.4,12.0,0.5) road=r spawn_road=na" in s for s in lines), lines
    assert all(s.endswith("-> keep") for s in lines), lines  # no deck and no map rotation here
    assert "collect low_yield paused after 3 cuts in a row" in text
    cuts = [json.loads(line) for line in (out / "cuts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [c["kind"] for c in cuts] == ["low_yield"] * LIVE_YIELD_MAX_CUTS
    rows = _rows(out)
    ids = [int(r["i"]) for r in rows]
    assert ids == list(range(len(ids))), "rows were dropped"
    assert all(c["i"] in ids and c["road"] == "r" and c["pos"] == [0.4, 12.0, 0.5] and c["action"] == "keep" for c in cuts), cuts
    gaps = [b["done_s"] - a["done_s"] for a, b in zip(cuts, cuts[1:])]
    assert all(59.0 <= g <= 61.5 for g in gaps), gaps
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    assert status.get("live_yield", {}).get("cuts") == LIVE_YIELD_MAX_CUTS, status
    code, text_off, out_off = _run_collector(args, ["--no-excursions"])
    assert code == 0 and "low_yield" not in text_off and not (out_off / "cuts.jsonl").exists()
    assert len(_rows(out_off)) == len(rows), (len(_rows(out_off)), len(rows))
    import scripts.collect_scene_lanes as collect

    previous = collect.painted_for
    collect.painted_for = lambda level: _Paint(lambda x, y: True)  # type: ignore[assignment]
    try:
        code, text_ok, out_ok = _run_collector(args, ["--no-excursions", "--min-live-yield", "0.25"])
    finally:
        collect.painted_for = previous  # type: ignore[assignment]
    assert code == 0 and "low_yield" not in text_ok, text_ok[-1500:]
    print(f"yield guard run: {len(lines)} cuts 60 s apart then paused, {len(rows)} rows kept, off by default")


# ------------------------------------------------- (f) road repeats, recheck
def check_road_repeats() -> None:
    """Picks and recorded seconds per spawn road; a resumed run (same seed) re-picks
    the same roads, and the third pick of a road is flagged."""
    log: list[str] = []
    deck = SpawnDeck(_pools(0, 4), painted=PAINT_SOUTH, min_paint_cov=0.5, log=log.append)
    first = [deck.take() for _ in range(2)]
    deck.note_time(first[0].road_id, 30.0)
    deck.note_time(first[1].road_id, 10.0)
    deck.note_time(None, 99.0)  # no spawn road: not credited
    r = deck.road_repeats()
    assert r["roads"] == 2 and r["picks"] == 2 and r["max_picks"] == 1 and r["flagged"] == [], r
    assert r["by_road"][first[0].road_id] == {"kind": CITY, "picks": 1, "time_s": 30.0, "time_share": 0.75}, r
    assert r["time_s"] == 40.0 and r["flagged_time_share"] == 0.0
    # Two resumes from collect.json: each new deck has the same shuffle.
    prior = r
    for _ in range(2):
        again = SpawnDeck(_pools(0, 4), painted=PAINT_SOUTH, min_paint_cov=0.5)
        assert again.seed_repeats(prior) == len(prior["by_road"])
        picks = [again.take() for _ in range(2)]
        assert [p.road_id for p in picks] == [p.road_id for p in first]
        again.note_time(picks[0].road_id, 30.0)
        prior = again.road_repeats()
    assert prior["max_picks"] == REPEAT_FLAG_PICKS == 3 and prior["picks"] == 6, prior
    assert prior["flagged"] == [first[0].road_id, first[1].road_id], prior["flagged"]
    assert prior["flagged_time_share"] == 1.0 and prior["by_road"][first[0].road_id]["time_s"] == 90.0
    lines = again.repeat_lines()
    assert lines[0] == "spawn repeats roads=2 picks=6 max_picks=3 flagged=2 (>= 3 picks) flagged_time=100% of 100s", lines
    assert lines[1] == f"spawn repeat flagged road={first[0].road_id} kind=city picks=3 time=90s share=90%", lines
    assert again.seed_repeats(None) == 0 and again.seed_repeats({"by_road": "x"}) == 0
    import scripts.collect_scene_lanes as collect

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "collect.json"
        collect.save_progress(path, 5.0, 60.0, {"road_repeats": prior})
        back = json.loads(path.read_text(encoding="utf-8"))["road_repeats"]
        assert back["flagged"] == prior["flagged"] and back["by_road"][first[0].road_id]["picks"] == 3
    print(f"road repeats ok: {lines[0]}")


def check_recheck_coverage_from_pose() -> None:
    """The recheck scores the spawn probe from the car's pose, along its heading."""
    a = _road("a", [(0, 0), (200, 0), (400, 0)])
    cross = _road("cross", [(50, -200), (50, 200)])  # a cross street through the car
    paint = _Paint(lambda x, y: x < 100.0 and abs(y) < 5.0)
    deck = SpawnDeck({CITY: [a, cross], HIGHWAY: [], RAMP: []}, painted=paint, min_paint_cov=0.5)
    # At (50, 0) heading +x: probes 50..340; a lane 10 m ahead for x < 90 -> 50..80 = 4 of 30.
    cov, rid = deck.coverage_from((50.0, 0.5, 10.0), (1.0, 0.0, 0.0))
    assert rid == "a" and abs(cov - 4 / 30) < 1e-9, (rid, cov)
    chain = chain_from_pose(a, (50.0, 0.5, 10.0), (1.0, 0.0), deck._by_id, deck._ends)
    assert abs(chain[0][0] - 50.0) < 1e-9 and abs(chain[0][1]) < 1e-9 and chain[-1][0] == 400.0
    # Heading -x on the two-way road: 50 m back to its end, 6 probes (0..50 m) all
    # painted, under 100 m -> 6/10.
    cov, rid = deck.coverage_from((50.0, 0.5, 10.0), (-1.0, 0.0, 0.0))
    assert rid == "a" and abs(cov - 0.6) < 1e-9, (rid, cov)
    # Heading +y at the same spot: the cross street, which is unpainted off y = 0.
    cov, rid = deck.coverage_from((50.0, 0.5, 10.0), (0.0, 1.0, 0.0))
    assert rid == "cross" and cov < 0.1, (rid, cov)
    # Off every road (a lot), the filter off, or no pose: not scored.
    assert deck.coverage_from((150.0, 60.0, 10.0), (1.0, 0.0, 0.0)) == (None, None)
    assert deck.coverage_from(None, (1.0, 0.0, 0.0)) == (None, None)
    assert deck.coverage_from((50.0, 0.0, 10.0), (0.0, 0.0, 1.0)) == (None, None)
    off = SpawnDeck({CITY: [a], HIGHWAY: [], RAMP: []}, painted=paint, min_paint_cov=0.0)
    assert off.coverage_from((50.0, 0.0, 10.0), (1.0, 0.0, 0.0)) == (None, None)
    # The recheck reads the same probe as the spawn: at the spawn pose it gives the spawn score.
    long_road = _road("long", [(0, 0), (500, 0)])
    every = SpawnDeck({CITY: [long_road], HIGHWAY: [], RAMP: []}, painted=_Paint(lambda x, y: x < 300.0), min_paint_cov=0.5)
    assert abs(every.coverage_from((200.0, 0.0, 10.0), (1.0, 0.0, 0.0))[0] - every.coverage(long_road)) < 1e-9
    import scripts.collect_scene_lanes as collect

    assert collect.build_parser().parse_args([]).recheck_cov_min == 0.0
    assert collect.build_parser().parse_args(["--recheck-cov-min", "3"]).recheck_cov_min == 3.0
    assert not collect.recheck_due(0.0, 9999.0, 0.0)
    assert not collect.recheck_due(180.0, 179.0, 0.0) and collect.recheck_due(180.0, 180.0, 0.0)
    assert collect.recheck_due(180.0, 400.0, 220.0) and not collect.recheck_due(180.0, 399.0, 220.0)
    assert collect.recheck_plan(0.2, 0.5) == "respawn" and collect.recheck_plan(0.5, 0.5) == "keep"
    assert collect.recheck_plan(None, 0.5) == "keep"
    rec = collect.recheck_record(t=1.0, i=7, segment="west_coast_usa#0", map_name="west_coast_usa", spawn_type=CITY,
                                 spawn_road="s", road="a", pos=[1.0, 2.0, 3.0], cov=0.12345, min_cov=0.5,
                                 recheck_min=3.0, action="respawn", done_s=180.0)
    assert rec["reason"] == "recheck_cov" and rec["kind"] == "recheck" and rec["paint_cov"] == 0.1235, rec
    print("recheck from pose ok: heading picks the road and direction, off-road and filter-off not scored")


def _run_west_coast(paint, pools: dict, extra: list[str], stop_at: int, root: Path | None = None,
                    speed_at=None) -> tuple[int, str, Path]:
    """The mocked collector on a west_coast_usa map rotation with a mocked deck.

    The car holds (0.4, 12, 0.5) heading +y at 10 m/s; a teleport does not move it.
    """
    import scripts.collect_scene_lanes as collect
    import python.collect.spawn_mix as spawn_mix
    from python.collect.level_session import OpenedLevel

    def enter(bng, level, cfg, cams):
        del level, cfg, cams
        return OpenedLevel(vehicle=bng.vehicles.vehicle, roads=[RoadPoly("r", [(0.0, 0.0), (0.0, 80.0)], 7.0)],
                           spawn=(0.4, 12.0, 0.5), cameras={"main": _Cam()})

    previous = (collect.enter_map, collect.painted_for, spawn_mix.SpawnDeck.load)
    collect.enter_map = enter  # type: ignore[assignment]
    collect.painted_for = lambda level: paint  # type: ignore[assignment]
    spawn_mix.SpawnDeck.load = classmethod(lambda cls, path, rng=None, **kw: cls(pools, rng, **kw))  # type: ignore[assignment]
    try:
        return _run_collector({"speed_at": speed_at or (lambda n: 10.0), "stop_at": stop_at},
                              ["--maps", "west_coast_usa", "--no-vary-env", "--no-excursions"] + list(extra), root)
    finally:
        collect.enter_map, collect.painted_for, spawn_mix.SpawnDeck.load = previous  # type: ignore[assignment]


def _west_pools() -> dict:
    # The car's road is unpainted (x < 1000); every spawn road far east is painted.
    here = _road("here", [(0.4, -300), (0.4, 600)])
    east = [_road(f"east{j}", [(5000 + 50 * j, 0), (5000 + 50 * j, 500)]) for j in range(12)]
    return {CITY: [here] + east, HIGHWAY: [], RAMP: []}


def check_recheck_and_repeats_run() -> None:
    """End to end: every 30 s the recheck scores the unpainted road under the car,
    respawns, and logs reason=recheck_cov; no row is dropped. road_repeats lands in
    collect.json and the end-of-run log. Off by default; on painted road it keeps."""
    east_paint = _Paint(lambda x, y: x > 1000.0)
    code, text, out = _run_west_coast(east_paint, _west_pools(), ["--recheck-cov-min", "0.5"], 1300)
    assert code == 0, (code, text[-2500:])
    assert "collect recheck_cov every 0.5 min on" in text, text[:3000]
    lines = [line for line in text.splitlines() if line.startswith("collect recheck_cov west_coast_usa#")]
    assert len(lines) >= 3, text[-3000:]
    assert all("paint_cov=0.00 min=0.50 road=here" in line and line.endswith("-> respawn") for line in lines), lines
    cuts = [json.loads(line) for line in (out / "cuts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(cuts) == len(lines) and all(c["reason"] == "recheck_cov" and c["action"] == "respawn" for c in cuts), cuts
    assert all(c["road"] == "here" and c["spawn_road"].startswith("east") and c["recheck_min"] == 0.5 for c in cuts), cuts
    gaps = [b["done_s"] - a["done_s"] for a, b in zip(cuts, cuts[1:])]
    assert all(29.0 <= g <= 34.5 for g in gaps), gaps  # 30 s, plus the 3 s settle when it lands on one
    # A respawn per recheck: the spawn road changes each time.
    assert len({c["spawn_road"] for c in cuts}) == len(cuts)
    rows = _rows(out)
    ids = [int(r["i"]) for r in rows]
    assert ids == list(range(len(ids))), "rows were dropped"
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    rr = status["road_repeats"]
    assert rr["picks"] == len(cuts) + 1 and rr["max_picks"] == 1 and rr["flagged"] == [], rr
    assert "here" not in {k for k, v in rr["by_road"].items() if v["picks"]}, rr  # unpainted: skipped, never picked
    assert abs(rr["time_s"] - 0.1 * len(rows)) < 2.0, (rr["time_s"], len(rows))
    assert "collect spawn repeats roads=" in text and f"picks={rr['picks']} max_picks=1 flagged=0" in text, text[-1500:]
    # Off by default: same rows, no rechecks, no cuts file.
    code, text_off, out_off = _run_west_coast(east_paint, _west_pools(), [], 1300)
    assert code == 0 and "recheck_cov" not in text_off and not (out_off / "cuts.jsonl").exists(), text_off[-1500:]
    assert len(_rows(out_off)) == len(rows), (len(_rows(out_off)), len(rows))
    assert json.loads((out_off / "collect.json").read_text(encoding="utf-8"))["road_repeats"]["picks"] == 1
    # Painted everywhere: each recheck is logged and kept.
    code, text_ok, out_ok = _run_west_coast(_Paint(lambda x, y: True), _west_pools(), ["--recheck-cov-min", "0.5"], 700)
    assert code == 0, text_ok[-1500:]
    kept = [json.loads(line) for line in (out_ok / "cuts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert kept and all(c["action"] == "keep" and c["paint_cov"] == 1.0 for c in kept), kept
    # The recheck needs the paint filter.
    code, text_nf, out_nf = _run_west_coast(east_paint, _west_pools(), ["--recheck-cov-min", "0.5", "--min-paint-cov", "0"], 700)
    assert code == 0 and "collect recheck_cov every 0.5 min off" in text_nf and not (out_nf / "cuts.jsonl").exists()
    print(f"recheck run: {len(cuts)} rechecks -> respawn, {len(rows)} rows kept; off by default; painted keeps")


# ------------------------------------------- (g) road types, resume skip
def _deck_pools() -> dict:
    # A highway passing 6 m over a city street at (200, 0); a ramp far away.
    hw = {"id": "hw", "pts": [(0.0, 0.0, 16.0), (200.0, 0.0, 16.0), (400.0, 0.0, 16.0)], "lanes": 4, "length": 400.0, "oneway": False}
    street = _road("street", [(200, -200), (200, 0), (200, 200)])
    ramp = _road("ramp", [(3000, 0), (3100, 0)], oneway=True)
    return {CITY: [street], HIGHWAY: [hw], RAMP: [ramp]}


def check_road_types() -> None:
    deck = SpawnDeck(_deck_pools())
    assert deck.road_type("hw", (205.0, 3.0)) == BRIDGE and deck.road_type("street", (198.0, 10.0)) == TUNNEL
    assert deck.road_type("hw", (20.0, 0.0)) == HIGHWAY and deck.road_type("street", (200.0, -150.0)) == CITY
    assert deck.road_type("ramp", (3050.0, 0.0)) == RAMP and deck.road_type("nope", (0.0, 0.0)) == CITY
    assert deck.road_type("hw") == HIGHWAY  # no pose: the pool kind
    # 3 m apart is a grade crossing, not a bridge.
    flat = {CITY: [_road("x", [(0, -50), (0, 50)]), _road("y", [(-50, 0), (50, 0)])], HIGHWAY: [], RAMP: []}
    flat[CITY][0]["pts"] = [(0.0, -50.0, 13.0), (0.0, 50.0, 13.0)]
    assert SpawnDeck(flat).road_type("x", (0.0, 0.0)) == CITY
    # Picks count by type and source; the spawn pose is 40% along (hw: x=160, 40 m from the crossing).
    deck = SpawnDeck(_deck_pools())
    a = deck.take()
    b = deck.take("recheck")
    assert a is not None and b is not None and deck.last_type is not None
    s = deck.type_summary()
    assert sum(s["spawn"].values()) == 1 and sum(s["recheck"].values()) == 1, s
    assert s["total"][deck.road_type(a.road_id, a.pos)] >= 1 and set(s["spawn"]) == {CITY, HIGHWAY, RAMP, BRIDGE, TUNNEL}
    lines = deck.type_lines()
    assert lines[0].startswith("spawn types spawn n=1 city=") and lines[1].startswith("spawn types recheck n=1 "), lines
    assert "bridge=" in lines[0] and "tunnel=" in lines[0]
    print(f"road types ok: {lines[0]} | {lines[1]}")


def check_resume_skips_recorded() -> None:
    """Resume: the reseeded deck would re-pick the same roads; skip_recorded moves it on."""
    deck = SpawnDeck(_pools(0, 6), painted=PAINT_SOUTH, min_paint_cov=0.5)
    first = [deck.take().road_id for _ in range(3)]
    prior = deck.road_repeats()
    again = SpawnDeck(_pools(0, 6), painted=PAINT_SOUTH, min_paint_cov=0.5)
    again.seed_repeats(prior)
    assert again.skip_recorded(prior) == 3 and again.resume_skipped == 3
    second = [again.take().road_id for _ in range(3)]
    assert not set(first) & set(second), (first, second)
    r = again.road_repeats()
    assert r["picks"] == 6 and r["max_picks"] == 1 and r["flagged"] == [], r
    assert again.take() is None  # 6 roads, all used: no silent repeat
    assert again.skip_recorded(None) == 0 and again.skip_recorded({"by_road": {"ghost": {"picks": 2}}}) == 0
    assert again.type_summary()["resume_skipped_roads"] == 3
    print(f"resume skip ok: first {first} then {second}")


def check_recheck_types_and_resume_run() -> None:
    """End to end: recheck lines and cuts.jsonl carry the road type and the respawn's type;
    the end-of-run log splits the type mix by spawn/recheck; a resumed run picks new roads."""
    east_paint = _Paint(lambda x, y: x > 1000.0)
    code, text, out = _run_west_coast(east_paint, _west_pools(), ["--recheck-cov-min", "0.5"], 1000)
    assert code == 0, text[-2000:]
    lines = [line for line in text.splitlines() if line.startswith("collect recheck_cov west_coast_usa#")]
    assert lines and all(" road=here road_type=city " in x and " new_type=city -> respawn" in x for x in lines), lines
    cuts = [json.loads(line) for line in (out / "cuts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(c["road_type"] == "city" and c["new_road_type"] == "city" and c["new_spawn_road"].startswith("east") for c in cuts), cuts
    assert f"collect spawn types spawn n=1 city=1 highway=0 ramp=0 bridge=0 tunnel=0" in text, text[-1500:]
    assert f"collect spawn types recheck n={len(cuts)} city={len(cuts)} " in text, text[-1500:]
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    assert status["spawn_types"]["recheck"]["city"] == len(cuts) and status["spawn_types"]["spawn"]["city"] == 1, status["spawn_types"]
    first = {k for k, v in status["road_repeats"]["by_road"].items() if v["picks"]}
    # Resume into the same folder (collect.json done_s < 5 min): new roads only.
    code, text2, out2 = _run_west_coast(east_paint, _west_pools(), ["--recheck-cov-min", "0.5"], 1000, root=out.parent)
    assert code == 0 and out2 == out, text2[-1500:]
    assert f"skipping {len(first)} already-picked roads" in text2, text2[:3000]
    status2 = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    rr = status2["road_repeats"]
    assert rr["max_picks"] == 1 and rr["picks"] > len(first), rr
    picked2 = [line.split(" road=")[1].split(" ")[0] for line in text2.splitlines() if line.startswith("collect spawn pick ")]
    assert picked2 and not set(picked2) & first, (picked2, first)
    print(f"recheck types run: {len(cuts)} recheck respawns typed; resume picked {len(picked2)} new roads, max_picks=1")

# ------------------------------------------- (h) spawn safety
def _pick(road_id: str, x: float, y: float, yaw_dx: float = 0.0, yaw_dy: float = 1.0) -> SpawnPick:
    from python.collect.spawn_mix import _yaw_quat

    return SpawnPick(CITY, road_id, (x, y, 0.4), _yaw_quat(yaw_dx, yaw_dy), (8.0, 12.0))


def check_traffic_slots() -> None:
    """Companion cars 60 m ahead / 40 m behind along the road; farthest spot or skip when short."""
    import math

    from python.collect.spawn_mix import TRAFFIC_AHEAD_M, TRAFFIC_BEHIND_M, traffic_lua

    assert (TRAFFIC_AHEAD_M, TRAFFIC_BEHIND_M) == (60.0, 40.0)
    # a: (0,0)->(0,50) joins b: (0,50)->(40,90)->(40,300). Ahead follows the bend, not the tangent.
    a = _road("a", [(0, 0), (0, 50)])
    b = _road("b", [(0, 50), (40, 90), (40, 300)])
    c = _road("c", [(0, -200), (0, 0)], oneway=True)  # one-way into a's start: valid behind
    deck = SpawnDeck({CITY: [a, b, c], HIGHWAY: [], RAMP: []})
    slots, notes = deck.traffic_slots(_pick("a", 0.0, 30.0), 2)
    assert notes == [] and [w for w, *_ in slots] == ["ahead", "behind"], (slots, notes)
    (_, pa, ra, la), (_, pb, rb, lb) = slots
    assert abs(la - 60.0) < 1e-6 and abs(lb - 40.0) < 1e-6
    # 20 m to the end of a, then 40 m along the 45-degree leg of b.
    assert abs(pa[0] - 40 / math.sqrt(2)) < 1e-6 and abs(pa[1] - (50 + 40 / math.sqrt(2))) < 1e-6 and pa[2] == b["pts"][0][2] + 0.5, pa  # road z + 0.5
    assert abs(pb[0]) < 1e-9 and abs(pb[1] + 10.0) < 1e-6, pb  # 30 m on a, 10 m back along c
    assert abs(rb[2] - _pick("a", 0, 0).rot[2]) < 1e-9 and abs(rb[3] - _pick("a", 0, 0).rot[3]) < 1e-9  # faces +y
    # Behind cannot enter a one-way road at its start (that is against its flow).
    d = _road("d", [(0, 0), (0, -200)], oneway=True)
    deck2 = SpawnDeck({CITY: [a, b, d], HIGHWAY: [], RAMP: []})
    slots2, notes2 = deck2.traffic_slots(_pick("a", 0.0, 30.0), 2)
    assert [w for w, *_ in slots2] == ["ahead", "behind"] and abs(slots2[1][3] - 30.0) < 1e-6, slots2
    assert notes2 == ["traffic behind fallback: farthest valid spot 30.0 m along the road (wanted 40 m)"], notes2
    # A dead end 5 m ahead: no valid spot, the car is skipped.
    lone = _road("lone", [(0, 0), (0, 100)])
    deck3 = SpawnDeck({CITY: [lone], HIGHWAY: [], RAMP: []})
    slots3, notes3 = deck3.traffic_slots(_pick("lone", 0.0, 95.0), 2)
    assert [w for w, *_ in slots3] == ["behind"] and "traffic ahead skipped: road walk reaches 5.0 m < 8 m" in notes3[0], notes3
    # Flags change the distances; count 1 is only the car ahead.
    slots4, _ = deck3.traffic_slots(_pick("lone", 0.0, 10.0), 1, 30.0, 25.0)
    assert len(slots4) == 1 and abs(slots4[0][1][1] - 40.0) < 1e-6
    _, notes5 = deck3.traffic_slots(_pick("ghost", 0.0, 10.0), 2)
    assert len(notes5) == 2 and "not in the deck" in notes5[0]
    # The Lua uses the slot poses; without slots it keeps the old 28 m / 22 m tangent offsets.
    lua = traffic_lua(_pick("a", 0.0, 30.0), 2, slots)[0]
    assert "gvd_traffic_0" in lua and "gvd_traffic_1" in lua and f"vec3({pa[0]:.2f},{pa[1]:.2f}," in lua and "0.00,-10.00," in lua, lua[:600]
    old = traffic_lua(_pick("a", 0.0, 30.0), 2)[0]
    assert "0.00,58.00,0.40)" in old and "0.00,8.00,0.40)" in old, old[:600]
    print(f"traffic slots ok: ahead {la:.0f} m at ({pa[0]:.1f},{pa[1]:.1f}), behind {lb:.0f} m; fallback and skip noted")


def check_stall_respawn_helpers() -> None:
    import scripts.collect_scene_lanes as collect

    a = collect.build_parser().parse_args([])
    assert (a.stall_respawn_s, a.traffic_ahead_m, a.traffic_behind_m) == (20.0, 60.0, 40.0)
    b = collect.build_parser().parse_args(["--stall-respawn-s", "0", "--traffic-ahead-m", "80", "--traffic-behind-m", "50"])
    assert (b.stall_respawn_s, b.traffic_ahead_m, b.traffic_behind_m) == (0.0, 80.0, 50.0)
    due = collect.stall_respawn_due
    assert due(20.0, 20.0, 5.0, True) and not due(20.0, 19.9, 5.0, True)
    assert not due(20.0, 25.0, SETTLE_FLAG_S - 0.1, True)  # inside the settle: skipped
    assert not due(20.0, 25.0, 5.0, False) and not due(0.0, 99.0, 99.0, True)
    hold = collect.defer_stationary
    assert hold("stationary", 20.0, 10.0, True) == "" and hold("stationary", 20.0, 20.0, True) == "stationary"
    assert hold("stationary", 0.0, 10.0, True) == "stationary" and hold("stationary", 20.0, 10.0, False) == "stationary"
    assert hold("tipped", 20.0, 10.0, True) == "tipped" and hold("", 20.0, 10.0, True) == ""
    print("stall respawn helpers ok")


def _small_pools(n: int) -> dict:
    here = _road("here", [(0.4, -300), (0.4, 600)])
    east = [_road(f"east{j}", [(5000 + 50 * j, 0), (5000 + 50 * j, 500)]) for j in range(n)]
    return {CITY: [here] + east, HIGHWAY: [], RAMP: []}


def check_deck_exhausted_run() -> None:
    """End to end: 3 painted roads, a recheck respawn every 30 s. The 4th pick finds the deck
    empty: one 'collect deck exhausted' line, collect.json says so, the car keeps driving."""
    east_paint = _Paint(lambda x, y: x > 1000.0)
    code, text, out = _run_west_coast(east_paint, _small_pools(3), ["--recheck-cov-min", "0.5"], 1600)
    assert code == 0, text[-2500:]
    lines = [x for x in text.splitlines() if x.startswith("collect deck exhausted west_coast_usa#")]
    assert len(lines) == 1, (lines, text[-3000:])
    assert "source=recheck picks=3 used=4/4 roads; keeping the current spawn road=east" in lines[0] and lines[0].endswith("(no respawn)"), lines
    assert "none left" not in text
    picks = [x for x in text.splitlines() if x.startswith("collect spawn pick ")]
    assert len(picks) == 3, picks
    rows = _rows(out)
    ids = [int(r["i"]) for r in rows]
    assert ids == list(range(len(ids))) and len(rows) >= 1500, len(rows)  # no stop: every tick still recorded
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    ex = status["deck_exhausted"]
    assert ex["exhausted"] and ex["source"] == "recheck" and ex["used"] == 4 and ex["roads"] == 4, ex
    t_ex = ex["done_s"]
    assert 0.1 * len(rows) > t_ex + 30.0, (t_ex, len(rows))  # well past the exhaustion
    assert status["spawn_mix"]["exhausted"] is True
    assert "collect deck exhausted at" in text
    print(f"deck exhausted run: line once at {t_ex:.0f}s, {len(rows)} rows, kept driving")


def check_stall_respawn_run() -> None:
    """End to end: the car eases to 0 m/s and stays there. After 20 s stationary it respawns
    from the deck (cuts.jsonl reason=stall_respawn), not the 10 s stationary abort."""
    paint = _Paint(lambda x, y: x > 1000.0)

    def speed(n: int) -> float:
        if n <= 100 or n > 550:
            return 10.0
        return max(0.0, 10.0 - 0.1 * (n - 100))  # 1 m/s2: not an impact

    code, text, out = _run_west_coast(paint, _west_pools(), [], 900, speed_at=speed)
    assert code == 0, text[-2500:]
    assert "collect abort stationary" not in text and "collect impact " not in text, text[-3000:]
    lines = [x for x in text.splitlines() if x.startswith("collect stall_respawn west_coast_usa#")]
    assert len(lines) == 1 and " stalled=20." in lines[0] and ">= 20s" in lines[0] and "-> respawn road=east" in lines[0], lines
    cuts = [json.loads(x) for x in (out / "cuts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(cuts) == 1 and cuts[0]["reason"] == "stall_respawn" and cuts[0]["action"] == "respawn", cuts
    assert cuts[0]["stall_respawn_s"] == 20.0 and cuts[0]["new_spawn_road"].startswith("east") and cuts[0]["stalled_s"] >= 20.0
    assert "collect spawn types stall n=1 city=1" in text, text[-1500:]
    rows = _rows(out)
    assert [int(r["i"]) for r in rows] == list(range(len(rows)))
    # Off: the old 10 s stationary abort and recover.
    code, text0, out0 = _run_west_coast(paint, _west_pools(), ["--stall-respawn-s", "0"], 900, speed_at=speed)
    assert code == 0 and "collect abort stationary" in text0 and "stall_respawn" not in text0, text0[-2000:]
    print(f"stall respawn run: {lines[0].split(' -> ')[0].split(' ', 3)[-1]}; off keeps the stationary abort")


def check_impact_traffic_run() -> None:
    """End to end on the west coast: traffic is placed along the road and logged; an impact
    keeps its decel line and impacts.jsonl event, with no traffic attribution and no
    BeamNG vehicle-state query (the mock has no get_states)."""
    paint = _Paint(lambda x, y: x > 1000.0)
    crash = lambda n: 10.0 if n <= 40 or n > 75 else 0.2  # noqa: E731
    assert not hasattr(_Api(None), "get_states")
    code, text, out = _run_west_coast(paint, _west_pools(), [], 300, speed_at=crash)
    assert code == 0, text[-2500:]
    placed = [x for x in text.splitlines() if x.startswith("collect traffic west_coast_usa#")]
    assert placed and "placed ahead=60.0m behind=40.0m (want ahead=60m behind=40m along the road)" in placed[0], placed
    line = [x for x in text.splitlines() if x.startswith("collect impact west_coast_usa#") and " decel=" in x]
    assert len(line) == 1 and line[0].endswith("decel=98.0 m/s2 v=10.0->0.2 excursion=cruise"), line
    assert "traffic=" not in line[0] and "state unavailable" not in text
    ev = [json.loads(x) for x in (out / "impacts.jsonl").read_text(encoding="utf-8").splitlines()][0]
    assert ev["kind"] == "impact" and ev["decel_mps2"] == 98.0, ev
    assert not any(k.startswith("traffic") for k in ev), ev
    # Respawn on impact: a deck pick on the next tick, logged, cut and counted.
    resp = [x for x in text.splitlines() if x.startswith("collect impact_respawn west_coast_usa#")]
    assert len(resp) == 1 and " -> respawn road=east" in resp[0], resp
    assert "respawning" not in text
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    assert status["impact_respawn"]["deck"] == 1 and status["impact_respawn"]["impacts"] == 1, status["impact_respawn"]
    print(f"impact run: {line[0].split(' ', 3)[-1]}; no traffic attribution; {resp[0]}")


def _line_paint(xs, y0: float = -100.0, y1: float = 300.0):
    """Real PaintedLanes over painted lines at world x = xs along +y (DecalRoad nodes, z 0.4)."""
    from python.debug.painted_lanes import PaintedLanes

    # Nodes every 10 m: PaintedLanes buckets segments in 25 m cells by their midpoint.
    ys = [y0 + 10.0 * k for k in range(int((y1 - y0) / 10.0) + 1)]
    return PaintedLanes([{"material": "line_white", "nodes": [[x, y, 0.4, 0.15] for y in ys]} for x in xs])


def check_impact_respawn_run() -> None:
    """A logged impact respawns on the next tick (recover in place off the deck), so nothing
    after the impact row is recorded from the damaged car; counted in collect.json."""
    code, text, out = _run_collector({}, ["--no-excursions"])
    assert code == 0, (code, text[-2000:])
    lines = [x for x in text.splitlines() if x.startswith("collect impact_respawn ")]
    assert lines == ["collect impact_respawn current#0 n=1 pos=(0.4,12.0,0.5) spawn_road=na -> recover in place"], lines
    assert "collect abort impact segment=current#0" in text and "collect recover current#0" in text
    assert "respawning" not in text, "no 2 s stall wait after the impact"
    events = [json.loads(x) for x in (out / "impacts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["kind"] for e in events] == ["impact", "impact_end"] and events[1]["how"] == "impact_respawn", events
    rows = _rows(out)
    i_hit = [int(r["i"]) for r in rows if r.get("impact")]
    i_rec = [int(r["i"]) for r in rows if r.get("recover")]
    assert len(i_hit) == 1 and i_rec == [i_hit[0] + 1], (i_hit, i_rec)
    # The mask is then just the 3 s run-up plus the impact row.
    assert i_hit[0] - events[0]["pre_i0"] == int(round(PRE_IMPACT_S / 0.1)), events[0]
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    assert status["impact_respawn"] == {"on": True, "impacts": 1, "deck": 0, "recover": 1, "abort_same_tick": 0}, status
    cuts = [json.loads(x) for x in (out / "cuts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [c["reason"] for c in cuts if c["kind"] == "impact"] == ["impact_respawn"], cuts
    # Opt out: the old drive-on / 2 s stall path.
    code, text_off, out_off = _run_collector({}, ["--no-excursions", "--no-impact-respawn"])
    assert code == 0 and "collect impact_respawn" not in text_off and "respawning" in text_off
    off = json.loads((out_off / "collect.json").read_text(encoding="utf-8"))["impact_respawn"]
    assert off == {"on": False, "impacts": 1, "deck": 0, "recover": 0, "abort_same_tick": 0}, off
    print(f"impact respawn: impact row i={i_hit[0]}, recover row i={i_rec[0]} (next row), counted; --no-impact-respawn keeps the old path")


def check_vehicle_width_runs() -> None:
    """The half width comes from the bbox once per level load; etk800 without a bbox is logged."""
    code, text, out = _run_collector({"speed_at": lambda n: 10.0, "stop_at": 60}, [])
    lines = [x for x in text.splitlines() if x.startswith("collect vehicle width ")]
    assert lines == ["collect vehicle width current#0 source=bbox width=1.90m half=0.95m"], lines
    assert json.loads((out / "collect.json").read_text(encoding="utf-8"))["ego_width"]["half_m"] == 0.95
    code, text, out = _run_collector({"speed_at": lambda n: 10.0, "stop_at": 60, "bbox_w": None}, [])
    lines = [x for x in text.splitlines() if x.startswith("collect vehicle width ")]
    assert lines == ["collect vehicle width current#0 source=etk800 width=1.80m half=0.90m "
                     "(bbox failed: RuntimeError: The request was not handled by BeamNG)"], lines
    code, text, out = _run_collector({"speed_at": lambda n: 10.0, "stop_at": 60, "bbox_w": 0.3}, [])
    assert "source=etk800 width=1.80m half=0.90m (bbox width 0.30m outside 1.2-3 m)" in text, text[-1500:]
    print("vehicle width: bbox 1.90 m -> 0.95 m, etk800 fallback logged on bbox failure / bad size")


def check_lane_edge_straddle_run() -> None:
    """Painted lines at x = -3.5, 0, 3.5 (the real PaintedLanes): a car at x=0.4 straddles the
    centerline, so its weave aborts with lane_edge at once; in its lane (x=1.75) it weaves."""
    import scripts.collect_scene_lanes as collect

    previous = collect.painted_for
    collect.painted_for = lambda level: _line_paint([-3.5, 0.0, 3.5])  # type: ignore[assignment]
    try:
        code, text, out = _run_collector({"speed_at": lambda n: 10.0, "stop_at": 300, "x": 0.4}, [])
        code_in, text_in, out_in = _run_collector({"speed_at": lambda n: 10.0, "stop_at": 300, "x": 1.75}, [])
    finally:
        collect.painted_for = previous  # type: ignore[assignment]
    assert code == 0 and code_in == 0, (text[-1500:], text_in[-1500:])
    assert "collect gate current#0 weave open" in text, text[-2000:]
    aborts = [r for r in _rows(out) if (r.get("excursion") or {}).get("phase") == "abort"]
    assert aborts and {r["excursion"]["reason"] for r in aborts} == {"lane_edge"}, {r["excursion"]["reason"] for r in aborts}
    weave = [r for r in _rows(out) if (r.get("excursion") or {}).get("phase") == "weave"]
    assert len(weave) <= 1, len(weave)
    weave_in = [r for r in _rows(out_in) if (r.get("excursion") or {}).get("phase") == "weave"]
    reasons_in = {(r.get("excursion") or {}).get("reason") for r in _rows(out_in)}
    assert len(weave_in) > 20 and "lane_edge" not in reasons_in, (len(weave_in), reasons_in)
    sent = [x for x in text_in.splitlines() if x.startswith("collect weave current#0 ")]
    # Planned at the measured 10 m/s, not the 8 m/s target: wave max(40, 5 * 10) = 50 m.
    assert sent and sent[0].startswith(
        "collect weave current#0 v_plan=10.0 v_meas=10.0 v_target=8.0 amp=0.50m wave=50m line=50m "
    ), sent  # amp 3.5/2 - 0.95 - 0.3
    print(f"lane_edge run: x=0.4 straddling -> abort lane_edge after {len(weave)} weave row; x=1.75 -> {len(weave_in)} weave rows; {sent[0]}")


def main() -> None:
    check_coverage_walks_ahead()
    check_short_probe_is_not_full_coverage()
    check_deck_skips_and_logs()
    check_deck_disabled_is_unchanged()
    check_deck_fallback_and_next_kind()
    check_real_west_coast_if_present()
    check_impact_watch()
    check_settle_window()
    check_tip_rules()
    check_weave_guard_helpers()
    check_paint_share_slot_order()
    check_spawn_mix_summary()
    check_damage_and_up_are_read()
    check_collector_impact_rows()
    check_impact_on_an_abort_tick_names_a_row()
    check_yield_guard()
    check_yield_guard_runs()
    check_weave_guard_runs()
    check_road_repeats()
    check_recheck_coverage_from_pose()
    check_recheck_and_repeats_run()
    check_road_types()
    check_resume_skips_recorded()
    check_recheck_types_and_resume_run()
    check_traffic_slots()
    check_stall_respawn_helpers()
    check_deck_exhausted_run()
    check_stall_respawn_run()
    check_impact_traffic_run()
    check_impact_respawn_run()
    check_vehicle_width_runs()
    check_lane_edge_straddle_run()
    print("collect spawn fix ok")


if __name__ == "__main__":
    main()
