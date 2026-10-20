#!/usr/bin/env python3
"""Collector fixes: paint-coverage spawns, impacts, settle rows, tip check, damage.

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
    ImpactWatch,
    Liveness,
    SettleWindow,
    car_is_driving,
    is_tipped,
)
from python.collect.spawn_mix import (  # noqa: E402
    CITY,
    HIGHWAY,
    RAMP,
    SpawnDeck,
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

    def lane_at(self, pos, direction, ys=(5.0, 10.0, 20.0, 30.0)):
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
    assert pre_i == 41 - int(round(PRE_IMPACT_S / 0.1)), hits  # first row inside the 2 s run-up
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

    def __init__(self, stop: Path, speed_at=None, stop_at: int = 120) -> None:
        self.stop = stop
        self.speed_at = speed_at or (lambda n: 10.0 if n <= 40 or n > 75 else 0.2)
        self.stop_at = int(stop_at)
        self.polls = 0
        self.commands: list[str] = []
        self.state: dict = {}
        self._set(10.0)

    def _set(self, speed: float) -> None:
        self.state = {
            "pos": [0.4, 12.0, 0.5],
            "dir": [0.0, 1.0, 0.0],
            "up": [0.0, 0.0, 1.0],
            "electrics": {"wheelspeed": speed, "steering_input": 0.0, "throttle_input": 0.3, "brake_input": 0.0},
        }

    def poll_sensors(self) -> None:
        self.polls += 1
        self._set(float(self.speed_at(self.polls)))
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


def _run_collector(vehicle_args: dict, extra: list[str]) -> tuple[int, str, Path]:
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

    root = Path(tempfile.mkdtemp())
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
    gt_logger.load_road_polys = lambda session: ([RoadPoly("r", [(0.0, 0.0), (0.0, 80.0)], 7.0)], "")  # type: ignore[assignment]
    collect._attach_cameras = lambda session, veh, cfg: {"main": _Cam()}  # type: ignore[assignment]
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
    code, text, out = _run_collector({}, ["--no-excursions"])
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
    print(f"collector impact rows ok impact i={i_hit} pre_i0={ev['pre_i0']} recover i={i_rec} settle={len(settled)} rows")


def check_weave_guard_runs() -> None:
    """No paint on this mocked level, so the guard keeps the weave from starting."""
    code, text, out = _run_collector({"speed_at": lambda n: 10.0, "stop_at": 260}, ["--weave-guard"])
    assert code == 0, (code, text)
    assert "collect gate current#0 weave blocked paint_ahead" in text, text[-3000:]
    assert "excursion=weave" not in text
    code, text, out = _run_collector({"speed_at": lambda n: 10.0, "stop_at": 260}, [])
    assert code == 0, (code, text)
    assert "excursion=weave" in text, text[-3000:]  # without the flag the weave is unchanged
    assert "line end" not in text
    # Painted everywhere: the guard lets the weave start, then ends it before
    # the 60 m line runs out (5.5 s at 10 m/s) instead of after the 12 s weave.
    import scripts.collect_scene_lanes as collect

    previous = collect.painted_for
    collect.painted_for = lambda level: _Paint(lambda x, y: True)  # type: ignore[assignment]
    try:
        code, text, out = _run_collector({"speed_at": lambda n: 10.0, "stop_at": 300}, ["--weave-guard"])
    finally:
        collect.painted_for = previous  # type: ignore[assignment]
    assert code == 0, (code, text)
    assert "collect gate current#0 weave open" in text, text[-3000:]
    assert "collect excursion current#0 weave line end, back to lane" in text, text[-3000:]
    after = text.split("line end, back to lane", 1)[1]
    assert "excursion=abort" in after, after[:400]
    rows = [json.loads(line) for line in (out / "state.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    weave_rows = [row for row in rows if (row.get("excursion") or {}).get("phase") == "weave"]
    assert weave_rows and 50 <= len(weave_rows) <= 58, len(weave_rows)
    print(f"weave guard: unpainted weave blocked, off by default, line end after {len(weave_rows)} rows")


def main() -> None:
    check_coverage_walks_ahead()
    check_deck_skips_and_logs()
    check_deck_disabled_is_unchanged()
    check_deck_fallback_and_next_kind()
    check_real_west_coast_if_present()
    check_impact_watch()
    check_settle_window()
    check_tip_rules()
    check_weave_guard_helpers()
    check_damage_and_up_are_read()
    check_collector_impact_rows()
    check_weave_guard_runs()
    print("collect spawn fix ok")


if __name__ == "__main__":
    main()
