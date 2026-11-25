#!/usr/bin/env python3
"""Map-load settle, post-teleport settle, double-teleport fix, settle metadata,
and AI traffic lane/heading/mode fixes (without altering ego spawn placement).

Offline; uses the fake BeamNG harness from test_collect_spawn_fix / test_recorder_v2.
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import scripts.collect_scene_lanes as collect  # noqa: E402
import scripts.test_collect_spawn_fix as sf  # noqa: E402
import python.collect.level_session as level_session  # noqa: E402
import python.collect.liveness as liveness  # noqa: E402
import python.collect.spawn_mix as spawn_mix  # noqa: E402
from python.debug.gt_geom import RoadPoly  # noqa: E402
from python.train.frame_quality import build_road_grid, classify_wrong_way, load_quality_roads  # noqa: E402


def _bng_spawn_actual_dir(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float]:
    """Simulate BeamNG core_vehicles.spawnNewVehicle -> spawn.lua:605 actual forward dir.

    Empirically verified across all 9 batch-1 clips (clip_000..clip_008):
    passing yaw quaternion (0, 0, sin(a/2), cos(a/2)) to spawnNewVehicle yields
    veh:getDirectionVector() at yaw = pi/2 - a.
    """
    alpha = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    actual_yaw = math.pi * 0.5 - alpha
    return (math.cos(actual_yaw), math.sin(actual_yaw))


def _parse_traffic_rot_quats(lua: str) -> list[tuple[float, float, float, float]]:
    out = []
    for m in re.finditer(r"rot=quat\(([-0-9.eE]+),([-0-9.eE]+),([-0-9.eE]+),([-0-9.eE]+)\)", lua):
        out.append(tuple(float(m.group(i)) for i in range(1, 5)))
    return out


# ----------------------------------------------------------- (1) Map load/switch settle

def check_level_load_settle_waits_for_steady_fps() -> None:
    """After level load/switch, wait for map load + steady FPS above threshold before placing/recording."""
    events: list[tuple[str, float]] = []
    cam_polls = {"n": 0}
    real_enter = collect.enter_map
    real_place = level_session.place_vehicle
    real_load = spawn_mix.SpawnDeck.load

    class _SlowStartCam:
        def poll(self):
            cam_polls["n"] += 1
            # First 6 camera polls run slow (1.5 Hz -> dt=0.67s), then recover to 8 Hz (0.125s).
            return {"colour": sf.np.full((8, 8, 3), 120, dtype=sf.np.uint8)}

    def enter(bng, level, cfg, cams):
        del cfg
        collect.drop_cameras(cams)
        events.append(("enter", collect.time.monotonic()))
        return level_session.OpenedLevel(
            vehicle=bng.vehicles.vehicle,
            roads=[RoadPoly("r", [(0.0, 0.0), (0.0, 80.0)], 7.0)],
            spawn=(0.4, 12.0, 0.5),
            cameras={"main": _SlowStartCam()},
        )

    def place(*args, **kwargs):
        events.append(("place", collect.time.monotonic()))
        return real_place(*args, **kwargs)

    pools = sf._west_pools()
    collect.enter_map = enter
    level_session.place_vehicle = place
    spawn_mix.SpawnDeck.load = classmethod(lambda cls, path, rng=None, **kw: cls(pools, rng, **kw))
    real_tick_sleep = collect.tick_sleep_s
    try:
        # Hook _run_collector internals by patching tick_sleep_s after _run_collector sets it?
        # _run_collector overwrites tick_sleep_s, so use a custom runner via _Cam + tick_sleep wrapper.
        import python.data.raw_queue as raw_queue
        import python.debug.gt_logger as gt_logger
        import python.sensors.tech as tech
        from python.collect.auto_mode import stop_file
        from gvd_test_env import collect_root

        with collect_root() as root:
            out = root / "scene"
            veh = sf._CrashVehicle(stop_file(out), speed_at=lambda n: 10.0, stop_at=25)
            bng = sf._Bng(veh)
            clock = {"t": 0.0}

            def sleep(sec: float) -> None:
                clock["t"] += float(sec or 0.0)

            def tick_sleep(_elapsed: float, _period: float = 0.1) -> float:
                # Simulate 1.5 Hz stutter for the first 6 polls, then steady 8 Hz.
                return 0.67 if cam_polls["n"] <= 6 else 0.125

            prev = (
                tech.open_tech_beamngpy, tech.widen_socket_timeout, tech.release_tech_beamngpy,
                gt_logger.load_road_polys, collect._attach_cameras,
                collect.time.sleep, collect.time.monotonic, collect.tick_sleep_s,
                raw_queue.RawWriteQueue,
            )

            class _DeepQueue(raw_queue.RawWriteQueue):
                def __init__(self, writer, max_depth: int = 64, **kw) -> None:
                    del max_depth
                    super().__init__(writer, 100000, **kw)

            tech.open_tech_beamngpy = lambda *a, **k: bng
            tech.widen_socket_timeout = lambda *a, **k: None
            tech.release_tech_beamngpy = lambda *a, **k: None
            gt_logger.load_road_polys = lambda s: ([RoadPoly("r", [(0.0, 0.0), (0.0, 80.0)], 7.0)], "")
            collect._attach_cameras = lambda s, v, c: {"main": _SlowStartCam()}
            collect.time.sleep = sleep
            collect.time.monotonic = lambda: clock["t"]
            collect.tick_sleep_s = tick_sleep
            raw_queue.RawWriteQueue = _DeepQueue
            import io
            buf = io.StringIO()
            old_out = sys.stdout
            sys.stdout = buf
            try:
                code = collect.main([
                    "--maps", "west_coast_usa", "--no-vary-env", "--no-excursions",
                    "--min-paint-cov", "0", "--minutes", "5", "--smoke-s", "0",
                    "--data-dir", str(root), "--out", str(out), "--host", "127.0.0.1", "--port", "1",
                ])
            finally:
                sys.stdout = old_out
                (
                    tech.open_tech_beamngpy, tech.widen_socket_timeout, tech.release_tech_beamngpy,
                    gt_logger.load_road_polys, collect._attach_cameras,
                    collect.time.sleep, collect.time.monotonic, collect.tick_sleep_s,
                    raw_queue.RawWriteQueue,
                ) = prev
            text = buf.getvalue()
            assert code == 0, text[-2000:]
            assert "collect level_settle " in text, f"missing level_settle log:\n{text[:2000]}"
            enter_t = next(t for k, t in events if k == "enter")
            place_t = next(t for k, t in events if k == "place")
            # 6 polls at 0.67s (~4.0s) + >= LEVEL_SETTLE_HOLD_S (2.0s) steady at 8 Hz before place_vehicle!
            assert place_t - enter_t >= 5.5, f"placed too early after level load: enter={enter_t} place={place_t}"
            rows = [json.loads(ln) for ln in (out / "state.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
            assert len(rows) >= 10
    finally:
        collect.enter_map = real_enter
        level_session.place_vehicle = real_place
        spawn_mix.SpawnDeck.load = real_load
        collect.tick_sleep_s = real_tick_sleep
    print("level_load_settle waits for steady FPS before placing and recording OK")


def check_level_load_settle_timeout_fails_loudly() -> None:
    """If FPS stays at 1.5 Hz past LEVEL_SETTLE_TIMEOUT_S, fail loudly without spawning or recording."""
    real_enter = collect.enter_map
    real_place = level_session.place_vehicle
    real_load = spawn_mix.SpawnDeck.load
    placed = []

    def enter(bng, level, cfg, cams):
        del cfg
        collect.drop_cameras(cams)
        return level_session.OpenedLevel(
            vehicle=bng.vehicles.vehicle,
            roads=[RoadPoly("r", [(0.0, 0.0), (0.0, 80.0)], 7.0)],
            spawn=(0.4, 12.0, 0.5),
            cameras={"main": sf._Cam()},
        )

    def place(*args, **kwargs):
        placed.append(1)
        return real_place(*args, **kwargs)

    pools = sf._west_pools()
    collect.enter_map = enter
    level_session.place_vehicle = place
    spawn_mix.SpawnDeck.load = classmethod(lambda cls, path, rng=None, **kw: cls(pools, rng, **kw))
    try:
        import python.data.raw_queue as raw_queue
        import python.debug.gt_logger as gt_logger
        import python.sensors.tech as tech
        from python.collect.auto_mode import stop_file
        from gvd_test_env import collect_root

        with collect_root() as root:
            out = root / "scene"
            veh = sf._CrashVehicle(stop_file(out), speed_at=lambda n: 10.0, stop_at=99999)
            bng = sf._Bng(veh)
            clock = {"t": 0.0}
            prev = (
                tech.open_tech_beamngpy, tech.widen_socket_timeout, tech.release_tech_beamngpy,
                gt_logger.load_road_polys, collect._attach_cameras,
                collect.time.sleep, collect.time.monotonic, collect.tick_sleep_s,
                raw_queue.RawWriteQueue,
            )

            class _DeepQueue(raw_queue.RawWriteQueue):
                def __init__(self, writer, max_depth: int = 64, **kw) -> None:
                    del max_depth
                    super().__init__(writer, 100000, **kw)

            tech.open_tech_beamngpy = lambda *a, **k: bng
            tech.widen_socket_timeout = lambda *a, **k: None
            tech.release_tech_beamngpy = lambda *a, **k: None
            gt_logger.load_road_polys = lambda s: ([RoadPoly("r", [(0.0, 0.0), (0.0, 80.0)], 7.0)], "")
            collect._attach_cameras = lambda s, v, c: {"main": sf._Cam()}
            collect.time.sleep = lambda sec: clock.__setitem__("t", clock["t"] + float(sec or 0.0))
            collect.time.monotonic = lambda: clock["t"]
            collect.tick_sleep_s = lambda _e, _p=0.1: 0.67  # stuck at 1.5 Hz forever!
            raw_queue.RawWriteQueue = _DeepQueue
            import io
            buf = io.StringIO()
            old_out = sys.stdout
            sys.stdout = buf
            try:
                code = collect.main([
                    "--maps", "west_coast_usa", "--no-vary-env", "--no-excursions",
                    "--min-paint-cov", "0", "--minutes", "5", "--smoke-s", "0",
                    "--data-dir", str(root), "--out", str(out), "--host", "127.0.0.1", "--port", "1",
                ])
            finally:
                sys.stdout = old_out
                (
                    tech.open_tech_beamngpy, tech.widen_socket_timeout, tech.release_tech_beamngpy,
                    gt_logger.load_road_polys, collect._attach_cameras,
                    collect.time.sleep, collect.time.monotonic, collect.tick_sleep_s,
                    raw_queue.RawWriteQueue,
                ) = prev
            text = buf.getvalue()
            assert code != 0, (code, text[-1500:])
            assert "collect stop reason=level_settle_timeout " in text, text[-2000:]
            status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
            assert status["stop_reason"] == "level_settle_timeout", status
            assert not placed, "must not place/spawn vehicle when level settle timed out"
            assert not (out / "state.jsonl").is_file() or not (out / "state.jsonl").read_text().strip()
    finally:
        collect.enter_map = real_enter
        level_session.place_vehicle = real_place
        spawn_mix.SpawnDeck.load = real_load
    print("level_load_settle_timeout fails loudly without spawning or recording OK")


# ------------------------------------------- (2) Post-teleport settle & no double teleport

def check_post_teleport_settle_and_no_double_teleport() -> None:
    """After every teleport/respawn, wait ~5s + FPS recovery before recording; no double teleport at low FPS / bounce."""
    real_enter = collect.enter_map
    real_place = level_session.place_vehicle
    real_load = spawn_mix.SpawnDeck.load
    place_times: list[float] = []
    cam_ticks = {"n": 0}

    def enter(bng, level, cfg, cams):
        del cfg
        collect.drop_cameras(cams)
        return level_session.OpenedLevel(
            vehicle=bng.vehicles.vehicle,
            roads=[RoadPoly("r", [(0.0, 0.0), (0.0, 80.0)], 7.0)],
            spawn=(0.4, 12.0, 0.5),
            cameras={"main": sf._Cam()},
        )

    def place(*args, **kwargs):
        place_times.append(collect.time.monotonic())
        return real_place(*args, **kwargs)

    pools = sf._west_pools()
    collect.enter_map = enter
    level_session.place_vehicle = place
    spawn_mix.SpawnDeck.load = classmethod(lambda cls, path, rng=None, **kw: cls(pools, rng, **kw))
    try:
        import python.data.raw_queue as raw_queue
        import python.debug.gt_logger as gt_logger
        import python.sensors.tech as tech
        from python.collect.auto_mode import stop_file
        from gvd_test_env import collect_root

        with collect_root() as root:
            out = root / "scene"
            # Speed bounces sharply on the first 3 polls after spawn/teleport (like clip_007/008!),
            # which previously triggered an immediate false impact -> double teleport!
            def speed_at(n: int) -> float:
                if n in (1, 2):
                    return 9.3 if n == 1 else 0.1  # huge decel right at start (would fire impact if not suppressed!)
                return 10.0

            veh = sf._CrashVehicle(stop_file(out), speed_at=speed_at, stop_at=120)
            bng = sf._Bng(veh)
            clock = {"t": 0.0}
            prev = (
                tech.open_tech_beamngpy, tech.widen_socket_timeout, tech.release_tech_beamngpy,
                gt_logger.load_road_polys, collect._attach_cameras,
                collect.time.sleep, collect.time.monotonic, collect.tick_sleep_s,
                raw_queue.RawWriteQueue,
            )

            class _DeepQueue(raw_queue.RawWriteQueue):
                def __init__(self, writer, max_depth: int = 64, **kw) -> None:
                    del max_depth
                    super().__init__(writer, 100000, **kw)

            def tick_sleep(_elapsed: float, _period: float = 0.1) -> float:
                cam_ticks["n"] += 1
                return 0.1

            tech.open_tech_beamngpy = lambda *a, **k: bng
            tech.widen_socket_timeout = lambda *a, **k: None
            tech.release_tech_beamngpy = lambda *a, **k: None
            gt_logger.load_road_polys = lambda s: ([RoadPoly("r", [(0.0, 0.0), (0.0, 80.0)], 7.0)], "")
            collect._attach_cameras = lambda s, v, c: {"main": sf._Cam()}
            collect.time.sleep = lambda sec: clock.__setitem__("t", clock["t"] + float(sec or 0.0))
            collect.time.monotonic = lambda: clock["t"]
            collect.tick_sleep_s = tick_sleep
            raw_queue.RawWriteQueue = _DeepQueue
            import io
            buf = io.StringIO()
            old_out = sys.stdout
            sys.stdout = buf
            try:
                code = collect.main([
                    "--maps", "west_coast_usa", "--no-vary-env", "--no-excursions",
                    "--min-paint-cov", "0", "--respawn-min", "0.1", "--minutes", "5", "--smoke-s", "0",
                    "--data-dir", str(root), "--out", str(out), "--host", "127.0.0.1", "--port", "1",
                ])
            finally:
                sys.stdout = old_out
                (
                    tech.open_tech_beamngpy, tech.widen_socket_timeout, tech.release_tech_beamngpy,
                    gt_logger.load_road_polys, collect._attach_cameras,
                    collect.time.sleep, collect.time.monotonic, collect.tick_sleep_s,
                    raw_queue.RawWriteQueue,
                ) = prev
            text = buf.getvalue()
            assert code == 0, text[-2000:]
            assert "collect impact " not in text, f"spurious post-teleport impact triggered:\n{text}"
            assert len(place_times) >= 2, place_times
            # Every consecutive teleport must be separated by at least respawn_s + settle (~11s), never a double teleport!
            for a, b in zip(place_times, place_times[1:]):
                assert b - a >= 9.0, f"double teleport detected: {a:.2f} -> {b:.2f}"
            rows = [json.loads(ln) for ln in (out / "state.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
            # Verify settle timestamps/flags in state.jsonl and collect.json!
            settled_rows = [r for r in rows if r.get("settle")]
            assert settled_rows, "expected settle-flagged rows in state.jsonl"
            for r in settled_rows:
                assert isinstance(r.get("settle_t0"), (int, float)), r
                assert isinstance(r.get("settle_source"), str) and r["settle_source"], r
                sw = r.get("settle_window")
                assert isinstance(sw, dict) and sw.get("wait_s", 0.0) >= 4.5, sw
                assert "t0" in sw and "wait_t0" in sw and "wait_t1" in sw, sw
            # Unsettled rows must NOT carry settle flags.
            unsettled = [r for r in rows if not r.get("settle")]
            assert unsettled and all("settle_window" not in r and "settle_t0" not in r for r in unsettled)
            status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
            windows = status.get("settle_windows")
            assert isinstance(windows, list) and len(windows) >= 2, windows
            for w in windows:
                assert w["wait_s"] >= 4.5 and w["t0"] <= w["t1"] and w["i0"] <= w["i1"], w
    finally:
        collect.enter_map = real_enter
        level_session.place_vehicle = real_place
        spawn_mix.SpawnDeck.load = real_load
    print("post-teleport settle, no double teleport, and settle metadata OK")


# ------------------------------------------- (3) AI traffic heading/lanes/AI mode

def check_traffic_heading_lanes_and_ego_unchanged() -> None:
    """Fix traffic heading/lanes/AI mode without touching ego spawn placement or speeds."""
    fixture = ROOT / "tests" / "fixtures" / "roads" / "west_coast_usa.json.gz"
    # (a) Ego spawn placement MUST remain byte-for-byte identical to main!
    deck = spawn_mix.SpawnDeck.load(fixture, random.Random(2003305365), only_kind="highway")
    pick = deck.take("spawn")
    assert pick is not None
    assert str(pick.road_id) == "893554.0", pick.road_id
    assert abs(pick.pos[0] - (-314.7055910441602)) < 1e-6, pick.pos
    assert abs(pick.pos[1] - 29.486259716891478) < 1e-6, pick.pos
    assert abs(pick.rot[2] - 0.39246405195054357) < 1e-6, pick.rot
    assert abs(pick.rot[3] - 0.9197673444554123) < 1e-6, pick.rot

    # (b) Traffic heading in traffic_lua must face the road direction across ALL quadrants!
    for deg in (0.0, 45.0, 90.0, 135.0, 180.0, -45.0, -90.0, -135.0):
        rad = math.radians(deg)
        fx, fy = math.cos(rad), math.sin(rad)
        p = spawn_mix.SpawnPick("city", "r0", (10.0, 20.0, 1.0), spawn_mix._yaw_quat(fx, fy), (8.0, 14.0))
        # Test both slots=None and explicit slots!
        lua_none = spawn_mix.traffic_lua(p, 2, None)[0]
        quats = _parse_traffic_rot_quats(lua_none)
        assert len(quats) == 2, (deg, lua_none)
        for q in quats:
            ax, ay = _bng_spawn_actual_dir(*q)
            dot = ax * fx + ay * fy
            assert dot > 0.999, f"deg={deg}: actual=({ax:.3f},{ay:.3f}) want=({fx:.3f},{fy:.3f}) dot={dot:.4f}"

    # (c) AI commands: traffic mode, driveInLane on, avoidCars on, and speed kept (~pick.speed midpoint)!
    lua_cmd = spawn_mix.traffic_lua(pick, 2, None)[0]
    assert 'ai.setMode("traffic")' in lua_cmd, lua_cmd
    assert 'ai.setMode("span")' not in lua_cmd, lua_cmd
    assert 'ai.driveInLane("on")' in lua_cmd, lua_cmd
    assert 'ai.setSpeedMode("limit")' in lua_cmd, lua_cmd
    expected_speed = (pick.speed[0] + pick.speed[1]) * 0.5
    assert f"ai.setSpeed({expected_speed:.1f})" in lua_cmd, lua_cmd

    # (d) Real west_coast_usa highway pick -> narrow ramp/two-way slots must stay in legal lanes!
    slots, _notes = deck.traffic_slots(pick, 6)
    assert slots, "expected traffic slots on west_coast_usa"
    qroads = load_quality_roads(fixture)
    grid = build_road_grid(qroads)
    for where, pos, rot, _along in slots:
        _qx, _qy, qz, qw = rot
        yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
        verdict = classify_wrong_way(pos[0], pos[1], yaw, qroads, grid, z=pos[2])
        assert verdict["verdict"] == "ok", (where, pos, verdict)

    # (e) Two-way road with pick.offset_m == 0.0 must still enforce legal lane placement for traffic!
    twoway = {
        "id": "tw",
        "pts": [(0.0, 0.0, 0.0), (0.0, 300.0, 0.0)],
        "widths": [8.0, 8.0],
        "lanes": 2,
        "lanes_left": 1,
        "lanes_right": 1,
        "length": 300.0,
        "oneway": False,
        "flip": False,
        "auto_lanes": False,
        "width": 8.0,
    }
    tw_deck = spawn_mix.SpawnDeck({spawn_mix.CITY: [twoway], spawn_mix.HIGHWAY: [], spawn_mix.RAMP: []})
    tw_pick = spawn_mix.SpawnPick("city", "tw", (0.0, 100.0, 0.5), spawn_mix._yaw_quat(0.0, 1.0), (8.0, 14.0), offset_m=0.0)
    tw_slots, _ = tw_deck.traffic_slots(tw_pick, 2)
    assert len(tw_slots) == 2, tw_slots
    tw_q, tw_g = tw_deck._ensure_quality()
    for where, pos, rot, _ in tw_slots:
        _qx, _qy, qz, qw = rot
        yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
        v = classify_wrong_way(pos[0], pos[1], yaw, tw_q, tw_g, z=pos[2])
        assert v["verdict"] == "ok" and v["reason"] == "twoway_correct_side", (where, pos, v)

    # (f) _walk_ahead respects flipDirection=True on joining one-way roads!
    r0 = {"id": "r0", "pts": [(0.0, 0.0, 0.0), (0.0, 100.0, 0.0)], "oneway": True, "flip": False, "lanes": 1, "length": 100.0, "width": 4.0}
    # r1's pts go (0, 200) -> (0, 100), flip=True means legal travel is (0, 100) -> (0, 200)!
    r1 = {"id": "r1", "pts": [(0.0, 200.0, 0.0), (0.0, 100.0, 0.0)], "oneway": True, "flip": True, "lanes": 1, "length": 100.0, "width": 4.0}
    fdeck = spawn_mix.SpawnDeck({spawn_mix.HIGHWAY: [r0, r1], spawn_mix.CITY: [], spawn_mix.RAMP: []})
    fpick = spawn_mix.SpawnPick("highway", "r0", (0.0, 70.0, 0.5), spawn_mix._yaw_quat(0.0, 1.0), (20.0, 30.0), offset_m=0.0)
    fslots, _ = fdeck.traffic_slots(fpick, 1, ahead_m=60.0)
    assert len(fslots) == 1 and abs(fslots[0][1][1] - 130.0) < 1e-6, fslots
    print("traffic heading (all quadrants), legal lanes, flip oneway, AI mode, and ego unchanged OK")


def check_legal_traffic_slot_keeps_unsnapped_pose() -> None:
    """A slot already in a legal lane keeps its chosen pose; _legal_traffic_pose is not called.

    Kills mutant S3, which always takes the _legal_traffic_pose branch and
    re-snaps to the first _offset_candidates lane, discarding pick.offset_m.
    """
    # 4-lane two-way, 14 m wide (LANE_W_M=3.5 so the tag is kept). Legal
    # right-hand centers are 1.75 m (inner) and 5.25 m (rightmost first).
    road = {
        "id": "tw4",
        "pts": [(0.0, 0.0, 0.0), (0.0, 300.0, 0.0)],
        "widths": [14.0, 14.0],
        "lanes": 4,
        "lanes_left": 2,
        "lanes_right": 2,
        "length": 300.0,
        "oneway": False,
        "flip": False,
        "auto_lanes": False,
        "width": 14.0,
    }
    deck = spawn_mix.SpawnDeck({spawn_mix.CITY: [road], spawn_mix.HIGHWAY: [], spawn_mix.RAMP: []})
    offset = 1.75
    pick = spawn_mix.SpawnPick(
        "city",
        "tw4",
        (0.0, 100.0, 0.5),
        spawn_mix._yaw_quat(0.0, 1.0),
        (8.0, 14.0),
        offset_m=offset,
    )
    assert deck._slot_is_legal(offset, 160.0, 0.5, 0.0, 1.0), "inner right-hand lane must already be legal"
    assert deck._slot_is_legal(-offset, 60.0, 0.5, 0.0, -1.0), "oncoming inner lane must already be legal"

    calls: list[tuple[float, float, float, float, float]] = []
    real = spawn_mix.SpawnDeck._legal_traffic_pose

    def wrapped(self, x, y, z, fx, fy):
        calls.append((x, y, z, fx, fy))
        return real(self, x, y, z, fx, fy)

    spawn_mix.SpawnDeck._legal_traffic_pose = wrapped  # type: ignore[method-assign]
    try:
        slots, notes = deck.traffic_slots(pick, 2)
    finally:
        spawn_mix.SpawnDeck._legal_traffic_pose = real  # type: ignore[method-assign]

    assert calls == [], f"_legal_traffic_pose must not be called for an already-legal slot: {calls} notes={notes}"
    assert len(slots) == 2, (slots, notes)

    hx, hy = 0.0, 1.0
    expected: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {}
    for where, want, back in (("ahead", spawn_mix.TRAFFIC_AHEAD_M, False), ("behind", spawn_mix.TRAFFIC_BEHIND_M, True)):
        heading = (-hx, -hy) if back else (hx, hy)
        chain = spawn_mix.chain_from_pose(
            deck._by_id["tw4"], pick.pos, heading, deck._by_id, deck._ends, want + 1.0, reverse=back
        )
        got = spawn_mix.point_along(chain, want)
        assert got is not None, (where, want)
        (x, y, z), (tx, ty), _along = got
        fx, fy = (-tx, -ty) if back else (tx, ty)
        rx, ry = fy, -fx
        cx, cy, cz = x + rx * offset, y + ry * offset, z + 0.5
        expected[where] = ((cx, cy, cz), spawn_mix._yaw_quat(fx, fy))

    got_slots = {where: (pos, rot) for where, pos, rot, _along in slots}
    assert set(got_slots) == set(expected), got_slots
    for where, (pos, rot) in expected.items():
        gpos, grot = got_slots[where]
        assert gpos == pos, f"{where} pos changed: got {gpos} want unsnapped {pos}"
        assert grot == rot, f"{where} rot changed: got {grot} want _yaw_quat {rot}"
    print("legal traffic slot keeps unsnapped pose; _legal_traffic_pose not called OK")


def check_begin_level_step_settle_and_timeout() -> None:
    """begin_level waits for vehicle step rate to settle before snap_to_road/place_vehicle, and fails loudly on timeout."""
    events: list[tuple[str, float]] = []
    clock = {"t": 0.0}
    real_mono = level_session.time.monotonic
    real_sleep = level_session.time.sleep
    real_snap = level_session.snap_to_road
    real_place = level_session.place_vehicle
    real_spawn = level_session.spawn_scenario
    real_player = level_session.player_vehicle
    real_roads = level_session._roads

    class _Veh:
        def __init__(self, dt_fn) -> None:
            self.sensors = {"state": object()}
            self.polls = 0
            self.dt_fn = dt_fn

        def poll_sensors(self) -> None:
            self.polls += 1
            clock["t"] += self.dt_fn(self.polls)

    class _FakeBng:
        def __init__(self, veh) -> None:
            self.veh = veh
            self.vehicles = type("V", (), {"vehicle": veh, "get_current": lambda s: {"ego": veh}, "get_player_vehicle_id": lambda s: "ego"})()

        def get_levels(self):
            return {"west_coast_usa": {"spawnPoints": [{"pos": [1.0, 2.0, 3.0]}]}}

    def _run_case(dt_fn):
        clock["t"] = 0.0
        events.clear()
        veh = _Veh(dt_fn)
        bng = _FakeBng(veh)
        level_session.time.monotonic = lambda: clock["t"]
        level_session.time.sleep = lambda s: clock.__setitem__("t", clock["t"] + float(s or 0.0))
        level_session.spawn_scenario = lambda b, l, p, r: ""
        level_session.player_vehicle = lambda b: (veh, "")
        level_session._roads = lambda b: ([RoadPoly("r", [(0.0, 0.0), (0.0, 80.0)], 7.0)], "")
        level_session.snap_to_road = lambda b, near: (events.append(("snap", clock["t"])), ((0.4, 12.0, 0.5), (0.0, 0.0, 0.0, 1.0)))[1]
        level_session.place_vehicle = lambda b, v, pos, rot: (events.append(("place", clock["t"])), "")[1]
        return level_session.begin_level(bng, "west_coast_usa"), veh

    try:
        opened_ok, _veh = _run_case(lambda n: 0.6 if n <= 5 else 0.1)
        assert opened_ok.error == "", opened_ok
        assert [k for k, _ in events] == ["snap", "place"], events
        assert events[0][1] >= 5.0, f"snapped before step rate settled: {events}"

        opened_bad, _veh2 = _run_case(lambda n: 0.6)
        assert "step rate did not settle" in opened_bad.error, opened_bad
        assert events == [], f"must not snap/place on timeout: {events}"
    finally:
        level_session.time.monotonic = real_mono
        level_session.time.sleep = real_sleep
        level_session.snap_to_road = real_snap
        level_session.place_vehicle = real_place
        level_session.spawn_scenario = real_spawn
        level_session.player_vehicle = real_player
        level_session._roads = real_roads
    print("begin_level step settle and timeout OK")


def check_stale_impact_respawn_due_cleared_on_spawn() -> None:
    """An impact on tick T followed by respawn_due on tick T+1 must not double-teleport."""
    real_enter = collect.enter_map
    real_place = level_session.place_vehicle
    real_load = spawn_mix.SpawnDeck.load
    place_times: list[float] = []

    def enter(bng, level, cfg, cams):
        del cfg
        collect.drop_cameras(cams)
        return level_session.OpenedLevel(
            vehicle=bng.vehicles.vehicle,
            roads=[RoadPoly("r", [(0.0, 0.0), (0.0, 80.0)], 7.0)],
            spawn=(0.4, 12.0, 0.5),
            cameras={"main": sf._Cam()},
        )

    def place(*args, **kwargs):
        place_times.append(collect.time.monotonic())
        return real_place(*args, **kwargs)

    pools = sf._west_pools()
    collect.enter_map = enter
    level_session.place_vehicle = place
    spawn_mix.SpawnDeck.load = classmethod(lambda cls, path, rng=None, **kw: cls(pools, rng, **kw))
    try:
        # At poll 60 (done_s ~ 5.9s, right before respawn-min=0.1 -> 6.0s!), inject a sharp decel impact.
        # On poll 61 (done_s >= 6.0s), respawn_due fires _apply_spawn(False), which must clear
        # impact_respawn_due so it does NOT immediately chain into _impact_respawn on the same tick!
        def speed_at(n: int) -> float:
            return 2.0 if n == 60 else 10.0

        code, text, out = sf._run_collector(
            {"speed_at": speed_at, "stop_at": 130},
            ["--maps", "west_coast_usa", "--no-vary-env", "--no-excursions",
             "--min-paint-cov", "0", "--respawn-min", "0.099"],
        )
        assert code == 0, text[-2000:]
        assert "collect impact " in text, text
        assert "collect impact_respawn " not in text, f"stale impact_respawn_due fired double teleport:\n{text}"
        for a, b in zip(place_times, place_times[1:]):
            assert b - a >= 9.0, f"double teleport: {a:.2f} -> {b:.2f}"
    finally:
        collect.enter_map = real_enter
        level_session.place_vehicle = real_place
        spawn_mix.SpawnDeck.load = real_load
    print("stale impact_respawn_due cleared on spawn (no double teleport) OK")


def main() -> int:
    check_traffic_heading_lanes_and_ego_unchanged()
    check_legal_traffic_slot_keeps_unsnapped_pose()
    check_level_load_settle_waits_for_steady_fps()
    check_level_load_settle_timeout_fails_loudly()
    check_post_teleport_settle_and_no_double_teleport()
    check_begin_level_step_settle_and_timeout()
    check_stale_impact_respawn_due_cleared_on_spawn()
    print("ALL test_map_settle CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
