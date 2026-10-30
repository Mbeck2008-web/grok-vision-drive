#!/usr/bin/env python3
"""Control-input logging: probe parsing, the file cap stop, auto-stops, no env commands.

Offline. The collector runs reuse the mocks in test_collect_spawn_fix.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import scripts.collect_scene_lanes as collect  # noqa: E402
import scripts.test_collect_spawn_fix as sf  # noqa: E402
import scripts.test_collect_stops as stops  # noqa: E402
from python.collect import control_log as cl  # noqa: E402

PROBE = {
    "src": ["ai", "steeringWheel"],
    "ai": {"steering": 0.12, "throttle": 0.3, "brake": 0.0, "parkingbrake": 0, "clutch": 0.0},
    "g": {"x": 1.1, "y": -0.2, "z": 9.7},
    "g2": {"x": 0.9, "y": -0.1, "z": 9.8},
    "w": {"yaw": 0.05, "pitch": -0.01, "roll": 0.002},
}
IMU_KEYS = {"gx", "gy", "gz", "g_smoothed", "yaw_rate", "pitch_rate", "roll_rate"}
PROBE_KEYS = {"steer_ai", "throttle_ai", "brake_ai", "parkingbrake_ai", "clutch_ai", "imu", "ai_in_src"}


def _all_null(got: dict) -> bool:
    imu = got["imu"]
    return (
        all(got[k] is None for k in PROBE_KEYS - {"imu"})
        and all(imu[k] is None for k in IMU_KEYS - {"g_smoothed"})
        and imu["g_smoothed"] == {"x": None, "y": None, "z": None}
    )


def check_schema() -> None:
    for reply in (json.dumps(PROBE), PROBE):
        got = cl.parse_ve_probe(reply)
        assert set(got) == PROBE_KEYS, sorted(got)
        assert set(got["imu"]) == IMU_KEYS, sorted(got["imu"])
        assert got["steer_ai"] == 0.12 and got["throttle_ai"] == 0.3 and got["brake_ai"] == 0.0
        assert got["parkingbrake_ai"] == 0.0 and got["clutch_ai"] == 0.0
        imu = got["imu"]
        assert (imu["gx"], imu["gy"], imu["gz"]) == (1.1, -0.2, 9.7), imu
        assert imu["g_smoothed"] == {"x": 0.9, "y": -0.1, "z": 9.8}, imu
        assert imu["gx"] != imu["g_smoothed"]["x"]
        assert (imu["yaw_rate"], imu["pitch_rate"], imu["roll_rate"]) == (0.05, -0.01, 0.002)
        assert got["ai_in_src"] == ["ai", "steeringWheel"]
        assert all(isinstance(v, float) for v in (imu["gx"], imu["g_smoothed"]["x"], got["steer_ai"]))
    # Garbage, None, a non-JSON string, an empty table: every value null, never 0, no raise.
    for bad in (None, "", "not json", "{}", "[]", 7, {"ai": "x", "g": [1, 2]}):
        got = cl.parse_ve_probe(bad)
        assert set(got) == PROBE_KEYS and _all_null(got), (bad, got)
    # A NaN or missing channel is null on its own; the rest stays.
    part = dict(PROBE, ai={"steering": float("nan"), "throttle": 0.5})
    got = cl.parse_ve_probe(part)
    assert got["steer_ai"] is None and got["throttle_ai"] == 0.5 and got["brake_ai"] is None
    assert cl.probe_nulls() == cl.parse_ve_probe(None)
    # Electrics: whitelist only, beamngpy names and raw Lua aliases both land on one key.
    el = {"gearIndex": 3, "gear": "D", "rpm": 2100.0, "engine_load": 0.4, "brake_lights": 1,
          "isShifting": False, "fuel_volume": 40.0, "steering": 4.5, "lights_state": 1}
    sub = cl.electrics_subset(el)
    assert sub == {"gear": "D", "gear_index": 3, "is_shifting": False, "rpm": 2100.0,
                   "engine_load": 0.4, "brakelights": 1}, sub
    assert set(cl.ELECTRICS_KEEP) == {
        "gear", "gear_index", "is_shifting", "rpm", "engine_load", "engine_throttle", "clutch_ratio",
        "wheelspeed", "airspeed", "avg_wheel_av", "abs_active", "esc_active", "tcs_active", "reverse",
        "turnsignal", "brakelights", "ignition", "running",
    }
    assert cl.electrics_subset(None) == {} and cl.electrics_subset({"rpm": float("inf")}) == {"rpm": None}
    # Traffic: GE reply list, empty Lua table, garbage.
    car = {"name": "gvd_traffic_0", "pos": [1, 2, 3], "vel": [0, 9.5, 0], "dir": [0, 1, 0]}
    assert cl.parse_traffic(json.dumps([car])) == [
        {"name": "gvd_traffic_0", "pos": [1.0, 2.0, 3.0], "vel": [0.0, 9.5, 0.0], "dir": [0.0, 1.0, 0.0]}
    ]
    assert cl.parse_traffic("{}") == [] and cl.parse_traffic("[]") == []
    assert cl.parse_traffic(None) is None and cl.parse_traffic("nope") is None
    assert "gvd_traffic_0" in cl.TRAFFIC_CHUNK and "gvd_traffic_1" in cl.TRAFFIC_CHUNK
    assert "return jsonEncode" in cl.TRAFFIC_CHUNK and "return jsonEncode" in cl.VE_PROBE
    assert "lastInputs" in cl.VE_PROBE and "gx2" in cl.VE_PROBE
    # Meta probe: lock and wheelbase, null on garbage.
    assert cl.parse_ve_meta('{"lock": 450, "wb": 2.91}') == {"steer_lock_deg": 450.0, "wheelbase_m": 2.91}
    assert cl.parse_ve_meta(None) == {"steer_lock_deg": None, "wheelbase_m": None}
    # ai_cmd: the collector's own last AI lines, no game call.
    state = cl.ai_cmd_state()
    assert state == {"mode": None, "set_speed_mps": None, "speed_mode": None, "line_active": False}
    for line in collect.ai_commands(12.0, None):
        state = cl.ai_cmd_state(state, line)
    assert state == {"mode": "span", "set_speed_mps": 12.0, "speed_mode": "limit", "line_active": False}, state
    for line in ('ai.setMode("manual")', "ai.setSpeed(9.5)", "ai.setLine({{pos={1,2,3}}}, true)"):
        state = cl.ai_cmd_state(state, line)
    assert state == {"mode": "manual", "set_speed_mps": 9.5, "speed_mode": "limit", "line_active": True}, state
    state = cl.ai_cmd_state(state, 'ai.setMode("span")')
    assert state["line_active"] is False and state["mode"] == "span"
    assert cl.ai_cmd_state(state, "core_weather.setWeather(1)") == state
    print(f"schema ok probe={sorted(PROBE_KEYS)} veh={len(cl.ELECTRICS_KEEP)} keys")


def _rows(out: Path) -> list[dict]:
    return [json.loads(x) for x in (out / "state.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def _files(out: Path) -> set[str]:
    return {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}


def check_file_cap_exit_7() -> None:
    assert collect.STOP_EXIT_FILE_CAP == 7 and collect.STOP_EXIT_EARLY == 6
    code, text, out = sf._run_collector({"speed_at": lambda n: 10.0, "stop_at": 99999},
                                        ["--no-excursions", "--max-file-gib", "0.000001"])
    # collect.main -> the real RunStop.finish(): 7 survives, it does not become 6.
    status = stops._check_stop(code, text, out, "file_cap", 7)
    assert status["done_s"] < status["target_s"], status
    # The detail names the top-level file that hit the cap (whichever log is biggest then).
    capped = str(status["stop_detail"]).split(" ")[0]
    assert (out / capped).is_file() and capped.endswith(".jsonl"), status["stop_detail"]
    rows = _rows(out)
    ids = [int(r["i"]) for r in rows]
    assert ids and ids == list(range(len(ids))), "rows were dropped"
    pngs = list((out / "raw" / "main").rglob("*.png"))
    assert len(pngs) >= len(rows), (len(pngs), len(rows))
    before = _files(out)
    # A second start in the same dir appends; nothing earlier is removed.
    code2, text2, _ = sf._run_collector({"speed_at": lambda n: 10.0, "stop_at": 30},
                                        ["--no-excursions"], root=out.parent)
    assert code2 == 0, text2[-1500:]
    assert before <= _files(out), sorted(before - _files(out))
    # Default cap 3.5 GiB never trips on a short run.
    assert collect.build_parser().parse_args([]).max_file_gib == 3.5
    print(f"file_cap ok exit={code} rows={len(rows)} done_s={status['done_s']:.1f}")


def check_auto_stops_delete_nothing() -> None:
    import python.data.data_root as data_root

    real = data_root.check_data_path
    calls = {"n": 0}

    def guard(path):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise data_root.DataDriveError("data disk below 20 GB free")
        return real(path)

    data_root.check_data_path = guard
    try:
        code, text, out = sf._run_collector({"speed_at": lambda n: 10.0, "stop_at": 99999}, ["--no-excursions"])
    finally:
        data_root.check_data_path = real
    stops._check_stop(code, text, out, "disk_low", 3)
    rows = _rows(out)
    assert [int(r["i"]) for r in rows] == list(range(len(rows))) and rows
    code, text, out = sf._run_collector({"speed_at": lambda n: 10.0, "stop_at": 60}, ["--no-excursions"])
    stops._check_stop(code, text, out, "requested", 0)
    assert (out / "STOP").is_file(), "the STOP file is left in place"
    rows = _rows(out)
    assert [int(r["i"]) for r in rows] == list(range(len(rows))) and rows
    assert len(list((out / "raw" / "main").rglob("*.png"))) >= len(rows)
    print("auto-stops ok disk_low=3 requested=0, nothing deleted")


LAUNCH = ("--maps west_coast_usa --no-vary-env --traffic 2 --segment-min 12 --smoke-s 60 "
          "--ctl-probe-every 1 --minutes 120").split()


def _west_env_run(extra: list[str], vary: bool) -> tuple[int, str, list[list[str]], list[str]]:
    """The mocked west-coast collector; records every environment_commands result and queued line."""
    import python.collect.auto_mode as auto_mode
    import python.collect.spawn_mix as spawn_mix
    from python.collect.level_session import OpenedLevel

    made: list[list[str]] = []
    queued: list[str] = []
    real_env, real_queue = auto_mode.environment_commands, collect._queue

    def env(tod, sky):
        got = real_env(tod, sky)
        made.append(list(got))
        return got

    def queue(vehicle, command):
        queued.append(str(command))
        return real_queue(vehicle, command)

    def enter(bng, level, cfg, cams):
        del level, cfg, cams
        return OpenedLevel(vehicle=bng.vehicles.vehicle, roads=[sf.RoadPoly("r", [(0.0, 0.0), (0.0, 80.0)], 7.0)],
                           spawn=(0.4, 12.0, 0.5), cameras={"main": sf._Cam()})

    previous = (collect.enter_map, collect.painted_for, spawn_mix.SpawnDeck.load)
    paint = sf._Paint(lambda x, y: True)
    pools = sf._west_pools()
    collect.enter_map = enter  # type: ignore[assignment]
    collect.painted_for = lambda level: paint  # type: ignore[assignment]
    spawn_mix.SpawnDeck.load = classmethod(lambda cls, path, rng=None, **kw: cls(pools, rng, **kw))  # type: ignore[assignment]
    auto_mode.environment_commands = env  # type: ignore[assignment]
    collect._queue = queue  # type: ignore[assignment]
    try:
        args = list(extra)
        if vary:
            # Control: vary the sky, and cut 1 min segments (the floor) so the run reaches
            # segment 2 (segment 1 always leaves the sky alone).
            args = [a for a in args if a != "--no-vary-env"]
            args[args.index("--segment-min") + 1] = "1"
        code, text, _out = sf._run_collector({"speed_at": lambda n: 10.0, "stop_at": 700 if vary else 200},
                                             ["--no-excursions"] + args + ["--smoke-s", "0"])
    finally:
        collect.enter_map, collect.painted_for, spawn_mix.SpawnDeck.load = previous  # type: ignore[assignment]
        auto_mode.environment_commands, collect._queue = real_env, real_queue  # type: ignore[assignment]
    return code, text, made, queued


def check_launch_argv_no_env() -> None:
    args = collect.build_parser().parse_args(LAUNCH + ["--out", "x"])
    assert args.no_vary_env and not str(args.time_of_day or "").strip() and not str(args.weather or "").strip()
    assert args.ctl_probe_every == 1 and args.traffic == 2 and args.max_file_gib == 3.5
    code, text, made, queued = _west_env_run(LAUNCH, vary=False)
    assert code == 0, text[-2000:]
    assert made and all(m == [] for m in made), made
    env_lines = [q for q in queued if "core_environment" in q or "core_weather" in q]
    assert env_lines == [], env_lines
    # Control: the same argv without --no-vary-env does send them, so the check above can fail.
    code, text, made_v, queued_v = _west_env_run(LAUNCH, vary=True)
    assert any(m for m in made_v), made_v
    assert any("core_environment" in q or "core_weather" in q for q in queued_v)
    print(f"env ok: launch argv calls={len(made)} commands=0; control commands={sum(len(m) for m in made_v)}")


def _respawn_run(same_object: bool) -> tuple[int, str, list[dict], int]:
    """Recheck respawns every 30 s; every pick lands on the same coordinates.

    same_object=False: each pick carries a new tuple with equal content.
    same_object=True: each pick carries the one tuple object, so id(spawn_pos) never
    changes (what a reused address would look like).
    """
    import dataclasses

    import python.collect.level_session as level_session
    import python.collect.spawn_mix as spawn_mix

    shared = (5000.0, 100.0, 0.5)
    places = {"n": 0}
    real_take, real_place = spawn_mix.SpawnDeck.take, level_session.place_vehicle

    def take(self, *a, **k):
        pick = real_take(self, *a, **k)
        if pick is None:
            return None
        pos = shared if same_object else tuple(float(v) for v in list(shared))
        return dataclasses.replace(pick, pos=pos)

    def place(bng, vehicle, pos, rot):
        got = real_place(bng, vehicle, pos, rot)
        if not got:
            places["n"] += 1
        return got

    spawn_mix.SpawnDeck.take = take  # type: ignore[assignment]
    level_session.place_vehicle = place  # type: ignore[assignment]
    try:
        paint = sf._Paint(lambda x, y: x > 1000.0)
        code, text, out = sf._run_west_coast(paint, sf._west_pools(), ["--recheck-cov-min", "0.5"], 1300)
    finally:
        spawn_mix.SpawnDeck.take, level_session.place_vehicle = real_take, real_place  # type: ignore[assignment]
    return code, text, _rows(out), places["n"]


def check_veh_meta_every_spawn() -> None:
    """veh_meta is keyed on a spawn counter, not on id(): it fires once per spawn even
    when a respawn reuses equal coordinates or the very same tuple object."""
    for same_object in (False, True):
        code, text, rows, places = _respawn_run(same_object)
        assert code == 0, text[-2500:]
        assert places >= 3, (places, text[-2500:])
        metas = [r for r in rows if "veh_meta" in r]
        # The level load and the first placement both land before row 0 (one veh_meta);
        # then one per recheck respawn.
        assert len(metas) == places, (same_object, len(metas), places)
        assert "veh_meta" in rows[0], rows[0].keys()
        print(f"veh_meta ok same_object={same_object} respawns={places} veh_meta_rows={len(metas)}")


def main() -> None:
    check_schema()
    check_file_cap_exit_7()
    check_auto_stops_delete_nothing()
    check_launch_argv_no_env()
    check_veh_meta_every_spawn()
    print("control log ok")


if __name__ == "__main__":
    main()
