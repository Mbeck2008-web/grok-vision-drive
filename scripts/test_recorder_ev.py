#!/usr/bin/env python3
"""Recorder ego car and steady boundaries: the eSBR spawn, weave-guard default,
a 60 s cruise between excursions, and chunk switches / timed teleports that
wait for cruise (capped at 30 s).

Offline. beamngpy is a fake module; the collector runs on the mocked harness
from test_collect_spawn_fix. No BeamNG is needed.
"""
from __future__ import annotations

import io
import json
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

POS = (10.0, 20.0, 0.5)
ROT = (0.0, 0.0, 0.0, 1.0)


class _FakeVehicle:
    made: list[dict] = []

    def __init__(self, vid, model, **kwargs) -> None:
        self.vid = vid
        self.model = model
        self.kwargs = dict(kwargs)
        _FakeVehicle.made.append({"vid": vid, "model": model, **kwargs})


class _FakeScenario:
    def __init__(self, level, name, **kwargs) -> None:
        del kwargs
        self.level, self.name = level, name
        self.added: list = []

    def add_vehicle(self, vehicle, **kwargs) -> None:
        self.added.append((vehicle, kwargs))


def _with_beamngpy(fn):
    fake = types.ModuleType("beamngpy")
    fake.Vehicle = _FakeVehicle  # type: ignore[attr-defined]
    fake.Scenario = _FakeScenario  # type: ignore[attr-defined]
    old = sys.modules.get("beamngpy", None)
    sys.modules["beamngpy"] = fake
    _FakeVehicle.made = []
    try:
        return fn()
    finally:
        if old is None:
            sys.modules.pop("beamngpy", None)
        else:
            sys.modules["beamngpy"] = old


def check_spawn_passes_model_and_config() -> None:
    import python.collect.level_session as ls

    assert ls.EGO_MODEL_DEFAULT == "sbr", ls.EGO_MODEL_DEFAULT
    assert ls.EGO_CONFIG_DEFAULT == "vehicles/sbr/electric_800.pc", ls.EGO_CONFIG_DEFAULT

    class _Bng:
        def __init__(self) -> None:
            self.spawned: list = []

        def spawn_vehicle(self, vehicle, pos, rot_quat=None, safe_spawn=True):
            self.spawned.append((vehicle, pos, rot_quat, safe_spawn))

    def run():
        scenario = ls.make_collect_scenario("west_coast_usa", POS, ROT, name="t")
        bng = _Bng()
        assert ls.spawn_direct(bng, POS, ROT) == ""
        return scenario, bng

    scenario, bng = _with_beamngpy(run)
    made = _FakeVehicle.made
    assert len(made) == 2, made
    for entry in made:
        assert entry["model"] == "sbr" and entry["part_config"] == ls.EGO_CONFIG_DEFAULT, entry
    assert scenario.added and scenario.added[0][0].model == "sbr"
    assert bng.spawned and bng.spawned[0][0].kwargs["part_config"] == ls.EGO_CONFIG_DEFAULT
    # set_ego switches every spawn site; an empty config passes no part_config.
    ls.set_ego("etk800", "")
    try:
        _with_beamngpy(run)
        assert all(e["model"] == "etk800" and "part_config" not in e for e in _FakeVehicle.made), _FakeVehicle.made
    finally:
        ls.set_ego(ls.EGO_MODEL_DEFAULT, ls.EGO_CONFIG_DEFAULT)
    assert ls.ego_spec() == {"model": "sbr", "part_config": ls.EGO_CONFIG_DEFAULT}
    print(f"spawn: Scenario and direct spawns pass model=sbr part_config={ls.EGO_CONFIG_DEFAULT} OK")


def check_lua_fallback_has_config() -> None:
    import python.collect.level_session as ls

    class _LuaBng:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def queue_lua_command(self, command: str) -> None:
            self.commands.append(command)

    old = sys.modules.get("beamngpy", None)
    sys.modules["beamngpy"] = None  # type: ignore[assignment]  # import fails: the Lua path
    try:
        bng = _LuaBng()
        assert ls.spawn_direct(bng, POS, ROT) == ""
    finally:
        if old is None:
            sys.modules.pop("beamngpy", None)
        else:
            sys.modules["beamngpy"] = old
    assert len(bng.commands) == 1, bng.commands
    cmd = bng.commands[0]
    assert cmd.startswith('core_vehicles.spawnNewVehicle("sbr", {'), cmd
    assert f'config="{ls.EGO_CONFIG_DEFAULT}"' in cmd and "pos=vec3(10.00,20.00,0.50)" in cmd, cmd
    print(f"lua fallback: {cmd} OK")


def check_ego_values_validated() -> None:
    """--ego-model / --ego-config reach a Lua string: only [\\w/.-] is accepted."""
    import contextlib
    import io

    import python.collect.level_session as ls
    import scripts.collect_scene_lanes as collect

    bad = [("sbr", 'vehicles/sbr/x.pc", color="1'), ("sbr", "vehicles\\sbr\\electric_800.pc"),
           ('sb"r', ""), ("sbr; rm", ""), ("sbr", "vehicles/sbr/electric 800.pc"),
           ("sbr;", ""), (";", ""), ("sbr", "vehicles/sbr/electric_800.pc;"), ("sbr", ";")]
    for model, config in bad:
        try:
            ls.set_ego(model, config)
        except ValueError:
            pass
        else:
            raise AssertionError((model, config))
        assert ls.ego_spec() == {"model": ls.EGO_MODEL_DEFAULT, "part_config": ls.EGO_CONFIG_DEFAULT}
    for model, config in [("sbr", "vehicles/sbr/electric_800.pc"), ("etk800", ""), ("sbr2", "a-b/c_d.pc")]:
        ls.set_ego(model, config)
    ls.set_ego(ls.EGO_MODEL_DEFAULT, ls.EGO_CONFIG_DEFAULT)
    # Every refused character exits 2 from the collector, on both flags.
    rejected = {"double quote": '"', "single quote": "'", "backslash": "\\", "space": " ", "semicolon": ";"}
    argvs = []
    for ch in rejected.values():
        argvs.append(["--ego-model", f"sb{ch}r"])
        argvs.append(["--ego-config", f"vehicles/sbr/x{ch}.pc"])
    for argv in argvs:
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                collect._main(argv, collect.RunStop())
        except SystemExit as exc:
            assert exc.code == 2, exc.code
        else:
            raise AssertionError(argv)
        assert "--ego-" in err.getvalue(), err.getvalue()
    assert ls.ego_spec() == {"model": ls.EGO_MODEL_DEFAULT, "part_config": ls.EGO_CONFIG_DEFAULT}
    print(f"ego values: {', '.join(rejected)} refused (set_ego, and the collector exits 2 on both flags) OK")


def check_paint_lime() -> None:
    """electric_800 spawns in its stock Lime (sbr info.json paints.Lime), on every spawn site."""
    import python.collect.level_session as ls

    lime = (0.42, 0.62, 0.05, 1.0)
    paint = ls.ego_paint()
    assert paint and paint["name"] == "Lime" and tuple(paint["base_color"]) == lime, paint
    assert (paint["metallic"], paint["roughness"], paint["clearcoat"], paint["clearcoat_roughness"]) == (
        0.0, 0.9, 1.0, 0.05), paint
    assert "info.json" in paint["source"], paint

    class _Bng:
        def spawn_vehicle(self, vehicle, pos, rot_quat=None, safe_spawn=True):
            del vehicle, pos, rot_quat, safe_spawn

    def run():
        ls.make_collect_scenario("west_coast_usa", POS, ROT, name="t")
        ls.spawn_direct(_Bng(), POS, ROT)

    _with_beamngpy(run)
    assert len(_FakeVehicle.made) == 2 and all(e["color"] == lime for e in _FakeVehicle.made), _FakeVehicle.made

    class _LuaBng:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def queue_lua_command(self, command: str) -> None:
            self.commands.append(command)

    old = sys.modules.get("beamngpy", None)
    sys.modules["beamngpy"] = None  # type: ignore[assignment]
    try:
        bng = _LuaBng()
        ls.spawn_direct(bng, POS, ROT)
    finally:
        if old is None:
            sys.modules.pop("beamngpy", None)
        else:
            sys.modules["beamngpy"] = old
    assert 'color="0.42 0.62 0.05 1.0"' in bng.commands[0], bng.commands
    # A config with no known paint keeps the spawn API default: no color passed.
    ls.set_ego("sbr", "vehicles/sbr/electric_300.pc")
    try:
        assert ls.ego_paint() is None
        _with_beamngpy(run)
        assert all("color" not in e for e in _FakeVehicle.made), _FakeVehicle.made
    finally:
        ls.set_ego(ls.EGO_MODEL_DEFAULT, ls.EGO_CONFIG_DEFAULT)
    print(f"paint: electric_800 spawns Lime {lime} (scenario, direct, Lua) OK")


def check_defaults() -> None:
    import scripts.collect_scene_lanes as collect
    from python.collect.excursion import CRUISE_S
    from python.collect.level_session import EGO_CONFIG_DEFAULT

    assert CRUISE_S == 60.0, CRUISE_S
    args = collect.build_parser().parse_args([])
    assert args.weave_guard is True, args.weave_guard
    assert args.ego_model == "sbr" and args.ego_config == EGO_CONFIG_DEFAULT, (args.ego_model, args.ego_config)
    args = collect.build_parser().parse_args(["--no-weave-guard", "--ego-model", "etk800", "--ego-config", ""])
    assert args.weave_guard is False and args.ego_model == "etk800" and args.ego_config == ""
    assert collect.build_parser().parse_args(["--weave-guard"]).weave_guard is True
    assert collect.CAMERA_RIG_MODEL == "etk800"
    print("defaults: weave guard on, CRUISE_S 60, ego sbr OK")


def check_carry_over_unit() -> None:
    """CarryOver: a held switch's wait is owed back evenly over the rest of the run."""
    from python.collect.boundary import CarryOver

    c = CarryOver()
    c.add(30.0, 90.0, 200.0)
    assert c.total_s == 30.0 and c.owed_s == 30.0
    assert c.repay(90.0) == 0.0
    assert abs(c.repay(145.0) - 15.0) < 1e-9 and abs(c.owed_s - 15.0) < 1e-9  # 30 s over 110 s
    c.add(5.0, 145.0, 200.0)  # a second hold: 20 s owed over the 55 s left
    assert c.total_s == 35.0 and abs(c.owed_s - 20.0) < 1e-9
    assert abs(c.repay(172.5) - 10.0) < 1e-9
    assert abs(c.repay(250.0) - 10.0) < 1e-9 and c.owed_s == 0.0  # never more than owed
    assert c.repay(260.0) == 0.0
    # An open-ended run (no total): nothing to keep inside, nothing repaid.
    o = CarryOver()
    o.add(10.0, 0.0, float("inf"))
    assert o.repay(1e6) == 0.0 and o.owed_s == 10.0 and o.total_s == 10.0
    assert c.summary() == {"carried_s": 35.0, "owed_s": 0.0}
    print("carry-over: owed back evenly over the rest of the run, never past the total OK")


def check_boundary_defer_unit() -> None:
    from python.collect.boundary import BOUNDARY_DEFER_CAP_S, BoundaryDefer, boundary_wait_reason

    assert BOUNDARY_DEFER_CAP_S == 30.0
    assert boundary_wait_reason("cruise", False) == ""
    assert boundary_wait_reason("cruise", True) == "settle"
    for phase in ("weave", "lane_change", "recover", "abort"):
        assert boundary_wait_reason(phase, False) == phase
    buf = io.StringIO()
    with redirect_stdout(buf):
        d = BoundaryDefer("segment")
        assert d.hold(100.0, "") is False  # steady: no wait, no log
        assert d.hold(100.0, "weave") is True
        assert d.hold(104.9, "recover") is True
        assert d.hold(105.0, "") is False  # released at cruise
        assert d.hold(200.0, "weave") is True
        assert d.hold(229.9, "weave") is True
        assert d.hold(230.0, "weave") is False  # cap: switch anyway
        assert d.hold(300.0, "lane_change") is True
        d.clear()  # the boundary went away (a forced switch)
        assert d.pending() is False
    lines = buf.getvalue().splitlines()
    assert lines == [
        "collect boundary_defer segment:weave 0.0",
        "collect boundary_defer segment:cruise 5.0",
        "collect boundary_defer segment:weave 0.0",
        "collect boundary_defer segment:cap 30.0",
        "collect boundary_defer segment:lane_change 0.0",
    ], lines
    print("boundary defer: held while weaving, released at cruise or the 30 s cap OK")


def _rows(out: Path) -> list[dict]:
    return [json.loads(x) for x in (out / "state.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def check_veh_meta_and_rig_warning() -> None:
    import python.collect.level_session as ls
    import scripts.test_collect_spawn_fix as sf
    from python.collect.level_session import EGO_CONFIG_DEFAULT

    base = sf._CrashVehicle

    class _Sbr(base):  # what an eSBR spawn reports
        def __init__(self, *a, **k) -> None:
            k.setdefault("model", "sbr")
            super().__init__(*a, **k)

    sf._CrashVehicle = _Sbr  # type: ignore[misc]
    try:
        code, text, out = _west(lambda n: "", ["--respawn-min", "0.1"], 200)
    finally:
        sf._CrashVehicle = base  # type: ignore[misc]
    assert code == 0, text[-2000:]
    metas = [r["veh_meta"] for r in _rows(out) if "veh_meta" in r]
    assert len(metas) >= 2, metas  # one per spawn
    assert all(m["model"] == "sbr" and m["part_config"] == EGO_CONFIG_DEFAULT for m in metas), metas
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    assert status["ego"] == {"model": "sbr", "part_config": EGO_CONFIG_DEFAULT, "camera_rig_model": "sbr",
                             "camera_overrides": status["ego"]["camera_overrides"],
                             "spawned": True, "paint": ls.EGO_PAINTS[EGO_CONFIG_DEFAULT]}, status.get("ego")
    # cameras.yaml has sbr mounts (vehicle_overrides): no rig warning for the spawned eSBR.
    assert "camera rig calibrated" not in text, text[:3000]
    # An sbr already in a single-level run keeps the etk800 rig, and is warned once.
    sf._CrashVehicle = _Sbr  # type: ignore[misc]
    try:
        code, text, out = sf._run_collector({"stop_at": 60}, ["--no-excursions"])
    finally:
        sf._CrashVehicle = base  # type: ignore[misc]
    warn = "collect WARNING camera rig calibrated on etk800, ego model is sbr"
    assert code == 0 and text.count(warn) == 1, text[:3000]
    # The etk800 opt-back: no rig warning.
    code, text, out = _west(lambda n: "", ["--ego-model", "etk800", "--ego-config", ""], 60)
    assert code == 0 and "camera rig calibrated" not in text, text[:2000]
    metas = [r["veh_meta"] for r in _rows(out) if "veh_meta" in r]
    assert metas and metas[0]["model"] == "etk800" and metas[0]["part_config"] == "", metas
    # A single-level run drives the car already there: no part_config claimed.
    code, text, out = sf._run_collector({"stop_at": 60}, ["--no-excursions"])
    metas = [r["veh_meta"] for r in _rows(out) if "veh_meta" in r]
    assert code == 0 and metas and "part_config" not in metas[0], metas
    ego = json.loads((out / "collect.json").read_text(encoding="utf-8"))["ego"]
    assert ego["spawned"] is False and ego["paint"] is None, ego
    assert ego["camera_rig_model"] == "etk800" and ego["camera_overrides"] == {}, ego
    print("veh_meta: model=sbr part_config recorded per spawn; collect.json ego; sbr rig, warning only off-rig OK")


def check_ego_model_reaches_spawn() -> None:
    """--ego-model sbr2 (neither the default nor the mock's etk800): the spawn is asked for sbr2,
    and veh_meta / collect.json carry the spawned model, not the live vehicle object's."""
    import python.collect.level_session as ls

    asked: list[dict] = []
    try:
        code, text, out = _west(lambda n: "", ["--ego-model", "sbr2", "--ego-config", "vehicles/sbr2/x.pc"], 60,
                                enter_hook=lambda level: asked.append(ls.ego_spec()))
    finally:
        ls.set_ego(ls.EGO_MODEL_DEFAULT, ls.EGO_CONFIG_DEFAULT)
    assert code == 0, text[-2000:]
    assert asked and all(a == {"model": "sbr2", "part_config": "vehicles/sbr2/x.pc"} for a in asked), asked
    metas = [r["veh_meta"] for r in _rows(out) if "veh_meta" in r]
    assert metas and metas[0]["model"] == "sbr2" and metas[0]["part_config"] == "vehicles/sbr2/x.pc", metas
    ego = json.loads((out / "collect.json").read_text(encoding="utf-8"))["ego"]
    assert ego["model"] == "sbr2" and ego["camera_rig_model"] == "etk800", ego
    assert text.count("collect WARNING camera rig calibrated on etk800, ego model is sbr2") == 1, text[:3000]
    print("ego model: --ego-model sbr2 reaches the spawn; veh_meta model is the spawned sbr2 OK")


def check_empty_ego_model_stays_in_sync() -> None:
    """--ego-model "" spawns the default (set_ego falls back to EGO_MODEL_DEFAULT): the camera
    rig, veh_meta and collect.json name that same model, not "" or the mock's etk800."""
    import python.collect.level_session as ls

    asked: list[dict] = []
    try:
        code, text, out = _west(lambda n: "", ["--ego-model", "", "--ego-config", ""], 60,
                                enter_hook=lambda level: asked.append(ls.ego_spec()))
    finally:
        ls.set_ego(ls.EGO_MODEL_DEFAULT, ls.EGO_CONFIG_DEFAULT)
    assert code == 0, text[-2000:]
    want = {"model": ls.EGO_MODEL_DEFAULT, "part_config": ""}
    assert asked and all(a == want for a in asked), asked
    metas = [r["veh_meta"] for r in _rows(out) if "veh_meta" in r]
    assert metas and all(m["model"] == want["model"] and m["part_config"] == "" for m in metas), metas
    ego = json.loads((out / "collect.json").read_text(encoding="utf-8"))["ego"]
    assert ego["model"] == want["model"] and ego["part_config"] == "", ego
    assert ego["camera_rig_model"] == want["model"], ego
    assert "camera rig calibrated" not in text, text[:3000]
    print(f"ego model: --ego-model '' spawns {want['model']}; rig, veh_meta and collect.json agree OK")


def check_sbr_without_bbox_stops() -> None:
    """The eSBR has no fallback width (only the etk800 has one): no bbox stops the run with
    vehicle_width, code 2, before a row is written. (The mocked level-load harness runs
    --no-excursions, which needs no width, so this drives the sbr already in the level.)"""
    import scripts.test_collect_spawn_fix as sf

    code, text, out = sf._run_collector({"speed_at": lambda n: 10.0, "stop_at": 120, "bbox_w": None, "model": "sbr"}, [])
    assert code == 2, (code, text[-2000:])
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    assert status["stop_reason"] == "vehicle_width", status.get("stop_reason")
    assert "bbox failed: RuntimeError" in status["stop_detail"] and "model=sbr is not etk800" in status["stop_detail"], status
    assert "collect stop vehicle width unknown:" in text, text[-2000:]
    assert not (out / "state.jsonl").is_file() or not (out / "state.jsonl").read_text(encoding="utf-8").strip()
    print("vehicle width: an sbr without a bbox stops with vehicle_width, code 2 OK")


def _respawn_rows(rows: list[dict]) -> list[float]:
    """done_s (0.1 s per row) where a settle run starts: one per spawn."""
    out, prev = [], False
    for r in rows:
        on = bool(r.get("settle"))
        if on and not prev:
            out.append(round(0.1 * int(r["i"]), 1))
        prev = on
    return out


def _west(pattern, extra: list[str], stop_at: int, enter_hook=None):
    import scripts.collect_scene_lanes as collect
    import scripts.test_collect_spawn_fix as sf
    from python.collect.spawn_mix import CITY

    calls = {"n": 0}
    real = collect.boundary_wait_reason

    def fake(phase, settling):
        calls["n"] += 1
        return pattern(calls["n"]) or real(phase, settling)

    collect.boundary_wait_reason = fake  # type: ignore[assignment]
    try:
        here = sf._road("here", [(0.4, -300), (0.4, 600)])
        east = [sf._road(f"east{j}", [(5000 + 50 * j, 0), (5000 + 50 * j, 500)]) for j in range(40)]
        paint = sf._Paint(lambda x, y: True)
        return sf._run_west_coast(paint, {CITY: [here] + east, "highway": [], "ramp": []}, extra, stop_at,
                                  enter_hook=enter_hook)
    finally:
        collect.boundary_wait_reason = real  # type: ignore[assignment]


def check_respawn_deferred_run() -> None:
    # Weaving for the first 50 ticks of each due teleport (5 s), then cruise.
    code, text, out = _west(lambda n: "weave" if n <= 50 else "", ["--respawn-min", "0.5"], 520)
    assert code == 0, text[-2000:]
    starts = _respawn_rows(_rows(out))
    assert "collect boundary_defer respawn:weave 0.0" in text, text[-3000:]
    assert "collect boundary_defer respawn:cruise 5.0" in text, text[-3000:]
    assert len(starts) >= 2 and abs(starts[1] - 35.0) <= 0.3, starts
    # Always weaving: the 30 s cap teleports anyway.
    code, text, out = _west(lambda n: "weave", ["--respawn-min", "0.5"], 700)
    assert code == 0, text[-2000:]
    starts = _respawn_rows(_rows(out))
    assert "collect boundary_defer respawn:cap 30.0" in text, text[-3000:]
    assert len(starts) >= 2 and abs(starts[1] - 60.0) <= 0.3, starts
    print(f"respawn-min: teleport waits for cruise (35 s) or the cap (60 s) OK")


def check_respawn_failed_logs_once() -> None:
    """A due respawn the cap releases but _apply_spawn cannot place (here place_vehicle fails
    after the first spawn) must not re-arm the hold and log again every 30 s."""
    import python.collect.level_session as ls

    real = ls.place_vehicle
    calls = {"n": 0}

    def place(*a, **k):
        calls["n"] += 1
        return real(*a, **k) if calls["n"] == 1 else "shift 9.9 m"

    ls.place_vehicle = place  # type: ignore[assignment]
    try:
        code, text, out = _west(lambda n: "weave", ["--respawn-min", "0.5"], 1500)
    finally:
        ls.place_vehicle = real  # type: ignore[assignment]
    assert code == 0, text[-2000:]
    holds = [ln for ln in text.splitlines() if ln.startswith("collect boundary_defer respawn:")]
    assert holds == ["collect boundary_defer respawn:weave 0.0", "collect boundary_defer respawn:cap 30.0"], holds
    assert calls["n"] >= 3 and "failed shift 9.9 m" in text, (calls, text[-2000:])
    print(f"respawn: a failed due respawn holds and logs once over 150 s (holds={len(holds)}, tries={calls['n']}) OK")


def check_segment_deferred_run() -> None:
    code, text, out = _west(lambda n: "lane_change" if n <= 40 else "", ["--segment-min", "1"], 760)
    assert code == 0, text[-2000:]
    assert "collect boundary_defer segment:lane_change 0.0" in text, text[-3000:]
    assert "collect boundary_defer segment:cruise 4.0" in text, text[-3000:]
    rows = _rows(out)
    first = next(r for r in rows if r.get("segment") == "west_coast_usa#1")
    assert abs(0.1 * int(first["i"]) - 64.0) <= 0.3, first["i"]
    # No wait reason: the switch lands on the minute as before.
    code, text, out = _west(lambda n: "", ["--segment-min", "1"], 760)
    assert code == 0 and "boundary_defer" not in text, text[-3000:]
    first = next(r for r in _rows(out) if r.get("segment") == "west_coast_usa#1")
    assert abs(0.1 * int(first["i"]) - 60.0) <= 0.3, first["i"]
    print("segment-min: chunk switch waits for cruise (64 s vs 60 s) OK")


def _segments_in_order(rows: list[dict]) -> list[str]:
    seen: list[str] = []
    for r in rows:
        seg = str(r.get("segment") or "")
        if seg and (not seen or seen[-1] != seg):
            seen.append(seg)
    return seen


def check_segment_cap_run() -> None:
    # Always in a lane change: the 30 s cap switches the chunk anyway, at 90 s instead of 60 s.
    code, text, out = _west(lambda n: "lane_change", ["--segment-min", "1"], 1000)
    assert code == 0, text[-2000:]
    assert "collect boundary_defer segment:lane_change 0.0" in text, text[-3000:]
    assert "collect boundary_defer segment:cap 30.0" in text, text[-3000:]
    first = next(r for r in _rows(out) if r.get("segment") == "west_coast_usa#1")
    assert abs(0.1 * int(first["i"]) - 90.0) <= 0.3, first["i"]
    print("segment-min: an always-busy chunk switch hits the 30 s cap (90 s) OK")


def check_segment_hold_one_step() -> None:
    """A hold that outlasts a short weighted chunk advances one index, not two.

    200 s at west_coast_usa=0.9: pieces west 60, italy 20, west 60, west 60. The switch due at
    60 s waits 30 s (cap) to 90 s, past italy#1's end at 80 s; it still loads italy#1 next."""
    import scripts.collect_scene_lanes as collect

    maps = ("west_coast_usa", "italy")
    weights = {"west_coast_usa": 0.9}
    assert [round(d) for _n, d in collect.weighted_pieces(maps, 60.0, weights, 200.0)] == [60, 20, 60, 60]
    assert collect.segment_at(90.0, maps, 60.0, weights, 200.0) == ("west_coast_usa", 2)
    flags = ["--maps", "west_coast_usa,italy", "--map-weights", "west_coast_usa=0.9", "--segment-min", "1",
             "--minutes", str(200.0 / 60.0)]
    code, text, out = _west(lambda n: "lane_change" if n <= 400 else "", flags, 2100)
    assert code == 0, text[-2000:]
    assert "collect boundary_defer segment:cap 30.0" in text, text[-3000:]
    order = _segments_in_order(_rows(out))
    assert order[:3] == ["west_coast_usa#0", "italy#1", "west_coast_usa#2"], order
    idx = [int(s.split("#")[1]) for s in order]
    assert all(b - a == 1 for a, b in zip(idx, idx[1:])), order
    first = next(r for r in _rows(out) if r.get("segment") == "italy#1")
    assert abs(0.1 * int(first["i"]) - 90.0) <= 0.3, first["i"]
    print(f"segment hold: one index per release {order} OK")


def _seg_spans(rows: list[dict]) -> dict[str, tuple[float, float]]:
    """segment -> (first, last) done_s of its rows."""
    out: dict[str, tuple[float, float]] = {}
    for r in rows:
        seg, t = str(r.get("segment") or ""), 0.1 * int(r["i"])
        out[seg] = (out[seg][0], t) if seg in out else (t, t)
    return out


def check_segment_hold_carries_over() -> None:
    """A held switch's wait is carried over, then owed back evenly over the rest of the run.

    Same plan as check_segment_hold_one_step (west 60, italy 20, west 60, west 60, 200 s). The
    switch due at 60 s is held 30 s (cap: 300 busy ticks, the 301st releases). italy#1 starts at
    90 s from its own start, not with 10 s left of its slot (3.1 s in the italy probe). The 30 s
    is then repaid over the last 110 s, so every later chunk runs 110/140 of its slot (about
    15.7, 47.1 and 47.1 s): the tail no longer pays it all (west#3 was 29.9 s of 60). This holds
    because the hold is EARLY (110 s of the run left to spread over); a late hold is pinned by
    check_segment_hold_final_slot (tail >= slot - 30 s)."""
    flags = ["--maps", "west_coast_usa,italy", "--map-weights", "west_coast_usa=0.9", "--segment-min", "1",
             "--minutes", str(200.0 / 60.0)]
    code, text, out = _west(lambda n: "lane_change" if n <= 301 else "", flags, 2100)
    assert code == 0, text[-2000:]
    spans = _seg_spans(_rows(out))
    lengths = {seg: round(b - a + 0.1, 1) for seg, (a, b) in spans.items()}
    first, _last = spans["italy#1"]
    assert abs(first - 90.0) <= 0.3, spans
    share = 110.0 / 140.0
    for seg, slot in (("italy#1", 20.0), ("west_coast_usa#2", 60.0), ("west_coast_usa#3", 60.0)):
        assert abs(lengths[seg] - share * slot) <= 0.5, (seg, lengths)
    # With this early hold the tail keeps at least 3/4 of its slot (not true of a late hold).
    assert lengths["west_coast_usa#3"] >= 45.0, lengths
    holds = [ln for ln in text.splitlines() if ln.startswith("collect boundary_defer segment:")]
    assert holds == ["collect boundary_defer segment:lane_change 0.0", "collect boundary_defer segment:cap 30.0"], holds
    carry = json.loads((out / "collect.json").read_text(encoding="utf-8"))["segment_carry"]
    assert carry["carried_s"] == 30.0 and carry["owed_s"] <= 0.2, carry
    print(f"segment hold: carried 30 s, repaid over the rest; chunk lengths {lengths} OK")


def check_segment_hold_final_slot() -> None:
    """A hold at the last boundary has no rest of the run to spread over, so the tail pays it.

    Same 200 s plan; the switches at 60 and 80 s go at once and the one into the last chunk is
    held the full 30 s (cap). The carry is then owed over the 30 s left, a repay rate of 1, so
    west#3 runs 30 of its 60 s. That is the floor: a chunk loses at most the 30 s cap per hold,
    so the tail keeps at least its slot - 30 s."""
    flags = ["--maps", "west_coast_usa,italy", "--map-weights", "west_coast_usa=0.9", "--segment-min", "1",
             "--minutes", str(200.0 / 60.0)]
    code, text, out = _west(lambda n: "lane_change" if n >= 3 else "", flags, 2100)
    assert code == 0, text[-2000:]
    rows = _rows(out)
    order = _segments_in_order(rows)
    assert order == ["west_coast_usa#0", "italy#1", "west_coast_usa#2", "west_coast_usa#3"], order
    lengths = {seg: round(b - a + 0.1, 1) for seg, (a, b) in _seg_spans(rows).items()}
    assert lengths["west_coast_usa#3"] >= 60.0 - 30.0 - 0.2, lengths  # tail >= slot - 30 s (the cap)
    assert lengths["west_coast_usa#2"] >= 89.7, lengths  # held 30 s past its 60 s slot
    holds = [ln for ln in text.splitlines() if ln.startswith("collect boundary_defer segment:")]
    assert holds == ["collect boundary_defer segment:lane_change 0.0", "collect boundary_defer segment:cap 30.0"], holds
    carry = json.loads((out / "collect.json").read_text(encoding="utf-8"))["segment_carry"]
    assert carry["carried_s"] == 30.0 and carry["owed_s"] <= 0.2, carry
    print(f"segment hold at the last boundary: tail {lengths['west_coast_usa#3']:.1f} s of 60 (>= slot - 30 s) OK")


def check_segment_carry_always_busy() -> None:
    """Worst case: every switch hits the 30 s cap (always in a lane change), 6 min of 1 min chunks.
    120 s is carried in total and repaid by the end. Each hold here still has at least 2 chunks
    of run left to spread over, so no chunk, the tail included, drops under 3/4 of its 60 s
    slot; a hold at the last boundary only keeps slot - 30 s (check_segment_hold_final_slot).
    collect.json records the carried seconds."""
    flags = ["--maps", "west_coast_usa,italy", "--segment-min", "1", "--minutes", "6"]
    code, text, out = _west(lambda n: "lane_change", flags, 3700)
    assert code == 0, text[-2000:]
    order = _segments_in_order(_rows(out))
    spans = _seg_spans(_rows(out))
    lengths = {seg: round(b - a + 0.1, 1) for seg, (a, b) in spans.items()}
    assert all(v >= 45.0 for v in lengths.values()), lengths
    assert lengths[order[-1]] >= 45.0, (order, lengths)
    carry = json.loads((out / "collect.json").read_text(encoding="utf-8"))["segment_carry"]
    assert carry["carried_s"] == 120.0 and carry["owed_s"] <= 0.2, carry
    print(f"segment carry: 4 capped holds, {carry['carried_s']:.0f} s carried and repaid; "
          f"shortest chunk {min(lengths.values()):.1f} s (tail {lengths[order[-1]]:.1f} s) OK")


def main() -> None:
    check_spawn_passes_model_and_config()
    check_lua_fallback_has_config()
    check_paint_lime()
    check_ego_values_validated()
    check_defaults()
    check_boundary_defer_unit()
    check_carry_over_unit()
    check_veh_meta_and_rig_warning()
    check_ego_model_reaches_spawn()
    check_empty_ego_model_stays_in_sync()
    check_sbr_without_bbox_stops()
    check_respawn_deferred_run()
    check_respawn_failed_logs_once()
    check_segment_deferred_run()
    check_segment_cap_run()
    check_segment_hold_one_step()
    check_segment_hold_carries_over()
    check_segment_hold_final_slot()
    check_segment_carry_always_busy()
    print("recorder ev ok")


if __name__ == "__main__":
    main()
