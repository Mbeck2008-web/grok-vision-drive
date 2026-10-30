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
           ('sb"r', ""), ("sbr; rm", ""), ("sbr", "vehicles/sbr/electric 800.pc")]
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
    for argv in (["--ego-config", 'vehicles/sbr/x".pc'], ["--ego-config", "vehicles\\sbr\\x.pc"],
                 ["--ego-model", "sb'r"]):
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
    print("ego values: quote, backslash path, space refused (set_ego and the collector, exit 2) OK")


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


def _respawn_rows(rows: list[dict]) -> list[float]:
    """done_s (0.1 s per row) where a settle run starts: one per spawn."""
    out, prev = [], False
    for r in rows:
        on = bool(r.get("settle"))
        if on and not prev:
            out.append(round(0.1 * int(r["i"]), 1))
        prev = on
    return out


def _west(pattern, extra: list[str], stop_at: int):
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
        return sf._run_west_coast(paint, {CITY: [here] + east, "highway": [], "ramp": []}, extra, stop_at)
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


def main() -> None:
    check_spawn_passes_model_and_config()
    check_lua_fallback_has_config()
    check_paint_lime()
    check_ego_values_validated()
    check_defaults()
    check_boundary_defer_unit()
    check_veh_meta_and_rig_warning()
    check_respawn_deferred_run()
    check_segment_deferred_run()
    check_segment_cap_run()
    check_segment_hold_one_step()
    print("recorder ev ok")


if __name__ == "__main__":
    main()
