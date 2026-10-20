#!/usr/bin/env python3
"""Every raw Lua call GVD sends must exist in BeamNG.tech 0.39.4.0.

``tests/fixtures/beamng_0.39.4.0_lua_exports.json`` is the export list of
each Lua module GVD calls, read from the 0.39.4.0 install on the dev PC
(``ai``, ``mapmgr``, ``input`` in the vehicle VM; ``core_environment``,
``core_weather``, ``core_vehicles``, ``freeroam_freeroam`` in the game-engine
VM). The collector sent ``ai.setLine`` for every weave: 0.39.4.0's ``ai`` has
no such function, the call failed in the vehicle VM and the car drifted. This
test would have failed on that.

1. Static: every string literal (not docstrings) in the collector,
   ``python/collect``, ``python/runtime`` and ``python/control`` is scanned
   for ``module.name(`` calls and ``module.name`` reads. A module in the
   fixture must export the name. Any other ``x.y(`` / ``x:y(`` must be a
   reviewed entry below (Lua stdlib, an engine binding, or Python text).
2. VM: the chunks the producers build are checked against the VM they are
   sent to (vehicle ``queue_lua_command`` vs game-engine ``queue_lua_command``).

``KNOWN_BROKEN`` lists flagged calls that are left as they are in this
change. The test fails if that list changes either way.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / "tests" / "fixtures" / "beamng_0.39.4.0_lua_exports.json"
SCOPE = [ROOT / "scripts" / "collect_scene_lanes.py"] + [
    p for d in ("python/collect", "python/runtime", "python/control") for p in sorted((ROOT / d).glob("*.py"))
]

CALL = re.compile(r"(?<![\w.:])([A-Za-z_]\w*)([.:])([A-Za-z_]\w*)\s*\(")
READ = re.compile(r"(?<![\w.:])([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b(?!\s*\()")

# ``debug``: the line readback reads the ai-local route with debug.getupvalue.
# The vehicle VM has it (0.39.4.0 ai.lua 169, powertrain.lua 1207 call it);
# only the game-engine techCore sets ``debug = nil``.
LUA_STDLIB = {"string", "math", "table", "os", "io", "coroutine", "debug"}
# ``x:y(`` receivers that are engine objects (C++ bindings, not Lua modules).
ENGINE_RECEIVERS = {"be", "veh", "v", "obj", "FS"}
# ``x.y(`` text in string literals that is not Lua sent to the game (log and
# error messages, file names, Python names in help text).
NOT_LUA = {
    "BeamNGpy.close": "help text",
    "Vehicle.get_bbox": "log text",
    "clock.start_move": "comment text",
    "gvd_ego.json": "file name",
    "scenario.make": "error text (beamngpy Scenario.make)",
    "vehicle.control": "help text (beamngpy)",
}
# Flagged and left alone in this change. Collector environment_commands():
# sent through the *vehicle* VM, where core_environment / core_weather are nil
# (beamng.log: "attempt to index global 'core_environment' (a nil value)"), and
# 0.39.4.0's core_weather has no setWeather (switchWeather / activate). The
# segment weather / time-of-day variation has never applied; fixing it changes
# what recordings look like, so it is a separate decision.
KNOWN_BROKEN = {
    "core_weather.setWeather": "not exported by 0.39.4.0 core_weather (switchWeather, activate)",
}
KNOWN_VM_MISMATCH = {
    "core_environment.setTimeOfDay": "GE module sent to the vehicle VM (environment_commands via the vehicle)",
    "core_weather.setWeather": "GE module sent to the vehicle VM (environment_commands via the vehicle)",
}


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _modules(fx: dict) -> dict[str, tuple[str, set[str], set[str]]]:
    """module -> (vm, functions, functions | fields | dynamic_fields)."""
    out = {}
    for vm, mods in fx["vms"].items():
        for name, info in mods.items():
            funcs = set(info["functions"])
            out[name] = (vm, funcs, funcs | set(info["fields"]) | set(info["dynamic_fields"]))
    return out


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def lua_strings(paths: list[Path]) -> list[tuple[str, int, str]]:
    found = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        skip = _docstring_ids(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
                found.append((str(path.relative_to(ROOT)), int(getattr(node, "lineno", 0)), node.value))
    return found


def audit(paths: list[Path], fx: dict) -> tuple[list[str], dict[str, set[str]], list[str]]:
    """(problems, every audited call -> where, reviewed non-module calls)."""
    mods = _modules(fx)
    problems: list[str] = []
    calls: dict[str, set[str]] = {}
    other: list[str] = []
    for where, line, text in lua_strings(paths):
        at = f"{where}:{line}"
        for m in CALL.finditer(text):
            mod, sep, name = m.groups()
            key = f"{mod}{sep}{name}"
            if sep == ":":
                if mod not in ENGINE_RECEIVERS:
                    problems.append(f"{at}: {key}( unreviewed method call")
                else:
                    other.append(f"{key} (engine binding) {at}")
                continue
            if mod in mods:
                calls.setdefault(key, set()).add(at)
                if name not in mods[mod][1]:
                    problems.append(f"{at}: {key}( not exported by 0.39.4.0 {mod} ({mods[mod][0]} VM)")
            elif mod in LUA_STDLIB:
                continue
            elif key in NOT_LUA:
                other.append(f"{key} ({NOT_LUA[key]}) {at}")
            else:
                problems.append(f"{at}: {key}( unreviewed: not a fixture module, Lua stdlib, or reviewed text")
        for m in READ.finditer(text):
            mod, name = m.groups()
            if mod in mods and name not in mods[mod][2]:
                problems.append(f"{at}: {mod}.{name} not a 0.39.4.0 {mod} export")
            elif mod in mods:
                calls.setdefault(f"{mod}.{name}", set()).add(at)
    return problems, calls, other


def _nested(chunk: str) -> tuple[str, list[str]]:
    """GE chunk without its ``veh:queueLuaCommand('...')`` bodies, and those bodies (vehicle VM)."""
    inner = re.findall(r"queueLuaCommand\('((?:[^'\\]|\\.)*)'\)", chunk)
    outer = re.sub(r"queueLuaCommand\('((?:[^'\\]|\\.)*)'\)", "queueLuaCommand(...)", chunk)
    return outer, inner


def vm_problems(chunk: str, vm: str, fx: dict, label: str) -> list[str]:
    mods = _modules(fx)
    out = []
    for m in CALL.finditer(chunk):
        mod, sep, name = m.groups()
        if sep != "." or mod not in mods:
            continue
        mvm, funcs, _all = mods[mod]
        if mvm != vm:
            out.append(f"{label}: {mod}.{name} is a {mvm}-VM module, sent to the {vm} VM")
        if name not in funcs:
            out.append(f"{label}: {mod}.{name} not exported by 0.39.4.0 {mod}")
    return out


def produced_chunks() -> list[tuple[str, str, str]]:
    """(label, vm, chunk) for every producer of Lua the collector / control send."""
    from python.collect.auto_mode import environment_commands
    from python.collect.excursion import excursion_commands
    from python.collect.spawn_mix import SpawnPick, traffic_lua
    from python.control.actuate import TECH_DRIVE_SHIFT_LUA
    import python.collect.level_session as level_session
    import scripts.collect_scene_lanes as collect

    out: list[tuple[str, str, str]] = []
    path = [(0.0, 0.0), (0.4, 4.0), (0.8, 8.0)]
    for phase in ("cruise", "weave", "lane_change", "recover", "abort"):
        for chunk in excursion_commands(12.0, phase, 1, path=path):
            out.append((f"excursion_commands({phase}, path)", "vehicle", chunk))
        for chunk in excursion_commands(12.0, phase, -1):
            out.append((f"excursion_commands({phase})", "vehicle", chunk))
    for waypoint in (None, "wp_a"):
        for traffic in (False, True):
            for chunk in collect.ai_commands(12.0, waypoint, traffic=traffic):
                out.append((f"ai_commands({waypoint}, traffic={traffic})", "vehicle", chunk))
    try:
        from python.collect.ai_line import line_state_lua, pre_line_commands
    except ImportError:  # before the set_line fix
        pass
    else:
        out += [("pre_line_commands", "vehicle", c) for c in pre_line_commands()]
        out.append(("line_state_lua", "vehicle", line_state_lua(3)))
    out.append(("TECH_DRIVE_SHIFT_LUA", "vehicle", TECH_DRIVE_SHIFT_LUA))
    # The collector queues these through the vehicle (``_queue(vehicle, ...)``).
    for chunk in environment_commands("0.35", "clear"):
        out.append(("environment_commands (sent via the vehicle)", "vehicle", chunk))
    pick = SpawnPick(kind="city", pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), speed=(8.0, 14.0), road_id="r", paint_cov=1.0)
    for chunk in traffic_lua(pick, 2):
        outer, inner = _nested(chunk)
        out.append(("traffic_lua", "ge", outer))
        out += [("traffic_lua veh:queueLuaCommand", "vehicle", c.replace('\\"', '"')) for c in inner]

    class _Ge:
        control = None

        def __init__(self) -> None:
            self.sent: list[str] = []

        def queue_lua_command(self, chunk: str, *a: object) -> None:
            self.sent.append(chunk)

    ge = _Ge()
    level_session.queue_level(ge, "west_coast_usa")
    out += [("queue_level", "ge", c) for c in ge.sent]
    return out


def main() -> None:
    fx = _fixture()
    assert fx["beamng_tech"] == "0.39.4.0"
    assert "setLine" not in fx["vms"]["vehicle"]["ai"]["functions"]
    assert "driveUsingPath" in fx["vms"]["vehicle"]["ai"]["functions"]

    problems, calls, other = audit(SCOPE, fx)
    print("audited calls (0.39.4.0 exports):")
    for key in sorted(calls):
        print(f"  {key}  {', '.join(sorted(calls[key]))}")
    print("reviewed, not audited against the fixture:")
    for line in sorted(set(other)):
        print(f"  {line}")
    flagged = {p.split(": ", 1)[1].split("(")[0] for p in problems}
    unexpected = [p for p in problems if p.split(": ", 1)[1].split("(")[0] not in KNOWN_BROKEN]
    for p in problems:
        print(("KNOWN  " if p not in unexpected else "FLAG   ") + p)

    vm_issues = []
    for label, vm, chunk in produced_chunks():
        vm_issues += vm_problems(chunk, vm, fx, label)
    vm_keys = {re.search(r": (\S+) ", p).group(1) for p in vm_issues}
    vm_unexpected = [p for p in vm_issues if re.search(r": (\S+) ", p).group(1) not in KNOWN_VM_MISMATCH]
    for p in sorted(set(vm_issues)):
        print(("KNOWN  " if p not in vm_unexpected else "FLAG   ") + p)

    assert not unexpected, "raw Lua calls not in 0.39.4.0:\n" + "\n".join(unexpected)
    assert not vm_unexpected, "raw Lua sent to the wrong VM / not exported:\n" + "\n".join(sorted(set(vm_unexpected)))
    # The known list is exact: a fix (or a new break of the same name) must update it.
    assert flagged == set(KNOWN_BROKEN), (flagged, set(KNOWN_BROKEN))
    assert {k for k in vm_keys if k in KNOWN_VM_MISMATCH} == set(KNOWN_VM_MISMATCH), vm_keys

    # Self-check: the scanner flags a setLine line like the one the collector used to send.
    bad = 'ai.setLine({{pos={1.00,2.00,0},speed=8.00}}, true)'
    assert vm_problems(bad, "vehicle", fx, "self-check") == ["self-check: ai.setLine not exported by 0.39.4.0 ai"]
    print("lua exports ok")


if __name__ == "__main__":
    main()
