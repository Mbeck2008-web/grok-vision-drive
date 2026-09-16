#!/usr/bin/env python3
"""Offline checks for the in-game GVD app and its viz bridge (no BeamNG required).

Guards what breaks silently: app.json wiring, the app.js <-> gvd_main Lua contract, the
"no FSD / Tesla chrome" rule, the M6 retail drive path surviving UI edits, and the road
model never inventing lanes it did not see.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "beamng_mod" / "ui" / "modules" / "apps" / "GVD"
LUA = ROOT / "beamng_mod" / "lua" / "ge" / "extensions" / "gvd" / "main.lua"
RUN_VISION = ROOT / "python" / "run_vision.py"

BANNED = [
    (r"tesla", "Tesla marks"),
    (r"\bfsd\b", "FSD label"),
    (r"full[\s-]*self[\s-]*driv", "Full Self-Driving label"),
    (r"autopilot", "Autopilot label"),
]


def main() -> None:
    app_json = json.loads((APP / "app.json").read_text(encoding="utf-8"))
    app_js = (APP / "app.js").read_text(encoding="utf-8")
    app_html = (APP / "app.html").read_text(encoding="utf-8")
    lua = LUA.read_text(encoding="utf-8")
    run_vision = RUN_VISION.read_text(encoding="utf-8")

    # app.json wiring: directive name MUST match app.js (camelCase). BeamNG
    # UiAppsService checks $injector.has(directive + 'Directive'); kebab-case
    # "gvd-app" looks up gvd-appDirective and the tile stays blank after app.js loads.
    assert app_json["name"] == "GVD", app_json["name"]
    assert app_json["directive"] == "gvdApp", app_json["directive"]
    assert app_json["domElement"] == "<gvd-app></gvd-app>", app_json["domElement"]
    css = app_json["css"]
    assert isinstance(css, dict), "app.json css must be an object (not a JSON string)"
    for key in ("width", "height"):
        assert css[key].endswith("px"), css
    assert ".directive('gvdApp'" in app_js, "directive gvdApp missing (must match app.json)"
    assert "/ui/modules/apps/GVD/app.html" in app_js, "templateUrl must be the absolute mod path"
    assert (APP / "app.png").is_file(), "app.png icon missing"
    strip_json = json.loads((ROOT / "beamng_mod" / "ui" / "modules" / "apps" / "gvd_strip"
                             / "app.json").read_text(encoding="utf-8"))
    assert strip_json["directive"] == "gvdStrip", strip_json["directive"]
    assert isinstance(strip_json["css"], dict), "gvd_strip css must be an object"

    # Titles stay GVD / VISION, and no Tesla / FSD chrome anywhere the player can see.
    assert "GVD" in app_html and "VISION" in app_html
    assert "ALT+G" in app_html and "ALT+A" not in app_html
    for name, text in (("app.js", app_js), ("app.html", app_html), ("main.lua", lua),
                       ("app.json", json.dumps(app_json))):
        low = text.lower()
        for pattern, label in BANNED:
            hit = re.search(pattern, low)
            assert hit is None, f"{name}: {label} found ({hit.group(0)!r})"

    # CEF / ActionMap / inputmap strings must stay ASCII (no BOM). BeamNG's
    # in-game CEF drops app.js on em-dash / middot / fancy quotes, and Windows
    # ActionMap addBinding fails when the action title/desc is missing or invalid.
    cef_paths = [
        APP / "app.js", APP / "app.html", APP / "app.json",
        ROOT / "beamng_mod" / "ui" / "modules" / "apps" / "gvd_strip" / "app.js",
        ROOT / "beamng_mod" / "ui" / "modules" / "apps" / "gvd_strip" / "app.html",
        ROOT / "beamng_mod" / "ui" / "modules" / "apps" / "gvd_strip" / "app.json",
        ROOT / "beamng_mod" / "lua" / "ge" / "extensions" / "core" / "input" / "actions" / "gvd.json",
        ROOT / "beamng_mod" / "settings" / "inputmaps" / "keyboardGvd.json",
        ROOT / "python" / "viz" / "monitors.py",
    ]
    for path in cef_paths:
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{path.name}: UTF-8 BOM"
        bad = sorted({ch for ch in raw.decode("utf-8") if ord(ch) > 127})
        assert not bad, f"{path.name}: non-ASCII {bad!r}"
        raw.decode("utf-8").encode("cp1252")

    # Every gvd_main call the app makes must exist in the Lua extension.
    called = set(re.findall(r"extensions\.gvd_main\.(\w+)\s*\(", app_js))
    defined = set(re.findall(r"function M\.(\w+)\s*\(", lua))
    missing = sorted(called - defined)
    assert not missing, f"app.js calls gvd_main functions that do not exist: {missing}"
    for required in ("toggleEngage", "setShowPath", "setShowAgentGhosts",
                     "requestPolicy", "requestVizScreen", "pushUiState"):
        assert required in defined, f"gvd_main.{required} missing"
        assert required in app_js, f"app.js never calls gvd_main.{required}"
    # Scene canvas is gone; Lua keeps setShowScene so old prefs do not error.
    assert "setShowScene" in defined
    assert "gvd-canvas" not in app_html and "drawGround" not in app_js
    assert "requestAnimationFrame" not in app_js

    # Angular template only binds handlers the directive actually publishes.
    bound = set(re.findall(r'ng-(?:click|change)="(\w+)\(', app_html))
    bound |= set(re.findall(r"\{\{\s*(\w+)\(", app_html))
    published = set(re.findall(r"scope\.(\w+)\s*=", app_js))
    unbound = sorted(bound - published)
    assert not unbound, f"app.html binds scope functions that app.js never defines: {unbound}"

    # Engage still has to be reachable without the app. Alt+A is stock BeamNG
    # toggleRangeStatus — GVD uses Alt+G (Ctrl+Alt+G fallback) and does not steal it.
    actions = json.loads((ROOT / "beamng_mod" / "lua" / "ge" / "extensions" / "core" / "input"
                          / "actions" / "gvd.json").read_text(encoding="utf-8"))
    title, desc = actions["gvd_toggle_engage"]["title"], actions["gvd_toggle_engage"]["desc"]
    assert title and desc and all(ord(ch) < 128 for ch in title + desc), (title, desc)
    assert "toggleEngage" in actions["gvd_toggle_engage"]["onDown"]
    keymap = json.loads((ROOT / "beamng_mod" / "settings" / "inputmaps"
                         / "keyboardGvd.json").read_text(encoding="utf-8"))
    controls = {b["control"] for b in keymap["bindings"]}
    assert "alt+g" in controls and "ctrl+alt+g" in controls, keymap
    assert "alt+a" not in controls, "do not steal stock toggleRangeStatus"
    # ActionMap: core_input_actions.load / loadActions, then
    # core_input_bindings.reloadBindings(). loadActions is not on bindings (#22).
    lua_exec = re.sub(r"--[^\n]*", "", lua)

    def lua_fn(name: str) -> str:
        m = re.search(rf"local function {re.escape(name)}\s*\([^)]*\).*?\nend\n", lua, re.S)
        assert m, f"missing lua function {name}"
        return m.group(0)

    refresh_fn = lua_fn("refreshEngageBindings")
    load_fn = lua_fn("loadEngageActions")
    reload_fn = lua_fn("reloadEngageBindings")
    retry_fn = lua_fn("retryEngageBindings")
    tick_fn = lua_fn("tickPush")
    try_fn = lua_fn("tryCall")
    clear_fn = lua_fn("clearActionsCache")
    lookup_fn = lua_fn("lookupEngageAction")
    wipe_fn = lua_fn("wipeActionsCacheTable")
    refresh_exec = re.sub(r"--[^\n]*", "", refresh_fn)
    load_exec = re.sub(r"--[^\n]*", "", load_fn)
    reload_exec = re.sub(r"--[^\n]*", "", reload_fn)
    retry_exec = re.sub(r"--[^\n]*", "", retry_fn)
    try_exec = re.sub(r"--[^\n]*", "", try_fn)
    clear_exec = re.sub(r"--[^\n]*", "", clear_fn)

    assert "core_input_actions" in lua and "getActiveActions" in lua
    assert "gvd_toggle_engage" in lua and "refreshEngageBindings" in lua
    assert "retryEngageBindings" in lua
    # Never the #22 / 1543777 call site.
    assert "b.loadActions" not in lua
    assert "bindings.loadActions" not in lua_exec
    assert "core_input_bindings.loadActions" not in lua_exec
    # Required sequence lives in executable code, not comments.
    assert "actions.load" in load_exec, "must call core_input_actions.load"
    assert "actions.loadActions" in load_exec, "must call core_input_actions.loadActions"
    assert "reloadBindings" in reload_exec, "must call core_input_bindings.reloadBindings"
    assert "loadEngageActions" in refresh_exec
    assert "reloadEngageBindings" in refresh_exec
    assert "clearActionsCache" in refresh_exec
    assert refresh_exec.find("clearActionsCache") < refresh_exec.find("loadEngageActions")
    assert refresh_exec.find("loadEngageActions") < refresh_exec.find("reloadEngageBindings")
    assert "elseif type(bindings.reloadBindings)" not in lua
    assert not re.search(
        r"onFileChanged[\s\S]{0,400}elseif[^\n]*reloadBindings", refresh_fn
    ), "onFileChanged must not skip reloadBindings"
    else_at = refresh_exec.find("else")
    reload_at = refresh_exec.find("reloadEngageBindings")
    assert else_at == -1 or reload_at < else_at, "reloadBindings must run before any onFileChanged fallback"
    # Always clear actionsCache before load -- do not skip the cache drop.
    assert "actionsCache" in clear_exec and "normalActionsCache" in clear_exec
    assert "onFileChanged" in clear_exec
    assert "cache[true]" in wipe_fn and "cache[false]" in wipe_fn
    # 0.36/0.39 dump shapes: flat map or boolean-keyed cache wrapper.
    assert "root[true]" in lookup_fn and "root[false]" in lookup_fn
    assert "root.actions" in lookup_fn and "root.normal" in lookup_fn
    # pcall failures are logged, not swallowed.
    assert "failed:" in try_exec
    assert "tryCall" in load_exec and "tryCall" in reload_exec
    # BeamNG onFileChanged arity is (filename, type).
    assert ", 'added')" in lua
    assert "'added'" in lua
    # title/desc from getActiveActions is not bindReady by itself; dump-shape walk gates ready.
    assert "engageActionListed" in refresh_exec
    assert "reloadStatus == 'ok' and engageActionListed()" in refresh_exec
    assert refresh_exec.count("engageActionListed()") >= 2
    assert "reloadStatus" in refresh_fn
    assert refresh_fn.find("reloadEngageBindings") < refresh_fn.find("markBindReady")
    assert "reloadStatus == 'ok'" in refresh_exec
    assert "reloadStatus == 'missing'" in refresh_exec
    assert "retryEngageBindings(dt)" in lua_exec or "retryEngageBindings(dt)" in retry_fn
    assert "BIND_DEFER_S" in lua
    assert "bindFileChangedRan" in retry_exec, "fail log only if onFileChanged actually ran"
    notify_fn = lua_fn("notifyBindingsFileChanged")
    notify_exec = re.sub(r"--[^\n]*", "", notify_fn)
    assert "return a or b" in notify_exec, "notifyBindingsFileChanged must succeed only when pcall succeeds"
    assert re.search(r"return true\b", notify_exec) is None, "do not mark notify success just because the export exists"
    bind_test = (ROOT / "scripts" / "test_gvd_bind_reload.lua").read_text(encoding="utf-8")
    assert "neither load nor loadActions" in bind_test
    assert "Test B still has loadActions" in bind_test

    # Retail Direct Drive echo on the slim HUD -- reuse egoFb, do not invent a bus.
    assert "WHEEL / PEDALS" in app_html
    for key in ("steerInput", "throttleInput", "brakeInput",
                "playerSteer", "playerThrottle", "playerBrake", "playerDevice"):
        assert f"{key} =" in lua or f"{key}=" in lua, f"gvdUi missing {key}"
        assert key in app_js, f"app.js never reads {key}"
    assert "axisTxt" in app_js and "inputNote" in app_js
    # Wheel HUD cadence: one 10 Hz tickPush path. Not 4 Hz strip + 10 Hz ego pusher.
    assert "UI_PUSH_S" in lua
    assert re.search(r"UI_PUSH_S\s*=\s*EGO_POLL_S\b", lua), "HUD push must match 10 Hz egoFb poll"
    assert re.search(r"EGO_POLL_S\s*=\s*0\.10\b", lua)
    ego_decl = next(ln for ln in lua.splitlines() if ln.startswith("local EGO_POLL_S"))
    assert "supervisor" not in ego_decl.lower(), "EGO_POLL_S comment must not say supervisor-only"
    assert "supervisorAlive" not in lua_fn("pollEgo")
    assert "0.25" not in tick_fn
    assert "UI_PUSH_S" in tick_fn
    assert "pushUi()" in tick_fn
    ego_fn = lua[lua.index("function M.onEgoFeedback"):lua.index("local function pollEgo")]
    assert "pushUi()" not in ego_fn, "onEgoFeedback must not be a second HUD pusher"
    assert "egoFb" in lua and "gvd_ego.json" in lua

    # One prefs bus: what Lua writes is what the supervisor reads.
    written = set(re.findall(r'"(show_path|show_agent_ghosts|show_scene|policy|viz_screen)"\s*:', lua))
    assert {"show_path", "show_agent_ghosts", "show_scene", "policy", "viz_screen"} <= written, written
    for key in ("show_path", "show_agent_ghosts", "policy", "viz_screen"):
        assert f'"{key}"' in run_vision, f"run_vision.py ignores gvd_ui_prefs {key}"
    # Session gate: a pref from a past run must not override --policy at launch.
    assert "_ui_request_is_live" in run_vision

    # The UI layer shares main.lua with the M6 retail drive: a HUD refactor must not eat it.
    for symbol in ("VE_APPLY_FMT", "VE_RELEASE", "VE_HOLD", "VE_ARCADE", "VE_FEEDBACK", "CMD_STALE_S",
                   "CMD_DEAD_S", "applyCmdJson", "releaseInputs", "syncEngageFromSupervisor",
                   "function M.onEgoFeedback", "gvd_ego.json", "setAllowedInputSource"):
        assert symbol in lua, f"main.lua lost the retail drive path: {symbol}"
    assert "input.event('steering'" in lua and "input.event('throttle'" in lua
    assert "input.event('brake'" in lua and "input.event('parkingbrake'" in lua and "input.event('clutch'" in lua
    assert "2,900,0,nil,'gvd'" in lua, "retail must apply as a secondary Direct Drive wheel"
    assert "2,0,0,nil,'gvd'" in lua, "retail must apply throttle/brake as Direct Drive pedals"
    assert "setAllowedInputSource('steering','local',false)" in lua, "retail must lock out the player wheel"
    assert "setAllowedInputSource('throttle','local',false)" in lua, "retail must lock out the player gas pedal"
    assert "setAllowedInputSource('brake','local',false)" in lua, "retail must lock out the player brake pedal"
    assert "setAllowedInputSource('parkingbrake','local',false)" in lua, "retail must lock out a resting handbrake"
    assert "setAllowedInputSource('clutch','local',false)" in lua, "retail must lock out a resting clutch"
    assert "applying = applying" in lua, "gvdUi must carry the drive flag for the DRIVE state"
    assert "'DRIVE'" in app_js and "steer to take over" in app_js
    assert "engage takes the wheel" in app_html

    # Path / ghosts keep drawing while OFF; engage is underglow + applyCmdJson only.
    start = lua.index("function M.drawPath(dt)")
    nxt = lua.index("\nlocal function r2", start)
    draw_path = lua[start:nxt]
    assert "if not showPath then return end" in draw_path
    assert "if not engaged then return end" not in draw_path, "drawPath must not hide ribbon when disengaged"
    assert "if engaged and veh" in draw_path, "underglow stays engage-only"
    assert "releaseInputs('disengaged')" in lua

    # Lua may still pack scene geometry for old layouts; the in-game app no longer draws it.
    for key in ("lanes", "edges", "signs", "fans", "tracks", "path"):
        assert re.search(rf"\b{key} = ", lua), f"gvdUi payload missing {key}"

    _check_road_model()
    _check_opencv_lexicon()
    print("test_gvd_ui_app: OK")


def _check_road_model() -> None:
    """Predicted lanes need a detected anchor; kerbs are never claimed as detected."""
    sys.path.insert(0, str(ROOT))
    from python.perception.detect import STATIC_CLASSES, coco_class_name
    from python.perception.road_model import LANE_CONF_MIN, NEIGHBOUR_LANES, lanes_ext, road_edges

    assert coco_class_name(9) == "traffic_light" and coco_class_name(11) == "stop_sign"
    assert coco_class_name(10) == "pole" and coco_class_name(12) == "pole"
    assert coco_class_name(2) == "vehicle" and coco_class_name(63) is None
    assert set(STATIC_CLASSES) == {"traffic_light", "stop_sign", "pole"}

    lanes = [[{"x": -1.8, "y": y} for y in range(2, 30, 4)],
             [{"x": 1.8, "y": y} for y in range(2, 30, 4)]]
    ext = lanes_ext(lanes, 0.8)
    kinds = [l["kind"] for l in ext]
    assert kinds.count("detected") == 2, ext
    assert kinds.count("predicted") == 2 * NEIGHBOUR_LANES, ext
    assert {l["index"] for l in ext} == {-1, 1, -2, 2, -3, 3}, ext
    # the fan must stay ordered outwards, so the app can fade by |index|
    for lane in ext:
        assert abs(lane["index"]) <= NEIGHBOUR_LANES + 1
    assert all(e["kind"] == "predicted" for e in road_edges(ext)), "kerbs are never detected"
    # No paint seen → nothing predicted, no kerbs. Weak fit → detected only.
    assert lanes_ext([], 0.9) == [] and road_edges([]) == []
    weak = lanes_ext(lanes, LANE_CONF_MIN - 0.01)
    assert all(l["kind"] == "detected" for l in weak) and len(weak) == 2

    # Road furniture must never reach the tracker (CIPV / AEB / ghosts read tracks).
    from python.perception.pipeline import ModularPerception
    import numpy as np

    out = ModularPerception(allow_synthetic=True, detector_id="synthetic").tick(
        np.zeros((240, 320, 3), dtype=np.uint8)
    )
    assert {s["cls"] for s in out.signs} == {"stop_sign", "traffic_light"}, out.signs
    assert all(t["class"] not in STATIC_CLASSES for t in out.tracks), out.tracks
    light = [s for s in out.signs if s["cls"] == "traffic_light"][0]
    assert light["state"] == "unknown", "no lamp-colour classifier exists — never guess"


def _check_opencv_lexicon() -> None:
    """Second-screen stage owns the VISION lexicon; titles stay GVD / VISION."""
    stage = (ROOT / "python" / "viz" / "stage.py").read_text(encoding="utf-8")
    nerd = (ROOT / "python" / "viz" / "nerd.py").read_text(encoding="utf-8")
    for name, text in (("stage.py", stage), ("nerd.py", nerd)):
        low = text.lower()
        for pattern, label in BANNED:
            hit = re.search(pattern, low)
            assert hit is None, f"{name}: {label} found ({hit.group(0)!r})"
    for symbol in ("lane_draw_mode", "is_slowing", "is_halted", "is_hazard",
                   "_draw_lanes", "_draw_edges", "_draw_signs",
                   "_draw_stop_bar", "_draw_tracks"):
        assert symbol in stage, f"stage.py lost lexicon helper {symbol}"
    from python.viz.stage import is_halted, is_hazard, is_slowing, lane_draw_mode

    assert lane_draw_mode("detected", smoke=False) == "solid"
    assert lane_draw_mode("predicted", smoke=False) == "dashed"
    assert lane_draw_mode("stub", smoke=False) is None
    assert lane_draw_mode("stub", smoke=True) == "dashed"
    st = {"planner": {"aeb": "off", "target_v": 8.0, "ttc_lead": 2.4, "cipv_id": 1},
          "ego": {"speed_mps": 12.0, "brake": 0.0}}
    assert is_slowing(st) and not is_halted(st)
    st["planner"]["aeb"] = "brake"
    assert is_halted(st)
    lead = {"id": 1, "class": "vehicle", "x": 0.1, "y": 14}
    assert is_hazard(lead, st, 1)
    assert not is_hazard({"id": 2, "x": 3, "y": 20}, st, 1)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"test_gvd_ui_app: FAIL - {e}")
        sys.exit(1)
