#!/usr/bin/env python3
"""Offline checks for the in-game GVD app (no BeamNG required).

Guards the three things that break silently: app.json wiring, the app.js <-> gvd_main
Lua contract, and the "no FSD / Tesla chrome" rule.
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

    # app.json wiring: directive name, template path, default geometry
    assert app_json["name"] == "GVD", app_json["name"]
    assert app_json["directive"] == "gvd-app", app_json["directive"]
    assert app_json["domElement"] == "<gvd-app></gvd-app>", app_json["domElement"]
    css = json.loads(app_json["css"])
    for key in ("width", "height"):
        assert css[key].endswith("px"), css
    assert "gvdApp" in app_js, "directive gvdApp missing (app.json says gvd-app)"
    assert "/ui/modules/apps/GVD/app.html" in app_js, "templateUrl must be the absolute mod path"
    assert (APP / "app.png").is_file(), "app.png icon missing"

    # Titles stay GVD / VISION, and no Tesla / FSD chrome anywhere the player can see.
    assert "GVD" in app_html and "VISION" in app_html
    for name, text in (("app.js", app_js), ("app.html", app_html), ("main.lua", lua),
                       ("app.json", json.dumps(app_json))):
        low = text.lower()
        for pattern, label in BANNED:
            hit = re.search(pattern, low)
            assert hit is None, f"{name}: {label} found ({hit.group(0)!r})"

    # Every gvd_main call the app makes must exist in the Lua extension.
    called = set(re.findall(r"extensions\.gvd_main\.(\w+)\s*\(", app_js))
    defined = set(re.findall(r"function M\.(\w+)\s*\(", lua))
    missing = sorted(called - defined)
    assert not missing, f"app.js calls gvd_main functions that do not exist: {missing}"
    for required in ("toggleEngage", "setShowPath", "setShowAgentGhosts", "setShowScene",
                     "requestPolicy", "requestVizScreen", "pushUiState"):
        assert required in defined, f"gvd_main.{required} missing"
        assert required in app_js, f"app.js never calls gvd_main.{required}"

    # Angular template only binds handlers the directive actually publishes.
    bound = set(re.findall(r'ng-(?:click|change)="(\w+)\(', app_html))
    bound |= set(re.findall(r"\{\{\s*(\w+)\(", app_html))
    published = set(re.findall(r"scope\.(\w+)\s*=", app_js))
    unbound = sorted(bound - published)
    assert not unbound, f"app.html binds scope functions that app.js never defines: {unbound}"

    # Engage still has to be reachable without the app (Alt+A action map).
    actions = json.loads((ROOT / "beamng_mod" / "lua" / "ge" / "extensions" / "core" / "input"
                          / "actions" / "gvd.json").read_text(encoding="utf-8"))
    assert "toggleEngage" in actions["gvd_toggle_engage"]["onDown"]
    keymap = json.loads((ROOT / "beamng_mod" / "settings" / "inputmaps"
                         / "keyboardGvd.json").read_text(encoding="utf-8"))
    assert any(b["control"] == "alt+a" for b in keymap["bindings"]), keymap

    # One prefs bus: what Lua writes is what the supervisor reads.
    written = set(re.findall(r'"(show_path|show_agent_ghosts|show_scene|policy|viz_screen)"\s*:', lua))
    assert {"show_path", "show_agent_ghosts", "show_scene", "policy", "viz_screen"} <= written, written
    for key in ("show_path", "show_agent_ghosts", "policy", "viz_screen"):
        assert f'"{key}"' in run_vision, f"run_vision.py ignores gvd_ui_prefs {key}"
    # Session gate: a pref from a past run must not override --policy at launch.
    assert "_ui_request_is_live" in run_vision

    print("test_gvd_ui_app: OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"test_gvd_ui_app: FAIL — {e}")
        sys.exit(1)
