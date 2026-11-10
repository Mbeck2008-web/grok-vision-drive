#!/usr/bin/env python3
"""GPS route intent (SPEC M1): {turn, next_jct_m} for display only.

Offline. No BeamNG, no GPU, no planner, no visualizer geometry.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from synthetic_roads import load_graph, track  # noqa: E402


def _iso_banned(path: Path, banned: tuple[str, ...]) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [alias.name for alias in node.names]
        for name in names:
            low = name.lower()
            assert not any(b in low for b in banned), f"{path.relative_to(ROOT)} imports {name}"


def check_isolation() -> None:
    """Planner, actuation, and perception never import route intent or navgraph."""
    banned = ("route_intent", "junction_grade", "navgraph", "navigraph", "beamngpy.tools")
    targets = [
        ROOT / "python" / "control" / "actuate.py",
        ROOT / "python" / "planning" / "path_predictor.py",
        ROOT / "python" / "train" / "scene_lane_runtime.py",
    ]
    targets.extend(sorted((ROOT / "python" / "perception").glob("*.py")))
    for path in targets:
        assert path.is_file(), path
        _iso_banned(path, banned)
    viz_banned = ("route_intent", "junction_grade", "navgraph", "navigraph", "beamngpy.tools")
    for path in (ROOT / "python" / "viz").glob("*.py"):
        _iso_banned(path, viz_banned)
    print(f"  isolation: {len(targets)} control files + viz import no route intent or navgraph")


def check_result_schema() -> None:
    from python.sensors.route_intent import RESULT_KEYS, TURNS, result_message

    msg = result_message("left", 12.5)
    assert set(msg) == RESULT_KEYS == {"turn", "next_jct_m"}
    assert msg["turn"] == "left" and msg["next_jct_m"] == 12.5
    none = result_message("none", None)
    assert none == {"turn": "none", "next_jct_m": None}
    for turn in TURNS:
        assert result_message(turn, None)["turn"] == turn
    print("  result schema is exactly {turn, next_jct_m}")


def check_queue_put_ast() -> None:
    """Every .put( in the module sends only the result schema (or a name bound to it)."""
    from python.sensors import route_intent as ri

    path = Path(ri.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    puts = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_put = (isinstance(func, ast.Attribute) and func.attr == "put") or (
            isinstance(func, ast.Name) and func.id == "put"
        )
        if not is_put:
            continue
        puts += 1
        assert node.args, "route_intent put() with no payload"
        arg = node.args[0]
        assert isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "result_message", (
            ast.dump(arg)
        )
    assert puts >= 1, "expected at least one result_message put"
    print(f"  AST: {puts} put() call(s) send result_message(...)")


def _intent_along(path_names: list[str], goal: str, *, at_m: float):
    from python.sensors.route_intent import Graph, intent_from_pose

    data = load_graph()
    graph = Graph.from_fixture(data, "synth_a")
    poses = track(data, "synth_a", path_names, speed_mps=10.0, dt_s=0.1)
    # poses are (x, y, heading_rad); pick the first whose travelled distance >= at_m
    travelled = 0.0
    chosen = poses[0]
    prev = poses[0]
    for p in poses[1:]:
        travelled += ((p[0] - prev[0]) ** 2 + (p[1] - prev[1]) ** 2) ** 0.5
        prev = p
        chosen = p
        if travelled >= at_m:
            break
    return intent_from_pose(graph, (chosen[0], chosen[1]), goal=goal)


def check_synthetic_turns() -> None:
    """S -> YL: right at T, straight at X, left at Y. next_jct_m falls toward each box."""
    from python.sensors.route_intent import Graph, intent_from_pose

    data = load_graph()
    graph = Graph.from_fixture(data, "synth_a")
    far = intent_from_pose(graph, (0.0, -150.0), goal="YL")
    assert far["turn"] == "right", far
    assert far["next_jct_m"] is not None and 140.0 <= far["next_jct_m"] <= 160.0, far
    near = intent_from_pose(graph, (0.0, -8.0), goal="YL")
    assert near["turn"] == "right", near
    assert near["next_jct_m"] is not None and near["next_jct_m"] <= 12.0, near
    assert near["next_jct_m"] < far["next_jct_m"]

    mid = intent_from_pose(graph, (80.0, 0.0), goal="YL")
    assert mid["turn"] == "straight", mid
    assert mid["next_jct_m"] is not None and 110.0 <= mid["next_jct_m"] <= 130.0, mid

    after_x = intent_from_pose(graph, (320.0, 0.0), goal="YL")
    assert after_x["turn"] == "left", after_x
    assert after_x["next_jct_m"] is not None and 70.0 <= after_x["next_jct_m"] <= 90.0, after_x
    print("  S->YL: right at T, straight at X, left at Y; next_jct_m falls toward T")


def check_no_route_and_missing() -> None:
    from python.sensors.route_intent import Graph, intent_from_pose, merge_nav_intent, result_message

    data = load_graph()
    graph = Graph.from_fixture(data, "synth_a")
    stuck = intent_from_pose(graph, (0.0, -150.0), goal="U1")
    assert stuck == result_message("none", None), stuck
    missing = merge_nav_intent({"mode": "missing", "pin": None}, None)
    assert missing["turn"] == "none" and missing["next_jct_m"] is None
    assert missing["route_src"] == "none" and missing["mode"] == "missing"
    print("  unreachable goal and missing GPS -> turn=none, no crash")


def check_reuse_on_empty_get() -> None:
    from python.sensors.route_intent import RouteInbox, result_message

    box = RouteInbox()
    first = box.observe(result_message("left", 40.0))
    assert first == {"turn": "left", "next_jct_m": 40.0} and box.route_age_ticks == 0
    reused = box.observe(None)
    assert reused == first and box.route_age_ticks == 1
    reused2 = box.observe(None)
    assert reused2 == first and box.route_age_ticks == 2
    fresh = box.observe(result_message("straight", 9.0))
    assert fresh == {"turn": "straight", "next_jct_m": 9.0} and box.route_age_ticks == 0
    print("  empty get reuses last result and counts route_age_ticks")


def check_inbox_rejects_extra_keys() -> None:
    from python.sensors.route_intent import RouteInbox

    box = RouteInbox()
    try:
        box.observe({"turn": "left", "next_jct_m": 1.0, "nodes": ["T"]})
    except ValueError as exc:
        assert "nodes" in str(exc) or "schema" in str(exc).lower()
    else:
        raise AssertionError("geometry on the result must be refused")
    print("  extra keys (geometry) refused")


def check_merge_is_display_only() -> None:
    from python.sensors.route_intent import merge_nav_intent, result_message

    nav = {
        "mode": "hint",
        "drive_to_pin": False,
        "gps": {"lat": 1.0, "lon": 2.0, "ok": True},
        "pin": {"lat": 1.1, "lon": 2.1, "name": "gate"},
    }
    out = merge_nav_intent(nav, result_message("right", 33.0), route_age_ticks=2)
    assert out["drive_to_pin"] is False
    assert out["turn"] == "right" and out["next_jct_m"] == 33.0
    assert out["route_src"] == "gps" and out["route_age_ticks"] == 2
    assert "nodes" not in out and "path" not in out
    print("  merge keeps drive_to_pin false and adds no geometry")


def check_nav_line_display() -> None:
    from python.sensors.route_intent import intent_caption

    assert intent_caption({"turn": "none", "next_jct_m": None}) == ""
    cap = intent_caption({"turn": "left", "next_jct_m": 18.4})
    assert cap == "intent left 18m", cap

    try:
        from python.viz.nerd import _nav_line
    except ImportError as exc:
        if "cv2" in str(exc):
            print("  intent caption OK; nerd/_nav_line skipped (no cv2 in this env)")
            return
        raise

    bare = _nav_line(
        {
            "capture_backend": "beamngpy",
            "nav": {
                "mode": "hint",
                "gps": {"lat": 53.09, "lon": 8.81, "ok": True},
                "pin": {"lat": 53.10, "lon": 8.82, "name": "west gate"},
                "range_m": 1234.0,
                "bearing_rel_deg": 45.0,
            },
        }
    )
    assert "nav hint" in bare and "west gate" in bare and "not routing" in bare
    shown = _nav_line(
        {
            "capture_backend": "beamngpy",
            "nav": {
                "mode": "hint",
                "gps": {"lat": 53.09, "lon": 8.81, "ok": True},
                "pin": {"name": "west gate"},
                "range_m": 100.0,
                "turn": "left",
                "next_jct_m": 18.4,
            },
        }
    )
    assert "intent left" in shown and "18m" in shown
    assert "not routing" not in shown
    print("  LIVE line shows intent left 18m and keeps the old pin wording without it")


def check_run_vision_imports_but_planner_does_not() -> None:
    text = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    assert "merge_nav_intent" in text
    assert "from python.sensors.route_intent import" in text
    print("  run_vision merges display intent")


def main() -> None:
    check_isolation()
    check_result_schema()
    check_queue_put_ast()
    check_synthetic_turns()
    check_no_route_and_missing()
    check_reuse_on_empty_get()
    check_inbox_rejects_extra_keys()
    check_merge_is_display_only()
    check_nav_line_display()
    check_run_vision_imports_but_planner_does_not()
    print("test_route_intent: OK")


if __name__ == "__main__":
    main()
