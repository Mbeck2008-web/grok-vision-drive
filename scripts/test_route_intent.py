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


BANNED_IMPORT_NEEDLES = ("route_intent", "junction_grade", "navgraph", "navigraph", "beamngpy.tools")
NAV_ARG_KEYS = frozenset({"nav", "turn", "next_jct_m", "route_src"})
# Control / engage calls in run_vision that must never see those keys (args or kwargs).
CONTROL_CALL_NAMES = frozenset({
    "tick",
    "shadow_tick",
    "plan_command",
    "predict_path",
    "apply",
    "engage_gate",
})
CONTROL_CALL_OWNERS = {
    "tick": frozenset({"perc"}),
    "apply": frozenset({"actuator"}),
}
# Functions in run_vision that may mention nav/turn/next_jct_m/route_src in arguments.
# Display-only (#203): merge_nav_intent and apply_nav_missing.
# The only sanctioned addition is the SceneNet --route-cond command encoder
# ("runtime_route_vec"). Planner, actuation, and engage can never be listed here.
DISPLAY_NAV_ARG_ALLOWLIST = frozenset({"merge_nav_intent", "apply_nav_missing"})
MODEL_ROUTE_INPUT_ALLOWLIST = frozenset()
HARMLESS_NAV_READS = frozenset({"str", "len", "bool", "isinstance", "print", "repr", "type"})


def _iso_banned_names(names: list[str], path: Path, banned: tuple[str, ...]) -> None:
    for name in names:
        low = name.lower()
        assert not any(b in low for b in banned), f"{path.relative_to(ROOT)} imports {name}"


def _call_func_parts(func: ast.AST) -> tuple[str | None, str | None]:
    """Return (owner, name) for perc.tick -> ('perc', 'tick') or (None, 'shadow_tick')."""
    if isinstance(func, ast.Name):
        return None, func.id
    if isinstance(func, ast.Attribute):
        owner = func.value.id if isinstance(func.value, ast.Name) else None
        return owner, func.attr
    return None, None


def _dyn_import_module(node: ast.Call) -> str | None:
    owner, name = _call_func_parts(node.func)
    dotted = f"{owner}.{name}" if owner else name
    if name not in {"import_module", "__import__"} and dotted not in {"importlib.import_module"}:
        return None
    if not node.args:
        return None
    arg0 = node.args[0]
    if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
        return arg0.value
    return None


def _collect_imported_modules(path: Path, tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                pkg = path.parent
                for _ in range(node.level - 1):
                    pkg = pkg.parent
                try:
                    prefix = ".".join(pkg.relative_to(ROOT).parts)
                except ValueError:
                    prefix = ""
                if node.module:
                    names.append(f"{prefix}.{node.module}" if prefix else node.module)
                elif prefix:
                    names.append(prefix)
            if node.module:
                names.append(node.module)
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            dyn = _dyn_import_module(node)
            if dyn:
                names.append(dyn)
    return names


def _py_module_path(mod: str) -> Path | None:
    if not mod.startswith("python.") and mod != "python":
        return None
    base = ROOT.joinpath(*mod.split("."))
    file = base.with_suffix(".py")
    if file.is_file():
        return file
    init = base / "__init__.py"
    return init if init.is_file() else None


def _walk_python_imports(roots: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    stack = list(roots)
    while stack:
        path = stack.pop()
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name in _collect_imported_modules(path, tree):
            target = _py_module_path(name)
            if target is not None and target.resolve() not in seen:
                stack.append(target)
    return sorted(seen)


def _expr_mentions_nav(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in NAV_ARG_KEYS:
            return True
        if isinstance(child, ast.Attribute) and child.attr in NAV_ARG_KEYS:
            return True
        if isinstance(child, ast.Constant) and isinstance(child.value, str) and child.value in NAV_ARG_KEYS:
            return True
    return False


def _is_control_call(owner: str | None, name: str | None) -> bool:
    if name not in CONTROL_CALL_NAMES:
        return False
    allowed_owners = CONTROL_CALL_OWNERS.get(name)
    if allowed_owners is None:
        return True
    return owner in allowed_owners


def check_isolation() -> None:
    """Planner, actuation, and perception never import route intent — transitively.

    The walk follows ``python.*`` imports from the banned roots, including
    intermediate modules such as ``python.planning.speed``. ``importlib.import_module``
    and ``__import__`` string literals are treated as imports too.

    ``run_vision.py`` may merge display intent and, later, encode it as the
    SceneNet ``--route-cond`` command (see ``MODEL_ROUTE_INPUT_ALLOWLIST``).
    ``perc.tick``, ``shadow_tick``, ``plan_command``, ``predict_path``,
    ``actuator.apply``, and ``engage_gate`` must never receive ``nav`` / ``turn``
    / ``next_jct_m`` / ``route_src``.
    """
    banned = BANNED_IMPORT_NEEDLES
    roots = [
        ROOT / "python" / "control" / "actuate.py",
        ROOT / "python" / "planning" / "path_predictor.py",
        ROOT / "python" / "train" / "scene_lane_runtime.py",
    ]
    roots.extend(sorted((ROOT / "python" / "perception").glob("*.py")))
    for path in roots:
        assert path.is_file(), path
    visited = _walk_python_imports(roots)
    for path in visited:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _iso_banned_names(_collect_imported_modules(path, tree), path, banned)
    viz_files = sorted((ROOT / "python" / "viz").glob("*.py"))
    for path in viz_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _iso_banned_names(_collect_imported_modules(path, tree), path, banned)

    allow = DISPLAY_NAV_ARG_ALLOWLIST | MODEL_ROUTE_INPUT_ALLOWLIST | HARMLESS_NAV_READS
    assert MODEL_ROUTE_INPUT_ALLOWLIST.isdisjoint(CONTROL_CALL_NAMES)
    assert MODEL_ROUTE_INPUT_ALLOWLIST.isdisjoint(DISPLAY_NAV_ARG_ALLOWLIST)
    vision = ROOT / "python" / "run_vision.py"
    tree = ast.parse(vision.read_text(encoding="utf-8"), filename=str(vision))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        owner, name = _call_func_parts(node.func)
        if name is None:
            continue
        mentions = any(_expr_mentions_nav(arg) for arg in node.args)
        mentions = mentions or any(
            (kw.arg in NAV_ARG_KEYS) or (kw.value is not None and _expr_mentions_nav(kw.value))
            for kw in node.keywords
        )
        if not mentions:
            continue
        if name in allow:
            continue
        if _is_control_call(owner, name):
            label = f"{owner}.{name}" if owner else name
            raise AssertionError(f"run_vision {label} must not take nav/turn/next_jct_m/route_src")
        # Any other call that uses a nav= / turn= keyword is also a leak unless allowlisted.
        if any(kw.arg in NAV_ARG_KEYS for kw in node.keywords):
            label = f"{owner}.{name}" if owner else name
            raise AssertionError(f"run_vision {label} must not take nav/turn/next_jct_m/route_src")
    print(
        f"  isolation: {len(visited)} transitive control modules + viz; "
        f"run_vision control calls have no nav args "
        f"(model allowlist={sorted(MODEL_ROUTE_INPUT_ALLOWLIST) or 'empty'})"
    )


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


def check_last_metre_snap() -> None:
    """1 m and 0.5 m before T and Y keep the same turn as the 8 m pose.

    The old nearest+1.0 m / remain=0 tie-break flipped S->YL to straight,
    S->W from left to right, and dropped X->YL to none.
    """
    from python.sensors.route_intent import Graph, intent_from_pose, result_message

    data = load_graph()
    graph = Graph.from_fixture(data, "synth_a")
    for dist in (1.0, 0.5):
        right = intent_from_pose(graph, (0.0, -dist), goal="YL")
        assert right["turn"] == "right", (dist, right)
        assert right["next_jct_m"] is not None and 0.0 < float(right["next_jct_m"]) <= 2.0, (dist, right)
        left = intent_from_pose(graph, (0.0, -dist), goal="W")
        assert left["turn"] == "left", (dist, left)
        assert left["next_jct_m"] is not None and 0.0 < float(left["next_jct_m"]) <= 2.0, (dist, left)
    for east in (399.0, 399.5):
        toward = intent_from_pose(graph, (east, 0.0), goal="YL")
        assert toward["turn"] == "left", (east, toward)
        assert toward["next_jct_m"] is not None and 0.0 < float(toward["next_jct_m"]) <= 2.0, (east, toward)
    # Non-finite ego must not raise.
    for xy in (
        (float("nan"), 0.0),
        (0.0, float("nan")),
        (float("inf"), 0.0),
        (0.0, float("-inf")),
        (float("nan"), float("inf")),
    ):
        assert intent_from_pose(graph, xy, goal="YL") == result_message("none", None), xy
    print("  last metre: S->YL stays right, S->W stays left, X->YL stays left; NaN/inf -> none")


def main() -> None:
    check_isolation()
    check_result_schema()
    check_queue_put_ast()
    check_synthetic_turns()
    check_last_metre_snap()
    check_no_route_and_missing()
    check_reuse_on_empty_get()
    check_inbox_rejects_extra_keys()
    check_merge_is_display_only()
    check_nav_line_display()
    check_run_vision_imports_but_planner_does_not()
    print("test_route_intent: OK")


if __name__ == "__main__":
    main()
