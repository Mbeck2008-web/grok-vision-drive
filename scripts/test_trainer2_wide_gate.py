#!/usr/bin/env python3
"""Trainer 2.0 PR 2: wide-gate parity + safety-path import isolation.

docs/specs/trainer_2.md, "Wide near-range gating" and Non-goals:

* check_wide_gate_same_as_live: Trainer 2.0 wide-derived lane points come from
  the live gate (``lanes._estimate_span_lanes(..., gate_wide=True)`` reached
  through ``lanes_from_view``). ``WIDE_NEAR_MAX_Y_M`` / ``WIDE_NEAR_MAX_X_M``
  are imported from ``python.perception.lanes``, never redefined, and no
  ``trainer2_*`` module hard-codes 9.0 / 5.0 as a gate. On the recorded
  fixture strips the trainer's wide points equal ``gate_wide=True`` and
  differ from ``gate_wide=False``.
* check_safety_path_no_trainer_imports: heartbeat / AEB / engage modules
  never reach ``python.train.*`` (AST walk, lazy imports included, plus a
  fresh-interpreter import), and no ``trainer2_*`` module is reachable from
  the live supervisor.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_occ_runtime_imports import chain, reachable  # noqa: E402

GATE_NAMES = ("WIDE_NEAR_MAX_Y_M", "WIDE_NEAR_MAX_X_M")
GATE_LITERALS = (9.0, 5.0)
# docs/specs/trainer_2.md Non-goals: the safety path (beamng_mod gvd/main.lua is Lua, not walked here),
# plus every python/control module (Appendix A: python/control/* is not touched) except
# CONTROL_NOT_SAFETY. gt_lane_loop is the GT-steer lane-model debug loop: it imports
# python.train.scene_lane_runtime on main 8ce34d5 (the live lane model, not a trainer).
CONTROL_NOT_SAFETY = ("gt_lane_loop",)
SAFETY_MODULES = tuple(sorted(
    {
        "python.control.actuate",
        "python.control.ego_speed",
        "python.control.echo_link",
        "python.sensors.live_hold",
        "python.perception.cipv",
    }
    | {f"python.control.{p.stem}" for p in (ROOT / "python" / "control").glob("*.py")
       if p.stem != "__init__" and p.stem not in CONTROL_NOT_SAFETY}
))
LIVE_ENTRY = "python.run_vision"
# Same sector crop and spans as scripts/test_wide_near_range.py (96 x 1336 fixture strips).
STITCH_CROP = (336, 840)
STITCH_SPANS = ((0, 160), (168, 328), (336, 496))

CHILD = """
import json, sys
sys.path.insert(0, {root!r})
import importlib
for name in {entries!r}:
    importlib.import_module(name)
print(json.dumps(sorted(m for m in sys.modules if m.startswith("python."))))
"""


def trainer2_modules() -> list[Path]:
    return sorted((ROOT / "python" / "train").glob("trainer2_*.py"))


def gate_violations(source: str) -> list[str]:
    """Where ``source`` redefines a wide-gate constant or hard-codes 9.0 / 5.0 as a bound."""
    tree = ast.parse(source)
    bad: list[str] = []

    def is_gate_literal(node: ast.AST) -> bool:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            node = node.operand
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
            and float(node.value) in GATE_LITERALS
        )

    for node in ast.walk(tree):
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for tgt in targets:
            for sub in ast.walk(tgt):
                name = sub.id if isinstance(sub, ast.Name) else sub.attr if isinstance(sub, ast.Attribute) else None
                if name in GATE_NAMES:
                    bad.append(f"l.{node.lineno}: assigns {name}")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in GATE_NAMES:
            bad.append(f"l.{node.lineno}: defines {node.name}")
        if isinstance(node, ast.Compare):
            for operand in [node.left, *node.comparators]:
                if is_gate_literal(operand):
                    bad.append(f"l.{node.lineno}: compares against {ast.unparse(operand)}")
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            if name in {"min", "max", "clip", "clamp", "minimum", "maximum"}:
                for arg in [*node.args, *(k.value for k in node.keywords)]:
                    if is_gate_literal(arg):
                        bad.append(f"l.{node.lineno}: {name}(..., {ast.unparse(arg)})")
    return bad


def check_gate_checker_catches_mutants() -> None:
    clean = "from python.perception.lanes import WIDE_NEAR_MAX_Y_M\nok = [p for p in pts if p['y'] <= WIDE_NEAR_MAX_Y_M]\n"
    assert gate_violations(clean) == [], gate_violations(clean)
    mutants = {
        "redefine": "WIDE_NEAR_MAX_Y_M = 9.0\n",
        "redefine_ann": "WIDE_NEAR_MAX_X_M: float = 5.0\n",
        "attr": "import python.perception.lanes as L\nL.WIDE_NEAR_MAX_Y_M = 12.0\n",
        "literal_y": "ok = [p for p in pts if p['y'] <= 9.0]\n",
        "literal_x": "ok = [p for p in pts if abs(p['x']) <= 5]\n",
        "literal_neg": "ok = [p for p in pts if p['x'] >= -5.0]\n",
        "min_clip": "y = min(y, 9.0)\n",
        "np_clip": "x = np.clip(x, -5.0, 5.0)\n",
    }
    for name, src in mutants.items():
        assert gate_violations(src), f"mutant {name} not caught"
    print(f"gate checker: clean source passes, {len(mutants)}/{len(mutants)} mutants caught OK")


def check_no_trainer2_gate_redefinition() -> None:
    mods = trainer2_modules()
    assert any(p.name == "trainer2_wide.py" for p in mods), [p.name for p in mods]
    for path in mods:
        bad = gate_violations(path.read_text(encoding="utf-8"))
        assert not bad, f"{path.name}: {bad}"
    from python.perception import lanes
    from python.train import trainer2_wide

    for name in GATE_NAMES:
        assert getattr(trainer2_wide, name) is getattr(lanes, name), name
    tree = ast.parse((ROOT / "python" / "train" / "trainer2_wide.py").read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "python.perception.lanes"
        for alias in node.names
    }
    assert set(GATE_NAMES) <= imported, imported
    assert trainer2_wide.wide_gate() == (lanes.WIDE_NEAR_MAX_Y_M, lanes.WIDE_NEAR_MAX_X_M)
    print(f"no gate redefinition in {len(mods)} trainer2_* module(s); trainer2_wide imports {GATE_NAMES} OK")


def _wide_points(lanes_bev) -> list[tuple[float, float]]:
    return [
        (round(float(pt["x"]), 6), round(float(pt["y"]), 6))
        for poly in lanes_bev
        for pt in poly
        if pt.get("src") == "wide"
    ]


def check_wide_points_match_live_gate() -> None:
    from python.perception import lanes
    from python.train import trainer2_wide

    n_frames = n_points = n_differ = 0
    for p in sorted((ROOT / "tests" / "fixtures").rglob("*.jpg")):
        img = cv2.imread(str(p))
        if img is None or img.shape[:2] != (96, 1336) or "esbr" in p.name:
            continue
        img = img.copy()
        img[:28, :420] = (30, 30, 32)
        crop = np.ascontiguousarray(img[:, STITCH_CROP[0] : STITCH_CROP[1]])
        got = _wide_points(trainer2_wide.wide_lane_points(img))
        gated = _wide_points(lanes._estimate_span_lanes(crop, lanes.STITCH_LANE_FAR_M, STITCH_SPANS, gate_wide=True).lanes_bev)
        ungated = _wide_points(lanes._estimate_span_lanes(crop, lanes.STITCH_LANE_FAR_M, STITCH_SPANS, gate_wide=False).lanes_bev)
        assert got == gated, (p.name, got[:4], gated[:4])
        for x, y in got:
            assert y <= lanes.WIDE_NEAR_MAX_Y_M and abs(x) <= lanes.WIDE_NEAR_MAX_X_M, (p.name, x, y)
        n_frames += 1
        n_points += len(got)
        n_differ += int(got != ungated)
    assert n_frames > 0, "no 96x1336 fixture strips found"
    assert n_points > 0, "fixtures gave no wide points; parity is untested"
    assert n_differ > 0, "gate_wide=True never differs from gate_wide=False on the fixtures; the gate is untested"
    print(f"wide parity: {n_frames} fixture strips, {n_points} wide points == gate_wide=True, {n_differ} differ from gate_wide=False OK")


def check_safety_path_static() -> None:
    seen = reachable(SAFETY_MODULES)
    bad = [chain(seen, m) for m in seen if m == "python.train" or m.startswith("python.train.")]
    assert not bad, bad
    probe = reachable(("python.train.trainer2_wide",))
    assert "python.perception.lanes" in probe, sorted(probe)
    live = reachable((LIVE_ENTRY,))
    leaked = [chain(live, m) for m in live if m.startswith("python.train.trainer2_")]
    assert not leaked, leaked
    print(f"static: {len(SAFETY_MODULES)} safety modules reach {len(seen)} modules, none in python.train; "
          f"no trainer2_* reachable from {LIVE_ENTRY} OK")


def check_safety_path_runtime() -> None:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    code = CHILD.format(root=str(ROOT), entries=SAFETY_MODULES)
    res = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=240)
    assert res.returncode == 0, res.stderr[-2000:]
    loaded = set(json.loads(res.stdout.strip().splitlines()[-1]))
    assert set(SAFETY_MODULES) <= loaded, set(SAFETY_MODULES) - loaded
    leaked = sorted(m for m in loaded if m == "python.train" or m.startswith("python.train."))
    assert not leaked, f"importing the safety path loaded {leaked}"
    print(f"runtime: importing {len(SAFETY_MODULES)} safety modules loads {len(loaded)} python.* modules, none in python.train OK")


def main() -> int:
    checks = (
        check_gate_checker_catches_mutants,
        check_no_trainer2_gate_redefinition,
        check_wide_points_match_live_gate,
        check_safety_path_static,
        check_safety_path_runtime,
    )
    failed = 0
    for check in checks:
        try:
            check()
        except Exception as exc:  # noqa: BLE001 - report every check
            failed += 1
            print(f"FAIL {check.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
