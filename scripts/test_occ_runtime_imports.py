#!/usr/bin/env python3
"""Ground truth stays out of the live path.

The training-only modules that read depth, semantics, or vehicle boxes
(``OCC_GT_MODULES``) must not load when a live entry point is imported, and
must not be reachable from one through any ``import`` statement (module
level or inside a function) in the ``python`` package.

* runtime: a fresh interpreter imports every live entry point and the
  runtime occupancy net; none of the GT modules (nor ``sensor_aux``) is in
  ``sys.modules``.
* static: the import graph from those entry points (AST walk, lazy imports
  included) never reaches a GT module.
* the walk itself is checked: it does reach ``occ_labels`` from ``occ_train``.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE_ENTRY_POINTS = (
    "python.run_vision",
    "python.train.scene_live",
    "python.train.scene_ckpt_runtime",
    "python.runtime.models",
    "python.perception.occ_sparse",
    "python.perception.occ_geometry",
)
OCC_GT_MODULES = (
    "python.train.occ_labels",
    "python.train.occ_train",
    "python.train.occ_sampler",
    "python.data.occ_shards",
)
# Older training-only depth/sem reader. ``scene_net`` imports it inside a
# training function (static path exists, pre-existing), so it is held to the
# runtime check only: importing the live path must not load it.
RUNTIME_ONLY_GT = ("python.train.sensor_aux",)

CHILD = """
import json, sys
sys.path.insert(0, {root!r})
import importlib
for name in {entries!r}:
    importlib.import_module(name)
print(json.dumps(sorted(m for m in sys.modules if m.startswith("python."))))
"""


def module_path(name: str) -> Path | None:
    base = ROOT.joinpath(*name.split("."))
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def imports_of(name: str) -> set[str]:
    path = module_path(name)
    if path is None:
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pkg = name if path.name == "__init__.py" else name.rpartition(".")[0]
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = pkg.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                mod = f"{base}.{node.module}" if node.module else base
            else:
                mod = node.module or ""
            out.add(mod)
            out.update(f"{mod}.{a.name}" for a in node.names)
    found = set()
    for m in out:
        if not m.startswith("python"):
            continue
        parts = m.split(".")
        for i in range(1, len(parts) + 1):
            sub = ".".join(parts[:i])
            if module_path(sub) is not None:
                found.add(sub)
    return found


def reachable(entries) -> dict[str, str]:
    """Every python.* module reachable from ``entries`` -> the module that imports it."""
    seen: dict[str, str] = {e: "<entry>" for e in entries}
    todo = list(entries)
    while todo:
        cur = todo.pop()
        for nxt in imports_of(cur):
            if nxt not in seen:
                seen[nxt] = cur
                todo.append(nxt)
    return seen


def chain(seen: dict[str, str], mod: str) -> str:
    out = [mod]
    while seen.get(out[-1], "<entry>") != "<entry>":
        out.append(seen[out[-1]])
    return " <- ".join(out)


def check_runtime() -> None:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    code = CHILD.format(root=str(ROOT), entries=LIVE_ENTRY_POINTS)
    res = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=240)
    assert res.returncode == 0, res.stderr[-2000:]
    loaded = set(json.loads(res.stdout.strip().splitlines()[-1]))
    assert set(LIVE_ENTRY_POINTS) <= loaded, set(LIVE_ENTRY_POINTS) - loaded
    leaked = sorted(loaded & (set(OCC_GT_MODULES) | set(RUNTIME_ONLY_GT)))
    assert not leaked, f"live import loaded GT modules: {leaked}"
    print(f"runtime: {len(loaded)} python.* modules after importing the live entry points, no GT module OK")


def check_static() -> None:
    seen = reachable(LIVE_ENTRY_POINTS)
    bad = [chain(seen, m) for m in OCC_GT_MODULES if m in seen]
    assert not bad, bad
    probe = reachable(("python.train.occ_train",))
    assert "python.train.occ_labels" in probe and "python.data.occ_shards" in probe, sorted(probe)
    print(f"static: {len(seen)} modules reachable from the live entry points, no GT module; walk finds occ_labels from occ_train OK")


def main() -> int:
    check_static()
    check_runtime()
    print("test_occ_runtime_imports: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
