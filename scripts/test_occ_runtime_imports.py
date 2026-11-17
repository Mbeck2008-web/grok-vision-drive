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
    "python.perception.occ_outputs",
    "python.perception.occ_ego",
    "python.viz.occ_viz",
)
# The occupancy runtime proper (net -> ego motion -> tracker -> viz). Besides the
# GT label modules it must not reach any sim-truth pose source.
OCC_RUNTIME = (
    "python.perception.occ_sparse", "python.perception.occ_geometry", "python.perception.occ_outputs",
    "python.perception.occ_ego", "python.perception.occ_out", "python.viz.occ_viz",
)
# Modules whose job is reading sim truth (pose, road graph, decals, signals).
# Config readers the live supervisor already loads (raw_frames, sensors.cameras)
# are not on this list; the spy below proves no pose is read through them.
GT_POSE_MODULES = (
    "python.debug.gt_geom", "python.debug.painted_lanes", "python.collect.signal_labels", "python.collect.route_hindsight",
    "python.collect.static_signs", "python.collect.road_dump", "python.collect.recorder_v4", "python.collect.road_loader",
)
# Identifiers that read sim-truth pose / labels; none may appear in the occupancy runtime source.
GT_POSE_NAMES = frozenset({"read_state_jsonl", "scenetree", "get_scenetree", "get_road_edges", "world_to_ego",
                           "vehicles_in_ego", "beamngpy", "state_jsonl", "gt_pose"})
GT_POSE_THIRD_PARTY = ("beamngpy",)
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


def third_party_of(name: str) -> set[str]:
    path = module_path(name)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            out.add(node.module.split(".")[0])
    return out


def names_in(name: str) -> set[str]:
    tree = ast.parse(module_path(name).read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.alias):
            out.add(node.name.split(".")[-1]); out.add(node.name.split(".")[0])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in GT_POSE_NAMES:
            out.add(node.value)
    return out


def check_gt_pose_trace() -> None:
    seen = reachable(OCC_RUNTIME)
    bad = [chain(seen, m) for m in seen if any(m == g or m.startswith(g + ".") for g in GT_POSE_MODULES)]
    assert not bad, bad
    # The occupancy modules themselves never import beamngpy. (The rig geometry is read
    # through raw_frames.camera_sidecar -> sensors.cameras, the live camera client,
    # which references beamngpy lazily for frames; the runtime check below proves
    # importing the occupancy runtime does not load it.)
    third = sorted({t for m in OCC_RUNTIME for t in third_party_of(m)} & set(GT_POSE_THIRD_PARTY))
    assert not third, third
    for m in OCC_RUNTIME:
        used = names_in(m) & GT_POSE_NAMES
        assert not used, (m, sorted(used))
    assert names_in("python.train.occ_labels") & GT_POSE_NAMES, "name scan must see occ_labels' world_to_ego"
    assert names_in("python.data.occ_shards") & GT_POSE_NAMES, "name scan must see the shard builder's state reader"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    code = CHILD.format(root=str(ROOT), entries=OCC_RUNTIME)
    res = subprocess.run([sys.executable, "-c", code + "\nprint(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in {third!r})))".format(third=GT_POSE_THIRD_PARTY)],
                         cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=240)
    assert res.returncode == 0, res.stderr[-2000:]
    lines = res.stdout.strip().splitlines()
    loaded, sim = set(json.loads(lines[-2])), json.loads(lines[-1])
    leaked = sorted(m for m in loaded if any(m == g or m.startswith(g + ".") for g in GT_POSE_MODULES))
    assert not leaked and not sim, (leaked, sim)
    print(f"GT pose: {len(seen)} modules reachable from the occupancy runtime, none is a pose/sim-truth source; "
          f"no pose-reading name in its source; beamngpy not imported OK")


class _Spy(dict):
    """Vehicle/state mapping that records every key read and raises on sim-truth keys."""

    FORBIDDEN = ("pos", "dir", "up", "rotation", "rot", "vel", "velocity", "yaw", "heading", "ego", "pose", "scenetree")

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.read: list[str] = []

    def _touch(self, k):
        self.read.append(k)
        if k in self.FORBIDDEN:
            raise AssertionError(f"runtime read sim-truth key {k!r}")

    def __getitem__(self, k):
        self._touch(k)
        return super().__getitem__(k)

    def get(self, k, d=None):
        self._touch(k)
        return super().get(k, d)

    def __contains__(self, k):
        self._touch(k)
        return super().__contains__(k)

    def keys(self):
        raise AssertionError("runtime enumerated the vehicle mapping")

    def items(self):
        raise AssertionError("runtime enumerated the vehicle mapping")

    def __iter__(self):
        raise AssertionError("runtime enumerated the vehicle mapping")


class _NoImage:
    """Stands in for every camera array downstream of the heads: any access raises."""

    def __getattr__(self, name):
        raise AssertionError(f"downstream code touched a camera array (.{name})")

    def __getitem__(self, k):
        raise AssertionError("downstream code indexed a camera array")

    def __array__(self, *a, **k):
        raise AssertionError("downstream code converted a camera array")

    def __iter__(self):
        raise AssertionError("downstream code iterated a camera array")

    def __len__(self):
        raise AssertionError("downstream code sized a camera array")


def planner_stub(msg) -> float | None:
    """Toy planner on the viz message only: nearest track ahead within 2 m of the lane."""
    ahead = [t["y"] for t in msg["tracks"] if t["y"] > 0 and abs(t["x"]) < 2.0]
    return min(ahead) if ahead else None


def check_flag_and_spy() -> None:
    import argparse

    import numpy as np

    sys.path.insert(0, str(ROOT))
    from python.perception.occ_ego import VEHICLE_SIGNAL_KEYS, EgoMotion, add_ego_args, runtime_frame
    from python.perception.occ_geometry import DEFAULT_GRID as G
    from python.perception.occ_outputs import OccTracker
    from python.viz.occ_viz import encode, occ_to_viz

    a = add_ego_args(argparse.ArgumentParser()).parse_args([])
    assert a.ego_from_vehicle is False
    assert EgoMotion.from_args(a).source == "camera_vo"
    on = add_ego_args(argparse.ArgumentParser()).parse_args(["--ego-from-vehicle"])
    assert EgoMotion.from_args(on).source == "vehicle_signals"

    class Pred:  # what the net hands downstream: cells only; image fields are guards
        def __init__(self, k, c):
            self.keys, self.cls = k, c
            self.images = _NoImage(); self.frames = _NoImage()

    car = np.array([[x, 12.0 + y, z] for x in (-0.6, -0.2, 0.2, 0.6) for y in (0.0, 0.4, 0.8, 1.2, 1.6) for z in (0.2, 0.6)])
    keys = G.keys_of(car); cls = np.full(keys.shape, 2, np.uint8)
    truth = {"pos": [1, 2, 3], "dir": [0, 1, 0], "rotation": [0, 0, 0, 1], "vel": [0, 9, 0], "wheelspeed_mps": 9.0,
             "imu_yaw_rate": 0.01, "scenetree": {}}
    for flag in (False, True):
        ego, tr = EgoMotion(from_vehicle=flag), OccTracker()
        for i in range(4):
            spy = _Spy(truth)
            out = runtime_frame(Pred(keys, cls), tr, ego, 0.1 * i, vehicle=spy)
            out["images"] = _NoImage()  # unknown keys in the outputs are never read either
            msg = occ_to_viz(out, seq=i)
            encode(msg)
            planner_stub(msg)
            if flag:
                assert set(spy.read) <= set(VEHICLE_SIGNAL_KEYS) and spy.read, spy.read
            else:
                assert spy.read == [], f"default (camera VO) read vehicle keys {spy.read}"
        assert out["ego"]["source"] == ("vehicle_signals" if flag else "camera_vo")
        if flag:
            assert out["ego"]["speed_mps"] == 9.0
    assert planner_stub(msg) is not None
    print("ego flag: --ego-from-vehicle defaults OFF (camera VO reads no vehicle key); ON reads only "
          f"{list(VEHICLE_SIGNAL_KEYS)}; outputs/viz/planner stub never touch a camera array OK")


def main() -> int:
    check_static()
    check_runtime()
    check_gt_pose_trace()
    check_flag_and_spy()
    print("test_occ_runtime_imports: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
