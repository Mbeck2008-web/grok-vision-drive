#!/usr/bin/env python3
"""run_vision starts without importing torch.

``scene_net`` imports torch at module load (about 0.8 s on a CPU box). The
supervisor took that hit at startup through ``live_strip_fields`` and
``SCENE_SHAPE_HOLD``, which ate into the m6 harness's timed windows (the
harness now also starts its clock at the first command). Those helpers now
live in the torch-free ``scene_live``; ``scene_net`` re-exports them.

1. ``scene_live`` has no torch import (source check, works without torch).
2. ``scene_net`` re-exports the very same objects.
3. With torch installed: ``import python.run_vision`` leaves torch out of
   ``sys.modules``, and a headless stub supervisor runs its loop (writes
   ``gvd_state.json`` and ``gvd_cmd.json``) for a few seconds without
   importing torch at any point.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gvd_test_env import isolated_home  # noqa: E402

TORCH = importlib.util.find_spec("torch") is not None

# Runs run_vision as __main__ and drops a marker file the moment anything imports torch.
_HOOK = r"""
import os, runpy, sys
marker = os.environ["GVD_TEST_TORCH_MARKER"]
class _Spy:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            with open(marker, "a", encoding="utf-8") as fh:
                fh.write(name + "\n")
        return None
sys.meta_path.insert(0, _Spy())
sys.argv = [sys.argv[1]] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def check_scene_live_source() -> None:
    tree = ast.parse((ROOT / "python" / "train" / "scene_live.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        assert not any(n == "torch" or n.startswith(("torch.", "python.train.scene_net")) for n in names), names
    rv = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    top = [line for line in rv.splitlines() if line.startswith(("from python.train.scene_net", "from python.train.scene_lane_runtime"))]
    assert not top, top
    print("scene_live: no torch import; run_vision has no top-level scene_net / scene_lane_runtime import OK")


def check_reexports() -> None:
    from python.train import scene_live

    from python.train import scene_lane_runtime, scene_net

    assert scene_net.live_strip_fields is scene_live.live_strip_fields
    assert scene_net._BAD_REASONS is scene_live._BAD_REASONS
    assert scene_lane_runtime.SCENE_SHAPE_HOLD == scene_live.SCENE_SHAPE_HOLD == "scene_hold:input_shape"
    fields = scene_live.live_strip_fields(lanes=[1], override_reason="player_brake", brake=0.4)
    assert fields["pedals"] == {"throttle": 0.0, "brake": 0.4, "reason": "player_brake"}, fields
    assert fields["tech"] == {"lanes": [1]}
    print("re-exports: scene_net and scene_lane_runtime share scene_live's objects OK")


def check_import_without_torch() -> None:
    code = "import sys, python.run_vision; print('torch' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), env=dict(os.environ, PYTHONPATH=str(ROOT)),
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "False", out.stdout
    print(f"import python.run_vision: torch not loaded (torch installed={TORCH}) OK")


def check_supervisor_loop_without_torch() -> None:
    from python.control.actuate import cmd_path
    from python.runtime import paths as gvd_paths
    from python.runtime.state_io import state_path

    with isolated_home(prefix="gvd-startup-") as home:
        marker = home / "torch_imports.txt"
        pin = gvd_paths.predicted_gvd_docs_path()
        pin.mkdir(parents=True, exist_ok=True)

        def touch_bus() -> None:
            (pin / "gvd_link.json").write_text(json.dumps({"lua_bus": str(pin.resolve())}), encoding="utf-8")

        touch_bus()
        env = dict(os.environ, PYTHONPATH=str(ROOT), GVD_TEST_TORCH_MARKER=str(marker))
        args = [sys.executable, "-u", "-c", _HOOK, str(ROOT / "python" / "run_vision.py"),
                "--backend", "stub", "--hz", "20", "--encode", "cpu"]
        t0 = time.time()
        proc = subprocess.Popen(args, cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        first_cmd = None
        try:
            end = t0 + 30.0
            while time.time() < end:
                assert proc.poll() is None, f"run_vision exited ({proc.returncode})"
                touch_bus()
                if first_cmd is None and cmd_path().exists():
                    first_cmd = time.time()
                    end = first_cmd + 2.0  # two seconds of loop after startup
                time.sleep(0.02)
        finally:
            proc.kill()
            proc.wait(timeout=10)
        assert first_cmd is not None, "run_vision wrote no gvd_cmd.json within 30 s"
        assert state_path().exists(), "the loop never wrote gvd_state.json"
        seq = json.loads(cmd_path().read_text(encoding="utf-8")).get("seq")
        loaded = marker.read_text(encoding="utf-8").split() if marker.exists() else []
        assert not loaded, f"run_vision imported torch: {loaded[:5]}"
        print(f"stub supervisor: first command {first_cmd - t0:.2f}s after spawn, seq {seq} after 2 s, no torch import OK")


def main() -> None:
    check_scene_live_source()
    check_reexports()
    check_import_without_torch()
    check_supervisor_loop_without_torch()
    print("test_run_vision_startup: OK")


if __name__ == "__main__":
    main()
