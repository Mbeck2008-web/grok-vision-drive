#!/usr/bin/env python3
"""C++ visualizer build + render tests (cpp/viz, docs/specs/viz_cpp.md).

* configure + build gvd_viz with CMake (or reuse GVD_VIZ_BUILD_DIR), then run
  its ctest pixel tests (white/yellow paint, no purple, red only edges, lead
  green, GT keys ignored, schema version enforced);
* end to end: the Python publisher's scene file -> ``gvd_viz --offline`` PNG ->
  pixels checked here, so the two sides of schema v1 agree.

A host with no CMake or no C++ compiler skips visibly (skip_check). CI
installs both, so CI always runs it. A build or test failure is a failure.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from gvd_test_env import skip_check  # noqa: E402

SRC = ROOT / "cpp" / "viz"
EXE_NAME = "gvd_viz.exe" if sys.platform == "win32" else "gvd_viz"


def _compiler() -> str | None:
    for env in ("CXX",):
        if os.environ.get(env):
            return os.environ[env]
    for name in ("c++", "g++", "clang++", "cl"):
        if shutil.which(name):
            return name
    return None


def _find_exe(build: Path) -> Path | None:
    for sub in ("", "Release", "Debug"):
        cand = build / sub / EXE_NAME
        if cand.is_file():
            return cand
    return None


def build_dir() -> Path | None:
    pre = os.environ.get("GVD_VIZ_BUILD_DIR")
    if pre:
        b = Path(pre)
        if _find_exe(b) is None:
            raise AssertionError(f"GVD_VIZ_BUILD_DIR={pre} has no {EXE_NAME}")
        return b
    cmake = shutil.which("cmake")
    if cmake is None or _compiler() is None:
        skip_check("cpp_build", f"no cmake ({cmake}) or C++ compiler ({_compiler()}) on this host")
        return None
    b = SRC / "build" / "suite"
    cfg = [cmake, "-S", str(SRC), "-B", str(b), "-DCMAKE_BUILD_TYPE=Release"]
    if shutil.which("ninja"):
        cfg += ["-G", "Ninja"]
    if not (b / "CMakeCache.txt").is_file():
        subprocess.run(cfg, check=True)
    subprocess.run([cmake, "--build", str(b), "--config", "Release", "--parallel"], check=True)
    return b


def check_ctest(b: Path) -> None:
    ctest = shutil.which("ctest") or str(Path(shutil.which("cmake") or "ctest").with_name("ctest"))
    r = subprocess.run([ctest, "--test-dir", str(b), "-C", "Release", "--output-on-failure"],
                       capture_output=True, text=True)
    print(r.stdout[-3000:])
    assert r.returncode == 0, r.stderr[-2000:]
    print("ctest: gvd_viz render tests OK")


def check_end_to_end(exe: Path) -> None:
    import cv2
    import numpy as np

    from python.viz.cpp_publisher import write_scene_file

    lanes = [
        {"points": [{"x": -1.8, "y": float(y)} for y in range(-10, 80, 2)], "kind": "detected", "side": "left", "paint": "yellow"},
        {"points": [{"x": 1.8, "y": float(y)} for y in range(-10, 80, 2)], "kind": "detected", "side": "right", "index": 1},
    ]
    state = {
        "engaged": True, "ego": {"speed_mps": 10.0}, "planner": {"cipv_id": 4},
        "lanes_ext": lanes,
        "road_edges": [{"points": [{"x": 5.5, "y": -10.0}, {"x": 5.5, "y": 80.0}], "kind": "predicted"}],
        "tracks": [{"id": 4, "class": "vehicle", "x": 0.0, "y": 15.0, "yaw": 1.5708}],
        # GT that must not be drawn: a yellow line at x=0 and a world path.
        "gt_lanes": [{"points": [{"x": 0.0, "y": 0.0}, {"x": 0.0, "y": 60.0}], "paint": "yellow"}],
        "path_world": [{"x": 0.5, "y": 0.0}, {"x": 0.5, "y": 50.0}],
        "debug_gt": True,
    }
    with tempfile.TemporaryDirectory() as tmp:
        sf = Path(tmp) / "scene.jsonl"
        assert write_scene_file(sf, [state]) == 1
        out = Path(tmp) / "out.png"
        subprocess.run([str(exe), "--offline", str(sf), "--png", str(out), "--size", "640x360", "--no-ego", "--no-hud"],
                       check=True, timeout=120)
        img = cv2.imread(str(out))
    assert img is not None and img.shape == (360, 640, 3), None if img is None else img.shape
    b, g, r = (img[..., i].astype(int) for i in range(3))
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    white = (mn > 185) & (mx - mn < 40)
    bright = mx > 110
    yellow = bright & (r > 170) & (g > 120) & (b < 0.55 * g) & (r - g < 110)
    purple = bright & (r > 120) & (b > 120) & (g < 0.6 * np.minimum(r, b))
    red = bright & (r > 150) & (g < 0.45 * r) & (b < 0.45 * r)
    green = bright & (g > 150) & (r < 0.6 * g) & (b < 0.6 * g)
    h, w = img.shape[:2]
    left, right = slice(0, w // 2), slice(w // 2, w)
    # yellow paint is the left line only, white the right; edge red on the right; lead green.
    assert yellow[:, left].sum() > 100, yellow[:, left].sum()
    assert yellow[:, right].sum() == 0, ("yellow on the right", yellow[:, right].sum())
    assert white[:, right].sum() > 100, white[:, right].sum()
    assert purple.sum() == 0, purple.sum()
    assert red[:, right].sum() > 60 and red[:, left].sum() == 0, (red[:, left].sum(), red[:, right].sum())
    assert green.sum() > 60, green.sum()
    # the GT yellow line at x=0 would be dead centre, below the lead box
    centre = yellow[int(h * 0.75):, w // 2 - 12: w // 2 + 12]
    assert centre.sum() == 0, ("GT line drawn", centre.sum())
    print(f"end to end: publisher -> gvd_viz PNG: yellow L={int(yellow[:, left].sum())} white R={int(white[:, right].sum())} "
          f"red R={int(red[:, right].sum())} green={int(green.sum())} purple=0, GT not drawn OK")



def check_stale_and_drain_wired() -> None:
    main = (SRC / "src" / "main.cpp").read_text(encoding="utf-8")
    assert "scene_to_show(cur, last_rx)" in main, "live loop must use scene_to_show(cur, last_rx)"
    assert "bool stale = false" not in main
    udp = (SRC / "src" / "udp.cpp").read_text(encoding="utf-8")
    assert "for (int i = 0; i < 1024; ++i)" in udp
    print("main.cpp/udp.cpp: stale helper and 1024-datagram drain wired OK")

def check_glfw_visible_after_init() -> None:
    """GLFW 3.4 drops hints set before glfwInit, and init resets them."""
    src = (SRC / "src" / "context.cpp").read_text(encoding="utf-8")
    head_start = src.find("bool GlContext::init_headless")
    win_start = src.find("bool GlContext::init_window")
    assert 0 <= head_start < win_start, "init_headless / init_window not found"
    headless = src[head_start:win_start]
    assert "glfwWindowHint" not in headless, "GLFW hints in init_headless run before glfwInit"
    assert "headless_glfw_visible()" in headless, "init_headless must pass visible=headless_glfw_visible()"
    win = src[win_start:]
    init_pos = win.find("glfwInit")
    vis_pos = win.find("GLFW_VISIBLE")
    assert init_pos != -1 and vis_pos != -1 and vis_pos > init_pos, (
        "GLFW_VISIBLE must be set after glfwInit (GLFW 3.4 resets hints at init)"
    )
    print("context.cpp: GLFW_VISIBLE is set after glfwInit OK")


def check_deps_pinned() -> None:
    cmake = (SRC / "CMakeLists.txt").read_text(encoding="utf-8")
    # glm / nlohmann_json are pinned tarballs only (no find_package).
    assert "find_package(glm" not in cmake
    assert "find_package(nlohmann_json" not in cmake
    assert "glm/archive/refs/tags/1.0.1.tar.gz" in cmake
    assert "json/releases/download/v3.11.3/json.tar.xz" in cmake
    # GLFW may use a system 3.3+ package, else the pinned 3.4 tarball.
    assert "find_package(glfw3 3.3" in cmake
    assert "glfw/releases/download/3.4/glfw-3.4.zip" in cmake
    print("CMakeLists: glm/json pinned-only; GLFW system-or-3.4 OK")


def main() -> None:
    check_glfw_visible_after_init()
    check_deps_pinned()
    check_stale_and_drain_wired()
    b = build_dir()
    if b is None:
        print("test_viz_cpp_render: skipped build (no toolchain)")
        return
    exe = _find_exe(b)
    assert exe is not None, f"no {EXE_NAME} in {b}"
    check_ctest(b)
    check_end_to_end(exe)
    print("test_viz_cpp_render: all OK")


if __name__ == "__main__":
    main()
