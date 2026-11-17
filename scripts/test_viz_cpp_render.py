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

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
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


def require_live() -> bool:
    """CI sets GVD_REQUIRE_LIVE=1 so a skipped live/GLFW check is a failure."""
    v = (os.environ.get("GVD_REQUIRE_LIVE") or "").strip().lower()
    return v not in ("", "0", "false", "no", "off")


def _glfw_headless_built(b: Path) -> bool:
    name = "gvd_viz_test_glfw_headless.exe" if sys.platform == "win32" else "gvd_viz_test_glfw_headless"
    return any((b / sub / name).is_file() for sub in ("", "Release", "Debug"))


def _no_glfw_build_error(msg: str) -> bool:
    text = (msg or "").lower()
    return "built without glfw" in text or "no window mode" in text


def check_ctest(b: Path) -> None:
    ctest = shutil.which("ctest") or str(Path(shutil.which("cmake") or "ctest").with_name("ctest"))
    r = subprocess.run([ctest, "--test-dir", str(b), "-C", "Release", "--output-on-failure"],
                       capture_output=True, text=True)
    print(r.stdout[-3000:])
    print(f"check_ctest: GVD_REQUIRE_LIVE={os.environ.get('GVD_REQUIRE_LIVE')!r} "
          f"DISPLAY={os.environ.get('DISPLAY')!r}")
    assert r.returncode == 0, (r.stderr or r.stdout)[-2000:]
    # A GLFW-fallback init_headless failure used to return 77 (ctest Skipped,
    # exit 0). If we have a display and a GLFW test binary, that skip is a fail.
    glfw_built = _glfw_headless_built(b)
    display = os.environ.get("DISPLAY")
    if glfw_built and (require_live() or display):
        skipped = [ln for ln in r.stdout.splitlines()
                   if "gvd_viz_glfw_headless" in ln and "Skipped" in ln]
        assert not skipped, (
            "gvd_viz_glfw_headless was skipped while GLFW is built and "
            f"DISPLAY={display!r} GVD_REQUIRE_LIVE={os.environ.get('GVD_REQUIRE_LIVE')!r}: "
            + " | ".join(skipped)
        )
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



def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_xvfb():
    """Return (display, proc) or (None, None) if Xvfb cannot be started."""
    xvfb = shutil.which("Xvfb")
    if xvfb is None:
        return None, None
    for n in range(20, 80):
        disp = f":{n}"
        lock = Path(f"/tmp/.X{n}-lock")
        if lock.exists():
            continue
        proc = subprocess.Popen(
            [xvfb, disp, "-screen", "0", "1280x720x24", "-nolisten", "tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.3)
        if proc.poll() is None:
            return disp, proc
        proc.wait(timeout=2)
    return None, None


def _grab_display_rgb(display: str):
    """Screenshot the X display with mss (requirements-retail).

    A hand-rolled libX11 ctypes grab SIGSEGVed the CI suite
    (test_viz_cpp_render.py exit -11) because untyped X11 calls
    truncate 64-bit Display* pointers. mss already sets argtypes.
    """
    import mss

    os.environ["DISPLAY"] = display
    with mss.MSS() as sct:
        mon = sct.monitors[0]
        shot = sct.grab(mon)
        return int(shot.width), int(shot.height), bytes(shot.rgb)


def _wait_yellow(display: str, timeout_s: float):
    """Poll the display until yellow lane pixels appear, or time out."""
    deadline = time.time() + timeout_s
    w = h = 0
    rgb = b""
    n = 0
    last_err = None
    while time.time() < deadline:
        try:
            w, h, rgb = _grab_display_rgb(display)
            n = _yellow_count(w, h, rgb)
            if n > 100:
                return w, h, rgb, n
        except Exception as exc:  # mss may briefly fail before the window maps
            last_err = exc
        time.sleep(0.15)
    if last_err is not None and n == 0 and not rgb:
        raise RuntimeError(f"mss screenshot failed on {display}: {last_err}")
    return w, h, rgb, n


def _yellow_count(w: int, h: int, rgb: bytes) -> int:
    n = 0
    for i in range(0, w * h * 3, 3):
        r, g, b = rgb[i], rgb[i + 1], rgb[i + 2]
        mx = max(r, g, b)
        if mx <= 110:
            continue
        if r > 170 and g > 120 and b < 0.55 * g and (r - g) < 110:
            n += 1
    return n


def _send_scene(port: int, body: dict) -> None:
    payload = json.dumps(body, separators=(",", ":")).encode()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(payload, ("127.0.0.1", port))
    finally:
        sock.close()


def check_live_uses_scene_to_show(exe: Path) -> None:
    """Behavioral check of main.cpp:167. After 1 s with no UDP, the live
    window must blank tracks/lanes (scene_to_show). `Scene show = cur`
    would keep the last yellow lanes on screen.
    Screenshots the Xvfb display with mss. Skips only when this build
    truly has no GLFW, or there is no display. GVD_REQUIRE_LIVE=1 (CI)
    turns those SKIPs into failures.
    """
    display = os.environ.get("DISPLAY")
    xvfb_proc = None
    if not display:
        display, xvfb_proc = _start_xvfb()
        if display is None:
            if require_live():
                raise AssertionError("GVD_REQUIRE_LIVE=1 but no DISPLAY / Xvfb")
            print("SKIP live scene_to_show wiring (no DISPLAY / Xvfb)")
            return
        os.environ["DISPLAY"] = display
    port = _free_port()
    env = os.environ.copy()
    env["DISPLAY"] = display
    proc = subprocess.Popen(
        [str(exe), "--listen", str(port), "--size", "640x360", "--no-hud", "--no-ego"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    try:
        # Fail fast. Only SKIP a no-GLFW *build*; any other early exit
        # (including a forced rx.open failure, exit 9) is a failure.
        # GVD_REQUIRE_LIVE=1 also fails the no-GLFW skip.
        time.sleep(0.4)
        if proc.poll() is not None:
            err = (proc.stderr.read() if proc.stderr else "") or ""
            out = (proc.stdout.read() if proc.stdout else "") or ""
            msg = (err + out).strip()
            no_glfw = _no_glfw_build_error(msg)
            if no_glfw and not require_live():
                print(f"SKIP live scene_to_show wiring (no GLFW build, "
                      f"exited {proc.returncode}): {msg}")
                return
            raise AssertionError(
                f"gvd_viz --listen exited {proc.returncode}: {msg}"
                + ("; GVD_REQUIRE_LIVE=1, missing GLFW is a failure" if no_glfw else "")
            )
        scene = {
            "v": 1, "type": "gvd_scene", "seq": 1, "t": 0,
            "lanes": [
                {"pts": [[-1.8, -12], [-1.8, 90]], "paint": "yellow"},
                {"pts": [[1.8, -12], [1.8, 90]], "paint": "yellow"},
            ],
        }
        # Give the window a moment to map, then send a distinctive scene.
        time.sleep(0.6)
        for _ in range(5):
            _send_scene(port, scene)
            time.sleep(0.05)
        # Poll until the yellow lanes are actually on screen (window map
        # + first frames), then wait past kStaleTimeout (1 s) in silence.
        w1, h1, rgb1, yellow_fresh = _wait_yellow(display, timeout_s=8.0)
        time.sleep(1.4)
        w2, h2, rgb2 = _grab_display_rgb(display)
        yellow_stale = _yellow_count(w2, h2, rgb2)
        assert yellow_fresh > 100, (
            f"live window did not show yellow lanes while fresh "
            f"(yellow={yellow_fresh} {w1}x{h1})"
        )
        assert yellow_stale < 20, (
            f"live window still shows yellow after 1 s silence "
            f"(fresh={yellow_fresh} stale={yellow_stale}); main.cpp must use scene_to_show"
        )
        print(
            f"live scene_to_show wiring: fresh yellow={yellow_fresh} "
            f"stale yellow={yellow_stale} ({w2}x{h2}) OK"
        )
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        if xvfb_proc is not None and xvfb_proc.poll() is None:
            xvfb_proc.terminate()
            try:
                xvfb_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                xvfb_proc.kill()


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
    print(f"GVD_REQUIRE_LIVE={os.environ.get('GVD_REQUIRE_LIVE') or ''} "
          f"(live/GLFW skips are {'FAIL' if require_live() else 'allowed'})")
    check_deps_pinned()
    b = build_dir()
    if b is None:
        print("test_viz_cpp_render: skipped build (no toolchain)")
        return
    exe = _find_exe(b)
    assert exe is not None, f"no {EXE_NAME} in {b}"
    check_ctest(b)
    check_end_to_end(exe)
    # Behavioral wiring checks first so a mutant is killed by observed
    # behaviour, not only by a source grep.
    check_live_uses_scene_to_show(exe)
    check_glfw_visible_after_init()
    check_stale_and_drain_wired()
    print("test_viz_cpp_render: all OK")


if __name__ == "__main__":
    main()
