#!/usr/bin/env python3
"""Run the offline GVD suite (no BeamNG, no GPU).

Every ``scripts/test_*.py`` and ``scripts/test_*.lua`` runs. A script that
cannot run on this host is skipped, and the skip is printed:

* ``# gvd-ci: requires windows`` — skipped unless ``sys.platform == "win32"``.
* ``# gvd-ci: requires beamng`` — skipped unless BeamNG.drive or BeamNG.tech
  is installed (a ``tech.key`` beside a Tech executable, or a Steam
  ``BeamNG.drive`` install). ``beamngpy`` on ``PYTHONPATH`` is not enough.

Lua scripts need ``lua5.1`` (then ``lua``). A missing interpreter fails the
suite; it is not a skip.

Each script runs with its own temp HOME / USERPROFILE / APPDATA /
LOCALAPPDATA / TMPDIR / TORCH_HOME (``gvd_test_env``), removed afterwards,
so GVD bus files (``gvd_state.json``, ``gvd_cmd.json``,
``gvd_strip_record.json``, ...) cannot leak from one test into the next or
into the real home. A script that leaves files in its torch hub folder
(downloaded weights) fails.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from gvd_test_env import isolated_env, torch_hub_files  # noqa: E402
MARKER_WINDOWS = "gvd-ci: requires windows"
MARKER_BEAMNG = "gvd-ci: requires beamng"
TIMEOUT_S = 300


def _head(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return ""


def _has_directive(path: Path, marker: str) -> bool:
    """True when a comment line carries the marker.

    A string inside a test (``"# gvd-ci: requires windows"``) is not a
    directive, so the skip check itself still runs.
    """
    for line in _head(path).splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("--"):
            if marker in stripped:
                return True
    return False


def _beamng_installed() -> bool:
    """True only when a game install is visible. A pip import is not an install."""
    candidates = [
        Path.home() / "BeamNG" / "BeamNG.tech" / "BeamNG.tech.exe",
        Path.home() / "BeamNG.tech" / "BeamNG.tech.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "BeamNG" / "BeamNG.tech" / "BeamNG.tech.exe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Steam"
        / "steamapps"
        / "common"
        / "BeamNG.drive"
        / "BeamNG.drive.exe",
    ]
    steam = os.environ.get("STEAM") or os.environ.get("SteamPath")
    if steam:
        candidates.append(Path(steam) / "steamapps" / "common" / "BeamNG.drive" / "BeamNG.drive.exe")
    for path in candidates:
        if path.is_file():
            return True
    return False


def _skip_reason(path: Path) -> str | None:
    if _has_directive(path, MARKER_WINDOWS) and sys.platform != "win32":
        return "windows-only (marker); this host is not Windows"
    if _has_directive(path, MARKER_BEAMNG) and not _beamng_installed():
        return "BeamNG-dependent (marker); BeamNG is not installed"
    return None


def _lua_exe() -> str | None:
    for name in ("lua5.1", "lua"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _run(cmd: list[str], label: str) -> tuple[str, int]:
    scratch = Path(tempfile.mkdtemp(prefix="gvd-suite-"))
    try:
        env = isolated_env(os.environ, scratch)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                env=env,
                timeout=TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            print(f"FAIL {label}: timed out after {TIMEOUT_S}s", flush=True)
            return "fail", 1
        downloaded = torch_hub_files(env)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if downloaded:
        print(f"FAIL {label}: downloaded weights into the torch hub: {', '.join(downloaded[:5])}", flush=True)
        return "fail", 1
    if proc.returncode == 0:
        print(f"PASS {label}", flush=True)
        return "pass", 0
    print(f"FAIL {label}: exit {proc.returncode}", flush=True)
    return "fail", proc.returncode


def main() -> None:
    py_tests = sorted(p for p in SCRIPTS.glob("test_*.py") if p.is_file())
    lua_tests = sorted(p for p in SCRIPTS.glob("test_*.lua") if p.is_file())
    if not py_tests and not lua_tests:
        raise SystemExit("run_offline_suite: no scripts/test_* found")

    passed = 0
    failed = 0
    skipped = 0
    lua = _lua_exe()

    for path in py_tests:
        reason = _skip_reason(path)
        if reason:
            skipped += 1
            print(f"SKIP {path.name}: {reason}", flush=True)
            continue
        status, _ = _run([sys.executable, str(path)], path.name)
        if status == "pass":
            passed += 1
        else:
            failed += 1

    if lua is None:
        print("FAIL lua: lua5.1 is not on PATH (install lua5.1; do not skip the Lua harness)", flush=True)
        failed += len(lua_tests) or 1
    else:
        for path in lua_tests:
            reason = _skip_reason(path)
            if reason:
                skipped += 1
                print(f"SKIP {path.name}: {reason}", flush=True)
                continue
            status, _ = _run([lua, str(path)], path.name)
            if status == "pass":
                passed += 1
            else:
                failed += 1

    print(
        f"run_offline_suite: {passed} passed, {failed} failed, {skipped} skipped",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
