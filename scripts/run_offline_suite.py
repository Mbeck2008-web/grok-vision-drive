#!/usr/bin/env python3
"""Run the offline GVD suite (no BeamNG, no GPU).

Every ``scripts/test_*.py`` and ``scripts/test_*.lua`` runs. A script that
cannot run on this host is skipped, and the skip is printed:

* ``# gvd-ci: requires windows`` — skipped unless ``sys.platform == "win32"``.
* ``# gvd-ci: requires beamng`` — skipped unless BeamNG.drive or BeamNG.tech
  is installed (a ``tech.key`` beside a Tech executable, or a Steam
  ``BeamNG.drive`` install). ``beamngpy`` on ``PYTHONPATH`` is not enough.
* ``# gvd-ci: requires torch`` — skipped unless this interpreter can import
  ``torch``. Only for a script with nothing to check without it; a script
  with torch-free parts skips those parts itself with
  ``gvd_test_env.skip_check``.

A check skipped inside a script (``skip_check``) is listed at the end and
counted on the summary line, so "passed" never hides a check that did not run.

Lua scripts need ``lua5.1`` (then ``lua``). A missing interpreter fails the
suite; it is not a skip.

Each script runs with its own temp HOME / USERPROFILE / APPDATA /
LOCALAPPDATA / TMPDIR / TORCH_HOME (``gvd_test_env``), removed afterwards,
so GVD bus files (``gvd_state.json``, ``gvd_cmd.json``,
``gvd_strip_record.json``, ...) cannot leak from one test into the next or
into the real home. A script that leaves files in its torch hub folder
(downloaded weights) fails.

Collector output goes under ``gvd_test_env.suite_data_root()``: each script
gets its own subfolder there (``gvd_test_env.collect_root`` reads it), removed
afterwards. The first lines print that folder and the temp root with their
filesystem and free space.

The first line logs the torch and torchvision versions the test scripts see
(read in a child with the same isolated environment). On Windows, the dev PC
target, the suite stops before running anything unless they are exactly the
pins below (a local suffix such as ``+cu126`` is allowed). Elsewhere (Linux
CI) the line is logged only.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from gvd_test_env import (  # noqa: E402
    SKIP_LOG_ENV,
    TEST_DATA_ENV,
    describe_root,
    isolated_env,
    read_skips,
    suite_data_root,
    torch_hub_files,
)

MARKER_WINDOWS = "gvd-ci: requires windows"
MARKER_BEAMNG = "gvd-ci: requires beamng"
MARKER_TORCH = "gvd-ci: requires torch"
TIMEOUT_S = 300
# Scripts that train several small models end to end need longer on CI runners.
SCRIPT_TIMEOUT_S = {"test_epoch_ckpt.py": 600, "test_pause_dir.py": 600, "test_seed.py": 600}


# The dev PC's torch / torchvision (the CI workflow installs the same versions).
TORCH_PINS = {"torch": "2.14.1", "torchvision": "0.29.1"}
_VERSIONS_CHILD = (
    "import importlib.metadata as m, json\n"
    "out = {}\n"
    "for name in %r:\n"
    "    try:\n"
    "        out[name] = m.version(name)\n"
    "    except m.PackageNotFoundError:\n"
    "        out[name] = None\n"
    "print(json.dumps(out))\n"
)


def pin_matches(version: str | None, pin: str) -> bool:
    """``version`` is exactly ``pin``, optionally with a local suffix (``2.14.1+cu126``)."""
    if version is None:
        return False
    return re.fullmatch(re.escape(pin) + r"(\+[0-9A-Za-z.]+)?", version) is not None


def torch_version_line(versions: dict[str, str | None]) -> str:
    parts = [f"{name} {versions.get(name) or 'not installed'}" for name in TORCH_PINS]
    return "[suite] " + ", ".join(parts)


def torch_pin_problems(versions: dict[str, str | None]) -> list[str]:
    return [
        f"{name} is {versions.get(name) or 'not installed'}, need {pin}"
        for name, pin in TORCH_PINS.items()
        if not pin_matches(versions.get(name), pin)
    ]


def torch_pins_fatal(versions: dict[str, str | None], platform: str) -> bool:
    """Only the Windows (dev PC) target enforces the pins; other hosts log the line."""
    return platform == "win32" and bool(torch_pin_problems(versions))


def _torch_versions() -> dict[str, str | None]:
    """Versions as the test scripts see them: a child with the same isolated env."""
    scratch = Path(tempfile.mkdtemp(prefix="gvd-suite-"))
    try:
        env = isolated_env(os.environ, scratch)
        proc = subprocess.run(
            [sys.executable, "-c", _VERSIONS_CHILD % (tuple(TORCH_PINS),)],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if proc.returncode != 0:
        raise SystemExit(f"run_offline_suite: could not read torch versions: {proc.stderr.strip()}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


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


def _torch_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("torch") is not None


def _skip_reason(path: Path) -> str | None:
    if _has_directive(path, MARKER_WINDOWS) and sys.platform != "win32":
        return "windows-only (marker); this host is not Windows"
    if _has_directive(path, MARKER_BEAMNG) and not _beamng_installed():
        return "BeamNG-dependent (marker); BeamNG is not installed"
    if _has_directive(path, MARKER_TORCH) and not _torch_installed():
        return "torch-dependent (marker); torch is not installed"
    return None


def _lua_exe() -> str | None:
    for name in ("lua5.1", "lua"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _run(cmd: list[str], label: str, data_root: Path | None = None) -> tuple[str, int, list[str]]:
    """Run one script isolated. Returns status, exit code, and the checks it skipped."""
    scratch = Path(tempfile.mkdtemp(prefix="gvd-suite-"))
    data_dir = Path(tempfile.mkdtemp(prefix="gvd-suite-", dir=str(data_root))) if data_root is not None else None
    skips: list[str] = []
    timeout_s = SCRIPT_TIMEOUT_S.get(label, TIMEOUT_S)
    try:
        env = isolated_env(os.environ, scratch)
        skip_log = scratch / "skipped_checks.txt"
        env[SKIP_LOG_ENV] = str(skip_log)
        if data_dir is not None:
            env[TEST_DATA_ENV] = str(data_dir)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                env=env,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            print(f"FAIL {label}: timed out after {timeout_s}s", flush=True)
            return "fail", 1, read_skips(skip_log)
        downloaded = torch_hub_files(env)
        skips = read_skips(skip_log)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        if data_dir is not None:
            shutil.rmtree(data_dir, ignore_errors=True)
    if downloaded:
        print(f"FAIL {label}: downloaded weights into the torch hub: {', '.join(downloaded[:5])}", flush=True)
        return "fail", 1, skips
    if proc.returncode == 0:
        note = f" ({len(skips)} check(s) skipped)" if skips else ""
        print(f"PASS {label}{note}", flush=True)
        return "pass", 0, skips
    print(f"FAIL {label}: exit {proc.returncode}", flush=True)
    return "fail", proc.returncode, skips


def main() -> None:
    py_tests = sorted(p for p in SCRIPTS.glob("test_*.py") if p.is_file())
    lua_tests = sorted(p for p in SCRIPTS.glob("test_*.lua") if p.is_file())
    if not py_tests and not lua_tests:
        raise SystemExit("run_offline_suite: no scripts/test_* found")
    versions = _torch_versions()
    print(torch_version_line(versions), flush=True)
    if torch_pins_fatal(versions, sys.platform):
        problems = "; ".join(torch_pin_problems(versions))
        print(f"FAIL torch pins: {problems} (Windows dev PC target)", flush=True)
        raise SystemExit(f"run_offline_suite: torch pins not met on Windows: {problems}")
    data_root = suite_data_root()
    print(f"[suite] temp {describe_root(tempfile.gettempdir())}", flush=True)

    passed = 0
    failed = 0
    skipped = 0
    check_skips: list[str] = []
    lua = _lua_exe()

    for path in py_tests:
        reason = _skip_reason(path)
        if reason:
            skipped += 1
            print(f"SKIP {path.name}: {reason}", flush=True)
            continue
        status, _, skips = _run([sys.executable, str(path)], path.name, data_root)
        check_skips.extend(skips)
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
            status, _, skips = _run([lua, str(path)], path.name, data_root)
            check_skips.extend(skips)
            if status == "pass":
                passed += 1
            else:
                failed += 1

    for line in check_skips:
        print(line, flush=True)
    print(
        f"run_offline_suite: {passed} passed, {failed} failed, {skipped} skipped"
        f" (+{len(check_skips)} checks skipped inside scripts)",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
