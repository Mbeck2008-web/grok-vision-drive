#!/usr/bin/env python3
"""Offline checks for Lua gvdDocsDir / Python Tech sandbox GVD (no BeamNG).

Lua gvdDocsDir + Python gvd_docs_dir (same resolve; #39 USERPROFILE Documents is the FAIL)
-----------------------------------------------------------------------------------------
1. env ``GVD_DOCS_DIR`` if set (full GVD root)
2. ``%LOCALAPPDATA%/BeamNG/BeamNG.tech/current/Documents/GVD``
3. else ``{USERPROFILE|HOME}/AppData/Local/BeamNG/BeamNG.tech/current/Documents/GVD``
   (synthesize LOCALAPPDATA — never USERPROFILE/Documents)
4. else FS:getUserPath / virtual2Native / getFileRealPath:
   - ``…/BeamNG/BeamNG.tech/current`` → that + ``/Documents/GVD``
   - ``…/AppData/Local`` → LOCALAPPDATA + Tech tail
5. else last resort ``Documents/GVD`` (never a bare ``gvd_*.json`` under
   BeamNG userfolder ``current\\``).

Not USERPROFILE\\Documents (Tech GELua cannot read it). Not OneDrive. No junctions.

Lua ``gvd_state`` LINKED (not path-only)
---------------------------------------
``readText`` uses absolute ``io.open`` **before** VFS ``FS:readFile``. ``pollStateFile``
sets ``lastGood`` and logs read ok / read fail / json fail distinctly. ``pushUiState``
and ``onExtensionLoaded`` poll + push ``gvdUi`` so CEF is not stuck on
"supervisor not running" when the disk file is fresh.

Run: ``PYTHONPATH=. python scripts/test_gvd_docs_dir.py``
Lua extract harness (when lua5.1/luajit is on PATH): ``lua5.1 scripts/test_gvd_docs_dir.lua``
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LUA = ROOT / "beamng_mod" / "lua" / "ge" / "extensions" / "gvd" / "main.lua"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TECH_TAIL = "BeamNG/BeamNG.tech/current/Documents/GVD"
TECH_BAT = r"%LOCALAPPDATA%\BeamNG\BeamNG.tech\current\Documents\GVD"


def _fn(src: str, name: str) -> str:
    m = re.search(rf"local function {re.escape(name)}\s*\(.*?\nend\n", src, re.S)
    assert m, f"missing lua function {name}"
    return m.group(0)


def local_app_from_path(p: str | None) -> str | None:
    """Python mirror of Lua ``_localAppFromPath``."""
    if not p:
        return None
    p = str(p).replace("\\", "/")
    m = re.match(r"^(.+/AppData/Local)", p)
    if m and m.group(1) and "onedrive" not in m.group(1).lower():
        return m.group(1)
    return None


def tech_current_from_path(p: str | None) -> str | None:
    """Python mirror of Lua ``_techCurrentFromPath``."""
    if not p:
        return None
    p = str(p).replace("\\", "/")
    m = re.match(r"^(.+/BeamNG/BeamNG.tech/current)", p)
    if m and m.group(1) and "onedrive" not in m.group(1).lower():
        return m.group(1)
    return None


def resolve_docs_dir(
    env: dict[str, str | None],
    fs_paths: list[str] | None = None,
) -> str:
    """Python mirror of gvdDocsDir env + FS reconstruction."""
    ov = env.get("GVD_DOCS_DIR")
    if ov and str(ov).strip():
        return str(ov).strip().replace("\\", "/")
    la = env.get("LOCALAPPDATA")
    if la and str(la).strip() and "onedrive" not in str(la).lower():
        return str(la).replace("\\", "/") + "/" + TECH_TAIL
    home = env.get("USERPROFILE") or env.get("HOME")
    if home and str(home).strip():
        synth = str(home).replace("\\", "/") + "/AppData/Local"
        if "onedrive" not in synth.lower():
            return synth + "/" + TECH_TAIL
    for p in fs_paths or []:
        cur = tech_current_from_path(p)
        if cur:
            return cur + "/Documents/GVD"
        la2 = local_app_from_path(p)
        if la2:
            return la2 + "/" + TECH_TAIL
    return "Documents/GVD"


def check_source_contracts(lua: str) -> None:
    local_fn = _fn(lua, "_localAppFromPath")
    tech_fn = _fn(lua, "_techCurrentFromPath")
    try_env = _fn(lua, "_tryEnvDocs")
    docs_fn = _fn(lua, "gvdDocsDir")
    file_fn = _fn(lua, "gvdFile")
    link_fn = _fn(lua, "linkState")

    assert "AppData/Local" in local_fn
    assert "BeamNG/BeamNG.tech/current" in tech_fn
    assert TECH_TAIL in lua

    try_exec = re.sub(r"--[^\n]*", "", try_env)
    ov = try_exec.find("GVD_DOCS_DIR")
    la = try_exec.find("LOCALAPPDATA")
    assert 0 <= ov < la, "GVD_DOCS_DIR then LOCALAPPDATA"
    assert "AppData/Local" in try_exec
    assert not re.search(r"USERPROFILE.+/Documents/GVD", try_exec)
    assert "/Documents/GVD" not in try_exec or "TECH_GVD_TAIL" in try_env or TECH_TAIL in lua

    docs_exec = re.sub(r"--[^\n]*", "", docs_fn)
    assert "/Documents/GVD" in docs_exec or "TECH_GVD_TAIL" in lua
    assert "directoryCreate" in docs_exec, "mkdir of resolved docs dir"
    assert "gvdDocsLogged" in docs_fn and "docs dir=" in docs_fn, "one-shot log"
    assert "dir = 'Documents/GVD'" in docs_exec, "last resort is Documents/GVD, never CWD or current\\"
    assert "_tryEnvDocs" in docs_exec and "_tryFsDocs" in docs_exec

    file_exec = re.sub(r"--[^\n]*", "", file_fn)
    assert "gvdDocsDir() .. '/' .. name" in file_exec
    assert not re.search(r"return\s+name\b", file_exec)

    # LINKED = gvd_state heartbeat (lastGood / hbAge). Not gvd_ego.json.
    link_exec = re.sub(r"--[^\n]*", "", link_fn)
    assert "lastGood" in link_exec and "hbAgeS" in link_exec
    assert "gvd_ego" not in link_exec and "egoFb" not in link_exec, link_fn
    assert "userEgoPath" not in link_exec

    helpers = local_fn + tech_fn + try_env + docs_fn
    assert "Accounts" not in helpers and "UserFolder" not in helpers
    assert "SHGetKnownFolderPath" not in helpers
    assert "FOLDERID" not in helpers
    assert "mklink" not in lua.lower()

    read_fn = _fn(lua, "readText")
    io_fn = _fn(lua, "_ioOpenRead")
    read_exec = re.sub(r"--[^\n]*", "", read_fn)
    io_exec = re.sub(r"--[^\n]*", "", io_fn)
    assert "io.open" in io_exec, "absolute io.open helper"
    assert "_isAbsDiskPath" in read_exec and "_ioReadAll" in read_exec
    assert 0 <= read_exec.find("_isAbsDiskPath") < read_exec.find("FS:readFile"), "io.open before FS:readFile"
    assert read_exec.find("_ioReadAll") < read_exec.find("FS:readFile")
    assert read_exec.find("_isAbsDiskPath") < read_exec.find("if readFile")
    assert "_isAbsDiskPath" in lua and r"%a:[/\\" in lua
    poll_fn = _fn(lua, "pollStateFile")
    assert "lastGood = st" in poll_fn
    assert "gvd_state read ok path=" in poll_fn
    assert "read fail path=" in poll_fn
    assert "json fail len=" in poll_fn
    assert "pushUi()" in poll_fn, "first lastGood must push gvdUi"
    loaded = re.search(r"function M\.onExtensionLoaded\(\).*?\nend\n", lua, re.S)
    assert loaded, "onExtensionLoaded missing"
    load_exec = re.sub(r"--[^\n]*", "", loaded.group(0))
    assert "pollStateFile()" in load_exec and "pushUi()" in load_exec
    poke = re.search(r"function M\.pushUiState\(\).*?\nend\n", lua, re.S)
    assert poke, "pushUiState missing"
    poke_exec = re.sub(r"--[^\n]*", "", poke.group(0))
    assert "pollStateFile()" in poke_exec and "pushUi()" in poke_exec
    assert "pollState(dt)" in lua
    assert re.search(r"function M\.onPreRender\(dt\).*?pollState\(dt\)", lua, re.S)
    assert re.search(r"function M\.onUpdate\(dt\).*?pollState\(dt\)", lua, re.S)

    for bat_name in ("install.bat", "play_gvd.bat", "play_gvd_tech.bat"):
        bat = (ROOT / bat_name).read_text(encoding="utf-8", errors="ignore")
        assert TECH_BAT in bat, bat_name
        assert "GVD_DOCS_DIR" in bat, bat_name
        assert 'set "GVD_DOCS_DIR=%USERPROFILE%\\Documents\\GVD"' not in bat, bat_name
        assert 'set "DOCS=%USERPROFILE%\\Documents\\GVD"' not in bat, bat_name
        assert "OneDrive" not in bat and "FOLDERID" not in bat and "mklink" not in bat.lower()


def check_python_mirror() -> None:
    spec = "C:/Users/Name/AppData/Local/" + TECH_TAIL
    assert local_app_from_path(r"C:\Users\Name\AppData\Local\BeamNG\BeamNG.tech\current") == (
        "C:/Users/Name/AppData/Local"
    )
    assert (
        tech_current_from_path("C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current")
        == "C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current"
    )
    assert tech_current_from_path("C:/Users/Name/AppData/Local/BeamNG.tech/current") is None
    assert local_app_from_path("C:/Users/Name/Documents/BeamNG.drive/current") is None
    assert local_app_from_path("D:/BeamNG.drive/current") is None

    assert (
        resolve_docs_dir(
            {"LOCALAPPDATA": r"C:\Users\Name\AppData\Local", "USERPROFILE": r"C:\Users\Name"},
            ["C:/Users/Name/Documents/BeamNG.drive/0.36/current"],
        )
        == spec
    )
    assert resolve_docs_dir({"USERPROFILE": r"C:\Users\Name"}, []) == spec
    assert resolve_docs_dir({"HOME": "/home/me"}, []) == "/home/me/AppData/Local/" + TECH_TAIL
    assert (
        resolve_docs_dir({}, ["C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current"])
        == spec
    )
    assert (
        resolve_docs_dir({}, ["C:/Users/Name/AppData/Local/BeamNG.drive/0.36"])
        == spec
    )
    # Documents userfolder cannot reconstruct LOCALAPPDATA → last resort (not USERPROFILE Documents).
    assert (
        resolve_docs_dir({}, ["C:/Users/Name/Documents/BeamNG.drive/0.36/current"])
        == "Documents/GVD"
    )
    last = resolve_docs_dir({}, [])
    assert last == "Documents/GVD"
    assert not last.endswith(".json")
    assert last != "current"

    for name in ("gvd_state.json", "gvd_ego.json", "gvd_engage.json", "gvd_cmd.json"):
        p = resolve_docs_dir({"LOCALAPPDATA": "C:/Users/Name/AppData/Local"}) + "/" + name
        assert p.endswith("/Documents/GVD/" + name)
        assert TECH_TAIL in p
        assert not p.startswith("gvd_")
        assert not p.endswith("/current/" + name)


def _slash(p: Path | str) -> str:
    return str(p).replace("\\", "/").rstrip("/")


@contextmanager
def _env(**kwargs: str | None):
    keys = list(kwargs)
    old = {k: os.environ.get(k) for k in keys}
    try:
        for k, v in kwargs.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def check_python_tech_sandbox() -> None:
    """Python GVD bus: Tech current\\Documents\\GVD; override wins; never USERPROFILE Documents."""
    from python.control.actuate import cmd_path, ego_path, engage_path
    from python.runtime import paths
    from python.runtime.state_io import gvd_docs_dir, state_path

    src = (ROOT / "python" / "runtime" / "paths.py").read_text(encoding="utf-8")
    state_src = (ROOT / "python" / "runtime" / "state_io.py").read_text(encoding="utf-8")
    assert "GVD_DOCS_DIR" in src
    assert "BeamNG.tech" in src and "current" in src
    assert "TECH_GVD_REL" in src
    assert "SHGetKnownFolderPath" not in src
    assert "FOLDERID" not in src
    assert "Accounts" not in src and "UserFolder" not in src, "no Personal-reg OneDrive tip"
    assert "expanduser" not in src and "expanduser" not in state_src
    assert "python.runtime.paths" in state_src and "gvd_docs_dir" in state_src
    assert "USERPROFILE" in src and "Documents" in src
    # USERPROFILE is only used to synthesize AppData\\Local, not Documents\\GVD.
    assert "AppData" in src and "Local" in src

    for rel in (
        "python/runtime/state_io.py",
        "python/control/actuate.py",
        "python/sensors/extras.py",
        "python/data/record.py",
        "python/train/train_e2e.py",
        "python/run_vision.py",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "gvd_docs_dir" in text, rel
        assert 'Path.home() / "Documents"' not in text, rel
        assert 'expanduser("~")' not in text, rel

    spec = "C:/Users/Name/AppData/Local/" + TECH_TAIL
    lua_spec = resolve_docs_dir({"LOCALAPPDATA": r"C:\Users\Name\AppData\Local"})
    assert lua_spec == spec, lua_spec

    assert paths.is_onedrive_path(Path("C:/Users/Name/OneDrive/Documents"))
    assert paths.is_onedrive_path(r"C:\Users\Name\OneDrive - Personal\Documents")
    assert paths.is_onedrive_path("C:/Users/Name/OneDrive - Contoso/Documents")
    assert not paths.is_onedrive_path(Path("C:/Users/Name/AppData/Local"))
    assert not paths.is_onedrive_path(None)
    assert not paths.is_onedrive_path("  ")

    # LOCALAPPDATA wins over USERPROFILE Documents (the live NO LINK path).
    with _env(
        LOCALAPPDATA="C:/Users/Name/AppData/Local",
        USERPROFILE="C:/Users/Name",
        HOME="/home/other",
        GVD_DOCS_DIR=None,
    ):
        got = paths.gvd_docs_path()
        assert _slash(got) == spec, got
        assert got.parts[-2:] == ("Documents", "GVD")
        assert "BeamNG.tech" in got.parts
        assert "OneDrive" not in _slash(got)
        assert _slash(got) == lua_spec
        assert "Users/Name/Documents/GVD" not in _slash(got)

    # USERPROFILE alone synthesizes AppData\\Local\\…Tech…\\Documents\\GVD.
    with _env(LOCALAPPDATA=None, USERPROFILE="C:/Users/Name", HOME=None, GVD_DOCS_DIR=None):
        got = paths.gvd_docs_path()
        assert _slash(got) == spec, got
        assert _slash(got) == resolve_docs_dir({"USERPROFILE": r"C:\Users\Name"})

    # OneDrive LOCALAPPDATA is rejected; fall through to USERPROFILE\\AppData\\Local.
    with _env(
        LOCALAPPDATA="C:/Users/Name/OneDrive",
        USERPROFILE="C:/Users/Name",
        GVD_DOCS_DIR=None,
    ):
        got = paths.gvd_docs_path()
        assert "OneDrive" not in _slash(got), got
        assert _slash(got) == spec

    # GVD_DOCS_DIR override wins over Tech sandbox and USERPROFILE.
    with tempfile.TemporaryDirectory() as td:
        override = Path(td) / "custom_gvd"
        with _env(
            GVD_DOCS_DIR=str(override),
            USERPROFILE="C:/Users/Name",
            LOCALAPPDATA="C:/Users/Name/AppData/Local",
        ):
            got = paths.gvd_docs_path()
            assert Path(got) == override, got
            root = gvd_docs_dir()
            assert root == override and root.is_dir()
            assert state_path().parent == root
            assert cmd_path().parent == engage_path().parent == ego_path().parent == root

    # Blank GVD_DOCS_DIR is ignored (fall through to Tech sandbox).
    with _env(
        GVD_DOCS_DIR="   ",
        LOCALAPPDATA="C:/Users/Name/AppData/Local",
        USERPROFILE="C:/Users/Name",
    ):
        assert _slash(paths.gvd_docs_path()) == spec

    # Last resort relative Documents/GVD (Lua last-resort contract).
    with _env(GVD_DOCS_DIR=None, LOCALAPPDATA=None, USERPROFILE=None, HOME=None):
        last = paths.gvd_docs_path()
        assert _slash(last) == "Documents/GVD"
        assert not str(last).endswith(".json")

    # Live mkdir: Tech sandbox under LOCALAPPDATA, never USERPROFILE\\Documents or OneDrive.
    with tempfile.TemporaryDirectory() as td:
        profile = Path(td) / "Users" / "Name"
        la = profile / "AppData" / "Local"
        onedrive = profile / "OneDrive" / "Documents"
        onedrive.mkdir(parents=True)
        local_docs = profile / "Documents"
        local_docs.mkdir(parents=True)
        with _env(
            LOCALAPPDATA=str(la),
            USERPROFILE=str(profile),
            HOME=None,
            GVD_DOCS_DIR=None,
        ):
            root = gvd_docs_dir()
            expected = la.joinpath(*paths.TECH_GVD_REL)
            assert root == expected and root.is_dir()
            assert "OneDrive" not in _slash(root)
            assert not (onedrive / "GVD").exists()
            assert not (local_docs / "GVD").exists()
            assert state_path().parent == root
            assert cmd_path().parent == engage_path().parent == ego_path().parent == root
            assert root.parts[-2:] == ("Documents", "GVD")
            assert "BeamNG.tech" in root.parts

    # USERPROFILE-only mkdir still lands on AppData\\Local Tech sandbox.
    with tempfile.TemporaryDirectory() as td:
        profile = Path(td) / "Users" / "Name"
        with _env(LOCALAPPDATA=None, USERPROFILE=str(profile), HOME=None, GVD_DOCS_DIR=None):
            root = gvd_docs_dir()
            expected = profile.joinpath("AppData", "Local", *paths.TECH_GVD_REL)
            assert root == expected and root.is_dir()
            assert not (profile / "Documents" / "GVD").exists()

    # Host fallback: Tech tail when LOCALAPPDATA/HOME can be synthesized.
    with _env(GVD_DOCS_DIR=None):
        live = paths.gvd_docs_path()
        assert live.parts[-2:] == ("Documents", "GVD"), live
        if os.environ.get("LOCALAPPDATA") or os.environ.get("USERPROFILE") or os.environ.get("HOME"):
            assert "BeamNG.tech" in live.parts, live


def check_python_known_folder() -> None:
    """Alias kept for test_m6_retail.check_gvd_docs_dir."""
    check_python_tech_sandbox()


def check_lua_harness() -> None:
    for exe in ("lua5.1", "luajit", "lua"):
        if shutil.which(exe):
            res = subprocess.run(
                [exe, "scripts/test_gvd_docs_dir.lua"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert res.returncode == 0 and "test_gvd_docs_dir: OK" in res.stdout, (
                res.stdout + res.stderr
            )
            link = subprocess.run(
                [exe, "scripts/test_gvd_state_link.lua"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert link.returncode == 0 and "test_gvd_state_link: OK" in link.stdout, (
                link.stdout + link.stderr
            )
            print(f"  lua harness via {exe}: OK")
            return
    print("  (lua interpreter not found — scripts/test_gvd_docs_dir.lua skipped)")


def main() -> None:
    lua = LUA.read_text(encoding="utf-8")
    check_source_contracts(lua)
    check_python_mirror()
    check_python_tech_sandbox()
    check_lua_harness()
    print("test_gvd_docs_dir: OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"test_gvd_docs_dir: FAIL - {e}")
        sys.exit(1)
