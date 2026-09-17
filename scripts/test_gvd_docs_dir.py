#!/usr/bin/env python3
"""Offline checks for Lua relative Documents/GVD reads / Python dual-path writes.

Python gvd_docs_dir (writers; dual-path #40)
------------------------------------------
1. env ``GVD_DOCS_DIR`` if set (full GVD root)
2. product sandbox under ``%LOCALAPPDATA%`` (or synthesized
   ``{USERPROFILE|HOME}/AppData/Local`` — never USERPROFILE/Documents):
     - ``GVD_PRODUCT=tech`` / ``GVD_BEAMNG=1`` / ``GVD_BACKEND=beamngpy`` → Tech
       ``%LOCALAPPDATA%\\BeamNG\\BeamNG.tech\\current\\Documents\\GVD``
     - else Drive / retail
       ``%LOCALAPPDATA%\\BeamNG\\BeamNG.drive\\current\\Documents\\GVD``
3. else last resort ``Documents/GVD`` (never a bare ``gvd_*.json`` under
   BeamNG userfolder ``current\\``).

Lua gvdDocsDir / readText (readers)
-----------------------------------
Relative ``Documents/GVD`` via VFS ``readFile`` / ``FS:readFile`` only.
No absolute ``io.open``. Absolute ``GVD_DOCS_DIR`` / LOCALAPPDATA Tech tails are
Python write roots — Tech GELua cannot ``io.open`` them. Relative ``GVD_DOCS_DIR``
still wins. Never USERPROFILE\\Documents. Never OneDrive. No junctions.

Lua ``gvd_state`` lastGood / gvdUi (offline ≠ live Apps LINK)
-----------------------------------------------------------
``readText`` uses relative ``Documents/GVD`` via VFS / ``FS:readFile`` only.
``pollStateFile`` sets ``lastGood`` and logs read ok / read fail / json fail
distinctly. ``pushUiState`` and ``onExtensionLoaded`` poll + push ``gvdUi``.
Live CEF LINK stays UNPROVEN.

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
DRIVE_TAIL = "BeamNG/BeamNG.drive/current/Documents/GVD"
TECH_BAT = r"%LOCALAPPDATA%\BeamNG\BeamNG.tech\current\Documents\GVD"
DRIVE_BAT = r"%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\Documents\GVD"
PRODUCT_ENV = ("GVD_DOCS_DIR", "GVD_PRODUCT", "GVD_BEAMNG", "GVD_BACKEND")


def _fn(src: str, name: str) -> str:
    m = re.search(rf"local function {re.escape(name)}\s*\(.*?\nend\n", src, re.S)
    assert m, f"missing lua function {name}"
    return m.group(0)


def local_app_from_path(p: str | None) -> str | None:
    """Python mirror of Lua ``_localAppFromPath``."""
    if not p:
        return None
    p = str(p).replace("\\", "/")
    if "onedrive" in p.lower():
        return None
    m = re.match(r"^(.+/AppData/Local)", p)
    if m and m.group(1):
        return m.group(1)
    return None


def tech_current_from_path(p: str | None) -> str | None:
    """Python mirror of Lua ``_techCurrentFromPath``."""
    if not p:
        return None
    p = str(p).replace("\\", "/")
    if "onedrive" in p.lower():
        return None
    m = re.match(r"^(.+/BeamNG/BeamNG.tech/current)", p)
    if m and m.group(1):
        return m.group(1)
    return None


def drive_current_from_path(p: str | None) -> str | None:
    """Python mirror of Lua ``_driveCurrentFromPath``."""
    if not p:
        return None
    p = str(p).replace("\\", "/")
    if "onedrive" in p.lower():
        return None
    m = re.match(r"^(.+/BeamNG/BeamNG.drive/current)", p)
    if m and m.group(1):
        return m.group(1)
    return None


def product_from_path(p: str | None) -> str | None:
    if not p:
        return None
    s = str(p).replace("\\", "/").lower()
    if "onedrive" in s:
        return None
    if "beamng.tech" in s:
        return "tech"
    if "beamng.drive" in s:
        return "drive"
    return None


def from_la(local_app: str | None, product: str) -> str | None:
    if not local_app or not str(local_app).strip():
        return None
    la = str(local_app).replace("\\", "/").strip()
    if not la or "onedrive" in la.lower():
        return None
    tail = TECH_TAIL if product == "tech" else DRIVE_TAIL
    return la + "/" + tail


def env_product(env: dict[str, str | None]) -> str:
    raw = env.get("GVD_PRODUCT")
    if raw and str(raw).strip():
        p = str(raw).strip().lower().replace(" ", "")
        if p in ("tech", "beamng.tech", "beamngtech"):
            return "tech"
        if p in ("drive", "retail", "beamng.drive", "beamngdrive"):
            return "drive"
    beamng = (env.get("GVD_BEAMNG") or "").strip().lower()
    if beamng in ("1", "true", "yes"):
        return "tech"
    backend = (env.get("GVD_BACKEND") or "").strip().lower()
    if backend in ("beamngpy", "tech"):
        return "tech"
    return "drive"


def resolve_docs_dir(
    env: dict[str, str | None],
    fs_paths: list[str] | None = None,
) -> str:
    """Python mirror of gvdDocsDir: override → FS product → env product sandbox."""
    ov = env.get("GVD_DOCS_DIR")
    if ov and str(ov).strip():
        return str(ov).strip().replace("\\", "/")
    home = env.get("USERPROFILE") or env.get("HOME")
    home_la = (str(home).replace("\\", "/") + "/AppData/Local") if home and str(home).strip() else None
    for p in fs_paths or []:
        cur = tech_current_from_path(p)
        if cur:
            return cur + "/Documents/GVD"
        cur = drive_current_from_path(p)
        if cur:
            return cur + "/Documents/GVD"
        product = product_from_path(p)
        if not product:
            continue
        got = from_la(local_app_from_path(p), product)
        if got:
            return got
        got = from_la(env.get("LOCALAPPDATA"), product) or from_la(home_la, product)
        if got:
            return got
    product = env_product(env)
    got = from_la(env.get("LOCALAPPDATA"), product) or from_la(home_la, product)
    if got:
        return got
    return "Documents/GVD"


def check_source_contracts(lua: str) -> None:
    try_env = _fn(lua, "_tryEnvDocs")
    docs_fn = _fn(lua, "gvdDocsDir")
    file_fn = _fn(lua, "gvdFile")
    link_fn = _fn(lua, "linkState")
    bus_fn = _fn(lua, "_gvdBusRel")
    vfs_fn = _fn(lua, "_vfsRead")

    assert TECH_TAIL in lua, "Lua comments still name the Python Tech write tail"
    assert DRIVE_TAIL in lua, "Lua comments still name the Python Drive write tail"
    assert "_tryFsDocs" not in lua
    assert "_tryEnvProductDocs" not in lua
    assert "_localAppFromPath" not in lua
    assert "_techCurrentFromPath" not in lua
    assert "_driveCurrentFromPath" not in lua

    try_exec = re.sub(r"--[^\n]*", "", try_env)
    ov = try_exec.find("GVD_DOCS_DIR")
    assert ov >= 0, "GVD_DOCS_DIR override"
    assert "_isAbsDiskPath" in try_exec, "absolute override is not a Lua read path"
    assert "LOCALAPPDATA" not in try_exec, "override must not append Tech tail from LOCALAPPDATA"
    assert not re.search(r"USERPROFILE.+/Documents/GVD", try_exec)

    docs_exec = re.sub(r"--[^\n]*", "", docs_fn)
    assert "/Documents/GVD" in docs_exec or "Documents/GVD" in docs_exec
    assert "directoryCreate" in docs_exec, "mkdir of resolved docs dir"
    assert "gvdDocsLogged" in docs_fn and "docs dir=" in docs_fn, "one-shot log"
    assert "dir = 'Documents/GVD'" in docs_exec, "Lua bus is Documents/GVD, never CWD or current\\"
    assert "_tryEnvDocs" in docs_exec
    assert "_tryFsDocs" not in docs_exec
    assert "_tryEnvProductDocs" not in docs_exec
    assert "dir = _tryEnvDocs()" in docs_exec
    assert "getUserPath" not in docs_exec, "gvdDocsDir must not io/read via getUserPath"

    lua_bus_fn = _fn(lua, "luaBusPath")
    same_fn = _fn(lua, "busesSame")
    lua_bus_exec = re.sub(r"--[^\n]*", "", lua_bus_fn)
    same_exec = re.sub(r"--[^\n]*", "", same_fn)
    assert "getUserPath" in lua_bus_exec
    assert "Documents/GVD" in lua_bus_exec
    assert "io.open" not in lua_bus_exec
    assert "onedrive" in same_exec
    write_fn = _fn(lua, "writeText")
    write_exec = re.sub(r"--[^\n]*", "", write_fn)
    assert "io.open" not in write_exec, "bus writes must not io.open (leftover current\\ gvd_*.json)"
    assert "FS:writeFile" in write_exec
    assert "tickBusIdentity" in lua
    assert "python_bus=" in lua and "lua_bus=" in lua and "product=" in lua
    assert "state_mtime=" in lua and "engage=" in lua and "seq=" in lua
    assert "link=MISMATCH" in lua

    file_exec = re.sub(r"--[^\n]*", "", file_fn)
    assert "gvdDocsDir() .. '/' .. name" in file_exec
    assert not re.search(r"return\s+name\b", file_exec)

    # lastGood / gvdUi = gvd_state heartbeat (hbAge). bus folder mismatch is louder than stale.
    link_exec = re.sub(r"--[^\n]*", "", link_fn)
    assert "lastGood" in link_exec and "hbAgeS" in link_exec
    assert "mismatch" in link_exec and "busMismatch" in lua
    assert "gvd_ego" not in link_exec and "egoFb" not in link_exec, link_fn
    assert "userEgoPath" not in link_exec

    helpers = try_env + docs_fn
    assert "Accounts" not in helpers and "UserFolder" not in helpers
    assert "SHGetKnownFolderPath" not in helpers
    assert "FOLDERID" not in helpers
    assert "mklink" not in lua.lower()

    read_fn = _fn(lua, "readText")
    read_exec = re.sub(r"--[^\n]*", "", read_fn)
    bus_exec = re.sub(r"--[^\n]*", "", bus_fn)
    vfs_exec = re.sub(r"--[^\n]*", "", vfs_fn)
    assert "_ioOpenRead" not in lua
    assert "_ioReadAll" not in lua
    assert "io.open" not in read_exec, "bus reads must not io.open"
    assert "io.open" not in bus_exec
    assert "io.open" not in vfs_exec
    assert "Documents/GVD" in bus_exec
    assert "readFile" in vfs_exec and "FS:readFile" in vfs_exec
    assert "_gvdBusRel" in read_exec and "_vfsRead" in read_exec
    assert 0 <= read_exec.find("_gvdBusRel") < read_exec.find("_vfsRead"), "relative Documents/GVD before VFS read"
    assert "FS:readFile" in vfs_exec
    poll_fn = _fn(lua, "pollStateFile")
    poll_exec = re.sub(r"--[^\n]*", "", poll_fn)
    assert "lastGood = st" in poll_fn
    assert "gvd_state read ok path=" in poll_fn
    assert "read fail path=" in poll_fn
    assert "json fail len=" in poll_fn
    assert "pushUi()" in poll_fn, "first lastGood must push gvdUi"
    assert "STATE_REL" in poll_exec, "poll reads relative Documents/GVD/gvd_state.json"
    assert "io.open" not in poll_exec
    assert "readText(STATE_REL)" in lua
    assert "readText(CMD_REL)" in lua
    assert "readText(ENGAGE_REL)" in lua
    assert "readText(UI_PREFS_REL)" in lua
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

    play = (ROOT / "play_gvd.bat").read_text(encoding="utf-8", errors="ignore")
    tech = (ROOT / "play_gvd_tech.bat").read_text(encoding="utf-8", errors="ignore")
    install = (ROOT / "install.bat").read_text(encoding="utf-8", errors="ignore")
    assert DRIVE_BAT in play and 'set "GVD_DOCS_DIR=%LOCALAPPDATA%\\BeamNG\\BeamNG.drive\\current\\Documents\\GVD"' in play
    assert "echo [GVD] bus:" in play
    assert "if not defined GVD_DOCS_DIR" not in play, "play_gvd.bat must pin Drive bus (no leftover Tech override)"
    assert 'set "GVD_PRODUCT=drive"' in play
    assert 'set "GVD_DOCS_DIR=%LOCALAPPDATA%\\BeamNG\\BeamNG.tech\\current\\Documents\\GVD"' not in play
    assert TECH_BAT in tech and 'set "GVD_DOCS_DIR=%LOCALAPPDATA%\\BeamNG\\BeamNG.tech\\current\\Documents\\GVD"' in tech
    assert "if not defined GVD_DOCS_DIR" not in tech, "play_gvd_tech.bat must pin Tech bus"
    assert 'set "GVD_PRODUCT=tech"' in tech
    assert 'set "DOCS=%LOCALAPPDATA%\\BeamNG\\BeamNG.drive\\current\\Documents\\GVD"' in install
    assert 'set "DOCS=%GVD_DOCS_DIR%"' not in install
    assert 'set "DOCS=%LOCALAPPDATA%\\BeamNG\\BeamNG.tech\\current\\Documents\\GVD"' not in install
    assert "gvdApp" in install
    assert "gvd-app" in install, "install must call out stale kebab tiles"
    assert "unpacked\\gvd" in install
    for bat_name, bat in (("install.bat", install), ("play_gvd.bat", play), ("play_gvd_tech.bat", tech)):
        assert 'set "GVD_DOCS_DIR=%USERPROFILE%\\Documents\\GVD"' not in bat, bat_name
        assert 'set "DOCS=%USERPROFILE%\\Documents\\GVD"' not in bat, bat_name
        assert "OneDrive" not in bat and "FOLDERID" not in bat and "mklink" not in bat.lower()
    assert "GVD_DOCS_DIR" in play and "GVD_DOCS_DIR" in tech

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    schema = (ROOT / "docs" / "gvd_state_schema.md").read_text(encoding="utf-8")
    assert r"%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\Documents\GVD" in readme
    assert r"%LOCALAPPDATA%\BeamNG\BeamNG.tech\current\Documents\GVD" in readme
    assert "also copied to %USERPROFILE%\\Documents\\GVD" not in readme
    assert r"%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\Documents\GVD" in schema
    assert r"%LOCALAPPDATA%\BeamNG\BeamNG.tech\current\Documents\GVD" in schema
    assert "Path: `%USERPROFILE%\\Documents\\GVD" not in schema
    assert "Lua and Python share `%USERPROFILE%" not in schema
    assert "FS:readFile" in readme
    assert "FS:readFile" in schema
    assert "io.open" in readme and "io.open" in schema


def check_python_mirror() -> None:
    tech = "C:/Users/Name/AppData/Local/" + TECH_TAIL
    drive = "C:/Users/Name/AppData/Local/" + DRIVE_TAIL
    assert local_app_from_path(r"C:\Users\Name\AppData\Local\BeamNG\BeamNG.tech\current") == (
        "C:/Users/Name/AppData/Local"
    )
    assert (
        tech_current_from_path("C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current")
        == "C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current"
    )
    assert (
        drive_current_from_path("C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current")
        == "C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current"
    )
    assert tech_current_from_path("C:/Users/Name/AppData/Local/BeamNG.tech/current") is None
    assert local_app_from_path("C:/Users/Name/Documents/BeamNG.drive/current") is None
    assert local_app_from_path("D:/BeamNG.drive/current") is None
    assert product_from_path("C:/Users/Name/AppData/Local/BeamNG.drive/0.36") == "drive"
    assert product_from_path("C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current") == "tech"

    la = {"LOCALAPPDATA": r"C:\Users\Name\AppData\Local", "USERPROFILE": r"C:\Users\Name"}
    assert (
        resolve_docs_dir(la, ["C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current"])
        == drive
    )
    assert (
        resolve_docs_dir(la, ["C:/Users/Name/Documents/BeamNG.drive/0.36/current"])
        == drive
    )
    assert resolve_docs_dir({"USERPROFILE": r"C:\Users\Name"}, []) == drive
    assert resolve_docs_dir({"HOME": "/home/me"}, []) == "/home/me/AppData/Local/" + DRIVE_TAIL
    assert (
        resolve_docs_dir({}, ["C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current"])
        == tech
    )
    assert (
        resolve_docs_dir(la, ["C:/Users/Name/AppData/Local/BeamNG.drive/0.36"])
        == drive
    ), "Drive FS must not reconstruct Tech tail"
    assert (
        resolve_docs_dir({"LOCALAPPDATA": r"C:\Users\Name\AppData\Local", "GVD_BEAMNG": "1"}, [])
        == tech
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
        assert DRIVE_TAIL in p
        assert TECH_TAIL not in p
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


def _clear_product(**extra: str | None):
    kwargs: dict[str, str | None] = {k: None for k in PRODUCT_ENV}
    kwargs.update(extra)
    return _env(**kwargs)


def check_python_product_sandbox() -> None:
    """Python GVD bus: Drive vs Tech current\\Documents\\GVD; override wins; never USERPROFILE Documents."""
    from python.control.actuate import cmd_path, ego_path, engage_path
    from python.runtime import paths
    from python.runtime.state_io import gvd_docs_dir, state_path

    src = (ROOT / "python" / "runtime" / "paths.py").read_text(encoding="utf-8")
    state_src = (ROOT / "python" / "runtime" / "state_io.py").read_text(encoding="utf-8")
    assert "GVD_DOCS_DIR" in src
    assert "BeamNG.tech" in src and "BeamNG.drive" in src and "current" in src
    assert "TECH_GVD_REL" in src and "DRIVE_GVD_REL" in src
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

    tech = "C:/Users/Name/AppData/Local/" + TECH_TAIL
    drive = "C:/Users/Name/AppData/Local/" + DRIVE_TAIL
    lua_drive = resolve_docs_dir({"LOCALAPPDATA": r"C:\Users\Name\AppData\Local"})
    lua_tech = resolve_docs_dir({"LOCALAPPDATA": r"C:\Users\Name\AppData\Local", "GVD_BEAMNG": "1"})
    assert lua_drive == drive, lua_drive
    assert lua_tech == tech, lua_tech

    assert paths.is_onedrive_path(Path("C:/Users/Name/OneDrive/Documents"))
    assert paths.is_onedrive_path(r"C:\Users\Name\OneDrive - Personal\Documents")
    assert paths.is_onedrive_path("C:/Users/Name/OneDrive - Contoso/Documents")
    assert not paths.is_onedrive_path(Path("C:/Users/Name/AppData/Local"))
    assert not paths.is_onedrive_path(None)
    assert not paths.is_onedrive_path("  ")

    # Retail default: LOCALAPPDATA Drive sandbox, never USERPROFILE Documents, never Tech.
    with _clear_product(
        LOCALAPPDATA="C:/Users/Name/AppData/Local",
        USERPROFILE="C:/Users/Name",
        HOME="/home/other",
    ):
        got = paths.gvd_docs_path()
        assert _slash(got) == drive, got
        assert got.parts[-2:] == ("Documents", "GVD")
        assert "BeamNG.drive" in got.parts
        assert "BeamNG.tech" not in got.parts
        assert "OneDrive" not in _slash(got)
        assert _slash(got) == lua_drive
        assert "Users/Name/Documents/GVD" not in _slash(got)
        assert paths.gvd_product() == "drive"

    # Tech env: GVD_BEAMNG=1 / GVD_BACKEND=beamngpy / GVD_PRODUCT=tech.
    with _clear_product(
        LOCALAPPDATA="C:/Users/Name/AppData/Local",
        USERPROFILE="C:/Users/Name",
        GVD_BEAMNG="1",
    ):
        got = paths.gvd_docs_path()
        assert _slash(got) == tech, got
        assert "BeamNG.tech" in got.parts
        assert paths.gvd_product() == "tech"
    with _clear_product(
        LOCALAPPDATA="C:/Users/Name/AppData/Local",
        GVD_BACKEND="beamngpy",
    ):
        assert _slash(paths.gvd_docs_path()) == tech
    with _clear_product(
        LOCALAPPDATA="C:/Users/Name/AppData/Local",
        GVD_PRODUCT="tech",
    ):
        assert _slash(paths.gvd_docs_path()) == tech

    # USERPROFILE alone synthesizes AppData\\Local Drive sandbox (retail default).
    with _clear_product(LOCALAPPDATA=None, USERPROFILE="C:/Users/Name", HOME=None):
        got = paths.gvd_docs_path()
        assert _slash(got) == drive, got
        assert _slash(got) == resolve_docs_dir({"USERPROFILE": r"C:\Users\Name"})

    # OneDrive LOCALAPPDATA is rejected; fall through to USERPROFILE\\AppData\\Local Drive.
    with _clear_product(
        LOCALAPPDATA="C:/Users/Name/OneDrive",
        USERPROFILE="C:/Users/Name",
    ):
        got = paths.gvd_docs_path()
        assert "OneDrive" not in _slash(got), got
        assert _slash(got) == drive

    # GVD_DOCS_DIR override wins over both sandboxes and USERPROFILE.
    with tempfile.TemporaryDirectory() as td:
        override = Path(td) / "custom_gvd"
        with _clear_product(
            GVD_DOCS_DIR=str(override),
            USERPROFILE="C:/Users/Name",
            LOCALAPPDATA="C:/Users/Name/AppData/Local",
            GVD_BEAMNG="1",
        ):
            got = paths.gvd_docs_path()
            assert Path(got) == override, got
            root = gvd_docs_dir()
            assert root == override and root.is_dir()
            assert state_path().parent == root
            assert cmd_path().parent == engage_path().parent == ego_path().parent == root
            ident = paths.bus_identity()
            assert ident.matched is False, "Python-only override must not pretend Lua shares that folder"
            assert ident.link == "MISMATCH"
            assert not paths.buses_same_folder(ident.python_bus, ident.lua_bus)

    # Blank GVD_DOCS_DIR is ignored (fall through to Drive sandbox).
    with _clear_product(
        GVD_DOCS_DIR="   ",
        LOCALAPPDATA="C:/Users/Name/AppData/Local",
        USERPROFILE="C:/Users/Name",
    ):
        assert _slash(paths.gvd_docs_path()) == drive

    # Last resort relative Documents/GVD (Lua last-resort contract).
    with _clear_product(LOCALAPPDATA=None, USERPROFILE=None, HOME=None):
        last = paths.gvd_docs_path()
        assert _slash(last) == "Documents/GVD"
        assert not str(last).endswith(".json")

    # Live mkdir: Drive sandbox under LOCALAPPDATA, never USERPROFILE\\Documents or OneDrive.
    with tempfile.TemporaryDirectory() as td:
        profile = Path(td) / "Users" / "Name"
        la = profile / "AppData" / "Local"
        onedrive = profile / "OneDrive" / "Documents"
        onedrive.mkdir(parents=True)
        local_docs = profile / "Documents"
        local_docs.mkdir(parents=True)
        with _clear_product(
            LOCALAPPDATA=str(la),
            USERPROFILE=str(profile),
            HOME=None,
        ):
            root = gvd_docs_dir()
            expected = la.joinpath(*paths.DRIVE_GVD_REL)
            assert root == expected and root.is_dir()
            assert "OneDrive" not in _slash(root)
            assert not (onedrive / "GVD").exists()
            assert not (local_docs / "GVD").exists()
            assert state_path().parent == root
            assert cmd_path().parent == engage_path().parent == ego_path().parent == root
            assert root.parts[-2:] == ("Documents", "GVD")
            assert "BeamNG.drive" in root.parts
            assert "BeamNG.tech" not in root.parts
            ident = paths.bus_identity()
            assert ident.matched is True
            assert ident.product == "drive"
            assert paths.buses_same_folder(ident.python_bus, ident.lua_bus)
            assert ident.python_bus == expected

    # Tech mkdir when GVD_BEAMNG=1.
    with tempfile.TemporaryDirectory() as td:
        la = Path(td) / "AppData" / "Local"
        with _clear_product(LOCALAPPDATA=str(la), USERPROFILE=str(Path(td)), HOME=None, GVD_BEAMNG="1"):
            root = gvd_docs_dir()
            expected = la.joinpath(*paths.TECH_GVD_REL)
            assert root == expected and root.is_dir()
            assert "BeamNG.tech" in root.parts
            assert not (Path(td) / "Documents" / "GVD").exists()
            ident = paths.bus_identity()
            assert ident.matched is True and ident.product == "tech"

    # USERPROFILE-only mkdir still lands on AppData\\Local Drive sandbox.
    with tempfile.TemporaryDirectory() as td:
        profile = Path(td) / "Users" / "Name"
        with _clear_product(LOCALAPPDATA=None, USERPROFILE=str(profile), HOME=None):
            root = gvd_docs_dir()
            expected = profile.joinpath("AppData", "Local", *paths.DRIVE_GVD_REL)
            assert root == expected and root.is_dir()
            assert not (profile / "Documents" / "GVD").exists()

    # Host fallback: product tail when LOCALAPPDATA/HOME can be synthesized.
    with _clear_product():
        live = paths.gvd_docs_path()
        assert live.parts[-2:] == ("Documents", "GVD"), live
        if os.environ.get("LOCALAPPDATA") or os.environ.get("USERPROFILE") or os.environ.get("HOME"):
            assert "BeamNG.drive" in live.parts or "BeamNG.tech" in live.parts, live
            assert "OneDrive" not in _slash(live)


def check_python_tech_sandbox() -> None:
    """Alias kept for older imports."""
    check_python_product_sandbox()


def check_python_known_folder() -> None:
    """Alias kept for test_m6_retail.check_gvd_docs_dir."""
    check_python_product_sandbox()


def lua_read_helper(path: str) -> str | None:
    """Mirror of Lua ``_gvdBusRel``: the folder GELua actually reads."""
    p = str(path).replace("\\", "/")
    name = p.rsplit("/", 1)[-1]
    if name.startswith("gvd_"):
        return "Documents/GVD/" + name
    if p == "Documents/GVD" or p.startswith("Documents/GVD/"):
        return p
    return None


def check_writer_lua_folder_agreement() -> None:
    """Fails if Python writers and the Lua read helper disagree on the bus folder."""
    from python.runtime import paths

    lua = LUA.read_text(encoding="utf-8")
    bus_fn = _fn(lua, "_gvdBusRel")
    bus_exec = re.sub(r"--[^\n]*", "", bus_fn)
    assert "Documents/GVD/" in bus_exec
    assert "return 'Documents/GVD/' .. name" in bus_exec or 'return "Documents/GVD/" .. name' in bus_exec
    assert not re.search(r"return\s+name\b", bus_exec), "Lua must not read a bare gvd_*.json under current\\"
    assert "/current/" not in bus_exec.replace("Documents/GVD", "")

    rel = lua_read_helper("gvd_state.json")
    assert rel == "Documents/GVD/gvd_state.json", rel
    leftover = lua_read_helper("current/gvd_state.json")
    assert leftover == "Documents/GVD/gvd_state.json", leftover
    assert lua_read_helper("Documents/GVD/gvd_cmd.json") == "Documents/GVD/gvd_cmd.json"

    drive = "C:/Users/Name/AppData/Local/" + DRIVE_TAIL
    tech = "C:/Users/Name/AppData/Local/" + TECH_TAIL
    assert paths.buses_same_folder(drive, drive.replace("/", "\\"))
    assert not paths.buses_same_folder(drive, tech), "Drive vs Tech must fail the agreement check"
    assert not paths.buses_same_folder(drive, "Documents/GVD"), "do not guess relative == absolute"
    assert not paths.buses_same_folder(drive, "C:/Users/Name/Documents/GVD")
    assert not paths.buses_same_folder(drive, "C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current")

    with _clear_product(
        LOCALAPPDATA="C:/Users/Name/AppData/Local",
        USERPROFILE="C:/Users/Name",
        HOME=None,
    ):
        py = paths.gvd_docs_path()
        lua_bus = paths.resolved_lua_bus("drive")
        ident = paths.bus_identity()
        writer_state = (py / "gvd_state.json").as_posix()
        reader_state = lua_read_helper("gvd_state.json")
        assert writer_state.endswith("/" + reader_state) or writer_state.endswith(reader_state), (
            writer_state,
            reader_state,
        )
        assert "BeamNG.drive/current/Documents/GVD" in writer_state.replace("\\", "/")
        assert ident.matched and paths.buses_same_folder(py, lua_bus), (py, lua_bus)
        assert ident.product == "drive"
        assert ident.lua_bus_s.replace("\\", "/").endswith("/Documents/GVD") or ident.lua_bus_s.replace("\\", "/") == "Documents/GVD"

    with _clear_product(
        LOCALAPPDATA="C:/Users/Name/AppData/Local",
        GVD_BEAMNG="1",
    ):
        py = paths.gvd_docs_path()
        assert "BeamNG.tech/current/Documents/GVD" in str(py).replace("\\", "/")
        assert paths.bus_identity().product == "tech"
        assert paths.bus_identity().matched

    with _clear_product(
        LOCALAPPDATA="C:/Users/Name/AppData/Local",
        GVD_DOCS_DIR="C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current/Documents/GVD",
        GVD_PRODUCT="drive",
    ):
        ident = paths.bus_identity()
        assert ident.product == "drive"
        assert ident.matched is False, "Tech write root while Drive product is a bus mismatch"
        assert ident.link == "MISMATCH"

    # Leftover gvd_state.json under current\\ while Documents/GVD is empty → MISMATCH.
    with tempfile.TemporaryDirectory() as td:
        la = Path(td) / "AppData" / "Local"
        current = la.joinpath("BeamNG", "BeamNG.drive", "current")
        current.mkdir(parents=True)
        (current / "gvd_state.json").write_text("{}", encoding="utf-8")
        with _clear_product(LOCALAPPDATA=str(la), USERPROFILE=str(Path(td)), HOME=None):
            ident = paths.bus_identity()
            assert ident.leftover
            assert ident.matched is False
            assert "leftover" in ident.note

    # Other product tree has the file, ours does not → MISMATCH.
    with tempfile.TemporaryDirectory() as td:
        la = Path(td) / "AppData" / "Local"
        tech_docs = la.joinpath(*paths.TECH_GVD_REL)
        tech_docs.mkdir(parents=True)
        (tech_docs / "gvd_state.json").write_text("{}", encoding="utf-8")
        with _clear_product(LOCALAPPDATA=str(la), USERPROFILE=str(Path(td)), HOME=None):
            ident = paths.bus_identity()
            assert ident.product == "drive"
            assert ident.other_state is not None
            assert ident.matched is False
            assert "other product" in ident.note

    line = paths.format_bus_identity_line(
        python_bus=drive,
        lua_bus=drive,
        product="drive",
        state_mtime=1,
        engage=False,
        seq=3,
        link="ok",
    )
    assert "python_bus=" in line and "lua_bus=" in line and "product=drive" in line
    assert "state_mtime=1" in line and "engage=false" in line and "seq=3" in line
    bad = paths.format_bus_identity_line(
        python_bus=tech,
        lua_bus=drive,
        product="drive",
        state_mtime=2,
        engage=True,
        seq=9,
        link="MISMATCH",
    )
    assert "link=MISMATCH" in bad and "engage=true" in bad

    from python.runtime.state_io import load_paint_state

    with tempfile.TemporaryDirectory() as td:
        la = Path(td) / "AppData" / "Local"
        with _clear_product(LOCALAPPDATA=str(la), USERPROFILE=str(Path(td)), HOME=None):
            paint = load_paint_state()
            assert paint.get("bus_link") == "MISMATCH"
            assert paint.get("path_ego") == []
            assert paint.get("lanes_ext") == []
            assert paint.get("engaged") is False


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
    check_python_product_sandbox()
    check_writer_lua_folder_agreement()
    check_lua_harness()
    print("test_gvd_docs_dir: OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"test_gvd_docs_dir: FAIL - {e}")
        sys.exit(1)
