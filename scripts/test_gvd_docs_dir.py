#!/usr/bin/env python3
"""Offline checks for Lua gvdDocsDir / Python dual-path GVD (no BeamNG).

Lua gvdDocsDir + Python gvd_docs_dir (same product sandbox; #39 USERPROFILE Documents is the FAIL)
------------------------------------------------------------------------------------------------
1. env ``GVD_DOCS_DIR`` if set (full GVD root)
2. Lua: FS:getUserPath / virtual2Native / getFileRealPath → running product
   ``current\\Documents\\GVD`` (Tech vs Drive). Steam does not inherit env.
3. else product sandbox under ``%LOCALAPPDATA%`` (or synthesized
   ``{USERPROFILE|HOME}/AppData/Local`` — never USERPROFILE/Documents):
     - ``GVD_PRODUCT=tech`` / ``GVD_BEAMNG=1`` / ``GVD_BACKEND=beamngpy`` → Tech
     - else Drive / retail
4. else last resort ``Documents/GVD`` (never a bare ``gvd_*.json`` under
   BeamNG userfolder ``current\\``).

Not USERPROFILE\\Documents. Not OneDrive. No junctions.
Drive FS must not reconstruct the Tech tail.

Lua ``gvd_state`` lastGood / gvdUi (offline ≠ live Apps LINK)
-----------------------------------------------------------
``readText`` uses absolute ``io.open`` **before** VFS ``FS:readFile``. ``pollStateFile``
sets ``lastGood`` and logs read ok / read fail / json fail distinctly. ``pushUiState``
and ``onExtensionLoaded`` poll + push ``gvdUi``. Live CEF LINK stays UNPROVEN.

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
    local_fn = _fn(lua, "_localAppFromPath")
    tech_fn = _fn(lua, "_techCurrentFromPath")
    try_env = _fn(lua, "_tryEnvDocs")
    docs_fn = _fn(lua, "gvdDocsDir")
    file_fn = _fn(lua, "gvdFile")
    link_fn = _fn(lua, "linkState")

    assert "AppData/Local" in local_fn
    assert "BeamNG/BeamNG.tech/current" in tech_fn
    assert TECH_TAIL in lua
    assert DRIVE_TAIL in lua
    assert "_driveCurrentFromPath" in docs_fn
    assert "beamng.drive" in docs_fn.lower()

    try_exec = re.sub(r"--[^\n]*", "", try_env)
    ov = try_exec.find("GVD_DOCS_DIR")
    assert ov >= 0, "GVD_DOCS_DIR override"
    assert "LOCALAPPDATA" not in try_exec, "override must not append Tech tail from LOCALAPPDATA"
    assert not re.search(r"USERPROFILE.+/Documents/GVD", try_exec)

    docs_exec = re.sub(r"--[^\n]*", "", docs_fn)
    assert "/Documents/GVD" in docs_exec
    assert "directoryCreate" in docs_exec, "mkdir of resolved docs dir"
    assert "gvdDocsLogged" in docs_fn and "docs dir=" in docs_fn, "one-shot log"
    assert "dir = 'Documents/GVD'" in docs_exec, "last resort is Documents/GVD, never CWD or current\\"
    assert "_tryEnvDocs" in docs_exec and "_tryFsDocs" in docs_exec
    assert "_tryEnvProductDocs" in docs_exec
    assert "dir = _tryEnvDocs() or _tryFsDocs() or _tryEnvProductDocs()" in docs_exec, (
        "override → FS product → env LOCALAPPDATA"
    )

    file_exec = re.sub(r"--[^\n]*", "", file_fn)
    assert "gvdDocsDir() .. '/' .. name" in file_exec
    assert not re.search(r"return\s+name\b", file_exec)

    # lastGood / gvdUi = gvd_state heartbeat (hbAge). Not gvd_ego.json. Live LINK UNPROVEN.
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

    play = (ROOT / "play_gvd.bat").read_text(encoding="utf-8", errors="ignore")
    tech = (ROOT / "play_gvd_tech.bat").read_text(encoding="utf-8", errors="ignore")
    install = (ROOT / "install.bat").read_text(encoding="utf-8", errors="ignore")
    assert DRIVE_BAT in play and 'set "GVD_DOCS_DIR=%LOCALAPPDATA%\\BeamNG\\BeamNG.drive\\current\\Documents\\GVD"' in play
    assert "echo [GVD] bus:" in play
    assert 'set "GVD_DOCS_DIR=%LOCALAPPDATA%\\BeamNG\\BeamNG.tech\\current\\Documents\\GVD"' not in play
    assert TECH_BAT in tech and 'set "GVD_DOCS_DIR=%LOCALAPPDATA%\\BeamNG\\BeamNG.tech\\current\\Documents\\GVD"' in tech
    assert 'set "DOCS=%LOCALAPPDATA%\\BeamNG\\BeamNG.drive\\current\\Documents\\GVD"' in install
    assert 'set "DOCS=%LOCALAPPDATA%\\BeamNG\\BeamNG.tech\\current\\Documents\\GVD"' not in install
    for bat_name, bat in (("install.bat", install), ("play_gvd.bat", play), ("play_gvd_tech.bat", tech)):
        assert "GVD_DOCS_DIR" in bat, bat_name
        assert 'set "GVD_DOCS_DIR=%USERPROFILE%\\Documents\\GVD"' not in bat, bat_name
        assert 'set "DOCS=%USERPROFILE%\\Documents\\GVD"' not in bat, bat_name
        assert "OneDrive" not in bat and "FOLDERID" not in bat and "mklink" not in bat.lower()

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    schema = (ROOT / "docs" / "gvd_state_schema.md").read_text(encoding="utf-8")
    assert r"%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\Documents\GVD" in readme
    assert r"%LOCALAPPDATA%\BeamNG\BeamNG.tech\current\Documents\GVD" in readme
    assert "also copied to %USERPROFILE%\\Documents\\GVD" not in readme
    assert r"%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\Documents\GVD" in schema
    assert r"%LOCALAPPDATA%\BeamNG\BeamNG.tech\current\Documents\GVD" in schema
    assert "Path: `%USERPROFILE%\\Documents\\GVD" not in schema
    assert "Lua and Python share `%USERPROFILE%" not in schema


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

    # Tech mkdir when GVD_BEAMNG=1.
    with tempfile.TemporaryDirectory() as td:
        la = Path(td) / "AppData" / "Local"
        with _clear_product(LOCALAPPDATA=str(la), USERPROFILE=str(Path(td)), HOME=None, GVD_BEAMNG="1"):
            root = gvd_docs_dir()
            expected = la.joinpath(*paths.TECH_GVD_REL)
            assert root == expected and root.is_dir()
            assert "BeamNG.tech" in root.parts
            assert not (Path(td) / "Documents" / "GVD").exists()

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
    check_lua_harness()
    print("test_gvd_docs_dir: OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"test_gvd_docs_dir: FAIL - {e}")
        sys.exit(1)
