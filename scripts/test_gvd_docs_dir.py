#!/usr/bin/env python3
"""Offline checks for Lua gvdDocsDir / Python Documents/GVD (no BeamNG).

Lua gvdDocsDir (unchanged USERPROFILE-first; no Personal-reg OneDrive tip)
-------------------------------------------------------------------------
1. USERPROFILE (preferred) → ``{USERPROFILE}/Documents/GVD``
2. else HOME → ``{HOME}/Documents/GVD``
3. else LOCALAPPDATA stripped to the profile (AppData parent) → ``…/Documents/GVD``
4. else FS:getUserPath / virtual2Native / getFileRealPath, stripped:
   - ``…/AppData/…`` → profile
   - ``…/Documents/…`` (BeamNG userfolder under Documents) → profile
   then ``{profile}/Documents/GVD``
5. else last resort ``Documents/GVD`` (never a bare ``gvd_*.json`` under
   BeamNG userfolder ``current\\``).

Python ``gvd_docs_dir`` (local Documents; #38 OneDrive FOLDERID is the FAIL)
---------------------------------------------------------------------------
1. env ``GVD_DOCS_DIR`` if set (full GVD root)
2. ``%USERPROFILE%/Documents/GVD`` (local, non-redirected)
3. ``SHGetKnownFolderPath(FOLDERID_Documents)`` as a probe only if USERPROFILE is
   missing — reject when the path contains ``OneDrive``
4. else ``expanduser`` / HOME Documents, else relative ``Documents/GVD``

Lua ``gvd_state`` LINKED (not path-only)
---------------------------------------
``readText`` uses absolute ``io.open`` **before** VFS ``FS:readFile``. ``pollStateFile``
sets ``lastGood`` and logs read ok / read fail / json fail distinctly. ``pushUiState``
and ``onExtensionLoaded`` poll + push ``gvdUi`` so CEF is not stuck on
"supervisor not running" when the disk file is fresh.

When FOLDERID points at OneDrive, Python must land on
``%USERPROFILE%\\Documents\\GVD`` — the same USERPROFILE-first folder Lua uses
when USERPROFILE is set. No junction. No OneDrive Personal targeting.

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


def _fn(src: str, name: str) -> str:
    m = re.search(rf"local function {re.escape(name)}\s*\(.*?\nend\n", src, re.S)
    assert m, f"missing lua function {name}"
    return m.group(0)


def strip_to_user_home(p: str | None) -> str | None:
    """Python mirror of Lua ``_stripToUserHome`` (backslash → slash, then parents)."""
    if not p:
        return None
    p = str(p).replace("\\", "/")
    for pat in (r"^(.+)/AppData/Local", r"^(.+)/AppData/Roaming", r"^(.+)/AppData"):
        m = re.match(pat, p)
        if m and m.group(1):
            return m.group(1)
    m = re.match(r"^(.*?)/Documents/", p) or re.match(r"^(.*?)/Documents$", p)
    if m and m.group(1):
        return m.group(1)
    return None


def resolve_docs_dir(
    env: dict[str, str | None],
    fs_paths: list[str] | None = None,
) -> str:
    """Python mirror of gvdDocsDir home pick + ``/Documents/GVD`` join."""
    home = env.get("USERPROFILE") or env.get("HOME")
    if not home:
        la = env.get("LOCALAPPDATA")
        home = strip_to_user_home(la) if la else None
    if not home:
        for p in fs_paths or []:
            home = strip_to_user_home(p)
            if home:
                break
    if home:
        return home.replace("\\", "/") + "/Documents/GVD"
    return "Documents/GVD"


def check_source_contracts(lua: str) -> None:
    strip_fn = _fn(lua, "_stripToUserHome")
    try_home = _fn(lua, "_tryHomeEnv")
    docs_fn = _fn(lua, "gvdDocsDir")
    file_fn = _fn(lua, "gvdFile")
    link_fn = _fn(lua, "linkState")

    assert "/Documents/" in strip_fn, "must strip on the /Documents/ parent"
    assert "^(.-)/Documents/" in strip_fn or "^(.*)/Documents/" in strip_fn, strip_fn
    assert "/AppData/Local" in strip_fn and "/AppData/Roaming" in strip_fn

    # USERPROFILE prefer, then HOME, then LOCALAPPDATA strip.
    try_exec = re.sub(r"--[^\n]*", "", try_home)
    up = try_exec.find("USERPROFILE")
    hm = try_exec.find("HOME")
    la = try_exec.find("LOCALAPPDATA")
    assert 0 <= up < hm < la, "USERPROFILE then HOME then LOCALAPPDATA"

    docs_exec = re.sub(r"--[^\n]*", "", docs_fn)
    assert "/Documents/GVD" in docs_exec
    assert "directoryCreate" in docs_exec, "mkdir of resolved docs dir"
    assert "gvdDocsLogged" in docs_fn and "docs dir=" in docs_fn, "one-shot log"
    assert "dir = 'Documents/GVD'" in docs_exec, "last resort is Documents/GVD, never CWD or current\\"

    file_exec = re.sub(r"--[^\n]*", "", file_fn)
    assert "gvdDocsDir() .. '/' .. name" in file_exec
    assert not re.search(r"return\s+name\b", file_exec)

    # LINKED = gvd_state heartbeat (lastGood / hbAge). Not gvd_ego.json.
    link_exec = re.sub(r"--[^\n]*", "", link_fn)
    assert "lastGood" in link_exec and "hbAgeS" in link_exec
    assert "gvd_ego" not in link_exec and "egoFb" not in link_exec, link_fn
    assert "userEgoPath" not in link_exec

    # No Personal-reg OneDrive tip (HKCU OneDrive Accounts\\Personal UserFolder).
    helpers = strip_fn + try_home + docs_fn
    assert "Accounts" not in helpers and "UserFolder" not in helpers
    assert "SHGetKnownFolderPath" not in helpers
    assert "FOLDERID" not in helpers

    read_fn = _fn(lua, "readText")
    io_fn = _fn(lua, "_ioOpenRead")
    read_exec = re.sub(r"--[^\n]*", "", read_fn)
    io_exec = re.sub(r"--[^\n]*", "", io_fn)
    assert "io.open" in io_exec, "absolute io.open helper"
    assert "_isAbsDiskPath" in read_exec and "_ioReadAll" in read_exec
    assert 0 <= read_exec.find("_isAbsDiskPath") < read_exec.find("FS:readFile"), "io.open before FS:readFile"
    assert read_exec.find("_ioReadAll") < read_exec.find("FS:readFile")
    assert read_exec.find("_isAbsDiskPath") < read_exec.find("if readFile")
    assert "_isAbsDiskPath" in lua and "%a:[/\\" in lua
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
        assert r"%USERPROFILE%\Documents\GVD" in bat, bat_name
        assert "GVD_DOCS_DIR" in bat, bat_name
        assert "OneDrive" not in bat and "FOLDERID" not in bat and "mklink" not in bat.lower()


def check_python_mirror() -> None:
    assert strip_to_user_home(r"C:\Users\Name\AppData\Local\BeamNG.drive") == "C:/Users/Name"
    assert (
        strip_to_user_home("C:/Users/Name/Documents/BeamNG.drive/0.36/current")
        == "C:/Users/Name"
    )
    assert strip_to_user_home("C:/Users/Name/Documents") == "C:/Users/Name"
    assert (
        strip_to_user_home("C:/Users/Name/Documents/BeamNG.drive/current/Documents/GVD")
        == "C:/Users/Name"
    )
    assert strip_to_user_home("D:/BeamNG.drive/current") is None

    # USERPROFILE prefer → …/Documents/GVD even when FS is the Documents userfolder.
    assert (
        resolve_docs_dir(
            {"USERPROFILE": r"C:\Users\Name", "HOME": "/home/other"},
            ["C:/Users/Name/Documents/BeamNG.drive/0.36/current"],
        )
        == "C:/Users/Name/Documents/GVD"
    )
    assert resolve_docs_dir({"HOME": "/home/me"}, []) == "/home/me/Documents/GVD"
    assert (
        resolve_docs_dir({"LOCALAPPDATA": r"C:\Users\Name\AppData\Local"})
        == "C:/Users/Name/Documents/GVD"
    )
    # Live bug: empty GELua env + BeamNG userfolder under Documents.
    assert (
        resolve_docs_dir({}, ["C:/Users/Name/Documents/BeamNG.drive/0.36/current"])
        == "C:/Users/Name/Documents/GVD"
    )
    last = resolve_docs_dir({}, [])
    assert last == "Documents/GVD"
    assert not last.endswith(".json")
    assert last != "current"

    for name in ("gvd_state.json", "gvd_ego.json", "gvd_engage.json", "gvd_cmd.json"):
        p = resolve_docs_dir({"USERPROFILE": "C:/Users/Name"}) + "/" + name
        assert p.endswith("/Documents/GVD/" + name)
        assert not p.startswith("gvd_")
        assert "/current/" + name not in p


def _slash(p: Path | str) -> str:
    return str(p).replace("\\", "/").rstrip("/")


@contextmanager
def _patch_attr(mod, name: str, value):
    orig = getattr(mod, name)
    setattr(mod, name, value)
    try:
        yield
    finally:
        setattr(mod, name, orig)


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


def check_python_known_folder() -> None:
    """Python Documents/GVD: reject OneDrive FOLDERID; USERPROFILE local; override wins."""
    import ctypes

    from python.control.actuate import cmd_path, ego_path, engage_path
    from python.runtime import paths
    from python.runtime.state_io import gvd_docs_dir, state_path

    src = (ROOT / "python" / "runtime" / "paths.py").read_text(encoding="utf-8")
    docs_body = src[src.find("def documents_dir"): src.find("def gvd_docs_path")]
    assert docs_body.find("env_documents_dir") < docs_body.find("windows_known_folder_documents"), (
        "USERPROFILE Documents before Known Folder probe"
    )
    state_src = (ROOT / "python" / "runtime" / "state_io.py").read_text(encoding="utf-8")
    assert "SHGetKnownFolderPath" in src, "Known Folder stays a probe"
    assert "FOLDERID_Documents" in src
    assert "FDD39AD0" in src.upper()
    assert "238F" in src.upper() and "46AF" in src.upper()
    assert "CoTaskMemFree" in src, "Known Folder PWSTR must be freed"
    assert "no pywin32" in src.lower()
    assert "GVD_DOCS_DIR" in src
    assert "is_onedrive_path" in src
    assert "Accounts" not in src and "UserFolder" not in src, "no Personal-reg OneDrive tip"
    assert "expanduser" not in state_src, "state_io must not expanduser Documents"
    assert "python.runtime.paths" in state_src and "gvd_docs_dir" in state_src

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

    data1, data2, data3, data4 = paths.folderid_documents_fields()
    assert data1 == 0xFDD39AD0 and data2 == 0x238F and data3 == 0x46AF
    assert tuple(data4) == (0xAD, 0xB4, 0x6C, 0x85, 0x48, 0x03, 0x69, 0xC7)
    assert paths.FOLDERID_Documents.upper() == "FDD39AD0-238F-46AF-ADB4-6C85480369C7"

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_uint32),
            ("Data2", ctypes.c_uint16),
            ("Data3", ctypes.c_uint16),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    guid = GUID(data1, data2, data3, (ctypes.c_ubyte * 8)(*data4))
    assert guid.Data1 == 0xFDD39AD0 and bytes(guid.Data4) == bytes(data4)

    assert paths.is_onedrive_path(Path("C:/Users/Name/OneDrive/Documents"))
    assert paths.is_onedrive_path(r"C:\Users\Name\OneDrive - Personal\Documents")
    assert paths.is_onedrive_path("C:/Users/Name/OneDrive - Contoso/Documents")
    assert not paths.is_onedrive_path(Path("C:/Users/Name/Documents"))
    assert not paths.is_onedrive_path(None)
    assert not paths.is_onedrive_path("  ")

    # ctypes / non-Windows: Known Folder helper returns None (fallback path).
    assert paths.windows_known_folder_documents() is None or sys.platform == "win32"

    onedrive_docs = Path("C:/Users/Name/OneDrive/Documents")
    local_gvd = "C:/Users/Name/Documents/GVD"
    lua_userprofile = resolve_docs_dir({"USERPROFILE": r"C:\Users\Name"})
    assert lua_userprofile == local_gvd, lua_userprofile
    # Lua FS-strip without USERPROFILE still lands on OneDrive Documents/GVD (#37).
    # Python must NOT follow that when FOLDERID is OneDrive — USERPROFILE-first.
    lua_onedrive = resolve_docs_dir(
        {},
        ["C:/Users/Name/OneDrive/Documents/BeamNG.drive/current"],
    )
    assert lua_onedrive == "C:/Users/Name/OneDrive/Documents/GVD", lua_onedrive

    # Mock FOLDERID → OneDrive: must NOT use it; resolve USERPROFILE\Documents\GVD.
    with _patch_attr(paths, "windows_known_folder_documents", lambda: onedrive_docs):
        with _env(USERPROFILE="C:/Users/Name", HOME="/home/other", GVD_DOCS_DIR=None):
            got = paths.gvd_docs_path()
            assert _slash(got) == local_gvd, got
            assert got.parts[-2:] == ("Documents", "GVD")
            assert "OneDrive" not in _slash(got)
            assert _slash(got) == lua_userprofile
            assert _slash(got) != lua_onedrive

    # OneDrive - Personal (and Business) folder names are also rejected.
    personal = Path("C:/Users/Name/OneDrive - Personal/Documents")
    with _patch_attr(paths, "windows_known_folder_documents", lambda: personal):
        with _env(USERPROFILE="C:/Users/Name", GVD_DOCS_DIR=None):
            got = paths.gvd_docs_path()
            assert _slash(got) == local_gvd, got
            assert "OneDrive" not in _slash(got)

    # ctypes SHGetKnownFolderPath returns the OneDrive Known Folder — probe only.
    with _patch_attr(
        paths,
        "_call_sh_get_known_folder_path",
        lambda: "C:/Users/Name/OneDrive/Documents",
    ):
        with _env(USERPROFILE="C:/Users/Name", GVD_DOCS_DIR=None):
            raw = paths.windows_known_folder_documents()
            assert _slash(raw) == "C:/Users/Name/OneDrive/Documents"
            assert paths.is_onedrive_path(raw)
            assert _slash(paths.gvd_docs_path()) == local_gvd

    # ctypes fails → %USERPROFILE%/Documents/GVD (classic retail).
    with _patch_attr(paths, "_call_sh_get_known_folder_path", lambda: None):
        assert paths.windows_known_folder_documents() is None
        with _env(USERPROFILE="C:/Users/Name", HOME="/home/other", GVD_DOCS_DIR=None):
            got = paths.gvd_docs_path()
            assert _slash(got) == local_gvd, got
            assert got.parts[-2:] == ("Documents", "GVD")

    # Classic Known Folder == USERPROFILE Documents (non-OneDrive retail).
    classic_docs = Path("C:/Users/Name/Documents")
    with _patch_attr(paths, "windows_known_folder_documents", lambda: classic_docs):
        with _env(GVD_DOCS_DIR=None, USERPROFILE="C:/Users/Name"):
            got = paths.gvd_docs_path()
            assert _slash(got) == local_gvd
            assert lua_onedrive != _slash(got)
            assert resolve_docs_dir({"USERPROFILE": "C:/Users/Name"}) == _slash(got)

    # USERPROFILE wins over a non-OneDrive FOLDERID redirect (system-disk local docs).
    other = Path("D:/Redirected/Documents")
    with _patch_attr(paths, "windows_known_folder_documents", lambda: other):
        with _env(USERPROFILE="C:/Users/Name", GVD_DOCS_DIR=None):
            got = paths.gvd_docs_path()
            assert _slash(got) == local_gvd, got
            assert "Redirected" not in _slash(got)

    # GVD_DOCS_DIR override wins over OneDrive FOLDERID and USERPROFILE.
    with tempfile.TemporaryDirectory() as td:
        override = Path(td) / "custom_gvd"
        with _patch_attr(paths, "windows_known_folder_documents", lambda: onedrive_docs):
            with _env(GVD_DOCS_DIR=str(override), USERPROFILE="C:/Users/Name"):
                got = paths.gvd_docs_path()
                assert Path(got) == override, got
                root = gvd_docs_dir()
                assert root == override and root.is_dir()
                assert state_path().parent == root
                assert cmd_path().parent == engage_path().parent == ego_path().parent == root

    # Blank GVD_DOCS_DIR is ignored (fall through to USERPROFILE Documents).
    with _patch_attr(paths, "windows_known_folder_documents", lambda: None):
        with _env(GVD_DOCS_DIR="   ", USERPROFILE="C:/Users/Name"):
            assert _slash(paths.gvd_docs_path()) == local_gvd

    # Last resort relative Documents/GVD (Lua last-resort contract).
    with _patch_attr(paths, "windows_known_folder_documents", lambda: None):
        with _patch_attr(paths, "env_documents_dir", lambda: None):
            with _env(GVD_DOCS_DIR=None):
                last = paths.gvd_docs_path()
                assert _slash(last) == "Documents/GVD"
                assert not str(last).endswith(".json")

    # Live mkdir: OneDrive FOLDERID probe must land on USERPROFILE\Documents\GVD.
    with tempfile.TemporaryDirectory() as td:
        profile = Path(td) / "Users" / "Name"
        known = profile / "OneDrive" / "Documents"
        known.mkdir(parents=True)
        local_docs = profile / "Documents"
        with _patch_attr(paths, "windows_known_folder_documents", lambda: known):
            with _env(USERPROFILE=str(profile), HOME=None, GVD_DOCS_DIR=None):
                root = gvd_docs_dir()
                assert root == local_docs / "GVD" and root.is_dir()
                assert "OneDrive" not in _slash(root)
                assert not (known / "GVD").exists()
                assert state_path().parent == root
                assert cmd_path().parent == engage_path().parent == ego_path().parent == root
                assert root.parts[-2:] == ("Documents", "GVD")

    # Classic non-OneDrive FOLDERID mkdir only when USERPROFILE/HOME are missing.
    with tempfile.TemporaryDirectory() as td:
        known = Path(td) / "Documents"
        known.mkdir(parents=True)
        with _patch_attr(paths, "windows_known_folder_documents", lambda: known):
            with _patch_attr(paths, "env_documents_dir", lambda: None):
                with _env(GVD_DOCS_DIR=None):
                    root = gvd_docs_dir()
                    assert root == known / "GVD" and root.is_dir()
                    assert state_path().parent == root
                    assert cmd_path().parent == engage_path().parent == ego_path().parent == root

    # Host fallback (Linux CI / non-OneDrive): still …/Documents/GVD unless override.
    with _env(GVD_DOCS_DIR=None):
        live = paths.gvd_docs_path()
        assert live.parts[-2:] == ("Documents", "GVD"), live


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
    check_python_known_folder()
    check_lua_harness()
    print("test_gvd_docs_dir: OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"test_gvd_docs_dir: FAIL - {e}")
        sys.exit(1)
