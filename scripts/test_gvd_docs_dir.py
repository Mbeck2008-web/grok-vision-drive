#!/usr/bin/env python3
"""Offline checks for Lua gvdDocsDir / Python Documents/GVD (no BeamNG).

Lua gvdDocsDir (unchanged #37 contract)
---------------------------------------
1. USERPROFILE (preferred) → ``{USERPROFILE}/Documents/GVD``
2. else HOME → ``{HOME}/Documents/GVD``
3. else LOCALAPPDATA stripped to the profile (AppData parent) → ``…/Documents/GVD``
4. else FS:getUserPath / virtual2Native / getFileRealPath, stripped:
   - ``…/AppData/…`` → profile
   - ``…/Documents/…`` (BeamNG userfolder under Documents) → profile
   then ``{profile}/Documents/GVD``
5. else last resort ``Documents/GVD`` (never a bare ``gvd_*.json`` under
   BeamNG userfolder ``current\\``).

Python ``gvd_docs_dir`` (this hotfix)
-------------------------------------
1. ``SHGetKnownFolderPath(FOLDERID_Documents)`` → ``{that}/GVD``
   (OneDrive Personal Known Folder is ``…/OneDrive/Documents`` — no junction)
2. else ``%USERPROFILE%/Documents/GVD`` or ``expanduser`` Documents
3. else relative ``Documents/GVD``

Lua OneDrive strip of ``…/OneDrive/Documents/BeamNG.drive/…`` lands on
``…/OneDrive/Documents/GVD``, which is the same path Python gets from
FOLDERID_Documents. CEF Apps LINKED is ``gvd_state`` heartbeat only.

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
    assert "gvdDocsLogged" in docs_fn and "docs dir:" in docs_fn, "one-shot log"
    assert "dir = 'Documents/GVD'" in docs_exec, "last resort is Documents/GVD, never CWD or current\\"

    file_exec = re.sub(r"--[^\n]*", "", file_fn)
    assert "gvdDocsDir() .. '/' .. name" in file_exec
    assert not re.search(r"return\s+name\b", file_exec)

    # LINKED = gvd_state heartbeat (lastGood / hbAge). Not gvd_ego.json.
    link_exec = re.sub(r"--[^\n]*", "", link_fn)
    assert "lastGood" in link_exec and "hbAgeS" in link_exec
    assert "gvd_ego" not in link_exec and "egoFb" not in link_exec, link_fn
    assert "userEgoPath" not in link_exec


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
    """Python Documents/GVD via FOLDERID_Documents; fallbacks; Lua contract match."""
    import ctypes

    from python.control.actuate import cmd_path, ego_path, engage_path
    from python.runtime import paths
    from python.runtime.state_io import gvd_docs_dir, state_path

    src = (ROOT / "python" / "runtime" / "paths.py").read_text(encoding="utf-8")
    state_src = (ROOT / "python" / "runtime" / "state_io.py").read_text(encoding="utf-8")
    assert "SHGetKnownFolderPath" in src, "must call SHGetKnownFolderPath"
    assert "FOLDERID_Documents" in src
    assert "FDD39AD0" in src.upper()
    assert "238F" in src.upper() and "46AF" in src.upper()
    assert "CoTaskMemFree" in src, "Known Folder PWSTR must be freed"
    assert "no pywin32" in src.lower()
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

    # ctypes / non-Windows: Known Folder helper returns None (fallback path).
    assert paths.windows_known_folder_documents() is None or sys.platform == "win32"

    onedrive_docs = Path("C:/Users/Name/OneDrive/Documents")
    lua_onedrive = resolve_docs_dir(
        {},
        ["C:/Users/Name/OneDrive/Documents/BeamNG.drive/current"],
    )
    assert lua_onedrive == "C:/Users/Name/OneDrive/Documents/GVD", lua_onedrive

    with _patch_attr(paths, "windows_known_folder_documents", lambda: onedrive_docs):
        got = paths.gvd_docs_path()
        assert _slash(got) == "C:/Users/Name/OneDrive/Documents/GVD"
        assert got.parts[-2:] == ("Documents", "GVD")
        # Same folder Lua gvdDocsDir lands on after #37 OneDrive Documents strip.
        assert _slash(got) == lua_onedrive
        # Must not be the non-redirected %USERPROFILE%\Documents (no junction).
        assert "OneDrive" in _slash(got)
        assert _slash(got) != "C:/Users/Name/Documents/GVD"

    # ctypes SHGetKnownFolderPath returns the OneDrive Known Folder.
    with _patch_attr(
        paths,
        "_call_sh_get_known_folder_path",
        lambda: "C:/Users/Name/OneDrive/Documents",
    ):
        assert _slash(paths.windows_known_folder_documents()) == "C:/Users/Name/OneDrive/Documents"
        assert _slash(paths.gvd_docs_path()) == lua_onedrive

    # ctypes fails → %USERPROFILE%/Documents/GVD (classic retail).
    with _patch_attr(paths, "_call_sh_get_known_folder_path", lambda: None):
        assert paths.windows_known_folder_documents() is None
        with _env(USERPROFILE="C:/Users/Name", HOME="/home/other"):
            got = paths.gvd_docs_path()
            assert _slash(got) == "C:/Users/Name/Documents/GVD", got
            assert got.parts[-2:] == ("Documents", "GVD")

    # Classic Known Folder == USERPROFILE Documents (non-OneDrive retail).
    classic_docs = Path("C:/Users/Name/Documents")
    with _patch_attr(paths, "windows_known_folder_documents", lambda: classic_docs):
        got = paths.gvd_docs_path()
        assert _slash(got) == "C:/Users/Name/Documents/GVD"
        assert lua_onedrive != _slash(got)
        assert resolve_docs_dir({"USERPROFILE": "C:/Users/Name"}) == _slash(got)

    # Last resort relative Documents/GVD (Lua last-resort contract).
    with _patch_attr(paths, "windows_known_folder_documents", lambda: None):
        with _patch_attr(paths, "env_documents_dir", lambda: None):
            last = paths.gvd_docs_path()
            assert _slash(last) == "Documents/GVD"
            assert not str(last).endswith(".json")

    # Live mkdir under a fake OneDrive Known Folder — bus files share that root.
    with tempfile.TemporaryDirectory() as td:
        known = Path(td) / "OneDrive" / "Documents"
        known.mkdir(parents=True)
        with _patch_attr(paths, "windows_known_folder_documents", lambda: known):
            root = gvd_docs_dir()
            assert root == known / "GVD" and root.is_dir()
            assert state_path().parent == root
            assert cmd_path().parent == engage_path().parent == ego_path().parent == root
            assert root.parts[-2:] == ("Documents", "GVD")

    # Host fallback (Linux CI / non-OneDrive): still …/Documents/GVD.
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
