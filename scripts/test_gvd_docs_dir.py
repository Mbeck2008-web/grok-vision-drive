#!/usr/bin/env python3
"""Offline checks for Lua gvdDocsDir / _stripToUserHome (no BeamNG).

Expected resolve
----------------
1. USERPROFILE (preferred) → ``{USERPROFILE}/Documents/GVD``
2. else HOME → ``{HOME}/Documents/GVD``
3. else LOCALAPPDATA stripped to the profile (AppData parent) → ``…/Documents/GVD``
4. else FS:getUserPath / virtual2Native / getFileRealPath, stripped:
   - ``…/AppData/…`` → profile
   - ``…/Documents/…`` (BeamNG userfolder under Documents) → profile
   then ``{profile}/Documents/GVD``
5. else last resort ``Documents/GVD`` (never a bare ``gvd_*.json`` under
   BeamNG userfolder ``current\\``).

CEF Apps LINKED is live from **gvd_state heartbeat only** (Lua ``linkState``).
``gvd_ego.json`` is not required for LINKED; this path fix is so Lua reads the
same ``gvd_state.json`` Python writes.

Run: ``PYTHONPATH=. python scripts/test_gvd_docs_dir.py``
Lua extract harness (when lua5.1/luajit is on PATH): ``lua5.1 scripts/test_gvd_docs_dir.lua``
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LUA = ROOT / "beamng_mod" / "lua" / "ge" / "extensions" / "gvd" / "main.lua"


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
    check_lua_harness()
    print("test_gvd_docs_dir: OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"test_gvd_docs_dir: FAIL - {e}")
        sys.exit(1)
