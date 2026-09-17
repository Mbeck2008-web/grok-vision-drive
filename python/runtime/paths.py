"""Resolve the GVD bus dir to the Tech GELua sandbox.

Tech Lua cannot read ``%USERPROFILE%\\Documents\\GVD`` (sandbox). Python and Lua
share:

  ``%LOCALAPPDATA%\\BeamNG\\BeamNG.tech\\current\\Documents\\GVD``

Not USERPROFILE Documents. Not OneDrive. No junctions.

Resolution:
  1. env override ``GVD_DOCS_DIR`` if set (full GVD root)
  2. ``%LOCALAPPDATA%/BeamNG/BeamNG.tech/current/Documents/GVD``
  3. else ``{USERPROFILE|HOME}/AppData/Local/`` + the same Tech tail
     (LOCALAPPDATA is usually USERPROFILE\\AppData\\Local — never Documents)
  4. else relative ``Documents/GVD``

Callers append nothing: ``gvd_docs_dir()`` is the GVD root (state/cmd/engage).
"""

from __future__ import annotations

import os
from pathlib import Path

TECH_GVD_REL = ("BeamNG", "BeamNG.tech", "current", "Documents", "GVD")
TECH_GVD_TAIL = "BeamNG/BeamNG.tech/current/Documents/GVD"


def is_onedrive_path(path: Path | str | None) -> bool:
    """True when *path* is a OneDrive redirect (Personal, Business, ``OneDrive - …``)."""
    if path is None:
        return False
    text = str(path).replace("\\", "/")
    if not text.strip():
        return False
    return "onedrive" in text.lower()


def env_override_gvd_docs_dir() -> Path | None:
    """``GVD_DOCS_DIR`` env override (full GVD root), or None if unset/blank."""
    raw = os.environ.get("GVD_DOCS_DIR")
    if raw is None:
        return None
    stripped = str(raw).strip()
    if not stripped:
        return None
    return Path(stripped)


def _env_nonempty(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    stripped = str(raw).strip()
    if not stripped:
        return None
    return stripped


def local_appdata_dir() -> Path | None:
    """``%LOCALAPPDATA%``, else ``{USERPROFILE|HOME}/AppData/Local``. Never OneDrive."""
    raw = _env_nonempty("LOCALAPPDATA")
    if raw:
        p = Path(raw)
        if not is_onedrive_path(p):
            return p
    profile = _env_nonempty("USERPROFILE") or _env_nonempty("HOME")
    if profile:
        p = Path(profile) / "AppData" / "Local"
        if not is_onedrive_path(p):
            return p
    return None


def tech_gvd_docs_path() -> Path | None:
    """Tech sandbox GVD root from LOCALAPPDATA (or synthesized AppData\\Local)."""
    la = local_appdata_dir()
    if la is None:
        return None
    return la.joinpath(*TECH_GVD_REL)


def gvd_docs_path() -> Path:
    """GVD bus root without mkdir (tests / path math). ``GVD_DOCS_DIR`` wins."""
    override = env_override_gvd_docs_dir()
    if override is not None:
        return override
    tech = tech_gvd_docs_path()
    if tech is not None:
        return tech
    return Path("Documents") / "GVD"


def gvd_docs_dir() -> Path:
    """Tech sandbox GVD root, created if missing — public writer/reader root."""
    d = gvd_docs_path()
    d.mkdir(parents=True, exist_ok=True)
    return d
