"""Resolve the GVD bus dir to the running product sandbox.

Tech GELua cannot ``io.open`` the Drive sandbox; Drive GELua cannot ``io.open``
the Tech sandbox. Steam does not inherit ``GVD_DOCS_DIR`` into GELua, so live
retail must use the Drive userfolder bus without an env override.

  Tech:  ``%LOCALAPPDATA%\\BeamNG\\BeamNG.tech\\current\\Documents\\GVD``
  Drive: ``%LOCALAPPDATA%\\BeamNG\\BeamNG.drive\\current\\Documents\\GVD``

Not USERPROFILE Documents. Not OneDrive. No junctions.

Resolution:
  1. env override ``GVD_DOCS_DIR`` if set (full GVD root)
  2. product sandbox under ``%LOCALAPPDATA%`` (or synthesized
     ``{USERPROFILE|HOME}/AppData/Local`` — never Documents):
       - ``GVD_PRODUCT=tech`` / ``GVD_BEAMNG=1`` / ``GVD_BACKEND=beamngpy`` → Tech
       - else Drive / retail (``play_gvd.bat`` / window backend)
  3. else relative ``Documents/GVD``

Lua additionally reconstructs the product from the running ``FS:getUserPath``
(before any LOCALAPPDATA default) so GELua matches the process that loaded it.

Callers append nothing: ``gvd_docs_dir()`` is the GVD root (state/cmd/engage).
"""

from __future__ import annotations

import os
from pathlib import Path

TECH_GVD_REL = ("BeamNG", "BeamNG.tech", "current", "Documents", "GVD")
DRIVE_GVD_REL = ("BeamNG", "BeamNG.drive", "current", "Documents", "GVD")
TECH_GVD_TAIL = "BeamNG/BeamNG.tech/current/Documents/GVD"
DRIVE_GVD_TAIL = "BeamNG/BeamNG.drive/current/Documents/GVD"


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


def gvd_product() -> str:
    """``tech`` or ``drive``. Explicit ``GVD_PRODUCT``, else Tech env, else Drive."""
    raw = _env_nonempty("GVD_PRODUCT")
    if raw:
        p = raw.strip().lower().replace(" ", "")
        if p in ("tech", "beamng.tech", "beamngtech"):
            return "tech"
        if p in ("drive", "retail", "beamng.drive", "beamngdrive"):
            return "drive"
    beamng = (_env_nonempty("GVD_BEAMNG") or "").strip().lower()
    if beamng in ("1", "true", "yes"):
        return "tech"
    backend = (_env_nonempty("GVD_BACKEND") or "").strip().lower()
    if backend in ("beamngpy", "tech"):
        return "tech"
    return "drive"


def gvd_rel_for_product(product: str | None = None) -> tuple[str, ...]:
    if (product or gvd_product()) == "tech":
        return TECH_GVD_REL
    return DRIVE_GVD_REL


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


def drive_gvd_docs_path() -> Path | None:
    """Drive / retail sandbox GVD root from LOCALAPPDATA (or synthesized AppData\\Local)."""
    la = local_appdata_dir()
    if la is None:
        return None
    return la.joinpath(*DRIVE_GVD_REL)


def product_gvd_docs_path(product: str | None = None) -> Path | None:
    """Product sandbox GVD root. ``product`` is ``tech`` or ``drive``."""
    la = local_appdata_dir()
    if la is None:
        return None
    return la.joinpath(*gvd_rel_for_product(product))


def gvd_docs_path() -> Path:
    """GVD bus root without mkdir (tests / path math). ``GVD_DOCS_DIR`` wins."""
    override = env_override_gvd_docs_dir()
    if override is not None:
        return override
    product = product_gvd_docs_path()
    if product is not None:
        return product
    return Path("Documents") / "GVD"


def gvd_docs_dir() -> Path:
    """Product sandbox GVD root, created if missing — public writer/reader root."""
    d = gvd_docs_path()
    d.mkdir(parents=True, exist_ok=True)
    return d
