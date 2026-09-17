"""Resolve the GVD bus dir to the running product sandbox (Python writers).

Python writes the product sandbox (dual-path #40). Lua reads the same folder
as userfolder-relative ``Documents/GVD`` via VFS / ``FS:readFile`` — never
absolute ``io.open``.

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

Callers append nothing: ``gvd_docs_dir()`` is the GVD root (state/cmd/engage).

Steam GELua does not inherit ``GVD_DOCS_DIR``. Identity (printed every second on
both sides) is the only link check: ``python_bus`` must be the same folder as
Lua's resolved ``Documents/GVD``. Drive vs Tech, a leftover override, or
``gvd_*.json`` under userfolder ``current\\`` is ``link=MISMATCH`` — no guess.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

TECH_GVD_REL = ("BeamNG", "BeamNG.tech", "current", "Documents", "GVD")
DRIVE_GVD_REL = ("BeamNG", "BeamNG.drive", "current", "Documents", "GVD")
TECH_GVD_TAIL = "BeamNG/BeamNG.tech/current/Documents/GVD"
DRIVE_GVD_TAIL = "BeamNG/BeamNG.drive/current/Documents/GVD"
LUA_BUS_REL = "Documents/GVD"


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


def norm_bus_folder(path: Path | str | None) -> str:
    """Slash-unified, case-folded folder identity. Empty if unset."""
    if path is None:
        return ""
    text = str(path).strip().replace("\\", "/")
    if not text:
        return ""
    while "//" in text:
        text = text.replace("//", "/")
    return text.rstrip("/").lower()


def buses_same_folder(python_bus: Path | str | None, lua_bus: Path | str | None) -> bool:
    """True only when both sides name the same folder. Do not guess relative==absolute."""
    a = norm_bus_folder(python_bus)
    b = norm_bus_folder(lua_bus)
    if not a or not b:
        return False
    if is_onedrive_path(a) or is_onedrive_path(b):
        return False
    return a == b


def ends_with_docs_gvd(path: Path | str | None) -> bool:
    s = norm_bus_folder(path)
    return s == "documents/gvd" or s.endswith("/documents/gvd")


def resolved_lua_bus(product: str | None = None) -> Path:
    """Folder relative ``Documents/GVD`` maps to for this product (never ``GVD_DOCS_DIR``)."""
    mapped = product_gvd_docs_path(product)
    if mapped is not None:
        return mapped
    return Path(LUA_BUS_REL)


def leftover_gvd_json(product: str | None = None) -> list[Path]:
    """``gvd_*.json`` sitting under userfolder ``current\\`` instead of ``Documents\\GVD``."""
    root = product_gvd_docs_path(product)
    if root is None:
        return []
    current = root.parent.parent
    hits: list[Path] = []
    try:
        for child in current.iterdir():
            name = child.name
            if child.is_file() and name.startswith("gvd_") and name.endswith(".json"):
                hits.append(child)
    except OSError:
        return []
    return hits


def other_product_name(product: str | None = None) -> str:
    return "tech" if (product or gvd_product()) == "drive" else "drive"


def other_product_state_path(product: str | None = None) -> Path | None:
    """``gvd_state.json`` in the other product tree, if present."""
    other = product_gvd_docs_path(other_product_name(product))
    if other is None:
        return None
    p = other / "gvd_state.json"
    try:
        if p.is_file():
            return p
    except OSError:
        return None
    return None


@dataclass(frozen=True)
class BusIdentity:
    python_bus: Path
    lua_bus: Path
    product: str
    matched: bool
    note: str
    leftover: tuple[Path, ...] = ()
    other_state: Path | None = None

    @property
    def python_bus_s(self) -> str:
        return str(self.python_bus)

    @property
    def lua_bus_s(self) -> str:
        return str(self.lua_bus)

    @property
    def link(self) -> str:
        return "ok" if self.matched else "MISMATCH"


def bus_identity() -> BusIdentity:
    """Python write root vs the folder Lua's ``Documents/GVD`` resolves to.

    ``GVD_DOCS_DIR`` is Python-only. If it is not the product sandbox, this is
    ``MISMATCH`` — Steam GELua will not follow the override.
    """
    product = gvd_product()
    python_bus = gvd_docs_path()
    lua_bus = resolved_lua_bus(product)
    leftover = tuple(leftover_gvd_json(product))
    other = other_product_state_path(product)
    note = "ok"
    matched = buses_same_folder(python_bus, lua_bus) and ends_with_docs_gvd(python_bus)
    if is_onedrive_path(python_bus) or is_onedrive_path(lua_bus):
        matched = False
        note = "OneDrive is not the GVD bus"
    elif not matched:
        note = "python_bus and lua_bus are not the same folder"
    our_state = python_bus / "gvd_state.json"
    try:
        our_exists = our_state.is_file()
    except OSError:
        our_exists = False
    if leftover and not our_exists:
        matched = False
        note = "leftover gvd_*.json under userfolder current\\ (not Documents/GVD)"
    if other is not None and not our_exists:
        matched = False
        note = "gvd_state.json is in the other product tree"
    return BusIdentity(
        python_bus=python_bus,
        lua_bus=lua_bus,
        product=product,
        matched=matched,
        note=note,
        leftover=leftover,
        other_state=other,
    )


def format_bus_identity_line(
    *,
    python_bus: Path | str,
    lua_bus: Path | str,
    product: str,
    state_mtime: object,
    engage: bool,
    seq: object,
    link: str | None = None,
) -> str:
    """Shared 1 Hz identity line (Python and Lua print the same fields)."""
    bits = [
        f"python_bus={python_bus}",
        f"lua_bus={lua_bus}",
        f"product={product}",
        f"state_mtime={state_mtime}",
        f"engage={str(bool(engage)).lower()}",
        f"seq={seq}",
    ]
    if link:
        bits.append(f"link={link}")
    return "[GVD] " + " ".join(bits)
