"""Resolve the GVD bus dir to the running product sandbox (Python writers).

Python writes the product sandbox (dual-path #40). Lua reads the same folder
as userfolder-relative ``Documents/GVD`` via VFS / ``FS:readFile`` — never
absolute ``io.open``.

  Tech:  ``%LOCALAPPDATA%\\BeamNG\\BeamNG.tech\\current\\Documents\\GVD``
  Drive: ``%LOCALAPPDATA%\\BeamNG\\BeamNG.drive\\current\\Documents\\GVD``

Not USERPROFILE Documents. Not OneDrive. No junctions.

Resolution:
  1. live ``lua_bus`` from a fresh (< 1 s) ``gvd_link.json`` / ``gvd_ego.json``
     at the predicted pin — source of truth (``FS:getUserPath()`` + Documents/GVD)
  2. else env override ``GVD_DOCS_DIR`` if set (full GVD root; bat pin only
     until Lua's first ``lua_bus`` lands)
  3. else product sandbox under ``%LOCALAPPDATA%`` (or synthesized
     ``{USERPROFILE|HOME}/AppData/Local`` — never Documents):
       - ``GVD_PRODUCT=tech`` / ``GVD_BEAMNG=1`` / ``GVD_BACKEND=beamngpy`` → Tech
       - else Drive / retail (``play_gvd.bat`` / window backend)
  4. else relative ``Documents/GVD``

Callers append nothing: ``gvd_docs_dir()`` is the GVD root (state/cmd/engage).

Steam GELua does not inherit ``GVD_DOCS_DIR``. Do not predict Lua's folder from
LOCALAPPDATA: a moved userfolder (Launcher → Manage User Folder) makes
``current\\Documents\\GVD`` the wrong string. Identity is the live ``lua_bus``
string vs the folder Python actually writes. Missing/stale/bad-tail ``lua_bus``
is ``link=MISMATCH`` — no OneDrive / USERPROFILE Documents fallback.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

TECH_GVD_REL = ("BeamNG", "BeamNG.tech", "current", "Documents", "GVD")
DRIVE_GVD_REL = ("BeamNG", "BeamNG.drive", "current", "Documents", "GVD")
TECH_GVD_TAIL = "BeamNG/BeamNG.tech/current/Documents/GVD"
DRIVE_GVD_TAIL = "BeamNG/BeamNG.drive/current/Documents/GVD"
LUA_BUS_REL = "Documents/GVD"
LUA_BUS_FRESH_S = 1.0
LUA_BUS_FILES = ("gvd_link.json", "gvd_ego.json")


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


def predicted_gvd_docs_path() -> Path:
    """Bat / LOCALAPPDATA guess. Not Lua's live ``getUserPath`` folder."""
    override = env_override_gvd_docs_dir()
    if override is not None:
        return override
    product = product_gvd_docs_path()
    if product is not None:
        return product
    return Path("Documents") / "GVD"


def gvd_docs_path() -> Path:
    """GVD bus root without mkdir (tests / path math). Live ``lua_bus`` wins."""
    live, _note = read_live_lua_bus()
    if live is not None:
        return live
    return predicted_gvd_docs_path()


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


def is_abs_disk_path(path: Path | str | None) -> bool:
    """True for a Windows drive / UNC / POSIX absolute path. Relative ``Documents/GVD`` is not."""
    if path is None:
        return False
    text = str(path).strip().replace("\\", "/")
    if not text:
        return False
    if len(text) >= 2 and text[1] == ":":
        return True
    if text.startswith("//") or text.startswith("/"):
        return True
    try:
        return Path(text).is_absolute()
    except (OSError, ValueError):
        return False


def is_userprofile_documents_gvd(path: Path | str | None) -> bool:
    """True when *path* is USERPROFILE/HOME Documents/GVD — not a BeamNG userfolder."""
    if path is None:
        return False
    profile = _env_nonempty("USERPROFILE") or _env_nonempty("HOME")
    if not profile:
        return False
    banned = Path(profile) / "Documents" / "GVD"
    return buses_same_folder(path, banned)


def accepted_lua_bus_folder(raw: Path | str | None) -> Path | None:
    """Live ``lua_bus`` only when it is a real absolute folder ending in Documents/GVD."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    p = Path(text)
    if not is_abs_disk_path(p):
        return None
    if is_onedrive_path(p) or is_userprofile_documents_gvd(p):
        return None
    if not ends_with_docs_gvd(p):
        return None
    try:
        if p.is_dir():
            return p
    except OSError:
        return None
    return None


def _probe_lua_bus_files() -> list[Path]:
    """Look for Lua's handshake at the bat pin / predicted sandbox only.

    Do not search USERPROFILE Documents or OneDrive.
    """
    files: list[Path] = []
    seen: set[str] = set()
    for root in (env_override_gvd_docs_dir(), product_gvd_docs_path()):
        if root is None:
            continue
        key = norm_bus_folder(root)
        if not key or key in seen:
            continue
        seen.add(key)
        for name in LUA_BUS_FILES:
            files.append(root / name)
    return files


def _read_lua_bus_field(path: Path, now: float) -> tuple[str | None, float | None]:
    """Return ``(lua_bus or None, age_s)``. Missing file → ``(None, None)``."""
    try:
        if not path.is_file():
            return None, None
        age = now - path.stat().st_mtime
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError, TypeError, ValueError):
        return None, None
    if not isinstance(data, dict):
        return None, age
    raw = data.get("lua_bus")
    if raw is None:
        return None, age
    text = str(raw).strip()
    if not text:
        return None, age
    return text, age


def read_live_lua_bus(*, now: float | None = None) -> tuple[Path | None, str]:
    """Fresh ``lua_bus`` from ``gvd_link.json`` / ``gvd_ego.json``, or ``(None, why)``."""
    t = time.time() if now is None else now
    saw_file = False
    saw_stale = False
    saw_bad = False
    for path in _probe_lua_bus_files():
        raw, age = _read_lua_bus_field(path, t)
        if age is None:
            continue
        saw_file = True
        if age >= LUA_BUS_FRESH_S:
            saw_stale = True
            continue
        accepted = accepted_lua_bus_folder(raw)
        if accepted is not None:
            return accepted, "ok"
        saw_bad = True
    if not saw_file:
        return None, "lua_bus missing or stale"
    if saw_bad:
        return None, "lua_bus is not a Documents/GVD folder"
    if saw_stale:
        return None, "lua_bus missing or stale"
    return None, "lua_bus missing or stale"


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
    """Python write root vs Lua's live ``getUserPath()`` folder.

    ``lua_bus`` from a fresh ego/link file is source of truth. The LOCALAPPDATA
    ``current\\Documents\\GVD`` guess is only a pin until that file lands.
    Missing, stale, OneDrive, USERPROFILE Documents, or a path that is not a
    real ``Documents/GVD`` folder is ``MISMATCH`` — do not predict a match.
    """
    product = gvd_product()
    leftover = tuple(leftover_gvd_json(product))
    other = other_product_state_path(product)
    live, live_note = read_live_lua_bus()
    if live is not None:
        return BusIdentity(
            python_bus=live,
            lua_bus=live,
            product=product,
            matched=True,
            note="ok",
            leftover=leftover,
            other_state=other,
        )
    predicted = predicted_gvd_docs_path()
    note = live_note
    if is_onedrive_path(predicted):
        note = "OneDrive is not the GVD bus"
    our_state = predicted / "gvd_state.json"
    try:
        our_exists = our_state.is_file()
    except OSError:
        our_exists = False
    if leftover and not our_exists:
        note = "leftover gvd_*.json under userfolder current\\ (not Documents/GVD)"
    elif other is not None and not our_exists:
        note = "gvd_state.json is in the other product tree"
    return BusIdentity(
        python_bus=predicted,
        lua_bus=Path(""),
        product=product,
        matched=False,
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
