"""Resolve Documents/GVD to local ``%USERPROFILE%\\Documents\\GVD``.

#38 followed ``SHGetKnownFolderPath(FOLDERID_Documents)`` onto OneDrive Personal.
Lua ``gvdDocsDir`` is USERPROFILE-first (``{USERPROFILE}/Documents/GVD``) and is
left as-is. Python must match that local folder — not the Known Folder redirect.

Resolution:
  1. env override ``GVD_DOCS_DIR`` if set (full GVD root)
  2. ``%USERPROFILE%/Documents`` (local, non-redirected; never OneDrive)
  3. ``SHGetKnownFolderPath(FOLDERID_Documents)`` probe only if USERPROFILE is missing
     (ctypes/shell32, no pywin32) — **reject** when the path contains ``OneDrive``
  4. ``expanduser(~)/Documents`` / ``HOME``; else relative ``Documents``

Callers append ``GVD`` through ``gvd_docs_dir()`` unless the override already is
the GVD root.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# FOLDERID_Documents = {FDD39AD0-238F-46AF-ADB4-6C85480369C7}
FOLDERID_Documents = "FDD39AD0-238F-46AF-ADB4-6C85480369C7"
_FOLDERID_DOCUMENTS_DATA1 = 0xFDD39AD0
_FOLDERID_DOCUMENTS_DATA2 = 0x238F
_FOLDERID_DOCUMENTS_DATA3 = 0x46AF
_FOLDERID_DOCUMENTS_DATA4 = (0xAD, 0xB4, 0x6C, 0x85, 0x48, 0x03, 0x69, 0xC7)


def folderid_documents_fields() -> tuple[int, int, int, tuple[int, ...]]:
    """FOLDERID_Documents GUID fields (for tests / ctypes Structure)."""
    return (
        _FOLDERID_DOCUMENTS_DATA1,
        _FOLDERID_DOCUMENTS_DATA2,
        _FOLDERID_DOCUMENTS_DATA3,
        _FOLDERID_DOCUMENTS_DATA4,
    )


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


def _call_sh_get_known_folder_path() -> str | None:
    """SHGetKnownFolderPath(FOLDERID_Documents) or None if ctypes/shell32 fails."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return None
    try:
        shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
        ole32 = ctypes.windll.ole32  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return None

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_uint32),
            ("Data2", ctypes.c_uint16),
            ("Data3", ctypes.c_uint16),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    data1, data2, data3, data4 = folderid_documents_fields()
    folder_id = GUID(data1, data2, data3, (ctypes.c_ubyte * 8)(*data4))

    # pywin32-free: ctypes to shell32.SHGetKnownFolderPath(FOLDERID_Documents)
    sh_get_known_folder_path = shell32.SHGetKnownFolderPath
    sh_get_known_folder_path.argtypes = [
        ctypes.POINTER(GUID),
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_wchar_p),
    ]
    sh_get_known_folder_path.restype = ctypes.c_long  # HRESULT

    path_ptr = ctypes.c_wchar_p()
    try:
        hr = sh_get_known_folder_path(
            ctypes.byref(folder_id),
            0,  # KF_FLAG_DEFAULT — probe only; OneDrive redirects are rejected
            None,
            ctypes.byref(path_ptr),
        )
        if hr != 0 or not path_ptr.value:
            return None
        return path_ptr.value
    except Exception:
        return None
    finally:
        if path_ptr:
            try:
                ole32.CoTaskMemFree(path_ptr)
            except Exception:
                pass


def windows_known_folder_documents() -> Path | None:
    """Raw Documents folder from ``SHGetKnownFolderPath(FOLDERID_Documents)`` (probe)."""
    raw = _call_sh_get_known_folder_path()
    if not raw or not str(raw).strip():
        return None
    return Path(str(raw).rstrip("\\/"))


def env_documents_dir() -> Path | None:
    """Fallback Documents: ``%USERPROFILE%/Documents`` or ``~/Documents``."""
    profile = os.environ.get("USERPROFILE")
    if profile:
        return Path(profile) / "Documents"
    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return Path(expanded) / "Documents"
    home = os.environ.get("HOME")
    if home:
        return Path(home) / "Documents"
    return None


def documents_dir() -> Path:
    """Local Documents: USERPROFILE first; never an OneDrive Known Folder redirect."""
    env_docs = env_documents_dir()
    if env_docs is not None and not is_onedrive_path(env_docs):
        return env_docs
    known = windows_known_folder_documents()
    if known is not None and not is_onedrive_path(known):
        return known
    if env_docs is not None:
        return env_docs
    return Path("Documents")


def gvd_docs_path() -> Path:
    """``{Documents}/GVD`` without mkdir (tests / path math). ``GVD_DOCS_DIR`` wins."""
    override = env_override_gvd_docs_dir()
    if override is not None:
        return override
    return documents_dir() / "GVD"


def gvd_docs_dir() -> Path:
    """``{Documents}/GVD``, created if missing — public writer/reader root."""
    d = gvd_docs_path()
    d.mkdir(parents=True, exist_ok=True)
    return d
