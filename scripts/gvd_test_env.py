"""Per-test home, AppData, and temp folders for the offline suite.

GVD finds its bus folder (``gvd_state.json``, ``gvd_cmd.json``,
``gvd_engage.json``, ``gvd_strip_record.json``, ...) from the environment:
``GVD_DOCS_DIR``, else ``LOCALAPPDATA``, else ``{USERPROFILE|HOME}/AppData/Local``
(``python/runtime/paths.py``). Pointing those at a fresh folder per test means
a file one test leaves behind cannot reach the next one, and nothing lands in
the real home of the PC running the suite.

``TORCH_HOME`` points into the same folder, so a test that downloads
pretrained weights is caught (``torch_hub_files``) instead of reusing a
cache. Not a test itself: the suite only runs ``test_*`` files.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator, Mapping

# Env names a leftover shell value could point at a shared folder with.
DROP = ("GVD_DOCS_DIR",)


def isolated_env(base: Mapping[str, str], root: Path) -> dict[str, str]:
    """``base`` with home, AppData, temp, and cache folders under ``root`` (created)."""
    root = Path(root)
    home = root / "home"
    roaming = home / "AppData" / "Roaming"
    local = home / "AppData" / "Local"
    tmp = root / "tmp"
    for folder in (roaming, local, tmp, home / ".cache", home / ".config"):
        folder.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in base.items() if k not in DROP}
    env.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "APPDATA": str(roaming),
            "LOCALAPPDATA": str(local),
            "TMPDIR": str(tmp),
            "TMP": str(tmp),
            "TEMP": str(tmp),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "TORCH_HOME": str(home / ".cache" / "torch"),
        }
    )
    return env


def torch_hub_files(env: Mapping[str, str]) -> list[str]:
    """Files under this env's torch hub folder. Empty means no weights were downloaded."""
    hub = Path(env["TORCH_HOME"]) / "hub"
    if not hub.is_dir():
        return []
    return sorted(str(p.relative_to(hub)) for p in hub.rglob("*") if p.is_file())


@contextlib.contextmanager
def isolated_home(prefix: str = "gvd-test-") -> Iterator[Path]:
    """In-process: swap ``os.environ`` to a fresh isolated folder, restore after.

    Child processes started inside inherit it. ``tempfile`` is re-pointed too.
    """
    saved = dict(os.environ)
    saved_tempdir = tempfile.tempdir
    root = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        env = isolated_env(saved, root)
        os.environ.clear()
        os.environ.update(env)
        tempfile.tempdir = env["TMPDIR"]
        yield root
    finally:
        os.environ.clear()
        os.environ.update(saved)
        tempfile.tempdir = saved_tempdir
        shutil.rmtree(root, ignore_errors=True)
