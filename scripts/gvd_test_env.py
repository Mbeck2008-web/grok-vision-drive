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

``skip_check`` is how a script skips one check it cannot run on this host
(torch missing, say): it prints the skip and appends it to the file named by
``GVD_SKIP_LOG``, which the suite sets per script and counts in its tally, so
a script that passes with checks skipped does not read as a plain pass.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Mapping

# Env names a leftover shell value could point at a shared folder with.
DROP = ("GVD_DOCS_DIR",)
# Thread pools a test process starts with (see isolated_env).
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
# Set by run_offline_suite per script: where skip_check records skipped checks.
SKIP_LOG_ENV = "GVD_SKIP_LOG"
SKIP_PREFIX = "SKIP-CHECK"


def skip_check(name: str, reason: str) -> None:
    """Skip one check, visibly: print it and record it for the suite tally."""
    script = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "?"
    line = f"{SKIP_PREFIX} {script} {name}: {reason}"
    print(line, flush=True)
    log = os.environ.get(SKIP_LOG_ENV)
    if log:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def read_skips(log: Path) -> list[str]:
    """The skip_check lines a script recorded (empty when it skipped nothing)."""
    try:
        text = Path(log).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    return [line for line in text.splitlines() if line.startswith(SKIP_PREFIX)]


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
    # One OpenMP / MKL thread per test process unless the caller already chose. torch sizes
    # its intra-op pool to every core; when the cores are shared (a CI neighbour, a parallel
    # suite) each parallel op waits at a barrier for pool threads that are descheduled.
    # test_train_options under 64 busy loops on 8 cores: sanity_overfit 1.6 s -> 451 s with 8
    # threads, 16.7 s with one; idle the script goes from about 6 s to 11 s.
    for name in THREAD_ENV:
        env.setdefault(name, "1")
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
