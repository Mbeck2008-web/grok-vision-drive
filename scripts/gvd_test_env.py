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

``collect_root`` is where a collector test puts its ``--data-dir``: a fresh
folder under ``GVD_TEST_DATA_DIR`` when that is set, else under the temp
root. The product's own ``data_root.check_data_path`` (C:/D: refused, 20 GB
floor) is applied to it unchanged, so a bad root fails before the test body.
This module is the only code that reads ``GVD_TEST_DATA_DIR``;
``run_offline_suite`` sets it per script from ``suite_data_root``.
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
# Folder (off C:/D:, >= 20 GB free) collector tests write under; see collect_root.
TEST_DATA_ENV = "GVD_TEST_DATA_DIR"
GIB = 1024.0 ** 3


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


def _fs_type(path: Path) -> str:
    """Filesystem name of the volume holding ``path`` ("?" when it cannot be read)."""
    if sys.platform == "win32":
        import ctypes

        kernel = ctypes.windll.kernel32
        volume = ctypes.create_unicode_buffer(261)
        if not kernel.GetVolumePathNameW(str(path), volume, len(volume)):
            return "?"
        name = ctypes.create_unicode_buffer(261)
        fs = ctypes.create_unicode_buffer(261)
        ok = kernel.GetVolumeInformationW(volume.value, name, len(name), None, None, None, fs, len(fs))
        return fs.value if ok and fs.value else "?"
    try:
        lines = Path("/proc/self/mounts").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "?"
    target = os.path.realpath(str(path))
    best, best_fs = "", "?"
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        mount = parts[1].replace("\\040", " ").replace("\\011", "\t").replace("\\134", "\\")
        inside = target == mount or target.startswith(mount.rstrip("/") + "/")
        if inside and len(mount) >= len(best):
            best, best_fs = mount, parts[2]
    return best_fs


def _existing_parent(path: Path) -> Path:
    current = Path(path)
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def describe_root(path: Path | str) -> str:
    """``<path> fs=<FS> free=<GB> GB`` for the suite log."""
    path = Path(path)
    probe = _existing_parent(path)
    try:
        free = f"{shutil.disk_usage(probe).free / GIB:.1f}"
    except OSError:
        free = "?"
    return f"{path} fs={_fs_type(probe)} free={free} GB"


def suite_data_root() -> Path | None:
    """The suite's collector-output folder from ``GVD_TEST_DATA_DIR``, logged with ``describe_root``.

    Windows refuses C:/D: data paths, so the temp root cannot stand in there: an unset
    variable stops the suite at start. Elsewhere unset means collector tests use the temp root.
    """
    raw = os.environ.get(TEST_DATA_ENV, "").strip()
    if not raw:
        if sys.platform == "win32":
            raise SystemExit(f"set {TEST_DATA_ENV} to a folder off C:/D: with \u226520 GB free")
        print(f"[suite] data-dir unset ({TEST_DATA_ENV}); collector tests use the temp root", flush=True)
        return None
    root = Path(raw)
    root.mkdir(parents=True, exist_ok=True)
    print(f"[suite] data-dir {describe_root(root)}", flush=True)
    return root


@contextlib.contextmanager
def collect_root(prefix: str = "gvd-collect-") -> Iterator[Path]:
    """A fresh folder for collector output, checked by ``data_root.check_data_path``, removed after.

    Under ``GVD_TEST_DATA_DIR`` when set, else under ``tempfile.gettempdir()``.
    """
    from python.data.data_root import check_data_path

    raw = os.environ.get(TEST_DATA_ENV, "").strip()
    base = Path(raw) if raw else Path(tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=prefix, dir=str(base)))
    try:
        check_data_path(root)
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)

