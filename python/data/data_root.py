"""Where large GVD files go.

Recordings, datasets, checkpoints, ONNX exports, training temp files, and
live ground-truth logs default to ``E:\\GVD_data``. ``GVD_DATA_DIR`` overrides
that. C: and D: are refused. Nothing here creates a folder on those drives.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

DEFAULT_DATA_DIR = Path(r"E:\GVD_data")
MIN_FREE_BYTES = 20 * 1024 * 1024 * 1024


class DataDriveError(RuntimeError):
    """The data path is on C: or D:, or the drive is below the free-space floor."""


def drive_letter(path: Path | str) -> str:
    text = str(path).replace("/", "\\")
    if len(text) >= 2 and text[1] == ":":
        return text[0].upper()
    drive = Path(path).drive
    if len(drive) >= 2 and drive[1] == ":":
        return drive[0].upper()
    return ""


def data_root() -> Path:
    raw = os.environ.get("GVD_DATA_DIR", "").strip()
    root = Path(raw) if raw else DEFAULT_DATA_DIR
    refuse_system_drive(root)
    return root


def use_data_disk() -> bool:
    """True when the data root is a real disk, not a relative ``E:`` path.

    Windows uses ``E:\\GVD_data`` unless ``GVD_DATA_DIR`` is set. Other
    systems only redirect when ``GVD_DATA_DIR`` is an absolute path, so a
    Linux test run does not create an ``E:`` folder inside the repo.
    """
    raw = os.environ.get("GVD_DATA_DIR", "").strip()
    if raw:
        path = Path(raw)
        return path.is_absolute() and drive_letter(path) not in ("C", "D")
    return os.name == "nt"


def refuse_system_drive(path: Path | str) -> None:
    letter = drive_letter(path)
    if letter in ("C", "D"):
        raise DataDriveError(
            f"refusing {path}: C: and D: are too small. "
            f"Set GVD_DATA_DIR to E:\\GVD_data (E: is the data disk)."
        )


def scene_lanes_dir() -> Path:
    return data_root() / "scene_lanes"


def logs_dir() -> Path:
    return data_root() / "logs"


def roads_dir() -> Path:
    """Live road-network dumps, one gzip per map."""
    return data_root() / "roads"


def models_dir() -> Path:
    return data_root() / "models"


def cache_dir() -> Path:
    return data_root() / "cache"


def tmp_dir() -> Path:
    return data_root() / "tmp"


def apply_cache_env() -> Path:
    """Point torch and temp files at the data disk. Does not create C: or D: dirs."""
    root = data_root()
    if not use_data_disk():
        return root
    cache = cache_dir()
    temp = tmp_dir()
    os.environ["GVD_DATA_DIR"] = str(root)
    os.environ["TORCH_HOME"] = str(cache / "torch")
    os.environ["TMP"] = str(temp)
    os.environ["TEMP"] = str(temp)
    os.environ["TMPDIR"] = str(temp)
    return root


def ensure_cache_dirs() -> Path:
    """Create cache and temp on the data disk after the drive check."""
    root = apply_cache_env()
    if not use_data_disk():
        return root
    cache_dir().mkdir(parents=True, exist_ok=True)
    (cache_dir() / "torch").mkdir(parents=True, exist_ok=True)
    tmp_dir().mkdir(parents=True, exist_ok=True)
    return root


def _existing(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            break
        current = parent
    return current if current.exists() else path


def free_bytes(path: Path | str) -> int:
    probe = _existing(Path(path))
    return int(shutil.disk_usage(probe).free)


def space_failure(free: int, *, minimum: int = MIN_FREE_BYTES) -> str | None:
    if int(free) >= int(minimum):
        return None
    have = int(free) / (1024.0 ** 3)
    need = int(minimum) / (1024.0 ** 3)
    return f"only {have:.1f} GB free, need {need:.0f} GB"


def check_data_path(path: Path | str) -> int:
    """Refuse C:/D: and a drive under 20 GB free. Returns free bytes."""
    refuse_system_drive(path)
    free = free_bytes(path)
    reason = space_failure(free)
    if reason is not None:
        raise DataDriveError(f"{path}: {reason}")
    return free
