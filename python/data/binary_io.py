"""Exact byte writes. Windows text mode must not touch PNG, JPEG, or npy bytes."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import BinaryIO

PNG_SIG = b"\x89PNG\r\n\x1a\n"
# Read size for the copy fallback of ``link_or_copy_new``.
COPY_CHUNK = 1 << 20


def os_binary() -> int:
    """``O_BINARY`` looked up at the call, so a test can install one.

    Platforms without text mode report 0. ``'wb'`` sets this same flag.
    """
    return int(getattr(os, "O_BINARY", 0))


def write_flags(extra: int) -> int:
    return int(extra) | os_binary()


def read_flags() -> int:
    return int(os.O_RDONLY) | os_binary()


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        wrote = os.write(fd, view)
        if wrote <= 0:
            raise OSError("binary write made no progress")
        view = view[wrote:]


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def verify_written(path: Path, payload: bytes) -> None:
    """Re-read the new file and refuse a text-mode newline expansion.

    The read is binary too. A text-mode read would hide the extra CR bytes
    in the PNG signature and the check would lie.
    """
    fd = os.open(str(path), read_flags())
    try:
        size = os.lseek(fd, 0, os.SEEK_END)
        os.lseek(fd, 0, os.SEEK_SET)
        sig = os.read(fd, 8)
    finally:
        os.close(fd)
    if payload.startswith(PNG_SIG) and sig != PNG_SIG:
        raise OSError(
            f"PNG signature mismatch for {path}: on-disk {sig!r}, "
            f"expected {PNG_SIG!r}. The writer expanded newlines (text mode)."
        )
    if size != len(payload):
        raise OSError(
            f"binary write size mismatch for {path}: on-disk {size} bytes, "
            f"encoded {len(payload)} bytes. The writer expanded newlines (text mode)."
        )


def write_new_file(path: Path, payload: bytes) -> None:
    """Create ``path`` with exactly ``payload``. Fails if the name exists.

    A failed verify deletes this new file. It does not delete any other path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), write_flags(os.O_CREAT | os.O_EXCL | os.O_WRONLY), 0o644)
    try:
        _write_all(fd, payload)
    except Exception:
        os.close(fd)
        _unlink_quiet(path)
        raise
    os.close(fd)
    try:
        verify_written(path, payload)
    except OSError:
        _unlink_quiet(path)
        raise


def write_atomic(path: Path, payload: bytes) -> None:
    """Write ``payload`` to a temp name in the same directory, then rename onto ``path``.

    The rename replaces ``path`` only. Callers must not pass an original capture
    they promised to keep.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_name(path)
    fd = os.open(str(tmp), write_flags(os.O_CREAT | os.O_EXCL | os.O_WRONLY), 0o644)
    try:
        _write_all(fd, payload)
    except Exception:
        os.close(fd)
        _unlink_quiet(tmp)
        raise
    os.close(fd)
    try:
        verify_written(tmp, payload)
    except OSError:
        _unlink_quiet(tmp)
        raise
    os.replace(tmp, path)


def _temp_name(path: Path) -> Path:
    base = path.with_name(path.name + ".tmp")
    if not base.exists():
        return base
    for n in range(1000):
        cand = path.with_name(f"{path.name}.{os.getpid()}.{n}.tmp")
        if not cand.exists():
            return cand
    raise OSError(f"no temp name next to {path}")


def _write_chunk(handle: BinaryIO, data: bytes) -> None:
    """One copy write of ``link_or_copy_new`` (a test seam)."""
    handle.write(data)


def link_or_copy_new(src: Path | str, dst: Path | str) -> None:
    """Move ``src`` to the new name ``dst``, create-only. Never renames.

    A hard link is tried first. A filesystem without hard links (FAT32, exFAT,
    some network mounts) gets a copy into ``dst`` opened with ``"xb"`` (O_EXCL),
    so an existing ``dst`` is refused at the final step and never replaced.
    ``FileExistsError`` is raised as is. A failed copy removes only the ``dst``
    it created and leaves ``src`` intact; ``src`` is removed only after ``dst``
    is complete and fsynced.
    """
    src = Path(src)
    dst = Path(dst)
    try:
        os.link(src, dst)
    except FileExistsError:
        raise
    except OSError:
        pass
    else:
        src.unlink()
        return
    handle = open(dst, "xb")
    created = True
    try:
        with open(src, "rb") as source:
            while True:
                data = source.read(COPY_CHUNK)
                if not data:
                    break
                _write_chunk(handle, data)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        want = os.stat(src).st_size
        got = os.stat(dst).st_size
        if got != want:
            raise OSError(errno.EIO, f"copy size mismatch for {dst}: {got} bytes, source {want}")
    except BaseException:
        handle.close()
        if created:
            _unlink_quiet(dst)
        raise
    src.unlink()
