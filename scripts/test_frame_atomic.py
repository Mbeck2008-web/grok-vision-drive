#!/usr/bin/env python3
"""Frame PNGs go through binary_io.write_atomic: temp file, verify, rename.

The PNG is on disk under its final name before its state line; a crash or a
failed verify mid-write leaves no torn frame and no state line; an existing
frame is never replaced. ``link_or_copy_new`` (ONNX exports) links, else
copies into an O_EXCL-created name: an existing name is refused, a failed
copy removes only the name it created, and the source goes only after a
complete copy. Offline.
"""
from __future__ import annotations

import errno
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import python.data.binary_io as binary_io  # noqa: E402
import python.data.raw_frames as raw_frames  # noqa: E402
from python.data.binary_io import PNG_SIG  # noqa: E402
from python.data.raw_frames import RawFrameWriter  # noqa: E402
from gvd_test_env import collect_root  # noqa: E402

SRC_BYTES = bytes(range(256)) * 3 + b"\r\n\n\r"


def _no_link(src, dst, *args, **kwargs):
    raise OSError(errno.EPERM, "hard links not supported (forced)")


class _ForcedFallback:
    """``os.link`` raises EPERM inside the block, as on FAT32/exFAT."""

    def __enter__(self):
        self.real = os.link
        os.link = _no_link
        return self

    def __exit__(self, *exc):
        os.link = self.real
        return False


class _RealLink:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


VARIANTS = (("link", _RealLink), ("copy fallback", _ForcedFallback))


def _img() -> np.ndarray:
    a = np.zeros((6, 9, 3), dtype=np.uint8)
    a[:, :, 0] = np.arange(9, dtype=np.uint8)
    a[2, 3] = (10, 13, 255)  # bytes 0x0a / 0x0d in the payload
    return a


def _state_lines(root: Path) -> int:
    path = root / "state.jsonl"
    return len(path.read_text(encoding="utf-8").splitlines()) if path.is_file() else 0


def check_temp_verify_rename_before_state() -> None:
    events: list[tuple] = []
    real_atomic, real_replace, real_verify = raw_frames.write_atomic, binary_io.os.replace, binary_io.verify_written
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def atomic(path, payload):
            events.append(("atomic", path.name, _state_lines(root)))
            return real_atomic(path, payload)

        def verify(path, payload):
            events.append(("verify", Path(path).name))
            return real_verify(path, payload)

        def replace(src, dst):
            events.append(("rename", Path(src).name, Path(dst).name, Path(dst).exists()))
            return real_replace(src, dst)

        raw_frames.write_atomic = atomic
        binary_io.verify_written = verify
        binary_io.os.replace = replace
        try:
            writer = RawFrameWriter(root, [])
            writer.write({"main": _img()}, t=1.0)
            writer.write({"main": _img()}, t=1.1)
        finally:
            raw_frames.write_atomic, binary_io.verify_written, binary_io.os.replace = real_atomic, real_verify, real_replace
        png = root / "raw" / "main" / "000" / "000001.png"
        assert png.read_bytes().startswith(PNG_SIG)
        assert not list((root / "raw" / "main" / "000").glob("*.tmp")), "temp files left behind"
        rows = [json.loads(x) for x in (root / "state.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [r["cams"]["main"]["file"] for r in rows] == ["raw/main/000/000000.png", "raw/main/000/000001.png"], rows
    # raw_layout.json (the folder size) goes through the same temp/verify/rename first.
    layout = [e for e in events if e[1].startswith("raw_layout.json")]
    assert layout == [("atomic", "raw_layout.json", 0), ("verify", "raw_layout.json.tmp"), ("rename", "raw_layout.json.tmp", "raw_layout.json", False)], events
    assert events[:3] == layout, events
    events = events[3:]
    assert events[:3] == [("atomic", "000000.png", 0), ("verify", "000000.png.tmp"), ("rename", "000000.png.tmp", "000000.png", False)], events
    assert events[3] == ("atomic", "000001.png", 1), events  # frame 1 written while state holds only row 0
    print(f"atomic ok: {events[:3]}")


def check_crash_mid_write_leaves_nothing() -> None:
    real = binary_io._write_all
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = RawFrameWriter(root, [])
        writer.write({"main": _img()}, t=1.0)

        def torn(fd, payload):
            real(fd, payload[: len(payload) // 2])
            raise OSError("disk yanked")

        binary_io._write_all = torn
        try:
            writer.write({"main": _img()}, t=1.1)
        except OSError as exc:
            assert "disk yanked" in str(exc)
        else:
            raise AssertionError("the write error must surface")
        finally:
            binary_io._write_all = real
        folder = root / "raw" / "main" / "000"
        assert sorted(p.name for p in folder.iterdir()) == ["000000.png"], list(folder.iterdir())
        assert _state_lines(root) == 1
    print("crash mid-write: no torn frame, no temp, no state line")


def check_failed_verify_and_no_replace() -> None:
    real = binary_io.verify_written
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = RawFrameWriter(root, [])

        def bad(path, payload):
            raise OSError("PNG signature mismatch (text mode)")

        binary_io.verify_written = bad
        try:
            writer.write({"main": _img()}, t=1.0)
        except OSError:
            pass
        else:
            raise AssertionError("a failed verify must surface")
        finally:
            binary_io.verify_written = real
        folder = root / "raw" / "main" / "000"
        assert not folder.exists() or not list(folder.iterdir()), list(folder.iterdir())
        assert _state_lines(root) == 0
        # An existing frame is never replaced.
        folder.mkdir(parents=True, exist_ok=True)
        keep = folder / "000000.png"
        writer2 = RawFrameWriter(root, [])
        keep_name = folder / f"{writer2._n:06d}.png"
        keep_name.write_bytes(b"keep me")
        try:
            writer2.write({"main": _img()}, t=2.0)
        except FileExistsError as exc:
            assert "refusing to replace" in str(exc)
        else:
            raise AssertionError("an existing frame must not be overwritten")
        assert keep_name.read_bytes() == b"keep me"
        del keep
    print("failed verify: final name never created; existing frame kept")


def check_link_or_copy_refuses_existing() -> None:
    for name, variant in VARIANTS:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "model.onnx.partial"
            dst = Path(tmp) / "model.onnx"
            src.write_bytes(SRC_BYTES)
            dst.write_bytes(b"OLD")
            with variant():
                try:
                    binary_io.link_or_copy_new(src, dst)
                except FileExistsError:
                    pass
                else:
                    raise AssertionError(f"{name}: an existing dst must be refused")
            assert dst.read_bytes() == b"OLD", name
            assert src.read_bytes() == SRC_BYTES, name
    print("link_or_copy_new: existing dst refused (link and copy), dst and src unchanged")


def check_link_or_copy_success() -> None:
    for name, variant in VARIANTS:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "model.onnx.partial"
            dst = Path(tmp) / "model.onnx"
            src.write_bytes(SRC_BYTES)
            with variant():
                binary_io.link_or_copy_new(src, dst)
            assert dst.read_bytes() == SRC_BYTES, name
            assert not src.exists(), name
            assert os.stat(dst).st_nlink == 1, (name, os.stat(dst).st_nlink)
    print("link_or_copy_new: dst holds the bytes and src is gone (link and copy)")


def check_link_or_copy_mid_copy_failure() -> None:
    real_chunk, real_write = binary_io.COPY_CHUNK, binary_io._write_chunk
    calls = {"n": 0}

    def failing_write(handle, data):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(errno.ENOSPC, "no space left (forced)")
        real_write(handle, data)

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "model.onnx.partial"
        dst = Path(tmp) / "model.onnx"
        src.write_bytes(SRC_BYTES)
        binary_io.COPY_CHUNK = 16
        binary_io._write_chunk = failing_write
        try:
            with _ForcedFallback():
                binary_io.link_or_copy_new(src, dst)
        except OSError as exc:
            assert exc.errno == errno.ENOSPC, exc
        else:
            raise AssertionError("a failed copy must raise")
        finally:
            binary_io.COPY_CHUNK, binary_io._write_chunk = real_chunk, real_write
        assert calls["n"] == 2, calls
        assert not dst.exists(), "a failed copy must remove the dst it created"
        assert src.read_bytes() == SRC_BYTES
    print("link_or_copy_new: mid-copy ENOSPC propagates, dst removed, src intact")


def check_link_or_copy_on_collect_root() -> None:
    with collect_root(prefix="gvd-link-") as root:
        src = root / "model.onnx.partial"
        dst = root / "model.onnx"
        src.write_bytes(SRC_BYTES)
        binary_io.link_or_copy_new(src, dst)
        assert dst.read_bytes() == SRC_BYTES
        assert not src.exists()
        assert os.stat(dst).st_nlink == 1, os.stat(dst).st_nlink
    print("link_or_copy_new: unforced run on the collector data root ok")


def main() -> None:
    check_temp_verify_rename_before_state()
    check_crash_mid_write_leaves_nothing()
    check_failed_verify_and_no_replace()
    check_link_or_copy_refuses_existing()
    check_link_or_copy_success()
    check_link_or_copy_mid_copy_failure()
    check_link_or_copy_on_collect_root()
    print("frame atomic ok")


if __name__ == "__main__":
    main()
