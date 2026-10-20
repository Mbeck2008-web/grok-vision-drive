#!/usr/bin/env python3
"""PNG writes stay binary, and text-mode captures can be repaired beside the originals."""

from __future__ import annotations

import contextlib
import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from python.data.binary_io import PNG_SIG  # noqa: E402
from python.data.png_repair import (  # noqa: E402
    crc32_reverse,
    png_chunks_ok,
    png_decodes,
    repair_dataset,
    repair_png_bytes,
    undo_inserted_cr,
)
from python.data.raw_frames import (  # noqa: E402
    RawFrameWriter,
    _png_bytes,
    load_training_bgr,
    read_image_file,
    read_image_with_repair,
)
from python.data.strip_writer import StripWriter, compose_strip, encode_jpeg  # noqa: E402
from python.debug.dataset_check import validate_dataset  # noqa: E402
from python.debug.lane_targets import tech_from_hit  # noqa: E402
from scripts.repair_text_mode_png import main as repair_main  # noqa: E402


def _dumb(data: bytes) -> bytes:
    return data.replace(b"\n", b"\r\n")


def _smart(data: bytes) -> bytes:
    """Insert CR before LF only when the previous output byte is not already CR."""
    out = bytearray()
    for byte in data:
        if byte == 0x0A and (not out or out[-1] != 0x0D):
            out.append(0x0D)
        out.append(byte)
    return bytes(out)


def _chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def _png_with_text(text: bytes) -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw = bytes([0, 17, 34, 51])
    idat = zlib.compress(raw, 0)
    if b"\n" in idat or b"\r" in idat:
        raise AssertionError("fixture IDAT contains a newline")
    return PNG_SIG + _chunk(b"IHDR", ihdr) + _chunk(b"tEXt", b"Comment\x00" + text) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def check_crc_reverse_and_dumb_inverse() -> None:
    blob = b"hello world, png chunk body\r\nmore\n"
    full = zlib.crc32(blob) & 0xFFFFFFFF
    cut = 11
    head, tail = blob[:cut], blob[cut:]
    head_crc = zlib.crc32(head) & 0xFFFFFFFF
    assert (zlib.crc32(tail, head_crc) & 0xFFFFFFFF) == full
    assert crc32_reverse(tail, full) == head_crc
    original = b"A\r\nB\nC\rD\r\n"
    assert undo_inserted_cr(_dumb(original)) == original
    print("crc reverse and dumb inverse ok")


def check_dumb_png_repairs_exact() -> None:
    import cv2

    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(32, 40, 3), dtype=np.uint8)
    encoded = _png_bytes(image)
    assert encoded is not None and encoded.startswith(PNG_SIG)
    corrupt = _dumb(encoded)
    assert corrupt.startswith(b"\x89PNG\r\r\n\x1a")
    assert not corrupt.startswith(PNG_SIG)
    result = repair_png_bytes(corrupt)
    assert result.ok and result.method == "drop_cr" and result.data == encoded
    assert png_chunks_ok(result.data)
    assert png_decodes(result.data)
    back = cv2.imdecode(np.frombuffer(result.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert back is not None and np.array_equal(back, image)
    print(f"dumb repair exact {len(encoded)} -> {len(corrupt)}")


def check_crc_search_keeps_original_cr() -> None:
    # Lone LFs become an inserted CR. Existing CRLF stays one CR. Drop-all
    # would delete that original CR, so the chunk CRC has to choose.
    text = b"a\nb\r\n" * 12
    original = _png_with_text(text)
    assert png_chunks_ok(original) and png_decodes(original)
    corrupt = _smart(original)
    assert undo_inserted_cr(corrupt) != original
    assert not png_chunks_ok(undo_inserted_cr(corrupt))
    result = repair_png_bytes(corrupt)
    assert result.ok and result.method == "crc_search", result.method
    assert result.data == original
    assert png_decodes(result.data)
    print(f"crc search restored {corrupt.count(bytes([13]))} CR decisions")


def check_dataset_copies_do_not_touch_originals() -> None:
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[:, :, 1] = 40
    image[0, 0] = (8, 16, 24)
    encoded = _png_bytes(image)
    assert encoded is not None
    tech = tech_from_hit(0.2, 0.0, 3.5, False)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = RawFrameWriter(root, [])
        writer.write({"main": image}, t=1.0, tech=tech, driver="beamng_ai")
        good = root / "raw" / "main" / "000" / "000000.png"
        original_good = good.read_bytes()
        assert original_good == encoded
        bad = root / "raw" / "main" / "000" / "000001.png"
        bad.write_bytes(_dumb(encoded))
        junk = root / "raw" / "main" / "000" / "000002.png"
        junk.write_bytes(b"not a png")
        before = {path.name: path.read_bytes() for path in (root / "raw" / "main" / "000").glob("*.png")}
        code = repair_main([str(root)])
        assert code == 1
        after = {path.name: path.read_bytes() for path in (root / "raw" / "main" / "000").glob("*.png")}
        assert after == before
        assert not (root / "raw_repaired" / "main" / "000" / "000000.png").exists()
        repaired = root / "raw_repaired" / "main" / "000" / "000001.png"
        assert repaired.is_file()
        assert repaired.read_bytes() == encoded
        assert not (root / "raw_repaired" / "main" / "000" / "000002.png").exists()
        assert bad.read_bytes().startswith(b"\x89PNG\r\r\n\x1a")
        assert read_image_file(bad) is None
        loaded = read_image_with_repair(bad)
        assert loaded is not None and np.array_equal(loaded, image)
        canvas = load_training_bgr(root, 1)
        assert canvas.shape[2] == 3
        try:
            load_training_bgr(root, 2)
        except FileNotFoundError as exc:
            assert ".jpg" not in str(exc)
            assert "raw" in str(exc)
        else:
            raise AssertionError("undecodable raw frame fell through")
        import cv2

        real_imread = cv2.imread

        def refuse_jpeg(path, flags=cv2.IMREAD_COLOR):
            if str(path).lower().endswith(".jpg"):
                raise AssertionError(f"validator opened {path}")
            return real_imread(path, flags)

        cv2.imread = refuse_jpeg
        try:
            report = validate_dataset(root, root / "validate_out", montage_n=1)
        finally:
            cv2.imread = real_imread
        text = "\n".join(report["issues"])
        assert ".jpg" not in text
        assert "strips" not in text
        assert any(item.startswith("unreadable 000002.png") for item in report["issues"])
        assert not any("000001" in item and "unreadable" in item for item in report["issues"])
        assert not any("000000" in item and "unreadable" in item for item in report["issues"])
        report2 = repair_dataset(root, dry_run=True)
        assert report2["rescued"] == 2 and report2["total"] == 3
        assert abs(float(report2["fraction"]) - (2.0 / 3.0)) < 1e-9
        print(f"dataset fraction {report2['fraction']:.4f}")


def check_validator_without_strips() -> None:
    tech = tech_from_hit(0.2, 0.0, 3.5, False)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = RawFrameWriter(root, [])
        colour = np.full((8, 12, 3), 30, dtype=np.uint8)
        writer.write({"main": colour}, t=3.0, tech=tech, driver="beamng_ai")
        import shutil

        shutil.rmtree(root / "raw")
        report = validate_dataset(root, root / "validate_out", montage_n=1)
        text = "\n".join(report["issues"])
        assert report["jpegs"] == 0
        assert ".jpg" not in text
        assert "no raw frames and no strips" in text
        assert "missing jpeg" not in text
        print(text)


@contextlib.contextmanager
def _windows_text_writes(binary_value: int):
    """Translate ``\\n`` to ``\\r\\n`` on writes whose flags omit ``O_BINARY``."""
    real_open = os.open
    real_write = os.write
    real_close = os.close
    had_binary = hasattr(os, "O_BINARY")
    previous = getattr(os, "O_BINARY", None)
    text_fds: set[int] = set()
    opened: list[tuple[str, int]] = []

    def open2(path, flags, mode=0o777, *, dir_fd=None):
        opened.append((str(path), int(flags)))
        binary = bool(int(flags) & int(getattr(os, "O_BINARY", 0)))
        real_flags = int(flags)
        if binary_value:
            real_flags &= ~int(binary_value)
        kwargs = {}
        if dir_fd is not None:
            kwargs["dir_fd"] = dir_fd
        fd = real_open(path, real_flags, mode, **kwargs)
        if not binary and (int(flags) & (os.O_WRONLY | os.O_RDWR)):
            text_fds.add(fd)
        return fd

    def write2(fd, data):
        buf = bytes(data)
        if fd in text_fds:
            expanded = buf.replace(b"\n", b"\r\n")
            wrote = real_write(fd, expanded)
            if wrote < len(expanded):
                raise OSError("partial text write")
            return len(buf)
        return real_write(fd, buf)

    def close2(fd):
        text_fds.discard(fd)
        return real_close(fd)

    os.O_BINARY = binary_value
    os.open = open2
    os.write = write2
    os.close = close2
    try:
        yield opened
    finally:
        os.open = real_open
        os.write = real_write
        os.close = real_close
        if had_binary:
            os.O_BINARY = previous
        else:
            try:
                del os.O_BINARY
            except AttributeError:
                pass


def check_windows_newline_regression() -> None:
    """A text-mode os.write must not be able to change PNG or JPEG bytes."""
    colour = np.zeros((8, 12, 3), dtype=np.uint8)
    colour[:, :, 0] = np.arange(12, dtype=np.uint8)
    colour[:, :, 1] = 9
    colour[:, :, 2] = 200
    odd = np.arange(12, dtype=np.float32).reshape(3, 4)
    encoded = _png_bytes(colour)
    assert encoded is not None
    sentinel = 1 << 15
    with tempfile.TemporaryDirectory() as tmp:
        with _windows_text_writes(sentinel) as opened:
            root = Path(tmp)
            writer = RawFrameWriter(root, [])
            writer.write({"main": colour, "wide": odd}, t=1.0)
            png = (root / "raw" / "main" / "000" / "000000.png").read_bytes()
            npy = (root / "raw" / "wide" / "000" / "000000.npy").read_bytes()
            assert png == encoded
            assert png.startswith(PNG_SIG)
            import io

            buf = io.BytesIO()
            np.save(buf, odd)
            assert npy == buf.getvalue()
            from python.data.raw_queue import RawWriteQueue

            queue = RawWriteQueue(writer, max_depth=2, workers=1)
            assert queue.submit({"main": colour}, t=2.0)
            queue.close(timeout_s=10)
            queued = (root / "raw" / "main" / "000" / "000001.png").read_bytes()
            assert queued == encoded
            strips = root / "strips_out"
            strip = StripWriter(strips)
            record = strip.write({"main": colour}, t=4.0)
            jpeg = record.path.read_bytes()
            assert jpeg == encode_jpeg(compose_strip({"main": colour}))
            assert jpeg.startswith(b"\xff\xd8")
            watched = (".png", ".jpg", ".npy", ".jsonl")
            for name, flags in opened:
                if name.lower().endswith(watched):
                    assert flags & sentinel, f"{name} opened without O_BINARY ({flags:#x})"
    print("windows newline regression ok")


def check_text_mode_fails_loud() -> None:
    colour = np.zeros((6, 6, 3), dtype=np.uint8)
    colour[0, 0] = (1, 2, 3)
    with tempfile.TemporaryDirectory() as tmp:
        with _windows_text_writes(0):
            writer = RawFrameWriter(Path(tmp), [])
            try:
                writer.write({"main": colour}, t=1.0)
            except OSError as exc:
                text = str(exc).lower()
                assert "text mode" in text
                assert "signature" in text or "newline" in text
            else:
                raise AssertionError("text-mode PNG write should fail")
    print("text-mode write fails loud")


def main() -> None:
    check_crc_reverse_and_dumb_inverse()
    check_dumb_png_repairs_exact()
    check_crc_search_keeps_original_cr()
    check_dataset_copies_do_not_touch_originals()
    check_validator_without_strips()
    check_windows_newline_regression()
    check_text_mode_fails_loud()
    print("png repair ok")


if __name__ == "__main__":
    main()
