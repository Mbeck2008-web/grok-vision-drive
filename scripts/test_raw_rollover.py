#!/usr/bin/env python3
"""Camera frames roll over into numbered subfolders; old flat recordings still load.

``raw/<cam>/<NNN>/<index>.png`` keeps every FAT32 folder far below its
65,536-entry limit. Recordings written before the subfolders keep
``raw/<cam>/<index>.png`` and every reader takes both.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from python.data import raw_frames  # noqa: E402
from python.data.raw_frames import (  # noqa: E402
    RAW_FRAMES_PER_DIR,
    RawFrameWriter,
    RawLayoutError,
    _used_indices,
    is_raw_frame_path,
    list_raw_images,
    load_raw_cameras,
    load_training_bgr,
    raw_camera_of,
    raw_frame_candidates,
    raw_image_counts,
    raw_layout_path,
    raw_subdir_name,
    read_raw_frames_per_dir,
    repaired_copies,
    row_has_decodable_frame,
    training_frame_bytes,
    write_raw_layout,
)
from python.debug.dataset_check import validate_dataset  # noqa: E402
from python.train.frame_cache import _read_cam  # noqa: E402

FAT32_DIR_ENTRIES = 65_536


def _img(value: int) -> np.ndarray:
    image = np.full((12, 16, 3), value, dtype=np.uint8)
    image[0, 0] = (1, 2, 3)
    return image


def _rows(root: Path) -> list[dict]:
    text = (root / "state.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def check_cap_fits_fat32() -> None:
    # Two entries per name (long-name entry + 8.3) on any driver, ``.`` and
    # ``..``, plus a .tmp per frame in flight (two entries each) for up to
    # 8 cameras x a queue of 64: still under the FAT32 folder limit.
    worst = 2 * RAW_FRAMES_PER_DIR + 2 + 2 * 8 * 64
    assert worst < FAT32_DIR_ENTRIES // 2, worst
    assert raw_subdir_name(0) == "000"
    assert raw_subdir_name(RAW_FRAMES_PER_DIR - 1) == "000"
    assert raw_subdir_name(RAW_FRAMES_PER_DIR) == "001"
    assert raw_subdir_name(123_456) == f"{123_456 // RAW_FRAMES_PER_DIR:03d}"
    assert raw_subdir_name(7, per_dir=3) == "002"
    print(f"cap {RAW_FRAMES_PER_DIR}: worst {worst} of {FAT32_DIR_ENTRIES} entries")


def check_writer_rolls_over_and_file_is_real() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = RawFrameWriter(root, [], per_dir=3)
        for n in range(7):
            writer.write({"main": _img(40 + n), "wide": np.full((3, 4), n, np.float32)}, t=1.0 + 0.1 * n)
        rows = _rows(root)
        assert [r["i"] for r in rows] == list(range(7))
        for row in rows:
            idx = int(row["i"])
            sub = f"{idx // 3:03d}"
            assert row["cams"]["main"]["file"] == f"raw/main/{sub}/{idx:06d}.png", row["cams"]["main"]
            assert row["cams"]["wide"]["file"] == f"raw/wide/{sub}/{idx:06d}.npy", row["cams"]["wide"]
            for meta in row["cams"].values():
                assert (root / meta["file"]).is_file(), meta["file"]
        folders = sorted(p.name for p in (root / "raw" / "main").iterdir())
        assert folders == ["000", "001", "002"], folders
        assert sorted(p.name for p in (root / "raw" / "main" / "002").iterdir()) == ["000006.png"]
        assert not list((root / "raw").rglob("*.tmp"))
        # A new writer on the same folder continues the index past the subfolders.
        again = RawFrameWriter(root, [], per_dir=3)
        assert again._n == 7, again._n
        assert _used_indices(root) == set(range(7))
    print("writer: raw/<cam>/<NNN>/<index> and cams.<cam>.file is the real path")


def check_default_writer_uses_000() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        RawFrameWriter(root, []).write({"main": _img(50)}, t=1.0)
        assert _rows(root)[0]["cams"]["main"]["file"] == "raw/main/000/000000.png"
        assert raw_frames.RAW_FRAMES_PER_DIR == RAW_FRAMES_PER_DIR
    print("default writer: raw/main/000/000000.png")


def _flat_recording(root: Path, n: int) -> None:
    """A recording written before subfolders: raw/<cam>/NNNNNN.png."""
    import cv2

    for cam in ("main", "narrow"):
        (root / "raw" / cam).mkdir(parents=True, exist_ok=True)
    lines = []
    for idx in range(n):
        cams = {}
        for cam in ("main", "narrow"):
            ok, buf = cv2.imencode(".png", _img(60 + idx))
            assert ok
            (root / "raw" / cam / f"{idx:06d}.png").write_bytes(buf.tobytes())
            cams[cam] = {"file": f"raw/{cam}/{idx:06d}.png", "t": 1.0 + idx * 0.1, "shape": [12, 16, 3], "dtype": "uint8"}
        lines.append(json.dumps({"i": idx, "t": 1.0 + idx * 0.1, "dt_s": None if idx == 0 else 0.1, "cams": cams}))
    (root / "state.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _check_readers(root: Path, n: int, label: str) -> None:
    images = list_raw_images(root)
    counts = raw_image_counts(images)
    assert counts["per_cam"] == {"main": n, "narrow": n}, (label, counts)
    assert counts["raw_frames"] == n, (label, counts)
    assert [int(p.stem) for p in images if raw_camera_of(p) == "main"] == list(range(n)), label
    for idx in range(n):
        frames = load_raw_cameras(root, idx)
        assert sorted(frames) == ["main", "narrow"], (label, idx, sorted(frames))
        assert int(frames["main"][5, 5, 0]) == 60 + idx, (label, idx)
        assert load_training_bgr(root, idx).ndim == 3
        assert row_has_decodable_frame(root, {"i": idx, "cams": {"main": {}}}), (label, idx)
        cam = _read_cam(root, "main", idx)
        assert cam is not None and int(cam[5, 5, 0]) == 60 + idx, (label, idx)
    assert not row_has_decodable_frame(root, {"i": n + 5}), label
    _bytes, frames = training_frame_bytes(root)
    assert frames == 2 * n, (label, frames)
    with tempfile.TemporaryDirectory() as out:
        report = validate_dataset(root, Path(out), montage_n=1)
    missing = [issue for issue in report["issues"] if "missing" in issue or "unreadable" in issue]
    assert not missing, (label, missing)
    assert report["per_cam"] == {"main": n, "narrow": n}, (label, report["per_cam"])


def check_flat_layout_still_reads() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _flat_recording(root, 4)
        _check_readers(root, 4, "flat")
    print("flat raw/<cam>/NNNNNN.png: loaders, validator, cache and counts unchanged")


def check_nested_layout_reads() -> None:
    import cv2

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = RawFrameWriter(root, [], per_dir=2)
        for idx in range(5):
            writer.write({"main": _img(60 + idx), "narrow": _img(60 + idx)}, t=1.0 + 0.1 * idx)
        # The loaders read per_dir=2 from raw_layout.json; nothing is listed.
        assert read_raw_frames_per_dir(root) == 2
        for idx in range(5):
            frames = load_raw_cameras(root, idx)
            assert sorted(frames) == ["main", "narrow"], (idx, sorted(frames))
            assert int(frames["main"][5, 5, 0]) == 60 + idx, idx
            cam = _read_cam(root, "narrow", idx)
            assert cam is not None and int(cam[5, 5, 0]) == 60 + idx, idx
        rows = _rows(root)
        for row in rows:
            assert row_has_decodable_frame(root, row), row["i"]
        images = list_raw_images(root)
        assert raw_image_counts(images)["per_cam"] == {"main": 5, "narrow": 5}
        _bytes, frames = training_frame_bytes(root)
        assert frames == 10, frames
        with tempfile.TemporaryDirectory() as out:
            report = validate_dataset(root, Path(out), montage_n=1)
        missing = [issue for issue in report["issues"] if "missing" in issue or "unreadable" in issue]
        assert not missing, missing
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = RawFrameWriter(root, [])
        for idx in range(4):
            writer.write({"main": _img(60 + idx), "narrow": _img(60 + idx)}, t=1.0 + 0.1 * idx)
        _check_readers(root, 4, "nested")
    print("nested raw/<cam>/<NNN>/NNNNNN.png: loaders, validator, cache and counts")


def check_mixed_layout_after_upgrade() -> None:
    """An old flat recording that a new writer appends to holds both layouts."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _flat_recording(root, 3)
        writer = RawFrameWriter(root, [])
        assert writer._n == 3
        writer.write({"main": _img(63), "narrow": _img(63)}, t=1.3)
        assert _rows(root)[-1]["cams"]["main"]["file"] == "raw/main/000/000003.png"
        _check_readers(root, 4, "mixed")
    print("mixed flat + subfolder recording reads as one")


def check_paths_and_repair_mirror() -> None:
    raw = Path("rec") / "raw"
    # No recording on disk, no raw_layout.json: a flat recording, one path.
    assert raw_frame_candidates(raw, "main", 12_345, ".png") == [raw / "main" / "012345.png"]
    flat = raw / "main" / "000001.png"
    nested = raw / "main" / "000" / "000001.png"
    assert raw_camera_of(flat) == "main" and raw_camera_of(nested) == "main"
    assert is_raw_frame_path(flat) and is_raw_frame_path(nested)
    assert not is_raw_frame_path(Path("rec") / "strips" / "000001.jpg")
    assert raw_camera_of(Path("rec") / "raw" / "main" / "notes" / "000001.png") is None
    assert Path("rec") / "raw_repaired" / "main" / "000001.png" in repaired_copies(flat)
    assert Path("rec") / "raw_repaired" / "main" / "000" / "000001.png" in repaired_copies(nested)
    print("paths: camera of either layout; raw_repaired mirrors the subfolder")


def _png(value: int) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", _img(value))
    assert ok
    return buf.tobytes()


def _snapshot(folder: Path) -> dict[str, tuple[int, int, int]]:
    """Name -> (size, mtime_ns, inode) of every file below ``folder``."""
    out = {}
    for path in folder.rglob("*"):
        if path.is_file():
            st = path.stat()
            out[str(path.relative_to(folder))] = (st.st_size, st.st_mtime_ns, st.st_ino)
    return out


def check_resume_across_folder_boundary() -> None:
    """A run that stopped at index 9,999 resumes in ``001/`` and keeps every frame.

    The last state line is lost on purpose (frame 9,999 is on disk, its
    state line is not), as after a crash between the two writes.
    """
    last = RAW_FRAMES_PER_DIR - 1
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        shard = root / "raw" / "main" / "000"
        shard.mkdir(parents=True)
        write_raw_layout(root, RAW_FRAMES_PER_DIR)  # what the stopped writer stored first
        blob = _png(70)
        lines = []
        for idx in range(last + 1):
            (shard / f"{idx:06d}.png").write_bytes(blob)
            if idx < last:
                cams = {"main": {"file": f"raw/main/000/{idx:06d}.png"}}
                lines.append(json.dumps({"i": idx, "t": 1.0 + idx * 0.1, "cams": cams}))
        (root / "state.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        before = _snapshot(root / "raw")
        assert len(before) == RAW_FRAMES_PER_DIR

        writer = RawFrameWriter(root, [])
        assert writer._n == RAW_FRAMES_PER_DIR, writer._n
        row = writer.write({"main": _img(71)}, t=9000.0)
        assert row["i"] == RAW_FRAMES_PER_DIR
        assert row["cams"]["main"]["file"] == "raw/main/001/010000.png", row["cams"]["main"]
        assert (root / "raw" / "main" / "001" / "010000.png").is_file()
        # A second resume continues in 001/.
        row = RawFrameWriter(root, []).write({"main": _img(72)}, t=9000.1)
        assert row["cams"]["main"]["file"] == "raw/main/001/010001.png", row["cams"]["main"]

        after = _snapshot(root / "raw")
        assert {k: after[k] for k in before} == before, "an existing frame changed"
        assert sorted(set(after) - set(before)) == [
            os.path.join("main", "001", "010000.png"),
            os.path.join("main", "001", "010001.png"),
        ]
        assert len(list(shard.iterdir())) == RAW_FRAMES_PER_DIR
        assert not list((root / "raw").rglob("*.tmp"))
        assert int(load_raw_cameras(root, last)["main"][5, 5, 0]) == 70
        assert int(load_raw_cameras(root, RAW_FRAMES_PER_DIR)["main"][5, 5, 0]) == 71
    print("resume at 9,999 -> raw/main/001/010000.png; all 10,000 old frames unchanged")


def check_writer_never_replaces_a_frame() -> None:
    """A writer pointed at an index that is already on disk refuses; nothing changes."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = RawFrameWriter(root, [], per_dir=3)
        for n in range(4):
            writer.write({"main": _img(80 + n)}, t=1.0 + 0.1 * n)
        before = _snapshot(root / "raw")
        state = (root / "state.jsonl").read_bytes()
        for idx in (0, 2, 3):
            writer._n = idx
            try:
                writer.write({"main": _img(99)}, t=5.0)
            except FileExistsError:
                pass
            else:
                raise AssertionError(f"writer replaced frame {idx}")
        assert _snapshot(root / "raw") == before
        assert (root / "state.jsonl").read_bytes() == state
        assert not list((root / "raw").rglob("*.tmp"))
    print("writer refuses an existing frame name: files and state.jsonl unchanged")


def _old_flat_counts(root: Path) -> dict:
    """``list_raw_images`` + ``raw_image_counts`` as they were before subfolders (fd48e2b)."""
    found = sorted(
        p
        for p in (root / "raw").rglob("*")
        if p.is_file() and p.suffix in {".png", ".npy"} and p.stem.isdigit()
    )
    per_cam: dict[str, int] = {}
    for path in found:
        per_cam[path.parent.name] = per_cam.get(path.parent.name, 0) + 1
    if "main" in per_cam:
        ticks = per_cam["main"]
    else:
        ticks = min(per_cam.values()) if per_cam else 0
    return {"raw_frames": ticks, "raw_files": sum(per_cam.values()), "per_cam": per_cam, "paths": found}


def check_rec5c_size_flat_then_mixed() -> None:
    """57,743 flat frames (rec5c's main camera) count as before; appended frames add up."""
    n_flat = 57_743
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        flat = root / "raw" / "main"
        flat.mkdir(parents=True)
        blob = _png(90)
        for idx in range(n_flat):
            (flat / f"{idx:06d}.png").write_bytes(blob)
        lines = [
            json.dumps({"i": idx, "t": 1.0 + idx * 0.1, "cams": {"main": {"file": f"raw/main/{idx:06d}.png"}}})
            for idx in range(n_flat)
        ]
        (root / "state.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

        old = _old_flat_counts(root)
        images = list_raw_images(root)
        counts = raw_image_counts(images)
        assert sorted(images) == old["paths"]
        assert counts == {k: old[k] for k in ("raw_frames", "raw_files", "per_cam")}, counts
        assert counts["raw_frames"] == n_flat and counts["per_cam"] == {"main": n_flat}

        # Loaders on the flat layout. A miss (no .npy, an index past the end)
        # costs a few stat calls: the 57,743-entry folder is never listed.
        listings = []
        real_scandir = os.scandir

        def counting_scandir(path):
            listings.append(str(path))
            return real_scandir(path)

        raw_frames.os.scandir = counting_scandir
        try:
            for idx in (0, 9_999, 10_000, 57_742):
                assert int(load_raw_cameras(root, idx)["main"][5, 5, 0]) == 90, idx
                cam = _read_cam(root, "main", idx)
                assert cam is not None and int(cam[5, 5, 0]) == 90, idx
                assert row_has_decodable_frame(root, {"i": idx, "cams": {"main": {}}}), idx
            for idx in range(n_flat, n_flat + 50):
                assert load_raw_cameras(root, idx) == {}, idx
                assert _read_cam(root, "main", idx) is None, idx
        finally:
            raw_frames.os.scandir = real_scandir
        assert listings.count(str(flat)) == 0, len(listings)

        # A new writer appends to it: mixed flat + raw/main/005/.
        writer = RawFrameWriter(root, [])
        assert writer._n == n_flat, writer._n
        for k in range(3):
            row = writer.write({"main": _img(91 + k)}, t=9000.0 + k)
            assert row["cams"]["main"]["file"] == f"raw/main/005/{n_flat + k:06d}.png", row["cams"]["main"]
        counts = raw_image_counts(list_raw_images(root))
        assert counts["per_cam"] == {"main": n_flat + 3} and counts["raw_frames"] == n_flat + 3, counts
        assert counts["raw_files"] == n_flat + 3, counts
        assert len([p for p in flat.iterdir() if p.is_file()]) == n_flat
        assert int(load_raw_cameras(root, n_flat - 1)["main"][5, 5, 0]) == 90
        assert int(load_raw_cameras(root, n_flat)["main"][5, 5, 0]) == 91
        cam = _read_cam(root, "main", n_flat + 2)
        assert cam is not None and int(cam[5, 5, 0]) == 93
    print(f"flat {n_flat:,} frames: counts equal the old code; + 3 nested = {n_flat + 3:,}")


def _count_listings():
    """Patch os.scandir / os.listdir; returns (calls list, restore)."""
    calls: list[str] = []
    real_scandir, real_listdir = os.scandir, os.listdir

    def scandir(path="."):
        calls.append(str(path))
        return real_scandir(path)

    def listdir(path="."):
        calls.append(str(path))
        return real_listdir(path)

    os.scandir, os.listdir = scandir, listdir

    def restore() -> None:
        os.scandir, os.listdir = real_scandir, real_listdir

    return calls, restore


def check_new_subfolder_found_with_frozen_mtime() -> None:
    """A frame in a new 00N/ is found even when no folder mtime moved (FAT32: 2 s steps, dirs stale)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = RawFrameWriter(root, [], per_dir=3)
        for n in range(3):
            writer.write({"main": _img(100 + n)}, t=1.0 + 0.1 * n)
        old = 1_000_000_000
        cam = root / "raw" / "main"
        for path in (root, root / "raw", cam, cam / "000"):
            os.utime(path, (old, old))
        # A reader looks before the next folder exists: a miss.
        assert load_raw_cameras(root, 3) == {}
        assert _read_cam(root, "main", 3) is None
        writer.write({"main": _img(103)}, t=1.3)
        assert (cam / "001" / "000003.png").is_file()
        for path in (root, root / "raw", cam, cam / "000", cam / "001"):
            os.utime(path, (old, old))
        assert cam.stat().st_mtime_ns == old * 10**9
        calls, restore = _count_listings()
        try:
            assert int(load_raw_cameras(root, 3)["main"][5, 5, 0]) == 103
            cam_img = _read_cam(root, "main", 3)
            assert cam_img is not None and int(cam_img[5, 5, 0]) == 103
            assert row_has_decodable_frame(root, {"i": 3, "cams": {"main": {}}})
        finally:
            restore()
        assert str(cam) not in calls, calls
    print("new 001/ found with every folder mtime frozen; camera folder not listed")


def check_non_default_size_from_metadata() -> None:
    """per_dir=7 is stored in raw_layout.json and every reader computes the path from it."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = RawFrameWriter(root, [], per_dir=7)
        for n in range(16):
            writer.write({"main": _img(n), "narrow": _img(n)}, t=1.0 + 0.1 * n)
        layout = json.loads(raw_layout_path(root).read_text(encoding="utf-8"))
        assert layout["raw_frames_per_dir"] == 7, layout
        assert sorted(p.name for p in (root / "raw" / "main").iterdir()) == ["000", "001", "002"]
        raw = root / "raw"
        assert raw_frame_candidates(raw, "main", 15, ".png") == [raw / "main" / "002" / "000015.png", raw / "main" / "000015.png"]
        assert raw_frame_candidates(raw, "main", 15, ".png") != raw_frame_candidates(raw, "main", 15, ".npy")
        calls, restore = _count_listings()
        try:
            for idx in range(16):
                assert int(load_raw_cameras(root, idx)["narrow"][5, 5, 0]) == idx, idx
                cam = _read_cam(root, "main", idx)
                assert cam is not None and int(cam[5, 5, 0]) == idx, idx
        finally:
            restore()
        assert not [c for c in calls if c.startswith(str(raw / "main")) or c.startswith(str(raw / "narrow"))], calls
        assert raw_image_counts(list_raw_images(root))["per_cam"] == {"main": 16, "narrow": 16}
        # Resuming keeps the stored size; asking for another one is refused.
        assert RawFrameWriter(root, [])._n == 16
        assert RawFrameWriter(root, []).per_dir == 7
        try:
            RawFrameWriter(root, [], per_dir=10)
        except ValueError as exc:
            assert "raw_frames_per_dir=7" in str(exc), exc
        else:
            raise AssertionError("a different per_dir was accepted")
    print("per_dir=7 read from raw_layout.json; no camera folder listed")


def check_unsized_subfolders_refused() -> None:
    """Numbered subfolders without raw_layout.json: every reader and the writer raise."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = RawFrameWriter(root, [])
        for n in range(3):
            writer.write({"main": _img(n), "narrow": _img(n)}, t=1.0 + 0.1 * n)
        raw_layout_path(root).unlink()
        before = _snapshot(root / "raw")
        attempts = {
            "load_raw_cameras": lambda: load_raw_cameras(root, 1),
            "load_raw_cameras(miss)": lambda: load_raw_cameras(root, 9),
            "_read_cam": lambda: _read_cam(root, "main", 1),
            "row_has_decodable_frame": lambda: row_has_decodable_frame(root, {"i": 1, "cams": {"main": {}}}),
            "list_raw_images": lambda: list_raw_images(root),
            "training_frame_bytes": lambda: training_frame_bytes(root),
            "RawFrameWriter": lambda: RawFrameWriter(root, []),
        }
        for label, call in attempts.items():
            try:
                call()
            except RawLayoutError as exc:
                text = str(exc)
                assert str(root) in text and "raw_layout.json" in text, (label, text)
                assert "'main'" in text or "'narrow'" in text, (label, text)
            else:
                raise AssertionError(f"{label} read unsized subfolders without an error")
        with tempfile.TemporaryDirectory() as out:
            try:
                validate_dataset(root, Path(out), montage_n=1)
            except RawLayoutError:
                pass
            else:
                raise AssertionError("validate_dataset read unsized subfolders without an error")
        assert not raw_layout_path(root).exists(), "the refusing writer stored a guessed size"
        assert _snapshot(root / "raw") == before
    print("numbered subfolders + no raw_layout.json: loaders, counts, validator and writer refuse")


def check_layout_written_before_frame_zero() -> None:
    """raw_layout.json exists, complete, before the first frame byte is written."""
    seen: list[tuple[str, int | None]] = []
    real = raw_frames._write_bytes

    def spy(path, payload):
        root = Path(path).parents[3]
        seen.append((Path(path).name, read_raw_frames_per_dir(root)))
        return real(path, payload)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = RawFrameWriter(root, [], per_dir=4)
        assert read_raw_frames_per_dir(root) == 4
        assert not list((root / "raw").rglob("*.png")), "a frame before any write"
        assert not list(root.glob("*.tmp")), "layout temp file left behind"
        raw_frames._write_bytes = spy
        try:
            writer.write({"main": _img(1)}, t=1.0)
        finally:
            raw_frames._write_bytes = real
        assert seen == [("000000.png", 4)], seen
    print("raw_layout.json is in place before frame 0 is written")


def main() -> None:
    check_cap_fits_fat32()
    check_writer_rolls_over_and_file_is_real()
    check_default_writer_uses_000()
    check_flat_layout_still_reads()
    check_nested_layout_reads()
    check_mixed_layout_after_upgrade()
    check_paths_and_repair_mirror()
    check_resume_across_folder_boundary()
    check_writer_never_replaces_a_frame()
    check_rec5c_size_flat_then_mixed()
    check_new_subfolder_found_with_frozen_mtime()
    check_non_default_size_from_metadata()
    check_unsized_subfolders_refused()
    check_layout_written_before_frame_zero()
    print("test_raw_rollover OK")


if __name__ == "__main__":
    main()
