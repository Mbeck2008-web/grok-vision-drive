#!/usr/bin/env python3
"""Camera frames roll over into numbered subfolders; old flat recordings still load.

``raw/<cam>/<NNN>/<index>.png`` keeps every FAT32 folder far below its
65,536-entry limit. Recordings written before the subfolders keep
``raw/<cam>/<index>.png`` and every reader takes both.
"""
from __future__ import annotations

import json
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
    _used_indices,
    is_raw_frame_path,
    list_raw_images,
    load_raw_cameras,
    load_training_bgr,
    raw_camera_of,
    raw_frame_candidates,
    raw_image_counts,
    raw_subdir_name,
    repaired_copies,
    row_has_decodable_frame,
    training_frame_bytes,
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
        # The loaders do not know per_dir=2. They search the numbered
        # subfolders when the default subfolder and the flat name miss.
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
    # No such folder on disk: only the two computed paths.
    assert raw_frame_candidates(raw, "main", 12_345, ".png") == [
        raw / "main" / raw_subdir_name(12_345) / "012345.png",
        raw / "main" / "012345.png",
    ]
    flat = raw / "main" / "000001.png"
    nested = raw / "main" / "000" / "000001.png"
    assert raw_camera_of(flat) == "main" and raw_camera_of(nested) == "main"
    assert is_raw_frame_path(flat) and is_raw_frame_path(nested)
    assert not is_raw_frame_path(Path("rec") / "strips" / "000001.jpg")
    assert raw_camera_of(Path("rec") / "raw" / "main" / "notes" / "000001.png") is None
    assert Path("rec") / "raw_repaired" / "main" / "000001.png" in repaired_copies(flat)
    assert Path("rec") / "raw_repaired" / "main" / "000" / "000001.png" in repaired_copies(nested)
    print("paths: camera of either layout; raw_repaired mirrors the subfolder")


def main() -> None:
    check_cap_fits_fat32()
    check_writer_rolls_over_and_file_is_real()
    check_default_writer_uses_000()
    check_flat_layout_still_reads()
    check_nested_layout_reads()
    check_mixed_layout_after_upgrade()
    check_paths_and_repair_mirror()
    print("test_raw_rollover OK")


if __name__ == "__main__":
    main()
