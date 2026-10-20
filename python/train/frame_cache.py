"""Decoded camera cache for scene-net training.

Each chunk is one ``.npy`` under 3.5 GB so a FAT32 data disk (4 GB per file)
can hold it. The loader memory-maps the chunk and copies one frame out.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from python.data.strip_writer import CANVAS_H
from python.train.cam_strip import canvas_width, parse_cams, scaled_layout, sectors_for

# Stay under the 4 GB FAT32 file limit with a little room for the header.
CHUNK_BYTES = 3_500_000_000
MANIFEST = "manifest.json"
# Free space the cache drive must keep (``--cache-min-free-gb``).
CACHE_MIN_FREE_GB = 8.0


class CacheMismatchError(ValueError):
    """The cache was not built from this strips folder's current state.jsonl."""


def _norm_path(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def state_fingerprint(strip_root: Path | str) -> dict[str, Any]:
    """Strips path, state.jsonl row count and SHA-256: what a cache is keyed to."""
    state = Path(strip_root) / "state.jsonl"
    digest = hashlib.sha256()
    rows = 0
    with open(state, "rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                rows += 1
    return {"strips": _norm_path(strip_root), "rows": rows, "state_sha256": digest.hexdigest()}


def check_cache_source(cache_root: Path | str, strip_root: Path | str) -> dict[str, Any]:
    """Refuse a cache built from another folder, or from an older or newer state.jsonl.

    The cache is keyed by row index only, so a cache from another recording
    would be read silently. A manifest without a ``source`` block (built
    before this check) is refused too: rebuild it into a new folder.
    """
    manifest = Path(cache_root) / MANIFEST
    if not manifest.is_file():
        raise CacheMismatchError(f"{cache_root}: no {MANIFEST}; build it with --build-cache")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    source = payload.get("source")
    if not isinstance(source, dict):
        raise CacheMismatchError(
            f"{cache_root}: manifest has no source (strips path, rows, state.jsonl hash); "
            "rebuild it into a new folder with --build-cache"
        )
    here = state_fingerprint(strip_root)
    for key in ("strips", "rows", "state_sha256"):
        if source.get(key) != here[key]:
            raise CacheMismatchError(
                f"{cache_root}: cache {key}={source.get(key)!r} but {strip_root} has {here[key]!r}; "
                "rebuild the cache into a new folder"
            )
    return here


def _sector_shape(cam_id: str) -> tuple[int, int, int]:
    sec = sectors_for((cam_id,))[0]
    return (CANVAS_H, int(sec.width), 3)


def frames_per_chunk(height: int, width: int) -> int:
    frame = max(1, int(height) * int(width) * 3)
    return max(1, CHUNK_BYTES // frame)


class FrameCache:
    """Read-only map of frame index to a packed camera canvas."""

    def __init__(self, root: Path | str, strips_root: Path | str | None = None) -> None:
        """``strips_root`` set: refuse (``CacheMismatchError``) a cache from another recording."""
        self.root = Path(root)
        if strips_root is not None:
            check_cache_source(self.root, strips_root)
        self.cams: tuple[str, ...] = ()
        self.canvas_w = 0
        self._open: dict[str, np.ndarray] = {}
        self._by_index: dict[int, list[tuple[str, str, int]]] = {}
        manifest = self.root / MANIFEST
        if not manifest.is_file():
            return
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.cams = parse_cams(",".join(payload.get("cams") or []))
        self.scale = float(payload.get("scale") or 1.0)
        self.sectors, _height, self.canvas_w = scaled_layout(self.cams, self.scale)
        for chunk in payload.get("chunks") or []:
            name = str(chunk.get("file") or "")
            cam = str(chunk.get("cam") or "")
            for slot, raw in enumerate(chunk.get("ids") or []):
                try:
                    index = int(raw)
                except (TypeError, ValueError):
                    continue
                self._by_index.setdefault(index, []).append((name, cam, slot))

    def __getstate__(self) -> dict[str, Any]:
        """Drop memory maps. A spawned worker reopens them on the first read."""
        state = dict(self.__dict__)
        state["_open"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open = {}

    def has(self, index: int) -> bool:
        return int(index) in self._by_index

    def canvas(self, index: int) -> np.ndarray | None:
        parts = self._by_index.get(int(index))
        if not parts:
            return None
        by_cam = {cam: (name, slot) for name, cam, slot in parts}
        height = CANVAS_H if self.scale >= 0.999 else max(1, int(round(CANVAS_H * self.scale)))
        image = np.zeros((height, self.canvas_w, 3), dtype=np.uint8)
        for sec in self.sectors:
            found = by_cam.get(sec.cam_id)
            if found is None:
                continue
            name, slot = found
            pixels = self._pixels(name)
            if pixels is None or slot >= int(pixels.shape[0]):
                continue
            image[:, sec.x0 : sec.x1] = pixels[slot]
        return image

    def _pixels(self, name: str) -> np.ndarray | None:
        cached = self._open.get(name)
        if cached is not None:
            return cached
        path = self.root / name
        if not path.is_file():
            return None
        array = np.load(path, mmap_mode="r")
        self._open[name] = array
        return array


def _fit_sector(image: np.ndarray, width: int) -> np.ndarray:
    from python.data.strip_writer import _fit

    return _fit(image, width, CANVAS_H)


def _read_cam(root: Path, cam: str, index: int) -> np.ndarray | None:
    from python.data.raw_frames import read_image_with_repair

    stem = f"{int(index):06d}"
    for suffix in (".png", ".npy"):
        path = root / "raw" / cam / f"{stem}{suffix}"
        if suffix == ".npy" and path.is_file():
            array = np.load(path)
            if isinstance(array, np.ndarray) and array.ndim >= 2:
                return array
            continue
        if path.is_file():
            image = read_image_with_repair(path)
            if image is not None:
                return image
    return None


def build_frame_cache(
    strip_root: Path | str,
    dest: Path | str,
    cams: tuple[str, ...] | str,
    *,
    scale: float = 1.0,
    progress: Any = None,
) -> dict[str, Any]:
    """Decode ``cams`` at sector size into chunked ``.npy`` files under ``dest``."""
    from python.data.raw_frames import read_state_jsonl

    root = Path(strip_root)
    source = state_fingerprint(root)
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    names = parse_cams(cams if isinstance(cams, str) else ",".join(cams))
    factor = 1.0 if float(scale) >= 0.999 else float(scale)
    rows = read_state_jsonl(root / "state.jsonl")
    indexes: list[int] = []
    for row in rows:
        try:
            indexes.append(int(row.get("i")))
        except (TypeError, ValueError):
            continue
    chunks: list[dict[str, Any]] = []
    written = 0
    for cam in names:
        height, width, _channels = _sector_shape(cam)
        if factor < 0.999:
            height = max(1, int(round(height * factor)))
            width = max(1, int(round(width * factor)))
        limit = frames_per_chunk(height, width)
        batch_ids: list[int] = []
        batch_px: list[np.ndarray] = []

        def flush() -> None:
            nonlocal written
            if not batch_ids:
                return
            stack = np.stack(batch_px, axis=0)
            file_name = f"{cam}_{len(chunks):03d}.npy"
            np.save(out / file_name, stack)
            chunks.append({"file": file_name, "cam": cam, "ids": list(batch_ids)})
            written += len(batch_ids)
            batch_ids.clear()
            batch_px.clear()
            if progress is not None:
                progress(written)

        for index in indexes:
            image = _read_cam(root, cam, index)
            if image is None:
                continue
            fitted = _fit_sector(np.asarray(image), int(_sector_shape(cam)[1]))
            if factor < 0.999:
                import cv2

                fitted = cv2.resize(fitted, (width, height), interpolation=cv2.INTER_AREA)
            batch_ids.append(index)
            batch_px.append(np.ascontiguousarray(fitted[:, :, :3]))
            if len(batch_ids) >= limit:
                flush()
        flush()
    manifest = {
        "cams": list(names),
        "canvas_h": CANVAS_H if factor >= 0.999 else max(1, int(round(CANVAS_H * factor))),
        "canvas_w": canvas_width(names) if factor >= 0.999 else max(1, int(round(canvas_width(names) * factor))),
        "scale": factor,
        "chunks": chunks,
        "frames": written // max(1, len(names)),
        "source": source,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
