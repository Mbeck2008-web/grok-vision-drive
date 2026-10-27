"""Decoded camera cache for scene-net training.

Each chunk is one ``.npy`` under 3.5 GB so a FAT32 data disk (4 GB per file)
can hold it. The loader memory-maps the chunk and copies one frame out.

Format ``gvd-frame-cache-2``: every sector is ``strip_writer.place_sector``
(fit, then the repeater ego-edge mask), the same call the live strip makes,
so a cached sector is byte-equal to ``compose_strip``'s. Format 1 kept the
real pixels in the repeater ego band that live writes as 0. A cache without
the version (or with a different one, or whose ``manifest.sha256`` does not
match) is stale: ``FrameCache.stale`` is True, it serves nothing, and the
trainer refuses it. A build is create-only: it goes into a new (or empty)
folder, every file is created exclusively, the manifest is written last and
its sha256 beside it. An existing cache is never overwritten in place.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from python.data.strip_writer import CANVAS_H
from python.train.cam_strip import canvas_width, parse_cams, scaled_layout, sectors_for

# Stay under the 4 GB FAT32 file limit with a little room for the header.
CHUNK_BYTES = 3_500_000_000
MANIFEST = "manifest.json"
MANIFEST_SHA = "manifest.sha256"
CACHE_FORMAT = "gvd-frame-cache-2"


class StaleCacheError(RuntimeError):
    """The cache folder is not this format. Rebuild it into a new folder."""


def _sector_shape(cam_id: str) -> tuple[int, int, int]:
    sec = sectors_for((cam_id,))[0]
    return (CANVAS_H, int(sec.width), 3)


def frames_per_chunk(height: int, width: int) -> int:
    frame = max(1, int(height) * int(width) * 3)
    return max(1, CHUNK_BYTES // frame)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FrameCache:
    """Read-only map of frame index to a packed camera canvas.

    Contract: every returned array is a copy; after close() no map is held.
    ``sector()`` and ``canvas()`` copy out of the chunk memory maps, so a caller
    may keep what they return after ``close()`` (or after the chunk is renamed or
    deleted). ``close()`` drops the maps; a read after it raises ``ValueError``.
    A pickled open cache (a spawned worker) reopens its maps on the first read.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.cams: tuple[str, ...] = ()
        self.canvas_w = 0
        self.scale = 1.0
        self.sectors: tuple[Any, ...] = ()
        self.stale = False
        self.reason = ""
        self.format: str | None = None
        self.manifest_sha256: str | None = None
        self._open: dict[str, np.ndarray] = {}
        self._closed = False
        self._by_index: dict[int, list[tuple[str, str, int]]] = {}
        manifest = self.root / MANIFEST
        if not manifest.is_file():
            return
        data = manifest.read_bytes()
        self.manifest_sha256 = _sha256(data)
        payload = json.loads(data.decode("utf-8"))
        self.format = payload.get("format")
        if self.format != CACHE_FORMAT:
            self.stale = True
            self.reason = (
                f"{self.root} is cache format {self.format or '1 (unversioned)'}, not {CACHE_FORMAT}: "
                "its repeater sectors keep the ego band live paints black. Rebuild it with "
                "--build-cache into a NEW folder (an old cache is never rewritten in place)."
            )
            return
        sha_file = self.root / MANIFEST_SHA
        want = sha_file.read_text(encoding="utf-8").split()[0] if sha_file.is_file() else ""
        if want != self.manifest_sha256:
            self.stale = True
            self.reason = f"{self.root}: {MANIFEST} sha256 does not match {MANIFEST_SHA}; rebuild into a new folder"
            return
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

    def require_fresh(self) -> None:
        """Raise ``StaleCacheError`` for a stale cache."""
        if self.stale:
            raise StaleCacheError(self.reason)

    def __getstate__(self) -> dict[str, Any]:
        """Drop memory maps. A spawned worker reopens them on the first read."""
        state = dict(self.__dict__)
        state["_open"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__dict__.setdefault("_closed", False)
        self._open = {}

    def close(self) -> None:
        """Drop every chunk memory map. Idempotent; reads after it raise ``ValueError``."""
        self._open = {}
        self._closed = True

    def __enter__(self) -> "FrameCache":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._closed:
            raise ValueError("FrameCache is closed")

    def has(self, index: int) -> bool:
        return int(index) in self._by_index

    def sector(self, cam: str, index: int) -> np.ndarray | None:
        """One camera's cached sector for tick ``index`` (a copy), or None."""
        self._check_open()
        for name, part_cam, slot in self._by_index.get(int(index)) or ():
            if part_cam != cam:
                continue
            pixels = self._pixels(name)
            if pixels is None or slot >= int(pixels.shape[0]):
                return None
            return np.array(pixels[slot], copy=True)
        return None

    def canvas(self, index: int) -> np.ndarray | None:
        """The packed canvas for tick ``index`` (a new array), or None."""
        self._check_open()
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


def _fit_sector(image: np.ndarray, cam: str) -> np.ndarray:
    from python.data.strip_writer import place_sector

    return place_sector(cam, image)


def _read_cam(root: Path, cam: str, index: int) -> np.ndarray | None:
    """The file the raw training path reads for this camera and tick."""
    from python.data.raw_frames import load_raw_camera

    return load_raw_camera(root, cam, index)


def _create(path: Path, data: bytes) -> None:
    with open(path, "xb") as handle:
        handle.write(data)


def build_frame_cache(
    strip_root: Path | str,
    dest: Path | str,
    cams: tuple[str, ...] | str,
    *,
    scale: float = 1.0,
    progress: Any = None,
) -> dict[str, Any]:
    """Decode ``cams`` at sector size into chunked ``.npy`` files under ``dest``.

    ``dest`` must be new or empty. Raises ``FileExistsError`` otherwise.
    """
    import io

    from python.data.raw_frames import read_state_jsonl

    root = Path(strip_root)
    out = Path(dest)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(
            f"{out} is not empty; a cache is built create-only into a new folder and never rewritten in place"
        )
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
            buffer = io.BytesIO()
            np.save(buffer, stack)
            _create(out / file_name, buffer.getvalue())
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
            fitted = _fit_sector(np.asarray(image), cam)
            if factor < 0.999:
                import cv2

                fitted = cv2.resize(fitted, (width, height), interpolation=cv2.INTER_AREA)
            batch_ids.append(index)
            batch_px.append(np.ascontiguousarray(fitted[:, :, :3]))
            if len(batch_ids) >= limit:
                flush()
        flush()
    manifest = {
        "format": CACHE_FORMAT,
        "sector": "strip_writer.place_sector (fit, then the repeater ego-edge mask)",
        "cams": list(names),
        "canvas_h": CANVAS_H if factor >= 0.999 else max(1, int(round(CANVAS_H * factor))),
        "canvas_w": canvas_width(names) if factor >= 0.999 else max(1, int(round(canvas_width(names) * factor))),
        "scale": factor,
        "chunks": chunks,
        "frames": written // max(1, len(names)),
    }
    data = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    _create(out / MANIFEST, data)
    digest = _sha256(data)
    _create(out / MANIFEST_SHA, f"{digest}  {MANIFEST}\n".encode("utf-8"))
    manifest["manifest_sha256"] = digest
    return manifest
