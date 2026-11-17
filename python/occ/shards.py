"""WebDataset-style tar shards for sparse occupancy samples (stdlib ``tarfile``).

One sample = members sharing a key: ``<key>.meta.json``, ``<key>.<cam>.png``
(lossless RGB) and ``<key>.<name>.npy`` (label arrays, no pickle). Shards are
written to ``*.tar.part`` and renamed on close, and each is capped below the
FAT32 4 GB limit. ``ShardDataset`` indexes the members once and then reads
by offset, so a sampler can draw random indices; it never opens loose image
files.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import tarfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

SHARD_MAX_BYTES = int(3.5 * 1024**3)
_KEY_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
_PART_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


class ShardError(ValueError):
    pass


def is_recovery_row(row: Mapping[str, Any] | None) -> bool:
    """A recorder tick is a recovery when it is tagged ``recover`` or its excursion phase is ``recover``."""
    if not isinstance(row, Mapping):
        return False
    tags = row.get("tags") if isinstance(row.get("tags"), Mapping) else row
    if tags.get("recover") is True:
        return True
    exc = tags.get("excursion")
    return isinstance(exc, Mapping) and exc.get("phase") == "recover"


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return buf.getvalue()


def _png_bytes(img: np.ndarray) -> bytes:
    import cv2

    a = np.asarray(img)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3 or a.shape[0] == 0 or a.shape[1] == 0:
        raise ShardError(f"rgb must be non-empty HxWx3 uint8, got {a.dtype} {a.shape}")
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(a), [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    if not ok:
        raise ShardError("png encode failed")
    return buf.tobytes()


def _png_decode(data: bytes) -> np.ndarray:
    import cv2

    arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ShardError("png decode failed")
    return arr


class ShardWriter:
    def __init__(self, out_dir: Path | str, prefix: str = "occ", *, max_bytes: int = SHARD_MAX_BYTES, max_count: int = 2000):
        if not _PART_RE.match(prefix):
            raise ShardError(f"bad shard prefix {prefix!r}")
        if max_bytes <= 0 or max_count <= 0:
            raise ShardError("max_bytes and max_count must be positive")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        old = sorted(self.out_dir.glob(f"{prefix}-*.tar")) + sorted(self.out_dir.glob(f"{prefix}-*.tar.part"))
        if old:
            # A second build must not overwrite part of the first and leave its
            # extra shards to be read as one set. Nothing is deleted here.
            raise ShardError(f"{self.out_dir} already has {len(old)} '{prefix}-*' shard file(s); use a new folder or prefix")
        self.prefix = prefix
        self.max_bytes = int(min(max_bytes, SHARD_MAX_BYTES))
        self.max_count = int(max_count)
        self.paths: list[Path] = []
        self._tar: tarfile.TarFile | None = None
        self._part: Path | None = None
        self._bytes = 0
        self._count = 0
        self._keys: set[str] = set()

    def _open(self) -> None:
        n = len(self.paths)
        final = self.out_dir / f"{self.prefix}-{n:06d}.tar"
        self._part = final.with_suffix(".tar.part")
        self._tar = tarfile.open(self._part, "w", format=tarfile.USTAR_FORMAT)
        self.paths.append(final)
        self._bytes = 0
        self._count = 0

    def _roll(self) -> None:
        if self._tar is not None:
            self._tar.close()
            os.replace(self._part, self.paths[-1])
            self._tar = None

    def write(self, key: str, *, meta: Mapping[str, Any], rgb: Mapping[str, np.ndarray], arrays: Mapping[str, np.ndarray]) -> None:
        if not isinstance(key, str) or not _KEY_RE.match(key):
            raise ShardError(f"bad key {key!r}: letters, digits, _ and - only")
        if key in self._keys:
            raise ShardError(f"duplicate key {key!r}")
        if not rgb:
            raise ShardError("sample needs at least one rgb camera")
        members: list[tuple[str, bytes]] = [
            (f"{key}.meta.json", json.dumps(dict(meta), sort_keys=True, allow_nan=False).encode("utf-8"))
        ]
        for cam, img in rgb.items():
            if not _PART_RE.match(str(cam)):
                raise ShardError(f"bad camera name {cam!r}")
            members.append((f"{key}.{cam}.png", _png_bytes(img)))
        for name, arr in arrays.items():
            if not _PART_RE.match(str(name)) or name in rgb:
                raise ShardError(f"bad array name {name!r}")
            members.append((f"{key}.{name}.npy", _npy_bytes(np.asarray(arr))))
        size = sum(512 + int(math.ceil(len(b) / 512)) * 512 for _, b in members)
        if size > self.max_bytes:
            raise ShardError(f"sample {key} is {size} bytes, over the shard cap {self.max_bytes}")
        if self._tar is None or self._count >= self.max_count or self._bytes + size > self.max_bytes:
            self._roll()
            self._open()
        assert self._tar is not None
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            self._tar.addfile(info, io.BytesIO(data))
        self._bytes += size
        self._count += 1
        self._keys.add(key)

    def close(self) -> list[Path]:
        self._roll()
        return list(self.paths)

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class ShardDataset:
    """Random access over tar shards. Only ``*.tar`` paths are accepted."""

    def __init__(self, paths: Sequence[Path | str]):
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ShardError("no shards")
        for p in self.paths:
            if p.suffix != ".tar":
                raise ShardError(f"not a shard: {p.name}")
        self._index: list[tuple[int, str, dict[str, tuple[int, int]]]] = []
        self.meta: list[dict[str, Any]] = []
        for si, p in enumerate(self.paths):
            groups: dict[str, dict[str, tuple[int, int]]] = {}
            order: list[str] = []
            with tarfile.open(p, "r") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    key, _, part = m.name.partition(".")
                    if not part:
                        raise ShardError(f"{p.name}: member without a field: {m.name}")
                    if key not in groups:
                        groups[key] = {}
                        order.append(key)
                    if part in groups[key]:
                        raise ShardError(f"{p.name}: duplicate member {m.name}")
                    groups[key][part] = (m.offset_data, m.size)
            with open(p, "rb") as fh:
                for key in order:
                    g = groups[key]
                    if "meta.json" not in g:
                        raise ShardError(f"{p.name}: sample {key} has no meta.json")
                    off, size = g["meta.json"]
                    fh.seek(off)
                    self.meta.append(json.loads(fh.read(size).decode("utf-8")))
                    self._index.append((si, key, g))

    def __len__(self) -> int:
        return len(self._index)

    def keys(self) -> list[str]:
        return [k for _, k, _ in self._index]

    def recovery_flags(self) -> list[bool]:
        return [bool(m.get("recover") is True) for m in self.meta]

    def __getitem__(self, i: int) -> dict[str, Any]:
        if not isinstance(i, (int, np.integer)) or isinstance(i, bool):
            raise TypeError("index must be an int")
        if i < 0 or i >= len(self._index):
            raise IndexError(i)
        si, key, g = self._index[int(i)]
        out: dict[str, Any] = {"key": key, "rgb": {}, "arrays": {}}
        with open(self.paths[si], "rb") as fh:
            for part, (off, size) in g.items():
                fh.seek(off)
                data = fh.read(size)
                name, _, ext = part.rpartition(".")
                if part == "meta.json":
                    out["meta"] = json.loads(data.decode("utf-8"))
                elif ext == "png":
                    out["rgb"][name] = _png_decode(data)
                elif ext == "npy":
                    out["arrays"][name] = np.load(io.BytesIO(data), allow_pickle=False)
                else:
                    raise ShardError(f"unknown member type {part}")
        return out

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]
