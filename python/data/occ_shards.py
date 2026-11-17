"""Binary shards for sparse-occupancy training (docs/specs/occ_sparse.md).

Format is WebDataset's: a plain POSIX tar whose members are
``<key>.<ext>``; members with the same key are one sample. Written with the
stdlib ``tarfile`` (no new dependency); the ``webdataset`` package can read
these files unchanged. Each sample holds the tick's encoded camera images
exactly as recorded (``<cam>.png``), the sparse labels (``occ.npz``, a few
kB instead of two uint16/uint8 planes per camera) and ``json`` metadata
(recording, tick index, ``recovery`` flag).

Random access does not unpack anything: :class:`ShardIndex` keeps the byte
offset of every member, so a training step reads only the bytes it needs
from the shard file on NVMe. Shards roll over below ``max_bytes`` (default
1 GiB; FAT32 holds 4 GiB per file).
"""
from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

DEFAULT_SHARD_BYTES = 1 << 30
_TAR_OVERHEAD = 1024


class ShardWriter:
    """Append samples to ``<prefix>-000000.tar``, rolling over before ``max_bytes``."""

    def __init__(self, out_dir: Path | str, prefix: str = "occ", max_bytes: int = DEFAULT_SHARD_BYTES) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.max_bytes = int(max_bytes)
        self.paths: list[Path] = []
        self._tar: tarfile.TarFile | None = None
        self._size = 0

    def _open(self) -> None:
        path = self.out_dir / f"{self.prefix}-{len(self.paths):06d}.tar"
        if path.exists():
            raise FileExistsError(f"shard exists, refusing to overwrite: {path}")
        self._tar = tarfile.open(path, "w", format=tarfile.USTAR_FORMAT)
        self.paths.append(path)
        self._size = 0

    def write(self, key: str, members: Mapping[str, bytes]) -> None:
        if "." in key or "/" in key:
            raise ValueError(f"shard key must not contain '.' or '/': {key!r}")
        need = sum(len(v) + 512 for v in members.values()) + _TAR_OVERHEAD
        if self._tar is None or (self._size > 0 and self._size + need > self.max_bytes):
            self.close_current()
            self._open()
        assert self._tar is not None
        for ext, data in sorted(members.items()):
            info = tarfile.TarInfo(f"{key}.{ext}")
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            self._tar.addfile(info, io.BytesIO(data))
            self._size += 512 + ((len(data) + 511) // 512) * 512

    def close_current(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None

    def close(self) -> list[Path]:
        self.close_current()
        return list(self.paths)


def _split(name: str) -> tuple[str, str]:
    base = name.rsplit("/", 1)[-1]
    key, _, ext = base.partition(".")
    return key, ext


def iter_samples(paths: Sequence[Path | str]) -> Iterator[tuple[str, dict[str, bytes]]]:
    """Sequential read: ``(key, {ext: bytes})`` in file order."""
    for path in paths:
        with tarfile.open(path, "r:") as tf:
            cur_key, cur = None, {}
            for info in tf:
                if not info.isfile():
                    continue
                key, ext = _split(info.name)
                if key != cur_key and cur_key is not None:
                    yield cur_key, cur
                    cur = {}
                cur_key = key
                fh = tf.extractfile(info)
                cur[ext] = fh.read() if fh is not None else b""
            if cur_key is not None:
                yield cur_key, cur


@dataclass
class _Member:
    shard: int
    offset: int
    size: int


class ShardIndex:
    """Byte-offset index over shard files for random access by sample number."""

    def __init__(self, paths: Sequence[Path | str]) -> None:
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise FileNotFoundError("no shard files given")
        self.keys: list[str] = []
        self.members: list[dict[str, _Member]] = []
        for si, path in enumerate(self.paths):
            if path.suffix != ".tar":
                raise ValueError(f"not a tar shard: {path}")
            with tarfile.open(path, "r:") as tf:
                for info in tf:
                    if not info.isfile():
                        continue
                    key, ext = _split(info.name)
                    if not self.keys or self.keys[-1] != key:
                        self.keys.append(key)
                        self.members.append({})
                    self.members[-1][ext] = _Member(si, info.offset_data, info.size)

    def __len__(self) -> int:
        return len(self.keys)

    def read(self, i: int, exts: Sequence[str] | None = None) -> dict[str, bytes]:
        out: dict[str, bytes] = {}
        for ext, m in self.members[i].items():
            if exts is not None and ext not in exts:
                continue
            with open(self.paths[m.shard], "rb") as fh:
                fh.seek(m.offset)
                out[ext] = fh.read(m.size)
        return out

    def meta(self, i: int) -> dict[str, Any]:
        raw = self.read(i, ["json"]).get("json")
        return json.loads(raw.decode("utf-8")) if raw else {}


def pack_labels(lab: Any) -> bytes:
    buf = io.BytesIO()
    extra = {} if lab.box is None else {"box": lab.box}
    np.savez_compressed(buf, occ_idx=lab.occ_idx, occ_cls=lab.occ_cls, flow=lab.flow, free_idx=lab.free_idx, **extra)
    return buf.getvalue()


def unpack_labels(blob: bytes) -> Any:
    from python.train.occ_labels import SparseOccLabels

    with np.load(io.BytesIO(blob)) as z:
        return SparseOccLabels(z["occ_idx"], z["occ_cls"], z["flow"], z["free_idx"], z["box"] if "box" in z.files else None)


def is_recovery_row(row: Mapping[str, Any] | None) -> bool:
    """A recovery tick: excursion phase ``recover`` or the recover-teleport flag."""
    tags = (row or {}).get("tags")
    if not isinstance(tags, Mapping):
        return False
    if tags.get("recover") is True:
        return True
    exc = tags.get("excursion")
    return isinstance(exc, Mapping) and exc.get("phase") == "recover"


def recovery_mix_weights(is_recovery: Sequence[bool] | np.ndarray, target: float = 0.25) -> np.ndarray:
    """Per-sample draw weights so recoveries are ``target`` of the expected draws.

    ``target`` must be in [0.20, 0.30]. With no recoveries (or only
    recoveries) the weights stay uniform: the mix cannot be met and nothing
    is invented; callers report the actual fraction.
    """
    if not 0.20 <= float(target) <= 0.30:
        raise ValueError(f"recovery target {target} outside 0.20..0.30")
    rec = np.asarray(is_recovery, bool)
    n_rec = int(rec.sum())
    n_other = int(len(rec) - n_rec)
    w = np.ones(len(rec), np.float64)
    if n_rec == 0 or n_other == 0:
        return w
    w[rec] = float(target) / n_rec
    w[~rec] = (1.0 - float(target)) / n_other
    return w / w.mean()
