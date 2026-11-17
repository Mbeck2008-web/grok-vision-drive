"""Occupancy training shards: WebDataset-format tar files, stdlib ``tarfile`` only.

One sample is a run of tar members sharing a key (WebDataset convention):

* ``<key>.<cam>.png``   lossless RGB of each rig camera present this tick
* ``<key>.keys.npy``    int64 occupied cell keys (sparse labels)
* ``<key>.cls.npy``     uint8 coarse class per key
* ``<key>.free.npy``    float16 free-space points (ego frame)
* ``<key>.meta.json``   t, session, recovery flag, cameras, label cameras

Beside each shard, ``<shard>.idx.json`` stores member offsets so a reader
seeks straight to one sample. Shards are capped at ``SHARD_MAX_BYTES`` (under
the 4 GB FAT32 file limit; NVMe wants large sequential files, not 400k loose
PNGs an hour; see ``docs/specs/occ_sparse.md``).

The reader opens ``.tar`` files only. It never falls back to a recording's
loose ``raw/<cam>/*.png`` (``OccShardDataset`` refuses directories).
Building shards from a recording (``recording_to_shards``) is the one step
that reads loose frames, offline. Shards hold training-only labels; the live
path never imports this module.
"""
from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SHARD_MAX_BYTES = 1 << 30
SHARD_PATTERN = "occ-{:06d}.tar"
INDEX_SUFFIX = ".idx.json"
INDEX_KIND = "gvd_occ_shard_index_1"


class ShardOnlyError(ValueError):
    """The loader was pointed at something that is not an occupancy tar shard."""


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return buf.getvalue()


def _png_bytes(img: np.ndarray) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", np.ascontiguousarray(img), [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    if not ok:
        raise ValueError("png encode failed")
    return buf.tobytes()


def is_recovery(row: Mapping[str, Any] | None) -> bool:
    """Recovery tick: excursion phase ``recover`` / ``abort`` or the collector's ``recover`` tag."""
    if not isinstance(row, Mapping):
        return False
    tags = row.get("tags") if isinstance(row.get("tags"), Mapping) else row
    if bool(tags.get("recover")):
        return True
    exc = tags.get("excursion")
    phase = exc.get("phase") if isinstance(exc, Mapping) else exc
    return str(phase or "") in ("recover", "abort")


class ShardWriter:
    """Create-only tar shards in ``out_dir``. Never overwrites a shard."""

    def __init__(self, out_dir: Path | str, *, max_bytes: int = SHARD_MAX_BYTES) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self._n = 0
        self._tar: tarfile.TarFile | None = None
        self._fh: Any = None
        self._path: Path | None = None
        self._index: list[dict[str, Any]] = []
        self.paths: list[Path] = []
        self.samples = 0

    def _open(self) -> None:
        while True:
            path = self.out_dir / SHARD_PATTERN.format(self._n)
            self._n += 1
            if not path.exists():
                break
        self._fh = open(path, "xb")
        self._tar = tarfile.open(fileobj=self._fh, mode="w", format=tarfile.USTAR_FORMAT)
        self._path = path
        self._index = []

    def _close_shard(self) -> None:
        if self._tar is None:
            return
        self._tar.close()
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        assert self._path is not None
        idx = {"kind": INDEX_KIND, "shard": self._path.name, "samples": self._index}
        with open(str(self._path) + INDEX_SUFFIX, "x", encoding="utf-8") as fh:
            json.dump(idx, fh)
        self.paths.append(self._path)
        self._tar = None

    def write(
        self,
        key: str,
        images: Mapping[str, np.ndarray],
        labels: Mapping[str, np.ndarray],
        meta: Mapping[str, Any],
    ) -> None:
        if "." in key or "/" in key:
            raise ValueError(f"shard key {key!r} may not contain '.' or '/'")
        members: list[tuple[str, bytes]] = []
        for cam, img in images.items():
            members.append((f"{key}.{cam}.png", _png_bytes(img)))
        members.append((f"{key}.keys.npy", _npy_bytes(np.asarray(labels["keys"], dtype=np.int64))))
        members.append((f"{key}.cls.npy", _npy_bytes(np.asarray(labels["cls"], dtype=np.uint8))))
        members.append((f"{key}.free.npy", _npy_bytes(np.asarray(labels["free"], dtype=np.float16))))
        m = dict(meta)
        m["cams"] = sorted(images)
        members.append((f"{key}.meta.json", json.dumps(m, sort_keys=True).encode("utf-8")))
        size = sum(len(b) + 1024 for _, b in members)
        if self._tar is None or (int(self._tar.offset) + size > self.max_bytes and self._index):
            self._close_shard()
            self._open()
        assert self._tar is not None
        entry: dict[str, Any] = {"key": key, "recovery": bool(m.get("recovery")), "members": {}}
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            header = len(info.tobuf(tarfile.USTAR_FORMAT, self._tar.encoding, self._tar.errors))
            data_at = int(self._tar.offset) + header
            self._tar.addfile(info, io.BytesIO(payload))
            entry["members"][name[len(key) + 1 :]] = [data_at, int(info.size)]
        self._index.append(entry)
        self.samples += 1

    def close(self) -> list[Path]:
        self._close_shard()
        return list(self.paths)

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _scan_index(path: Path) -> list[dict[str, Any]]:
    """Rebuild the member index from the tar itself (no sidecar)."""
    out: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    with tarfile.open(path, mode="r:") as tar:
        for info in tar:
            if not info.isfile():
                continue
            key, _, rest = info.name.partition(".")
            if key not in out:
                out[key] = {"key": key, "recovery": False, "members": {}}
                order.append(key)
            out[key]["members"][rest] = [int(info.offset_data), int(info.size)]
            if rest == "meta.json":
                meta = json.loads(tar.extractfile(info).read().decode("utf-8"))
                out[key]["recovery"] = bool(meta.get("recovery"))
    return [out[k] for k in order]


def shard_paths(spec: Path | str | Sequence[Path | str]) -> list[Path]:
    """Expand a shard folder or list into ``.tar`` paths. A folder must hold shards."""
    items = [spec] if isinstance(spec, (str, Path)) else list(spec)
    out: list[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            tars = sorted(p.glob("*.tar"))
            if not tars:
                raise ShardOnlyError(f"{p} holds no .tar shards (loose recordings are not read; build shards first)")
            out.extend(tars)
        elif p.suffix == ".tar" and p.is_file():
            out.append(p)
        else:
            raise ShardOnlyError(f"{p} is not a .tar shard")
    return out


class OccShardDataset:
    """Random access over occupancy shards. Map-style; torch DataLoader friendly."""

    def __init__(self, shards: Path | str | Sequence[Path | str], *, cams: Sequence[str] | None = None) -> None:
        self.paths = shard_paths(shards)
        self.cams = tuple(cams) if cams else None
        self.items: list[tuple[int, dict[str, Any]]] = []
        for si, p in enumerate(self.paths):
            idx_path = Path(str(p) + INDEX_SUFFIX)
            if idx_path.is_file():
                payload = json.loads(idx_path.read_text(encoding="utf-8"))
                if payload.get("kind") != INDEX_KIND:
                    raise ShardOnlyError(f"{idx_path} is not a {INDEX_KIND} index")
                samples = payload["samples"]
            else:
                samples = _scan_index(p)
            self.items.extend((si, s) for s in samples)
        self._fh: dict[int, Any] = {}
        self._pid = os.getpid()

    def __len__(self) -> int:
        return len(self.items)

    def recovery_flags(self) -> np.ndarray:
        return np.array([bool(s["recovery"]) for _, s in self.items], dtype=bool)

    def _read(self, si: int, off: int, size: int) -> bytes:
        if self._pid != os.getpid():  # a forked DataLoader worker gets its own handles
            self._fh, self._pid = {}, os.getpid()
        fh = self._fh.get(si)
        if fh is None:
            fh = open(self.paths[si], "rb")
            self._fh[si] = fh
        fh.seek(off)
        data = fh.read(size)
        if len(data) != size:
            raise OSError(f"short read in {self.paths[si]} at {off}")
        return data

    def raw(self, i: int) -> dict[str, bytes]:
        si, s = self.items[i]
        return {name: self._read(si, off, size) for name, (off, size) in s["members"].items()}

    def __getitem__(self, i: int) -> dict[str, Any]:
        import cv2

        blobs = self.raw(i)
        meta = json.loads(blobs["meta.json"].decode("utf-8"))
        cams = self.cams or tuple(meta["cams"])
        images = {}
        for cam in cams:
            b = blobs.get(f"{cam}.png")
            if b is None:
                raise KeyError(f"sample {self.items[i][1]['key']} has no {cam} frame")
            img = cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"undecodable {cam} frame in {self.items[i][1]['key']}")
            images[cam] = img
        load = lambda n: np.load(io.BytesIO(blobs[n]), allow_pickle=False)  # noqa: E731
        return {"images": images, "keys": load("keys.npy"), "cls": load("cls.npy"),
                "free": load("free.npy").astype(np.float32), "meta": meta}

    def close(self) -> None:
        for fh in self._fh.values():
            fh.close()
        self._fh = {}


def recording_to_shards(
    root: Path | str,
    out_dir: Path | str,
    cams_cfg: Mapping[str, Any],
    *,
    rgb_cams: Sequence[str] | None = None,
    max_bytes: int = SHARD_MAX_BYTES,
    limit: int | None = None,
) -> list[Path]:
    """Offline: one recording's loose frames + depth/sem/traffic -> shards.

    A tick is written when every ``rgb_cams`` frame is present; labels use
    every ``<cam>_depth`` / ``<cam>_sem`` plane recorded on that tick.
    """
    from python.data.raw_frames import read_image_file
    from python.data.strip_writer import read_state_jsonl
    from python.perception.occ_geometry import rig_cams
    from python.train.occ_labels import build_occ_labels, vehicles_in_ego
    from python.train.sensor_aux import read_sem_classes

    root = Path(root)
    geo = {c.cam_id: c for c in rig_cams(cams_cfg)}
    want = tuple(rgb_cams) if rgb_cams else tuple(geo)
    sem_classes = read_sem_classes(root)
    rows = read_state_jsonl(root / "state.jsonl")
    with ShardWriter(out_dir, max_bytes=max_bytes) as w:
        for row in rows:
            if limit is not None and w.samples >= int(limit):
                break
            files = row.get("cams") or {}
            def path_of(name: str) -> Path | None:
                ent = files.get(name)
                f = ent.get("file") if isinstance(ent, Mapping) else None
                return (root / f) if f else None
            imgs = {}
            for cam in want:
                p = path_of(cam)
                img = read_image_file(p) if p else None
                if img is None:
                    break
                imgs[cam] = img
            if len(imgs) != len(want):
                continue
            planes = {}
            for cam in geo:
                dp, sp = path_of(f"{cam}_depth"), path_of(f"{cam}_sem")
                d = read_image_file(dp) if dp else None
                if d is None:
                    continue
                s = read_image_file(sp) if sp else None
                planes[cam] = (d, s)
            if not planes:
                continue
            tags = row.get("tags") if isinstance(row.get("tags"), Mapping) else {}
            veh = vehicles_in_ego(tags.get("traffic"), row.get("ego"))
            lab = build_occ_labels(planes, geo, sem_classes=sem_classes, vehicles=veh,
                                   rng=np.random.default_rng(int(row.get("i") or 0)))
            meta = {"t": row.get("t"), "i": row.get("i"), "session": row.get("session"),
                    "recovery": is_recovery(row), "label_cams": list(lab.cams)}
            w.write(f"{root.name.replace('.', '_')}_{int(row.get('i') or 0):08d}", imgs, lab.as_arrays(), meta)
    return w.paths
