"""12x16 depth/sem aux targets next to the RGB frame cache.

Train-only supervision: these tensors never become inference inputs. The
memmap stores the same arrays ``load_sensor_aux_targets`` would compute
from the raw PNGs, so the aux loss does not change. An old RGB-only cache
still loads; a missing aux map falls back to the PNG reader. A present but
stale aux map is refused with a rebuild hint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from python.train.sensor_aux import AUX_H, AUX_W

AUX_FORMAT = "gvd-aux-cache-1"
AUX_DIR = "aux"
MANIFEST = "manifest.json"
MANIFEST_SHA = "manifest.sha256"
DEPTH_M = "main_depth_m.npy"
DEPTH_VALID = "main_depth_valid.npy"
SEM_ID = "main_sem_id.npy"
SEM_VALID = "main_sem_valid.npy"
IDS = "ids.npy"
SOURCES = "sources.npy"


class StaleAuxCacheError(RuntimeError):
    """The aux cache folder is not this format. Rebuild it into a new folder."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def aux_dir(cache_root: Path | str) -> Path:
    return Path(cache_root) / AUX_DIR


class AuxCache:
    """Read-only memmap of per-tick 12x16 aux targets beside an RGB cache."""

    def __init__(self, cache_root: Path | str) -> None:
        self.root = Path(cache_root)
        self.folder = aux_dir(self.root)
        self.stale = False
        self.reason = ""
        self.format: str | None = None
        self.sem_classes: dict[str, Any] = {}
        self.cam = "main"
        self._closed = False
        self._by_index: dict[Any, int] = {}
        self._depth_m: np.ndarray | None = None
        self._depth_valid: np.ndarray | None = None
        self._sem_id: np.ndarray | None = None
        self._sem_valid: np.ndarray | None = None
        manifest = self.folder / MANIFEST
        if not manifest.is_file():
            return
        data = manifest.read_bytes()
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            self.stale = True
            self.reason = f"{self.folder}: cannot read {MANIFEST}: {exc}; rebuild with --build-cache into a NEW folder"
            return
        self.format = payload.get("format")
        if self.format != AUX_FORMAT:
            self.stale = True
            self.reason = (
                f"{self.folder} is aux cache format {self.format or 'unknown'}, not {AUX_FORMAT}. "
                "Rebuild with --build-cache into a NEW folder (an old cache is never rewritten in place)."
            )
            return
        sha_file = self.folder / MANIFEST_SHA
        want = sha_file.read_text(encoding="utf-8").split()[0] if sha_file.is_file() else ""
        if want != _sha256(data):
            self.stale = True
            self.reason = f"{self.folder}: {MANIFEST} sha256 does not match {MANIFEST_SHA}; rebuild into a new folder"
            return
        if int(payload.get("aux_h") or 0) != AUX_H or int(payload.get("aux_w") or 0) != AUX_W:
            self.stale = True
            self.reason = (
                f"{self.folder}: aux grid {payload.get('aux_h')}x{payload.get('aux_w')} "
                f"is not {AUX_H}x{AUX_W}; rebuild into a new folder"
            )
            return
        self.cam = str(payload.get("cam") or "main")
        classes = payload.get("sem_classes") or {}
        self.sem_classes = dict(classes) if isinstance(classes, Mapping) else {}
        ids_path = self.folder / IDS
        if not ids_path.is_file():
            self.stale = True
            self.reason = f"{self.folder}: missing {IDS}; rebuild into a new folder"
            return
        ids = np.load(ids_path, mmap_mode="r")
        sources_path = self.folder / SOURCES
        sources = np.load(sources_path, mmap_mode="r") if sources_path.is_file() else None
        for slot, raw in enumerate(np.asarray(ids).reshape(-1)):
            try:
                index = int(raw)
            except (TypeError, ValueError):
                continue
            if sources is not None and slot < len(sources):
                key = str(sources[slot])
                self._by_index[(key, index)] = slot
            else:
                self._by_index[index] = slot
        self._depth_m = self._map(DEPTH_M)
        self._depth_valid = self._map(DEPTH_VALID)
        self._sem_id = self._map(SEM_ID)
        self._sem_valid = self._map(SEM_VALID)

    def _map(self, name: str) -> np.ndarray | None:
        path = self.folder / name
        if not path.is_file():
            return None
        return np.load(path, mmap_mode="r")

    def present(self) -> bool:
        return bool(self._by_index) and not self.stale

    def require_fresh(self) -> None:
        if self.stale:
            raise StaleAuxCacheError(self.reason)

    def close(self) -> None:
        self._depth_m = None
        self._depth_valid = None
        self._sem_id = None
        self._sem_valid = None
        self._closed = True

    def _index_key(self, index: int, source: str | None = None) -> Any:
        if source is not None:
            return (str(source), int(index))
        return int(index)

    def has(self, index: int, source: str | None = None) -> bool:
        if self.stale or self._closed:
            return False
        key = self._index_key(index, source)
        if key in self._by_index:
            return True
        # Single-root map addressed with a source= that format-2 RGB caches refuse.
        if source is not None and int(index) in self._by_index:
            return True
        return False

    def get(self, index: int, source: str | None = None, cam: str = "main") -> dict[str, Any] | None:
        """Copy of the stored targets, or None on a miss (PNG fallback)."""
        if self.stale or self._closed:
            return None
        if cam and str(cam) != self.cam:
            return None
        key = self._index_key(index, source)
        slot = self._by_index.get(key)
        if slot is None and source is not None:
            slot = self._by_index.get(int(index))
        if slot is None:
            return None
        empty_d = np.zeros((AUX_H, AUX_W), np.float32)
        empty_v = np.zeros((AUX_H, AUX_W), np.float32)
        empty_s = np.zeros((AUX_H, AUX_W), np.int64)
        depth = empty_d if self._depth_m is None else np.array(self._depth_m[slot], copy=True, dtype=np.float32)
        dvalid = empty_v if self._depth_valid is None else np.array(self._depth_valid[slot], copy=True, dtype=np.float32)
        sem = empty_s if self._sem_id is None else np.array(self._sem_id[slot], copy=True, dtype=np.int64)
        svalid = empty_v if self._sem_valid is None else np.array(self._sem_valid[slot], copy=True, dtype=np.float32)
        return {
            "depth_m": depth,
            "depth_valid": dvalid,
            "sem_id": sem,
            "sem_valid": svalid,
            "sem_classes": dict(self.sem_classes),
        }


def build_aux_cache(
    strip_roots: Sequence[Path | str] | Path | str,
    dest: Path | str,
    *,
    cam: str = "main",
) -> dict[str, Any] | None:
    """Write 12x16 aux memmaps under ``dest/aux`` from raw depth/sem PNGs.

    ``dest/aux`` must not already exist. Returns None when no tick has depth or
    sem (an RGB-only recording). Uses ``load_sensor_aux_targets`` so the stored
    tensors match the PNG path bit-for-bit.
    """
    from python.data.raw_frames import read_state_jsonl
    from python.train.sensor_aux import load_sensor_aux_targets
    from python.train.strip_sources import source_key_for

    if isinstance(strip_roots, (str, Path)):
        roots = [Path(strip_roots).resolve()]
    else:
        roots = [Path(r).resolve() for r in strip_roots]
    if not roots:
        raise ValueError("build_aux_cache needs at least one strip root")
    out = aux_dir(dest)
    if out.exists():
        raise FileExistsError(
            f"{out} already exists; an aux cache is built create-only and never rewritten in place"
        )
    ids: list[int] = []
    sources: list[str] = []
    depth_m: list[np.ndarray] = []
    depth_valid: list[np.ndarray] = []
    sem_id: list[np.ndarray] = []
    sem_valid: list[np.ndarray] = []
    sem_classes: dict[str, Any] = {}
    saw_plane = False
    multi = len(roots) > 1
    for root in roots:
        key = source_key_for(root) if multi else ""
        rows = read_state_jsonl(root / "state.jsonl")
        for row in rows:
            try:
                index = int(row.get("i"))
            except (TypeError, ValueError):
                continue
            aux = load_sensor_aux_targets(root, index, cam=cam)
            if float(np.asarray(aux["depth_valid"]).max(initial=0.0)) > 0.0 or float(
                np.asarray(aux["sem_valid"]).max(initial=0.0)
            ) > 0.0:
                saw_plane = True
            if aux.get("sem_classes"):
                sem_classes = dict(aux["sem_classes"])
            ids.append(index)
            sources.append(key)
            depth_m.append(np.asarray(aux["depth_m"], dtype=np.float32))
            depth_valid.append(np.asarray(aux["depth_valid"], dtype=np.float32))
            sem_id.append(np.asarray(aux["sem_id"], dtype=np.int16))
            sem_valid.append(np.asarray(aux["sem_valid"], dtype=np.float32))
    if not ids or not saw_plane:
        return None
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / IDS, np.asarray(ids, dtype=np.int32))
    if multi:
        np.save(out / SOURCES, np.asarray(sources))
    np.save(out / DEPTH_M, np.stack(depth_m, axis=0))
    np.save(out / DEPTH_VALID, np.stack(depth_valid, axis=0))
    np.save(out / SEM_ID, np.stack(sem_id, axis=0))
    np.save(out / SEM_VALID, np.stack(sem_valid, axis=0))
    manifest = {
        "format": AUX_FORMAT,
        "cam": str(cam),
        "aux_h": AUX_H,
        "aux_w": AUX_W,
        "frames": len(ids),
        "multi": multi,
        "sem_classes": sem_classes,
    }
    data = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    (out / MANIFEST).write_bytes(data)
    digest = _sha256(data)
    (out / MANIFEST_SHA).write_text(f"{digest}  {MANIFEST}\n", encoding="utf-8")
    manifest["manifest_sha256"] = digest
    return manifest
