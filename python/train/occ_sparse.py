"""Sparse occupancy: training-only labels, binary shards, and a query head.

Labels v4 pattern (``docs/specs/labels_v4.md``, ``python/train/sensor_aux.py``):
depth and annotation are training ground truth only. The live net stays RGB.
The head reads the GRU feature the other scene heads read and is evaluated only
at the queried voxels, so no dense 3D grid is ever built or exported.

Ego frame is the GVD vehicle frame: x right, y forward, z up (metres).
Depth is the recorder's uint16 centimetre plane, read as view-Z along the
optical axis (``python/data/raw_frames.py`` documents the scale as unverified
until the live spot-check). Depth 0 is unknown and never a point.

A voxel's class is ``unknown`` unless its semantic id is on the paved
allowlist (``DRIVABLE_CLASS_ALLOWLIST``); a voxel with any non-allowlisted
hit stays ``unknown``. ``vehicle`` comes from traffic rows: the centre column
only, because the recorder does not measure length or width.
"""
from __future__ import annotations

import io
import json
import math
import tarfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

# 0.5 m voxels. x in [-16, 16), y in [-16, 40), z in [-1, 3) ego metres.
# x and the forward 40 m match the obstacles BEV in labels_v4; 16 m behind
# and the z band cover the side and rear cameras.
VOXEL_M = 0.5
GRID_MIN = (-16.0, -16.0, -1.0)
GRID_MAX = (16.0, 40.0, 3.0)
GRID_SHAPE = tuple(int(round((hi - lo) / VOXEL_M)) for lo, hi in zip(GRID_MIN, GRID_MAX))
OCC_CLASSES = ("unknown", "drivable", "vehicle")
# Free samples along each ray, as fractions of the hit range.
FREE_FRACTIONS = (0.25, 0.5, 0.75)
VEHICLE_Z_M = (0.0, 1.5)
PE_FREQS = 6
HEAD_WIDTH = 128
# FAT32 holds 4 GB per file; the cache keeps chunks under 3.5 GB (AGENTS.md).
SHARD_MAX_BYTES = 3_500_000_000
RECOVERY_SHARE = (0.20, 0.30)


def _basis(yaw_deg: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from python.sensors.cameras import yaw_pitch_to_dir_up

    fwd, up = (np.asarray(v, dtype=np.float64) for v in yaw_pitch_to_dir_up(float(yaw_deg), float(pitch_deg)))
    right = np.cross(fwd, up)
    return right, up, fwd


def unproject_depth(depth_cm: np.ndarray, cam: Mapping[str, Any], *, stride: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Ego-frame points ``(N, 3)`` and the ``(H', W')`` keep mask of the strided grid."""
    depth = np.asarray(depth_cm)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError(f"depth must be 2-D uint16 cm, got {depth.dtype} {depth.shape}")
    h, w = depth.shape
    fy = (h / 2.0) / math.tan(math.radians(float(cam.get("fov_v") or 70.0)) / 2.0)
    vs = np.arange(0, h, int(stride))
    us = np.arange(0, w, int(stride))
    sub = depth[np.ix_(vs, us)].astype(np.float64) / 100.0
    keep = sub > 0
    uu, vv = np.meshgrid(us + 0.5 - w / 2.0, vs + 0.5 - h / 2.0)
    z = sub[keep]
    xc = uu[keep] / fy * z
    yc = -vv[keep] / fy * z
    right, up, fwd = _basis(float(cam.get("yaw_deg") or 0.0), float(cam.get("pitch_deg") or 0.0))
    pos = np.asarray(cam.get("pos_m") or [0.0, 0.0, 0.0], dtype=np.float64)
    pts = pos + np.outer(xc, right) + np.outer(yc, up) + np.outer(z, fwd)
    return pts, keep


def voxel_index(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat voxel ids of the in-grid points and their row mask."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    ijk = np.floor((pts - np.asarray(GRID_MIN)) / VOXEL_M).astype(np.int64)
    inside = np.all((ijk >= 0) & (ijk < np.asarray(GRID_SHAPE)), axis=1)
    flat = np.ravel_multi_index(tuple(ijk[inside].T), GRID_SHAPE) if inside.any() else np.zeros(0, np.int64)
    return flat.astype(np.int32), inside


def voxel_centers(flat: np.ndarray) -> np.ndarray:
    ijk = np.stack(np.unravel_index(np.asarray(flat, dtype=np.int64), GRID_SHAPE), axis=1)
    return np.asarray(GRID_MIN) + (ijk + 0.5) * VOXEL_M


def _vehicle_points(row: Mapping[str, Any] | None) -> np.ndarray:
    if not isinstance(row, Mapping):
        return np.zeros((0, 3))
    from python.train.vehicle_labels import _boxes, _traffic_field

    field = _traffic_field(row)
    if not isinstance(field, list):
        return np.zeros((0, 3))
    zs = np.arange(VEHICLE_Z_M[0] + VOXEL_M / 2.0, VEHICLE_Z_M[1], VOXEL_M)
    pts = [(b["x"], b["y"], z) for b in _boxes(row, field) for z in zs]
    return np.asarray(pts, dtype=np.float64).reshape(-1, 3)


def occ_labels_for_tick(
    planes: Mapping[str, Mapping[str, Any]],
    cams: Sequence[Mapping[str, Any]],
    *,
    sem_names: Mapping[int, str],
    row: Mapping[str, Any] | None,
    stride: int = 8,
) -> dict[str, Any]:
    """Sparse labels for one tick from every camera that has a depth plane.

    Returns ``occ_idx`` (int32, unique), ``occ_cls`` (uint8 index into
    ``OCC_CLASSES``), ``free_idx`` (int32, disjoint from ``occ_idx``) and
    ``cams`` (sorted ids that contributed).
    """
    from python.train.sensor_aux import DRIVABLE_CLASS_ALLOWLIST

    drivable_ids = {int(k) for k, v in (sem_names or {}).items() if str(v) in DRIVABLE_CLASS_ALLOWLIST}
    unk, drv, veh = (OCC_CLASSES.index(n) for n in OCC_CLASSES)
    occ_parts: list[np.ndarray] = []
    cls_parts: list[np.ndarray] = []
    free_parts: list[np.ndarray] = []
    used: list[str] = []
    for cam in cams:
        cid = str(cam.get("id"))
        plane = planes.get(cid) or {}
        depth = plane.get("depth_cm")
        if depth is None:
            continue
        pts, keep = unproject_depth(depth, cam, stride=stride)
        used.append(cid)
        sem = plane.get("sem")
        if sem is not None:
            ids = np.asarray(sem)[:: int(stride), :: int(stride)][: keep.shape[0], : keep.shape[1]][keep]
            cls = np.where(np.isin(ids, list(drivable_ids)), drv, unk).astype(np.uint8)
        else:
            cls = np.full(len(pts), unk, np.uint8)
        flat, inside = voxel_index(pts)
        occ_parts.append(flat)
        cls_parts.append(cls[inside])
        origin = np.asarray(cam.get("pos_m") or [0.0, 0.0, 0.0], dtype=np.float64)
        for frac in FREE_FRACTIONS:
            fflat, _ = voxel_index(origin + (pts - origin) * float(frac))
            free_parts.append(fflat)
    vpts = _vehicle_points(row)
    if len(vpts):
        vflat, _ = voxel_index(vpts)
        occ_parts.append(vflat)
        cls_parts.append(np.full(len(vflat), veh, np.uint8))
    occ = np.concatenate(occ_parts) if occ_parts else np.zeros(0, np.int32)
    cls = np.concatenate(cls_parts) if cls_parts else np.zeros(0, np.uint8)
    # Per voxel: vehicle beats unknown beats drivable. One stray unknown hit
    # keeps a voxel out of drivable.
    rank = np.array([1, 0, 2], dtype=np.int8)[cls] if len(cls) else np.zeros(0, np.int8)
    order = np.lexsort((-rank, occ))
    occ_sorted, cls_sorted = occ[order], cls[order]
    first = np.ones(len(occ_sorted), dtype=bool)
    first[1:] = occ_sorted[1:] != occ_sorted[:-1]
    occ_u = occ_sorted[first].astype(np.int32)
    cls_u = cls_sorted[first].astype(np.uint8)
    free = np.concatenate(free_parts) if free_parts else np.zeros(0, np.int32)
    free_u = np.setdiff1d(np.unique(free), occ_u).astype(np.int32)
    return {"occ_idx": occ_u, "occ_cls": cls_u, "free_idx": free_u, "cams": sorted(used)}


def is_recovery_row(row: Mapping[str, Any] | None) -> bool:
    tags = (row or {}).get("tags")
    exc = tags.get("excursion") if isinstance(tags, Mapping) else None
    return isinstance(exc, Mapping) and exc.get("phase") == "recover"


def recovery_weights(rows: Sequence[Mapping[str, Any]], *, target: float = 0.25) -> np.ndarray:
    """Per-row sampling weights so recover rows are ``target`` of the draws.

    ``target`` must be in [0.20, 0.30]. With no recover rows (or only
    recover rows) the weights stay uniform: no recovery is invented.
    """
    if not RECOVERY_SHARE[0] <= float(target) <= RECOVERY_SHARE[1]:
        raise ValueError(f"recovery share {target} outside {RECOVERY_SHARE}")
    flags = np.array([is_recovery_row(r) for r in rows], dtype=bool)
    n, n_r = len(flags), int(flags.sum())
    if n == 0 or n_r in (0, n):
        return np.ones(n, dtype=np.float64)
    w = np.where(flags, float(target) / n_r, (1.0 - float(target)) / (n - n_r))
    return w * n


_ARRAYS = ("occ_idx", "occ_cls", "free_idx")


def _npy(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr), allow_pickle=False)
    return buf.getvalue()


class OccShardWriter:
    """WebDataset-layout tar shards: ``<key>.occ_idx.npy``, ``.occ_cls.npy``, ``.free_idx.npy``, ``.json``.

    Plain stdlib ``tarfile``; no new dependency. A shard rotates before it
    would pass ``max_bytes`` so a FAT32 data disk is never asked for a 4 GB file.
    """

    def __init__(self, out_dir: Path | str, *, max_bytes: int = SHARD_MAX_BYTES, prefix: str = "occ") -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.prefix = prefix
        self._n = 0
        self._tar: tarfile.TarFile | None = None
        self._bytes = 0
        self._count = 0
        self.paths: list[Path] = []

    def _open(self) -> None:
        path = self.dir / f"{self.prefix}-{self._n:06d}.tar"
        self._n += 1
        self._tar = tarfile.open(path, "w")
        self._bytes = 1024
        self._count = 0
        self.paths.append(path)

    def add(self, sample: Mapping[str, Any]) -> None:
        key = str(sample["key"])
        if "." in key or "/" in key:
            raise ValueError(f"shard key must not contain '.' or '/': {key!r}")
        members = [(f"{key}.{name}.npy", _npy(sample[name])) for name in _ARRAYS]
        meta = {"recover": bool(sample.get("recover", False)), "cams": list(sample.get("cams") or [])}
        members.append((f"{key}.json", json.dumps(meta).encode("utf-8")))
        size = sum(512 + ((len(b) + 511) // 512) * 512 for _, b in members)
        if self._tar is None or (self._count and self._bytes + size > self.max_bytes):
            self.close()
            self._open()
        assert self._tar is not None
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            self._tar.addfile(info, io.BytesIO(payload))
        self._bytes += size
        self._count += 1

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None

    def __enter__(self) -> "OccShardWriter":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def read_occ_shards(paths: Iterable[Path | str]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with tarfile.open(path, "r") as tf:
            current: dict[str, Any] = {}
            for member in tf:
                key, _, rest = member.name.partition(".")
                if current and current.get("key") != key:
                    yield current
                    current = {}
                current["key"] = key
                data = tf.extractfile(member).read()  # type: ignore[union-attr]
                if rest == "json":
                    current.update(json.loads(data.decode("utf-8")))
                else:
                    current[rest[: -len(".npy")]] = np.load(io.BytesIO(data), allow_pickle=False)
            if current:
                yield current


def build_occ_shards(root: Path | str, out_dir: Path | str, *, stride: int = 8, max_bytes: int = SHARD_MAX_BYTES) -> list[Path]:
    """One recording's ``raw/<cam>_depth`` / ``raw/<cam>_sem`` + ``state.jsonl`` traffic -> tar shards."""
    from python.data.raw_frames import load_raw_camera, load_scene_rows
    from python.train.sensor_aux import read_sem_classes

    root = Path(root)
    cams = list(json.loads((root / "cameras.json").read_text(encoding="utf-8")).get("cameras") or [])
    palette = read_sem_classes(root).get("classes") or {}
    sem_names = {int(k): str(v.get("name") if isinstance(v, Mapping) else v) for k, v in palette.items()}
    tag = root.name.replace(".", "_")
    with OccShardWriter(out_dir, max_bytes=max_bytes, prefix=f"occ-{tag}") as writer:
        for row in load_scene_rows(root):
            try:
                index = int(row.get("i"))
            except (TypeError, ValueError):
                continue
            planes = {}
            for cam in cams:
                cid = str(cam.get("id"))
                depth = load_raw_camera(root, f"{cid}_depth", index)
                if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
                    continue
                sem = load_raw_camera(root, f"{cid}_sem", index)
                planes[cid] = {"depth_cm": depth, "sem": sem if sem is not None and sem.ndim == 2 else None}
            lab = occ_labels_for_tick(planes, cams, sem_names=sem_names, row=row, stride=stride)
            if not lab["cams"] and not len(lab["occ_idx"]):
                continue
            writer.add({"key": f"{tag}_{index:08d}", **lab, "recover": is_recovery_row(row)})
        return list(writer.paths)


def build_queries(labels: Sequence[Mapping[str, Any]], *, k: int = 4096, seed: int = 0) -> tuple[Any, Any, Any, Any]:
    """``(B, k)`` voxel ids, occ target, class target, mask. Half occupied, rest free; short rows are masked."""
    import torch

    rng = np.random.default_rng(int(seed))
    b = len(labels)
    q = np.zeros((b, k), np.int64)
    t_occ = np.zeros((b, k), np.float32)
    t_cls = np.zeros((b, k), np.int64)
    m = np.zeros((b, k), np.float32)
    for i, lab in enumerate(labels):
        occ = np.asarray(lab["occ_idx"], np.int64)
        cls = np.asarray(lab["occ_cls"], np.int64)
        free = np.asarray(lab["free_idx"], np.int64)
        n_occ = min(len(occ), k // 2)
        n_free = min(len(free), k - n_occ)
        pick_o = rng.choice(len(occ), n_occ, replace=False) if n_occ else np.zeros(0, np.int64)
        pick_f = rng.choice(len(free), n_free, replace=False) if n_free else np.zeros(0, np.int64)
        n = n_occ + n_free
        q[i, :n] = np.concatenate([occ[pick_o], free[pick_f]])
        t_occ[i, :n_occ] = 1.0
        t_cls[i, :n_occ] = cls[pick_o]
        m[i, :n] = 1.0
    return tuple(torch.as_tensor(a) for a in (q, t_occ, t_cls, m))  # type: ignore[return-value]


def occ_loss(logits: Any, t_occ: Any, t_cls: Any, mask: Any) -> Any:
    """Masked BCE on occupancy plus CE on the class of occupied queries. Zero with no labels."""
    import torch.nn.functional as functional

    occ_logit = logits[..., 0]
    m = mask.to(occ_logit.dtype)
    denom = m.sum()
    if float(denom) == 0.0:
        return (logits * 0.0).sum()
    bce = functional.binary_cross_entropy_with_logits(occ_logit.float(), t_occ.float(), reduction="none")
    loss = (bce * m).sum() / denom
    hit = (t_occ > 0.5) & (m > 0.5)
    if bool(hit.any()):
        loss = loss + functional.cross_entropy(logits[..., 1:][hit].float(), t_cls[hit])
    return loss


try:
    import torch
    from torch import nn

    class SparseOccHead(nn.Module):
        """Implicit occupancy at queried voxels: MLP(scene feature, Fourier(voxel centre)).

        Cost is O(queries), never O(grid). Output per query: 1 occupancy
        logit + one logit per ``OCC_CLASSES`` entry.
        """

        def __init__(self, feat_dim: int, width: int = HEAD_WIDTH) -> None:
            super().__init__()
            pe_dim = 3 + 3 * 2 * PE_FREQS
            self.mlp = nn.Sequential(
                nn.Linear(int(feat_dim) + pe_dim, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU(),
                nn.Linear(width, 1 + len(OCC_CLASSES)),
            )
            lo = torch.tensor(GRID_MIN)
            hi = torch.tensor(GRID_MAX)
            self.register_buffer("grid_lo", lo, persistent=False)
            self.register_buffer("grid_span", hi - lo, persistent=False)
            self.register_buffer("freqs", (2.0 ** torch.arange(PE_FREQS)) * math.pi, persistent=False)

        def encode(self, q: Any) -> Any:
            q = q.long()
            sz = GRID_SHAPE[1] * GRID_SHAPE[2]
            ix = torch.div(q, sz, rounding_mode="floor")
            iy = torch.div(q % sz, GRID_SHAPE[2], rounding_mode="floor")
            iz = q % GRID_SHAPE[2]
            ijk = torch.stack([ix, iy, iz], dim=-1).to(self.grid_lo.dtype)
            xyz = self.grid_lo + (ijk + 0.5) * VOXEL_M
            unit = (xyz - self.grid_lo) / self.grid_span * 2.0 - 1.0
            ang = unit.unsqueeze(-1) * self.freqs
            return torch.cat([unit, torch.sin(ang).flatten(-2), torch.cos(ang).flatten(-2)], dim=-1)

        def forward(self, feat: Any, q: Any) -> Any:
            pe = self.encode(q).to(feat.dtype)
            f = feat.unsqueeze(1).expand(-1, q.shape[1], -1)
            return self.mlp(torch.cat([f, pe], dim=-1))

except ImportError:  # pragma: no cover - torch-free hosts
    class SparseOccHead:  # type: ignore[no-redef]
        def __init__(self, *_a: Any, **_k: Any) -> None:
            raise RuntimeError("PyTorch is required for the occupancy head")
