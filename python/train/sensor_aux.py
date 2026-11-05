"""Training-only depth / semantic targets and separate road-edge labels.

Deployed perception stays RGB-only. These tensors supervise training; they are
never inference inputs. Painted ``lane_gt`` stays authoritative for paint;
``road_edge`` is a separately named capability and must not be copied into
paint slots. Semantic class names come from ``sem_classes.json``; unknown /
non-allowlisted classes are not treated as drivable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from python.debug.lane_targets import LOOKAHEADS_M

# Stem spatial map for a 192x256 main crop is 12x16; keep aux at that bound.
AUX_H = 12
AUX_W = 16
SEM_CLASS_MAX = 64
# Left/right road boundary (not paint) at each lookahead.
ROAD_EDGE_DIM = len(LOOKAHEADS_M) * 2

# Explicit paved-like names only. Dirt / gravel / unknown are never implied.
DRIVABLE_CLASS_ALLOWLIST = frozenset({"ROAD", "Asphalt", "ASPHALT", "Highway", "HIGHWAY"})


class AuxTargetError(ValueError):
    """Depth/sem alignment or ``sem_classes.json`` schema failure."""


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def read_sem_classes(root: Path | str) -> dict[str, Any]:
    """Load and validate ``sem_classes.json``. Missing file → empty dict."""
    path = Path(root) / "sem_classes.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuxTargetError(f"sem_classes schema: cannot read {path.name}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise AuxTargetError("sem_classes schema: root must be an object")
    if payload.get("kind") != "gvd_sem_classes":
        raise AuxTargetError("sem_classes schema: kind must be gvd_sem_classes")
    if payload.get("train_only") is not True:
        raise AuxTargetError("sem_classes schema: train_only must be true")
    classes = payload.get("classes")
    if not isinstance(classes, Mapping) or "0" not in classes:
        raise AuxTargetError("sem_classes schema: classes must map ids including 0")
    return dict(payload)


def _downsample_u16_depth_m(depth_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average depth to AUX_H x AUX_W metres; zero-cm samples are invalid."""
    import cv2

    valid = (depth_cm > 0).astype(np.float32)
    metres = depth_cm.astype(np.float32) / 100.0
    metres = np.where(valid > 0.5, metres, 0.0)
    # Area average; validity is the fraction of valid source pixels in the bin.
    d = cv2.resize(metres, (AUX_W, AUX_H), interpolation=cv2.INTER_AREA)
    v = cv2.resize(valid, (AUX_W, AUX_H), interpolation=cv2.INTER_AREA)
    mask = (v >= 0.5).astype(np.float32)
    return d.astype(np.float32), mask


def _downsample_sem(sem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour class ids; class 0 (unknown) is invalid for CE pull."""
    import cv2

    ids = cv2.resize(sem.astype(np.uint8), (AUX_W, AUX_H), interpolation=cv2.INTER_NEAREST)
    mask = (ids > 0).astype(np.float32)
    return ids.astype(np.int64), mask


def load_sensor_aux_targets(
    root: Path | str,
    index: int,
    *,
    cam: str = "main",
) -> dict[str, Any]:
    """Load train-only depth/sem for one tick into fixed AUX_H x AUX_W targets.

    Missing files yield all-masked targets (no fabricated labels). Present
    files with bad dtype/shape or a bad ``sem_classes.json`` raise
    ``AuxTargetError``.
    """
    from python.data.raw_frames import is_train_only_cam, load_raw_camera

    root = Path(root)
    depth_cam = f"{cam}_depth"
    sem_cam = f"{cam}_sem"
    if not is_train_only_cam(depth_cam) or not is_train_only_cam(sem_cam):
        raise AuxTargetError(f"refusing non-train-only cams {depth_cam}/{sem_cam}")

    empty_d = np.zeros((AUX_H, AUX_W), np.float32)
    empty_v = np.zeros((AUX_H, AUX_W), np.float32)
    empty_s = np.zeros((AUX_H, AUX_W), np.int64)
    out: dict[str, Any] = {
        "depth_m": empty_d.copy(),
        "depth_valid": empty_v.copy(),
        "sem_id": empty_s.copy(),
        "sem_valid": empty_v.copy(),
        "sem_classes": {},
    }

    depth = load_raw_camera(root, depth_cam, int(index))
    sem = load_raw_camera(root, sem_cam, int(index))
    if depth is None and sem is None:
        return out

    if depth is not None:
        if depth.ndim != 2 or depth.dtype != np.uint16:
            raise AuxTargetError(f"depth dtype/shape must be uint16 HxW, got {depth.dtype} {getattr(depth, 'shape', None)}")
    if sem is not None:
        if sem.ndim != 2 or sem.dtype != np.uint8:
            raise AuxTargetError(f"sem dtype/shape must be uint8 HxW, got {sem.dtype} {getattr(sem, 'shape', None)}")

    if depth is not None and sem is not None and depth.shape != sem.shape:
        raise AuxTargetError(f"depth/sem alignment shape mismatch: {depth.shape} vs {sem.shape}")

    # Validate schema whenever either plane is present (recording claimed train_gt).
    classes = read_sem_classes(root)
    out["sem_classes"] = classes

    if depth is not None:
        d, v = _downsample_u16_depth_m(depth)
        out["depth_m"], out["depth_valid"] = d, v
    if sem is not None:
        s, v = _downsample_sem(sem)
        if int(s.max()) >= SEM_CLASS_MAX:
            raise AuxTargetError(f"sem class id {int(s.max())} exceeds SEM_CLASS_MAX={SEM_CLASS_MAX}")
        out["sem_id"], out["sem_valid"] = s, v
    return out


def pack_road_edge(
    *,
    offset_m: float | None,
    heading_err_rad: float | None,
    road_width_m: float | None,
    off_road: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ego-frame left/right **road** boundaries at ``LOOKAHEADS_M``.

    Not paint. Off-road or missing geometry → all-masked. Does not write
    ``lane_gt``.
    """
    values = np.full((ROAD_EDGE_DIM,), np.nan, dtype=np.float32)
    valid = np.zeros((ROAD_EDGE_DIM,), dtype=np.float32)
    if off_road is True:
        return values, valid
    off = _finite(offset_m)
    width = _finite(road_width_m)
    if off is None or width is None or width <= 0.0:
        return values, valid
    heading = _finite(heading_err_rad)
    h = 0.0 if heading is None else float(heading)
    half = float(width) * 0.5
    for k, y in enumerate(LOOKAHEADS_M):
        # Same lateral sign as gt_geom.ego_center_x: +offset means car right of center.
        center = -float(off) + float(y) * math.sin(h)
        values[2 * k] = center - half
        values[2 * k + 1] = center + half
        valid[2 * k : 2 * k + 2] = 1.0
    return values, valid


def road_edge_from_record(record: Mapping[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    """Read explicit ``road_edge`` fields, else pack from road width/offset.

    Never reads DecalRoad paint. Never mutates ``lane_gt``.
    """
    record = record or {}
    raw = record.get("road_edge")
    mask = record.get("road_edge_valid")
    if raw is not None and mask is not None:
        values = np.zeros((ROAD_EDGE_DIM,), dtype=np.float32)
        valid = np.zeros((ROAD_EDGE_DIM,), dtype=np.float32)
        src = np.asarray(raw, dtype=np.float32).reshape(-1)
        msrc = np.asarray(mask, dtype=np.float32).reshape(-1)
        n = min(ROAD_EDGE_DIM, int(src.size), int(msrc.size))
        values[:n] = src[:n]
        valid[:n] = msrc[:n]
        return values, valid
    return pack_road_edge(
        offset_m=_finite(record.get("road_offset_m", record.get("offset_m"))),
        heading_err_rad=_finite(record.get("heading_err_rad", record.get("road_heading_err_rad"))),
        road_width_m=_finite(record.get("road_width_m")),
        off_road=record.get("off_road") if isinstance(record.get("off_road"), bool) else None,
    )


def drivable_mask_from_sem(sem_id: np.ndarray, classes: Mapping[str, Any]) -> np.ndarray:
    """Binary mask of allowlisted paved class ids only; dirt/unknown stay 0."""
    id_to_name: dict[int, str] = {}
    raw_classes = classes.get("classes", classes) if isinstance(classes, Mapping) else {}
    if isinstance(raw_classes, Mapping):
        for key, name in raw_classes.items():
            try:
                id_to_name[int(key)] = str(name)
            except (TypeError, ValueError):
                continue
    allow = {i for i, name in id_to_name.items() if name in DRIVABLE_CLASS_ALLOWLIST}
    arr = np.asarray(sem_id)
    out = np.zeros(arr.shape, dtype=np.float32)
    for cid in allow:
        out[arr == cid] = 1.0
    return out


def attach_aux_to_labels(
    labels: dict[str, np.ndarray],
    *,
    root: Path | str | None = None,
    index: int | None = None,
    row: Mapping[str, Any] | None = None,
    cam: str = "main",
    want_depth: bool = False,
    want_sem: bool = False,
    want_road_edge: bool = False,
) -> dict[str, np.ndarray]:
    """Optionally attach aux / road_edge tensors onto a labels dict (copy)."""
    out = dict(labels)
    if (want_depth or want_sem) and root is not None and index is not None:
        aux = load_sensor_aux_targets(root, int(index), cam=cam)
        if want_depth:
            out["aux_depth"] = aux["depth_m"]
            out["aux_depth_valid"] = aux["depth_valid"]
        if want_sem:
            out["aux_sem"] = aux["sem_id"]
            out["aux_sem_valid"] = aux["sem_valid"]
    if want_road_edge:
        record: dict[str, Any] = {}
        if isinstance(row, Mapping):
            record.update(row)
            tech = row.get("tech")
            if isinstance(tech, Mapping):
                record.update(tech)
        values, valid = road_edge_from_record(record)
        out["road_edge"] = values
        out["road_edge_valid"] = valid
    return out
