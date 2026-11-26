"""Reality View splat depth sort: 16-bit depth key, stable radix order (numpy only).

Drop-in for the bench's ``np.argsort(-d, kind="stable")`` (farthest first). Depths are
quantized to a uint16 key over ``[near, far]`` and ordered with NumPy's stable sort of
16-bit integers (a radix sort). Contract and the quantum-tie behavior change are in
``docs/specs/rv_splat_sort_notes.md``. Pure CPU, deterministic, no flag, no GPU, no I/O.
"""
from __future__ import annotations

import math

import numpy as np

KEY_MAX = 65535


def _check_range(near, far) -> tuple[float, float]:
    for name, v in (("near", near), ("far", far)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number, got {v!r}")
    near, far = float(near), float(far)
    if not (math.isfinite(near) and math.isfinite(far) and near < far and math.isfinite(far - near)):
        raise ValueError(f"need finite near < far with finite span, got near={near!r}, far={far!r}")
    return near, far


def _check_depth(depth) -> np.ndarray:
    d = np.asarray(depth)
    if d.ndim != 1:
        raise ValueError(f"depth must be 1-D, got shape {d.shape}")
    if d.dtype.kind not in "fiu":
        raise ValueError(f"depth must be a real numeric array, got dtype {d.dtype}")
    return d


def quantum_m(near: float, far: float) -> float:
    """Depth width of one key step, metres: ``(far - near) / 65535``."""
    near, far = _check_range(near, far)
    return (far - near) / KEY_MAX


def _quantize(d: np.ndarray, near: float, far: float) -> np.ndarray:
    c = np.clip(d.astype(np.float64), near, far)
    t = (c - near) / (far - near)
    return np.floor(t * float(KEY_MAX) + 0.5).astype(np.uint16)


def quantize_depth(depth, near: float, far: float) -> np.ndarray:
    """uint16 key: ``floor(clip(d, near, far) -> t in [0, 1], t * 65535 + 0.5)``; NaN is rejected."""
    near, far = _check_range(near, far)
    d = _check_depth(depth)
    if np.isnan(d).any():
        raise ValueError("NaN has no depth key (sort_splats puts NaN last)")
    return _quantize(d, near, far)


def sort_splats(depth, near: float, far: float, descending: bool = True) -> np.ndarray:
    """Index order of splats by 16-bit depth key; returns a permutation of ``0..N-1`` (intp).

    ``descending=True`` is painter order (farthest first). Equal keys keep input-index order
    (stable); NaN entries go last in input order in both directions. The input is not modified.
    """
    near, far = _check_range(near, far)
    if not isinstance(descending, (bool, np.bool_)):
        raise ValueError(f"descending must be a bool, got {descending!r}")
    d = _check_depth(depth)
    nan = np.isnan(d) if d.dtype.kind == "f" else None
    if nan is not None and nan.any():
        valid = np.flatnonzero(~nan)
        key = _quantize(d[valid], near, far)
        if descending:
            key = ~key
        return np.concatenate([valid[np.argsort(key, kind="stable")], np.flatnonzero(nan)])
    key = _quantize(d, near, far)
    if descending:
        key = ~key
    return np.argsort(key, kind="stable")
