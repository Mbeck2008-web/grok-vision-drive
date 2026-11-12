"""Training-only joins for intersection sidecars.

``signs.jsonl`` and ``stoplines.jsonl`` are attached by ``t`` inside the
row's own recording. A missing tick, an ``err`` row, or a mask bar contributes
no loss. Nothing here is a live input.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from python.collect.signal_labels import SIGN_CLASSES

STOP_LINE_DIM = 2


def _tkey(stamp: Any) -> str:
    try:
        return f"{float(stamp):.4f}"
    except (TypeError, ValueError):
        return str(stamp)


def _index_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict) and "t" in row:
                out.setdefault(_tkey(row.get("t")), row)
    return out


def stop_line_target(bars: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(y_m, valid)`` and a loss mask. Only a ``src=stop_line`` bar trains.

    The nearest valid bar (smallest ``y_m``) is the scalar. A mask bar, a
    missing list, or a non-finite distance leaves the mask at 0.
    """
    values = np.zeros((STOP_LINE_DIM,), dtype=np.float32)
    mask = np.zeros((STOP_LINE_DIM,), dtype=np.float32)
    nearest: float | None = None
    for bar in bars or []:
        if not isinstance(bar, dict):
            continue
        if str(bar.get("src") or "") != "stop_line" or bar.get("valid") is not True:
            continue
        try:
            y_m = float(bar["y_m"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(y_m):
            continue
        if nearest is None or y_m < nearest:
            nearest = y_m
    if nearest is None:
        return values, mask
    values[0] = float(nearest)
    values[1] = 1.0
    mask[:] = 1.0
    return values, mask


def stop_line_arrays(row: Mapping[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    """Arrays already joined onto ``row``, else zeros (loss weight 0)."""
    row = row or {}
    if "stop_line" in row and "stop_line_valid" in row:
        values = np.asarray(row["stop_line"], dtype=np.float32).reshape(-1)
        valid = np.asarray(row["stop_line_valid"], dtype=np.float32).reshape(-1)
        out_v = np.zeros((STOP_LINE_DIM,), dtype=np.float32)
        out_m = np.zeros((STOP_LINE_DIM,), dtype=np.float32)
        n = min(STOP_LINE_DIM, int(values.size), int(valid.size))
        out_v[:n] = values[:n]
        out_m[:n] = valid[:n]
        return out_v, out_m
    if isinstance(row.get("bars"), list):
        return stop_line_target(row.get("bars"))
    return stop_line_target(None)


def _write_signs(row: dict[str, Any], signs: list[Any] | None) -> None:
    tech = row.get("tech")
    if isinstance(tech, dict):
        copied = dict(tech)
        if signs is None:
            copied.pop("signs", None)
        else:
            copied["signs"] = signs
        row["tech"] = copied
        return
    if signs is None:
        row.pop("signs", None)
    else:
        row["signs"] = signs


def attach_sign_labels(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Copy ``signs.jsonl`` onto rows. ``err`` and a missing ``t`` clear signs.

    A recording with no sidecar is left alone: there is nothing to join and
    nothing is invented. Join stays inside ``source_root``.
    """
    caches: dict[str, dict[str, dict[str, Any]] | None] = {}
    counts = {"parsed": 0, "masked": 0, "absent": 0}
    for name in SIGN_CLASSES:
        counts[name] = 0
    for row in rows:
        root = str(row.get("source_root") or "")
        if not root:
            counts["absent"] += 1
            continue
        if root not in caches:
            path = Path(root) / "signs.jsonl"
            caches[root] = _index_jsonl(path) if path.is_file() else None
        index = caches[root]
        if index is None:
            counts["absent"] += 1
            continue
        rec = index.get(_tkey(row.get("t")))
        if not isinstance(rec, dict) or rec.get("err") or "signs" not in rec:
            _write_signs(row, None)
            counts["masked"] += 1
            continue
        signs = rec.get("signs")
        copied = list(signs) if isinstance(signs, list) else []
        _write_signs(row, copied)
        counts["parsed"] += 1
        for sign in copied:
            cls = sign.get("cls") if isinstance(sign, dict) else None
            if cls in counts:
                counts[str(cls)] += 1
    return counts


def _write_stop(row: dict[str, Any], values: np.ndarray, mask: np.ndarray) -> None:
    row["stop_line"] = np.asarray(values, dtype=np.float32).reshape(STOP_LINE_DIM).copy()
    row["stop_line_valid"] = np.asarray(mask, dtype=np.float32).reshape(STOP_LINE_DIM).copy()


def attach_stop_lines(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Join ``stoplines.jsonl``. Missing file or tick: mask 0, not a guessed bar."""
    caches: dict[str, dict[str, dict[str, Any]] | None] = {}
    counts = {"valid": 0, "masked": 0, "absent": 0}
    for row in rows:
        root = str(row.get("source_root") or "")
        if not root:
            _write_stop(row, *stop_line_target(None))
            counts["masked"] += 1
            continue
        if root not in caches:
            path = Path(root) / "stoplines.jsonl"
            caches[root] = _index_jsonl(path) if path.is_file() else None
        index = caches[root]
        if index is None:
            _write_stop(row, *stop_line_target(None))
            counts["absent"] += 1
            continue
        rec = index.get(_tkey(row.get("t")))
        bars = rec.get("bars") if isinstance(rec, dict) else None
        values, mask = stop_line_target(bars)
        _write_stop(row, values, mask)
        if float(mask[0]) > 0.5:
            counts["valid"] += 1
        else:
            counts["masked"] += 1
    return counts


def sign_counts_line(counts: Mapping[str, int]) -> str:
    """Row joins, then how many sign objects of each class were copied on."""
    parts = " ".join(f"{name}={int(counts.get(name, 0))}" for name in ("parsed", "masked", "absent"))
    classes = " ".join(f"{name}={int(counts.get(name, 0))}" for name in SIGN_CLASSES)
    return f"[GVD] signs: {parts} {classes}"


def stop_counts_line(counts: Mapping[str, int]) -> str:
    parts = " ".join(f"{name}={int(counts.get(name, 0))}" for name in ("valid", "masked", "absent"))
    return f"[GVD] stop-line: {parts}"


def stop_line_loss(pred: Any, goal: Any, mask: Any) -> Any:
    """Masked smooth-L1 on ``y_m`` plus BCE on ``valid``. Mask 0 adds nothing."""
    import torch
    import torch.nn.functional as functional

    pred = pred.to(dtype=torch.float32).reshape(-1, STOP_LINE_DIM)
    goal = torch.as_tensor(goal, device=pred.device, dtype=torch.float32).reshape(-1, STOP_LINE_DIM)
    valid = torch.as_tensor(mask, device=pred.device, dtype=torch.float32).reshape(-1, STOP_LINE_DIM)
    if not bool((valid > 0.5).any()):
        return pred.new_zeros(())
    y_m = valid[:, 0]
    diff = functional.smooth_l1_loss(pred[:, 0], torch.nan_to_num(goal[:, 0], nan=0.0), reduction="none")
    y_term = (diff * y_m).sum() / y_m.sum().clamp(min=1.0)
    v_m = valid[:, 1]
    prob = pred[:, 1].clamp(1e-6, 1.0 - 1e-6)
    bce = functional.binary_cross_entropy(prob, goal[:, 1].clamp(0.0, 1.0), reduction="none")
    v_term = (bce * v_m).sum() / v_m.sum().clamp(min=1.0)
    return y_term + v_term
