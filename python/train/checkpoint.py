"""Pause file for a scene-training run.

The file sits next to the strip folder, not inside it. A wipe of training
strips deletes JPEGs, ``state.jsonl``, and ``meta.json`` in that folder only.
It does not delete this checkpoint.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PAUSE_SUFFIX = ".gvd-pause.npz"


def checkpoint_path(strip_root: Path | str) -> Path:
    """Sibling of the strip folder: ``<parent>/<name>.gvd-pause.npz``."""
    root = Path(strip_root)
    name = root.name
    if name in ("", ".", ".."):
        return root.parent / f"gvd-train{PAUSE_SUFFIX}"
    return root.parent / f"{name}{PAUSE_SUFFIX}"


def _outside_strip_folder(strip_root: Path, path: Path) -> None:
    root = Path(strip_root)
    try:
        root_resolved = root.resolve()
        path_resolved = path.resolve()
    except OSError:
        return
    try:
        path_resolved.relative_to(root_resolved)
    except ValueError:
        return
    raise ValueError("pause checkpoint must stay outside the strip folder")


def _pack(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(value, np.ndarray):
        key = f"a{len(arrays)}"
        arrays[key] = np.array(value, copy=True)
        return {"$array": key}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _pack(item, arrays) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack(item, arrays) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot store {type(value).__name__} in a pause checkpoint")


def _unpack(value: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    if isinstance(value, dict) and set(value.keys()) == {"$array"}:
        key = str(value["$array"])
        if key not in arrays:
            raise ValueError(f"pause checkpoint is missing array {key}")
        return np.array(arrays[key], copy=True)
    if isinstance(value, dict):
        return {str(key): _unpack(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack(item, arrays) for item in value]
    return value


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)
    return str(value)


def save_checkpoint(strip_root: Path | str, payload: Mapping[str, Any]) -> Path:
    """Write optimizer, step, epoch, and weights next to ``strip_root``.

    Does not create, rewrite, or delete files inside the strip folder.
    Refuses a payload with no weights, so a pause cannot drop the model.
    """
    weights = payload.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("pause needs model weights")
    path = checkpoint_path(strip_root)
    _outside_strip_folder(Path(strip_root), path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    meta = _pack(dict(payload), arrays)
    partial = path.with_name(path.name + ".partial.npz")
    np.savez(partial, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(partial, path)
    return path


def load_checkpoint(strip_root: Path | str) -> dict[str, Any] | None:
    """Read a pause file. ``None`` when this run has not been paused."""
    path = checkpoint_path(strip_root)
    if not path.is_file():
        return None
    with np.load(path) as blob:
        meta = json.loads(_as_text(blob["meta"]))
        arrays = {key: np.array(blob[key], copy=True) for key in blob.files if key != "meta"}
    loaded = _unpack(meta, arrays)
    if not isinstance(loaded, dict):
        raise ValueError("pause checkpoint is not an object")
    return loaded


def clear_checkpoint(strip_root: Path | str) -> None:
    """Remove the pause file after a finished run. Strip files stay."""
    path = checkpoint_path(strip_root)
    _outside_strip_folder(Path(strip_root), path)
    try:
        if path.is_file() and not path.is_symlink():
            path.unlink()
    except OSError:
        return


def resume_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Step, loss, and weights a Start continues from.

    Raises when the weights are missing so resume cannot invent a new model.
    """
    weights = payload.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("pause checkpoint has no weights")
    raw_loss = payload.get("loss")
    loss = None if raw_loss is None else float(raw_loss)
    losses: list[float] = []
    for item in payload.get("losses") or []:
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number >= 0:
            losses.append(number)
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, dict):
        optimizer = {}
    return {
        "step": int(payload.get("step") or 0),
        "epoch": int(payload.get("epoch") or 0),
        "cursor": int(payload.get("cursor") or 0),
        "loss": loss,
        "lr": float(payload.get("lr") or 0.0),
        "epochs": int(payload.get("epochs") or 1),
        "batch": int(payload.get("batch") or 1),
        "device": str(payload.get("device") or "cpu"),
        "recommended": bool(payload.get("recommended", True)),
        "losses": losses,
        "weights": weights,
        "optimizer": optimizer,
        "scaler": payload.get("scaler"),
    }
