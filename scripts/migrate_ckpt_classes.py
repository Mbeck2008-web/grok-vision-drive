#!/usr/bin/env python3
"""Opt-in migration of a pre-v2 epoch checkpoint (96 object-class outputs) to the
current 80-output head. Run it by hand; nothing in python/ calls it.

    python scripts/migrate_ckpt_classes.py OLD.npz NEW.npz [--force]

Why 96 and 80
-------------
``object_class_head`` is ``nn.Linear(hidden, N_OBJECTS * n_classes)`` and the net
reads it as ``view(width, N_OBJECTS, n_classes)``, so output row ``slot * n_classes
+ class`` (slot-major). ``29f6f0f`` (recorder v2, PR #239, 2026-11-23, Michael's
decision: no people class) cut OBJECT_CLASSES from 6 to 5 names. ``N_OBJECTS`` is
16 on both sides: 16 x 6 = 96 -> 16 x 5 = 80. Main refuses a 96 checkpoint by
name (``python/train/scene_net.py::check_object_class_head``). That guard is the
default and is not changed by this tool.

Where the old class list comes from
-----------------------------------
A ``gvd-epoch-ckpt-1`` file carries no class list (meta has run/epoch/step/cams/
arch/..., no classes). The old list is therefore the ``OBJECT_CLASSES`` constant
in the code at ``4e41994`` (the commit the ckpts of that era were built at). It
is unchanged from there to ``29f6f0f^`` (``git log -G'^OBJECT_CLASSES\\s*='``:
29f6f0f is the only commit that touches it). The source is recorded in the
output's ``meta["migration"]``.

Mapping rule (by NAME, never by position)
-----------------------------------------
Each old name has an explicit policy in ``CLASS_POLICY``: a current class name
(kept, row copied bit-for-bit) or ``None`` (dropped). No merges exist in the
history, so nothing is averaged. An old name with no policy, a policy target
that is not in the current class list, or a current class with no old source
raises ``MappingError`` naming it. Nothing is silently dropped or initialised.

Tensors touched (found by inspection, ``scene_head_sizes``)
-----------------------------------------------------------
Only ``object_class_head.weight`` (rows) and ``object_class_head.bias``. Every
other tensor is copied untouched. Any *other* tensor with a leading dim of 96 is
refused rather than guessed at.

The result is a warm start / evaluation file for the 5-class head. It was not
trained with the people class removed, so treat its class logits accordingly.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

MAPPING_ID = "objclass-6to5-drop-people-v1"

# The pre-v2 class list: python/train/scene_net.py at 4e41994 (the ckpt has none).
OLD_CLASSES = ("vehicle", "car", "truck", "bus", "pedestrian", "bike")  # gvd-allow-pedestrian-word
OLD_CLASSES_SOURCE = {
    "kind": "code constant OBJECT_CLASSES (checkpoint carries no class list)",
    "file": "python/train/scene_net.py",
    "sha": "4e41994b508b3ca2f77ce0727ce93d75bf77756d",
    "same_through": "29f6f0fddc9c6f59a9cf585d915da66934515777^",
    "changed_by": "29f6f0fddc9c6f59a9cf585d915da66934515777",
    "n_objects": 16,
}
# old name -> current name it is copied into, or None = dropped.
CLASS_POLICY: dict[str, str | None] = {
    "vehicle": "vehicle",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "pedestrian": None,  # gvd-allow-pedestrian-word  (removed by 29f6f0f; no replacement class)
    "bike": "bike",
}
HEAD_WEIGHT = "object_class_head.weight"
HEAD_BIAS = "object_class_head.bias"
# Tensors whose leading dim is the class-output count. Everything else must not be 96 wide.
CLASS_TENSORS = (HEAD_WEIGHT, HEAD_BIAS)


class MigrationRefused(RuntimeError):
    """The input or output is not something this tool will touch."""


class MappingError(ValueError):
    """The old and current class lists cannot be mapped by name."""


def _current_classes() -> tuple[tuple[str, ...], int]:
    from python.train.scene_net import N_OBJECTS, OBJECT_CLASSES

    return tuple(OBJECT_CLASSES), int(N_OBJECTS)


def build_mapping(
    old_classes: Sequence[str],
    new_classes: Sequence[str],
    policy: Mapping[str, str | None] = CLASS_POLICY,
) -> dict[str, Any]:
    """Name-based mapping. ``new_from_old[j]`` is the old class index copied into new class j."""
    old = list(old_classes)
    new = list(new_classes)
    if len(set(old)) != len(old) or len(set(new)) != len(new):
        raise MappingError(f"duplicate class names: old={old} new={new}")
    for name in old:
        if name not in policy:
            raise MappingError(f"old class {name!r} has no entry in the mapping policy; refusing to guess or drop it")
    taken: dict[str, str] = {}
    dropped: list[str] = []
    for name in old:
        target = policy[name]
        if target is None:
            dropped.append(name)
            continue
        if target not in new:
            raise MappingError(f"old class {name!r} maps to {target!r}, which is not in the current class list {new}")
        if target in taken:
            raise MappingError(f"old classes {taken[target]!r} and {name!r} both map to {target!r}: a merge needs an explicit rule")
        taken[target] = name
    for name in new:
        if name not in taken:
            raise MappingError(f"current class {name!r} has no source in the old class list {old}")
    return {
        "id": MAPPING_ID,
        "old_classes": old,
        "new_classes": new,
        "new_from_old": [old.index(taken[name]) for name in new],
        "kept": [{"old": taken[name], "new": name} for name in new],
        "dropped": dropped,
        "merged": [],
    }


def row_map(mapping: Mapping[str, Any], n_objects: int) -> np.ndarray:
    """Old flat output row for every new flat row (slot-major, as ``view(width, N_OBJECTS, C)``)."""
    n_old = len(mapping["old_classes"])
    rows = [slot * n_old + i for slot in range(int(n_objects)) for i in mapping["new_from_old"]]
    return np.asarray(rows, dtype=np.int64)


def default_output_path(src: Path | str) -> Path:
    """``X.npz`` -> ``X.migrated5.npz``. Not a parseable epoch-ckpt name, so --export-epoch lookups skip it."""
    src = Path(src)
    return src.with_name(src.stem + ".migrated5.npz")


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def migrate_weights(
    weights: Mapping[str, Any], mapping: Mapping[str, Any], n_objects: int
) -> dict[str, np.ndarray]:
    n_old, n_new = len(mapping["old_classes"]), len(mapping["new_classes"])
    want_old, want_new = n_objects * n_old, n_objects * n_new
    for key in CLASS_TENSORS:
        if key not in weights:
            raise MigrationRefused(f"checkpoint has no {key}; this is not a scene-net checkpoint this tool can migrate")
    rows = {key: int(np.shape(weights[key])[0]) for key in CLASS_TENSORS}
    for key, got in rows.items():
        if got == want_new:
            raise MigrationRefused(f"{key} already has {want_new} outputs (the current head); nothing to migrate")
        if got != want_old:
            raise MigrationRefused(
                f"{key} has {got} outputs; this tool only migrates {want_old} ({n_objects} slots x {n_old} classes) to {want_new}"
            )
    if np.ndim(weights[HEAD_WEIGHT]) != 2 or np.ndim(weights[HEAD_BIAS]) != 1:
        raise MigrationRefused(f"unexpected head ranks: {HEAD_WEIGHT} {np.shape(weights[HEAD_WEIGHT])}, {HEAD_BIAS} {np.shape(weights[HEAD_BIAS])}")
    for key, value in weights.items():
        shape = np.shape(value)
        if key not in CLASS_TENSORS and shape and int(shape[0]) == want_old:
            raise MigrationRefused(f"{key} {tuple(shape)} also has a leading dim of {want_old}; not a known class tensor, refusing to guess")
    pick = row_map(mapping, n_objects)
    out: dict[str, np.ndarray] = {}
    for key, value in weights.items():
        arr = np.array(value, copy=True)
        out[str(key)] = np.ascontiguousarray(arr[pick]) if key in CLASS_TENSORS else arr
    return out


def migrate_file(
    src: Path | str,
    dst: Path | str,
    *,
    force: bool = False,
    old_classes: Sequence[str] = OLD_CLASSES,
    old_classes_source: Mapping[str, Any] = OLD_CLASSES_SOURCE,
) -> dict[str, Any]:
    """Write a migrated copy of ``src`` at ``dst``. ``src`` is never modified."""
    from python.train.epoch_ckpt import load_epoch_ckpt, serialise

    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise MigrationRefused(f"input {src} is not a file")
    if dst.exists() and src.exists() and os.path.samefile(src, dst):
        raise MigrationRefused(f"output is the input file ({src}); the input is never overwritten")
    if dst.exists() and not force:
        raise MigrationRefused(f"output {dst} already exists (pass --force to replace it)")
    new_classes, n_objects = _current_classes()
    mapping = build_mapping(old_classes, new_classes)  # unknown names fail here, before any file is written
    src_bytes = src.read_bytes()
    loaded = load_epoch_ckpt(src)
    meta = dict(loaded["meta"])
    if "migration" in meta:
        raise MigrationRefused(f"{src.name} was already migrated (meta.migration present)")
    new_weights = migrate_weights(loaded["weights"], mapping, n_objects)
    meta["migration"] = {
        "mapping_id": mapping["id"],
        "migrated_from_outputs": n_objects * len(mapping["old_classes"]),
        "migrated_to_outputs": n_objects * len(mapping["new_classes"]),
        "old_classes": mapping["old_classes"],
        "new_classes": mapping["new_classes"],
        "old_classes_source": dict(old_classes_source),
        "kept": mapping["kept"],
        "dropped": mapping["dropped"],
        "merged": mapping["merged"],
        "tensors": list(CLASS_TENSORS),
        "layout": "slot-major: row = slot * n_classes + class",
        "source_file": src.name,
        "source_sha256": _sha256_bytes(src_bytes),
        "tool": "scripts/migrate_ckpt_classes.py",
    }
    meta.pop("format", None)  # serialise() re-stamps it
    meta.pop("weight_names", None)
    blob = serialise(new_weights, meta)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".partial-{os.getpid()}")
    try:
        with open(tmp, "xb") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        if force:
            os.replace(tmp, dst)
        else:
            os.link(tmp, dst)  # create-only: fails if dst appeared meanwhile
            os.unlink(tmp)
    except FileExistsError as exc:
        raise MigrationRefused(f"output {dst} already exists") from exc
    finally:
        if tmp.exists():
            tmp.unlink()
    return {
        "input": str(src),
        "input_sha256": _sha256_bytes(src_bytes),
        "output": str(dst),
        "output_sha256": _sha256_bytes(blob),
        "old_outputs": n_objects * len(mapping["old_classes"]),
        "new_outputs": n_objects * len(mapping["new_classes"]),
        "mapping": mapping,
    }


def format_summary(info: Mapping[str, Any]) -> str:
    m = info["mapping"]
    lines = [
        f"migrated {info['old_outputs']} -> {info['new_outputs']} object-class outputs ({m['id']})",
        f"  input : {info['input']}  sha256={info['input_sha256']}",
        f"  output: {info['output']}  sha256={info['output_sha256']}",
        f"  old classes ({len(m['old_classes'])}): {', '.join(m['old_classes'])}",
        f"  new classes ({len(m['new_classes'])}): {', '.join(m['new_classes'])}",
        f"  kept   : {', '.join(k['old'] + ' -> ' + k['new'] for k in m['kept'])}",
        f"  dropped: {', '.join(m['dropped']) or '-'}",
        f"  merged : {', '.join(m['merged']) or '-'}",
        "  The input is untouched. Load the output by passing its path explicitly; nothing auto-migrates.",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Opt-in: migrate a pre-v2 (96-output) epoch checkpoint to the 80-output object-class head.")
    p.add_argument("input", help="gvd-epoch-ckpt-1 .npz with 96 object_class outputs (never modified)")
    p.add_argument("output", nargs="?", help="new .npz (default: <input>.migrated5.npz); refused if it exists")
    p.add_argument("--force", action="store_true", help="replace an existing OUTPUT (never the input)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.output) if args.output else default_output_path(args.input)
    try:
        info = migrate_file(args.input, out, force=args.force)
    except (MigrationRefused, MappingError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(format_summary(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
