"""Multi-recording strip roots for scene-net training.

Single-root mode preserves today's row shape and holdout selection.
Multi-root mode namespaces ``session`` and tags every row with ``source`` /
``source_root`` so colliding folder names and frame indexes cannot merge.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from python.train.continual import pass_plan
from python.train.scene_net import MEMORY_S


def source_key_for(root: Path | str) -> str:
    """Stable short key from the resolved absolute path."""
    resolved = str(Path(root).resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]


def parse_strips_arg(values: Sequence[str] | None) -> tuple[Path, ...]:
    """Normalize ``--strips`` append / comma-joined tokens into unique resolved paths."""
    if not values:
        raise ValueError("--strips is required")
    paths: list[Path] = []
    for raw in values:
        text = str(raw).strip()
        if not text:
            raise ValueError("empty --strips value")
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            raise ValueError("empty --strips value")
        for part in parts:
            paths.append(Path(part))
    resolved = [p.resolve() for p in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate --strips paths after resolve")
    return tuple(resolved)


def resolve_holdout_source(
    roots: Sequence[Path],
    holdout_source: Path | str | None,
) -> Path:
    """Default holdout source is the first root. Must be one of ``roots``."""
    if not roots:
        raise ValueError("no strip roots")
    if holdout_source is None or str(holdout_source).strip() == "":
        return Path(roots[0]).resolve()
    want = Path(holdout_source).resolve()
    allowed = {Path(r).resolve() for r in roots}
    if want not in allowed:
        raise ValueError(f"--holdout-source {want} is not in --strips")
    return want


def load_multi_source_rows(
    roots: Sequence[Path | str],
    *,
    holdout_source: Path | str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load ``state.jsonl`` (+ GT merge) from each root.

    Single root: rows unchanged (no ``source`` key; session left as recorded).
    Multi root: every row gets ``source``, ``source_root``, and a namespaced
    ``session``.
    """
    from python.data.raw_frames import load_scene_rows
    from python.train.scene_loader import recompute_dt

    resolved = [Path(r).resolve() for r in roots]
    if not resolved:
        raise ValueError("no strip roots")
    holdout = resolve_holdout_source(resolved, holdout_source)
    holdout_key = source_key_for(holdout)
    multi = len(resolved) > 1
    rows: list[dict[str, Any]] = []
    train_only_keys: list[str] = []
    for root in resolved:
        key = source_key_for(root)
        if root != holdout:
            train_only_keys.append(key)
        loaded = load_scene_rows(root)
        if not multi:
            rows.extend(dict(r) for r in loaded)
            continue
        for raw in loaded:
            item = dict(raw)
            original = item.get("session")
            item["source"] = key
            item["source_root"] = str(root)
            item["session"] = f"{key}:{'' if original is None else original}"
            rows.append(item)
    rows = recompute_dt(rows)
    meta = {
        "multi": multi,
        "holdout_key": holdout_key if multi else None,
        "holdout_root": str(holdout),
        "train_only_keys": tuple(train_only_keys),
        "roots": tuple(str(r) for r in resolved),
    }
    return rows, meta


def combined_pass_plan(
    rows: list[Mapping[str, Any]],
    *,
    length: int,
    stride: int,
    blocks: int = 1,
    holdout_key: str | None,
    memory_s: float | None = None,
) -> list[dict[str, Any]]:
    """Pass plan with fixed holdout on ``holdout_key`` only; others train-only.

    When ``holdout_key`` is None (single-root), delegates to ``pass_plan`` with
    the same blocks/fraction so baseline metrics match.
    """
    kwargs: dict[str, Any] = {"blocks": int(blocks)}
    if memory_s is not None:
        kwargs["memory_s"] = float(memory_s)
    if holdout_key is None:
        return pass_plan(list(rows), int(length), int(stride), **kwargs)

    prior = [r for r in rows if r.get("source") == holdout_key]
    others_by_key: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        key = str(r.get("source") or "")
        if key == holdout_key:
            continue
        others_by_key.setdefault(key, []).append(r)

    plan: list[dict[str, Any]] = []
    # Prior train windows first (full pass_plan, then strip holdout for later).
    prior_full = pass_plan(list(prior), int(length), int(stride), **kwargs)
    prior_train = [item for item in prior_full if item["kind"] == "train"]
    prior_hold = [item for item in prior_full if item["kind"] == "holdout"]
    plan.extend(prior_train)
    mem = float(kwargs["memory_s"]) if "memory_s" in kwargs else float(MEMORY_S)
    # Train-only sources in first-seen order.
    for _key, group in others_by_key.items():
        plan.extend(
            pass_plan(list(group), int(length), int(stride), memory_s=mem, fraction=0.0, blocks=1)
        )
    plan.extend(prior_hold)
    return plan



def apply_mask_files_multi(roots, rows, files):
    """Re-export: mask files scoped per source_root (spec C.5)."""
    from python.train.event_mask import apply_mask_files_multi as _apply

    return _apply(roots, rows, files)
