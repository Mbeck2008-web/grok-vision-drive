"""Opt-in per-epoch hard-example subset sampling (W5).

Off by default. Ranks **train** rows by previous-epoch loss, keeps the hardest
fraction plus a random floor. Holdout rows are never selected.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def row_frame_key(row: Mapping[str, Any]) -> tuple[str, int]:
    """Identity across combined ``--strips``: ``(source, i)``, not bare ``i``."""
    return (str(row.get("source") or ""), int(row["i"]))


def unique_rows_by_frame_key(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen: set[tuple[str, int]] = set()
    out: list[Mapping[str, Any]] = []
    for row in rows:
        key = row_frame_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _loss_value(row: Mapping[str, Any], losses: Mapping[Any, float]) -> float:
    key = row_frame_key(row)
    if key in losses:
        val = float(losses[key])
    elif not any(isinstance(item, tuple) for item in losses) and int(row["i"]) in losses:
        # Single-source tests still pass a bare-int map.
        val = float(losses[int(row["i"])])
    else:
        return float("inf")
    if not np.isfinite(val):
        return float("inf")
    return val


def select_active_subset(
    rows: Sequence[Mapping[str, Any]],
    losses: Mapping[Any, float],
    *,
    hard_frac: float = 0.3,
    random_floor: float = 0.1,
    rng: np.random.RandomState | None = None,
    holdout_kinds: set[str] | None = None,
) -> list[int]:
    """Return indices into ``rows`` for the next train epoch.

    ``losses`` is keyed by ``(source, i)`` (or bare ``i`` for a single source).
    Missing losses sort as hardest (+inf). Holdout kinds are excluded before
    ranking. The holdout itself is untouched.
    """
    rng = rng or np.random.RandomState(0)
    held = set(holdout_kinds or {"holdout"})
    train_idx = [n for n, row in enumerate(rows) if str(row.get("kind") or "train") not in held]
    if not train_idx:
        return []
    hard_frac = min(1.0, max(0.0, float(hard_frac)))
    random_floor = min(1.0, max(0.0, float(random_floor)))
    n = len(train_idx)
    n_hard = int(np.ceil(n * hard_frac)) if hard_frac > 0 else 0
    n_rand = int(np.ceil(n * random_floor)) if random_floor > 0 else 0

    def score(pos: int) -> float:
        return _loss_value(rows[pos], losses)

    ranked = sorted(train_idx, key=score, reverse=True)
    hard = ranked[:n_hard]
    remain = [p for p in ranked[n_hard:]]
    if n_rand > 0 and remain:
        pick = min(n_rand, len(remain))
        chosen_rand = list(rng.choice(remain, size=pick, replace=False))
    else:
        chosen_rand = []
    out = list(dict.fromkeys(hard + chosen_rand))  # stable unique
    return out


def filter_plan_whole_windows(
    plan_steps: Sequence[Mapping[str, Any]],
    allowed: set[tuple[str, int]],
) -> list[Any]:
    """Keep a train window entirely if any of its frames is selected.

    Holdout windows are appended unchanged. Partial frame lists are never
    written, so BPTT sequences are not hole-punched.
    """
    filtered: list[Any] = []
    for item in plan_steps:
        if item.get("kind") == "holdout":
            filtered.append(item)
            continue
        rows = list(item.get("rows") or [])
        if not rows:
            continue
        if any(row_frame_key(row) in allowed for row in rows):
            filtered.append(item)
    return filtered
