"""Opt-in per-epoch hard-example subset sampling (W5).

Off by default. Ranks **train** rows by previous-epoch loss, keeps the hardest
fraction plus a random floor. Holdout rows are never selected.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def select_active_subset(
    rows: Sequence[Mapping[str, Any]],
    losses: Mapping[int, float],
    *,
    hard_frac: float = 0.3,
    random_floor: float = 0.1,
    rng: np.random.RandomState | None = None,
    holdout_kinds: set[str] | None = None,
) -> list[int]:
    """Return indices into ``rows`` for the next train epoch.

    ``losses`` is keyed by ``row['i']``. Missing losses sort as hardest (+inf).
    Holdout kinds are excluded before ranking. The holdout itself is untouched.
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
        row_i = int(rows[pos]["i"])
        if row_i not in losses:
            return float("inf")
        val = float(losses[row_i])
        if not np.isfinite(val):
            return float("inf")
        return val

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
