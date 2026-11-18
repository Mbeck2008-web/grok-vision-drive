"""Batch sampler that enforces the recovery mix (20-30% recovery ticks).

Each batch holds exactly ``k`` recovery samples, where ``k / batch_size`` is
the target ratio rounded to the nearest count the batch can hold. A target
outside ``[MIX_LO, MIX_HI]``, a batch size with no legal ``k``, or a dataset
without recovery ticks is refused: the mix is never silently 0%.
Recoveries are drawn without replacement per pass and reshuffled when the
pool runs out (the pool is usually far smaller than the cruise pool).
"""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

MIX_LO = 0.20
MIX_HI = 0.30
MIX_DEFAULT = 0.25


class RecoveryMixError(ValueError):
    pass


def recovery_count(batch_size: int, ratio: float = MIX_DEFAULT) -> int:
    """Recovery samples per batch. Refuses a mix that cannot sit in [20%, 30%]."""
    b = int(batch_size)
    r = float(ratio)
    if b < 1:
        raise RecoveryMixError("batch_size must be >= 1")
    if not (MIX_LO <= r <= MIX_HI):
        raise RecoveryMixError(f"recovery ratio {r} is outside [{MIX_LO}, {MIX_HI}]")
    legal = [k for k in range(b + 1) if MIX_LO - 1e-9 <= k / b <= MIX_HI + 1e-9]
    if not legal:
        raise RecoveryMixError(f"batch size {b} cannot hold a {MIX_LO:.0%}-{MIX_HI:.0%} recovery mix")
    return min(legal, key=lambda k: (abs(k / b - r), k))


class RecoveryMixSampler:
    """Yields index lists (torch ``batch_sampler``)."""

    def __init__(
        self,
        recovery: Sequence[bool] | np.ndarray,
        batch_size: int,
        *,
        ratio: float = MIX_DEFAULT,
        seed: int = 0,
        batches: int | None = None,
    ) -> None:
        flags = np.asarray(recovery, dtype=bool)
        self.rec = np.flatnonzero(flags)
        self.cruise = np.flatnonzero(~flags)
        self.batch_size = int(batch_size)
        self.k = recovery_count(self.batch_size, ratio)
        if self.k > 0 and self.rec.size == 0:
            raise RecoveryMixError("no recovery ticks in the training set; the 20-30% mix cannot be met")
        if self.batch_size - self.k > 0 and self.cruise.size == 0:
            raise RecoveryMixError("no non-recovery ticks in the training set")
        n_cruise_batches = self.cruise.size // max(1, self.batch_size - self.k)
        self.batches = int(batches) if batches is not None else max(1, n_cruise_batches)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.batches

    @staticmethod
    def _stream(pool: np.ndarray, rng: np.random.Generator) -> Iterator[int]:
        while True:
            for i in rng.permutation(pool):
                yield int(i)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        rec = self._stream(self.rec, rng) if self.k else iter(())
        cru = self._stream(self.cruise, rng) if self.batch_size - self.k else iter(())
        for _ in range(self.batches):
            batch = [next(rec) for _ in range(self.k)] + [next(cru) for _ in range(self.batch_size - self.k)]
            yield [int(i) for i in rng.permutation(batch)]
