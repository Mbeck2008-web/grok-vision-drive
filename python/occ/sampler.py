"""Batch sampler that enforces a 20-30% recovery mix.

Recoveries are rare in a recording, so they are drawn from their own pool
(cycling through shuffled passes, repeating only when the pool is smaller
than its quota) and placed into the stream at a fixed share.

Guarantee (tested by property tests):

* **Batch mode** when a batch of ``B`` can hold a share in [lo, hi]
  (``ceil(lo*B) <= floor(hi*B)``, e.g. B=4, 5, 7+): every batch has
  ``ceil(lo*B) <= k <= floor(hi*B)`` recoveries and the epoch share stays in
  [lo, hi] (error diffusion toward ``target``).
* **Window mode** otherwise (B = 1, 2, 3, 6): batches are cut from a stream
  in which every aligned window of ``WINDOW`` = 10 samples has 2 or 3
  recoveries (alternating, so each 20 has exactly 5).

An epoch has a whole number of units (batches or windows), at least
``len(dataset)`` samples. A dataset with no recoveries, or nothing but
recoveries, cannot meet the mix and raises ``RecoveryMixError``.
"""
from __future__ import annotations

import math
from typing import Iterator, Sequence

import numpy as np

FRAC_LO = 0.20
FRAC_HI = 0.30
TARGET = 0.25
WINDOW = 10


class RecoveryMixError(ValueError):
    pass


def batch_bounds(batch_size: int, lo: float = FRAC_LO, hi: float = FRAC_HI) -> tuple[int, int] | None:
    k_lo = math.ceil(lo * batch_size - 1e-9)
    k_hi = math.floor(hi * batch_size + 1e-9)
    return (k_lo, k_hi) if k_lo <= k_hi else None


class _Pool:
    def __init__(self, idx: np.ndarray, rng: np.random.Generator):
        self.idx = idx
        self.rng = rng
        self.perm = rng.permutation(idx)
        self.pos = 0

    def take(self, n: int) -> list[int]:
        out: list[int] = []
        while len(out) < n:
            if self.pos >= len(self.perm):
                self.perm = self.rng.permutation(self.idx)
                self.pos = 0
            m = min(n - len(out), len(self.perm) - self.pos)
            out.extend(int(v) for v in self.perm[self.pos : self.pos + m])
            self.pos += m
        return out


class RecoveryMixSampler:
    def __init__(self, recovery_flags: Sequence[bool], batch_size: int, *, seed: int = 0):
        flags = np.asarray([bool(f) for f in recovery_flags], dtype=bool)
        if flags.size == 0:
            raise RecoveryMixError("empty dataset")
        if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
            raise RecoveryMixError(f"batch_size must be a positive int, got {batch_size!r}")
        self.batch_size = int(batch_size)
        self.rec = np.nonzero(flags)[0]
        self.norm = np.nonzero(~flags)[0]
        if self.rec.size == 0:
            raise RecoveryMixError("no recovery samples: cannot reach a 20-30% mix")
        if self.norm.size == 0:
            raise RecoveryMixError("only recovery samples: cannot reach a 20-30% mix")
        self.n = int(flags.size)
        self.seed = int(seed)
        self.bounds = batch_bounds(self.batch_size)
        self.mode = "batch" if self.bounds is not None else "window"
        unit = self.batch_size if self.mode == "batch" else math.lcm(WINDOW, self.batch_size)
        self.length = max(1, math.ceil(self.n / unit)) * unit

    def __len__(self) -> int:
        return self.length // self.batch_size

    def _counts(self, rng: np.random.Generator) -> list[tuple[int, int]]:
        """(slot size, recoveries) per unit of the stream."""
        if self.mode == "batch":
            assert self.bounds is not None
            lo, hi = self.bounds
            out, placed = [], 0
            for b in range(self.length // self.batch_size):
                want = TARGET * (b + 1) * self.batch_size - placed
                k = int(min(hi, max(lo, round(want))))
                out.append((self.batch_size, k))
                placed += k
            return out
        first = int(rng.integers(0, 2))
        return [(WINDOW, 2 + ((w + first) % 2)) for w in range(self.length // WINDOW)]

    def stream(self, epoch: int = 0) -> list[int]:
        rng = np.random.default_rng([self.seed, int(epoch)])
        rec, norm = _Pool(self.rec, rng), _Pool(self.norm, rng)
        out: list[int] = []
        for size, k in self._counts(rng):
            slot = rec.take(k) + norm.take(size - k)
            out.extend(int(slot[j]) for j in rng.permutation(size))
        return out

    def batches(self, epoch: int = 0) -> list[list[int]]:
        s = self.stream(epoch)
        b = self.batch_size
        return [s[i : i + b] for i in range(0, len(s), b)]

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.batches(0))
