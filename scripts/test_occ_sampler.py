#!/usr/bin/env python3
"""Recovery mix: every batch is 20-30% recovery ticks, enforced in the sampler.

* exact count per batch for several batch sizes and ratios;
* ratios outside [0.2, 0.3], batch sizes with no legal count, and a set
  without recovery ticks are refused;
* a small recovery pool is cycled without repeats inside one pass;
* ``is_recovery`` reads the collector tags (``recover`` / excursion phase).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from python.data.occ_shards import is_recovery  # noqa: E402
from python.train.occ_sampler import MIX_HI, MIX_LO, RecoveryMixError, RecoveryMixSampler, recovery_count  # noqa: E402


def _refused(fn) -> bool:
    try:
        fn()
    except RecoveryMixError:
        return True
    return False


def check_counts() -> None:
    expect = {(4, 0.25): 1, (5, 0.25): 1, (8, 0.25): 2, (10, 0.25): 2, (10, 0.3): 3, (10, 0.2): 2, (20, 0.25): 5, (7, 0.25): 2}
    for (b, r), k in expect.items():
        assert recovery_count(b, r) == k, (b, r, recovery_count(b, r), k)
        assert MIX_LO <= k / b <= MIX_HI + 1e-9, (b, k)
    print("recovery_count: exact k per batch size OK")


def check_batches() -> None:
    rng = np.random.default_rng(1)
    flags = rng.random(1000) < 0.05  # 5% recoveries in the raw data
    for b in (4, 5, 8, 10):
        s = RecoveryMixSampler(flags, b, ratio=0.25, seed=3, batches=200)
        seen = 0
        for batch in s:
            assert len(batch) == b, batch
            n = int(flags[batch].sum())
            assert n == s.k, (b, n, s.k)
            assert MIX_LO <= n / b <= MIX_HI + 1e-9
            seen += 1
        assert seen == 200 == len(s)
    print("sampler: every batch holds exactly k recoveries (5% raw -> 20-30%) OK")


def check_refusals() -> None:
    flags = np.array([True] * 5 + [False] * 95)
    assert _refused(lambda: RecoveryMixSampler(flags, 4, ratio=0.1))
    assert _refused(lambda: RecoveryMixSampler(flags, 4, ratio=0.35))
    assert _refused(lambda: RecoveryMixSampler(flags, 3, ratio=0.25))  # 0/3, 1/3 both outside
    assert _refused(lambda: RecoveryMixSampler(flags, 2, ratio=0.25))
    assert _refused(lambda: RecoveryMixSampler(np.zeros(100, bool), 4))
    assert _refused(lambda: RecoveryMixSampler(np.ones(100, bool), 4))
    print("sampler: bad ratio, impossible batch size, missing pool refused OK")


def check_cycle() -> None:
    flags = np.array([True] * 3 + [False] * 60)
    s = RecoveryMixSampler(flags, 4, seed=0, batches=6)
    recs = [i for batch in s for i in batch if flags[i]]
    assert len(recs) == 6
    assert sorted(recs[:3]) == [0, 1, 2] and sorted(recs[3:]) == [0, 1, 2], recs
    cru = Counter(i for batch in RecoveryMixSampler(flags, 4, seed=0, batches=20) for i in batch if not flags[i])
    assert max(cru.values()) == 1, "cruise ticks repeat inside one pass"
    a = [b for b in RecoveryMixSampler(flags, 4, seed=5, batches=5)]
    b = [b for b in RecoveryMixSampler(flags, 4, seed=5, batches=5)]
    assert a == b, "same seed, same batches"
    print("sampler: recovery pool cycles without repeats per pass; deterministic per seed OK")


def check_tags() -> None:
    assert is_recovery({"tags": {"recover": True}})
    assert is_recovery({"tags": {"excursion": {"phase": "recover"}}})
    assert is_recovery({"tags": {"excursion": {"phase": "abort"}}})
    assert not is_recovery({"tags": {"excursion": {"phase": "weave"}}})
    assert not is_recovery({"tags": {"excursion": {"phase": "cruise"}}})
    assert not is_recovery({"tags": {}}) and not is_recovery(None)
    print("is_recovery: recover tag / recover+abort phases only OK")


def main() -> int:
    check_counts()
    check_batches()
    check_refusals()
    check_cycle()
    check_tags()
    print("test_occ_sampler: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
