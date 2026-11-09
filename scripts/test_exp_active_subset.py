"""CPU tests for --exp-active-subset (W5). Holdout stays untouched."""

from __future__ import annotations

import sys
import unittest

import numpy as np

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ExpActiveSubsetTests(unittest.TestCase):
    def test_flag_off_by_default(self) -> None:
        from python.train.train_scene import build_parser

        args = build_parser().parse_args([])
        self.assertFalse(bool(getattr(args, "exp_active_subset", False)))

    def test_holdout_never_selected(self) -> None:
        from python.train.exp_active_subset import select_active_subset

        rows = [{"i": i, "kind": "train" if i < 8 else "holdout"} for i in range(10)]
        losses = {i: float(10 - i) for i in range(10)}  # lower i = harder
        rng = np.random.RandomState(0)
        chosen = select_active_subset(
            rows, losses, hard_frac=0.3, random_floor=0.1, rng=rng
        )
        kinds = {rows[i]["kind"] for i in chosen}
        self.assertNotIn("holdout", kinds)
        self.assertTrue(all(rows[i]["kind"] == "train" for i in chosen))

    def test_hard_plus_random_floor_size(self) -> None:
        from python.train.exp_active_subset import select_active_subset

        rows = [{"i": i, "kind": "train"} for i in range(100)]
        losses = {i: float(i) for i in range(100)}
        rng = np.random.RandomState(1)
        chosen = select_active_subset(
            rows, losses, hard_frac=0.3, random_floor=0.1, rng=rng
        )
        # 30 hard + ~10 random from remaining = about 40, unique
        self.assertEqual(len(chosen), len(set(chosen)))
        self.assertGreaterEqual(len(chosen), 30)
        self.assertLessEqual(len(chosen), 40)
        # hardest 30 by loss desc are i=99..70
        hard = set(range(70, 100))
        self.assertTrue(hard.issubset(set(chosen)))

    def test_missing_loss_treated_as_hard(self) -> None:
        from python.train.exp_active_subset import select_active_subset

        rows = [{"i": i, "kind": "train"} for i in range(5)]
        losses = {0: 1.0, 1: 2.0}  # 2,3,4 missing
        chosen = select_active_subset(
            rows, losses, hard_frac=0.4, random_floor=0.0, rng=np.random.RandomState(0)
        )
        # With missing -> +inf hard rank, top 40% of 5 = 2 ids from the missing/high
        self.assertEqual(len(chosen), 2)


if __name__ == "__main__":
    unittest.main()
