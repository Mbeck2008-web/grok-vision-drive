"""CPU tests for --exp-distill-feats (W2). Off by default; holdout never cached."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ExpDistillFeatsTests(unittest.TestCase):
    def test_flag_off_by_default(self) -> None:
        from python.train.train_scene import build_parser

        args = build_parser().parse_args([])
        self.assertFalse(bool(getattr(args, "exp_distill_feats", False)))
        self.assertEqual(str(getattr(args, "exp_distill_cache", "") or ""), "")

    def test_cache_skips_holdout_rows(self) -> None:
        from python.train.exp_distill import build_feat_cache, load_feat_cache

        rows = [
            {"i": 0, "kind": "train", "dt_s": 0.1},
            {"i": 1, "kind": "holdout", "dt_s": 0.1},
            {"i": 2, "kind": "train", "dt_s": 0.1},
        ]
        tokens = {
            0: np.zeros(64, dtype=np.float32),
            1: np.ones(64, dtype=np.float32),  # must not be written
            2: np.full(64, 2.0, dtype=np.float32),
        }

        def encode(row):
            return tokens[int(row["i"])]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feats"
            written = build_feat_cache(path, rows, encode=encode, holdout_kinds={"holdout"})
            self.assertEqual(sorted(written), [0, 2])
            cache = load_feat_cache(path)
            self.assertIn(0, cache.indices)
            self.assertIn(2, cache.indices)
            self.assertNotIn(1, cache.indices)
            np.testing.assert_array_equal(cache.token(0), tokens[0])
            np.testing.assert_array_equal(cache.token(2), tokens[2])

    def test_student_step_uses_cached_token_not_image(self) -> None:
        from python.train.exp_distill import StudentFromFeats, student_lane_loss

        import torch

        from python.debug.lane_targets import LANE_GT_DIM

        student = StudentFromFeats(token_dim=64, hidden=32)
        token = torch.zeros(1, 64)
        target = torch.zeros(1, int(LANE_GT_DIM))
        loss = student_lane_loss(student, token, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


if __name__ == "__main__":
    unittest.main()
