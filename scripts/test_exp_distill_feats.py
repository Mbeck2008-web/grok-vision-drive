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


def _teacher_token(row):
    return np.full(64, float(row["i"]) + 1.0, dtype=np.float32)


def _teacher_target(row, dim: int):
    return np.full(dim, float(row["i"]) + 3.0, dtype=np.float32)


class ExpDistillFeatsTests(unittest.TestCase):
    def test_flag_off_by_default(self) -> None:
        from python.train.train_scene import build_parser

        args = build_parser().parse_args([])
        self.assertFalse(bool(getattr(args, "exp_distill_feats", False)))
        self.assertEqual(str(getattr(args, "exp_distill_cache", "") or ""), "")
        self.assertEqual(str(getattr(args, "exp_distill_teacher", "") or ""), "")

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

    def test_cache_refuses_missing_kind(self) -> None:
        from python.train.exp_distill import DistillError, build_feat_cache

        rows = [{"i": 0}, {"i": 1, "kind": "train"}]

        def encode(row):
            return np.ones(64, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DistillError) as ctx:
                build_feat_cache(Path(tmp) / "feats", rows, encode=encode)
            self.assertIn("no kind", str(ctx.exception))

    def test_split_cache_uses_187_plan_and_refuses_raw(self) -> None:
        from python.train.exp_distill import DistillError, build_feat_cache, split_tagged_rows

        raw = [{"i": i, "t": float(i), "dt_s": 0.1, "session": "rec"} for i in range(50)]
        train_rows, holdout_rows = split_tagged_rows(raw, {"multi": False}, holdout_blocks=1)
        self.assertTrue(train_rows)
        self.assertTrue(holdout_rows)
        self.assertTrue(all(r["kind"] == "train" for r in train_rows))
        self.assertTrue(all(r["kind"] == "holdout" for r in holdout_rows))
        train_ids = {int(r["i"]) for r in train_rows}
        hold_ids = {int(r["i"]) for r in holdout_rows}
        self.assertFalse(train_ids & hold_ids)
        # Raw untagged rows must not be written.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DistillError):
                build_feat_cache(Path(tmp) / "bad", raw, encode=lambda r: np.ones(64, dtype=np.float32))
            written = build_feat_cache(
                Path(tmp) / "ok",
                train_rows,
                encode=lambda r: np.full(64, float(r["i"]) + 1.0, dtype=np.float32),
            )
            self.assertEqual(sorted(written), sorted(train_ids))
            self.assertTrue(set(written).isdisjoint(hold_ids))

    def test_missing_teacher_fails_loudly(self) -> None:
        from python.train.exp_distill import DistillError, load_teacher

        with self.assertRaises(DistillError):
            load_teacher("")
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.json"
            with self.assertRaises(DistillError) as ctx:
                load_teacher(missing)
            self.assertIn("not found", str(ctx.exception).lower())

    def test_student_targets_come_from_teacher(self) -> None:
        from python.debug.lane_targets import LANE_GT_DIM
        from python.train.exp_distill import (
            build_feat_cache,
            load_feat_cache,
            load_teacher,
            run_student_epochs,
            write_teacher_features,
        )

        rows = [
            {"i": 0, "kind": "train"},
            {"i": 1, "kind": "holdout"},
            {"i": 2, "kind": "train"},
            {"i": 3, "kind": "holdout"},
        ]
        entries = [
            {"i": r["i"], "token": _teacher_token(r), "target": _teacher_target(r, LANE_GT_DIM)}
            for r in rows
        ]
        seen_targets: list[np.ndarray] = []
        with tempfile.TemporaryDirectory() as tmp:
            teacher_path = write_teacher_features(Path(tmp) / "teacher.json", entries)
            teacher = load_teacher(teacher_path)
            cache_dir = Path(tmp) / "feats"
            written = build_feat_cache(
                cache_dir,
                rows,
                encode=teacher.encode,
                target_fn=teacher.target,
            )
            self.assertEqual(sorted(written), [0, 2])
            cache = load_feat_cache(cache_dir)
            np.testing.assert_array_equal(cache.token(0), _teacher_token({"i": 0}))
            np.testing.assert_array_equal(cache.target(0), _teacher_target({"i": 0}, LANE_GT_DIM))
            self.assertFalse(np.allclose(cache.target(0), 0))
            self.assertFalse(np.allclose(cache.token(0), 0))

            def target_fn(row_i, feat_cache):
                tgt = feat_cache.target(row_i)
                seen_targets.append(np.asarray(tgt).copy())
                return tgt

            hold_rows = [r for r in rows if r["kind"] == "holdout"]
            from python.train.exp_distill import encode_rows_to_cache

            holdout = encode_rows_to_cache(hold_rows, encode=teacher.encode, target_fn=teacher.target)
            stats = run_student_epochs(
                cache,
                target_fn=target_fn,
                epochs=2,
                lr=1e-2,
                seed=73,
                hidden=32,
                holdout=holdout,
                holdout_note="synthetic",
            )
            self.assertGreaterEqual(len(seen_targets), 2)
            allowed = [_teacher_target({"i": i}, LANE_GT_DIM) for i in (0, 1, 2, 3)]
            for tgt in seen_targets:
                self.assertFalse(np.allclose(tgt, 0.0))
                # Targets match the teacher file, not a zero vector.
                self.assertTrue(any(np.allclose(tgt, one) for one in allowed))
            # Training steps themselves used the train-row teacher targets.
            train_seen = seen_targets[: 2 * 2]
            self.assertTrue(
                all(
                    np.allclose(tgt, _teacher_target({"i": 0}, LANE_GT_DIM))
                    or np.allclose(tgt, _teacher_target({"i": 2}, LANE_GT_DIM))
                    for tgt in train_seen
                )
            )
            self.assertIn("holdout_mae", stats)
            self.assertEqual(int(stats["holdout_n"]), 2)
            self.assertEqual(stats["holdout_note"], "synthetic")
            self.assertTrue(np.isfinite(float(stats["holdout_mae"])))
            # The previous zero-target stub reported last_loss=0.4848 on this seed.
            # A real teacher target of 3.0 / 5.0 cannot reproduce that path.
            self.assertNotAlmostEqual(float(stats["last_loss"]), 0.4848, places=3)

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

    def test_run_student_epochs_refuses_missing_target_fn(self) -> None:
        from python.train.exp_distill import DistillError, FeatCache, run_student_epochs

        tokens = np.ones((1, 64), dtype=np.float32)
        cache = FeatCache(Path("memory"), [0], tokens, {"token_dim": 64, "indices": [0], "n": 1})
        with self.assertRaises(DistillError):
            run_student_epochs(cache, target_fn=None, epochs=1)


if __name__ == "__main__":
    unittest.main()
