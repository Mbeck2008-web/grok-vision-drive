"""Trainer 2.0 PR 1: split manifest + pre-registration (docs/specs/trainer_2.md).

Checks: check_trainer2_split_no_overlap, check_manifest_hashed_before_train,
check_no_seed_reroll, check_heldout_map_excluded. Pure Python (no torch).
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rows(source: str, n: int, *, t0: float = 0.0, dt: float = 0.1, session: str | None = None):
    return [
        {"source": source, "i": i, "t": round(t0 + i * dt, 6), "session": session or source}
        for i in range(n)
    ]


def _plan(train_rows, hold_rows):
    plan = []
    if train_rows:
        plan.append({"kind": "train", "rows": list(train_rows)})
        # A rare chunk listed twice must not duplicate keys in the manifest.
        plan.append({"kind": "train", "rows": list(train_rows[:3])})
    if hold_rows:
        plan.append({"kind": "holdout", "rows": list(hold_rows)})
    return plan


MAPS = {"recA": "west_coast_usa", "recB": "italy", "recU": "utah"}


class SplitNoOverlap(unittest.TestCase):
    """check_trainer2_split_no_overlap"""

    def test_keys_sorted_disjoint_and_hashed(self) -> None:
        from python.train import trainer2_split as ts

        rows = _rows("recA", 200)
        plan = _plan(rows[:150], rows[190:])  # 4 s gap between t=14.9 and t=19.0
        man = ts.build_split_manifest(plan, source_maps=MAPS, heldout_map="utah")
        train = [tuple(k[:2]) for k in man["train"]]
        hold = [tuple(k[:2]) for k in man["holdout"]]
        self.assertEqual(train, sorted(set(train)))
        self.assertEqual(hold, sorted(set(hold)))
        self.assertFalse(set(train) & set(hold))
        self.assertEqual(len(train), 150)
        self.assertEqual(man["sha256"], ts.manifest_sha256(man))
        ts.check_split_manifest(man)

    def test_real_pass_plan_blocks(self) -> None:
        from python.train import trainer2_split as ts
        from python.train.continual import CHUNK_MAX, pass_plan

        raw = [dict(r, dt_s=0.1) for r in _rows("recA", 3000)]
        for blocks in (1, 3):
            plan = pass_plan(raw, CHUNK_MAX, CHUNK_MAX // 2, blocks=blocks)
            man = ts.build_split_manifest(plan, source_maps=MAPS, heldout_map="utah")
            self.assertTrue(man["train"] and man["holdout"], blocks)
            # blocks=3 already leaves the 4 s gap; the untagged blocks=1 tail cut
            # (split_prefix_holdout) leaves none, so the manifest purges 39 rows.
            self.assertEqual(man["purged_gap_rows"], 0 if blocks == 3 else 39, blocks)
            ts.check_split_manifest(man)

    def test_overlap_refused(self) -> None:
        from python.train import trainer2_split as ts

        rows = _rows("recA", 100)
        plan = _plan(rows[:60], rows[59:])  # row 59 in both
        with self.assertRaises(ts.SplitError) as ctx:
            ts.build_split_manifest(plan, source_maps=MAPS, heldout_map="utah")
        self.assertIn("both", str(ctx.exception))

    def test_train_row_inside_gap_is_purged(self) -> None:
        from python.train import trainer2_split as ts

        rows = _rows("recA", 200)
        plan = _plan(rows[:150], rows[150:])  # adjacent: no gap from the plan
        man = ts.build_split_manifest(plan, source_maps=MAPS, heldout_map="utah")
        hold_t0 = min(k[2] for k in man["holdout"])
        self.assertTrue(all(hold_t0 - k[2] >= ts.HOLDOUT_GAP_S for k in man["train"]))
        self.assertEqual(man["purged_gap_rows"], 39)  # t in (11.0, 14.9]
        ts.check_split_manifest(man)

    def test_gap_is_per_recording(self) -> None:
        from python.train import trainer2_split as ts

        a = _rows("recA", 100)
        b = _rows("recB", 100)  # same timestamps, other recording: no purge
        plan = [{"kind": "train", "rows": b}, {"kind": "holdout", "rows": a}]
        man = ts.build_split_manifest(plan, source_maps=MAPS, heldout_map="utah")
        self.assertEqual(man["purged_gap_rows"], 0)
        self.assertEqual(len(man["train"]), 100)

    def test_check_rejects_tampered_manifest(self) -> None:
        from python.train import trainer2_split as ts

        rows = _rows("recA", 200)
        man = ts.build_split_manifest(_plan(rows[:150], rows[190:]), source_maps=MAPS, heldout_map="utah")
        bad = json.loads(json.dumps(man))
        bad["holdout"].append(bad["train"][0])
        with self.assertRaises(ts.SplitError):
            ts.check_split_manifest(bad)
        bad2 = json.loads(json.dumps(man))
        bad2["train"].append(["recA", 189, 18.9, "west_coast_usa", "recA"])
        bad2["train"].sort()
        bad2["sha256"] = ts.manifest_sha256(bad2)
        with self.assertRaises(ts.SplitError) as ctx:
            ts.check_split_manifest(bad2)
        self.assertIn("gap", str(ctx.exception))

    def test_ssl_rows_subset_of_train(self) -> None:
        from python.train import trainer2_split as ts

        rows = _rows("recA", 200)
        man = ts.build_split_manifest(_plan(rows[:150], rows[190:]), source_maps=MAPS, heldout_map="utah")
        kept = ts.filter_rows(rows, man, "train")
        self.assertEqual([r["i"] for r in kept], list(range(150)))
        ts.check_subset_of_train(kept, man, what="ssl")
        with self.assertRaises(ts.SplitError):
            ts.check_subset_of_train(rows[140:160], man, what="ssl")

    def test_temporal_target_must_stay_in_split(self) -> None:
        from python.train import trainer2_split as ts

        rows = _rows("recA", 200)
        man = ts.build_split_manifest(_plan(rows[:150], rows[190:]), source_maps=MAPS, heldout_map="utah")
        self.assertTrue(ts.same_split(man, ("recA", 10), ("recA", 20)))
        self.assertFalse(ts.same_split(man, ("recA", 149), ("recA", 195)))
        self.assertFalse(ts.same_split(man, ("recA", 149), ("recA", 170)))  # purged/absent


class HeldoutMapExcluded(unittest.TestCase):
    """check_heldout_map_excluded"""

    def test_heldout_map_rows_never_in_train_or_holdout(self) -> None:
        from python.train import trainer2_split as ts

        a = _rows("recA", 200)
        u = _rows("recU", 200)
        plan = [
            {"kind": "train", "rows": a[:150] + u[:150]},
            {"kind": "holdout", "rows": a[190:] + u[190:]},
        ]
        man = ts.build_split_manifest(plan, source_maps=MAPS, heldout_map="utah")
        self.assertEqual(man["heldout_map"], "utah")
        for split in ("train", "holdout"):
            self.assertFalse([k for k in man[split] if k[0] == "recU"], split)
            self.assertFalse([k for k in man[split] if k[3] == "utah"], split)
        self.assertEqual(man["heldout_map_rows"], 160)
        self.assertEqual(ts.filter_rows(u, man, "train"), [])
        self.assertEqual(ts.filter_rows(u, man, "holdout"), [])
        ts.check_split_manifest(man)
        for what in ("ssl", "teacher", "distill_cache"):
            with self.assertRaises(ts.SplitError):
                ts.check_subset_of_train(u[:5], man, what=what)

    def test_row_map_field_wins_and_unknown_map_refused(self) -> None:
        from python.train import trainer2_split as ts

        rows = [dict(r, map="utah") for r in _rows("recA", 50)]
        man = ts.build_split_manifest([{"kind": "train", "rows": rows}], source_maps=MAPS, heldout_map="utah")
        self.assertEqual(man["train"], [])
        with self.assertRaises(ts.SplitError) as ctx:
            ts.build_split_manifest([{"kind": "train", "rows": _rows("recZ", 5)}], source_maps=MAPS, heldout_map="utah")
        self.assertIn("map", str(ctx.exception))

    def test_tampered_heldout_map_row_fails_check(self) -> None:
        from python.train import trainer2_split as ts

        a = _rows("recA", 200)
        man = ts.build_split_manifest(_plan(a[:150], a[190:]), source_maps=MAPS, heldout_map="west_coast_usa")
        self.assertEqual(man["train"], [])
        man2 = ts.build_split_manifest(_plan(a[:150], a[190:]), source_maps=MAPS, heldout_map="utah")
        bad = json.loads(json.dumps(man2))
        bad["heldout_map"] = "west_coast_usa"
        bad["sha256"] = ts.manifest_sha256(bad)
        with self.assertRaises(ts.SplitError) as ctx:
            ts.check_split_manifest(bad)
        self.assertIn("held-out map", str(ctx.exception))

    def test_2_0_claim_needs_heldout_map(self) -> None:
        from python.train import trainer2_split as ts

        a = _rows("recA", 200)
        man = ts.build_split_manifest(_plan(a[:150], a[190:]), source_maps=MAPS, heldout_map=None)
        self.assertIsNone(man["heldout_map"])
        with self.assertRaises(ts.SplitError):
            ts.require_heldout_map(man)
        with tempfile.TemporaryDirectory() as tmp:
            sp = ts.write_split_manifest(Path(tmp) / "split_manifest.json", man)
            with self.assertRaises(ts.SplitError):
                ts.preregister(Path(tmp), split_path=sp, git_sha="a" * 40, config={"lr": 1e-3}, seed=0,
                               ssl_pretrain=True, decision_tier="2.0")


class ManifestBeforeTrain(unittest.TestCase):
    """check_manifest_hashed_before_train"""

    def _split(self, tmp: Path) -> Path:
        from python.train import trainer2_split as ts

        a = _rows("recA", 200)
        man = ts.build_split_manifest(_plan(a[:150], a[190:]), source_maps=MAPS, heldout_map="utah")
        return ts.write_split_manifest(tmp / "split_manifest.json", man)

    def test_require_before_step0(self) -> None:
        from python.train import trainer2_split as ts

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertRaises(ts.SplitError):
                ts.require_preregistered(out)
            sp = self._split(out)
            run = ts.preregister(out, split_path=sp, git_sha="b" * 40, config={"lr": 1e-3, "epochs": 2},
                                 seed=7, ssl_pretrain=True, decision_tier="iteration")
            self.assertEqual(run["split_sha256"], json.loads(sp.read_text())["sha256"])
            self.assertEqual(run["heldout_map"], "utah")
            self.assertTrue(run["ssl_pretrain"])
            self.assertEqual(run["decision_tier"], "iteration")
            self.assertEqual(run["sha256"], ts.manifest_sha256(run))
            got = ts.require_preregistered(out)
            self.assertEqual(got["sha256"], run["sha256"])
            self.assertFalse(list(out.glob("*.partial")))

    def test_tampered_run_or_split_manifest_refused(self) -> None:
        from python.train import trainer2_split as ts

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            sp = self._split(out)
            ts.preregister(out, split_path=sp, git_sha="b" * 40, config={"lr": 1e-3}, seed=7,
                           ssl_pretrain=False, decision_tier="iteration")
            rp = out / ts.RUN_MANIFEST
            data = json.loads(rp.read_text())
            data["seed"] = 8
            rp.write_text(json.dumps(data))
            with self.assertRaises(ts.SplitError):
                ts.require_preregistered(out)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            sp = self._split(out)
            ts.preregister(out, split_path=sp, git_sha="b" * 40, config={"lr": 1e-3}, seed=7,
                           ssl_pretrain=False, decision_tier="iteration")
            data = json.loads(sp.read_text())
            data["train"] = data["train"][1:]
            data["sha256"] = ts.manifest_sha256(data)
            sp.write_text(json.dumps(data))
            with self.assertRaises(ts.SplitError) as ctx:
                ts.require_preregistered(out)
            self.assertIn("split", str(ctx.exception))

    def test_split_manifest_not_overwritten(self) -> None:
        from python.train import trainer2_split as ts

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            sp = self._split(out)
            a = _rows("recA", 200)
            other = ts.build_split_manifest(_plan(a[:100], a[190:]), source_maps=MAPS, heldout_map="utah")
            with self.assertRaises(ts.SplitError):
                ts.write_split_manifest(sp, other)
            same = json.loads(sp.read_text())
            ts.write_split_manifest(sp, same)  # identical content is a no-op

    def test_bad_tier_refused(self) -> None:
        from python.train import trainer2_split as ts

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            sp = self._split(out)
            with self.assertRaises(ts.SplitError):
                ts.preregister(out, split_path=sp, git_sha="b" * 40, config={}, seed=0,
                               ssl_pretrain=True, decision_tier="best-of-3")


class NoSeedReroll(unittest.TestCase):
    """check_no_seed_reroll"""

    def test_second_seed_same_manifest_refused(self) -> None:
        from python.train import trainer2_split as ts

        with tempfile.TemporaryDirectory() as tmp:
            reg = Path(tmp) / "registry"
            a = _rows("recA", 200)
            man = ts.build_split_manifest(_plan(a[:150], a[190:]), source_maps=MAPS, heldout_map="utah")
            run1 = Path(tmp) / "run1"
            run1.mkdir()
            sp1 = ts.write_split_manifest(run1 / "split_manifest.json", man)
            cfg = {"lr": 1e-3, "epochs": 30}
            ts.preregister(run1, split_path=sp1, git_sha="c" * 40, config=cfg, seed=1,
                           ssl_pretrain=True, decision_tier="iteration", registry=reg)
            # Same arm, same seed (a resume) is fine.
            ts.preregister(run1, split_path=sp1, git_sha="c" * 40, config=cfg, seed=1,
                           ssl_pretrain=True, decision_tier="iteration", registry=reg)
            run2 = Path(tmp) / "run2"
            run2.mkdir()
            sp2 = ts.write_split_manifest(run2 / "split_manifest.json", man)
            with self.assertRaises(ts.SplitError) as ctx:
                ts.preregister(run2, split_path=sp2, git_sha="c" * 40, config=cfg, seed=2,
                               ssl_pretrain=True, decision_tier="iteration", registry=reg)
            self.assertIn("re-roll", str(ctx.exception))
            self.assertFalse((run2 / ts.RUN_MANIFEST).exists())
            # A different config is a different arm, not a re-roll.
            ts.preregister(run2, split_path=sp2, git_sha="c" * 40, config=dict(cfg, kd=True), seed=2,
                           ssl_pretrain=True, decision_tier="iteration", registry=reg)

    def test_existing_run_manifest_with_other_seed_refused(self) -> None:
        from python.train import trainer2_split as ts

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            a = _rows("recA", 200)
            man = ts.build_split_manifest(_plan(a[:150], a[190:]), source_maps=MAPS, heldout_map="utah")
            sp = ts.write_split_manifest(out / "split_manifest.json", man)
            ts.preregister(out, split_path=sp, git_sha="c" * 40, config={}, seed=1,
                           ssl_pretrain=True, decision_tier="iteration")
            with self.assertRaises(ts.SplitError):
                ts.preregister(out, split_path=sp, git_sha="c" * 40, config={}, seed=3,
                               ssl_pretrain=True, decision_tier="iteration")


if __name__ == "__main__":
    unittest.main()
