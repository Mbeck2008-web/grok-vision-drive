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
        man = ts.build_split_manifest(plan, source_maps=MAPS, heldout_map="utah", bar_heldout_from_holdout=True)
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
                               ssl_pretrain=True, decision_tier="2.0",
                           registry=Path(tmp) / "registry")


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
                                 seed=7, ssl_pretrain=True, decision_tier="iteration",
                           registry=Path(tmp) / "registry")
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
                           ssl_pretrain=False, decision_tier="iteration",
                           registry=Path(tmp) / "registry")
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
                           ssl_pretrain=False, decision_tier="iteration",
                           registry=Path(tmp) / "registry")
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
                               ssl_pretrain=True, decision_tier="best-of-3",
                           registry=Path(tmp) / "registry")


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
                           ssl_pretrain=True, decision_tier="iteration",
                           registry=Path(tmp) / "registry")
            with self.assertRaises(ts.SplitError) as ctx:
                ts.preregister(out, split_path=sp, git_sha="c" * 40, config={}, seed=3,
                               ssl_pretrain=True, decision_tier="iteration",
                           registry=Path(tmp) / "registry")
            self.assertIn("re-roll", str(ctx.exception))


class RegistryRequired(unittest.TestCase):
    """check_no_seed_reroll across run folders (the guard may not depend on a caller remembering it)."""

    def _split(self, tmp: Path) -> Path:
        from python.train import trainer2_split as ts

        a = _rows("recA", 200)
        man = ts.build_split_manifest(_plan(a[:150], a[190:]), source_maps=MAPS, heldout_map="utah")
        return ts.write_split_manifest(tmp / "split_manifest.json", man)

    def test_registry_is_a_required_argument(self) -> None:
        import inspect

        from python.train import trainer2_split as ts

        param = inspect.signature(ts.preregister).parameters["registry"]
        self.assertIs(param.default, inspect.Parameter.empty)
        with tempfile.TemporaryDirectory() as tmp:
            sp = self._split(Path(tmp))
            with self.assertRaises(TypeError):
                ts.preregister(Path(tmp) / "run1", split_path=sp, git_sha="c" * 40, config={}, seed=1,
                               ssl_pretrain=True, decision_tier="iteration")
            for bad in (None, ""):
                with self.assertRaises(ts.SplitError) as ctx:
                    ts.preregister(Path(tmp) / "run1", split_path=sp, git_sha="c" * 40, config={}, seed=1,
                                   ssl_pretrain=True, decision_tier="iteration", registry=bad)
                self.assertIn("registry is required", str(ctx.exception))
            self.assertFalse((Path(tmp) / "run1" / ts.RUN_MANIFEST).exists())

    def test_reroll_in_a_new_folder_refused_through_shared_registry(self) -> None:
        from python.train import trainer2_split as ts

        with tempfile.TemporaryDirectory() as tmp:
            sp = self._split(Path(tmp))
            reg = Path(tmp) / "registry"
            ts.preregister(Path(tmp) / "run1", split_path=sp, git_sha="c" * 40, config={"x": 1}, seed=1,
                           ssl_pretrain=True, decision_tier="iteration", registry=reg)
            with self.assertRaises(ts.SplitError) as ctx:
                ts.preregister(Path(tmp) / "run2", split_path=sp, git_sha="c" * 40, config={"x": 1}, seed=2,
                               ssl_pretrain=True, decision_tier="iteration", registry=reg)
            self.assertIn("re-roll", str(ctx.exception))
            self.assertFalse((Path(tmp) / "run2").exists())
            # SSL on/off is a different arm: not a re-roll.
            ts.preregister(Path(tmp) / "run3", split_path=sp, git_sha="c" * 40, config={"x": 1}, seed=2,
                           ssl_pretrain=False, decision_tier="iteration", registry=reg)
            self.assertEqual(len(list(reg.glob("*.json"))), 2)


class RowsNeedFiniteTime(unittest.TestCase):
    """A row without a finite t cannot be purged or checked, so it is refused, not kept."""

    def _no_t(self, rows, *, bad=None):
        out = []
        for row in rows:
            row = dict(row)
            if bad == "missing":
                del row["t"]
            else:
                row["t"] = bad
            out.append(row)
        return out

    def test_missing_nan_inf_t_refused_when_recording_has_holdout(self) -> None:
        from python.train import trainer2_split as ts

        rows = _rows("recA", 100)
        for bad in ("missing", float("nan"), float("inf"), "abc", None):
            with self.subTest(bad=bad):
                train = self._no_t(rows[:50], bad=bad)  # train rows adjacent to the holdout
                with self.assertRaises(ts.SplitError) as ctx:
                    ts.build_split_manifest(
                        [{"kind": "train", "rows": train}, {"kind": "holdout", "rows": rows[50:]}],
                        source_maps=MAPS, heldout_map=None)
                self.assertIn("gap cannot be enforced", str(ctx.exception))  # refused at build, not only at check
                hold = self._no_t(rows[50:], bad=bad)  # holdout rows without t
                with self.assertRaises(ts.SplitError) as ctx:
                    ts.build_split_manifest(
                        [{"kind": "train", "rows": rows[:50]}, {"kind": "holdout", "rows": hold}],
                        source_maps=MAPS, heldout_map=None)
                self.assertIn("gap cannot be enforced", str(ctx.exception))

    def test_one_bad_row_among_good_ones_is_refused(self) -> None:
        from python.train import trainer2_split as ts

        rows = _rows("recA", 200)
        rows[10] = dict(rows[10], t=float("nan"))  # far from the holdout, still unprovable
        with self.assertRaises(ts.SplitError) as ctx:
            ts.build_split_manifest(_plan(rows[:100], rows[150:]), source_maps=MAPS, heldout_map=None)
        self.assertIn("gap cannot be enforced", str(ctx.exception))

    def test_recording_without_holdout_rows_may_lack_t(self) -> None:
        from python.train import trainer2_split as ts

        a = _rows("recA", 200)
        b = self._no_t(_rows("recB", 30), bad="missing")  # no holdout row in recB: nothing to purge against
        man = ts.build_split_manifest(
            [{"kind": "train", "rows": a[:100] + b}, {"kind": "holdout", "rows": a[150:]}],
            source_maps=MAPS, heldout_map=None)
        self.assertEqual(len([k for k in man["train"] if k[0] == "recB"]), 30)
        ts.check_split_manifest(man)

    def test_check_refuses_a_tampered_manifest_with_null_t(self) -> None:
        from python.train import trainer2_split as ts

        a = _rows("recA", 200)
        man = ts.build_split_manifest(_plan(a[:100], a[150:]), source_maps=MAPS, heldout_map=None)
        for split in ("train", "holdout"):
            bad = json.loads(json.dumps(man))
            bad[split][0][2] = None
            bad["sha256"] = ts.manifest_sha256(bad)
            with self.assertRaises(ts.SplitError) as ctx:
                ts.check_split_manifest(bad)
            self.assertIn("no finite t", str(ctx.exception))


def _pos_rows(source: str, xs, *, t0: float = 0.0, dt: float = 1.0, y: float = 10.0, session: str | None = None):
    return [
        {"source": source, "i": i, "t": round(t0 + i * dt, 6), "session": session or source,
         "ego": {"pos": [float(x), y]}}
        for i, x in enumerate(xs)
    ]


# One 250 m holdout cell (0, 0) with the #187 50 m buffer.
CELLS = {"cell_m": 250.0, "buffer_m": 50.0, "cells": [{"cx": 0, "cy": 0}]}


class CellsDisjoint(unittest.TestCase):
    """check_trainer2_split_no_overlap: 'cells disjoint when on' (spec, Data split and holdout)."""

    def _manifest(self):
        from python.train import trainer2_split as ts

        # t spacing 100 s keeps the 4 s time gap out of this test.
        train = _pos_rows("recA", [-600.0, -500.0, -300.0, -260.0, 400.0, 600.0], dt=100.0)
        near = _pos_rows("recA", [-30.0], t0=700.0)  # cell (-1, 0), 30 m from the holdout cell: buffer
        near[0]["i"] = 6
        nopos = [{"source": "recA", "i": 7, "t": 800.0, "session": "recA"}]  # no position: unprovable, purged
        hold = _pos_rows("recA", [10.0, 100.0, 240.0], t0=900.0)
        for k, row in enumerate(hold):
            row["i"] = 8 + k
        plan = [{"kind": "train", "rows": train + near + nopos}, {"kind": "holdout", "rows": hold}]
        return ts.build_split_manifest(plan, source_maps=MAPS, heldout_map=None, cells=CELLS)

    def test_cells_split_is_disjoint_and_buffered(self) -> None:
        from python.train import trainer2_split as ts

        man = self._manifest()
        self.assertEqual([k[1] for k in man["holdout"]], [8, 9, 10])
        self.assertEqual([k[1] for k in man["train"]], [0, 1, 2, 3, 4, 5])
        self.assertEqual(man["purged_buffer_rows"], 2)  # the 30 m row and the row with no position
        block = man["cells"]
        self.assertEqual(block["listed"], [[0, 0]])
        self.assertEqual(block["holdout"], [[0, 0]] * 3)
        self.assertFalse({tuple(c) for c in block["train"]} & {tuple(c) for c in block["listed"]})
        ts.check_split_manifest(man)

    def test_train_row_in_a_listed_cell_refused_at_build(self) -> None:
        from python.train import trainer2_split as ts

        train = _pos_rows("recA", [-600.0, 120.0], dt=100.0)
        hold = _pos_rows("recA", [10.0], t0=900.0)
        hold[0]["i"] = 5
        with self.assertRaises(ts.SplitError) as ctx:
            ts.build_split_manifest([{"kind": "train", "rows": train}, {"kind": "holdout", "rows": hold}],
                                    source_maps=MAPS, heldout_map=None, cells=CELLS)
        self.assertIn("inside a listed holdout cell", str(ctx.exception))

    def test_holdout_row_outside_every_listed_cell_refused_at_build(self) -> None:
        from python.train import trainer2_split as ts

        train = _pos_rows("recA", [-600.0], dt=100.0)
        hold = _pos_rows("recA", [10.0, 800.0], t0=900.0)
        hold[0]["i"], hold[1]["i"] = 5, 6
        with self.assertRaises(ts.SplitError) as ctx:
            ts.build_split_manifest([{"kind": "train", "rows": train}, {"kind": "holdout", "rows": hold}],
                                    source_maps=MAPS, heldout_map=None, cells=CELLS)
        self.assertIn("not inside a listed holdout cell", str(ctx.exception))

    def test_check_refuses_tampered_cells(self) -> None:
        from python.train import trainer2_split as ts

        man = self._manifest()

        def tampered(edit):
            bad = json.loads(json.dumps(man))
            edit(bad)
            bad["sha256"] = ts.manifest_sha256(bad)
            return bad

        cases = {
            "train cell moved into a listed cell": (lambda m: m["cells"]["train"].__setitem__(0, [0, 0]), "inside holdout cell"),
            "holdout cell outside the listed cells": (lambda m: m["cells"]["holdout"].__setitem__(0, [5, 5]), "not a listed"),
            "cells block shorter than the rows": (lambda m: m["cells"]["train"].pop(), "cells for"),
            "empty listed cells": (lambda m: m["cells"].__setitem__("listed", []), "lists no holdout cell"),
        }
        for name, (edit, needle) in cases.items():
            with self.subTest(name):
                with self.assertRaises(ts.SplitError) as ctx:
                    ts.check_split_manifest(tampered(edit))
                self.assertIn(needle, str(ctx.exception))

    def test_malformed_cells_spec_refused(self) -> None:
        from python.train import trainer2_split as ts

        hold = _pos_rows("recA", [10.0])
        for spec in ({"cell_m": 250.0, "cells": []}, {"cell_m": 0.0, "cells": [{"cx": 0, "cy": 0}]},
                     {"cells": [{"cx": "a", "cy": 0}]}):
            with self.subTest(spec=spec):
                with self.assertRaises(ts.SplitError) as ctx:
                    ts.build_split_manifest([{"kind": "holdout", "rows": hold}], source_maps=MAPS,
                                            heldout_map=None, cells=spec)
                self.assertIn("cells spec", str(ctx.exception))

    def test_real_choose_holdout_cells_spec(self) -> None:
        from python.train import trainer2_split as ts
        from python.train.continual import choose_holdout_cells

        rows = _pos_rows("recA", [float(x) for x in range(-1000, 1000, 10)], dt=100.0)
        for row in rows:
            row["ego"]["speed_mps"] = 8.0
        spec = choose_holdout_cells(rows, seed=3, fraction=0.2)
        listed = {(c["cx"], c["cy"]) for c in spec["cells"]}
        self.assertTrue(listed)
        hold = [r for r in rows if (int(r["ego"]["pos"][0] // 250), int(r["ego"]["pos"][1] // 250)) in listed]
        train = [r for r in rows if r not in hold]
        man = ts.build_split_manifest([{"kind": "train", "rows": train}, {"kind": "holdout", "rows": hold}],
                                      source_maps=MAPS, heldout_map=None, cells=spec)
        self.assertEqual(len(man["holdout"]), len(hold))
        self.assertGreater(man["purged_buffer_rows"], 0)
        self.assertEqual(len(man["train"]) + man["purged_buffer_rows"], len(train))
        ts.check_split_manifest(man)


class SortedUniqueGuard(unittest.TestCase):
    """check_split_manifest refuses unsorted or duplicate keys even when the hash is recomputed."""

    def test_unsorted_and_duplicate_keys_refused(self) -> None:
        from python.train import trainer2_split as ts

        a = _rows("recA", 200)
        man = ts.build_split_manifest(_plan(a[:100], a[150:]), source_maps=MAPS, heldout_map=None)
        for split in ("train", "holdout"):
            unsorted = json.loads(json.dumps(man))
            unsorted[split].reverse()
            unsorted["sha256"] = ts.manifest_sha256(unsorted)
            with self.assertRaises(ts.SplitError) as ctx:
                ts.check_split_manifest(unsorted)
            self.assertIn("sorted and unique", str(ctx.exception))
            dup = json.loads(json.dumps(man))
            dup[split].insert(1, list(dup[split][0]))  # still sorted, key repeated
            dup["sha256"] = ts.manifest_sha256(dup)
            with self.assertRaises(ts.SplitError) as ctx:
                ts.check_split_manifest(dup)
            self.assertIn("sorted and unique", str(ctx.exception))


class HeldoutMapBarFlag(unittest.TestCase):
    """DECIDED (Michael 2026-11-26, "for now"): the held-out map is NOT barred from the matched holdout.

    Held-out-map rows stay out of train only; the held-out-map test comes later. Both settings
    stay pinned (``bar_heldout_from_holdout=True`` still bars it from the holdout).
    """

    def _plan(self):
        a = _rows("recA", 200)
        u = _rows("recU", 200)
        return [
            {"kind": "train", "rows": a[:150] + u[:150]},
            {"kind": "holdout", "rows": a[190:] + u[190:]},
        ], u

    def test_current_default_is_decided(self) -> None:
        from python.train import trainer2_split as ts

        self.assertIs(ts.BAR_HELDOUT_MAP_FROM_HOLDOUT, False)  # DECIDED (Michael 2026-11-26)
        plan, _u = self._plan()
        man = ts.build_split_manifest(plan, source_maps=MAPS, heldout_map="utah")
        self.assertIs(man["heldout_map_bars_holdout"], False)
        self.assertFalse([k for k in man["train"] if k[3] == "utah"])
        self.assertEqual(len([k for k in man["holdout"] if k[3] == "utah"]), 10)

    def test_barring_keeps_heldout_map_out_of_both(self) -> None:
        from python.train import trainer2_split as ts

        plan, _u = self._plan()
        man = ts.build_split_manifest(plan, source_maps=MAPS, heldout_map="utah", bar_heldout_from_holdout=True)
        self.assertIs(man["heldout_map_bars_holdout"], True)
        self.assertFalse([k for k in man["holdout"] if k[3] == "utah"])

    def test_not_barring_keeps_heldout_map_in_holdout_only(self) -> None:
        from python.train import trainer2_split as ts

        plan, u = self._plan()
        man = ts.build_split_manifest(plan, source_maps=MAPS, heldout_map="utah", bar_heldout_from_holdout=False)
        self.assertIs(man["heldout_map_bars_holdout"], False)
        self.assertFalse([k for k in man["train"] if k[3] == "utah"])
        self.assertEqual(len([k for k in man["holdout"] if k[3] == "utah"]), 10)
        self.assertEqual(man["heldout_map_rows"], 150)  # only the train rows are excluded
        ts.check_split_manifest(man)
        for what in ("ssl", "teacher", "distill_cache"):
            with self.assertRaises(ts.SplitError):
                ts.check_subset_of_train(u[190:195], man, what=what)

    def test_check_enforces_the_recorded_setting(self) -> None:
        from python.train import trainer2_split as ts

        plan, _u = self._plan()
        loose = ts.build_split_manifest(plan, source_maps=MAPS, heldout_map="utah", bar_heldout_from_holdout=False)
        flipped = json.loads(json.dumps(loose))
        flipped["heldout_map_bars_holdout"] = True  # claims the bar, holds utah rows
        flipped["sha256"] = ts.manifest_sha256(flipped)
        with self.assertRaises(ts.SplitError) as ctx:
            ts.check_split_manifest(flipped)
        self.assertIn("held-out map", str(ctx.exception))
        sneaky = json.loads(json.dumps(loose))  # utah row smuggled into train under either setting
        sneaky["train"].append(["recU", 1000, 100.0, "utah", "recU"])
        sneaky["train"].sort()
        sneaky["sha256"] = ts.manifest_sha256(sneaky)
        with self.assertRaises(ts.SplitError):
            ts.check_split_manifest(sneaky)

    def test_check_refuses_a_row_with_no_map_under_a_heldout_map(self) -> None:
        from python.train import trainer2_split as ts

        plan, _u = self._plan()
        man = ts.build_split_manifest(plan, source_maps=MAPS, heldout_map="utah")
        for split in ("train", "holdout"):
            bad = json.loads(json.dumps(man))
            bad[split][0][3] = None
            bad["sha256"] = ts.manifest_sha256(bad)
            with self.assertRaises(ts.SplitError) as ctx:
                ts.check_split_manifest(bad)
            self.assertIn("has no map", str(ctx.exception))

    def test_unknown_map_message_names_the_missing_map(self) -> None:
        from python.train import trainer2_split as ts

        with self.assertRaises(ts.SplitError) as ctx:
            ts.build_split_manifest([{"kind": "train", "rows": _rows("recZ", 5)}], source_maps=MAPS, heldout_map="utah")
        self.assertIn("no source_maps entry", str(ctx.exception))  # the build refuses, not just the later check


if __name__ == "__main__":
    unittest.main()
