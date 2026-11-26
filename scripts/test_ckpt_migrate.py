#!/usr/bin/env python3
"""Opt-in 96 -> 80 object-class checkpoint migration (scripts/migrate_ckpt_classes.py).

Main refuses a pre-v2 checkpoint (6 classes x 16 slots = 96 outputs) by name.
The migration tool is a separate, explicit step; these checks pin both sides:

* default: the loaders still refuse the 96 ckpt, with the unchanged message, even
  when a migrated sibling sits next to it; no python/ code imports the tool.
* tool: row j of the new head is exactly row map[j] of the old head; the dropped
  class is gone; the input file is byte-identical; output is create-only.
* mapping: built from names; covers every current class exactly once; an old
  class name with no policy fails loudly (names it, writes nothing).

Synthetic checkpoint: a real tiny CPU SceneNet state_dict with the object-class
head widened to 96 rows of known values.
"""
# gvd-ci: requires torch
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import traceback
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

OLD_NAMES = ("vehicle", "car", "truck", "bus", "pedestrian", "bike")  # gvd-allow-pedestrian-word
OLD_WORD = "pedestrian"  # gvd-allow-pedestrian-word
MSG_96 = "object_class has 96 outputs, this build expects 80"
HEAD_KEYS = ("object_class_head.weight", "object_class_head.bias")
META_FIELDS = {"run": "20261107T011144Z", "epoch": 11, "step": 77, "cams": ["main"], "scale": 0.25, "backbone": "scratch"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_old_ckpt(dirpath: Path, name: str = "strips.run20261107T011144Z.ckpt_ep11_step77.npz") -> Path:
    """A real tiny SceneNet state_dict whose object_class_head is 96 rows (distinct, known values)."""
    from python.train.epoch_ckpt import build_net, write_epoch_ckpt

    net = build_net(META_FIELDS)  # the same constructor the loader uses for this meta
    weights = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
    hidden = weights["object_class_head.weight"].shape[1]
    weights["object_class_head.weight"] = (np.arange(96)[:, None] * 1000 + np.arange(hidden)[None, :]).astype(np.float32)
    weights["object_class_head.bias"] = (np.arange(96) + 0.5).astype(np.float32)
    path = dirpath / name
    write_epoch_ckpt(path, weights, dict(META_FIELDS), min_free_bytes=0)
    return path


def _default_loader_refuses(path: Path) -> str:
    """Load through the unchanged epoch-ckpt path; return the refusal message."""
    from python.train.epoch_ckpt import build_net, load_epoch_ckpt, load_into_net
    from python.train.scene_net import ObjectClassCountError

    loaded = load_epoch_ckpt(path)
    net = build_net(loaded["meta"], loaded["weights"])
    try:
        load_into_net(net, loaded["weights"])
    except ObjectClassCountError as exc:
        return str(exc)
    raise AssertionError(f"{path.name}: default loader accepted a 96-output checkpoint")


# ------------------------------------------------------------------ default stays refused


def check_default_still_refuses() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        old = make_old_ckpt(Path(tmp))
        text = _default_loader_refuses(old)
        assert MSG_96 in text, text
        assert "checkpoint" in text and "object_class_head.weight" in text, text
        assert "Retrain." in text, text
    print(f"default load refuses the 96 ckpt: {text[:96]}...")


def check_no_load_path_hook() -> None:
    hits = []
    for path in sorted((ROOT / "python").rglob("*.py")):
        if "migrate_ckpt_classes" in path.read_text(encoding="utf-8"):
            hits.append(str(path.relative_to(ROOT)))
    assert not hits, f"load path must not know the migration tool: {hits}"
    # nothing in the guarded loaders may auto-migrate: they only call the checker.
    text = (ROOT / "python/train/epoch_ckpt.py").read_text(encoding="utf-8")
    assert "migrat" not in text.lower(), "epoch_ckpt must not migrate"
    print("no python/ module imports or names the migration tool")


def check_sibling_does_not_unrefuse() -> None:
    from python.train.epoch_ckpt import find_epoch_ckpt
    import migrate_ckpt_classes as mig

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        old = make_old_ckpt(d)
        new = mig.default_output_path(old)
        mig.migrate_file(old, new)
        assert new.exists() and new.parent == old.parent
        found = find_epoch_ckpt(d, 11)  # what --export-epoch / eval would pick
        assert found == old, (found, old)
        assert MSG_96 in _default_loader_refuses(found)
        assert _default_loader_refuses(old)
    print("migrated sibling next to the 96 ckpt: default lookup still returns the 96 file and still refuses it")


# ------------------------------------------------------------------ mapping


def check_mapping_covers_current_classes() -> None:
    import migrate_ckpt_classes as mig
    from python.train.scene_net import OBJECT_CLASSES

    assert mig.OLD_CLASSES == OLD_NAMES, mig.OLD_CLASSES
    assert mig.OLD_CLASSES_SOURCE["sha"].startswith("4e41994"), mig.OLD_CLASSES_SOURCE
    assert "code" in mig.OLD_CLASSES_SOURCE["kind"], mig.OLD_CLASSES_SOURCE
    m = mig.build_mapping(mig.OLD_CLASSES, OBJECT_CLASSES)
    assert tuple(m["new_classes"]) == tuple(OBJECT_CLASSES)  # a class-list change breaks this test
    src = m["new_from_old"]
    assert len(src) == len(OBJECT_CLASSES) and len(set(src)) == len(src), src
    for j, i in enumerate(src):
        assert mig.OLD_CLASSES[i] == OBJECT_CLASSES[j], (j, i)
    assert m["dropped"] == [OLD_WORD] and m["merged"] == [], m
    assert sorted(src + [mig.OLD_CLASSES.index(OLD_WORD)]) == list(range(6)), src  # every old class kept or dropped, once
    print(f"mapping {m['id']}: {len(src)} kept, dropped={m['dropped']}, merged={m['merged']}")


def check_unmapped_class_fails_loudly() -> None:
    import migrate_ckpt_classes as mig
    from python.train.scene_net import OBJECT_CLASSES

    with tempfile.TemporaryDirectory() as tmp:
        old = make_old_ckpt(Path(tmp))
        out = Path(tmp) / "never.npz"
        before = _sha(old)
        weird = OLD_NAMES + ("tram",)
        try:
            mig.build_mapping(weird, OBJECT_CLASSES)
        except mig.MappingError as exc:
            assert "tram" in str(exc), exc
        else:
            raise AssertionError("unknown old class did not raise")
        # same through the file API: no output, input untouched
        try:
            mig.migrate_file(old, out, old_classes=weird)
        except mig.MappingError as exc:
            assert "tram" in str(exc), exc
        else:
            raise AssertionError("migrate_file accepted an unmapped class")
        assert not out.exists(), "output written despite unmapped class"
        assert not list(Path(tmp).glob("*.partial*")) and _sha(old) == before
        # a NEW class with no old source is just as loud (never random rows)
        try:
            mig.build_mapping(mig.OLD_CLASSES, OBJECT_CLASSES + ("tram",))
        except mig.MappingError as exc:
            assert "tram" in str(exc), exc
        else:
            raise AssertionError("new class without source did not raise")
        # a reordered old list is mapped by NAME, not position
        flipped = tuple(reversed(OLD_NAMES))
        mm = mig.build_mapping(flipped, OBJECT_CLASSES)
        assert [flipped[i] for i in mm["new_from_old"]] == list(OBJECT_CLASSES)
    print("unmapped old/new class name raises MappingError naming it ('tram'); no output file, input unchanged")


# ------------------------------------------------------------------ tool


def check_migrate_values_and_loads() -> None:
    import migrate_ckpt_classes as mig
    from python.train.epoch_ckpt import build_net, load_epoch_ckpt, load_into_net
    from python.train.scene_net import N_OBJECTS, OBJECT_CLASSES

    with tempfile.TemporaryDirectory() as tmp:
        old = make_old_ckpt(Path(tmp))
        out = Path(tmp) / "new.migrated5.npz"
        before = _sha(old)
        summary = mig.migrate_file(old, out)
        assert _sha(old) == before, "input file changed"
        assert summary["old_outputs"] == 96 and summary["new_outputs"] == 80, summary
        a, b = load_epoch_ckpt(old), load_epoch_ckpt(out)
        # every non-class tensor identical, same key order
        assert list(a["weights"]) == list(b["weights"])
        for key in a["weights"]:
            if key not in HEAD_KEYS:
                assert np.array_equal(a["weights"][key], b["weights"][key]), key
        n_new, n_old = len(OBJECT_CLASSES), len(OLD_NAMES)
        for key in HEAD_KEYS:
            ow, nw = a["weights"][key], b["weights"][key]
            assert nw.shape[0] == 80 and ow.shape[0] == 96 and nw.shape[1:] == ow.shape[1:], key
            assert nw.dtype == ow.dtype, key
            for slot in range(N_OBJECTS):
                for j, name in enumerate(OBJECT_CLASSES):
                    i = OLD_NAMES.index(name)
                    assert np.array_equal(nw[slot * n_new + j], ow[slot * n_old + i]), (key, slot, name)
        # dropped class: none of its old rows (values encode the old row) survive
        w = b["weights"]["object_class_head.weight"]
        gone = {slot * n_old + OLD_NAMES.index(OLD_WORD) for slot in range(N_OBJECTS)}
        kept_ids = {int(row[0] // 1000) for row in w}
        assert not (kept_ids & gone), kept_ids & gone
        assert len(kept_ids) == 80
        # metadata
        meta = b["meta"]
        mg = meta["migration"]
        assert mg["migrated_from_outputs"] == 96 and mg["migrated_to_outputs"] == 80, mg
        assert mg["dropped"] == [OLD_WORD] and mg["merged"] == [], mg
        assert mg["old_classes"] == list(OLD_NAMES) and mg["new_classes"] == list(OBJECT_CLASSES), mg
        assert mg["old_classes_source"]["sha"].startswith("4e41994") and mg["mapping_id"], mg
        assert mg["source_file"] == old.name and mg["source_sha256"] == before, mg
        assert meta["epoch"] == 11 and meta["step"] == 77 and meta["run"] == META_FIELDS["run"]
        # loads under the unchanged guard
        net = build_net(meta, b["weights"])
        load_into_net(net, b["weights"])
        got = net.state_dict()["object_class_head.weight"].numpy()
        assert np.array_equal(got, w)
    print("migrated ckpt: row j == old row map[j] for weight+bias x16 slots, dropped rows gone, input sha unchanged, loads under guard")


def check_refusals() -> None:
    import migrate_ckpt_classes as mig
    from python.train.epoch_ckpt import build_net, write_epoch_ckpt

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        old = make_old_ckpt(d)
        out = d / "o.npz"
        mig.migrate_file(old, out)
        first = _sha(out)
        # overwrite refused unless --force; input never overwritten
        try:
            mig.migrate_file(old, out)
        except mig.MigrationRefused as exc:
            assert "exists" in str(exc), exc
        else:
            raise AssertionError("overwrite not refused")
        assert _sha(out) == first
        mig.migrate_file(old, out, force=True)
        for same in (old, d / "." / old.name):
            try:
                mig.migrate_file(old, same, force=True)
            except mig.MigrationRefused as exc:
                assert "input" in str(exc), exc
            else:
                raise AssertionError("output == input not refused")
        # already 80: refused
        net = build_net(META_FIELDS)
        w80 = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
        ok80 = d / "ok80.npz"
        write_epoch_ckpt(ok80, w80, dict(META_FIELDS), min_free_bytes=0)
        try:
            mig.migrate_file(ok80, d / "x80.npz")
        except mig.MigrationRefused as exc:
            assert "80" in str(exc) and "already" in str(exc), exc
        else:
            raise AssertionError("80-output input accepted")
        # wrong counts / shapes / keys
        for rows, why in ((95, "95"), (97, "97"), (64, "64")):
            bad = dict(w80)
            bad["object_class_head.weight"] = np.zeros((rows, w80["object_class_head.weight"].shape[1]), np.float32)
            bad["object_class_head.bias"] = np.zeros((rows,), np.float32)
            p = d / f"bad{rows}.npz"
            write_epoch_ckpt(p, bad, dict(META_FIELDS), min_free_bytes=0)
            try:
                mig.migrate_file(p, d / f"bad{rows}.out.npz")
            except mig.MigrationRefused as exc:
                assert why in str(exc), exc
            else:
                raise AssertionError(f"{rows}-output input accepted")
            assert not (d / f"bad{rows}.out.npz").exists()
        # weight 96 but bias 80: mismatch refused
        mixed = dict(w80)
        hid = w80["object_class_head.weight"].shape[1]
        mixed["object_class_head.weight"] = np.zeros((96, hid), np.float32)
        p = d / "mixed.npz"
        write_epoch_ckpt(p, mixed, dict(META_FIELDS), min_free_bytes=0)
        try:
            mig.migrate_file(p, d / "mixed.out.npz")
        except mig.MigrationRefused:
            pass
        else:
            raise AssertionError("weight/bias mismatch accepted")
        # head missing
        nohead = {k: v for k, v in w80.items() if k not in HEAD_KEYS}
        p = d / "nohead.npz"
        write_epoch_ckpt(p, nohead, dict(META_FIELDS), min_free_bytes=0)
        try:
            mig.migrate_file(p, d / "nohead.out.npz")
        except mig.MigrationRefused as exc:
            assert "object_class_head" in str(exc), exc
        else:
            raise AssertionError("ckpt without head accepted")
        # a different tensor with a 96-wide leading dim is not silently passed through as 'class-like'
        odd = {k: v.copy() for k, v in make_old_weights().items()}
        odd["extra_class_head.weight"] = np.zeros((96, 4), np.float32)
        p = d / "odd.npz"
        write_epoch_ckpt(p, odd, dict(META_FIELDS), min_free_bytes=0)
        try:
            mig.migrate_file(p, d / "odd.out.npz")
        except mig.MigrationRefused as exc:
            assert "extra_class_head.weight" in str(exc), exc
        else:
            raise AssertionError("unknown 96-row tensor passed through")
        # not an epoch ckpt
        junk = d / "junk.npz"
        np.savez(junk, a=np.zeros(3))
        try:
            mig.migrate_file(junk, d / "junk.out.npz")
        except (mig.MigrationRefused, ValueError, KeyError):
            pass
        else:
            raise AssertionError("non-checkpoint accepted")
    print("refusals: overwrite, output==input, already-80, 95/97/64 rows, weight/bias mismatch, no head, unknown 96-row tensor, non-ckpt")


def make_old_weights() -> dict:
    from python.train.epoch_ckpt import load_epoch_ckpt

    with tempfile.TemporaryDirectory() as tmp:
        return load_epoch_ckpt(make_old_ckpt(Path(tmp)))["weights"]


def check_cli() -> None:
    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        old = make_old_ckpt(d)
        out = d / "cli.migrated5.npz"
        cmd = [sys.executable, str(ROOT / "scripts/migrate_ckpt_classes.py"), str(old), str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "96 -> 80" in r.stdout and OLD_WORD in r.stdout and out.name in r.stdout, r.stdout
        again = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=120)
        assert again.returncode != 0 and "exists" in (again.stdout + again.stderr), again.stdout + again.stderr
    print("cli: exit 0 with summary; second run refused (exit != 0)")


CHECKS = (
    check_default_still_refuses,
    check_no_load_path_hook,
    check_mapping_covers_current_classes,
    check_unmapped_class_fails_loudly,
    check_migrate_values_and_loads,
    check_refusals,
    check_sibling_does_not_unrefuse,
    check_cli,
)


def main() -> None:
    failed = []
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except BaseException as exc:  # report every check, not just the first failure
            failed.append(fn.__name__)
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {str(exc)[:300]}")
            traceback.print_exc(limit=2)
    if failed:
        print(f"test_ckpt_migrate: {len(failed)} FAILED: {', '.join(failed)}")
        sys.exit(1)
    print("test_ckpt_migrate: all OK")


if __name__ == "__main__":
    main()
