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


def _same_bits(a, b) -> bool:
    """Same dtype, same shape, same bytes (np.array_equal would pass 0.0 == -0.0 and widen dtypes)."""
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def make_old_ckpt(dirpath: Path, name: str = "strips.run20261107T011144Z.ckpt_ep11_step77.npz") -> Path:
    """A real tiny SceneNet state_dict whose object_class_head is 96 rows (distinct, known values)."""
    from python.train.epoch_ckpt import build_net, write_epoch_ckpt

    net = build_net(META_FIELDS)  # the same constructor the loader uses for this meta
    weights = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
    hidden = weights["object_class_head.weight"].shape[1]
    weights["object_class_head.weight"] = (np.arange(96)[:, None] * 1000 + np.arange(hidden)[None, :]).astype(np.float32)
    weights["object_class_head.bias"] = (np.arange(96) + 0.5).astype(np.float32)
    weights["object_class_head.weight"][0, 0] = np.float32(-0.0)  # a value np.array_equal cannot tell from +0.0
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
                assert _same_bits(a["weights"][key], b["weights"][key]), key
        n_new, n_old = len(OBJECT_CLASSES), len(OLD_NAMES)
        for key in HEAD_KEYS:
            ow, nw = a["weights"][key], b["weights"][key]
            assert nw.shape[0] == 80 and ow.shape[0] == 96 and nw.shape[1:] == ow.shape[1:], key
            assert nw.dtype == ow.dtype, key
            for slot in range(N_OBJECTS):
                for j, name in enumerate(OBJECT_CLASSES):
                    i = OLD_NAMES.index(name)
                    assert _same_bits(nw[slot * n_new + j], ow[slot * n_old + i]), (key, slot, name)
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
        assert _same_bits(got, w)
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
        # an existing output is always refused; input never overwritten
        try:
            mig.migrate_file(old, out)
        except mig.MigrationRefused as exc:
            assert "exists" in str(exc), exc
        else:
            raise AssertionError("overwrite not refused")
        assert _sha(out) == first
        for same in (old, d / "." / old.name):
            try:
                mig.migrate_file(old, same)
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
        cmd = [sys.executable, str(ROOT / "scripts/migrate_ckpt_classes.py"), str(old), str(out), "--min-free-gb", "0"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "96 -> 80" in r.stdout and OLD_WORD in r.stdout and out.name in r.stdout, r.stdout
        again = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=120)
        assert again.returncode == 2 and "REFUSED" in again.stderr and "exists" in again.stderr, (again.returncode, again.stdout, again.stderr)
    print("cli: exit 0 with summary; second run REFUSED exit 2")


# ------------------------------------------------------------------ rework (#266): never replace, refuse bad input


def _cli(*args: str):
    import subprocess

    cmd = [sys.executable, str(ROOT / "scripts/migrate_ckpt_classes.py"), *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=120)


def _plain_ckpt(path: Path) -> bytes:
    """A valid 80-output (non-migrated) epoch ckpt: stands in for a real model file."""
    from python.train.epoch_ckpt import build_net, write_epoch_ckpt

    net = build_net(META_FIELDS)
    w = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
    write_epoch_ckpt(path, w, dict(META_FIELDS), min_free_bytes=0)
    return path.read_bytes()


def check_existing_output_never_replaced() -> None:
    import inspect

    import migrate_ckpt_classes as mig

    assert "force" not in inspect.signature(mig.migrate_file).parameters, "migrate_file still has a force switch"
    assert "force" not in mig.build_parser().format_help().lower(), "--force still in --help"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        old = make_old_ckpt(d)
        src_before = _sha(old)
        migrated = d / "was_migrated.npz"
        mig.migrate_file(old, migrated)
        plain = d / "real_model.npz"
        _plain_ckpt(plain)
        garbage = d / "not_a_ckpt.npz"
        garbage.write_bytes(b"precious bytes, not an npz")
        for dst in (plain, migrated, garbage):
            before = dst.read_bytes()
            # API: refused, bytes identical
            try:
                mig.migrate_file(old, dst)
            except mig.MigrationRefused as exc:
                assert "exists" in str(exc), exc
            else:
                raise AssertionError(f"{dst.name}: existing output not refused")
            assert dst.read_bytes() == before, f"{dst.name} changed"
            # CLI: exit 2, REFUSED, bytes identical
            r = _cli(str(old), str(dst), "--min-free-gb", "0")
            assert r.returncode == 2 and "REFUSED" in r.stderr and "exists" in r.stderr, (dst.name, r.returncode, r.stdout, r.stderr)
            assert dst.read_bytes() == before and hashlib.sha256(before).hexdigest() == _sha(dst), f"{dst.name} changed via CLI"
            # --force no longer exists: argparse error, exit 2, nothing replaced
            r = _cli(str(old), str(dst), "--force", "--min-free-gb", "0")
            assert r.returncode == 2 and "--force" in r.stderr and "unrecognized" in r.stderr, (dst.name, r.returncode, r.stderr)
            assert dst.read_bytes() == before, f"{dst.name} changed by --force"
        assert _sha(old) == src_before
        assert not list(d.glob("*.partial*"))
    print("existing output (non-migrated ckpt, migrated ckpt, garbage) refused exit 2, bytes identical; --force is an argparse error exit 2")


def check_dangling_symlink_output_refused() -> None:
    import migrate_ckpt_classes as mig

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        old = make_old_ckpt(d)
        link = d / "out.npz"
        try:
            os.symlink(d / "nowhere.npz", link)
        except (OSError, NotImplementedError):
            print("symlink not permitted here; skipped")
            return
        try:
            mig.migrate_file(old, link)
        except mig.MigrationRefused:
            pass
        else:
            raise AssertionError("dangling symlink output accepted")
        assert not (d / "nowhere.npz").exists()
    print("dangling-symlink output refused, target not created")


def check_two_old_classes_one_new_class_refused() -> None:
    import migrate_ckpt_classes as mig
    from python.train.scene_net import OBJECT_CLASSES

    policy = dict(mig.CLASS_POLICY)
    policy["bus"] = "truck"  # truck <- truck and bus
    try:
        mig.build_mapping(mig.OLD_CLASSES, OBJECT_CLASSES, policy)
    except mig.MappingError as exc:
        text = str(exc)
        assert "truck" in text and "bus" in text and "both map" in text and "merge" in text, text
    else:
        raise AssertionError("two old classes mapping to one new class was accepted")
    # same through the file API: nothing written
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        old = make_old_ckpt(d)
        saved = dict(mig.CLASS_POLICY)
        mig.CLASS_POLICY["bus"] = "truck"
        try:
            try:
                mig.migrate_file(old, d / "never.npz")
            except mig.MappingError:
                pass
            else:
                raise AssertionError("migrate_file accepted a merge")
        finally:
            mig.CLASS_POLICY.clear()
            mig.CLASS_POLICY.update(saved)
        assert not (d / "never.npz").exists() and not list(d.glob("*.partial*"))
    print("two old classes -> one new class: MappingError naming both, no output")


def check_output_appearing_after_precheck_is_refused() -> None:
    import migrate_ckpt_classes as mig
    from python.train import epoch_ckpt

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        old = make_old_ckpt(d)
        dst = d / "race.npz"
        precious = b"someone else's file, created after the exists() check"
        real_serialise = epoch_ckpt.serialise

        def serialise_then_race(weights, meta):
            blob = real_serialise(weights, meta)
            dst.write_bytes(precious)  # appears after every pre-check, before the create step
            return blob

        epoch_ckpt.serialise = serialise_then_race
        try:
            try:
                mig.migrate_file(old, dst)
            except mig.MigrationRefused as exc:
                text = str(exc)
                assert "already exists" in text, exc
                assert "write failed" not in text, exc
            else:
                raise AssertionError("an output that appeared after the pre-check was replaced")
        finally:
            epoch_ckpt.serialise = real_serialise
        assert dst.read_bytes() == precious, "racing file was overwritten"
        assert not list(d.glob("*.partial*")), "temp file left behind"
    print("output created after the pre-check: refused with 'already exists' (not 'write failed'), file intact, no temp left")


def check_unreadable_input_is_refused_exit_2() -> None:
    import migrate_ckpt_classes as mig

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        good = make_old_ckpt(d)
        blob = good.read_bytes()
        garbage = d / "garbage.npz"
        garbage.write_bytes(b"this is not a zip file at all" * 10)
        truncated = d / "truncated.npz"
        truncated.write_bytes(blob[: len(blob) // 2])
        empty = d / "empty.npz"
        empty.write_bytes(b"")
        missing = d / "missing.npz"
        folder = d / "folder.npz"
        folder.mkdir()
        for bad in (garbage, truncated, empty, missing, folder):
            out = d / (bad.stem + ".out.npz")
            try:
                mig.migrate_file(bad, out)
            except mig.MigrationRefused:
                pass
            else:
                raise AssertionError(f"{bad.name}: accepted")
            assert not out.exists() and not list(d.glob("*.partial*")), bad.name
            r = _cli(str(bad), str(out), "--min-free-gb", "0")
            assert r.returncode == 2 and "REFUSED" in r.stderr and "Traceback" not in r.stderr, (bad.name, r.returncode, r.stderr)
            assert not out.exists(), bad.name
        # valid JSON meta that is not an object, or a bad weight_names value: refusal, not a traceback
        from python.train.epoch_ckpt import FORMAT

        def _write_meta(path: Path, meta_obj) -> None:
            _write_meta_arrays(path, meta_obj, w0=np.zeros(3, np.float32))

        def _write_meta_arrays(path: Path, meta_obj, **arrays) -> None:
            import json

            np.savez(path, meta=np.array(json.dumps(meta_obj)), **arrays)

        malformed = {
            "meta_list.npz": ["not", "an", "object"],
            "meta_null.npz": None,
            "meta_string.npz": "not an object",
            "names_noniter.npz": {"format": FORMAT, "weight_names": 123},
            "names_nonstr.npz": {"format": FORMAT, "weight_names": [1, 2, 3]},
        }
        for name, meta_obj in malformed.items():
            bad = d / name
            _write_meta(bad, meta_obj)
            out = d / (bad.stem + ".out.npz")
            try:
                mig.migrate_file(bad, out)
            except mig.MigrationRefused as exc:
                if name.startswith("names_"):
                    assert "weight_names" in str(exc), (name, exc)
            else:
                raise AssertionError(f"{name}: accepted")
            assert not out.exists() and not list(d.glob("*.partial*")), name
            r = _cli(str(bad), str(out), "--min-free-gb", "0")
            assert r.returncode == 2 and "REFUSED" in r.stderr and "Traceback" not in r.stderr, (name, r.returncode, r.stderr)
            if name.startswith("names_"):
                assert "weight_names" in r.stderr, (name, r.stderr)
            assert not out.exists(), name
        # JSON has no tuples, so a tuple-of-str weight_names cannot be written into an npz meta.
        # Pin the validation at load_epoch_ckpt: after json.loads, swap the list for a tuple.
        from python.train import epoch_ckpt
        from python.train.epoch_ckpt import load_epoch_ckpt

        # include w0 and w1 so a dropped check would load successfully rather than die later on a missing array
        tuple_path = d / "names_tuple.npz"
        _write_meta_arrays(tuple_path, {"format": FORMAT, "weight_names": ["a", "b"]}, w0=np.zeros(3, np.float32), w1=np.zeros(3, np.float32))
        real_loads = epoch_ckpt.json.loads

        def loads_as_tuple(text, *a, **k):
            meta = real_loads(text, *a, **k)
            if isinstance(meta, dict) and isinstance(meta.get("weight_names"), list):
                meta["weight_names"] = tuple(meta["weight_names"])  # ('a', 'b')
            return meta

        epoch_ckpt.json.loads = loads_as_tuple
        try:
            try:
                load_epoch_ckpt(tuple_path)
            except ValueError as exc:
                assert "weight_names" in str(exc), exc
            else:
                raise AssertionError("tuple-of-str weight_names was accepted by load_epoch_ckpt")
        finally:
            epoch_ckpt.json.loads = real_loads
    print("garbage / truncated / empty / missing / directory / list-null-string meta / bad weight_names (message names weight_names) / tuple-of-str at load_epoch_ckpt: REFUSED exit 2, no traceback, no output")


def check_low_disk_space_is_refused() -> None:
    import shutil

    import migrate_ckpt_classes as mig
    from python.train.epoch_ckpt import DEFAULT_MIN_FREE_GB, GB

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        old = make_old_ckpt(d)
        dst = d / "low.npz"
        real = shutil.disk_usage
        shutil.disk_usage = lambda p: shutil._ntuple_diskusage(10 * GB, 10 * GB - int(DEFAULT_MIN_FREE_GB * GB) + 1000, int(DEFAULT_MIN_FREE_GB * GB) - 1000)
        try:
            try:
                mig.migrate_file(old, dst)
            except mig.MigrationRefused as exc:
                assert "low space" in str(exc), exc
            else:
                raise AssertionError("low-space write accepted")
            assert not dst.exists() and not list(d.glob("*.partial*"))
            # the floor is the trainer's: lowering it explicitly lets the same write through
            mig.migrate_file(old, dst, min_free_bytes=0)
            assert dst.exists()
        finally:
            shutil.disk_usage = real
    print("low free space: refused with the epoch-ckpt message, nothing written; floor 0 lets it through")


def check_link_oserror_is_refused() -> None:
    import errno

    import migrate_ckpt_classes as mig

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        old = make_old_ckpt(d)
        dst = d / "perm.npz"
        real_link = os.link

        def boom(src, dst_, *a, **k):
            raise OSError(errno.EPERM, "Operation not permitted")

        os.link = boom
        try:
            try:
                mig.migrate_file(old, dst)
            except mig.MigrationRefused as exc:
                assert "perm.npz" in str(exc), exc
            else:
                raise AssertionError("OSError from os.link escaped as a non-refusal")
        finally:
            os.link = real_link
        assert not dst.exists() and not list(d.glob("*.partial*")), "temp or output left behind"
    print("os.link OSError (not EEXIST): MigrationRefused, temp file cleaned")


def check_input_read_once_hash_matches_loaded_bytes() -> None:
    import migrate_ckpt_classes as mig
    from python.train.epoch_ckpt import load_epoch_ckpt

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        old = make_old_ckpt(d)
        original = old.read_bytes()
        other = Path(tmp) / "other"
        other.mkdir()
        alt = make_old_ckpt(other)
        w = load_epoch_ckpt(alt)
        # a different checkpoint (distinct head values) that will replace the input mid-run
        from python.train.epoch_ckpt import write_epoch_ckpt

        w["weights"]["object_class_head.bias"] = w["weights"]["object_class_head.bias"] + 1000
        alt.unlink()
        write_epoch_ckpt(alt, w["weights"], dict(w["meta"]), min_free_bytes=0)
        replacement = alt.read_bytes()
        assert replacement != original
        real_load = np.load
        state = {"swapped": False}

        def swap_then_load(*a, **k):
            if not state["swapped"]:
                state["swapped"] = True
                old.write_bytes(replacement)  # the file changes under the tool after it was first read
            return real_load(*a, **k)

        np.load = swap_then_load
        out = d / "out.npz"
        try:
            mig.migrate_file(old, out)
        finally:
            np.load = real_load
        got = load_epoch_ckpt(out)
        assert got["meta"]["migration"]["source_sha256"] == hashlib.sha256(original).hexdigest(), "source_sha256 is not the hash of the bytes loaded"
        ref = load_epoch_ckpt_from(original, d / "ref.npz")
        assert _same_bits(got["weights"]["object_class_head.bias"][:5], ref["object_class_head.bias"][[0, 1, 2, 3, 5]]), "migrated weights came from different bytes than the hash"
    print("input read once: source_sha256 is the hash of exactly the bytes that were migrated, even if the file changes mid-run")


def load_epoch_ckpt_from(blob: bytes, path: Path) -> dict:
    from python.train.epoch_ckpt import load_epoch_ckpt

    path.write_bytes(blob)
    return load_epoch_ckpt(path)["weights"]


def check_min_free_gb_cli() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        old = make_old_ckpt(d)
        for bad in ("nan", "inf", "-1"):
            out = d / f"bad_{bad}.npz"
            r = _cli(str(old), str(out), "--min-free-gb", bad)
            assert r.returncode == 2 and "Traceback" not in r.stderr + r.stdout, (bad, r.returncode, r.stderr, r.stdout)
            assert "min-free-gb" in r.stderr or "REFUSED" in r.stderr, (bad, r.stderr)
            assert not out.exists() and not list(d.glob("*.partial*")), bad
        out = d / "zero.npz"
        r = _cli(str(old), str(out), "--min-free-gb", "0")
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert out.exists() and "96 -> 80" in r.stdout
    print("--min-free-gb nan/inf/-1: exit 2, no traceback, no output; 0 is allowed and writes")


def check_disk_floor_boundary() -> None:
    import shutil

    import migrate_ckpt_classes as mig
    from python.train.epoch_ckpt import DEFAULT_MIN_FREE_GB, GB

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        old = make_old_ckpt(d)
        measured = d / "measured.npz"
        mig.migrate_file(old, measured, min_free_bytes=0)
        size = measured.stat().st_size
        floor = int(DEFAULT_MIN_FREE_GB * GB)
        real = shutil.disk_usage
        try:
            # free - size == floor: exactly at the floor, must write
            shutil.disk_usage = lambda p: shutil._ntuple_diskusage(floor + size, 0, floor + size)
            exact = d / "exact.npz"
            mig.migrate_file(old, exact)
            assert exact.exists() and exact.stat().st_size == size
            # one byte short of that: refused, nothing written
            shutil.disk_usage = lambda p: shutil._ntuple_diskusage(floor + size - 1, 0, floor + size - 1)
            short = d / "short.npz"
            try:
                mig.migrate_file(old, short)
            except mig.MigrationRefused as exc:
                assert "low space" in str(exc), exc
            else:
                raise AssertionError("one byte under the floor was accepted")
            assert not short.exists() and not list(d.glob("*.partial*"))
        finally:
            shutil.disk_usage = real
    print("disk floor boundary: free-size == floor writes; one byte less is refused")


CHECKS = (
    check_default_still_refuses,
    check_no_load_path_hook,
    check_mapping_covers_current_classes,
    check_unmapped_class_fails_loudly,
    check_migrate_values_and_loads,
    check_refusals,
    check_sibling_does_not_unrefuse,
    check_cli,
    check_existing_output_never_replaced,
    check_dangling_symlink_output_refused,
    check_two_old_classes_one_new_class_refused,
    check_output_appearing_after_precheck_is_refused,
    check_unreadable_input_is_refused_exit_2,
    check_low_disk_space_is_refused,
    check_disk_floor_boundary,
    check_min_free_gb_cli,
    check_link_oserror_is_refused,
    check_input_read_once_hash_matches_loaded_bytes,
)


def main() -> None:
    import shutil

    failed = []
    real_usage = shutil.disk_usage
    for fn in CHECKS:
        # the tool keeps write_epoch_ckpt's 8 GB free-space floor: pretend a roomy disk so the checks do not depend on the host
        shutil.disk_usage = real_usage if fn is check_low_disk_space_is_refused else (lambda p: shutil._ntuple_diskusage(10**13, 0, 10**13))
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except BaseException as exc:  # report every check, not just the first failure
            failed.append(fn.__name__)
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {str(exc)[:300]}")
            traceback.print_exc(limit=2)
    shutil.disk_usage = real_usage
    if failed:
        print(f"test_ckpt_migrate: {len(failed)} FAILED: {', '.join(failed)}")
        sys.exit(1)
    print("test_ckpt_migrate: all OK")


if __name__ == "__main__":
    main()
