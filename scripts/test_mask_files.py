#!/usr/bin/env python3
"""--mask-files: offline mask sidecars leave rows out of training AND the holdout.

(1) A missing mask file, a line without an integer ``i``, an ``i`` the
    recording does not have, a file listed twice, a malformed ``@sha`` and a
    sha256 mismatch all raise ``MaskFileError``. With torch, the unknown-id and
    sha-mismatch refusals also stop ``train_scene.main`` (exit 2) before any
    frame is measured.
(2) ``--mask-files`` defaults to empty.
(3) A masked run is one gap: over MEMORY_S it splits ``recording_spans``; a
    short run does not.
(4) A masked id is gone from both the train chunks and the holdout of
    ``pass_plan`` (the trainer drops the rows before the holdout split), and
    without the mask the same ids are in both.
(5) The ``[GVD] selection:`` line counts empty tech, each file with its
    sha256, the union, and the rows left.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gvd_test_env import collect_root, skip_check  # noqa: E402


def _write_jsonl(path: Path, items: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_loud_failures() -> None:
    from python.train.event_mask import MaskFileError, mask_file_sources

    with collect_root("gvd-mask-") as tmp:
        rows = [{"i": i, "t": 0.1 * i, "segment": "s"} for i in range(50)]
        _write_jsonl(tmp / "ok.jsonl", [{"i": 3}, {"i": 4}])
        _write_jsonl(tmp / "unknown.jsonl", [{"i": 3}, {"i": 50}, {"i": 99}])
        (tmp / "bad.jsonl").write_text('{"i": 3}\n{"t": 1.0}\n', encoding="utf-8")
        (tmp / "float.jsonl").write_text('{"i": 3.0}\n', encoding="utf-8")
        (tmp / "broken.jsonl").write_text('{"i": 3}\nnot json\n', encoding="utf-8")
        sha = _sha(tmp / "ok.jsonl")
        wrong = ("0" if sha[0] != "0" else "1") + sha[1:12]
        cases = {
            "missing.jsonl": "not found",
            "unknown.jsonl": "2 row id(s) not in the recording (50,99)",
            "bad.jsonl": "line 2",
            "float.jsonl": "line 1",
            "broken.jsonl": "line 2",
            "ok.jsonl,ok.jsonl": "twice",
            f"ok.jsonl,ok.jsonl@{sha[:8]}": "twice",
            f"ok.jsonl@{wrong}": f"does not start with the expected {wrong}",
            "ok.jsonl@abc": "name@sha256prefix",
            "ok.jsonl@zzzzzzzz": "name@sha256prefix",
            f"missing.jsonl@{sha[:8]}": "not found",
        }
        for files, needle in cases.items():
            try:
                mask_file_sources(tmp, rows, files)
            except MaskFileError as exc:
                assert needle in str(exc), (files, str(exc))
            else:
                raise AssertionError(f"{files} did not fail")
        for files in ("ok.jsonl", f"ok.jsonl@{sha[:8]}", f"ok.jsonl@{sha.upper()}"):
            got = mask_file_sources(tmp, rows, files)
            assert [(s["name"], s["sha256"], s["ids"]) for s in got] == [("ok.jsonl", sha, {3, 4})], got
        assert mask_file_sources(tmp, rows, "") == []
        cli = _cli_refusals(tmp)
    print("(1) missing file, non-integer i, unparseable line, unknown i, a file listed twice, a bad @sha and a "
          f"sha256 mismatch raise MaskFileError; trainer CLI {cli} OK")


def _cli_refusals(tmp: Path) -> str:
    try:
        import torch  # noqa: F401
    except ImportError:
        skip_check("mask-files CLI refusals", "torch not installed (train_scene.main needs it)")
        return "skipped (no torch)"
    from python.data import data_root
    from python.train import train_scene

    _write_jsonl(tmp / "state.jsonl", [{"i": i, "t": 1000.0 + 0.1 * i, "segment": "s", "cams": {}} for i in range(50)])
    (tmp / "raw" / "main").mkdir(parents=True, exist_ok=True)
    sha = _sha(tmp / "ok.jsonl")
    wrong = ("0" if sha[0] != "0" else "1") + sha[1:12]
    real_free = getattr(data_root, "free_bytes", None)
    if real_free is not None:
        data_root.free_bytes = lambda _path: 500 * 1024 ** 3
    argv = sys.argv
    try:
        for files, needle in (("unknown.jsonl", "2 row id(s) not in the recording (50,99)"),
                              (f"ok.jsonl@{wrong}", "does not start with the expected")):
            out = io.StringIO()
            sys.argv = ["train_scene", "--strips", str(tmp), "--no-panel", "--cams", "main", "--mask-files", files]
            code = None
            with contextlib.redirect_stdout(out):
                try:
                    train_scene.main()
                except SystemExit as exc:
                    code = exc.code
            text = out.getvalue()
            assert code == 2, (files, code, text)
            assert needle in text, (files, text)
            assert "scene index: measuring frames" not in text, "a bad mask file must stop before frames are measured"
    finally:
        sys.argv = argv
        if real_free is not None:
            data_root.free_bytes = real_free
    return "exits 2 on both before measuring frames"


def check_default() -> None:
    try:
        from python.train.train_scene import build_parser
    except ImportError as exc:
        skip_check("--mask-files default", f"train_scene import failed ({exc})")
        print("(2) --mask-files default: skipped (train_scene import)")
        return
    assert build_parser().parse_args([]).mask_files == ""
    assert build_parser().parse_args(["--mask-files", "a.jsonl,b.jsonl@0123abcd"]).mask_files == "a.jsonl,b.jsonl@0123abcd"
    print("(2) --mask-files defaults to empty OK")


def check_masked_gap() -> None:
    from python.train.continual import recording_spans
    from python.train.event_mask import drop_masked, mask_file_sources
    from python.train.scene_net import MEMORY_S

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        rows = [{"i": i, "t": 1000.0 + 0.1 * i, "segment": "s", "dt_s": 0.1 if i else None} for i in range(260)]
        # 20 s masked run (rows 20-219) from a sidecar, and a 2 s run (rows 230-249) from another.
        _write_jsonl(tmp / "long.jsonl", [{"i": i} for i in range(20, 220)])
        _write_jsonl(tmp / "short.jsonl", [{"i": i} for i in range(230, 250)])
        sources = mask_file_sources(tmp, rows, "long.jsonl,short.jsonl")
        union = set().union(*(s["ids"] for s in sources))
        kept = drop_masked(rows, union)
        by_i = {r["i"]: r for r in kept}
        assert abs(by_i[220]["dt_s"] - 20.1) < 1e-6 and by_i[220]["dt_s"] > MEMORY_S, by_i[220]["dt_s"]
        assert abs(by_i[250]["dt_s"] - 2.1) < 1e-6, by_i[250]["dt_s"]
        spans = recording_spans(kept)
        k220 = [r["i"] for r in kept].index(220)
        assert spans == [(0, k220), (k220, len(kept))], spans
        stale = [r for r in rows if r["i"] not in union]
        assert len(recording_spans(stale)) == 1  # without the dt_s rewrite the GRU would carry over
    print(f"(3) masked 20 s run -> dt_s 20.1 s > MEMORY_S={MEMORY_S:.0f} s: spans {spans}; "
          "2 s masked run -> dt_s 2.1 s, no split OK")


def _plan_ids(rows: list[dict]) -> tuple[set[int], set[int]]:
    from python.train.continual import CHUNK_MAX, pass_plan

    train: set[int] = set()
    hold: set[int] = set()
    for item in pass_plan(rows, CHUNK_MAX, CHUNK_MAX // 2):
        (hold if item["kind"] == "holdout" else train).update(int(r["i"]) for r in item["rows"])
    return train, hold


def check_train_and_holdout() -> None:
    from python.train.event_mask import drop_masked, mask_file_sources

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        # Five short segments: pass_plan holds out whole segments.
        rows = []
        for i in range(1500):
            seg = i // 300
            rows.append({"i": i, "t": 1000.0 + 0.1 * i + 100.0 * seg, "segment": f"seg{seg}",
                         "dt_s": None if i % 300 == 0 else 0.1})
        train0, hold0 = _plan_ids(rows)
        assert train0 and hold0 and not (train0 & hold0), (len(train0), len(hold0))
        masked = sorted(range(5, 1500, 37))
        assert set(masked) & train0 and set(masked) & hold0, "the mask must hit both sides before it is applied"
        _write_jsonl(tmp / "m.jsonl", [{"i": i, "mask": "test"} for i in masked])
        sources = mask_file_sources(tmp, rows, "m.jsonl@" + _sha(tmp / "m.jsonl")[:10])
        kept = drop_masked(rows, sources[0]["ids"])
        train1, hold1 = _plan_ids(kept)
        assert not (set(masked) & train1), sorted(set(masked) & train1)[:5]
        assert not (set(masked) & hold1), sorted(set(masked) & hold1)[:5]
        assert train1 and hold1
        assert (train1 | hold1) <= (train0 | hold0) - set(masked)
        n_train = len(set(masked) & train0)
        n_hold = len(set(masked) & hold0)
    print(f"(4) {len(masked)} masked ids ({n_train} in train, {n_hold} in holdout unmasked) are in neither the "
          "train chunks nor the holdout after --mask-files OK")


def check_selection_line() -> None:
    from python.train.event_mask import drop_masked, mask_file_sources, selection_line

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        state = []
        for i in range(100):
            tech = {} if i in (10, 11, 12, 90) else {"lane_gt": [0.0] * 8}
            state.append({"i": i, "t": 0.1 * i, "tech": tech})
        before = [r for r in state if r["i"] != 90]  # row 90 has no frame
        _write_jsonl(tmp / "a.jsonl", [{"i": i} for i in (10, 11, 12, 90)])
        _write_jsonl(tmp / "b.jsonl", [{"i": i} for i in (12, 40, 41)])
        sources = mask_file_sources(tmp, state, "a.jsonl,b.jsonl")
        after = drop_masked(before, set().union(*(s["ids"] for s in sources)))
        line, meta = selection_line(state, before, after, sources)
        want = (
            "[GVD] selection: empty tech (no loss, no score; kept as GRU context) 4 of 100 state rows, "
            f"3 of 99 rows with frames; file a.jsonl sha256 {_sha(tmp / 'a.jsonl')} 4 rows (3 with frames); "
            f"file b.jsonl sha256 {_sha(tmp / 'b.jsonl')} 3 rows (3 with frames); files union 6 rows; "
            "left out 5 of 99 rows with frames; 94 remain"
        )
        assert line == want, (line, want)
        assert line.count("\n") == 0
        assert meta["files_union"] == 6 and meta["after"] == 94, meta
        bare, _ = selection_line(state, before, before, [])
        assert bare.endswith("; no mask files; left out 0 of 99 rows with frames; 99 remain"), bare
    print("(5) one selection line: empty tech, each file with sha256 and rows, union, rows left OK")


def main() -> None:
    check_loud_failures()
    check_default()
    check_masked_gap()
    check_train_and_holdout()
    check_selection_line()
    print("test_mask_files: OK")


if __name__ == "__main__":
    main()
