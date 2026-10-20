#!/usr/bin/env python3
"""--mask-files / --mask-phases: the trainer's mask union, and masked runs as GRU gaps.

(1) On a synthetic recording (impacts, settle/stall flags, excursion phases,
    and four mask sidecars), ``row_masks`` + ``mask_report`` give the same
    masked / trainable main / trainable valid counts as the offline union
    script (event mask on every row, phases weave/lane_change/abort/recover,
    sidecar ids; main = rows with ``cams.main``; valid = 5 m and 10 m centre).
    With torch, the same counts come out of the real trainer path
    (``train_scene.main --mask-report``, labels joined from GVD_GT_FILE).
(2) A missing mask file, a line without an integer ``i``, an ``i`` the
    recording does not have, and a file listed twice all fail loudly.
(3) ``--mask-phases`` defaults on with --mask-events, off without it, and
    --no-mask-phases turns it off.
(4) A masked run is one gap: over MEMORY_S it splits ``recording_spans`` and
    ``eval_resets`` clears every stream at that row; a short run keeps the state.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import random
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gvd_test_env import skip_check  # noqa: E402

PHASES = ("weave", "lane_change", "abort", "recover")
FILES = ("gt_x.jumpmask.jsonl", "gt_x.postreset_mask_v3.jsonl", "gt_x.jumpmask_v2b.jsonl", "gt_x.narrowpair_mask.jsonl")


def _write_jsonl(path: Path, items: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")


def _synthetic(tmp: Path, n: int = 3000, seed: int = 7) -> dict:
    """A recording folder; returns the rows and the ids each sidecar lists."""
    rng = random.Random(seed)
    rows, labels = [], []
    phase_run = None
    for i in range(n):
        seg = f"seg{i // 1000}"
        t = 1000.0 + 0.1 * i + (50.0 if i >= 1000 else 0.0) + (80.0 if i >= 2000 else 0.0)
        cams = {"main": f"raw/main/{i:06d}.png"} if rng.random() < 0.65 else {"pillarL": f"raw/pillarL/{i:06d}.png"}
        if phase_run is None and rng.random() < 0.004:
            phase_run = [rng.choice(PHASES + ("cruise",)), rng.randint(5, 60), rng.choice(("lane_edge", "no_path", ""))]
        exc = None
        if phase_run is not None:
            exc = {"phase": phase_run[0], "reason": phase_run[2]}
            phase_run[1] -= 1
            if phase_run[1] <= 0:
                phase_run = None
        row = {"i": i, "t": t, "segment": seg, "cams": cams, "excursion": exc}
        if i % 1000 < 30:
            row["settle"] = True
        if rng.random() < 0.01:
            row["stall"] = True
        rows.append(row)
        valid = [1.0 if rng.random() < 0.7 else 0.0 for _ in range(4)] + [1.0]
        labels.append({"i": i, "tech": {"lane_gt": [0.0] * 8, "lane_gt_valid": valid}})
    with open(tmp / "state.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    _write_jsonl(tmp / "gt_x.jsonl", labels)
    impacts = []
    for k in (400, 1500, 2600):
        t = rows[k]["t"]
        impacts.append({"kind": "impact", "t": t, "pre_t0": t - 1.0, "segment": rows[k]["segment"], "i": k})
        impacts.append({"kind": "impact_end", "t": t + 2.0, "segment": rows[k]["segment"]})
    _write_jsonl(tmp / "impacts.jsonl", impacts)
    listed = {}
    for name in FILES:
        ids = sorted(rng.sample(range(n), 120))
        listed[name] = set(ids)
        _write_jsonl(tmp / name, [{"i": i, "t": rows[i]["t"], "mask": name} for i in ids])
    return {"rows": rows, "labels": labels, "impacts": impacts, "listed": listed}


def _reference(data: dict) -> dict:
    """The offline union script's arithmetic, written out independently."""
    from python.train.event_mask import event_mask

    rows, labels = data["rows"], data["labels"]
    emask, _ = event_mask(rows, data["impacts"])
    main = [k for k, r in enumerate(rows) if (r.get("cams") or {}).get("main")]
    files = set().union(*data["listed"].values())
    masked = set()
    for k in main:
        phase = (rows[k].get("excursion") or {}).get("phase")
        if emask[k] or phase in PHASES or rows[k]["i"] in files:
            masked.add(k)

    def v510(k: int) -> bool:
        m = labels[k]["tech"]["lane_gt_valid"]
        return m[0] > 0.5 and m[3] > 0.5

    return {"main": len(main), "masked_main": len(masked), "trainable_main": len(main) - len(masked),
            "trainable_valid": sum(1 for k in main if k not in masked and v510(k))}


def check_union_counts() -> None:
    from python.data import raw_frames
    from python.train.event_mask import mask_report, row_masks

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        (tmp / "raw" / "main").mkdir(parents=True)
        data = _synthetic(tmp)
        ref = _reference(data)
        os.environ["GVD_GT_FILE"] = "gt_x.jsonl"
        try:
            rows = raw_frames.load_scene_rows(tmp)
        finally:
            os.environ.pop("GVD_GT_FILE", None)
        sources, union, _counts = row_masks(tmp, rows, events=True, phases=True, files=",".join(FILES))
        names = [name for name, _ids in sources]
        assert names == ["events", "phases(" + ",".join(PHASES) + ")"] + [f"file {f}" for f in FILES], names
        for name in FILES:
            assert dict(sources)[f"file {name}"] == data["listed"][name]
        lines, totals = mask_report(rows, sources, union)
        assert len(lines) == len(sources) + 1, lines
        got = {k: totals[k] for k in ("main", "masked_main", "trainable_main", "trainable_valid")}
        assert got == ref, (got, ref)
        assert 0 < ref["masked_main"] < ref["main"] and ref["trainable_valid"] > 0, ref
        # Phases alone (no --mask-events) mask exactly the phase rows.
        p_sources, p_union, _ = row_masks(tmp, rows, phases=True)
        assert p_union == {r["i"] for r in rows if (r.get("excursion") or {}).get("phase") in PHASES}
        cli = _cli_counts(tmp)
    print(f"(1) union: masked {ref['masked_main']} of {ref['main']} main, trainable main {ref['trainable_main']}, "
          f"trainable valid {ref['trainable_valid']} = offline union script; trainer --mask-report {cli} OK")


def _cli_counts(tmp: Path) -> str:
    try:
        import torch  # noqa: F401
    except ImportError:
        skip_check("mask-report CLI", "torch not installed (train_scene.main needs it)")
        return "skipped (no torch)"
    from python.data import data_root
    from python.train import train_scene

    ref = _reference_from_disk(tmp)
    real_free = data_root.free_bytes
    data_root.free_bytes = lambda _path: 500 * 1024 ** 3
    argv = sys.argv
    out = io.StringIO()
    os.environ["GVD_GT_FILE"] = "gt_x.jsonl"
    try:
        sys.argv = ["train_scene", "--strips", str(tmp), "--no-panel", "--cams", "main", "--mask-events",
                    "--mask-files", ",".join(FILES), "--mask-report"]
        with contextlib.redirect_stdout(out):
            train_scene.main()
    finally:
        sys.argv = argv
        data_root.free_bytes = real_free
        os.environ.pop("GVD_GT_FILE", None)
    text = out.getvalue()
    per_source = [line for line in text.splitlines() if line.startswith("[GVD] mask source ")]
    assert len(per_source) == 2 + len(FILES), text
    m = re.search(r"\[GVD\] mask union: \d+ of \d+ rows; (\d+) of (\d+) main frames \([\d.]+%\); "
                  r"trainable main (\d+), trainable valid \(5 m \+ 10 m centre\) (\d+)", text)
    assert m, text
    got = dict(zip(("masked_main", "main", "trainable_main", "trainable_valid"), map(int, m.groups())))
    assert got == {k: ref[k] for k in got}, (got, ref)
    assert "scene index: measuring frames" not in text, "--mask-report must exit before any frame is read"
    return "prints the same counts"


def _reference_from_disk(tmp: Path) -> dict:
    def rd(name: str) -> list[dict]:
        with open(tmp / name, "r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    listed = {name: {item["i"] for item in rd(name)} for name in FILES}
    return _reference({"rows": rd("state.jsonl"), "labels": rd("gt_x.jsonl"), "impacts": rd("impacts.jsonl"),
                       "listed": listed})


def check_loud_failures() -> None:
    from python.train.event_mask import MaskFileError, row_masks

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        rows = [{"i": i, "t": 0.1 * i, "segment": "s"} for i in range(50)]
        _write_jsonl(tmp / "ok.jsonl", [{"i": 3}, {"i": 4}])
        _write_jsonl(tmp / "unknown.jsonl", [{"i": 3}, {"i": 50}, {"i": 99}])
        (tmp / "bad.jsonl").write_text('{"i": 3}\n{"t": 1.0}\n', encoding="utf-8")
        (tmp / "float.jsonl").write_text('{"i": 3.0}\n', encoding="utf-8")
        (tmp / "broken.jsonl").write_text('{"i": 3}\nnot json\n', encoding="utf-8")
        cases = {
            "missing.jsonl": "not found",
            "unknown.jsonl": "2 row id(s) not in the recording (50,99)",
            "bad.jsonl": "line 2",
            "float.jsonl": "line 1",
            "broken.jsonl": "line 2",
            "ok.jsonl,ok.jsonl": "twice",
        }
        for files, needle in cases.items():
            try:
                row_masks(tmp, rows, files=files)
            except MaskFileError as exc:
                assert needle in str(exc), (files, str(exc))
            else:
                raise AssertionError(f"{files} did not fail")
        _sources, union, _ = row_masks(tmp, rows, files="ok.jsonl")
        assert union == {3, 4}
    print("(2) missing file, non-integer i, unparseable line, unknown i and a file listed twice raise MaskFileError OK")


def check_phase_default() -> None:
    from python.train.train_scene import build_parser

    parser = build_parser()
    assert parser.parse_args(["--mask-events"]).mask_phases is None
    assert parser.parse_args(["--mask-events", "--no-mask-phases"]).mask_phases is False
    assert parser.parse_args(["--mask-phases"]).mask_phases is True
    assert parser.parse_args([]).mask_files == ""
    print("(3) --mask-phases: unset (on with --mask-events), --no-mask-phases off, --mask-phases on OK")


def check_masked_gap() -> None:
    from python.train.continual import recording_spans
    from python.train.event_mask import drop_masked, row_masks
    from python.train.scene_net import MEMORY_S

    try:
        from python.train.train_scene import eval_resets
    except ImportError as exc:  # pragma: no cover - torch-free hosts
        eval_resets = None
        skip_check("masked gap eval_resets", f"train_scene import failed ({exc})")
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        rows = [{"i": i, "t": 1000.0 + 0.1 * i, "segment": "s", "dt_s": 0.1 if i else None} for i in range(260)]
        # 20 s masked run (rows 20-219) from a sidecar, and a 2 s run (rows 230-249) from another.
        _write_jsonl(tmp / "long.jsonl", [{"i": i} for i in range(20, 220)])
        _write_jsonl(tmp / "short.jsonl", [{"i": i} for i in range(230, 250)])
        _s, union, _ = row_masks(tmp, rows, files="long.jsonl,short.jsonl")
        kept = drop_masked(rows, union)
        by_i = {r["i"]: r for r in kept}
        assert abs(by_i[220]["dt_s"] - 20.1) < 1e-6 and by_i[220]["dt_s"] > MEMORY_S, by_i[220]["dt_s"]
        assert abs(by_i[250]["dt_s"] - 2.1) < 1e-6, by_i[250]["dt_s"]
        spans = recording_spans(kept)
        k220 = [r["i"] for r in kept].index(220)
        assert spans == [(0, k220), (k220, len(kept))], spans
        stale = [r for r in rows if r["i"] not in union]
        assert len(recording_spans(stale)) == 1  # without the dt_s rewrite the GRU would carry over
        if eval_resets is not None:
            hits = [k for k in range(1, len(kept))
                    if eval_resets(k, kept[k], kept[k - 1], reset_every=0, width=2) == (0, 1)]
            assert hits == [k220], hits
    print(f"(4) masked 20 s run -> dt_s 20.1 s > MEMORY_S={MEMORY_S:.0f} s: spans {spans}, eval resets both streams "
          "at that row; 2 s masked run -> dt_s 2.1 s, no reset OK")


def main() -> None:
    check_union_counts()
    check_loud_failures()
    check_phase_default()
    check_masked_gap()
    print("test_mask_files: OK")


if __name__ == "__main__":
    main()
