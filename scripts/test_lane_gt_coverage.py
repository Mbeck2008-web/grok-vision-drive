#!/usr/bin/env python3
"""Batches without a lane_gt target are counted and logged each epoch.

A mixed epoch only logs the share. An epoch where every batch is missing
lane_gt, with the lane_gt head on, raises ``LaneGtMissingError``. Without
torch, ``_lane_gt_loss`` hands ``zero`` back for a target-less batch.
CPU only; the end-to-end runs need torch. The coverage runs pass
``min_valid=0`` so target-less windows reach the optimizer. With the default
``min_valid`` (0.3) a recording without lane centers loses every window to
the valid filter before batching; that is logged (windows, rows, recordings,
why) and raises ``LaneGtMissingError`` instead of training zero steps.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gvd_test_env import skip_check  # noqa: E402

import numpy as np  # noqa: E402

from python.debug.lane_targets import LANE_GT_DIM  # noqa: E402
from python.train.scene_net import LaneGtCoverage, LaneGtMissingError, has_lane_gt, labels_for_row  # noqa: E402

N_ROWS = 240


def check_counter() -> None:
    log = io.StringIO()
    seen = LaneGtCoverage(enabled=True)
    for has in (True, False, False, True):
        seen.note(has)
    with contextlib.redirect_stdout(log):
        summary = seen.close_epoch(3)
    assert summary == {"epoch": 3, "batches": 4, "missing": 2, "share": 0.5}, summary
    assert "lane_gt missing epoch=3 batches=2/4 share=50.0%" in log.getvalue(), log.getvalue()
    # Reset after closing: the next epoch counts from zero.
    assert seen.batches == 0 and seen.missing == 0

    for _ in range(3):
        seen.note(False)
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            seen.close_epoch(4)
    except LaneGtMissingError as exc:
        assert "all 3 training batches" in str(exc) and "--heads" in str(exc), exc
    else:
        raise AssertionError("an epoch with no lane_gt target at all did not raise")
    assert "batches=3/3 share=100.0%" in log.getvalue(), log.getvalue()

    off = LaneGtCoverage(enabled=False)
    off.note(False)
    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert off.close_epoch(0)["share"] == 1.0
    assert "(lane_gt head off)" in out.getvalue()
    # Nothing trained this epoch (paused, empty plan): no log, no raise.
    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert LaneGtCoverage(enabled=True).close_epoch(0)["share"] is None
    assert out.getvalue() == ""

    assert has_lane_gt(labels_for_row({"tech": _lane_tech()}))
    assert not has_lane_gt(labels_for_row({"tech": {}}))
    assert not has_lane_gt(None)
    print("counter: mixed epoch logs 2/4, all-missing raises, head off and empty epochs only log OK")


def _lane_tech() -> dict:
    values = [0.2, -1.6, 2.0] * 4 + [0.0, 1.0]
    return {"lane_gt": values, "lane_gt_valid": [1.0] * LANE_GT_DIM}


def _fixture(root: Path, with_lane) -> None:
    import cv2

    cam = root / "raw" / "main"
    cam.mkdir(parents=True)
    rng = np.random.default_rng(0)
    state_lines = []
    gt_lines = []
    for index in range(N_ROWS):
        frame = (rng.random((48, 64, 3)) * 200 + 30).astype(np.uint8)
        ok, buf = cv2.imencode(".png", frame)
        assert ok
        (cam / f"{index:06d}.png").write_bytes(buf.tobytes())
        state_lines.append(
            {
                "i": index,
                "t": float(index) * 0.1,
                "dt_s": None if index == 0 else 0.1,
                "segment": "west_coast_usa#0",
                "cams": {"main": {"file": f"raw/main/{index:06d}.png"}},
                "wheel": {"steer": 0.1},
                "pedals": {"throttle": 0.2, "brake": 0.0},
            }
        )
        # A row without lane_gt still has a tech dict, so loss_mask keeps it.
        tech = _lane_tech() if with_lane(index) else {"speed_mps": 10.0}
        gt_lines.append({"i": index, "t": float(index) * 0.1, "tech": tech})
    (root / "state.jsonl").write_text("\n".join(json.dumps(r) for r in state_lines) + "\n", encoding="utf-8")
    (root / "gt.jsonl").write_text("\n".join(json.dumps(r) for r in gt_lines) + "\n", encoding="utf-8")


_DEFAULT = object()


def _train(root: Path, heads, min_valid=0.0):
    from python.train.train_scene import train_directory

    # The default 0.3 drops windows without lane centers before they are
    # batches; 0 lets target-less windows reach the optimizer. ``_DEFAULT``
    # leaves the argument out so train_directory's own default applies.
    extra = {} if min_valid is _DEFAULT else {"min_valid": float(min_valid)}
    return train_directory(
        root,
        epochs=1,
        lr=1e-3,
        capability=None,
        export_path=None,
        free_vram_bytes=0,
        free_ram_bytes=8_000_000_000,
        resume=False,
        open_panel=False,
        autostart=True,
        cams=("main",),
        heads=heads,
        augment=False,
        workers=0,
        status_root=root / "_status",
        **extra,
    )


def check_end_to_end() -> None:
    from python.train.scene_net import torch_ready

    if not torch_ready():
        skip_check("end_to_end", "torch missing")
        return
    import torch

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        # Half the rows, in blocks of 16, carry lane_gt: a mixed epoch only logs.
        mixed = Path(tmp) / "mixed"
        _fixture(mixed, lambda i: (i // 16) % 2 == 0)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            _train(mixed, ("lane_gt",))
        lines = [line for line in log.getvalue().splitlines() if "lane_gt missing epoch=" in line]
        assert lines, log.getvalue()[-2000:]
        assert "share=100.0%" not in lines[0], lines
        print(f"mixed: {lines[0].replace('[GVD] ', '')} and training finished OK")

        # No row has lane_gt and the head is on: the first epoch raises.
        bare = Path(tmp) / "bare"
        _fixture(bare, lambda i: False)
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                _train(bare, ("lane_gt",))
        except LaneGtMissingError as exc:
            assert "share=100.0%" in log.getvalue(), log.getvalue()[-2000:]
            print(f"missing: {str(exc)[:90]}... OK")
        else:
            raise AssertionError("a lane_gt run with no lane_gt target in any batch finished")

        # Default min_valid, a recording with no lane centers: the valid
        # filter removes the whole recording before batching. That used to
        # train zero steps silently; it must log the removal and raise.
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                _train(bare, ("lane_gt",), min_valid=_DEFAULT)
        except LaneGtMissingError as exc:
            text = log.getvalue()
            lines = [line for line in text.splitlines() if "min-valid filter" in line]
            assert lines, text[-2000:]
            assert "recordings=1/1" in lines[0] and "why: lane-center valid share < 0.3" in lines[0], lines
            assert "1 recording(s) (bare)" in str(exc) and "--min-valid 0" in str(exc), exc
            # Raised before any optimizer step: no epoch summary was logged.
            assert "lane_gt missing epoch=" not in text, text[-2000:]
            print(f"default min_valid: {lines[0].replace('[GVD] ', '')} -> raised OK")
        else:
            raise AssertionError("default min_valid on a recording without lane centers trained silently")


def check_zero_without_target() -> None:
    from python.train.scene_net import ScenePrediction, _lane_gt_loss

    fields = (
        "lanes lane_valid curbs curb_valid objects object_class object_valid "
        "signs sign_class sign_state sign_valid lane_gt"
    ).split()
    pred = ScenePrediction(**{name: None for name in fields})
    token = object()
    assert _lane_gt_loss(None, pred, {}, None, None, token, conf_weight=0.1) is token
    print("zero: a target-less batch returns zero unweighted OK")


def _row(index: int, segment: str, lane: bool) -> dict:
    tech = _lane_tech() if lane else {"speed_mps": 10.0}
    return {"i": index, "t": index * 0.1, "segment": segment, "tech": tech}


def check_valid_filter() -> None:
    """``epoch_windows`` counts what --min-valid removes; a lost recording raises."""
    import random

    from python.train.continual import epoch_windows, report_valid_filter

    good = [_row(i, "a#0", True) for i in range(16)]
    half = [_row(i, "a#0", i % 4 == 0) for i in range(16, 32)]  # valid share 0.25
    bare = [_row(i, "b#0", False) for i in range(32, 64)]
    plan = [
        {"kind": "train", "rows": good[:8]},
        {"kind": "train", "rows": good[8:]},
        {"kind": "train", "rows": half},
        {"kind": "train", "rows": bare[:16]},
        {"kind": "train", "rows": bare[8:24]},
        {"kind": "train", "rows": bare[16:]},
        {"kind": "holdout", "rows": bare[24:]},
    ]
    stats: dict = {}
    out = epoch_windows(plan, random.Random(0), min_valid=0.3, stats=stats)
    assert sum(1 for item in out if item["kind"] == "train") == 2
    assert stats["windows_in"] == 6 and stats["windows_dropped"] == 4, stats
    assert stats["rows_dropped"] == 16 + 32, stats
    assert stats["rows_dropped_no_center"] == 12 + 32, stats
    assert stats["recordings"] == 2 and stats["recordings_removed"] == ["b#0"], stats
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            report_valid_filter(stats, epoch=0)
    except LaneGtMissingError as exc:
        assert "b#0" in str(exc) and "1 recording" in str(exc), exc
    else:
        raise AssertionError("a recording the filter removed whole did not raise")
    assert "removed windows=4/6 rows=48 recordings=1/2; why: lane-center valid share < 0.3" in log.getvalue(), log.getvalue()

    # Only part of a recording removed: logged once, no raise; quiet after that.
    stats = {}
    epoch_windows(plan[:3], random.Random(0), min_valid=0.3, stats=stats)
    assert stats["recordings_removed"] == [] and stats["windows_dropped"] == 1
    with contextlib.redirect_stdout(io.StringIO()) as out_log:
        report_valid_filter(stats, epoch=0)
        report_valid_filter(stats, epoch=1, quiet=True)
    assert out_log.getvalue().count("min-valid filter") == 1, out_log.getvalue()

    # min_valid 0 removes nothing.
    stats = {}
    epoch_windows(plan, random.Random(0), min_valid=0.0, stats=stats)
    assert stats["windows_dropped"] == 0 and stats["recordings_removed"] == []
    with contextlib.redirect_stdout(io.StringIO()):
        report_valid_filter(stats, epoch=0)

    # Planned windows but zero counted lane_gt steps with the head on raises.
    seen = LaneGtCoverage(enabled=True)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            seen.close_epoch(2, planned=3)
        except LaneGtMissingError as exc:
            assert "zero lane_gt steps" in str(exc), exc
        else:
            raise AssertionError("an epoch with planned windows and zero lane_gt steps did not raise")
        assert LaneGtCoverage(enabled=False).close_epoch(2, planned=3)["batches"] == 0
    print("valid filter: windows 4/6, rows 48, recording b#0 removed -> raises; partial logs only OK")


def main() -> None:
    check_counter()
    check_valid_filter()
    check_zero_without_target()
    check_end_to_end()
    print("test_lane_gt_coverage: OK")


if __name__ == "__main__":
    main()
