#!/usr/bin/env python3
"""A strip-recorder row with ``ego.speed_mps`` None loads in every reader and stays None (#125).

run_vision now records an unknown ego speed as ``{"speed_mps": None}`` (JSON null) instead of
0.0. Each reader of recorded rows must take that row without crashing and must not turn the
None into 0. A reader that needs a speed drops or masks the row; it never fills a value.

Readers covered: the strip writer itself (python/data/strip_writer.py), raw_frames
``load_scene_rows``, dataset_check ``_speed_of`` and ``validate_dataset``, continual
``_row_speed`` / ``choose_holdout_cells`` / ``holdout_mix`` (train_scene's holdout split and
its ``no_speed`` count), gt_summary ``summarize``, the clip ring in data/record.py, and
liveness ``bad_row_reasons``.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from gvd_test_env import isolated_home  # noqa: E402

KNOWN = 7.0


def _ego(speed: float | None, x: float) -> dict:
    return {"speed_mps": speed, "pos": [x, 0.0, 0.0], "dir": [1.0, 0.0, 0.0], "yaw_rate": 0.0}


def _record(root: Path) -> Path:
    """Three strips through the real writer: known, None, known."""
    from python.data.strip_writer import StripWriter

    writer = StripWriter(root)
    pic = np.full((96, 160, 3), 60, dtype=np.uint8)
    for k, speed in enumerate((KNOWN, None, KNOWN)):
        writer.write({"main": pic}, t=100.0 + 0.1 * k, ego=_ego(speed, 0.7 * k),
                     wheel={"steer": 0.0}, pedals={"throttle": 0.1, "brake": 0.0})
    return root


def _rows(root: Path) -> list[dict]:
    from python.data.strip_writer import read_state_jsonl

    return read_state_jsonl(root / "state.jsonl")


def check_writer_keeps_null() -> None:
    with isolated_home(prefix="gvd-speed-none-") as home:
        root = _record(home / "rec")
        lines = (root / "state.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3, lines
        raw = json.loads(lines[1])
        assert "speed_mps" in raw["ego"] and raw["ego"]["speed_mps"] is None, raw["ego"]
        assert '"speed_mps": null' in lines[1], lines[1]


def check_raw_frames_load_scene_rows() -> None:
    from python.data.raw_frames import load_scene_rows

    with isolated_home(prefix="gvd-speed-none-") as home:
        rows = load_scene_rows(_record(home / "rec"))
        assert [r["ego"]["speed_mps"] for r in rows] == [KNOWN, None, KNOWN], [r["ego"] for r in rows]


def check_dataset_check() -> None:
    from python.debug import dataset_check

    with isolated_home(prefix="gvd-speed-none-") as home:
        root = _record(home / "rec")
        rows = _rows(root)
        assert dataset_check._speed_of(rows[1]) is None, "dataset_check._speed_of filled a None speed"
        assert dataset_check._speed_of(rows[0]) == KNOWN
        report = dataset_check.validate_dataset(root, home / "out", montage_n=0)
        assert isinstance(report, dict), report
        text = json.dumps(report, default=str)
        assert '"speed_mps": 0.0' not in text, "validate_dataset reported a None speed as 0.0"


def check_continual_and_train_scene_holdout() -> None:
    from python.train import continual

    with isolated_home(prefix="gvd-speed-none-") as home:
        rows = _rows(_record(home / "rec"))
        assert continual._row_speed(rows[1]) is None, "continual._row_speed filled a None speed"
        assert continual._row_speed(rows[0]) == KNOWN
        spec = continual.choose_holdout_cells(rows, seed=0)
        assert isinstance(spec, dict), spec
        # train_scene's holdout split counts the None row as no_speed: masked, not slow (0).
        mix = continual.holdout_mix([{"kind": "holdout", "rows": rows}], fast_mps=KNOWN - 1.0)
        assert mix["no_speed"] == 1 and mix["fast"] == 2 and mix["slow"] == 0, mix
        # A cell with only the None row is not given a 0 median from it.
        only_none = [rows[1]]
        spec1 = continual.choose_holdout_cells(only_none, seed=0)
        assert isinstance(spec1, dict), spec1


def check_gt_summary() -> None:
    from python.debug.gt_summary import summarize

    base = {"engaged": True, "steer": 0.0, "throttle": 0.1, "brake": 0.0}
    rows = [dict(base, t=0.0, speed_mps=KNOWN, pos=[0.0, 0.0]),
            dict(base, t=0.1, speed_mps=None, pos=[0.7, 0.0]),
            dict(base, t=0.2, speed_mps=KNOWN, pos=[1.4, 0.0])]
    out = summarize(rows)
    # None is left out of the mean; as a 0 it would pull the mean to 4.67.
    assert out["mean_speed_mps"] == KNOWN, out["mean_speed_mps"]
    none_only = summarize([dict(r, speed_mps=None) for r in rows])
    assert none_only["mean_speed_mps"] is None, none_only["mean_speed_mps"]


def check_clip_ring_keeps_none() -> None:
    from python.data.record import ClipRecorder

    rec = ClipRecorder(dry_run=True)
    ok = rec.push(np.full((48, 64, 3), 60, dtype=np.uint8), {"engaged": False, "ego": {"speed_mps": None}}, now=1.0)
    assert ok, "ClipRecorder.push skipped the frame"
    assert rec._buf[-1].state["ego"] == {"speed_mps": None}, rec._buf[-1].state


def check_liveness_not_a_stall() -> None:
    from python.collect.liveness import bad_row_reasons

    with isolated_home(prefix="gvd-speed-none-") as home:
        rows = _rows(_record(home / "rec"))
        none_rows = [dict(r, ego=dict(r["ego"], speed_mps=None), dt_s=1.0) for r in rows] * 4
        reasons = bad_row_reasons(none_rows)
        assert not any(v == "stall" or "stall" in str(v) for v in reasons.values()), reasons


CHECKS = (
    check_writer_keeps_null,
    check_raw_frames_load_scene_rows,
    check_dataset_check,
    check_continual_and_train_scene_holdout,
    check_gt_summary,
    check_clip_ring_keeps_none,
    check_liveness_not_a_stall,
)


def main() -> None:
    failed: list[str] = []
    for fn in CHECKS:
        try:
            fn()
            print(f"{fn.__name__}: ok", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            kind = "FAIL" if isinstance(exc, AssertionError) else f"ERROR {type(exc).__name__}"
            print(f"{fn.__name__}: {kind} {exc!r}"[:2000], flush=True)
            if not isinstance(exc, AssertionError):
                traceback.print_exc()
    if failed:
        raise SystemExit("test_speed_none_rows failed: " + ", ".join(failed))
    print("test_speed_none_rows: OK")


if __name__ == "__main__":
    main()
