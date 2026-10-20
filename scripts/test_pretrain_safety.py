#!/usr/bin/env python3
"""Pre-training safety fixes. CPU only, no torch.

(f) with all cameras (no --cams), a row without a decodable main picture is
    dropped (``dropped["no_main"]``), not stitched with a black front sector.
(b) ``dt_s`` is recomputed after every removal: a dropped no-main row and an
    event-masked run each become one gap, and a gap over MEMORY_S resets the
    GRU (``recording_spans``) while a short one does not.
(a) --mask-events masks every ``excursion.phase == "abort"`` row whatever its
    reason (a lane_edge abort with no impact included), unless the reason is
    on --keep-abort-reasons; counts are logged per reason.
(c) the holdout forward (``_eval_sequence`` via ``eval_resets``) clears every
    stream wherever ``should_reset_state`` is true, at the same rows
    ``recording_spans`` starts a training span.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def _png(path: Path, value: int = 128) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", np.full((8, 8, 3), value, dtype=np.uint8))
    assert ok
    path.write_bytes(buf.tobytes())


def _row(i: int, t: float, cams: tuple[str, ...], segment: str = "s#0") -> dict:
    return {
        "i": i,
        "t": t,
        "segment": segment,
        "ego": {"speed_mps": 10.0, "damage": 0.0, "dir": [0.0, 1.0, 0.0]},
        "cams": {cam: {"file": f"raw/{cam}/{i:06d}.png"} for cam in cams},
    }


def _recording(tmp: Path, layout: list[tuple[str, ...]], *, step: float = 0.1) -> list[dict]:
    rows = []
    for i, cams in enumerate(layout):
        for cam in cams:
            _png(tmp / "raw" / cam / f"{i:06d}.png")
        rows.append(_row(i, 1000.0 + step * i, cams))
    return rows


def check_no_main_dropped() -> None:
    from python.train.scene_loader import select_training_rows

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        main = ("main", "narrow", "wide")
        side = ("pillarL", "pillarR")
        rows = _recording(tmp, [main, side, main, side, main])
        dropped: dict = {}
        kept, missing, _excluded = select_training_rows(tmp, rows, cams=None, dropped=dropped)
        assert [r["i"] for r in kept] == [0, 2, 4], [r["i"] for r in kept]
        assert missing == 2 and dropped.get("no_main") == 2, (missing, dropped)
        # A camera list without main keeps the side-only rows.
        kept_side, _m, _e = select_training_rows(tmp, rows, cams=("pillarL",), dropped={})
        assert [r["i"] for r in kept_side] == [1, 3], [r["i"] for r in kept_side]
    print("(f) no-main rows dropped with all cameras: kept [0, 2, 4], dropped no_main=2 OK")


def check_dt_after_removals() -> None:
    from python.train.continual import recording_spans
    from python.train.event_mask import mask_event_rows
    from python.train.scene_loader import recompute_dt, select_training_rows
    from python.train.scene_net import MEMORY_S

    main_cams = ("main",)
    side = ("pillarL",)
    # 10 Hz: rows 0-9 drive, row 5 has no main picture, rows 10-209 (20 s)
    # are a settle run the event mask removes, rows 210-219 drive on.
    layout = [side if i == 5 else main_cams for i in range(220)]
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        rows = _recording(tmp, layout)
        for row in rows[10:210]:
            row["settle"] = True
        kept, _missing, _excluded = select_training_rows(tmp, rows, cams=None, dropped={})
        by_i = {r["i"]: r for r in kept}
        assert 5 not in by_i and abs(by_i[6]["dt_s"] - 0.2) < 1e-6, by_i[6]["dt_s"]
        masked, counts = mask_event_rows(tmp, kept)
        assert counts["settle"] == 200, counts
        stale = {r["i"]: r for r in masked}[210]["dt_s"]
        assert abs(stale - 0.1) < 1e-6, stale  # what the GRU saw before this fix
        assert len(recording_spans(masked)) == 1
        fixed = recompute_dt(masked)
        f_i = {r["i"]: r for r in fixed}
        gap = f_i[210]["dt_s"]
        assert abs(gap - 20.1) < 1e-6 and gap > MEMORY_S, gap
        assert abs(f_i[6]["dt_s"] - 0.2) < 1e-6
        spans = recording_spans(fixed)
        assert spans == [(0, 9), (9, 19)], spans
        assert fixed[0]["dt_s"] is None
        # A short masked run (2 s) is one gap but keeps the state.
        short = recompute_dt([r for r in kept if not 10 <= r["i"] < 30])
        assert len(recording_spans(short)) == 1
        assert abs({r["i"]: r for r in short}[30]["dt_s"] - 2.1) < 1e-6
    print(
        f"(b) dropped row dt_s=0.2 s (no reset); masked 20 s run dt_s {stale:.1f} -> {gap:.1f} s "
        f"> MEMORY_S={MEMORY_S:.0f} s: spans {spans} (GRU reset); 2 s mask keeps one span OK"
    )


def check_abort_mask() -> None:
    from python.train.event_mask import describe, event_mask

    def run(reason_of):
        return [
            {"i": i, "t": 100 + 0.1 * i, "segment": "s#0",
             "excursion": {"phase": "abort" if reason_of(i) else ("weave" if i < 50 else "cruise"),
                           "reason": reason_of(i) or ""}}
            for i in range(200)
        ]

    # r7: a lane_edge abort run (rows 50-79) with no impact.
    rows = run(lambda i: "lane_edge" if 50 <= i < 80 else None)
    mask, counts = event_mask(rows, [])
    hit = sum(m for r, m in zip(rows, mask) if r["excursion"]["phase"] == "abort")
    assert hit == 30 and counts["abort"] == 30 and counts["masked"] == 30, counts
    assert counts["abort_reasons"] == {"lane_edge": 30}, counts
    assert not any(m for r, m in zip(rows, mask) if r["excursion"]["phase"] != "abort")
    # Every reason is masked by default; --keep-abort-reasons keeps only the listed ones.
    reasons = {50: "lane_edge", 51: "no_path", 52: "line_end", 53: "overspeed", 54: "no_paint", 55: ""}
    mixed = run(lambda i: reasons.get(i) if i in reasons else None)
    mixed[55]["excursion"]["phase"] = "abort"
    _m, all_counts = event_mask(mixed, [])
    assert all_counts["abort"] == 6, all_counts
    assert all_counts["abort_reasons"] == {"lane_edge": 1, "no_path": 1, "line_end": 1, "overspeed": 1,
                                           "no_paint": 1, "(none)": 1}, all_counts
    kmask, kept = event_mask(mixed, [], keep_abort_reasons="no_path, line_end")
    assert kept["abort"] == 4 and kept["abort_kept"] == 2 and not kmask[51] and not kmask[52] and kmask[50], kept
    line = describe(kept, len(mixed))
    assert "abort 4 (by reason: (none)=1,lane_edge=1,no_paint=1,overspeed=1" in line and "2 kept" in line, line
    print(f"(a) lane_edge abort with no impact: {hit}/30 masked; all 6 reasons masked by default; "
          f"--keep-abort-reasons no_path,line_end keeps 2 OK\n    {line}")


def check_eval_resets() -> None:
    from python.train.continual import recording_spans
    from python.train.scene_loader import recompute_dt
    from python.train.train_scene import eval_resets

    # 40 rows at 10 Hz, a 20 s hole after row 9 and a 3 s hole after row 24.
    stamps = []
    t = 0.0
    for k in range(40):
        stamps.append(t)
        t += 20.0 if k == 9 else (3.0 if k == 24 else 0.1)
    rows = recompute_dt([{"i": k, "t": s_} for k, s_ in enumerate(stamps)])
    span_starts = {start for start, _end in recording_spans(rows)} - {0}
    for width in (1, 2):
        for every in (0, 16):
            all_reset = set()
            chunk_only = set()
            prev = None
            for k, row in enumerate(rows):
                streams = eval_resets(k, row, prev, reset_every=every, width=width)
                if streams == tuple(range(width)) and k > 0 and (every == 0 or k % every or width > 1):
                    all_reset.add(k)
                if streams == (0,) and width == 2:
                    chunk_only.add(k)
                prev = row
            assert all_reset == span_starts == {10}, (width, every, all_reset, span_starts)
            if every and width == 2:
                assert chunk_only == {0, 16, 32}, chunk_only
    assert eval_resets(0, rows[0], None, reset_every=16, width=2) == (0,)
    assert eval_resets(0, rows[0], None, reset_every=0, width=1) == ()
    assert eval_resets(5, {"dt_s": None}, rows[4], reset_every=0, width=2) == (0, 1)
    assert eval_resets(5, {"dt_s": 0.1, "session": "b"}, {"session": "a"}, reset_every=0, width=1) == (0,)
    print("(c) eval resets both streams at the recording_spans boundary (row 10, 20 s gap), not at the 3 s gap; "
          "chunk resets stay every 16 rows on stream 0 OK")


def main() -> None:
    check_eval_resets()
    check_abort_mask()
    check_no_main_dropped()
    check_dt_after_removals()
    print("test_pretrain_safety: OK")


if __name__ == "__main__":
    main()
