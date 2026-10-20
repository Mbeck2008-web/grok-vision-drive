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
(d) the effective LR per group (from ``opt.param_groups``) is logged every
    epoch and in the val line and metrics record; on resume ``lr_mult`` is
    re-set from --backbone-lr-mult before the rate is pinned, so a paused
    0.1 multiplier cannot leave the backbone at 1e-4 under --lr 1e-3.
(e) a frame cache records its strips path, row count and state.jsonl hash;
    another recording, a grown state.jsonl or a manifest without a source is
    refused; the cache drive is checked with its own free-space floor.
(g) a row missing any other trained camera (wide, narrow, side, rear) is
    dropped, and a trained sector that loads near-black is skipped in
    ``_frame_row``; with no --cams a recording that lacks a camera is refused.
    The net has no per-camera validity input, so a black sector never trains.
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
        main = ("main", "narrow", "wide", "pillarL", "pillarR")
        side = ("pillarL", "pillarR")
        rows = _recording(tmp, [main, side, main, side, main])
        dropped: dict = {}
        kept, missing, _excluded = select_training_rows(tmp, rows, cams=None, dropped=dropped)
        assert [r["i"] for r in kept] == [0, 2, 4], [r["i"] for r in kept]
        assert missing == 2 and dropped.get("no_main") == 2, (missing, dropped)
        # A camera list without main keeps the side-only rows.
        kept_side, _m, _e = select_training_rows(tmp, rows, cams=("pillarL",), dropped={})
        assert [r["i"] for r in kept_side] == [0, 1, 2, 3, 4], [r["i"] for r in kept_side]
    print("(f) no-main rows dropped with all cameras: kept [0, 2, 4], dropped no_main=2 OK")


def check_dt_after_removals() -> None:
    from python.train.continual import recording_spans
    from python.train.event_mask import mask_event_rows
    from python.train.scene_loader import recompute_dt, select_training_rows
    from python.train.scene_net import MEMORY_S

    main_cams = ("main", "pillarL")
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


def check_lr_logging() -> None:
    from python.train import train_scene as ts

    class Opt:
        def __init__(self) -> None:
            # What load_state_dict restores from a pause made with --backbone-lr-mult 0.1.
            self.param_groups = [
                {"name": "head", "lr": 1e-3, "lr_mult": 1.0},
                {"name": "backbone", "lr": 1e-4, "lr_mult": 0.1},
            ]

    stale = Opt()
    ts._pin_optimizer_lr(stale, 1e-3)
    assert ts.effective_lrs(stale) == {"head": 1e-3, "backbone": 1e-4}  # the old resume path
    fixed = Opt()
    ts.set_lr_mults(fixed, 1.0)
    ts._pin_optimizer_lr(fixed, 1e-3)
    rates = ts.effective_lrs(fixed)
    assert rates == {"head": 1e-3, "backbone": 1e-3}, rates
    assert ts.format_lrs(rates) == "head:0.001,backbone:0.001"
    half = Opt()
    ts.set_lr_mults(half, 0.5)
    ts._pin_optimizer_lr(half, 1e-3)
    assert ts.effective_lrs(half) == {"head": 1e-3, "backbone": 5e-4}
    plain = type("O", (), {"param_groups": [{"lr": 1e-3}]})()
    ts.set_lr_mults(plain, 0.1)
    ts._pin_optimizer_lr(plain, 1e-3)
    assert ts.effective_lrs(plain) == {"group0": 1e-3}, "scratch net: one group, never lowered"
    source = Path(ts.__file__).read_text(encoding="utf-8")
    assert '"lr": lr_now,' in source and "lr={format_lrs(lr_now)}" in source
    assert source.count("set_lr_mults(holder[\"opt\"], float(backbone_lr_mult))") == 1
    real = ""
    try:
        import torch
    except ImportError:
        real = " (torch not installed: AdamW round trip skipped)"
    else:
        a = torch.nn.Linear(2, 2)
        b = torch.nn.Linear(2, 2)
        saved = torch.optim.AdamW([{"params": a.parameters(), "name": "head", "lr": 1e-3, "lr_mult": 1.0},
                                   {"params": b.parameters(), "name": "backbone", "lr": 1e-4, "lr_mult": 0.1}])
        opt = torch.optim.AdamW([{"params": a.parameters(), "name": "head", "lr": 1e-3, "lr_mult": 1.0},
                                 {"params": b.parameters(), "name": "backbone", "lr": 1e-3, "lr_mult": 1.0}])
        opt.load_state_dict(saved.state_dict())
        assert opt.param_groups[1]["lr_mult"] == 0.1, "load_state_dict restores the paused lr_mult"
        ts.set_lr_mults(opt, 1.0)
        ts._pin_optimizer_lr(opt, 1e-3)
        assert ts.effective_lrs(opt) == {"head": 1e-3, "backbone": 1e-3}
        real = "; real AdamW load_state_dict round trip"
    print(f"(d) resume with a paused lr_mult 0.1: backbone 1e-4 before, {rates['backbone']:g} after; "
          f"lr logged per epoch, in the val line and metrics record{real} OK")


def check_cache_manifest() -> None:
    import json

    from python.data.data_root import DataDriveError, check_data_path
    from python.data.raw_frames import RawFrameWriter
    from python.train.frame_cache import CacheMismatchError, FrameCache, build_frame_cache

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)

        def rec(name: str, value: int, n: int = 3) -> Path:
            writer = RawFrameWriter(tmp / name, [])
            for _ in range(n):
                writer.write({"main": np.full((48, 64, 3), value, np.uint8)}, t=1.0, tech={"lane_gt": [0]})
            return tmp / name

        a, b = rec("rec5", 10), rec("rec5b", 200)
        man = build_frame_cache(a, tmp / "cache", ("main",))
        assert set(man["source"]) == {"strips", "rows", "state_sha256"} and man["source"]["rows"] == 3, man
        assert FrameCache(tmp / "cache", strips_root=a).has(1)  # same recording: accepted
        refused = ""
        try:
            FrameCache(tmp / "cache", strips_root=b)  # r3: another recording, same row indexes
        except CacheMismatchError as exc:
            refused = str(exc)
        assert "strips=" in refused, refused
        with open(a / "state.jsonl", "ab") as handle:  # the recording grew after the cache was built
            handle.write(b'{"i": 3, "t": 1.3}\n')
        try:
            FrameCache(tmp / "cache", strips_root=a)
            raise AssertionError("grown state.jsonl accepted")
        except CacheMismatchError as exc:
            assert "rows=3" in str(exc), exc
        payload = json.loads((tmp / "cache" / "manifest.json").read_text())
        payload.pop("source")
        (tmp / "cache" / "manifest.json").write_text(json.dumps(payload))
        try:
            FrameCache(tmp / "cache", strips_root=a)
            raise AssertionError("old manifest accepted")
        except CacheMismatchError as exc:
            assert "no source" in str(exc), exc
        assert FrameCache(tmp / "cache").has(1), "a worker reopening a checked cache is not re-hashed"
        try:
            check_data_path(tmp, minimum=1 << 62)
            raise AssertionError("free-space floor not applied")
        except DataDriveError as exc:
            assert "free" in str(exc), exc
    print(f"(e) cache from rec5 used for rec5b refused: {refused.split(': ', 1)[1][:60]}...; grown state.jsonl, "
          "source-less manifest and a full cache drive refused OK")


def check_other_cams() -> None:
    from python.data.strip_writer import compose_strip
    from python.train import train_scene as ts
    from python.train.scene_loader import recorded_cams, select_training_rows

    fast = ("main", "narrow", "wide")
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        # rec5b-like: 10 Hz cameras every tick, pillars every 5th, rear every 10th.
        layout = []
        for i in range(20):
            cams = fast + (("pillarL", "pillarR") if i % 5 == 0 else ()) + (("rear",) if i % 10 == 0 else ())
            layout.append(cams)
        rows = _recording(tmp, layout)
        assert set(recorded_cams(tmp)) == {"main", "narrow", "wide", "pillarL", "pillarR", "rear"}, recorded_cams(tmp)
        dropped: dict = {}
        kept, missing, _e = select_training_rows(tmp, rows, cams=None, dropped=dropped)
        assert [r["i"] for r in kept] == [0, 10] and missing == 18, ([r["i"] for r in kept], dropped)
        assert dropped["missing_cam"] == 18 and dropped["missing_cam:pillarL"] == 16 and dropped["missing_cam:rear"] == 18
        three, _m, _e = select_training_rows(tmp, rows, cams=fast, dropped={})
        assert len(three) == 20, "the 10 Hz cameras keep every tick"
        side, _m, _e = select_training_rows(tmp, rows, cams=("main", "pillarL"), dropped=(d := {}))
        assert [r["i"] for r in side] == [0, 5, 10, 15] and d == {"missing_cam": 16, "missing_cam:pillarL": 16}, d

    # A file that exists but is black (dead camera, or a cache with no chunk for it).
    lit = np.random.RandomState(0).randint(30, 200, size=(48, 64, 3)).astype(np.uint8)
    full = compose_strip({cam: lit for cam in ("main", "narrow", "wide", "pillarL", "pillarR", "rear", "repeatL", "repeatR")})
    assert ts.blank_sectors(full, None) == ()
    dead = compose_strip({cam: lit for cam in ("main", "narrow", "pillarL", "pillarR", "rear", "repeatL", "repeatR")}
                         | {"wide": np.zeros((48, 64, 3), np.uint8)})
    assert ts.blank_sectors(dead, None) == ("wide",), ts.blank_sectors(dead, None)

    class Cache:
        def __init__(self, frames):
            self.frames = frames

        def has(self, index):
            return index in self.frames

        def canvas(self, index):
            return self.frames[index]

    before = ts.BLANK_CAM_SKIPS
    main_before = ts.BLANK_MAIN_SKIPS
    assert ts._frame_row(Path("."), {"i": 1}, cams=None, cache=Cache({1: dead})) is None
    assert ts._frame_row(Path("."), {"i": 2}, cams=None, cache=Cache({2: full})) is not None
    assert ts.BLANK_CAM_SKIPS == before + 1 and ts.BLANK_MAIN_SKIPS == main_before
    refused = "(torch not installed: train_directory refusal not run)"
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            _recording(tmp, [("main",)] * 4)
            try:
                ts.train_directory(tmp, epochs=1, lr=1e-3, capability=None, export_path=None,
                                   open_panel=False, autostart=True, dry_run=True)
                raise AssertionError("main-only recording trained with all eight sectors")
            except SystemExit as exc:
                assert "--cams main" in str(exc), exc
                refused = "train_directory refuses no --cams on a main-only recording"
    print(f"(g) rows missing pillar/rear dropped (18 of 20 with all cameras, 0 with --cams main,narrow,wide); "
          f"a black wide sector skipped by _frame_row; {refused} OK")


def main() -> None:
    check_other_cams()
    check_cache_manifest()
    check_lr_logging()
    check_eval_resets()
    check_abort_mask()
    check_no_main_dropped()
    check_dt_after_removals()
    print("test_pretrain_safety: OK")


if __name__ == "__main__":
    main()
