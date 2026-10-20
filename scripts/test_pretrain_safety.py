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
(g) a row missing any other trained camera (wide, narrow, side, rear: no file
    written) is dropped; with no --cams a recording that lacks a camera is
    refused. A near-black sector that was written is only counted
    (``dark_sector_count``) and still trains: darkness is not a missing flag.
(e2) only the --cache folder may be on C: (free-space floor still applies);
    a recording/strips path on C: or D:, and a cache on D:, stay refused.
(h) camera contract: the trained camera list goes into metrics.jsonl (run
    record) and the ONNX ``gvd_cams``; the live runtime refuses a model with
    no list or with a camera the supervisor does not feed.
(i) several recordings train together through a combined folder of hard
    links (byte copies across volumes; never re-encoded); each row keeps its
    recording id, a per-recording usable_until cut is honoured, and the
    metadata records the cut, its reason and whether the run finished.
(j) the holdout splits by road area (250 m map cells), the same cells for
    every recording, so no road has frames in both train and val; it is the
    default when more than one recording is trained.
(k) trained-heads contract: gvd_heads lists only heads that had labels; the
    live runtime reads no other output, so untrained objects/signs/curbs are
    never drawn or fed to the planner; a live model with no gvd_heads is
    refused like one with no gvd_cams.
(l) hard links share the raw bytes: combining, building a frame cache over
    the combined folder and reading it for training leave every source
    file's sha256, size and mtime unchanged.
(m) the train/val composition (rows, buffer rows, phase and speed-band
    shares) is computed per split, a >2x val/train share gap warns, and the
    cell seed is a fixed-default flag recorded in the run metadata.
(n) the run metadata states the road-area split's limit: the 50 m buffer
    covers the 30 m labels, but cameras see held-out road past 50 m, so it is
    reduced leakage, not no leakage.
(o) a head is stamped only when its labels clear LABEL_FLOOR_SHARE (1%) and
    LABEL_FLOOR_ROWS (100); labelled_rows/share are recorded for every head.
(p) the label count runs on the rows that train (after the holdout split and
    --min-valid), not all rows: a head labelled only in val rows, or only in
    windows --min-valid drops, is not stamped.
(q) an export-only run stamps exactly the heads in the checkpoint's run
    record (saved with the weights); a checkpoint without one, no
    checkpoint, or a disagreeing --heads is refused.
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

    before = ts.DARK_SECTOR_COUNT
    main_before = ts.DARK_MAIN_COUNT
    assert ts._frame_row(Path("."), {"i": 1}, cams=None, cache=Cache({1: dead})) is not None, "dark is not dropped"
    assert ts._frame_row(Path("."), {"i": 2}, cams=None, cache=Cache({2: full})) is not None
    assert ts.DARK_SECTOR_COUNT == before + 1 and ts.DARK_MAIN_COUNT == main_before
    assert not hasattr(ts, "BLANK_CAM_SKIPS")
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
          f"a dark wide sector is counted (dark_sector_count) and kept; {refused} OK")


def check_cache_on_c() -> None:
    import contextlib
    import io

    from python.data import data_root
    from python.data.data_root import DataDriveError, check_cache_path, check_data_path
    from python.train import train_scene as ts

    gb = 1024 ** 3
    real_free = data_root.free_bytes
    free = {"bytes": 50 * gb}
    data_root.free_bytes = lambda _path: free["bytes"]
    try:
        assert check_cache_path(r"C:\GVD_cache\rec5b_7162fd559a6a", minimum=8 * gb) == 50 * gb
        free["bytes"] = 5 * gb
        try:
            check_cache_path(r"C:\GVD_cache\rec5b_7162fd559a6a", minimum=8 * gb)
            raise AssertionError("C: cache below the 8 GB floor accepted")
        except DataDriveError as exc:
            assert "need 8 GB" in str(exc), exc
        free["bytes"] = 50 * gb
        for bad in (r"C:\GVD_data\scene_lanes_rec5b", r"D:\GVD_data\scene_lanes_rec5b"):
            try:
                check_data_path(bad)
                raise AssertionError(f"recording on {bad[:2]} accepted")
            except DataDriveError as exc:
                assert "C: and D:" in str(exc), exc
        try:
            check_cache_path(r"D:\GVD_cache\x", minimum=8 * gb)
            raise AssertionError("D: cache accepted")
        except DataDriveError:
            pass

        def run(argv: list[str]) -> str:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                try:
                    ts.main(argv)
                    raise AssertionError(f"main accepted {argv}")
                except SystemExit as exc:
                    assert exc.code == 2, exc.code
            return out.getvalue()

        msg = run(["--cache", r"C:\GVD_cache\rec5b", "--strips", r"C:\GVD_data\scene_lanes_rec5b"])
        assert "C: and D:" in msg and r"C:\GVD_data" in msg, msg
        msg = run(["--build-cache", r"C:\GVD_cache\rec5b", "--strips", r"C:\GVD_data\scene_lanes_rec5b"])
        assert "C: and D:" in msg and r"C:\GVD_data" in msg, msg
        with tempfile.TemporaryDirectory() as raw:
            # Strips off C:, cache on C:: the drive checks pass and the manifest check decides.
            msg = run(["--cache", r"C:\GVD_cache\rec5b_none", "--strips", raw])
            assert "no manifest.json" in msg and "C: and D:" not in msg, msg
            data_root.free_bytes = lambda path: 5 * gb if str(path).startswith("C:") else 50 * gb
            msg = run(["--cache", r"C:\GVD_cache\rec5b_none", "--strips", raw, "--cache-min-free-gb", "8"])
            assert "C:" in msg and "need 8 GB" in msg and "C: and D:" not in msg, msg
    finally:
        data_root.free_bytes = real_free
    print("(e2) C: cache allowed at 50 GB free, refused under the 8 GB floor; C:/D: recording and D: cache refused; "
          "main() refuses --strips on C: with a C: cache OK")


def _tiny_scene_onnx(path: Path, cams: tuple[str, ...] | None, *, heads: tuple[str, ...] | None = ("lane_gt",),
                     extra: tuple[str, ...] = ()) -> bool:
    """A constant-output graph with the scene runtime's strip input. False without onnx."""
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return False
    from python.data.strip_writer import CANVAS_H
    from python.debug.lane_targets import LANE_GT_DIM
    from python.train.cam_strip import metadata_dict, scaled_layout

    names = cams or ("main", "narrow", "wide", "pillarL", "pillarR", "repeatL", "repeatR", "rear")
    _sectors, _h, width = scaled_layout(tuple(names), 1.0)
    value = helper.make_tensor("v", TensorProto.FLOAT, [1, LANE_GT_DIM], [0.0] * LANE_GT_DIM)
    nodes = [helper.make_node("Constant", [], ["lane_gt"], value=value)]
    outs = [helper.make_tensor_value_info("lane_gt", TensorProto.FLOAT, [1, LANE_GT_DIM])]
    for name in extra:  # untrained heads: confident garbage
        nodes.append(helper.make_node("Constant", [], [name],
                                      value=helper.make_tensor(name + "_v", TensorProto.FLOAT, [1, 4], [9.0] * 4)))
        outs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 4]))
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("strip", TensorProto.FLOAT, [1, 3, CANVAS_H, int(width)])],
        outs,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    if cams is not None:
        for key, val in metadata_dict(tuple(cams), heads, height=CANVAS_H, width=int(width)).items():
            if key == "gvd_heads" and heads is None:
                continue
            entry = model.metadata_props.add()
            entry.key, entry.value = key, val
    onnx.save(model, str(path))
    return True


def check_camera_contract() -> None:
    from python.train.scene_lane_runtime import (
        SceneCameraContractError,
        SceneInputShapeError,
        SceneLaneGuard,
        SceneLaneRuntime,
        check_camera_contract,
        live_cams_from_config,
    )

    eight = ("narrow", "main", "wide", "pillarL", "pillarR", "repeatL", "repeatR", "rear")
    three = {"gvd_cams": "main,narrow,wide"}
    assert check_camera_contract(three, eight) == ("wide", "main", "narrow")
    assert check_camera_contract(three, ("main", "narrow", "wide")) == ("wide", "main", "narrow")
    for meta, live, why in (({}, eight, "no gvd_cams"), (three, ("main", "wide"), "missing narrow"),
                            ({"gvd_cams": "main,rear"}, ("main", "narrow", "wide"), "missing rear")):
        try:
            check_camera_contract(meta, live)
            raise AssertionError(f"accepted {meta} with live {live}")
        except SceneCameraContractError as exc:
            assert why.split()[-1] in str(exc), exc
    assert issubclass(SceneCameraContractError, SceneInputShapeError), "the guard must refuse Engage on it"
    cfg = {"cameras": [{"id": "narrow"}, {"id": "main"}, {"id": "wide"}, {"id": "main"}, {"note": "x"}]}
    assert live_cams_from_config(cfg) == ("narrow", "main", "wide")
    try:
        import yaml  # noqa: F401
    except ImportError:
        pass
    else:
        from python.sensors.cameras import load_camera_config

        assert set(live_cams_from_config(load_camera_config())) == set(eight)

    root = Path(__file__).resolve().parents[1]
    vision = (root / "python" / "run_vision.py").read_text(encoding="utf-8")
    assert "SceneLaneRuntime(path, live_cams=live_cams)" in vision and "live_cams_from_config(load_camera_config())" in vision
    trainer = (root / "python" / "train" / "train_scene.py").read_text(encoding="utf-8")
    assert '{"kind": "run", "t": run_started, "cams": run_cams,' in trainer and '"cams": run_cams,' in trainer
    exporter = (root / "python" / "train" / "scene_net.py").read_text(encoding="utf-8")
    assert "or tuple(cams or ()) or tuple(STRIP_ORDER)" in exporter, "export always stamps gvd_cams"

    live_note = "(onnx/onnxruntime not installed: runtime load not run)"
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        if _tiny_scene_onnx(tmp / "three.onnx", ("main", "narrow", "wide")):
            try:
                import onnxruntime  # noqa: F401
            except ImportError:
                pass
            else:
                _tiny_scene_onnx(tmp / "old.onnx", None)
                _tiny_scene_onnx(tmp / "rear.onnx", ("main", "rear"))
                ok = SceneLaneRuntime(tmp / "three.onnx", live_cams=eight)
                assert ok.ready and ok.cams == ("wide", "main", "narrow"), ok.reason
                for name, live in (("three.onnx", ("main", "wide")), ("old.onnx", eight), ("rear.onnx", ("main", "narrow", "wide"))):
                    guard = SceneLaneGuard()
                    got = guard.load(lambda path, live=live: SceneLaneRuntime(path, live_cams=live), tmp / name)
                    assert got is None and guard.refused, (name, live)
                legacy = SceneLaneRuntime(tmp / "old.onnx")
                assert legacy.ready, "no live_cams given: old callers keep the old behaviour"
                live_note = "runtime + SceneLaneGuard refuse Engage on a mismatched or unlabelled model"
    print(f"(h) gvd_cams contract: main,narrow,wide accepted on the 8-camera live feed, refused without narrow; "
          f"no gvd_cams refused; {live_note} OK")


def _write_recording(root: Path, n: int, *, t0: float, pos=lambda i: (0.0, float(i)), value: int = 100,
                     segment: str = "west_coast_usa#0", dt: float = 0.1) -> None:
    import json

    rows = []
    for i in range(n):
        for cam in ("main", "narrow", "wide"):
            _png(root / "raw" / cam / f"{i:06d}.png", value)
        x, y = pos(i)
        rows.append({"i": i, "t": t0 + dt * i, "segment": segment, "map": "west_coast_usa",
                     "ego": {"speed_mps": 10.0, "damage": 0.0, "dir": [0.0, 1.0, 0.0], "pos": [x, y, 0.0]},
                     "cams": {cam: {"file": f"raw/{cam}/{i:06d}.png"} for cam in ("main", "narrow", "wide")}})
    (root / "state.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (root / "gt.jsonl").write_text("".join(json.dumps({"i": r["i"], "t": r["t"], "tech": {"lane_gt": [0.0]}}) + "\n"
                                           for r in rows), encoding="utf-8")
    (root / "cameras.json").write_text(json.dumps({"cameras": [{"id": "main"}, {"id": "narrow"}, {"id": "wide"}]}),
                                       encoding="utf-8")


def check_multi_recording() -> None:
    import json
    import os

    from python.train.combine_recordings import (
        CombineError,
        combine_recordings,
        describe_recording,
        parse_usable_until,
        recordings_metadata,
    )
    from python.train.continual import recording_spans
    from python.train.scene_loader import select_training_rows

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        a, b = tmp / "rec5b", tmp / "rec5c"
        _write_recording(a, 12, t0=1000.0, value=90)
        _write_recording(b, 8, t0=1000.5, value=200)  # overlapping clock: only the session splits them
        (a / "collect.json").write_text(json.dumps({"state": "running", "done_s": 3605.0, "target_s": 14400.0}))
        (b / "collect.json").write_text(json.dumps({"state": "done", "done_s": 14400.0, "target_s": 14400.0}))
        (a / "impacts.jsonl").write_text(json.dumps({"kind": "impact", "t": 1000.3, "segment": "west_coast_usa#0"}) + "\n"
                                         + json.dumps({"kind": "impact", "t": 1001.05, "segment": "west_coast_usa#0"}) + "\n")
        (b / "usable_until.json").write_text(json.dumps({"max_i": 6, "reason": "sidecar cut"}))
        flags = parse_usable_until(["rec5b=9:stopped early (commit-thread failure), not finished"])
        out = tmp / "combined"
        man = combine_recordings([a, b], out, usable_until=flags)
        assert man["rows"] == 10 + 7, man["rows"]
        recs = {r["id"]: r for r in man["recordings"]}
        assert recs["rec5b"]["cut"] == {"max_i": 9, "reason": "stopped early (commit-thread failure), not finished",
                                        "source": "flag"}
        assert recs["rec5c"]["cut"]["max_i"] == 6 and recs["rec5c"]["cut"]["source"] == "usable_until.json"
        assert recs["rec5b"]["finished"] is False and recs["rec5c"]["finished"] is True
        line = describe_recording(recs["rec5b"])
        assert line == ("rec5b: cut at row 9 (stopped early (commit-thread failure), not finished), "
                        "not finished, 10 of 12 rows"), line
        # Frames are links to the very same bytes: nothing decoded or re-encoded.
        src = a / "raw" / "main" / "000003.png"
        dst = out / "raw" / "main" / "000003.png"
        assert dst.read_bytes() == src.read_bytes()
        assert os.path.samefile(src, dst) and man["files"]["hardlink"] == 17 * 3 and man["files"]["copy"] == 0
        assert (out / "raw" / "main" / "000010.png").read_bytes() == (b / "raw" / "main" / "000000.png").read_bytes()
        rows = [json.loads(line) for line in (out / "state.jsonl").read_text().splitlines()]
        assert [r["recording"] for r in rows] == ["rec5b"] * 10 + ["rec5c"] * 7
        assert rows[10]["src_i"] == 0 and rows[10]["i"] == 10 and rows[10]["segment"] == "rec5c/west_coast_usa#0"
        assert rows[10]["cams"]["main"]["file"] == "raw/main/000010.png"
        gts = [json.loads(line) for line in (out / "gt.jsonl").read_text().splitlines()]
        assert [g["i"] for g in gts] == list(range(17)) and gts[12]["recording"] == "rec5c"
        imps = [json.loads(line) for line in (out / "impacts.jsonl").read_text().splitlines()]
        assert [i["segment"] for i in imps] == ["rec5b/west_coast_usa#0"], "impact after the cut dropped"
        kept, missing, _e = select_training_rows(out, rows)
        assert missing == 0 and len(kept) == 17 and kept[10]["dt_s"] is None, kept[10]["dt_s"]
        assert recording_spans(kept) == [(0, 10), (10, 17)]
        assert recordings_metadata(out) == man["recordings"]
        # Same sources and cuts: reused. Another cut into the same folder: refused, nothing overwritten.
        assert combine_recordings([a, b], out, usable_until=flags) == man
        try:
            combine_recordings([a, b], out, usable_until={})
            raise AssertionError("different combination written over the old one")
        except CombineError as exc:
            assert "different combination" in str(exc)
        try:
            combine_recordings([a, tmp / "x" / "rec5b"], tmp / "c2")
            raise AssertionError("duplicate ids")
        except CombineError:
            pass
        # A plain folder honours its own sidecar cut in the metadata too.
        meta = recordings_metadata(b, rows=[{}] * 7)
        assert meta[0]["cut"]["max_i"] == 6 and meta[0]["finished"] is True
    from python.train.train_scene import build_parser

    args = build_parser().parse_args(["--recording", "A", "--recording", "B", "--combined-out", "C",
                                      "--usable-until", "rec5b=9:x", "--usable-until", "rec5c=6"])
    assert args.recording == ["A", "B"] and args.combined_out == "C" and len(args.usable_until) == 2
    print(f"(i) rec5b+rec5c combined by hard link (17x3 links, 0 re-encodes); ids kept per row; GRU resets "
          f"between recordings; cut logged as '{line}' OK")


def check_road_area_holdout() -> None:
    import json
    import math

    from python.train.combine_recordings import combine_recordings
    from python.train.continual import (
        HOLDOUT_BUFFER_M,
        HOLDOUT_CELL_M,
        cell_holdout_labels,
        cell_of,
        choose_holdout_cells,
    )
    from python.train.train_scene import build_parser, count_recordings, resolve_holdout_cells

    def loop(i: int, phase: float) -> tuple[float, float]:
        # Both recordings drive the same 2 km x 1.2 km loop at about 26 m/s, from different points.
        a = (i * 0.004 + phase) % 1.0
        per = 2 * (2000 + 1200)
        d = a * per
        if d < 2000:
            return d, 0.0
        if d < 3200:
            return 2000.0, d - 2000
        if d < 5200:
            return 2000 - (d - 3200), 1200.0
        return 0.0, 1200 - (d - 5200)

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        _write_recording(tmp / "rec5b", 600, t0=1000.0, pos=lambda i: loop(i, 0.0), dt=1.0)
        _write_recording(tmp / "rec5c", 500, t0=1010.0, pos=lambda i: loop(i, 0.37), dt=1.0)
        out = tmp / "combined"
        combine_recordings([tmp / "rec5b", tmp / "rec5c"], out)
        rows = [json.loads(line) for line in (out / "state.jsonl").read_text().splitlines()]
        spec = choose_holdout_cells(rows, seed=3)
        # The trainer's own checks: one map, though each recording prefixes its segments.
        from python.train.continual import check_holdout_cells, row_maps, segment_map

        assert row_maps(rows) == ["west_coast_usa"] and spec["map"] == "west_coast_usa", row_maps(rows)
        assert segment_map("rec5b/west_coast_usa#3") == segment_map("west_coast_usa#3") == "west_coast_usa"
        check_holdout_cells(spec, rows, source="chosen cells")
        labels = cell_holdout_labels(rows, spec)
        sides: dict[tuple[int, int], set[str]] = {}
        by_rec: dict[str, dict[str, set]] = {}
        for row, lab in zip(rows, labels):
            xy = (row["ego"]["pos"][0], row["ego"]["pos"][1])
            sides.setdefault(cell_of(xy), set()).add(lab)
            by_rec.setdefault(row["recording"], {}).setdefault(lab, set()).add(cell_of(xy))
        assert all(not {"train", "hold"} <= s for s in sides.values()), "a road area on both sides"
        assert by_rec["rec5b"]["hold"] & by_rec["rec5c"]["hold"], "the same road held out in both recordings"
        assert not (by_rec["rec5b"]["train"] | by_rec["rec5c"]["train"]) & (
            by_rec["rec5b"]["hold"] | by_rec["rec5c"]["hold"]), "a held cell trains in another recording"
        trains = [r["ego"]["pos"][:2] for r, lab in zip(rows, labels) if lab == "train"]
        holds = [r["ego"]["pos"][:2] for r, lab in zip(rows, labels) if lab == "hold"]
        assert trains and holds
        nearest = min(math.dist(a, b) for a in trains[::3] for b in holds[::3])
        assert nearest >= HOLDOUT_BUFFER_M - 10.0, nearest  # sampled rows are about 8 m apart
        n_cells = len({cell_of(tuple(h)) for h in holds})
        n_shared = len(by_rec["rec5b"]["hold"] & by_rec["rec5c"]["hold"])
        assert count_recordings(out) == 2 and count_recordings(tmp / "rec5b") == 1
    assert resolve_holdout_cells(None, 2, 1) is True
    assert resolve_holdout_cells(None, 1, 1) is False
    assert resolve_holdout_cells(None, 2, 4) is False, "explicit time blocks win"
    assert resolve_holdout_cells(False, 2, 1) is False and resolve_holdout_cells(True, 1, 1) is True
    assert build_parser().parse_args([]).holdout_cells is None
    assert build_parser().parse_args(["--no-holdout-cells"]).holdout_cells is False
    print(f"(j) two recordings on one loop: {n_cells} held {int(HOLDOUT_CELL_M)} m cells ({n_shared} driven by both), no cell "
          f"in train and val, nearest train row {nearest:.0f} m from val; default on for 2 recordings OK")


def check_label_floor(row: dict) -> None:
    """(o) both sides of the label floor, on share and on row count."""
    from python.train.scene_net import LABEL_FLOOR_ROWS, LABEL_FLOOR_SHARE, labelled_heads

    assert (LABEL_FLOOR_SHARE, LABEL_FLOOR_ROWS) == (0.01, 100)
    blank = {"tech": {}}
    cases = (
        (100, 9900, True, "100 rows = 1.00%: on both floors"),
        (99, 9801, False, "99 rows = 1.00%: under the row floor"),
        (100, 10001, False, "100 rows = 0.9999%: under the share floor"),
        (150, 0, True, "150 rows = 100%"),
        (5, 0, False, "5 rows = 100%: share is not enough"),
    )
    for good, bad, want, why in cases:
        stamped, below, counts = labelled_heads([row] * good + [blank] * bad, ("lane_gt",))
        assert (stamped == ("lane_gt",)) is want and (below == ("lane_gt",)) is not want, why
        assert counts["lane_gt"]["labelled_rows"] == good and counts["lane_gt"]["stamped"] is want, (why, counts)
    print(f"(o) label floor {LABEL_FLOOR_SHARE:.0%} and {LABEL_FLOOR_ROWS} rows: "
          + "; ".join(f"{why} -> {'stamped' if want else 'left off'}" for _g, _b, want, why in cases) + " OK")


def check_heads_contract() -> None:
    from python.debug.lane_targets import LANE_GT_DIM
    from python.train.cam_strip import metadata_dict
    from python.train.scene_lane_runtime import (
        SceneHeadsContractError,
        SceneInputShapeError,
        SceneLaneGuard,
        SceneLaneRuntime,
        check_heads_contract,
        trained_outputs,
    )
    from python.train.scene_net import OUTPUT_FIELDS, labelled_heads

    assert issubclass(SceneHeadsContractError, SceneInputShapeError), "the guard must refuse Engage on it"
    assert check_heads_contract({"gvd_heads": "lane_gt"}) == ("lane_gt",)
    assert check_heads_contract({}, required=False) == ("lane_gt",), "off the live path: lane-only"
    for meta in ({}, {"gvd_heads": ""}, {"gvd_heads": "objects,signs"}, {"gvd_heads": "lane_gt,bogus"}):
        try:
            check_heads_contract(meta)
            raise AssertionError(f"accepted {meta}")
        except SceneHeadsContractError:
            pass
    # Every graph output, untrained heads included: only lane_gt and the state survive.
    everything = {name: [9.0] for name in OUTPUT_FIELDS + ("h", "tokens", "mask")}
    kept = trained_outputs(everything, ("lane_gt",))
    assert set(kept) == {"lane_gt", "h", "tokens", "mask"}, sorted(kept)
    for head in ("objects", "signs", "curbs", "lanes"):
        assert not any(name.startswith(head[:-1]) or name == head for name in kept if name != "lane_gt"), head
    assert set(trained_outputs(everything, ("lane_gt", "lanes"))) == {"lane_gt", "lanes", "lane_valid", "h", "tokens", "mask"}

    # The trainer stamps only heads that had a label (rec5b-like rows: lane_gt and lane polylines, nothing else).
    row = {"tech": {"lane_gt": [0.1] * LANE_GT_DIM, "lane_gt_valid": [1] * LANE_GT_DIM,
                    "lanes": [[{"x": 0.0, "y": 5.0}, {"x": 0.0, "y": 30.0}]]}}
    stamped, unlabelled, counts = labelled_heads([row] * 500, None)
    assert "lane_gt" in stamped and {"objects", "signs", "curbs"} <= set(unlabelled), (stamped, unlabelled)
    assert counts["objects"] == {"labelled_rows": 0, "labelled_share": 0.0, "stamped": False}
    assert counts["lane_gt"] == {"labelled_rows": 500, "labelled_share": 1.0, "stamped": True}
    assert labelled_heads([{"tech": {}}] * 500, ("lane_gt",))[:2] == ((), ("lane_gt",))
    check_label_floor(row)
    assert metadata_dict(("main",), (), height=1, width=1)["gvd_heads"] == "", "no labelled head stamps nothing"
    assert metadata_dict(("main",), ("lane_gt",), height=1, width=1)["gvd_heads"] == "lane_gt"

    # Nothing outside the runtime reads scene-net outputs: run_vision only takes lane_x/conf through the guard.
    import re

    root = Path(__file__).resolve().parents[1]
    vision = (root / "python" / "run_vision.py").read_text(encoding="utf-8")
    used = set(re.findall(r"scene_lane\.(\w+)", vision))
    assert used <= {"path", "ready", "reason", "cams", "heads"}, used
    for rel in ("python/run_vision.py", "python/viz", "python/planning", "python/perception", "python/control"):
        target = root / rel
        files = [target] if target.is_file() else sorted(target.rglob("*.py"))
        for file in files:
            text = file.read_text(encoding="utf-8")
            assert "last_outputs" not in text and "SceneLaneRuntime(" not in text.replace(
                "SceneLaneRuntime(path, live_cams=live_cams)", ""), file

    note = "(onnx/onnxruntime not installed: runtime load not run)"
    eight = ("narrow", "main", "wide", "pillarL", "pillarR", "repeatL", "repeatR", "rear")
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        extra = ("objects", "object_valid", "signs", "sign_valid", "curbs", "curb_valid")
        if _tiny_scene_onnx(tmp / "lane.onnx", ("main", "narrow", "wide"), extra=extra):
            try:
                import onnxruntime  # noqa: F401
            except ImportError:
                pass
            else:
                import numpy as np
                from python.data.strip_writer import CANVAS_H

                _tiny_scene_onnx(tmp / "nohead.onnx", ("main", "narrow", "wide"), heads=None, extra=extra)
                _tiny_scene_onnx(tmp / "objonly.onnx", ("main", "narrow", "wide"), heads=("objects",), extra=extra)
                rt = SceneLaneRuntime(tmp / "lane.onnx", live_cams=eight)
                assert rt.ready and rt.heads == ("lane_gt",), rt.reason
                strip = np.zeros((CANVAS_H, rt.canvas_w, 3), np.uint8)
                lane_x, _conf = rt.observe(strip, 0.1)
                assert lane_x is not None and set(rt.last_outputs) == {"lane_gt"}, sorted(rt.last_outputs)
                for name in ("nohead.onnx", "objonly.onnx"):
                    guard = SceneLaneGuard()
                    got = guard.load(lambda path: SceneLaneRuntime(path, live_cams=eight), tmp / name)
                    assert got is None and guard.refused and "gvd_heads" in guard.refused, (name, guard.refused)
                legacy = SceneLaneRuntime(tmp / "nohead.onnx")
                assert legacy.ready and legacy.heads == ("lane_gt",), "off the live path: lane-only"
                legacy.observe(strip, 0.1)
                assert set(legacy.last_outputs) == {"lane_gt"}
                note = ("live graph with objects/signs/curbs outputs: only lane_gt read; no gvd_heads / no lane_gt "
                        "refused by SceneLaneGuard")
    print(f"(k) gvd_heads contract: untrained objects/signs/curbs outputs dropped at the runtime, never reach "
          f"run_vision/viz/planner; trainer stamps labelled heads only; {note} OK")


def _snapshot(*roots: Path) -> dict[str, tuple[str, int, int]]:
    import hashlib

    out = {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                st = path.stat()
                out[str(path)] = (hashlib.sha256(path.read_bytes()).hexdigest(), st.st_size, st.st_mtime_ns)
    return out


def check_links_read_only() -> None:
    import json
    import os
    import time

    import python.train.train_scene as ts
    from python.data.raw_frames import load_raw_cameras, read_image_with_repair
    from python.train.combine_recordings import CombineError, _link, combine_recordings
    from python.train.frame_cache import FrameCache, build_frame_cache
    from python.train.scene_loader import load_canvas, select_training_rows

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        a, b = tmp / "rec5b", tmp / "rec5c"
        _write_recording(a, 6, t0=1000.0, value=60)
        _write_recording(b, 5, t0=2000.0, value=180)
        _png(a / "raw_repaired" / "main" / "000002.png", 61)
        old = time.time() - 86400  # a day old, so any touch would show
        for path in list(a.rglob("*")) + list(b.rglob("*")):
            if path.is_file():
                os.utime(path, (old, old))
        before = _snapshot(a, b)
        out = tmp / "combined"
        man = combine_recordings([a, b], out)
        assert man["files"]["hardlink"] == 11 * 3 + 1, man["files"]
        assert os.path.samefile(a / "raw_repaired" / "main" / "000002.png", out / "raw_repaired" / "main" / "000002.png")
        rows = [json.loads(line) for line in (out / "state.jsonl").read_text().splitlines()]
        kept, _missing, _e = select_training_rows(out, rows, cams=("main", "narrow", "wide"))
        assert len(kept) == 11
        cache_dir = tmp / "cache"
        build_frame_cache(out, cache_dir, "main,narrow,wide")
        cache = FrameCache(cache_dir, strips_root=out)
        for row in kept:
            assert ts._frame_row(out, row, cams=("main", "narrow", "wide"), cache=cache) is not None
            assert ts._frame_row(out, row, cams=("main", "narrow", "wide"), cache=None) is not None
            assert load_raw_cameras(out, int(row["i"]))
            load_canvas(out, int(row["i"]))
        assert read_image_with_repair(out / "raw" / "main" / "000002.png") is not None
        # A second write onto a linked frame is refused, never written through to the source.
        try:
            _link(b / "raw" / "main" / "000000.png", out / "raw" / "main" / "000000.png", {"hardlink": 0, "copy": 0})
            raise AssertionError("wrote over a linked frame")
        except CombineError:
            pass
        after = _snapshot(a, b)
        assert after == before, [k for k in before if before[k] != after.get(k)][:3]
        assert os.path.samefile(a / "raw" / "wide" / "000005.png", out / "raw" / "wide" / "000005.png")
    print(f"(l) {len(before)} source files (frames, raw_repaired, jsonl) identical in sha256, size and mtime after "
          "combine, cache build, cached + raw training reads and a refused re-link OK")


def check_val_composition() -> None:
    from python.train.continual import (
        cell_holdout_labels,
        choose_holdout_cells,
        format_composition,
        lopsided_warnings,
        pass_plan,
        speed_band,
        split_composition,
    )
    from python.train.train_scene import build_parser

    def row(i: int, phase: str | None, spd: float) -> dict:
        out = {"i": i, "t": float(i), "ego": {"speed_mps": spd}}
        if phase:
            out["excursion"] = {"phase": phase}
        return out

    rows = [row(i, "cruise", 20.0) for i in range(60)] + [row(60 + i, "weave", 8.0) for i in range(20)]
    rows += [row(80 + i, None, 2.0) for i in range(10)] + [row(90 + i, "cruise", 30.0) for i in range(10)]
    plan = [
        {"kind": "train", "rows": rows[0:40]},
        {"kind": "train", "rows": rows[20:70]},  # overlap: counted once
        {"kind": "holdout", "rows": rows[70:80] + rows[90:100]},
    ]
    comp = split_composition(rows, plan, buffer_rows=7)
    assert comp["rows"] == {"train": 70, "val": 20, "dropped": 10} and comp["buffer_rows"] == 7, comp["rows"]
    assert comp["train"]["phase"] == {"cruise": round(60 / 70, 4), "weave": round(10 / 70, 4)}
    assert comp["val"]["phase"] == {"cruise": 0.5, "weave": 0.5}
    assert comp["val"]["speed"] == {"25+": 0.5, "5-15": 0.5} and comp["train"]["speed"]["15-25"] == round(60 / 70, 4)
    warns = lopsided_warnings(comp)
    assert any(w.startswith("phase weave") for w in warns) and any(w.startswith("speed 25+") for w in warns), warns
    assert any(w.startswith("phase cruise") for w in warns) is False, "0.86 vs 0.5 is within 2x"
    even = split_composition(rows, [{"kind": "train", "rows": rows[::2]}, {"kind": "holdout", "rows": rows[1::2]}])
    assert lopsided_warnings(even) == [], lopsided_warnings(even)
    assert speed_band({"ego": {}}) == "no_speed" and "buffer=7" in format_composition(comp)

    # On the real cell split: every row lands in exactly one of train / val / dropped.
    loop_rows = [{"i": i, "t": float(i), "map": "m", "segment": "s#0", "excursion": {"phase": "cruise"},
                  "ego": {"speed_mps": 25.0, "pos": [float((i * 26) % 2000), float(((i * 26) // 2000) * 300)]}}
                 for i in range(800)]
    spec = choose_holdout_cells(loop_rows, seed=0)
    marks = cell_holdout_labels(loop_rows, spec)
    cplan = pass_plan(loop_rows, 64, 32, cells=spec)
    ccomp = split_composition(loop_rows, cplan, buffer_rows=marks.count("gap"))
    assert sum(ccomp["rows"].values()) == 800 and ccomp["rows"]["val"] == marks.count("hold"), ccomp["rows"]
    assert ccomp["rows"]["dropped"] >= marks.count("gap")
    assert spec["seed"] == 0 and choose_holdout_cells(loop_rows, seed=0) == spec, "fixed seed, same draw"
    assert build_parser().parse_args([]).holdout_seed == 0
    trainer = (Path(__file__).resolve().parents[1] / "python" / "train" / "train_scene.py").read_text(encoding="utf-8")
    assert '"holdout": holdout_meta' in trainer and '"seed": None if cell_spec is None else cell_spec.get("seed")' in trainer
    print(f"(m) composition: {format_composition(comp)}; lopsided: {warns}; cell split {ccomp['rows']} "
          f"buffer={ccomp['buffer_rows']}; seed fixed (default 0) and recorded OK")


def check_leakage_note() -> None:
    from python.debug.lane_targets import LOOKAHEADS_M
    from python.train.continual import HOLDOUT_BUFFER_M, holdout_leakage_note

    note = holdout_leakage_note()
    assert max(LOOKAHEADS_M) < HOLDOUT_BUFFER_M, "the buffer must still cover the farthest label"
    assert "50 m buffer" in note and "30 m" in note and "not no leakage" in note.lower(), note
    trainer = (Path(__file__).resolve().parents[1] / "python" / "train" / "train_scene.py").read_text(encoding="utf-8")
    assert 'holdout_meta["limit"] = holdout_leakage_note(' in trainer and '"holdout": holdout_meta' in trainer
    print(f"(n) run metadata holdout.limit: {note!r} OK")


def check_labels_on_training_rows() -> None:
    import random

    from python.debug.lane_targets import LANE_GT_DIM
    from python.train.continual import (
        cell_holdout_labels,
        choose_holdout_cells,
        epoch_windows,
        pass_plan,
        training_rows,
    )
    from python.train.scene_net import labelled_heads

    def same_as_epoch_windows(plan: list, min_valid: float) -> int:
        """training_rows and the rows epoch_windows trains on are one set (balance and shuffle too)."""
        mine = {int(r["i"]) for r in training_rows(plan, min_valid=min_valid)}
        for balance in (False, True):
            yielded = epoch_windows(plan, random.Random(7), min_valid=min_valid, balance=balance)
            trained = {int(r["i"]) for item in yielded if item.get("kind") != "holdout" for r in item["rows"]}
            assert mine == trained, (min_valid, balance, sorted(mine ^ trained)[:5])
        return len(mine)

    line = [[{"x": 0.0, "y": 5.0}, {"x": 0.0, "y": 30.0}]]

    def tech(lane_ok: bool, **extra) -> dict:
        out = {"lane_gt": [0.1] * LANE_GT_DIM, "lane_gt_valid": [1 if lane_ok else 0] * LANE_GT_DIM}
        out.update(extra)
        return out

    def loop_rows(n: int) -> list[dict]:
        return [{"i": i, "t": float(i), "map": "m", "segment": "s#0",
                 "ego": {"speed_mps": 25.0, "pos": [float((i * 26) % 2000), float(((i * 26) // 2000) * 300)]}}
                for i in range(n)]

    # Holdout: curbs labelled only on held-out road.
    rows = loop_rows(800)
    spec = choose_holdout_cells(rows, seed=0)
    marks = cell_holdout_labels(rows, spec)
    for row, mark in zip(rows, marks):
        row["tech"] = tech(True, curbs=line) if mark == "hold" else tech(True)
    held = marks.count("hold")
    everywhere = labelled_heads(rows, ("lane_gt", "curbs"))
    assert everywhere[0] == ("lane_gt", "curbs"), "on all rows curbs would clear the floor"
    cell_plan = pass_plan(rows, 64, 32, cells=spec)
    same_as_epoch_windows(cell_plan, 0.3)
    train = training_rows(cell_plan, min_valid=0.3)
    stamped, below, counts = labelled_heads(train, ("lane_gt", "curbs"))
    assert stamped == ("lane_gt",) and below == ("curbs",) and counts["curbs"]["labelled_rows"] == 0, counts
    assert not {int(r["i"]) for r in train} & {int(r["i"]) for r, m in zip(rows, marks) if m != "train"}

    # --min-valid: lane polylines only in a stretch whose lane centre is invalid.
    rows = loop_rows(800)
    for row in rows:
        bad = 100 <= int(row["i"]) < 400
        row["tech"] = tech(False, lanes=line) if bad else tech(True)
    assert labelled_heads(rows, ("lane_gt", "lanes"))[0] == ("lane_gt", "lanes")
    plan = pass_plan(rows, 64, 32)
    n_kept = same_as_epoch_windows(plan, 0.3)
    n_loose = same_as_epoch_windows(plan, 0.0)
    assert n_kept < n_loose, "--min-valid dropped windows on the (p) data"
    kept = training_rows(plan, min_valid=0.3)
    stamped2, below2, counts2 = labelled_heads(kept, ("lane_gt", "lanes"))
    assert stamped2 == ("lane_gt",) and below2 == ("lanes",), counts2
    assert counts2["lanes"]["labelled_rows"] < 100, counts2
    loose = labelled_heads(training_rows(plan, min_valid=0.0), ("lane_gt", "lanes"))
    assert "lanes" in loose[0], "without --min-valid those windows would train it"

    trainer = (Path(__file__).resolve().parents[1] / "python" / "train" / "train_scene.py").read_text(encoding="utf-8")
    call = trainer.index("label_rows = training_rows(")
    assert "pass_plan(rows, chunk_len, stride, blocks=int(holdout_blocks), cells=cell_spec), min_valid=float(min_valid)" \
        in trainer[call:call + 300]
    assert "labelled_heads(label_rows, train_heads)" in trainer and "labelled_heads(rows, train_heads)" not in trainer
    print(f"(p) curbs labelled on {held} held-out rows only: 0 in training rows, not stamped; lanes labelled only "
          f"where --min-valid drops the windows: {counts2['lanes']['labelled_rows']} training rows, not stamped; "
          f"training_rows == epoch_windows' training rows (holdout, min_valid 0.3 and 0, balance on/off) OK")


def check_export_only_heads() -> None:
    import contextlib
    import io

    import numpy as np

    from python.data import data_root
    from python.train import train_scene as ts
    from python.train.checkpoint import checkpoint_path, load_checkpoint_file, save_checkpoint

    run = {"cams": ["wide", "main", "narrow"], "heads": ["lane_gt"], "heads_requested": "all",
           "heads_unlabelled": ["lanes", "curbs", "objects", "signs"],
           "head_labels": {"lane_gt": {"labelled_rows": 900, "labelled_share": 0.9, "stamped": True},
                           "objects": {"labelled_rows": 0, "labelled_share": 0.0, "stamped": False}},
           "label_floor": {"share": 0.01, "rows": 100, "training_rows": 1000},
           "net": {"backbone": "", "scale": 1.0}}
    heads, cams, net = ts.export_contract_from_checkpoint({"run": run})
    assert heads == ("lane_gt",) and cams == ("wide", "main", "narrow") and net["backbone"] == ""
    assert ts.export_contract_from_checkpoint({"run": run}, heads_flag="lane-gt", cams_flag="main,narrow,wide")[0] == heads
    refusals = (
        ({}, "", "no run record"),
        ({"run": {"cams": ["main"]}}, "", "no heads list"),
        ({"run": dict(run, heads=[])}, "", "lists no trained head"),
        ({"run": run}, "lane_gt,objects", "disagrees"),
        ({"run": run}, "objects", "disagrees"),
    )
    for payload, flag, why in refusals:
        try:
            ts.export_contract_from_checkpoint(payload, heads_flag=flag)
            raise AssertionError(f"export accepted: {why}")
        except ts.ExportRefused as exc:
            assert why in str(exc), (why, exc)
    try:
        ts.export_contract_from_checkpoint({"run": run}, cams_flag="main")
        raise AssertionError("--cams disagreement accepted")
    except ts.ExportRefused:
        pass

    gb = 1024 ** 3
    real_free = data_root.free_bytes
    data_root.free_bytes = lambda _path: 50 * gb
    try:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            # The run record travels inside the pause file.
            save_checkpoint(tmp / "rec_a", {"weights": {"w": np.zeros(2, np.float32)}, "run": run})
            assert load_checkpoint_file(checkpoint_path(tmp / "rec_a"))["run"] == run
            save_checkpoint(tmp / "rec_old", {"weights": {"w": np.zeros(2, np.float32)}})

            def refused(argv: list[str]) -> str:
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    try:
                        ts.main(argv)
                        raise AssertionError(f"main accepted {argv}")
                    except SystemExit as exc:
                        assert exc.code == 2, exc.code
                return out.getvalue()

            onnx_out = tmp / "out" / "e2e_scene.onnx"
            msg = refused(["--export", str(onnx_out)])
            assert "needs --checkpoint" in msg, msg
            msg = refused(["--export", str(onnx_out), "--checkpoint", str(checkpoint_path(tmp / "rec_old"))])
            assert "no run record" in msg, msg
            msg = refused(["--export", str(onnx_out), "--checkpoint", str(checkpoint_path(tmp / "rec_a")),
                           "--heads", "lane_gt,objects"])
            assert "disagrees" in msg, msg
            assert not onnx_out.exists(), "a refused export wrote a file"
            note = "(torch not installed: full export not run)"
            if ts.torch_ready():
                try:
                    import onnx
                except ImportError:
                    onnx = None
                net = ts.build_export_net(("wide", "main", "narrow"), {"backbone": "", "scale": 1.0})
                payload = ts.stamp_pause(
                    {"weights": {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}, "run": run},
                    tuple(getattr(net, "cams", ()) or ()) or None,
                    getattr(net, "pause_arch", None),
                )
                save_checkpoint(tmp / "rec_t", payload)
                with contextlib.redirect_stdout(io.StringIO()):
                    ts.main(["--export", str(onnx_out), "--checkpoint", str(checkpoint_path(tmp / "rec_t"))])
                assert onnx_out.is_file()
                if onnx is not None:
                    meta = {p.key: p.value for p in onnx.load(str(onnx_out)).metadata_props}
                    assert meta["gvd_heads"] == "lane_gt", meta
                    assert meta["gvd_cams"] == "wide,main,narrow", meta
                    note = "torch export from the checkpoint stamped gvd_heads=lane_gt, gvd_cams=wide,main,narrow"
    finally:
        data_root.free_bytes = real_free
    print(f"(q) export-only takes gvd_heads from the checkpoint's run record; no checkpoint, no record, no heads "
          f"or a disagreeing --heads/--cams refused, nothing written; {note} OK")


def main() -> None:
    check_multi_recording()
    check_road_area_holdout()
    check_heads_contract()
    check_links_read_only()
    check_val_composition()
    check_leakage_note()
    check_labels_on_training_rows()
    check_export_only_heads()
    check_camera_contract()
    check_cache_on_c()
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
