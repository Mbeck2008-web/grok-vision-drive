#!/usr/bin/env python3
"""Hold-last side cameras (``--cam-age``): the shared rule, the parity of the
live strip and the frame cache, the ``ages`` ONNX input and the runtime gate.

* no look-ahead: a later frame is never picked, by the tracker or by the plan.
* no boundary crossing: respawn, segment, null dt and a train/holdout change reset.
* an age over the 1.5 s cap is a black sector with age -1.
* ages sit on the 1/15 s supervisor tick.
* main is never held.
* the trainer (raw and cache path) and a live tracker give byte-equal sectors
  and equal ages on one synthetic timestamp sequence.
* parity: ``compose_strip`` and the frame cache canvas are byte-equal on all
  eight cameras (repeaters included); an old-format cache is stale; a build is
  create-only.
* ONNX round trip with ``ages`` (torch), the runtime refuses a flag mismatch,
  and an old model (no ``gvd_cam_age``) still loads and runs.
* the read-only report helper's numbers on the fixture.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from gvd_test_env import skip_check  # noqa: E402
from python.data.cam_hold import (  # noqa: E402
    CAM_AGE_ORDER,
    HOLD_CAP_S,
    HOLD_TICK_S,
    NO_FRAME_AGE,
    CamHold,
    ages_vector,
    hold_plan,
    hold_sources,
    held_sector,
    plan_labels,
    quantise_age,
    respawn_starts,
    row_now,
)
from python.data.strip_writer import SECTORS, STRIP_ORDER, compose_strip  # noqa: E402

# Per-camera render period in ticks (10 Hz ticks): front every tick, pillars
# every third, repeaters and rear every fifth, the recorder's slow cameras.
PERIOD = {"main": 1, "narrow": 1, "wide": 1, "pillarL": 3, "pillarR": 3, "repeatL": 5, "repeatR": 5, "rear": 5}
OFFSET = {"pillarL": 0.004, "pillarR": 0.006, "repeatL": 0.009, "repeatR": 0.011, "rear": 0.013}
T0 = 1_792_000_000.0
N_TICKS = 90


def _shape(cam: str) -> tuple[int, int, int]:
    return (54, 96, 3) if cam in ("wide", "rear") else (48, 64, 3)


def _ticks() -> list[dict]:
    """Synthetic recording ticks: seg#0 with a respawn (settle run) at 40, seg#1 from 60, a 2 s hole at 75."""
    out = []
    t = T0
    for k in range(N_TICKS):
        if k > 0:
            t += 2.0 if k == 75 else 0.1
        tags: dict = {"segment": "west_coast_usa#0" if k < 60 else "west_coast_usa#1"}
        if 40 <= k < 46 or k == 60:
            tags["settle"] = True
        cams = {cam: round(t + OFFSET.get(cam, 0.0), 6) for cam, p in PERIOD.items() if k % p == 0}
        # pillarL misses a long run in seg#0: 1.6 s without a render.
        if 10 < k <= 26:
            cams.pop("pillarL", None)
        out.append({"k": k, "t": round(t, 6), "cams": cams, "tags": tags})
    return out


def _pixels(cam: str, k: int) -> np.ndarray:
    rng = np.random.default_rng(1000 * (STRIP_ORDER.index(cam) + 1) + k)
    return (rng.random(_shape(cam)) * 200 + 30).astype(np.uint8)


def _write_recording(root: Path) -> list[dict]:
    import cv2

    rows = []
    prev_t = None
    for tick in _ticks():
        k = tick["k"]
        cams = {}
        for cam, stamp in tick["cams"].items():
            folder = root / "raw" / cam
            folder.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".png", _pixels(cam, k))
            assert ok
            (folder / f"{k:06d}.png").write_bytes(buf.tobytes())
            cams[cam] = {"file": f"raw/{cam}/{k:06d}.png", "t": stamp}
        row = {"i": k, "t": tick["t"], "dt_s": None if prev_t is None else round(tick["t"] - prev_t, 6), "cams": cams}
        row.update(tick["tags"])
        prev_t = tick["t"]
        rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    (root / "state.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    gt = []
    for row in rows:
        center = 0.5 * math.sin(row["i"] / 7.0)
        lane = []
        for _ in range(4):
            lane += [center, center - 1.8, center + 1.8]
        gt.append({"i": row["i"], "t": row["t"], "tech": {"lane_gt": lane + [0.0, 1.0], "lane_gt_valid": [1.0] * 14}})
    (root / "gt.jsonl").write_text("\n".join(json.dumps(r) for r in gt) + "\n", encoding="utf-8")
    return rows


def _live(rows: list[dict]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """What a live supervisor does: note each new render with its pixels, reset at a respawn, pick at the tick."""
    hold = CamHold()
    prev = None
    out = {}
    for row in rows:
        if respawn_starts(row, prev):
            hold.reset()
        prev = row
        for cam, meta in row["cams"].items():
            hold.note(cam, meta["t"], _pixels(cam, row["i"]))
        picks = hold.pick(row_now(row))
        frames = {cam: frame for cam, (frame, _age) in picks.items() if frame is not None}
        out[row["i"]] = (compose_strip(frames), ages_vector(picks))
    return out


def check_quantise_and_cap() -> None:
    assert quantise_age(0.0) == 0.0
    assert quantise_age(0.03) == 0.0  # under half a tick
    assert quantise_age(0.04) == 1 / 15
    assert quantise_age(0.1) == 2 / 15  # 1.5 ticks rounds up
    assert quantise_age(0.3004) == 5 / 15
    for raw in (0.0, 0.033, 0.034, 0.1, 0.2, 0.31, 0.5, 1.0, 1.4):
        _frame, age = held_sector("pillarL", 10.0, "f", 10.0 + raw)
        assert abs(age * 15 - round(age * 15)) < 1e-9 and abs(age - raw) <= HOLD_TICK_S / 2 + 1e-9, (raw, age)
    assert held_sector("rear", 0.0, "f", 22 / 15) == ("f", 22 / 15)
    assert held_sector("rear", 0.0, "f", 1.49) == ("f", 22 / 15)
    # The cap is on the quantised age: 1.5 s is 22.5 ticks, rounds to 23 (1.533 s) and is over it.
    assert held_sector("rear", 0.0, "f", HOLD_CAP_S) == (None, NO_FRAME_AGE)
    frame, age = held_sector("rear", 0.0, "f", 23 / 15)
    assert frame is None and age == NO_FRAME_AGE, (frame, age)
    frame, age = held_sector("pillarR", 0.0, "f", 1.6)
    assert frame is None and age == -1.0
    assert held_sector("pillarR", None, "f", 1.0) == (None, -1.0)
    assert held_sector("pillarR", 0.0, None, 0.0) == (None, -1.0)
    print("quantise: ages on the 1/15 s tick; > 1.5 s cap is black with age -1 OK")


def check_main_never_held() -> None:
    assert held_sector("main", 5.0, "m", 5.0) == ("m", 0.0)
    assert held_sector("main", 5.0, "m", 5.02) == ("m", 0.0)  # same tick, a few ms
    assert held_sector("main", 5.0, "m", 5.1) == (None, -1.0)
    hold = CamHold()
    hold.note("main", 1.0, "m0")
    hold.note("pillarL", 1.0, "p0")
    picks = hold.pick(1.1)
    assert picks["main"] == (None, -1.0) and picks["pillarL"] == ("p0", 2 / 15), picks
    print("main never held: previous-tick main is black, this tick's is age 0 OK")


def check_no_look_ahead() -> None:
    assert held_sector("pillarL", 2.0, "late", 1.9) == (None, -1.0)
    hold = CamHold()
    hold.note("pillarL", 1.0, "a")
    hold.note("pillarL", 0.9, "older")  # an older stamp never replaces a newer one
    assert hold.last("pillarL") == (1.0, "a")
    rows = _ticks_as_rows()
    plan = hold_plan(rows)
    for row in rows:
        for cam in CAM_AGE_ORDER:
            src, age = plan[row["i"]][cam]
            if src is None:
                continue
            assert src <= row["i"], (row["i"], cam, src)
            assert age >= 0.0
    # pillarL renders at 0, 3, 6, 9 then not until 27: rows 10..26 hold 9 until the cap.
    assert plan[10]["pillarL"][0] == 9 and plan[11]["pillarL"][0] == 9
    print("no look-ahead: every source is at or before its row; older stamps never win OK")


def _ticks_as_rows() -> list[dict]:
    rows = []
    prev_t = None
    for tick in _ticks():
        row = {"i": tick["k"], "t": tick["t"], "dt_s": None if prev_t is None else round(tick["t"] - prev_t, 6)}
        row["cams"] = {cam: {"t": stamp} for cam, stamp in tick["cams"].items()}
        row.update(tick["tags"])
        prev_t = tick["t"]
        rows.append(row)
    return rows


def check_boundaries() -> None:
    rows = _ticks_as_rows()
    plan = hold_plan(rows)
    # Respawn: a settle run starts at 40. Repeaters rendered at 35; 36..39 hold it, 40 renders again.
    assert plan[39]["repeatL"][0] == 35
    assert plan[41]["repeatL"][0] == 40
    # pillarL renders at 39 (before the respawn) and 42; 41 holds 40? No: 40 is not a pillar tick,
    # so 40 and 41 have no pillar frame after the reset, never 39's.
    assert plan[39]["pillarL"][0] == 39 and plan[40]["pillarL"] == [None, -1.0] and plan[41]["pillarL"] == [None, -1.0]
    # Segment change at 60 (also a settle start): 59 holds 55's repeater, 61 holds 60's.
    assert plan[59]["repeatL"][0] == 55 and plan[61]["repeatL"][0] == 60
    # A row whose only earlier pillar is across the respawn boundary: drop the 40 render and check 41.
    cut = [dict(r, cams={c: m for c, m in r["cams"].items() if not (r["i"] == 40 and c != "main")}) for r in rows]
    plan_cut = hold_plan(cut)
    assert plan_cut[41]["repeatL"] == [None, -1.0], plan_cut[41]["repeatL"]
    # The 2 s hole at 75 is over the cap anyway; a null dt is a hard reset too.
    nulled = [dict(r, dt_s=None) if r["i"] == 52 else r for r in rows]
    assert hold_plan(nulled)[53]["repeatL"] == [None, -1.0]
    assert hold_plan(rows)[53]["repeatL"][0] == 50
    # Train/holdout: rows 0..29 train, 30..39 holdout. 31 must not show 30's predecessor 27/25.
    fake_plan = [
        {"kind": "train", "rows": [r for r in rows if r["i"] < 30]},
        {"kind": "holdout", "rows": [r for r in rows if 30 <= r["i"] < 40]},
    ]
    labelled = hold_plan(rows, plan_labels(fake_plan))
    assert hold_plan(rows)[31]["pillarL"][0] == 30
    assert labelled[31]["pillarL"][0] == 30 and labelled[34]["repeatL"][0] == 30
    assert labelled[30]["pillarR"][0] == 30
    assert labelled[32]["repeatL"][0] == 30 and hold_plan(rows)[29]["repeatL"][0] == 25
    # 30 is a holdout row; drop its side renders: holdout must not reach back to train frame 25/27.
    cut30 = [dict(r, cams={c: m for c, m in r["cams"].items() if not (r["i"] == 30 and c != "main")}) for r in rows]
    lab30 = hold_plan(cut30, plan_labels(fake_plan))
    assert hold_plan(cut30)[31]["repeatL"][0] == 25  # without labels it would hold the train frame
    assert lab30[31]["repeatL"] == [None, -1.0] and lab30[31]["pillarL"] == [None, -1.0], lab30[31]
    assert 45 not in lab30 and 10 in lab30  # unlabelled rows are not targets
    print("no boundary crossing: respawn, segment, null dt and train/holdout runs reset the hold OK")


def check_train_live_identical() -> None:
    from python.data.raw_frames import load_held_bgr
    from python.train.frame_cache import FrameCache, build_frame_cache
    from python.train.train_scene import held_canvas

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "rec"
        rows = _write_recording(root)
        live = _live(rows)
        plan = hold_plan(rows)
        build_frame_cache(root, Path(tmp) / "cache", ",".join(STRIP_ORDER))
        cache = FrameCache(Path(tmp) / "cache")
        assert not cache.stale, cache.reason
        held = black = 0
        for row in rows:
            row = dict(row, cam_hold=plan[row["i"]])
            sources = hold_sources(row)
            canvas, ages = load_held_bgr(root, sources)
            want_canvas, want_ages = live[row["i"]]
            assert np.array_equal(ages, want_ages), (row["i"], ages, want_ages)
            assert canvas.dtype == np.uint8 and np.array_equal(canvas, want_canvas), row["i"]
            cached, cached_ages = held_canvas(root, row, cache)
            assert np.array_equal(cached, want_canvas) and np.array_equal(cached_ages, want_ages), row["i"]
            held += int(((ages > 0)).sum())
            black += int((ages < 0).sum())
        assert held > 100 and black >= 4, (held, black)
    print(f"train == live: raw path, cache path and a live tracker agree on {len(rows)} ticks "
          f"({held} held sectors, {black} black) OK")


def check_parity_compose_vs_cache() -> None:
    import cv2

    from python.train.frame_cache import CACHE_FORMAT, MANIFEST, FrameCache, build_frame_cache

    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "rec"
        frames = {}
        for cam in STRIP_ORDER:
            (root / "raw" / cam).mkdir(parents=True)
            h, w = (180, 320) if cam in ("wide", "rear") else (150, 200)
            image = (rng.random((h, w, 3)) * 255).astype(np.uint8)
            if cam == "narrow":
                image = (rng.random((192, 256, 3)) * 255).astype(np.uint8)  # already sector size
            cv2.imwrite(str(root / "raw" / cam / "000000.png"), image)
            frames[cam] = image
        (root / "state.jsonl").write_text(json.dumps({"i": 0, "t": 0.0}) + "\n", encoding="utf-8")
        dest = Path(tmp) / "cache"
        manifest = build_frame_cache(root, dest, ",".join(STRIP_ORDER))
        assert manifest["format"] == CACHE_FORMAT
        cache = FrameCache(dest)
        cached = cache.canvas(0)
        live = compose_strip(frames)
        for sec in SECTORS:
            a = live[:, sec.x0 : sec.x1]
            b = cached[:, sec.x0 : sec.x1]
            assert a.tobytes() == b.tobytes(), (sec.cam_id, int((a != b).any(axis=2).sum()))
            assert np.array_equal(cache.sector(sec.cam_id, 0), a), sec.cam_id
        # The repeater ego band is black in both (it was the format-1 difference).
        rl = [s for s in SECTORS if s.cam_id == "repeatL"][0]
        rr = [s for s in SECTORS if s.cam_id == "repeatR"][0]
        assert not cached[:, rl.x0 : rl.x0 + 64].any() and cached[:, rl.x0 + 64 : rl.x1].any()
        assert not cached[:, rr.x1 - 64 : rr.x1].any() and cached[:, rr.x0 : rr.x1 - 64].any()
        assert cached.tobytes() == live.tobytes()
        # Create-only: a second build into the same folder is refused and leaves it as it was.
        before = (dest / MANIFEST).read_bytes()
        try:
            build_frame_cache(root, dest, "main")
        except FileExistsError:
            pass
        else:
            raise AssertionError("a cache build overwrote an existing cache")
        assert (dest / MANIFEST).read_bytes() == before
        # An old (format 1, unversioned) manifest is stale and serves nothing.
        old = Path(tmp) / "old"
        old.mkdir()
        payload = json.loads(before)
        payload.pop("format")
        (old / MANIFEST).write_text(json.dumps(payload), encoding="utf-8")
        stale = FrameCache(old)
        assert stale.stale and not stale.has(0) and "NEW folder" in stale.reason, stale.reason
        # A tampered manifest (sha mismatch) is stale too.
        bad = Path(tmp) / "bad"
        bad.mkdir()
        (bad / MANIFEST).write_bytes(before.replace(b'"frames": 1', b'"frames": 2'))
        (bad / "manifest.sha256").write_bytes((dest / "manifest.sha256").read_bytes())
        assert FrameCache(bad).stale
    print("parity: compose_strip and the frame cache are byte-equal on all 8 sectors (repeater ego band too); "
          "old caches stale; builds create-only OK")


def check_report_helper() -> None:
    from cam_hold_report import main as report_main, report

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "rec"
        _write_recording(root)
        before = {p: p.stat().st_mtime_ns for p in root.rglob("*")}
        result = report(root, cams=tuple(STRIP_ORDER), holdout_blocks=1)
        assert sorted(result["black_pct"]) == ["holdout", "holdout_scored", "train"]
        assert set(result["black_pct"]["train"]) == set(STRIP_ORDER)  # all 8, main included
        assert result["black_pct"]["train"]["main"] == 0.0
        for name in ("train", "holdout"):
            assert result["black_pct"][name]["wide"] == 0.0 and result["black_pct"][name]["narrow"] == 0.0
        assert result["black_pct"]["train"]["pillarL"] > 0.0  # the 1.6 s pillarL gap
        assert result["train_age_s"]["repeatL"]["p95"] <= HOLD_CAP_S
        assert result["train_age_s"]["wide"]["median"] == 0.0
        # Black counted at the cap (1.533 s): the pillarL gap pulls its capped p95 up to the cap;
        # n is every train row, not only the held ones.
        from python.data.cam_hold import BLACK_STAT_AGE_S, stat_age

        assert abs(BLACK_STAT_AGE_S - 23 / 15) < 1e-12 and stat_age(-1.0) == BLACK_STAT_AGE_S and stat_age(None) == BLACK_STAT_AGE_S
        assert stat_age(0.4) == 0.4
        capped = result["train_age_capped_s"]
        assert capped["pillarL"]["n"] == result["rows"]["train"] > result["train_age_s"]["pillarL"]["n"]
        assert capped["pillarL"]["p95"] == round(BLACK_STAT_AGE_S, 4) > result["train_age_s"]["pillarL"]["p95"]
        assert capped["wide"] == dict(result["train_age_s"]["wide"], n=result["rows"]["train"])
        assert result["rows"]["holdout"] > 0 and result["rows"]["train"] > 0
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert report_main(["--strips", str(root), "--holdout-blocks", "1"]) == 0
        assert "train black %" in out.getvalue() and "repeatL" in out.getvalue()
        after = {p: p.stat().st_mtime_ns for p in root.rglob("*")}
        assert before == after, "the report wrote into the recording"
    print("report: per-camera black % (all 8; train / holdout / scored), train age median/p95 held-only and with "
          "black at the 1.533 s cap, read-only OK")


def _onnx_ready() -> bool:
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    from python.train.scene_net import torch_ready

    return torch_ready()


def _set_meta(path: Path, drop: tuple[str, ...] = (), put: dict | None = None) -> None:
    import onnx

    model = onnx.load(str(path))
    kept = [(p.key, p.value) for p in model.metadata_props if p.key not in drop and p.key not in (put or {})]
    del model.metadata_props[:]
    for key, value in kept + list((put or {}).items()):
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(model, str(path))


def check_onnx_round_trip_and_runtime() -> None:
    if not _onnx_ready():
        skip_check("onnx_round_trip_and_runtime", "torch / onnx / onnxruntime not installed")
        return
    import onnxruntime as ort
    import torch

    from python.data.strip_writer import CANVAS_H, CANVAS_W
    from python.train.scene_lane_runtime import SceneCamAgeError, SceneLaneGuard, SceneLaneRuntime
    from python.train.scene_net import SceneNet, export_scene_onnx

    torch.manual_seed(0)
    cams = ("pillarL", "main")
    with tempfile.TemporaryDirectory() as tmp:
        net = SceneNet(cams=cams, cam_age=True).eval()
        assert net.token_dim == 2 * 64 + 8 and net.pause_arch.endswith(":cam_age=8")
        aged = Path(tmp) / "aged.onnx"
        export_scene_onnx(aged, net)
        session = ort.InferenceSession(str(aged), providers=["CPUExecutionProvider"])
        names = [item.name for item in session.get_inputs()]
        assert "ages" in names, names
        meta = dict(session.get_modelmeta().custom_metadata_map)
        assert meta["gvd_cam_age"] == "1" and meta["gvd_cam_age_order"] == ",".join(CAM_AGE_ORDER), meta
        rng = np.random.default_rng(3)
        strip = (rng.random((CANVAS_H, net.canvas_w, 3)) * 255).astype(np.uint8)
        state = net.initial_state()
        outs = []
        for ages in (np.array([[-1, 0.2, 0, 0, 0, 1 / 15, -1, 1.4]], np.float32), np.full((1, 8), -1, np.float32)):
            with torch.no_grad():
                pred, _ = net.forward(strip, torch.tensor([0.1]), state, torch.from_numpy(ages))
            feeds = {
                "strip": np.ascontiguousarray(np.transpose(strip.astype(np.float32) / 255.0, (2, 0, 1))[None]),
                "dt": np.array([0.1], np.float32),
                "h": state.h.numpy(),
                "tokens": state.tokens.numpy(),
                "mask": state.mask.numpy(),
                "ages": ages,
            }
            got = session.run(None, {i.name: feeds[i.name.split(".", 1)[0]] for i in session.get_inputs()})
            by = {o.name.split(".", 1)[0]: v for o, v in zip(session.get_outputs(), got)}
            diff = float(np.abs(by["lane_gt"].reshape(-1) - pred.lane_gt.numpy().reshape(-1)).max())
            assert diff < 1e-4, diff
            outs.append(by["lane_gt"].reshape(-1))
        assert float(np.abs(outs[0] - outs[1]).max()) > 1e-6, "ages do not reach the output"
        try:
            net.forward(strip, torch.tensor([0.1]), state)
        except ValueError as exc:
            assert "cam-age" in str(exc)
        else:
            raise AssertionError("a cam-age net ran without ages")

        # Runtime: today's supervisor feeds no ages and refuses this model; the guard refuses Engage.
        try:
            SceneLaneRuntime(aged)
        except SceneCamAgeError as exc:
            assert "gvd_cam_age=1" in str(exc), exc
        else:
            raise AssertionError("a runtime that cannot feed ages loaded a gvd_cam_age=1 model")
        guard = SceneLaneGuard()
        assert guard.load(SceneLaneRuntime, aged) is None and guard.status()["state"] == "refused"
        fed = SceneLaneRuntime(aged, feeds_cam_age=True)
        assert fed.ready and fed.cam_age
        full = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
        x, conf = fed.observe(full, 0.1, ages=np.zeros(8, np.float32))
        assert x is not None and conf is not None
        try:
            fed.observe(full, 0.1)
        except SceneCamAgeError:
            pass
        else:
            raise AssertionError("an ages model ran without ages")
        # Flag mismatches: metadata says 0 on an ages graph, or an unknown value.
        for put in ({"gvd_cam_age": "0"}, {"gvd_cam_age": "2"}, {"gvd_cam_age_cap_s": "2.0"}):
            bad = Path(tmp) / f"bad_{len(list(Path(tmp).iterdir()))}.onnx"
            bad.write_bytes(aged.read_bytes())
            _set_meta(bad, put=put)
            try:
                SceneLaneRuntime(bad, feeds_cam_age=True)
            except SceneCamAgeError:
                pass
            else:
                raise AssertionError(f"runtime loaded a model with {put}")

        # Old model: no ages input, and no gvd_cam_age key at all.
        plain_net = SceneNet(cams=cams).eval()
        plain = Path(tmp) / "plain.onnx"
        export_scene_onnx(plain, plain_net)
        assert dict(ort.InferenceSession(str(plain)).get_modelmeta().custom_metadata_map)["gvd_cam_age"] == "0"
        _set_meta(plain, drop=("gvd_cam_age",))
        assert "gvd_cam_age" not in dict(ort.InferenceSession(str(plain)).get_modelmeta().custom_metadata_map)
        for feeds_ages in (False, True):
            runtime = SceneLaneRuntime(plain, feeds_cam_age=feeds_ages)
            assert runtime.ready and not runtime.cam_age, runtime.reason
            x, _conf = runtime.observe(full, 0.1)
            assert x is not None
            x2, _conf = SceneLaneGuard().observe(runtime, full, 0.1, engaged=True)
            assert x2 is not None
        # gvd_cam_age=1 stamped on a graph with no ages input is refused.
        lie = Path(tmp) / "lie.onnx"
        lie.write_bytes(plain.read_bytes())
        _set_meta(lie, put=dict((k, v) for k, v in meta.items() if k.startswith("gvd_cam_age")))
        try:
            SceneLaneRuntime(lie, feeds_cam_age=True)
        except SceneCamAgeError as exc:
            assert "has no 'ages' input" in str(exc), exc
        else:
            raise AssertionError("runtime loaded gvd_cam_age=1 on a graph without ages")
    print("onnx: ages input round trip < 1e-4, metadata gvd_cam_age=1; runtime refuses flag mismatches; "
          "an old model (no key) loads and runs OK")


def main() -> None:
    check_quantise_and_cap()
    check_main_never_held()
    check_no_look_ahead()
    check_boundaries()
    check_train_live_identical()
    check_parity_compose_vs_cache()
    check_report_helper()
    check_onnx_round_trip_and_runtime()
    print("test_cam_hold: all OK")


if __name__ == "__main__":
    main()
